import numpy as np
import pandas as pd
import sys
import os
import multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.oil_rate_equations import OILSimulator
from data.feature_extraction import extract_features
from utils.seeds import get_seed
import joblib

def run_drift_simulation(drift_type='aging'):
    simulator = OILSimulator()
    t_max = 1e-6
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 1000)
    y0 = [1.0, 0.0, 1e8]
    
    base_E_ref = 1.0
    base_phi_ref = 0.0
    
    def current_func(t): return 2.0e17
    
    if drift_type == 'aging':
        # Slow laser-aging drift in mean photon number
        drift_rate = np.random.uniform(-0.2, 0.2)
        def E_ref_func(t): return base_E_ref * (1 + drift_rate * (t / t_max))
        def phi_ref_func(t): return base_phi_ref
        
    elif drift_type == 'thermal':
        # Thermal recalibration transients (transiently raises delta_phi)
        # Slower timescale variation than TWIRL
        drift_amp = np.random.uniform(0.1, 0.5)
        def E_ref_func(t): return base_E_ref
        def phi_ref_func(t): return base_phi_ref + drift_amp * np.sin(2 * np.pi * (1e6) * t)
        
    elif drift_type == 'calibration':
        # Routine watchdog recalibration events
        def E_ref_func(t): return base_E_ref * (1 + 0.05 * np.sin(2 * np.pi * 5e5 * t))
        def phi_ref_func(t): return base_phi_ref + 0.05 * np.cos(2 * np.pi * 5e5 * t)
    else:
        def E_ref_func(t): return base_E_ref
        def phi_ref_func(t): return base_phi_ref
        
    sol = simulator.simulate(t_span, y0, E_ref_func, phi_ref_func, current_func, t_eval)
    
    features = extract_features(sol.t, sol.y[0], sol.y[1], sol.y[2])
    
    # If calibration drift, artificially inject a small baseline QBER shift
    if drift_type == 'calibration':
        features[:, 4] += np.random.uniform(0.005, 0.015, size=features.shape[0])
        
    return features

def _worker(args):
    i, seed_base = args
    np.random.seed(seed_base + i)
    
    drift_types = ['aging', 'thermal', 'calibration']
    d_type = drift_types[i % 3]
    try:
        feats = run_drift_simulation(d_type)
        # Label is 0 (Normal) because these are benign drifts
        return feats, [0] * len(feats)
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None, None

def generate_drift_dataset(num_samples=5000):
    seed = get_seed("drift_generation")
    np.random.seed(seed)
    
    cpu_cores = mp.cpu_count()
    print(f"Generating {num_samples} drift samples using {cpu_cores} cores...")
    
    runs_needed = num_samples // 10
    all_features = []
    all_labels = []
    
    with mp.Pool(processes=cpu_cores) as pool:
        args_iter = [(i, seed) for i in range(runs_needed)]
        for i, result in enumerate(pool.imap_unordered(_worker, args_iter)):
            if i % 100 == 0 and i > 0:
                print(f"Progress: {i}/{runs_needed} runs completed")
            feats, labels = result
            if feats is not None:
                all_features.append(feats)
                all_labels.extend(labels)
                
    X = np.vstack(all_features)[:num_samples]
    y = np.array(all_labels)[:num_samples]
    
    df = pd.DataFrame(X, columns=['mu', 'sigma2', 'Psb', 'delta_phi', 'qber'])
    df['label'] = y
    
    os.makedirs('data', exist_ok=True)
    df.to_parquet('data/drift_dataset.parquet')
    print(f"Drift dataset saved to data/drift_dataset.parquet with shape {df.shape}")
    
def evaluate_fpr():
    df = pd.read_parquet('data/drift_dataset.parquet')
    X = df[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
    
    models = ['xgboost', 'randomforest', 'svm']
    
    print("\n--- False Positive Rate (FPR) Stress Test ---")
    print("Evaluating models on non-adversarial benign drift data.")
    
    for model_name in models:
        path = f'checkpoints/{model_name}.pkl'
        if os.path.exists(path):
            model = joblib.load(path)
            preds = model.predict(X)
            # FPR = FP / (FP + TN)
            # True labels are all 0 (Normal). So any prediction > 0 is a False Positive.
            fp = np.sum(preds > 0)
            tn = np.sum(preds == 0)
            fpr = fp / (fp + tn)
            print(f"{model_name.capitalize()} FPR on Drift Data: {fpr:.4f} ({fp} false alarms)")
        else:
            print(f"Model {model_name} not found in checkpoints.")

if __name__ == "__main__":
    generate_drift_dataset(5000)
    evaluate_fpr()
