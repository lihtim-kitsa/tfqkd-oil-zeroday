import numpy as np
import pandas as pd
import sys
import os
import multiprocessing as mp

# Add parent dir to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.oil_rate_equations import OILSimulator
from sim.attack_injectors import FIMInjector, TWIRLInjector
from data.feature_extraction import extract_features
from utils.seeds import get_seed

def run_simulation(attack_class='clean'):
    simulator = OILSimulator()
    t_span = (0, 1e-6)
    t_eval = np.linspace(0, 1e-6, 1000) # 1000 points
    y0 = [1.0, 0.0, 1e8] # Initial E, phi, N
    
    base_E_ref = 1.0
    base_phi_ref = 0.0
    
    def current_func(t): return 2.0e17
    
    if attack_class == 'fim':
        m = np.random.uniform(0.05, 0.40)
        f_fim = np.random.uniform(1e8, 1e10)
        injector = FIMInjector(m=m, f_fim=f_fim)
        def E_ref_func(t): return injector.inject_E_ref(t, base_E_ref)
        def phi_ref_func(t): return base_phi_ref
        
    elif attack_class == 'twirl':
        delta_lambda = np.random.uniform(0.1, 5.0)
        injector = TWIRLInjector(delta_lambda=delta_lambda)
        def E_ref_func(t): return base_E_ref
        def phi_ref_func(t): return injector.inject_phi_ref(t, base_phi_ref)
        
    else: # clean
        def E_ref_func(t): return base_E_ref
        def phi_ref_func(t): return base_phi_ref
        
    sol = simulator.simulate(t_span, y0, E_ref_func, phi_ref_func, current_func, t_eval)
    
    # Extract features
    features = extract_features(sol.t, sol.y[0], sol.y[1], sol.y[2])
    
    # We get multiple windows per simulation, take them all as samples
    # 1000 points / 100 window_size = 10 samples per run
    return features

def _worker(args):
    i, seed_base = args
    # Ensure different random state per run
    np.random.seed(seed_base + i)
    
    classes = ['clean', 'fim', 'twirl']
    label_name = classes[i % 3]
    label_idx = classes.index(label_name)
    try:
        feats = run_simulation(label_name)
        return feats, [label_idx] * len(feats)
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None, None

def generate_dataset(num_samples=40000):
    seed = get_seed("dataset_generation")
    np.random.seed(seed)
    
    cpu_cores = mp.cpu_count()
    print(f"Generating {num_samples} samples using {cpu_cores} CPU cores...")
    
    # Since each simulation run gives 10 windows (samples), we need num_samples / 10 runs
    runs_needed = num_samples // 10
    
    all_features = []
    all_labels = []
    
    # Use multiprocessing pool to parallelize the simulation runs
    with mp.Pool(processes=cpu_cores) as pool:
        args_iter = [(i, seed) for i in range(runs_needed)]
        
        for i, result in enumerate(pool.imap_unordered(_worker, args_iter)):
            if i % 100 == 0 and i > 0:
                print(f"Progress: {i}/{runs_needed} runs completed")
                
            feats, labels = result
            if feats is not None:
                all_features.append(feats)
                all_labels.extend(labels)
            
    # Flatten
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    # Trim to exact num_samples if we overshot
    X = X[:num_samples]
    y = y[:num_samples]
    
    df = pd.DataFrame(X, columns=['mu', 'sigma2', 'Psb', 'delta_phi', 'qber'])
    df['label'] = y
    
    os.makedirs('data', exist_ok=True)
    df.to_parquet('data/dataset_v1.parquet')
    print(f"Dataset saved to data/dataset_v1.parquet with shape {df.shape}")

if __name__ == "__main__":
    generate_dataset(40000)
