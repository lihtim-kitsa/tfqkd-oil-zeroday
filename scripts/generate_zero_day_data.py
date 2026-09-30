import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.oil_rate_equations import OILSimulator
from sim.attack_injectors import FSKInjector
from data.feature_extraction import extract_features
from utils.seeds import get_seed

def run_fsk_simulation(idx):
    simulator = OILSimulator()
    t_span = (0, 1e-6)
    t_eval = np.linspace(0, 1e-6, 1000)
    y0 = [1.0, 0.0, 1e8]
    
    base_E_ref = 1.0
    base_phi_ref = 0.0
    
    def current_func(t): return 2.0e17
    
    f_mod = np.random.uniform(5e8, 5e9)
    max_detuning = np.random.uniform(1e9, 5e10)
    injector = FSKInjector(f_mod=f_mod, max_detuning=max_detuning)
    
    def E_ref_func(t): return base_E_ref
    def phi_ref_func(t): return injector.inject_phi_ref(t, base_phi_ref)
    
    sol = simulator.simulate(t_span, y0, E_ref_func, phi_ref_func, current_func, t_eval)
    feats = extract_features(sol.t, sol.y[0], sol.y[1], sol.y[2])
    return feats

def main():
    runs = 500
    print(f"Generating {runs} zero-day (FSK) simulation runs using multiprocessing...")
    
    all_features = []
    pool = mp.Pool(processes=mp.cpu_count())
    
    results = []
    for i in range(runs):
        results.append(pool.apply_async(run_fsk_simulation, args=(i,)))
        
    for i, res in enumerate(results):
        feats = res.get()
        all_features.append(feats)
        if (i+1) % 50 == 0:
            print(f"Progress: {i+1}/{runs} runs completed")
            
    pool.close()
    pool.join()
    
    # Flatten
    X = np.vstack(all_features)
    # Give it label 3 (unknown)
    y = np.full(len(X), 3)
    
    df = pd.DataFrame(X, columns=['mu', 'sigma2', 'Psb', 'delta_phi', 'qber'])
    df['label'] = y
    
    os.makedirs('data', exist_ok=True)
    df.to_parquet('data/zero_day_dataset.parquet')
    print(f"Zero-Day dataset saved to data/zero_day_dataset.parquet with shape {df.shape}")

if __name__ == '__main__':
    main()
