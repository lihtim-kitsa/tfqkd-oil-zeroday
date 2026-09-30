import os
import json
import yaml
import torch
import numpy as np
import dill as pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from simulator.tf_qkd import TFQKDProtocol, estimate_decoy_parameters, calculate_key_rate
from simulator.attacks import FIM, attack_params_for_mode
from models.deep_svdd import DeepSVDD
from models.features import extract_features

def evaluate_attack(fim_mod_depth, fim_delta_f_hz, svdd, distance_km=100, n_pulses=100000):
    """
    Simulates the attack, extracts features, and computes the anomaly score and key rate.
    Uses 100 km distance as the Deep SVDD model was trained on 100 km nominal data.
    """
    proto = TFQKDProtocol(distance_km, n_pulses=n_pulses, use_oil=True)
    
    # Run the simulator
    result = proto.run(
        attack_mode=FIM,
        fim_mod_depth=fim_mod_depth,
        fim_f_mod_hz=50e6,      # Fix modulation frequency at 50 MHz
        fim_delta_f_hz=fim_delta_f_hz
    )
    
    # Calculate Key Rate
    Y1, e1, Qmm, Emm = estimate_decoy_parameters(result)
    key_rate = calculate_key_rate(Y1, e1, Qmm, Emm)
    
    # Extract features (requires raw array of shape (N, 7))
    X_raw = np.column_stack([
        result['phi_A'],
        result['phi_B'],
        result['I_A'],
        result['I_B'],
        (result['phi_A'] - result['phi_B']) % (2 * np.pi),
        result['clicks_D0'],
        result['clicks_D1']
    ])
    
    # Step=n_pulses, window_size=n_pulses -> 1 window for 100k pulses
    features = extract_features(X_raw, window_size=n_pulses, step=n_pulses)
    
    # Score features
    anomaly_scores = svdd.score_samples(features)
    score = np.mean(anomaly_scores)
    
    return score, key_rate

def objective(x, fim_mod_depth, svdd, nominal_score_threshold):
    """
    Objective function for Nelder-Mead.
    We want to minimize the anomaly score, but apply a massive penalty if the attack 
    fails to degrade the key rate (i.e. if key rate is still positive).
    x[0] is the frequency offset in MHz (scaled for optimization).
    """
    freq_offset_hz = x[0] * 1e6
    
    score, key_rate = evaluate_attack(fim_mod_depth, freq_offset_hz, svdd)
    
    # Penalty if key rate isn't dropped to 0
    penalty = 0
    if key_rate > 0:
        penalty = 1e6 * key_rate
        
    return score + penalty

def main():
    print("=" * 60)
    print("Adaptive Attacker Optimization (Nelder-Mead)")
    print("=" * 60)
    
    # 1. Load Hyperparams and Deep SVDD model
    with open("hyperparams.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)
        
    with open("splits.json", "r") as f:
        splits = json.load(f)
        
    svdd = DeepSVDD(**hyperparams["deep_svdd"])
    
    state_dict = torch.load("models/saved/deep_svdd.pth", map_location=svdd.device)
    svdd.net.load_state_dict(state_dict)
    
    with open("models/saved/deep_svdd_scaler.pkl", "rb") as f:
        svdd.scaler = pickle.load(f)
        
    print("Re-initializing Deep SVDD center 'c' from training data...")
    train_data = np.load("results/dataset_train.npz")
    X_train_features = extract_features(train_data['X'], window_size=splits["window_size"], step=splits["window_size"])
    X_train_scaled = svdd.scaler.transform(X_train_features)
    
    tensor_X = torch.tensor(X_train_scaled, dtype=torch.float32).to(svdd.device)
    svdd.net.eval()
    with torch.no_grad():
        outputs = svdd.net(tensor_X)
        c = torch.mean(outputs, dim=0)
        eps = 0.1
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c >= 0)] = eps
        svdd.c = c
        
    # Get nominal scores to set a threshold (e.g. 96th percentile for 4% FPR)
    nom_scores = svdd.score_samples(X_train_features)
    nominal_threshold = np.percentile(nom_scores, 96)
    print(f"Deep SVDD Nominal 4% FPR Threshold: {nominal_threshold:.4f}")
    
    # 2. Optimization Sweep (Grid Search for speed)
    mod_depths = [0.0, 0.1, 0.2, 0.3, 0.4]
    freq_offsets_mhz = [10.0, 30.0, 50.0] # Grid search over 3 offsets
    results = []
    
    for depth in mod_depths:
        print(f"\n--- Optimizing for FIM Mod Depth = {depth} ---")
        if depth == 0.0:
            score, kr = evaluate_attack(0.0, 0.0, svdd)
            print(f"Nominal (No Attack): Score = {score:.4f}, Key Rate = {kr:.4e}")
            results.append({'mod_depth': depth, 'opt_delta_f_mhz': 0.0, 'anomaly_score': float(score), 'key_rate': float(kr)})
            continue
            
        best_score = float('inf')
        best_freq = None
        best_kr = None
        
        for freq_mhz in freq_offsets_mhz:
            score, kr = evaluate_attack(depth, freq_mhz * 1e6, svdd)
            # We want to minimize score, subject to attack succeeding (kr == 0)
            # If key rate > 0, apply massive penalty
            penalty = 1e6 * kr if kr > 0 else 0
            obj_val = score + penalty
            
            if obj_val < best_score:
                best_score = obj_val
                best_freq = freq_mhz
                best_kr = kr
        
        print(f"Optimized Freq Offset: {best_freq:.2f} MHz")
        # best_score includes penalty, but we print actual score:
        # Since best_kr should be 0, actual score is best_score - penalty.
        actual_score = best_score - (1e6 * best_kr if best_kr > 0 else 0)
        print(f"Resulting Score: {actual_score:.4f} (Threshold: {nominal_threshold:.4f})")
        print(f"Resulting Key Rate: {best_kr:.4e}")
        
        results.append({
            'mod_depth': depth,
            'opt_delta_f_mhz': float(best_freq),
            'anomaly_score': float(actual_score),
            'key_rate': float(best_kr)
        })
        
    # 3. Save Results and Plot
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/adaptive_attacker_results.csv", index=False)
    print("\nResults saved to results/adaptive_attacker_results.csv")
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['mod_depth'], df['anomaly_score'], marker='o', linewidth=2, label='Adaptive FIM Score')
    plt.axhline(y=nominal_threshold, color='r', linestyle='--', label='4% FPR Threshold')
    
    plt.title('Deep SVDD Anomaly Score vs Attacker Adaptation Strength')
    plt.xlabel('FIM Modulation Depth (Adaptation Strength)')
    plt.ylabel('Deep SVDD Anomaly Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/adaptive_tradeoff.png", dpi=150)
    print("Tradeoff plot saved to results/adaptive_tradeoff.png")

if __name__ == "__main__":
    main()
