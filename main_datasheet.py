"""
main_datasheet.py
=================
Phase 6: Generates a datasheet for the dataset, detailing statistics, class balance,
and parameter histograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def print_balance(y, name):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\n{name} split ({total:,} samples):")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c:,} ({(c/total)*100:.1f}%)")

def generate_datasheet(results_dir="results"):
    print("=== Dataset Datasheet ===")
    
    # 1. Class Balance & Counts
    for split in ["train", "val", "test", "test_50km", "test_150km"]:
        f = os.path.join(results_dir, f"dataset_{split}.npz")
        if os.path.exists(f):
            data = np.load(f)
            y = data["y"]
            print_balance(y, split.upper())
            
    # 2. Parameter Distribution Summaries
    print("\n=== Attack Strength Summaries (Test Split) ===")
    test_f = os.path.join(results_dir, "dataset_test.npz")
    if os.path.exists(test_f):
        data = np.load(test_f)
        s = data["strength"]
        y = data["y"]
        # FIM
        if len(s[y == 1]) > 0:
            s_fim = s[y == 1]
            print(f"FIM: Mean Strength = {np.mean(s_fim):.4f}, Std = {np.std(s_fim):.4f}, Min = {np.min(s_fim):.4f}, Max = {np.max(s_fim):.4f}")
        # TWIRL
        if len(s[y == 2]) > 0:
            s_twirl = s[y == 2]
            print(f"TWIRL: Mean Strength = {np.mean(s_twirl):.4f}, Std = {np.std(s_twirl):.4f}, Min = {np.min(s_twirl):.4f}, Max = {np.max(s_twirl):.4f}")
        # ZD
        if len(s[y == 3]) > 0:
            s_zd = s[y == 3]
            print(f"ZD: Mean Strength = {np.mean(s_zd):.4f}, Std = {np.std(s_zd):.4f}, Min = {np.min(s_zd):.4f}, Max = {np.max(s_zd):.4f}")

    print("\nDatasheet generation complete.")

if __name__ == "__main__":
    generate_datasheet()
