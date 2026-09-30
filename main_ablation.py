import os
import yaml
import numpy as np
import time
import pandas as pd
from sklearn.metrics import roc_auc_score
from models.features import extract_features
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def extract_labels(y, window_size, step):
    N = len(y)
    n_windows = (N - window_size) // step + 1
    labels = []
    for i in range(n_windows):
        window_labels = y[i*step : i*step + window_size]
        counts = np.bincount(window_labels.astype(int))
        labels.append(np.argmax(counts))
    return np.array(labels)

def main():
    print("Phase 5: Running Window-Size Ablations...")
    results_dir = "results"
    
    print("Loading raw NPZ files...")
    try:
        train_data = np.load(os.path.join(results_dir, "dataset_train.npz"))
        val_data = np.load(os.path.join(results_dir, "dataset_val.npz"))
        test_data = np.load(os.path.join(results_dir, "dataset_test.npz"))
    except FileNotFoundError:
        print("Dataset not found. Please wait for the dataset generation to finish.")
        return
    
    with open("hyperparams.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)
        
    window_sizes = [1000, 10000, 50000, 100000]
    step = 1000
    
    results = []
    
    for ws in window_sizes:
        print(f"\n--- Evaluating Window Size: {ws} ---")
        
        t0 = time.time()
        X_train = extract_features(train_data['X'], window_size=ws, step=step)
        y_train = extract_labels(train_data['y'], window_size=ws, step=step)
        
        X_val = extract_features(val_data['X'], window_size=ws, step=step)
        y_val = extract_labels(val_data['y'], window_size=ws, step=step)
        
        X_test = extract_features(test_data['X'], window_size=ws, step=step)
        y_test = extract_labels(test_data['y'], window_size=ws, step=step)
        feature_time = time.time() - t0
        print(f"Feature extraction took {feature_time:.2f}s")
        
        # Combine train + val for baseline training, masking out Zero-Day
        X_baseline_train = np.vstack([X_train, X_val])
        y_baseline_train = np.concatenate([y_train, y_val])
        mask = y_baseline_train != 3
        X_baseline_train = X_baseline_train[mask]
        y_baseline_train = y_baseline_train[mask]
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(**hyperparams["xgboost"], use_label_encoder=False, eval_metric='logloss', random_state=42)
        
        t0 = time.time()
        xgb_model.fit(X_baseline_train, y_baseline_train)
        train_time_xgb = time.time() - t0
        
        t0 = time.time()
        xgb_score = xgb_model.predict_proba(X_test)[:, 1]
        inf_time_xgb = time.time() - t0
        
        y_test_binary = (y_test != 0).astype(int)
        
        if len(np.unique(y_test_binary)) > 1:
            xgb_auroc = roc_auc_score(y_test_binary, xgb_score)
        else:
            xgb_auroc = 0.5
            
        results.append({
            "Window Size": ws,
            "XGB AUROC": xgb_auroc,
            "XGB Train Time (s)": train_time_xgb,
            "XGB Inference Time (s)": inf_time_xgb
        })
        
        print(f"XGB AUROC: {xgb_auroc:.4f}")
        
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/ablation_window_size.csv", index=False)
    with open("results/ablation_window_size.md", "w") as f:
        f.write("# Phase 5: Window-Size Ablation\n\n")
        f.write(df.to_markdown(index=False))
        
    print("\nAblation complete. Results saved to results/ablation_window_size.md")

if __name__ == "__main__":
    main()
