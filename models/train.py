import os
import json
import yaml
import numpy as np
import dill as pickle
import torch
from models.features import extract_features
from models.deep_svdd import DeepSVDD
from models.baselines import StatisticalThreshold, QSVM, VQC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def load_and_preprocess_data(results_dir="results", window_size=1000):
    train_data = np.load(os.path.join(results_dir, "dataset_train.npz"))
    val_data = np.load(os.path.join(results_dir, "dataset_val.npz"))
    test_data = np.load(os.path.join(results_dir, "dataset_test.npz"))
    
    # Feature extraction with sliding window
    # 100,000 pulses window size, step of 1,000 to get a large number of stable windows
    window_size = 100000
    step = 1000
    
    X_train = extract_features(train_data['X'], window_size=window_size, step=step)
    
    def extract_labels(y, window_size, step):
        N = len(y)
        n_windows = (N - window_size) // step + 1
        labels = []
        for i in range(n_windows):
            window_labels = y[i*step : i*step + window_size]
            counts = np.bincount(window_labels.astype(int))
            labels.append(np.argmax(counts))
        return np.array(labels)

    y_train = extract_labels(train_data['y'], window_size, step)
    
    X_val = extract_features(val_data['X'], window_size=window_size, step=step)
    y_val = extract_labels(val_data['y'], window_size, step)
    
    X_test = extract_features(test_data['X'], window_size=window_size, step=step)
    y_test = extract_labels(test_data['y'], window_size, step)
    
    # Extract ZD subtypes
    zd_raw = np.load(os.path.join(results_dir, "zeroday_subtypes_test.npy"))
    def extract_zd_subtypes(zd, window_size, step):
        N = len(zd)
        n_windows = (N - window_size) // step + 1
        subs = []
        for i in range(n_windows):
            window_zd = zd[i*step : i*step + window_size]
            vals, counts = np.unique(window_zd, return_counts=True)
            subs.append(vals[np.argmax(counts)])
        return np.array(subs)
    
    zd_subtypes_test = extract_zd_subtypes(zd_raw, window_size, step)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, zd_subtypes_test

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dims", type=str, default="32,16", help="Comma separated list of hidden dimensions")
    args = parser.parse_args()
    
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    print("Loading data...")
    # Using window_size=100000, step=1000 to drastically reduce variance
    window_size = 100000 
    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_and_preprocess_data(window_size=window_size)
    print(f"Data loaded. X_train shape: {X_train.shape}")
    
    # Freeze splits
    splits_info = {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "window_size": window_size,
        "features_dim": X_train.shape[1]
    }
    with open("splits.json", "w") as f:
        json.dump(splits_info, f, indent=4)
        
    # Freeze hyperparams
    hyperparams = {
        "deep_svdd": {
            "input_dim": X_train.shape[1],
            "hidden_dims": hidden_dims,
            "rep_dim": 8,
            "lr": 1e-3,
            "epochs": 100,
            "batch_size": 128,
            "margin": 10.0
        },
        "statistical_threshold": {
            "threshold_std": 3.0
        },
        "qsvm": {
            "n_qubits": 10,
            "C": 1.0
        },
        "vqc": {
            "n_qubits": 10,
            "n_layers": 4,
            "epochs": 10,
            "lr": 0.1,
            "batch_size": 32
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": None
        }
    }
    with open("hyperparams.yaml", "w") as f:
        yaml.dump(hyperparams, f)
        
    # For supervised and semi-supervised models, we need both nominal and attack data. 
    # Since train is nominal only, we use train + val
    X_baseline_train = np.vstack([X_train, X_val])
    y_baseline_train = np.concatenate([y_train, y_val])

    # True OOD Generalization Test: Filter out Zero-Day attacks (label 3) from the training set
    mask = y_baseline_train != 3
    X_baseline_train = X_baseline_train[mask]
    y_baseline_train = y_baseline_train[mask]

    # --- Multi-Seed Training Pipeline ---
    N_SEEDS = 5
    os.makedirs("models/saved", exist_ok=True)
    
    for seed in range(1, N_SEEDS + 1):
        print(f"\n{'='*40}\nTraining Seed {seed}/{N_SEEDS}\n{'='*40}")
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"\nTraining Deep SAD (MLP {hidden_dims})...")
        svdd = DeepSVDD(**hyperparams["deep_svdd"])
        svdd.fit(X_baseline_train, y_baseline_train)
        
        # PyTorch model
        torch.save(svdd.net.state_dict(), f"models/saved/deep_svdd_seed{seed}.pth")
        with open(f"models/saved/deep_svdd_scaler_seed{seed}.pkl", "wb") as f:
            pickle.dump(svdd.scaler, f)
        
        print("\nTraining Baselines...")
        thresh = StatisticalThreshold(**hyperparams["statistical_threshold"])
        thresh.fit(X_baseline_train, y_baseline_train)
            
        print("Training Classical Baselines (XGBoost, RF)...")
        y_binary = (y_baseline_train != 0).astype(int)
        xgb_model = xgb.XGBClassifier(**hyperparams["xgboost"], use_label_encoder=False, eval_metric='logloss', random_state=seed)
        xgb_model.fit(X_baseline_train, y_binary)
        
        rf_model = RandomForestClassifier(**hyperparams["random_forest"], random_state=seed)
        rf_model.fit(X_baseline_train, y_binary)

        # Save models
        with open(f"models/saved/thresh_seed{seed}.pkl", "wb") as f:
            pickle.dump(thresh, f)
        with open(f"models/saved/xgboost_seed{seed}.pkl", "wb") as f:
            pickle.dump(xgb_model, f)
        with open(f"models/saved/rf_seed{seed}.pkl", "wb") as f:
            pickle.dump(rf_model, f)
            
    print("\nTraining complete and all seeded models saved.")

if __name__ == "__main__":
    main()
