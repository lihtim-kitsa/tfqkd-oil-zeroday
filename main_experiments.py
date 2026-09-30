import os
import json
import yaml
import time
import numpy as np
import dill as pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from models.train import load_and_preprocess_data
from models.deep_svdd import DeepSVDD

def get_tpr_at_fpr(y_true, y_score, target_fpr=0.04):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Find the index of the largest FPR that is <= target_fpr
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return tpr[idx[-1]]

def evaluate_model(model, X, y, model_name):
    # Subsample quantum models for faster evaluation
    if model_name in ["qsvm", "vqc"] and len(X) > 200:
        idx = np.random.choice(len(X), 200, replace=False)
        X_eval = X[idx]
        y_eval = y[idx]
    else:
        X_eval = X
        y_eval = y
        
    start_time = time.time()
    
    if model_name == "deep_svdd":
        y_score = model.score_samples(X_eval)
    elif model_name == "thresh":
        probs = model.predict_proba(X_eval)
        y_score = probs[:, 1]
    else:
        probs = model.predict_proba(X_eval)
        y_score = probs[:, 1]
        
    latency = (time.time() - start_time) / len(X_eval)
    
    auroc = roc_auc_score(y_eval, y_score)
    tpr_at_fpr = get_tpr_at_fpr(y_eval, y_score, target_fpr=0.04)
    
    return y_score, auroc, tpr_at_fpr, latency, y_eval

def main():
    print("Starting Phase 3 Evaluation...")
    window_size = 1000
    
    print("Loading data...")
    # Load all data to recreate DeepSVDD center and get test set
    X_train, y_train, _, _, X_test, y_test, _ = load_and_preprocess_data(window_size=window_size)
    print(f"Test set loaded: X={X_test.shape}, y={y_test.shape}")
    
    # Split test set by attack class
    test_sets = {}
    X_test_nom = X_test[y_test == 0]
    for attack_class, name in [(1, "FIM"), (2, "TWIRL"), (3, "Zero-Day")]:
        X_test_att = X_test[y_test == attack_class]
        y_nom = np.zeros(len(X_test_nom))
        y_att = np.ones(len(X_test_att))
        
        X_combined = np.vstack([X_test_nom, X_test_att])
        y_combined = np.concatenate([y_nom, y_att])
        
        test_sets[name] = (X_combined, y_combined)
        
    # Load hyperparams
    with open("hyperparams.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)
        
    models = {}
    print("Loading models...")
    for m in ["thresh", "xgboost", "rf"]:
        with open(f"models/saved/{m}.pkl", "rb") as f:
            models[m] = pickle.load(f)
            
    # Load and initialize Deep SVDD
    svdd = DeepSVDD(**hyperparams["deep_svdd"])
    svdd.net.load_state_dict(torch.load("models/saved/deep_svdd.pth"))
    with open("models/saved/deep_svdd_scaler.pkl", "rb") as f:
        svdd.scaler = pickle.load(f)
    # Reinitialize center c using train set (nominal data)
    print("Reinitializing Deep SVDD center c...")
    X_train_nom = X_train[y_train == 0]
    X_train_nom_scaled = svdd.scaler.transform(X_train_nom)
    tensor_X = torch.tensor(X_train_nom_scaled, dtype=torch.float32)
    tensor_y = torch.zeros(len(tensor_X), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hyperparams["deep_svdd"]["batch_size"], shuffle=False)
    svdd.init_center_c(train_loader)

    # Upgrade scoring: fit Mahalanobis covariance on nominal train embeddings
    print("Fitting Mahalanobis covariance on nominal embeddings...")
    svdd.net.eval()
    with torch.no_grad():
        nom_embeddings = svdd.net(tensor_X.to(svdd.device)).cpu().numpy()
    cov = np.cov(nom_embeddings, rowvar=False) + np.eye(svdd.rep_dim) * 1e-4
    cov_inv = np.linalg.inv(cov)
    c_np = svdd.c.cpu().numpy()

    def mahal_score(X, _net=svdd.net, _scaler=svdd.scaler, _c=c_np, _Sinv=cov_inv, _dev=svdd.device):
        _net.eval()
        X_sc = _scaler.transform(X)
        tx = torch.tensor(X_sc, dtype=torch.float32).to(_dev)
        with torch.no_grad():
            z = _net(tx).cpu().numpy()
        diff = z - _c
        return np.einsum('ni,ij,nj->n', diff, _Sinv, diff)

    svdd.score_samples = mahal_score
    models["deep_svdd"] = svdd
    
    results = []
    models.pop("qsvm", None)
    models.pop("vqc", None)
    
    
    # Pre-compute ROC curves to plot
    roc_data = {name: {} for name in test_sets.keys()}
    
    print("Evaluating models...")
    for attack_name, (X_eval, y_eval) in test_sets.items():
        print(f"  --> Attack: {attack_name}")
        for model_name, model in models.items():
            print(f"      Model: {model_name}")
            y_score, auroc, tpr_at_fpr, latency, y_eval_sub = evaluate_model(model, X_eval, y_eval, model_name)
            
            results.append({
                "Model": model_name,
                "Attack": attack_name,
                "AUROC": round(auroc, 4),
                "TPR@4%FPR": round(tpr_at_fpr, 4),
                "Latency (ms/pulse)": round(latency * 1000, 4)
            })
            
            fpr, tpr, _ = roc_curve(y_eval_sub, y_score)
            roc_data[attack_name][model_name] = (fpr, tpr)
            
    # Save ablation table
    df = pd.DataFrame(results)
    print("\nResults Table:")
    print(df)
    df.to_csv("results/ablation_table.csv", index=False)
    print("Saved results/ablation_table.csv")
    
    # Plot ROC curves
    print("Plotting ROC curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    attack_names = ["FIM", "TWIRL", "Zero-Day"]
    
    for i, attack_name in enumerate(attack_names):
        ax = axes[i]
        for model_name, (fpr, tpr) in roc_data[attack_name].items():
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={df[(df['Attack']==attack_name) & (df['Model']==model_name)]['AUROC'].values[0]:.3f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f"ROC - {attack_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        
    plt.tight_layout()
    plt.savefig("results/roc_curves.png")
    print("Saved results/roc_curves.png")

if __name__ == "__main__":
    main()
