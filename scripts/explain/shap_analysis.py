import os
import sys
import pandas as pd
import numpy as np
import shap
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_shap():
    if not os.path.exists('checkpoints/xgboost.pkl'):
        print("Model not found. Run train_models.py first.")
        return
        
    print("Loading XGBoost model for SHAP analysis...")
    model = joblib.load('checkpoints/xgboost.pkl')
    df = pd.read_parquet('data/dataset_v1.parquet')
    X = df[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
    feature_names = ['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    # Use a small sample to generate the alert fast
    shap_values = explainer.shap_values(X[:100])
    
    # Aggregate importance
    importance = np.abs(shap_values).mean(axis=0)
    if importance.ndim > 1: # Multiclass
        importance = importance.mean(axis=0)
        
    top_indices = np.argsort(importance)[::-1]
    
    print("\n--- SHAP Feature Attribution ---")
    for idx in top_indices:
        print(f"{feature_names[idx]}: {importance[idx]:.4f}")
        
    top_2 = [feature_names[top_indices[0]], feature_names[top_indices[1]]]
    
    print("\n--- Diagnostic Alert Generation ---")
    print(f"Tier 3 ALERT: Anomaly detected! Driven primarily by variations in {top_2[0]} and {top_2[1]}.")

if __name__ == "__main__":
    run_shap()
