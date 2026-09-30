import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classical import get_xgb_model, get_rf_model, get_svm_model
from scripts.stats.statistical_tests import bootstrap_metrics

def train_hardened_pipeline(seed=42):
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('TFQKD_ML_Detection')
    
    with mlflow.start_run(run_name=f'Hardened_Pipeline_Seed_{seed}'):
        mlflow.log_param('random_seed', seed)
        mlflow.log_param('dataset', 'dataset_v1 + drift_dataset')
        
        print("Loading base dataset...")
        df_base = pd.read_parquet('data/dataset_v1.parquet')
        
        print("Loading drift dataset...")
        df_drift = pd.read_parquet('data/drift_dataset.parquet')
        
        # Combine datasets to create a hardened training set
        df_combined = pd.concat([df_base, df_drift], ignore_index=True)
        
        X = df_combined[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
        y = df_combined['label'].values
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        models = {
            'XGBoost_Hardened': get_xgb_model(),
            'RandomForest_Hardened': get_rf_model(),
            'SVM_Hardened': get_svm_model()
        }
        
        models_preds = {}
        
        # Train Classical
        for name, model in models.items():
            print(f"Training {name} on combined data...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            models_preds[name] = preds
            
            joblib.dump(model, f'checkpoints/{name.lower()}.pkl')
            if 'XGBoost' in name:
                mlflow.sklearn.log_model(model, name.lower(), skops_trusted_types=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier'])
            else:
                mlflow.sklearn.log_model(model, name.lower())
                
        # Evaluate on Drift Specifically
        print("\nEvaluating Hardened Models on purely Drift Data (FPR check)...")
        X_drift_only = df_drift[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
        
        for name, model in models.items():
            drift_preds = model.predict(X_drift_only)
            fp = np.sum(drift_preds > 0)
            tn = np.sum(drift_preds == 0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            print(f"{name} FPR on Drift Data: {fpr:.4f} ({fp} false alarms)")
            mlflow.log_metric(f'{name}_drift_fpr', fpr)
            
        print("\nHardened Training Pipeline Complete!")

if __name__ == "__main__":
    train_hardened_pipeline()
