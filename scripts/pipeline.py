import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import QuantileTransformer
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classical import get_xgb_model, get_rf_model, get_svm_model, get_logreg_model
from models.one_class import DeepSVDD
from scripts.stats.statistical_tests import bootstrap_metrics, run_mcnemar_tests


def train_and_eval_pipeline(data_path='data/dataset_v1.parquet', zero_day_path='data/zero_day_dataset.parquet', seed=42):
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('TFQKD_ML_Detection')

    with mlflow.start_run(run_name=f'Baseline_Pipeline_Seed_{seed}'):
        mlflow.log_param('random_seed', seed)
        mlflow.log_param('data_path', data_path)

        print("Loading main dataset...")
        df = pd.read_parquet(data_path)
        X = df[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
        y = df['label'].values

        print("Loading Zero-Day dataset...")
        df_zd = pd.read_parquet(zero_day_path)
        X_zd = df_zd[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
        y_zd = df_zd['label'].values  # Expected to be 3

        # Split Main Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        os.makedirs('checkpoints', exist_ok=True)

        # ------------------------------------------------------------------
        # Scaler: fit ONLY on normal (label-0) training rows.
        # ------------------------------------------------------------------
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train[y_train == 0])
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)
        X_zd = scaler.transform(X_zd)
        joblib.dump(scaler, 'checkpoints/scaler.pkl')

        models = {
            'LogReg': get_logreg_model(), # Baseline multinomial regression
            'XGBoost': get_xgb_model(),
            'RandomForest': get_rf_model()
        }

        models_preds = {}

        # Train Classical Base Models
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            joblib.dump(model, f'checkpoints/{name.lower()}.pkl')
            if name == 'XGBoost':
                mlflow.sklearn.log_model(
                    model, name.lower(),
                    skops_trusted_types=['xgboost.core.Booster',
                                         'xgboost.sklearn.XGBClassifier']
                )
            else:
                mlflow.sklearn.log_model(model, name.lower(), serialization_format="cloudpickle")

        # ==================================================================
        # Train DeepSVDD Autoencoder (One-Class, on normal data only)
        # ==================================================================
        print("\nTraining Deep SVDD (Zero-Day Detector)...")
        X_clean = torch.tensor(X_train[y_train == 0], dtype=torch.float32)
        deep_svdd = DeepSVDD()
        optimizer = optim.Adam(deep_svdd.parameters(), lr=1e-3, weight_decay=1e-6)

        deep_svdd.train()
        epochs = 500
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = deep_svdd(X_clean)
            loss = torch.mean(torch.sum((outputs - X_clean) ** 2, dim=1))
            loss.backward()
            optimizer.step()
            scheduler.step()

        mlflow.log_metric('deep_svdd_final_loss', loss.item())
        torch.save(deep_svdd.state_dict(), 'checkpoints/deep_svdd.pth')
        mlflow.pytorch.log_model(
            deep_svdd, 'deep_svdd',
            input_example=X_clean[:1].numpy(),
            serialization_format='pickle'
        )
        
        deep_svdd.eval()
        with torch.no_grad():
            clean_recon = deep_svdd(X_clean)
            clean_errors = torch.sum((clean_recon - X_clean) ** 2, dim=1).numpy()
            threshold = float(np.percentile(clean_errors, 95))
            torch.save(torch.tensor(threshold), 'checkpoints/deep_svdd_threshold.pth')
            print(f"Deep SVDD trained. Threshold: {threshold:.4f}")

        # ==================================================================
        # Combine test datasets (Main Test + Zero-Day) to evaluate unified logic
        # ==================================================================
        X_combined = np.vstack([X_test, X_zd])
        y_combined = np.concatenate([y_test, y_zd])
        
        X_combined_tensor = torch.tensor(X_combined, dtype=torch.float32)
        with torch.no_grad():
            test_recon = deep_svdd(X_combined_tensor)
            distances = torch.sum((test_recon - X_combined_tensor) ** 2, dim=1).numpy()
            preds_svdd = (distances > threshold).astype(int)

        # Store NN raw binary metrics before unified logic
        y_combined_binary = (y_combined > 0).astype(int) # Anything >0 is an attack
        
        print("\n--- Neural Network (Deep SVDD) Standalone Metrics ---")
        from sklearn.metrics import precision_score, recall_score, f1_score
        nn_p = precision_score(y_combined_binary, preds_svdd)
        nn_r = recall_score(y_combined_binary, preds_svdd)
        nn_f1 = f1_score(y_combined_binary, preds_svdd)
        print(f"Deep SVDD Binary Detection - Precision: {nn_p:.4f}, Recall: {nn_r:.4f}, F1: {nn_f1:.4f}")

        # Apply unified classification for each classical model
        print("\nEvaluating unified classifiers (Class 3 = Zero-Day) ...")
        for name, model in models.items():
            base_preds = model.predict(X_combined)
            
            # Unified Logic: Override to class 3 if base model says 0 but SVDD flags anomaly
            final_preds = np.copy(base_preds)
            override_mask = (base_preds == 0) & (preds_svdd == 1)
            final_preds[override_mask] = 3
            
            models_preds[f"Unified_{name}"] = final_preds

        # ==================================================================
        # Statistical Validation
        # ==================================================================
        print("\nRunning Bootstrap Metrics (Unified 4-Class Data)...")
        for name, preds in models_preds.items():
            metrics = bootstrap_metrics(y_combined, preds, random_seed=seed)

            for metric_name, (mean_val, lower, upper) in metrics.items():
                mlflow.log_metric(f'{name}_{metric_name}_mean', mean_val)
                mlflow.log_metric(f'{name}_{metric_name}_lower_95ci', lower)
                mlflow.log_metric(f'{name}_{metric_name}_upper_95ci', upper)
                if metric_name in ['accuracy', 'f1_macro']:
                    print(f"{name} - {metric_name}: {mean_val:.4f} "
                          f"(95% CI: [{lower:.4f}, {upper:.4f}])")

        print("\nRunning McNemar Tests with Holm-Bonferroni correction...")
        mcnemar_results = run_mcnemar_tests(models_preds, y_combined)
        for comp, res in mcnemar_results.items():
            print(f"Comparison: {comp}")
            print(f"  p_raw: {res['p_raw']:.4e}")
            print(f"  p_holm: {res['p_holm_corrected']:.4e}")
            print(f"  significant: {res['significant']}")
            print(f"  odds_ratio: {res['odds_ratio_effect_size']:.4f}")

            mlflow.log_metric(
                f'mcnemar_{comp.replace(" ", "_")}_p_holm',
                res['p_holm_corrected']
            )

        print("\nPipeline Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset_v1.parquet')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_and_eval_pipeline(args.data, seed=args.seed)
