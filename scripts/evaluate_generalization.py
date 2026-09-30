import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.one_class import DeepSVDD


def evaluate_generalization():
    print("Loading FSK Zero-Day dataset...")
    df = pd.read_parquet('data/zero_day_dataset.parquet')
    X_zero_day = df[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values

    # Load scaler (fit on normal-only) and transform data
    scaler = joblib.load('checkpoints/scaler.pkl')
    X_zero_day = scaler.transform(X_zero_day)

    # ==================================================================
    # Evaluate XGBoost
    # ==================================================================
    print("\n--- Evaluating XGBoost ---")
    xgb_model = joblib.load('checkpoints/xgboost.pkl')
    preds = xgb_model.predict(X_zero_day)

    unique, counts = np.unique(preds, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Classified as Class {u}: {c} samples ({(c/len(preds))*100:.1f}%)")

    print("\nObservation: XGBoost misclassifies the unseen zero-day attack "
          "into existing categories.")
    if 0 in unique:
        n_evaded = counts[np.where(unique == 0)[0][0]]
        print(f"CRITICAL EVASION: {n_evaded} samples evaded detection as 'Clean'!")

    # ==================================================================
    # Evaluate Deep SVDD  (autoencoder reconstruction error)
    # ==================================================================
    print("\n--- Evaluating Deep SVDD ---")
    model = DeepSVDD()
    model.load_state_dict(torch.load('checkpoints/deep_svdd.pth', weights_only=True))
    model.eval()

    threshold = torch.load('checkpoints/deep_svdd_threshold.pth', weights_only=True).item()

    X_tensor = torch.tensor(X_zero_day, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
        distances = torch.sum((outputs - X_tensor) ** 2, dim=1)
        mean_dist = torch.mean(distances).item()
        preds_svdd = (distances > threshold).numpy().astype(int)

    svdd_detection_rate = preds_svdd.mean() * 100
    print(f"Mean Reconstruction Error: {mean_dist:.4f} (Threshold: {threshold:.4f})")
    print(f"Per-sample detection rate: {svdd_detection_rate:.1f}% "
          f"({preds_svdd.sum()}/{len(preds_svdd)} flagged as anomalous)")

    if mean_dist > threshold:
        print("SUCCESS: Deep SVDD robustly flagged the unseen FSK attack as an Anomaly!")
    else:
        print("FAILURE: Deep SVDD failed to distinguish the FSK attack from nominal data.")

    # ==================================================================
    # Unified Classifier Logic  (Class 3 = Zero-Day)
    # ==================================================================
    print("\n--- Unified Classifier Logic (Class 3: Zero-Day) ---")
    # Logic:
    #   XGBoost predicts 1 or 2  →  keep (known attack type)
    #   XGBoost predicts 0 AND DeepSVDD flags anomaly  →  Class 3 (Zero-Day)
    #   XGBoost predicts 0 AND DeepSVDD says normal   →  Class 0 (Nominal)
    #
    # Additionally, if DeepSVDD flags an anomaly for samples where XGBoost
    # says 1 or 2, it is still worth reporting — the attack was detected by
    # at least one model.

    final_preds = np.copy(preds)
    override_mask = (preds == 0) & (preds_svdd == 1)
    final_preds[override_mask] = 3

    unique_unified, counts_unified = np.unique(final_preds, return_counts=True)
    for u, c_count in zip(unique_unified, counts_unified):
        print(f"Unified Classifier Output Class {u}: {c_count} samples "
              f"({(c_count/len(final_preds))*100:.1f}%)")

    # Summary: total detection (either model flags it as non-normal)
    detected_by_either = ((preds != 0) | (preds_svdd == 1)).sum()
    print(f"\nTotal detected by either model: {detected_by_either}/{len(preds)} "
          f"({detected_by_either/len(preds)*100:.1f}%)")


if __name__ == "__main__":
    evaluate_generalization()
