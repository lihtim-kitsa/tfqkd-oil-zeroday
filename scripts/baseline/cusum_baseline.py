import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def cusum_detect(series, drift=0.1, threshold=1.0):
    """Simple univariate CUSUM detector."""
    pos_cusum = np.zeros_like(series)
    neg_cusum = np.zeros_like(series)
    alerts = np.zeros_like(series)
    
    mean_val = np.mean(series)
    std_val = np.std(series) + 1e-9
    norm_series = (series - mean_val) / std_val
    
    for i in range(1, len(series)):
        pos_cusum[i] = max(0, pos_cusum[i-1] + norm_series[i] - drift)
        neg_cusum[i] = max(0, neg_cusum[i-1] - norm_series[i] - drift)
        
        if pos_cusum[i] > threshold or neg_cusum[i] > threshold:
            alerts[i] = 1
            pos_cusum[i] = 0
            neg_cusum[i] = 0
            
    return alerts

if __name__ == "__main__":
    print("Running Classical Baseline (CUSUM)...")
    df = pd.read_parquet('data/dataset_v1.parquet')
    
    # Run CUSUM on the most vulnerable feature to phase shifts
    qber_alerts = cusum_detect(df['qber'].values, drift=0.5, threshold=3.0)
    
    # Compare with actual labels (0: clean, 1: fim, 2: twirl)
    y_true = df['label'].values
    
    # Overall Accuracy
    actual_binary = (y_true > 0).astype(int)
    acc = accuracy_score(actual_binary, qber_alerts)
    
    print(f"CUSUM Baseline Accuracy on QBER: {acc:.4f}\n")
    
    # Reliability Check per Class
    print("--- CUSUM Reliability Analysis ---")
    
    # Class 0: Nominal (False Positive Check)
    mask_0 = (y_true == 0)
    fpr = np.mean(qber_alerts[mask_0])
    print(f"Nominal (Class 0) False Positive Rate: {fpr*100:.2f}%")
    
    # Class 1: FIM
    mask_1 = (y_true == 1)
    recall_1 = np.mean(qber_alerts[mask_1])
    print(f"FIM Attack (Class 1) Detection Rate: {recall_1*100:.2f}%")
    
    # Class 2: TWIRL
    mask_2 = (y_true == 2)
    recall_2 = np.mean(qber_alerts[mask_2])
    print(f"TWIRL Attack (Class 2) Detection Rate: {recall_2*100:.2f}%")
    
    print("\nConclusion: CUSUM is highly unreliable for TWIRL (stealth attack) because TWIRL induces minimal QBER disturbance, failing to trigger the threshold.")
