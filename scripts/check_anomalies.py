"""Diagnostic: compare the feature distributions of Normal vs FSK zero-day data
after scaling, to see whether they are separable at all."""
import pandas as pd, numpy as np, joblib

df_train = pd.read_parquet('data/dataset_v1.parquet')
df_zd    = pd.read_parquet('data/zero_day_dataset.parquet')

cols = ['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']

X_normal = df_train[df_train['label'] == 0][cols].values
X_attack1 = df_train[df_train['label'] == 1][cols].values
X_attack2 = df_train[df_train['label'] == 2][cols].values
X_zd     = df_zd[cols].values

from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer(output_distribution='normal', random_state=42)
scaler.fit(X_normal)   # fit only on normal

X_n_s  = scaler.transform(X_normal)
X_a1_s = scaler.transform(X_attack1)
X_a2_s = scaler.transform(X_attack2)
X_zd_s = scaler.transform(X_zd)

print("=== RAW feature ranges ===")
for i, c in enumerate(cols):
    print(f"\n--- {c} ---")
    print(f"  Normal   : mean={X_normal[:,i].mean():.4f}  std={X_normal[:,i].std():.4f}  min={X_normal[:,i].min():.4f}  max={X_normal[:,i].max():.4f}")
    print(f"  Attack-1 : mean={X_attack1[:,i].mean():.4f}  std={X_attack1[:,i].std():.4f}  min={X_attack1[:,i].min():.4f}  max={X_attack1[:,i].max():.4f}")
    print(f"  Attack-2 : mean={X_attack2[:,i].mean():.4f}  std={X_attack2[:,i].std():.4f}  min={X_attack2[:,i].min():.4f}  max={X_attack2[:,i].max():.4f}")
    print(f"  FSK(ZD)  : mean={X_zd[:,i].mean():.4f}  std={X_zd[:,i].std():.4f}  min={X_zd[:,i].min():.4f}  max={X_zd[:,i].max():.4f}")

print("\n=== SCALED feature ranges (QuantileTransformer fit on Normal only) ===")
for i, c in enumerate(cols):
    print(f"\n--- {c} ---")
    print(f"  Normal   : mean={X_n_s[:,i].mean():.4f}  std={X_n_s[:,i].std():.4f}  min={X_n_s[:,i].min():.4f}  max={X_n_s[:,i].max():.4f}")
    print(f"  Attack-1 : mean={X_a1_s[:,i].mean():.4f}  std={X_a1_s[:,i].std():.4f}  min={X_a1_s[:,i].min():.4f}  max={X_a1_s[:,i].max():.4f}")
    print(f"  Attack-2 : mean={X_a2_s[:,i].mean():.4f}  std={X_a2_s[:,i].std():.4f}  min={X_a2_s[:,i].min():.4f}  max={X_a2_s[:,i].max():.4f}")
    print(f"  FSK(ZD)  : mean={X_zd_s[:,i].mean():.4f}  std={X_zd_s[:,i].std():.4f}  min={X_zd_s[:,i].min():.4f}  max={X_zd_s[:,i].max():.4f}")

# Also check if we can train a simple logistic regression to distinguish Normal from FSK
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
Xc = np.vstack([X_n_s, X_zd_s])
yc = np.array([0]*len(X_n_s) + [1]*len(X_zd_s))
lr = LogisticRegression(max_iter=1000)
lr.fit(Xc, yc)
print(f"\nLogistic Regression (Normal vs FSK, on scaled data): accuracy = {lr.score(Xc, yc):.4f}")
print(f"  Coefficients: {dict(zip(cols, lr.coef_[0]))}")
