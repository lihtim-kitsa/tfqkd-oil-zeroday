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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classical import get_xgb_model, get_rf_model, get_svm_model
from models.one_class import DeepSVDD


def train_classical():
    """
    Train XGBoost and RandomForest on the nominal and known attack data.
    """
    print("Loading dataset...")
    df = pd.read_parquet('data/dataset_v1.parquet')

    X = df[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs('checkpoints', exist_ok=True)

    # ------------------------------------------------------------------
    # Scaler: fit ONLY on normal (label-0) training rows.
    # This is critical — the autoencoder must see the normal-only
    # quantile mapping so that attack / zero-day data maps to
    # out-of-distribution regions and produces high reconstruction error.
    # ------------------------------------------------------------------
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    scaler.fit(X_train[y_train == 0])
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, 'checkpoints/scaler.pkl')

    from models.classical import get_xgb_model, get_rf_model, get_svm_model, get_logreg_model
    models = {
        'LogReg': get_logreg_model(),
        'XGBoost': get_xgb_model(),
        'RandomForest': get_rf_model(),
        'SVM': get_svm_model()
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))
        joblib.dump(model, f'checkpoints/{name.lower()}.pkl')

    return X_train, y_train


def train_deep_svdd(X_train, y_train):
    """
    Train the Deep SVDD autoencoder on clean (label-0) data only.
    Uses reconstruction error as the anomaly score.
    """
    print("\nTraining Deep SVDD (Zero-Day Detector)...")
    X_clean = torch.tensor(X_train[y_train == 0], dtype=torch.float32)

    model = DeepSVDD()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    epochs = 500
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_clean)

        # MSE reconstruction loss
        loss = torch.mean(torch.sum((outputs - X_clean) ** 2, dim=1))

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {loss.item():.6f}")

    # ------------------------------------------------------------------
    # Compute threshold on the training-set normal data
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        recon = model(X_clean)
        errors = torch.sum((recon - X_clean) ** 2, dim=1).numpy()
    threshold = float(np.percentile(errors, 95))

    print(f"Deep SVDD trained. Final Loss: {loss.item():.4f}")
    print(f"  95th-percentile threshold on training normals: {threshold:.6f}")
    torch.save(model.state_dict(), 'checkpoints/deep_svdd.pth')
    torch.save(torch.tensor(threshold), 'checkpoints/deep_svdd_threshold.pth')


if __name__ == "__main__":
    X_train, y_train = train_classical()
    train_deep_svdd(X_train, y_train)
    print("\nTraining pipeline complete! (Quantum models deferred to dedicated scripts/GPU)")
