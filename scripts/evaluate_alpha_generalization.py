"""Evaluate 100 km detector models across linewidth enhancement factor alpha."""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from models.deep_svdd import DeepSVDD
from models.features import extract_features
from simulator.attacks import FIM, TWIRL, ZERO_DAY_SUBTYPES, sample_attack_params
from simulator.dataset import _run_protocol_batch


ALPHAS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
PULSES_PER_BATCH = 100_000
BATCHES_PER_CLASS = 3
WINDOW_SIZE = 100_000
STEP = 1_000
N_SEEDS = 5


def labels_by_window(y):
    starts = np.arange(0, len(y) - WINDOW_SIZE + 1, STEP)
    return y[starts + WINDOW_SIZE // 2]


def tpr_at_fpr(y, score, max_fpr=0.04):
    fpr, tpr, _ = roc_curve(y, score)
    eligible = np.flatnonzero(fpr <= max_fpr)
    return float(tpr[eligible[-1]]) if len(eligible) else 0.0


def model_scores(X, X_train_nominal, hyperparams, seed):
    with open(f"models/saved/xgboost_seed{seed}.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    xgb_scores = xgb_model.predict_proba(X)[:, 1]

    svdd = DeepSVDD(**hyperparams["deep_svdd"])
    svdd.net.load_state_dict(torch.load(
        f"models/saved/deep_svdd_seed{seed}.pth",
        map_location=svdd.device,
        weights_only=True,
    ))
    with open(f"models/saved/deep_svdd_scaler_seed{seed}.pkl", "rb") as f:
        svdd.scaler = pickle.load(f)
    # Keep the one-class center and score covariance fixed from training data.
    # Re-estimating either from each target-alpha nominal set would leak target
    # distribution information into the cross-parameter generalization result.
    scaled_nom = svdd.scaler.transform(X_train_nominal)
    tensor_nom = torch.tensor(scaled_nom, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            tensor_nom, torch.zeros(len(tensor_nom), dtype=torch.float32)
        ),
        batch_size=hyperparams["deep_svdd"]["batch_size"],
        shuffle=False,
    )
    svdd.init_center_c(loader)
    svdd.net.eval()
    with torch.no_grad():
        z_nom = svdd.net(tensor_nom.to(svdd.device)).cpu().numpy()
        covariance_inverse = np.linalg.inv(
            np.cov(z_nom, rowvar=False) + np.eye(svdd.rep_dim) * 1e-4
        )
        center = svdd.c.cpu().numpy()
        tensor_eval = torch.tensor(
            svdd.scaler.transform(X), dtype=torch.float32
        ).to(svdd.device)
        z_eval = svdd.net(tensor_eval).cpu().numpy()
    delta = z_eval - center
    svdd_scores = np.einsum("ni,ij,nj->n", delta, covariance_inverse, delta)
    return {"deep_svdd": svdd_scores, "xgboost": xgb_scores}


def main():
    with open("hyperparams.yaml", "r", encoding="utf-8") as f:
        hyperparams = yaml.safe_load(f)
    rng = np.random.default_rng(69029)
    os.makedirs("results/alpha_generalization_data", exist_ok=True)
    train_archive = np.load("results/dataset_train.npz")
    X_train_nominal = extract_features(train_archive["X"], WINDOW_SIZE, STEP)
    rows = []

    for alpha in ALPHAS:
        print(f"\n--- alpha={alpha:g} ---", flush=True)
        feature_by_class = {}
        label_by_class = {}
        for class_name, mode in (("Nominal", 0), ("FIM", FIM), ("TWIRL", TWIRL), ("Zero-Day", None)):
            raw_batches = []
            for batch_id in range(BATCHES_PER_CLASS):
                actual_mode = (
                    int(rng.choice(ZERO_DAY_SUBTYPES)) if mode is None else mode
                )
                params = sample_attack_params(actual_mode, rng) if actual_mode else {}
                np.random.seed(int(rng.integers(0, 2**31)))
                cache_path = (
                    f"results/alpha_generalization_data/alpha_{alpha:g}_"
                    f"{class_name.lower().replace('-', '_')}_{batch_id}.npz"
                )
                if os.path.exists(cache_path):
                    with np.load(cache_path) as cached:
                        raw = cached["X"].copy()
                else:
                    raw = _run_protocol_batch(
                        100, PULSES_PER_BATCH, actual_mode, params, alpha=alpha
                    )
                    temp_path = cache_path + ".tmp.npz"
                    np.savez_compressed(temp_path, X=raw)
                    os.replace(temp_path, cache_path)
                raw_batches.append(raw)

            raw_class = np.concatenate(raw_batches, axis=0)
            feature_by_class[class_name] = extract_features(raw_class, WINDOW_SIZE, STEP)
            label_value = 0 if class_name == "Nominal" else 1
            label_by_class[class_name] = np.full(
                len(feature_by_class[class_name]), label_value, dtype=np.int8
            )

        X_nom = feature_by_class["Nominal"]
        for class_name in ("FIM", "TWIRL", "Zero-Day"):
            X_case = np.vstack([X_nom, feature_by_class[class_name]])
            y_case = np.concatenate([label_by_class["Nominal"], label_by_class[class_name]])
            seed_values = {m: {"auroc": [], "tpr": [], "ap": []}
                           for m in ("deep_svdd", "xgboost")}
            for seed in range(1, N_SEEDS + 1):
                scores = model_scores(X_case, X_train_nominal, hyperparams, seed)
                for model, score in scores.items():
                    seed_values[model]["auroc"].append(roc_auc_score(y_case, score))
                    seed_values[model]["tpr"].append(tpr_at_fpr(y_case, score))
                    seed_values[model]["ap"].append(average_precision_score(y_case, score))

            for model, metrics in seed_values.items():
                rows.append({
                    "alpha": alpha,
                    "attack": class_name,
                    "model": model,
                    "n_windows": len(y_case),
                    "positive_prevalence": float(np.mean(y_case)),
                    "auroc_mean": float(np.mean(metrics["auroc"])),
                    "auroc_std": float(np.std(metrics["auroc"])),
                    "tpr_at_4pct_fpr_mean": float(np.mean(metrics["tpr"])),
                    "tpr_at_4pct_fpr_std": float(np.std(metrics["tpr"])),
                    "average_precision_mean": float(np.mean(metrics["ap"])),
                    "average_precision_std": float(np.std(metrics["ap"])),
                })
                print(
                    f"{class_name:10s} {model:9s} AUROC "
                    f"{np.mean(metrics['auroc']):.3f} ± {np.std(metrics['auroc']):.3f}",
                    flush=True,
                )

    output = pd.DataFrame(rows)
    output.to_csv("results/alpha_generalization.csv", index=False)
    with open("results/alpha_generalization.md", "w", encoding="utf-8") as f:
        f.write("# Laser Linewidth Enhancement Factor Generalization\n\n")
        f.write(
            "Detectors were trained on the randomized 100 km dataset generated "
            "at alpha=3.0 and evaluated on independent simulations at each listed "
            "alpha. Each condition uses three 100,000-pulse batches per class; "
            "features use 100,000-pulse windows stepped by 1,000 (99% overlap). "
            "Metrics summarize five model seeds. Since overlapping windows are "
            "correlated, standard deviations describe seed variability and are not "
            "independent-window confidence intervals.\n\n"
        )
        f.write(output.to_markdown(index=False))
    print("Saved results/alpha_generalization.csv and .md", flush=True)


if __name__ == "__main__":
    main()
