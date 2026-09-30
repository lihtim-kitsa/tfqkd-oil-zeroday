"""Evaluate 100 km models on the randomized 50 km and 150 km test sets."""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from models.deep_svdd import DeepSVDD
from models.features import extract_features
from simulator.attacks import ATTACK_NAMES, ZERO_DAY_SUBTYPES


WINDOW_SIZE = 100_000
STEP = 1_000
N_SEEDS = 5


def window_labels(values):
    """Use each window's center label; source traces are contiguous by batch."""
    starts = np.arange(0, len(values) - WINDOW_SIZE + 1, STEP)
    return values[starts + WINDOW_SIZE // 2]


def tpr_at_fpr(y_true, scores, target_fpr=0.04):
    fpr, tpr, _ = roc_curve(y_true, scores)
    eligible = np.flatnonzero(fpr <= target_fpr)
    return float(tpr[eligible[-1]]) if len(eligible) else 0.0


def evaluate_case(X, y_binary, train_nominal, hyperparams, seed, model_dir):
    """Return score vectors for both models for one seed and test case."""
    xgb_path = os.path.join(model_dir, f"xgboost_seed{seed}.pkl")
    with open(xgb_path, "rb") as stream:
        xgb_model = pickle.load(stream)
    xgb_scores = xgb_model.predict_proba(X)[:, 1]

    svdd = DeepSVDD(**hyperparams["deep_svdd"])
    weights_path = os.path.join(model_dir, f"deep_svdd_seed{seed}.pth")
    svdd.net.load_state_dict(torch.load(weights_path, map_location=svdd.device, weights_only=True))
    scaler_path = os.path.join(model_dir, f"deep_svdd_scaler_seed{seed}.pkl")
    with open(scaler_path, "rb") as stream:
        svdd.scaler = pickle.load(stream)

    scaled_nominal = svdd.scaler.transform(train_nominal)
    nominal_tensor = torch.tensor(scaled_nominal, dtype=torch.float32)
    labels = torch.zeros(len(nominal_tensor), dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(nominal_tensor, labels),
        batch_size=hyperparams["deep_svdd"]["batch_size"],
        shuffle=False,
    )
    svdd.init_center_c(loader)

    svdd.net.eval()
    with torch.no_grad():
        embeddings = svdd.net(nominal_tensor.to(svdd.device)).cpu().numpy()
        covariance = np.cov(embeddings, rowvar=False)
        covariance += np.eye(svdd.rep_dim) * 1e-4
        covariance_inverse = np.linalg.inv(covariance)
        center = svdd.c.cpu().numpy()

        scaled_test = svdd.scaler.transform(X)
        test_tensor = torch.tensor(scaled_test, dtype=torch.float32).to(svdd.device)
        test_embeddings = svdd.net(test_tensor).cpu().numpy()
    delta = test_embeddings - center
    deep_scores = np.einsum("ni,ij,nj->n", delta, covariance_inverse, delta)
    return {"deep_svdd": deep_scores, "xgboost": xgb_scores}


def main():
    results_dir = "results"
    model_dir = "models/saved"
    with open("hyperparams.yaml", "r", encoding="utf-8") as stream:
        hyperparams = yaml.safe_load(stream)

    train_archive = np.load(os.path.join(results_dir, "dataset_train.npz"))
    X_train = extract_features(train_archive["X"], WINDOW_SIZE, STEP)
    y_train = window_labels(train_archive["y"])
    train_nominal = X_train[y_train == 0]
    rows = []

    for distance in (50, 150):
        path = os.path.join(results_dir, f"dataset_test_{distance}km.npz")
        archive = np.load(path)
        X_all = extract_features(archive["X"], WINDOW_SIZE, STEP)
        y_all = window_labels(archive["y"])
        subtype_raw = np.load(os.path.join(
            results_dir, f"zeroday_subtypes_test_{distance}km.npy"
        ))
        subtype_all = window_labels(subtype_raw)
        nominal_mask = y_all == 0

        cases = [
            ("FIM", y_all == 1),
            ("TWIRL", y_all == 2),
            ("Zero-Day (All)", y_all == 3),
        ]
        for subtype in ZERO_DAY_SUBTYPES:
            cases.append((
                f"ZD: {ATTACK_NAMES[subtype]}",
                (y_all == 3) & (subtype_all == subtype),
            ))

        for case_name, attack_mask in cases:
            if not np.any(attack_mask):
                continue
            X_case = np.vstack([X_all[nominal_mask], X_all[attack_mask]])
            y_case = np.concatenate([
                np.zeros(np.sum(nominal_mask), dtype=np.int8),
                np.ones(np.sum(attack_mask), dtype=np.int8),
            ])
            per_model = {"deep_svdd": [], "xgboost": []}
            for seed in range(1, N_SEEDS + 1):
                score_sets = evaluate_case(
                    X_case, y_case, train_nominal, hyperparams, seed, model_dir
                )
                for model, scores in score_sets.items():
                    per_model[model].append({
                        "auroc": roc_auc_score(y_case, scores),
                        "tpr_at_4pct_fpr": tpr_at_fpr(y_case, scores),
                        "average_precision": average_precision_score(y_case, scores),
                    })

            for model, seed_metrics in per_model.items():
                row = {
                    "distance_km": distance,
                    "attack": case_name,
                    "model": model,
                    "n_windows": int(len(y_case)),
                    "positive_prevalence": float(np.mean(y_case)),
                }
                for metric in ("auroc", "tpr_at_4pct_fpr", "average_precision"):
                    values = [item[metric] for item in seed_metrics]
                    row[f"{metric}_mean"] = float(np.mean(values))
                    row[f"{metric}_std"] = float(np.std(values))
                rows.append(row)
                print(
                    f"{distance} km | {case_name:18s} | {model:9s} | "
                    f"AUROC={row['auroc_mean']:.3f}±{row['auroc_std']:.3f} | "
                    f"TPR@4%={row['tpr_at_4pct_fpr_mean']:.3f}±"
                    f"{row['tpr_at_4pct_fpr_std']:.3f}"
                )

    output = pd.DataFrame(rows)
    output.to_csv(os.path.join(results_dir, "cross_distance_ood.csv"), index=False)
    with open(os.path.join(results_dir, "cross_distance_ood.md"), "w", encoding="utf-8") as stream:
        stream.write("# Cross-Distance OOD Evaluation\n\n")
        stream.write(
            "Models were trained at 100 km and evaluated on randomized 50 km and "
            "150 km test sets. Each result is mean ± standard deviation across five "
            "model seeds. Features use 100,000-pulse windows with a 1,000-pulse "
            "step (99% overlap); windows are correlated, so these descriptive "
            "standard deviations are not independent-window confidence intervals.\n\n"
        )
        stream.write(output.to_markdown(index=False))
    print("Saved results/cross_distance_ood.csv and results/cross_distance_ood.md")


if __name__ == "__main__":
    main()
