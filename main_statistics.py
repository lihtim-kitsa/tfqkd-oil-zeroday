import os
import yaml
import pickle
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from scipy.stats import norm
import warnings
import shap

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from models.deep_svdd import DeepSVDD
from models.train import load_and_preprocess_data

def bootstrap_metrics(y_true, y_scores, n_bootstraps=1000, target_fpr=0.04):
    np.random.seed(42)
    bootstrapped_auroc = []
    bootstrapped_tpr = []
    
    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_auroc.append(score)
        
        fpr, tpr, thresholds = roc_curve(y_true[indices], y_scores[indices])
        idx = np.where(fpr <= target_fpr)[0]
        if len(idx) > 0:
            bootstrapped_tpr.append(tpr[idx[-1]])
        else:
            bootstrapped_tpr.append(0.0)
            
    return np.mean(bootstrapped_auroc), np.std(bootstrapped_auroc), np.mean(bootstrapped_tpr), np.std(bootstrapped_tpr)

def mcnemar_test(y_true, preds1, preds2):
    b = np.sum((preds1 != y_true) & (preds2 == y_true))
    c = np.sum((preds1 == y_true) & (preds2 != y_true))
    if b + c == 0:
        return 1.0 
    statistic = ((np.abs(b - c) - 1)**2) / (b + c)
    import scipy.stats
    return scipy.stats.chi2.sf(statistic, 1)

def holm_bonferroni(p_values):
    sorted_indices = np.argsort(p_values)
    m = len(p_values)
    alpha = 0.05
    significant = np.zeros(m, dtype=bool)
    for k, idx in enumerate(sorted_indices):
        adjusted_alpha = alpha / (m - k)
        if p_values[idx] <= adjusted_alpha:
            significant[idx] = True
        else:
            break
    return significant

def get_mahal_scorer(svdd, X_train_nom_scaled):
    svdd.net.eval()
    tensor_X = torch.tensor(X_train_nom_scaled, dtype=torch.float32)
    with torch.no_grad():
        nom_embeddings = svdd.net(tensor_X.to(svdd.device)).cpu().numpy()
    cov = np.cov(nom_embeddings, rowvar=False) + np.eye(svdd.rep_dim) * 1e-4
    cov_inv = np.linalg.inv(cov)
    c_np = svdd.c.cpu().numpy()

    def mahal_score(X):
        svdd.net.eval()
        X_sc = svdd.scaler.transform(X)
        tx = torch.tensor(X_sc, dtype=torch.float32).to(svdd.device)
        with torch.no_grad():
            z = svdd.net(tx).cpu().numpy()
        diff = z - c_np
        return np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
    return mahal_score

def permutation_importance_svdd(score_fn, X, y, metric='auroc'):
    baseline_score = roc_auc_score(y, score_fn(X))
    importances = []
    rng = np.random.default_rng(42)
    
    for i in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        perm_score = roc_auc_score(y, score_fn(X_perm))
        importances.append(baseline_score - perm_score)
    return np.array(importances)

def main():
    print("Starting Comprehensive Statistical Validation (Phases 2-4)...")
    print("Loading data...")
    # Matches training window size
    window_size = 100000
    X_train, y_train, X_val, y_val, X_test, y_test, zd_subtypes_test = load_and_preprocess_data(window_size=window_size)
    
    with open("hyperparams.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)
        
    N_SEEDS = 5
    
    # Define test sets
    # We want to test on Nominal (0), FIM (1), TWIRL (2), and Zero-Day (3)
    test_cases = [
        ("FIM", y_test == 1),
        ("TWIRL", y_test == 2),
        ("Zero-Day (All)", y_test == 3)
    ]
    print(f"Running evaluation across {N_SEEDS} seeds...")
    
    # ── Phase 3: Zero-Day Subtype Mapping ──────────────────────────
    from simulator.attacks import ATTACK_NAMES, ZERO_DAY_SUBTYPES
    
    # We will test all attacks, plus each individual ZD subtype
    for subtype in ZERO_DAY_SUBTYPES:
        mask = (y_test == 3) & (zd_subtypes_test == subtype)
        if np.sum(mask) > 0:
            test_cases.append((f"ZD: {ATTACK_NAMES[subtype]}", mask))
            
    # Nominal mask
    nom_mask = (y_test == 0)
    
    results = []
    mcnemar_comparisons = []
    feature_importance_rows = []
    pr_summary_rows = []
    pr_curve_rows = []
    pr_plot_data = {}
    
    # Prepare feature names for Phase 4
    feature_names = [
        "transmittance", "var_I_B", "rms_phi", "click_rate", "qber",
        "phase_autocorr", "intensity_xcorr", "phase_kurtosis", "click_asymmetry",
        "spectral_power_max"
    ]
    
    # Store aggregated scores for each model/case across all seeds
    all_seed_scores = {case: {m: [] for m in ["deep_svdd", "xgboost"]} for case, _ in test_cases}
    
    for seed in range(1, N_SEEDS + 1):
        print(f"\n--- Evaluating Seed {seed} ---")
        try:
            # Load models
            svdd_params = hyperparams["deep_svdd"].copy()
            svdd = DeepSVDD(**svdd_params)
            svdd.net.load_state_dict(torch.load(f"models/saved/deep_svdd_seed{seed}.pth"))
            with open(f"models/saved/deep_svdd_scaler_seed{seed}.pkl", "rb") as f:
                svdd.scaler = pickle.load(f)
                
            X_train_nom_scaled = svdd.scaler.transform(X_train[y_train == 0])
            tensor_X = torch.tensor(X_train_nom_scaled, dtype=torch.float32)
            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_X, torch.zeros(len(tensor_X))), batch_size=128, shuffle=False)
            svdd.init_center_c(train_loader)
            svdd.score_samples = get_mahal_scorer(svdd, X_train_nom_scaled)
            
            with open(f"models/saved/xgboost_seed{seed}.pkl", "rb") as f:
                xgb_model = pickle.load(f)
                
        except FileNotFoundError:
            print(f"Warning: Models for seed {seed} not found. Skipping.")
            continue
            
        for case_name, att_mask in test_cases:
            if not np.any(att_mask):
                continue
                
            y_case = np.concatenate([np.zeros(np.sum(nom_mask)), np.ones(np.sum(att_mask))])
            X_case = np.vstack([X_test[nom_mask], X_test[att_mask]])
            
            # Scores
            svdd_score = svdd.score_samples(X_case)
            xgb_score = xgb_model.predict_proba(X_case)[:, 1]
            
            all_seed_scores[case_name]["deep_svdd"].append((y_case, svdd_score))
            all_seed_scores[case_name]["xgboost"].append((y_case, xgb_score))

        # Phase 4: Feature Importance (run only on seed 1 to save time)
        if seed == 1:
            print("\nComputing Feature Importances on Seed 1 (Zero-Day All)...")
            zd_mask = y_test == 3
            X_zd = X_test[zd_mask]
            zd_subtypes = zd_subtypes_test[zd_mask]
            
            X_zd = np.vstack([X_test[nom_mask], X_test[y_test == 3]])
            y_zd = np.concatenate([np.zeros(np.sum(nom_mask)), np.ones(np.sum(y_test == 3))])
            
            print("\nComputing Feature Importances on Seed 1 (Zero-Day All)...\n")
            
            # XGBoost SHAP
            explainer = shap.TreeExplainer(xgb_model)
            # Sample to avoid massive computation
            idx = np.random.default_rng(42).choice(
                len(X_zd), min(1000, len(X_zd)), replace=False
            )
            shap_values = explainer.shap_values(X_zd[idx])
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # positive class
            elif len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Deep SAD Permutation
            perm_importances = permutation_importance_svdd(svdd.score_samples, X_zd[idx], y_zd[idx])

            for fname, shap_value, perm_value in zip(
                feature_names, mean_abs_shap, perm_importances
            ):
                feature_importance_rows.append({
                    "feature": fname,
                    "mean_abs_shap_xgboost": float(shap_value),
                    "deep_svdd_auroc_drop_permutation": float(perm_value),
                    "evaluation": "nominal vs pooled zero-day; seed 1; sampled 1000 rows",
                })
            
            print("Feature Importance Rankings:")
            for i, fname in enumerate(feature_names):
                print(f"{fname:18s} | XGB SHAP: {mean_abs_shap[i]:.4f} | SVDD Perm Drop: {perm_importances[i]:.4f}")
            
    # ── Phase 2 Aggregation & McNemar ──────────────────────────────
    print("\nAggregating metrics across all seeds...")
    
    for case_name, _ in test_cases:
        case_results = {"Attack": case_name}
        
        for model in ["deep_svdd", "xgboost"]:
            aurocs = []
            tprs = []
            ensemble_scores = None
            ensemble_y = None
            
            for y_true, y_score in all_seed_scores[case_name][model]:
                aurocs.append(roc_auc_score(y_true, y_score))
                
                fpr, tpr, thresh = roc_curve(y_true, y_score)
                idx = np.where(fpr <= 0.04)[0]
                tprs.append(tpr[idx[-1]] if len(idx) > 0 else 0.0)
                
                if ensemble_scores is None:
                    ensemble_scores = y_score / N_SEEDS
                    ensemble_y = y_true
                else:
                    ensemble_scores += y_score / N_SEEDS
                    
            if len(aurocs) == 0:
                continue
                
            case_results[f"{model}_AUROC"] = f"{np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}"
            case_results[f"{model}_TPR@4%"] = f"{np.mean(tprs):.4f} ± {np.std(tprs):.4f}"

            precision, recall, pr_thresholds = precision_recall_curve(
                ensemble_y, ensemble_scores
            )
            average_precision = average_precision_score(
                ensemble_y, ensemble_scores
            )
            prevalence = float(np.mean(ensemble_y))
            pr_summary_rows.append({
                "attack": case_name,
                "model": model,
                "average_precision": float(average_precision),
                "positive_prevalence_baseline": prevalence,
                "n_windows": int(len(ensemble_y)),
                "n_seeds": int(len(all_seed_scores[case_name][model])),
            })
            for point, (p, r) in enumerate(zip(precision, recall)):
                pr_curve_rows.append({
                    "attack": case_name,
                    "model": model,
                    "point": point,
                    "precision": float(p),
                    "recall": float(r),
                })
            if case_name in {"FIM", "TWIRL", "Zero-Day (All)"}:
                pr_plot_data[(case_name, model)] = (
                    precision, recall, average_precision, prevalence
                )
            
            # Prepare binary predictions for McNemar using the ensemble
            fpr, tpr, thresh = roc_curve(ensemble_y, ensemble_scores)
            idx = np.where(fpr <= 0.04)[0]
            decision_thresh = thresh[idx[-1]] if len(idx) > 0 else thresh[0]
            all_seed_scores[case_name][f"{model}_ensemble_preds"] = (ensemble_scores >= decision_thresh).astype(int)
            all_seed_scores[case_name]["ensemble_y"] = ensemble_y
            
        if "deep_svdd_AUROC" in case_results:
            results.append(case_results)
            
            p_val = mcnemar_test(
                all_seed_scores[case_name]["ensemble_y"], 
                all_seed_scores[case_name]["xgboost_ensemble_preds"], 
                all_seed_scores[case_name]["deep_svdd_ensemble_preds"]
            )
            mcnemar_comparisons.append({
                "Attack": case_name,
                "p-value": p_val
            })
            
    print("\nApplying Holm-Bonferroni correction to McNemar p-values...")
    if len(mcnemar_comparisons) > 0:
        significance = holm_bonferroni([comp["p-value"] for comp in mcnemar_comparisons])
        for i, comp in enumerate(mcnemar_comparisons):
            comp["Significant (alpha=0.05)"] = significance[i]
            
    # Save as Markdown
    os.makedirs("results", exist_ok=True)
    if pr_summary_rows:
        pd.DataFrame(pr_summary_rows).to_csv(
            "results/precision_recall_summary.csv", index=False
        )
        pd.DataFrame(pr_curve_rows).to_csv(
            "results/precision_recall_curves.csv", index=False
        )
        plot_cases = ["FIM", "TWIRL", "Zero-Day (All)"]
        fig, axes = plt.subplots(1, len(plot_cases), figsize=(16, 5), sharey=True)
        for ax, case_name in zip(axes, plot_cases):
            for model in ["deep_svdd", "xgboost"]:
                curve = pr_plot_data.get((case_name, model))
                if curve is None:
                    continue
                precision, recall, ap, prevalence = curve
                ax.plot(recall, precision, label=f"{model} (AP={ap:.3f})")
            if curve is not None:
                ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1,
                           label=f"Prevalence={prevalence:.2f}")
            ax.set_title(case_name)
            ax.set_xlabel("Recall")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            ax.legend(loc="lower left", fontsize=8)
        axes[0].set_ylabel("Precision")
        fig.suptitle("Precision–Recall Curves (five-seed mean scores)")
        fig.tight_layout()
        fig.savefig("results/precision_recall_curves.png", dpi=180)
        plt.close(fig)

        with open("results/precision_recall.md", "w") as f:
            f.write("# Precision–Recall Results\n\n")
            f.write(
                "Average precision is reported with the positive-class prevalence "
                "as its no-skill baseline. Test windows overlap by 99% (100,000-pulse "
                "windows stepped by 1,000), so window-level points are correlated; "
                "interpret these as descriptive curves, not independent-sample CIs.\n\n"
            )
            f.write(pd.DataFrame(pr_summary_rows).to_markdown(index=False))

    with open("results/statistical_validation.md", "w") as f:
        f.write("# Phase 2 & 3: Statistical Validation & Subtype Breakdown\n\n")
        f.write("## Multi-Seed Performance (Mean ± Std over 5 seeds)\n\n")
        df_stats = pd.DataFrame(results)
        if not df_stats.empty:
            f.write(df_stats.to_markdown(index=False))
        else:
            f.write("No seeded models found. Train models first using `models/train.py`.\n")
            
        f.write("\n\n## McNemar's Test (XGBoost vs Deep SAD Ensemble)\n\n")
        df_mcnemar = pd.DataFrame(mcnemar_comparisons)
        if not df_mcnemar.empty:
            f.write(df_mcnemar.to_markdown(index=False))

    if feature_importance_rows:
        importance = pd.DataFrame(feature_importance_rows).sort_values(
            "mean_abs_shap_xgboost", ascending=False
        )
        importance.to_csv("results/feature_importance.csv", index=False)
        with open("results/feature_importance.md", "w") as f:
            f.write("# Feature Importance: Nominal vs Pooled Zero-Day\n\n")
            f.write(
                "Seed-1 XGBoost mean absolute SHAP values and Deep SVDD AUROC "
                "drop after independently permuting each feature. The latter "
                "is a sensitivity diagnostic, not a causal attribution. "
                "Scores use a 1,000-row sample of overlapping 100,000-pulse "
                "windows; rows are therefore not independent observations.\n\n"
            )
            f.write(importance.to_markdown(index=False))
        print("Feature importance saved to results/feature_importance.csv and .md")
            
    print("\nDone! Results saved to results/statistical_validation.md")

if __name__ == "__main__":
    main()
