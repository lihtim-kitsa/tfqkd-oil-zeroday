# Research checklist status

Updated 2026-09-30.

| Work item | Status | Artifact / note |
|---|---|---|
| Randomized full dataset regeneration | Complete | `dataset_train.npz`, `dataset_val.npz`, `dataset_test.npz`; attack parameter metadata is stored with generated records. |
| Five-seed retraining and statistical validation | Complete | `statistical_validation.md`; fixed data split, so reported dispersion reflects model seeds only. |
| Cross-distance OOD evaluation | Complete | `cross_distance_ood.md` and `.csv`; models trained at 100 km, evaluated at 50 and 150 km. |
| SHAP and permutation feature importance | Complete | `feature_importance.md` and `.csv`. |
| Precision--recall curves and average precision | Complete | `precision_recall.md`, CSV tables, and figure. |
| Laser alpha generalization | Complete (exploratory) | `alpha_generalization.md` and `.csv`; fixed training reference, three simulation batches per class/alpha. Saved scaler artifacts warn of scikit-learn 1.7.2 vs current 1.9.0. |
| Gradient-based adaptive attacker | Complete (exploratory) | `adaptive_surrogate.md`, design/candidate CSVs, and figure. Seven candidates were simulator-validated; search did not consistently beat its initial design. Library-version warning applies. |
| Manuscript draft | Drafted | `../manuscript_draft.tex`; includes alpha and adaptive findings with limitations. Built-in LaTeX compilation could not run because its compiler reported that it could not find standard platform directories. |

All model evaluation uses heavily overlapping 100,000-pulse windows stepped by 1,000 pulses. Window-level metrics should not be interpreted as independent-sample confidence intervals. Results describe this simulator and do not establish hardware validity or a QKD security proof.
