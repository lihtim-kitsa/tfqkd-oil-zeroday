# Feature Importance: Nominal vs Pooled Zero-Day

Seed-1 XGBoost mean absolute SHAP values and Deep SVDD AUROC drop after independently permuting each feature. The latter is a sensitivity diagnostic, not a causal attribution. Scores use a 1,000-row sample of overlapping 100,000-pulse windows; rows are therefore not independent observations.

| feature            |   mean_abs_shap_xgboost |   deep_svdd_auroc_drop_permutation | evaluation                                            |
|:-------------------|------------------------:|-----------------------------------:|:------------------------------------------------------|
| phase_kurtosis     |                2.86812  |                         0.179892   | nominal vs pooled zero-day; seed 1; sampled 1000 rows |
| rms_phi            |                1.21273  |                         0.0225431  | nominal vs pooled zero-day; seed 1; sampled 1000 rows |
| phase_autocorr     |                1.11132  |                         0.0489958  | nominal vs pooled zero-day; seed 1; sampled 1000 rows |
| intensity_xcorr    |                0.571073 |                        -0.00477516 | nominal vs pooled zero-day; seed 1; sampled 1000 rows |
| var_I_B            |                0.431442 |                        -0.00671768 | nominal vs pooled zero-day; seed 1; sampled 1000 rows |
| transmittance      |                0.389374 |                         0.0056997  | nominal vs pooled zero-day; seed 1; sampled 1000 rows |
| click_rate         |                0.25209  |                         0.00159336 | nominal vs pooled zero-day; seed 1; sampled 1000 rows |
| click_asymmetry    |                0.204557 |                         0.00345228 | nominal vs pooled zero-day; seed 1; sampled 1000 rows |
| qber               |                0.197327 |                         0.00307361 | nominal vs pooled zero-day; seed 1; sampled 1000 rows |
| spectral_power_max |                0.050483 |                         0.00268019 | nominal vs pooled zero-day; seed 1; sampled 1000 rows |