# Gradient-Based Adaptive Attacker (Surrogate Search)

The non-differentiable Lang-Kobayashi simulator was sampled at 48 Latin-hypercube FIM parameter settings. A small neural score surrogate was optimized with Adam to lower either Deep SVDD distance or the XGBoost attack probability, then each distinct candidate was rerun in the simulator. This is an exploratory adaptive attack: surrogate fit/generalization uncertainty is not yet a formal confidence bound, and the simulated key-rate estimator is asymptotic.

The simulator reruns did not show consistent improvement over the initial random design. The smallest Deep SVDD score among the validated optimized candidates was 4.80, while a Latin-hypercube design point scored 0.996. The smallest XGBoost attack probability among the optimized candidates was 0.0076 (from a candidate optimized against Deep SVDD); the best candidate optimized against XGBoost scored 0.026, compared with 0.010 in the initial design. Thus the fitted surrogate did not reliably find a better true-simulator blind spot than the starting design. This is an inconclusive negative result for this small surrogate/search budget, not evidence of robustness.

All results use the seed-1 detector artifacts. The loaded RobustScaler was serialized with scikit-learn 1.7.2 and loaded in 1.9.0, which emitted a compatibility warning; XGBoost also warned about loading serialized models across versions. Reproduction under the artifact-generation environment is needed before drawing strong conclusions. The search objective targeted detector scores only: it did not constrain key-rate impact or compare against a nominal key-rate reference. The listed key rates are the code's simplified asymptotic estimates, not finite-key security results.

## Surrogate-design samples

| parameter        |   lower_bound |   upper_bound |
|:-----------------|--------------:|--------------:|
| fim_mod_depth    |        0.05   |        0.5    |
| log_fim_f_mod_hz |       16.1181 |       18.8261 |
| fim_delta_f_hz   |        5e+06  |        1e+08  |

## Simulator-validated candidate points

| optimized_against   |   fim_mod_depth |   fim_f_mod_hz |   fim_delta_f_hz |   deep_svdd_score |   xgboost_attack_probability |   simulated_key_rate | surrogate_validation   |
|:--------------------|----------------:|---------------:|-----------------:|------------------:|-----------------------------:|---------------------:|:-----------------------|
| deep_svdd           |       0.365052  |    1.5e+08     |      1.41444e+07 |           4.79944 |                   0.00763482 |           0          | real simulator rerun   |
| deep_svdd           |       0.5       |    1.21375e+08 |      8.05521e+07 |        1525.51    |                   0.529602   |           0.00155929 | real simulator rerun   |
| xgboost             |       0.329787  |    1.5e+08     |      4.81938e+07 |          16.4258  |                   0.146891   |           0          | real simulator rerun   |
| xgboost             |       0.0989489 |    7.10983e+07 |      1e+08       |        6134.81    |                   0.960311   |           0.00440379 | real simulator rerun   |
| xgboost             |       0.5       |    1.06973e+08 |      7.2756e+07  |         144.172   |                   0.0262149  |           0.00433993 | real simulator rerun   |
| xgboost             |       0.0950826 |    1.63172e+07 |      1.30253e+07 |         231.131   |                   0.150858   |           0          | real simulator rerun   |
| xgboost             |       0.05      |    1.71039e+07 |      7.7431e+07  |          99.1735  |                   0.84431    |           0.00343426 | real simulator rerun   |
