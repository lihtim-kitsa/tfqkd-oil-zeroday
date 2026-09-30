# Precision–Recall Results

Average precision is reported with the positive-class prevalence as its no-skill baseline. Test windows overlap by 99% (100,000-pulse windows stepped by 1,000), so window-level points are correlated; interpret these as descriptive curves, not independent-sample CIs.

| attack         | model     |   average_precision |   positive_prevalence_baseline |   n_windows |   n_seeds |
|:---------------|:----------|--------------------:|-------------------------------:|------------:|----------:|
| FIM            | deep_svdd |           0.963778  |                      0.714082  |        3501 |         5 |
| FIM            | xgboost   |           0.958138  |                      0.714082  |        3501 |         5 |
| TWIRL          | deep_svdd |           0.815519  |                      0.714082  |        3501 |         5 |
| TWIRL          | xgboost   |           0.864743  |                      0.714082  |        3501 |         5 |
| Zero-Day (All) | deep_svdd |           0.974964  |                      0.709939  |        3451 |         5 |
| Zero-Day (All) | xgboost   |           0.977994  |                      0.709939  |        3451 |         5 |
| ZD: PNI (ZD-A) | deep_svdd |           0.959652  |                      0.28551   |        1401 |         5 |
| ZD: PNI (ZD-A) | xgboost   |           0.967003  |                      0.28551   |        1401 |         5 |
| ZD: CDA (ZD-B) | deep_svdd |           0.0688039 |                      0.0908265 |        1101 |         5 |
| ZD: CDA (ZD-B) | xgboost   |           0.0629866 |                      0.0908265 |        1101 |         5 |
| ZD: DSI (ZD-C) | deep_svdd |           0.746891  |                      0.166528  |        1201 |         5 |
| ZD: DSI (ZD-C) | xgboost   |           0.794966  |                      0.166528  |        1201 |         5 |
| ZD: CFI (ZD-E) | deep_svdd |           0.623435  |                      0.19984   |        1251 |         5 |
| ZD: CFI (ZD-E) | xgboost   |           0.738841  |                      0.19984   |        1251 |         5 |
| ZD: CAM (ZD-F) | deep_svdd |           0.967675  |                      0.374766  |        1601 |         5 |
| ZD: CAM (ZD-F) | xgboost   |           0.990118  |                      0.374766  |        1601 |         5 |
| ZD: DPJ (ZD-G) | deep_svdd |           0.69093   |                      0.0908265 |        1101 |         5 |
| ZD: DPJ (ZD-G) | xgboost   |           0.745389  |                      0.0908265 |        1101 |         5 |
| ZD: SAM (ZD-H) | deep_svdd |           0.925912  |                      0.166528  |        1201 |         5 |
| ZD: SAM (ZD-H) | xgboost   |           0.994454  |                      0.166528  |        1201 |         5 |
| ZD: RTN (ZD-I) | deep_svdd |           0.983549  |                      0.28551   |        1401 |         5 |
| ZD: RTN (ZD-I) | xgboost   |           0.990596  |                      0.28551   |        1401 |         5 |
| ZD: MTC (ZD-J) | deep_svdd |           0.994634  |                      0.166528  |        1201 |         5 |
| ZD: MTC (ZD-J) | xgboost   |           0.999589  |                      0.166528  |        1201 |         5 |