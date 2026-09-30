# Pre-Registered Evaluation Metrics

As per Phase 0, all evaluation metrics are frozen prior to Phase 2 to prevent metric-shopping and ensure statistically valid comparisons.

The following metrics will be used for evaluating all anomaly detection models (Deep SVDD and baselines):

## 1. Area Under the Receiver Operating Characteristic Curve (AUROC)
- **Primary Metric:** AUROC will serve as the primary threshold-independent measure of discriminative power for each attack class against the nominal baseline.
- **Reporting:** To be computed separately for FIM, TWIRL, and the held-out Zero-Day family.

## 2. False-Positive Rate (FPR) at Fixed Key-Rate Degradation Threshold
- **Constraint:** Based on the finite-key framework extending SNS TF-QKD, we set a conservative **4% false-positive rate** (FPR) threshold to maintain a 96% baseline transmission efficiency.
- **Reporting:** We will report the True Positive Rate (Recall) for each attack class at this strict 4% FPR operating point.

## 3. Detection Latency
- **Metric:** Measured in milliseconds per inference and decoy-pulse count.
- **Reporting:** Ensures that the model operates within the practical O(1) ms coherence timescale for distributed feedback lasers and avoids large buffer backlogs.
