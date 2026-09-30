# ML-Based Attack Detection for Twin-Field QKD with Optical Injection Locking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the simulation environment, datasets, and machine learning (ML) models developed to detect continuous-wave analog side-channel attacks against Twin-Field Quantum Key Distribution (TF-QKD) systems utilizing Optical Injection Locking (OIL).

## Overview

Physical countermeasures against side-channel leakage in TF-QKD have been studied extensively, but the continuous-wave analog side channels introduced by Optical Injection Locking (OIL) have received comparatively little attention. 

This repository implements a full ML-based anomaly detection framework targeting:
1. **FIM (Fast Intensity Modulation) attacks:** Eve inflates effective photon numbers using modulation faster than standard watchdog photodiodes.
2. **TWIRL (Trojan-wavelength injection) attacks:** Eve smuggles detuned probes into the slave laser phase dynamics.
3. **FSK (Frequency Shift Keying) Zero-Day attacks:** A held-out, unseen attack for testing zero-day anomaly detectors.

Instead of relying on basic Quantum Bit Error Rate (QBER) thresholding—which these stealthy attacks are designed to evade—this framework performs classification over a 5-dimensional physically-motivated feature space derived from the Lang-Kobayashi rate equations.

## Key Features & Contributions

- **Lang-Kobayashi Simulation Pipeline:** Simulates stiff optical phase/intensity equations using Backward Differentiation Formula (BDF) solvers to generate synthetic nominal, attack, and environmental drift data.
- **Physical Feature Extraction:** Instead of raw time-series data, models are trained on distinct observables: Pulse Mean Photon Number ($\mu$), Photon Number Variance ($\sigma^2_\mu$), Spectral Sideband Power ($P_{sb}$), RMS Phase Decoherence ($\Delta\phi$), and QBER.
- **Unified ML Architecture:** Compares Classical Tree Ensembles (XGBoost, Random Forest), FedAvg Neural Networks, and distance-based SVMs.
- **Drift Hardening:** Mitigates the 99.8% False Positive Rate (FPR) typically seen under benign operational drift (e.g., thermal variance, fiber aging) down to near-zero.
- **Zero-Day Detection:** Uses a generative Deep SVDD (Support Vector Data Description) model that maps nominal conditions to a tight hypersphere, allowing successful detection of unseen zero-day attacks (like FSK) without labeled training.
- **SHAP-Based Explainability:** Provides a Tiered Response Controller that converts binary alarms into transparent tiered alerts (Warning, Suspicious, Attack) based on which physical feature drove the model's decision.

## Repository Structure

```text
.
├── data/                  # Generated synthetic datasets (.parquet)
├── figures/               # Generated evaluation plots (ROC, SHAP, etc.)
├── models/                # ML Architectures (XGBoost, FedAvg MLP, Deep SVDD, QML)
├── scripts/               # Core execution scripts
│   ├── generate_data.py             # Generates nominal, FIM, and TWIRL datasets
│   ├── generate_drift_data.py       # Generates benign operational drift data
│   ├── generate_zero_day_data.py    # Generates zero-day FSK attack data
│   ├── train_pipeline.py            # Main MLOps training/evaluation pipeline
│   └── evaluate_generalization.py   # Evaluates model robustness against drift & zero-days
├── sim/                   # Physical simulation backend (Lang-Kobayashi equations)
├── utils/                 # Helper functions for MLOps tracking, metrics, and plotting
├── Dockerfile             # Docker container definition
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup & Installation

It is recommended to run this project inside a virtual environment (e.g., `venv` or `conda`).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ml-attack-detection-tfqkd-oil.git
   cd ml-attack-detection-tfqkd-oil
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For Quantum Machine Learning models, ensure `qiskit` and relevant IBM Quantum backends are installed).*

## Usage Guide

The pipeline is split into distinct stages that can be run sequentially via the `scripts/` directory.

### 1. Generate Synthetic Data
First, generate the physically-simulated dataset from the Lang-Kobayashi differential equations. This will sweep across modulation depths ($m$) and detunings ($\Delta\lambda$) to create Nominal, FIM, and TWIRL samples.
```bash
python scripts/generate_data.py
python scripts/generate_drift_data.py
python scripts/generate_zero_day_data.py
```

### 2. Train the Models
Train the suite of Classical, Federated, and One-Class classifiers. The pipeline automatically standardizes features (Z-score), performs cross-validated grid search for hyperparameter tuning, tracks metrics using `MLflow`, and plots the confusion matrices and SHAP summaries.
```bash
python scripts/train_pipeline.py
```

### 3. Evaluate Generalization & Zero-Day Robustness
Test how the trained models handle real-world challenges: non-adversarial environmental drift (evaluating False Positive reduction via Drift Hardening) and completely novel zero-day attacks (evaluating Deep SVDD anomaly boundary mapping).
```bash
python scripts/evaluate_generalization.py
```

## Results Summary

* **Performance Thresholds:** The XGBoost classifier achieved 88.22% overall accuracy, significantly outperforming distance-based SVMs (68%) and standard FedAvg Neural Networks (33%). 
* **Geometric Discontinuity:** Due to the non-convex, overlapping class regions of the Lang-Kobayashi feature space, tree-based orthogonal partitioning drastically outperforms gradient-descent optimizers on this data geometry.
* **Explainability:** SHAP analysis demonstrated that QBER ranks lowest in feature attribution against continuous-wave side channels. Phase Decoherence ($\Delta\phi$) and Spectral Sideband Power ($P_{sb}$) primarily define the adversarial boundary.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
**Author:** Mithil Hardik Astik  
*Department of Physics, BITS Pilani Hyderabad Campus, Hyderabad, India*
