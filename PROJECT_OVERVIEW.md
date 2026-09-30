# ML-Based Side-Channel Attack Detection for OIL-Assisted Twin-Field QKD
## A Comprehensive Technical Project Report

**Author:** Mithil Hardik Astik  
**Affiliation:** Department of Physics, BITS Pilani Hyderabad Campus, Hyderabad, India  
**Target Venue:** *npj Quantum Information* (primary) | *Physical Review Applied* (backup) | *IEEE TIFS* (security-framing backup)  
**Repository:** `tfqkd-deep-svdd`

---

## Table of Contents

1. [Motivation and Problem Statement](#1-motivation-and-problem-statement)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Physical Simulation Backend](#3-physical-simulation-backend)
4. [The Attack Taxonomy](#4-the-attack-taxonomy)
5. [Feature Engineering](#5-feature-engineering)
6. [Machine Learning Models](#6-machine-learning-models)
7. [Training and Evaluation Pipeline](#7-training-and-evaluation-pipeline)
8. [Experimental Results and Ablation Studies](#8-experimental-results-and-ablation-studies)
9. [Statistical Validation](#9-statistical-validation)
10. [Adaptive Attacker Study](#10-adaptive-attacker-study)
11. [MLOps and Reproducibility Infrastructure](#11-mlops-and-reproducibility-infrastructure)
12. [Open Research Questions and Roadmap](#12-open-research-questions-and-roadmap)
13. [Summary Checklist of Completed Work](#13-summary-checklist-of-completed-work)

---

## 1. Motivation and Problem Statement

### 1.1 Background: Twin-Field QKD and Optical Injection Locking

Twin-Field QKD (TF-QKD) is a breakthrough quantum cryptography protocol designed to overcome the repeaterless key-rate bound (the PLOB bound) by using a central, untrusted relay "Charlie" to perform joint single-photon interference. Two parties, Alice and Bob, co-phase-lock their independent semiconductor lasers to a master reference laser using a technique called **Optical Injection Locking (OIL)**.

OIL achieves phase coherence across long fiber distances by injecting a small fraction of the master laser's field into the slave lasers. The slave laser then "locks" to the master's phase, allowing Alice and Bob's pulses to interfere at Charlie's beam splitter.

### 1.2 The Security Gap

While hardware-level countermeasures against standard photon-number-splitting or intercept-resend attacks are well-studied for TF-QKD, the **continuous-wave analog side-channels** introduced by the OIL process have received comparatively little attention.

The OIL locking dynamics create a set of persistent, continuously observable physical signals -- the locking range, phase noise transfer, and frequency detuning response -- that are richer in information than discrete photon-click events. A sophisticated attacker ("Eve") can exploit this by subtly perturbing the slave laser's injection parameters in ways that:

1. **Pass standard QBER thresholds**, because the stealthy attacks are specifically designed to operate within the QBER watchdog's tolerance.
2. **Degrade or control the key generation**, by manipulating the OIL dynamics to either harvest the secret key or deny key generation entirely.

### 1.3 Project Goal

This repository implements a complete, end-to-end ML-based anomaly detection framework that replaces naive QBER-thresholding with a physically-grounded, multi-dimensional classification system. The core contribution is demonstrating that:

- **Supervised Classifiers** (XGBoost, Random Forest) can reliably catch *known* attacks with high AUROC, leveraging the physical feature space.
- **Generative One-Class Models** (Deep SVDD) can catch *zero-day, unseen attacks* by learning a tight boundary around nominal OIL dynamics, without ever seeing labeled attack data.

---

## 2. System Architecture Overview

The project is a multi-stage pipeline with clearly separated concerns:

```
tfqkd-deep-svdd/
|
|-- simulator/              # Physical simulation backend
|   |-- config.py           # TF-QKD and OIL laser parameters (frozen)
|   |-- lk_dynamics.py      # Numba-JIT Lang-Kobayashi SDE solver
|   |-- attacks.py          # Attack mode constants, defaults, sampling
|   |-- tf_qkd.py           # TF-QKD protocol (key-rate, QBER, decoy states)
|   +-- dataset.py          # Dataset generation, windowing, splitting
|
|-- models/                 # ML Model definitions
|   |-- deep_svdd.py        # Deep SVDD / Deep SAD (PyTorch)
|   |-- baselines.py        # QSVM, VQC (PennyLane), StatisticalThreshold
|   |-- classical.py        # XGBoost / RF wrappers
|   |-- features.py         # Physical feature extraction (10-D)
|   |-- federated_mlp.py    # FedAvg neural network
|   |-- one_class.py        # One-class SVM wrapper
|   +-- train.py            # Unified training logic
|
|-- scripts/                # Execution scripts
|   |-- pipeline.py         # Main MLOps training/evaluation pipeline
|   |-- generate_drift_data.py
|   |-- generate_zero_day_data.py
|   |-- evaluate_generalization.py
|   |-- evasion_simulator.py
|   +-- evaluate_clustering.py
|
|-- results/                # All output artifacts
|   |-- statistical_validation.md
|   |-- adaptive_robustness.md
|   |-- ablation_table.csv
|   |-- ablation_window_size.md
|   +-- (datasets, figures, etc.)
|
|-- adaptive_attacker.py    # Adaptive FIM attacker implementation
|-- main_statistics.py      # Multi-seed statistical validation runner
|-- main_experiments.py     # Core experiment runner
|-- main_ablation.py        # Ablation study runner
|-- hyperparams.yaml        # Frozen model hyperparameters
|-- splits.json             # Frozen train/val/test splits
+-- metrics.md              # Pre-registered evaluation metrics
```

> [!IMPORTANT]
> All model hyperparameters (`hyperparams.yaml`) and dataset splits (`splits.json`) were frozen *before* Phase 3 experiments, following a strict pre-registration protocol. No metric-shopping or test-set leakage was possible by design.

---

## 3. Physical Simulation Backend

### 3.1 Protocol Definition (`simulator/config.py`)

The simulation implements the **Phase-Matching (PM) TF-QKD variant**, using decoy-state intensity randomization. Key frozen protocol parameters:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Signal intensity mu | 0.5 photons/pulse | Mean photon number for signal state |
| Decoy intensity nu | 0.1 photons/pulse | Mean photon number for decoy state |
| P_mu | 0.60 | Probability of sending signal state |
| P_nu | 0.30 | Probability of sending decoy state |
| P_0 | 0.10 | Probability of sending vacuum state |
| Detector efficiency eta_d | 0.50 | Charlie single-photon detector efficiency |
| Dark count rate p_d | 1e-8 per pulse | |
| Fiber attenuation | 0.2 dB/km | Standard telecom fiber |
| Phase slices | 16 | Phase randomization grid for PM-QKD |

**OIL / Lang-Kobayashi Laser Parameters (from `config.py`):**

| Parameter | Symbol | Value | Description |
| :--- | :--- | :--- | :--- |
| Linewidth enhancement factor | alpha | 3.0 | Amplitude-phase coupling |
| Photon decay rate | gamma_p | 1e12 /s | ~1 ps photon lifetime |
| Carrier decay rate | gamma_e | 1e9 /s | ~1 ns carrier lifetime |
| Differential gain | g_N | 1e-4 /s | |
| Transparency carrier number | N_0 | 1e7 | |
| Coupling coefficient | kappa | 1e11 /s | ~500 MHz locking bandwidth at -10 dB |
| Injection ratio | r_inj | 0.1 (-10 dB) | Fraction of master power injected |
| Nominal frequency detuning | Delta_f | 10 MHz | |
| Spontaneous emission rate | R_sp | 1e12 /s | Quantum noise source |

### 3.2 Lang-Kobayashi SDE Solver (`simulator/lk_dynamics.py`)

The physical heart of the simulator is the **`solve_lk_sde`** function -- a Numba-JIT compiled integration of the stochastic Lang-Kobayashi equations describing the slave laser's complex electric field E(t) = E_re(t) + i*E_im(t) and carrier number N(t) under optical injection.

The governing equations (Euler-Maruyama discretization at dt = 0.1 ps):

```
dE/dt = 0.5*(G - gamma_p)*(1 + i*alpha)*E + kappa*E_inj(t) + F(t)
dN/dt = J - gamma_e*N - G*|E|^2
```

Where:
- `G = g_N*(N - N_0)` is the material gain
- `kappa*E_inj(t)` is the injection field (attack-mode dependent; see Section 4)
- `F(t)` is Langevin white-noise spontaneous emission with amplitude sqrt(R_sp / 2*dt)

**Implementation Details:**
- Time step: `dt = 1e-13 s` (0.1 ps), satisfying `dt << gamma_p^{-1}` (100 ps) for numerical stability
- Entire integration loop is **Numba JIT-compiled** (`@numba.njit(cache=True)`) for near-C execution speed
- All 12 attack modes are inlined inside the Numba kernel (no Python callbacks), maintaining JIT compatibility
- Phase is sampled at each 1 ns pulse boundary via `arctan2(E_im, E_re)` relative to the master frame
- First-run JIT compilation takes ~20 min; subsequent runs use the compiled cache

The nominal simulator output is validated against known PLOB-bound behavior (`results/key_rate_vs_distance.png`).

---

## 4. The Attack Taxonomy

All attacks are implemented as perturbations to `E_inj(t)` inside the `solve_lk_sde` kernel. `simulator/attacks.py` defines mode constants, default parameters, and per-mode randomized samplers via `sample_attack_params(mode, rng)`.

### 4.1 Known Attacks (Seen During Supervised Training)

**Mode 1 -- FIM (Fast Intensity Modulation):**  
Sinusoidally modulates injection amplitude at f_mod (too fast for watchdog photodiodes) plus a static frequency offset:

```
A_inj(t) = A_inj_0 * (1 + m * sin(2*pi*f_mod*t))
delta_omega(t) = delta_omega_base + 2*pi*delta_f_FIM
```

| Parameter | Default | Randomized Range | Distribution |
| :--- | :--- | :--- | :--- |
| Modulation depth m | 0.4 | [0.05, 0.5] | Uniform |
| Modulation frequency f_mod | 50 MHz | [10, 150] MHz | Log-uniform |
| Frequency offset delta_f_FIM | 30 MHz | [5, 100] MHz | Uniform |

**Mode 2 -- TWIRL (Trojan-Wavelength Injection with Ramp-Like Locking):**  
Sweeps the master injection frequency in a triangular wave across the full locking bandwidth:

```
delta_omega(t) = delta_omega_base + triangle(t/T_sweep) * omega_sweep
```

| Parameter | Default | Randomized Range | Distribution |
| :--- | :--- | :--- | :--- |
| Sweep range omega_sweep | 200 MHz | [50, 450] MHz | Uniform |
| Sweep period T_sweep | 100 ns | [20, 500] ns | Log-uniform |

### 4.2 Zero-Day Attack Family (Entirely Held Out from Training)

Dataset label `3` (generic anomaly). Sub-type indices stored separately in `results/zeroday_subtypes_test.npy`. Full family: **10 physically distinct sub-types (ZD-A through ZD-J)**:

| ID | Code | Mode # | Physical Mechanism | Key Parameters |
| :-- | :--- | :--- | :--- | :--- |
| ZD-A | PNI | 3 | Phase-Noise Injection: Wiener-process phase perturbation added to slave field each step | sigma in [0.05, 1.5] rad/sqrt(step) |
| ZD-B | CDA | 4 | Carrier Density Attack: sinusoidal perturbation into carrier rate equation: dN += A_CDA*N_th*sin(omega_CDA*t) | A_CDA in [0.02, 0.3], f_CDA in [10, 200] MHz |
| ZD-C | DSI | 5 | Double-Sideband Injection: three-tone field (carrier + two sidebands at +/-delta_f_DSI) | delta_f in [20, 300] MHz, ratio in [0.05, 0.9] |
| ZD-D | CC | 6 | Coherence Collapse: burst of very high injection ratio within each pulse interval | boost in [2, 20]x, duty in [0.05, 0.5] |
| ZD-E | CFI | 7 | Chirped Frequency Injection: TWIRL at sub-nanosecond sweep periods | sweep in [20, 150] MHz, T in [0.5, 5] ns |
| ZD-F | CAM | 8 | Chaotic Amplitude Modulation: AR(1) autoregressive process drives injection amplitude | sigma in [0.01, 0.15] |
| ZD-G | DPJ | 9 | Discrete Phase Jumps: random discrete phase jumps +/-delta_phi with probability p each step | delta_phi in [pi/4, pi], p in [0.005, 0.05] |
| ZD-H | SAM | 10 | Sawtooth Amplitude Modulation: linear sawtooth waveform on injection amplitude | amplitude in [0.1, 0.5], f in [10, 150] MHz |
| ZD-I | RTN | 11 | Random Telegraph Noise: injection amplitude switches between two levels stochastically | amplitude in [0.1, 0.5], flip prob in [0.01, 0.1] |
| ZD-J | MTC | 12 | Multi-Tone Comb: carrier plus 4 sidebands at k = +/-1, +/-2 times comb spacing | delta_f in [20, 150] MHz, ratio in [0.05, 0.5] |

> [!NOTE]
> **Status update (2026-09-29):** The historical note below has been superseded. The full 50/100/150-km datasets were regenerated with randomized attack parameters sampled per batch; see `results/current_status.md` for the current checklist and caveats.

---

## 5. Feature Engineering

### 5.1 Window Sizing Ablation

Raw pulse-phase data is not fed directly to models. A sliding-window feature extraction step (`models/features.py`) converts traces into compact feature vectors. The window size was empirically chosen via ablation (`results/ablation_window_size.md`):

| Window Size (pulses) | XGBoost AUROC | Train Time (s) | Inference Time (s) |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.581 | 0.270 | 0.0069 |
| 10,000 | 0.710 | 0.153 | 0.0041 |
| 50,000 | 0.734 | 0.154 | 0.0056 |
| **100,000** | **0.799** | **0.153** | **0.0040** |

**W = 100,000 pulses** (100 us at 1 GHz clock) was selected: best AUROC, sub-millisecond inference, satisfying the O(1 ms) coherence timescale constraint.

### 5.2 The 10-Dimensional Feature Vector

Each window extracts features from four raw channels per pulse: intensity at Alice (I_A), intensity at Bob (I_B), differential phase (delta_phi), and click bits (c_d0, c_d1):

| Index | Feature | Physical Motivation |
| :- | :--- | :--- |
| 0 | Transmittance = mean(I_B) / mean(I_A) | Overall channel transmission; attacks that steal or reroute light alter this |
| 1 | Intensity Variance at Bob = Var(I_B) | Modulation attacks (FIM, SAM) directly increase this |
| 2 | RMS Phase Decoherence = sqrt(mean(delta_phi^2)) | Primary OIL anomaly signal; nearly all attacks disrupt phase locking |
| 3 | Click Rate = mean(c_d0 + c_d1) | Elevated click rates indicate higher photon flux (FIM-type attacks) |
| 4 | Sifted QBER | Standard QKD monitor; SHAP shows this is the LEAST informative feature |
| 5 | Lag-1 Phase Autocorrelation rho(delta_phi, 1) | TWIRL's triangular sweep creates high positive autocorrelation |
| 6 | Intensity Cross-Correlation Corr(I_A, I_B) | Normal OIL locking gives high correlation; one-sided attacks reduce this |
| 7 | Phase Kurtosis (excess, Fisher) | DPJ/PNI produce non-Gaussian phase distributions detectable via kurtosis |
| 8 | Click Asymmetry = abs(mean(c_d0) - mean(c_d1)) | Basis-biasing attacks create detector channel imbalance |
| 9 | Max Spectral Power = max(abs(FFT(I_B))[1:]) | Anti-FIM feature: sinusoidal injection creates sharp spectral peak invisible in direct intensity |

> [!IMPORTANT]
> SHAP analysis confirmed that QBER (Feature 4) ranks **lowest** in feature attribution across all attack classes. Phase Decoherence (Feature 2) and Max Spectral Power (Feature 9) are the primary discriminators. This is the key empirical validation of the project's core thesis: standard QBER thresholding is blind to these OIL side-channel attacks.

`models/features.py` also implements a 3-channel STFT spectrogram extractor (over I_A, I_B, and delta_phi) for potential future CNN-based model work.

---

## 6. Machine Learning Models

### 6.1 Deep SVDD / Deep SAD (`models/deep_svdd.py`)

The cornerstone model. **Deep Support Vector Data Description** maps nominal data to a tight hypersphere in an 8-dimensional latent space. Any sample falling outside is flagged anomalous.

**Architecture (DeepSVDDNet):**
```
Input (10-D)
  -> Linear(10, 32, bias=False) -> BatchNorm1d(32, affine=False) -> LeakyReLU(0.1)
  -> Linear(32, 16, bias=False) -> BatchNorm1d(16, affine=False) -> LeakyReLU(0.1)
  -> Linear(16, 8, bias=False)
Output (8-D representation space)
```

Bias-free layers are critical -- bias terms allow the trivial hypersphere collapse solution.

**Deep SAD Loss (Semi-Supervised Extension):**

The loss function for the semi-supervised Deep SAD mode:

```
L = (1/N) * sum_i [ 1[y_i=0] * ||z_i - c||^2  +  1[y_i>0] * max(0, M - ||z_i - c||^2) ]
```

- Nominal samples (y=0): pulled toward center c, minimizing their distance
- Anomaly samples (y>0): pushed away from c by at least margin M = 10.0
- Center c initialized as the mean representation of nominal training data

**Training Configuration (frozen in `hyperparams.yaml`):**
- Input dim: 10, Hidden: [32, 16], Representation dim: 8, Margin: 10.0
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5) with Cosine Annealing LR schedule
- Epochs: 100, Batch size: 128
- Preprocessing: RobustScaler (fitted on training data only)
- **Detection threshold:** 0.4229 (calibrated at 4% FPR on nominal validation data)

**Inference:** Anomaly score = ||z_i - c||^2. Higher score = more anomalous.

### 6.2 XGBoost and Random Forest (`models/classical.py`)

The primary **supervised** baselines. Trained with labels for all four classes (Nominal, FIM, TWIRL, Zero-Day).

**Why tree ensembles succeed here:** The LK feature space has non-convex, heavily overlapping class regions. XGBoost's orthogonal decision splits can partition this overlapping space precisely in ways that distance/density-based methods cannot -- for example, a rule like "if spectral_power > X AND rms_phase < Y, then FIM attack" is effective even when the two classes overlap continuously in a 2D projection.

Frozen hyperparameters:
- XGBoost: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`
- Random Forest: `n_estimators=100`, `max_depth=None`

### 6.3 Quantum ML Baselines (`models/baselines.py`, PennyLane)

**QSVM (Quantum SVM):** Uses an IQP (Instantaneous Quantum Polynomial) kernel embedding with 10 qubits. Kernel matrix entry K(x1, x2) is the fidelity between two quantum states. Subsampled to 200 training points due to O(N^2) kernel computation cost. Config: n_qubits=10, C=1.0.

**VQC (Variational Quantum Classifier):** IQP data encoding + StronglyEntanglingLayers variational circuit (4 layers, 10 qubits). Trained with PennyLane Adam optimizer, MSE loss on {-1, +1} labels. Config: epochs=10, lr=0.1, batch_size=32.

> [!WARNING]
> QSVM is subsampled to 200 training points vs. the full training set for classical models. VQC uses ~2000 points. This asymmetry must be explicitly flagged as a comparison-fairness caveat whenever quantum vs. classical AUROC figures are presented.

### 6.4 Statistical Threshold Baseline

A naive 3-sigma outlier detector. Flags samples where any feature deviates more than 3 standard deviations from the nominal distribution. Config: threshold_std=3.0. Represents the prior-art QBER-thresholding approach.

### 6.5 Federated MLP (`models/federated_mlp.py`)

A standard MLP trained with the **FedAvg** federated learning algorithm. Included to test whether privacy-preserving distributed training compromises detection capability. Historically achieved only ~33% accuracy, confirming MLP architectural limitations on this non-convex feature geometry.

---

## 7. Training and Evaluation Pipeline

### 7.1 Complete Pipeline Flow

**Step 1 -- Data Generation**  
`scripts/generate_drift_data.py` and `scripts/generate_zero_day_data.py` run `solve_lk_sde` across all modes. Output:
- `results/dataset_{train,val,test}.npz` (primary 100 km dataset)
- `results/dataset_test_50km.npz` and `results/dataset_test_150km.npz` (OOD cross-distance datasets)
- `results/zeroday_subtypes_{test,val}.npy` (ZD sub-type labels, isolated from main labels to prevent leakage)

**Step 2 -- Feature Extraction**  
`scripts/pipeline.py` loads raw `.npz` pulse data, applies the 10-D sliding window extractor from `models/features.py`, and fits+applies `RobustScaler` using training data only.

**Step 3 -- Model Training**  
`models/train.py` loads frozen hyperparameters from `hyperparams.yaml`, trains all models on the training split, saves to `models/saved/`.

**Step 4 -- Evaluation**  
`scripts/evaluate_generalization.py` computes AUROC and TPR@4%FPR for each attack class and sub-type. Generates ROC curves (`results/roc_curves.png`).

**Step 5 -- Statistical Validation**  
`main_statistics.py` repeats the full pipeline for 5 independent seeds (42, 123, 2024, 7, 999), computes mean+/-std across seeds, runs 1000+ bootstrap resamples for confidence intervals, and runs McNemar's test with Holm-Bonferroni correction.

**Step 6 -- MLflow Tracking**  
Integrated throughout. All runs logged to `mlflow.db` and `mlruns/`. Hyperparameters, metrics, and artifacts versioned automatically.

---

## 8. Experimental Results and Ablation Studies

### 8.1 Primary Ablation Table (Single-Split, All Models)

From `results/ablation_table.csv`:

| Model | Attack | AUROC | TPR@4%FPR | Latency (ms/pulse) |
| :--- | :--- | ---: | ---: | ---: |
| Statistical Threshold | FIM | 0.739 | 0.286 | 0.0002 |
| XGBoost | FIM | 0.942 | 0.765 | 0.0035 |
| Random Forest | FIM | **0.944** | 0.740 | 0.0054 |
| Deep SVDD | FIM | 0.815 | 0.548 | **0.0006** |
| Statistical Threshold | TWIRL | 0.599 | 0.134 | 0.0002 |
| XGBoost | TWIRL | 0.429 | 0.083 | 0.0010 |
| Random Forest | TWIRL | 0.503 | 0.048 | 0.0051 |
| Deep SVDD | TWIRL | 0.484 | 0.140 | 0.0004 |
| Statistical Threshold | Zero-Day | 0.802 | 0.254 | 0.0002 |
| XGBoost | Zero-Day | 0.895 | 0.789 | 0.0010 |
| Random Forest | Zero-Day | **0.940** | 0.778 | 0.0051 |
| Deep SVDD | Zero-Day | 0.882 | 0.611 | **0.0004** |

> [!NOTE]
> TWIRL detection is globally poor across ALL models (AUROC ~0.43-0.60). This is the documented "inlier anomaly" pathology: TWIRL's triangular frequency sweep paradoxically *regularizes* phase statistics, making the feature vector appear *more nominal* than actual nominal data in certain dimensions.

### 8.2 Multi-Seed Statistical Results (5 seeds: 42, 123, 2024, 7, 999)

From `results/statistical_validation.md`:

| Attack | Deep SVDD AUROC | Deep SVDD TPR@4% | XGBoost AUROC | XGBoost TPR@4% |
| :--- | :--- | :--- | :--- | :--- |
| FIM | 0.9048 +/- 0.0057 | 0.4683 +/- 0.0655 | 0.9276 +/- 0.0000 | 0.7804 +/- 0.0000 |
| TWIRL | 0.7356 +/- 0.0137 | 0.0660 +/- 0.0336 | 0.7967 +/- 0.0000 | 0.3188 +/- 0.0000 |
| Zero-Day (All) | 0.9289 +/- 0.0074 | 0.4488 +/- 0.0800 | 0.9744 +/- 0.0000 | 0.8882 +/- 0.0000 |

### 8.3 Per-Subtype Zero-Day Breakdown (Multi-Seed)

| Zero-Day Sub-Type | Deep SVDD AUROC | Deep SVDD TPR@4% | XGBoost AUROC | XGBoost TPR@4% |
| :--- | :--- | :--- | :--- | :--- |
| ZD-A: PNI | 0.9502 +/- 0.0087 | 0.4245 +/- 0.0766 | 0.9966 +/- 0.0000 | 0.9950 +/- 0.0000 |
| ZD-B: CDA | **0.6030 +/- 0.0210** | **0.0000 +/- 0.0000** | **0.6302 +/- 0.0000** | **0.0000 +/- 0.0000** |
| ZD-C: DSI | 0.9357 +/- 0.0151 | 0.3370 +/- 0.1062 | 0.9782 +/- 0.0000 | 0.8250 +/- 0.0000 |
| ZD-E: CFI | 0.9365 +/- 0.0100 | 0.4752 +/- 0.1253 | 0.9714 +/- 0.0000 | 0.8440 +/- 0.0000 |
| ZD-F: CAM | 0.9531 +/- 0.0095 | 0.6270 +/- 0.1571 | 0.9957 +/- 0.0000 | 0.9500 +/- 0.0000 |
| ZD-G: DPJ | 0.9380 +/- 0.0119 | 0.1280 +/- 0.0527 | 0.9999 +/- 0.0000 | 1.0000 +/- 0.0000 |
| ZD-H: SAM | 0.9115 +/- 0.0152 | 0.3520 +/- 0.0427 | 0.9606 +/- 0.0000 | 0.7150 +/- 0.0000 |
| ZD-I: RTN | 0.9408 +/- 0.0108 | 0.4610 +/- 0.0466 | 0.9946 +/- 0.0000 | 0.9725 +/- 0.0000 |
| ZD-J: MTC | 0.9500 +/- 0.0047 | 0.4990 +/- 0.0120 | 0.9985 +/- 0.0000 | 1.0000 +/- 0.0000 |

> [!IMPORTANT]
> **CDA (ZD-B) is the hardest attack for both models.** AUROC ~0.60 and TPR@4% = 0.0 for both Deep SVDD and XGBoost indicates complete detection failure. This is the clearest "inlier anomaly" case: the carrier density perturbation at certain amplitudes actively *stabilizes* the feature distribution, pushing feature vectors closer to the nominal cluster than actual nominal data. This is a genuinely important open problem.

---

## 9. Statistical Validation

### 9.1 Pre-Registered Evaluation Metrics (`metrics.md`)

All evaluation criteria were frozen before Phase 3 experiments:

1. **AUROC** -- Threshold-independent discriminative power, computed separately per attack class.
2. **TPR @ 4% FPR** -- True positive rate at the strict 4% false positive rate operating point. Derived from the finite-key SNS TF-QKD framework: false positives trigger session resets, and 96% baseline transmission efficiency is the minimum acceptable threshold.
3. **Detection Latency (ms/pulse)** -- Inference must be within O(1 ms) to fit within the coherence timescale of distributed feedback lasers.

### 9.2 McNemar's Test Results (XGBoost vs. Deep SAD, Holm-Bonferroni corrected)

From `results/statistical_validation.md`:

| Attack | p-value | Significant (alpha=0.05) |
| :--- | ---: | :--- |
| FIM | 3.98e-170 | Yes |
| TWIRL | 6.67e-145 | Yes |
| Zero-Day (All) | 1.58e-236 | Yes |
| ZD-A: PNI | 1.60e-51 | Yes |
| ZD-B: CDA | 9.08e-09 | Yes |
| ZD-C: DSI | 4.28e-25 | Yes |
| ZD-E: CFI | 1.68e-20 | Yes |
| ZD-F: CAM | 4.73e-41 | Yes |
| ZD-G: DPJ | 3.14e-22 | Yes |
| ZD-H: SAM | 1.54e-16 | Yes |
| ZD-I: RTN | 6.19e-47 | Yes |
| ZD-J: MTC | 5.53e-24 | Yes |

Every pairwise comparison is statistically significant at extraordinary confidence levels, even after multiple-comparison correction.

---

## 10. Adaptive Attacker Study

### 10.1 Motivation

The adaptive attacker study addresses the core security question: *"What happens when the attacker knows the detector exists and adapts their strategy?"* (`adaptive_attacker.py`, `results/adaptive_robustness.md`)

### 10.2 Methodology

Grid search over the FIM frequency offset parameter at five modulation depths. The attacker has a dual objective:
- **Minimize:** Deep SAD anomaly score (evade the 0.4229 detection threshold)
- **Constraint:** Secure key rate must equal zero (the attack must still succeed)

Each grid point simulates 100,000 laser pulses -- matching the training feature extraction window exactly -- ensuring feature vectors are computed identically to the training environment.

### 10.3 Results

| Attacker Strength (mod_depth) | Optimal Freq Offset | Anomaly Score | Threshold | Evaded? | Secure Key Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.0 (Nominal baseline) | 0.00 MHz | 0.1632 | 0.4229 | N/A | 0.0 |
| 0.1 (Weak) | 30.00 MHz | 6.4742 | 0.4229 | NO -- Caught | 7.07e-4 |
| **0.2 (Moderate)** | **30.00 MHz** | **0.0564** | 0.4229 | **YES -- Evaded** | **0.0** |
| **0.3 (Strong)** | **50.00 MHz** | **0.0110** | 0.4229 | **YES -- Evaded** | **0.0** |
| 0.4 (Extreme) | 30.00 MHz | 1.7487 | 0.4229 | NO -- Caught | 0.0 |

### 10.4 Critical Finding: Blind Spots Exist

At Mod Depth 0.2 and 0.3, the anomaly scores (0.0564 and 0.0110) are **lower than the nominal baseline of 0.1632**. The optimized attacker parameters produce a feature vector that sits *closer to the center of the nominal hypersphere* than actual nominal data -- while completely destroying the secure key rate.

This reveals a fundamental vulnerability of hypersphere-based anomaly detection against a knowledgeable adversary: the exterior of the hypersphere contains both genuine anomalies AND adversarially crafted "super-nominal" points that exploit the learned boundary.

> [!CAUTION]
> This finding does NOT invalidate zero-day detection capability for non-adaptive adversaries. It motivates ensemble approaches, adversarial training augmentation, or dynamic thresholding as next steps. The result is an honest, reportable finding that pre-empts the "what if the attacker adapts?" reviewer critique.

---

## 11. MLOps and Reproducibility Infrastructure

| Component | Implementation | Details |
| :--- | :--- | :--- |
| Experiment Tracking | MLflow | All hyperparameters, metrics, artifacts versioned in `mlflow.db` + `mlruns/` |
| Frozen Splits | `splits.json` | Train/val/test indices locked before Phase 3 |
| Frozen Hyperparameters | `hyperparams.yaml` | All model configs frozen before Phase 3 |
| Pre-registered Metrics | `metrics.md` | AUROC, TPR@4%FPR, latency -- written before any results |
| Multi-seed Protocol | 5 seeds (42, 123, 2024, 7, 999) | Mean +/- std for all headline numbers |
| ZD Sub-type Isolation | `results/zeroday_subtypes_test.npy` | Separate from dataset labels; prevents leakage to models |
| Containerization | `Dockerfile` | Reproducible runtime environment |
| Unit Testing | `tests/` directory | 25 passing tests covering simulator, feature extractor, dataset |
| Cross-distance Datasets | 50 km, 100 km, 150 km `.npz` | Generated and stored; formal cross-distance eval pending |

---

## 12. Open Research Questions and Roadmap

Full roadmap: `research_strengthening_plan.md`. Prioritized open tasks:

### Priority 1 -- Attack Parameter Diversity (Critical, XL effort)
`sample_attack_params()` is implemented but the dataset was generated with fixed defaults. Regenerate using randomized per-batch or per-window parameters. This transforms "one unseen waveform instance" into "a genuine distribution of unseen waveforms" -- the single most important structural fix before any paper draft is finalized.

### Priority 2 -- CDA / TWIRL Inlier Anomaly Analysis (High, S effort)
CDA (ZD-B) shows AUROC 0.60 and TPR@4% = 0 for both models. Diagnose whether carrier-density perturbation actively regularizes feature statistics to appear super-nominal. Data already exists; this is analysis only, no new experiments required.

### Priority 3 -- Extended Adaptive Attacker Study (High, L effort)
Current study: FIM only, Deep SAD only, grid search. Needed additions:
- TWIRL and CDA/CC as attack targets
- XGBoost as a second target detector
- Gradient-based (PGD/Adam on surrogate model) rather than grid search
- Adaptive evasion of the spectral-feature countermeasure itself

### Priority 4 -- Cross-Distance OOD Evaluation (Medium, M effort)
50 km and 150 km test datasets exist (`results/dataset_test_50km.npz`, `results/dataset_test_150km.npz`). A formal evaluation (train @100 km, test @50/150 km) for the top models has not yet been run or reported.

### Priority 5 -- Physical/Hardware Generalization (Medium, M effort)
Sweep alpha in {2.0, 2.5, 3.0, 3.5, 4.0, 5.0} to test whether trained models generalize across real DFB laser parameter variability, or require per-device calibration.

### Priority 6 -- Paper Writing (Ongoing)
Target: npj Quantum Information (methods-paper format). A results-based draft is in `manuscript_draft.tex`; `deep_sad_upgrade.tex` is an earlier proposal and does not describe the completed experiments. Remaining items include alpha/adaptive results, related-work expansion, calibration reliability diagrams, and hardware/environment specifications for latency claims.

---

## 13. Summary Checklist of Completed Work

| Task | Status |
| :--- | :--- |
| Lang-Kobayashi SDE solver (Numba-JIT, all 12 attack modes) | COMPLETE |
| TF-QKD protocol simulation (decoy-state, PM variant, PLOB bound validation) | COMPLETE |
| Full attack taxonomy defined (FIM, TWIRL, 10 Zero-Day sub-types) | COMPLETE |
| Randomized attack parameter sampler `sample_attack_params()` | COMPLETE |
| 10-D physical feature extractor with sliding window | COMPLETE |
| Window-size ablation (empirical justification) | COMPLETE |
| Deep SVDD / Deep SAD (PyTorch, semi-supervised margin loss) | COMPLETE |
| XGBoost, Random Forest supervised baselines | COMPLETE |
| QSVM, VQC quantum baselines (PennyLane) | COMPLETE |
| Federated MLP baseline (FedAvg) | COMPLETE |
| Statistical threshold baseline (3-sigma) | COMPLETE |
| Frozen splits + hyperparameters + pre-registered metrics | COMPLETE |
| Multi-seed retraining (5 seeds, mean +/- std for all headlines) | COMPLETE |
| McNemar's test with Holm-Bonferroni correction (all attack classes) | COMPLETE |
| Per-subtype zero-day breakdown (9 sub-types formally evaluated) | COMPLETE |
| Adaptive attacker study (FIM, grid search, Deep SAD) | COMPLETE |
| Unsupervised clustering evaluation (K-Means, GMM, PCA visualization) | COMPLETE |
| Multi-distance dataset generation (50 km, 100 km, 150 km) | COMPLETE |
| MLflow tracking throughout pipeline | COMPLETE |
| Unit test suite (25 tests) | COMPLETE |
| Full dataset regeneration with randomized attack parameters | COMPLETE |
| Gradient-based adaptive attacker (surrogate model) | COMPLETE (exploratory; seven candidates validated) |
| Cross-distance OOD evaluation (formal results table) | COMPLETE |
| Laser-parameter (alpha) generalization study | COMPLETE (exploratory; model-artifact version caveat) |
| SHAP feature importance analysis | COMPLETE |
| Precision-recall curves + PR-AUC | COMPLETE |
| Full manuscript draft | DRAFTED (all checklist results integrated; compilation unavailable) |

---

*Updated from repository state: 2026-09-30*  
*See also:*  
*- [`research_strengthening_plan.md`](research_strengthening_plan.md) -- full reviewer-proofing roadmap*  
*- [`tf-qkd-deep-svdd-roadmap.md`](tf-qkd-deep-svdd-roadmap.md) -- phase-by-phase execution plan*
