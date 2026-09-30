# ML-Based Attack Detection for Optical-Injection-Locked Twin-Field QKD

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains a simulation and machine-learning research project exploring detection of optical side-channel attacks against twin-field quantum key distribution (TF-QKD) systems that use optical injection locking (OIL). It includes a Lang–Kobayashi laser simulator, attack scenarios, feature extraction, supervised and anomaly-detection models, experiment scripts, and saved evaluation artifacts.

**Research scope:** results in this repository are produced by a simulator. They do not establish performance on physical hardware, constitute a security proof for a QKD protocol, or demonstrate deployment readiness. See the limitations below and the detailed reports before interpreting metrics.

## Contents

- [Research overview](#research-overview)
- [Repository layout](#repository-layout)
- [Environment setup](#environment-setup)
- [Data and reproducibility](#data-and-reproducibility)
- [Running the project](#running-the-project)
- [Saved results](#saved-results)
- [Limitations and interpretation](#limitations-and-interpretation)
- [References within the repository](#references-within-the-repository)
- [License](#license)

## Research overview

The simulator models OIL laser dynamics and TF-QKD observables, then evaluates whether learned detectors can distinguish nominal operation from simulated attacks and operating-condition changes. The attack families represented in the code include fast intensity modulation (FIM), TWIRL, and a collection of held-out zero-day scenarios. The precise attack definitions and parameters are implemented in `simulator/attacks.py` and `simulator/tf_qkd.py`.

The model code includes XGBoost and Random Forest classifiers, Deep SVDD-style reconstruction-based anomaly detection, one-class and statistical baselines, clustering, a federated MLP, and PennyLane quantum baselines. Not every model is used by every experiment script; consult each script and its saved report for the actual evaluation protocol.

The physical simulation produces pulse-level quantities such as phase, intensity, and detector clicks. Feature extraction and windowing turn those traces into model inputs. Dataset generation, split definitions, model settings, and evaluation scripts are in the repository so the experiments can be inspected and rerun, subject to the caveats below.

## Repository layout

| Path | Contents |
| --- | --- |
| `simulator/` | TF-QKD and Lang–Kobayashi simulation, attack definitions, configuration, and dataset generation. |
| `models/` | Feature processing and model implementations, including classical, one-class, Deep SVDD, federated, clustering, and quantum baselines. |
| `scripts/` | Dataset generation, training, evaluation, explainability, statistical analysis, and figure-generation entry points. |
| `data/` | Parquet datasets used by the earlier feature-table pipeline. |
| `results/` | Generated pulse-level datasets, split/subtype files, metrics, tables, plots, and experiment reports. |
| `figures/` | Selected figures generated for analysis and reporting. |
| `checkpoints/`, `models/saved/` | Locally generated model artifacts when present. |
| `mlruns/`, `mlflow.db` | MLflow tracking records and artifacts when present. |
| `hyperparams.yaml`, `splits.json` | Experiment configuration and split metadata used by parts of the project. |
| `metrics.md` | Metric definitions and evaluation notes. |
| `PROJECT_OVERVIEW.md` | Longer technical report and project notes. |
| `manuscript_draft.tex` | Research manuscript draft. |
| `requirements.txt` | Python package dependencies. |

There are multiple generations of the pipeline in this repository. In particular, `data/*.parquet` is used by scripts such as `scripts/pipeline.py` and `scripts/train_models.py`, while the newer physical-trace workflow uses `.npz` and `.npy` files under `results/` and code in `simulator/dataset.py`. Check the input paths in the script you intend to run; these pipelines are not interchangeable.

## Environment setup

The project targets Python 3.10 (the included Dockerfile uses `python:3.10-slim`). A virtual environment is recommended.

```bash
git clone https://github.com/lihtim-kitsa/tfqkd-oil-zeroday.git
cd tfqkd-oil-zeroday

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some environments may need platform-specific PyTorch installation instructions. The quantum experiment also depends on PennyLane; `scripts/run_ibm_quantum.py` may require separate IBM Quantum credentials and provider setup. Those credentials are not included in this repository.

## Data and reproducibility

The checked-in data and results are research artifacts, not required software dependencies. Regenerating the physical simulation data can be computationally expensive. The newer generator in `simulator/dataset.py` uses batches and a cache under `results/`; inspect its function arguments and constants before launching a full run. For the existing feature-table pipeline, `scripts/generate_dataset_v1.py` creates `data/dataset_v1.parquet`; drift and zero-day dataset generation have separate scripts.

Several scripts write files to relative paths such as `results/`, `checkpoints/`, or the repository root. Run them from the repository root. Seeds, split definitions, and model hyperparameters are recorded in the relevant scripts and files, but exact reproduction can depend on package versions, platform, and generated or serialized artifacts.

## Running the project

The project has distinct entry points rather than one universal command. These common commands show the intended script paths; some stages expect datasets or artifacts from an earlier stage.

### Generate the earlier feature-table datasets

```bash
python scripts/generate_dataset_v1.py
python scripts/generate_drift_data.py
python scripts/generate_zero_day_data.py
```

The first script writes the nominal/FIM/TWIRL feature table under `data/`. The other scripts generate drift and zero-day data used by related pipeline stages. Review their source for default sample counts and output paths before starting long simulations.

### Train and evaluate the feature-table pipeline

```bash
python scripts/pipeline.py
```

This pipeline reads `data/dataset_v1.parquet` and `data/zero_day_dataset.parquet`, writes checkpoints, and records experiments in MLflow. `scripts/train_models.py` provides a separate training path; it should not be assumed to use exactly the same evaluation protocol.

### Generate and evaluate the pulse-level physical dataset

The physical dataset generator is implemented in `simulator/dataset.py`. Its defaults define a 100 km simulation with batched traces and train/validation/test partitions. Inspect the module or call `generate_dataset(...)` from Python to choose an output directory, seed, distance, batch size, or batch counts. This can require substantial computation.

After the expected `.npz`/`.npy` files exist under `results/`, the repository includes scripts for training and evaluation, including:

```bash
python scripts/evaluate_generalization.py
python scripts/evaluate_cross_distance.py
python scripts/evaluate_clustering.py
python scripts/evaluate_alpha_generalization.py
```

These analysis scripts may also expect trained model/scaler artifacts. Read their configuration and required paths before running; the saved reports in `results/` document completed runs and their caveats.

### Other analysis entry points

- `main.py`, `main_experiments.py`, `main_ablation.py`, and `main_statistics.py` run project-level simulations or studies.
- `scripts/train_federated.py` and `scripts/train_hardened.py` cover other training experiments.
- `scripts/adaptive_surrogate.py` and `scripts/evasion_simulator.py` investigate adaptive attacker scenarios.
- `scripts/generate_paper_figures.py` and related figure scripts regenerate selected plots.
- `scripts/run_ibm_quantum.py` is the IBM Quantum-related entry point and requires external account configuration.

Script arguments and defaults can change; use `python <script> --help` where supported and inspect the script before a long run.

## Saved results

The `results/` directory contains reports and artifacts for multiple studies, including:

- `statistical_validation.md` — multi-seed metrics and statistical comparisons for the documented run.
- `precision_recall.md` — average-precision summaries and positive-class baselines.
- `cross_distance_ood.md` — evaluation of models trained at 100 km on simulated 50 km and 150 km traces.
- `alpha_generalization.md` — exploratory laser-parameter generalization analysis.
- `adaptive_robustness.md` and `adaptive_surrogate.md` — adaptive-attacker analyses.
- `feature_importance.md` — feature-importance analysis.
- `ablation_window_size.md` — window-size ablation.
- `current_status.md` — research checklist and important interpretation notes.

Metrics in different reports may come from different pipeline generations, data versions, seeds, and evaluation protocols. Do not compare values across reports as if they came from one controlled benchmark; use each report's stated methodology.

## Limitations and interpretation

- **Simulation only:** the included studies use modeled physical dynamics and synthetic data. Hardware validation and protocol-level security proofs are outside this repository's demonstrated results.
- **Correlated windows:** some evaluations use 100,000-pulse windows stepped by 1,000 pulses, giving 99% overlap. Window-level metrics and seed variation are descriptive and should not be interpreted as independent-sample confidence intervals.
- **Detection gaps:** saved evaluations report weak detection for some attack families and subtypes. In particular, results for TWIRL and the CDA zero-day subtype show substantial limitations; consult the current reports rather than relying on headline aggregate scores.
- **Adaptive attackers:** the repository includes simulations where adaptive attack parameters can evade a detector. These experiments are exploratory and underscore that anomaly detection alone is not a security guarantee.
- **Artifact compatibility:** loading serialized estimators can produce warnings when scikit-learn versions differ from those used to save them. Some exploratory reports explicitly note such version mismatches.
- **Resource requirements:** physical simulation and data regeneration can be slow and produce large files. The checked-in environment, caches, model runs, and result data are not a lightweight package installation.

## References within the repository

- `PROJECT_OVERVIEW.md` — detailed architecture, methods, and research notes.
- `metrics.md` — metric definitions and evaluation criteria.
- `research_strengthening_plan.md` and `tf-qkd-deep-svdd-roadmap.md` — research planning documents.
- `manuscript_draft.tex` — current manuscript draft; `deep_sad_upgrade.tex` is an earlier proposal.
- `LICENSE` — MIT License.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
