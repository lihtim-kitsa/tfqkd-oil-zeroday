# Research Roadmap: Statistically Validated Zero-Day Anomaly Detection for OIL-Based TF-QKD via Deep SVDD

**Target venue:** npj Quantum Information (primary) · Physical Review Applied (backup) · IEEE TIFS (backup, security-framing rewrite)
**Paper type:** Simulation-based ML security paper, methods-focused
**Core claim:** A Deep SVDD anomaly detector trained only on nominal OIL-locking dynamics generalizes to *unseen* attack classes (FIM, TWIRL, held-out third variant) on TF-QKD systems, outperforming supervised baselines (QSVM, VQC, threshold detectors) with statistically validated significance and demonstrated robustness to an attacker adaptive to the detector's existence.

---

## 0. Non-negotiable design decisions (lock these before writing code)

- [ ] **Evaluation metrics pre-registered**: AUROC, detection latency (ms / decoy-pulse count), false-positive rate at fixed key-rate degradation threshold. No metric-shopping after results are in — write these into a frozen `metrics.md` before Phase 2 begins.
- [x] **Attack taxonomy fixed**: FIM (Frequency/Intensity Manipulation), TWIRL, and a **Zero-Day Family** of 4 physically distinct sub-types (PNI, CDA, DSI, CC) — randomly sampled at test time, none seen during training. Implemented in `simulator/attacks.py`.
- [ ] **Simulation-only scope stated explicitly** in the paper's limitations section — no hardware access, so all channel/injection-locking dynamics are simulated. This is acceptable in-subfield if flagged early and the simulator is benchmarked against known PLOB-bound behavior.
- [ ] **One venue, one bar**: write to npj Quantum Information's methods-paper conventions from week 1, don't try to hedge across all three.

---

## Phase 1 — Simulator & Threat Model (Weeks 1–3)

**Goal:** A physically credible OIL-based TF-QKD channel simulator that produces both nominal and attacked traces.

- ~~Week 1: Implement decoy-state TF-QKD protocol simulation (intensity/phase encoding, basis reconciliation, key-rate calculation).~~ ✅ **DONE** — `simulator/tf_qkd.py`, `simulator/config.py`
- ~~Week 2: Implement Optical Injection Locking dynamics model (locking range, phase noise transfer, frequency detuning response).~~ ✅ **DONE** — `simulator/lk_dynamics.py` (Lang-Kobayashi SDE, Numba-JIT)
- ~~Week 3: Implement attack injection — FIM, TWIRL, and the held-out third attack — as perturbations to the OIL locking signal. Validate nominal-channel output against known PLOB bound as a sanity check.~~ ✅ **DONE** — `simulator/attacks.py` (6 modes), `simulator/dataset.py`, `main_attacks.py`, 25/25 tests passing.

**Deliverable:** ✅ `simulator/` module complete. `results/key_rate_vs_distance.png` confirms PLOB-bound behavior. `results/key_rate_under_attacks.png` shows all 7 modes. Dataset splits in `results/dataset_{train,val,test}.npz`.

> **Note:** First run of `main_attacks.py` or `pytest` triggers Numba JIT compilation (~20 min). Subsequent runs use the cache in `simulator/__pycache__/`.

---

## Phase 2 — Detector Implementation (Weeks 4–6)

**Goal:** Deep SVDD anomaly detector + all baseline comparators, trained under identical conditions.

- Week 4: Implement Deep SVDD (trained only on nominal OIL traces — no attack data seen during training).
- Week 5: Implement baselines: QSVM, VQC (variational quantum classifier), and a simple statistical threshold detector — all trained *with* labeled attack data (supervised), since these are the natural point of comparison for "does zero-day generalization beat supervised specificity."
- Week 6: Freeze train/test splits. Freeze hyperparameters. No further tuning after this week — write this down explicitly in the reproducibility checklist (below) so reviewers can't accuse you of test-set leakage.

**Deliverable:** `models/` module, frozen `splits.json`, frozen `hyperparams.yaml`.

---

## Phase 3 — Core Experiments (Weeks 7–9)

**Goal:** Generate the primary results table.

- Week 7: Run Deep SVDD and all baselines on FIM and TWIRL (attacks the supervised baselines *have* seen in training).
- Week 8: Run all detectors on the held-out third attack class (zero-day test) — this is the headline result.
- Week 9: Collect AUROC, detection latency, and false-positive-at-fixed-key-rate-degradation for every detector × every attack class. Build the master ablation table (Deep SVDD vs QSVM vs VQC vs threshold, same data, same splits).

**Deliverable:** `results/ablation_table.csv` + first-draft Figure 1 (ROC curves per detector per attack).

---

## Phase 4 — Statistical Validation (Weeks 10–11)

**Goal:** Make the results reviewer-proof.

- Week 10: Bootstrap confidence intervals (≥1000 resamples) on all headline metrics. Apply Holm-Bonferroni correction across the multiple attack-class comparisons to control family-wise error rate.
- Week 11: McNemar's test for pairwise detector comparison on shared test instances (Deep SVDD vs each baseline, per attack class).

**Deliverable:** `results/statistical_validation.md` — every claimed superiority must have a CI and a corrected p-value attached before it goes in the paper.

---

## Phase 5 — Adaptive Robustness Study (Weeks 12–13)

**Goal:** Pre-empt the single most likely "yeah but" from a QKD security reviewer — what if the attacker knows the detector exists?

- Week 12: Implement an adaptive FIM attacker that optimizes its perturbation to stay under the Deep SVDD decision boundary (gradient-based or black-box optimization against a surrogate of the detector).
- Week 13: Re-run detection metrics against the adaptive attacker. Report the degradation honestly — this section is what separates a complete paper from an incremental one, even if the numbers aren't flattering.

**Deliverable:** `results/adaptive_robustness.md` + Figure showing detection rate vs. attacker adaptation strength.

---

## Phase 6 — Writing (Weeks 14–15)

**Goal:** Full manuscript draft, structured to npj Quantum Information conventions.

- Week 14: Draft Introduction, Related Work (comparison table against prior ML-QKD security papers), Methods (simulator + detector architecture), Results.
- Week 15: Draft Discussion, Limitations (simulation-only scope, stated plainly), Dual-use ethics statement, Reproducibility checklist, Conclusion. Full pass for citation accuracy — every claim traceable to a source or to your own results table.

**Deliverable:** Full manuscript draft, `manuscript.docx` or LaTeX source.

---

## Phase 7 — Internal Review & Submission Prep (Week 16)

- [ ] Reproducibility checklist complete (seeds, hyperparameters, splits, hardware/software versions all documented).
- [ ] Every figure has a corresponding data file in `results/`.
- [ ] Every statistical claim has a CI and corrected p-value.
- [ ] Limitations section explicitly states simulation-only scope.
- [ ] Dual-use ethics statement present (this is an attack-detection paper — venue reviewers will expect it).
- [ ] Related-work table checked against most recent literature (re-search before submission — don't rely on the version from Phase 6).
- [ ] Format to npj Quantum Information submission guidelines (word count, figure limits, supplementary material structure).
- [ ] Prepare rebuttal-ready appendix material: ablation table, adaptive robustness numbers, statistical test details — anticipate reviewer #2 asking for exactly these.

**Deliverable:** Submission-ready package.

---

## Reviewer-Proofing Checklist (carry through every phase)

| Risk | Mitigation | Phase |
|---|---|---|
| "Just another classifier" critique | Zero-day framing: train only on nominal data, test on unseen attack class | 2, 3 |
| Metric-shopping accusation | Pre-registered metrics frozen before experiments | 0 |
| Test-set leakage | Frozen splits + hyperparameters before Phase 3, documented | 2 |
| "What if attacker adapts?" | Dedicated adaptive robustness study | 5 |
| Simulation-only skepticism | PLOB-bound validation of simulator + explicit limitations statement | 1, 6 |
| Statistical rigor questioned | Bootstrapped CIs + Holm-Bonferroni + McNemar's test | 4 |
| Reproducibility concerns | Full checklist: seeds, splits, hyperparameters, software versions | 7 |

---

## Open decisions to revisit

- Final choice of held-out third attack class (candidates: a phase-noise injection variant not in current FIM/TWIRL taxonomy — needs literature check for precedent).
- Whether VQC baseline is worth the barren-plateau debugging overhead vs. dropping it and noting the omission with justification.
- Whether Physical Review Applied is worth a parallel-track rewrite if npj Quantum Information rejects, or whether IEEE TIFS's security-framing is a better fallback given the attack-detection angle.
