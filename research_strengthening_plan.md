# Research Strengthening Plan
## OOD Generalization for Optical Injection-Locking Side-Channels in TF-QKD via Deep Semi-Supervised Anomaly Detection

**Purpose of this document:** a concrete, reviewer-proofing roadmap to take the current PHY F266 study-project report to publication quality (target: a security-focused venue such as *Physical Review A/Applied*, *PRX Quantum*, *IEEE TIFS*, or a QKD/quantum-crypto workshop). Every item below is written as an actionable task with a rationale (what reviewer objection it defends against), a concrete method, and an estimate of effort.

> **Status note (2026-09-30):** This is the original planning roadmap; its checkboxes are historical and have not been reconciled task-by-task. See [`results/current_status.md`](results/current_status.md) for the completed alpha/adaptive work, generated artifacts, and known caveats.

---

## 0. How to use this document

Each major section = one weakness class in the current report. Within each section:
- **Reviewer risk** — the specific criticism a referee would write.
- **Current state** — what the report does today (with section references to the existing report).
- **Target state** — what needs to exist for the claim to be defensible.
- **Concrete tasks** — step-by-step, implementable directly against the existing `simulator` package (`config.py`, `lk_dynamics.py`, `attacks.py`, `tf_qkd.py`, `dataset.py`).
- **New artifacts** — figures/tables this produces, which map directly onto new paper sections.
- **Effort** — rough sizing (S = <1 day, M = 1–3 days, L = 3–7 days, XL = 1–2 weeks) assuming the simulator already runs.

Sections are ordered by priority (highest-impact-per-effort first), not by where they'd appear in the final paper. A suggested paper table of contents is given in Section 10.

---

## 1. Attack Parameter Diversity (Critical — do this first)

### Reviewer risk
"All six attacks are injected at a single fixed default parameter configuration (Table 4.1). The reported 'zero-day generalization' is therefore generalization to one specific unseen *waveform instance*, not to a *class* of physically plausible attacks. This substantially overstates the claim in the abstract."

This is the single most damaging gap in the current report, because it undermines the headline number (Deep SAD 0.7959 AUROC on zero-day; XGBoost 0.9444 on the 9-D feature set). Fix this before investing further in anything downstream, since every subsequent experiment should be re-run on top of it.

### Current state
- `attacks.py` implements each of FIM, TWIRL, PNI, CDA, DSI, CC with one hardcoded default parameter set (Table 4.1: e.g., FIM at m=0.4, fmod=50 MHz, Δf=30 MHz).
- Section 5.1 confirms every attack batch uses these defaults uniformly.
- The only parameter variation anywhere in the report is the Chapter 12 adaptive-attacker grid search over FIM modulation depth {0.1, 0.2, 0.3, 0.4} and frequency offset — and that is a post-hoc robustness probe, not part of the main dataset.

### Target state
Each attack instance (per pulse-window, or per batch — see below) draws its defining parameters from a distribution, not a constant. Detectors are trained and evaluated across the resulting parameter range, and performance is reported **stratified by attack strength**, not only as a single pooled AUROC.

### Concrete tasks

1. **Define parameter distributions per attack** (physically motivated ranges, not arbitrary):

   | Attack | Parameter | Current default | Proposed range | Distribution |
   |---|---|---|---|---|
   | FIM | modulation depth m | 0.4 | [0.05, 0.5] | Uniform |
   | FIM | mod. frequency fmod | 50 MHz | [10, 150] MHz | Log-uniform |
   | FIM | freq. offset ΔfFIM | 30 MHz | [5, 100] MHz | Uniform |
   | TWIRL | sweep range Δfsweep | 200 MHz | [50, 450] MHz | Uniform |
   | TWIRL | sweep period Tsweep | 100 ns | [20, 500] ns | Log-uniform |
   | PNI | phase noise σ | 0.5 rad/√step | [0.05, 1.5] rad/√step | Log-uniform |
   | CDA | amplitude ACDA | 0.1 | [0.02, 0.3] | Uniform |
   | CDA | frequency fCDA | 80 MHz | [10, 200] MHz | Log-uniform |
   | DSI | sideband offset ΔfDSI | 120 MHz | [20, 300] MHz | Uniform |
   | DSI | power ratio rDSI | 0.5 | [0.05, 0.9] | Uniform |
   | CC | boost βboost | 8.0 | [2, 20] | Log-uniform |
   | CC | duty cycle fduty | 0.2 | [0.05, 0.5] | Uniform |

   Lower bounds should be chosen so the attack is still *in principle* detectable-in-theory (i.e. produces *some* physical deviation above shot-noise floor established in Section 8.1); upper bounds should stay inside regimes where the Lang-Kobayashi integrator remains numerically stable (retain the `dt ≪ γp⁻¹` constraint).

2. **Implement a `sample_attack_params(attack_type, rng)` function** in `attacks.py` that returns a parameter dict per batch (or, for finer granularity, per pulse-window) drawn from the above distributions, seeded from the existing dataset-generation RNG (seed 42 lineage) so this remains fully reproducible.

3. **Decide and document the sampling granularity.** Two options, pick one and justify it in the methods section:
   - *Per-batch* (500,000 pulses share one draw) — simpler, matches current architecture, but only gives ~23 distinct attack instances total.
   - *Per-window* (each 100,000-pulse feature window gets its own draw) — much richer coverage (thousands of distinct instances), more realistic, but requires re-plumbing the batch generator. **Recommended** if engineering time allows; it is what turns "one unseen waveform" into "a genuine distribution of unseen waveforms."

4. **Regenerate the full 11.5M-pulse dataset** (or scale up — see Section 6 on dataset size) with randomized parameters, keeping the existing 70/15/15 nominal split and 50/50 val/test attack split logic (Section 5.2) unchanged.

5. **Add an "attack strength" scalar per sample** to the persisted metadata — e.g., normalized modulation depth / boost factor / noise σ — so every downstream evaluation can be stratified by it without needing to re-derive it later.

6. **Re-run the CUSUM validation (Section 5.3) and the 25-test unit suite (Section 5.4) against the randomized generator**, adding new unit tests that check the sampled parameters fall within the declared bounds and that boundary cases (min/max of each range) still produce the expected qualitative attack signature.

### New artifacts
- A new **Table**: attack parameter distributions (the table above, refined).
- A new **Figure**: AUROC vs. attack strength (per attack class, per detector) — this becomes one of the most important figures in the paper, since it directly shows the operating envelope of each detector ("Deep SAD is reliable above modulation depth 0.15 but degrades below that").
- Updated Chapter 9/11 result tables, now reporting **mean AUROC ± std across the strength distribution**, plus the stratified curve above.

### Effort
**XL** (this is the structural foundation for everything else — budget 1–2 weeks, most of it in re-plumbing `attacks.py`/`dataset.py` and regenerating the dataset).

---

## 2. Statistical Rigor Applied Uniformly (Critical, low effort relative to impact)

### Reviewer risk
"Bootstrap CIs and McNemar's test are applied to the 9-D feature set (Chapter 11) but *not* to the headline 5-D result in Chapter 9, where the central claim of the paper — Deep SAD (0.7959) beating Random Forest (0.7568) on zero-day AUROC — rests on a single train/test split with no uncertainty quantification at all. A 0.039 AUROC gap on ~3,300 test samples may well be noise."

### Current state
- Chapter 9 (5-D, the abstract's headline result): single point estimates, no CIs, no significance testing.
- Chapter 11 (9-D feature set): bootstrap 95% CIs (Table 11.2) and Holm-Bonferroni-corrected McNemar's test (Table 11.3) — done properly.
- All experiments across the entire report use **one fixed train/val/test split** (seed 42). No repeated-seed variance is reported anywhere.

### Target state
Every headline number in the paper — Ch. 9, Ch. 10, Ch. 11, Ch. 12 — carries a confidence interval, and every pairwise "model A beats model B" claim carries a significance test with multiple-comparison correction. Additionally, variance from *data generation and training stochasticity* (not just test-set sampling) is quantified.

### Concrete tasks

1. **Bootstrap the Chapter 9 results identically to Chapter 11's Table 11.2** — 1,000+ resamples of the test set, 95% percentile CIs on AUROC and TPR@4%FPR, for all six detectors × three attack classes.

2. **Add McNemar's test (Holm-Bonferroni corrected) for the Chapter 9 pairwise comparisons that the abstract actually leans on:**
   - Deep SAD vs. Random Forest (the "beats fully supervised RF" claim)
   - Deep SAD vs. XGBoost (the "trails XGBoost by only 0.03" claim)
   - Deep SAD vs. QSVM / VQC (the "substantially exceeds quantum baselines" claim)

3. **Multi-seed retraining.** This is the more important addition, because bootstrap CIs on a single test split only capture *test-sampling* variance, not variance from:
   - Different train/val/test partitions (reshuffling with different seeds),
   - Different network weight initializations for Deep SVDD/SAD, QSVM, VQC,
   - Stochastic optimizer behavior (Adam minibatch order).

   Protocol: repeat the **entire pipeline** (dataset split → feature extraction → all six detector trainings → evaluation) for **≥5 independent seeds** (e.g., 42, 123, 2024, 7, 999). Report every headline metric as **mean ± std across seeds**, and additionally run a paired t-test or Wilcoxon signed-rank test *across seeds* (not just across test samples) for the key model comparisons. This is a stronger and more standard form of evidence than single-split bootstrap CIs and directly preempts "how do we know this isn't a lucky/unlucky split" reviews.

4. **k-fold cross-validation as a secondary check**, given the sample-level dataset is modest (4,900 train / 3,299 val / 3,300 test *windows*, not raw pulses). A 5-fold CV over the pooled val+test windows (retraining supervised models each fold) will tighten the effective sample size story and is cheap to run given how fast the classical/deep models train (Table 9.1 latency figures suggest full retraining is seconds-to-minutes per model).

5. **Effect size reporting**, not just p-values — report Cohen's h (for AUROC proportion-like comparisons) or the AUROC difference with its CI, since p < 0.05 alone is a known point of criticism in ML venues (statistically significant but practically negligible differences).

### New artifacts
- Table 9-CI: bootstrap CIs for all Chapter 9 numbers (mirrors Table 11.2 exactly).
- Table 9-MN: McNemar's test results for Chapter 9 (mirrors Table 11.3).
- **New Table**: multi-seed summary — mean ± std AUROC/TPR across 5 seeds, for every model/attack/feature-set combination reported in the paper.
- **New Figure**: box/violin plots of AUROC across seeds per model, per attack class — visually communicates variance far better than a table.

### Effort
**M–L** (mostly compute time / orchestration scripting; the statistical machinery from Chapter 11 can be reused directly — this is largely "run the Chapter 11 recipe on Chapter 9's data").

---

## 3. Per-Subtype Zero-Day Breakdown

### Reviewer risk
"The four zero-day sub-types (PNI, CDA, DSI, CC) are physically very different — Table 5.1 itself shows PNI/DSI collapse the key rate to exactly zero everywhere, while CDA/CC produce subtle non-zero, near-nominal key-rate profiles. Yet every downstream AUROC number (Ch. 9–12) reports 'Zero-Day' as one pooled class. It's entirely possible the detector aces PNI/DSI (which are the 'easy' anomalies) and fails badly on CDA/CC (the 'hard' ones), and the pooled number is hiding this. This materially changes the operational conclusion."

### Current state
- `dataset.py` already stores sub-type assignments separately for leakage-prevention purposes (Section 4.2, "with sub-type assignments stored separately to prevent feature leakage") — the information needed for this analysis **already exists in the pipeline** and is simply not being used for evaluation breakdowns. This is a low-effort, high-value fix.

### Target state
Every zero-day AUROC/TPR number in the paper is reported both pooled *and* broken out per sub-type (PNI, CDA, DSI, CC individually), for every detector and every feature-set version (5-D, 9-D, 10-D).

### Concrete tasks

1. **Modify the evaluation script** to slice the zero-day test predictions by the already-stored sub-type label before computing AUROC/TPR, in addition to the pooled computation.

2. **Re-run for**: Chapter 9 (5-D, OOD experiment), Chapter 11 (9-D, full model comparison + bootstrap CIs), and Chapter 12 (adaptive study, if extended to zero-day sub-types per Section 7 below).

3. **Produce a per-subtype heatmap**: rows = detector, columns = {PNI, CDA, DSI, CC}, cell = AUROC. This single figure will likely be the most-cited figure in the paper because it directly exposes detector blind spots (recall Section 8.3's "inlier anomaly" discussion for TWIRL — CDA/CC plausibly exhibit similar pathologies given their near-nominal key-rate profiles in Table 5.1, and this needs to be checked explicitly rather than assumed).

4. **Cross-reference with Chapter 8's diagnostic framework.** If CDA/CC turn out to be "inlier anomalies" the way TWIRL was, this is a genuinely new and interesting finding — that the inlier-anomaly pathology isn't specific to TWIRL's phase-sweep mechanism but is a general property of any attack that *regularizes* rather than *randomizes* the channel statistics. Worth a dedicated discussion paragraph if confirmed.

### New artifacts
- **New Table**: per-subtype AUROC/TPR@4%FPR for every detector, at every feature-set stage.
- **New Figure**: detector × sub-type AUROC heatmap.
- Updated discussion in Ch. 9.3.2/Ch. 11 acknowledging which sub-types drive the pooled number.

### Effort
**S** (data already exists; this is almost purely an analysis/reporting task, no retraining required — do this immediately, it's the best effort-to-insight ratio in this entire plan).

---

## 4. Feature Importance / Interpretability for the 9-D→10-D Result

### Reviewer risk
"Chapter 10 asserts that 'encoding higher-order physical invariants into the feature space is a more robust route to generalization than deepening the anomaly-detection architecture' — but this is argued narratively from a single before/after AUROC comparison (0.7568 → 0.9444), not demonstrated. Which of the four new features (autocorrelation, cross-correlation, kurtosis, click asymmetry) is actually doing the work? Is it possible only one feature matters and the other three are noise?"

### Current state
No feature-importance analysis anywhere in the report, despite XGBoost natively supporting this at near-zero marginal cost.

### Target state
A quantitative decomposition of which features drive the zero-day generalization result, both for the supervised (XGBoost) and semi-supervised (Deep SAD) paradigms.

### Concrete tasks

1. **XGBoost native feature importance** (gain-based and split-count-based) for the 9-D and 10-D models, reported as a bar chart, specifically on the zero-day-holdout-trained model from Chapter 10/12.

2. **SHAP values** (TreeExplainer, essentially free for XGBoost) computed on the zero-day test set specifically — this shows not just *global* importance but *how each feature pushes individual zero-day windows toward "anomalous"*, which is a much stronger form of evidence than gain-based importance alone and is increasingly expected by reviewers for any "we added features and it worked" claim.

3. **Ablation study**: retrain XGBoost on the 9-D set with each of the 4 new features held out one at a time (leave-one-feature-out), plus the reverse (each new feature added individually to the original 5-D set). This directly answers "which feature(s) are necessary/sufficient" rather than relying on SHAP alone. Produces a clean table:

   | Feature set | Zero-day AUROC |
   |---|---|
   | Original 5-D | 0.8274 (baseline, from Ch.9) |
   | 5-D + autocorrelation only | ? |
   | 5-D + cross-correlation only | ? |
   | 5-D + kurtosis only | ? |
   | 5-D + click asymmetry only | ? |
   | 9-D minus autocorrelation | ? |
   | 9-D minus cross-correlation | ? |
   | 9-D minus kurtosis | ? |
   | 9-D minus click asymmetry | ? |
   | Full 9-D | 0.9444 (from Ch.10) |

4. **For Deep SAD**, since SHAP is less natural for the hypersphere-distance objective, use **integrated gradients** or a simpler **permutation-importance-on-anomaly-score** approach (shuffle one feature at a time in the test set, measure AUROC degradation) to produce an analogous ranking. This also directly probes the representation-overfitting hypothesis from Section 10.2: if permutation importance shows Deep SAD's anomaly score becomes *dominated* by 1–2 of the new features (rather than distributing across all 9), that's direct quantitative evidence for the "overfits to known-attack geometry" narrative currently only argued qualitatively.

5. **Tie this to physical interpretation.** For each feature that turns out to be important, write a paragraph connecting the SHAP/ablation result back to the physics (Section 10.1 already gives physical motivations per feature — close the loop by showing the *learned* importance matches the *intended* physical mechanism, e.g., "lag-1 phase autocorrelation, designed to capture TWIRL's deterministic sweep, indeed shows the highest SHAP magnitude on TWIRL-adjacent zero-day sub-types [CDA/CC if they involve periodic forcing], confirming the feature is functioning as physically intended.").

### New artifacts
- **New Figure**: SHAP summary plot (beeswarm) for XGBoost 9-D/10-D model on zero-day test set.
- **New Figure**: permutation importance for Deep SAD.
- **New Table**: leave-one-out / add-one-in ablation (above).
- Strengthened Section 10.2/10.3 narrative backed by quantitative evidence rather than assertion.

### Effort
**M** (SHAP/permutation importance are cheap given trained models already exist; the leave-one-out ablation requires 8 additional XGBoost retrains, trivial given Table 9.1's reported latency).

---

## 5. Window-Size and Architecture Ablations (Decoupling Confounded Variables)

### Reviewer risk
"Section 8.1 fixes the observation window at W=100,000 pulses after showing W=1,000 is shot-noise-dominated, but no intermediate values are explored — why 100,000 and not 10,000 or 50,000? Separately, Chapter 10 changes *both* the feature set (5-D→9-D) *and* implicitly discusses architecture capacity ('even after making the network shallower') when explaining Deep SAD's regression — these two changes are confounded. It's unclear whether representation overfitting is caused by the richer features, the architecture change, or an interaction between them."

### Current state
- Window size: only two values tested (1,000 and 100,000), justified by a back-of-envelope shot-noise calculation (Section 8.1) rather than an empirical sweep.
- Architecture: Chapter 7 uses R⁵→R¹⁶→R⁸→R⁴; Chapter 8.4.3 deepens to R⁵→R¹²⁸→R⁶⁴→R³²→R¹⁶ for the Deep SAD upgrade; Chapter 10.2 mentions "making the network shallower (reducing hidden dimensions to 32 and 16)" as a fix attempt for representation overfitting — three different architectures appear across the report without a controlled comparison isolating architecture from feature-set effects.

### Target state
1. A window-size sweep showing AUROC (and the QBER shot-noise variance directly) as a continuous function of W, empirically justifying the chosen operating point rather than asserting it from a single calculation.
2. A 2×2 (or more) factorial design: {5-D features, 9-D features} × {shallow architecture (16-8-4), deep architecture (128-64-32-16)}, fully crossed, to cleanly attribute the representation-overfitting effect (Section 10.2) to its true cause.

### Concrete tasks

1. **Window sweep**: W ∈ {1,000, 5,000, 10,000, 25,000, 50,000, 100,000, 200,000, 500,000}. For each, regenerate the feature-extracted dataset and re-run Deep SVDD/SAD + XGBoost + Random Forest, recording AUROC and TPR@4%FPR per attack class. Also directly report the empirical variance of the QBER feature per window size (validating the theoretical 1/n shot-noise scaling claimed in Eq. 8.2's discussion) — this turns the current back-of-envelope argument into an empirically validated one, and the AUROC-vs-W curve will show whether there's a *sweet spot* (too small → shot noise; too large → each window straddles multiple physical regimes / smooths out transient attack signatures — worth checking whether huge windows also hurt, which the current single-jump-to-100k methodology cannot reveal).

2. **Architecture × feature-set factorial**:
   - Shallow (16-8-4) + 5-D
   - Shallow (16-8-4) + 9-D
   - Deep (128-64-32-16) + 5-D
   - Deep (128-64-32-16) + 9-D
   - (optionally) an intermediate architecture (64-32-16) at both feature-set sizes for a finer capacity gradient.

   Report zero-day AUROC for each of the 4 (or 6) cells in a clean table. This directly tests Chapter 10's implicit hypothesis: if representation overfitting is a **feature-space** phenomenon, AUROC should drop 5-D→9-D *regardless* of architecture. If it's an **architecture-capacity** phenomenon, the deep architecture should underperform the shallow one *regardless* of feature set. The factorial design is the only way to distinguish these cleanly, and it's cheap since both networks train in what Table 9.1 suggests is well under a minute.

3. **Report training/validation loss curves** for each Deep SVDD/SAD configuration (currently the report only gives a single final-loss number, "≈3.2×10⁻⁵," for the original Deep SVDD in Section 7.1) — a loss curve, plus the anomaly-score distribution histogram (nominal vs. each attack class, overlaid) for each configuration, would let a reviewer visually verify the hypersphere-collapse and representation-overfitting claims rather than taking them on faith.

### New artifacts
- **New Figure**: AUROC vs. window size (multi-panel: one per detector, or overlaid with attack class as color).
- **New Figure**: empirical QBER-feature variance vs. window size, with the theoretical 1/n curve overlaid for validation.
- **New Table**: architecture × feature-set factorial AUROC grid.
- **New Figure**: anomaly-score histograms (nominal vs. attack) for each of the 4 factorial cells — directly visualizes representation overfitting / hypersphere collapse / inlier anomalies.

### Effort
**L** (window sweep requires re-extracting features at 8 window sizes — cheap per Table 9.1's latency numbers, but requires re-running the full raw-pulse dataset through the feature extractor 8 times; the factorial architecture study is cheap once the 5-D/9-D datasets already exist from Section 1's work).

---

## 6. Dataset Scale and Diversity

### Reviewer risk
"11.5M raw pulses sounds large, but after windowing (W=100,000) this collapses to only 4,900 training / 3,299 validation / 3,300 test *samples* — and worse, after Section 1's parameter randomization, this needs to cover a much larger attack-parameter space with the same or fewer effective samples per configuration. Is the dataset large enough to support the granularity of stratified/ablation analysis proposed above?"

### Current state
- 23 batches × 500,000 pulses = 11.5M raw pulses (Section 5.1), yielding 4,900/3,299/3,300 windowed samples at W=100,000 (Section 6.2).
- Single fixed distance (L=100 km) for the entire dataset; the distance sweep (Section 5.5) is a separate, much smaller validation-only exercise (30,000 pulses × 5 distances × 7 modes = only ~1M pulses, not windowed into a training set).

### Target state
A dataset sized and structured to support (a) the parameter-randomization of Section 1, (b) reliable per-subtype breakdowns (Section 3), and (c) a multi-distance generalization test (see below), without straining compute budgets unreasonably.

### Concrete tasks

1. **Scale up attack batches.** With parameter randomization (Section 1), you want enough distinct attack instances per class to populate a meaningful strength-stratified AUROC curve (Section 1's figure) and a reliable per-subtype breakdown (Section 3). Recommend increasing each attack class from 3 batches (1.5M pulses) to at least 8–10 batches (4–5M pulses) per class, and correspondingly increasing nominal batches to preserve the current class-balance ratios. This roughly doubles total raw pulses to ~20–25M — still tractable given the Numba-JIT compiled integrator (Section 3.4.3) already runs "within minutes on a standard CPU."

2. **Add a multi-distance training/test axis.** Currently every result in Ch. 9–12 is at a fixed L=100 km. Extend dataset generation to at least 3 distances (e.g., 50, 100, 150 km) so you can report:
   - **Within-distance generalization** (train and test at the same distance — current setup), and
   - **Cross-distance (OOD) generalization** (train at 100 km, test at 50/150 km) — this is a second, independent axis of "out-of-distribution" beyond attack-type novelty, and directly strengthens the "deployable real-world monitor" claim, since a real network won't have every node at exactly the same span. This reuses the exact same OOD *methodology* already built for Chapter 9, just swapping the held-out variable from attack-class to distance.

3. **Document dataset statistics thoroughly** in the paper's dataset section: class balance, attack-parameter distribution histograms (post Section 1), pulse count per split, window count per split, per-distance breakdown — this is standard "datasheet for datasets" practice increasingly expected by reviewers and costs almost nothing to produce given the metadata already persisted (Section 5.2 mentions "all metadata... is persisted for reproducibility").

4. **Consider releasing the dataset generation code and a fixed dataset snapshot** (or at minimum the generation scripts + seeds) alongside the paper — see Section 9 (Reproducibility) below.

### New artifacts
- **New Table**: dataset statistics / "datasheet" (class balance, pulse/window counts, parameter distribution summary, per-distance breakdown).
- **New Figure**: histograms of sampled attack parameters actually realized in the generated dataset (sanity-check that Section 1's intended distributions were correctly realized).
- **New Table**: cross-distance generalization results (train @100km, test @50/150km) for the strongest 2–3 detectors.

### Effort
**L–XL** (mostly compute time for regeneration; the multi-distance extension reuses existing simulator infrastructure directly, since distance is already a first-class parameter in `tf_qkd.py` per Section 3.2.2).

---

## 7. Extending the Adaptive Robustness Study (Chapter 12)

### Reviewer risk
"The adaptive attacker is evaluated against exactly one detector (Deep SAD) on exactly one attack (FIM), searching over exactly one parameter (frequency offset) via grid search. This is a weak threat model — the paper's own Future Work section (13.3) explicitly flags TWIRL/zero-day extension and a richer attack surface as not-yet-done, and grid search is a much weaker adversary than what the adversarial-ML literature (your own citations [16], [17]) treats as standard."

### Current state
- Chapter 12: FIM only, Deep SAD only, grid search over frequency offset at 5 fixed modulation depths, evaluated at one distance (100 km).
- The 10-D spectral countermeasure closes the FIM blind spot, but per the report's own honest caveat, "this should not be read as a general robustness guarantee against attackers who adapt to the spectral feature itself" — an open question the report explicitly leaves unresolved.

### Target state
A broader, more standard adversarial-robustness evaluation: multiple attacks, multiple detectors, a stronger (gradient-based) attacker, and — critically — a test of whether the spectral-feature countermeasure itself can be adaptively evaded (closing the report's own explicitly flagged open question).

### Concrete tasks

1. **Extend to TWIRL and the strongest zero-day sub-type** (likely CDA or CC, given Table 5.1's near-nominal key-rate profiles — these are your "hardest to catch" attacks and therefore the most interesting adaptive-attack targets). Grid/gradient search over each attack's own native parameters (sweep range/period for TWIRL; amplitude/frequency for CDA; boost/duty for CC) rather than reusing FIM's frequency axis.

2. **Extend to XGBoost as a second target detector.** Post-Chapter-11, XGBoost is your best-performing model on the 9-D/10-D feature set — an adaptive attacker who doesn't also test against the *actual best detector* is testing a strawman. This also lets you compare adaptive-robustness properties *across paradigms* (hypersphere-distance vs. tree-ensemble decision boundaries), which is a genuinely novel angle not covered by prior QKD-ML security work per your own Related Work chapter.

3. **Implement a gradient-based adaptive attack**, not just grid search, since Deep SAD's anomaly score is differentiable end-to-end w.r.t. the attack parameters (through the Lang-Kobayashi simulator, if it's differentiable/JIT-friendly — if the simulator itself isn't differentiable, use a differentiable *surrogate*: fit a small neural net or Gaussian process regressor mapping attack parameters → feature vector → anomaly score, and run PGD/Adam-based optimization against that surrogate, then validate the discovered "blind spot" parameters against the real simulator). This is the standard methodology in the adversarial robustness literature your own citations [16], [17] draw from, and a grid search over 4–5 points will read as under-powered by comparison.

4. **Test adaptive evasion of the spectral-feature countermeasure itself** (directly closing the caveat left open at the end of Chapter 12): does an attacker who knows the 10-D detector includes an FFT-based max-spectral-power feature have a way to spread the attack's spectral energy (e.g., chirped/frequency-hopped modulation instead of single-tone) to stay below the spectral feature's detection threshold while still neutralizing the key rate? This is a natural and important extension — if the countermeasure can itself be adaptively evaded, that's an important (and honestly reportable) negative result; if it can't (within a reasonable attacker budget), that's a much stronger robustness claim than currently supported.

5. **Report results as a standard robustness curve**: anomaly score / detection rate vs. attacker "budget" (e.g., number of grid/gradient-search iterations, or compute time), rather than only a handful of discrete table rows — this is the standard presentation in the adversarial ML literature and makes the "evaded / caught" table (Tables 12.1–12.2) much more informative.

### New artifacts
- **New Table/Figure**: adaptive-attack results across {FIM, TWIRL, best zero-day subtype} × {Deep SAD, XGBoost} × {9-D, 10-D feature sets}.
- **New Figure**: gradient-based attacker convergence / robustness curve.
- **New subsection**: adaptive evasion attempt against the spectral countermeasure itself, with an honest reporting of success/failure.
- Updated Discussion/Limitations section reflecting the (likely more nuanced) real robustness picture.

### Effort
**L** (grid-search extensions are cheap reuses of Chapter 12's existing methodology; the gradient-based/surrogate-model attacker is the most novel engineering work here and the main driver of the L effort estimate).

---

## 8. Physical/Domain Generalization Beyond Attack Type

### Reviewer risk
"Every experiment uses a single fixed set of laser parameters (Table 3.2: α=3.0, γp=1e12, γe=1e9, etc.) and a single fixed distance in the main results. Given the paper is framed around laser physics as much as ML, a physics reviewer will ask whether the detector generalizes across plausible variation in real DFB laser characteristics, not just across attack type."

### Current state
- Table 3.2 fixes all laser parameters as constants for the entire study; no sensitivity analysis or cross-laser-parameter generalization test exists anywhere in the report.
- The only "physics generalization" already present is the distance sweep in Section 5.5 (validation-only, not a trained-and-tested OOD axis) and the locking-bandwidth sweep in Section 3.4.4 (a sanity check on the nominal simulator, not on attack detection).

### Target state
At least one experiment demonstrating (or honestly characterizing the limits of) generalization across realistic laser-parameter variation, in addition to the multi-distance extension already proposed in Section 6.

### Concrete tasks

1. **Identify the 2–3 most operationally variable laser parameters** across real DFB lasers used in QKD systems (likely candidates: linewidth enhancement factor α, which varies significantly across laser designs typically in the range 2–6; injection ratio ηinj, which is a deployment/engineering choice rather than a fixed physical constant; frequency detuning Δf, which drifts with temperature in real systems).

2. **Generate a small supplementary dataset** varying one parameter at a time (e.g., α ∈ {2.0, 2.5, 3.0, 3.5, 4.0, 5.0}, holding everything else at Table 3.2's nominal values) at the standard 100 km distance, for nominal + FIM + TWIRL (the two known attacks are enough for this check; doesn't need the full zero-day family).

3. **Evaluate the Chapter 9/11 detectors, trained only at the nominal α=3.0, against test data generated at each off-nominal α** — this is a direct, cheap OOD-generalization test along a *physical* axis rather than an *attack-taxonomy* axis, and a natural complement to the paper's existing OOD framing.

4. **If generalization degrades significantly**, this motivates (and the paper can propose as future work, or better, actually implement if time allows) either (a) domain-randomization during training — training on a *mixture* of α values rather than a single nominal value, analogous to Section 1's attack-parameter randomization, or (b) an explicit "laser-parameter calibration" preprocessing step that normalizes features by known device characteristics before feeding the detector.

### New artifacts
- **New Table/Figure**: detector AUROC vs. off-nominal laser parameter (α, ηinj, or Δf drift), for the 2 known attacks, at fixed distance.
- A short new **Discussion** paragraph honestly characterizing whether the detector is a "per-device-calibrated" monitor or a "universally transferable" one — this level of honesty about scope is exactly what turns a borderline review into an accept, since overclaiming generality is a much bigger risk than a well-characterized limitation.

### Effort
**M** (reuses Section 6's multi-distance infrastructure pattern; only 2 known attacks needed, not the full six-attack taxonomy, keeping this comparatively cheap).

---

## 9. Reproducibility, Fairness Caveats, and Presentation Polish

### Reviewer risk
"No code/data release is mentioned. The quantum baselines are trained on a small, undisclosed-as-a-caveat subsample (1,000–2,000 points vs. millions available to classical models) — buried in Section 7.2 rather than flagged as a fairness caveat where quantum results are first presented. No precision-recall analysis is given despite a real deployment being a highly imbalanced (mostly-nominal) setting where PR curves are more informative than ROC/AUROC."

### Current state
- No repository/code-release statement anywhere in the report.
- QSVM subsampled to 1,000 points, VQC to 2,000 points (Sections 7.2.1, 7.2.2) — mentioned only as an implementation detail ("for computational feasibility"), not flagged as a comparison-fairness caveat at the point results are first presented (Chapter 9).
- Only AUROC and TPR@4%FPR reported throughout — no precision-recall curves, despite Section 6.1 itself motivating the 4%-FPR operating point via a *base-rate-sensitive* security argument (finite-key SNS TF-QKD), which is exactly the situation where PR curves matter more than ROC/AUROC.
- Related Work (Chapter 2) is comparatively thin (roughly one page) relative to the depth of the experimental apparatus — this is a common and easily-fixed reviewer complaint ("insufficient positioning against prior QKD-ML security work").

### Concrete tasks

1. **Prepare a public code repository** (GitHub or institutional archive) containing: the `simulator` package (config/lk_dynamics/attacks/tf_qkd/dataset modules), the 25-test unit suite (Section 5.4), the training/evaluation pipeline, and either the generated dataset itself (if size permits — else the generation script + all seeds needed to reproduce it exactly, given Section 3.4.3 already documents seed=42 as controlling all split/subtype/partition determinism) and trained model weights for the headline detectors. Include a `requirements.txt`/environment file pinning exact versions of Numba, XGBoost, scikit-learn, and whatever quantum-computing framework (PennyLane/Qiskit) was used for QSVM/VQC — this single addition meaningfully de-risks reproducibility concerns, which are an increasingly common explicit review criterion.

2. **Move the quantum-baseline subsampling caveat up-front.** Add an explicit sentence in the Chapter 9 results discussion (Section 9.3.1) the first time QSVM/VQC numbers are presented: "QSVM and VQC are trained on subsamples of 1,000/2,000 points respectively due to the O(n²) kernel-evaluation cost of the quantum kernel, versus the full training+validation set available to classical baselines; this asymmetry should be kept in mind when comparing quantum and classical AUROC figures directly." This single-sentence fix removes an easy "the comparison isn't fair" objection.

3. **Add precision-recall curves and PR-AUC** for the top 2–3 detectors (Deep SAD, XGBoost, Random Forest) at each feature-set stage, alongside the existing ROC curves (Figure 11.1's counterpart). Given the extreme class imbalance in a real deployment (the paper's own Section 6.1 argues for the 4%-FPR operating point specifically because of a finite-key security cost of false positives), a PR curve materially strengthens the "operationally deployable" claim in a way ROC/AUROC alone cannot.

4. **Expand Related Work** into a proper comparison table against citations [11]–[15] (the closest prior QKD-ML-security work), with columns: {protocol type (DV/CV/TF), feature type (discrete detection-event stats vs. continuous telemetry), attack types considered, zero-day/OOD generalization tested (Y/N), latency reported (Y/N)}. This both strengthens positioning and makes the "gap this paper fills" claim (currently asserted in prose at the end of Chapter 2) visually self-evident.

5. **Confidence/calibration check for the anomaly scores**, not just discrimination (AUROC only measures ranking, not calibration) — a reliability diagram or Brier score for whichever detector is proposed as the deployment candidate would strengthen the "sub-millisecond, deployable" operational claim, since a real system needs a calibrated threshold, not just a well-ranked score.

6. **Report compute/hardware environment** for all latency numbers (Table 9.1, 9.2, 11.1) — currently "sub-millisecond" and "≈750ms for QSVM" are reported with no CPU/GPU spec, which a reviewer will ask for since latency claims are hardware-dependent and currently unverifiable/unreproducible as stated.

### New artifacts
- Public repository link + reproducibility statement (new subsection, likely in Limitations or a dedicated "Reproducibility" section).
- Updated Section 9.3.1 with the fairness caveat inline.
- **New Figure**: precision-recall curves (companion to Figure 11.1).
- **New Table**: related-work comparison matrix.
- **New Figure/Table**: calibration reliability diagram + Brier scores for the top 2–3 detectors.
- Hardware/environment specification added to all latency tables' captions or a methods subsection.

### Effort
**M** (mostly writing/packaging effort; PR curves and calibration diagrams are cheap given predictions already exist from prior experiments; the related-work table requires re-reading citations [11]–[15] carefully but no new experiments).

---

## 10. Suggested Revised Paper Structure

Given all of the above, here's how the current chapter structure maps onto a strengthened paper outline. New/heavily-revised sections are marked **[NEW]**.

1. **Introduction** (largely as-is; Sections 1.1–1.3)
2. **Related Work** — **[EXPANDED]** with comparison table (Section 9, task 4 above)
3. **Simulation Framework** (as-is; Chapter 3)
4. **Attack Taxonomy** — **[REVISED]** to present parameter *distributions*, not fixed defaults (Section 1 above)
5. **Dataset Generation and Validation** — **[EXPANDED]** with dataset "datasheet," parameter-distribution histograms, multi-distance generation (Sections 1, 6 above)
6. **Feature Extraction and Evaluation Protocol** — **[EXPANDED]** with the window-size sweep as a pre-registered, empirically-justified choice rather than a single calculation (Section 5 above)
7. **Detector Architectures** (largely as-is; Chapter 7)
8. **Diagnostic Failure Analysis and the Deep SAD Upgrade** (as-is; Chapter 8)
9. **Out-of-Distribution Generalization Experiment** — **[STRENGTHENED]** with bootstrap CIs + McNemar's test + multi-seed variance (Section 2), and per-subtype breakdown (Section 3)
10. **Feature Engineering and True OOD Generalization** — **[STRENGTHENED]** with SHAP/permutation feature importance and the architecture × feature-set factorial ablation, cleanly separating "it's the features" from "it's the architecture" (Sections 4, 5)
11. **Statistical Validation** — merge with strengthened Section 9/10 rigor above rather than keeping as a separate afterthought chapter; make CIs/significance testing a running practice throughout, not a dedicated late chapter
12. **Physical/Cross-Distance Generalization** — **[NEW]** dedicated section for the multi-distance and laser-parameter generalization studies (Sections 6, 8 above)
13. **Adaptive Robustness Study** — **[EXPANDED]** to multiple attacks, multiple detectors, gradient-based attacker, and adaptive evasion of the countermeasure itself (Section 7 above)
14. **Discussion, Limitations, and Future Work** — updated to reflect what's now actually done vs. genuinely left open (dual-use caveat as-is; reproducibility statement moved here or to its own section)
15. **Conclusion**
16. **Reproducibility Statement** — **[NEW]** (Section 9 above)

---

## 11. Prioritized Execution Order (if time-constrained)

If you cannot do everything above before a submission deadline, here is the order that maximizes review-risk reduction per unit effort:

1. **Section 3 (per-subtype zero-day breakdown)** — S effort, data already exists, immediately closes a real gap.
2. **Section 9, tasks 2–3 (fairness caveat sentence + PR curves)** — S–M effort, cheap wins using existing predictions.
3. **Section 2 (statistical rigor on Chapter 9 + multi-seed reruns)** — M–L effort, directly defends the headline claim.
4. **Section 4 (feature importance / SHAP / ablation)** — M effort, converts a narrative claim into quantitative evidence.
5. **Section 1 (attack parameter diversity)** — XL effort, but the single most important structural fix; do this before finalizing any paper draft, since it changes what every other number in the paper actually means. If truly time-constrained, a reduced version (randomize just 2 parameters for just FIM/TWIRL/one zero-day subtype, per-batch not per-window) is far better than skipping this entirely.
6. **Section 5 (window-size sweep + architecture factorial)** — L effort, strengthens methodology defensibility.
7. **Section 7 (extended adaptive robustness)** — L effort, high value if adversarial robustness is emphasized in the target venue.
8. **Section 6 (dataset scale) and Section 8 (physical generalization)** — L–XL effort each; valuable but most deferrable to a follow-up paper or extended/journal version if page/time budget is tight.
9. **Section 9, tasks 1, 4–6 (repo release, related-work table, calibration, hardware specs)** — spread throughout final writing pass.

---

## 12. Summary Checklist

- [ ] Attack parameters randomized per-batch or per-window, dataset regenerated (Sec. 1)
- [ ] Bootstrap CIs + McNemar's test applied to Chapter 9 exactly as done in Chapter 11 (Sec. 2)
- [ ] ≥5-seed retraining with mean±std reported for every headline number (Sec. 2)
- [ ] Per-subtype (PNI/CDA/DSI/CC) zero-day breakdown for all detectors/feature-sets (Sec. 3)
- [ ] SHAP + permutation feature importance for the 9-D/10-D result (Sec. 4)
- [ ] Leave-one-feature-out / add-one-feature-in ablation table (Sec. 4)
- [ ] Window-size sweep (empirical, not just theoretical justification) (Sec. 5)
- [ ] Architecture × feature-set factorial to decouple representation overfitting causes (Sec. 5)
- [ ] Anomaly-score distribution histograms per configuration (Sec. 5)
- [ ] Dataset scaled up + "datasheet" table produced (Sec. 6)
- [ ] Multi-distance training/testing (cross-distance OOD) (Sec. 6)
- [ ] Adaptive study extended to TWIRL + best zero-day subtype (Sec. 7)
- [ ] Adaptive study extended to XGBoost as a second target (Sec. 7)
- [ ] Gradient-based (not just grid-search) adaptive attacker (Sec. 7)
- [ ] Adaptive evasion attempt against the spectral countermeasure itself (Sec. 7)
- [ ] Laser-parameter (α / ηinj / Δf) generalization study (Sec. 8)
- [ ] Public code/data repository + reproducibility statement (Sec. 9)
- [ ] Quantum-baseline subsampling fairness caveat moved inline (Sec. 9)
- [ ] Precision-recall curves + PR-AUC (Sec. 9)
- [ ] Related-work comparison table vs. citations [11]–[15] (Sec. 9)
- [ ] Calibration reliability diagrams / Brier scores (Sec. 9)
- [ ] Hardware/environment specs added to all latency claims (Sec. 9)

---

*This plan is intended to be worked through iteratively — Sections 1–4 form the highest-priority core; Sections 5–9 substantially raise the ceiling on both rigor and the paper's perceived thoroughness. Each section's "New artifacts" list maps directly onto figures/tables you can build incrementally and slot into the existing chapter structure without needing to rewrite the report from scratch.*
