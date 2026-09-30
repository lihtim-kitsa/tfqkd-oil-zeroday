# Phase 5: Adaptive Robustness Study

## Overview
This document evaluates the robustness of the trained Deep SAD anomaly detection model against an **Adaptive Attacker**. The Deep SAD model was trained on 100 km data using a semi-supervised objective, meaning it explicitly learned to map the *default* Frequency Intensity Modulation (FIM) attack outside of its nominal hypersphere. 

The goal of this study is to answer a critical security reviewer question: **"What happens if the attacker knows the Deep SAD decision boundary and adapts their attack?"**

## Methodology
We implemented an adaptive attacker (`adaptive_attacker.py`) that systematically searches for a FIM parameter configuration (a "blind spot") that successfully evades detection while still neutralizing the QKD key rate.

- **Objective:** Minimize the Deep SAD anomaly score.
- **Constraint:** The resulting secure key rate must be exactly 0 (complete denial of key).
- **Optimization Strategy:** A grid search over the FIM frequency offset (`fim_delta_f_hz`) for varying attacker strengths (`fim_mod_depth` from 0.1 to 0.4).
- **Simulation:** Evaluated at 100 km, generating 100,000 laser pulses per optimization step to precisely match the feature extraction window used during model training.

## Results

The Deep SAD Nominal 4% False Positive Rate (FPR) Threshold was determined to be **`0.4229`**. Any FIM attack resulting in an anomaly score below this threshold successfully evades the detector.

| Attacker Strength (`fim_mod_depth`) | Optimal Frequency Offset | Anomaly Score | Threshold | Evaded? | Secure Key Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.0 (Nominal)** | 0.00 MHz | 0.1632 | 0.4229 | N/A | 0.0000e+00 |
| **0.1 (Weak)** | 30.00 MHz | 6.4742 | 0.4229 | ❌ Caught | 7.0659e-04 |
| **0.2 (Moderate)** | 30.00 MHz | 0.0564 | 0.4229 | ✅ **Evaded** | 0.0000e+00 |
| **0.3 (Strong)** | 50.00 MHz | 0.0110 | 0.4229 | ✅ **Evaded** | 0.0000e+00 |
| **0.4 (Extreme)** | 30.00 MHz | 1.7487 | 0.4229 | ❌ Caught | 0.0000e+00 |

### Tradeoff Analysis
![Adaptive Tradeoff Plot](results/adaptive_tradeoff.png)

## Conclusion
The results show a critical vulnerability in the Deep SAD model when facing an adaptive attacker:
1. **Blind Spots Exist:** At specific combinations of modulation depth and frequency offsets (e.g., Mod Depth 0.2/0.3 and Offset 30/50 MHz), the extracted 9D feature vectors actually fall *closer* to the center of the Deep SAD hypersphere than standard nominal data (Anomaly scores of `0.0564` and `0.0110` vs the nominal `0.1632`). 
2. **Key Rate Degradation:** Despite the features appearing perfectly nominal to the detector, the attack completely destroyed the secure key generation, driving it to exactly zero. 

By adapting its parameters, the FIM attacker effectively bypasses the ML security layer. This highlights the fundamental limitation of using hypersphere-based anomaly detection against a knowledgeable adversary in continuous-variable quantum channels.
