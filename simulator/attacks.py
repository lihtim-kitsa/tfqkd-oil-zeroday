"""
simulator/attacks.py
====================
Attack-mode constants and parameter definitions for the OIL-TF-QKD simulator.

Attack taxonomy
---------------
NOMINAL  (0) : Clean OIL dynamics — training data for Deep SVDD.
FIM      (1) : Frequency/Intensity Manipulation — sinusoidal injection-amplitude
               modulation + static frequency offset. Seen by supervised baselines.
TWIRL    (2) : Master-frequency sweep across the locking bandwidth. Seen by
               supervised baselines.
--- Zero-Day Family (label 3 in dataset, never seen during training) ---
PNI      (3) : Phase-Noise Injection — additive Wiener-process phase perturbation
               on the slave field.
CDA      (4) : Carrier Density Attack — sinusoidal perturbation injected directly
               into carrier density N.
DSI      (5) : Double-Sideband Injection — two-tone injection at ±Δf around the
               nominal carrier frequency.
CC       (6) : Coherence Collapse — sudden injection-ratio spike that pushes the
               slave laser to the coherence-collapse boundary.

In the dataset, ZD sub-types (3-6) are all assigned label=3 (zero-day family).
Sub-type assignments are stored separately in results/zeroday_subtypes_test.npy
for reproducibility without leaking information to the model.
"""

import numpy as np

# ── Attack mode integers ──────────────────────────────────────────────────────
NOMINAL = 0
FIM     = 1
TWIRL   = 2
PNI     = 3   # ZD-A
CDA     = 4   # ZD-B
DSI     = 5   # ZD-C
CC      = 6   # ZD-D
CFI     = 7   # ZD-E
CAM     = 8   # ZD-F
DPJ     = 9   # ZD-G
SAM     = 10  # ZD-H
RTN     = 11  # ZD-I
MTC     = 12  # ZD-J

# Human-readable names for logging / plotting
ATTACK_NAMES = {
    NOMINAL: "Nominal",
    FIM:     "FIM",
    TWIRL:   "TWIRL",
    PNI:     "PNI (ZD-A)",
    CDA:     "CDA (ZD-B)",
    DSI:     "DSI (ZD-C)",
    CC:      "CC  (ZD-D)",
    CFI:     "CFI (ZD-E)",
    CAM:     "CAM (ZD-F)",
    DPJ:     "DPJ (ZD-G)",
    SAM:     "SAM (ZD-H)",
    RTN:     "RTN (ZD-I)",
    MTC:     "MTC (ZD-J)",
}

# Zero-day sub-type IDs (internal modes; mapped to label=3 in the dataset)
ZERO_DAY_SUBTYPES = [PNI, CDA, DSI, CC, CFI, CAM, DPJ, SAM, RTN, MTC]

# ── Default attack parameters ─────────────────────────────────────────────────
# These are "strong but plausible" defaults — enough to clearly degrade QKD
# performance while staying within the locking-range physics.

FIM_DEFAULTS = dict(
    fim_mod_depth   = 0.4,      # fractional amplitude modulation depth (0–1)
    fim_f_mod_hz    = 50e6,     # modulation frequency (50 MHz)
    fim_delta_f_hz  = 30e6,     # additional static frequency offset (30 MHz)
)

TWIRL_DEFAULTS = dict(
    twirl_sweep_hz  = 200e6,    # total sweep range (Hz) across the locking bandwidth
    twirl_T_sweep_s = 100e-9,   # sweep period (100 ns → back-and-forth)
)

PNI_DEFAULTS = dict(
    pni_sigma       = 0.5,      # phase noise std (rad/√step); scales Wiener increment
)

CDA_DEFAULTS = dict(
    cda_amplitude   = 0.1,      # fractional carrier-density perturbation amplitude
    cda_freq_hz     = 80e6,     # perturbation frequency (80 MHz)
)

DSI_DEFAULTS = dict(
    dsi_delta_hz    = 120e6,    # sideband offset from carrier (Hz)
    dsi_ratio       = 0.5,      # power ratio of each sideband vs. main tone (0–1)
)

CC_DEFAULTS = dict(
    cc_ratio_boost  = 8.0,      # injection-ratio multiplier during collapse burst
    cc_onset_frac   = 0.3,      # fraction of pulse interval at which burst starts
    cc_duty_frac    = 0.2,      # fraction of pulse interval for which burst lasts
)

CFI_DEFAULTS = dict(
    cfi_sweep_hz    = 50e6,
    cfi_T_sweep_s   = 1e-9,
)

CAM_DEFAULTS = dict(
    cam_sigma       = 0.05,
)

DPJ_DEFAULTS = dict(
    dpj_phase_shift = np.pi/2,
    dpj_probability = 0.01,
)

SAM_DEFAULTS = dict(
    sam_amplitude   = 0.4,
    sam_freq_hz     = 50e6,
)

RTN_DEFAULTS = dict(
    rtn_amplitude   = 0.3,
    rtn_flip_prob   = 0.05,
)

MTC_DEFAULTS = dict(
    mtc_delta_hz    = 50e6,
    mtc_ratio       = 0.2,
)

# All defaults merged into one flat dict for convenient unpacking
ALL_DEFAULTS = {
    **FIM_DEFAULTS,
    **TWIRL_DEFAULTS,
    **PNI_DEFAULTS,
    **CDA_DEFAULTS,
    **DSI_DEFAULTS,
    **CC_DEFAULTS,
    **CFI_DEFAULTS,
    **CAM_DEFAULTS,
    **DPJ_DEFAULTS,
    **SAM_DEFAULTS,
    **RTN_DEFAULTS,
    **MTC_DEFAULTS,
}


def sample_zero_day_mode(rng: np.random.Generator | None = None) -> int:
    """
    Randomly draws one zero-day sub-type mode integer from {PNI, CDA, DSI, CC}.

    Parameters
    ----------
    rng : np.random.Generator, optional
        If provided, uses this generator for reproducibility.

    Returns
    -------
    int
        One of ZERO_DAY_SUBTYPES.
    """
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(ZERO_DAY_SUBTYPES))


def attack_params_for_mode(mode: int) -> dict:
    """
    Returns the default parameter dict for a given attack mode.
    Only the parameters relevant to that mode are included.

    Parameters
    ----------
    mode : int
        One of the attack-mode constants above.

    Returns
    -------
    dict
        Keyword arguments suitable for passing to solve_lk_sde / TFQKDProtocol.run.
    """
    if mode == FIM:
        return {**FIM_DEFAULTS}
    elif mode == TWIRL:
        return {**TWIRL_DEFAULTS}
    elif mode == PNI:
        return {**PNI_DEFAULTS}
    elif mode == CDA:
        return {**CDA_DEFAULTS}
    elif mode == DSI:
        return {**DSI_DEFAULTS}
    elif mode == CC:
        return {**CC_DEFAULTS}
    elif mode == CFI:
        return {**CFI_DEFAULTS}
    elif mode == CAM:
        return {**CAM_DEFAULTS}
    elif mode == DPJ:
        return {**DPJ_DEFAULTS}
    elif mode == SAM:
        return {**SAM_DEFAULTS}
    elif mode == RTN:
        return {**RTN_DEFAULTS}
    elif mode == MTC:
        return {**MTC_DEFAULTS}
    else:
        return {}  # NOMINAL — no extra params

def sample_attack_params(mode: int, rng: np.random.Generator = None) -> dict:
    """
    Randomly draws parameters for a given attack mode based on predefined distributions.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    if mode == FIM:
        return dict(
            fim_mod_depth   = rng.uniform(0.05, 0.5),
            fim_f_mod_hz    = np.exp(rng.uniform(np.log(10e6), np.log(150e6))),
            fim_delta_f_hz  = rng.uniform(5e6, 100e6)
        )
    elif mode == TWIRL:
        return dict(
            twirl_sweep_hz  = rng.uniform(50e6, 450e6),
            twirl_T_sweep_s = np.exp(rng.uniform(np.log(20e-9), np.log(500e-9)))
        )
    elif mode == PNI:
        return dict(
            pni_sigma       = np.exp(rng.uniform(np.log(0.05), np.log(1.5)))
        )
    elif mode == CDA:
        return dict(
            cda_amplitude   = rng.uniform(0.02, 0.3),
            cda_freq_hz     = np.exp(rng.uniform(np.log(10e6), np.log(200e6)))
        )
    elif mode == DSI:
        return dict(
            dsi_delta_hz    = rng.uniform(20e6, 300e6),
            dsi_ratio       = rng.uniform(0.05, 0.9)
        )
    elif mode == CC:
        return dict(
            cc_ratio_boost  = np.exp(rng.uniform(np.log(2.0), np.log(20.0))),
            cc_onset_frac   = rng.uniform(0.1, 0.5),
            cc_duty_frac    = rng.uniform(0.05, 0.5)
        )
    elif mode == CFI:
        return dict(
            cfi_sweep_hz    = rng.uniform(20e6, 150e6),
            cfi_T_sweep_s   = np.exp(rng.uniform(np.log(0.5e-9), np.log(5e-9)))
        )
    elif mode == CAM:
        return dict(
            cam_sigma       = np.exp(rng.uniform(np.log(0.01), np.log(0.15)))
        )
    elif mode == DPJ:
        return dict(
            dpj_phase_shift = rng.uniform(np.pi/4, np.pi),
            dpj_probability = rng.uniform(0.005, 0.05)
        )
    elif mode == SAM:
        return dict(
            sam_amplitude   = rng.uniform(0.1, 0.5),
            sam_freq_hz     = np.exp(rng.uniform(np.log(10e6), np.log(150e6)))
        )
    elif mode == RTN:
        return dict(
            rtn_amplitude   = rng.uniform(0.1, 0.5),
            rtn_flip_prob   = rng.uniform(0.01, 0.1)
        )
    elif mode == MTC:
        return dict(
            mtc_delta_hz    = rng.uniform(20e6, 150e6),
            mtc_ratio       = rng.uniform(0.05, 0.5)
        )
    else:
        return {}
