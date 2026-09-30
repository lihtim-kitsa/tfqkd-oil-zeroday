import numpy as np

# Protocol Definition: Phase-Matching (PM) TF-QKD variant
# This variant relies on random phase slices and post-selecting matching/anti-matching phases.

# Decoy-state probabilities
P_MU = 0.6  # Probability of sending signal state (mu)
P_NU = 0.3  # Probability of sending decoy state (nu)
P_0 = 0.1   # Probability of sending vacuum state (0)

# Intensities
MU = 0.5    # Signal intensity (photons/pulse)
NU = 0.1    # Decoy intensity
OMEGA = 0.0 # Vacuum intensity

# Phase Randomization
NUM_PHASE_SLICES = 16
PHASE_SLICES = np.linspace(0, 2 * np.pi, NUM_PHASE_SLICES, endpoint=False)

# Detector Parameters (Charlie)
DETECTOR_EFFICIENCY = 0.5  # eta_d
DARK_COUNT_RATE = 1e-8     # p_d per pulse

# Fiber Parameters
ATTENUATION_COEF = 0.2     # dB/km (standard telecom fiber)

# Simulation scale (default number of pulses)
DEFAULT_N_PULSES = 10_000_000

# Error Correction Efficiency
F_EC = 1.15

# --- Optical Injection Locking (OIL) Parameters ---
# Lang-Kobayashi baseline parameters for a typical semiconductor DFB laser
LK_ALPHA = 3.0           # Linewidth enhancement factor
LK_GAMMA_P = 1e12        # Photon decay rate (s^-1) ~ 1 ps lifetime
LK_GAMMA_E = 1e9         # Carrier decay rate (s^-1) ~ 1 ns lifetime
LK_G_N = 1e-4            # Differential gain (s^-1)
LK_N0 = 1e7              # Transparency carrier number

# Injection parameters
INJ_RATIO_DB = -10.0     # Injection ratio (dB)
INJ_RATIO = 10 ** (INJ_RATIO_DB / 10.0)

# The coupling rate kappa is proportional to 1 / (2 * tau_in) where tau_in is the round-trip time.
# For simplicity, we define a coupling rate that results in ~500 MHz locking bandwidth at -10dB.
LK_KAPPA = 1e11          # Coupling coefficient (s^-1)

LK_DETUNING_HZ = 10e6    # Initial frequency detuning (10 MHz)

# Noise parameters
# White frequency noise corresponding to intrinsic linewidth
# Spontaneous emission rate R_sp
LK_R_SP = 1e12           

