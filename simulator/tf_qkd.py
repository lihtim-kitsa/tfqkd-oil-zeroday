import numpy as np
from simulator.config import *
from simulator.lk_dynamics import solve_lk_sde
from simulator.attacks import ALL_DEFAULTS

class Transmitter:
    def __init__(self, name, use_oil=False, alpha=LK_ALPHA):
        self.name = name
        self.use_oil = use_oil
        self.alpha = alpha

    def generate_pulses(self, n_pulses, attack_mode=0, **attack_params):
        """
        Generate intensity and phase values for n_pulses.

        Parameters
        ----------
        n_pulses : int
        attack_mode : int
            0=Nominal (default), 1=FIM, 2=TWIRL, 3=PNI, 4=CDA, 5=DSI, 6=CC.
        **attack_params
            Override any attack parameter from attacks.ALL_DEFAULTS.
        """
        # 1. Randomly choose intensity
        rands = np.random.rand(n_pulses)
        intensities = np.zeros(n_pulses)
        intensities[rands < P_0] = OMEGA
        intensities[(rands >= P_0) & (rands < P_0 + P_NU)] = NU
        intensities[rands >= P_0 + P_NU] = MU

        # 2. Randomly choose phase from discrete slices
        phases = np.random.choice(PHASE_SLICES, size=n_pulses)

        if self.use_oil:
            # Merge defaults with any caller overrides
            lk_kwargs = {**ALL_DEFAULTS, **attack_params}
            phase_errors = solve_lk_sde(
                n_pulses,
                alpha=self.alpha,
                gamma_p=LK_GAMMA_P,
                gamma_e=LK_GAMMA_E,
                g_n=LK_G_N,
                N0=LK_N0,
                kappa=LK_KAPPA,
                detuning_hz=LK_DETUNING_HZ,
                R_sp=LK_R_SP,
                inj_ratio=INJ_RATIO,
                attack_mode=attack_mode,
                fim_mod_depth=lk_kwargs["fim_mod_depth"],
                fim_f_mod_hz=lk_kwargs["fim_f_mod_hz"],
                fim_delta_f_hz=lk_kwargs["fim_delta_f_hz"],
                twirl_sweep_hz=lk_kwargs["twirl_sweep_hz"],
                twirl_T_sweep_s=lk_kwargs["twirl_T_sweep_s"],
                pni_sigma=lk_kwargs["pni_sigma"],
                cda_amplitude=lk_kwargs["cda_amplitude"],
                cda_freq_hz=lk_kwargs["cda_freq_hz"],
                dsi_delta_hz=lk_kwargs["dsi_delta_hz"],
                dsi_ratio=lk_kwargs["dsi_ratio"],
                cc_ratio_boost=lk_kwargs["cc_ratio_boost"],
                cc_onset_frac=lk_kwargs["cc_onset_frac"],
                cc_duty_frac=lk_kwargs["cc_duty_frac"],
            )
            phases = (phases + phase_errors) % (2 * np.pi)

        return intensities, phases

class ChannelNode:
    def __init__(self, distance_km):
        # Distance is total distance L. Distance to Charlie is L/2.
        self.distance_to_charlie = distance_km / 2.0
        # Transmittance to Charlie
        self.eta_c = 10 ** (-ATTENUATION_COEF * self.distance_to_charlie / 10.0)

    def simulate_detection(self, intensities_A, phases_A, intensities_B, phases_B):
        # Fields arriving at Charlie
        # E_A = sqrt(eta_c * I_A) * exp(i * phi_A)
        # However, interference intensity is:
        # I_D0 = 0.5 * eta_c * (I_A + I_B + 2*sqrt(I_A * I_B)*cos(phi_A - phi_B))
        # I_D1 = 0.5 * eta_c * (I_A + I_B - 2*sqrt(I_A * I_B)*cos(phi_A - phi_B))
        
        phase_diff = phases_A - phases_B
        interference_term = 2 * np.sqrt(intensities_A * intensities_B) * np.cos(phase_diff)
        
        I_D0 = 0.5 * self.eta_c * (intensities_A + intensities_B + interference_term)
        I_D1 = 0.5 * self.eta_c * (intensities_A + intensities_B - interference_term)
        
        # Click probabilities (assuming Poisson photon statistics)
        p_click_D0 = 1 - (1 - DARK_COUNT_RATE) * np.exp(-DETECTOR_EFFICIENCY * I_D0)
        p_click_D1 = 1 - (1 - DARK_COUNT_RATE) * np.exp(-DETECTOR_EFFICIENCY * I_D1)
        
        # Sample detection events
        clicks_D0 = np.random.rand(len(I_D0)) < p_click_D0
        clicks_D1 = np.random.rand(len(I_D1)) < p_click_D1
        
        return clicks_D0, clicks_D1

class TFQKDProtocol:
    def __init__(self, distance_km, n_pulses=DEFAULT_N_PULSES, use_oil=False, alpha=LK_ALPHA):
        self.alice = Transmitter("Alice", use_oil=use_oil, alpha=alpha)
        self.bob = Transmitter("Bob", use_oil=use_oil, alpha=alpha)
        self.charlie = ChannelNode(distance_km)
        self.n_pulses = n_pulses

    def run(self, attack_mode=0, **attack_params):
        """
        Run the TF-QKD protocol for one distance.

        Parameters
        ----------
        attack_mode : int
            0=Nominal, 1=FIM, 2=TWIRL, 3=PNI, 4=CDA, 5=DSI, 6=CC.
        **attack_params
            Override any attack parameter.
        """
        # 1. State preparation
        I_A, phi_A = self.alice.generate_pulses(self.n_pulses, attack_mode, **attack_params)
        I_B, phi_B = self.bob.generate_pulses(self.n_pulses, attack_mode, **attack_params)
        
        # 2. Transmission & Detection
        clicks_D0, clicks_D1 = self.charlie.simulate_detection(I_A, phi_A, I_B, phi_B)
        
        # 3. Sifting (Phase Matching)
        # Valid events: exactly one detector clicks
        valid_events = clicks_D0 ^ clicks_D1
        
        # In PM-TFQKD, Alice and Bob announce their phases.
        # They sift events where phase difference is 0 or pi (mod 2pi)
        phase_diff = (phi_A - phi_B) % (2 * np.pi)
        
        # Due to floating point issues, we should check closeness
        is_diff_0 = np.isclose(phase_diff, 0, atol=1e-5) | np.isclose(phase_diff, 2*np.pi, atol=1e-5)
        is_diff_pi = np.isclose(phase_diff, np.pi, atol=1e-5)
        
        phase_matched = is_diff_0 | is_diff_pi
        
        sifted_idx = valid_events & phase_matched
        
        # 4. Error mapping
        # If phase diff is 0, D0 should click. If D1 clicks, it's an error.
        # If phase diff is pi, D1 should click. If D0 clicks, it's an error.
        is_error = np.zeros(self.n_pulses, dtype=bool)
        
        # For diff = 0, D1 click is error
        is_error[is_diff_0 & clicks_D1] = True
        
        # For diff = pi, D0 click is error
        is_error[is_diff_pi & clicks_D0] = True
        
        return {
            "I_A": I_A, "I_B": I_B,
            "phi_A": phi_A, "phi_B": phi_B,
            "clicks_D0": clicks_D0,
            "clicks_D1": clicks_D1,
            "sifted_idx": sifted_idx,
            "is_error": is_error,
            "phase_matched": phase_matched
        }

def estimate_decoy_parameters(results):
    """
    Implements 3-intensity decoy-state estimation.
    This estimates the single-photon yield Y_11 and error rate e_11.
    For simplicity, we compute gains (Q) and quantum bit error rates (E)
    for intensity pairs (I_A, I_B).
    """
    I_A = results["I_A"]
    I_B = results["I_B"]
    sifted_idx = results["sifted_idx"]
    is_error = results["is_error"]
    
    n_pulses = len(I_A)
    # The total number of pulses sent with matching phases is n_pulses / NUM_PHASE_SLICES * 2
    # But let's just count them directly.
    phase_matched = results["phase_matched"]
    
    def get_stats(intensity_A, intensity_B):
        # Count total pulses sent with this intensity pair AND matching phases
        sent_mask = (I_A == intensity_A) & (I_B == intensity_B) & phase_matched
        n_sent = np.sum(sent_mask)
        if n_sent == 0:
            return 0, 0, 0
        
        # Count how many of these were sifted (single click)
        n_sifted = np.sum(sifted_idx & sent_mask)
        
        # Count how many of the sifted were errors
        n_error = np.sum(is_error & sifted_idx & sent_mask)
        
        gain_Q = n_sifted / n_sent
        error_E = n_error / n_sifted if n_sifted > 0 else 0
        return gain_Q, error_E, n_sent

    # Calculate statistics for different intensity pairs
    Q_mumu, E_mumu, N_mumu = get_stats(MU, MU)
    Q_nunu, E_nunu, N_nunu = get_stats(NU, NU)
    Q_00, E_00, N_00 = get_stats(OMEGA, OMEGA)
    
    # In standard TF-QKD, vacuum yield is 2 * p_d (since either detector can click from dark counts)
    Y_0 = Q_00
    
    # Linear bounds for Y_11 and e_11 (simplified Hwang method for symmetric pairs)
    # Note: rigorous finite-key analysis or complex bounds are usually needed for real experiments.
    # Here we use the standard analytical asymptotic bounds.
    # We want to bound Y_11 (yield of 1-photon from Alice and 1-photon from Bob) or the overall single-photon equivalent.
    # Wait, in PM-TFQKD, the "single-photon" state actually means a state with 1 photon TOTAL across both modes,
    # or more accurately, the relative phase encoded state. 
    # For Phase-Matching QKD (Ma et al. or Wang et al.), the yield of interest is Y_1 (single photon in the coherent superposition).
    # A simplified lower bound on Y_1 using mu and nu (assuming mu > nu > 0) is:
    
    # Q_mu = exp(-mu) * sum (mu^n / n!) Y_n
    # Q_nu = exp(-nu) * sum (nu^n / n!) Y_n
    
    # This is slightly different for TF-QKD because the total intensity is 2*mu.
    # Using the standard decoy state equation where intensity is the single-arm intensity:
    
    if Q_nunu == 0 or Q_mumu == 0:
        return 0, 0, Q_mumu, E_mumu
    
    try:
        # Standard decoy state bound formula applied to the effective twin-field intensity
        # I_eff = 2*I
        Y_1_lower = (MU**2 * np.exp(2*NU) * Q_nunu - NU**2 * np.exp(2*MU) * Q_mumu - (MU**2 - NU**2) * Y_0) / (MU * NU * (MU - NU))
        Y_1_lower = max(Y_1_lower, 0)
        
        e_1_upper = (E_nunu * Q_nunu * np.exp(2*NU) - 0.5 * Y_0) / (NU * Y_1_lower) if Y_1_lower > 0 else 0.5
        e_1_upper = min(e_1_upper, 0.5)
        e_1_upper = max(e_1_upper, 0)
    except:
        Y_1_lower = 0
        e_1_upper = 0.5

    return Y_1_lower, e_1_upper, Q_mumu, E_mumu

def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def calculate_key_rate(Y_1_lower, e_1_upper, Q_mumu, E_mumu):
    """
    Computes asymptotic secure key rate (bits per pulse) using GLLP.
    R = Q_mumu * ( p1 * Y_1 * (1 - H(e_1)) - f_ec * H(E_mumu) ) / Q_mumu ...
    Actually R = P_{11} Y_11 (1 - H(e_11)) - Q_mumu * f_ec * H(E_mumu)
    where P_{11} is the probability of emitting the single-photon component from the signal states.
    For signal state intensity MU on both sides, the single-photon total intensity is 2*MU.
    The Poisson probability of 1 photon total is (2*MU) * exp(-2*MU).
    """
    if Q_mumu <= 0 or Y_1_lower <= 0:
        return 0.0
    
    # Probability of 1 photon total given both chose MU
    P_1 = (2 * MU) * np.exp(-2 * MU)
    
    term1 = P_1 * Y_1_lower * (1 - binary_entropy(e_1_upper))
    term2 = Q_mumu * F_EC * binary_entropy(E_mumu)
    
    R = term1 - term2
    
    # The key is only extracted when both Alice and Bob chose MU and matching phases.
    # The factor for basis choice is P_MU * P_MU * (2 / NUM_PHASE_SLICES)
    # We multiply by this to get the rate per total sent pulse.
    # (assuming we only extract key from signal-signal, phase-matched pulses)
    
    basis_sifting_factor = (P_MU ** 2) * (2 / NUM_PHASE_SLICES)
    
    return max(0.0, basis_sifting_factor * R)
