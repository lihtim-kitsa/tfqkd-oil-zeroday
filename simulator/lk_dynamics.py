import numpy as np
import numba

@numba.njit(cache=True)
def solve_lk_sde(
    n_pulses,
    pulse_interval_s = 1e-9,   # 1 ns between pulses (1 GHz clock)
    dt               = 1e-13,  # 0.1 ps integration step for stability
    alpha  = 3.0,
    gamma_p= 1e12,
    gamma_e= 1e9,
    g_n    = 1e-4,
    N0     = 1e7,
    kappa  = 1e11,
    detuning_hz = 10e6,
    R_sp   = 1e12,
    inj_ratio = 0.1,
    # ── Attack parameters ─────────────────────────────────────────────────
    # attack_mode: 0=Nominal, 1=FIM, 2=TWIRL, 3=PNI, 4=CDA, 5=DSI, 6=CC
    attack_mode     = 0,
    # FIM (mode 1)
    fim_mod_depth   = 0.0,
    fim_f_mod_hz    = 50e6,
    fim_delta_f_hz  = 0.0,
    # TWIRL (mode 2)
    twirl_sweep_hz  = 0.0,
    twirl_T_sweep_s = 100e-9,
    # PNI — Phase-Noise Injection (mode 3, ZD-A)
    pni_sigma       = 0.0,
    # CDA — Carrier Density Attack (mode 4, ZD-B)
    cda_amplitude   = 0.0,
    cda_freq_hz     = 80e6,
    # DSI — Double-Sideband Injection (mode 5, ZD-C)
    dsi_delta_hz    = 120e6,
    dsi_ratio       = 0.0,
    # CC — Coherence Collapse (mode 6, ZD-D)
    cc_ratio_boost  = 1.0,
    cc_onset_frac   = 0.3,
    cc_duty_frac    = 0.2,
    # CFI — Chirped Frequency Injection (mode 7, ZD-E)
    cfi_sweep_hz    = 50e6,
    cfi_T_sweep_s   = 1e-9,
    # CAM — Chaotic Amplitude Modulation (mode 8, ZD-F)
    cam_sigma       = 0.05,
    # DPJ — Discrete Phase Jumps (mode 9, ZD-G)
    dpj_phase_shift = np.pi/2,
    dpj_probability = 0.01,
    # SAM — Sawtooth Amplitude Modulation (mode 10, ZD-H)
    sam_amplitude   = 0.4,
    sam_freq_hz     = 50e6,
    # RTN — Random Telegraph Noise (mode 11, ZD-I)
    rtn_amplitude   = 0.3,
    rtn_flip_prob   = 0.05,
    # MTC — Multi-Tone Comb (mode 12, ZD-J)
    mtc_delta_hz    = 50e6,
    mtc_ratio       = 0.2,
):
    """
    Solves the Lang-Kobayashi SDE for a slave laser under optical injection,
    with optional attack perturbations injected inside the integration loop.

    Attacks are inlined (no Python callbacks) so the function remains
    Numba-JIT compatible.

    Parameters
    ----------
    n_pulses : int
        Number of pulses to simulate.
    attack_mode : int
        0 = Nominal, 1 = FIM, 2 = TWIRL,
        3 = PNI, 4 = CDA, 5 = DSI, 6 = CC,
        7 = CFI, 8 = CAM, 9 = DPJ, 10 = SAM, 11 = RTN, 12 = MTC.
    (all other attack_* params)
        Attack-specific floats; defaults are 0 / passthrough for Nominal.

    Returns
    -------
    phases : ndarray, shape (n_pulses,)
        Instantaneous phase of the slave laser (relative to master frame)
        sampled once per pulse interval.
    """

    # ── Pre-compute constants ────────────────────────────────────────────────
    N_th = N0 + gamma_p / g_n
    J    = 2.0 * gamma_e * N_th          # Pump at 2× threshold
    S0   = (J - gamma_e * N_th) / gamma_p
    A0   = np.sqrt(S0)

    # Nominal injection amplitude
    A_inj_nominal = np.sqrt(inj_ratio * S0)

    # Angular detuning
    delta_omega_base = 2.0 * np.pi * detuning_hz

    # FIM extra offset
    fim_delta_omega = 2.0 * np.pi * fim_delta_f_hz

    # TWIRL angular sweep rate (linear ramp that reverses every half-period)
    twirl_omega_range = 2.0 * np.pi * twirl_sweep_hz
    twirl_T           = twirl_T_sweep_s

    # DSI sideband
    dsi_delta_omega = 2.0 * np.pi * dsi_delta_hz

    # CDA
    cda_omega = 2.0 * np.pi * cda_freq_hz

    # Noise scaling
    noise_std = np.sqrt(R_sp / (2.0 * dt))

    steps_per_pulse = int(pulse_interval_s / dt)
    total_steps     = n_pulses * steps_per_pulse

    # CC burst window (in steps within a pulse interval)
    cc_onset_step = int(cc_onset_frac  * steps_per_pulse)
    cc_end_step   = int((cc_onset_frac + cc_duty_frac) * steps_per_pulse)

    # Initial states
    E_re = A0
    E_im = 0.0
    N    = N_th

    pulse_phases = np.empty(n_pulses, dtype=np.float64)
    pulse_idx    = 0

    # State variables for new attacks
    cam_current_amp = A_inj_nominal
    dpj_current_phase = 0.0
    rtn_current_state = 1.0

    for step in range(total_steps):
        t = step * dt

        # Step within the current pulse interval (used by CC)
        step_in_pulse = step % steps_per_pulse

        # ── Build the effective injection field for this timestep ────────────

        if attack_mode == 1:
            # FIM: sinusoidally modulated amplitude + static frequency offset
            A_inj_t = A_inj_nominal * (1.0 + fim_mod_depth * np.sin(2.0 * np.pi * fim_f_mod_hz * t))
            delta_omega_t = delta_omega_base + fim_delta_omega
            inj_phase  = delta_omega_t * t
            inj_re = A_inj_t * np.cos(inj_phase)
            inj_im = A_inj_t * np.sin(inj_phase)

        elif attack_mode == 2:
            # TWIRL: linearly sweeping master frequency (triangle wave)
            # phase = integral of omega(t) over [0, t]
            # omega(t) = base + sweep * triangle(t/T)
            # For simplicity, compute instantaneous frequency and accumulate below.
            # Here we use an approximate closed form: triangle sweep.
            half_T = twirl_T / 2.0
            t_mod  = t % twirl_T
            if t_mod < half_T:
                sweep_offset = (twirl_omega_range / half_T) * t_mod - twirl_omega_range / 2.0
            else:
                sweep_offset = twirl_omega_range / 2.0 - (twirl_omega_range / half_T) * (t_mod - half_T)
            # Cumulative phase via instantaneous frequency (Euler approx)
            delta_omega_t = delta_omega_base + sweep_offset
            inj_phase = delta_omega_t * t
            inj_re = A_inj_nominal * np.cos(inj_phase)
            inj_im = A_inj_nominal * np.sin(inj_phase)

        elif attack_mode == 5:
            # DSI: two tones at (base ± dsi_delta_omega)
            inj_phase_main = delta_omega_base * t
            inj_phase_lo   = (delta_omega_base - dsi_delta_omega) * t
            inj_phase_hi   = (delta_omega_base + dsi_delta_omega) * t
            A_side = A_inj_nominal * np.sqrt(dsi_ratio)
            inj_re = (A_inj_nominal * np.cos(inj_phase_main)
                      + A_side * np.cos(inj_phase_lo)
                      + A_side * np.cos(inj_phase_hi))
            inj_im = (A_inj_nominal * np.sin(inj_phase_main)
                      + A_side * np.sin(inj_phase_lo)
                      + A_side * np.sin(inj_phase_hi))

        elif attack_mode == 6:
            # CC: inject-ratio boost during burst window within each pulse interval
            inj_phase = delta_omega_base * t
            if cc_onset_step <= step_in_pulse < cc_end_step:
                A_inj_t = A_inj_nominal * np.sqrt(cc_ratio_boost)
            else:
                A_inj_t = A_inj_nominal
            inj_re = A_inj_t * np.cos(inj_phase)
            inj_im = A_inj_t * np.sin(inj_phase)

        elif attack_mode == 7:
            # CFI: extremely fast chirped frequency injection
            half_T = cfi_T_sweep_s / 2.0
            t_mod = t % cfi_T_sweep_s
            cfi_omega_range = 2.0 * np.pi * cfi_sweep_hz
            if t_mod < half_T:
                sweep_offset = (cfi_omega_range / half_T) * t_mod - cfi_omega_range / 2.0
            else:
                sweep_offset = cfi_omega_range / 2.0 - (cfi_omega_range / half_T) * (t_mod - half_T)
            delta_omega_t = delta_omega_base + sweep_offset
            inj_phase = delta_omega_t * t
            inj_re = A_inj_nominal * np.cos(inj_phase)
            inj_im = A_inj_nominal * np.sin(inj_phase)

        elif attack_mode == 8:
            # CAM: Chaotic amplitude modulation (AR(1) process on amplitude)
            cam_current_amp = 0.999 * cam_current_amp + 0.001 * A_inj_nominal * (1.0 + cam_sigma * np.random.randn())
            inj_phase = delta_omega_base * t
            inj_re = cam_current_amp * np.cos(inj_phase)
            inj_im = cam_current_amp * np.sin(inj_phase)

        elif attack_mode == 9:
            # DPJ: Random discrete phase jumps
            if np.random.rand() < dpj_probability:
                dpj_current_phase += dpj_phase_shift * (1.0 if np.random.rand() < 0.5 else -1.0)
            inj_phase = delta_omega_base * t + dpj_current_phase
            inj_re = A_inj_nominal * np.cos(inj_phase)
            inj_im = A_inj_nominal * np.sin(inj_phase)

        elif attack_mode == 10:
            # SAM: Sawtooth amplitude modulation
            T_sam = 1.0 / sam_freq_hz
            t_mod = t % T_sam
            amp_mod = 1.0 + sam_amplitude * (2.0 * t_mod / T_sam - 1.0)
            A_inj_t = A_inj_nominal * amp_mod
            inj_phase = delta_omega_base * t
            inj_re = A_inj_t * np.cos(inj_phase)
            inj_im = A_inj_t * np.sin(inj_phase)

        elif attack_mode == 11:
            # RTN: Random telegraph noise on amplitude
            if np.random.rand() < rtn_flip_prob:
                rtn_current_state = -rtn_current_state
            A_inj_t = A_inj_nominal * (1.0 + rtn_amplitude * rtn_current_state)
            inj_phase = delta_omega_base * t
            inj_re = A_inj_t * np.cos(inj_phase)
            inj_im = A_inj_t * np.sin(inj_phase)

        elif attack_mode == 12:
            # MTC: Multi-tone comb (carrier + 2 sidebands each side)
            inj_phase_main = delta_omega_base * t
            mtc_delta_omega = 2.0 * np.pi * mtc_delta_hz
            A_side = A_inj_nominal * np.sqrt(mtc_ratio)
            inj_re = A_inj_nominal * np.cos(inj_phase_main)
            for k in (-2, -1, 1, 2):
                phase_k = (delta_omega_base + k * mtc_delta_omega) * t
                inj_re += A_side * np.cos(phase_k)
            inj_im = A_inj_nominal * np.sin(inj_phase_main)
            for k in (-2, -1, 1, 2):
                phase_k = (delta_omega_base + k * mtc_delta_omega) * t
                inj_im += A_side * np.sin(phase_k)

        else:
            # Nominal (0), PNI (3), CDA (4): standard injection field
            inj_phase = delta_omega_base * t
            inj_re = A_inj_nominal * np.cos(inj_phase)
            inj_im = A_inj_nominal * np.sin(inj_phase)

        # ── Lang-Kobayashi ODE step ──────────────────────────────────────────
        G       = g_n * (N - N0)
        net_gain = G - gamma_p

        term1_re = 0.5 * net_gain * (E_re - alpha * E_im)
        term1_im = 0.5 * net_gain * (E_im + alpha * E_re)

        dW_re = np.random.randn() * noise_std
        dW_im = np.random.randn() * noise_std

        dE_re = term1_re + kappa * inj_re + dW_re
        dE_im = term1_im + kappa * inj_im + dW_im

        # CDA (mode 4): sinusoidal carrier-density perturbation
        S  = E_re * E_re + E_im * E_im
        dN = J - gamma_e * N - G * S
        if attack_mode == 4:
            dN += cda_amplitude * N_th * np.sin(cda_omega * t)

        # Euler update
        E_re += dE_re * dt
        E_im += dE_im * dt
        N    += dN    * dt

        # PNI (mode 3): additive phase noise on slave field (post-update)
        if attack_mode == 3:
            phi_noise = pni_sigma * np.random.randn()
            # Rotate (E_re, E_im) by phi_noise
            cos_phi = np.cos(phi_noise)
            sin_phi = np.sin(phi_noise)
            E_re_new = E_re * cos_phi - E_im * sin_phi
            E_im_new = E_re * sin_phi + E_im * cos_phi
            E_re = E_re_new
            E_im = E_im_new

        # ── Sample phase at pulse boundary ───────────────────────────────────
        if (step + 1) % steps_per_pulse == 0:
            phase = np.arctan2(E_im, E_re)
            # Phase relative to master frame
            slave_phase_rel = phase - inj_phase
            # Wrap to [-pi, pi]
            slave_phase_rel = (slave_phase_rel + np.pi) % (2 * np.pi) - np.pi
            pulse_phases[pulse_idx] = slave_phase_rel
            pulse_idx += 1

    return pulse_phases
