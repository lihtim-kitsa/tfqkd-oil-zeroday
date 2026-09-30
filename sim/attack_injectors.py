import numpy as np

class FIMInjector:
    def __init__(self, m=0.1, f_fim=1e9):
        """
        Fast Intensity Modulation (FIM) attack.
        m: modulation index (0.05 to 0.40)
        f_fim: modulation frequency (100 MHz to 10 GHz)
        """
        self.m = m
        self.f_fim = f_fim

    def inject_E_ref(self, t, base_E_ref):
        """Modulates the base reference amplitude."""
        return base_E_ref * (1.0 + self.m * np.sin(2 * np.pi * self.f_fim * t))


class TWIRLInjector:
    def __init__(self, delta_lambda=1.0, power_ratio=0.01):
        """
        TWIRL attack (Trojan-wavelength injection).
        delta_lambda: wavelength difference in nm
        power_ratio: ratio of trojan power to reference power
        """
        self.delta_lambda = delta_lambda
        # convert delta_lambda to frequency shift (approximate for 1550nm)
        # c = f * lambda => df = -(c/lambda^2) * dlambda
        # For lambda=1550nm, df ~ 125 GHz / nm
        self.delta_f = 125e9 * delta_lambda
        self.power_ratio = power_ratio
        self.trojan_amplitude = np.sqrt(power_ratio)

    def inject_phi_ref(self, t, base_phi_ref):
        """Adds phase beating from the trojan signal."""
        # Simplified: the beating causes a phase ripple
        return base_phi_ref + self.delta_lambda * t * 1e11 # simple linear phase ramp

class FSKInjector:
    def __init__(self, f_mod=1e9, max_detuning=1e10):
        self.f_mod = f_mod
        self.max_detuning = max_detuning
        
    def inject_phi_ref(self, t, base_phi_ref):
        # Rapid sinusoidal frequency modulation (FSK-like)
        # Integral of angular frequency gives phase
        return base_phi_ref - (self.max_detuning / (2 * np.pi * self.f_mod)) * np.cos(2 * np.pi * self.f_mod * t)
