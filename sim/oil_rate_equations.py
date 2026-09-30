import numpy as np
from scipy.integrate import solve_ivp

class OILSimulator:
    def __init__(self, kappa=1e9, alpha=3.0, gamma=1e11, delta_omega=0.0):
        self.kappa = kappa        # Injection rate
        self.alpha = alpha        # Linewidth enhancement factor
        self.gamma = gamma        # Cavity decay rate
        self.delta_omega = delta_omega # Detuning
        
        # Gain model parameters
        self.g0 = 1.2e11          # Linear gain coefficient
        self.N_tr = 1e8           # Transparency carrier number
        
    def _rate_equations(self, t, y, E_ref_func, phi_ref_func, current_func):
        # y = [E, phi, N] where E is amplitude, phi is phase, N is carrier number
        E, phi, N = y
        
        # Evaluate time-dependent inputs
        E_ref = E_ref_func(t)
        phi_ref = phi_ref_func(t)
        I_inj = current_func(t)
        
        # Simple gain model
        G = self.g0 * (N - self.N_tr) / self.N_tr
        
        # Rate equations
        dE_dt = 0.5 * (G - self.gamma) * E + self.kappa * E_ref * np.cos(phi_ref - phi)
        dphi_dt = 0.5 * self.alpha * (G - self.gamma) - self.delta_omega + self.kappa * (E_ref / (E + 1e-12)) * np.sin(phi_ref - phi)
        
        # Carrier dynamics (simplified)
        tau_e = 1e-9 # Carrier lifetime
        dN_dt = I_inj - N / tau_e - G * E**2
        
        return [dE_dt, dphi_dt, dN_dt]

    def simulate(self, t_span, y0, E_ref_func, phi_ref_func, current_func, t_eval=None):
        """Simulate the OIL system."""
        sol = solve_ivp(
            self._rate_equations, 
            t_span, 
            y0, 
            args=(E_ref_func, phi_ref_func, current_func),
            t_eval=t_eval,
            method='BDF',
            rtol=1e-3,
            atol=1e-6
        )
        return sol
