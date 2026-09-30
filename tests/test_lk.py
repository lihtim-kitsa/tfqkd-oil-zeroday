import numpy as np
from simulator.lk_dynamics import solve_lk_sde

def test_lk_locking():
    # If detuning is 0, the phase should lock quickly and stay near 0 (with zero noise).
    phases = solve_lk_sde(
        n_pulses=10000, 
        detuning_hz=0.0, 
        R_sp=0.0 # turn off noise to test pure deterministic locking
    )
    
    # After a transient, phase should be constant (locked)
    steady_state_phases = phases[5000:]
    assert np.var(steady_state_phases) < 1e-4

def test_lk_unlocked():
    # If detuning is very large, phase slips continuously
    phases = solve_lk_sde(
        n_pulses=10000, 
        detuning_hz=1e12, # 1 THz detuning, well outside 500 MHz locking bandwidth
        R_sp=0.0
    )
    
    # Phase should not be constant. It should drift and wrap around.
    # The variance should be large because it's wrapping uniformly over [-pi, pi].
    steady_state_phases = phases[5000:]
    assert np.var(steady_state_phases) > 1.0
