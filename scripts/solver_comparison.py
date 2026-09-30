import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import os

# Ensure we can import from sim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim.oil_rate_equations import OILSimulator

def run_comparison():
    sim = OILSimulator()
    
    # 10 ns simulation
    t_span = (0, 10e-9)
    y0 = [1.0, 0.0, 0.0]
    
    # Dummy inputs
    def E_ref_func(t): return 1.0
    def phi_ref_func(t): return 0.0
    def current_func(t): return 1.5e8
    
    print("Running with BDF (Implicit)...")
    start_time = time.time()
    sol_bdf = solve_ivp(
        sim._rate_equations, 
        t_span, 
        y0, 
        args=(E_ref_func, phi_ref_func, current_func),
        method='BDF',
        rtol=1e-3,
        atol=1e-6
    )
    bdf_time = time.time() - start_time
    
    print("Running with RK45 (Explicit)...")
    start_time = time.time()
    # Adding a timeout or just letting it run to see if it takes too long
    # We might expect it to take many more steps
    sol_rk45 = solve_ivp(
        sim._rate_equations, 
        t_span, 
        y0, 
        args=(E_ref_func, phi_ref_func, current_func),
        method='RK45',
        rtol=1e-3,
        atol=1e-6
    )
    rk45_time = time.time() - start_time
    
    print("\n--- Solver Comparison Results ---")
    print(f"BDF Method:")
    print(f"  Success: {sol_bdf.success}")
    print(f"  Number of Function Evaluations (nfev): {sol_bdf.nfev}")
    print(f"  Number of Time Steps: {len(sol_bdf.t)}")
    print(f"  Compute Time: {bdf_time:.4f} seconds")
    
    print(f"\nRK45 Method:")
    print(f"  Success: {sol_rk45.success}")
    print(f"  Number of Function Evaluations (nfev): {sol_rk45.nfev}")
    print(f"  Number of Time Steps: {len(sol_rk45.t)}")
    print(f"  Compute Time: {rk45_time:.4f} seconds")
    
    # Let's plot step sizes
    plt.figure(figsize=(10, 5))
    
    steps_bdf = np.diff(sol_bdf.t)
    steps_rk45 = np.diff(sol_rk45.t)
    
    plt.plot(sol_bdf.t[:-1] * 1e9, steps_bdf, label='BDF Steps', alpha=0.8)
    plt.plot(sol_rk45.t[:-1] * 1e9, steps_rk45, label='RK45 Steps', alpha=0.8)
    
    plt.yscale('log')
    plt.xlabel('Simulation Time (ns)')
    plt.ylabel('Step Size (s)')
    plt.title('Solver Step Size Comparison: BDF vs RK45')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/solver_comparison.png')
    print("\nSaved step size plot to figures/solver_comparison.png")

if __name__ == "__main__":
    run_comparison()
