import numpy as np
import matplotlib.pyplot as plt
from simulator.tf_qkd import TFQKDProtocol, estimate_decoy_parameters, calculate_key_rate
import simulator.config as config

def main():
    # We will test at a fixed distance (e.g. 50 km) where rate is typically decent.
    # We'll sweep detuning from 0 up to 1 GHz.
    # The locking bandwidth is around ~500 MHz, so we should see rate drop to 0 as detuning approaches and exceeds that.
    
    distance = 50.0 # km
    detunings_mhz = np.linspace(0, 1000, 20)
    
    rates = []
    
    # Numba compilation happens on first run, so the first iteration might be slow.
    # Reduce n_pulses for quick sweeping in this demonstration.
    n_pulses = 10_000
    
    print(f"Sweeping detuning at {distance} km with {n_pulses} pulses (OIL Enabled)")
    print(f"{'Detuning (MHz)':>15} | {'Key Rate (bps/pulse)':>25} | {'e_1 (upper)':>15}")
    print("-" * 65)
    
    for det_mhz in detunings_mhz:
        config.LK_DETUNING_HZ = det_mhz * 1e6
        
        protocol = TFQKDProtocol(distance, n_pulses=n_pulses, use_oil=True)
        results = protocol.run()
        
        Y_1_lower, e_1_upper, Q_mumu, E_mumu = estimate_decoy_parameters(results)
        rate = calculate_key_rate(Y_1_lower, e_1_upper, Q_mumu, E_mumu)
        rates.append(rate)
        
        print(f"{det_mhz:15.1f} | {rate:25.2e} | {e_1_upper:15.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(detunings_mhz, rates, 'r-o', label='Simulated Key Rate with OIL')
    plt.axvline(x=500, color='k', linestyle='--', label='Approx. Locking Boundary (500 MHz)')
    plt.title('TF-QKD Key Rate vs. Master-Slave Frequency Detuning')
    plt.xlabel('Detuning (MHz)')
    plt.ylabel('Secure Key Rate (bits/pulse)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('results/key_rate_vs_detuning.png')
    print("\nSaved plot to results/key_rate_vs_detuning.png")

if __name__ == "__main__":
    main()
