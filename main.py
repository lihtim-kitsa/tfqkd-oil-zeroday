import numpy as np
import matplotlib.pyplot as plt
from simulator.tf_qkd import TFQKDProtocol, estimate_decoy_parameters, calculate_key_rate
from simulator.config import ATTENUATION_COEF, DETECTOR_EFFICIENCY, DARK_COUNT_RATE

def plob_bound(distance_km):
    """
    Computes the PLOB bound: R = -log2(1 - eta)
    where eta is the total transmittance.
    """
    # Total loss in dB
    loss_dB = ATTENUATION_COEF * distance_km
    eta = 10 ** (-loss_dB / 10.0)
    return -np.log2(1 - eta)

def repeaterless_bound(distance_km):
    """
    Standard repeaterless QKD bound scaling as O(eta).
    Approximately R ~ eta. (Using 1.44 * eta as per some literature bounds)
    """
    loss_dB = ATTENUATION_COEF * distance_km
    eta = 10 ** (-loss_dB / 10.0)
    # The rate for standard QKD typically scales as eta
    return eta

def main():
    distances = np.arange(10, 310, 20)  # km (this implies losses up to 60 dB)
    # The max distance 300 km * 0.2 dB/km = 60 dB total loss.
    
    simulated_rates = []
    
    print(f"{'Distance (km)':>15} | {'Key Rate (bps/pulse)':>25} | {'Y_1 (lower)':>15} | {'e_1 (upper)':>15}")
    print("-" * 78)
    
    # We use a large number of pulses to get stable statistics, 
    # particularly at high distances where clicks are rare.
    # At 300km, eta_c = 10^(-30dB/10) = 0.001.
    # With N=50,000,000, we expect enough clicks to compute the rate.
    n_pulses = 10_000_000 
    
    for d in distances:
        # Increase n_pulses for longer distances to maintain precision
        current_n = n_pulses
        if d > 150:
            current_n = 30_000_000
        if d >= 250:
            current_n = 50_000_000
            
        protocol = TFQKDProtocol(d, n_pulses=current_n)
        results = protocol.run()
        
        Y_1_lower, e_1_upper, Q_mumu, E_mumu = estimate_decoy_parameters(results)
        rate = calculate_key_rate(Y_1_lower, e_1_upper, Q_mumu, E_mumu)
        simulated_rates.append(rate)
        
        print(f"{d:15.1f} | {rate:25.2e} | {Y_1_lower:15.2e} | {e_1_upper:15.4f}")

    # Plotting
    plob_rates = [plob_bound(d) for d in distances]
    repeaterless_rates = [repeaterless_bound(d) for d in distances]
    
    plt.figure(figsize=(10, 7))
    plt.semilogy(distances, plob_rates, 'k--', label=r'PLOB bound $O(\eta)$')
    plt.semilogy(distances, repeaterless_rates, 'g-.', label=r'Repeaterless bound $O(\eta)$')
    
    # Only plot positive rates
    valid_distances = [d for d, r in zip(distances, simulated_rates) if r > 0]
    valid_rates = [r for r in simulated_rates if r > 0]
    
    plt.semilogy(valid_distances, valid_rates, 'b-o', label='Simulated TF-QKD')
    
    plt.title('TF-QKD Simulated Key Rate vs Distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Secure Key Rate (bits/pulse)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plt.savefig('results/key_rate_vs_distance.png')
    print("\nSaved plot to results/key_rate_vs_distance.png")

if __name__ == "__main__":
    main()
