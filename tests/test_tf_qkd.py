import pytest
import numpy as np
from simulator.tf_qkd import Transmitter, ChannelNode, TFQKDProtocol
from simulator.config import MU, NU, OMEGA, P_MU, P_NU, P_0, NUM_PHASE_SLICES

def test_transmitter():
    t = Transmitter("Alice")
    intensities, phases = t.generate_pulses(1000)
    
    assert len(intensities) == 1000
    assert len(phases) == 1000
    
    # Check that intensities are from the defined set
    unique_intensities = set(np.unique(intensities))
    assert unique_intensities.issubset({MU, NU, OMEGA})
    
    # Check proportions roughly
    p_mu = np.sum(intensities == MU) / 1000
    p_nu = np.sum(intensities == NU) / 1000
    p_0 = np.sum(intensities == OMEGA) / 1000
    
    assert 0.5 < p_mu < 0.7  # expected 0.6
    assert 0.2 < p_nu < 0.4  # expected 0.3
    assert 0.0 < p_0 < 0.2   # expected 0.1

def test_channel_node():
    node = ChannelNode(0)  # 0 km distance
    # With 0 km, eta_c = 1.
    assert node.eta_c == 1.0
    
    # Interference test: construct signals so phase diff is 0
    I_A = np.array([1.0, 1.0])
    I_B = np.array([1.0, 1.0])
    phi_A = np.array([0.0, np.pi])
    phi_B = np.array([0.0, np.pi])
    
    # Simulate multiple times to check probabilities?
    # This is random, but we can verify the math inside.
    # D0 intensity = 0.5 * 1.0 * (1 + 1 + 2*1*1) = 2.0
    # D1 intensity = 0.5 * 1.0 * (1 + 1 - 2*1*1) = 0.0
    clicks_D0, clicks_D1 = node.simulate_detection(I_A, phi_A, I_B, phi_B)
    
    # For D1, intensity is 0, so clicks should only be from dark counts (very rare)
    assert not np.any(clicks_D1)

def test_tf_qkd_protocol():
    # Small number of pulses for quick test
    protocol = TFQKDProtocol(10.0, n_pulses=10000)
    results = protocol.run()
    
    assert "I_A" in results
    assert "clicks_D0" in results
    assert "sifted_idx" in results
    
    # Sifted events should have matching phases
    phase_diff = (results["phi_A"] - results["phi_B"]) % (2*np.pi)
    sifted_diff = phase_diff[results["sifted_idx"]]
    
    # Phase difference must be 0 or pi
    valid_0 = np.isclose(sifted_diff, 0, atol=1e-4) | np.isclose(sifted_diff, 2*np.pi, atol=1e-4)
    valid_pi = np.isclose(sifted_diff, np.pi, atol=1e-4)
    assert np.all(valid_0 | valid_pi)
