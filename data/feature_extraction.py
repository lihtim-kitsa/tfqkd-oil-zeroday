import numpy as np

def extract_features(t, E, phi, N, window_size=100):
    """
    Extract 5 features for each window: mu, sigma2, Psb, delta_phi, QBER.
    Assuming E, phi are arrays of length N.
    """
    # Number of photons is proportional to |E|^2
    # This is a normalized representation
    photon_counts = np.abs(E)**2 
    
    features = []
    
    # Process in windows
    num_windows = len(t) // window_size
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        
        window_E = E[start:end]
        window_phi = phi[start:end]
        window_photons = photon_counts[start:end]
        
        # Feature 1: Mean photon number mu
        mu = np.mean(window_photons)
        
        # Feature 2: Photon number variance sigma^2
        sigma2 = np.var(window_photons)
        
        # Feature 3: Spectral sideband power Psb (simplified via FFT)
        # We take FFT of E * exp(i*phi) to get spectrum
        complex_field = window_E * np.exp(1j * window_phi)
        spectrum = np.abs(np.fft.fft(complex_field))**2
        # Assume carrier is at DC (index 0), sidebands are elsewhere
        # We sum power excluding the DC component and low frequencies
        Psb = np.sum(spectrum[5:-5]) # Crude approximation for out-of-band power
        
        # Feature 4: Phase decoherence delta_phi (RMS phase error)
        # Mean phase drift
        mean_phi = np.mean(window_phi)
        delta_phi = np.sqrt(np.mean((window_phi - mean_phi)**2))
        
        # Feature 5: QBER
        # QBER increases when phase error or photon variance deviates significantly
        # A simple synthetic mapping for the dataset generator:
        # QBER base is 0.01 (1%), increases with delta_phi and FIM variance
        qber = 0.01 + 0.1 * delta_phi + 0.05 * (sigma2 / (mu + 1e-9))
        
        features.append([mu, sigma2, Psb, delta_phi, qber])
        
    features = np.array(features)
    
    # We return the raw physical features.
    # The machine learning pipeline (StandardScaler) will handle global normalization.
    return features
