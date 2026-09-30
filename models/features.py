import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

def extract_features(X_batch, window_size=10000, step=1000):
    N = len(X_batch)
    if N < window_size:
        raise ValueError("Batch size is smaller than window size")
        
    n_windows = (N - window_size) // step + 1
    
    features_list = []
    
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window = X_batch[start:end]
        
        I_A   = window[:, 2]
        I_B   = window[:, 3]
        d_phi = window[:, 4]
        c_d0  = window[:, 5]
        c_d1  = window[:, 6]

        # 0. Transmittance
        transmittance = np.mean(I_B) / (np.mean(I_A) + 1e-9)

        # 1. Intensity variance at Bob
        var_I_B = np.var(I_B)

        # 2. RMS phase decoherence
        rms_phi = np.sqrt(np.mean(d_phi ** 2))

        # 3. Click rate
        click_rate = np.mean(c_d0 + c_d1)

        # 4. Sifted QBER
        is_diff_0  = (d_phi < 0.1) | (d_phi > 2 * np.pi - 0.1)
        is_diff_pi = np.abs(d_phi - np.pi) < 0.1
        errors     = (is_diff_0 & (c_d1 > 0.5)) | (is_diff_pi & (c_d0 > 0.5))
        sifted     = (is_diff_0 | is_diff_pi) & ((c_d0 > 0.5) != (c_d1 > 0.5))
        qber       = np.sum(errors) / (np.sum(sifted) + 1e-9)

        # 5. Lag-1 phase autocorrelation
        phi_centered = d_phi - np.mean(d_phi)
        phi_var = np.var(d_phi) + 1e-12
        phase_autocorr = float(np.mean(phi_centered[:-1] * phi_centered[1:]) / phi_var)

        # 6. Intensity cross-correlation
        IA_c = I_A - np.mean(I_A)
        IB_c = I_B - np.mean(I_B)
        denom = float(np.std(I_A) * np.std(I_B)) + 1e-12
        intensity_xcorr = float(np.mean(IA_c * IB_c) / denom)

        # 7. Phase kurtosis
        phase_kurtosis = float(scipy_kurtosis(d_phi, fisher=True, bias=False))
        phase_kurtosis = float(np.clip(phase_kurtosis, -10.0, 10.0))

        # 8. Click asymmetry
        click_asymmetry = float(abs(np.mean(c_d0) - np.mean(c_d1)))

        # 9. Maximum Spectral Power
        # Perform FFT on the mean-centered intensity at Bob
        IB_centered = I_B - np.mean(I_B)
        fft_IB = np.abs(np.fft.rfft(IB_centered))
        # Skip DC component (index 0), get max spectral power
        spectral_power_max = float(np.max(fft_IB[1:]) if len(fft_IB) > 1 else 0.0)

        features_list.append([
            transmittance, var_I_B, rms_phi, click_rate, qber,
            phase_autocorr, intensity_xcorr, phase_kurtosis, click_asymmetry,
            spectral_power_max
        ])
        
    return np.array(features_list)

from scipy.signal import stft

def extract_spectrogram(X_batch, window_size=10000, step=1000, nperseg=256):
    """
    Extracts a 3-channel time-frequency spectrogram representation from the physical traces.
    Returns array of shape (n_windows, 3, F, T).
    """
    N = len(X_batch)
    if N < window_size:
        raise ValueError("Batch size is smaller than window size")
        
    n_windows = (N - window_size) // step + 1
    spectrograms = []
    
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window = X_batch[start:end]
        
        I_A   = window[:, 2]
        I_B   = window[:, 3]
        d_phi = window[:, 4]
        
        # Center the signals before STFT
        I_A_c = I_A - np.mean(I_A)
        I_B_c = I_B - np.mean(I_B)
        d_phi_c = d_phi - np.mean(d_phi)
        
        # Compute STFT
        _, _, Zxx_IA = stft(I_A_c, nperseg=nperseg)
        _, _, Zxx_IB = stft(I_B_c, nperseg=nperseg)
        _, _, Zxx_phi = stft(d_phi_c, nperseg=nperseg)
        
        # Stack channels (3, F, T)
        spec = np.stack([
            np.abs(Zxx_IA),
            np.abs(Zxx_IB),
            np.abs(Zxx_phi)
        ], axis=0)
        
        # Downsample 2x in freq, 4x in time to prevent MemoryError (OOM)
        spec = spec[:, ::2, ::4]
        
        spectrograms.append(spec)
        
    return np.array(spectrograms, dtype=np.float32)
