"""
Feature extraction for Track 1 (LSTM Autoencoder anomaly detection).

Converts a PREPROCESSED AE signal window into the (32, 30) temporal feature
tensor the Track 1 model expects. Run preprocess_signal.py (shared with
Track 2) on the raw signal FIRST, then pass the result here.

Pipeline: raw signal -> [preprocess_signal.py: bandpass + standardize]
          -> [this file: DWT temporal features]
Requires: numpy, scipy, PyWavelets
    pip install numpy scipy PyWavelets
"""

import numpy as np
import pywt
from scipy.stats import entropy, kurtosis

WAVELET = 'db4'
DWT_LEVEL = 5
N_SEGMENTS = 32       # timesteps fed to the LSTM


def extract_features_temporal(window, n_segments=N_SEGMENTS):
    """
    window: 1D numpy array, ALREADY preprocessed via preprocess_signal.py
            (bandpass filtered + standardized). Do NOT pass a raw signal here.
    Returns: np.ndarray of shape (n_segments, 30) -- 5 stats x 6 DWT subbands
             per timestep, subbands ordered [A5, D5, D4, D3, D2, D1]
             (pywt.wavedec's natural output order).
    """
    coeffs = pywt.wavedec(window, wavelet=WAVELET, level=DWT_LEVEL)

    timesteps = []
    for segment_idx in range(n_segments):
        segment_features = []
        for subband in coeffs:
            segments = np.array_split(subband, n_segments)
            seg = segments[segment_idx]

            mean = np.mean(seg)
            std = np.std(seg)
            energy = np.sum(seg ** 2)
            ent = entropy(np.abs(seg) + 1e-10)
            kurt = kurtosis(seg)

            segment_features.extend([mean, std, energy, ent, kurt])
        timesteps.append(segment_features)

    return np.array(timesteps, dtype=np.float32)  # (32, 30)


def prepare_track1_input(preprocessed_signal, scaler, top_k_indices):
    """
    Track 1 feature pipeline: preprocessed signal -> model-ready input.

    preprocessed_signal: 1D numpy array, output of preprocess_signal.py
                         (bandpass filtered + standardized). NOT a raw signal.
    scaler: the loaded StandardScaler object, fit on all 30 features
            during training (track1_temporal_scaler_v3.pkl).
    top_k_indices: numpy array of 20 feature-column indices (out of 30)
                   selected by MCFS during training
                   (ranked_indices_temporal_v3.npy[:20]).

    Returns: np.ndarray of shape (1, 32, 20), ready for the TFLite model.
    """
    feats_30 = extract_features_temporal(preprocessed_signal)  # (32, 30)

    # Scale using the 30-feature scaler BEFORE selecting columns -- the
    # scaler's mean/std were computed on all 30 features, so scaling must
    # happen first, then column selection. Doing this in the other order
    # silently produces wrong numbers.
    feats_scaled = scaler.transform(feats_30.reshape(-1, 30)).reshape(feats_30.shape)
    feats_20 = feats_scaled[:, top_k_indices]              # (32, 20)

    return np.expand_dims(feats_20, axis=0).astype(np.float32)  # (1, 32, 20)
