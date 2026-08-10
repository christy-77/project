"""
Feature extraction for Track 2 fault classification (XGBoost).

IMPORTANT: this expects a PREPROCESSED signal window as input, NOT a raw
signal straight from the sensor. "Preprocessed" means:
    1. Butterworth bandpass filter, 20 kHz - 400 kHz, 2nd order
    2. Z-score standardization (zero mean, unit variance)
This is the SAME preprocessing Track 1 applies. The model was trained on
signals that went through exactly this pipeline (fault_preprocessed_v3.npz)
-- feeding it a raw, unfiltered signal will silently produce wrong feature
values and wrong predictions, with no error or crash to warn you.

Converts a preprocessed AE signal window into the 9-feature vector the
production model expects. Must run on incoming signal windows AFTER
preprocessing and BEFORE calling the model's .predict().

Requires: PyWavelets, numpy, scipy
    pip install PyWavelets numpy scipy
"""

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, hilbert

FS = 1_000_000          # sampling rate (Hz) the model was trained on
WAVELET = 'db4'
DWT_LEVEL = 5
DWT_SUBBANDS_KEPT = ['D3', 'D2']  # A5/D5 excluded - near-zero energy, no separating info

# Exact order the model expects features in. Do not reorder.
FEATURE_NAMES = [
    'dwt_frac_D3',
    'dwt_frac_D2',
    'wavelet_entropy',
    'frac_150_400',
    'frac_60_150',
    'spec_centroid',
    'spec_spread',
    'peak_freq',
    'mean_envelope_width',
]


def bandpass(signal, low, high, fs, order=2):
    b, a = butter(order, [low, high], btype='band', fs=fs)
    return filtfilt(b, a, signal)


def extract_features(signal, fs=FS):
    """
    signal: 1D numpy array, a single AE window sampled at `fs`.
    Returns: dict of the 9 features (see FEATURE_NAMES for required order
             when building the model input array).
    """
    feats = {}

    # --- DWT subband energies (all 6 bands computed, only D3/D2 kept as
    #     individual features; entropy below uses all 6) ---
    coeffs = pywt.wavedec(signal, WAVELET, level=DWT_LEVEL)
    subband_names = ['A5', 'D5', 'D4', 'D3', 'D2', 'D1']
    energies = {name: np.sum(c ** 2) for name, c in zip(subband_names, coeffs)}
    total_dwt_energy = sum(energies.values()) + 1e-10

    for name in DWT_SUBBANDS_KEPT:
        feats[f'dwt_frac_{name}'] = energies[name] / total_dwt_energy

    # --- Wavelet Shannon entropy over ALL 6 subbands (correctly normalized) ---
    all_probs = np.array([energies[name] for name in subband_names]) / total_dwt_energy
    all_probs = all_probs[all_probs > 0]
    if len(all_probs) > 0:
        feats['wavelet_entropy'] = -np.sum(all_probs * np.log2(all_probs))
    else:
        feats['wavelet_entropy'] = 0.0

    # --- Bandpass energy ratios ---
    total_energy_safe = np.sum(signal ** 2) + 1e-10
    bp_mech = bandpass(signal, 150e3, 400e3, fs)
    feats['frac_150_400'] = np.sum(bp_mech ** 2) / total_energy_safe
    bp_pd = bandpass(signal, 60e3, 150e3, fs)
    feats['frac_60_150'] = np.sum(bp_pd ** 2) / total_energy_safe

    # --- FFT-based spectral features ---
    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
    mag = np.abs(np.fft.rfft(signal))

    centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-10)
    feats['spec_centroid'] = centroid
    feats['spec_spread'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / (np.sum(mag) + 1e-10))

    peak_idx = np.argmax(mag)
    feats['peak_freq'] = freqs[peak_idx]

    # --- Hilbert envelope burst duration ---
    envelope = np.abs(hilbert(signal))
    mean_env = np.mean(envelope)
    std_env = np.std(envelope)
    block_threshold = mean_env + (2.0 * std_env)
    above_threshold = (envelope > block_threshold).astype(int)
    edges = np.diff(above_threshold)
    rising_edges = np.where(edges == 1)[0]
    falling_edges = np.where(edges == -1)[0]
    if len(falling_edges) > 0 and len(rising_edges) > 0:
        if falling_edges[0] < rising_edges[0]:
            falling_edges = falling_edges[1:]
        min_len = min(len(rising_edges), len(falling_edges))
        if min_len > 0:
            pulse_durations = (falling_edges[:min_len] - rising_edges[:min_len]) / fs * 1000
            feats['mean_envelope_width'] = np.mean(pulse_durations)
        else:
            feats['mean_envelope_width'] = 0.0
    else:
        feats['mean_envelope_width'] = 0.0

    return feats


def features_to_vector(feats):
    """Convert the feats dict into a correctly-ordered list for model input."""
    return [feats[name] for name in FEATURE_NAMES]
