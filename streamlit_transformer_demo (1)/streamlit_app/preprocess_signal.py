"""
Shared preprocessing for BOTH Track 1 and Track 2.

Both tracks were trained on signals that went through this SAME pipeline:
    1. Butterworth bandpass filter, 20 kHz - 400 kHz, 2nd order
    2. Z-score standardization (zero mean, unit variance)

Run this ONCE per incoming raw signal, then pass the result to both
run_track1() and classify_track2(). Do not preprocess twice, and do not
feed a raw/unpreprocessed signal directly to either track's model --
both expect this exact preprocessing to have already been applied.

Requires: numpy, scipy
    pip install numpy scipy
"""

import numpy as np
from scipy.signal import butter, filtfilt

FS = 1_000_000    # sampling rate (Hz) both tracks were trained on
LOWCUT = 20e3     # Hz
HIGHCUT = 400e3   # Hz


def preprocess_signal(raw_signal, fs=FS):
    """
    raw_signal: 1D numpy array, RAW signal straight from the sensor.
    Returns: 1D numpy array, bandpass-filtered + standardized -- this is
             the ONLY form of the signal either track's model has ever
             seen during training.
    """
    b, a = butter(2, [LOWCUT, HIGHCUT], btype='band', fs=fs)
    filtered = filtfilt(b, a, raw_signal)
    standardized = (filtered - np.mean(filtered)) / np.std(filtered)
    return standardized
