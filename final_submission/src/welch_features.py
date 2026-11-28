"""Welch power-spectrum features used by the selected v5 model."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

N_BANDS = 32
NPERSEG = 512
NOVERLAP = 256
LOG_FLOOR = 1e-12
EPS = np.finfo(float).eps


def extract_welch_features(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Return 32 log10 relative-power features from one vibration CSV."""
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    if time.ndim != 1 or values.ndim != 1 or time.size != values.size:
        raise ValueError("time and signal must be equal-length one-dimensional arrays")
    if values.size < NPERSEG or not np.isfinite(time).all() or not np.isfinite(values).all():
        raise ValueError(f"Expected at least {NPERSEG} finite measurements")

    intervals = np.diff(time)
    if np.any(intervals <= 0):
        raise ValueError("time values must be strictly increasing")
    fs = 1.0 / float(np.median(intervals))
    freqs, psd = welch(
        values - values.mean(),
        fs=fs,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )

    positive = freqs > 0
    band_index = np.ceil((freqs[positive] / (fs / 2.0)) * N_BANDS - 1e-9).astype(int) - 1
    if band_index.min() < 0 or band_index.max() != N_BANDS - 1:
        raise RuntimeError("Unexpected Welch frequency-band assignment")
    band_power = np.bincount(band_index, weights=psd[positive], minlength=N_BANDS)
    total_power = float(band_power.sum())
    if total_power <= EPS:
        relative_power = np.full(N_BANDS, 1.0 / N_BANDS)
    else:
        relative_power = band_power / total_power
    return np.log10(np.maximum(relative_power, LOG_FLOOR))
