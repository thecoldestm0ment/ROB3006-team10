"""V4 (week 3) feature sets for the fixed-model feature comparison.

A. v3 baseline: the 16 amplitude-invariant features chosen by the final v3 model.
B. Welch 16 bands: log10 of relative band power from a Welch PSD.
C. Symlet-8 wavelet packet: relative energy of the 32 level-5 nodes.

Only the extraction functions live here; data loading and evaluation are in
``run_experiments_v4``. Every function takes one CSV's arrays and returns a
fixed-length dict, so features are computed once per CSV and reused.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import welch

from .preprocess_v2 import extract_features_v2
from .preprocess_v3 import AMPLITUDE_INVARIANT_FEATURES_V3, extract_shape_features_v3

EPS = np.finfo(float).eps

# ---------------------------------------------------------------- A. v3 baseline
FEATURE_NAMES_A = tuple(AMPLITUDE_INVARIANT_FEATURES_V3)


def extract_features_a(time: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    """v3 amplitude-invariant 16 features, using the existing v2/v3 extractors."""
    row = extract_features_v2(time, signal)
    row.update(extract_shape_features_v3(signal))
    return {name: row[name] for name in FEATURE_NAMES_A}


# ---------------------------------------------------------------- B. Welch 16 bands
WELCH_NPERSEG = 512
WELCH_NOVERLAP = 256
WELCH_BANDS = 16
WELCH_LOG_FLOOR = 1e-12  # relative power below this is clipped before log10
FEATURE_NAMES_B = tuple(f"welch_band_{i:02d}_log10_rel_power" for i in range(1, WELCH_BANDS + 1))


def sampling_rate_from_time(time: np.ndarray) -> float:
    """Sampling rate from the measured time column (same evidence as v2)."""
    intervals = np.diff(np.asarray(time, dtype=np.float64))
    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        raise ValueError("time values must be strictly increasing")
    return 1.0 / median_interval


def welch_band_edges_normalized() -> np.ndarray:
    """Band edges as fractions of Nyquist: 0, 1/16, ..., 1 (DC itself excluded)."""
    return np.linspace(0.0, 1.0, WELCH_BANDS + 1)


def extract_features_b(time: np.ndarray, signal: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    """Welch PSD (Hann 512, overlap 256) -> 16 equal-width bands -> log10(band / total).

    Returns the features and flags used to report zero-power handling.
    Band j (0-based) holds positive-frequency bins k with (j/16)·Nyquist < f_k <= ((j+1)/16)·Nyquist,
    so DC (k = 0) is excluded and the Nyquist bin is in the last band.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.size < WELCH_NPERSEG:
        raise ValueError(f"signal shorter than nperseg={WELCH_NPERSEG}: {values.size}")
    fs = sampling_rate_from_time(time)
    centered = values - values.mean()
    freqs, psd = welch(
        centered, fs=fs, window="hann", nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP,
        detrend=False, scaling="density", return_onesided=True,
    )
    nyquist = fs / 2.0
    positive = freqs > 0
    normalized = freqs[positive] / nyquist
    power = psd[positive]
    band_index = np.ceil(normalized * WELCH_BANDS - 1e-9).astype(int) - 1
    if band_index.min() < 0 or band_index.max() != WELCH_BANDS - 1:
        raise RuntimeError("unexpected Welch band assignment")
    band_power = np.bincount(band_index, weights=power, minlength=WELCH_BANDS)
    total = float(band_power.sum())
    flags = {"zero_total_power": 0, "floored_bands": 0}
    if total <= EPS:
        relative = np.full(WELCH_BANDS, 1.0 / WELCH_BANDS)
        flags["zero_total_power"] = 1
    else:
        relative = band_power / total
    flags["floored_bands"] = int(np.sum(relative < WELCH_LOG_FLOOR))
    features = np.log10(np.maximum(relative, WELCH_LOG_FLOOR))
    return dict(zip(FEATURE_NAMES_B, features.tolist())), flags


def welch_bins_per_band(n_samples: int = 1024) -> np.ndarray:
    """Number of PSD bins in each band for a given length (for the record)."""
    freqs = np.fft.rfftfreq(WELCH_NPERSEG, d=1.0)  # normalized; Nyquist = 0.5
    normalized = freqs[freqs > 0] / 0.5
    band_index = np.ceil(normalized * WELCH_BANDS - 1e-9).astype(int) - 1
    return np.bincount(band_index, minlength=WELCH_BANDS)


# ---------------------------------------------------------------- C. Symlet-8 wavelet packet
WPT_WAVELET = "sym8"
WPT_MODE = "periodization"
WPT_LEVEL = 5
WPT_NODES = 2 ** WPT_LEVEL
FEATURE_NAMES_C = tuple(f"wpt_sym8_l5_node_{i:02d}_rel_energy" for i in range(1, WPT_NODES + 1))


def wpt_node_paths() -> list[str]:
    """Level-5 node paths in frequency order (lowest band first)."""
    wp = pywt.WaveletPacket(data=np.zeros(1024), wavelet=WPT_WAVELET, mode=WPT_MODE, maxlevel=WPT_LEVEL)
    return [node.path for node in wp.get_level(WPT_LEVEL, order="freq")]


def extract_features_c(signal: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    """Mean removal + RMS normalization -> sym8 packet, level 5 -> node energy / total energy."""
    values = np.asarray(signal, dtype=np.float64)
    centered = values - values.mean()
    rms = float(np.sqrt(np.mean(np.square(centered))))
    flags = {"zero_signal": 0}
    if rms <= EPS:
        flags["zero_signal"] = 1
        relative = np.full(WPT_NODES, 1.0 / WPT_NODES)
        return dict(zip(FEATURE_NAMES_C, relative.tolist())), flags
    normalized = centered / rms
    wp = pywt.WaveletPacket(data=normalized, wavelet=WPT_WAVELET, mode=WPT_MODE, maxlevel=WPT_LEVEL)
    nodes = wp.get_level(WPT_LEVEL, order="freq")
    if len(nodes) != WPT_NODES:
        raise RuntimeError(f"expected {WPT_NODES} nodes, got {len(nodes)}")
    energy = np.array([float(np.sum(np.square(node.data))) for node in nodes])
    total = float(energy.sum())
    if total <= EPS:
        flags["zero_signal"] = 1
        relative = np.full(WPT_NODES, 1.0 / WPT_NODES)
    else:
        relative = energy / total
    return dict(zip(FEATURE_NAMES_C, relative.tolist())), flags
