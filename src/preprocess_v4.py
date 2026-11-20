"""V4 (week 3) feature sets for the fixed-model feature comparison.

A.  v3 baseline: the 16 amplitude-invariant features chosen by the final v3 model.
B.  Welch bands: log10 of relative band power from a Welch PSD (16 or 32 bands).
C.  Symlet-8 wavelet packet: relative energy of the level-4 (16) or level-5 (32) nodes.

B and C are each run with 16 and 32 bands so that, with the same model and
splits, method effects (same band count) and band-count effects (same method)
can be read separately. Every function takes one CSV's arrays and returns a
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


def extract_features_a(time: np.ndarray, signal: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    """v3 amplitude-invariant 16 features, using the existing v2/v3 extractors."""
    row = extract_features_v2(time, signal)
    row.update(extract_shape_features_v3(signal))
    return {name: row[name] for name in FEATURE_NAMES_A}, {}


# ---------------------------------------------------------------- B. Welch bands
WELCH_NPERSEG = 512
WELCH_NOVERLAP = 256
WELCH_LOG_FLOOR = 1e-12  # relative power below this is clipped before log10


def welch_feature_names(n_bands: int) -> tuple[str, ...]:
    return tuple(f"welch{n_bands}_band_{i:02d}_log10_rel_power" for i in range(1, n_bands + 1))


def sampling_rate_from_time(time: np.ndarray) -> float:
    """Sampling rate from the measured time column (same evidence as v2)."""
    intervals = np.diff(np.asarray(time, dtype=np.float64))
    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        raise ValueError("time values must be strictly increasing")
    return 1.0 / median_interval


def _welch_band_index(normalized_freq: np.ndarray, n_bands: int) -> np.ndarray:
    """Band j (0-based) holds bins with (j/n)·Nyquist < f <= ((j+1)/n)·Nyquist."""
    return np.ceil(normalized_freq * n_bands - 1e-9).astype(int) - 1


def welch_bins_per_band(n_bands: int) -> list[int]:
    """PSD bins per band for nperseg=512 (DC excluded, Nyquist in the last band)."""
    freqs = np.fft.rfftfreq(WELCH_NPERSEG, d=1.0)
    normalized = freqs[freqs > 0] / 0.5
    return np.bincount(_welch_band_index(normalized, n_bands), minlength=n_bands).tolist()


def extract_features_b(time: np.ndarray, signal: np.ndarray, n_bands: int) -> tuple[dict[str, float], dict[str, int]]:
    """Mean removal -> Welch PSD (Hann 512, overlap 256) -> n equal bands -> log10(band / total)."""
    values = np.asarray(signal, dtype=np.float64)
    if values.size < WELCH_NPERSEG:
        raise ValueError(f"signal shorter than nperseg={WELCH_NPERSEG}: {values.size}")
    fs = sampling_rate_from_time(time)
    freqs, psd = welch(
        values - values.mean(), fs=fs, window="hann", nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP, detrend=False, scaling="density", return_onesided=True,
    )
    positive = freqs > 0
    band_index = _welch_band_index(freqs[positive] / (fs / 2.0), n_bands)
    if band_index.min() < 0 or band_index.max() != n_bands - 1:
        raise RuntimeError("unexpected Welch band assignment")
    band_power = np.bincount(band_index, weights=psd[positive], minlength=n_bands)
    total = float(band_power.sum())
    flags = {"zero_total_power": 0, "floored_bands": 0}
    if total <= EPS:
        relative = np.full(n_bands, 1.0 / n_bands)
        flags["zero_total_power"] = 1
    else:
        relative = band_power / total
    flags["floored_bands"] = int(np.sum(relative < WELCH_LOG_FLOOR))
    features = np.log10(np.maximum(relative, WELCH_LOG_FLOOR))
    return dict(zip(welch_feature_names(n_bands), features.tolist())), flags


# ---------------------------------------------------------------- C. Symlet-8 wavelet packet
WPT_WAVELET = "sym8"
WPT_MODE = "periodization"


def wpt_feature_names(level: int) -> tuple[str, ...]:
    return tuple(f"wpt_sym8_l{level}_node_{i:02d}_rel_energy" for i in range(1, 2 ** level + 1))


def wpt_node_paths(level: int) -> list[str]:
    """Node paths at a level in frequency order (lowest band first)."""
    wp = pywt.WaveletPacket(data=np.zeros(1024), wavelet=WPT_WAVELET, mode=WPT_MODE, maxlevel=level)
    return [node.path for node in wp.get_level(level, order="freq")]


def extract_features_c(signal: np.ndarray, level: int) -> tuple[dict[str, float], dict[str, int]]:
    """Mean removal + RMS normalization -> sym8 packet to `level` -> node energy / total energy."""
    n_nodes = 2 ** level
    values = np.asarray(signal, dtype=np.float64)
    centered = values - values.mean()
    rms = float(np.sqrt(np.mean(np.square(centered))))
    flags = {"zero_signal": 0}
    if rms <= EPS:
        flags["zero_signal"] = 1
        return dict(zip(wpt_feature_names(level), [1.0 / n_nodes] * n_nodes)), flags
    wp = pywt.WaveletPacket(data=centered / rms, wavelet=WPT_WAVELET, mode=WPT_MODE, maxlevel=level)
    nodes = wp.get_level(level, order="freq")
    if len(nodes) != n_nodes:
        raise RuntimeError(f"expected {n_nodes} nodes, got {len(nodes)}")
    energy = np.array([float(np.sum(np.square(node.data))) for node in nodes])
    total = float(energy.sum())
    if total <= EPS:
        flags["zero_signal"] = 1
        relative = np.full(n_nodes, 1.0 / n_nodes)
    else:
        relative = energy / total
    return dict(zip(wpt_feature_names(level), relative.tolist())), flags
