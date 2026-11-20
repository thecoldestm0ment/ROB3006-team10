"""V3 features and motor-folder spectra for vibration classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess_v2 import DATA_ROOT, FEATURE_NAMES_V2, LABELS, extract_features_v2

SHAPE_FEATURE_NAMES_V3 = (
    "waveform_skewness",
    "waveform_excess_kurtosis",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
)
FEATURE_NAMES_V3 = FEATURE_NAMES_V2 + SHAPE_FEATURE_NAMES_V3
AMPLITUDE_INVARIANT_FEATURES_V3 = tuple(
    name
    for name in FEATURE_NAMES_V2
    if name not in {"mean", "std", "rms", "max", "min", "peak_to_peak"}
) + SHAPE_FEATURE_NAMES_V3


def extract_shape_features_v3(signal: np.ndarray) -> dict[str, float]:
    """Return offset- and gain-invariant waveform-shape features."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 4 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite 1D array with at least 4 samples")

    centered = values - np.mean(values)
    absolute = np.abs(centered)
    rms = float(np.sqrt(np.mean(np.square(centered))))
    mean_absolute = float(np.mean(absolute))
    mean_sqrt_absolute_squared = float(np.square(np.mean(np.sqrt(absolute))))
    if rms <= np.finfo(float).eps:
        skewness = excess_kurtosis = crest_factor = 0.0
    else:
        standardized = centered / rms
        skewness = float(np.mean(np.power(standardized, 3)))
        excess_kurtosis = float(np.mean(np.power(standardized, 4)) - 3.0)
        crest_factor = float(np.max(absolute) / rms)

    peak = float(np.max(absolute))
    return {
        "waveform_skewness": skewness,
        "waveform_excess_kurtosis": excess_kurtosis,
        "crest_factor": crest_factor,
        "shape_factor": float(rms / mean_absolute) if mean_absolute else 0.0,
        "impulse_factor": float(peak / mean_absolute) if mean_absolute else 0.0,
        "clearance_factor": (
            float(peak / mean_sqrt_absolute_squared)
            if mean_sqrt_absolute_squared
            else 0.0
        ),
    }


def _normalized_spectrum(signal: np.ndarray) -> np.ndarray:
    centered = signal - np.mean(signal)
    power = np.square(np.abs(np.fft.rfft(centered)))
    power[0] = 0.0
    total = float(power.sum())
    if total <= np.finfo(float).eps:
        return np.zeros_like(power)
    return power / total


def load_dataset_v3(data_root: str | Path = DATA_ROOT, progress: bool = True):
    """Load only development motors 11–40 and collect per-motor spectra.

    Folder number is the grouping key. The same number under normal/abnormal is
    deliberately kept in one group, a conservative choice while motor identity
    correspondence between the two class trees is undocumented.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    folder_ids = tuple(range(11, 41))
    if any(folder_id < 11 or folder_id > 40 for folder_id in folder_ids):
        raise ValueError("V3 development loader must exclude test folders 1–10")

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    records: list[dict[str, object]] = []
    spectra_sum: dict[tuple[str, int], np.ndarray] = {}
    spectra_count: dict[tuple[str, int], int] = {}
    representatives: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    files = [
        (class_name, label, folder_id, csv_path)
        for class_name, label in LABELS.items()
        for folder_id in folder_ids
        for csv_path in sorted(
            (root / class_name / str(folder_id) / "time_vib").glob("*.csv")
        )
    ]
    if not files:
        raise ValueError(f"No CSV files found under {root}")

    for count, (class_name, label, folder_id, csv_path) in enumerate(files, start=1):
        frame = pd.read_csv(csv_path, header=None)
        if frame.shape[1] != 2:
            raise ValueError(
                f"Expected two columns (time, signal) in {csv_path}; found {frame.shape[1]}"
            )
        time = frame.iloc[:, 0].to_numpy(dtype=np.float64)
        signal = frame.iloc[:, 1].to_numpy(dtype=np.float64)
        try:
            row = extract_features_v2(time, signal)
            row.update(extract_shape_features_v3(signal))
        except ValueError as error:
            raise ValueError(f"Feature extraction failed for {csv_path}: {error}") from error

        spectrum = _normalized_spectrum(signal)
        key = (class_name, folder_id)
        spectra_sum[key] = spectra_sum.get(key, np.zeros_like(spectrum)) + spectrum
        spectra_count[key] = spectra_count.get(key, 0) + 1
        if folder_id == 11 and class_name not in representatives:
            representatives[class_name] = (time.copy(), signal.copy(), spectrum.copy())

        rows.append(row)
        labels.append(label)
        records.append(
            {
                "class_name": class_name,
                "folder_id": folder_id,
                "source": csv_path.relative_to(root).as_posix(),
            }
        )
        if progress and count % 1000 == 0:
            print(f"Loaded {count:,}/{len(files):,} development CSV files", flush=True)

    features = pd.DataFrame(rows, columns=FEATURE_NAMES_V3)
    target = np.asarray(labels, dtype=np.int8)
    metadata = pd.DataFrame(records)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature extraction produced NaN or infinite values")

    feature_sets = {
        "v2_all_16": features.loc[:, FEATURE_NAMES_V2].copy(),
        "v3_all_22": features.loc[:, FEATURE_NAMES_V3].copy(),
        "v3_amplitude_invariant_16": features.loc[
            :, AMPLITUDE_INVARIANT_FEATURES_V3
        ].copy(),
    }
    motor_spectra = {
        key: value / spectra_count[key] for key, value in spectra_sum.items()
    }
    if progress:
        print(
            f"Loaded {len(features):,} development CSV files; "
            f"folders 1-10 were not read.",
            flush=True,
        )
    return feature_sets, target, metadata, motor_spectra, representatives
