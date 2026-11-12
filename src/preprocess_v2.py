"""Version 2 features: time-domain statistics plus normalized FFT features."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("IC_PBL_DATA_ROOT", PROJECT_ROOT / "data" / "raw")
).expanduser()

LABELS = {"normal": 0, "abnormal": 1}
FEATURE_NAMES_V2 = (
    "mean",
    "std",
    "rms",
    "max",
    "min",
    "peak_to_peak",
    "dominant_frequency_hz",
    "dominant_power_ratio",
    "spectral_centroid_hz",
    "spectral_spread_hz",
    "spectral_entropy",
    "spectral_rolloff_85_hz",
    "band_power_0_0p1_nyquist",
    "band_power_0p1_0p25_nyquist",
    "band_power_0p25_0p5_nyquist",
    "band_power_0p5_1_nyquist",
)


def extract_features_v2(time: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    """Extract six time-domain and ten frequency-domain features.

    Frequency features use the measured time column, not a hard-coded sampling
    rate. Four relative power bands are expressed as fractions of Nyquist.
    """
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    if time.ndim != 1 or values.ndim != 1 or time.size != values.size or time.size < 4:
        raise ValueError("time and signal must be equal-length 1D arrays with >= 4 rows")
    if not np.isfinite(time).all() or not np.isfinite(values).all():
        raise ValueError("time or signal contains NaN or infinite values")

    intervals = np.diff(time)
    median_interval = float(np.median(intervals))
    if median_interval <= 0 or np.any(intervals <= 0):
        raise ValueError("time values must be strictly increasing")
    relative_jitter = float(np.max(np.abs(intervals - median_interval)) / median_interval)
    if relative_jitter > 0.02:
        raise ValueError(
            f"sampling interval varies by {relative_jitter:.1%}; FFT assumes uniform sampling"
        )

    centered = values - np.mean(values)
    power = np.square(np.abs(np.fft.rfft(centered)))
    frequencies = np.fft.rfftfreq(values.size, d=median_interval)
    # Exclude DC: the signal offset is represented by the time-domain mean.
    positive = frequencies > 0
    frequencies = frequencies[positive]
    power = power[positive]
    total_power = float(np.sum(power))
    if frequencies.size == 0 or total_power <= np.finfo(float).eps:
        spectral = {
            "dominant_frequency_hz": 0.0,
            "dominant_power_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_spread_hz": 0.0,
            "spectral_entropy": 0.0,
            "spectral_rolloff_85_hz": 0.0,
            "band_power_0_0p1_nyquist": 0.0,
            "band_power_0p1_0p25_nyquist": 0.0,
            "band_power_0p25_0p5_nyquist": 0.0,
            "band_power_0p5_1_nyquist": 0.0,
        }
    else:
        probabilities = power / total_power
        centroid = float(np.sum(frequencies * probabilities))
        entropy = -float(
            np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))
            / np.log(probabilities.size)
        ) if probabilities.size > 1 else 0.0
        cumulative = np.cumsum(probabilities)
        rolloff_index = min(int(np.searchsorted(cumulative, 0.85)), frequencies.size - 1)
        nyquist = float(frequencies[-1])
        normalized_frequency = frequencies / nyquist
        band_edges = (0.0, 0.1, 0.25, 0.5, 1.0)
        band_names = (
            "band_power_0_0p1_nyquist",
            "band_power_0p1_0p25_nyquist",
            "band_power_0p25_0p5_nyquist",
            "band_power_0p5_1_nyquist",
        )
        band_powers = {
            name: float(np.sum(probabilities[(normalized_frequency >= low) & (normalized_frequency < high)]))
            for name, low, high in zip(band_names, band_edges[:-1], band_edges[1:])
        }
        # Include the Nyquist bin in the final band when the sample count is even.
        band_powers[band_names[-1]] += float(
            np.sum(probabilities[normalized_frequency == 1.0])
        )
        peak_index = int(np.argmax(power))
        spectral = {
            "dominant_frequency_hz": float(frequencies[peak_index]),
            "dominant_power_ratio": float(probabilities[peak_index]),
            "spectral_centroid_hz": centroid,
            "spectral_spread_hz": float(
                np.sqrt(np.sum(np.square(frequencies - centroid) * probabilities))
            ),
            "spectral_entropy": entropy,
            "spectral_rolloff_85_hz": float(frequencies[rolloff_index]),
            **band_powers,
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "peak_to_peak": float(np.max(values) - np.min(values)),
        **spectral,
    }


def load_dataset_v2(data_root: str | Path = DATA_ROOT, progress: bool = True):
    """Load all recordings and return (features, labels, metadata)."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data root not found: {root}. Set IC_PBL_DATA_ROOT or pass --data-root."
        )

    feature_rows: list[dict[str, float]] = []
    labels: list[int] = []
    records: list[dict[str, object]] = []
    total_files = sum(
        len(list((root / class_name).glob("*/time_vib/*.csv")))
        for class_name in LABELS
    )
    processed = 0

    for class_name, label in LABELS.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        folder_ids = {
            int(folder.name)
            for folder in class_dir.iterdir()
            if folder.is_dir() and folder.name.isdigit()
        }
        missing_ids = set(range(1, 41)) - folder_ids
        if missing_ids:
            raise ValueError(
                f"{class_dir} is missing numbered folders: {sorted(missing_ids)}"
            )

        for folder_id in range(1, 41):
            signal_dir = class_dir / str(folder_id) / "time_vib"
            if not signal_dir.is_dir():
                raise FileNotFoundError(f"Expected data folder not found: {signal_dir}")
            csv_files = sorted(signal_dir.glob("*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in {signal_dir}")

            for csv_path in csv_files:
                frame = pd.read_csv(csv_path, header=None)
                if frame.shape[1] != 2:
                    raise ValueError(
                        f"Expected two columns (time, signal) in {csv_path}; "
                        f"found {frame.shape[1]}"
                    )
                time = frame.iloc[:, 0].to_numpy(dtype=np.float64)
                signal = frame.iloc[:, 1].to_numpy(dtype=np.float64)
                try:
                    row = extract_features_v2(time, signal)
                except ValueError as error:
                    raise ValueError(f"Feature extraction failed for {csv_path}: {error}") from error
                feature_rows.append(row)
                labels.append(label)
                records.append(
                    {
                        "class_name": class_name,
                        "folder_id": folder_id,
                        "source": csv_path.relative_to(root).as_posix(),
                    }
                )
                processed += 1
                if progress and processed % 1000 == 0:
                    print(f"Loaded {processed:,}/{total_files:,} CSV files")

    features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES_V2)
    target = np.asarray(labels, dtype=np.int8)
    metadata = pd.DataFrame(records)
    if features.empty:
        raise ValueError(f"No CSV files found under {root}")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature extraction produced NaN or infinite values")
    if progress:
        print(
            f"Loaded {len(features):,} CSV files: "
            f"{int(np.sum(target == 0)):,} normal, {int(np.sum(target == 1)):,} abnormal"
        )
    return features, target, metadata
