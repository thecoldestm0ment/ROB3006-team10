"""Load motor vibration CSVs and create one feature row per recording."""

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
SPLITS = {
    "train": tuple(range(11, 35)),
    "validation": tuple(range(35, 41)),
    "final_train": tuple(range(11, 41)),
    "test": tuple(range(1, 11)),
}
RANDOM_STATE = 42

FEATURE_NAMES = (
    "mean",
    "std",
    "rms",
    "max",
    "min",
    "peak_to_peak",
)


def extract_features(signal: np.ndarray) -> dict[str, float]:
    """Extract the shared six statistical features from one CSV recording."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("signal contains NaN or infinite values")

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "peak_to_peak": float(np.max(values) - np.min(values)),
    }


def load_dataset(data_root: str | Path = DATA_ROOT, progress: bool = True):
    """Load all recordings once.

    Returns (features, labels, metadata). Each CSV is one labeled sample;
    metadata keeps the source class, numbered folder, and relative CSV path.
    """
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
                signal = frame.iloc[:, 1].to_numpy(dtype=np.float64)
                feature_rows.append(extract_features(signal))
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

    features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    target = np.asarray(labels, dtype=np.int8)
    metadata = pd.DataFrame(records)
    if features.empty:
        raise ValueError(f"No CSV files found under {root}")
    if progress:
        print(
            f"Loaded {len(features):,} CSV files: "
            f"{int(np.sum(target == 0)):,} normal, {int(np.sum(target == 1)):,} abnormal"
        )
    return features, target, metadata


def load_split(split: str, data_root: str | Path = DATA_ROOT):
    """Load one documented split and return its features, labels, and metadata."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {tuple(SPLITS)}; got {split!r}")
    features, labels, metadata = load_dataset(data_root=data_root, progress=False)
    mask = metadata["folder_id"].isin(SPLITS[split]).to_numpy()
    return (
        features.loc[mask].reset_index(drop=True),
        labels[mask],
        metadata.loc[mask].reset_index(drop=True),
    )
