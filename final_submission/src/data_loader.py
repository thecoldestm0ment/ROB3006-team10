"""Load labeled CSV recordings from the normal/abnormal folder structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"normal": 0, "abnormal": 1}
TRAIN_IDS = tuple(range(11, 41))
TEST_IDS = tuple(range(1, 11))


def load_dataset(root: Path, motor_ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Load Welch features and binary labels for the requested motor folders."""
    from .welch_features import extract_welch_features

    rows, labels = [], []
    for class_name, label in LABELS.items():
        for motor in motor_ids:
            folder = root / class_name / str(motor) / "time_vib"
            files = sorted(folder.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No CSV files found in {folder}")
            for path in files:
                frame = pd.read_csv(path, header=None)
                values = frame.to_numpy(dtype=np.float64)
                if values.ndim != 2 or values.shape[1] != 2:
                    raise ValueError(f"Expected 2 columns (time, vibration) in {path}")
                features = extract_welch_features(values[:, 0], values[:, 1])
                rows.append(features)
                labels.append(label)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int8)
