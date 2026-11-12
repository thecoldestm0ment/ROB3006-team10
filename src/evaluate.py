"""Shared binary-classification metrics and CSV result persistence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

METRIC_COLUMNS = (
    "experiment_id",
    "split",
    "f1",
    "precision",
    "recall",
    "tn",
    "fp",
    "fn",
    "tp",
)


def evaluate_predictions(y_true, y_pred) -> dict[str, float | int]:
    """Evaluate binary predictions with abnormal (1) as the positive class."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def save_results(
    result: dict[str, float | int],
    output_dir: str | Path,
    experiment_id: str,
    split: str,
) -> Path:
    """Upsert one result row in the model folder's metrics.csv."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "metrics.csv"
    row = {
        "experiment_id": experiment_id,
        "split": split,
        **{name: result[name] for name in METRIC_COLUMNS[2:]},
    }
    new_row = pd.DataFrame([row], columns=METRIC_COLUMNS)

    if path.exists() and path.stat().st_size:
        existing = pd.read_csv(path)
        if not existing.empty:
            for column in METRIC_COLUMNS:
                if column not in existing.columns:
                    existing[column] = pd.NA
            existing = existing.loc[
                ~(
                    (existing["experiment_id"] == experiment_id)
                    & (existing["split"] == split)
                ),
                list(METRIC_COLUMNS),
            ]
            new_row = pd.concat([existing, new_row], ignore_index=True)

    new_row.to_csv(path, index=False, columns=METRIC_COLUMNS)
    return path
