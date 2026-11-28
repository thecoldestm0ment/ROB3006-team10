"""CSV-level binary classification metrics."""

from __future__ import annotations

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict[str, float | int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
