"""Tune logistic regression and random forest, then run the final held-out test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import evaluate_predictions, save_results
from .preprocess import DATA_ROOT, FEATURE_NAMES, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_IDS = set(range(11, 35))
VALIDATION_IDS = set(range(35, 41))
FINAL_TRAIN_IDS = set(range(11, 41))
TEST_IDS = set(range(1, 11))
RANDOM_STATE = 42


def _metrics_for(model, x, y) -> dict[str, float | int]:
    return evaluate_predictions(y, model.predict(x))


def _select_by_validation(candidates, x_train, y_train, x_validation, y_validation):
    records = []
    best_model = None
    best_params = None
    best_metrics = None
    best_key = None

    for params, model in candidates:
        model.fit(x_train, y_train)
        metrics = _metrics_for(model, x_validation, y_validation)
        records.append({"params": params, **metrics})
        score_key = (metrics["f1"], metrics["recall"], metrics["precision"])
        if best_key is None or score_key > best_key:
            best_key = score_key
            best_model = model
            best_params = params
            best_metrics = metrics
        print(
            f"  validation {params}: F1={metrics['f1']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}"
        )

    return best_model, best_params, best_metrics, records


def _plot_confusion_matrix(metrics, model_name: str, path: Path) -> None:
    matrix = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, f"{value:,}", ha="center", va="center", fontsize=12)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal (0)", "Abnormal (1)"],
        yticklabels=["Normal (0)", "Abnormal (1)"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{model_name} — held-out test",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_experiment_record(
    path: Path,
    title: str,
    best_params: dict,
    validation_metrics: dict,
    validation_records: list[dict],
    test_metrics: dict,
) -> None:
    rows = [
        "| Parameters | F1 | Precision | Recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for record in validation_records:
        rows.append(
            f"| `{json.dumps(record['params'], sort_keys=True)}` "
            f"| {record['f1']:.4f} | {record['precision']:.4f} "
            f"| {record['recall']:.4f} |"
        )
    text = f"""# {title} experiment

## Experiment information

- Experiment ID: `{title.lower().replace(' ', '_')}_stats_v1`
- Sample unit: one CSV recording
- Features: {', '.join(FEATURE_NAMES)}
- Labels: normal=0, abnormal=1 (positive class)
- Validation: train folder IDs 11-34, validate on IDs 35-40
- Final test: refit on IDs 11-40, evaluate once on IDs 1-10
- Random state: {RANDOM_STATE}
- Python dependencies: scikit-learn {sklearn.__version__}, pandas {pd.__version__}
- Model selection used validation F1; test data was not used to choose parameters.

## Validation parameter search

{chr(10).join(rows)}

Selected parameters: `{json.dumps(best_params, sort_keys=True)}`

Selected validation metrics: F1={validation_metrics['f1']:.4f}, \
Precision={validation_metrics['precision']:.4f}, \
Recall={validation_metrics['recall']:.4f}.

## Final held-out test

- F1: {test_metrics['f1']:.4f}
- Precision: {test_metrics['precision']:.4f}
- Recall: {test_metrics['recall']:.4f}
- Confusion matrix counts: TN={test_metrics['tn']}, FP={test_metrics['fp']}, \
FN={test_metrics['fn']}, TP={test_metrics['tp']}
- Figure: `confusion_matrix.png`

The final-test result is reported separately from validation and was not used for
model selection. Confirm with the instructor whether the final reporting unit
and F1 averaging rule should differ from per-CSV binary F1.
"""
    path.write_text(text, encoding="utf-8")


def _run_model(
    model_name: str,
    output_folder: str,
    candidates,
    features: pd.DataFrame,
    labels: np.ndarray,
    metadata: pd.DataFrame,
):
    folder_ids = metadata["folder_id"]
    train_mask = folder_ids.isin(TRAIN_IDS).to_numpy()
    validation_mask = folder_ids.isin(VALIDATION_IDS).to_numpy()
    final_train_mask = folder_ids.isin(FINAL_TRAIN_IDS).to_numpy()
    test_mask = folder_ids.isin(TEST_IDS).to_numpy()

    x_train, y_train = features.loc[train_mask], labels[train_mask]
    x_validation, y_validation = features.loc[validation_mask], labels[validation_mask]
    x_final_train, y_final_train = features.loc[final_train_mask], labels[final_train_mask]
    x_test, y_test = features.loc[test_mask], labels[test_mask]
    if not all(len(values) for values in (y_train, y_validation, y_test)):
        raise ValueError("A split is empty; check the numbered data folders.")

    print(f"\n=== {model_name} ===")
    best_model, best_params, validation_metrics, records = _select_by_validation(
        candidates, x_train, y_train, x_validation, y_validation
    )

    # The selected model is re-fit on IDs 11-40 before the untouched test IDs.
    final_model = best_model
    final_model.fit(x_final_train, y_final_train)
    test_metrics = _metrics_for(final_model, x_test, y_test)

    output_dir = PROJECT_ROOT / "results" / "v1" / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = f"{output_folder}_stats_v1"
    save_results(validation_metrics, output_dir, experiment_id, "validation_35_40")
    save_results(test_metrics, output_dir, experiment_id, "final_test_1_10")
    _plot_confusion_matrix(
        test_metrics, model_name, output_dir / "confusion_matrix.png"
    )
    _write_experiment_record(
        output_dir / "experiment.md",
        model_name,
        best_params,
        validation_metrics,
        records,
        test_metrics,
    )
    print(
        f"Selected: {best_params}\n"
        f"Final test: F1={test_metrics['f1']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}"
    )
    return {
        "model": model_name,
        "experiment_id": experiment_id,
        "selected_params": json.dumps(best_params, sort_keys=True),
        "validation_f1": validation_metrics["f1"],
        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
    }


def _logistic_candidates():
    for c_value in (0.01, 0.1, 1.0, 10.0, 100.0):
        params = {"C": c_value, "max_iter": 2000}
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value, max_iter=2000, random_state=RANDOM_STATE
                    ),
                ),
            ]
        )
        yield params, model


def _random_forest_candidates():
    for n_estimators in (200, 500):
        for max_depth in (None, 20):
            for min_samples_leaf in (1, 2):
                params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "random_state": RANDOM_STATE,
                }
                model = RandomForestClassifier(
                    **params,
                    n_jobs=-1,
                )
                yield params, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Folder containing normal/ and abnormal/ (default: IC_PBL_DATA_ROOT or data/raw)",
    )
    args = parser.parse_args()

    print(f"Data root: {args.data_root}")
    features, labels, metadata = load_dataset(args.data_root)
    print(
        "Split sizes: "
        f"train={int(metadata['folder_id'].isin(TRAIN_IDS).sum()):,}, "
        f"validation={int(metadata['folder_id'].isin(VALIDATION_IDS).sum()):,}, "
        f"final_train={int(metadata['folder_id'].isin(FINAL_TRAIN_IDS).sum()):,}, "
        f"test={int(metadata['folder_id'].isin(TEST_IDS).sum()):,} CSVs"
    )

    results = [
        _run_model(
            "Logistic Regression",
            "logistic_regression",
            _logistic_candidates(),
            features,
            labels,
            metadata,
        ),
        _run_model(
            "Random Forest",
            "random_forest",
            _random_forest_candidates(),
            features,
            labels,
            metadata,
        ),
    ]

    comparison_path = PROJECT_ROOT / "results" / "v1" / "model_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(comparison_path, index=False)
    print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
