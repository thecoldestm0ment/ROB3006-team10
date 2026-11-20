"""Motor-grouped nested CV, FFT plots, and F1-based final model selection."""

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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluate import evaluate_predictions
from .preprocess_v2 import DATA_ROOT
from .preprocess_v3 import load_dataset_v3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_IDS = set(range(11, 41))
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
FINAL_CV_FOLDS = 5

MODEL_SPECS = {
    "Logistic Regression": {
        "output_name": "logistic_regression",
        "score_type": "probability",
        "thresholds": (0.35, 0.50, 0.65),
        "candidates": [{"C": c} for c in (0.001, 0.01)],
    },
    "Random Forest": {
        "output_name": "random_forest",
        "score_type": "probability",
        "thresholds": (0.35, 0.50, 0.65),
        "candidates": [
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": leaf}
            for leaf in (1, 2)
        ],
    },
    "RBF SVM": {
        "output_name": "rbf_svm",
        "score_type": "decision",
        "thresholds": (-0.50, 0.00, 0.50),
        "candidates": [{"C": c, "gamma": "scale"} for c in (1.0, 10.0)],
    },
}


def _make_model(model_name: str, params: dict):
    if model_name == "Logistic Regression":
        model = LogisticRegression(
            **params, max_iter=3000, random_state=RANDOM_STATE
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", model)])
    if model_name == "Random Forest":
        return RandomForestClassifier(
            **params, random_state=RANDOM_STATE, n_jobs=-1
        )
    return Pipeline([("scaler", StandardScaler()), ("classifier", SVC(**params))])


def _scores(model, features, score_type):
    if score_type == "probability":
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def _predict(scores, threshold):
    return (scores >= threshold).astype(np.int8)


def _search_one(model_name, feature_name, x, y, groups, n_splits):
    """Select parameters and threshold using grouped CV, without held-out groups."""
    spec = MODEL_SPECS[model_name]
    folds = list(GroupKFold(n_splits=n_splits).split(x, y, groups=groups))
    rows = []
    best = None
    for params in spec["candidates"]:
        fold_targets, fold_scores = [], []
        for fold_number, (fit_idx, hold_idx) in enumerate(folds, start=1):
            model = _make_model(model_name, params)
            model.fit(x.iloc[fit_idx], y[fit_idx])
            fold_targets.append(y[hold_idx])
            fold_scores.append(_scores(model, x.iloc[hold_idx], spec["score_type"]))

        for threshold in spec["thresholds"]:
            metrics = [
                evaluate_predictions(target, _predict(score, threshold))
                for target, score in zip(fold_targets, fold_scores)
            ]
            mean_f1 = float(np.mean([item["f1"] for item in metrics]))
            std_f1 = float(np.std([item["f1"] for item in metrics]))
            mean_recall = float(np.mean([item["recall"] for item in metrics]))
            mean_precision = float(np.mean([item["precision"] for item in metrics]))
            row = {
                "feature_set": feature_name,
                "model": model_name,
                "parameters": json.dumps(params, sort_keys=True),
                "threshold": float(threshold),
                "mean_fold_f1": mean_f1,
                "std_fold_f1": std_f1,
                "mean_fold_precision": mean_precision,
                "mean_fold_recall": mean_recall,
            }
            rows.append(row)
            default_threshold = 0.5 if spec["score_type"] == "probability" else 0.0
            key = (mean_f1, -std_f1, mean_recall, -abs(threshold - default_threshold))
            if best is None or key > best[0]:
                best = (key, params, row)

    search = pd.DataFrame(rows).sort_values(
        ["mean_fold_f1", "std_fold_f1", "mean_fold_recall"],
        ascending=[False, True, False],
    )
    return best[1], best[2], search


def _nested_motor_cv(feature_sets, labels, metadata):
    """Estimate each model family with feature selection inside grouped CV."""
    groups_all = metadata["folder_id"].to_numpy()
    outer = GroupKFold(n_splits=OUTER_FOLDS)
    predictions_by_model = {
        name: np.full(labels.shape, -1, dtype=np.int8) for name in MODEL_SPECS
    }
    fold_rows, motor_rows, prediction_tables = [], [], []

    for model_name in MODEL_SPECS:
        predictions = predictions_by_model[model_name]
        for outer_fold, (train_idx, hold_idx) in enumerate(
            outer.split(next(iter(feature_sets.values())), labels, groups=groups_all),
            start=1,
        ):
            train_groups = groups_all[train_idx]
            hold_groups = sorted(np.unique(groups_all[hold_idx]).tolist())
            feature_candidates = []
            for feature_name, all_features in feature_sets.items():
                x_train = all_features.iloc[train_idx].reset_index(drop=True)
                params, cv_row, _ = _search_one(
                    model_name, feature_name, x_train, labels[train_idx],
                    train_groups, INNER_FOLDS
                )
                feature_candidates.append(
                    (cv_row["mean_fold_f1"], -cv_row["std_fold_f1"],
                     feature_name, params, cv_row)
                )

            _, _, feature_name, params, cv_row = max(
                feature_candidates, key=lambda item: (item[0], item[1])
            )
            all_features = feature_sets[feature_name]
            model = _make_model(model_name, params)
            model.fit(all_features.iloc[train_idx], labels[train_idx])
            hold_scores = _scores(
                model, all_features.iloc[hold_idx], MODEL_SPECS[model_name]["score_type"]
            )
            hold_pred = _predict(hold_scores, cv_row["threshold"])
            predictions[hold_idx] = hold_pred
            metrics = evaluate_predictions(labels[hold_idx], hold_pred)
            fold_rows.append({
                "model": model_name,
                "outer_fold": outer_fold,
                "held_out_folder_ids": ",".join(str(i) for i in hold_groups),
                "selected_feature_set": feature_name,
                "selected_params": json.dumps(params, sort_keys=True),
                "threshold": cv_row["threshold"],
                "inner_cv_mean_f1": cv_row["mean_fold_f1"],
                **metrics,
            })
            for folder_id in hold_groups:
                mask = groups_all[hold_idx] == folder_id
                motor_metrics = evaluate_predictions(
                    labels[hold_idx][mask], hold_pred[mask]
                )
                motor_rows.append({
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "folder_id": int(folder_id),
                    "selected_feature_set": feature_name,
                    "n_csv": int(mask.sum()),
                    **motor_metrics,
                })
            print(
                f"Nested {model_name} fold {outer_fold}/{OUTER_FOLDS}: "
                f"held out motors {hold_groups}; {feature_name}, F1={metrics['f1']:.4f}",
                flush=True,
            )

        if np.any(predictions < 0):
            raise RuntimeError(f"Nested CV missed predictions for {model_name}")
        prediction_tables.append(pd.DataFrame({
            "model": model_name,
            "source": metadata["source"],
            "folder_id": groups_all,
            "class_name": metadata["class_name"],
            "y_true": labels,
            "y_pred": predictions,
        }))

    summary_rows = []
    for model_name, predictions in predictions_by_model.items():
        pooled = evaluate_predictions(labels, predictions)
        model_folds = [row for row in fold_rows if row["model"] == model_name]
        fold_f1 = [row["f1"] for row in model_folds]
        summary_rows.append({
            "model": model_name,
            **pooled,
            "outer_folds": OUTER_FOLDS,
            "mean_outer_fold_f1": float(np.mean(fold_f1)),
            "std_outer_fold_f1": float(np.std(fold_f1)),
            "n_csv": int(len(labels)),
            "n_motor_groups": int(len(np.unique(groups_all))),
        })
    summary = pd.DataFrame(summary_rows).sort_values("f1", ascending=False)
    return (
        summary,
        pd.DataFrame(fold_rows),
        pd.DataFrame(motor_rows),
        pd.concat(prediction_tables, ignore_index=True),
    )

def _final_cv_comparison(feature_sets, labels, metadata):
    groups = metadata["folder_id"].to_numpy()
    comparison = []
    searches = []
    for model_name in MODEL_SPECS:
        for feature_name, features in feature_sets.items():
            params, best_row, search = _search_one(
                model_name,
                feature_name,
                features.reset_index(drop=True),
                labels,
                groups,
                FINAL_CV_FOLDS,
            )
            comparison.append(
                {
                    "model": model_name,
                    "feature_set": feature_name,
                    "selected_params": json.dumps(params, sort_keys=True),
                    "threshold": best_row["threshold"],
                    "cv_mean_f1": best_row["mean_fold_f1"],
                    "cv_std_f1": best_row["std_fold_f1"],
                    "cv_mean_precision": best_row["mean_fold_precision"],
                    "cv_mean_recall": best_row["mean_fold_recall"],
                    "selected_as_final": False,
                }
            )
            searches.append(search)
            print(
                f"Full development CV {model_name}/{feature_name}: "
                f"mean F1={best_row['mean_fold_f1']:.4f}",
                flush=True,
            )
    table = pd.DataFrame(comparison).sort_values(
        ["cv_mean_f1", "cv_std_f1", "cv_mean_recall"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    return table, pd.concat(searches, ignore_index=True)


def _save_spectrum_figures(motor_spectra, representatives, output_dir):
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, 1.0, 513)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, class_name, color in zip(axes, ("normal", "abnormal"), ("#2878B5", "#D9534F")):
        per_motor = [
            spectrum for (label, _folder), spectrum in sorted(motor_spectra.items())
            if label == class_name
        ]
        matrix = np.vstack(per_motor)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        for spectrum in matrix:
            ax.plot(x, spectrum, color=color, alpha=0.10, linewidth=0.6)
        ax.plot(x, mean, color=color, linewidth=2.0, label="Motor mean")
        ax.fill_between(x, np.maximum(mean - std, 0), mean + std, color=color, alpha=0.18)
        ax.set_title(f"{class_name.title()} ({len(matrix)} motor folders)")
        ax.set_xlabel("Normalized frequency (fraction of Nyquist)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Mean normalized spectral power per motor")
    fig.suptitle("Motor-level mean vibration spectra — folders 11–40 only")
    fig.tight_layout()
    fig.savefig(figure_dir / "motor_spectra_by_class.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for column, class_name in enumerate(("normal", "abnormal")):
        time, signal, spectrum = representatives[class_name]
        axes[0, column].plot(time, signal, linewidth=0.8)
        axes[0, column].set_title(f"{class_name.title()} example — folder 11")
        axes[0, column].set_xlabel("Time (s)")
        axes[0, column].set_ylabel("Vibration signal")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].plot(x, spectrum, linewidth=0.9)
        axes[1, column].set_xlabel("Normalized frequency (fraction of Nyquist)")
        axes[1, column].set_ylabel("Normalized spectral power")
        axes[1, column].grid(alpha=0.2)
    fig.suptitle("Example waveforms and FFT spectra (development data only)")
    fig.tight_layout()
    fig.savefig(figure_dir / "representative_waveform_fft.png", dpi=170)
    plt.close(fig)


def _save_confusion(metrics, path, title="Nested motor-group cross-validation"):
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
        title=title,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_experiment_record(path, nested_summary, selected_row, history):
    comparison_rows = [
        "| 모델 | Nested OOF F1 | 정밀도 | 재현율 | Outer fold 평균 F1 ± 표준편차 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in nested_summary.to_dict("records"):
        comparison_rows.append(
            f"| {row['model']} | {row['f1']:.4f} | {row['precision']:.4f} "
            f"| {row['recall']:.4f} | {row['mean_outer_fold_f1']:.4f} "
            f"± {row['std_outer_fold_f1']:.4f} |"
        )
    best_nested = nested_summary.iloc[0]
    path.write_text(
        f"""# 진동 분류 v3 — 모터 단위 3개 모델 비교

## 데이터와 평가 방법

- 개발 데이터: 11~40번 폴더 (30개 번호 그룹, 6,308 CSV)
- 같은 번호의 normal/abnormal 폴더는 관계가 불명확해 보수적으로 하나의 그룹으로 묶었습니다.
- 각 모델의 바깥 5-fold에서 모터 폴더를 통째로 보류하고, 안쪽 3-fold에서 특징 세트·설정·판정 기준을 선택했습니다.
- F1 양성 클래스: abnormal=1. 최종 후보 비교 기준은 motor-group nested OOF F1입니다.
- scikit-learn {sklearn.__version__}, pandas {pd.__version__}

## 로지스틱 회귀·랜덤포레스트·RBF SVM 비교

{chr(10).join(comparison_rows)}

최고 nested F1 모델 계열: **{best_nested['model']}** (F1={best_nested['f1']:.4f}).

## 최종 모델 설정

- 모델: {selected_row['model']}
- 특징 세트: `{selected_row['feature_set']}`
- 설정: `{selected_row['selected_params']}`
- 기준값: {selected_row['threshold']:.3f}
- 전체 11~40번 그룹 5-fold 선택용 평균 F1: {selected_row['cv_mean_f1']:.4f} (표준편차 {selected_row['cv_std_f1']:.4f})

## 중간 과정 그림

- `figures/motor_spectra_by_class.png`: 모터별 평균 스펙트럼과 클래스 평균·편차
- `figures/representative_waveform_fft.png`: 대표 파형 및 FFT
- `figures/confusion_matrix_*.png`: 모델별 nested OOF 혼동행렬
- `motor_group_metrics.csv`: 각 보류 모터의 모델별 F1

## 사후검증 방지

이전 실험에서 1~10번 테스트 점수를 확인해 개발 방향을 논의했으므로, 이 v3 실행에서는 해당 폴더를 읽지 않았습니다. 과거 점수는 `exploratory_test_1_10.csv`에 탐색 기록으로만 남겼고 최종 시험 점수라고 주장하지 않습니다. 새 독립 모터 데이터가 확보되면 최종 시험에 사용할 수 있습니다.
""",
        encoding="utf-8",
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "results" / "v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets, labels, metadata, motor_spectra, representatives = load_dataset_v3(
        args.data_root
    )
    groups = metadata["folder_id"].to_numpy()
    if set(np.unique(groups)) != DEVELOPMENT_IDS:
        raise ValueError("V3 expects only motor folder groups 11-40")

    _save_spectrum_figures(motor_spectra, representatives, output_dir)
    nested, nested_folds, motor_metrics, predictions = _nested_motor_cv(
        feature_sets, labels, metadata
    )
    candidate_comparison, full_search = _final_cv_comparison(
        feature_sets, labels, metadata
    )

    # Choose the model family by its independently nested motor-group F1.
    best_nested_model = nested.iloc[0]["model"]
    family_rows = []
    for model_name in MODEL_SPECS:
        family = candidate_comparison.loc[
            candidate_comparison["model"] == model_name
        ].sort_values(
            ["cv_mean_f1", "cv_std_f1", "cv_mean_recall"],
            ascending=[False, True, False],
        )
        best_candidate = family.iloc[0].to_dict()
        nested_row = nested.loc[nested["model"] == model_name].iloc[0].to_dict()
        summary_row = {
            **best_candidate,
            "nested_cv_f1": nested_row["f1"],
            "nested_cv_precision": nested_row["precision"],
            "nested_cv_recall": nested_row["recall"],
            "nested_cv_mean_outer_f1": nested_row["mean_outer_fold_f1"],
            "nested_cv_std_outer_f1": nested_row["std_outer_fold_f1"],
            "selected_as_final": model_name == best_nested_model,
        }
        family_rows.append(summary_row)
    model_comparison = pd.DataFrame(family_rows).sort_values(
        ["nested_cv_f1", "nested_cv_mean_outer_f1"], ascending=False
    )
    selected = model_comparison.loc[
        model_comparison["model"] == best_nested_model
    ].iloc[0].to_dict()

    candidate_comparison.to_csv(output_dir / "candidate_comparison.csv", index=False)
    model_comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    full_search.to_csv(output_dir / "cv_search.csv", index=False)
    nested.to_csv(output_dir / "nested_cv_summary.csv", index=False)
    nested_folds.to_csv(output_dir / "nested_cv_folds.csv", index=False)
    motor_metrics.to_csv(output_dir / "motor_group_metrics.csv", index=False)
    motor_metrics.to_csv(output_dir / "folder_metrics.csv", index=False)
    predictions.to_csv(output_dir / "nested_cv_predictions.csv", index=False)

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for row in nested.to_dict("records"):
        name = MODEL_SPECS[row["model"]]["output_name"]
        _save_confusion(
            row,
            figure_dir / f"confusion_matrix_{name}.png",
            title=f"{row['model']} motor-group nested CV",
        )
    _save_confusion(
        nested.loc[nested["model"] == best_nested_model].iloc[0].to_dict(),
        output_dir / "confusion_matrix.png",
        title=f"Selected: {best_nested_model} motor-group nested CV",
    )

    # Historical 1-10 results are retained as exploratory evidence only.
    history = pd.DataFrame(
        [
            {"source": "v2 logistic regression", "f1_1_10": 0.8698924731, "status": "exploratory_only_test_was_seen"},
            {"source": "v2 random forest", "f1_1_10": 0.9367728882, "status": "exploratory_only_test_was_seen"},
            {"source": "v3 initial logistic regression", "f1_1_10": 0.9154639175, "status": "exploratory_only_test_was_seen"},
            {"source": "v3 initial random forest", "f1_1_10": 0.9484742371, "status": "exploratory_only_test_was_seen"},
            {"source": "v3 initial RBF SVM", "f1_1_10": 0.9588905399, "status": "exploratory_only_test_was_seen"},
            {"source": "v3 validation-tuned logistic regression", "f1_1_10": 0.3007275667, "status": "exploratory_only_test_was_seen"},
        ]
    )
    history.to_csv(output_dir / "exploratory_test_1_10.csv", index=False)
    _write_experiment_record(output_dir / "experiment.md", nested, selected, history)

    # Replace earlier per-family summaries; none retains a current test score.
    for model_name, spec in MODEL_SPECS.items():
        row = model_comparison.loc[
            model_comparison["model"] == model_name
        ].iloc[0].to_dict()
        model_dir = output_dir / spec["output_name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(model_dir / "model_cv_comparison.csv", index=False)
        full_search.loc[
            (full_search["model"] == model_name)
            & (full_search["feature_set"] == row["feature_set"])
        ].to_csv(model_dir / "cv_search.csv", index=False)
        pd.DataFrame([{
            "experiment_id": f"{spec['output_name']}_{row['feature_set']}_v3",
            "split": "nested_group_cv_11_40",
            "f1": row["nested_cv_f1"],
            "precision": row["nested_cv_precision"],
            "recall": row["nested_cv_recall"],
            "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
        }]).to_csv(model_dir / "metrics.csv", index=False)
        (model_dir / "experiment.md").write_text(
            f"# {model_name} — 모터 단위 nested CV\n\n"
            f"- 특징 세트: `{row['feature_set']}`\n"
            f"- 설정: `{row['selected_params']}`\n"
            f"- nested OOF F1: {row['nested_cv_f1']:.4f}\n"
            f"- 정밀도: {row['nested_cv_precision']:.4f}; 재현율: {row['nested_cv_recall']:.4f}\n"
            f"- 전체 3개 모델 비교는 상위 `model_comparison.csv`를 확인하세요.\n"
            f"- 1~10번은 이번 실행에서 읽지 않았습니다.\n",
            encoding="utf-8",
        )
        _save_confusion(
            nested.loc[nested["model"] == model_name].iloc[0].to_dict(),
            model_dir / "confusion_matrix.png",
            title=f"{model_name} motor-group nested CV",
        )
        model_comparison.loc[
            model_comparison["model"] == model_name
        ].to_csv(output_dir / f"{spec['output_name']}_cv_summary.csv", index=False)

    print("Motor-group nested CV comparison:", flush=True)
    for row in model_comparison.to_dict("records"):
        print(
            f"  {row['model']}: F1={row['nested_cv_f1']:.4f}, "
            f"precision={row['nested_cv_precision']:.4f}, "
            f"recall={row['nested_cv_recall']:.4f}",
            flush=True,
        )
    print(
        f"Final model selected by nested F1: {selected['model']} / "
        f"{selected['feature_set']}; 1-10 folders were not read.",
        flush=True,
    )


if __name__ == "__main__":
    main()
