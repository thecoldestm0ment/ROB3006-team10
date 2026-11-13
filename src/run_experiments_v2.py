"""Run version 2 features with folder-grouped tuning and per-folder diagnostics."""

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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import evaluate_predictions, save_results
from .preprocess_v2 import DATA_ROOT, FEATURE_NAMES_V2, load_dataset_v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_IDS = set(range(11, 35))
VALIDATION_IDS = set(range(35, 41))
FINAL_TRAIN_IDS = set(range(11, 41))
TEST_IDS = set(range(1, 11))
RANDOM_STATE = 42
CV_FOLDS = 4


def _make_model(model_name: str, params: dict):
    if model_name == "Logistic Regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        **params,
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    return RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)


def _candidate_params(model_name: str) -> list[dict]:
    if model_name == "Logistic Regression":
        return [{"C": c_value} for c_value in (0.01, 0.1, 1.0, 10.0, 100.0)]
    return [
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }
        for n_estimators in (200, 500)
        for max_depth in (None, 20)
        for min_samples_leaf in (1, 2)
    ]


def _metrics_for(model, x, y):
    return evaluate_predictions(y, model.predict(x))


def _group_cv_search(model_name, x_train, y_train, groups):
    splitter = GroupKFold(n_splits=CV_FOLDS)
    search_rows = []
    best_params = None
    best_key = None
    best_cv_summary = None
    best_oof_metrics = None

    for params in _candidate_params(model_name):
        fold_metrics = []
        oof_true = np.empty_like(y_train)
        oof_pred = np.empty_like(y_train)
        for fold, (fit_idx, validation_idx) in enumerate(
            splitter.split(x_train, y_train, groups=groups), start=1
        ):
            model = _make_model(model_name, params)
            model.fit(x_train.iloc[fit_idx], y_train[fit_idx])
            prediction = model.predict(x_train.iloc[validation_idx])
            metrics = evaluate_predictions(y_train[validation_idx], prediction)
            fold_metrics.append(metrics)
            oof_true[validation_idx] = y_train[validation_idx]
            oof_pred[validation_idx] = prediction

        mean_f1 = float(np.mean([metrics["f1"] for metrics in fold_metrics]))
        std_f1 = float(np.std([metrics["f1"] for metrics in fold_metrics], ddof=0))
        mean_precision = float(
            np.mean([metrics["precision"] for metrics in fold_metrics])
        )
        mean_recall = float(np.mean([metrics["recall"] for metrics in fold_metrics]))
        search_rows.append(
            {
                "parameters": json.dumps(params, sort_keys=True),
                "mean_fold_f1": mean_f1,
                "std_fold_f1": std_f1,
                "mean_fold_precision": mean_precision,
                "mean_fold_recall": mean_recall,
            }
        )
        print(
            f"  grouped CV {params}: mean F1={mean_f1:.4f} "
            f"(std={std_f1:.4f}), mean recall={mean_recall:.4f}"
        )
        # Prefer a higher mean F1; use lower fold variation, then recall, to break ties.
        key = (mean_f1, -std_f1, mean_recall)
        if best_key is None or key > best_key:
            best_key = key
            best_params = params
            best_cv_summary = {
                "mean_fold_f1": mean_f1,
                "std_fold_f1": std_f1,
                "mean_fold_precision": mean_precision,
                "mean_fold_recall": mean_recall,
            }
            best_oof_metrics = evaluate_predictions(oof_true, oof_pred)

    return best_params, best_cv_summary, best_oof_metrics, pd.DataFrame(search_rows)


def _folder_metrics(model_name, split_name, y_true, y_pred, metadata):
    rows = []
    folder_ids = metadata["folder_id"].to_numpy()
    for folder_id in sorted(np.unique(folder_ids)):
        mask = folder_ids == folder_id
        metrics = evaluate_predictions(y_true[mask], y_pred[mask])
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "folder_id": int(folder_id),
                "n_csv": int(mask.sum()),
                **metrics,
            }
        )
    return rows


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
        title=f"{model_name} — v2 held-out test",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_experiment_record(
    path,
    model_name,
    best_params,
    cv_summary,
    cv_oof_metrics,
    search_table,
    validation_metrics,
    test_metrics,
):
    korean_name = {
        "Logistic Regression": "로지스틱 회귀",
        "Random Forest": "랜덤 포레스트",
    }[model_name]
    feature_labels = (
        "평균(mean)",
        "표준편차(std)",
        "RMS",
        "최댓값(max)",
        "최솟값(min)",
        "진폭 범위(peak_to_peak)",
        "최대 파워 주파수(dominant_frequency_hz)",
        "최대 파워 비율(dominant_power_ratio)",
        "스펙트럼 중심(spectral_centroid_hz)",
        "스펙트럼 퍼짐(spectral_spread_hz)",
        "스펙트럼 엔트로피(spectral_entropy)",
        "85% 스펙트럼 롤오프(spectral_rolloff_85_hz)",
        "나이퀴스트 0~10% 대역 파워",
        "나이퀴스트 10~25% 대역 파워",
        "나이퀴스트 25~50% 대역 파워",
        "나이퀴스트 50~100% 대역 파워",
    )
    search_rows = [
        "| 설정 | 교차검증 평균 F1 | F1 표준편차 | 평균 정밀도 | 평균 재현율 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in search_table.to_dict("records"):
        search_rows.append(
            f"| `{row['parameters']}` | {row['mean_fold_f1']:.4f} "
            f"| {row['std_fold_f1']:.4f} | {row['mean_fold_precision']:.4f} "
            f"| {row['mean_fold_recall']:.4f} |"
        )
    text = f"""# {korean_name} 실험 기록 (v2)

## 실험 조건 및 특징

- 실험 ID: `{model_name.lower().replace(' ', '_')}_fft_v2`
- 샘플 단위: CSV 파일 1개
- 라벨: 정상=0, 불량=1 (양성 클래스)
- 특징 수: {len(FEATURE_NAMES_V2)}개 (시간 영역 6개 + 주파수 영역 10개)
- 주파수 특징: CSV의 시간 간격으로 FFT를 계산하고 DC 성분을 제외했습니다. 나이퀴스트 주파수 기준으로 4개 대역의 파워를 계산했습니다.
- 특징 목록: {', '.join(feature_labels)}
- 그룹 분리: 같은 번호 폴더의 CSV가 서로 다른 교차검증 fold에 나뉘지 않도록 했습니다.
- 난수 시드: {RANDOM_STATE}
- Python 라이브러리: scikit-learn {sklearn.__version__}, pandas {pd.__version__}

## 학습 데이터(11~34번) 그룹 교차검증

폴더 번호를 기준으로 4개 그룹 fold를 구성했습니다. fold 평균 F1이 가장 높은 설정을 선택하고, 동률이면 fold별 F1 편차가 낮은 설정을 우선했습니다.

{chr(10).join(search_rows)}

선택한 설정: `{json.dumps(best_params, sort_keys=True)}`

- fold 평균 F1: {cv_summary['mean_fold_f1']:.4f} (표준편차 {cv_summary['std_fold_f1']:.4f})
- 교차검증 전체 예측(OOF) F1: {cv_oof_metrics['f1']:.4f}

## 별도 검증 및 최종 테스트

- 검증: 11~34번 폴더로 학습하고 35~40번 폴더로 평가
- 최종 테스트: 11~40번 폴더로 재학습하고 1~10번 폴더로 한 번 평가
- 검증 F1: {validation_metrics['f1']:.4f}, 정밀도={validation_metrics['precision']:.4f}, 재현율={validation_metrics['recall']:.4f}
- 최종 테스트 F1: {test_metrics['f1']:.4f}, 정밀도={test_metrics['precision']:.4f}, 재현율={test_metrics['recall']:.4f}
- 최종 테스트 혼동행렬: TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TP={test_metrics['tp']}
- 폴더별 검증·테스트 점수: `folder_metrics.csv`
- 최종 테스트 혼동행렬 그림: `confusion_matrix.png`

전체 F1만으로 일반화 성능을 판단하지 말고 `folder_metrics.csv`의 폴더별 결과도 함께 확인해야 합니다. 교차검증과 35~40번 검증 결과는 최종 테스트 데이터와 분리했습니다. 최종 평가 단위와 F1 평균 방식은 수업 피드백에 따라 확인해야 합니다.
"""
    path.write_text(text, encoding="utf-8")


def _run_model(model_name, output_name, features, labels, metadata):
    folder_ids = metadata["folder_id"].to_numpy()
    train_mask = np.isin(folder_ids, list(TRAIN_IDS))
    validation_mask = np.isin(folder_ids, list(VALIDATION_IDS))
    final_train_mask = np.isin(folder_ids, list(FINAL_TRAIN_IDS))
    test_mask = np.isin(folder_ids, list(TEST_IDS))

    x_train, y_train = features.loc[train_mask].reset_index(drop=True), labels[train_mask]
    groups = folder_ids[train_mask]
    params, cv_summary, cv_oof_metrics, search_table = _group_cv_search(
        model_name, x_train, y_train, groups
    )

    validation_model = _make_model(model_name, params)
    validation_model.fit(x_train, y_train)
    x_validation = features.loc[validation_mask]
    y_validation = labels[validation_mask]
    meta_validation = metadata.loc[validation_mask].reset_index(drop=True)
    validation_prediction = validation_model.predict(x_validation)
    validation_metrics = evaluate_predictions(y_validation, validation_prediction)

    final_model = _make_model(model_name, params)
    x_final_train = features.loc[final_train_mask]
    y_final_train = labels[final_train_mask]
    x_test = features.loc[test_mask]
    y_test = labels[test_mask]
    meta_test = metadata.loc[test_mask].reset_index(drop=True)
    final_model.fit(x_final_train, y_final_train)
    test_prediction = final_model.predict(x_test)
    test_metrics = evaluate_predictions(y_test, test_prediction)

    output_dir = PROJECT_ROOT / "results" / "v2" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = f"{output_name}_fft_v2"
    save_results(cv_oof_metrics, output_dir, experiment_id, "group_cv_oof_11_34")
    save_results(validation_metrics, output_dir, experiment_id, "validation_35_40")
    save_results(test_metrics, output_dir, experiment_id, "final_test_1_10")
    search_table.to_csv(output_dir / "cv_search.csv", index=False)
    _plot_confusion_matrix(test_metrics, model_name, output_dir / "confusion_matrix.png")
    _write_experiment_record(
        output_dir / "experiment.md",
        model_name,
        params,
        cv_summary,
        cv_oof_metrics,
        search_table,
        validation_metrics,
        test_metrics,
    )

    folder_rows = _folder_metrics(
        model_name, "validation_35_40", y_validation, validation_prediction, meta_validation
    )
    folder_rows += _folder_metrics(
        model_name, "final_test_1_10", y_test, test_prediction, meta_test
    )
    print(
        f"Selected {model_name} settings: {params}\n"
        f"  group CV mean F1={cv_summary['mean_fold_f1']:.4f} "
        f"(std={cv_summary['std_fold_f1']:.4f})\n"
        f"  validation F1={validation_metrics['f1']:.4f}; "
        f"final test F1={test_metrics['f1']:.4f}"
    )
    summary = {
        "model": model_name,
        "experiment_id": experiment_id,
        "selected_params": json.dumps(params, sort_keys=True),
        "cv_mean_f1": cv_summary["mean_fold_f1"],
        "cv_std_f1": cv_summary["std_fold_f1"],
        "cv_oof_f1": cv_oof_metrics["f1"],
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
    return summary, folder_rows


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
    features, labels, metadata = load_dataset_v2(args.data_root)
    print(
        f"Split sizes: train={int(metadata['folder_id'].isin(TRAIN_IDS).sum()):,}, "
        f"validation={int(metadata['folder_id'].isin(VALIDATION_IDS).sum()):,}, "
        f"test={int(metadata['folder_id'].isin(TEST_IDS).sum()):,} CSVs"
    )

    summaries = []
    all_folder_rows = []
    for model_name, output_name in (
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
    ):
        summary, folder_rows = _run_model(
            model_name, output_name, features, labels, metadata
        )
        summaries.append(summary)
        all_folder_rows.extend(folder_rows)

    v2_summary = pd.DataFrame(summaries)
    baseline_path = PROJECT_ROOT / "results" / "v1" / "model_comparison.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path).set_index("model")
        v2_summary["v1_validation_f1"] = v2_summary["model"].map(
            baseline["validation_f1"]
        )
        v2_summary["v1_test_f1"] = v2_summary["model"].map(baseline["test_f1"])
        v2_summary["test_f1_delta_vs_v1"] = (
            v2_summary["test_f1"] - v2_summary["v1_test_f1"]
        )

    results_root = PROJECT_ROOT / "results" / "v2"
    results_root.mkdir(parents=True, exist_ok=True)
    v2_summary.to_csv(results_root / "model_comparison.csv", index=False)
    pd.DataFrame(all_folder_rows).to_csv(
        results_root / "folder_metrics.csv", index=False
    )
    print(f"v2 summary saved to {results_root / 'model_comparison.csv'}")
    print(f"Per-folder diagnostics saved to {results_root / 'folder_metrics.csv'}")


if __name__ == "__main__":
    main()
