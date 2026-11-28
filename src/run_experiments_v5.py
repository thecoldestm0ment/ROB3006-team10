"""Compare Welch band counts on the same motor folds as v4."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import __version__ as scipy_version
from sklearn.ensemble import RandomForestClassifier

from .evaluate import evaluate_predictions
from .preprocess_v2 import DATA_ROOT
from .preprocess_v4 import (
    FEATURE_NAMES_A,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    extract_features_a,
    extract_features_b,
    welch_bins_per_band,
    welch_feature_names,
)
from .run_experiments_v4 import (
    PROJECT_ROOT,
    DEVELOPMENT_IDS,
    RF_PARAMS,
    THRESHOLD,
    list_development_files,
    reshuffled_folds,
    v3_outer_folds,
)

BAND_COUNTS = (16, 32, 64)
COMBINED_METHOD = "welch_32_plus_v3"
METHODS = {
    **{f"welch_{bands}": (bands, f"Welch {bands}대역") for bands in BAND_COUNTS},
    COMBINED_METHOD: (32, "Welch 32대역 + 기존 v3 특징 16개"),
}


def load_features(root):
    files = list_development_files(root)
    groups, labels, baseline_rows = [], [], []
    rows = {bands: [] for bands in BAND_COUNTS}
    for count, (_, label, group, path) in enumerate(files, 1):
        values = pd.read_csv(path, header=None).to_numpy(dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError(f"Expected two finite numeric columns in {path}")
        time, signal = values.T
        for bands in BAND_COUNTS:
            features, _ = extract_features_b(time, signal, bands)
            rows[bands].append(features)
        baseline, _ = extract_features_a(time, signal)
        baseline_rows.append(baseline)
        groups.append(group)
        labels.append(label)
        if count % 1000 == 0:
            print(f"Loaded {count:,}/{len(files):,} CSVs", flush=True)

    group_labels = set(zip(groups, labels))
    if group_labels != {(group, label) for group in DEVELOPMENT_IDS for label in (0, 1)}:
        raise ValueError("Expected normal and abnormal recordings for every group 11-40")
    tables = {
        f"welch_{bands}": pd.DataFrame(records, columns=welch_feature_names(bands))
        for bands, records in rows.items()
    }
    tables[COMBINED_METHOD] = pd.concat(
        [tables["welch_32"], pd.DataFrame(baseline_rows, columns=FEATURE_NAMES_A)], axis=1
    )
    if not all(np.isfinite(table.to_numpy()).all() for table in tables.values()):
        raise ValueError("Welch extraction produced non-finite features")
    return tables, np.asarray(labels), np.asarray(groups)


def evaluate(features, labels, groups, splits):
    fold_rows, split_rows = [], []
    for method, table in features.items():
        bands, _ = METHODS[method]
        for split_name, folds in splits.items():
            oof = np.full(len(labels), -1)
            training_scores = []
            for fold, held_out in enumerate(folds, 1):
                valid = np.isin(groups, held_out)
                train = ~valid
                model = RandomForestClassifier(**RF_PARAMS)
                model.fit(table.loc[train], labels[train])
                train_pred = model.predict_proba(table.loc[train])[:, 1] >= THRESHOLD
                oof[valid] = model.predict_proba(table.loc[valid])[:, 1] >= THRESHOLD
                train_f1 = evaluate_predictions(labels[train], train_pred)["f1"]
                training_scores.append(train_f1)
                fold_rows.append({
                    "method": method, "bands": bands, "n_features": table.shape[1],
                    "split": split_name, "fold": fold,
                    "validation_groups": ",".join(map(str, held_out)),
                    "train_f1": train_f1,
                    **evaluate_predictions(labels[valid], oof[valid]),
                })
            if np.any(oof < 0):
                raise ValueError(f"Incomplete validation predictions for {split_name}")
            metrics = evaluate_predictions(labels, oof)
            mean_train = float(np.mean(training_scores))
            split_rows.append({
                "method": method, "bands": bands, "n_features": table.shape[1],
                "split": split_name, "mean_train_f1": mean_train,
                "train_oof_gap": mean_train - metrics["f1"], **metrics,
            })
            print(f"{method} / {split_name}: OOF F1={metrics['f1']:.4f}", flush=True)
    return pd.DataFrame(fold_rows), pd.DataFrame(split_rows)


def write_report(output, summary, splits, sample_count):
    split_names = list(splits)
    header = "| 특징 조합 | " + " | ".join(split_names) + " | 평균 OOF F1 | 평균 학습 F1 | 평균 격차 |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(split_names) + 3)) + " |"
    lines = [header, separator]
    for row in summary.to_dict("records"):
        values = [f"{row[name]:.4f}" for name in split_names]
        values.extend(f"{row[name]:.4f}" for name in ("mean_oof_f1", "mean_train_f1", "mean_train_oof_gap"))
        lines.append(f"| {METHODS[row['method']][1]} | " + " | ".join(values) + " |")

    best = summary.loc[summary["mean_oof_f1"].idxmax()]
    scores = summary.set_index("method")["mean_oof_f1"]
    combined_delta = scores.loc[COMBINED_METHOD] - scores.loc["welch_32"]
    (output / "experiment.md").write_text(
        f"""# v5 Welch 대역 수와 특징 조합 비교

11~40번 모터 번호 그룹의 CSV {sample_count:,}개를 사용했다. 같은 번호의 정상·불량 폴더는 같은 그룹으로 묶었다. v4에서 사용한 모터 단위 5-fold 분할 세 가지와 랜덤포레스트 설정을 그대로 사용했다. 불량 판정 기준은 {THRESHOLD}이다. 1~10번 시험 폴더는 이번 실행에서 읽지 않았다.

Welch 설정은 Hann 창, 시간 구간 {WELCH_NPERSEG}샘플, 겹침 {WELCH_NOVERLAP}샘플로 고정했다. 16·32·64는 주파수 대역 수이며, 양의 주파수 빈 256개를 대역마다 각각 16·8·4개씩 담는다. v4와 같은 방식으로 대역 상대 전력을 계산하고 로그를 취했다.

복합 조합은 Welch 32대역 특징에 기존 v3의 진폭 불변 특징 16개(FFT 10개, 파형 모양 6개)를 이어 붙여 총 48개를 사용한다.

{chr(10).join(lines)}

비교한 설정 중 평균 OOF F1이 가장 높은 것은 **{METHODS[best['method']][1]}({best['mean_oof_f1']:.4f})**이었다. 기존 특징을 더한 경우 Welch 32대역 단독 대비 평균 OOF F1 변화는 **{combined_delta:+.4f}**였다.

OOF F1은 분할마다 보류한 모터의 CSV 예측을 합쳐 계산했다. 표의 격차는 학습 F1에서 OOF F1을 뺀 값이다. 격차가 크면 학습 폴더에서 얻은 점수가 다른 폴더에 그대로 이어지지 않았다는 의미이며, 과적합의 원인을 단독으로 증명하지는 않는다. 세 분할은 동일한 30개 번호를 재사용한 개발 검증이다.

실행 명령

```bash
python -m src.run_experiments_v5 --data-root "데이터 폴더 경로"
```

scikit-learn {sklearn.__version__}, SciPy {scipy_version}
""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "v5_welch_comparison")
    args = parser.parse_args()
    splits = {"v3_outer_folds": v3_outer_folds()}
    splits.update({f"seed_{seed}": reshuffled_folds(seed) for seed in (42, 73)})
    features, labels, groups = load_features(args.data_root)
    folds, metrics = evaluate(features, labels, groups, splits)
    summary = metrics.pivot(index="method", columns="split", values="f1").loc[list(METHODS), list(splits)]
    summary["mean_oof_f1"] = summary.mean(axis=1)
    summary["mean_train_f1"] = metrics.groupby("method")["mean_train_f1"].mean()
    summary["mean_train_oof_gap"] = metrics.groupby("method")["train_oof_gap"].mean()
    summary.insert(0, "bands", [METHODS[method][0] for method in summary.index])
    summary.insert(1, "n_features", [features[method].shape[1] for method in summary.index])
    summary["psd_bins_per_band"] = summary["bands"].map(lambda bands: welch_bins_per_band(bands)[0])
    summary = summary.reset_index()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    metrics.to_csv(output / "split_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    write_report(output, summary, splits, len(labels))


if __name__ == "__main__":
    main()
