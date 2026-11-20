"""Week-3 feature comparison with a fixed Random Forest (motors 11–40 only).

Compares A (v3 amplitude-invariant 16), B (Welch 16 bands) and C (sym8 wavelet
packet 32 nodes) on three motor-grouped 5-fold split sets: the v3 outer folds,
and two reshuffled motor splits (seeds 42, 73). Every method sees exactly the
same train/held-out motors within a split set. Folders 1–10 are never read.

    python -m src.run_experiments_v4 --data-root "<folder with normal/ and abnormal/>"
"""

from __future__ import annotations

import argparse
import json
import platform
import time as timer
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier

from .evaluate import evaluate_predictions
from .preprocess_v2 import DATA_ROOT, LABELS
from .preprocess_v4 import (
    FEATURE_NAMES_A, FEATURE_NAMES_B, FEATURE_NAMES_C, WELCH_BANDS, WELCH_LOG_FLOOR,
    WELCH_NOVERLAP, WELCH_NPERSEG, WPT_LEVEL, WPT_MODE, WPT_WAVELET,
    extract_features_a, extract_features_b, extract_features_c,
    sampling_rate_from_time, welch_band_edges_normalized, welch_bins_per_band, wpt_node_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_IDS = tuple(range(11, 41))
V3_FOLDS_CSV = PROJECT_ROOT / "results" / "v3" / "nested_cv_folds.csv"
RESHUFFLE_SEEDS = (42, 73)
N_FOLDS = 5
THRESHOLD = 0.5
RF_PARAMS = {  # fixed for all methods; random_state/n_jobs as in v3 _make_model
    "n_estimators": 200, "max_depth": None, "min_samples_leaf": 2,
    "random_state": 42, "n_jobs": -1,
}
METHODS = {
    "A_v3_amp_invariant_16": FEATURE_NAMES_A,
    "B_welch_16band": FEATURE_NAMES_B,
    "C_wpt_sym8_32node": FEATURE_NAMES_C,
}


# ---------------------------------------------------------------- data + features
def list_development_files(root: Path) -> list[tuple[str, int, int, Path]]:
    files = []
    for class_name, label in LABELS.items():
        for motor in DEVELOPMENT_IDS:
            if not 11 <= motor <= 40:
                raise ValueError("development loader must exclude test motors 1–10")
            files.extend(
                (class_name, label, motor, path)
                for path in sorted((root / class_name / str(motor) / "time_vib").glob("*.csv"))
            )
    if not files:
        raise FileNotFoundError(f"No development CSV files under {root}")
    return files


def build_feature_table(root: Path):
    """Read each development CSV once and compute A, B, C features."""
    files = list_development_files(root)
    rows, seconds = [], {"read_csv": 0.0, "A": 0.0, "B": 0.0, "C": 0.0}
    flags = {"welch_zero_total_power": 0, "welch_floored_bands": 0, "wpt_zero_signal": 0}
    lengths, rates = [], []
    for count, (class_name, label, motor, path) in enumerate(files, start=1):
        t0 = timer.perf_counter()
        frame = pd.read_csv(path, header=None)
        if frame.shape[1] != 2:
            raise ValueError(f"Expected 2 columns in {path}")
        time_col = frame.iloc[:, 0].to_numpy(dtype=np.float64)
        signal = frame.iloc[:, 1].to_numpy(dtype=np.float64)
        t1 = timer.perf_counter()
        feat_a = extract_features_a(time_col, signal)
        t2 = timer.perf_counter()
        feat_b, flag_b = extract_features_b(time_col, signal)
        t3 = timer.perf_counter()
        feat_c, flag_c = extract_features_c(signal)
        t4 = timer.perf_counter()
        seconds["read_csv"] += t1 - t0
        seconds["A"] += t2 - t1
        seconds["B"] += t3 - t2
        seconds["C"] += t4 - t3
        flags["welch_zero_total_power"] += flag_b["zero_total_power"]
        flags["welch_floored_bands"] += flag_b["floored_bands"]
        flags["wpt_zero_signal"] += flag_c["zero_signal"]
        lengths.append(signal.size)
        rates.append(sampling_rate_from_time(time_col))
        rows.append({
            "source": path.relative_to(root).as_posix(), "class_name": class_name,
            "motor": motor, "label": label, **feat_a, **feat_b, **feat_c,
        })
        if count % 1000 == 0:
            print(f"features: {count:,}/{len(files):,} CSVs", flush=True)
    table = pd.DataFrame(rows)
    feature_cols = [c for names in METHODS.values() for c in names]
    if not np.isfinite(table[feature_cols].to_numpy()).all():
        raise ValueError("non-finite feature values")
    info = {
        "n_csv": len(table), "seconds": seconds, "flags": flags,
        "length_min": int(min(lengths)), "length_max": int(max(lengths)),
        "fs_min": float(min(rates)), "fs_median": float(np.median(rates)), "fs_max": float(max(rates)),
    }
    return table, info


# ---------------------------------------------------------------- split sets
def v3_outer_folds() -> list[list[int]]:
    folds = pd.read_csv(V3_FOLDS_CSV)
    rf = folds.loc[folds["model"] == "Random Forest"].sort_values("outer_fold")
    per_model = {
        model: tuple(group.sort_values("outer_fold")["held_out_folder_ids"])
        for model, group in folds.groupby("model")
    }
    if len(set(per_model.values())) != 1:
        raise ValueError("v3 outer folds differ between models")
    result = [sorted(int(x) for x in ids.split(",")) for ids in rf["held_out_folder_ids"]]
    _check_partition(result)
    return result


def reshuffled_folds(seed: int) -> list[list[int]]:
    order = np.random.default_rng(seed).permutation(np.array(DEVELOPMENT_IDS))
    result = [sorted(int(x) for x in part) for part in np.array_split(order, N_FOLDS)]
    _check_partition(result)
    return result


def _check_partition(folds: list[list[int]]) -> None:
    flat = [m for fold in folds for m in fold]
    if len(folds) != N_FOLDS or sorted(flat) != list(DEVELOPMENT_IDS):
        raise ValueError(f"folds must partition motors 11–40 into {N_FOLDS}: {folds}")


# ---------------------------------------------------------------- evaluation
def run_split(table: pd.DataFrame, split_name: str, folds: list[list[int]]):
    y = table["label"].to_numpy()
    motors = table["motor"].to_numpy()
    oof_rows, fold_rows, pred_frames, fit_seconds = [], [], [], {}
    for method, columns in METHODS.items():
        x = table.loc[:, list(columns)].to_numpy()
        proba = np.full(len(table), np.nan)
        start = timer.perf_counter()
        for fold_no, held in enumerate(folds, start=1):
            hold = np.isin(motors, held)
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(x[~hold], y[~hold])
            proba[hold] = model.predict_proba(x[hold])[:, 1]
            fold_pred = (proba[hold] >= THRESHOLD).astype(np.int8)
            fold_rows.append({"split_set": split_name, "method": method, "fold": fold_no,
                              "held_out_motors": ",".join(map(str, held)),
                              **evaluate_predictions(y[hold], fold_pred)})
        fit_seconds[method] = timer.perf_counter() - start
        if np.isnan(proba).any():
            raise RuntimeError(f"missing OOF predictions for {method}/{split_name}")
        pred = (proba >= THRESHOLD).astype(np.int8)
        fold_f1 = [r["f1"] for r in fold_rows if r["method"] == method and r["split_set"] == split_name]
        oof_rows.append({"split_set": split_name, "method": method, **evaluate_predictions(y, pred),
                         "mean_fold_f1": float(np.mean(fold_f1)), "std_fold_f1": float(np.std(fold_f1)),
                         "train_predict_seconds": fit_seconds[method]})
        pred_frames.append(pd.DataFrame({
            "split_set": split_name, "method": method, "source": table["source"],
            "motor": motors, "y_true": y, "proba_abnormal": proba, "y_pred": pred,
        }))
        print(f"{split_name} / {method}: OOF F1={oof_rows[-1]['f1']:.4f} "
              f"({fit_seconds[method]:.1f}s)", flush=True)
    return oof_rows, fold_rows, pred_frames


def motor_error_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, motor), g in pred.groupby(["method", "motor"]):
        rows.append({
            "method": method, "motor": int(motor),
            "n_normal": int((g["y_true"] == 0).sum()), "n_abnormal": int((g["y_true"] == 1).sum()),
            "fp": int(((g["y_true"] == 0) & (g["y_pred"] == 1)).sum()),
            "fn": int(((g["y_true"] == 1) & (g["y_pred"] == 0)).sum()),
        })
    long = pd.DataFrame(rows)
    wide = long.pivot(index="motor", columns="method", values=["fp", "fn"])
    wide.columns = [f"{metric}_{method.split('_')[0]}" for metric, method in wide.columns]
    counts = long.drop_duplicates("motor").set_index("motor")[["n_normal", "n_abnormal"]]
    out = counts.join(wide).reset_index()
    for letter in ("B", "C"):
        out[f"fn_change_{letter}_vs_A"] = out[f"fn_{letter}"] - out["fn_A"]
        out[f"fp_change_{letter}_vs_A"] = out[f"fp_{letter}"] - out["fp_A"]
    return out.sort_values(["fn_A", "motor"], ascending=[False, True])


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "v4_features")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    total_start = timer.perf_counter()
    table, info = build_feature_table(root)
    table.to_csv(out / "feature_cache.csv.gz", index=False)

    split_sets = {"v3_outer_folds": v3_outer_folds()}
    for seed in RESHUFFLE_SEEDS:
        split_sets[f"motor_shuffle_seed{seed}"] = reshuffled_folds(seed)
    pd.DataFrame([
        {"split_set": name, "fold": i, "held_out_motors": ",".join(map(str, held))}
        for name, folds in split_sets.items() for i, held in enumerate(folds, start=1)
    ]).to_csv(out / "split_definitions.csv", index=False)

    oof_rows, fold_rows, pred_frames = [], [], []
    for name, folds in split_sets.items():
        o, f, p = run_split(table, name, folds)
        oof_rows += o
        fold_rows += f
        pred_frames += p
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(out / "oof_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_csv(out / "oof_predictions.csv.gz", index=False)

    f1 = oof.pivot(index="method", columns="split_set", values="f1")[list(split_sets)]
    summary = f1.copy()
    summary["mean_f1"] = f1.mean(axis=1)
    summary["min_f1"] = f1.min(axis=1)
    summary["mean_delta_vs_A"] = summary["mean_f1"] - summary.loc["A_v3_amp_invariant_16", "mean_f1"]
    for split in split_sets:
        summary[f"rank_{split}"] = f1[split].rank(ascending=False, method="min").astype(int)
    summary.sort_values("mean_f1", ascending=False).to_csv(out / "summary.csv")

    motor_errors = motor_error_table(predictions.loc[predictions["split_set"] == "v3_outer_folds"])
    motor_errors.to_csv(out / "motor_errors_v3_outer_folds.csv", index=False)

    timing = pd.DataFrame([
        {"method": m,
         "feature_seconds_6308_csv": info["seconds"][m[0]],
         "train_predict_seconds_all_splits": float(oof.loc[oof["method"] == m, "train_predict_seconds"].sum())}
        for m in METHODS
    ])
    timing.to_csv(out / "timing.csv", index=False)

    run_info = {
        "data_root": str(root), "motors": [DEVELOPMENT_IDS[0], DEVELOPMENT_IDS[-1]],
        "n_csv": info["n_csv"], "csv_length_min": info["length_min"], "csv_length_max": info["length_max"],
        "fs_from_time_column_hz": {"min": info["fs_min"], "median": info["fs_median"], "max": info["fs_max"]},
        "csv_read_seconds": info["seconds"]["read_csv"], "flags": info["flags"],
        "welch": {"window": "hann", "nperseg": WELCH_NPERSEG, "noverlap": WELCH_NOVERLAP, "detrend": False,
                  "bands": WELCH_BANDS, "edges_fraction_of_nyquist": welch_band_edges_normalized().tolist(),
                  "bins_per_band_1024": welch_bins_per_band().tolist(), "log_floor": WELCH_LOG_FLOOR},
        "wpt": {"wavelet": WPT_WAVELET, "mode": WPT_MODE, "level": WPT_LEVEL, "node_paths_freq_order": wpt_node_paths()},
        "rf_params": RF_PARAMS, "threshold": THRESHOLD, "split_sets": split_sets,
        "total_seconds": timer.perf_counter() - total_start,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "pywavelets": pywt.__version__},
    }
    (out / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)
    print(f"done in {run_info['total_seconds']:.1f}s; folders 1-10 were not read.", flush=True)


if __name__ == "__main__":
    main()
