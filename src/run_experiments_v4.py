"""Week-3 feature comparison with a fixed Random Forest (motors 11–40 only).

Methods: A (v3 amplitude-invariant 16), Welch 16/32 bands, sym8 wavelet packet
16/32 nodes. All use the same Random Forest and, within a split set, exactly
the same train/held-out motors. The Welch/WPT x 16/32 grid separates the
method effect (same band count) from the band-count effect (same method).
Split sets: the v3 outer folds and two reshuffled motor splits (seeds 42, 73).
Folders 1–10 are never read.

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
    FEATURE_NAMES_A, WELCH_LOG_FLOOR, WELCH_NOVERLAP, WELCH_NPERSEG, WPT_MODE, WPT_WAVELET,
    extract_features_a, extract_features_b, extract_features_c, sampling_rate_from_time,
    welch_bins_per_band, welch_feature_names, wpt_feature_names, wpt_node_paths,
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
# key -> (short label, feature columns, extractor(time, signal) -> (features, flags))
METHODS = {
    "A_v3_amp_invariant_16": ("A", FEATURE_NAMES_A, extract_features_a),
    "W16_welch_16band": ("W16", welch_feature_names(16), lambda t, s: extract_features_b(t, s, 16)),
    "W32_welch_32band": ("W32", welch_feature_names(32), lambda t, s: extract_features_b(t, s, 32)),
    "P16_wpt_sym8_l4_16node": ("P16", wpt_feature_names(4), lambda t, s: extract_features_c(s, 4)),
    "P32_wpt_sym8_l5_32node": ("P32", wpt_feature_names(5), lambda t, s: extract_features_c(s, 5)),
}
GRID = {("Welch", 16): "W16_welch_16band", ("Welch", 32): "W32_welch_32band",
        ("WPT", 16): "P16_wpt_sym8_l4_16node", ("WPT", 32): "P32_wpt_sym8_l5_32node"}
BASELINE = "A_v3_amp_invariant_16"


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
    """Read each development CSV once and compute every method's features."""
    files = list_development_files(root)
    rows = []
    seconds = {"read_csv": 0.0, **{key: 0.0 for key in METHODS}}
    flags: dict[str, int] = {}
    lengths, rates = [], []
    for count, (class_name, label, motor, path) in enumerate(files, start=1):
        t0 = timer.perf_counter()
        frame = pd.read_csv(path, header=None)
        if frame.shape[1] != 2:
            raise ValueError(f"Expected 2 columns in {path}")
        time_col = frame.iloc[:, 0].to_numpy(dtype=np.float64)
        signal = frame.iloc[:, 1].to_numpy(dtype=np.float64)
        seconds["read_csv"] += timer.perf_counter() - t0
        row = {"source": path.relative_to(root).as_posix(), "class_name": class_name,
               "motor": motor, "label": label}
        for key, (short, _, extractor) in METHODS.items():
            t1 = timer.perf_counter()
            features, method_flags = extractor(time_col, signal)
            seconds[key] += timer.perf_counter() - t1
            row.update(features)
            for name, value in method_flags.items():
                flags[f"{short}_{name}"] = flags.get(f"{short}_{name}", 0) + value
        lengths.append(signal.size)
        rates.append(sampling_rate_from_time(time_col))
        rows.append(row)
        if count % 1000 == 0:
            print(f"features: {count:,}/{len(files):,} CSVs", flush=True)
    table = pd.DataFrame(rows)
    feature_cols = [c for _, cols, _ in METHODS.values() for c in cols]
    if not np.isfinite(table[feature_cols].to_numpy()).all():
        raise ValueError("non-finite feature values")
    info = {
        "n_csv": len(table), "seconds": seconds, "flags": flags,
        "length_min": int(min(lengths)), "length_max": int(max(lengths)),
        "fs_min": float(min(rates)), "fs_median": float(np.median(rates)), "fs_max": float(max(rates)),
        "fs_counts": pd.Series(np.round(rates, 3)).value_counts().sort_index().to_dict(),
    }
    return table, info


# ---------------------------------------------------------------- split sets
def v3_outer_folds() -> list[list[int]]:
    folds = pd.read_csv(V3_FOLDS_CSV)
    per_model = {
        model: tuple(group.sort_values("outer_fold")["held_out_folder_ids"])
        for model, group in folds.groupby("model")
    }
    if len(set(per_model.values())) != 1:
        raise ValueError("v3 outer folds differ between models")
    rf = folds.loc[folds["model"] == "Random Forest"].sort_values("outer_fold")
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
    oof_rows, fold_rows, pred_frames = [], [], []
    for key, (_, columns, _) in METHODS.items():
        x = table.loc[:, list(columns)].to_numpy()
        proba = np.full(len(table), np.nan)
        method_folds = []
        start = timer.perf_counter()
        for fold_no, held in enumerate(folds, start=1):
            hold = np.isin(motors, held)
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(x[~hold], y[~hold])
            proba[hold] = model.predict_proba(x[hold])[:, 1]
            metrics = evaluate_predictions(y[hold], (proba[hold] >= THRESHOLD).astype(np.int8))
            method_folds.append(metrics["f1"])
            fold_rows.append({"split_set": split_name, "method": key, "fold": fold_no,
                              "held_out_motors": ",".join(map(str, held)), **metrics})
        elapsed = timer.perf_counter() - start
        if np.isnan(proba).any():
            raise RuntimeError(f"missing OOF predictions for {key}/{split_name}")
        pred = (proba >= THRESHOLD).astype(np.int8)
        oof_rows.append({"split_set": split_name, "method": key, **evaluate_predictions(y, pred),
                         "mean_fold_f1": float(np.mean(method_folds)),
                         "std_fold_f1": float(np.std(method_folds)),
                         "train_predict_seconds": elapsed})
        pred_frames.append(pd.DataFrame({
            "split_set": split_name, "method": key, "source": table["source"],
            "motor": motors, "y_true": y, "proba_abnormal": proba, "y_pred": pred,
        }))
        print(f"{split_name} / {key}: OOF F1={oof_rows[-1]['f1']:.4f} ({elapsed:.1f}s)", flush=True)
    return oof_rows, fold_rows, pred_frames


def grid_table(f1: pd.DataFrame) -> pd.DataFrame:
    """Rows: Welch, WPT, WPT−Welch. Columns: 16, 32, 32−16 per split set and mean."""
    rows = []
    for split in f1.columns:
        cell = {(m, b): float(f1.loc[GRID[(m, b)], split]) for (m, b) in GRID}
        for method in ("Welch", "WPT"):
            rows.append({"split_set": split, "row": method, "bands_16": cell[(method, 16)],
                         "bands_32": cell[(method, 32)],
                         "effect_32_minus_16": cell[(method, 32)] - cell[(method, 16)]})
        rows.append({"split_set": split, "row": "WPT_minus_Welch",
                     "bands_16": cell[("WPT", 16)] - cell[("Welch", 16)],
                     "bands_32": cell[("WPT", 32)] - cell[("Welch", 32)],
                     "effect_32_minus_16": np.nan})
    return pd.DataFrame(rows)


def motor_error_table(pred: pd.DataFrame) -> pd.DataFrame:
    shorts = {key: short for key, (short, _, _) in METHODS.items()}
    rows = []
    for (method, motor), g in pred.groupby(["method", "motor"]):
        rows.append({
            "short": shorts[method], "motor": int(motor),
            "n_normal": int((g["y_true"] == 0).sum()), "n_abnormal": int((g["y_true"] == 1).sum()),
            "fp": int(((g["y_true"] == 0) & (g["y_pred"] == 1)).sum()),
            "fn": int(((g["y_true"] == 1) & (g["y_pred"] == 0)).sum()),
        })
    long = pd.DataFrame(rows)
    order = [short for short, _, _ in METHODS.values()]
    wide = long.pivot(index="motor", columns="short", values=["fn", "fp"])
    wide.columns = [f"{metric}_{short}" for metric, short in wide.columns]
    wide = wide[[f"{m}_{s}" for m in ("fn", "fp") for s in order]]
    counts = long.drop_duplicates("motor").set_index("motor")[["n_normal", "n_abnormal"]]
    out = counts.join(wide).reset_index()
    for short in order[1:]:
        out[f"fn_change_{short}_vs_A"] = out[f"fn_{short}"] - out["fn_A"]
        out[f"fp_change_{short}_vs_A"] = out[f"fp_{short}"] - out["fp_A"]
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

    f1 = oof.pivot(index="method", columns="split_set", values="f1").loc[list(METHODS), list(split_sets)]
    f1["mean"] = f1[list(split_sets)].mean(axis=1)
    summary = f1.rename(columns={"mean": "mean_f1"}).copy()
    summary["min_f1"] = f1[list(split_sets)].min(axis=1)
    summary["mean_delta_vs_A"] = summary["mean_f1"] - summary.loc[BASELINE, "mean_f1"]
    for split in split_sets:
        summary[f"rank_{split}"] = f1[split].rank(ascending=False, method="min").astype(int)
    summary.sort_values("mean_f1", ascending=False).to_csv(out / "summary.csv")
    grid_table(f1).to_csv(out / "band_method_grid.csv", index=False)

    motor_error_table(predictions.loc[predictions["split_set"] == "v3_outer_folds"]).to_csv(
        out / "motor_errors_v3_outer_folds.csv", index=False)

    pd.DataFrame([
        {"method": key, "n_features": len(cols),
         "feature_seconds_all_csv": info["seconds"][key],
         "train_predict_seconds_all_splits": float(oof.loc[oof["method"] == key, "train_predict_seconds"].sum())}
        for key, (_, cols, _) in METHODS.items()
    ]).to_csv(out / "timing.csv", index=False)

    run_info = {
        "data_root": str(root), "motors": [DEVELOPMENT_IDS[0], DEVELOPMENT_IDS[-1]],
        "n_csv": info["n_csv"], "csv_length_min": info["length_min"], "csv_length_max": info["length_max"],
        "fs_from_time_column_hz": {"min": info["fs_min"], "median": info["fs_median"], "max": info["fs_max"],
                                   "counts": {str(k): int(v) for k, v in info["fs_counts"].items()}},
        "csv_read_seconds": info["seconds"]["read_csv"], "flags": info["flags"],
        "welch": {"window": "hann", "nperseg": WELCH_NPERSEG, "noverlap": WELCH_NOVERLAP, "detrend": False,
                  "band_edges": "band j = ((j-1)/n, j/n] x Nyquist; DC excluded, Nyquist in last band",
                  "bins_per_band": {16: welch_bins_per_band(16), 32: welch_bins_per_band(32)},
                  "log_floor": WELCH_LOG_FLOOR},
        "wpt": {"wavelet": WPT_WAVELET, "mode": WPT_MODE,
                "node_paths_freq_order": {4: wpt_node_paths(4), 5: wpt_node_paths(5)}},
        "methods": {key: len(cols) for key, (_, cols, _) in METHODS.items()},
        "rf_params": RF_PARAMS, "threshold": THRESHOLD, "split_sets": split_sets,
        "total_seconds": timer.perf_counter() - total_start,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "pywavelets": pywt.__version__},
    }
    (out / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(), flush=True)
    print(f"done in {run_info['total_seconds']:.1f}s; folders 1-10 were not read.", flush=True)


if __name__ == "__main__":
    main()
