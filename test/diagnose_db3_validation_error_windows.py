"""Build a validation-only raw-EMG error-window table for DB3 development.

This independent diagnostic trains the current Key10 KinematicTCN on raw EMG
and records validation-window errors plus EMG-only summary features.  It never
predicts, scores, or ranks the held-out test repetitions.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np
import torch

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config, set_seed


DISTANCE_FEATURES = (
    "emg_mean",
    "emg_std",
    "temporal_delta",
    "near_zero_ratio",
    "envelope_modulation_power",
    "envelope_band_low_ratio",
    "envelope_band_mid_ratio",
    "envelope_band_high_ratio",
    "burst_fraction",
    "burst_max_duration_ms",
    "flat_fraction",
    "flat_max_duration_ms",
    "mean_abs_channel_corr",
    "max_abs_channel_corr",
)


def _load_exp3_helpers():
    path = PROJECT_ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("exp3_validation_window_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_subjects(value: str) -> list[int]:
    output: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = (int(item) for item in token.split("-", 1))
            output.extend(range(lo, hi + 1))
        else:
            output.append(int(token))
    return sorted(set(output))


def _maximum_run_duration(mask: np.ndarray, target_fs: float) -> np.ndarray:
    # Return the longest true run across channels for each window, in ms.
    n_windows, _, n_channels = mask.shape
    longest = np.zeros(n_windows, dtype=np.int32)
    for window_index in range(n_windows):
        for channel_index in range(n_channels):
            run = 0
            for value in mask[window_index, :, channel_index]:
                if value:
                    run += 1
                    longest[window_index] = max(longest[window_index], run)
                else:
                    run = 0
    return longest.astype(np.float64) * 1000.0 / target_fs


def _band_ratio(power: np.ndarray, frequencies: np.ndarray, lo: float, hi: float) -> np.ndarray:
    active = frequencies > 0.0
    band = (frequencies >= lo) & (frequencies < hi)
    numerator = power[:, band, :].sum(axis=(1, 2))
    denominator = power[:, active, :].sum(axis=(1, 2))
    return numerator / np.maximum(denominator, 1e-12)


def _channel_correlation_features(emg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_abs = np.zeros(len(emg), dtype=np.float64)
    max_abs = np.zeros(len(emg), dtype=np.float64)
    for index, window in enumerate(emg):
        centered = window - window.mean(axis=0, keepdims=True)
        norms = np.sqrt(np.sum(centered**2, axis=0))
        valid = norms > 1e-8
        if valid.sum() < 2:
            continue
        corr = (centered[:, valid].T @ centered[:, valid]) / np.outer(norms[valid], norms[valid])
        off_diagonal = np.abs(corr[np.triu_indices_from(corr, k=1)])
        mean_abs[index] = off_diagonal.mean()
        max_abs[index] = off_diagonal.max()
    return mean_abs, max_abs


def _fit_feature_reference(train_emg: np.ndarray, target_fs: float) -> dict[str, np.ndarray | float | list[str]]:
    deltas = np.abs(np.diff(train_emg, axis=1))
    reference: dict[str, np.ndarray | float | list[str]] = {
        "burst_threshold": np.quantile(train_emg, 0.90, axis=(0, 1)),
        "flat_delta_threshold": np.quantile(deltas, 0.10, axis=(0, 1)),
        "target_fs": float(target_fs),
    }
    train_features = _window_features(train_emg, reference)
    matrix = np.column_stack([train_features[name] for name in DISTANCE_FEATURES])
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0)
    reference["distance_feature_names"] = list(DISTANCE_FEATURES)
    reference["distance_center"] = center
    reference["distance_scale"] = np.maximum(1.4826 * mad, 1e-6)
    return reference


def _window_features(emg: np.ndarray, reference: dict[str, np.ndarray | float | list[str]]) -> dict[str, np.ndarray]:
    target_fs = float(reference["target_fs"])
    channel_mean = emg.mean(axis=1)
    channel_std = emg.std(axis=1)
    temporal_delta = np.abs(np.diff(emg, axis=1)).mean(axis=(1, 2))
    centered = emg - emg.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(emg.shape[1], d=1.0 / target_fs)
    burst_mask = emg >= np.asarray(reference["burst_threshold"])[None, None, :]
    flat_mask = np.abs(np.diff(emg, axis=1, prepend=emg[:, :1, :])) <= np.asarray(
        reference["flat_delta_threshold"]
    )[None, None, :]
    mean_abs_corr, max_abs_corr = _channel_correlation_features(emg)
    features = {
        "emg_mean": emg.mean(axis=(1, 2)),
        "emg_std": emg.std(axis=(1, 2)),
        "temporal_delta": temporal_delta,
        "near_zero_ratio": (emg <= 1e-6).mean(axis=(1, 2)),
        "envelope_modulation_power": np.mean(centered**2, axis=(1, 2)),
        "envelope_band_low_ratio": _band_ratio(power, frequencies, 0.8, 3.0),
        "envelope_band_mid_ratio": _band_ratio(power, frequencies, 3.0, 8.0),
        "envelope_band_high_ratio": _band_ratio(power, frequencies, 8.0, 20.0),
        "burst_fraction": burst_mask.mean(axis=(1, 2)),
        "burst_max_duration_ms": _maximum_run_duration(burst_mask, target_fs),
        "flat_fraction": flat_mask.mean(axis=(1, 2)),
        "flat_max_duration_ms": _maximum_run_duration(flat_mask, target_fs),
        "mean_abs_channel_corr": mean_abs_corr,
        "max_abs_channel_corr": max_abs_corr,
        "channel_mean_min": channel_mean.min(axis=1),
        "channel_mean_median": np.median(channel_mean, axis=1),
        "channel_std_min": channel_std.min(axis=1),
        "channel_std_median": np.median(channel_std, axis=1),
    }
    if "distance_center" in reference:
        matrix = np.column_stack([features[name] for name in DISTANCE_FEATURES])
        center = np.asarray(reference["distance_center"])
        scale = np.asarray(reference["distance_scale"])
        features["train_robust_distance"] = np.sqrt(np.mean(((matrix - center) / scale) ** 2, axis=1))
    else:
        features["train_robust_distance"] = np.zeros(len(emg), dtype=np.float64)
    return features


def _feature_associations(features: dict[str, np.ndarray], high_error: np.ndarray) -> list[dict[str, float | str]]:
    positive = np.flatnonzero(high_error)
    negative = np.flatnonzero(~high_error)
    associations: list[dict[str, float | str]] = []
    for name, values in features.items():
        comparisons = (values[positive, None] > values[negative]).mean()
        ties = (values[positive, None] == values[negative]).mean()
        auc = float(comparisons + 0.5 * ties)
        associations.append({
            "feature": name,
            "high_error_median": float(np.median(values[positive])),
            "remaining_median": float(np.median(values[negative])),
            "auc": auc,
            "separation_auc": float(max(auc, 1.0 - auc)),
            "high_error_direction": "higher" if auc >= 0.5 else "lower",
        })
    return sorted(associations, key=lambda item: float(item["separation_auc"]), reverse=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only DB3 raw-EMG high-error-window diagnostic")
    parser.add_argument("--run-dir", required=True, help="Existing outputs/run/<run_id>")
    parser.add_argument("--subjects", default="8,9", help="Development subjects; test repetitions remain unused")
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--high-error-quantile", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if not 0.5 < args.high_error_quantile < 1.0:
        raise ValueError("high-error-quantile must be between 0.5 and 1.0")

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    config = flatten_pipeline_config(load_yaml_config(PROJECT_ROOT))
    exp3 = _load_exp3_helpers()
    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    output_dir = run_dir / "06_diagnostics" / "db3_validation_error_windows"
    for name in ("metrics", "predictions", "checkpoints"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "validation-only raw-EMG KinematicTCN error-window diagnostic",
        "test_policy": "test repetitions 2/5 are neither predicted nor scored",
        "subjects": [],
    }

    for subject_id in _parse_subjects(args.subjects):
        print(f"\n[validation error windows] S{subject_id:02d}")
        emg, angle, _, repetitions = prepare_kinematics_data(
            data_loader, [subject_id], config, exercises=[args.exercise], db="db3"
        )
        train_idx, val_idx, test_idx = make_rep_split(
            repetitions, train_reps=(1, 3, 4, 6), test_reps=(2, 5),
            val_ratio=float(config["regressor_val_ratio"]), seed=int(config["regressor_random_seed"]) + subject_id,
        )
        set_seed(args.seed + subject_id)
        model, best_val, epochs = exp3.train_tcn_on_emg(emg, angle, train_idx, val_idx, config, config["device"])
        torch.save(model.state_dict(), output_dir / "checkpoints" / f"S{subject_id:02d}_raw_validation_best.pth")
        pred, target = exp3.predict_on_set(model, emg, angle, val_idx, config, config["device"])
        errors = np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))
        dynamic_scores = np.max(np.ptp(target, axis=1), axis=1)
        cutoff = float(np.quantile(errors, args.high_error_quantile))
        high_error = errors >= cutoff
        target_fs = float(config["target_fs"])
        feature_reference = _fit_feature_reference(emg[train_idx], target_fs)
        features = _window_features(emg[val_idx], feature_reference)
        feature_associations = _feature_associations(features, high_error)
        rows = []
        for local_index, source_index in enumerate(val_idx):
            row = {
                "subject_id": subject_id,
                "validation_local_index": int(local_index),
                "source_window_index": int(source_index),
                "repetition": int(repetitions[source_index]),
                "window_rmse": float(errors[local_index]),
                "dynamic_score": float(dynamic_scores[local_index]),
                "is_high_error": bool(high_error[local_index]),
            }
            row.update({name: float(values[local_index]) for name, values in features.items()})
            rows.append(row)
        rows.sort(key=lambda row: row["window_rmse"], reverse=True)
        _write_csv(output_dir / "metrics" / f"S{subject_id:02d}_validation_windows.csv", rows)
        np.savez(
            output_dir / "predictions" / f"S{subject_id:02d}_validation_windows.npz",
            emg=emg[val_idx], target=target, prediction=pred, errors=errors,
            dynamic_scores=dynamic_scores, validation_indices=val_idx, repetitions=repetitions[val_idx],
            high_error=high_error,
        )
        top_rows = rows[:args.top_k]
        print(f"  train={len(train_idx)} val={len(val_idx)} test_unused={len(test_idx)} "
              f"best_val={best_val:.5f} high_error={high_error.sum()}/{len(high_error)} cutoff={cutoff:.4f}")
        for row in top_rows[:5]:
            print(f"  val_idx={row['validation_local_index']:>3} rep={row['repetition']} "
                  f"rmse={row['window_rmse']:.4f} dyn={row['dynamic_score']:.4f} "
                  f"mean={row['emg_mean']:.4f} std={row['emg_std']:.4f}")
        report["subjects"].append({
            "subject_id": subject_id,
            "split": {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test_unused": int(len(test_idx))},
            "best_validation_loss": float(best_val),
            "epochs": int(epochs),
            "high_error_quantile": float(args.high_error_quantile),
            "high_error_cutoff": cutoff,
            "n_high_error": int(high_error.sum()),
            "validation_rmse": {
                "median": float(np.median(errors)), "p90": float(np.percentile(errors, 90)), "max": float(np.max(errors)),
            },
            "emg_feature_policy": {
                "source": "200 Hz rectified envelope windows",
                "spectral_features": "envelope modulation power ratios: 0.8-3, 3-8, and 8-20 Hz",
                "reference_split": "train windows only",
                "distance": "robust standardized Euclidean distance to train feature median",
            },
            "feature_reference": {
                "burst_threshold_per_channel": np.asarray(feature_reference["burst_threshold"]).tolist(),
                "flat_delta_threshold_per_channel": np.asarray(feature_reference["flat_delta_threshold"]).tolist(),
                "distance_feature_names": feature_reference["distance_feature_names"],
                "distance_center": np.asarray(feature_reference["distance_center"]).tolist(),
                "distance_scale": np.asarray(feature_reference["distance_scale"]).tolist(),
            },
            "feature_associations": feature_associations,
            "top_windows": top_rows,
            "csv": str(output_dir / "metrics" / f"S{subject_id:02d}_validation_windows.csv"),
            "npz": str(output_dir / "predictions" / f"S{subject_id:02d}_validation_windows.npz"),
        })
    with (output_dir / "metrics" / "validation_error_windows_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\n[diagnostic] saved: {output_dir}")


if __name__ == "__main__":
    main()
