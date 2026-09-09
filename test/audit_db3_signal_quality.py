"""Read-only audit of DB3 raw sEMG quality and downstream error association.

This diagnostic intentionally separates acquisition-failure evidence from weak
physiological activity.  It never invokes MCIA or trains a kinematic regressor.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config
from utils.rule_anomaly_detector import LABEL_NAMES, RuleAnomalyDetector


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= min_len]


def _event_summary(mask: np.ndarray, action_fraction: np.ndarray, patch_ms: float,
                   min_patches: int) -> tuple[dict, np.ndarray]:
    """Summarize channel-wise candidate events and return their patch coverage."""
    n_patches, n_channels = mask.shape
    coverage = np.zeros_like(mask, dtype=bool)
    rows = []
    for channel in range(n_channels):
        events = _runs(mask[:, channel], min_patches)
        for start, end in events:
            coverage[start:end, channel] = True
        lengths = np.asarray([end - start for start, end in events], dtype=float)
        weighted_action = sum(
            float(action_fraction[start:end].sum()) for start, end in events
        )
        total_patches = int(lengths.sum()) if len(lengths) else 0
        rows.append({
            "channel": int(channel + 1),
            "event_count": int(len(events)),
            "total_ms": float(total_patches * patch_ms),
            "max_ms": float(lengths.max() * patch_ms) if len(lengths) else 0.0,
            "median_ms": float(np.median(lengths) * patch_ms) if len(lengths) else 0.0,
            "action_overlap": float(weighted_action / total_patches) if total_patches else 0.0,
        })
    return {"per_channel": rows, "event_count": int(sum(r["event_count"] for r in rows))}, coverage


def _raw_subject(subject_id: int, exercises: list[int], db3_path: Path) -> tuple[np.ndarray, np.ndarray]:
    subject_dir = db3_path / f"s{subject_id}" / f"DB3_s{subject_id}"
    emg_list, label_list = [], []
    for exercise in exercises:
        path = subject_dir / f"S{subject_id}_E{exercise}_A1.mat"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = sio.loadmat(path)
        emg = np.asarray(payload["emg"], dtype=np.float32)
        label = np.asarray(payload.get("restimulus", payload.get("stimulus"))).reshape(-1)
        n = min(len(emg), len(label))
        emg_list.append(emg[:n])
        label_list.append(label[:n])
    return np.vstack(emg_list), np.concatenate(label_list)


def _quality_from_raw(raw: np.ndarray, labels: np.ndarray, fs: int, patch_ms: float,
                      clear_min_ms: float) -> dict:
    patch_size = max(1, int(round(fs * patch_ms / 1000.0)))
    usable = len(raw) // patch_size * patch_size
    raw, labels = raw[:usable], labels[:usable]
    patches = raw.reshape(-1, patch_size, raw.shape[1])
    label_patches = labels.reshape(-1, patch_size)
    action_fraction = (label_patches != 0).mean(axis=1)
    rms = np.sqrt(np.mean(patches**2, axis=1))
    mad = np.mean(np.abs(patches - patches.mean(axis=1, keepdims=True)), axis=1)
    ptp = np.ptp(patches, axis=1)
    global_ptp = np.ptp(raw, axis=0)
    flat_tol = np.maximum(global_ptp * 1e-3, np.finfo(np.float32).eps)

    # A physiologic rest signal remains noisy.  Only a sustained nearly constant
    # raw waveform is called a clear acquisition-failure *candidate*.
    flat_patch = ptp <= flat_tol[None, :]
    clear_min_patches = max(1, int(np.ceil(clear_min_ms / patch_ms)))
    clear_summary, clear_coverage = _event_summary(
        flat_patch, action_fraction, patch_ms, clear_min_patches
    )

    # Repeated record-wide extrema can indicate ADC clipping, but without amplifier
    # rail metadata this remains a saturation candidate, not confirmed failure.
    lo, hi = raw.min(axis=0), raw.max(axis=0)
    atol = np.maximum(global_ptp * 1e-6, np.finfo(np.float32).eps)
    at_extreme = (np.isclose(patches, lo[None, None, :], atol=atol[None, None, :]) |
                  np.isclose(patches, hi[None, None, :], atol=atol[None, None, :]))
    clip_patch = at_extreme.mean(axis=1) >= 0.95
    clip_summary, clip_coverage = _event_summary(clip_patch, action_fraction, patch_ms, 1)

    # Weak activity is descriptive. Thresholds are channel-specific movement P5,
    # which prevents cross-subject amplitude from being interpreted as failure.
    movement = action_fraction > 0.0
    rms_p5 = np.percentile(rms[movement], 5, axis=0) if movement.any() else np.zeros(raw.shape[1])
    mad_p5 = np.percentile(mad[movement], 5, axis=0) if movement.any() else np.zeros(raw.shape[1])
    weak_patch = (rms <= rms_p5[None, :]) & (mad <= mad_p5[None, :])
    weak_summary, weak_coverage = _event_summary(weak_patch, action_fraction, patch_ms, 1)

    per_channel = []
    for channel in range(raw.shape[1]):
        clear = clear_summary["per_channel"][channel]
        clip = clip_summary["per_channel"][channel]
        weak = weak_summary["per_channel"][channel]
        per_channel.append({
            "channel": channel + 1,
            "raw_range": float(global_ptp[channel]),
            "raw_zero_sample_ratio": float(np.mean(raw[:, channel] == 0.0)),
            "movement_rms_p5": float(rms_p5[channel]),
            "movement_mad_p5": float(mad_p5[channel]),
            "clear_flatline_candidate": clear,
            "possible_clipping_candidate": clip,
            "weak_activity": weak,
        })

    def _sample_mask(coverage: np.ndarray) -> np.ndarray:
        return np.repeat(coverage, patch_size, axis=0)[:len(raw)]

    return {
        "patch_ms": patch_ms,
        "patch_size_samples": patch_size,
        "clear_failure_definition": (
            f"raw within-patch peak-to-peak <= 0.1% of channel full range, "
            f"sustained >= {clear_min_ms:g} ms; candidate only"
        ),
        "weak_activity_definition": "per-subject/channel movement-patch RMS and MAD both <= P5; not a failure label",
        "per_channel": per_channel,
        "overall": {
            "n_raw_samples": int(len(raw)),
            "action_sample_ratio": float(np.mean(labels != 0)),
            "clear_failure_candidate_ratio": float(clear_coverage.mean()),
            "possible_clipping_candidate_ratio": float(clip_coverage.mean()),
            "weak_activity_ratio": float(weak_coverage.mean()),
            "clear_failure_action_overlap": float(
                (clear_coverage * (action_fraction[:, None] > 0)).sum() /
                max(1, clear_coverage.sum())
            ),
            "weak_activity_action_overlap": float(
                (weak_coverage * (action_fraction[:, None] > 0)).sum() /
                max(1, weak_coverage.sum())
            ),
        },
        "sample_masks": {
            "clear": _sample_mask(clear_coverage),
            "weak": _sample_mask(weak_coverage),
        },
    }


def _correlation(x: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 8 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return {"n": int(len(x)), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {"n": int(len(x)), "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp)}


def _window_coverage(sample_mask: np.ndarray, starts_down: np.ndarray, downsample: int,
                     window_down: int) -> np.ndarray:
    values = []
    raw_len = len(sample_mask)
    for start in starts_down:
        lo, hi = int(start) * downsample, (int(start) + window_down) * downsample
        values.append(float(sample_mask[max(0, lo):min(raw_len, hi)].mean()) if hi > lo else np.nan)
    return np.asarray(values, dtype=float)


def _downstream_association(subject_id: int, prediction_dir: Path, config: dict,
                            loader: NinaProDataLoader, quality: dict) -> dict:
    path = prediction_dir / f"S{subject_id:02d}_angle_predictions.npz"
    if not path.exists():
        return {"available": False, "reason": f"prediction missing: {path}"}
    artifact = np.load(path, allow_pickle=False)
    required = {"target", "pred_A", "test_window_starts"}
    missing = sorted(required - set(artifact.files))
    if missing:
        return {"available": False, "reason": f"prediction missing fields: {missing}"}

    emg, _, _, reps = prepare_kinematics_data(loader, [subject_id], config, exercises=[1], db="db3")
    train_idx, _, test_idx = make_rep_split(
        reps, val_ratio=float(config.get("regressor_val_ratio", 0.2)), seed=42
    )
    starts = np.asarray(artifact["test_window_starts"], dtype=np.int64)
    target = np.asarray(artifact["target"], dtype=float)
    if len(test_idx) != len(starts) or len(starts) != len(target):
        return {"available": False, "reason": "test window count does not match current preparation"}

    # Count matching alone would not prove a window-level quality/error association.
    source = loader.load_db3_subject(subject_id, exercises=[1])
    factor = int(config["orig_fs"] / config["target_fs"])
    labels = np.asarray(source["restimulus"])[::factor]
    repetitions = np.asarray(source["repetition"])[::factor]
    usable = min(len(labels), len(repetitions))
    candidate_starts = np.asarray([
        start for start in range(0, usable - int(config["window_size"]) + 1, int(config["stride"]))
        if np.any(labels[start:start + int(config["window_size"])] != 0)
    ], dtype=np.int64)
    if len(candidate_starts) != len(reps) or not np.array_equal(candidate_starts[test_idx], starts):
        return {"available": False, "reason": "test window time indices do not match reconstructed DB3 E1 windows"}

    detector = RuleAnomalyDetector(
        patch_size=int(config.get("patch_size", 8)),
        group_indices=config.get("group_indices"),
    ).fit(emg[train_idx])
    detected = detector.detect_batch(emg[test_idx])
    mask_ratio = 1.0 - detected["mask"].mean(axis=(1, 2))
    weak_coverage = _window_coverage(
        quality["sample_masks"]["weak"], starts, factor, int(config["window_size"])
    )
    clear_coverage = _window_coverage(
        quality["sample_masks"]["clear"], starts, factor, int(config["window_size"])
    )

    results = {"available": True, "prediction_path": str(path), "legacy_target_dim": int(target.shape[-1]),
               "test_windows": int(len(target)), "mapping_status": "time-indices-validated"}
    window_errors = {}
    for group in ("A", "B"):
        key = f"pred_{group}"
        if key not in artifact.files:
            continue
        error = np.sqrt(np.mean((np.asarray(artifact[key], dtype=float) - target) ** 2, axis=(1, 2)))
        window_errors[group] = error
        results[f"group_{group}_rmse"] = float(error.mean())
        results[f"group_{group}_association"] = {
            "rule_mask_ratio": _correlation(mask_ratio, error),
            "weak_activity_ratio": _correlation(weak_coverage, error),
            "clear_failure_ratio": _correlation(clear_coverage, error),
        }
    if "A" in window_errors:
        for group in ("B",):
            if group in window_errors:
                improvement = window_errors["A"] - window_errors[group]
                results[f"group_{group}_minus_A_association"] = {
                    "rule_mask_ratio": _correlation(mask_ratio, improvement),
                    "weak_activity_ratio": _correlation(weak_coverage, improvement),
                }
    results["rule_mask_ratio_mean"] = float(mask_ratio.mean())
    results["rule_dead_channels_1based"] = [int(c + 1) for c in detected["dead_channels"]]
    return results


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only DB3 raw sEMG quality audit")
    parser.add_argument("--run-dir", required=True, help="Existing outputs/run/<run_id> for diagnostic output")
    parser.add_argument("--prediction-run", required=True, help="Existing run containing DB3 prediction NPZ files")
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--exercises", default="1")
    parser.add_argument("--patch-ms", type=float, default=40.0)
    parser.add_argument("--clear-min-ms", type=float, default=200.0)
    args = parser.parse_args()

    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    subjects = [int(value) for value in args.subjects.split(",") if value.strip()]
    exercises = [int(value) for value in args.exercises.split(",") if value.strip()]
    run_dir = Path(args.run_dir).resolve()
    prediction_dir = Path(args.prediction_run).resolve() / "03_angle_prediction" / "predictions"
    out_dir = run_dir / "06_diagnostics" / "db3_signal_quality_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=int(config["orig_fs"]))

    report = {
        "scope": "read-only DB3 E1 raw sEMG quality audit; no MCIA inference or TCN training",
        "prediction_source": str(prediction_dir),
        "definitions": {
            "clear_failure": "sustained near-flat raw waveform candidate; not confirmed hardware failure without acquisition metadata",
            "weak_activity": "relative low activity only; explicitly not a failure or imputation label",
            "possible_clipping": "repeated record-wide extrema candidate; not confirmed ADC saturation without rail metadata",
        },
        "subjects": [],
    }
    subject_rows, channel_rows = [], []
    for subject_id in subjects:
        print(f"[audit] DB3 S{subject_id:02d}")
        raw, labels = _raw_subject(subject_id, exercises, Path(config["db3_path"]))
        quality = _quality_from_raw(raw, labels, int(config["orig_fs"]), args.patch_ms, args.clear_min_ms)
        relation = _downstream_association(subject_id, prediction_dir, config, loader, quality)
        report["subjects"].append({"subject_id": subject_id, "raw_quality": {k: v for k, v in quality.items() if k != "sample_masks"},
                                   "downstream_association": relation})
        overall = quality["overall"]
        subject_rows.append({"subject_id": subject_id, **overall,
                             "rule_mask_ratio_mean": relation.get("rule_mask_ratio_mean"),
                             "raw_A_window_rmse": relation.get("group_A_rmse"),
                             "association_available": relation["available"]})
        for row in quality["per_channel"]:
            channel_rows.append({"subject_id": subject_id, "channel": row["channel"],
                                 "raw_range": row["raw_range"], "raw_zero_sample_ratio": row["raw_zero_sample_ratio"],
                                 "movement_rms_p5": row["movement_rms_p5"], "movement_mad_p5": row["movement_mad_p5"],
                                 "clear_events": row["clear_flatline_candidate"]["event_count"],
                                 "clear_total_ms": row["clear_flatline_candidate"]["total_ms"],
                                 "clear_action_overlap": row["clear_flatline_candidate"]["action_overlap"],
                                 "clip_events": row["possible_clipping_candidate"]["event_count"],
                                 "weak_events": row["weak_activity"]["event_count"],
                                 "weak_total_ms": row["weak_activity"]["total_ms"],
                                 "weak_action_overlap": row["weak_activity"]["action_overlap"]})
        print(f"  clear={overall['clear_failure_candidate_ratio']:.4%} | weak={overall['weak_activity_ratio']:.2%} | "
              f"action overlap={overall['weak_activity_action_overlap']:.2%} | association={relation['available']}")

    def _finite(values):
        return np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)

    rule_rho = _finite([
        subject["downstream_association"].get("group_A_association", {})
        .get("rule_mask_ratio", {}).get("spearman_r")
        for subject in report["subjects"]
    ])
    weak_rho = _finite([
        subject["downstream_association"].get("group_A_association", {})
        .get("weak_activity_ratio", {}).get("spearman_r")
        for subject in report["subjects"]
    ])
    group_deltas = {}
    for group in ("B",):
        values = _finite([
            subject["downstream_association"].get(f"group_{group}_rmse", np.nan)
            - subject["downstream_association"].get("group_A_rmse", np.nan)
            for subject in report["subjects"]
        ])
        group_deltas[group] = {
            "n": int(len(values)),
            "mean_rmse_delta_vs_A": float(values.mean()) if len(values) else None,
            "median_rmse_delta_vs_A": float(np.median(values)) if len(values) else None,
            "improved_subjects": int((values < 0).sum()),
            "worse_subjects": int((values > 0).sum()),
        }
    report["aggregate"] = {
        "n_subjects": int(len(report["subjects"])),
        "subjects_with_clear_failure_candidates": [
            subject["subject_id"] for subject in report["subjects"]
            if subject["raw_quality"]["overall"]["clear_failure_candidate_ratio"] > 0
        ],
        "median_weak_activity_ratio": float(np.median([
            subject["raw_quality"]["overall"]["weak_activity_ratio"] for subject in report["subjects"]
        ])),
        "group_A_window_error_association": {
            "rule_mask_spearman_median": float(np.median(rule_rho)) if len(rule_rho) else None,
            "rule_mask_positive_subjects": int((rule_rho > 0).sum()),
            "rule_mask_negative_subjects": int((rule_rho < 0).sum()),
            "weak_activity_spearman_median": float(np.median(weak_rho)) if len(weak_rho) else None,
            "weak_activity_positive_subjects": int((weak_rho > 0).sum()),
            "weak_activity_negative_subjects": int((weak_rho < 0).sum()),
        },
        "legacy_prediction_group_rmse_delta_vs_A": group_deltas,
        "interpretation_limit": (
            "Window-level association uses legacy 22-D historical predictions only; it diagnoses the anomaly rule, " 
            "not the current Key10 main-result magnitude."
        ),
    }

    with (out_dir / "db3_signal_quality_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    _write_csv(out_dir / "subject_summary.csv", subject_rows)
    _write_csv(out_dir / "channel_summary.csv", channel_rows)
    print(f"[audit] saved: {out_dir}")


if __name__ == "__main__":
    main()
