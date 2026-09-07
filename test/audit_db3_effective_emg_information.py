"""Read-only DB2/DB3 audit for EMG channels with little action-related information.

This is deliberately not a missing-data detector.  It looks for DB3 channels
whose EMG remains rest-like during objectively dynamic glove periods and whose
subject-level action coupling is unusually low compared with DB2.  Any result
is an MCIA masking candidate for follow-up validation, never a claim that the
original EMG was corrupted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np
import scipy.io as sio

from data.dataset_db2_emg import moving_average
from data.ninapro_loader import NinaProDataLoader
from utils.kinematic_target import KEY10_GLOVE_INDICES
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config


EPS = np.finfo(np.float64).eps


def _parse_subjects(value: str) -> list[int]:
    subjects: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = (int(part) for part in token.split("-", 1))
            subjects.extend(range(lo, hi + 1))
        else:
            subjects.append(int(token))
    return sorted(set(subjects))


def _mat_path(root: Path, db: str, subject_id: int, exercise: int) -> Path:
    if db == "db2":
        return root / f"DB2_s{subject_id}" / f"S{subject_id}_E{exercise}_A1.mat"
    return root / f"s{subject_id}" / f"DB3_s{subject_id}" / f"S{subject_id}_E{exercise}_A1.mat"


def _load_mat(root: Path, db: str, subject_id: int, exercise: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _mat_path(root, db, subject_id, exercise)
    if not path.exists():
        raise FileNotFoundError(path)
    data = sio.loadmat(path)
    emg = np.asarray(data["emg"], dtype=np.float32)
    glove = np.asarray(data["glove"], dtype=np.float32)
    labels = np.asarray(data.get("restimulus", data.get("stimulus"))).reshape(-1)
    n = min(len(emg), len(glove), len(labels))
    return emg[:n], glove[:n], labels[:n]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10 or np.std(x) <= EPS or np.std(y) <= EPS:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def _max_lag_abs_corr(signal: np.ndarray, target: np.ndarray, max_lag: int) -> float:
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            value = _safe_corr(signal[:lag], target[-lag:])
        elif lag > 0:
            value = _safe_corr(signal[lag:], target[:-lag])
        else:
            value = _safe_corr(signal, target)
        best = max(best, abs(value))
    return best


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    overall = float(np.mean(values))
    total = float(np.sum((values - overall) ** 2))
    if total <= EPS:
        return 0.0
    between = 0.0
    for action in np.unique(labels):
        select = labels == action
        between += float(select.sum()) * (float(np.mean(values[select])) - overall) ** 2
    return float(np.clip(between / total, 0.0, 1.0))


def _normalise_glove(glove: np.ndarray) -> np.ndarray:
    lo = glove.min(axis=0, keepdims=True)
    scale = glove.max(axis=0, keepdims=True) - lo + EPS
    return (glove - lo) / scale


def _preprocess(emg: np.ndarray, glove: np.ndarray, labels: np.ndarray,
                loader: NinaProDataLoader, factor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filtered = loader.notch_filter(loader.bandpass_filter(emg * 1000.0))
    envelope = moving_average(np.abs(filtered), factor)[::factor]
    glove_down = _normalise_glove(glove[::factor])[:, KEY10_GLOVE_INDICES]
    labels_down = labels[::factor]
    n = min(len(envelope), len(glove_down), len(labels_down))
    return envelope[:n], glove_down[:n], labels_down[:n]


def _subject_metrics(envelope: np.ndarray, glove: np.ndarray, labels: np.ndarray,
                     epoch_samples: int, max_lag_samples: int,
                     target_fs: int) -> tuple[list[dict], list[dict]]:
    movement = labels != 0
    rest = labels == 0
    velocity = np.linalg.norm(np.diff(glove, axis=0, prepend=glove[:1]), axis=1)
    dynamic_cutoff = float(np.percentile(velocity[movement], 75)) if movement.any() else np.inf
    rows: list[dict] = []
    epoch_rows: list[dict] = []

    for channel in range(envelope.shape[1]):
        signal = envelope[:, channel]
        rest_level = float(np.median(signal[rest])) if rest.any() else float(np.median(signal))
        action_level = float(np.median(signal[movement])) if movement.any() else rest_level
        action_signal = signal[movement]
        action_velocity = velocity[movement]
        action_labels = labels[movement]
        rows.append({
            "channel": channel + 1,
            "action_to_rest_ratio": action_level / (rest_level + EPS),
            "action_eta_squared": _eta_squared(action_signal, action_labels),
            "max_abs_glove_velocity_corr": _max_lag_abs_corr(action_signal, action_velocity, max_lag_samples),
            "rest_envelope_median": rest_level,
            "action_envelope_median": action_level,
        })

        padded = np.concatenate(([-1], labels.astype(np.int64), [-1]))
        changes = np.flatnonzero(np.diff(padded) != 0)
        for lo, hi in zip(changes[:-1], changes[1:]):
            action = int(labels[lo])
            if action == 0:
                continue
            for start in range(int(lo), int(hi) - epoch_samples + 1, epoch_samples):
                stop = start + epoch_samples
                epoch_velocity = velocity[start:stop]
                if float(np.median(epoch_velocity)) < dynamic_cutoff:
                    continue
                epoch_signal = signal[start:stop]
                epoch_rows.append({
                    "channel": channel + 1,
                    "action": action,
                    "start_sample_200hz": start,
                    "start_seconds": start / target_fs,
                    "median_velocity": float(np.median(epoch_velocity)),
                    "action_to_rest_ratio": float(np.median(epoch_signal) / (rest_level + EPS)),
                    "abs_velocity_corr": abs(_safe_corr(epoch_signal, epoch_velocity)),
                })
    return rows, epoch_rows


def _reference_percentile(value: float, reference: np.ndarray) -> float:
    return float(np.mean(reference <= value))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only EMG effective-information audit")
    parser.add_argument("--run-dir", required=True, help="Existing outputs/run/<run_id>")
    parser.add_argument("--db2-subjects", default="1-40")
    parser.add_argument("--db3-subjects", default="1-11")
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--epoch-ms", type=float, default=1000.0)
    parser.add_argument("--max-lag-ms", type=float, default=200.0)
    args = parser.parse_args()

    config = flatten_pipeline_config(load_yaml_config(PROJECT_ROOT))
    factor = int(config["orig_fs"] / config["target_fs"])
    target_fs = int(config["target_fs"])
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=int(config["orig_fs"]))
    epoch_samples = int(round(args.epoch_ms / 1000.0 * target_fs))
    max_lag_samples = int(round(args.max_lag_ms / 1000.0 * target_fs))
    out_dir = Path(args.run_dir).resolve() / "06_diagnostics" / "db3_effective_emg_information_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    db2_by_channel: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    db2_subjects = _parse_subjects(args.db2_subjects)
    for subject_id in db2_subjects:
        print(f"[db2 reference] S{subject_id:02d}")
        raw, glove, labels = _load_mat(Path(config["db2_path"]), "db2", subject_id, args.exercise)
        envelope, glove_down, labels_down = _preprocess(raw, glove, labels, loader, factor)
        rows, _ = _subject_metrics(envelope, glove_down, labels_down, epoch_samples, max_lag_samples, target_fs)
        for row in rows:
            for metric in ("action_to_rest_ratio", "action_eta_squared", "max_abs_glove_velocity_corr"):
                db2_by_channel[row["channel"]][metric].append(float(row[metric]))

    db3_subjects = _parse_subjects(args.db3_subjects)
    db3_rows, candidate_epochs, subject_summaries = [], [], []
    for subject_id in db3_subjects:
        print(f"[db3 audit] S{subject_id:02d}")
        raw, glove, labels = _load_mat(Path(config["db3_path"]), "db3", subject_id, args.exercise)
        envelope, glove_down, labels_down = _preprocess(raw, glove, labels, loader, factor)
        rows, epochs = _subject_metrics(envelope, glove_down, labels_down, epoch_samples, max_lag_samples, target_fs)
        by_channel = {row["channel"]: row for row in rows}
        low_information_channels = []
        for row in rows:
            for metric in ("action_to_rest_ratio", "action_eta_squared", "max_abs_glove_velocity_corr"):
                row[f"db2_percentile_{metric}"] = _reference_percentile(
                    float(row[metric]), np.asarray(db2_by_channel[row["channel"]][metric])
                )
            low_information = all(
                row[f"db2_percentile_{metric}"] <= 0.10
                for metric in ("action_to_rest_ratio", "action_eta_squared", "max_abs_glove_velocity_corr")
            )
            row["low_effective_information_candidate"] = bool(low_information)
            row["subject_id"] = subject_id
            if low_information:
                low_information_channels.append(row["channel"])
        for epoch in epochs:
            channel = by_channel[epoch["channel"]]
            epoch["subject_id"] = subject_id
            epoch["channel_low_effective_information"] = channel["low_effective_information_candidate"]
            epoch["low_information_dynamic_epoch_candidate"] = bool(
                channel["low_effective_information_candidate"]
                and epoch["action_to_rest_ratio"] <= 1.25
                and epoch["abs_velocity_corr"] <= 0.10
            )
            if epoch["low_information_dynamic_epoch_candidate"]:
                candidate_epochs.append(epoch)
        db3_rows.extend(rows)
        subject_summaries.append({
            "subject_id": subject_id,
            "low_effective_information_channels": low_information_channels,
            "n_low_information_dynamic_epochs": int(sum(
                row["subject_id"] == subject_id for row in candidate_epochs
            )),
        })
        print(f"  low_information_channels={low_information_channels} "
              f"dynamic_epoch_candidates={subject_summaries[-1]['n_low_information_dynamic_epochs']}")

    report = {
        "scope": "read-only DB2 reference / DB3 E1 effective-information audit",
        "definition": {
            "low_effective_information_channel": "all three channel metrics fall in the lowest 10% of matched DB2 channel distributions",
            "metrics": ["action-to-rest envelope ratio", "action-label eta squared", "maximum absolute glove-velocity correlation over +/- max-lag-ms"],
            "dynamic_epoch_candidate": "candidate channel, high-glove-velocity epoch, rest-like envelope and weak within-epoch velocity correlation",
            "important_limit": "this identifies masking candidates only; it does not prove corrupted EMG or justify MCIA changes without a separate no-leakage completion ablation",
        },
        "db2_reference_subjects": db2_subjects,
        "subjects": subject_summaries,
    }
    with (out_dir / "db3_effective_emg_information_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    _write_csv(out_dir / "db3_channel_information.csv", db3_rows)
    _write_csv(out_dir / "db3_low_information_dynamic_epochs.csv", candidate_epochs)
    print(f"[audit] saved: {out_dir}")


if __name__ == "__main__":
    main()
