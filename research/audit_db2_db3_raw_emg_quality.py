"""Read-only 2 kHz sEMG quality comparison for matched DB2/DB3 actions.

The script treats DB2 as a reference distribution, not as an absolute amplitude
threshold for DB3.  It never trains a model, invokes MCIA, or changes data.
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


def _load_raw(root: Path, db: str, subject_id: int, exercise: int) -> tuple[np.ndarray, np.ndarray]:
    path = _mat_path(root, db, subject_id, exercise)
    if not path.exists():
        raise FileNotFoundError(path)
    data = sio.loadmat(path)
    emg = np.asarray(data["emg"], dtype=np.float32)
    labels = np.asarray(data.get("restimulus", data.get("stimulus"))).reshape(-1)
    n = min(len(emg), len(labels))
    return emg[:n], labels[:n]


def _action_epochs(labels: np.ndarray, epoch_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return full, non-overlapping action-only epoch starts and movement IDs."""
    starts: list[int] = []
    actions: list[int] = []
    padded = np.concatenate(([-1], labels.astype(np.int64), [-1]))
    changes = np.flatnonzero(np.diff(padded) != 0)
    for lo, hi in zip(changes[:-1], changes[1:]):
        action = int(labels[lo])
        if action == 0:
            continue
        for start in range(int(lo), int(hi) - epoch_samples + 1, epoch_samples):
            starts.append(start)
            actions.append(action)
    return np.asarray(starts, dtype=np.int64), np.asarray(actions, dtype=np.int16)


def _band_power(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    select = (freqs >= lo) & (freqs <= hi)
    return power[:, select, :].sum(axis=1)


def _epoch_features(raw: np.ndarray, labels: np.ndarray, fs: int, epoch_samples: int) -> dict:
    starts, actions = _action_epochs(labels, epoch_samples)
    if len(starts) == 0:
        raise ValueError("No full action epochs")
    offsets = np.arange(epoch_samples, dtype=np.int64)
    epochs = raw[starts[:, None] + offsets[None, :]]
    centered = epochs.astype(np.float64) - epochs.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1))
    spectrum = np.fft.rfft(centered * np.hanning(epoch_samples)[None, :, None], axis=1)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(epoch_samples, d=1.0 / fs)
    emg_power = _band_power(power, freqs, 20.0, 450.0) + EPS
    low_motion_ratio = _band_power(power, freqs, 0.5, 20.0) / emg_power
    line_ratio = _band_power(power, freqs, 49.0, 51.0) / emg_power
    cumulative = np.cumsum(power, axis=1)
    total = cumulative[:, -1:, :] + EPS
    median_index = np.argmax(cumulative >= total * 0.5, axis=1)
    median_frequency = freqs[median_index]
    return {
        "actions": actions,
        "starts": starts,
        "rms": rms.astype(np.float32),
        "low_motion_ratio": low_motion_ratio.astype(np.float32),
        "line_ratio": line_ratio.astype(np.float32),
        "median_frequency": median_frequency.astype(np.float32),
        "raw_zero_ratio": np.mean(raw == 0.0, axis=0).astype(np.float32),
        "raw_ptp": np.ptp(raw, axis=0).astype(np.float32),
    }


def _median_mad(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=axis)
    mad = np.median(np.abs(values - np.expand_dims(median, axis=axis)), axis=axis)
    return median, np.maximum(mad, EPS)


def _robust_z(values: np.ndarray, median: np.ndarray, mad: np.ndarray) -> np.ndarray:
    return 0.6745 * (values - median) / mad


def _build_db2_reference(records: list[dict]) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    by_action: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for action in np.unique(record["actions"]):
            select = record["actions"] == action
            for feature in ("low_motion_ratio", "line_ratio", "median_frequency"):
                by_action[int(action)][feature].append(record[feature][select])
    reference = {}
    for action, features in by_action.items():
        reference[action] = {}
        for feature, blocks in features.items():
            values = np.concatenate(blocks, axis=0)
            median, mad = _median_mad(values)
            reference[action][feature] = {
                "median": median,
                "mad": mad,
                "p95": np.percentile(values, 95, axis=0),
                "n_epochs": int(len(values)),
            }
    return reference


def _within_action_z(values: np.ndarray, actions: np.ndarray) -> np.ndarray:
    z = np.zeros_like(values, dtype=np.float64)
    for action in np.unique(actions):
        select = actions == action
        median, mad = _median_mad(values[select])
        z[select] = _robust_z(values[select], median, mad)
    return z


def _candidate_rows(subject_id: int, features: dict, reference: dict, epoch_samples: int,
                    fs: int) -> tuple[list[dict], dict, list[dict]]:
    actions = features["actions"]
    n_epochs, n_channels = features["rms"].shape
    low_db2_z = np.full((n_epochs, n_channels), np.nan)
    line_db2_z = np.full((n_epochs, n_channels), np.nan)
    for action in np.unique(actions):
        if int(action) not in reference:
            continue
        select = actions == action
        for feature, target in (("low_motion_ratio", low_db2_z), ("line_ratio", line_db2_z)):
            profile = reference[int(action)][feature]
            target[select] = _robust_z(features[feature][select], profile["median"], profile["mad"])

    low_self_z = _within_action_z(features["low_motion_ratio"], actions)
    line_self_z = _within_action_z(features["line_ratio"], actions)
    log_rms = np.log(features["rms"] + 1e-12)
    rms_self_z = _within_action_z(log_rms, actions)

    # A spectral artifact must be an outlier both relative to the matched DB2
    # movement and relative to the DB3 subject's own instances of that movement.
    low_candidate = (low_db2_z > 3.5) & (low_self_z > 3.5)
    line_candidate = (line_db2_z > 3.5) & (line_self_z > 3.5)

    # A contact-change candidate requires an abrupt amplitude change plus a
    # simultaneous low-frequency outlier. Amplitude alone is not considered bad.
    previous_contiguous = np.zeros(n_epochs, dtype=bool)
    if n_epochs > 1:
        previous_contiguous[1:] = (
            (actions[1:] == actions[:-1]) &
            (features["starts"][1:] - features["starts"][:-1] == epoch_samples)
        )
    rms_jump = np.zeros((n_epochs, n_channels), dtype=float)
    rms_jump[1:] = np.abs(log_rms[1:] - log_rms[:-1])
    jump_baseline = np.median(rms_jump[previous_contiguous], axis=0) if previous_contiguous.any() else np.ones(n_channels)
    jump_mad = np.median(np.abs(rms_jump[previous_contiguous] - jump_baseline), axis=0) if previous_contiguous.any() else np.ones(n_channels)
    abrupt_contact = previous_contiguous[:, None] & (rms_jump > (jump_baseline + 4.0 * np.maximum(jump_mad, 1e-6))) & low_candidate

    rows = []
    for channel in range(n_channels):
        rows.append({
            "subject_id": int(subject_id),
            "channel": int(channel + 1),
            "confirmed_all_zero": bool(features["raw_zero_ratio"][channel] == 1.0),
            "raw_zero_ratio": float(features["raw_zero_ratio"][channel]),
            "raw_peak_to_peak": float(features["raw_ptp"][channel]),
            "low_frequency_candidates": int(low_candidate[:, channel].sum()),
            "powerline_candidates": int(line_candidate[:, channel].sum()),
            "abrupt_contact_candidates": int(abrupt_contact[:, channel].sum()),
        })
    detail_rows = []
    for epoch in range(n_epochs):
        for channel in range(n_channels):
            if not (low_candidate[epoch, channel] or line_candidate[epoch, channel] or abrupt_contact[epoch, channel]):
                continue
            detail_rows.append({
                "subject_id": int(subject_id),
                "action": int(actions[epoch]),
                "start_sample_2khz": int(features["starts"][epoch]),
                "start_seconds": float(features["starts"][epoch] / fs),
                "channel": int(channel + 1),
                "low_frequency_candidate": bool(low_candidate[epoch, channel]),
                "powerline_candidate": bool(line_candidate[epoch, channel]),
                "abrupt_contact_candidate": bool(abrupt_contact[epoch, channel]),
                "low_frequency_db2_z": float(low_db2_z[epoch, channel]),
                "low_frequency_self_z": float(low_self_z[epoch, channel]),
                "powerline_db2_z": float(line_db2_z[epoch, channel]),
                "powerline_self_z": float(line_self_z[epoch, channel]),
                "rms": float(features["rms"][epoch, channel]),
            })

    summary = {
        "n_action_epochs": int(n_epochs),
        "epoch_ms": float(epoch_samples / fs * 1000.0),
        "low_frequency_candidate_ratio": float(low_candidate.mean()),
        "powerline_candidate_ratio": float(line_candidate.mean()),
        "abrupt_contact_candidate_ratio": float(abrupt_contact.mean()),
        "candidate_actions": {
            "low_frequency": sorted(set(int(a) for a in actions[np.where(low_candidate.any(axis=1))[0]])),
            "powerline": sorted(set(int(a) for a in actions[np.where(line_candidate.any(axis=1))[0]])),
            "abrupt_contact": sorted(set(int(a) for a in actions[np.where(abrupt_contact.any(axis=1))[0]])),
        },
    }
    return rows, summary, detail_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _true_runs(mask: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    """Return inclusive/exclusive True runs with a minimum length."""
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)) != 0)
    return [
        (int(start), int(stop))
        for start, stop in zip(changes[::2], changes[1::2])
        if stop - start >= min_samples
    ]


def _dominant_action(labels: np.ndarray, start: int, stop: int) -> int:
    segment = labels[start:stop]
    if segment.size == 0:
        return -1
    values, counts = np.unique(segment, return_counts=True)
    return int(values[np.argmax(counts)])


def _raw_integrity_scan(raw: np.ndarray, labels: np.ndarray, fs: int, domain: str,
                        subject_id: int, keep_events: bool) -> tuple[list[dict], list[dict]]:
    """Find exact raw-sample failures without treating weak EMG as missing."""
    n_samples, n_channels = raw.shape
    min_plateau = max(4, int(round(0.002 * fs)))
    min_hold = max(8, int(round(0.004 * fs)))
    channel_rows: list[dict] = []
    events: list[dict] = []

    def add_event(channel: int, event_type: str, start: int, stop: int,
                  value: float | None = None, peak_z: float | None = None) -> None:
        if not keep_events:
            return
        event = {
            "domain": domain, "subject_id": int(subject_id), "channel": int(channel + 1),
            "event_type": event_type, "start_sample_2khz": int(start),
            "stop_sample_2khz": int(stop), "start_seconds": float(start / fs),
            "duration_ms": float((stop - start) / fs * 1000.0),
            "dominant_restimulus": _dominant_action(labels, start, stop),
        }
        if value is not None:
            event["plateau_value"] = float(value)
        if peak_z is not None:
            event["peak_derivative_robust_z"] = float(peak_z)
        events.append(event)

    for channel in range(n_channels):
        signal = raw[:, channel]
        zero_runs = _true_runs(signal == 0.0, min_hold)
        nonzero_plateaus = []
        for value in (float(np.min(signal)), float(np.max(signal))):
            if value == 0.0:
                continue
            for start, stop in _true_runs(signal == value, min_plateau):
                nonzero_plateaus.append((start, stop, value))
                add_event(channel, "adc_rail_or_constant_plateau", start, stop, value=value)

        equal_previous = np.concatenate(([False], signal[1:] == signal[:-1]))
        constant_holds = [
            (start, stop) for start, stop in _true_runs(equal_previous, min_hold)
            if not np.all(signal[start:stop] == 0.0)
        ]
        for start, stop in constant_holds:
            add_event(channel, "single_channel_hold", start, stop, value=float(signal[start]))

        derivative = np.abs(np.diff(signal.astype(np.float64)))
        derivative_median, derivative_mad = _median_mad(derivative)
        robust_z = _robust_z(derivative, derivative_median, derivative_mad)
        threshold = max(float(np.percentile(derivative, 99.995)),
                        float(derivative_median + 25.0 * derivative_mad))
        positions = np.flatnonzero((derivative >= threshold) & (robust_z >= 25.0))
        for index in positions:
            add_event(channel, "transient_impulse_review", int(index),
                      min(n_samples, int(index) + 2), peak_z=float(robust_z[index]))

        channel_rows.append({
            "domain": domain, "subject_id": int(subject_id), "channel": int(channel + 1),
            "zero_hold_runs": int(len(zero_runs)),
            "nonzero_rail_or_plateau_runs": int(len(nonzero_plateaus)),
            "single_channel_hold_runs": int(len(constant_holds)),
            "transient_impulse_review_events": int(len(positions)),
        })

    repeated_count = np.sum(raw[1:] == raw[:-1], axis=1)
    repeated_mask = repeated_count >= max(3, int(np.ceil(n_channels / 2)))
    multi_hold_runs = _true_runs(repeated_mask, min_plateau)
    channel_multi_counts = np.zeros(n_channels, dtype=np.int64)
    for start, stop in multi_hold_runs:
        held_channels = np.flatnonzero(np.all(raw[start:stop + 1] == raw[start], axis=0))
        for channel in held_channels:
            channel_multi_counts[channel] += 1
            add_event(int(channel), "multi_channel_repeated_frame", start, stop + 1,
                      value=float(raw[start, channel]))
    for row in channel_rows:
        row["multi_channel_repeated_frame_runs"] = int(channel_multi_counts[row["channel"] - 1])
    return channel_rows, events


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only 2 kHz DB2/DB3 sEMG quality audit")
    parser.add_argument("--run-dir", required=True, help="Existing outputs/run/<run_id>")
    parser.add_argument("--db2-subjects", default="1-40")
    parser.add_argument("--db3-subjects", default="1-11")
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--epoch-ms", type=float, default=1000.0)
    args = parser.parse_args()

    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    fs = int(config["orig_fs"])
    epoch_samples = int(round(args.epoch_ms / 1000.0 * fs))
    if epoch_samples < 100:
        raise ValueError("epoch-ms must provide at least 100 samples")
    out_dir = Path(args.run_dir).resolve() / "06_diagnostics" / "db2_db3_raw_emg_quality_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    db2_subjects, db3_subjects = _parse_subjects(args.db2_subjects), _parse_subjects(args.db3_subjects)

    db2_records, db2_integrity_rows = [], []
    for subject_id in db2_subjects:
        print(f"[db2 reference] S{subject_id:02d}")
        raw, labels = _load_raw(Path(config["db2_path"]), "db2", subject_id, args.exercise)
        integrity_rows, _ = _raw_integrity_scan(raw, labels, fs, "db2", subject_id, keep_events=False)
        db2_integrity_rows.extend(integrity_rows)
        db2_records.append(_epoch_features(raw, labels, fs, epoch_samples))
    reference = _build_db2_reference(db2_records)

    profile_rows = []
    for action, features in sorted(reference.items()):
        for feature, profile in features.items():
            for channel in range(len(profile["median"])):
                profile_rows.append({
                    "action": action, "channel": channel + 1, "feature": feature,
                    "median": float(profile["median"][channel]), "mad": float(profile["mad"][channel]),
                    "p95": float(profile["p95"][channel]), "n_epochs": profile["n_epochs"],
                })

    subjects, channel_rows, candidate_detail_rows = [], [], []
    db3_integrity_rows, db3_integrity_events = [], []
    for subject_id in db3_subjects:
        print(f"[db3 audit] S{subject_id:02d}")
        raw, labels = _load_raw(Path(config["db3_path"]), "db3", subject_id, args.exercise)
        integrity_rows, integrity_events = _raw_integrity_scan(raw, labels, fs, "db3", subject_id, keep_events=True)
        db3_integrity_rows.extend(integrity_rows)
        db3_integrity_events.extend(integrity_events)
        features = _epoch_features(raw, labels, fs, epoch_samples)
        rows, summary, details = _candidate_rows(subject_id, features, reference, epoch_samples, fs)
        channel_rows.extend(rows)
        candidate_detail_rows.extend(details)
        integrity_summary = {
            key: int(sum(row[key] for row in integrity_rows))
            for key in ("zero_hold_runs", "nonzero_rail_or_plateau_runs", "single_channel_hold_runs",
                        "multi_channel_repeated_frame_runs", "transient_impulse_review_events")
        }
        subjects.append({"subject_id": subject_id, "summary": summary, "channels": rows,
                         "raw_integrity": integrity_summary})
        confirmed = [row["channel"] for row in rows if row["confirmed_all_zero"]]
        print(f"  zero_channels={confirmed} low_freq={summary['low_frequency_candidate_ratio']:.3%} "
              f"line={summary['powerline_candidate_ratio']:.3%} contact={summary['abrupt_contact_candidate_ratio']:.3%} "
              f"plateau={integrity_summary['nonzero_rail_or_plateau_runs']} "
              f"repeat={integrity_summary['multi_channel_repeated_frame_runs']}")

    report = {
        "scope": "read-only raw 2 kHz DB2/DB3 E1 matched-restimulus quality audit",
        "method": {
            "confirmed_failure": "entire raw channel exactly zero",
            "low_frequency_artifact": "DB2 matched-action and DB3 within-action robust z > 3.5",
            "powerline_artifact": "49-51 Hz ratio, DB2 matched-action and DB3 within-action robust z > 3.5",
            "contact_change": "abrupt RMS jump plus low-frequency artifact; amplitude alone is not abnormal",
            "raw_integrity": "exact raw-sample plateaus, repeated multi-channel frames, and extreme derivative impulses",
            "important_limit": "impulses and spectral artifacts require waveform review or acquisition metadata before being called ground-truth failures",
        },
        "db2_reference_subjects": db2_subjects,
        "db3_subjects": subjects,
    }
    with (out_dir / "db2_db3_raw_emg_quality_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    _write_csv(out_dir / "db2_action_reference.csv", profile_rows)
    _write_csv(out_dir / "db3_channel_candidates.csv", channel_rows)
    _write_csv(out_dir / "db3_epoch_candidates.csv", candidate_detail_rows)
    _write_csv(out_dir / "db2_raw_integrity_summary.csv", db2_integrity_rows)
    _write_csv(out_dir / "db3_raw_integrity_summary.csv", db3_integrity_rows)
    _write_csv(out_dir / "db3_raw_integrity_candidates.csv", db3_integrity_events)
    print(f"[audit] saved: {out_dir}")


if __name__ == "__main__":
    main()
