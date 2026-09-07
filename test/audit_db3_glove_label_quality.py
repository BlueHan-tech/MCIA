"""Read-only DB3 E1 CyberGlove label-quality audit for all 22 source channels.

This diagnostic does not train a model, alter a split, or write any dataset
files. It measures whether raw source glove labels, and the fixed Key10 target
derived from them, contain usable continuous variation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from utils.kinematic_target import (
    KEY10_CHANNEL_NAMES,
    KEY10_GLOVE_INDICES,
    key10_target_metadata,
    select_key10_angles,
)
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config

EPS = 1e-8


def _parse_subjects(value: str) -> list[int]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lower, upper = (int(part) for part in token.split("-", 1))
            result.extend(range(lower, upper + 1))
        else:
            result.append(int(token))
    return sorted(set(result))


def _load_subject(root: Path, subject_id: int, exercise: int) -> tuple[np.ndarray, np.ndarray]:
    path = root / f"s{subject_id}" / f"DB3_s{subject_id}" / f"S{subject_id}_E{exercise}_A1.mat"
    if not path.exists():
        raise FileNotFoundError(path)
    data = sio.loadmat(path)
    glove = np.asarray(data["glove"], dtype=np.float64)
    labels = np.asarray(data.get("restimulus", data.get("stimulus"))).reshape(-1)
    if glove.ndim != 2 or glove.shape[1] != 22:
        raise ValueError(f"expected raw glove shape (N, 22), received {glove.shape}")
    n_samples = min(len(glove), len(labels))
    return glove[:n_samples], labels[:n_samples]


def _boundary_proximity(update_positions: np.ndarray, labels: np.ndarray, radius_samples: int) -> float:
    boundaries = np.flatnonzero(np.diff(labels) != 0) + 1
    if len(update_positions) == 0 or len(boundaries) == 0:
        return 0.0
    insertion = np.searchsorted(boundaries, update_positions)
    left = np.maximum(insertion - 1, 0)
    right = np.minimum(insertion, len(boundaries) - 1)
    distance = np.minimum(
        np.abs(update_positions - boundaries[left]),
        np.abs(update_positions - boundaries[right]),
    )
    return float(np.mean(distance <= radius_samples))


def _channel_rows(subject_id: int, glove: np.ndarray, labels: np.ndarray,
                  downsample_factor: int, epsilon: float, boundary_radius: int) -> list[dict]:
    raw_delta = np.abs(np.diff(glove, axis=0))
    down = glove[::downsample_factor]
    down_delta = np.abs(np.diff(down, axis=0))
    movement = labels != 0
    result: list[dict] = []
    for channel in range(glove.shape[1]):
        updates = np.flatnonzero(raw_delta[:, channel] > epsilon) + 1
        movement_values = glove[movement, channel]
        rest_values = glove[~movement, channel]
        key10_index = KEY10_GLOVE_INDICES.index(channel) if channel in KEY10_GLOVE_INDICES else None
        result.append({
            "subject_id": subject_id,
            "source_glove_index_1based": channel + 1,
            "is_key10": channel in KEY10_GLOVE_INDICES,
            "key10_index": key10_index,
            "key10_name": KEY10_CHANNEL_NAMES[key10_index] if key10_index is not None else "",
            "raw_min": float(np.min(glove[:, channel])),
            "raw_max": float(np.max(glove[:, channel])),
            "raw_ptp": float(np.ptp(glove[:, channel])),
            "raw_std": float(np.std(glove[:, channel])),
            "raw_n_updates": int(len(updates)),
            "raw_update_fraction": float(len(updates) / max(1, len(glove) - 1)),
            "raw_constant": bool(len(updates) == 0),
            "down_n_updates": int(np.sum(down_delta[:, channel] > epsilon)),
            "down_update_fraction": float(np.mean(down_delta[:, channel] > epsilon)),
            "movement_ptp": float(np.ptp(movement_values)) if len(movement_values) else 0.0,
            "movement_std": float(np.std(movement_values)) if len(movement_values) else 0.0,
            "rest_ptp": float(np.ptp(rest_values)) if len(rest_values) else 0.0,
            "updates_near_label_boundary_fraction": _boundary_proximity(updates, labels, boundary_radius),
        })
    return result


def _window_metrics(glove: np.ndarray, labels: np.ndarray, config: dict,
                    downsample_factor: int, dynamic_min_ptp: float) -> dict:
    glove_down = glove[::downsample_factor]
    labels_down = labels[::downsample_factor]
    n_samples = min(len(glove_down), len(labels_down))
    glove_down = glove_down[:n_samples]
    labels_down = labels_down[:n_samples]
    low = glove_down.min(axis=0, keepdims=True)
    scale = glove_down.max(axis=0, keepdims=True) - low
    constant_source = scale.reshape(-1) <= EPS
    normalized = (glove_down - low) / (scale + EPS)
    key10 = select_key10_angles(normalized)
    w_len = int(config["window_size"])
    stride = int(config["stride"])
    scores: list[float] = []
    full_scores: list[float] = []
    dynamic_contains_label_boundary: list[bool] = []
    for start in range(0, n_samples - w_len + 1, stride):
        stop = start + w_len
        if np.any(labels_down[start:stop] != 0):
            score = float(np.max(np.ptp(key10[start:stop], axis=0)))
            full_score = float(np.max(np.ptp(normalized[start:stop], axis=0)))
            scores.append(score)
            full_scores.append(full_score)
            dynamic_contains_label_boundary.append(
                bool(score >= dynamic_min_ptp and np.any(np.diff(labels_down[start:stop]) != 0))
            )
    scores_np = np.asarray(scores, dtype=np.float64)
    full_scores_np = np.asarray(full_scores, dtype=np.float64)
    return {
        "n_action_windows": int(len(scores_np)),
        "n_dynamic_windows": int(np.sum(scores_np >= dynamic_min_ptp)),
        "dynamic_window_ratio": float(np.mean(scores_np >= dynamic_min_ptp)) if len(scores_np) else 0.0,
        "full22_n_dynamic_windows": int(np.sum(full_scores_np >= dynamic_min_ptp)),
        "full22_dynamic_window_ratio": float(np.mean(full_scores_np >= dynamic_min_ptp)) if len(full_scores_np) else 0.0,
        "full22_dynamic_score_median": float(np.median(full_scores_np)) if len(full_scores_np) else 0.0,
        "full22_dynamic_score_p90": float(np.percentile(full_scores_np, 90)) if len(full_scores_np) else 0.0,
        "full22_dynamic_score_max": float(np.max(full_scores_np)) if len(full_scores_np) else 0.0,
        "n_dynamic_windows_with_label_boundary": int(sum(dynamic_contains_label_boundary)),
        "dynamic_windows_with_label_boundary_fraction": (
            float(sum(dynamic_contains_label_boundary) / np.sum(scores_np >= dynamic_min_ptp))
            if np.any(scores_np >= dynamic_min_ptp) else 0.0
        ),
        "dynamic_score_median": float(np.median(scores_np)) if len(scores_np) else 0.0,
        "dynamic_score_p90": float(np.percentile(scores_np, 90)) if len(scores_np) else 0.0,
        "dynamic_score_max": float(np.max(scores_np)) if len(scores_np) else 0.0,
        "constant_source_channels_normalized_to_zero": [int(index + 1) for index in np.flatnonzero(constant_source)],
        "constant_key10_source_channels_normalized_to_zero": [
            int(index + 1) for index in KEY10_GLOVE_INDICES if constant_source[index]
        ],
    }


def _quality_flag(rows: list[dict], windows: dict) -> str:
    key10 = [row for row in rows if row["is_key10"]]
    active = sum(not row["raw_constant"] for row in key10)
    if windows["full22_dynamic_window_ratio"] >= 0.10 and windows["dynamic_window_ratio"] < 0.10:
        return "key10_misses_available_glove_dynamics"
    if active == 0:
        return "invalid_all_key10_sources_constant"
    if active <= 2:
        return "invalid_most_key10_sources_constant"
    if windows["dynamic_window_ratio"] < 0.01:
        return "very_sparse_key10_dynamics"
    if windows["dynamic_window_ratio"] < 0.10:
        return "sparse_key10_dynamics"
    return "key10_dynamics_available"


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_update_heatmap(path: Path, rows: list[dict]) -> None:
    subjects = sorted({int(row["subject_id"]) for row in rows})
    values = np.full((len(subjects), 22), np.nan, dtype=np.float64)
    key10_mask = np.zeros(22, dtype=bool)
    key10_mask[KEY10_GLOVE_INDICES] = True
    for row in rows:
        values[subjects.index(int(row["subject_id"])), int(row["source_glove_index_1based"]) - 1] = max(
            float(row["raw_update_fraction"]), 1e-7
        )
    fig, ax = plt.subplots(figsize=(12, 5.0))
    image = ax.imshow(np.log10(values), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(np.arange(22))
    ax.set_xticklabels(np.arange(1, 23))
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels([f"S{subject_id:02d}" for subject_id in subjects])
    ax.set_xlabel("raw glove source channel (1-based); bold = current Key10")
    ax.set_ylabel("DB3 subject")
    for index, label in enumerate(ax.get_xticklabels()):
        if key10_mask[index]:
            label.set_fontweight("bold")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("log10(raw adjacent-sample update fraction)")
    ax.set_title("DB3 raw CyberGlove temporal update density")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only DB3 CyberGlove label-quality audit")
    parser.add_argument("--run-dir", required=True, help="Existing outputs/run/<run_id>")
    parser.add_argument("--subjects", default="1-11")
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--dynamic-min-ptp", type=float, default=None)
    parser.add_argument("--change-epsilon", type=float, default=1e-8)
    parser.add_argument("--boundary-ms", type=float, default=500.0)
    args = parser.parse_args()

    config = flatten_pipeline_config(load_yaml_config(PROJECT_ROOT))
    dynamic_min_ptp = float(config.get("dynamic_min_ptp", 0.05) if args.dynamic_min_ptp is None else args.dynamic_min_ptp)
    downsample_factor = int(config["orig_fs"] / config["target_fs"])
    boundary_radius = int(round(args.boundary_ms / 1000.0 * int(config["orig_fs"])))
    output_dir = Path(args.run_dir).resolve() / "06_diagnostics" / "db3_glove_label_quality_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    subject_reports: list[dict] = []
    for subject_id in _parse_subjects(args.subjects):
        print(f"[glove audit] S{subject_id:02d}", flush=True)
        glove, labels = _load_subject(Path(config["db3_path"]), subject_id, args.exercise)
        rows = _channel_rows(subject_id, glove, labels, downsample_factor, float(args.change_epsilon), boundary_radius)
        windows = _window_metrics(glove, labels, config, downsample_factor, dynamic_min_ptp)
        key10_rows = [row for row in rows if row["is_key10"]]
        report = {
            "subject_id": subject_id,
            "raw_samples": int(len(glove)),
            "duration_seconds": float(len(glove) / int(config["orig_fs"])),
            "n_active_raw_glove_channels": int(sum(not row["raw_constant"] for row in rows)),
            "n_active_key10_channels": int(sum(not row["raw_constant"] for row in key10_rows)),
            "key10_mean_raw_update_fraction": float(np.mean([row["raw_update_fraction"] for row in key10_rows])),
            "key10_min_raw_update_fraction": float(np.min([row["raw_update_fraction"] for row in key10_rows])),
            "key10_max_raw_update_fraction": float(np.max([row["raw_update_fraction"] for row in key10_rows])),
            "quality_flag": _quality_flag(rows, windows),
            "window_metrics": windows,
        }
        subject_reports.append(report)
        all_rows.extend(rows)
        print(
            f"  Key10 active={report['n_active_key10_channels']}/10 dynamic={windows['n_dynamic_windows']}/{windows['n_action_windows']} ({windows['dynamic_window_ratio']:.1%}) flag={report['quality_flag']}",
            flush=True,
        )

    report = {
        "scope": "raw DB3 E1 glove label audit; no model training or MCIA completion",
        "angle_target": key10_target_metadata(),
        "definitions": {
            "raw_update": f"absolute adjacent raw sample difference > {args.change_epsilon:g}",
            "dynamic_window": f"current Key10 normalized global max ptp >= {dynamic_min_ptp:g}",
            "quality_flag": "descriptive label-availability flag, not an EMG or model-quality verdict",
            "normalization_warning": "source channels with zero raw range become all zero in the current min-max normalization",
        },
        "subjects": subject_reports,
    }
    with (output_dir / "db3_glove_label_quality_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    _write_csv(output_dir / "db3_glove_channel_metrics.csv", all_rows)
    _plot_update_heatmap(output_dir / "db3_glove_raw_update_heatmap.png", all_rows)
    print(f"[glove audit] saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
