"""Dev-only DB3 raw sEMG panels grouped by exercise/task/rep.

This script is intentionally not connected to run_all_experiments.py. It reads
one DB3 subject through the existing NinaProDataLoader path, applies the same lightweight
preprocessing used by prepare_data_db3, and writes raw/original sEMG figures only.
No checkpoints, caches, metrics, masks, predictions, or run outputs are changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import yaml

from data.dataset_db2_emg import moving_average
from data.ninapro_loader import NinaProDataLoader

SUBJECT_ID = 1
SUBJECT_LABEL = f"S{SUBJECT_ID:02d}"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dev" / f"db3_s{SUBJECT_ID:02d}_raw_semg_by_task"
EXERCISES = (1, 2, 3)
MAX_TASKS_PER_EXERCISE = 3
MAX_WINDOWS_PER_TASK = 2


def load_config() -> dict:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    flat = {
        "db2_path": cfg["paths"]["db2"],
        "db3_path": cfg["paths"]["db3"],
        "orig_fs": cfg["signal"]["orig_fs"],
        "target_fs": cfg["signal"]["target_fs"],
        "window_size": cfg["signal"]["window_size"],
        "stride": cfg["signal"]["stride"],
    }
    return flat


def preprocess_and_segment_exercise(data_loader: NinaProDataLoader, subject_id: int,
                                    exercise: int, config: dict):
    data = data_loader.load_db3_subject(subject_id, exercises=[exercise])
    downsample_factor = int(config["orig_fs"] / config["target_fs"])

    emg = data["emg"] * 1000.0
    emg = data_loader.bandpass_filter(emg)
    emg = data_loader.notch_filter(emg)
    emg_rect = np.abs(emg)
    emg_env = moving_average(emg_rect, downsample_factor)
    emg_down = emg_env[::downsample_factor]

    labels_source = data.get("restimulus", data.get("stimulus"))
    reps_source = data.get("repetition", np.zeros(data["emg"].shape[0], dtype=np.int32))
    labels_down = labels_source[::downsample_factor]
    reps_down = reps_source[::downsample_factor]

    mu = 255.0
    emg_max = emg_down.max()
    if emg_max > 0:
        emg_normalized_temp = emg_down / emg_max
        emg_down = (np.log1p(mu * emg_normalized_temp) / np.log1p(mu)) * emg_max

    q05 = np.percentile(emg_down, 5)
    q99 = np.percentile(emg_down, 99)
    emg_norm = (emg_down - q05) / ((q99 - q05) + 1e-8)
    emg_norm = np.clip(emg_norm, 0.0, 1.0)

    segments = []
    metadata = []
    w_len = int(config["window_size"])
    stride = int(config["stride"])
    center_offset = w_len // 2
    n_samples = emg_norm.shape[0]
    for start in range(0, n_samples - w_len, stride):
        end = start + w_len
        window_labels = labels_down[start:end]
        if not np.any(window_labels != 0):
            continue
        center = min(start + center_offset, len(labels_down) - 1)
        active = window_labels[window_labels != 0]
        task = int(np.bincount(active.astype(np.int64)).argmax()) if len(active) else int(labels_down[center])
        rep = int(reps_down[center])
        segments.append(emg_norm[start:end, :])
        metadata.append({
            "exercise": int(exercise),
            "task": task,
            "rep": rep,
            "window_index": len(segments) - 1,
            "start_sample_200hz": int(start),
            "end_sample_200hz": int(end),
        })
    return np.asarray(segments, dtype=np.float32), metadata


def select_representative_indices(metadata: list[dict],
                                  max_tasks_per_exercise: int = MAX_TASKS_PER_EXERCISE,
                                  max_windows_per_task: int = MAX_WINDOWS_PER_TASK) -> list[int]:
    by_exercise_task: dict[tuple[int, int], list[int]] = {}
    for idx, item in enumerate(metadata):
        key = (int(item["exercise"]), int(item["task"]))
        by_exercise_task.setdefault(key, []).append(idx)

    selected = []
    tasks_by_exercise: dict[int, list[int]] = {}
    for exercise, task in sorted(by_exercise_task):
        tasks_by_exercise.setdefault(exercise, []).append(task)

    for exercise in sorted(tasks_by_exercise):
        for task in tasks_by_exercise[exercise][:max_tasks_per_exercise]:
            indices = by_exercise_task[(exercise, task)]
            if len(indices) <= max_windows_per_task:
                selected.extend(indices)
                continue
            positions = np.linspace(0, len(indices) - 1, max_windows_per_task, dtype=int)
            selected.extend([indices[int(pos)] for pos in positions])
    return selected


def plot_raw_semg_panel(segment: np.ndarray, save_path: Path, title: str, header_text: str) -> None:
    t_steps, n_channels = segment.shape
    n_cols = 3
    n_rows = int(np.ceil(n_channels / n_cols))
    fig = plt.figure(figsize=(7.5 * n_cols, 2.6 * n_rows + 1.0))
    gs = GridSpec(
        n_rows + 1,
        n_cols,
        figure=fig,
        hspace=0.55,
        wspace=0.22,
        height_ratios=[0.25] + [1.0] * n_rows,
    )

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    header_ax.text(
        0.5,
        0.5,
        header_text,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        bbox=dict(boxstyle="round", fc="#f0f7ff", ec="#4a90e2"),
    )

    xs = np.arange(t_steps)
    raw_color = "#2ca02c"
    for ch in range(n_channels):
        row, col = divmod(ch, n_cols)
        ax = fig.add_subplot(gs[row + 1, col])
        ax.plot(xs, segment[:, ch], color=raw_color, label="Raw sEMG", linewidth=1.2, alpha=0.9)
        ax.set_title(f"Ch{ch + 1} | raw/original sEMG", fontsize=10.5, fontweight="bold", color="#2e7d32")
        ax.grid(True, alpha=0.25)
        if ch == 0:
            ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = load_config()
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])

    all_segments = []
    all_metadata = []
    for exercise in EXERCISES:
        segments, metadata = preprocess_and_segment_exercise(loader, SUBJECT_ID, exercise, config)
        offset = len(all_segments)
        all_segments.extend(list(segments))
        for item in metadata:
            item = dict(item)
            item["global_window_index"] = offset + int(item["window_index"])
            all_metadata.append(item)
        print(f"S{SUBJECT_ID:02d} E{exercise}: {len(segments)} active windows")

    if not all_segments:
        raise RuntimeError(f"No active DB3 {SUBJECT_LABEL} windows found.")

    segments_arr = np.asarray(all_segments, dtype=np.float32)
    selected = select_representative_indices(all_metadata)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = []
    for idx in selected:
        meta = all_metadata[idx]
        exercise = int(meta["exercise"])
        task = int(meta["task"])
        rep = int(meta["rep"])
        global_idx = int(meta["global_window_index"])
        filename = f"{SUBJECT_LABEL}_E{exercise}_task{task:02d}_rep{rep:02d}_win{global_idx:04d}.png"
        path = OUTPUT_DIR / filename
        title = f"DB3 subject={SUBJECT_LABEL} | exercise/task=E{exercise}/task{task:02d} | rep={rep} | window={global_idx}"
        header = (
            f"raw/original sEMG only    subject={SUBJECT_LABEL}    exercise=E{exercise}    "
            f"task={task:02d}    rep={rep}    window={global_idx}"
        )
        plot_raw_semg_panel(segments_arr[idx], path, title=title, header_text=header)
        saved.append(path)

    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Generated PNG: {len(saved)}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
