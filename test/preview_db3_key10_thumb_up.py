"""Independent current-Key10 raw-EMG Thumb-up preview for DB3 S05/S06."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.kinematic_output_postprocess import postprocess_window_predictions
from utils.kinematic_target import KEY10_PLOT_GROUPS
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config, set_seed


def _exp3():
    path = PROJECT_ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("exp3_preview", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Exp3 helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_subjects(value: str) -> list[int]:
    return sorted({int(item) for item in value.split(",") if item.strip()})


def _select_contiguous(starts, stimulus, repetitions, scores, stimulus_id, stride):
    candidates = np.flatnonzero(stimulus == stimulus_id)
    if len(candidates) == 0:
        raise RuntimeError(f"No held-out test windows have stimulus={stimulus_id}")
    blocks = []
    begin = 0
    for pos in range(1, len(candidates) + 1):
        split = pos == len(candidates)
        if not split:
            prev_idx, curr_idx = candidates[pos - 1], candidates[pos]
            split = starts[curr_idx] - starts[prev_idx] != stride or repetitions[curr_idx] != repetitions[prev_idx]
        if split:
            blocks.append(candidates[begin:pos])
            begin = pos
    return max(blocks, key=lambda block: (len(block), float(np.max(scores[block]))))


def _save_plot(path, subject_id, repetition, target, prediction, times, metrics, best_val, epochs, target_fs):
    group = KEY10_PLOT_GROUPS["key10"]
    fig, axes = plt.subplots(*group["shape"], figsize=(16.5, 14.0), sharex=True)
    seconds = (times - times[0]) / target_fs
    for axis, (name, channel) in zip(axes.reshape(-1), group["items"]):
        axis.plot(seconds, target[:, channel], color="#222222", lw=1.7, label="Ground Truth")
        axis.plot(seconds, prediction[:, channel], color="#1f77b4", lw=1.5, label="Raw EMG KinematicTCN")
        axis.set_title(name, fontsize=10, fontweight="bold")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.24)
    axes.reshape(-1)[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time within held-out Thumb-up segment (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("normalized angle")
    value = metrics["global"]
    fig.suptitle(
        f"DB3 S{subject_id:02d} | stimulus=1 Thumb up | held-out repetition {repetition} | current Key10 raw-EMG preview\n"
        f"continuous RMSE={value['rmse']:.4f}, Pearson={value['pearson']:.3f}, validation MSE={best_val:.5f}, epochs={epochs}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="DB3 current-Key10 Thumb-up raw-EMG preview")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subjects", default="5,6")
    parser.add_argument("--stimulus", type=int, default=1)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not os.environ.get("MCIA_RUN_DIR"):
        raise RuntimeError("MCIA_RUN_DIR must be set for this single diagnostic step")
    if Path(os.environ["MCIA_RUN_DIR"]).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")

    config = flatten_pipeline_config(load_yaml_config(PROJECT_ROOT))
    set_seed(int(config["regressor_random_seed"]))
    exp3 = _exp3()
    output_dir = run_dir / "06_diagnostics" / "db3_key10_thumb_up_preview"
    figure_dir = output_dir / "figures"
    prediction_dir = output_dir / "predictions"
    figure_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=int(config["orig_fs"]))
    factor = int(config["orig_fs"] / config["target_fs"])
    results = {"scope": "Group A current Key10 raw-EMG preview only; no MCIA and no Exp3 artifact writes", "stimulus": args.stimulus, "subjects": []}

    for subject_id in _parse_subjects(args.subjects):
        print(f"[preview] S{subject_id:02d} train raw KinematicTCN", flush=True)
        emg, angle, _, reps = prepare_kinematics_data(loader, [subject_id], config, exercises=[1], db="db3")
        train_idx, val_idx, test_idx = make_rep_split(
            reps, train_reps=(1, 3, 4, 6), test_reps=(2, 5),
            val_ratio=float(config["regressor_val_ratio"]), seed=int(config["regressor_random_seed"]) + subject_id,
        )
        model, best_val, epochs = exp3.train_tcn_on_emg(emg, angle, train_idx, val_idx, config, config["device"])
        val_pred, val_target = exp3.predict_on_set(model, emg, angle, val_idx, config, config["device"])
        calibration = exp3._fit_channelwise_affine(val_pred, val_target)
        test_pred, test_target = exp3.predict_on_set(model, emg, angle, test_idx, config, config["device"])
        test_pred = exp3._apply_channelwise_affine(test_pred, calibration)

        starts = exp3._reconstruct_db3_window_starts(loader, subject_id, config, reps, exercises=[1])
        test_starts = starts[test_idx]
        raw = loader.load_db3_subject(subject_id, [1])
        labels = np.asarray(raw.get("restimulus", raw["stimulus"])).reshape(-1)[::factor]
        raw_reps = np.asarray(raw["repetition"]).reshape(-1)[::factor]
        centers = test_starts + int(config["window_size"]) // 2
        center_labels, center_reps = labels[centers], raw_reps[centers]
        scores = exp3.compute_dynamic_angle_scores(test_target)
        chosen = _select_contiguous(test_starts, center_labels, center_reps, scores, args.stimulus, int(config["stride"]))
        local_dynamic = np.flatnonzero(scores[chosen] >= float(config["regressor_dynamic_min_ptp"]))
        packed = postprocess_window_predictions(
            test_pred[chosen], test_target[chosen], test_starts[chosen], local_dynamic,
            savgol_window=int(config["regressor_output_postprocess_savgol_window"]),
            polyorder=int(config["regressor_output_postprocess_savgol_polyorder"]),
            fusion=str(config["regressor_output_postprocess_fusion"]),
            filter_kind=str(config["regressor_output_postprocess_filter"]),
            target_fs=int(config["target_fs"]), lowpass_hz=float(config["regressor_output_postprocess_lowpass_hz"]),
        )
        target, prediction, times = packed["continuous_target"], packed["continuous_prediction"], packed["continuous_time_indices"]
        metrics = exp3.evaluate_subsets(prediction[None, :, :], target[None, :, :])
        repetition = int(center_reps[chosen[0]])
        figure = figure_dir / f"S{subject_id:02d}_stimulus1_thumb_up_rep{repetition}_key10_raw_tcn.png"
        _save_plot(figure, subject_id, repetition, target, prediction, times, metrics, best_val, epochs, int(config["target_fs"]))
        prediction_file = prediction_dir / f"S{subject_id:02d}_stimulus1_thumb_up_key10_raw_tcn.npz"
        np.savez(prediction_file, target=target, prediction=prediction, time_indices=times, overlap_counts=packed["continuous_overlap_counts"], test_local_window_indices=chosen, test_window_starts=test_starts[chosen], center_stimulus=center_labels[chosen], center_repetition=center_reps[chosen])
        item = {
            "subject_id": subject_id, "best_validation_mse": float(best_val), "epochs": int(epochs),
            "split_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "selected_test_local_window_indices": [int(value) for value in chosen],
            "selected_repetition": repetition, "selected_duration_seconds": float(len(target) / int(config["target_fs"])),
            "continuous_metrics": metrics, "figure": str(figure), "prediction": str(prediction_file),
        }
        results["subjects"].append(item)
        print(f"  Thumb up rep={repetition}, duration={item['selected_duration_seconds']:.2f}s, RMSE={metrics['global']['rmse']:.4f}", flush=True)

    report = output_dir / "db3_key10_thumb_up_preview_results.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[preview] saved: {report}", flush=True)


if __name__ == "__main__":
    main()
