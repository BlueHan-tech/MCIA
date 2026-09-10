"""Validation-only MC-dropout uncertainty screen against DB3 all-channel Oracle gains."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np
import torch

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config


def load_exp3():
    path = ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("mc_dropout_gain_exp3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ranks(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return result


def spearman(left, right):
    left_rank, right_rank = ranks(left), ranks(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def top_gain_summary(confidence, gain):
    count = max(1, int(np.ceil(0.10 * len(confidence))))
    top = np.argsort(confidence)[-count:]
    return {
        "top_10pct_gain_mean": float(gain[top].mean()),
        "top_10pct_positive_gain_fraction": float((gain[top] > 0.0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="DB3 validation-only MC-dropout uncertainty versus Oracle-gain screen"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mc_samples < 2:
        raise ValueError("mc-samples must be at least 2")
    run_dir = Path(args.run_dir).resolve()
    env_run_dir = os.environ.get("MCIA_RUN_DIR")
    if not env_run_dir or Path(env_run_dir).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")

    oracle_path = (
        run_dir
        / "06_diagnostics"
        / "db3_completion_oracle"
        / "predictions"
        / f"S{args.subject:02d}_oracle.npz"
    )
    if not oracle_path.exists():
        raise FileNotFoundError(f"Run Oracle diagnostic first: {oracle_path}")
    oracle = np.load(oracle_path)
    validation_indices = oracle["validation_indices"]
    gain = oracle["candidate_gain"]

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    device = config["device"]
    batch_size = args.batch_size or int(config["regressor_batch_size"])
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    exp3 = load_exp3()
    model, checkpoint = exp3.load_healthy_prior_mcia(config, device)
    if model is None:
        raise FileNotFoundError("healthy-prior MCIA checkpoint")
    loader = NinaProDataLoader(
        config["db2_path"], config["db3_path"], fs=int(config["orig_fs"])
    )
    emg, _, _, repetitions = prepare_kinematics_data(
        loader, [args.subject], config, exercises=[args.exercise], db="db3"
    )
    _, _, test_idx = make_rep_split(
        repetitions,
        train_reps=(1, 3, 4, 6),
        test_reps=(2, 5),
        val_ratio=float(config["regressor_val_ratio"]),
        seed=int(config["regressor_random_seed"]) + args.subject,
    )
    windows = emg[validation_indices]
    n_windows, n_steps, n_channels = windows.shape
    patch_size = int(config["patch_size"])
    if n_steps % patch_size:
        raise ValueError("window length must be divisible by patch_size")
    n_patches = n_steps // patch_size
    if gain.shape != (n_windows, n_patches):
        raise ValueError(
            f"Oracle gain has shape {gain.shape}, expected {(n_windows, n_patches)}"
        )

    window_ids = np.repeat(np.arange(n_windows), n_patches)
    patch_ids = np.tile(np.arange(n_patches), n_windows)
    uncertainty = np.empty((n_windows, n_patches), dtype=np.float64)

    print(
        f"[mc dropout gain screen] S{args.subject:02d} validation={n_windows} "
        f"candidates={len(window_ids)} mc_samples={args.mc_samples} "
        f"batch={batch_size} test_unused={len(test_idx)}"
    )
    model.train()
    with torch.inference_mode():
        for start in range(0, len(window_ids), batch_size):
            stop = min(start + batch_size, len(window_ids))
            batch_window_ids = window_ids[start:stop]
            batch_patch_ids = patch_ids[start:stop]
            raw_np = windows[batch_window_ids]
            mask_np = np.ones_like(raw_np, dtype=np.float32)
            for row, patch_id in enumerate(batch_patch_ids):
                patch_start = patch_id * patch_size
                mask_np[row, patch_start:patch_start + patch_size] = 0.0
            raw = torch.as_tensor(raw_np, dtype=torch.float32, device=device)
            mask = torch.as_tensor(mask_np, dtype=torch.float32, device=device)
            channel_valid = (mask.mean(dim=1) > 0.5).float()
            predictions = []
            for _ in range(args.mc_samples):
                predictions.append(
                    model(
                        raw * mask,
                        raw_time_mask=mask,
                        chan_valid_mask=channel_valid,
                    )
                )
            std = torch.stack(predictions, dim=0).std(dim=0, unbiased=False)
            std_np = std.cpu().numpy()
            for row, patch_id in enumerate(batch_patch_ids):
                patch_start = patch_id * patch_size
                uncertainty[batch_window_ids[row], patch_id] = float(
                    std_np[row, patch_start:patch_start + patch_size].mean()
                )
    model.eval()

    confidence = -uncertainty
    global_spearman = spearman(confidence.reshape(-1), gain.reshape(-1))
    window_spearman = np.asarray(
        [spearman(confidence[row], gain[row]) for row in range(n_windows)],
        dtype=np.float64,
    )
    finite = np.isfinite(window_spearman)
    report = {
        "scope": "DB3 S05 validation-only MC-dropout uncertainty versus existing Oracle gains",
        "test_policy": "test repetitions 2/5 are not predicted, scored, or used",
        "feature_policy": (
            "MC-dropout uncertainty uses only raw EMG, a fixed all-channel single-time-patch "
            "mask, and frozen MCIA weights. Glove labels are only embedded in the precomputed "
            "Oracle gain used afterward for association."
        ),
        "subject_id": args.subject,
        "reused": {
            "oracle_gain": str(oracle_path),
            "healthy_prior_mcia": str(checkpoint),
        },
        "candidate_constraint": {
            "candidate_unit": "one all-channel time patch",
            "patch_size_samples": patch_size,
            "patch_duration_ms": 1000.0 * patch_size / float(config["target_fs"]),
            "all_patch_positions_per_window": n_patches,
            "max_patches_per_window": 1,
        },
        "mc_dropout": {
            "samples": args.mc_samples,
            "model_mode": "train for dropout only, inference_mode with no gradients or optimizer",
        },
        "association_rule": "global Spearman of negative uncertainty versus Oracle gain greater than 0.1 is a pass",
        "global_spearman_negative_uncertainty_vs_gain": global_spearman,
        "window_spearman": {
            "finite_window_count": int(finite.sum()),
            "mean": float(window_spearman[finite].mean()) if finite.any() else float("nan"),
            "median": float(np.median(window_spearman[finite])) if finite.any() else float("nan"),
            "positive_fraction": float((window_spearman[finite] > 0.0).mean()) if finite.any() else float("nan"),
        },
        "top_confidence_gain": top_gain_summary(confidence.reshape(-1), gain.reshape(-1)),
        "pass": bool(np.isfinite(global_spearman) and global_spearman > 0.1),
        "interpretation_limit": (
            "A pass is only a S05 validation association. It requires independent S06 validation "
            "confirmation before any gate calibration or untouched test evaluation."
        ),
    }
    out = run_dir / "06_diagnostics" / "db3_mc_dropout_oracle_gain"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    out_file = out / "metrics" / f"S{args.subject:02d}_mc_dropout_oracle_gain.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / f"S{args.subject:02d}_mc_dropout_oracle_gain.npz",
        validation_indices=validation_indices,
        uncertainty=uncertainty,
        oracle_gain=gain,
        window_spearman=window_spearman,
    )
    print(
        f"  global Spearman(-uncertainty, gain)={global_spearman:+.4f} "
        f"window_mean={report['window_spearman']['mean']:+.4f} "
        f"window_median={report['window_spearman']['median']:+.4f} "
        f"positive_windows={report['window_spearman']['positive_fraction']:.3f} "
        f"pass={report['pass']}"
    )
    print(f"[diagnostic] saved: {out_file}")


if __name__ == "__main__":
    main()
