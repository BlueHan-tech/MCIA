"""Validation-only screen of MCIA-derived features against fixed S05 oracle gains."""
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
    spec = importlib.util.spec_from_file_location("oracle_feature_exp3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def complete(mcia, windows, masks, device, batch_size):
    result = np.empty_like(windows)
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        x = torch.as_tensor(windows[start:stop], dtype=torch.float32, device=device)
        mask = torch.as_tensor(masks[start:stop], dtype=torch.float32, device=device)
        valid = (mask.mean(dim=1) > 0.5).float()
        pred = mcia(x * mask, raw_time_mask=mask, chan_valid_mask=valid)
        result[start:stop] = (pred * (1.0 - mask) + x * mask).cpu().numpy()
    return result


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
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def association(name, values, gain):
    count = max(1, int(np.ceil(0.10 * len(values))))
    top = np.argsort(values)[-count:]
    return {
        "feature": name,
        "spearman_with_oracle_gain": spearman(values, gain),
        "top_10pct_gain_mean": float(gain[top].mean()),
        "all_gain_mean": float(gain.mean()),
        "top_10pct_positive_gain_fraction": float((gain[top] > 0.0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="DB3 validation-only MCIA oracle-feature screen")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not os.environ.get("MCIA_RUN_DIR") or Path(os.environ["MCIA_RUN_DIR"]).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")

    oracle_path = run_dir / "06_diagnostics" / "db3_completion_oracle" / "predictions" / f"S{args.subject:02d}_oracle.npz"
    if not oracle_path.exists():
        raise FileNotFoundError(f"Run the corresponding oracle diagnostic first: {oracle_path}")
    oracle = np.load(oracle_path)
    validation_indices = oracle["validation_indices"]
    oracle_gain = oracle["candidate_gain"].reshape(-1)

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    exp3 = load_exp3()
    mcia, mcia_checkpoint = exp3.load_healthy_prior_mcia(config, config["device"])
    if mcia is None:
        raise FileNotFoundError("healthy-prior MCIA checkpoint")
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    emg, _, _, repetitions = prepare_kinematics_data(loader, [args.subject], config, exercises=[args.exercise], db="db3")
    train_idx, _, test_idx = make_rep_split(repetitions, train_reps=(1, 3, 4, 6), test_reps=(2, 5), val_ratio=float(config["regressor_val_ratio"]), seed=int(config["regressor_random_seed"]) + args.subject)
    windows = emg[validation_indices]
    patch_size = int(config["patch_size"])
    n_windows, n_steps, n_channels = windows.shape
    n_patches = n_steps // patch_size
    if n_steps % patch_size:
        raise ValueError("window length is not divisible by patch size")
    if len(oracle_gain) != n_windows * n_patches:
        raise ValueError("oracle gain shape does not match validation windows")

    window_ids = np.repeat(np.arange(n_windows), n_patches)
    patch_ids = np.tile(np.arange(n_patches), n_windows)
    candidates = windows[window_ids]
    masks = np.ones_like(candidates, dtype=np.float32)
    for index, patch_id in enumerate(patch_ids):
        masks[index, patch_id * patch_size:(patch_id + 1) * patch_size] = 0.0
    print(f"[oracle feature screen] S{args.subject:02d} candidates={len(candidates)} test_unused={len(test_idx)}")
    completed = complete(mcia, candidates, masks, config["device"], int(config["regressor_batch_size"]))
    missing = masks < 0.5
    difference = np.abs(completed - candidates)
    completion_change = (difference * missing).sum(axis=(1, 2)) / missing.sum(axis=(1, 2))
    train_center = np.median(emg[train_idx], axis=(0, 1))
    train_mad = np.median(np.abs(emg[train_idx] - train_center[None, None, :]), axis=(0, 1))
    train_scale = np.maximum(1.4826 * train_mad, 1e-6)
    patch_mean = np.empty((len(candidates), n_channels), dtype=np.float64)
    for index, patch_id in enumerate(patch_ids):
        start = patch_id * patch_size
        patch_mean[index] = candidates[index, start:start + patch_size].mean(axis=0)
    channel_z = (patch_mean - train_center[None, :]) / train_scale[None, :]
    neighbor_inconsistency = np.abs(np.diff(channel_z, axis=1)).mean(axis=1)
    channel_dispersion = np.abs(channel_z - channel_z.mean(axis=1, keepdims=True)).mean(axis=1)

    rows = [
        association("mcia_completion_change_mean", completion_change, oracle_gain),
        association("neighbor_channel_inconsistency", neighbor_inconsistency, oracle_gain),
        association("cross_channel_dispersion", channel_dispersion, oracle_gain),
    ]
    rows.sort(key=lambda row: abs(row["spearman_with_oracle_gain"] or 0.0), reverse=True)
    report = {
        "scope": "validation-only feature screen against precomputed oracle gains",
        "test_policy": "test repetitions 2/5 are not predicted, scored, or used for feature screening",
        "subject_id": args.subject,
        "reused": {"oracle_gain": str(oracle_path), "healthy_prior_mcia": str(mcia_checkpoint)},
        "feature_definitions": {
            "mcia_completion_change_mean": "mean absolute MCIA replacement change inside the masked 40 ms patch",
            "neighbor_channel_inconsistency": "mean adjacent-channel difference after train-only robust per-channel normalization",
            "cross_channel_dispersion": "mean absolute deviation from the simultaneous cross-channel normalized mean",
        },
        "candidate_count": int(len(candidates)),
        "association_rule": "absolute Spearman correlation greater than 0.1 is a feature-screen pass",
        "associations": rows,
        "note": "MCIA dropout uncertainty is intentionally excluded from this first screen.",
    }
    out = run_dir / "06_diagnostics" / "db3_oracle_feature_screen"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "metrics" / f"S{args.subject:02d}_feature_screen.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(f"  {row['feature']}: spearman={row['spearman_with_oracle_gain']:+.4f} top10_gain={row['top_10pct_gain_mean']:+.6f}")
    print(f"[diagnostic] saved: {out}")


if __name__ == "__main__":
    main()