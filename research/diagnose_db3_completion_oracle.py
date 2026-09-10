"""Validation-only oracle upper bound for fixed-budget MCIA patch completion."""
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
    spec = importlib.util.spec_from_file_location("oracle_exp3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tcn(exp3, checkpoint, config, channels, device):
    model = exp3.build_tcn(config, channels, device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    return model.eval()


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


def per_window_rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))


def per_window_mse(pred, target):
    return np.mean((pred - target) ** 2, axis=(1, 2))


def main():
    parser = argparse.ArgumentParser(description="DB3 validation-only MCIA oracle upper bound")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--max-val-windows", type=int, default=0, help="0 uses all validation windows")
    parser.add_argument("--fractions", default="0.10,0.25,0.50,1.00")
    args = parser.parse_args()
    fractions = [float(value) for value in args.fractions.split(",")]
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("fractions must be in (0, 1]")
    run_dir = Path(args.run_dir).resolve()
    if not os.environ.get("MCIA_RUN_DIR") or Path(os.environ["MCIA_RUN_DIR"]).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    device = config["device"]
    exp3 = load_exp3()
    checkpoint = run_dir / "06_diagnostics" / "db3_validation_error_windows" / "checkpoints" / f"S{args.subject:02d}_raw_validation_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    mcia, mcia_checkpoint = exp3.load_healthy_prior_mcia(config, device)
    if mcia is None:
        raise FileNotFoundError("healthy-prior MCIA checkpoint")
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    emg, angle, _, repetitions = prepare_kinematics_data(loader, [args.subject], config, exercises=[args.exercise], db="db3")
    _, val_idx, test_idx = make_rep_split(repetitions, train_reps=(1, 3, 4, 6), test_reps=(2, 5), val_ratio=float(config["regressor_val_ratio"]), seed=int(config["regressor_random_seed"]) + args.subject)
    if args.max_val_windows:
        val_idx = val_idx[:args.max_val_windows]
    windows, target = emg[val_idx], angle[val_idx]
    patch_size = int(config["patch_size"])
    if windows.shape[1] % patch_size:
        raise ValueError("window length is not divisible by patch size")
    n_patches = windows.shape[1] // patch_size
    tcn = load_tcn(exp3, checkpoint, config, windows.shape[-1], device)
    raw_pred, raw_target = exp3.predict_on_set(tcn, windows, target, np.arange(len(windows)), config, device)
    raw_rmse = per_window_rmse(raw_pred, raw_target)
    raw_mse = per_window_mse(raw_pred, raw_target)

    window_ids = np.repeat(np.arange(len(windows)), n_patches)
    patch_ids = np.tile(np.arange(n_patches), len(windows))
    candidates = windows[window_ids]
    masks = np.ones_like(candidates, dtype=np.float32)
    for index, patch_id in enumerate(patch_ids):
        masks[index, patch_id * patch_size:(patch_id + 1) * patch_size] = 0.0
    print(f"[oracle] S{args.subject:02d} val={len(windows)} candidates={len(candidates)} patches_per_window={n_patches} test_unused={len(test_idx)}")
    completed = complete(mcia, candidates, masks, device, int(config["regressor_batch_size"]))
    repeated_target = target[window_ids]
    completed_pred, completed_target = exp3.predict_on_set(tcn, completed, repeated_target, np.arange(len(completed)), config, device)
    if not np.allclose(completed_target, repeated_target):
        raise RuntimeError("candidate target order changed")
    candidate_rmse = per_window_rmse(completed_pred, completed_target).reshape(len(windows), n_patches)
    candidate_mse = per_window_mse(completed_pred, completed_target).reshape(len(windows), n_patches)
    gain = raw_rmse[:, None] - candidate_rmse
    best_patch = np.argmax(gain, axis=1)
    best_gain = gain[np.arange(len(windows)), best_patch]
    best_mse = candidate_mse[np.arange(len(windows)), best_patch]
    order = np.argsort(best_gain)[::-1]
    budgets = []
    for fraction in sorted(set(fractions)):
        count = max(1, int(np.ceil(fraction * len(windows))))
        selected = order[:count]
        oracle_mse = raw_mse.copy()
        oracle_mse[selected] = best_mse[selected]
        item = {
            "window_fraction": fraction,
            "selected_windows": int(count),
            "masked_emg_fraction": float(count * patch_size / (len(windows) * windows.shape[1])),
            "selected_best_patch_gain_mean": float(best_gain[selected].mean()),
            "mean_window_rmse": float(np.mean(np.sqrt(oracle_mse))),
            "global_rmse": float(np.sqrt(oracle_mse.mean())),
            "global_rmse_gain": float(np.sqrt(raw_mse.mean()) - np.sqrt(oracle_mse.mean())),
        }
        budgets.append(item)
        print(f"  budget={fraction:.0%} windows={count} global_rmse={item['global_rmse']:.6f} gain={item['global_rmse_gain']:+.6f}")
    report = {
        "scope": "validation-only oracle upper bound; no deployable mask selector",
        "test_policy": "test repetitions 2/5 not predicted, scored, or used for oracle selection",
        "subject_id": args.subject,
        "split": {"validation_used": int(len(windows)), "test_unused": int(len(test_idx))},
        "reused": {"raw_tcn": str(checkpoint), "healthy_prior_mcia": str(mcia_checkpoint)},
        "oracle_constraint": {"candidate_unit": "one all-channel time patch", "patch_size_samples": patch_size, "patch_duration_ms": 1000.0 * patch_size / float(config["target_fs"]), "all_patch_positions_per_window": n_patches, "max_patches_per_window": 1, "selection_uses": "validation angle labels and is oracle-only"},
        "raw": {"mean_window_rmse": float(raw_rmse.mean()), "global_rmse": float(np.sqrt(raw_mse.mean()))},
        "candidate_gain": {"mean": float(gain.mean()), "positive_fraction": float((gain > 0.0).mean()), "best_per_window_mean": float(best_gain.mean()), "best_per_window_positive_fraction": float((best_gain > 0.0).mean())},
        "budgets": budgets,
        "interpretation_limit": "This is only an upper bound for one 40 ms all-channel patch per window under the frozen MCIA and TCN. It cannot be used as a test metric or as evidence of a deployable selector.",
    }
    out = run_dir / "06_diagnostics" / "db3_completion_oracle"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "metrics" / f"S{args.subject:02d}_oracle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / "predictions" / f"S{args.subject:02d}_oracle.npz", validation_indices=val_idx, best_patch=best_patch, best_gain=best_gain, candidate_gain=gain, raw_rmse=raw_rmse, raw_mse=raw_mse, best_mse=best_mse)
    print(f"[diagnostic] saved: {out}")


if __name__ == "__main__":
    main()