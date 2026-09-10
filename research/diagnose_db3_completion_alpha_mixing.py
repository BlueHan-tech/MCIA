"""Train-split-only screen for conservative MCIA alpha mixing."""
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
    spec = importlib.util.spec_from_file_location("alpha_mixing_exp3", path)
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
def raw_window_mse(model, windows, targets, device, batch_size):
    result = np.empty(len(windows), dtype=np.float64)
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        x = torch.as_tensor(windows[start:stop], dtype=torch.float32, device=device)
        y = torch.as_tensor(targets[start:stop], dtype=torch.float32, device=device)
        pred = model(x)
        result[start:stop] = ((pred - y) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return result


def parse_alphas(text):
    values = [float(item) for item in text.split(",")]
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise ValueError("alphas must be in (0, 1]")
    return sorted(set(values))


def alpha_report(alpha, mse_by_candidate, raw_mse):
    candidate_rmse = np.sqrt(mse_by_candidate)
    raw_rmse = np.sqrt(raw_mse)[:, None]
    best_mse = mse_by_candidate.min(axis=1)
    return {
        "alpha": alpha,
        "all_candidates_global_rmse": float(np.sqrt(mse_by_candidate.mean())),
        "all_candidates_mean_window_rmse": float(candidate_rmse.mean()),
        "all_candidates_mean_rmse_gain": float((raw_rmse - candidate_rmse).mean()),
        "all_candidates_positive_gain_fraction": float((raw_rmse > candidate_rmse).mean()),
        "oracle_one_patch_per_window_global_rmse": float(np.sqrt(best_mse.mean())),
        "oracle_one_patch_per_window_global_rmse_gain": float(
            np.sqrt(raw_mse.mean()) - np.sqrt(best_mse.mean())
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="DB3 train-split-only conservative MCIA alpha-mixing screen"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.5,0.7,1.0")
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=0,
        help="0 uses every repetition-1/3/4/6 training window",
    )
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    alphas = parse_alphas(args.alphas)
    run_dir = Path(args.run_dir).resolve()
    env_run_dir = os.environ.get("MCIA_RUN_DIR")
    if not env_run_dir or Path(env_run_dir).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    device = config["device"]
    batch_size = args.batch_size or int(config["regressor_batch_size"])
    exp3 = load_exp3()
    tcn_checkpoint = (
        run_dir
        / "06_diagnostics"
        / "db3_validation_error_windows"
        / "checkpoints"
        / f"S{args.subject:02d}_raw_validation_best.pth"
    )
    if not tcn_checkpoint.exists():
        raise FileNotFoundError(tcn_checkpoint)
    mcia, mcia_checkpoint = exp3.load_healthy_prior_mcia(config, device)
    if mcia is None:
        raise FileNotFoundError("healthy-prior MCIA checkpoint")

    loader = NinaProDataLoader(
        config["db2_path"], config["db3_path"], fs=config["orig_fs"]
    )
    emg, angle, _, repetitions = prepare_kinematics_data(
        loader, [args.subject], config, exercises=[args.exercise], db="db3"
    )
    train_idx, val_idx, test_idx = make_rep_split(
        repetitions,
        train_reps=(1, 3, 4, 6),
        test_reps=(2, 5),
        val_ratio=float(config["regressor_val_ratio"]),
        seed=int(config["regressor_random_seed"]) + args.subject,
    )
    if args.max_train_windows:
        train_idx = train_idx[: args.max_train_windows]
    windows, targets = emg[train_idx], angle[train_idx]
    n_windows, n_steps, _ = windows.shape
    patch_size = int(config["patch_size"])
    if n_steps % patch_size:
        raise ValueError("window length must be divisible by patch_size")
    n_patches = n_steps // patch_size
    tcn = load_tcn(exp3, tcn_checkpoint, config, windows.shape[-1], device)
    raw_mse = raw_window_mse(tcn, windows, targets, device, batch_size)

    n_candidates = n_windows * n_patches
    window_ids = np.repeat(np.arange(n_windows), n_patches)
    patch_ids = np.tile(np.arange(n_patches), n_windows)
    candidate_mse = {
        alpha: np.empty((n_windows, n_patches), dtype=np.float64) for alpha in alphas
    }

    print(
        f"[alpha mixing] S{args.subject:02d} train={n_windows} "
        f"candidates={n_candidates} patches_per_window={n_patches} "
        f"val_unused={len(val_idx)} test_unused={len(test_idx)}"
    )
    with torch.no_grad():
        for start in range(0, n_candidates, batch_size):
            stop = min(start + batch_size, n_candidates)
            batch_window_ids = window_ids[start:stop]
            batch_patch_ids = patch_ids[start:stop]
            raw_np = windows[batch_window_ids]
            target_np = targets[batch_window_ids]
            masks_np = np.ones_like(raw_np, dtype=np.float32)
            for row, patch_id in enumerate(batch_patch_ids):
                patch_start = patch_id * patch_size
                masks_np[row, patch_start:patch_start + patch_size] = 0.0
            raw_t = torch.as_tensor(raw_np, dtype=torch.float32, device=device)
            target_t = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(masks_np, dtype=torch.float32, device=device)
            channel_valid = (mask_t.mean(dim=1) > 0.5).float()
            completed_t = mcia(
                raw_t * mask_t,
                raw_time_mask=mask_t,
                chan_valid_mask=channel_valid,
            )
            completed_t = completed_t * (1.0 - mask_t) + raw_t * mask_t
            difference_t = completed_t - raw_t
            for alpha in alphas:
                pred_t = tcn(raw_t + alpha * difference_t)
                mse = ((pred_t - target_t) ** 2).mean(dim=(1, 2)).cpu().numpy()
                candidate_mse[alpha][batch_window_ids, batch_patch_ids] = mse

    rows = [alpha_report(alpha, candidate_mse[alpha], raw_mse) for alpha in alphas]
    rows.sort(key=lambda item: item["all_candidates_global_rmse"])
    best_all_candidates = rows[0]["alpha"]
    best_oracle = min(
        rows, key=lambda item: item["oracle_one_patch_per_window_global_rmse"]
    )["alpha"]
    report = {
        "scope": "train-split-only mechanism screen; not a validation or test result",
        "test_policy": "test repetitions 2/5 are not predicted, scored, or used",
        "validation_policy": (
            "validation windows are not predicted or scored; the frozen checkpoint was "
            "previously selected by validation and is reused unchanged"
        ),
        "subject_id": args.subject,
        "split": {
            "train_used": int(n_windows),
            "validation_unused": int(len(val_idx)),
            "test_unused": int(len(test_idx)),
        },
        "candidate_constraint": {
            "candidate_unit": "one all-channel time patch",
            "patch_size_samples": patch_size,
            "patch_duration_ms": 1000.0 * patch_size / float(config["target_fs"]),
            "all_patch_positions_per_window": n_patches,
            "max_patches_per_window": 1,
        },
        "mixing_rule": "enhanced = raw + alpha * (mcia_completed - raw)",
        "reused": {
            "raw_tcn": str(tcn_checkpoint),
            "healthy_prior_mcia": str(mcia_checkpoint),
        },
        "raw": {
            "global_rmse": float(np.sqrt(raw_mse.mean())),
            "mean_window_rmse": float(np.sqrt(raw_mse).mean()),
        },
        "alpha_results": rows,
        "best_alpha_by_all_candidates_global_rmse": best_all_candidates,
        "best_alpha_by_oracle_one_patch_global_rmse": best_oracle,
        "interpretation_limit": (
            "This screen changes only the MCIA replacement amplitude for every "
            "enumerated training candidate. It does not establish a deployable mask "
            "selector or a generalization result."
        ),
    }
    out = run_dir / "06_diagnostics" / "db3_completion_alpha_mixing"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    out_file = out / "metrics" / f"S{args.subject:02d}_alpha_mixing.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  raw global_rmse={report['raw']['global_rmse']:.6f}")
    for row in rows:
        print(
            f"  alpha={row['alpha']:.1f} all_rmse={row['all_candidates_global_rmse']:.6f} "
            f"all_gain={row['all_candidates_mean_rmse_gain']:+.6f} "
            f"oracle_gain={row['oracle_one_patch_per_window_global_rmse_gain']:+.6f}"
        )
    print(f"[diagnostic] saved: {out_file}")


if __name__ == "__main__":
    main()
