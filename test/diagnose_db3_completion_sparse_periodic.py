"""Validation-only fixed periodic MCIA completion screen."""
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
    spec = importlib.util.spec_from_file_location("sparse_periodic_exp3", path)
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
def predict(model, windows, targets, device, batch_size):
    result = np.empty_like(targets)
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        x = torch.as_tensor(windows[start:stop], dtype=torch.float32, device=device)
        result[start:stop] = model(x).cpu().numpy()
    return result


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


def parse_values(text, cast):
    values = [cast(item) for item in text.split(",")]
    if not values:
        raise ValueError("values must not be empty")
    return values


def periodic_mask(windows, patch_size, interval, offset):
    _, n_steps, _ = windows.shape
    n_patches = n_steps // patch_size
    mask = np.ones_like(windows, dtype=np.float32)
    ids = list(range(offset, n_patches, interval))
    for patch_id in ids:
        start = patch_id * patch_size
        mask[:, start:start + patch_size] = 0.0
    return mask, ids


def linear_interpolation(raw, mask):
    result = raw.copy()
    n_steps = raw.shape[1]
    time = np.arange(n_steps)
    for window_id in range(len(raw)):
        observed = mask[window_id, :, 0] > 0.5
        if observed.sum() < 2:
            raise ValueError("periodic interpolation needs at least two observed samples")
        observed_time = time[observed]
        for channel_id in range(raw.shape[2]):
            interpolated = np.interp(
                time, observed_time, raw[window_id, observed, channel_id]
            )
            result[window_id, ~observed, channel_id] = interpolated[~observed]
    return result


def global_metrics(pred, target):
    mse = np.mean((pred - target) ** 2)
    window_mse = np.mean((pred - target) ** 2, axis=(1, 2))
    return {
        "global_rmse": float(np.sqrt(mse)),
        "mean_window_rmse": float(np.sqrt(window_mse).mean()),
    }


def offset_summary(rows):
    values = np.array([row["global_rmse"] for row in rows], dtype=np.float64)
    return {
        "offset_results": rows,
        "mean_global_rmse": float(values.mean()),
        "std_global_rmse": float(values.std(ddof=0)),
        "min_global_rmse_not_for_selection": float(values.min()),
        "max_global_rmse_not_for_selection": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="DB3 validation-only fixed periodic completion screen"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--offsets", default="0,1,2,3")
    parser.add_argument("--alphas", default="0.02,0.05,0.1,1.0")
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    offsets = parse_values(args.offsets, int)
    alphas = parse_values(args.alphas, float)
    if args.interval < 2 or any(offset < 0 or offset >= args.interval for offset in offsets):
        raise ValueError("offsets must be in [0, interval)")
    if any(alpha <= 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("alphas must be in (0, 1]")

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
    _, val_idx, test_idx = make_rep_split(
        repetitions,
        train_reps=(1, 3, 4, 6),
        test_reps=(2, 5),
        val_ratio=float(config["regressor_val_ratio"]),
        seed=int(config["regressor_random_seed"]) + args.subject,
    )
    windows, targets = emg[val_idx], angle[val_idx]
    patch_size = int(config["patch_size"])
    if windows.shape[1] % patch_size:
        raise ValueError("window length must be divisible by patch_size")
    n_patches = windows.shape[1] // patch_size
    if any(offset >= n_patches for offset in offsets):
        raise ValueError("offset must be smaller than patches per window")

    tcn = load_tcn(exp3, tcn_checkpoint, config, windows.shape[-1], device)
    baseline = global_metrics(predict(tcn, windows, targets, device, batch_size), targets)
    per_mode = {"mcia": {}, "zero_replacement": {}, "linear_interpolation": {}}
    print(
        f"[sparse periodic] S{args.subject:02d} validation={len(windows)} "
        f"interval={args.interval} offsets={offsets} patches={n_patches} "
        f"test_unused={len(test_idx)}"
    )
    print(f"  baseline global_rmse={baseline['global_rmse']:.6f}")

    for offset in offsets:
        mask, masked_ids = periodic_mask(
            windows, patch_size, args.interval, offset
        )
        mcia_completed = complete(mcia, windows, mask, device, batch_size)
        zero_completed = windows * mask
        linear_completed = linear_interpolation(windows, mask)
        completed_by_mode = {
            "mcia": mcia_completed,
            "zero_replacement": zero_completed,
            "linear_interpolation": linear_completed,
        }
        print(
            f"  offset={offset} masked={masked_ids} "
            f"mask_fraction={1.0 - float(mask.mean()):.3f}"
        )
        for mode, completed in completed_by_mode.items():
            for alpha in alphas:
                mixed = windows + alpha * (completed - windows)
                metrics = global_metrics(
                    predict(tcn, mixed, targets, device, batch_size), targets
                )
                metrics["offset"] = offset
                metrics["masked_patch_ids"] = masked_ids
                key = f"{alpha:.6g}"
                per_mode[mode].setdefault(key, []).append(metrics)

    summarized = {
        mode: {alpha: offset_summary(rows) for alpha, rows in by_alpha.items()}
        for mode, by_alpha in per_mode.items()
    }
    mcia_effective_alphas = [
        float(alpha)
        for alpha, result in summarized["mcia"].items()
        if result["mean_global_rmse"] < baseline["global_rmse"]
    ]
    report = {
        "scope": "validation-only fixed periodic sparse completion screen",
        "test_policy": "test repetitions 2/5 are not predicted, scored, or used",
        "selection_policy": (
            "all requested offsets are reported as a mean and standard deviation; "
            "the best offset is explicitly not selected"
        ),
        "subject_id": args.subject,
        "split": {
            "validation_used": int(len(windows)),
            "test_unused": int(len(test_idx)),
        },
        "periodic_mask": {
            "patch_size_samples": patch_size,
            "patch_duration_ms": 1000.0 * patch_size / float(config["target_fs"]),
            "patches_per_window": n_patches,
            "interval": args.interval,
            "offsets": offsets,
            "masked_patch_count_per_offset": len(range(0, n_patches, args.interval)),
        },
        "mixing_rule": "enhanced = raw + alpha * (replacement - raw)",
        "reused": {
            "raw_tcn": str(tcn_checkpoint),
            "healthy_prior_mcia": str(mcia_checkpoint),
        },
        "baseline": baseline,
        "methods": summarized,
        "mcia_alphas_beating_baseline_by_offset_mean": mcia_effective_alphas,
        "control_interpretation": (
            "zero replacement tests whether MCIA is better than direct signal removal. "
            "Linear interpolation tests whether any improvement is specific to MCIA "
            "rather than a simple local smoothing replacement."
        ),
        "interpretation_limit": (
            "This is a validation-only fixed-rule screen, not a test result. "
            "A passing rule requires independent S06 validation confirmation before "
            "any untouched test evaluation."
        ),
    }
    out = run_dir / "06_diagnostics" / "db3_completion_sparse_periodic"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    out_file = out / "metrics" / f"S{args.subject:02d}_sparse_periodic.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, by_alpha in summarized.items():
        for alpha, result in sorted(by_alpha.items(), key=lambda item: float(item[0])):
            print(
                f"  {mode} alpha={alpha} mean_rmse={result['mean_global_rmse']:.6f} "
                f"std={result['std_global_rmse']:.6f}"
            )
    print(f"[diagnostic] saved: {out_file}")


if __name__ == "__main__":
    main()
