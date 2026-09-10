"""Small validation-only feasibility test for EMG-only completion benefit prediction."""
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
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config


def exp3_module():
    path = ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("gain_exp3", path)
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


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))


def layout(n_windows, n_patches, stride):
    positions = np.arange(0, n_patches, stride, dtype=np.int64)
    return np.repeat(np.arange(n_windows), len(positions)), np.tile(positions, n_windows)


def masks_for(patch_ids, time_steps, channels, patch_size):
    masks = np.ones((len(patch_ids), time_steps, channels), dtype=np.float32)
    for index, patch in enumerate(patch_ids):
        masks[index, patch * patch_size:(patch + 1) * patch_size] = 0.0
    return masks


def features(windows, window_ids, patch_ids, patch_size):
    out = []
    for window_id, patch_id in zip(window_ids, patch_ids):
        window = windows[window_id]
        start = patch_id * patch_size
        patch = window[start:start + patch_size]
        context = np.concatenate((window[max(0, start - patch_size):start], window[start + patch_size:start + 2 * patch_size]))
        if not len(context):
            context = window
        means, stds = patch.mean(axis=0), patch.std(axis=0)
        out.append((
            patch.mean(), patch.std(), np.abs(np.diff(patch, axis=0)).mean(),
            np.abs(means - context.mean(axis=0)).mean(), (patch <= 1e-6).mean(),
            means.min(), np.median(means), stds.min(), np.median(stds),
            (patch.mean() - window.mean()) / max(window.std(), 1e-6),
            patch_id / max((len(window) // patch_size) - 1, 1),
        ))
    return np.asarray(out, dtype=np.float64)


def gains(exp3, tcn, mcia, windows, targets, window_ids, patch_ids, patch_size, config, device):
    index = np.arange(len(windows))
    raw_pred, raw_target = exp3.predict_on_set(tcn, windows, targets, index, config, device)
    raw_error = rmse(raw_pred, raw_target)
    candidate_windows = windows[window_ids]
    candidate_targets = targets[window_ids]
    completed = complete(mcia, candidate_windows, masks_for(patch_ids, windows.shape[1], windows.shape[2], patch_size), device, int(config["regressor_batch_size"]))
    completed_pred, completed_target = exp3.predict_on_set(tcn, completed, candidate_targets, np.arange(len(completed)), config, device)
    if not np.allclose(completed_target, candidate_targets):
        raise RuntimeError("candidate target order changed")
    return raw_error[window_ids] - rmse(completed_pred, completed_target), raw_pred


def main():
    parser = argparse.ArgumentParser(description="S05 validation-only completion gain feasibility")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--train-windows", type=int, default=64)
    parser.add_argument("--val-windows", type=int, default=48)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--selection-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if not 0 < args.selection_fraction <= 0.5 or args.patch_stride <= 0:
        raise ValueError("invalid selection fraction or patch stride")
    run_dir = Path(args.run_dir).resolve()
    if not os.environ.get("MCIA_RUN_DIR") or Path(os.environ["MCIA_RUN_DIR"]).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    device = config["device"]
    exp3 = exp3_module()
    checkpoint = run_dir / "06_diagnostics" / "db3_validation_error_windows" / "checkpoints" / f"S{args.subject:02d}_raw_validation_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    mcia, mcia_checkpoint = exp3.load_healthy_prior_mcia(config, device)
    if mcia is None:
        raise FileNotFoundError("healthy-prior MCIA checkpoint")
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    emg, angle, _, repetitions = prepare_kinematics_data(loader, [args.subject], config, exercises=[args.exercise], db="db3")
    train_idx, val_idx, test_idx = make_rep_split(repetitions, train_reps=(1, 3, 4, 6), test_reps=(2, 5), val_ratio=float(config["regressor_val_ratio"]), seed=int(config["regressor_random_seed"]) + args.subject)
    rng = np.random.default_rng(args.seed + args.subject)
    train_source = np.sort(rng.choice(train_idx, args.train_windows, replace=False))
    val_source = np.sort(rng.choice(val_idx, args.val_windows, replace=False))
    train_x, train_y = emg[train_source], angle[train_source]
    val_x, val_y = emg[val_source], angle[val_source]
    patch_size = int(config["patch_size"])
    n_patches = train_x.shape[1] // patch_size
    if train_x.shape[1] % patch_size:
        raise ValueError("window length is not divisible by patch size")
    train_windows, train_patches = layout(len(train_x), n_patches, args.patch_stride)
    val_windows, val_patches = layout(len(val_x), n_patches, args.patch_stride)
    tcn = load_tcn(exp3, checkpoint, config, emg.shape[-1], device)
    print(f"[gain feasibility] S{args.subject:02d} train={len(train_x)} val={len(val_x)} candidates={len(train_windows)}/{len(val_windows)} test_unused={len(test_idx)}")
    train_gain, _ = gains(exp3, tcn, mcia, train_x, train_y, train_windows, train_patches, patch_size, config, device)
    head = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(features(train_x, train_windows, train_patches, patch_size), train_gain)
    val_gain, raw_val_pred = gains(exp3, tcn, mcia, val_x, val_y, val_windows, val_patches, patch_size, config, device)
    val_score = head.predict(features(val_x, val_windows, val_patches, patch_size))
    positive = val_gain > 0.0
    auc = float(roc_auc_score(positive, val_score)) if len(np.unique(positive)) == 2 else None
    n_select = max(1, int(np.ceil(args.selection_fraction * len(val_score))))
    selected = np.argsort(val_score)[-n_select:]
    selected_masks = np.ones_like(val_x, dtype=np.float32)
    for candidate in selected:
        start = val_patches[candidate] * patch_size
        selected_masks[val_windows[candidate], start:start + patch_size] = 0.0
    completed_x = complete(mcia, val_x, selected_masks, device, int(config["regressor_batch_size"]))
    completed_pred, completed_target = exp3.predict_on_set(tcn, completed_x, val_y, np.arange(len(val_x)), config, device)
    raw_mean = float(rmse(raw_val_pred, val_y).mean())
    completed_mean = float(rmse(completed_pred, completed_target).mean())
    pearson = None if np.std(val_score) <= 1e-12 or np.std(val_gain) <= 1e-12 else float(np.corrcoef(val_score, val_gain)[0, 1])
    report = {
        "scope": "bounded validation-only EMG-only completion benefit feasibility",
        "test_policy": "test repetitions 2/5 not predicted, scored, or used for selection",
        "subject_id": args.subject,
        "split": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test_unused": int(len(test_idx))},
        "reused": {"raw_tcn": str(checkpoint), "healthy_prior_mcia": str(mcia_checkpoint)},
        "candidate": {"unit": "all-channel time patch", "patch_size_samples": patch_size, "duration_ms": 1000.0 * patch_size / float(config["target_fs"]), "gain": "raw RMSE minus completed RMSE; positive is beneficial", "head_input": "raw EMG patch and context only", "head": "standardized Ridge"},
        "sample": {"train_windows": len(train_x), "val_windows": len(val_x), "train_candidates": len(train_gain), "val_candidates": len(val_gain)},
        "validation_ranking": {"pearson": pearson, "positive_gain_auc": auc, "all_gain_mean": float(val_gain.mean()), "positive_fraction": float(positive.mean()), "selected_count": n_select, "selected_actual_gain_mean": float(val_gain[selected].mean()), "selected_positive_fraction": float((val_gain[selected] > 0).mean())},
        "validation_combined": {"masked_fraction": float((selected_masks < 0.5).mean()), "raw_rmse": raw_mean, "completed_rmse": completed_mean, "gain": raw_mean - completed_mean},
        "limit": "Train gain labels are generated with a frozen TCN trained on the train split. This is a bounded direction test; a final method requires cross-fitted gain labels before test reporting.",
    }
    out = run_dir / "06_diagnostics" / "db3_completion_gain_predictor_feasibility"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "metrics" / f"S{args.subject:02d}_completion_gain_feasibility.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(out / "predictions" / f"S{args.subject:02d}_completion_gain_feasibility.npz", val_target=val_y, pred_raw=raw_val_pred, pred_completed=completed_pred, masks=selected_masks, candidate_window_ids=val_windows, candidate_patch_ids=val_patches, predicted_gain=val_score, actual_gain=val_gain, selected_candidates=selected)
    print(f"  ranking pearson={pearson} auc={auc} selected_gain={val_gain[selected].mean():+.6f}")
    print(f"  validation raw={raw_mean:.6f} completed={completed_mean:.6f} gain={raw_mean - completed_mean:+.6f}")
    print(f"[diagnostic] saved: {out}")


if __name__ == "__main__":
    main()