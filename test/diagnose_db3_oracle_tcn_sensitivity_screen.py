"""Validation-only label-free TCN sensitivity screen against Oracle gains."""
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
    spec = importlib.util.spec_from_file_location("tcn_sensitivity_exp3", path)
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


def association(values, gain):
    count = max(1, int(np.ceil(0.10 * len(values))))
    top = np.argsort(values)[-count:]
    return {
        "spearman_with_oracle_gain": spearman(values, gain),
        "top_10pct_gain_mean": float(gain[top].mean()),
        "all_gain_mean": float(gain.mean()),
        "top_10pct_positive_gain_fraction": float((gain[top] > 0.0).mean()),
    }


def patch_sensitivity(model, windows, patch_size, device, n_probes, seed):
    rng = np.random.default_rng(seed)
    n_windows, n_steps, n_channels = windows.shape
    n_patches = n_steps // patch_size
    values = np.empty((n_windows, n_patches), dtype=np.float64)
    for index in range(n_windows):
        squared_grad = np.zeros((n_steps, n_channels), dtype=np.float64)
        for _ in range(n_probes):
            x = torch.as_tensor(
                windows[index:index + 1], dtype=torch.float32, device=device
            ).requires_grad_(True)
            output = model(x)
            signs = rng.choice(
                np.array([-1.0, 1.0], dtype=np.float32), size=tuple(output.shape)
            )
            probe = torch.as_tensor(signs, dtype=output.dtype, device=device)
            scalar = (output * probe).sum()
            gradient = torch.autograd.grad(scalar, x, only_inputs=True)[0][0]
            squared_grad += gradient.detach().cpu().numpy().astype(np.float64) ** 2
        squared_grad /= float(n_probes)
        for patch_id in range(n_patches):
            start = patch_id * patch_size
            values[index, patch_id] = np.sqrt(
                squared_grad[start:start + patch_size].sum()
            )
    return values


def main():
    parser = argparse.ArgumentParser(
        description="DB3 validation-only label-free TCN sensitivity screen"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--n-probes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if args.n_probes < 1:
        raise ValueError("n-probes must be positive")
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
    oracle_gain = oracle["candidate_gain"]

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    exp3 = load_exp3()
    checkpoint = (
        run_dir
        / "06_diagnostics"
        / "db3_validation_error_windows"
        / "checkpoints"
        / f"S{args.subject:02d}_raw_validation_best.pth"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    loader = NinaProDataLoader(
        config["db2_path"], config["db3_path"], fs=config["orig_fs"]
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
    patch_size = int(config["patch_size"])
    if windows.shape[1] % patch_size:
        raise ValueError("window length must be divisible by patch_size")
    n_patches = windows.shape[1] // patch_size
    if oracle_gain.shape != (len(windows), n_patches):
        raise ValueError("Oracle gain shape does not match validation windows")

    tcn = load_tcn(exp3, checkpoint, config, windows.shape[-1], config["device"])
    print(
        f"[tcn sensitivity] S{args.subject:02d} validation={len(windows)} "
        f"patches_per_window={n_patches} probes={args.n_probes} "
        f"test_unused={len(test_idx)}"
    )
    sensitivity = patch_sensitivity(
        tcn, windows, patch_size, config["device"], args.n_probes, args.seed
    )
    results = association(sensitivity.reshape(-1), oracle_gain.reshape(-1))
    report = {
        "scope": "validation-only label-free TCN output-sensitivity feature screen",
        "test_policy": "test repetitions 2/5 are not predicted, scored, or used",
        "feature_policy": (
            "The feature is a Hutchinson estimate of the output Jacobian norm and "
            "uses only raw EMG and frozen TCN weights. Glove labels are used only "
            "afterward to calculate the Oracle-gain association."
        ),
        "subject_id": args.subject,
        "reused": {
            "oracle_gain": str(oracle_path),
            "raw_tcn": str(checkpoint),
        },
        "candidate_constraint": {
            "candidate_unit": "one all-channel time patch",
            "patch_size_samples": patch_size,
            "patch_duration_ms": 1000.0 * patch_size / float(config["target_fs"]),
            "all_patch_positions_per_window": n_patches,
        },
        "n_validation_windows": int(len(windows)),
        "n_probes": args.n_probes,
        "seed": args.seed,
        "association_rule": "Spearman correlation greater than 0.1 is a feature-screen pass",
        "association": results,
        "interpretation_limit": (
            "This is a feature screen only. A passing feature still requires "
            "out-of-fold training-label construction and an untouched test evaluation."
        ),
    }
    out = run_dir / "06_diagnostics" / "db3_oracle_tcn_sensitivity_screen"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    out_file = out / "metrics" / f"S{args.subject:02d}_tcn_sensitivity.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / f"S{args.subject:02d}_tcn_sensitivity.npz",
        validation_indices=validation_indices,
        patch_sensitivity=sensitivity,
    )
    print(
        f"  sensitivity Spearman={results['spearman_with_oracle_gain']:+.4f} "
        f"top10_gain={results['top_10pct_gain_mean']:+.6f}"
    )
    print(f"[diagnostic] saved: {out_file}")


if __name__ == "__main__":
    main()
