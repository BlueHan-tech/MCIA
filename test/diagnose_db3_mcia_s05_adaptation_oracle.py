"""Train-only S05 MCIA adapter check with a fixed validation-only Oracle screen."""
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
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset_db2_emg import EMGCompletionDataset
from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import (
    build_mask_generator,
    build_structural_loss,
    flatten_pipeline_config,
    load_yaml_config,
    set_seed,
    train_mcia_epoch,
)


def load_module(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_window_mse(pred, target):
    return np.mean((pred - target) ** 2, axis=(1, 2))


@torch.no_grad()
def complete(mcia, windows, masks, device, batch_size, domain_id=None):
    result = np.empty_like(windows)
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        x = torch.as_tensor(windows[start:stop], dtype=torch.float32, device=device)
        mask = torch.as_tensor(masks[start:stop], dtype=torch.float32, device=device)
        channel_valid = (mask.mean(dim=1) > 0.5).float()
        domain = None
        if domain_id is not None:
            domain = torch.full((len(x),), int(domain_id), dtype=torch.long, device=device)
        pred = mcia(
            x * mask,
            raw_time_mask=mask,
            chan_valid_mask=channel_valid,
            domain_id=domain,
        )
        result[start:stop] = (pred * (1.0 - mask) + x * mask).cpu().numpy()
    return result


def fixed_oracle(exp3, tcn, mcia, windows, target, config, device, batch_size, domain_id):
    patch_size = int(config["patch_size"])
    n_patches = windows.shape[1] // patch_size
    window_ids = np.repeat(np.arange(len(windows)), n_patches)
    patch_ids = np.tile(np.arange(n_patches), len(windows))
    candidates = windows[window_ids]
    masks = np.ones_like(candidates, dtype=np.float32)
    for row, patch_id in enumerate(patch_ids):
        masks[row, patch_id * patch_size:(patch_id + 1) * patch_size] = 0.0
    raw_pred, raw_target = exp3.predict_on_set(
        tcn, windows, target, np.arange(len(windows)), config, device
    )
    raw_mse = per_window_mse(raw_pred, raw_target)
    completed = complete(mcia, candidates, masks, device, batch_size, domain_id)
    completed_pred, completed_target = exp3.predict_on_set(
        tcn, completed, target[window_ids], np.arange(len(completed)), config, device
    )
    if not np.allclose(completed_target, target[window_ids]):
        raise RuntimeError("candidate target order changed")
    candidate_mse = per_window_mse(completed_pred, completed_target).reshape(len(windows), n_patches)
    candidate_gain = np.sqrt(raw_mse)[:, None] - np.sqrt(candidate_mse)
    best_mse = candidate_mse.min(axis=1)
    return {
        "raw_global_rmse": float(np.sqrt(raw_mse.mean())),
        "candidate_gain_mean": float(candidate_gain.mean()),
        "candidate_positive_gain_fraction": float((candidate_gain > 0.0).mean()),
        "best_patch_global_rmse": float(np.sqrt(best_mse.mean())),
        "best_patch_global_rmse_gain": float(np.sqrt(raw_mse.mean()) - np.sqrt(best_mse.mean())),
        "best_patch_positive_window_fraction": float((candidate_gain.max(axis=1) > 0.0).mean()),
        "best_patch_index": np.argmin(candidate_mse, axis=1).astype(np.int16),
        "candidate_gain": candidate_gain.astype(np.float32),
    }


def train_adapter(model, train_windows, config, device, epochs, learning_rate, seed):
    set_seed(seed)
    dataset = EMGCompletionDataset(train_windows)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    mask_generator = build_mask_generator(config)
    criterion = build_structural_loss(config, device)
    optimizer = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
        weight_decay=1e-5,
    )
    best_loss = float("inf")
    best_state = None
    losses = []
    for epoch in range(int(epochs)):
        loss = train_mcia_epoch(
            model,
            loader,
            optimizer,
            device,
            mask_generator,
            criterion,
            scenario=None,
            cfg_dropout_prob=float(config.get("cfg_dropout_prob", 0.0)),
            domain_id=1,
            epoch=epoch,
        )
        losses.append(float(loss))
        print(f"  adaptation epoch {epoch + 1}/{epochs}: train_loss={loss:.6f}")
        if loss < best_loss:
            best_loss = float(loss)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return losses, best_loss


def pack_oracle(result):
    return {key: value for key, value in result.items() if key not in {"best_patch_index", "candidate_gain"}}


def main():
    parser = argparse.ArgumentParser(description="S05 train-only MCIA adaptation plus fixed validation Oracle")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-val-windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    if args.subject != 5:
        raise ValueError("This fast diagnostic is intentionally fixed to S05")
    if args.epochs < 1 or args.max_val_windows < 1:
        raise ValueError("epochs and max-val-windows must be positive")

    run_dir = Path(args.run_dir).resolve()
    if not os.environ.get("MCIA_RUN_DIR") or Path(os.environ["MCIA_RUN_DIR"]).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    out = run_dir / "06_diagnostics" / "db3_mcia_s05_adaptation_oracle"
    for name in ("metrics", "checkpoints", "predictions", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    device = config["device"]
    batch_size = args.batch_size or int(config["regressor_batch_size"])
    exp3 = load_module("s05_adaptation_exp3", "04_eval_db3_angle_raw_vs_augmented.py")
    exp2 = load_module("s05_adaptation_exp2", "02_finetune_mcia_db3_amputee.py")
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
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
    fixed_val_idx = val_idx[: min(int(args.max_val_windows), len(val_idx))]
    tcn_checkpoint = (
        run_dir / "06_diagnostics" / "db3_validation_error_windows" / "checkpoints"
        / f"S{args.subject:02d}_raw_validation_best.pth"
    )
    if not tcn_checkpoint.exists():
        raise FileNotFoundError(tcn_checkpoint)
    healthy_mcia, healthy_checkpoint = exp3.load_healthy_prior_mcia(config, device)
    if healthy_mcia is None:
        raise FileNotFoundError("healthy-prior MCIA checkpoint")
    tcn = exp3.build_tcn(config, emg.shape[-1], device)
    try:
        state = torch.load(tcn_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(tcn_checkpoint, map_location=device)
    tcn.load_state_dict(state)
    tcn.eval()

    print(
        f"[S05 adapter oracle] train={len(train_idx)} fixed_val={len(fixed_val_idx)} "
        f"test_unused={len(test_idx)} epochs={args.epochs}"
    )
    healthy_oracle = fixed_oracle(
        exp3, tcn, healthy_mcia, emg[fixed_val_idx], angle[fixed_val_idx],
        config, device, batch_size, domain_id=None,
    )
    print(
        f"  healthy Oracle gain={healthy_oracle['best_patch_global_rmse_gain']:+.6f} "
        f"positive={healthy_oracle['candidate_positive_gain_fraction']:.2%}"
    )

    adapted_mcia, _ = exp3.load_healthy_prior_mcia(config, device)
    exp2.activate_adapters(adapted_mcia, device, n_adapter_blocks=2, bottleneck_dim=32)
    losses, best_loss = train_adapter(
        adapted_mcia,
        emg[train_idx],
        config,
        device,
        args.epochs,
        args.learning_rate,
        int(config["regressor_random_seed"]) + args.subject,
    )
    checkpoint = out / "checkpoints" / "S05_train_only_adapter_best.pth"
    torch.save(
        {
            "model": adapted_mcia.state_dict(),
            "subject_id": args.subject,
            "scope": "train-repetitions-only diagnostic adapter",
            "epochs": args.epochs,
        },
        checkpoint,
    )
    adapted_mcia.eval()
    adapted_oracle = fixed_oracle(
        exp3, tcn, adapted_mcia, emg[fixed_val_idx], angle[fixed_val_idx],
        config, device, batch_size, domain_id=1,
    )
    print(
        f"  adapted Oracle gain={adapted_oracle['best_patch_global_rmse_gain']:+.6f} "
        f"positive={adapted_oracle['candidate_positive_gain_fraction']:.2%}"
    )
    np.savez_compressed(
        out / "predictions" / "S05_fixed64_oracle.npz",
        validation_indices=fixed_val_idx,
        healthy_best_patch=healthy_oracle["best_patch_index"],
        healthy_candidate_gain=healthy_oracle["candidate_gain"],
        adapted_best_patch=adapted_oracle["best_patch_index"],
        adapted_candidate_gain=adapted_oracle["candidate_gain"],
    )
    report = {
        "scope": "S05 train-only MCIA adapter diagnostic with fixed validation-only Oracle",
        "leakage_control": {
            "mcia_adaptation": "only train repetitions 1/3/4/6",
            "oracle_windows": "first fixed validation indices from train repetitions",
            "test_repetitions": [2, 5],
            "test_policy": "not loaded into model, Oracle, or metric computation",
        },
        "split_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "fixed_val": int(len(fixed_val_idx)), "test_unused": int(len(test_idx))},
        "adaptation": {
            "method": "healthy-prior MCIA plus last-two-block adapters",
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "scenario": "configured ScenarioMix sampled per batch",
            "train_losses": losses,
            "best_train_loss": float(best_loss),
            "checkpoint": str(checkpoint),
        },
        "reused": {"healthy_mcia_checkpoint": str(healthy_checkpoint), "raw_tcn_checkpoint": str(tcn_checkpoint)},
        "oracle_constraint": {
            "candidate_unit": "one all-channel time patch",
            "patch_size_samples": int(config["patch_size"]),
            "patch_duration_ms": 1000.0 * int(config["patch_size"]) / float(config["target_fs"]),
            "max_patches_per_window": 1,
            "selection_uses": "validation angle labels; upper bound only",
        },
        "healthy_prior": pack_oracle(healthy_oracle),
        "train_only_adapted": pack_oracle(adapted_oracle),
        "interpretation_limit": "An Oracle gain does not establish a deployable mask policy. Adaptation is promising only if it improves this validation-only upper bound before selector development.",
    }
    out_file = out / "metrics" / "S05_train_only_adapter_oracle.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diagnostic] saved: {out_file}")


if __name__ == "__main__":
    main()
