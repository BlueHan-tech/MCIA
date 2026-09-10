"""Train-only EMG patch-selector screen against a fixed validation Oracle."""
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config, set_seed


def load_exp3():
    path = ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("oracle_selector_exp3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PatchSelector(nn.Module):
    """Small full-window dilated CNN that emits one score per time patch."""

    def __init__(self, channels, n_patches):
        super().__init__()
        width = 32
        layers = [nn.Conv1d(channels, width, 3, padding=1), nn.GELU()]
        for dilation in (1, 2, 4, 8, 16, 32):
            layers += [nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation), nn.GELU()]
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv1d(width, 1, 1)
        self.n_patches = int(n_patches)

    def forward(self, x):
        score = self.head(self.features(x.transpose(1, 2))).squeeze(1)
        return score.reshape(len(score), self.n_patches, -1).mean(dim=-1)


@torch.no_grad()
def complete(mcia, windows, masks, device, batch_size):
    out = np.empty_like(windows)
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        x = torch.as_tensor(windows[start:stop], dtype=torch.float32, device=device)
        mask = torch.as_tensor(masks[start:stop], dtype=torch.float32, device=device)
        valid = (mask.mean(dim=1) > 0.5).float()
        pred = mcia(x * mask, raw_time_mask=mask, chan_valid_mask=valid)
        out[start:stop] = (pred * (1.0 - mask) + x * mask).cpu().numpy()
    return out


def mse(pred, target):
    return np.mean((pred - target) ** 2, axis=(1, 2))


def oracle(exp3, tcn, mcia, windows, target, config, device, batch_size):
    patch_size = int(config["patch_size"])
    n_patches = windows.shape[1] // patch_size
    ids = np.repeat(np.arange(len(windows)), n_patches)
    patches = np.tile(np.arange(n_patches), len(windows))
    candidates = windows[ids]
    masks = np.ones_like(candidates, dtype=np.float32)
    for row, patch in enumerate(patches):
        masks[row, patch * patch_size:(patch + 1) * patch_size] = 0.0
    raw_pred, raw_target = exp3.predict_on_set(tcn, windows, target, np.arange(len(windows)), config, device)
    raw_mse = mse(raw_pred, raw_target)
    completed = complete(mcia, candidates, masks, device, batch_size)
    pred, actual = exp3.predict_on_set(tcn, completed, target[ids], np.arange(len(completed)), config, device)
    if not np.allclose(actual, target[ids]):
        raise RuntimeError("candidate target order changed")
    candidate_mse = mse(pred, actual).reshape(len(windows), n_patches)
    return raw_mse, candidate_mse


def train_selector(windows, labels, channels, n_patches, device, epochs, batch_size, learning_rate, seed):
    set_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(windows, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    model = PatchSelector(channels, n_patches).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total = correct = count = 0
        for x, y in loader:
            logits = model(x.to(device))
            y = y.to(device)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(x)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            count += len(x)
        item = {"epoch": epoch + 1, "train_cross_entropy": total / count, "train_top1": correct / count}
        history.append(item)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"  selector epoch {epoch + 1}/{epochs}: loss={item['train_cross_entropy']:.4f} top1={item['train_top1']:.2%}")
    return model.eval(), history


@torch.no_grad()
def select(model, windows, device, batch_size):
    selected = []
    for start in range(0, len(windows), batch_size):
        x = torch.as_tensor(windows[start:start + batch_size], dtype=torch.float32, device=device)
        selected.append(model(x).argmax(dim=1).cpu().numpy())
    return np.concatenate(selected).astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description="S05 EMG-only Oracle patch selector feasibility")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-val-windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    if args.subject != 5:
        raise ValueError("This rapid screen is intentionally fixed to S05")
    if args.epochs < 1 or args.max_val_windows < 1:
        raise ValueError("epochs and max-val-windows must be positive")
    run_dir = Path(args.run_dir).resolve()
    if not os.environ.get("MCIA_RUN_DIR") or Path(os.environ["MCIA_RUN_DIR"]).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")
    out = run_dir / "06_diagnostics" / "db3_oracle_patch_selector"
    for name in ("metrics", "checkpoints", "predictions", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    device = config["device"]
    batch_size = args.batch_size or int(config["regressor_batch_size"])
    exp3 = load_exp3()
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    emg, angle, _, repetitions = prepare_kinematics_data(loader, [5], config, exercises=[args.exercise], db="db3")
    train_idx, val_idx, test_idx = make_rep_split(
        repetitions, train_reps=(1, 3, 4, 6), test_reps=(2, 5),
        val_ratio=float(config["regressor_val_ratio"]), seed=int(config["regressor_random_seed"]) + 5,
    )
    val_idx = val_idx[:min(args.max_val_windows, len(val_idx))]
    tcn_path = run_dir / "06_diagnostics" / "db3_validation_error_windows" / "checkpoints" / "S05_raw_validation_best.pth"
    if not tcn_path.exists():
        raise FileNotFoundError(tcn_path)
    mcia, mcia_path = exp3.load_healthy_prior_mcia(config, device)
    if mcia is None:
        raise FileNotFoundError("healthy-prior MCIA checkpoint")
    tcn = exp3.build_tcn(config, emg.shape[-1], device)
    try:
        state = torch.load(tcn_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(tcn_path, map_location=device)
    tcn.load_state_dict(state)
    tcn.eval()

    print(f"[oracle selector] S05 train={len(train_idx)} fixed_val={len(val_idx)} test_unused={len(test_idx)}")
    train_raw_mse, train_candidate_mse = oracle(exp3, tcn, mcia, emg[train_idx], angle[train_idx], config, device, batch_size)
    train_label = train_candidate_mse.argmin(axis=1)
    print("  train Oracle labels generated")
    model, history = train_selector(
        emg[train_idx], train_label, emg.shape[-1], train_candidate_mse.shape[1], device,
        args.epochs, batch_size, args.learning_rate, int(config["regressor_random_seed"]) + 5,
    )
    model_path = out / "checkpoints" / "S05_emg_patch_selector.pth"
    torch.save(model.state_dict(), model_path)
    val_raw_mse, val_candidate_mse = oracle(exp3, tcn, mcia, emg[val_idx], angle[val_idx], config, device, batch_size)
    selected = select(model, emg[val_idx], device, batch_size)
    selected_mse = val_candidate_mse[np.arange(len(val_idx)), selected]
    best_mse = val_candidate_mse.min(axis=1)
    best_patch = val_candidate_mse.argmin(axis=1)
    metrics = {
        "raw_global_rmse": float(np.sqrt(val_raw_mse.mean())),
        "selected_global_rmse": float(np.sqrt(selected_mse.mean())),
        "selected_global_rmse_gain": float(np.sqrt(val_raw_mse.mean()) - np.sqrt(selected_mse.mean())),
        "selected_positive_gain_fraction": float((selected_mse < val_raw_mse).mean()),
        "selector_exact_oracle_top1": float((selected == best_patch).mean()),
        "oracle_global_rmse": float(np.sqrt(best_mse.mean())),
        "oracle_global_rmse_gain": float(np.sqrt(val_raw_mse.mean()) - np.sqrt(best_mse.mean())),
    }
    np.savez_compressed(
        out / "predictions" / "S05_fixed64_selector.npz",
        validation_indices=val_idx,
        selected_patch=selected,
        oracle_best_patch=best_patch,
        candidate_mse=val_candidate_mse,
        raw_mse=val_raw_mse,
    )
    report = {
        "scope": "S05 train-only EMG selector feasibility diagnostic",
        "leakage_control": {
            "oracle_labels": "training repetitions only",
            "selector_input": "raw EMG only",
            "validation": "fixed validation windows, not used for training",
            "test_repetitions": [2, 5],
            "test_policy": "not loaded or scored",
        },
        "split_sizes": {"train": int(len(train_idx)), "fixed_val": int(len(val_idx)), "test_unused": int(len(test_idx))},
        "models": {"healthy_mcia": str(mcia_path), "raw_tcn": str(tcn_path), "selector": str(model_path)},
        "candidate_contract": {"unit": "one all-channel time patch", "patch_count": int(train_candidate_mse.shape[1]), "patch_size_samples": int(config["patch_size"])},
        "selector_training": {"architecture": "small full-window dilated temporal CNN", "epochs": args.epochs, "learning_rate": args.learning_rate, "history": history},
        "validation": metrics,
        "interpretation_limit": "Frozen raw-TCN selector feasibility only. A positive result still requires a separate enhanced-input TCN evaluation.",
    }
    out_file = out / "metrics" / "S05_oracle_patch_selector.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  validation selected gain={metrics['selected_global_rmse_gain']:+.6f}; top1={metrics['selector_exact_oracle_top1']:.2%}; oracle={metrics['oracle_global_rmse_gain']:+.6f}")
    print(f"[diagnostic] saved: {out_file}")


if __name__ == "__main__":
    main()
