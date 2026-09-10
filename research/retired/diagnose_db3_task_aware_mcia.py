"""Train-only task-aware MCIA adapter check on S05 rule-mask completion."""
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config, set_seed
from utils.rule_anomaly_detector import RuleAnomalyDetector


def load_script(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_raw_tcn(exp3, checkpoint, config, channels, device):
    model = exp3.build_tcn(config, channels, device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def domain_tensor(size, device):
    return torch.ones(size, dtype=torch.long, device=device)


def enhanced_forward(mcia, raw, mask, domain_id=None):
    valid = (mask.mean(dim=1) > 0.5).float()
    pred = mcia(raw * mask, raw_time_mask=mask, chan_valid_mask=valid, domain_id=domain_id)
    return pred * (1.0 - mask) + raw * mask, pred


@torch.no_grad()
def enhance(mcia, windows, masks, device, batch_size, use_domain):
    result = np.empty_like(windows)
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        raw = torch.as_tensor(windows[start:stop], dtype=torch.float32, device=device)
        mask = torch.as_tensor(masks[start:stop], dtype=torch.float32, device=device)
        domain = domain_tensor(len(raw), device) if use_domain else None
        completed, _ = enhanced_forward(mcia, raw, mask, domain)
        result[start:stop] = completed.cpu().numpy()
    return result


def metric_payload(exp3, pred, target, config):
    scores = exp3.compute_dynamic_angle_scores(target, str(config["regressor_dynamic_score"]))
    dynamic_idx = np.flatnonzero(scores >= float(config["regressor_dynamic_min_ptp"]))
    return {
        "subsets": exp3.evaluate_subsets(pred, target),
        "dynamic_subsets": exp3.evaluate_subsets(pred[dynamic_idx], target[dynamic_idx]) if len(dynamic_idx) else {},
        "n_dynamic": int(len(dynamic_idx)),
        "n_total": int(len(target)),
    }


def train_task_aware(mcia, tcn, windows, angle, masks, device, epochs, batch_size, learning_rate, lambda_angle, seed):
    set_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(windows, dtype=torch.float32),
            torch.as_tensor(angle, dtype=torch.float32),
            torch.as_tensor(masks, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = optim.AdamW(
        [parameter for parameter in mcia.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-5,
    )
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        mcia.train()
        total = recon_total = angle_total = count = 0.0
        for raw, target, mask in loader:
            raw, target, mask = raw.to(device), target.to(device), mask.to(device)
            completed, pred = enhanced_forward(mcia, raw, mask, domain_tensor(len(raw), device))
            missing = 1.0 - mask
            recon = ((pred - raw).square() * missing).sum() / missing.sum().clamp_min(1.0)
            angle_loss = F.mse_loss(tcn(completed), target)
            loss = recon + lambda_angle * angle_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in mcia.parameters() if parameter.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            total += float(loss.item()) * len(raw)
            recon_total += float(recon.item()) * len(raw)
            angle_total += float(angle_loss.item()) * len(raw)
            count += len(raw)
        row = {
            "epoch": epoch + 1,
            "total_loss": total / count,
            "masked_reconstruction_mse": recon_total / count,
            "angle_mse": angle_total / count,
        }
        history.append(row)
        print(
            f"  task-aware epoch {epoch + 1}/{epochs}: total={row['total_loss']:.6f} "
            f"recon={row['masked_reconstruction_mse']:.6f} angle={row['angle_mse']:.6f}"
        )
        if row["total_loss"] < best_loss:
            best_loss = row["total_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in mcia.state_dict().items()}
    if best_state is not None:
        mcia.load_state_dict(best_state)
    return history, best_loss


def main():
    parser = argparse.ArgumentParser(description="S05 task-aware MCIA completion diagnostic")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-angle", type=float, default=5.0)
    parser.add_argument("--max-val-windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    if args.subject != 5:
        raise ValueError("This rapid diagnostic is intentionally fixed to S05")
    if args.epochs < 1 or args.max_val_windows < 1 or args.lambda_angle <= 0.0:
        raise ValueError("epochs, max-val-windows, and lambda-angle must be positive")
    run_dir = Path(args.run_dir).resolve()
    if not os.environ.get("MCIA_RUN_DIR") or Path(os.environ["MCIA_RUN_DIR"]).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    device = config["device"]
    batch_size = args.batch_size or int(config["regressor_batch_size"])
    out = run_dir / "06_diagnostics" / "db3_task_aware_mcia"
    for name in ("metrics", "checkpoints", "predictions", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)
    exp3 = load_script("task_aware_exp3", "04_eval_db3_angle_raw_vs_augmented.py")
    exp2 = load_script("task_aware_exp2", "02_finetune_mcia_db3_amputee.py")
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    emg, angle, _, repetitions = prepare_kinematics_data(loader, [5], config, exercises=[args.exercise], db="db3")
    train_idx, val_idx, test_idx = make_rep_split(
        repetitions, train_reps=(1, 3, 4, 6), test_reps=(2, 5),
        val_ratio=float(config["regressor_val_ratio"]), seed=int(config["regressor_random_seed"]) + 5,
    )
    val_idx = val_idx[:min(args.max_val_windows, len(val_idx))]
    detector = RuleAnomalyDetector(
        patch_size=int(config["patch_size"]), group_indices=config.get("group_indices")
    ).fit(emg[train_idx])
    train_masks = detector.detect_batch(emg[train_idx])["mask"]
    val_masks = detector.detect_batch(emg[val_idx])["mask"]
    tcn_path = run_dir / "06_diagnostics" / "db3_validation_error_windows" / "checkpoints" / "S05_raw_validation_best.pth"
    if not tcn_path.exists():
        raise FileNotFoundError(tcn_path)
    healthy, healthy_path = exp3.load_healthy_prior_mcia(config, device)
    if healthy is None:
        raise FileNotFoundError("healthy-prior MCIA checkpoint")
    tcn = load_raw_tcn(exp3, tcn_path, config, emg.shape[-1], device)
    print(
        f"[task-aware] S05 train={len(train_idx)} fixed_val={len(val_idx)} "
        f"test_unused={len(test_idx)} mask_train={float((train_masks < 0.5).mean()):.2%}"
    )
    healthy_val = enhance(healthy, emg[val_idx], val_masks, device, batch_size, False)

    task_mcia, _ = exp3.load_healthy_prior_mcia(config, device)
    exp2.activate_adapters(task_mcia, device, n_adapter_blocks=2, bottleneck_dim=32)
    history, best_loss = train_task_aware(
        task_mcia, tcn, emg[train_idx], angle[train_idx], train_masks, device,
        args.epochs, batch_size, args.learning_rate, args.lambda_angle,
        int(config["regressor_random_seed"]) + 5,
    )
    task_mcia.eval()
    task_val = enhance(task_mcia, emg[val_idx], val_masks, device, batch_size, True)
    raw_pred, target = exp3.predict_on_set(tcn, emg[val_idx], angle[val_idx], np.arange(len(val_idx)), config, device)
    healthy_pred, _ = exp3.predict_on_set(tcn, healthy_val, angle[val_idx], np.arange(len(val_idx)), config, device)
    task_pred, _ = exp3.predict_on_set(tcn, task_val, angle[val_idx], np.arange(len(val_idx)), config, device)
    results = {
        "raw": metric_payload(exp3, raw_pred, target, config),
        "healthy_rule_completion": metric_payload(exp3, healthy_pred, target, config),
        "task_aware_rule_completion": metric_payload(exp3, task_pred, target, config),
    }
    checkpoint = out / "checkpoints" / "S05_task_aware_adapter_best.pth"
    torch.save({"model": task_mcia.state_dict(), "scope": "train-only task-aware diagnostic"}, checkpoint)
    np.savez_compressed(
        out / "predictions" / "S05_fixed64_task_aware.npz",
        validation_indices=val_idx, target=target, pred_raw=raw_pred,
        pred_healthy=healthy_pred, pred_task_aware=task_pred, rule_mask=val_masks,
    )
    report = {
        "scope": "S05 train-only task-aware MCIA adapter diagnostic",
        "leakage_control": {
            "detector_fit": "training repetitions only",
            "mcia_training": "training repetitions only",
            "frozen_downstream": "raw KinematicTCN trained without task-aware MCIA",
            "validation": "fixed validation windows only",
            "test_repetitions": [2, 5],
            "test_policy": "not loaded or scored",
        },
        "split_sizes": {"train": int(len(train_idx)), "fixed_val": int(len(val_idx)), "test_unused": int(len(test_idx))},
        "training": {
            "loss": "masked reconstruction MSE + lambda_angle * frozen-TCN angle MSE",
            "lambda_angle": float(args.lambda_angle),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "history": history,
            "best_train_loss": float(best_loss),
            "checkpoint": str(checkpoint),
        },
        "inputs": {"healthy_mcia": str(healthy_path), "raw_tcn": str(tcn_path), "rule_mask_train_ratio": float((train_masks < 0.5).mean())},
        "validation": results,
        "interpretation_limit": "This is a frozen-TCN mechanism check. A positive result would still require a separately trained enhanced-input KinematicTCN comparison.",
    }
    out_file = out / "metrics" / "S05_task_aware_mcia.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in results.items():
        print(f"  {name}: global RMSE={result['subsets']['global']['rmse']:.6f}")
    print(f"[diagnostic] saved: {out_file}")


if __name__ == "__main__":
    main()
