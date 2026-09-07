"""
Level 0：在干净 vs 退化残肢 sEMG 下的 DB3 直接回归基线。

本实验刻意排除生成式补全、迁移学习、解剖感知掩码与结构损失，
仅回答：截肢残肢 sEMG 被掩码后，直接关节角度回归会退化多少。
"""

import sys
import time
import json
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_kinematics import KinematicsDataset, make_subject_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from models.prediction.kinematic_regressor import KinematicBiLSTM, KinematicCNNGRU, KinematicTCN
from utils.kinematic_target import KEY10_CHANNEL_NAMES, KEY10_DIM, KEY10_GLOVE_INDICES, key10_prediction_metadata, key10_target_metadata
from utils.metrics_kinematics import evaluate_kinematics


def load_yaml_config(project_root):
    with open(project_root / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_level0_config(cfg):
    level0 = cfg.get("level0_baseline", {})
    signal = cfg["signal"]
    paths = cfg["paths"]
    config = {
        "db2_path": paths["db2"],
        "db3_path": paths["db3"],
        "output_dir": paths["output"],
        "orig_fs": signal["orig_fs"],
        "target_fs": signal["target_fs"],
        "n_channels": signal["n_channels"],
        "window_size": signal["window_size"],
        "stride": signal["stride"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "db3_subjects": level0.get("db3_subjects", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        "models": level0.get("models", ["tcn", "bilstm"]),
        "train_ratio": float(level0.get("train_ratio", 0.60)),
        "val_ratio": float(level0.get("val_ratio", 0.20)),
        "random_seed": int(level0.get("random_seed", 42)),
        "batch_size": int(level0.get("batch_size", 64)),
        "learning_rate": float(level0.get("learning_rate", 1e-3)),
        "num_epochs": int(level0.get("num_epochs", 100)),
        "patience": int(level0.get("patience", 20)),
        "lr_patience": int(level0.get("lr_patience", 5)),
        "lr_factor": float(level0.get("lr_factor", 0.5)),
        "hidden_dim": int(level0.get("hidden_dim", 64)),
        "n_layers": int(level0.get("n_layers", 4)),
        "bilstm_hidden_dim": int(level0.get("bilstm_hidden_dim", 128)),
        "bilstm_layers": int(level0.get("bilstm_layers", 2)),
        "cnn_gru_layers": int(level0.get("cnn_gru_layers", 2)),
        "kernel_size": int(level0.get("kernel_size", 3)),
        "model_dropout": float(level0.get("model_dropout", 0.1)),
        "n_angle_channels": int(level0.get("n_angle_channels", 10)),
        "channel_mask_ratio": float(level0.get("channel_mask_ratio", 0.20)),
        "temporal_patch_mask_ratio": float(level0.get("temporal_patch_mask_ratio", 0.15)),
        "patch_size": int(level0.get("patch_size", signal.get("patch_size", 8))),
        "results_path": level0.get(
            "results_path",
            str(Path(paths["output"]) / "level0_db3_baselines" / "results.json"),
        ),
    }
    if config["n_angle_channels"] != KEY10_DIM:
        raise ValueError(f"Level0 requires fixed Key10 output dimension {KEY10_DIM}.")
    return config


def random_channel_mask(shape, ratio, rng):
    """返回形状为 (N,T,C) 的通道掩码，1=可见，0=退化。"""
    n_samples, _, n_channels = shape
    keep = np.ones((n_samples, 1, n_channels), dtype=np.float32)
    for i in range(n_samples):
        n_drop = int(round(n_channels * ratio))
        if n_drop <= 0:
            continue
        drop_idx = rng.choice(n_channels, size=min(n_drop, n_channels), replace=False)
        keep[i, 0, drop_idx] = 0.0
    return np.repeat(keep, shape[1], axis=1)


def random_temporal_patch_mask(shape, ratio, patch_size, rng):
    """返回 patch 对齐的时间掩码，形状 (N,T,C)，1=可见。"""
    n_samples, time_steps, n_channels = shape
    if time_steps % patch_size != 0:
        raise ValueError(f"time_steps={time_steps} must be divisible by patch_size={patch_size}")
    n_patches = time_steps // patch_size
    patch_keep = (rng.random((n_samples, n_patches, n_channels)) >= ratio).astype(np.float32)
    return np.repeat(patch_keep, patch_size, axis=1)


def make_degraded_emg(emg, config, seed):
    rng = np.random.default_rng(seed)
    ch_mask = random_channel_mask(emg.shape, config["channel_mask_ratio"], rng)
    time_mask = random_temporal_patch_mask(
        emg.shape,
        config["temporal_patch_mask_ratio"],
        config["patch_size"],
        rng,
    )
    mask = ch_mask * time_mask
    return emg * mask, mask


def build_regressor(model_name, config, n_emg_channels, device):
    if model_name == "tcn":
        return KinematicTCN(
            n_emg_channels=n_emg_channels,
            n_angle_channels=config["n_angle_channels"],
            hidden_dim=config["hidden_dim"],
            n_layers=config["n_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["model_dropout"],
        ).to(device)
    if model_name == "bilstm":
        return KinematicBiLSTM(
            n_emg_channels=n_emg_channels,
            n_angle_channels=config["n_angle_channels"],
            hidden_dim=config["bilstm_hidden_dim"],
            n_layers=config["bilstm_layers"],
            dropout=config["model_dropout"],
        ).to(device)
    if model_name == "cnn_gru":
        return KinematicCNNGRU(
            n_emg_channels=n_emg_channels,
            n_angle_channels=config["n_angle_channels"],
            hidden_dim=config["hidden_dim"],
            n_layers=config["cnn_gru_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["model_dropout"],
        ).to(device)
    raise ValueError(f"Unknown Level 0 baseline model: {model_name}")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in loader:
        emg = batch["emg"].to(device)
        angle = batch["angle"].to(device)
        pred = model(emg)
        loss = criterion(pred, angle)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    preds, targets = [], []
    for batch in loader:
        emg = batch["emg"].to(device)
        angle = batch["angle"].to(device)
        pred = model(emg)
        total += float(criterion(pred, angle).item())
        preds.append(pred.cpu().numpy())
        targets.append(angle.cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    metrics = evaluate_kinematics(target_np, pred_np)
    metrics["loss"] = total / max(len(loader), 1)
    return metrics, pred_np, target_np


def train_and_test(model_name, emg, angle, split, config, device):
    train_idx, val_idx, test_idx = split
    train_set = KinematicsDataset(emg[train_idx], angle[train_idx])
    val_set = KinematicsDataset(emg[val_idx], angle[val_idx])
    test_set = KinematicsDataset(emg[test_idx], angle[test_idx])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    model = build_regressor(model_name, config, emg.shape[-1], device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["lr_factor"],
        patience=config["lr_patience"],
        min_lr=1e-7,
    )

    best_state = None
    best_val = float("inf")
    no_improve = 0
    history = []
    for epoch in range(config["num_epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        history.append({
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_r2": float(val_metrics["r2"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_mae": float(val_metrics["mae"]),
            "val_corr": float(val_metrics["corr"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= config["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, pred, target = evaluate(model, test_loader, criterion, device)
    test_metrics["best_val_loss"] = float(best_val)
    test_metrics["epochs"] = int(epoch + 1)
    return test_metrics, pred, target, history


def save_trace_plot(path, pred, target, title):
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(pred) == 0:
        return
    plt.figure(figsize=(8, 3))
    plt.plot(target[0, :, 0], label="true", linewidth=1.5)
    plt.plot(pred[0, :, 0], label="pred", linewidth=1.2)
    plt.title(title)
    plt.xlabel("sample")
    plt.ylabel("normalized angle")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_trace_grid(path, pred, target, title, max_channels=10):
    if plt is None or len(pred) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    n_channels = min(target.shape[-1], max_channels)
    n_cols = 2
    n_rows = int(np.ceil(n_channels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(8, n_rows * 1.45)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for c in range(n_channels):
        axes[c].plot(target[0, :, c], label="true", linewidth=1.0)
        axes[c].plot(pred[0, :, c], label="pred", linewidth=0.9, alpha=0.85)
        axes[c].set_title(KEY10_CHANNEL_NAMES[c], fontsize=9)
        axes[c].tick_params(labelsize=8)
    for ax in axes[n_channels:]:
        ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.supxlabel("sample")
    fig.supylabel("normalized angle")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_per_channel_metric_plot(path, metrics, title):
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = np.arange(1, len(metrics["r2_per_channel"]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(channels, metrics["r2_per_channel"])
    axes[0].set_ylabel("R2")
    axes[1].bar(channels, metrics["rmse_per_channel"])
    axes[1].set_ylabel("RMSE")
    axes[2].bar(channels, metrics["corr_per_channel"])
    axes[2].set_ylabel("Pearson r")
    axes[2].set_xlabel("joint channel")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_scatter_plot(path, pred, target, title, max_points=20000):
    if plt is None or len(pred) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = target.reshape(-1)
    y_pred = pred.reshape(-1)
    if len(y_true) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y_true), size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=3, alpha=0.25)
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
    plt.xlabel("true normalized angle")
    plt.ylabel("predicted normalized angle")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_error_histogram(path, pred, target, title):
    if plt is None or len(pred) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    err = (pred - target).reshape(-1)
    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=80, alpha=0.85)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("prediction error")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_training_curve(path, history, title):
    if plt is None or not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [h["train_loss"] for h in history], label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def metric_rows(subject_id, run_key, metrics):
    rows = []
    n_channels = len(metrics["r2_per_channel"])
    for c in range(n_channels):
        rows.append({
            "subject_id": subject_id,
            "run": run_key,
            "joint_channel": c + 1,
            "source_glove_index": KEY10_GLOVE_INDICES[c],
            "channel_name": KEY10_CHANNEL_NAMES[c],
            "r2": "" if not np.isfinite(metrics["r2_per_channel"][c]) else float(metrics["r2_per_channel"][c]),
            "r2_valid": bool(metrics["r2_valid_per_channel"][c]),
            "rmse": float(metrics["rmse_per_channel"][c]),
            "mae": float(metrics["mae_per_channel"][c]),
            "corr": float(metrics["corr_per_channel"][c]),
        })
    return rows


def save_level0_tables(out_dir, report):
    run_rows = []
    channel_rows = []
    for subject in report["subjects"]:
        if subject.get("status") != "ok":
            continue
        for run_key, result in subject.get("results", {}).items():
            if result.get("status") != "ok":
                continue
            metrics = result["metrics"]
            run_rows.append({
                "subject_id": subject["subject_id"],
                "run": run_key,
                "condition": run_key.rsplit("_", 1)[0],
                "model": run_key.rsplit("_", 1)[1],
                "r2": float(metrics["r2"]),
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "corr": float(metrics["corr"]),
                "loss": float(metrics["loss"]),
                "best_val_loss": float(metrics["best_val_loss"]),
                "epochs": int(metrics["epochs"]),
                "r2_valid_channel_count": int(metrics.get("r2_valid_channel_count", 0)),
                "n_segments": int(subject["n_segments"]),
                "test_segments": int(subject["split_sizes"]["test"]),
                "actual_mask_ratio": float(subject["actual_mask_ratio"]),
            })
            channel_rows.extend(metric_rows(subject["subject_id"], run_key, metrics))

    save_csv(
        out_dir / "run_metrics.csv",
        run_rows,
        ["subject_id", "run", "condition", "model", "r2", "rmse", "mae", "corr", "loss",
         "best_val_loss", "epochs", "r2_valid_channel_count", "n_segments", "test_segments", "actual_mask_ratio"],
    )
    save_csv(
        out_dir / "per_channel_metrics.csv",
        channel_rows,
        ["subject_id", "run", "joint_channel", "source_glove_index", "channel_name", "r2", "r2_valid", "rmse", "mae", "corr"],
    )


def summarize(report):
    ok_subjects = [s for s in report["subjects"] if s.get("status") == "ok"]
    summary = {}
    for condition in report["conditions"]:
        for model_name in report["models"]:
            key = f"{condition}_{model_name}"
            vals = [
                s["results"][key]["metrics"]
                for s in ok_subjects
                if key in s.get("results", {}) and s["results"][key].get("status") == "ok"
            ]
            if not vals:
                continue
            summary[key] = {
                "r2": float(np.mean([v["r2"] for v in vals])),
                "rmse": float(np.mean([v["rmse"] for v in vals])),
                "mae": float(np.mean([v["mae"] for v in vals])),
                "corr": float(np.mean([v["corr"] for v in vals])),
                "n_subjects": int(len(vals)),
            }
    if "clean_tcn" in summary and "degraded_tcn" in summary:
        summary["delta_degraded_minus_clean_tcn"] = {
            "r2": summary["degraded_tcn"]["r2"] - summary["clean_tcn"]["r2"],
            "rmse": summary["degraded_tcn"]["rmse"] - summary["clean_tcn"]["rmse"],
            "mae": summary["degraded_tcn"]["mae"] - summary["clean_tcn"]["mae"],
            "corr": summary["degraded_tcn"]["corr"] - summary["clean_tcn"]["corr"],
        }
    report["summary"] = summary


def save_summary_plot(path, report):
    if plt is None:
        return
    summary = report.get("summary", {})
    if not summary:
        return
    keys = [k for k in summary.keys() if not k.startswith("delta_")]
    if not keys:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(keys))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, metric in zip(axes.reshape(-1), ["r2", "rmse", "mae", "corr"]):
        ax.bar(x, [summary[k][metric] for k in keys])
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_ylabel(metric.upper() if metric == "r2" else metric)
    fig.suptitle("Level 0 DB3 clean/degraded summary")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_level0_config(cfg)
    device = config["device"]
    set_seed(config["random_seed"])

    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    out_path = Path(config["results_path"])
    viz_dir = out_path.parent / "viz"
    pred_dir = out_path.parent / "predictions"
    table_dir = out_path.parent / "tables"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "level0_db3_direct_regression_baselines",
        "angle_target": key10_target_metadata(),
        "conditions": ["clean", "degraded"],
        "models": config["models"],
        "config.yaml": {
            "db3_subjects": config["db3_subjects"],
            "train_ratio": config["train_ratio"],
            "val_ratio": config["val_ratio"],
            "random_seed": config["random_seed"],
            "channel_mask_ratio": config["channel_mask_ratio"],
            "temporal_patch_mask_ratio": config["temporal_patch_mask_ratio"],
            "patch_size": config["patch_size"],
        },
        "subjects": [],
    }

    start_time = time.time()
    print("=" * 80)
    print("[Level 0] DB3 clean vs degraded direct-regression baselines")
    print("=" * 80)

    for subject_id in config["db3_subjects"]:
        print(f"\n--- DB3 S{subject_id:02d} ---")
        try:
            clean_emg, angle, _, _reps = prepare_kinematics_data(
                data_loader, [subject_id], config, exercises=[1], db="db3"
            )
        except Exception as exc:
            print(f"  skipped: no usable glove data ({exc})")
            report["subjects"].append({"subject_id": subject_id, "status": "skipped", "reason": str(exc)})
            continue

        if len(clean_emg) < 5:
            reason = f"too few paired windows: {len(clean_emg)}"
            print(f"  skipped: {reason}")
            report["subjects"].append({"subject_id": subject_id, "status": "skipped", "reason": reason})
            continue

        split = make_subject_split(
            len(clean_emg),
            train_ratio=config["train_ratio"],
            val_ratio=config["val_ratio"],
            seed=config["random_seed"] + subject_id,
        )
        degraded_emg, degradation_mask = make_degraded_emg(
            clean_emg,
            config,
            seed=config["random_seed"] + 1000 + subject_id,
        )
        actual_mask_ratio = float((degradation_mask < 0.5).mean())
        subject_result = {
            "subject_id": subject_id,
            "status": "ok",
            "n_segments": int(len(clean_emg)),
            "split_sizes": {
                "train": int(len(split[0])),
                "val": int(len(split[1])),
                "test": int(len(split[2])),
            },
            "actual_mask_ratio": actual_mask_ratio,
            "angle_target": key10_target_metadata(),
            "results": {},
        }
        print(f"  paired windows={len(clean_emg)} | degraded mask={actual_mask_ratio:.2%}")

        inputs = {"clean": clean_emg, "degraded": degraded_emg}
        for condition, emg_input in inputs.items():
            for model_name in config["models"]:
                run_key = f"{condition}_{model_name}"
                print(f"  training {run_key}...")
                try:
                    metrics, pred, target, history = train_and_test(model_name, emg_input, angle, split, config, device)
                except Exception as exc:
                    print(f"    failed: {exc}")
                    subject_result["results"][run_key] = {"status": "failed", "reason": str(exc)}
                    continue
                subject_result["results"][run_key] = {"status": "ok", "metrics": metrics}
                np.savez_compressed(
                    pred_dir / f"S{subject_id:02d}_{run_key}_predictions.npz",
                    **key10_prediction_metadata(),
                    pred=pred,
                    target=target,
                    test_indices=split[2],
                    metrics=json_ready(metrics),
                )
                save_csv(
                    table_dir / "training_history" / f"S{subject_id:02d}_{run_key}_history.csv",
                    history,
                    ["epoch", "train_loss", "val_loss", "val_r2", "val_rmse", "val_mae", "val_corr", "lr"],
                )
                save_training_curve(
                    viz_dir / "training_curves" / f"S{subject_id:02d}_{run_key}_loss.png",
                    history,
                    f"S{subject_id:02d} {run_key} loss",
                )
                print(
                    f"    R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f} "
                    f"MAE={metrics['mae']:.4f} CC={metrics['corr']:.4f}"
                )
                if model_name == "tcn":
                    save_trace_plot(
                        viz_dir / f"S{subject_id:02d}_{run_key}_angle_trace.png",
                        pred,
                        target,
                        f"S{subject_id:02d} {run_key}",
                    )
                    save_trace_grid(
                        viz_dir / f"S{subject_id:02d}_{run_key}_angle_trace_grid.png",
                        pred,
                        target,
                        f"S{subject_id:02d} {run_key}: all Key10 traces",
                    )
                    save_per_channel_metric_plot(
                        viz_dir / f"S{subject_id:02d}_{run_key}_per_channel_metrics.png",
                        metrics,
                        f"S{subject_id:02d} {run_key}: per-joint metrics",
                    )
                    save_scatter_plot(
                        viz_dir / f"S{subject_id:02d}_{run_key}_true_vs_pred_scatter.png",
                        pred,
                        target,
                        f"S{subject_id:02d} {run_key}: true vs predicted",
                    )
                    save_error_histogram(
                        viz_dir / f"S{subject_id:02d}_{run_key}_error_hist.png",
                        pred,
                        target,
                        f"S{subject_id:02d} {run_key}: prediction error",
                    )

        if (
            "clean_tcn" in subject_result["results"]
            and "degraded_tcn" in subject_result["results"]
            and subject_result["results"]["clean_tcn"].get("status") == "ok"
            and subject_result["results"]["degraded_tcn"].get("status") == "ok"
        ):
            clean_m = subject_result["results"]["clean_tcn"]["metrics"]
            degraded_m = subject_result["results"]["degraded_tcn"]["metrics"]
            subject_result["delta_degraded_minus_clean_tcn"] = {
                "r2": degraded_m["r2"] - clean_m["r2"],
                "rmse": degraded_m["rmse"] - clean_m["rmse"],
                "mae": degraded_m["mae"] - clean_m["mae"],
                "corr": degraded_m["corr"] - clean_m["corr"],
            }

        report["subjects"].append(subject_result)
        summarize(report)
        report["elapsed_min"] = (time.time() - start_time) / 60.0
        save_json(out_path, report)
        save_level0_tables(table_dir, report)
        save_summary_plot(viz_dir / "level0_summary_metrics.png", report)

    summarize(report)
    report["elapsed_min"] = (time.time() - start_time) / 60.0
    save_json(out_path, report)
    save_level0_tables(table_dir, report)
    save_summary_plot(viz_dir / "level0_summary_metrics.png", report)
    print(f"\nSaved Level 0 report to: {out_path}")


if __name__ == "__main__":
    main()
