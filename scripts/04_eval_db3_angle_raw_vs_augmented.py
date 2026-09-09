"""
实验 3：DB3 受试者内连续关节角度估计。

各组在每个受试者内使用相同的基于重复次数的训练/验证/测试集划分：
A：原始 EMG 训练 / 原始 EMG 验证 / 原始 EMG 测试。
B：健康先验 MCIA 增强 EMG 训练/验证/测试，TCN 从零开始训练。

该设计消除了原有原始训练/增强测试之间的分布不匹配问题，
用于检验一致的增强 EMG 表征是否有助于提升下游运动学估计性能。
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset_kinematics import KinematicsDataset, make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from models.prediction.kinematic_regressor import KinematicTCN
from utils.paper_pipeline import (
    build_mcia,
    flatten_pipeline_config,
    load_mcia_state_dict,
    load_yaml_config,
    patch_boundary_crossfade,
    safe_pearson_np as _safe_pearson,
    save_json,
    set_seed,
)
from utils.kinematic_output_postprocess import (
    continuous_trajectory_quality,
    postprocess_window_predictions,
    window_overlap_prediction_consistency,
)
from utils.kinematic_target import (
    KEY10_CHANNEL_NAMES,
    KEY10_DIM,
    KEY10_PLOT_GROUPS,
    KEY10_SUBSETS,
    assert_key10_prediction_payload,
    assert_key10_target,
    key10_prediction_metadata,
    key10_target_metadata,
)


ANGLE_SUBSETS = KEY10_SUBSETS
ANGLE_PLOT_GROUPS = KEY10_PLOT_GROUPS
def build_tcn(config, n_emg_channels, device):
    if int(config["regressor_n_angle_channels"]) != KEY10_DIM:
        raise ValueError(f"Exp3 requires fixed Key10 output dimension {KEY10_DIM}.")
    return KinematicTCN(
        n_emg_channels=n_emg_channels,
        n_angle_channels=config["regressor_n_angle_channels"],
        hidden_dim=config["regressor_hidden_dim"],
        n_layers=config["regressor_n_layers"],
        kernel_size=config["regressor_kernel_size"],
        dropout=config["regressor_model_dropout"],
    ).to(device)


def _train_loop(model, train_loader, val_loader, optimizer, criterion, scheduler,
                num_epochs, patience, device):
    best_val = float("inf")
    best_state = None
    no_improve = 0
    epoch = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            emg = batch["emg"].to(device)
            angle = batch["angle"].to(device)
            loss = criterion(model(emg), angle)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += float(criterion(model(batch["emg"].to(device)),
                                            batch["angle"].to(device)).item())
        val_loss /= max(len(val_loader), 1)
        if scheduler is not None:
            scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_val), epoch + 1


def train_tcn_on_emg(emg, angle, train_idx, val_idx, config, device):
    train_set = KinematicsDataset(emg[train_idx], angle[train_idx])
    val_set = KinematicsDataset(emg[val_idx], angle[val_idx])
    bs = config["regressor_batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=0)

    model = build_tcn(config, emg.shape[-1], device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["regressor_learning_rate"],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["regressor_lr_factor"],
        patience=config["regressor_lr_patience"],
        min_lr=1e-7,
    )
    return _train_loop(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        int(config["regressor_num_epochs"]), int(config["regressor_patience"]), device,
    )


def _exp1_checkpoint_candidates(config):
    candidates = [
        Path(config.get("exp1_dir", "")) / "checkpoints" / "best_model.pth",
        Path(config["checkpoints_dir"]) / "exp1_mcia_db2" / "best_model.pth",
    ]
    output_dir = Path(config["output_dir"]) / "exp1_mcia_db2"
    if output_dir.exists():
        candidates += sorted(output_dir.glob("run_*/best_model.pth"), reverse=True)
    return candidates


def _load_mcia_from_checkpoint(config, device, checkpoint_path, label):
    model = build_mcia(config, device)
    load_mcia_state_dict(model, checkpoint_path, device)
    model.eval()
    print(f"  [MCIA:{label}] Loaded: {checkpoint_path}")
    return model, checkpoint_path


def load_healthy_prior_mcia(config, device):
    for path in _exp1_checkpoint_candidates(config):
        if path.exists():
            return _load_mcia_from_checkpoint(config, device, path, "healthy_prior")
    print("  [MCIA:B] No healthy-prior checkpoint found; Group B will be skipped")
    return None, None


@torch.no_grad()
def apply_mcia(mcia_model, emg_windows, masks, device, batch_size=64, domain_id=None,
               patch_size=8):
    N, T, C = emg_windows.shape
    enhanced = np.empty_like(emg_windows)
    for start in range(0, N, batch_size):
        batch_np = emg_windows[start:start + batch_size]
        B = len(batch_np)
        masks_np = masks[start:start + B]
        mask_t = torch.FloatTensor(masks_np).to(device)
        emg_t = torch.FloatTensor(batch_np).to(device)
        emg_masked = emg_t * mask_t
        chan_valid = (mask_t.mean(dim=1) > 0.5).float()
        domain_t = (
            torch.full((B,), int(domain_id), dtype=torch.long, device=device)
            if domain_id is not None else None
        )
        pred = mcia_model(
            emg_masked, raw_time_mask=mask_t, chan_valid_mask=chan_valid, domain_id=domain_t
        )
        # 交付规则（2026-09-09 采纳）：clip 后对 patch 边界做三点淡化，再复制回观测值。
        pred = patch_boundary_crossfade(pred.clamp(0.0, 1.0), patch_size)
        enh = pred * (1.0 - mask_t) + emg_t * mask_t
        enhanced[start:start + B] = enh.cpu().numpy()
    return enhanced


def make_enhanced_pool(mcia_model, raw_emg, train_idx, val_idx, test_idx, masks, device, domain_id=None,
                       patch_size=8):
    enhanced = raw_emg.copy()
    for idx in (train_idx, val_idx, test_idx):
        if len(idx) > 0:
            enhanced[idx] = apply_mcia(
                mcia_model, raw_emg[idx], masks[idx], device, domain_id=domain_id,
                patch_size=patch_size,
            )
    return enhanced


@torch.no_grad()
def predict_on_set(model, emg, angle, idx, config, device):
    dataset = KinematicsDataset(emg[idx], angle[idx])
    loader = DataLoader(dataset, batch_size=config["regressor_batch_size"], shuffle=False, num_workers=0)
    model.eval()
    preds, targets = [], []
    for batch in loader:
        preds.append(model(batch["emg"].to(device)).cpu().numpy())
        targets.append(batch["angle"].numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def evaluate_subsets(pred_np, target_np):
    assert_key10_target(pred_np, "Exp3 prediction")
    assert_key10_target(target_np, "Exp3 target")
    p_flat = pred_np.reshape(-1, pred_np.shape[-1])
    t_flat = target_np.reshape(-1, target_np.shape[-1])

    results = {}
    for name, dims in ANGLE_SUBSETS.items():
        p = p_flat[:, dims]
        t = t_flat[:, dims]
        rmse_val = float(np.sqrt(np.mean((p - t) ** 2)))
        mae_val = float(np.mean(np.abs(p - t)))
        corrs, r2_vals, nan_dims = [], [], []
        for i, dim in enumerate(dims):
            ss_res = float(np.sum((p[:, i] - t[:, i]) ** 2))
            ss_tot = float(np.sum((t[:, i] - t[:, i].mean()) ** 2))
            if ss_tot < 1e-8:
                corrs.append(float("nan"))
                r2_vals.append(float("nan"))
                nan_dims.append(dim)
            elif np.std(p[:, i]) > 1e-8 and np.std(t[:, i]) > 1e-8:
                c = _safe_pearson(p[:, i], t[:, i])
                corrs.append(c if np.isfinite(c) else float("nan"))
                r2_vals.append(1.0 - ss_res / ss_tot)
            else:
                corrs.append(float("nan"))
                r2_vals.append(float("nan"))
                nan_dims.append(dim)
        pearson = float(np.nanmean(corrs)) if any(np.isfinite(c) for c in corrs) else float("nan")
        r2 = float(np.nanmean(r2_vals)) if any(np.isfinite(v) for v in r2_vals) else float("nan")
        results[name] = {
            "rmse": rmse_val,
            "mae": mae_val,
            "pearson": pearson,
            "r2": r2,
            "nan_dims": nan_dims,
        }
    return results


def _continuous_output_config(config: dict) -> dict:
    settings = {
        "enabled": bool(config.get("regressor_output_postprocess_enabled", True)),
        "fusion": str(config.get("regressor_output_postprocess_fusion", "uniform_overlap_add")),
        "filter_kind": str(config.get("regressor_output_postprocess_filter", "savgol")),
        "savgol_window": int(config.get("regressor_output_postprocess_savgol_window", 13)),
        "savgol_polyorder": int(config.get("regressor_output_postprocess_savgol_polyorder", 2)),
        "lowpass_hz": float(config.get("regressor_output_postprocess_lowpass_hz", 8.0)),
        "calibration": str(config.get("regressor_output_postprocess_calibration", "none")),
    }
    if settings["fusion"] not in {"uniform_overlap_add", "triangular_overlap_add"}:
        raise ValueError(f"Unsupported continuous-output fusion: {settings['fusion']}")
    if settings["filter_kind"] not in {"none", "savgol", "butterworth"}:
        raise ValueError(f"Unsupported continuous-output filter: {settings['filter_kind']}")
    if settings["filter_kind"] == "savgol" and settings["savgol_window"] < 3:
        raise ValueError("regressor_output_postprocess_savgol_window must be at least 3")
    if settings["filter_kind"] == "butterworth" and settings["lowpass_hz"] <= 0:
        raise ValueError("regressor_output_postprocess_lowpass_hz must be positive")
    if settings["calibration"] not in {"none", "affine"}:
        raise ValueError(f"Unsupported continuous-output calibration: {settings['calibration']}")
    return settings


def continuous_output_label(config: dict) -> str:
    settings = _continuous_output_config(config)
    if settings["filter_kind"] == "savgol":
        filter_label = f"SG({settings['savgol_window']})"
    elif settings["filter_kind"] == "butterworth":
        filter_label = f"Butterworth({settings['lowpass_hz']:g}Hz)"
    else:
        filter_label = "no_filter"
    calibration_label = " + validation_affine" if settings["calibration"] == "affine" else ""
    return f"continuous output: {settings['fusion']} + {filter_label}{calibration_label}"


def _fit_channelwise_affine(pred: np.ndarray, target: np.ndarray) -> dict:
    """Fit a bounded per-channel correction using validation predictions only."""
    x = np.asarray(pred, dtype=np.float64).reshape(-1, pred.shape[-1])
    y = np.asarray(target, dtype=np.float64).reshape(-1, target.shape[-1])
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    centered_x = x - x_mean
    denominator = np.sum(centered_x ** 2, axis=0)
    numerator = np.sum(centered_x * (y - y_mean), axis=0)
    gain = np.ones_like(x_mean)
    valid = denominator > 1e-10
    gain[valid] = numerator[valid] / denominator[valid]
    gain = np.clip(gain, 0.25, 4.0)
    bias = y_mean - gain * x_mean
    return {
        "gain": gain.astype(np.float32),
        "bias": bias.astype(np.float32),
        "fit_scope": "validation_predictions_only",
        "gain_min": float(gain.min()),
        "gain_median": float(np.median(gain)),
        "gain_max": float(gain.max()),
    }


def _apply_channelwise_affine(pred: np.ndarray, calibration: dict) -> np.ndarray:
    return np.asarray(pred) * calibration["gain"][None, None, :] + calibration["bias"][None, None, :]


def _exercise_separated_window_starts(metadata: dict, config: dict) -> np.ndarray:
    """Map within-exercise starts to one sparse timeline without cross-exercise fusion."""
    exercises = np.asarray(metadata["exercise"], dtype=np.int16)
    starts = np.asarray(metadata["start"], dtype=np.int64)
    if len(exercises) != len(starts):
        raise ValueError("Kinematics window metadata is inconsistent")
    offsets = {}
    cursor = 0
    gap = 2 * int(config["window_size"])
    for exercise in np.unique(exercises):
        local = starts[exercises == exercise]
        offsets[int(exercise)] = cursor
        cursor += int(local.max()) + gap
    return np.asarray([start + offsets[int(exercise)] for exercise, start in zip(exercises, starts)], dtype=np.int64)


def _continuous_output_from_groups(group_preds: dict, target: np.ndarray,
                                   test_window_starts: np.ndarray,
                                   dynamic_test_indices: np.ndarray,
                                   config: dict,
                                   group_calibrations: dict | None = None) -> dict:
    """Apply the fixed prediction-only continuous-output rule to every Exp3 group."""
    settings = _continuous_output_config(config)
    if not settings["enabled"]:
        return {"enabled": False, "group_predictions": group_preds, "group_results": {}, "payload": {}}
    processed_groups, group_results, payload = {}, {}, {}
    reference = None
    group_calibrations = group_calibrations or {}
    for group_name, pred in group_preds.items():
        calibration = group_calibrations.get(group_name)
        working_pred = np.asarray(pred)
        calibration_metadata = None
        if settings["calibration"] == "affine":
            if calibration is None:
                calibration_metadata = {"applied": False, "reason": "validation_calibration_unavailable"}
            else:
                working_pred = _apply_channelwise_affine(working_pred, calibration)
                calibration_metadata = {
                    "applied": True,
                    "fit_scope": calibration["fit_scope"],
                    "gain_min": calibration["gain_min"],
                    "gain_median": calibration["gain_median"],
                    "gain_max": calibration["gain_max"],
                }
        packed = postprocess_window_predictions(
            working_pred, target, test_window_starts, dynamic_test_indices,
            savgol_window=settings["savgol_window"],
            polyorder=settings["savgol_polyorder"],
            fusion=settings["fusion"],
            filter_kind=settings["filter_kind"],
            target_fs=int(config["target_fs"]),
            lowpass_hz=settings["lowpass_hz"],
        )
        continuous_pred = packed["continuous_prediction"]
        continuous_target = packed["continuous_target"]
        continuous_times = packed["continuous_time_indices"]
        dynamic_mask = packed["continuous_dynamic_mask"]
        overlap_counts = packed["continuous_overlap_counts"]
        if reference is None:
            reference = (continuous_target, continuous_times, dynamic_mask, overlap_counts)
            payload.update({
                "test_window_starts": np.asarray(test_window_starts, dtype=np.int64),
                "continuous_target": continuous_target,
                "continuous_time_indices": continuous_times,
                "continuous_dynamic_mask": dynamic_mask,
                "continuous_overlap_counts": overlap_counts,
                "continuous_output_fusion": np.asarray(settings["fusion"]),
                "continuous_output_filter": np.asarray(settings["filter_kind"]),
                "continuous_output_savgol_window": np.asarray(settings["savgol_window"], dtype=np.int64),
                "continuous_output_savgol_polyorder": np.asarray(settings["savgol_polyorder"], dtype=np.int64),
                "continuous_output_lowpass_hz": np.asarray(settings["lowpass_hz"], dtype=np.float32),
                "continuous_output_calibration": np.asarray(settings["calibration"]),
            })
        else:
            ref_target, ref_times, ref_dynamic, ref_overlap_counts = reference
            if not (np.array_equal(continuous_times, ref_times) and np.array_equal(dynamic_mask, ref_dynamic)
                    and np.array_equal(overlap_counts, ref_overlap_counts)
                    and np.allclose(continuous_target, ref_target)):
                raise RuntimeError("Continuous target/time alignment differs between Exp3 groups")
        processed_groups[group_name] = packed["processed_windows"]
        payload[f"pred_{group_name}_continuous_windows"] = packed["processed_windows"]
        payload[f"pred_{group_name}_continuous"] = continuous_pred
        if calibration is not None:
            payload[f"continuous_calibration_{group_name}_gain"] = calibration["gain"]
            payload[f"continuous_calibration_{group_name}_bias"] = calibration["bias"]
        group_results[group_name] = {
            "continuous_subsets": evaluate_subsets(continuous_pred[None, :, :], continuous_target[None, :, :]),
            "continuous_dynamic_subsets": (
                evaluate_subsets(continuous_pred[dynamic_mask][None, :, :], continuous_target[dynamic_mask][None, :, :])
                if np.any(dynamic_mask) else {}
            ),
            "continuous_trajectory_quality": continuous_trajectory_quality(
                continuous_pred, continuous_target, continuous_times
            ),
            "continuous_overlap_consistency": window_overlap_prediction_consistency(
                working_pred, test_window_starts
            ),
            "continuous_output_postprocess": packed["metadata"],
            "continuous_output_calibration": calibration_metadata,
        }
    return {
        "enabled": True,
        "settings": settings,
        "group_predictions": processed_groups,
        "group_results": group_results,
        "payload": payload,
    }


def compute_dynamic_angle_scores(angle_windows: np.ndarray, score_name: str = "global_max_ptp") -> np.ndarray:
    """Return one dynamic-angle score per window without changing the split."""
    if angle_windows is None or len(angle_windows) == 0:
        return np.asarray([], dtype=np.float32)
    channel_ptp = np.ptp(angle_windows, axis=1)
    if score_name == "global_mean_ptp":
        scores = channel_ptp.mean(axis=1)
    elif score_name == "global_max_ptp":
        scores = channel_ptp.max(axis=1)
    else:
        raise ValueError(f"Unknown regressor_dynamic_score: {score_name}")
    return np.asarray(scores, dtype=np.float32)


def _score_distribution(scores: np.ndarray) -> dict:
    if scores is None or len(scores) == 0:
        return {"median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(scores)),
        "p90": float(np.percentile(scores, 90)),
        "max": float(np.max(scores)),
    }


def dynamic_split_diagnostics(angle: np.ndarray, split_indices: dict, config: dict) -> dict:
    score_name = str(config.get("regressor_dynamic_score", "global_max_ptp"))
    threshold = float(config.get("regressor_dynamic_min_ptp", 0.05))
    min_train_windows = int(config.get("regressor_dynamic_min_train_windows", 8))
    splits = {}

    for split_name, idx in split_indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        scores = compute_dynamic_angle_scores(angle[idx], score_name)
        dynamic_local = np.flatnonzero(scores >= threshold).astype(np.int64)
        total = int(len(scores))
        count = int(len(dynamic_local))
        splits[split_name] = {
            "dynamic": count,
            "total": total,
            "ratio": float(count / total) if total > 0 else float("nan"),
            "score_summary": _score_distribution(scores),
            "dynamic_score_summary": _score_distribution(scores[dynamic_local]),
            "dynamic_indices": dynamic_local.tolist(),
        }

    train_dynamic = int(splits.get("train", {}).get("dynamic", 0))
    return {
        "enabled": bool(config.get("regressor_dynamic_subset_enabled", True)),
        "score": score_name,
        "dynamic_min_ptp": threshold,
        "low_dynamic_train_min_windows": min_train_windows,
        "low_dynamic_train_coverage": bool(train_dynamic < min_train_windows),
        "splits": splits,
    }


def _format_dynamic_split(split: dict) -> str:
    summary = split.get("score_summary", {})
    return (
        f"{split.get('dynamic', 0)}/{split.get('total', 0)} "
        f"({split.get('ratio', float('nan')):.3f}) "
        f"score med/p90/max="
        f"{_metric_str(summary.get('median', float('nan')))}"
        f"/{_metric_str(summary.get('p90', float('nan')))}"
        f"/{_metric_str(summary.get('max', float('nan')))}"
    )


def print_dynamic_diagnostics(subject_id: int, diagnostics: dict) -> None:
    splits = diagnostics.get("splits", {})
    parts = []
    for name in ("train", "val", "test"):
        if name in splits:
            parts.append(f"{name}={_format_dynamic_split(splits[name])}")
    suffix = " LOW_DYNAMIC_TRAIN_COVERAGE" if diagnostics.get("low_dynamic_train_coverage") else ""
    print(
        f"  Dynamic-angle coverage S{subject_id:02d} "
        f"threshold={diagnostics.get('dynamic_min_ptp')} "
        f"score={diagnostics.get('score')} :: " + "  ".join(parts) + suffix
    )


def _metric_str(v):
    return f"{v:.4f}" if np.isfinite(v) else "nan"


def print_subject_results(label, subsets):
    parts = []
    for k, v in subsets.items():
        s = (f"{k}: RMSE={v['rmse']:.4f} MAE={v['mae']:.4f} "
             f"CC={_metric_str(v['pearson'])} R2={_metric_str(v['r2'])}")
        if v.get("nan_dims"):
            s += f"(nan dims={v['nan_dims']})"
        parts.append(s)
    print(f"    [{label}] {'  '.join(parts)}")


def _safe_dim_metrics(pred: np.ndarray, target: np.ndarray, dims: list[int]) -> dict:
    p = pred[:, dims].reshape(-1, len(dims))
    t = target[:, dims].reshape(-1, len(dims))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    corrs = []
    for i in range(len(dims)):
        if np.std(p[:, i]) > 1e-8 and np.std(t[:, i]) > 1e-8:
            corrs.append(_safe_pearson(p[:, i], t[:, i]))
    return {"rmse": rmse, "pearson": float(np.nanmean(corrs)) if corrs else float("nan")}


def _contiguous_local_groups(time_indices: np.ndarray) -> list[np.ndarray]:
    if len(time_indices) == 0:
        return []
    local_indices = np.arange(len(time_indices), dtype=np.int64)
    split_points = np.flatnonzero(np.diff(time_indices) > 1) + 1
    return [group for group in np.split(local_indices, split_points) if len(group)]


def build_abc_visualization_selection(
    target: np.ndarray,
    time_indices: np.ndarray,
    overlap_counts: np.ndarray,
    dynamic_mask: np.ndarray,
    config: dict,
    low_dynamic_coverage: bool = False,
) -> dict:
    """Select non-overlapping, physically continuous 1280-point test sections."""
    target = np.asarray(target)
    times = np.asarray(time_indices, dtype=np.int64)
    overlaps = np.asarray(overlap_counts, dtype=np.int16)
    dynamics = np.asarray(dynamic_mask, dtype=bool)
    if target.ndim != 2 or len(target) != len(times):
        raise ValueError("continuous target and time indices must be aligned 2-D arrays")
    if len(overlaps) != len(times) or len(dynamics) != len(times):
        raise ValueError("continuous overlap and dynamic masks must align with time indices")

    n_requested = max(0, int(config.get("regressor_viz_trials_per_subject", 2)))
    dynamic_top_k = max(0, int(config.get("regressor_dynamic_viz_top_k", n_requested)))
    n_points = int(config.get("regressor_continuous_viz_points", 1280))
    stride = int(config["stride"])
    target_fs = int(config["target_fs"])
    min_overlap = max(1, int(config.get("regressor_continuous_viz_min_overlap", 2)))
    threshold = float(config.get("regressor_dynamic_min_ptp", 0.05))
    score_name = str(config.get("regressor_dynamic_score", "global_max_ptp"))

    base = {
        "mode": "continuous_dynamic_preferred",
        "score": score_name,
        "dynamic_min_ptp": threshold,
        "n_requested": n_requested,
        "dynamic_viz_top_k": dynamic_top_k,
        "segment_points": n_points,
        "segment_seconds": float(n_points / target_fs),
        "target_fs": target_fs,
        "min_overlap_windows": min_overlap,
        "n_contiguous_sections": 0,
        "n_eligible_segments": 0,
        "n_dynamic_segments": 0,
        "selected_indices": [],
        "selected_time_indices": [],
        "selected_scores": [],
        "selected_modes": [],
        "selected_dynamic_ranks": [],
        "selected_segments": [],
        "low_dynamic_coverage": bool(low_dynamic_coverage),
    }
    if n_requested <= 0 or len(target) < n_points:
        return base

    candidates = []
    contiguous_groups = _contiguous_local_groups(times)
    base["n_contiguous_sections"] = int(len(contiguous_groups))
    for group in contiguous_groups:
        if len(group) < n_points:
            continue
        for left in range(0, len(group) - n_points + 1, stride):
            local = group[left:left + n_points]
            if len(local) != n_points or int(np.min(overlaps[local])) < min_overlap:
                continue
            segment = target[local]
            score = float(compute_dynamic_angle_scores(segment[None, ...], score_name)[0])
            candidates.append({
                "local_start": int(local[0]),
                "local_end_exclusive": int(local[-1] + 1),
                "time_start": int(times[local[0]]),
                "time_end_exclusive": int(times[local[-1]] + 1),
                "dynamic_score": score,
                "dynamic_sample_ratio": float(np.mean(dynamics[local])),
                "min_overlap": int(np.min(overlaps[local])),
                "mean_overlap": float(np.mean(overlaps[local])),
            })

    base["n_eligible_segments"] = int(len(candidates))
    dynamic_candidates = [item for item in candidates if item["dynamic_score"] >= threshold]
    base["n_dynamic_segments"] = int(len(dynamic_candidates))
    ranked_dynamic = sorted(dynamic_candidates, key=lambda item: item["dynamic_score"], reverse=True)
    dynamic_rank = {item["local_start"]: rank for rank, item in enumerate(ranked_dynamic, start=1)}

    selected = []
    selected_intervals = []
    for mode, ranked in (
        ("dynamic-selected", ranked_dynamic[:min(dynamic_top_k, n_requested)]),
        ("full-selected", sorted(candidates, key=lambda item: item["dynamic_score"], reverse=True)),
    ):
        for item in ranked:
            interval = (item["time_start"], item["time_end_exclusive"])
            if any(interval[0] < used_end and used_start < interval[1] for used_start, used_end in selected_intervals):
                continue
            item = dict(item)
            item["selection_mode"] = mode
            item["dynamic_rank"] = dynamic_rank.get(item["local_start"])
            selected.append(item)
            selected_intervals.append(interval)
            if len(selected) >= n_requested:
                break
        if len(selected) >= n_requested:
            break

    base["selected_segments"] = selected
    base["selected_indices"] = [item["local_start"] for item in selected]
    base["selected_time_indices"] = [item["time_start"] for item in selected]
    base["selected_scores"] = [item["dynamic_score"] for item in selected]
    base["selected_modes"] = [item["selection_mode"] for item in selected]
    base["selected_dynamic_ranks"] = [item["dynamic_rank"] for item in selected]
    return base


ANGLE_PLOT_STYLES = {
    "target": ("#111111", "Ground Truth", "-", 1.45, 0.95),
    "A": ("#1f77b4", "A Raw", "-", 1.05, 0.88),
    "B": ("#d62728", "B Healthy-prior", "-", 1.05, 0.82),
}


def _plot_dim_trace(ax, xs: np.ndarray, dim: int, target_trial: np.ndarray,
                    group_trials: dict) -> None:
    color, label, linestyle, width, alpha = ANGLE_PLOT_STYLES["target"]
    ax.plot(xs, target_trial[:, dim], color=color, lw=width, alpha=alpha,
            linestyle=linestyle, label=label)
    for grp in ("A", "B"):
        pred = group_trials.get(grp)
        if pred is None:
            continue
        color, label, linestyle, width, alpha = ANGLE_PLOT_STYLES[grp]
        ax.plot(xs, pred[:, dim], color=color, lw=width, alpha=alpha,
                linestyle=linestyle, label=label)
    ax.set_title(KEY10_CHANNEL_NAMES[dim], fontsize=9.5, fontweight="bold", pad=3)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)


def _save_abc_anatomy_plot(path: Path, target_segment: np.ndarray, group_segments: dict,
                           group_spec: dict, group_name: str, subject_id: int,
                           segment_rank: int, selection_info: dict,
                           target_fs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = group_spec["shape"]
    fig_w, fig_h = group_spec.get(
        "figsize",
        (max(8.8, 4.2 * n_cols), max(3.4, 2.25 * n_rows + 1.4)),
    )
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(18.0, fig_w), fig_h), sharex=True)
    axes_arr = np.asarray(axes, dtype=object).reshape(n_rows, n_cols)
    xs = np.arange(target_segment.shape[0], dtype=np.float32) / float(target_fs)

    dims = []
    for slot, (anatomy_label, dim) in enumerate(group_spec["items"]):
        row, col = divmod(slot, n_cols)
        ax = axes_arr[row, col]
        _plot_dim_trace(ax, xs, dim, target_segment, group_segments)
        ax.set_ylabel(anatomy_label if col == 0 else "", fontsize=9)
        dims.append(dim)
    for slot in range(len(group_spec["items"]), n_rows * n_cols):
        row, col = divmod(slot, n_cols)
        axes_arr[row, col].axis("off")
    for ax in axes_arr[-1, :]:
        if ax.has_data():
            ax.set_xlabel("time within continuous test segment (s)", fontsize=9)

    handles, labels = axes_arr.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.975))
    metric_parts = []
    for grp in ("A", "B"):
        pred = group_segments.get(grp)
        if pred is not None:
            metric_parts.append(f"{grp} RMSE={_safe_dim_metrics(pred, target_segment, dims)['rmse']:.4f}")
    header = (
        f"DB3 S{subject_id:02d} | Key10 proxy | {len(target_segment) / target_fs:.2f}s "
        f"continuous offline test trajectory | Segment {segment_rank:02d} | "
        f"time idx {selection_info['time_start']}-{selection_info['time_end_exclusive'] - 1} | "
        f"{group_name.replace('_', ' ').title()} A/B comparison"
    )
    detail_parts = [
        f"{selection_info['selection_mode']} | dynamic score={_metric_str(selection_info['dynamic_score'])} | "
        f"dynamic rank={selection_info['dynamic_rank'] if selection_info['dynamic_rank'] is not None else 'NA'}",
        f"overlap windows min/mean={selection_info['min_overlap']}/{selection_info['mean_overlap']:.2f}",
    ]
    if selection_info.get("low_dynamic_coverage"):
        detail_parts.append("low dynamic coverage")
    if selection_info.get("output_postprocess"):
        detail_parts.append(str(selection_info["output_postprocess"]))
    fig.suptitle(header + "\n" + "    ".join(metric_parts) + "\n" + " | ".join(detail_parts),
                 fontsize=12.5, fontweight="bold", y=0.955)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.07, top=0.79,
                        hspace=0.55, wspace=0.22)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _score_for_filename(score: float) -> str:
    if not np.isfinite(score):
        return "nan"
    return f"{float(score):.3f}".replace(".", "p")


def save_abc_comparison_figures(
    subject_id: int,
    group_predictions: dict,
    continuous_target: np.ndarray,
    continuous_time_indices: np.ndarray,
    continuous_overlap_counts: np.ndarray,
    continuous_dynamic_mask: np.ndarray,
    out_dir: Path,
    n_trials: int = 2,
    config: dict | None = None,
    low_dynamic_coverage: bool = False,
    output_postprocess: str | None = None,
    return_selection: bool = False,
):
    cfg = dict(config or {})
    if "regressor_viz_trials_per_subject" not in cfg:
        cfg["regressor_viz_trials_per_subject"] = n_trials
    target = np.asarray(continuous_target)
    times = np.asarray(continuous_time_indices, dtype=np.int64)
    overlaps = np.asarray(continuous_overlap_counts, dtype=np.int16)
    dynamics = np.asarray(continuous_dynamic_mask, dtype=bool)
    required_groups = {"A", "B"}
    if target.ndim != 2 or not required_groups.issubset(group_predictions):
        raise ValueError("continuous A/B figures require target and A/B continuous predictions")
    if any(np.asarray(group_predictions[group]).shape != target.shape for group in required_groups):
        raise ValueError("continuous A/B predictions must align with the continuous target")
    selection = build_abc_visualization_selection(
        target, times, overlaps, dynamics, cfg, low_dynamic_coverage
    )
    subject_dir = out_dir / f"S{subject_id:02d}"
    if subject_dir.exists():
        for stale_path in subject_dir.glob(f"S{subject_id:02d}_*_AB.png"):
            stale_path.unlink()
    saved = []
    target_fs = int(cfg["target_fs"])
    for rank, segment in enumerate(selection["selected_segments"], start=1):
        left = segment["local_start"]
        right = segment["local_end_exclusive"]
        selection_info = {
            **segment,
            "low_dynamic_coverage": selection.get("low_dynamic_coverage", False),
            "output_postprocess": output_postprocess,
        }
        group_segments = {group: np.asarray(values)[left:right] for group, values in group_predictions.items()}
        score_tag = _score_for_filename(float(segment["dynamic_score"]))
        for group_name, group_spec in ANGLE_PLOT_GROUPS.items():
            path = (
                subject_dir /
                f"S{subject_id:02d}_continuous_{rank:02d}_"
                f"t{segment['time_start']:06d}-{segment['time_end_exclusive'] - 1:06d}_"
                f"score{score_tag}_{group_name}_AB.png"
            )
            _save_abc_anatomy_plot(
                path, target[left:right], group_segments, group_spec,
                group_name, subject_id, rank, selection_info, target_fs,
            )
            saved.append(str(path))
    return (saved, selection) if return_selection else saved


def print_tcn_info(config, n_emg_channels=12, fs=200):
    if int(config["regressor_n_angle_channels"]) != KEY10_DIM:
        raise ValueError(f"Exp3 requires fixed Key10 output dimension {KEY10_DIM}.")
    k = config["regressor_kernel_size"]
    n_layers = config["regressor_n_layers"]
    model = KinematicTCN(
        n_emg_channels=n_emg_channels,
        n_angle_channels=config["regressor_n_angle_channels"],
        hidden_dim=config["regressor_hidden_dim"],
        n_layers=n_layers,
        kernel_size=k,
        dropout=config["regressor_model_dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    rf = 1 + 2 * (k - 1) * sum(2 ** i for i in range(n_layers))
    print(f"\n{'='*70}")
    print(f"KinematicTCN Key10={KEY10_DIM}  hidden={config['regressor_hidden_dim']}  "
          f"n_layers={n_layers}  kernel={k}  non-causal same-padding")
    print(f"  Total params : {total_params:,}")
    print(f"  Receptive field: {rf} samples = {rf/fs*1000:.0f} ms @ {fs} Hz")
    print(f"{'='*70}\n")


def _fmt_nan_summary(vals):
    arr = np.asarray(vals, dtype=float)
    n = int(np.sum(np.isfinite(arr)))
    if n == 0:
        return "nan+/-nan(n=0)"
    return f"{np.nanmean(arr):.4f}+/-{np.nanstd(arr):.4f}(n={n})"


def aggregate_and_print_summary(all_subject_results, groups=("A", "B")):
    print(f"\n{'='*70}")
    print("Cross-subject summary (nanmean +/- nanstd)")
    print(f"{'='*70}")
    for grp in groups:
        rows = [r[grp] for r in all_subject_results if r.get(grp) is not None]
        if not rows:
            continue
        print(f"  Group {grp}:")
        for subset in ANGLE_SUBSETS:
            sub_rows = [r[subset] for r in rows if r.get(subset)]
            if not sub_rows:
                continue
            rmses = [v["rmse"] for v in sub_rows]
            maes = [v["mae"] for v in sub_rows]
            pearsons = [v["pearson"] for v in sub_rows]
            r2s = [v["r2"] for v in sub_rows]
            print(
                f"    {subset:<8}  RMSE={_fmt_nan_summary(rmses)}  "
                f"MAE={_fmt_nan_summary(maes)}  "
                f"CC={_fmt_nan_summary(pearsons)}  "
                f"R2={_fmt_nan_summary(r2s)}"
            )


def aggregate_continuous_summary(report: dict, groups=("A", "B")) -> None:
    print("\n" + "=" * 70)
    print("Cross-subject continuous-output summary (nanmean +/- nanstd)")
    print("=" * 70)
    for grp in groups:
        rows = [
            subject.get("groups", {}).get(grp, {}) for subject in report.get("subjects", [])
            if subject.get("status") == "ok" and subject.get("groups", {}).get(grp, {}).get("continuous_subsets")
        ]
        if not rows:
            continue
        print(f"  Group {grp}:")
        for subset in ANGLE_SUBSETS:
            vals = [row["continuous_subsets"].get(subset) for row in rows if row["continuous_subsets"].get(subset)]
            if not vals:
                continue
            print(
                f"    {subset:<8} RMSE={_fmt_nan_summary([v['rmse'] for v in vals])} "
                f"MAE={_fmt_nan_summary([v['mae'] for v in vals])} "
                f"CC={_fmt_nan_summary([v['pearson'] for v in vals])} "
                f"R2={_fmt_nan_summary([v['r2'] for v in vals])}"
            )


def print_dynamic_compact_summary(report: dict) -> None:
    rows = []
    for subject in report.get("subjects", []):
        if subject.get("status") != "ok":
            continue
        diag = subject.get("dynamic_angle_coverage", {})
        splits = diag.get("splits", {})
        test = splits.get("test", {})
        train = splits.get("train", {})
        if not test:
            continue
        rows.append((
            int(subject.get("subject_id", 0)),
            float(train.get("ratio", float("nan"))),
            int(test.get("dynamic", 0)),
            int(test.get("total", 0)),
            float(test.get("ratio", float("nan"))),
            bool(subject.get("low_dynamic_train_coverage", False)),
        ))
    if not rows:
        return
    print(f"\n{'='*70}")
    print("Dynamic-angle compact summary")
    print(f"{'='*70}")
    print("  subject  train_ratio  test_dynamic/test_total  test_ratio  display_note")
    for sid, train_ratio, test_dyn, test_total, test_ratio, low_train in rows:
        if low_train:
            note = "low train dynamic coverage"
        elif test_ratio >= 0.45:
            note = "good continuous-trajectory candidate"
        elif test_ratio >= 0.15:
            note = "moderate dynamic coverage"
        else:
            note = "sparse dynamic coverage"
        print(
            f"  S{sid:02d}      {_metric_str(train_ratio):>8}     "
            f"{test_dyn:>4}/{test_total:<4}             {_metric_str(test_ratio):>8}  {note}"
        )


def evaluate_group(label, emg, angle, train_idx, val_idx, test_idx, config, device,
                   ckpt_path, subject_id, metadata=None):
    print(f"  Training Group {label} TCN from scratch...")
    model, best_val, epochs = train_tcn_on_emg(emg, angle, train_idx, val_idx, config, device)
    torch.save(model.state_dict(), ckpt_path)
    pred, tgt = predict_on_set(model, emg, angle, test_idx, config, device)
    output_calibration = None
    settings = _continuous_output_config(config)
    if settings["enabled"] and settings["calibration"] == "affine":
        val_pred, val_target = predict_on_set(model, emg, angle, val_idx, config, device)
        output_calibration = _fit_channelwise_affine(val_pred, val_target)
    subsets = evaluate_subsets(pred, tgt)
    print(f"  Group {label} best_val={best_val:.5f} epochs={epochs}")
    print_subject_results(label, subsets)
    result = {
        "best_val_loss": float(best_val),
        "epochs": int(epochs),
        "subsets": subsets,
    }
    if output_calibration is not None:
        result["continuous_output_calibration"] = {
            "fit_scope": output_calibration["fit_scope"],
            "gain_min": output_calibration["gain_min"],
            "gain_median": output_calibration["gain_median"],
            "gain_max": output_calibration["gain_max"],
        }
    if metadata:
        result.update(metadata)
    return result, subsets, pred, tgt, output_calibration


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(log_path: Path, message: str) -> None:
    line = f"[{_now()}] {message}"
    print(line)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def _assert_key10_report(report: dict, context: str) -> None:
    metadata = report.get("angle_target")
    expected = key10_target_metadata()
    if metadata != expected:
        raise ValueError(
            f"{context} is a legacy or incompatible angle report. Create a new run for "
            f"the fixed {KEY10_DIM}-D Key10 target."
        )


def _load_existing_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    _assert_key10_report(report, str(path))
    return report


def _base_report() -> dict:
    return {
        "design": {
            "A": "raw train/val/test; TCN trained from scratch",
            "B": "healthy-prior MCIA enhanced train/val/test; TCN trained from scratch",
        },
        "angle_target": key10_target_metadata(),
        "subjects": [],
    }


def _upsert_subject(report: dict, subject_result: dict) -> None:
    sid = int(subject_result["subject_id"])
    subjects = [s for s in report.get("subjects", []) if int(s.get("subject_id", -1)) != sid]
    subjects.append(subject_result)
    subjects.sort(key=lambda row: int(row.get("subject_id", 0)))
    report["subjects"] = subjects


def _save_progress(out_path: Path, report: dict, start_ts: float) -> None:
    report["elapsed_min"] = (time.time() - start_ts) / 60.0
    save_json(out_path, report)


def _merge_prediction_payload(npz_path: Path, updates: dict) -> None:
    """Add deterministic postprocessing fields while preserving legacy prediction fields."""
    with np.load(npz_path) as data:
        payload = {key: data[key] for key in data.files}
    payload.update(updates)
    np.savez(npz_path, **payload)


def _prediction_has_groups(npz_path: Path, run_groups: set[str]) -> bool:
    if not npz_path.exists():
        return False
    with np.load(npz_path) as data:
        assert_key10_prediction_payload(data, str(npz_path))
        return all(f"pred_{grp}" in data for grp in run_groups)


def _subject_from_prediction(
    subject_id: int,
    npz_path: Path,
    run_groups: set[str],
    dynamic_test_indices: np.ndarray | None = None,
    test_window_starts: np.ndarray | None = None,
    config: dict | None = None,
) -> tuple[dict, dict, dict]:
    with np.load(npz_path) as data:
        assert_key10_prediction_payload(data, str(npz_path))
        target = data["target"]
        if dynamic_test_indices is None and "dynamic_test_indices" in data:
            dynamic_test_indices = np.asarray(data["dynamic_test_indices"], dtype=np.int64)
        dynamic_test_indices = np.asarray(
            dynamic_test_indices if dynamic_test_indices is not None else [], dtype=np.int64
        )
        subj_subsets, groups, raw_preds, group_calibrations = {}, {}, {}, {}
        for grp in sorted(run_groups):
            pred_key = f"pred_{grp}"
            if pred_key not in data:
                continue
            pred = data[pred_key]
            raw_preds[grp] = pred
            gain_key = f"continuous_calibration_{grp}_gain"
            bias_key = f"continuous_calibration_{grp}_bias"
            if gain_key in data.files and bias_key in data.files:
                gain = np.asarray(data[gain_key], dtype=np.float32)
                bias = np.asarray(data[bias_key], dtype=np.float32)
                group_calibrations[grp] = {
                    "gain": gain,
                    "bias": bias,
                    "fit_scope": "validation_predictions_only",
                    "gain_min": float(gain.min()),
                    "gain_median": float(np.median(gain)),
                    "gain_max": float(gain.max()),
                }
            subsets = evaluate_subsets(pred, target)
            subj_subsets[grp] = subsets
            group_result = {"subsets": subsets, "resumed_from_prediction": True}
            if len(dynamic_test_indices) > 0:
                group_result["dynamic_subsets"] = evaluate_subsets(
                    pred[dynamic_test_indices], target[dynamic_test_indices]
                )
            groups[grp] = group_result
    updates = {}
    continuous_output = None
    if raw_preds and test_window_starts is not None and config is not None:
        continuous_output = _continuous_output_from_groups(
            raw_preds, target, test_window_starts, dynamic_test_indices, config,
            group_calibrations=group_calibrations,
        )
        if continuous_output.get("enabled"):
            updates = continuous_output["payload"]
            for grp, details in continuous_output["group_results"].items():
                groups[grp].update(details)
    subject = {
        "subject_id": int(subject_id), "status": "ok", "angle_target": key10_target_metadata(), "groups": groups,
        "prediction_file": str(npz_path), "resumed_from_prediction": True,
    }
    if continuous_output is not None:
        subject["continuous_output"] = continuous_output.get("settings", {"enabled": False})
    return subject, subj_subsets, updates


def _subsets_from_report(report: dict) -> list[dict]:
    rows = []
    for subject in report.get("subjects", []):
        if subject.get("status") != "ok":
            continue
        subj = {}
        for grp, result in subject.get("groups", {}).items():
            if isinstance(result, dict) and "subsets" in result:
                subj[grp] = result["subsets"]
        if subj:
            rows.append(subj)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Exp3 DB3 angle estimation")
    parser.add_argument(
        "--groups",
        default="A,B",
        help="Comma-separated groups to run. A=raw, B=healthy-prior enhanced.",
    )
    args = parser.parse_args()
    run_groups = {g.strip().upper() for g in args.groups.split(",") if g.strip()}
    valid_groups = {"A", "B"}
    unknown = run_groups - valid_groups
    if unknown:
        raise ValueError(f"Unknown Exp3 groups: {sorted(unknown)}")

    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    device = config["device"]
    if int(config["regressor_n_angle_channels"]) != KEY10_DIM:
        raise ValueError(f"config.yaml must set exp3_regressor.n_angle_channels to {KEY10_DIM} for Key10.")
    set_seed(int(config["regressor_random_seed"]))

    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    out_path = Path(config["regressor_results_path"])
    run_dir = Path(config.get("run_dir", out_path.parent.parent.parent))
    ckpt_dir = run_dir / "03_angle_prediction" / "checkpoints"
    viz_dir = run_dir / "03_angle_prediction" / "figures" / "comparison"
    summary_dir = run_dir / "03_angle_prediction" / "figures" / "summary"
    pred_dir = run_dir / "03_angle_prediction" / "predictions"
    log_path = run_dir / "03_angle_prediction" / "logs" / "exp3_subjects.log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print_tcn_info(config, n_emg_channels=config.get("n_channels", 12), fs=config["target_fs"])

    healthy_mcia, healthy_ckpt = (None, None)
    if "B" in run_groups:
        healthy_mcia, healthy_ckpt = load_healthy_prior_mcia(config, device)

    report = _load_existing_report(out_path) or _base_report()
    start_ts = time.time()
    print("=" * 70)
    print("[Exp3] DB3 fixed-Key10 proxy angle estimation - groups A / B")
    print("=" * 70)
    _log(log_path, f"Exp3 started | groups={','.join(sorted(run_groups))} | run_dir={run_dir}")

    for subject_id in config["regressor_db3_subjects"]:
        print(f"\n--- DB3 S{subject_id:02d} ---")
        npz_path = pred_dir / f"S{subject_id:02d}_angle_predictions.npz"
        if npz_path.exists():
            with np.load(npz_path) as existing_prediction:
                assert_key10_prediction_payload(existing_prediction, str(npz_path))
        if _prediction_has_groups(npz_path, run_groups):
            try:
                _log(log_path, f"S{subject_id:02d} prediction exists; loading split diagnostics")
                raw_emg, angle, _, reps, window_meta = prepare_kinematics_data(
                    data_loader, [subject_id], config,
                    exercises=config["regressor_exercises"], db="db3", return_metadata=True,
                )
                train_idx, val_idx, test_idx = make_rep_split(reps)
                dynamic_diag = dynamic_split_diagnostics(
                    angle,
                    {"train": train_idx, "val": val_idx, "test": test_idx},
                    config,
                )
                print_dynamic_diagnostics(subject_id, dynamic_diag)
                dynamic_test_indices = np.asarray(
                    dynamic_diag.get("splits", {}).get("test", {}).get("dynamic_indices", []),
                    dtype=np.int64,
                )
                window_starts = _exercise_separated_window_starts(window_meta, config)
                test_window_starts = window_starts[test_idx]
                subj_result, _, resume_updates = _subject_from_prediction(
                    subject_id, npz_path, run_groups, dynamic_test_indices=dynamic_test_indices,
                    test_window_starts=test_window_starts, config=config,
                )
                if resume_updates:
                    _merge_prediction_payload(npz_path, resume_updates)
                    _log(log_path, f"S{subject_id:02d} continuous-output fields rebuilt from prediction")
                subj_result["n_segments"] = int(len(angle))
                subj_result["split_sizes"] = {
                    "train": len(train_idx), "val": len(val_idx), "test": len(test_idx),
                }
                subj_result["split_method"] = "rep_based"
                subj_result["dynamic_angle_coverage"] = dynamic_diag
                subj_result["low_dynamic_train_coverage"] = bool(dynamic_diag.get("low_dynamic_train_coverage"))
                if dynamic_diag.get("low_dynamic_train_coverage"):
                    _log(log_path, f"WARNING S{subject_id:02d} low_dynamic_train_coverage=true")
                with np.load(npz_path) as data:
                    required_continuous = {
                        "continuous_target", "continuous_time_indices",
                        "continuous_overlap_counts", "continuous_dynamic_mask",
                        *(f"pred_{grp}_continuous" for grp in ("A", "B")),
                    }
                    missing_continuous = sorted(required_continuous.difference(data.files))
                    if missing_continuous:
                        raise RuntimeError(
                            f"S{subject_id:02d} has no complete continuous plotting payload: {missing_continuous}"
                        )
                    group_preds_for_fig = {
                        grp: np.asarray(data[f"pred_{grp}_continuous"]) for grp in ("A", "B")
                    }
                    continuous_target_for_fig = np.asarray(data["continuous_target"])
                    continuous_times_for_fig = np.asarray(data["continuous_time_indices"], dtype=np.int64)
                    continuous_overlaps_for_fig = np.asarray(data["continuous_overlap_counts"], dtype=np.int16)
                    continuous_dynamic_for_fig = np.asarray(data["continuous_dynamic_mask"], dtype=bool)
                figure_files, selection = save_abc_comparison_figures(
                    subject_id, group_preds_for_fig, continuous_target_for_fig,
                    continuous_times_for_fig, continuous_overlaps_for_fig, continuous_dynamic_for_fig, viz_dir,
                    n_trials=int(config.get("regressor_viz_trials_per_subject", 2)), config=config,
                    low_dynamic_coverage=bool(dynamic_diag.get("low_dynamic_train_coverage")),
                    output_postprocess=continuous_output_label(config),
                    return_selection=True,
                )
                subj_result["abc_comparison_figures"] = figure_files
                subj_result["abc_visualization_selection"] = selection
                print(
                    f"  A/B visualization selected S{subject_id:02d}: "
                    f"indices={selection.get('selected_indices', [])} "
                    f"scores={[round(float(s), 4) for s in selection.get('selected_scores', [])]}"
                )
                _upsert_subject(report, subj_result)
                _save_progress(out_path, report, start_ts)
                _log(log_path, f"S{subject_id:02d} resumed prediction and regenerated {len(figure_files)} continuous-output figures")
                continue
            except Exception as exc:
                _log(log_path, f"WARNING S{subject_id:02d} failed prediction-resume diagnostics: {repr(exc)}")
                subj_result, _, _ = _subject_from_prediction(subject_id, npz_path, run_groups)
                _upsert_subject(report, subj_result)
                _save_progress(out_path, report, start_ts)
                _log(log_path, f"S{subject_id:02d} skipped training: complete prediction exists at {npz_path}")
                continue

        try:
            _log(log_path, f"S{subject_id:02d} start")
            raw_emg, angle, _, reps, window_meta = prepare_kinematics_data(
                data_loader, [subject_id], config,
                exercises=config["regressor_exercises"], db="db3", return_metadata=True,
            )
            train_idx, val_idx, test_idx = make_rep_split(reps)
            print(f"  rep-split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
                  "(train=[1,3,4] val=[6] test=[2,5])")
            dynamic_diag = dynamic_split_diagnostics(
                angle,
                {"train": train_idx, "val": val_idx, "test": test_idx},
                config,
            )
            print_dynamic_diagnostics(subject_id, dynamic_diag)
            if dynamic_diag.get("low_dynamic_train_coverage"):
                _log(log_path, f"WARNING S{subject_id:02d} low_dynamic_train_coverage=true")

            subj_result = {
                "subject_id": subject_id,
                "status": "ok",
                "angle_target": key10_target_metadata(),
                "n_segments": int(len(raw_emg)),
                "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
                "split_method": "fixed_repetition_1_3_4__6__2_5",
                "source_exercises": [int(value) for value in config["regressor_exercises"]],
                "normalization": window_meta["normalization"],
                "window_policy": window_meta["window_policy"],
                "dynamic_angle_coverage": dynamic_diag,
                "low_dynamic_train_coverage": bool(dynamic_diag.get("low_dynamic_train_coverage")),
                "groups": {},
            }
            subj_subsets = {}
            group_preds = {}
            group_calibrations = {}
            group_target = None

            quality_masks = window_meta["quality_mask"]
            subj_result["quality_mask_rule"] = "hard_zero_train_1_3_4_or_gronlund_2005_mqp_p_gt_0_20"
            subj_result["quality_mask_ratio"] = float((quality_masks < 0.5).mean())

            if "A" in run_groups:
                _log(log_path, f"S{subject_id:02d} group A train")
                subj_ckpt_dir = ckpt_dir / f"S{subject_id:02d}"
                subj_ckpt_dir.mkdir(parents=True, exist_ok=True)
                result_A, subs_A, pred_A, tgt_A, calibration_A = evaluate_group(
                    "A", raw_emg, angle, train_idx, val_idx, test_idx, config, device,
                    subj_ckpt_dir / "tcn_A_raw_best.pth",
                    subject_id,
                    {"input": "raw"},
                )
                subj_result["groups"]["A"] = result_A
                subj_subsets["A"] = subs_A
                group_preds["A"] = pred_A
                if calibration_A is not None:
                    group_calibrations["A"] = calibration_A
                group_target = tgt_A
                _log(log_path, f"S{subject_id:02d} group A done")

            if "B" in run_groups and healthy_mcia is not None:
                _log(log_path, f"S{subject_id:02d} group B enhance/train")
                print("  Building Group B enhanced EMG with healthy-prior MCIA...")
                enh_B = make_enhanced_pool(
                    healthy_mcia, raw_emg, train_idx, val_idx, test_idx, quality_masks, device, domain_id=None,
                    patch_size=int(config["patch_size"]),
                )
                subj_ckpt_dir = ckpt_dir / f"S{subject_id:02d}"
                subj_ckpt_dir.mkdir(parents=True, exist_ok=True)
                result_B, subs_B, pred_B, tgt_B, calibration_B = evaluate_group(
                    "B", enh_B, angle, train_idx, val_idx, test_idx, config, device,
                    subj_ckpt_dir / "tcn_B_healthy_prior_best.pth",
                    subject_id,
                    {"input": "enhanced", "mcia_source": "healthy_prior", "mcia_checkpoint": str(healthy_ckpt),
                     "mcia_domain_id": None, "mcia_adapters_loaded": False},
                )
                subj_result["groups"]["B"] = result_B
                subj_subsets["B"] = subs_B
                group_preds["B"] = pred_B
                if calibration_B is not None:
                    group_calibrations["B"] = calibration_B
                if group_target is None:
                    group_target = tgt_B
                _log(log_path, f"S{subject_id:02d} group B done")

            dynamic_test_indices = np.asarray(
                dynamic_diag.get("splits", {}).get("test", {}).get("dynamic_indices", []),
                dtype=np.int64,
            )
            dynamic_test_scores = None
            if group_target is not None:
                dynamic_test_scores = compute_dynamic_angle_scores(
                    group_target, str(config.get("regressor_dynamic_score", "global_max_ptp"))
                )
            if group_target is not None and len(dynamic_test_indices) > 0:
                for grp, pred in group_preds.items():
                    if grp in subj_result["groups"]:
                        subj_result["groups"][grp]["dynamic_subsets"] = evaluate_subsets(
                            pred[dynamic_test_indices], group_target[dynamic_test_indices]
                        )
            elif group_target is not None:
                for grp in group_preds:
                    if grp in subj_result["groups"]:
                        subj_result["groups"][grp]["dynamic_subsets"] = {}

            continuous_output = None
            continuous_plot_payload = None
            pred_payload = {
                **key10_prediction_metadata(),
                "angle_data_schema": np.asarray("e1_e2_exercise_separated_train_only_normalization_v1"),
                "source_exercises": np.asarray(config["regressor_exercises"], dtype=np.int16),
                "target": group_target,
            }
            if group_target is not None:
                window_starts = _exercise_separated_window_starts(window_meta, config)
                test_window_starts = window_starts[test_idx]
                continuous_output = _continuous_output_from_groups(
                    group_preds, group_target, test_window_starts, dynamic_test_indices, config,
                    group_calibrations=group_calibrations,
                )
                if continuous_output.get("enabled"):
                    continuous_plot_payload = continuous_output["payload"]
                    for grp, details in continuous_output["group_results"].items():
                        if grp in subj_result["groups"]:
                            subj_result["groups"][grp].update(details)
                    subj_result["continuous_output"] = continuous_output["settings"]
                    pred_payload.update(continuous_output["payload"])
                pred_payload["dynamic_test_indices"] = dynamic_test_indices
                pred_payload["dynamic_test_scores"] = dynamic_test_scores
            for grp, pred in group_preds.items():
                pred_payload[f"pred_{grp}"] = pred
            if group_target is not None:
                np.savez(npz_path, **pred_payload)
                _log(log_path, f"S{subject_id:02d} prediction saved: {npz_path}")
            figure_files = []
            subj_result["prediction_file"] = str(npz_path)
            subj_result["abc_comparison_figures"] = figure_files
            _upsert_subject(report, subj_result)
            _save_progress(out_path, report, start_ts)

            if group_target is not None:
                try:
                    if continuous_plot_payload is None:
                        raise RuntimeError("continuous output is required for 1280-point A/B figures")
                    figure_files, selection = save_abc_comparison_figures(
                        subject_id,
                        {grp: continuous_plot_payload[f"pred_{grp}_continuous"] for grp in ("A", "B")},
                        continuous_plot_payload["continuous_target"],
                        continuous_plot_payload["continuous_time_indices"],
                        continuous_plot_payload["continuous_overlap_counts"],
                        continuous_plot_payload["continuous_dynamic_mask"],
                        viz_dir,
                        n_trials=int(config.get("regressor_viz_trials_per_subject", 2)),
                        config=config,
                        low_dynamic_coverage=bool(dynamic_diag.get("low_dynamic_train_coverage")),
                        output_postprocess=continuous_output_label(config),
                        return_selection=True,
                    )
                    subj_result["abc_comparison_figures"] = figure_files
                    subj_result["abc_visualization_selection"] = selection
                    print(
                        f"  A/B visualization selected S{subject_id:02d}: "
                        f"indices={selection.get('selected_indices', [])} "
                        f"scores={[round(float(s), 4) for s in selection.get('selected_scores', [])]}"
                    )
                    _upsert_subject(report, subj_result)
                    _save_progress(out_path, report, start_ts)
                    _log(log_path, f"S{subject_id:02d} A/B anatomy figures saved: {len(figure_files)}")
                except Exception as fig_exc:
                    _log(log_path, f"WARNING S{subject_id:02d} A/B anatomy figures skipped: {fig_exc}")
            _log(log_path, f"S{subject_id:02d} done")

        except Exception as exc:
            tb = traceback.format_exc()
            print(tb)
            failed = {
                "subject_id": subject_id,
                "status": "failed",
                "reason": repr(exc),
                "traceback": tb,
            }
            _upsert_subject(report, failed)
            _save_progress(out_path, report, start_ts)
            _log(log_path, f"S{subject_id:02d} failed: {repr(exc)}")
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(tb + "\n")
            continue

    _save_progress(out_path, report, start_ts)
    print(f"\nReport saved to: {out_path}")
    all_subj_results = _subsets_from_report(report)
    aggregate_and_print_summary(all_subj_results, groups=tuple(sorted(run_groups)))
    aggregate_continuous_summary(report, groups=tuple(sorted(run_groups)))
    print_dynamic_compact_summary(report)
    failures = [s for s in report.get("subjects", []) if s.get("status") == "failed"]
    if failures:
        raise RuntimeError(f"Exp3 finished with failed subjects: {[s['subject_id'] for s in failures]}")

if __name__ == "__main__":
    main()
