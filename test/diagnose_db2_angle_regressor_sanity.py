"""Independent DB2 continuous-angle regressor sanity experiment.

This diagnostic is intentionally decoupled from Exp1/Exp2/Exp3. It trains one
KinematicTCN per DB2 subject on DB2 E1 EMG + same-side glove labels, using the
same kinematics windowing, repetition split style, model shape, and metric
helpers as Exp3. Outputs are written under 06_diagnostics/db2_angle_sanity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import scipy.io as sio
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
import yaml
from torch.utils.data import DataLoader, Dataset

from data.dataset_kinematics import KinematicsDataset, make_rep_split, prepare_kinematics_data
from data.dataset_db2_emg import moving_average
from data.ninapro_loader import NinaProDataLoader
from models.prediction.kinematic_regressor import KinematicBiLSTM, KinematicTCN
from utils.paper_pipeline import set_seed
from utils.kinematic_target import KEY10_CHANNEL_NAMES, KEY10_DIM, key10_prediction_metadata, key10_target_metadata
from utils.kinematic_output_postprocess import (
    continuous_trajectory_quality,
    postprocess_window_predictions,
    window_overlap_prediction_consistency,
)


def _load_exp3_helpers():
    path = PROJECT_ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("exp3_angle_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EXP3 = _load_exp3_helpers()
ANGLE_SUBSETS = EXP3.ANGLE_SUBSETS


def _parse_int_list(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def _make_strict_rep_split(repetitions: np.ndarray, validation_rep: int,
                           train_reps: tuple[int, ...] = (1, 3, 4, 6),
                           test_reps: tuple[int, ...] = (2, 5)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out one complete training repetition for validation without window overlap."""
    if int(validation_rep) not in train_reps:
        raise ValueError(f"validation_rep={validation_rep} must be one of train_reps={train_reps}")
    reps = np.asarray(repetitions, dtype=np.int32)
    val_idx = np.flatnonzero(reps == int(validation_rep)).astype(np.int64)
    train_idx = np.flatnonzero(np.isin(reps, train_reps) & (reps != int(validation_rep))).astype(np.int64)
    test_idx = np.flatnonzero(np.isin(reps, test_reps)).astype(np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Strict repetition split produced an empty train, validation, or test set")
    return train_idx, val_idx, test_idx


def _latest_run_dir(output_root: Path) -> Path | None:
    run_root = output_root / "run"
    if not run_root.exists():
        return None
    runs = sorted([p for p in run_root.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _resolve_run_dir(project_root: Path, cfg: dict, arg_run_dir: str | None) -> Path:
    if arg_run_dir:
        return Path(arg_run_dir)
    if os.environ.get("MCIA_RUN_DIR"):
        return Path(os.environ["MCIA_RUN_DIR"])
    output_root = Path(cfg["paths"]["output"])
    if not output_root.is_absolute():
        output_root = project_root / output_root
    latest = _latest_run_dir(output_root)
    if latest is not None:
        return latest
    raise RuntimeError(
        "No run directory found. Set --run-dir or MCIA_RUN_DIR; this diagnostic will not create a new run."
    )


def _load_config() -> tuple[dict, dict]:
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    exp3 = raw.get("exp3_regressor", {})
    diag = raw.get("diagnostics", {}).get("db2_angle_sanity", {})
    config = {
        "db2_path": raw["paths"]["db2"],
        "db3_path": raw["paths"]["db3"],
        "orig_fs": raw["signal"]["orig_fs"],
        "target_fs": raw["signal"]["target_fs"],
        "n_channels": raw["signal"]["n_channels"],
        "window_size": raw["signal"]["window_size"],
        "stride": raw["signal"]["stride"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    for key, value in exp3.items():
        config[f"regressor_{key}"] = value
    config["diag_subjects"] = diag.get("subjects", [1, 2])
    config["diag_exercises"] = diag.get("exercises", [1])
    # DB2 sanity inherits the locked Exp3 visualization and dynamic settings.
    config["diag_viz_trials_per_subject"] = int(config["regressor_viz_trials_per_subject"])
    config["diag_dynamic_min_ptp"] = float(config["regressor_dynamic_min_ptp"])
    config["diag_dynamic_viz_top_k"] = int(config["regressor_dynamic_viz_top_k"])
    return raw, config


def _log(log_path: Path, message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def _build_tcn(config: dict, n_emg_channels: int, device: str) -> KinematicTCN:
    return KinematicTCN(
        n_emg_channels=n_emg_channels,
        n_angle_channels=int(config["regressor_n_angle_channels"]),
        hidden_dim=int(config["regressor_hidden_dim"]),
        n_layers=int(config["regressor_n_layers"]),
        kernel_size=int(config["regressor_kernel_size"]),
        dropout=float(config["regressor_model_dropout"]),
    ).to(device)




def _build_bilstm(config: dict, n_emg_channels: int, device: str) -> KinematicBiLSTM:
    return KinematicBiLSTM(
        n_emg_channels=n_emg_channels,
        n_angle_channels=int(config["regressor_n_angle_channels"]),
        hidden_dim=int(config.get("regressor_bilstm_hidden_dim", 128)),
        n_layers=int(config.get("regressor_bilstm_layers", 2)),
        dropout=float(config["regressor_model_dropout"]),
    ).to(device)


def _shift_angle_windows(angle: np.ndarray, lag_samples: int) -> np.ndarray:
    lag = int(lag_samples)
    if lag == 0:
        return angle.copy()
    t = angle.shape[1]
    src = np.clip(np.arange(t) + lag, 0, t - 1)
    return angle[:, src, :].copy()


def _lowpass_angle_windows(angle: np.ndarray, target_fs: int, cutoff_hz: float = 10.0) -> np.ndarray:
    if angle.shape[1] < 8:
        return angle.copy()
    nyq = 0.5 * float(target_fs)
    cutoff = min(float(cutoff_hz), nyq * 0.95)
    sos = signal.butter(4, cutoff / nyq, btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, angle, axis=1).astype(angle.dtype, copy=False)


def _center_mean_target_metrics(pred_traj: np.ndarray, target_traj: np.ndarray,
                                dynamic_indices: np.ndarray) -> dict:
    center = target_traj.shape[1] // 2
    window_mean_pred = pred_traj.mean(axis=1)
    window_mean_target = target_traj.mean(axis=1)
    center_pred = pred_traj[:, center, :]
    center_target = target_traj[:, center, :]
    return {
        "trajectory_center_point": _point_method_result(center_pred, center_target, dynamic_indices),
        "trajectory_window_mean": _point_method_result(window_mean_pred, window_mean_target, dynamic_indices),
    }



class CenterPointDataset(Dataset):
    def __init__(self, emg: np.ndarray, angle_center: np.ndarray):
        self.emg = torch.as_tensor(emg, dtype=torch.float32)
        self.angle = torch.as_tensor(angle_center, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.emg.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"emg": self.emg[idx], "angle": self.angle[idx]}


class CenterPointTCN(nn.Module):
    def __init__(self, config: dict, n_emg_channels: int, center_idx: int):
        super().__init__()
        self.backbone = KinematicTCN(
            n_emg_channels=n_emg_channels,
            n_angle_channels=int(config["regressor_n_angle_channels"]),
            hidden_dim=int(config["regressor_hidden_dim"]),
            n_layers=int(config["regressor_n_layers"]),
            kernel_size=int(config["regressor_kernel_size"]),
            dropout=float(config["regressor_model_dropout"]),
        )
        self.center_idx = int(center_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[:, self.center_idx, :]


def _subset_metrics_points(pred: np.ndarray, target: np.ndarray) -> dict:
    return EXP3.evaluate_subsets(pred[:, None, :], target[:, None, :])


def _window_features(emg_windows: np.ndarray) -> np.ndarray:
    mean = emg_windows.mean(axis=1)
    rms = np.sqrt(np.mean(emg_windows ** 2, axis=1))
    std = emg_windows.std(axis=1)
    slope = (emg_windows[:, -1, :] - emg_windows[:, 0, :]) / max(emg_windows.shape[1] - 1, 1)
    return np.concatenate([mean, rms, std, slope], axis=1)


def _train_center_tcn(emg: np.ndarray, angle: np.ndarray, train_idx: np.ndarray,
                      val_idx: np.ndarray, test_idx: np.ndarray, config: dict,
                      device: str) -> dict:
    center_idx = angle.shape[1] // 2
    train_set = CenterPointDataset(emg[train_idx], angle[train_idx, center_idx, :])
    val_set = CenterPointDataset(emg[val_idx], angle[val_idx, center_idx, :])
    train_loader = DataLoader(train_set, batch_size=int(config["regressor_batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(config["regressor_batch_size"]), shuffle=False, num_workers=0)
    model = CenterPointTCN(config, emg.shape[-1], center_idx).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config["regressor_learning_rate"]), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["regressor_lr_factor"]),
        patience=int(config["regressor_lr_patience"]),
        min_lr=1e-7,
    )
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []
    for epoch in range(int(config["regressor_num_epochs"])):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pred = model(batch["emg"].to(device))
            loss = criterion(pred, batch["angle"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item())
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += float(criterion(model(batch["emg"].to(device)), batch["angle"].to(device)).item())
        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(config["regressor_patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loader = DataLoader(
        CenterPointDataset(emg[test_idx], angle[test_idx, center_idx, :]),
        batch_size=int(config["regressor_batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            preds.append(model(batch["emg"].to(device)).cpu().numpy())
            targets.append(batch["angle"].numpy())
    pred_arr = np.concatenate(preds, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    return {
        "center_index": int(center_idx),
        "training": {"best_val_loss": float(best_val), "epochs": int(len(history)), "history": history},
        "subsets": _subset_metrics_points(pred_arr, target_arr),
        "pred": pred_arr,
        "target": target_arr,
    }


def _run_center_baselines(emg: np.ndarray, angle: np.ndarray, train_idx: np.ndarray,
                          test_idx: np.ndarray, rf_trees: int, skip_rf: bool) -> dict:
    center_idx = angle.shape[1] // 2
    y_train = angle[train_idx, center_idx, :]
    y_test = angle[test_idx, center_idx, :]
    x_train = _window_features(emg[train_idx])
    x_test = _window_features(emg[test_idx])
    results = {}
    predictions = {}
    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(test_idx), axis=0)
    predictions["train_mean"] = mean_pred
    results["train_mean"] = _subset_metrics_points(mean_pred, y_test)

    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(x_train)
    x_test_z = scaler.transform(x_test)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_z, y_train)
    ridge_pred = ridge.predict(x_test_z)
    predictions["ridge"] = ridge_pred
    results["ridge"] = _subset_metrics_points(ridge_pred, y_test)

    if not skip_rf:
        rf = RandomForestRegressor(n_estimators=int(rf_trees), random_state=42, n_jobs=-1, min_samples_leaf=2)
        rf.fit(x_train, y_train)
        rf_pred = rf.predict(x_test)
        predictions["random_forest"] = rf_pred
        results["random_forest"] = _subset_metrics_points(rf_pred, y_test)
    else:
        results["random_forest"] = {"skipped": True}
    return {
        "center_index": int(center_idx),
        "feature_dim": int(x_train.shape[1]),
        "subsets": results,
        "predictions": predictions,
        "target": y_test,
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _lag_corr_curve(source: np.ndarray, target: np.ndarray, lag_max: int) -> dict:
    lags = np.arange(-int(lag_max), int(lag_max) + 1, dtype=np.int64)
    corrs = []
    for lag in lags:
        if lag > 0:
            # Positive lag means source/EMG leads target/glove by lag samples.
            sx, ty = source[:-lag], target[lag:]
        elif lag < 0:
            sx, ty = source[-lag:], target[:lag]
        else:
            sx, ty = source, target
        corrs.append(_safe_corr(sx, ty))
    corr_arr = np.asarray(corrs, dtype=np.float64)
    valid = np.where(np.isfinite(corr_arr), np.abs(corr_arr), -np.inf)
    best_i = int(np.argmax(valid)) if len(valid) else 0
    return {
        "lags": lags.tolist(),
        "correlations": corr_arr.tolist(),
        "best_lag_samples": int(lags[best_i]) if len(lags) else 0,
        "best_correlation": float(corr_arr[best_i]) if len(corr_arr) else float("nan"),
        "best_abs_correlation": float(abs(corr_arr[best_i])) if len(corr_arr) else float("nan"),
    }


def _distribution(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": float("nan"), "p90": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _lowpass_then_downsample(glove: np.ndarray, orig_fs: int, target_fs: int, cutoff_hz: float) -> np.ndarray:
    downsample_factor = max(int(orig_fs // target_fs), 1)
    nyq = 0.5 * float(orig_fs)
    cutoff = min(float(cutoff_hz), nyq * 0.95)
    sos = signal.butter(4, cutoff / nyq, btype="lowpass", output="sos")
    filt = signal.sosfiltfilt(sos, glove, axis=0)
    return filt[::downsample_factor]


def _preprocess_emg_continuous(loader: NinaProDataLoader, emg: np.ndarray, config: dict) -> np.ndarray:
    emg_mv = emg.astype(np.float64) * 1000.0
    filtered = loader.bandpass_filter(emg_mv)
    filtered = loader.notch_filter(filtered)
    downsample_factor = max(int(config["orig_fs"] // config["target_fs"]), 1)
    envelope = moving_average(np.abs(filtered), downsample_factor)
    return envelope[::downsample_factor]


def _load_db2_raw_exercise(loader: NinaProDataLoader, subject_id: int, exercise_id: int) -> dict:
    subject_dir = Path(loader.db2_path) / f"DB2_s{subject_id}"
    mat_path = subject_dir / f"S{subject_id}_E{exercise_id}_A1.mat"
    mat = sio.loadmat(mat_path)
    return {
        "path": str(mat_path),
        "emg": np.asarray(mat["emg"], dtype=np.float64),
        "glove": np.asarray(mat["glove"], dtype=np.float64),
        "restimulus": np.asarray(mat.get("restimulus", mat.get("stimulus"))).reshape(-1),
        "rerepetition": np.asarray(mat.get("rerepetition", mat.get("repetition"))).reshape(-1),
    }


def _plot_lag_scan(path: Path, lag_scan: dict, target_fs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for name, curve in lag_scan["curves"].items():
        lags = np.asarray(curve["lags"], dtype=float)
        ms = lags / float(target_fs) * 1000.0
        ax.plot(ms, curve["correlations"], lw=1.4, label=name.replace("_", " "))
        ax.axvline(curve["best_lag_ms"], ls="--", lw=0.8, alpha=0.35)
    ax.axhline(0, color="#666666", lw=0.8, alpha=0.6)
    ax.set_xlabel("lag ms (positive: EMG leads glove)")
    ax.set_ylabel("Pearson correlation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_glove_downsample(path: Path, raw: np.ndarray, current_down: np.ndarray,
                           lowpass_down: np.ndarray, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    span_down = min(int(config["target_fs"] * 5), current_down.shape[0])
    downsample_factor = max(int(config["orig_fs"] // config["target_fs"]), 1)
    span_raw = min(span_down * downsample_factor, raw.shape[0])
    channels = np.argsort(np.ptp(current_down[:span_down], axis=0))[-3:][::-1]
    fig, axes = plt.subplots(len(channels), 1, figsize=(8.5, 5.5), sharex=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for ax, ch in zip(axes, channels):
        raw_t = np.arange(span_raw) / float(config["orig_fs"])
        down_t = np.arange(span_down) / float(config["target_fs"])
        ax.plot(raw_t, raw[:span_raw, ch], color="#999999", lw=0.8, alpha=0.6, label="raw glove")
        ax.plot(down_t, current_down[:span_down, ch], color="#111111", lw=1.1, label="glove[::factor]")
        ax.plot(down_t, lowpass_down[:span_down, ch], color="#1f77b4", lw=1.1, label="lowpass then downsample")
        ax.set_ylabel(f"Dim {int(ch)}", fontsize=9)
        ax.grid(True, alpha=0.22)
    axes[-1].set_xlabel("seconds")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("DB2 glove downsample diagnostic", y=0.97, fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.84, hspace=0.35)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _continuous_diagnostics(loader: NinaProDataLoader, subject_id: int, config: dict,
                            figures_dir: Path, lag_max: int, lowpass_hz: float) -> dict:
    results = {"exercises": []}
    for exercise_id in config["diag_exercises"]:
        raw = _load_db2_raw_exercise(loader, subject_id, int(exercise_id))
        emg_down = _preprocess_emg_continuous(loader, raw["emg"], config)
        downsample_factor = max(int(config["orig_fs"] // config["target_fs"]), 1)
        glove_raw = raw["glove"]
        glove_down = glove_raw[::downsample_factor]
        glove_low_down = _lowpass_then_downsample(glove_raw, int(config["orig_fs"]), int(config["target_fs"]), lowpass_hz)
        n = min(emg_down.shape[0], glove_down.shape[0], glove_low_down.shape[0])
        emg_down = emg_down[:n]
        glove_down = glove_down[:n]
        glove_low_down = glove_low_down[:n]
        emg_rms = np.sqrt(np.mean(emg_down ** 2, axis=1))
        emg_z = StandardScaler().fit_transform(emg_down)
        pca1 = PCA(n_components=1, random_state=42).fit_transform(emg_z).reshape(-1)
        glove_vel = np.diff(glove_down, axis=0, prepend=glove_down[:1])
        glove_vel_rms = np.sqrt(np.mean(glove_vel ** 2, axis=1))
        glove_dyn = np.max(np.abs(glove_vel), axis=1)
        curves = {
            "emg_rms_vs_glove_velocity": _lag_corr_curve(emg_rms, glove_vel_rms, lag_max),
            "emg_pca1_vs_glove_velocity": _lag_corr_curve(pca1, glove_vel_rms, lag_max),
            "emg_rms_vs_angle_dynamic_score": _lag_corr_curve(emg_rms, glove_dyn, lag_max),
            "emg_pca1_vs_angle_dynamic_score": _lag_corr_curve(pca1, glove_dyn, lag_max),
        }
        for curve in curves.values():
            curve["best_lag_ms"] = float(curve["best_lag_samples"] / float(config["target_fs"]) * 1000.0)
        lag_scan = {
            "lag_definition": "positive lag means EMG/envelope leads glove target by lag samples",
            "lag_max_samples": int(lag_max),
            "target_fs": int(config["target_fs"]),
            "curves": curves,
        }
        lag_fig = figures_dir / f"S{subject_id:02d}" / "lag_scan" / f"S{subject_id:02d}_E{exercise_id}_lag_scan.png"
        _plot_lag_scan(lag_fig, lag_scan, int(config["target_fs"]))

        def one_glove_block(arr: np.ndarray) -> dict:
            vel = np.diff(arr, axis=0)
            return {
                "ptp_per_channel": np.ptp(arr, axis=0).astype(float).tolist(),
                "std_per_channel": np.std(arr, axis=0).astype(float).tolist(),
                "ptp_distribution": _distribution(np.ptp(arr, axis=0)),
                "std_distribution": _distribution(np.std(arr, axis=0)),
                "velocity_abs_distribution": _distribution(np.abs(vel)),
                "velocity_rms_distribution": _distribution(np.sqrt(np.mean(vel ** 2, axis=1))),
            }
        glove_diag = {
            "downsample_factor": int(downsample_factor),
            "lowpass_hz": float(lowpass_hz),
            "raw_glove": one_glove_block(glove_raw),
            "glove_down": one_glove_block(glove_down),
            "lowpass_then_downsample_glove": one_glove_block(glove_low_down),
        }
        glove_fig = figures_dir / f"S{subject_id:02d}" / "glove_downsample" / f"S{subject_id:02d}_E{exercise_id}_glove_downsample.png"
        _plot_glove_downsample(glove_fig, glove_raw, glove_down, glove_low_down, config)
        results["exercises"].append({
            "exercise": int(exercise_id),
            "mat_path": raw["path"],
            "lag_scan": lag_scan,
            "glove_downsample_diagnostics": glove_diag,
            "lag_scan_figure": str(lag_fig),
            "glove_downsample_figure": str(glove_fig),
        })
    return results


def _trajectory_quality(pred: np.ndarray, target: np.ndarray) -> dict:
    pred_vel = np.diff(pred, axis=1)
    target_vel = np.diff(target, axis=1)
    pred_jerk = np.diff(pred, n=2, axis=1)
    target_jerk = np.diff(target, n=2, axis=1)
    pred_fft = np.abs(np.fft.rfft(pred - pred.mean(axis=1, keepdims=True), axis=1))
    target_fft = np.abs(np.fft.rfft(target - target.mean(axis=1, keepdims=True), axis=1))
    pred_spec = pred_fft.mean(axis=(0, 2))
    target_spec = target_fft.mean(axis=(0, 2))
    pred_spec_n = pred_spec / (np.sum(pred_spec) + 1e-12)
    target_spec_n = target_spec / (np.sum(target_spec) + 1e-12)
    split = max(1, len(pred_spec_n) // 3)
    return {
        "velocity_rms_gt": float(np.sqrt(np.mean(target_vel ** 2))),
        "velocity_rms_pred": float(np.sqrt(np.mean(pred_vel ** 2))),
        "velocity_rms_ratio_pred_over_gt": float(np.sqrt(np.mean(pred_vel ** 2)) / (np.sqrt(np.mean(target_vel ** 2)) + 1e-12)),
        "jerk_rms_gt": float(np.sqrt(np.mean(target_jerk ** 2))),
        "jerk_rms_pred": float(np.sqrt(np.mean(pred_jerk ** 2))),
        "jerk_rms_ratio_pred_over_gt": float(np.sqrt(np.mean(pred_jerk ** 2)) / (np.sqrt(np.mean(target_jerk ** 2)) + 1e-12)),
        "spectrum_l1_normalized": float(np.mean(np.abs(pred_spec_n - target_spec_n))),
        "high_frequency_energy_ratio_gt": float(np.sum(target_spec_n[split:])),
        "high_frequency_energy_ratio_pred": float(np.sum(pred_spec_n[split:])),
    }



def _trajectory_loss(pred: torch.Tensor, target: torch.Tensor,
                     lambda_velocity: float, lambda_smooth: float) -> tuple[torch.Tensor, dict]:
    mse = nn.functional.mse_loss(pred, target)
    vel = nn.functional.mse_loss(torch.diff(pred, dim=1), torch.diff(target, dim=1))
    jerk = nn.functional.mse_loss(torch.diff(pred, n=2, dim=1), torch.diff(target, n=2, dim=1))
    total = mse + float(lambda_velocity) * vel + float(lambda_smooth) * jerk
    return total, {"mse": float(mse.item()), "velocity": float(vel.item()), "smooth": float(jerk.item())}


def _smooth_predictions(pred: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    if pred.shape[1] < 3:
        return pred.copy()
    win = int(window)
    if win % 2 == 0:
        win += 1
    win = max(3, min(win, pred.shape[1] if pred.shape[1] % 2 == 1 else pred.shape[1] - 1))
    if win <= int(polyorder):
        win = int(polyorder) + 3
        if win % 2 == 0:
            win += 1
        win = min(win, pred.shape[1] if pred.shape[1] % 2 == 1 else pred.shape[1] - 1)
    if win < 3:
        return pred.copy()
    return signal.savgol_filter(pred, window_length=win, polyorder=min(int(polyorder), win - 1), axis=1, mode="interp")


def _reconstruct_window_starts(data_loader: NinaProDataLoader, subject_id: int,
                               exercises: list[int], config: dict,
                               expected_repetitions: np.ndarray) -> np.ndarray:
    """Rebuild prepare_kinematics_data window starts without changing that helper."""
    data = data_loader.load_db2_subject(subject_id, exercises)
    factor = int(config["orig_fs"] / config["target_fs"])
    labels = data.get("restimulus", data["stimulus"])[::factor]
    repetitions = data["repetition"][::factor]
    n_samples = min(len(labels), len(repetitions), len(data["emg"][::factor]))
    starts, rebuilt_reps = [], []
    window_size = int(config["window_size"])
    stride = int(config["stride"])
    center = window_size // 2
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        if np.any(labels[start:end] != 0):
            starts.append(start)
            rebuilt_reps.append(int(repetitions[start + center]))
    starts_np = np.asarray(starts, dtype=np.int64)
    rebuilt_reps_np = np.asarray(rebuilt_reps, dtype=np.int32)
    if not np.array_equal(rebuilt_reps_np, np.asarray(expected_repetitions, dtype=np.int32)):
        raise RuntimeError("Reconstructed DB2 window starts do not match prepare_kinematics_data repetitions")
    return starts_np


def _smooth_continuous_prediction(values: np.ndarray, valid_mask: np.ndarray,
                                  window: int, polyorder: int = 2) -> np.ndarray:
    """Smooth each contiguous predicted timeline separately, never across activity gaps."""
    result = values.copy()
    valid_positions = np.flatnonzero(valid_mask)
    if len(valid_positions) == 0:
        return result
    split_points = np.flatnonzero(np.diff(valid_positions) > 1) + 1
    for group in np.split(valid_positions, split_points):
        if len(group) < 3:
            continue
        segment = result[group][None, :, :]
        result[group] = _smooth_predictions(segment, window=window, polyorder=polyorder)[0]
    return result


def _overlap_fuse_predictions(pred: np.ndarray, target: np.ndarray,
                              window_starts: np.ndarray, smooth_window: int | None = None) -> tuple[np.ndarray, dict]:
    """Uniform overlap-add fusion using only model outputs and known window positions."""
    if len(pred) != len(window_starts) or len(target) != len(window_starts):
        raise ValueError("Prediction, target, and window-start counts must match")
    if len(pred) == 0:
        return pred.copy(), {"n_unique_samples": 0, "max_overlap": 0, "post_smooth_window": smooth_window}

    sequence_length = int(pred.shape[1])
    start_min = int(np.min(window_starts))
    start_max = int(np.max(window_starts)) + sequence_length
    n_time = start_max - start_min
    channels = int(pred.shape[2])
    pred_sum = np.zeros((n_time, channels), dtype=np.float64)
    target_sum = np.zeros((n_time, channels), dtype=np.float64)
    weights = np.zeros(n_time, dtype=np.float64)
    for trial_pred, trial_target, start in zip(pred, target, window_starts):
        left = int(start) - start_min
        right = left + sequence_length
        pred_sum[left:right] += trial_pred
        target_sum[left:right] += trial_target
        weights[left:right] += 1.0

    valid = weights > 0
    fused_pred = np.zeros_like(pred_sum)
    fused_target = np.zeros_like(target_sum)
    fused_pred[valid] = pred_sum[valid] / weights[valid, None]
    fused_target[valid] = target_sum[valid] / weights[valid, None]
    if smooth_window is not None:
        fused_pred = _smooth_continuous_prediction(fused_pred, valid, int(smooth_window))

    fused_windows = np.empty_like(pred)
    for trial_idx, start in enumerate(window_starts):
        left = int(start) - start_min
        fused_windows[trial_idx] = fused_pred[left:left + sequence_length]
    metadata = {
        "mode": "uniform_overlap_add",
        "post_smooth_window": int(smooth_window) if smooth_window is not None else None,
        "n_unique_samples": int(valid.sum()),
        "mean_overlap": float(weights[valid].mean()),
        "max_overlap": int(weights.max()),
        "continuous_full_metrics": EXP3.evaluate_subsets(fused_pred[valid][None, :, :], fused_target[valid][None, :, :]),
        "continuous_trajectory_quality": _trajectory_quality(
            fused_pred[valid][None, :, :], fused_target[valid][None, :, :]
        ),
    }
    return fused_windows, metadata


def _dynamic_metrics_for_trajectory(pred: np.ndarray, target: np.ndarray, dynamic_indices: np.ndarray) -> dict:
    if len(dynamic_indices) == 0:
        return {"skipped": True, "reason": "no_dynamic_test_windows", "n_dynamic": 0, "n_test": int(len(target))}
    return {
        "skipped": False,
        "n_dynamic": int(len(dynamic_indices)),
        "n_test": int(len(target)),
        "subsets": EXP3.evaluate_subsets(pred[dynamic_indices], target[dynamic_indices]),
    }


def _dynamic_metrics_for_points(pred: np.ndarray, target: np.ndarray, dynamic_indices: np.ndarray) -> dict:
    if len(dynamic_indices) == 0:
        return {"skipped": True, "reason": "no_dynamic_test_windows", "n_dynamic": 0, "n_test": int(len(target))}
    return {
        "skipped": False,
        "n_dynamic": int(len(dynamic_indices)),
        "n_test": int(len(target)),
        "subsets": _subset_metrics_points(pred[dynamic_indices], target[dynamic_indices]),
    }


def _trajectory_method_result(pred: np.ndarray, target: np.ndarray, dynamic_indices: np.ndarray,
                              training: dict | None = None, extra: dict | None = None) -> dict:
    result = {
        "task": "trajectory",
        "full_metrics": EXP3.evaluate_subsets(pred, target),
        "dynamic_subset_metrics": _dynamic_metrics_for_trajectory(pred, target, dynamic_indices),
        "trajectory_quality": _trajectory_quality(pred, target),
    }
    if training is not None:
        result["training"] = training
    if extra:
        result.update(extra)
    return result


def _build_overlap_validation_candidates(pred: np.ndarray, target: np.ndarray,
                                         window_starts: np.ndarray, dynamic_min_ptp: float,
                                         smooth_windows: list[int]) -> dict:
    """Evaluate predeclared fusion/post-smoothing candidates on validation output only."""
    scores = EXP3.compute_dynamic_angle_scores(target, "global_max_ptp")
    dynamic_indices = np.flatnonzero(scores >= float(dynamic_min_ptp)).astype(np.int64)
    candidates = {}
    fused_pred, fused_meta = _overlap_fuse_predictions(pred, target, window_starts, smooth_window=None)
    candidates["overlap_fused"] = _trajectory_method_result(
        fused_pred, target, dynamic_indices, None,
        {"source_method": "trajectory_tcn_mse", "overlap_fusion": fused_meta},
    )
    for window in sorted(set(int(v) for v in smooth_windows)):
        if window < 3: raise ValueError("overlap post-smoothing candidate windows must be at least 3")
        fused_smooth_pred, fused_smooth_meta = _overlap_fuse_predictions(
            pred, target, window_starts, smooth_window=window
        )
        candidates[f"overlap_fused_post_smooth_w{window}"] = _trajectory_method_result(
            fused_smooth_pred, target, dynamic_indices, None,
            {"source_method": "trajectory_tcn_mse", "overlap_fusion": fused_smooth_meta},
        )
    return {
        "selection_scope": "validation_only",
        "selection_metric": "continuous_full_metrics.global.rmse",
        "tie_breaker": "within_1_percent_rmse_choose_lower_continuous_jerk_ratio",
        "dynamic_guard": "dynamic_global_rmse_must_not_exceed_overlap_fused_by_more_than_1_percent",
        "dynamic_score": "global_max_ptp", "dynamic_min_ptp": float(dynamic_min_ptp),
        "candidate_windows": [int(v) for v in sorted(set(smooth_windows))], "candidates": candidates,
    }


def _parse_output_filter_spec(value: str) -> dict:
    text = str(value).strip().lower()
    if text == "none":
        return {"filter_kind": "none", "savgol_window": 13, "lowpass_hz": 8.0}
    kind, separator, parameter = text.partition(":")
    if kind == "savgol":
        return {"filter_kind": "savgol", "savgol_window": int(parameter or 13), "lowpass_hz": 8.0}
    if kind in {"butter", "butterworth"}:
        return {"filter_kind": "butterworth", "savgol_window": 13, "lowpass_hz": float(parameter or 8.0)}
    raise ValueError(f"Unsupported output filter candidate: {value}")


def _parse_output_pipeline(value: str) -> dict:
    parts = [part.strip() for part in str(value).split("|")]
    if len(parts) != 3:
        raise ValueError("fixed output pipeline must be fusion|filter|calibration")
    fusion, filter_value, calibration = parts
    if fusion not in {"uniform_overlap_add", "triangular_overlap_add"}:
        raise ValueError(f"Unsupported output fusion: {fusion}")
    if calibration not in {"none", "affine"}:
        raise ValueError(f"Unsupported output calibration: {calibration}")
    return {"fusion": fusion, "filter": _parse_output_filter_spec(filter_value), "calibration": calibration}


def _fit_channelwise_affine(pred: np.ndarray, target: np.ndarray) -> dict:
    """Fit y=a*x+b on validation output only; it never sees test labels."""
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
    # Bound a low-capacity correction so one validation repetition cannot explode a channel.
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


def _run_output_pipeline(pred: np.ndarray, target: np.ndarray, window_starts: np.ndarray,
                         dynamic_indices: np.ndarray, spec: dict, config: dict,
                         calibration: dict | None = None) -> tuple[np.ndarray, dict, dict | None]:
    used_calibration = calibration
    working_pred = np.asarray(pred)
    if spec["calibration"] == "affine":
        used_calibration = calibration or _fit_channelwise_affine(working_pred, target)
        working_pred = _apply_channelwise_affine(working_pred, used_calibration)
    filter_spec = spec["filter"]
    packed = postprocess_window_predictions(
        working_pred, target, window_starts, dynamic_indices,
        savgol_window=int(filter_spec["savgol_window"]),
        polyorder=2,
        fusion=str(spec["fusion"]),
        filter_kind=str(filter_spec["filter_kind"]),
        target_fs=int(config["target_fs"]),
        lowpass_hz=float(filter_spec["lowpass_hz"]),
    )
    return packed["processed_windows"], packed["metadata"], used_calibration


def _build_output_pipeline_validation_candidates(pred: np.ndarray, target: np.ndarray,
                                                 window_starts: np.ndarray,
                                                 dynamic_min_ptp: float, config: dict,
                                                 fusions: list[str], filters: list[str],
                                                 calibrations: list[str]) -> dict:
    """Compare output-only candidates on validation windows; test labels are not accessed."""
    scores = EXP3.compute_dynamic_angle_scores(target, "global_max_ptp")
    dynamic_indices = np.flatnonzero(scores >= float(dynamic_min_ptp)).astype(np.int64)
    candidates = {}
    for fusion in fusions:
        for filter_value in filters:
            for calibration in calibrations:
                spec = {
                    "fusion": str(fusion),
                    "filter": _parse_output_filter_spec(filter_value),
                    "calibration": str(calibration),
                }
                name = f"{fusion}|{filter_value}|{calibration}"
                processed, metadata, fitted = _run_output_pipeline(
                    pred, target, window_starts, dynamic_indices, spec, config
                )
                result = _trajectory_method_result(processed, target, dynamic_indices)
                result["postprocess"] = metadata
                result["pipeline_spec"] = {
                    "fusion": spec["fusion"], "filter": filter_value, "calibration": spec["calibration"]
                }
                if fitted is not None:
                    result["calibration"] = {
                        key: value for key, value in fitted.items() if key not in {"gain", "bias"}
                    }
                candidates[name] = result
    return {
        "selection_scope": "validation_only",
        "selection_metric": "dynamic_subset_metrics.subsets.global.rmse",
        "full_metric_guard": "full_global_rmse_must_not_exceed_uniform_savgol13_none_by_more_than_1_percent",
        "tie_breaker": "lower_trajectory_jerk_ratio",
        "dynamic_score": "global_max_ptp",
        "dynamic_min_ptp": float(dynamic_min_ptp),
        "candidates": candidates,
    }


def _point_method_result(pred: np.ndarray, target: np.ndarray, dynamic_indices: np.ndarray,
                         training: dict | None = None, extra: dict | None = None) -> dict:
    result = {
        "task": "center_point",
        "full_metrics": _subset_metrics_points(pred, target),
        "dynamic_subset_metrics": _dynamic_metrics_for_points(pred, target, dynamic_indices),
    }
    if training is not None:
        result["training"] = training
    if extra:
        result.update(extra)
    return result

def _train_subject_model(emg: np.ndarray, angle: np.ndarray, train_idx: np.ndarray,
                         val_idx: np.ndarray, config: dict, device: str,
                         lambda_velocity: float = 0.0,
                         lambda_smooth: float = 0.0,
                         model_type: str = "tcn") -> tuple[nn.Module, dict]:
    train_set = KinematicsDataset(emg[train_idx], angle[train_idx])
    val_set = KinematicsDataset(emg[val_idx], angle[val_idx])
    batch_size = int(config["regressor_batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    if model_type == "bilstm":
        model = _build_bilstm(config, emg.shape[-1], device)
    else:
        model = _build_tcn(config, emg.shape[-1], device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config["regressor_learning_rate"]), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["regressor_lr_factor"]),
        patience=int(config["regressor_lr_patience"]),
        min_lr=1e-7,
    )

    best_val = float("inf")
    best_state = None
    no_improve = 0
    epochs_run = 0
    history = []
    for epoch in range(int(config["regressor_num_epochs"])):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            pred = model(batch["emg"].to(device))
            target_batch = batch["angle"].to(device)
            if float(lambda_velocity) or float(lambda_smooth):
                loss, _loss_parts = _trajectory_loss(pred, target_batch, lambda_velocity, lambda_smooth)
            else:
                loss = criterion(pred, target_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["emg"].to(device))
                target_batch = batch["angle"].to(device)
                if float(lambda_velocity) or float(lambda_smooth):
                    loss, _loss_parts = _trajectory_loss(pred, target_batch, lambda_velocity, lambda_smooth)
                else:
                    loss = criterion(pred, target_batch)
                val_loss += float(loss.item())
        train_loss /= max(train_batches, 1)
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        epochs_run = epoch + 1
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(config["regressor_patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_val_loss": float(best_val),
        "epochs": int(epochs_run),
        "history": history,
        "lambda_velocity": float(lambda_velocity),
        "lambda_smooth": float(lambda_smooth),
        "model_type": str(model_type),
    }


@torch.no_grad()
def _predict(model: nn.Module, emg: np.ndarray, angle: np.ndarray, idx: np.ndarray,
             config: dict, device: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = KinematicsDataset(emg[idx], angle[idx])
    loader = DataLoader(dataset, batch_size=int(config["regressor_batch_size"]), shuffle=False, num_workers=0)
    model.eval()
    preds, targets = [], []
    for batch in loader:
        preds.append(model(batch["emg"].to(device)).cpu().numpy())
        targets.append(batch["angle"].numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def _dynamic_diagnostics(angle: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                         test_idx: np.ndarray, config: dict) -> dict:
    helper_config = {
        "regressor_dynamic_score": "global_max_ptp",
        "regressor_dynamic_min_ptp": float(config["diag_dynamic_min_ptp"]),
        "regressor_dynamic_min_train_windows": int(config.get("regressor_dynamic_min_train_windows", 8)),
    }
    return EXP3.dynamic_split_diagnostics(
        angle,
        {"train": train_idx, "val": val_idx, "test": test_idx},
        helper_config,
    )


def _select_visualization_indices(target: np.ndarray, config: dict) -> dict:
    helper_config = {
        "regressor_viz_trials_per_subject": int(config["diag_viz_trials_per_subject"]),
        "regressor_dynamic_viz_top_k": int(config["diag_dynamic_viz_top_k"]),
        "regressor_dynamic_min_ptp": float(config["diag_dynamic_min_ptp"]),
        "regressor_dynamic_score": "global_max_ptp",
    }
    scores = EXP3.compute_dynamic_angle_scores(target, "global_max_ptp")
    dynamic_indices = np.flatnonzero(scores >= float(config["diag_dynamic_min_ptp"])).astype(np.int64)
    return EXP3.build_abc_visualization_selection(target, helper_config, dynamic_indices, scores, False)


def _plot_subset_trace(path: Path, target_trial: np.ndarray, method_trials: dict[str, np.ndarray],
                       dims: list[int], subset_name: str, subject_id: int, test_idx: int,
                       dynamic_score: float, title_stats: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(target_trial.shape[0])
    fig, axes = plt.subplots(len(dims), 1, figsize=(9.4, max(3.2, 1.95 * len(dims))), sharex=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    styles = {
        "trajectory_tcn_mse": {"color": "#1f77b4", "lw": 1.0, "alpha": 0.82, "label": "trajectory_tcn_mse"},
        "trajectory_tcn_smooth": {"color": "#d62728", "lw": 1.1, "alpha": 0.88, "label": "trajectory_tcn_smooth"},
        "trajectory_tcn_mse_post_smooth": {"color": "#2ca02c", "lw": 1.0, "alpha": 0.82, "label": "post_smooth"},
        "trajectory_tcn_mse_overlap_fused": {"color": "#9467bd", "lw": 1.0, "alpha": 0.84, "label": "overlap_fused"},
        "trajectory_tcn_mse_overlap_fused_post_smooth": {"color": "#17becf", "lw": 1.15, "alpha": 0.90, "label": "overlap_fused_post_smooth"},
    }
    for ax, dim in zip(axes, dims):
        ax.plot(xs, target_trial[:, dim], color="#111111", lw=1.45, alpha=0.95, label="Ground Truth")
        for name, trial in method_trials.items():
            style = styles.get(name, {"color": None, "lw": 1.0, "alpha": 0.8, "label": name})
            ax.plot(xs, trial[:, dim], color=style["color"], lw=style["lw"], alpha=style["alpha"], label=style["label"])
        ax.set_ylabel(KEY10_CHANNEL_NAMES[dim], fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("sample", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.985))
    mse_stats = title_stats.get("trajectory_tcn_mse", {})
    smooth_stats = title_stats.get("trajectory_tcn_smooth", {})
    post_stats = title_stats.get("trajectory_tcn_mse_post_smooth", {})
    fused_stats = title_stats.get("trajectory_tcn_mse_overlap_fused", {})
    combo_stats = title_stats.get("trajectory_tcn_mse_overlap_fused_post_smooth", {})
    title = (
        f"DB2 S{subject_id:02d} | idx {test_idx} | {subset_name.upper()} | score={dynamic_score:.4f}\n"
        f"MSE rmse={mse_stats.get('rmse', float('nan')):.4f} v={mse_stats.get('velocity_ratio', float('nan')):.2f} j={mse_stats.get('jerk_ratio', float('nan')):.2f} | "
        f"Smooth rmse={smooth_stats.get('rmse', float('nan')):.4f} v={smooth_stats.get('velocity_ratio', float('nan')):.2f} j={smooth_stats.get('jerk_ratio', float('nan')):.2f} | "
        f"Post rmse={post_stats.get('rmse', float('nan')):.4f} v={post_stats.get('velocity_ratio', float('nan')):.2f} j={post_stats.get('jerk_ratio', float('nan')):.2f}\n"
        f"Fuse rmse={fused_stats.get('rmse', float('nan')):.4f} v={fused_stats.get('velocity_ratio', float('nan')):.2f} j={fused_stats.get('jerk_ratio', float('nan')):.2f} | "
        f"Fuse+Post rmse={combo_stats.get('rmse', float('nan')):.4f} v={combo_stats.get('velocity_ratio', float('nan')):.2f} j={combo_stats.get('jerk_ratio', float('nan')):.2f}"
    )
    fig.suptitle(title, fontsize=10.0, fontweight="bold", y=0.94)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.76, hspace=0.42)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _trial_quality(pred_trial: np.ndarray, target_trial: np.ndarray) -> dict:
    quality = _trajectory_quality(pred_trial[None, :, :], target_trial[None, :, :])
    return {
        "rmse": float(np.sqrt(np.mean((pred_trial - target_trial) ** 2))),
        "velocity_ratio": quality["velocity_rms_ratio_pred_over_gt"],
        "jerk_ratio": quality["jerk_rms_ratio_pred_over_gt"],
    }


def _save_figures(subject_id: int, method_preds: dict[str, np.ndarray], target: np.ndarray,
                  selection: dict, figures_dir: Path) -> list[str]:
    saved = []
    plot_subsets = {k: ANGLE_SUBSETS[k] for k in ("mcp", "pip")}
    for rank, idx in enumerate(selection.get("selected_indices", []), start=1):
        score = float(selection["selected_scores"][rank - 1])
        trial_preds = {name: pred[idx] for name, pred in method_preds.items()}
        for subset_name, dims in plot_subsets.items():
            title_stats = {
                name: _trial_quality(pred[idx, :, dims], target[idx, :, dims])
                for name, pred in method_preds.items()
            }
            path = (
                figures_dir / f"S{subject_id:02d}" /
                f"S{subject_id:02d}_trial_{rank:02d}_idx{idx:04d}_score{score:.3f}_{subset_name}_DB2_sanity_compare.png"
            )
            _plot_subset_trace(path, target[idx], trial_preds, dims, subset_name, subject_id, int(idx), score, title_stats)
            saved.append(str(path))
    return saved


def _format_metric(v: float) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) and np.isfinite(float(v)) else "nan"


def _print_subject_report(subject_id: int, subsets: dict, dynamic: dict) -> None:
    print(f"\nDB2 S{subject_id:02d} metrics")
    for subset_name, vals in subsets.items():
        print(
            f"  {subset_name:<6} RMSE={_format_metric(vals['rmse'])} "
            f"MAE={_format_metric(vals['mae'])} Pearson={_format_metric(vals['pearson'])} "
            f"R2={_format_metric(vals['r2'])} nan_dims={vals.get('nan_dims', [])}"
        )
    splits = dynamic.get("splits", {})
    for split in ("train", "val", "test"):
        row = splits.get(split, {})
        if row:
            print(f"  dynamic {split:<5} {row['dynamic']}/{row['total']} ratio={row['ratio']:.4f}")


def _build_arg_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DB2 validation of the fixed Exp3 Key10 KinematicTCN downstream"
    )
    parser.add_argument("--subjects", default=",".join(str(s) for s in defaults["diag_subjects"]))
    parser.add_argument("--exercises", default=",".join(str(e) for e in defaults["diag_exercises"]))
    parser.add_argument("--viz-trials", type=int, default=int(defaults["diag_viz_trials_per_subject"]))
    parser.add_argument("--dynamic-min-ptp", type=float, default=float(defaults["diag_dynamic_min_ptp"]))
    parser.add_argument("--dynamic-viz-top-k", type=int, default=int(defaults["diag_dynamic_viz_top_k"]))
    parser.add_argument("--run-dir", default=None, help="Existing run dir. Defaults to MCIA_RUN_DIR or latest outputs/run/run_*.")
    parser.add_argument("--output-tag", default=None, help="Optional child directory under db2_angle_sanity/runs.")
    parser.add_argument("--seed", type=int, default=int(defaults.get("regressor_random_seed", 42)))
    return parser


def _continuous_method_result(pred: np.ndarray, target: np.ndarray, packed: dict,
                              calibration: dict, settings: dict,
                              dynamic_window_indices: np.ndarray,
                              window_starts: np.ndarray, training: dict) -> dict:
    continuous_pred = packed["continuous_prediction"]
    continuous_target = packed["continuous_target"]
    dynamic_mask = packed["continuous_dynamic_mask"]
    window_dynamic = (
        EXP3.evaluate_subsets(pred[dynamic_window_indices], target[dynamic_window_indices])
        if len(dynamic_window_indices) else {}
    )
    return {
        "model": "KinematicTCN",
        "loss": "mse",
        "training": training,
        "subsets": EXP3.evaluate_subsets(pred, target),
        "dynamic_subsets": window_dynamic,
        "continuous_subsets": EXP3.evaluate_subsets(
            continuous_pred[None, :, :], continuous_target[None, :, :]
        ),
        "continuous_dynamic_subsets": (
            EXP3.evaluate_subsets(
                continuous_pred[dynamic_mask][None, :, :],
                continuous_target[dynamic_mask][None, :, :],
            )
            if np.any(dynamic_mask) else {}
        ),
        "continuous_trajectory_quality": continuous_trajectory_quality(
            continuous_pred, continuous_target, packed["continuous_time_indices"]
        ),
        "continuous_overlap_consistency": window_overlap_prediction_consistency(
            pred, window_starts
        ),
        "continuous_output_postprocess": packed["metadata"],
        "continuous_output_calibration": {
            "applied": True,
            "fit_scope": calibration["fit_scope"],
            "gain_min": calibration["gain_min"],
            "gain_median": calibration["gain_median"],
            "gain_max": calibration["gain_max"],
        },
        "continuous_output_settings": settings,
    }


def _save_main_style_continuous_figures(subject_id: int, prediction: np.ndarray,
                                        packed: dict, selection: dict,
                                        figures_dir: Path, output_label: str,
                                        low_dynamic_coverage: bool) -> list[str]:
    target = np.asarray(packed["continuous_target"])
    times = np.asarray(packed["continuous_time_indices"], dtype=np.int64)
    fs = int(packed["metadata"]["target_fs"])
    saved = []
    for rank, segment in enumerate(selection.get("selected_segments", []), start=1):
        left = int(segment["local_start"])
        right = int(segment["local_end_exclusive"])
        target_segment = target[left:right]
        pred_segment = prediction[left:right]
        xs = np.arange(len(target_segment), dtype=np.float32) / float(fs)
        fig, axes = plt.subplots(5, 2, figsize=(18.0, 12.0), sharex=True)
        axes_arr = np.asarray(axes, dtype=object).reshape(5, 2)
        for dim, (name, ax) in enumerate(zip(KEY10_CHANNEL_NAMES, axes_arr.flat)):
            ax.plot(xs, target_segment[:, dim], color="#111111", lw=1.45, alpha=0.95, label="Ground Truth")
            ax.plot(xs, pred_segment[:, dim], color="#1f77b4", lw=1.1, alpha=0.88, label="DB2 Pred")
            ax.set_title(name, fontsize=9.5, fontweight="bold", pad=3)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)
            if dim % 2 == 0:
                ax.set_ylabel(name, fontsize=9)
        for ax in axes_arr[-1, :]:
            ax.set_xlabel("time within continuous test segment (s)", fontsize=9)
        handles, labels = axes_arr.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 0.975))
        rmse = float(np.sqrt(np.mean((pred_segment - target_segment) ** 2)))
        title = (
            f"DB2 S{subject_id:02d} | Key10 | {len(target_segment) / fs:.2f}s continuous offline "
            f"test trajectory | Segment {rank:02d} | "
            f"time idx {segment['time_start']}-{segment['time_end_exclusive'] - 1}"
        )
        detail = (
            f"{segment['selection_mode']} | dynamic score={segment['dynamic_score']:.4f} | "
            f"dynamic rank={segment['dynamic_rank'] if segment['dynamic_rank'] is not None else 'NA'} | "
            f"segment RMSE={rmse:.4f} | overlap windows min/mean="
            f"{segment['min_overlap']}/{segment['mean_overlap']:.2f}"
        )
        if low_dynamic_coverage:
            detail += " | low dynamic coverage"
        fig.suptitle(title + "\n" + detail + "\n" + output_label,
                     fontsize=12.5, fontweight="bold", y=0.955)
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.07, top=0.80,
                            hspace=0.55, wspace=0.22)
        path = (
            figures_dir / f"S{subject_id:02d}" /
            f"S{subject_id:02d}_continuous_{rank:02d}_"
            f"t{segment['time_start']:06d}-{segment['time_end_exclusive'] - 1:06d}_"
            f"score{EXP3._score_for_filename(float(segment['dynamic_score']))}_key10_DB2_sanity.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))
    return saved


def _write_ablation_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        vals = []
        for key in keys:
            val = row.get(key, "")
            if isinstance(val, float):
                vals.append(f"{val:.8g}")
            else:
                vals.append(str(val))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_summary_row(name: str, method: dict, ablation_type: str, extra: dict | None = None) -> dict:
    full = method["full_metrics"]["global"]
    dyn = method["dynamic_subset_metrics"]
    dyn_global = dyn.get("subsets", {}).get("global", {}) if not dyn.get("skipped") else {}
    q = method.get("trajectory_quality", {})
    row = {
        "ablation": ablation_type,
        "method": name,
        "full_rmse": full.get("rmse", float("nan")),
        "full_mae": full.get("mae", float("nan")),
        "full_pearson": full.get("pearson", float("nan")),
        "full_r2": full.get("r2", float("nan")),
        "dynamic_rmse": dyn_global.get("rmse", float("nan")),
        "dynamic_pearson": dyn_global.get("pearson", float("nan")),
        "dynamic_r2": dyn_global.get("r2", float("nan")),
        "velocity_ratio": q.get("velocity_rms_ratio_pred_over_gt", float("nan")),
        "jerk_ratio": q.get("jerk_rms_ratio_pred_over_gt", float("nan")),
        "spectral_error": q.get("spectrum_l1_normalized", float("nan")),
    }
    if extra:
        row.update(extra)
    return row


def _run_trajectory_variant(emg: np.ndarray, angle_variant: np.ndarray, train_idx: np.ndarray,
                            val_idx: np.ndarray, test_idx: np.ndarray, config: dict, device: str,
                            lambda_velocity: float, lambda_smooth: float,
                            method_name: str, model_type: str = "tcn") -> tuple[dict, np.ndarray, np.ndarray, dict]:
    model, train_info = _train_subject_model(
        emg,
        angle_variant,
        train_idx,
        val_idx,
        config,
        device,
        lambda_velocity=lambda_velocity,
        lambda_smooth=lambda_smooth,
        model_type=model_type,
    )
    pred, target = _predict(model, emg, angle_variant, test_idx, config, device)
    scores = EXP3.compute_dynamic_angle_scores(target, "global_max_ptp")
    dyn_idx = np.flatnonzero(scores >= float(config["diag_dynamic_min_ptp"])).astype(np.int64)
    result = _trajectory_method_result(
        pred,
        target,
        dyn_idx,
        train_info,
        {"model_type": model_type, "method_name": method_name},
    )
    return result, pred, target, {"dynamic_scores": scores, "dynamic_test_indices": dyn_idx}


def _save_ablation_figures(subject_id: int, ablation_name: str, variant_name: str,
                           method_preds: dict[str, np.ndarray], target: np.ndarray,
                           selection: dict, figures_dir: Path) -> list[str]:
    out_dir = figures_dir / f"S{subject_id:02d}" / ablation_name / variant_name
    return _save_figures(subject_id, method_preds, target, selection, out_dir.parent)


def _run_lag_ablation(subject_id: int, emg: np.ndarray, angle: np.ndarray, train_idx: np.ndarray,
                      val_idx: np.ndarray, test_idx: np.ndarray, config: dict, device: str,
                      figures_dir: Path, lambda_velocity: float, lambda_smooth: float) -> dict:
    lag_ms_values = [-600, -400, -200, 0, 200]
    variants = []
    rows = []
    for lag_ms in lag_ms_values:
        lag_samples = int(round(float(lag_ms) / 1000.0 * float(config["target_fs"])))
        angle_lag = _shift_angle_windows(angle, lag_samples)
        method, pred, target, dyn = _run_trajectory_variant(
            emg, angle_lag, train_idx, val_idx, test_idx, config, device,
            lambda_velocity, lambda_smooth, f"lag_{lag_ms}ms"
        )
        selection = _select_visualization_indices(target, config)
        fig_files = _save_figures(
            subject_id,
            {f"lag_{lag_ms}ms": pred},
            target,
            selection,
            figures_dir / f"S{subject_id:02d}" / "ablation_lag" / f"lag_{lag_ms}ms",
        )
        row = _method_summary_row(f"lag_{lag_ms}ms", method, "lag", {"lag_ms": lag_ms, "lag_samples": lag_samples})
        rows.append(row)
        variants.append({
            "lag_ms": int(lag_ms),
            "lag_samples": int(lag_samples),
            "method": method,
            "figure_files": fig_files,
        })
    rows_sorted = sorted(rows, key=lambda r: (r["dynamic_rmse"], r["jerk_ratio"]))
    return {"variants": variants, "summary_rows": rows_sorted, "best": rows_sorted[0] if rows_sorted else None}


def _run_glove_preprocess_ablation(subject_id: int, emg: np.ndarray, angle: np.ndarray, train_idx: np.ndarray,
                                   val_idx: np.ndarray, test_idx: np.ndarray, config: dict, device: str,
                                   figures_dir: Path, lambda_velocity: float, lambda_smooth: float,
                                   selected: str) -> dict:
    variants_map = {
        "direct": angle.copy(),
        "lowpass": _lowpass_angle_windows(angle, int(config["target_fs"]), 10.0),
    }
    if selected in variants_map:
        run_items = [(selected, variants_map[selected])]
    else:
        run_items = list(variants_map.items())
    variants = []
    rows = []
    for name, target_angle in run_items:
        method, pred, target, dyn = _run_trajectory_variant(
            emg, target_angle, train_idx, val_idx, test_idx, config, device,
            lambda_velocity, lambda_smooth, f"glove_{name}"
        )
        selection = _select_visualization_indices(target, config)
        fig_files = _save_figures(
            subject_id,
            {f"glove_{name}": pred},
            target,
            selection,
            figures_dir / f"S{subject_id:02d}" / "ablation_glove_preprocess" / name,
        )
        row = _method_summary_row(f"glove_{name}", method, "glove_preprocess", {"glove_preprocess": name})
        rows.append(row)
        variants.append({"glove_preprocess": name, "method": method, "figure_files": fig_files})
    rows_sorted = sorted(rows, key=lambda r: (r["dynamic_rmse"], r["jerk_ratio"]))
    return {"variants": variants, "summary_rows": rows_sorted, "best": rows_sorted[0] if rows_sorted else None}


def _run_smooth_loss_ablation(subject_id: int, emg: np.ndarray, angle: np.ndarray, train_idx: np.ndarray,
                              val_idx: np.ndarray, test_idx: np.ndarray, config: dict, device: str,
                              figures_dir: Path) -> dict:
    velocity_grid = [0.05, 0.1, 0.2]
    smooth_grid = [0.005, 0.01, 0.05]
    variants = []
    rows = []
    for lv in velocity_grid:
        for ls in smooth_grid:
            name = f"lv{lv:g}_ls{ls:g}".replace(".", "p")
            method, pred, target, dyn = _run_trajectory_variant(
                emg, angle, train_idx, val_idx, test_idx, config, device, lv, ls, name
            )
            row = _method_summary_row(name, method, "smooth_loss", {"lambda_velocity": lv, "lambda_smooth": ls})
            rows.append(row)
            variants.append({"lambda_velocity": lv, "lambda_smooth": ls, "method": method})
    rows_sorted = sorted(rows, key=lambda r: (r["dynamic_rmse"], r["jerk_ratio"]))
    if rows_sorted:
        best = rows_sorted[0]
        best_name = best["method"]
        best_variant = next(v for v in variants if f"lv{v['lambda_velocity']:g}_ls{v['lambda_smooth']:g}".replace(".", "p") == best_name)
        method, pred, target, dyn = _run_trajectory_variant(
            emg, angle, train_idx, val_idx, test_idx, config, device,
            best_variant["lambda_velocity"], best_variant["lambda_smooth"], f"best_{best_name}"
        )
        selection = _select_visualization_indices(target, config)
        fig_files = _save_figures(
            subject_id,
            {f"smooth_best_{best_name}": pred},
            target,
            selection,
            figures_dir / f"S{subject_id:02d}" / "ablation_smooth_loss" / best_name,
        )
        best_variant["figure_files"] = fig_files
    return {"variants": variants, "summary_rows": rows_sorted, "best": rows_sorted[0] if rows_sorted else None}


def main() -> None:
    raw_cfg, config = _load_config()
    parser = _build_arg_parser(config)
    args = parser.parse_args()
    config["diag_subjects"] = _parse_int_list(args.subjects)
    config["diag_exercises"] = _parse_int_list(args.exercises)
    config["diag_viz_trials_per_subject"] = int(args.viz_trials)
    config["diag_dynamic_min_ptp"] = float(args.dynamic_min_ptp)
    config["diag_dynamic_viz_top_k"] = int(args.dynamic_viz_top_k)
    config["regressor_random_seed"] = int(args.seed)
    if int(config["regressor_n_angle_channels"]) != KEY10_DIM:
        raise ValueError(f"DB2 sanity requires fixed Key10 output dimension {KEY10_DIM}.")

    settings = EXP3._continuous_output_config(config)
    if not settings["enabled"]:
        raise RuntimeError("DB2 sanity requires the locked Exp3 continuous output rule.")

    run_dir = _resolve_run_dir(PROJECT_ROOT, raw_cfg, args.run_dir)
    out_root = run_dir / "06_diagnostics" / "db2_angle_sanity"
    if args.output_tag:
        output_tag = str(args.output_tag).strip()
        if not output_tag or Path(output_tag).name != output_tag:
            raise ValueError("--output-tag must be a single directory name")
        out_root = out_root / "runs" / output_tag
    metrics_dir = out_root / "metrics"
    pred_dir = out_root / "predictions"
    fig_dir = out_root / "figures"
    ckpt_dir = out_root / "checkpoints"
    log_path = out_root / "logs" / "db2_angle_sanity.log"
    for path in (metrics_dir, pred_dir, fig_dir, ckpt_dir, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["regressor_random_seed"]))
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=int(config["orig_fs"]))
    device = str(config["device"])
    start_ts = time.time()
    report = {
        "diagnostic": "db2_angle_regressor_sanity",
        "purpose": "DB2 validation of the fixed Exp3 downstream KinematicTCN",
        "angle_target": key10_target_metadata(),
        "run_dir": str(run_dir),
        "output_dir": str(out_root),
        "subjects": [],
        "config": {
            "subjects": config["diag_subjects"],
            "exercises": config["diag_exercises"],
            "train_reps": [1, 3, 4, 6],
            "test_reps": [2, 5],
            "split_method": "make_rep_split",
            "model": "KinematicTCN",
            "loss": "mse",
            "continuous_viz_points": int(config["regressor_continuous_viz_points"]),
            "continuous_viz_min_overlap": int(config["regressor_continuous_viz_min_overlap"]),
            "viz_trials_per_subject": int(config["diag_viz_trials_per_subject"]),
            "dynamic_min_ptp": float(config["diag_dynamic_min_ptp"]),
            "dynamic_viz_top_k": int(config["diag_dynamic_viz_top_k"]),
            "continuous_output": settings,
            "device": device,
        },
    }

    _log(log_path, f"DB2 fixed downstream validation started | run_dir={run_dir}")
    for subject_id in config["diag_subjects"]:
        subject_seed = int(config["regressor_random_seed"]) + int(subject_id)
        set_seed(subject_seed)
        try:
            emg, angle, _sids, reps = prepare_kinematics_data(
                loader, [subject_id], config, exercises=config["diag_exercises"], db="db2"
            )
            train_idx, val_idx, test_idx = make_rep_split(
                reps, train_reps=(1, 3, 4, 6), test_reps=(2, 5),
                val_ratio=float(config["regressor_val_ratio"]), seed=subject_seed,
            )
            dynamic = _dynamic_diagnostics(angle, train_idx, val_idx, test_idx, config)
            dynamic_test_indices = np.asarray(
                dynamic["splits"]["test"]["dynamic_indices"], dtype=np.int64
            )
            model, best_val, epochs = EXP3.train_tcn_on_emg(
                emg, angle, train_idx, val_idx, config, device
            )
            training = {
                "best_val_loss": float(best_val),
                "epochs": int(epochs),
                "model": "KinematicTCN",
                "loss": "mse",
                "training_rule": "Exp3.train_tcn_on_emg",
            }
            subject_ckpt_dir = ckpt_dir / f"S{subject_id:02d}"
            subject_ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = subject_ckpt_dir / "db2_angle_sanity_kinematic_tcn_best.pth"
            torch.save(model.state_dict(), ckpt_path)

            pred_raw, target = EXP3.predict_on_set(model, emg, angle, test_idx, config, device)
            pred_val, target_val = EXP3.predict_on_set(model, emg, angle, val_idx, config, device)
            calibration = EXP3._fit_channelwise_affine(pred_val, target_val)
            pred_calibrated = EXP3._apply_channelwise_affine(pred_raw, calibration)
            window_starts = _reconstruct_window_starts(
                loader, subject_id, config["diag_exercises"], config, reps
            )
            test_window_starts = window_starts[test_idx]
            packed = postprocess_window_predictions(
                pred_calibrated,
                target,
                test_window_starts,
                dynamic_test_indices,
                savgol_window=settings["savgol_window"],
                polyorder=settings["savgol_polyorder"],
                fusion=settings["fusion"],
                filter_kind=settings["filter_kind"],
                target_fs=int(config["target_fs"]),
                lowpass_hz=settings["lowpass_hz"],
            )
            method = _continuous_method_result(
                pred_raw, target, packed, calibration, settings,
                dynamic_test_indices, test_window_starts, training
            )
            selection_config = dict(config)
            selection_config["regressor_viz_trials_per_subject"] = int(config["diag_viz_trials_per_subject"])
            selection_config["regressor_dynamic_min_ptp"] = float(config["diag_dynamic_min_ptp"])
            selection_config["regressor_dynamic_viz_top_k"] = int(config["diag_dynamic_viz_top_k"])
            selection = EXP3.build_abc_visualization_selection(
                packed["continuous_target"],
                packed["continuous_time_indices"],
                packed["continuous_overlap_counts"],
                packed["continuous_dynamic_mask"],
                selection_config,
                bool(dynamic["low_dynamic_train_coverage"]),
            )
            figure_files = _save_main_style_continuous_figures(
                subject_id,
                packed["continuous_prediction"],
                packed,
                selection,
                fig_dir,
                EXP3.continuous_output_label(config),
                bool(dynamic["low_dynamic_train_coverage"]),
            )
            pred_path = pred_dir / f"S{subject_id:02d}_angle_sanity_predictions.npz"
            np.savez(
                pred_path,
                **key10_prediction_metadata(),
                target=target,
                pred=pred_raw,
                pred_continuous_windows=packed["processed_windows"],
                pred_continuous=packed["continuous_prediction"],
                continuous_target=packed["continuous_target"],
                continuous_time_indices=packed["continuous_time_indices"],
                continuous_dynamic_mask=packed["continuous_dynamic_mask"],
                continuous_overlap_counts=packed["continuous_overlap_counts"],
                continuous_output_fusion=np.asarray(settings["fusion"]),
                continuous_output_filter=np.asarray(settings["filter_kind"]),
                continuous_output_lowpass_hz=np.asarray(settings["lowpass_hz"], dtype=np.float32),
                continuous_output_calibration=np.asarray(settings["calibration"]),
                test_idx=test_idx,
                test_window_starts=test_window_starts,
                dynamic_test_indices=dynamic_test_indices,
                dynamic_test_scores=EXP3.compute_dynamic_angle_scores(target, "global_max_ptp"),
                continuous_calibration_gain=calibration["gain"],
                continuous_calibration_bias=calibration["bias"],
            )
            subject_result = {
                "subject_id": int(subject_id),
                "subject_seed": int(subject_seed),
                "status": "ok",
                "n_segments": int(len(emg)),
                "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
                "split_method": "rep_based_make_rep_split",
                "angle_target": key10_target_metadata(),
                "dynamic_angle_coverage": dynamic,
                "low_dynamic_train_coverage": bool(dynamic["low_dynamic_train_coverage"]),
                "methods": {"kinematic_tcn": method},
                "subsets": method["subsets"],
                "dynamic_subsets": method["dynamic_subsets"],
                "continuous_subsets": method["continuous_subsets"],
                "continuous_dynamic_subsets": method["continuous_dynamic_subsets"],
                "continuous_trajectory_quality": method["continuous_trajectory_quality"],
                "continuous_overlap_consistency": method["continuous_overlap_consistency"],
                "continuous_output": method["continuous_output_settings"],
                "visualization_selection": selection,
                "prediction_file": str(pred_path),
                "checkpoint_file": str(ckpt_path),
                "figure_files": figure_files,
            }
            report["subjects"].append(subject_result)
            _save_json(metrics_dir / "db2_angle_sanity_results.json", report)
            _print_subject_report(subject_id, method["subsets"], dynamic)
            continuous_global = method["continuous_subsets"]["global"]
            consistency = method["continuous_overlap_consistency"]
            print(
                f"  fixed continuous global RMSE={continuous_global['rmse']:.4f} "
                f"Pearson={_format_metric(continuous_global['pearson'])} "
                f"overlap_RMSE={_format_metric(consistency['overlap_rmse'])} "
                f"figures={len(figure_files)}"
            )
            _log(log_path, f"S{subject_id:02d} done | figures={len(figure_files)} | prediction={pred_path}")
        except Exception as exc:
            failed = {"subject_id": int(subject_id), "status": "failed", "reason": repr(exc)}
            report["subjects"].append(failed)
            _save_json(metrics_dir / "db2_angle_sanity_results.json", report)
            _log(log_path, f"S{subject_id:02d} failed: {repr(exc)}")
            print(f"DB2 S{subject_id:02d} failed: {exc}")

    report["elapsed_min"] = (time.time() - start_ts) / 60.0
    _save_json(metrics_dir / "db2_angle_sanity_results.json", report)
    _log(log_path, f"DB2 fixed downstream validation finished | output={out_root}")
    print("\nOutput run dir:", run_dir)
    print("Metrics JSON:", metrics_dir / "db2_angle_sanity_results.json")
    print("Figures dir:", fig_dir)


if __name__ == "__main__":
    main()
