"""Formal DB3 48-class gesture-recognition main experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from data.dataset_db2_emg import moving_average
from data.ninapro_loader import NinaProDataLoader
from models.prediction.kinematic_regressor import TemporalBlock
from utils.paper_pipeline import build_mcia, flatten_pipeline_config, load_mcia_state_dict, load_yaml_config, patch_boundary_crossfade, set_seed
from utils.db3_quality_mask import db3_quality_mask


def set_reproducible_seed(seed: int) -> None:
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class GestureWindows:
    emg: np.ndarray
    labels: np.ndarray
    repetitions: np.ndarray
    starts: np.ndarray
    quality_mask: np.ndarray


class GestureDataset(Dataset):
    def __init__(self, emg: np.ndarray, labels: np.ndarray):
        self.emg = torch.as_tensor(emg, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.emg)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"emg": self.emg[index], "label": self.labels[index]}


class GestureTCN(nn.Module):
    """Small subject-specific temporal classifier with global time pooling."""

    def __init__(self, n_emg_channels: int, n_classes: int, hidden_dim: int, n_layers: int,
                 kernel_size: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = n_emg_channels
        for level in range(n_layers):
            layers.append(TemporalBlock(in_channels, hidden_dim, kernel_size, 2 ** level, dropout))
            in_channels = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.tcn(emg.transpose(1, 2)).mean(dim=-1))


def load_db3_windows(loader: NinaProDataLoader, subject_id: int, exercise: int, config: dict,
                     min_label_ratio: float, min_repetition_ratio: float) -> GestureWindows:
    """Create action-pure windows using train-repetition-only input scaling."""
    data = loader.load_db3_subject(subject_id, [exercise])
    quality_keep, quality_report = db3_quality_mask(
        data["emg"], data.get("restimulus", data.get("stimulus")), data["repetition"],
        int(config["target_fs"]), tuple(int(v) for v in config["gesture_action_ids"]),
    )
    emg = loader.notch_filter(loader.bandpass_filter(data["emg"].astype(np.float32) * 1000.0))
    factor = int(config["orig_fs"] / config["target_fs"])
    emg_down = moving_average(np.abs(emg), factor)[::factor]
    labels = np.asarray(data.get("restimulus", data.get("stimulus")))[::factor].reshape(-1)
    repetitions = np.asarray(data["repetition"])[::factor].reshape(-1)
    n_samples = min(len(emg_down), len(labels), len(repetitions))
    size, stride, center = int(config["window_size"]), int(config["stride"]), int(config["window_size"]) // 2
    raw_windows, window_labels, window_reps, starts, masks = [], [], [], [], []
    for start in range(0, n_samples - size + 1, stride):
        end = start + size
        label_window, rep_window = labels[start:end], repetitions[start:end]
        label, repetition = int(label_window[center]), int(rep_window[center])
        if label <= 0 or repetition <= 0:
            continue
        if np.mean(label_window == label) < min_label_ratio or np.mean(rep_window == repetition) < min_repetition_ratio:
            continue
        raw_windows.append(emg_down[start:end].astype(np.float32, copy=False))
        masks.append(quality_keep[start:end])
        window_labels.append(label); window_reps.append(repetition); starts.append(start)
    if not raw_windows:
        raise ValueError(f"S{subject_id:02d} has no action-pure gesture windows")
    raw_windows = np.stack(raw_windows)
    window_reps = np.asarray(window_reps, dtype=np.int64)
    train_values = raw_windows[np.isin(window_reps, (1, 3, 4))].reshape(-1, raw_windows.shape[-1])
    emg_max = float(train_values.max())
    def compress(values):
        return values if emg_max <= 0.0 else np.log1p(255.0 * values / emg_max) / np.log1p(255.0) * emg_max
    train_values = compress(train_values)
    q05, q99 = np.percentile(train_values, [5, 99])
    emg_norm = np.clip((compress(raw_windows) - q05) / (q99 - q05 + 1e-8), 0.0, 1.0).astype(np.float32)
    return GestureWindows(emg_norm, np.asarray(window_labels, dtype=np.int64),
                          window_reps, np.asarray(starts, dtype=np.int64),
                          np.stack(masks).astype(np.float32)), quality_report


def split_indices(repetitions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.flatnonzero(np.isin(repetitions, (1, 3, 4)))
    validation = np.flatnonzero(repetitions == 6)
    test = np.flatnonzero(np.isin(repetitions, (2, 5)))
    if not len(train) or not len(validation) or not len(test):
        raise ValueError(f"Invalid split train={len(train)}, val={len(validation)}, test={len(test)}")
    return train, validation, test


def class_mapping(train_labels: np.ndarray) -> tuple[dict[int, int], list[int]]:
    action_ids = sorted(int(value) for value in np.unique(train_labels) if value > 0)
    return {action: index for index, action in enumerate(action_ids)}, action_ids


def encode_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    missing = sorted(set(int(value) for value in labels) - set(mapping))
    if missing:
        raise ValueError(f"Actions missing from train repetitions: {missing}")
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)


def healthy_checkpoint(config: dict) -> Path:
    candidates = [
        Path(config.get("exp1_dir", "")) / "checkpoints" / "best_model.pth",
        Path(config["checkpoints_dir"]) / "exp1_mcia_db2" / "best_model.pth",
    ]
    output_dir = Path(config["output_dir"]) / "exp1_mcia_db2"
    if output_dir.exists():
        candidates.extend(sorted(output_dir.glob("run_*/best_model.pth"), reverse=True))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No healthy-prior MCIA checkpoint was found")


@torch.no_grad()
def apply_mcia(mcia: nn.Module, raw_windows: np.ndarray, masks: np.ndarray,
               device: str, batch_size: int, patch_size: int = 8) -> tuple[np.ndarray, dict]:
    enhanced = np.empty_like(raw_windows)
    for start in range(0, len(raw_windows), batch_size):
        stop = min(start + batch_size, len(raw_windows))
        raw = torch.as_tensor(raw_windows[start:stop], dtype=torch.float32, device=device)
        mask = torch.as_tensor(masks[start:stop], dtype=torch.float32, device=device)
        valid_channels = (mask.mean(dim=1) > 0.5).float()
        completed = mcia(raw * mask, raw_time_mask=mask, chan_valid_mask=valid_channels)
        # 交付规则（2026-09-09 采纳）：clip 后对 patch 边界做三点淡化，再复制回观测值。
        completed = patch_boundary_crossfade(completed.clamp(0.0, 1.0), patch_size)
        enhanced[start:stop] = (completed * (1.0 - mask) + raw * mask).cpu().numpy()
    return enhanced, {"mask": masks}


def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).tolist(),
    }


def trial_majority_metrics(y_true: np.ndarray, y_pred: np.ndarray, repetitions: np.ndarray,
                           n_classes: int) -> dict:
    trial_true, trial_pred = [], []
    for repetition in np.unique(repetitions):
        for action in np.unique(y_true[repetitions == repetition]):
            values = y_pred[(repetitions == repetition) & (y_true == action)]
            if len(values):
                trial_true.append(int(action))
                trial_pred.append(int(np.bincount(values, minlength=n_classes).argmax()))
    return {"n_action_repetition_trials": int(len(trial_true)),
            **metrics(np.asarray(trial_true), np.asarray(trial_pred), n_classes)}


def train_classifier(emg: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                     config: dict, device: str, seed: int) -> tuple[GestureTCN, dict]:
    set_reproducible_seed(seed)
    model = GestureTCN(emg.shape[-1], int(labels.max()) + 1, int(config["regressor_hidden_dim"]),
                       int(config["regressor_n_layers"]), int(config["regressor_kernel_size"]),
                       float(config["regressor_model_dropout"])).to(device)
    batch_size = int(config["regressor_batch_size"])
    train_loader = DataLoader(GestureDataset(emg[train_idx], labels[train_idx]), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(GestureDataset(emg[val_idx], labels[val_idx]), batch_size=batch_size,
                            shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config["regressor_learning_rate"]), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(config["regressor_lr_factor"]),
        patience=int(config["regressor_lr_patience"]))
    best_state, best_macro_f1, no_improve, history = None, -np.inf, 0, []
    n_classes = int(labels.max()) + 1
    for epoch in range(int(config["regressor_num_epochs"])):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            logits = model(batch["emg"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(logits)
            seen += len(logits)
        model.eval()
        true_values, predicted_values = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["emg"].to(device))
                predicted_values.append(logits.argmax(dim=1).cpu().numpy())
                true_values.append(batch["label"].numpy())
        validation = metrics(np.concatenate(true_values), np.concatenate(predicted_values), n_classes)
        scheduler.step(validation["macro_f1"])
        history.append({"epoch": epoch + 1, "train_cross_entropy": float(total_loss / max(seen, 1)),
                        "val_macro_f1": validation["macro_f1"]})
        if validation["macro_f1"] > best_macro_f1 + 1e-8:
            best_macro_f1 = validation["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(config["regressor_patience"]):
            break
    if best_state is None:
        raise RuntimeError("GestureTCN produced no validation state")
    model.load_state_dict(best_state)
    return model, {"best_validation_macro_f1": float(best_macro_f1), "epochs": int(len(history)), "history": history}


@torch.no_grad()
def predict(model: GestureTCN, emg: np.ndarray, indices: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    loader = DataLoader(GestureDataset(emg[indices], np.zeros(len(indices), dtype=np.int64)),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    return np.concatenate([model(batch["emg"].to(device)).argmax(dim=1).cpu().numpy() for batch in loader])


def save_confusion_plot(matrix: np.ndarray, action_ids: list[int], title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=axis, label="window count")
    ticks = np.arange(len(action_ids))
    axis.set_xticks(ticks, [str(value) for value in action_ids], rotation=45, ha="right")
    axis.set_yticks(ticks, [str(value) for value in action_ids])
    axis.set_xlabel("Predicted action")
    axis.set_ylabel("True action")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def mask_summary(metadata: dict) -> dict:
    mask = metadata["mask"]
    return {
        "masked_token_ratio": float((mask < 0.5).mean()),
        "windows_with_any_mask_ratio": float((mask < 0.5).any(axis=(1, 2)).mean()),
    }


def main() -> None:
    config = flatten_pipeline_config(load_yaml_config(ROOT))
    device = config["device"]
    run_dir = Path(config["run_dir"])
    out_dir = run_dir / "04_gesture_recognition"
    for name in ("metrics", "predictions", "figures", "checkpoints", "logs"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    healthy_path = healthy_checkpoint(config)
    healthy_mcia = build_mcia(config, device)
    load_mcia_state_dict(healthy_mcia, healthy_path, device)
    healthy_mcia.eval()
    action_ids_expected = [int(value) for value in config["gesture_action_ids"]]
    report = {
        "task": "DB3 subject-internal 48-class gesture recognition",
        "source_exercises": [int(value) for value in config["gesture_exercises"]],
        "actions": action_ids_expected,
        "excluded_actions": [49],
        "split": {"train_repetitions": [1, 3, 4], "validation_repetitions": [6], "test_repetitions": [2, 5]},
        "leakage_control": {
            "window_policy": "action/repetition pure windows; no cross-exercise windows",
            "quality_mask": "hard zero channels fitted on training repetitions 1/3/4; fixed MQP p>0.20 evaluated per input second",
            "normalization": "per subject and exercise, train-free EMG preprocessing; no labels enter enhancement",
            "model_selection": "validation macro-F1 only",
        },
        "model": {"name": "GestureTCN", "same_architecture_for_A_B": True},
        "subjects": [],
    }
    for subject_id in config["gesture_subjects"]:
        print(f"\n[Gesture] DB3 S{subject_id:02d}", flush=True)
        raw_parts, b_parts, labels_parts, reps_parts, starts_parts, quality_parts = [], [], [], [], [], []
        detector_report = {}
        for exercise in config["gesture_exercises"]:
            one, quality_report = load_db3_windows(loader, subject_id, int(exercise), config,
                                   float(config["gesture_min_label_ratio"]),
                                   float(config["gesture_min_repetition_ratio"]))
            keep = np.isin(one.labels, action_ids_expected)
            one = GestureWindows(one.emg[keep], one.labels[keep], one.repetitions[keep], one.starts[keep], one.quality_mask[keep])
            b_values, b_meta = apply_mcia(healthy_mcia, one.emg, one.quality_mask, device,
                                          int(config["regressor_batch_size"]),
                                          patch_size=int(config["patch_size"]))
            raw_parts.append(one.emg); b_parts.append(b_values)
            labels_parts.append(one.labels); reps_parts.append(one.repetitions); starts_parts.append(one.starts); quality_parts.append(one.quality_mask)
            detector_report[f"E{exercise}"] = {
                "two_layer_quality_mask": quality_report,
                "healthy_mask": mask_summary(b_meta),
            }
        windows = GestureWindows(np.concatenate(raw_parts), np.concatenate(labels_parts),
                                 np.concatenate(reps_parts), np.concatenate(starts_parts),
                                 np.concatenate(quality_parts))
        train_idx, val_idx, test_idx = split_indices(windows.repetitions)
        mapping, action_ids = class_mapping(windows.labels[train_idx])
        if action_ids != action_ids_expected:
            raise ValueError(f"S{subject_id:02d} does not have the locked 48-class train set: {action_ids}")
        encoded = encode_labels(windows.labels, mapping)
        conditions = {"A_raw": np.concatenate(raw_parts), "B_healthy_prior": np.concatenate(b_parts)}
        condition_report = {}
        for condition, emg in conditions.items():
            model, training = train_classifier(emg, encoded, train_idx, val_idx, config, device,
                                               int(config["gesture_random_seed"]) + subject_id)
            prediction = predict(model, emg, test_idx, int(config["regressor_batch_size"]), device)
            truth = encoded[test_idx]
            test_metrics = metrics(truth, prediction, len(action_ids))
            trial_metrics = trial_majority_metrics(truth, prediction, windows.repetitions[test_idx], len(action_ids))
            ckpt = out_dir / "checkpoints" / f"S{subject_id:02d}_{condition}.pth"
            pred_path = out_dir / "predictions" / f"S{subject_id:02d}_{condition}_test.npz"
            fig_path = out_dir / "figures" / f"S{subject_id:02d}_{condition}_48class_test_confusion.png"
            torch.save(model.state_dict(), ckpt)
            np.savez_compressed(pred_path, original_window_index=test_idx, repetition=windows.repetitions[test_idx],
                                true_action=windows.labels[test_idx],
                                predicted_action=np.asarray([action_ids[value] for value in prediction]),
                                true_class=truth, predicted_class=prediction)
            save_confusion_plot(np.asarray(test_metrics["confusion_matrix"]), action_ids,
                                f"DB3 S{subject_id:02d} {condition}: 48-class test confusion", fig_path)
            condition_report[condition] = {
                "training": training, "test_window_metrics": test_metrics,
                "test_trial_majority_metrics": trial_metrics, "checkpoint": str(ckpt),
                "prediction_file": str(pred_path), "confusion_figure": str(fig_path)}
            print(f"  {condition}: test macro-F1={test_metrics['macro_f1']:.4f} accuracy={test_metrics['accuracy']:.4f}", flush=True)
        report["subjects"].append({
            "subject_id": int(subject_id), "status": "ok", "n_windows": int(len(windows.emg)),
            "split_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "rule_detector": detector_report, "groups": condition_report,
            "delta_macro_f1_vs_A": {
                "B": float(condition_report["B_healthy_prior"]["test_window_metrics"]["macro_f1"] - condition_report["A_raw"]["test_window_metrics"]["macro_f1"]),
            }})
    path = Path(config["gesture_results_path"])
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Gesture] results: {path}", flush=True)


if __name__ == "__main__":
    main()
