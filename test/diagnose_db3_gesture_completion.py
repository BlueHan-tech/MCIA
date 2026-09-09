"""Isolated DB3 gesture-recognition diagnostic for raw versus MCIA-enhanced EMG.

This script does not modify Exp1, Exp2, or Exp3. It evaluates a fixed,
subject-specific GestureTCN on DB3 E1 actions using a repetition-held-out split:
train repetitions 1/3/4, validation repetition 6, and test repetitions 2/5.
"""
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
from utils.paper_pipeline import build_mcia, flatten_pipeline_config, load_mcia_state_dict, load_yaml_config, set_seed
from utils.rule_anomaly_detector import RuleAnomalyDetector


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
    """Reuse Exp3 EMG preprocessing while retaining exact action-window metadata."""
    data = loader.load_db3_subject(subject_id, [exercise])
    emg = data["emg"] * 1000.0
    emg = loader.bandpass_filter(emg)
    emg = loader.notch_filter(emg)
    factor = int(config["orig_fs"] / config["target_fs"])
    emg_down = moving_average(np.abs(emg), factor)[::factor]
    emg_max = float(emg_down.max())
    if emg_max > 0.0:
        emg_down = np.log1p(255.0 * emg_down / emg_max) / np.log1p(255.0) * emg_max
    q05, q99 = np.percentile(emg_down, [5, 99])
    emg_norm = np.clip((emg_down - q05) / (q99 - q05 + 1e-8), 0.0, 1.0)

    labels = np.asarray(data.get("restimulus", data.get("stimulus")))[::factor].reshape(-1)
    repetitions = np.asarray(data.get("repetition"))[::factor].reshape(-1)
    n_samples = min(len(emg_norm), len(labels), len(repetitions))
    size, stride, center = int(config["window_size"]), int(config["stride"]), int(config["window_size"]) // 2
    emg_windows, window_labels, window_reps, starts = [], [], [], []
    for start in range(0, n_samples - size + 1, stride):
        end = start + size
        label_window, rep_window = labels[start:end], repetitions[start:end]
        label, repetition = int(label_window[center]), int(rep_window[center])
        if label <= 0 or repetition <= 0:
            continue
        if np.mean(label_window == label) < min_label_ratio:
            continue
        if np.mean(rep_window == repetition) < min_repetition_ratio:
            continue
        emg_windows.append(emg_norm[start:end].astype(np.float32, copy=False))
        window_labels.append(label)
        window_reps.append(repetition)
        starts.append(start)
    if not emg_windows:
        raise ValueError(f"S{subject_id:02d} has no action-pure gesture windows")
    return GestureWindows(np.stack(emg_windows), np.asarray(window_labels, dtype=np.int64),
                          np.asarray(window_reps, dtype=np.int64), np.asarray(starts, dtype=np.int64))


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
def apply_mcia(mcia: nn.Module, raw_windows: np.ndarray, detector: RuleAnomalyDetector,
               device: str, batch_size: int) -> tuple[np.ndarray, dict]:
    metadata = detector.detect_batch(raw_windows)
    masks = metadata["mask"].astype(np.float32, copy=False)
    enhanced = np.empty_like(raw_windows)
    for start in range(0, len(raw_windows), batch_size):
        stop = min(start + batch_size, len(raw_windows))
        raw = torch.as_tensor(raw_windows[start:stop], dtype=torch.float32, device=device)
        mask = torch.as_tensor(masks[start:stop], dtype=torch.float32, device=device)
        valid_channels = (mask.mean(dim=1) > 0.5).float()
        completed = mcia(raw * mask, raw_time_mask=mask, chan_valid_mask=valid_channels)
        enhanced[start:stop] = (completed * (1.0 - mask) + raw * mask).cpu().numpy()
    return enhanced, metadata


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
    labels = metadata["labels"]
    return {
        "masked_token_ratio": float((labels != 0).mean()),
        "windows_with_any_mask_ratio": float((labels != 0).any(axis=(1, 2)).mean()),
        "labels": {str(key): float((labels == key).mean()) for key in range(1, 5)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated DB3 raw-versus-MCIA gesture-recognition diagnostic")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subjects", default="6")
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--min-label-ratio", type=float, default=0.80)
    parser.add_argument("--min-repetition-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-name", default="db3_gesture_completion_single_seed.json")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not os.environ.get("MCIA_RUN_DIR") or Path(os.environ["MCIA_RUN_DIR"]).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir for this single diagnostic")
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if not 0.0 < args.min_label_ratio <= 1.0 or not 0.0 < args.min_repetition_ratio <= 1.0:
        raise ValueError("Window purity ratios must be in (0, 1]")

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    device = config["device"]
    out_dir = run_dir / "06_diagnostics" / "db3_gesture_completion"
    for name in ("metrics", "predictions", "figures", "checkpoints", "logs"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    checkpoint = healthy_checkpoint(config)
    mcia = build_mcia(config, device)
    load_mcia_state_dict(mcia, checkpoint, device)
    mcia.eval()
    print(f"[MCIA:B] healthy prior: {checkpoint}", flush=True)

    subjects_report = {}
    for subject_id in [int(value) for value in args.subjects.split(",") if value.strip()]:
        print(f"\n[gesture] S{subject_id:02d}: loading action-pure windows", flush=True)
        windows = load_db3_windows(loader, subject_id, args.exercise, config,
                                   args.min_label_ratio, args.min_repetition_ratio)
        train_idx, val_idx, test_idx = split_indices(windows.repetitions)
        mapping, action_ids = class_mapping(windows.labels[train_idx])
        encoded = encode_labels(windows.labels, mapping)
        detector = RuleAnomalyDetector(patch_size=int(config["patch_size"]),
                                       group_indices=config.get("group_indices")).fit(windows.emg[train_idx])
        enhanced, metadata = apply_mcia(mcia, windows.emg, detector, device, int(config["regressor_batch_size"]))
        summary = mask_summary(metadata)
        print(f"  windows train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}; "
              f"actions={len(action_ids)}; masked={summary['masked_token_ratio']:.2%}", flush=True)
        report_conditions = {}
        for name, emg in {"A_raw": windows.emg, "B_healthy_prior": enhanced}.items():
            model, training = train_classifier(emg, encoded, train_idx, val_idx, config, device,
                                               args.seed + subject_id)
            test_prediction = predict(model, emg, test_idx, int(config["regressor_batch_size"]), device)
            test_truth = encoded[test_idx]
            test_metrics = metrics(test_truth, test_prediction, len(action_ids))
            trial_metrics = trial_majority_metrics(test_truth, test_prediction,
                                                   windows.repetitions[test_idx], len(action_ids))
            checkpoint_path = out_dir / "checkpoints" / f"S{subject_id:02d}_{name}.pth"
            prediction_path = out_dir / "predictions" / f"S{subject_id:02d}_{name}_test.npz"
            figure_path = out_dir / "figures" / f"S{subject_id:02d}_{name}_test_confusion.png"
            torch.save(model.state_dict(), checkpoint_path)
            np.savez_compressed(prediction_path, test_local_index=np.arange(len(test_idx), dtype=np.int64),
                                original_window_index=test_idx, downsample_start=windows.starts[test_idx],
                                repetition=windows.repetitions[test_idx], true_action=windows.labels[test_idx],
                                predicted_action=np.asarray([action_ids[value] for value in test_prediction]),
                                true_class=test_truth, predicted_class=test_prediction)
            save_confusion_plot(np.asarray(test_metrics["confusion_matrix"]), action_ids,
                                f"DB3 S{subject_id:02d} {name}: test confusion matrix", figure_path)
            report_conditions[name] = {
                "training": training, "test_window_metrics": test_metrics,
                "test_trial_majority_metrics": trial_metrics, "checkpoint": str(checkpoint_path),
                "prediction_file": str(prediction_path), "confusion_figure": str(figure_path)}
            print(f"  {name}: validation macro-F1={training['best_validation_macro_f1']:.4f}; "
                  f"test macro-F1={test_metrics['macro_f1']:.4f}; "
                  f"balanced accuracy={test_metrics['balanced_accuracy']:.4f}", flush=True)
        subjects_report[f"S{subject_id:02d}"] = {
            "n_windows": int(len(windows.emg)),
            "split_sizes": {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))},
            "actions": action_ids, "mask_coverage": summary,
            "rule_detector_dead_channels_one_based": [
                int(value) + 1 for value in np.flatnonzero(detector.dead_channels_)
            ],
            "conditions": report_conditions,
            "healthy_prior_delta_test_macro_f1": float(
                report_conditions["B_healthy_prior"]["test_window_metrics"]["macro_f1"] -
                report_conditions["A_raw"]["test_window_metrics"]["macro_f1"]) }

    report = {
        "scope": "isolated DB3 gesture-recognition diagnostic; Exp1/Exp2/Exp3 are unchanged",
        "purpose": "test whether existing healthy-prior MCIA enhancement improves subject-specific DB3 E1 gesture recognition",
        "leakage_control": {
            "window_policy": "center action and repetition must each occupy at least the configured purity ratio",
            "train_repetitions": [1, 3, 4], "validation_repetitions": [6], "test_repetitions": [2, 5],
            "detector_fit": "training repetitions only",
            "models": "A raw and B enhanced train independently on matching train inputs",
            "model_selection": "early stopping by validation macro-F1 only"},
        "model": {"name": "GestureTCN", "hidden_dim": int(config["regressor_hidden_dim"]),
                  "n_layers": int(config["regressor_n_layers"]), "kernel_size": int(config["regressor_kernel_size"]),
                  "dropout": float(config["regressor_model_dropout"]), "pooling": "global temporal mean"},
        "window_policy": {"window_size": int(config["window_size"]), "stride": int(config["stride"]),
                          "target_fs": int(config["target_fs"]), "min_label_ratio": float(args.min_label_ratio),
                          "min_repetition_ratio": float(args.min_repetition_ratio)},
        "mcia": {"group": "B_healthy_prior", "checkpoint": str(checkpoint)}, "subjects": subjects_report}
    result_path = out_dir / "metrics" / args.result_name
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[gesture] results: {result_path}", flush=True)


if __name__ == "__main__":
    main()
