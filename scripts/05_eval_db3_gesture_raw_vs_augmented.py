"""DB3 49-class A/B/C gesture experiment and legacy waveform readers."""
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
from torch.utils.data import DataLoader, Dataset, Subset

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
    """采用全局时间池化的小型被试专属时序分类器。"""

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
    """仅用训练 repetition 拟合输入缩放，并创建动作纯净窗口。"""
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
        raise ValueError(f"S{subject_id:02d} 没有动作纯净的手势窗口")
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
        raise ValueError(f"无效划分：训练={len(train)}，验证={len(validation)}，测试={len(test)}")
    return train, validation, test


def class_mapping(train_labels: np.ndarray) -> tuple[dict[int, int], list[int]]:
    action_ids = sorted(int(value) for value in np.unique(train_labels) if value > 0)
    return {action: index for index, action in enumerate(action_ids)}, action_ids


def encode_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    missing = sorted(set(int(value) for value in labels) - set(mapping))
    if missing:
        raise ValueError(f"训练 repetition 中缺少以下动作：{missing}")
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
    raise FileNotFoundError("未找到健康先验 MCIA 检查点")


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
        # 交付规则（2026-09-09 采纳）：裁剪后对补丁边界做三点淡化，再复制回观测值。
        completed = patch_boundary_crossfade(completed.clamp(0.0, 1.0), patch_size)
        enhanced[start:stop] = (completed * (1.0 - mask) + raw * mask).cpu().numpy()
    return enhanced, {"mask": masks}


def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    labels = np.arange(n_classes) if n_classes == 49 else None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
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
    train_data = GestureDataset(emg[train_idx], labels[train_idx])
    val_data = GestureDataset(emg[val_idx], labels[val_idx])
    return fit_classifier(model, train_data, val_data, int(labels.max()) + 1, config, device)


def fit_classifier(model, train_data, val_data, n_classes, config, device):
    batch_size = int(config['regressor_batch_size'])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config["regressor_learning_rate"]), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(config["regressor_lr_factor"]),
        patience=int(config["regressor_lr_patience"]))
    best_state, best_macro_f1, no_improve, history = None, -np.inf, 0, []
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
        raise RuntimeError('Gesture classifier produced no validation checkpoint')
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
    fig.colorbar(image, ax=axis, label="窗口数")
    ticks = np.arange(len(action_ids))
    axis.set_xticks(ticks, [str(value) for value in action_ids], rotation=45, ha="right")
    axis.set_yticks(ticks, [str(value) for value in action_ids])
    axis.set_xlabel("预测动作")
    axis.set_ylabel("真实动作")
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


def main():
    from data.waveform import WAVEFORM_FORMAT, ChannelStats
    from models.prediction.dsranet import DSRANet
    from utils.paper_pipeline import save_json
    from utils.waveform_db3 import abc_waveforms, load_abc_models, prepare_db3_exercise
    from utils.waveform_downstream import GestureSpectra, frozen_development_subject, paired_summary, split_windows
    from utils.waveform_pipeline import setup_waveform

    parser = argparse.ArgumentParser(description='DB3 49-class DSRANet A/B/C on the 2 kHz protocol')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--final', action='store_true', help='Use locked protocol and evaluate repetitions 2/5')
    args = parser.parse_args()
    if args.final and args.smoke:
        parser.error('--final and --smoke are mutually exclusive')
    cfg, run, loader, device = setup_waveform(ROOT, smoke=args.smoke)
    config = flatten_pipeline_config(cfg)
    options = cfg['exp4_gesture']
    if args.smoke:
        config['regressor_num_epochs'] = 1
        config['regressor_batch_size'] = 4
    scope = 'final' if args.final else 'development'
    output = run / '04_gesture_recognition'
    path = output / 'metrics' / f'waveform_{scope}.json'
    report = (json.loads(path.read_text(encoding='utf-8')) if path.is_file() else
              {'format': WAVEFORM_FORMAT, 'scope': scope, 'smoke': args.smoke, 'subjects': [], 'classes': 49})
    for subject in [5] if args.smoke else options['subjects']:
        if any(row['subject_id'] == subject and row['status'] == 'ok' for row in report['subjects']):
            continue
        result = {'subject_id': subject, 'status': 'failed'}
        try:
            models, stats, model_ids = load_abc_models(cfg, run, device, subject, smoke=args.smoke)
            result['model_sha256'] = model_ids
            frozen = {}
            if args.final:
                frozen_development_subject(output, subject, model_ids)
                for arm in 'ABC':
                    ckpt = output / 'checkpoints' / f'S{subject:02d}_{arm}_waveform_development.pt'
                    payload = torch.load(ckpt, map_location='cpu', weights_only=True)
                    if (payload['format'] != WAVEFORM_FORMAT or not np.array_equal(payload['db2_std'].numpy(), stats.std)):
                        raise ValueError(f'Incompatible frozen gesture checkpoint: {ckpt}')
                    frozen[arm] = payload
            waves = {group: [] for group in 'ABC'}
            starts, labels, reps, exercise_ids = [], [], [], []
            for record_index, exercise in enumerate([1, 3] if args.smoke else options['exercises']):
                rec, quality = prepare_db3_exercise(cfg, loader, subject, exercise,
                                                    smoke=args.smoke, final=args.final, run=run)
                delivery = abc_waveforms(run, subject, exercise, rec, quality, models, stats, model_ids,
                                         device, smoke=args.smoke, final=args.final)
                windows = rec.gestures
                take = np.arange(len(windows.starts))
                if args.smoke:
                    take = np.r_[np.flatnonzero(np.isin(windows.repetitions, (1, 3, 4)))[:4],
                                 np.flatnonzero(windows.repetitions == 6)[:4]]
                for group in waves:
                    waves[group].append(delivery[group])
                starts.extend((record_index, start) for start in windows.starts[take])
                labels.extend(windows.actions[take] - 1)
                reps.extend(windows.repetitions[take])
                exercise_ids.extend([exercise] * len(take))
            y, rep = np.asarray(labels), np.asarray(reps)
            tr, va, evaluate = split_windows(rep, final=args.final)
            split_sizes = {'train': len(tr), 'validation': len(va), 'evaluation': len(evaluate)}
            coverage = {name: np.unique(y[idx] + 1).tolist()
                        for name, idx in [('train', tr), ('validation', va), ('evaluation', evaluate)]}
            result.update(action_coverage=coverage, split_sizes=split_sizes)
            if not args.smoke and any(len(classes) != 49 for classes in coverage.values()):
                raise ValueError('At least one required partition does not cover all 49 actions')
            groups = {}
            payload = {'format': np.asarray(WAVEFORM_FORMAT), 'target_action': y[evaluate] + 1,
                       'repetition': rep[evaluate], 'exercise': np.asarray(exercise_ids)[evaluate],
                       'window_start_2khz': np.asarray(starts)[evaluate, 1]}
            for group, signals in waves.items():
                locked = frozen.get(group)
                group_stats = ChannelStats(locked['mean'].numpy(), locked['std'].numpy()) if locked is not None else None
                data = GestureSpectra(signals, starts, y, stats.std, tr, frozen_stats=group_stats)
                set_reproducible_seed(options['random_seed'])
                model = DSRANet().to(device)
                ckpt = output / 'checkpoints' / f'S{subject:02d}_{group}_waveform_development.pt'
                if locked is None:
                    model, info = fit_classifier(model, Subset(data, tr.tolist()), Subset(data, va.tolist()), 49, config, device)
                    torch.save({'model': model.state_dict(), 'format': WAVEFORM_FORMAT,
                                'mean': torch.from_numpy(data.stats.mean), 'std': torch.from_numpy(data.stats.std),
                                'db2_std': torch.from_numpy(stats.std)}, ckpt)
                else:
                    model.load_state_dict(locked['model'])
                    info = {'checkpoint': str(ckpt)}
                model.eval()
                batches = DataLoader(Subset(data, evaluate.tolist()), batch_size=config['regressor_batch_size'])
                with torch.inference_mode():
                    pred = np.concatenate([model(batch['emg'].to(device)).argmax(1).cpu().numpy() for batch in batches])
                window = metrics(y[evaluate], pred, 49)
                trial = trial_majority_metrics(y[evaluate], pred, rep[evaluate], 49)
                figure = output / 'figures' / scope / f'S{subject:02d}_{group}_confusion.png'
                figure.parent.mkdir(parents=True, exist_ok=True)
                save_confusion_plot(np.asarray(window['confusion_matrix']), list(range(1, 50)),
                                    f'S{subject:02d} {group} {scope}', figure)
                payload[f'{group}_prediction_action'] = pred + 1
                groups[group] = dict(training=info, window_metrics=window, trial_metrics=trial, figure=str(figure))
            prediction_path = output / 'predictions' / f'S{subject:02d}_waveform_{scope}.npz'
            np.savez_compressed(prediction_path, **payload)
            values = {group: groups[group]['window_metrics']['macro_f1'] for group in 'ABC'}
            result.update(status='ok', groups=groups, predictions=str(prediction_path),
                          improvements={group: values[group] - values['A'] for group in 'BC'})
        except Exception as exc:
            result['reason'] = str(exc)
            print(f'S{subject:02d}: {exc}', flush=True)
        report['subjects'] = [row for row in report['subjects'] if row['subject_id'] != subject] + [result]
        report['paired_summary'] = paired_summary(
            report['subjects'], lambda row, group: row['groups'][group]['window_metrics']['macro_f1'],
            higher_is_better=True)
        save_json(path, report)
    if any(row['status'] != 'ok' for row in report['subjects']):
        raise RuntimeError(f'Gesture evaluation has failed subjects. See {path}')


if __name__ == "__main__":
    main()
