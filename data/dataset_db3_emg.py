"""
(实验二) DB3 截肢者肌电数据集

功能：
- DB3EMGDataset: 加载 DB3 截肢者 sEMG，生成辅助退化掩码
- prepare_data_db3: DB3 预处理流水线（与 DB2 一致的滤波→包络→归一化流程）

辅助退化掩码规则：
- emg <= weak_threshold (微弱值)  -> 标记为需要补全
- emg >= abnormal_threshold (异常值) -> 标记为需要补全
- 被标记的位点置零，mask 设为 0（需补全）
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from utils.db3_quality_mask import db3_quality_mask

from data.dataset_db2_emg import moving_average


class DB3EMGDataset(Dataset):
    """
    DB3 截肢者 EMG 数据集

    与 DB2 EMGCompletionDataset 不同：
    - mask 可作为辅助退化区域，不作为论文主实验的核心标注
    - 每个样本返回: data(原始), data_masked(异常区域置零), mask(1=正常, 0=需补全)
    """

    def __init__(self, segments, weak_threshold=0.02, abnormal_threshold=0.95):
        """
        Args:
            segments: (N, T, C) 预处理后的 EMG 片段
            weak_threshold: 低于此值作为弱信号辅助退化候选
            abnormal_threshold: 高于此值作为高幅值辅助退化候选
        """
        self.segments = torch.FloatTensor(segments)
        self.weak_threshold = weak_threshold
        self.abnormal_threshold = abnormal_threshold

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        emg = self.segments[idx]  # (T, C)

        # 辅助退化检测：逐通道判定
        # 掩码：1=保留，0=候选增强区域
        mask = torch.ones_like(emg)

        # 规则1: 信号极弱 -> 弱信号候选
        mask[emg <= self.weak_threshold] = 0.0
        # 规则2: 信号异常高 -> 高幅值候选
        mask[emg >= self.abnormal_threshold] = 0.0

        # 通道级掩码（如果一个通道超过50%的时间点被标记，则作为候选退化通道）
        channel_anomaly_ratio = (mask == 0).float().mean(dim=0)  # (C,)
        channel_mask = (channel_anomaly_ratio < 0.5).float()     # (C,) 1=正常通道
        mask_1d = channel_mask  # (C,) 用于模型的通道级掩码

        emg_masked = emg * mask

        return {
            'data': emg,
            'data_masked': emg_masked,
            'mask': mask,           # (T, C) 时间点级掩码
            'mask_1d': mask_1d,     # (C,)   通道级掩码
        }


def prepare_data_db3(data_loader, subject_ids, config, exercises=None, return_metadata=False):
    """Prepare DB3 EMG windows without crossing exercises or leaking test scaling."""
    exercises = list(exercises or config.get("transfer_exercises", [1]))
    factor = int(config["orig_fs"] / config["target_fs"])
    window_size, stride = int(config["window_size"]), int(config["stride"])
    train_reps = tuple(config.get("transfer_train_repetitions", (1, 3, 4)))
    all_segments, all_subject_ids, all_reps, all_exercises, all_starts, all_masks = [], [], [], [], [], []
    print(f"\n[DB3 Data Prep] exercises={exercises}, {config['orig_fs']}Hz -> {config['target_fs']}Hz")

    for subject_id in subject_ids:
        try:
            parts = []
            for exercise in exercises:
                data = data_loader.load_db3_subject(subject_id, [exercise])
                quality_keep, quality_report = db3_quality_mask(
                    data["emg"], data.get("restimulus", data["stimulus"]),
                    data["repetition"], int(config["target_fs"])
                )
                filtered = data_loader.notch_filter(data_loader.bandpass_filter(data["emg"].astype(np.float32) * 1000.0))
                emg_down = moving_average(np.abs(filtered), factor)[::factor].astype(np.float32)
                labels = np.asarray(data.get("restimulus", data["stimulus"]))[::factor].reshape(-1)
                repetitions = np.asarray(data["repetition"])[::factor].reshape(-1)
                n = min(len(emg_down), len(labels), len(repetitions))
                rows = []
                for start in range(0, n - window_size + 1, stride):
                    end = start + window_size
                    nonzero_reps = np.unique(repetitions[start:end][repetitions[start:end] > 0])
                    if len(nonzero_reps) == 1 and np.any(labels[start:end] != 0):
                        rows.append((start, int(nonzero_reps[0])))
                parts.append({
                    "exercise": int(exercise), "emg": emg_down[:n],
                    "quality_mask": quality_keep[:n], "quality_report": quality_report,
                    "rows": rows,
                })

            train_values = np.concatenate([
                item["emg"][start:start + window_size] for item in parts
                for start, rep in item["rows"] if rep in train_reps
            ], axis=0)
            emg_max = float(train_values.max())
            def compress(values):
                return values if emg_max <= 0.0 else np.log1p(255.0 * values / emg_max) / np.log1p(255.0) * emg_max
            q05, q99 = np.percentile(compress(train_values), [5, 99])
            subject_count = 0
            for item in parts:
                norm = np.clip((compress(item["emg"]) - q05) / (q99 - q05 + 1e-8), 0.0, 1.0)
                for start, rep in item["rows"]:
                    all_segments.append(norm[start:start + window_size].astype(np.float32, copy=False))
                    all_subject_ids.append(int(subject_id)); all_reps.append(int(rep))
                    all_exercises.append(item["exercise"]); all_starts.append(int(start)); subject_count += 1
                    all_masks.append(item["quality_mask"][start:start + window_size].astype(np.float32, copy=False))
            print(f"  S{subject_id:02d}: {subject_count} segments")
        except Exception as exc:
            print(f"  S{subject_id}: FAILED - {exc}")

    if not all_segments:
        raise ValueError("No valid DB3 data")
    segments = np.stack(all_segments)
    subject_array = np.asarray(all_subject_ids, dtype=np.int32)
    if not return_metadata:
        return segments, subject_array
    return segments, subject_array, {
        "repetition": np.asarray(all_reps, dtype=np.int32),
        "exercise": np.asarray(all_exercises, dtype=np.int16),
        "start": np.asarray(all_starts, dtype=np.int64),
        "quality_mask": np.stack(all_masks).astype(np.float32),
        "normalization": "train_repetitions_only",
        "window_policy": "exercise_separated_single_nonzero_repetition",
    }
