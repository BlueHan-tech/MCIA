"""
(实验二) DB3 截肢者肌电数据集

功能：
- prepare_data_db3: DB3 预处理流水线（与 DB2 一致的滤波→包络→归一化流程），
  质量掩码来自 utils.db3_quality_mask 的两层规则（硬零 + WC-BQD）。

历史注记：旧 DB3EMGDataset（weak/abnormal 阈值辅助退化掩码）已随规则掩码
主链路化移除（2026-09-10 清理，删除前经 rg 确认零消费方）。
"""

import numpy as np
import torch
from pathlib import Path
from utils.db3_quality_mask import db3_quality_mask

from data.dataset_db2_emg import moving_average


def prepare_data_db3(data_loader, subject_ids, config, exercises=None, return_metadata=False):
    """准备 DB3 EMG 窗口；不跨 exercise，且不泄漏测试集缩放统计量。"""
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
