"""
(实验二) DB3 辅助退化掩码生成

规则：
1. 弱信号候选: emg <= weak_threshold
2. 高幅值候选: emg >= abnormal_threshold
3. 通道级候选: 若通道超过 ratio_threshold 比例被标记，整通道作为候选增强区域

该模块仅作为辅助工程工具。论文主实验优先使用可控伪缺失掩码，
以便在可观测 DB3 通道上保留 真值标注。
"""

import numpy as np
import torch


def detect_anomaly_mask(emg, weak_threshold=0.02, abnormal_threshold=0.95):
    """
    检测 EMG 信号中的辅助退化候选位点，生成增强掩码

    Args:
        emg: (T, C) 或 (B, T, C) EMG 信号（已归一化到 [0, 1]）
        weak_threshold: 低于此值作为弱信号候选
        abnormal_threshold: 高于此值作为高幅值候选

    Returns:
        mask: 与 emg 同形状，1=保留，0=候选增强区域
    """
    if isinstance(emg, torch.Tensor):
        mask = torch.ones_like(emg)
        mask[emg <= weak_threshold] = 0.0
        mask[emg >= abnormal_threshold] = 0.0
    else:
        mask = np.ones_like(emg, dtype=np.float32)
        mask[emg <= weak_threshold] = 0.0
        mask[emg >= abnormal_threshold] = 0.0

    return mask


def channel_level_mask(mask, ratio_threshold=0.5):
    """
    从时间点级掩码生成通道级掩码

    如果某通道超过 ratio_threshold 比例的时间点被标记，则整通道作为候选增强通道。

    Args:
        mask: (T, C) 或 (B, T, C) 时间点级掩码
        ratio_threshold: 候选区域比例阈值

    Returns:
        channel_mask: (C,) 或 (B, C)，1=保留通道，0=候选增强通道
    """
    if isinstance(mask, torch.Tensor):
        if mask.dim() == 2:
            anomaly_ratio = (mask == 0).float().mean(dim=0)
            return (anomaly_ratio < ratio_threshold).float()
        elif mask.dim() == 3:
            anomaly_ratio = (mask == 0).float().mean(dim=1)
            return (anomaly_ratio < ratio_threshold).float()
    else:
        if mask.ndim == 2:
            anomaly_ratio = (mask == 0).astype(np.float32).mean(axis=0)
            return (anomaly_ratio < ratio_threshold).astype(np.float32)
        elif mask.ndim == 3:
            anomaly_ratio = (mask == 0).astype(np.float32).mean(axis=1)
            return (anomaly_ratio < ratio_threshold).astype(np.float32)


def compute_db2_thresholds(db2_segments, weak_percentile=2, abnormal_percentile=98):
    """
    基于 DB2 健康人数据计算辅助退化阈值

    思路：健康人信号的极端值可以作为辅助候选区域参考边界。

    Args:
        db2_segments: (N, T, C) DB2 健康人 EMG 片段（已归一化）
        weak_percentile: 微弱值对应的百分位
        abnormal_percentile: 异常值对应的百分位

    Returns:
        weak_threshold: float
        abnormal_threshold: float
    """
    flat = db2_segments.reshape(-1)
    weak_threshold = np.percentile(flat, weak_percentile)
    abnormal_threshold = np.percentile(flat, abnormal_percentile)

    print(f"  DB2 auxiliary thresholds: weak <= {weak_threshold:.4f}, high >= {abnormal_threshold:.4f}")
    return weak_threshold, abnormal_threshold
