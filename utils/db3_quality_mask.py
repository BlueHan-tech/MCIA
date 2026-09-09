"""DB3 two-layer quality masks: hard dropouts plus Gronlund MQP.

The MQP equations/constants reproduce Gronlund et al. (2005).  The selected
project decision rule is p_j > 0.20; it is deliberately kept separate from
the paper's original p_j > 0.05 reporting threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

PAPER_FS = 2048
SHORT_SAMPLES = 8
LONG_SAMPLES = 256
OUTER_MULTIPLIER = 1.6
MQP_MASK_THRESHOLD = 0.20


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _mqp_flags(observations: np.ndarray) -> np.ndarray:
    center = np.median(observations, axis=0)
    subset = observations[np.argsort(np.linalg.norm(observations - center, axis=1), kind="stable")[:len(observations) // 2]]
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(subset, rowvar=False, ddof=1))
    if not np.all(np.isfinite(eigenvalues)) or np.min(eigenvalues) <= 0:
        return np.zeros(len(observations), dtype=bool)
    principal = observations @ eigenvectors
    lower, middle, upper = np.percentile(principal, [11.7, 50.0, 88.3], axis=0)
    upper_radius = OUTER_MULTIPLIER * (upper - middle)
    lower_radius = OUTER_MULTIPLIER * (middle - lower)
    if np.any(upper_radius <= 0) or np.any(lower_radius <= 0):
        return np.zeros(len(observations), dtype=bool)
    centered = principal - middle
    angle = np.arctan2(centered[:, 1], centered[:, 0])
    a = np.where(np.cos(angle) >= 0, upper_radius[0], lower_radius[0])
    b = np.where(np.sin(angle) >= 0, upper_radius[1], lower_radius[1])
    boundary = np.sqrt((b * np.sin(angle)) ** 2 + (a * np.cos(angle)) ** 2)
    return np.linalg.norm(centered, axis=1) > boundary


def mqp_probability(one_second_2000: np.ndarray) -> np.ndarray:
    """Exact MQP descriptor windows and robust outlier calculation for one second."""
    if one_second_2000.shape[0] != 2000:
        raise ValueError("MQP requires one 2000-Hz second")
    values = resample_poly(one_second_2000, up=128, down=125, axis=0)
    n_channels = values.shape[1]
    short = values.reshape(-1, SHORT_SAMPLES, n_channels).std(axis=1, ddof=0)
    long = values.reshape(-1, LONG_SAMPLES, n_channels).std(axis=1, ddof=0)
    observations = np.stack((short, np.repeat(long, LONG_SAMPLES // SHORT_SAMPLES, axis=0)), axis=-1)
    return np.stack([_mqp_flags(observations[k]) for k in range(len(observations))]).mean(axis=0)


def db3_quality_mask(raw: np.ndarray, labels: np.ndarray, repetitions: np.ndarray,
                     target_fs: int, action_ids: tuple[int, ...] = tuple(range(1, 49))) -> tuple[np.ndarray, dict]:
    """Return a 200-Hz keep mask and auditable two-layer mask metadata.

    Hard channels are fitted only from train repetitions 1/3/4. MQP is a
    fixed per-second calculation and is evaluated independently on all input
    repetitions, without labels other than selecting active action intervals.
    """
    raw = np.asarray(raw, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    repetitions = np.asarray(repetitions).reshape(-1)
    n = min(len(raw), len(labels), len(repetitions))
    raw, labels, repetitions = raw[:n], labels[:n], repetitions[:n]
    valid_action = np.isin(labels, action_ids)
    train = valid_action & np.isin(repetitions, (1, 3, 4))
    hard = np.all(raw[train] == 0.0, axis=0) if train.any() else np.zeros(raw.shape[1], dtype=bool)
    raw_keep = np.ones_like(raw, dtype=bool)
    raw_keep[:, hard] = False
    centered = raw - raw.mean(axis=0, keepdims=True)
    mqp_segments = []
    for start, end in _runs(valid_action):
        for cursor in range(start, end - 2000 + 1, 2000):
            if np.all(repetitions[cursor:cursor + 2000] == repetitions[cursor]):
                probability = mqp_probability(centered[cursor:cursor + 2000])
                flagged = probability > MQP_MASK_THRESHOLD
                raw_keep[cursor:cursor + 2000, flagged] = False
                mqp_segments.append((int(cursor), int(repetitions[cursor]), probability, flagged))
    factor = 2000 // int(target_fs)
    usable = (len(raw_keep) // factor) * factor
    down_keep = raw_keep[:usable].reshape(-1, factor, raw.shape[1]).all(axis=1)
    mqp_count = int(sum(int(x[3].sum()) for x in mqp_segments))
    return down_keep.astype(np.float32), {
        "hard_channels_one_based": (np.flatnonzero(hard) + 1).astype(int).tolist(),
        "mqp_threshold": MQP_MASK_THRESHOLD,
        "mqp_one_second_segments": len(mqp_segments),
        "mqp_channel_second_alerts": mqp_count,
        "mask_rule": "hard_zero_train_1_3_4 OR gronlund_2005_mqp_p_gt_0_20",
    }
