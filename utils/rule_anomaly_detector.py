"""
与场景对齐的 DB3 异常掩码检测器。

规则与 DB2 ScenarioMix 掩码对齐：
S1 短时局部丢失、S2 死通道/不可用通道、S3 肌群异常、S4 多通道时间丢失。
掩码约定：1=保留，0=送入 MCIA。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import numpy as np

LABEL_KEEP = 0
LABEL_S1 = 1
LABEL_S2 = 2
LABEL_S3 = 3
LABEL_S4 = 4
LABEL_UNRECOVERABLE = 9
LABEL_NAMES = {
    LABEL_KEEP: "keep",
    LABEL_S1: "s1_short_dropout",
    LABEL_S2: "s2_dead_channel",
    LABEL_S3: "s3_group_abnormal",
    LABEL_S4: "s4_time_dropout",
    LABEL_UNRECOVERABLE: "unrecoverable",
}


@dataclass
class RuleMaskConfig:
    patch_size: int = 8
    min_valid_channels: int = 3
    max_mask_ratio: float = 0.75
    active_rms_ratio: float = 0.20
    dead_rms_ratio: float = 0.20
    dead_mad_ratio: float = 0.25
    dead_active_ratio: float = 0.15
    clipping_value: float = 0.995
    clipping_ratio: float = 0.50
    patch_percentile: float = 5.0
    patch_threshold_scale: float = 0.80
    patch_abs_ratio: float = 0.03
    s1_min_run_patches: int = 1
    s1_max_run_patches: int = 3
    s3_min_group_channels: int = 2
    s3_max_global_low_ratio: float = 0.50
    s4_min_low_channel_ratio: float = 0.50
    s4_min_run_patches: int = 2


class RuleAnomalyDetector:
    """拟合被试级质量统计并生成类 ScenarioMix 掩码。"""

    def __init__(self, patch_size: int = 8,
                 group_indices: Optional[Dict[str, Iterable[int]]] = None,
                 config: Optional[RuleMaskConfig] = None,
                 **overrides):
        self.cfg = config or RuleMaskConfig(patch_size=patch_size)
        self.cfg.patch_size = int(patch_size)
        for key, value in overrides.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
        self.group_indices = self._normalize_groups(group_indices)
        self.dead_channels_: Optional[np.ndarray] = None
        self.rms_threshold_: Optional[np.ndarray] = None
        self.mad_threshold_: Optional[np.ndarray] = None
        self.fit_stats_: Dict[str, np.ndarray | float] = {}
        self.theta_dead: Optional[np.ndarray] = None
        self.theta_weak: Optional[np.ndarray] = None

    @staticmethod
    def _normalize_groups(group_indices: Optional[Dict[str, Iterable[int]]]) -> Dict[str, List[int]]:
        if group_indices is None:
            group_indices = {
                "flexor": [0, 1, 2, 3, 8],
                "extensor": [4, 5, 6, 7, 9],
                "upper_arm": [10, 11],
            }
        return {name: [int(c) for c in channels] for name, channels in group_indices.items()}

    def fit(self, emg_windows: np.ndarray) -> "RuleAnomalyDetector":
        windows = self._trim_windows(np.asarray(emg_windows, dtype=np.float32))
        n_items, _, n_channels = windows.shape
        patch_size = self.cfg.patch_size
        n_patches = windows.shape[1] // patch_size
        rms_win = np.sqrt(np.mean(windows ** 2, axis=1))
        mad_win = np.mean(np.abs(windows - windows.mean(axis=1, keepdims=True)), axis=1)
        med_rms = np.median(rms_win, axis=0)
        med_mad = np.median(mad_win, axis=0)
        pos_rms = med_rms[med_rms > 1e-8]
        pos_mad = med_mad[med_mad > 1e-8]
        global_rms = float(np.median(pos_rms)) if len(pos_rms) else 1e-6
        global_mad = float(np.median(pos_mad)) if len(pos_mad) else 1e-6
        active_thr = max(1e-6, self.cfg.active_rms_ratio * global_rms)
        active_ratio = np.mean(rms_win > active_thr, axis=0)
        clip_ratio = np.mean(windows >= self.cfg.clipping_value, axis=(0, 1))
        dead = ((med_rms < self.cfg.dead_rms_ratio * global_rms)
                & (med_mad < self.cfg.dead_mad_ratio * global_mad)
                & (active_ratio < self.cfg.dead_active_ratio))
        dead = dead | (clip_ratio > self.cfg.clipping_ratio)
        patch = windows.reshape(n_items, n_patches, patch_size, n_channels)
        patch_rms = np.sqrt(np.mean(patch ** 2, axis=2))
        patch_mad = np.mean(np.abs(patch - patch.mean(axis=2, keepdims=True)), axis=2)
        rms_thr = np.maximum(
            np.percentile(patch_rms, self.cfg.patch_percentile, axis=(0, 1)) * self.cfg.patch_threshold_scale,
            self.cfg.patch_abs_ratio * global_rms,
        )
        mad_thr = np.maximum(
            np.percentile(patch_mad, self.cfg.patch_percentile, axis=(0, 1)) * self.cfg.patch_threshold_scale,
            self.cfg.patch_abs_ratio * global_mad,
        )
        self.dead_channels_ = dead.astype(bool)
        self.rms_threshold_ = rms_thr.astype(np.float32)
        self.mad_threshold_ = mad_thr.astype(np.float32)
        self.theta_dead = self.dead_channels_
        self.theta_weak = self.rms_threshold_
        self.fit_stats_ = {
            "median_rms": med_rms.astype(np.float32),
            "median_mad": med_mad.astype(np.float32),
            "global_rms": float(global_rms),
            "global_mad": float(global_mad),
            "active_ratio": active_ratio.astype(np.float32),
            "clipping_ratio": clip_ratio.astype(np.float32),
            "rms_threshold": self.rms_threshold_,
            "mad_threshold": self.mad_threshold_,
        }
        return self

    def detect(self, window: np.ndarray) -> np.ndarray:
        return self.detect_with_metadata(window)["mask"]

    def detect_with_metadata(self, window: np.ndarray) -> Dict[str, np.ndarray | float]:
        self._require_fit()
        x = self._trim_window(np.asarray(window, dtype=np.float32))
        time_steps, n_channels = x.shape
        patch_size = self.cfg.patch_size
        n_patches = time_steps // patch_size
        patch = x.reshape(n_patches, patch_size, n_channels)
        patch_rms = np.sqrt(np.mean(patch ** 2, axis=1))
        patch_mad = np.mean(np.abs(patch - patch.mean(axis=1, keepdims=True)), axis=1)
        low = (patch_rms < self.rms_threshold_[None, :]) & (patch_mad < self.mad_threshold_[None, :])
        low[:, self.dead_channels_] = False
        mask_p = np.ones((n_patches, n_channels), dtype=np.float32)
        labels_p = np.zeros((n_patches, n_channels), dtype=np.uint8)
        self._apply_s2(mask_p, labels_p)
        self._apply_s3(mask_p, labels_p, low)
        self._apply_s4(mask_p, labels_p, low)
        self._apply_s1(mask_p, labels_p, low)
        valid_channels = (mask_p > 0.5).sum(axis=1)
        patch_mask_ratio = (mask_p < 0.5).mean(axis=1)
        unrecoverable = ((valid_channels < self.cfg.min_valid_channels)
                         | (patch_mask_ratio > self.cfg.max_mask_ratio))
        mask = np.repeat(mask_p, patch_size, axis=0)
        labels = np.repeat(labels_p, patch_size, axis=0)
        return {
            "mask": mask.astype(np.float32),
            "labels": labels.astype(np.uint8),
            "patch_mask": mask_p.astype(np.float32),
            "patch_labels": labels_p.astype(np.uint8),
            "unrecoverable_patches": unrecoverable.astype(bool),
            "mask_ratio": float((mask < 0.5).mean()),
            "unrecoverable_ratio": float(unrecoverable.mean()),
            "dead_channels": np.where(self.dead_channels_)[0].astype(np.int32),
        }

    def detect_batch(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        trimmed = self._trim_windows(np.asarray(windows, dtype=np.float32))
        outputs = [self.detect_with_metadata(trimmed[i]) for i in range(len(trimmed))]
        return {
            "mask": np.stack([o["mask"] for o in outputs]).astype(np.float32),
            "labels": np.stack([o["labels"] for o in outputs]).astype(np.uint8),
            "patch_mask": np.stack([o["patch_mask"] for o in outputs]).astype(np.float32),
            "patch_labels": np.stack([o["patch_labels"] for o in outputs]).astype(np.uint8),
            "unrecoverable_patches": np.stack([o["unrecoverable_patches"] for o in outputs]).astype(bool),
            "mask_ratio": np.asarray([o["mask_ratio"] for o in outputs], dtype=np.float32),
            "unrecoverable_ratio": np.asarray([o["unrecoverable_ratio"] for o in outputs], dtype=np.float32),
            "dead_channels": np.where(self.dead_channels_)[0].astype(np.int32),
        }

    def _apply_s2(self, mask_p: np.ndarray, labels_p: np.ndarray) -> None:
        if self.dead_channels_ is None:
            return
        mask_p[:, self.dead_channels_] = 0.0
        labels_p[:, self.dead_channels_] = LABEL_S2

    def _apply_s3(self, mask_p: np.ndarray, labels_p: np.ndarray, low: np.ndarray) -> None:
        for channels in self.group_indices.values():
            idxs = [c for c in channels if c < low.shape[1]]
            for p in range(low.shape[0]):
                group_low = [c for c in idxs if low[p, c]]
                if len(group_low) >= self.cfg.s3_min_group_channels and low[p].mean() < self.cfg.s3_max_global_low_ratio:
                    mask_p[p, group_low] = 0.0
                    for c in group_low:
                        if labels_p[p, c] == LABEL_KEEP:
                            labels_p[p, c] = LABEL_S3

    def _apply_s4(self, mask_p: np.ndarray, labels_p: np.ndarray, low: np.ndarray) -> None:
        global_low = low.mean(axis=1) >= self.cfg.s4_min_low_channel_ratio
        p = 0
        while p < len(global_low):
            if not global_low[p]:
                p += 1
                continue
            start = p
            while p < len(global_low) and global_low[p]:
                p += 1
            if p - start >= self.cfg.s4_min_run_patches:
                channels = np.where(low[start:p].any(axis=0))[0]
                mask_p[start:p, channels] = 0.0
                for pp in range(start, p):
                    for c in channels:
                        if labels_p[pp, c] == LABEL_KEEP:
                            labels_p[pp, c] = LABEL_S4

    def _apply_s1(self, mask_p: np.ndarray, labels_p: np.ndarray, low: np.ndarray) -> None:
        n_patches, n_channels = low.shape
        for c in range(n_channels):
            if self.dead_channels_ is not None and self.dead_channels_[c]:
                continue
            p = 0
            while p < n_patches:
                if not low[p, c] or mask_p[p, c] < 0.5:
                    p += 1
                    continue
                start = p
                while p < n_patches and low[p, c] and mask_p[p, c] > 0.5:
                    p += 1
                run_len = p - start
                before_ok = start > 0 and not low[start - 1, c]
                after_ok = p < n_patches and not low[p, c]
                if self.cfg.s1_min_run_patches <= run_len <= self.cfg.s1_max_run_patches and before_ok and after_ok:
                    mask_p[start:p, c] = 0.0
                    labels_p[start:p, c] = LABEL_S1

    def _require_fit(self) -> None:
        if self.dead_channels_ is None or self.rms_threshold_ is None or self.mad_threshold_ is None:
            raise RuntimeError("Call fit(emg_windows) before detect().")

    def _trim_windows(self, windows: np.ndarray) -> np.ndarray:
        if windows.ndim != 3:
            raise ValueError(f"Expected windows with shape (N,T,C), got {windows.shape}")
        usable = (windows.shape[1] // self.cfg.patch_size) * self.cfg.patch_size
        if usable <= 0:
            raise ValueError("Window length must be at least one patch.")
        return windows[:, :usable, :]

    def _trim_window(self, window: np.ndarray) -> np.ndarray:
        if window.ndim != 2:
            raise ValueError(f"Expected window with shape (T,C), got {window.shape}")
        usable = (window.shape[0] // self.cfg.patch_size) * self.cfg.patch_size
        if usable <= 0:
            raise ValueError("Window length must be at least one patch.")
        return window[:usable, :]
