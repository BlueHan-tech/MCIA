"""Prediction-only postprocessing for windowed continuous kinematic output."""

from __future__ import annotations

import numpy as np
from scipy import signal


def _normalized_savgol_window(length: int, requested: int, polyorder: int) -> int | None:
    if length < 3:
        return None
    window = int(requested)
    if window % 2 == 0:
        window += 1
    max_window = length if length % 2 == 1 else length - 1
    window = min(window, max_window)
    minimum = int(polyorder) + 1
    if minimum % 2 == 0:
        minimum += 1
    if window < minimum:
        return None
    return window


def _contiguous_groups(time_indices: np.ndarray) -> list[np.ndarray]:
    if len(time_indices) == 0:
        return []
    split_points = np.flatnonzero(np.diff(time_indices) > 1) + 1
    return [group for group in np.split(time_indices, split_points) if len(group)]


def _overlap_weights(length: int, fusion: str) -> np.ndarray:
    """Deterministic window weights; no target values are used."""
    if fusion == "uniform_overlap_add":
        return np.ones(length, dtype=np.float64)
    if fusion == "triangular_overlap_add":
        # Keep nonzero edges so samples covered by one window remain defined.
        positions = np.linspace(-1.0, 1.0, length, dtype=np.float64)
        return 0.1 + 0.9 * (1.0 - np.abs(positions))
    raise ValueError(f"Unsupported fusion mode: {fusion}")


def _smooth_segment(values: np.ndarray, filter_kind: str, savgol_window: int,
                    polyorder: int, target_fs: int, lowpass_hz: float) -> np.ndarray:
    if filter_kind == "none":
        return values
    if filter_kind == "savgol":
        window = _normalized_savgol_window(len(values), savgol_window, polyorder)
        if window is None:
            return values
        return signal.savgol_filter(values, window_length=window,
                                    polyorder=min(int(polyorder), window - 1),
                                    axis=0, mode="interp")
    if filter_kind == "butterworth":
        nyquist = float(target_fs) / 2.0
        if not 0.0 < float(lowpass_hz) < nyquist or len(values) < 16:
            return values
        sos = signal.butter(4, float(lowpass_hz) / nyquist, btype="lowpass", output="sos")
        padlen = 3 * (2 * len(sos) + 1)
        if len(values) <= padlen:
            return values
        return signal.sosfiltfilt(sos, values, axis=0)
    raise ValueError(f"Unsupported filter kind: {filter_kind}")


def _quality_for_sequence(pred: np.ndarray, target: np.ndarray) -> dict | None:
    if len(pred) < 3:
        return None
    pred_vel = np.diff(pred, axis=0)
    target_vel = np.diff(target, axis=0)
    pred_jerk = np.diff(pred, n=2, axis=0)
    target_jerk = np.diff(target, n=2, axis=0)
    pred_fft = np.abs(np.fft.rfft(pred - pred.mean(axis=0, keepdims=True), axis=0))
    target_fft = np.abs(np.fft.rfft(target - target.mean(axis=0, keepdims=True), axis=0))
    pred_spec = pred_fft.mean(axis=1)
    target_spec = target_fft.mean(axis=1)
    pred_spec = pred_spec / (pred_spec.sum() + 1e-12)
    target_spec = target_spec / (target_spec.sum() + 1e-12)
    split = max(1, len(pred_spec) // 3)
    return {
        "weight": int(len(pred)),
        "velocity_rms_gt": float(np.sqrt(np.mean(target_vel ** 2))),
        "velocity_rms_pred": float(np.sqrt(np.mean(pred_vel ** 2))),
        "jerk_rms_gt": float(np.sqrt(np.mean(target_jerk ** 2))),
        "jerk_rms_pred": float(np.sqrt(np.mean(pred_jerk ** 2))),
        "spectrum_l1_normalized": float(np.mean(np.abs(pred_spec - target_spec))),
        "high_frequency_energy_ratio_gt": float(np.sum(target_spec[split:])),
        "high_frequency_energy_ratio_pred": float(np.sum(pred_spec[split:])),
    }


def continuous_trajectory_quality(pred: np.ndarray, target: np.ndarray,
                                  time_indices: np.ndarray) -> dict:
    """Quality over contiguous sections only, avoiding derivative jumps across gaps."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    times = np.asarray(time_indices, dtype=np.int64)
    if pred.shape != target.shape or len(pred) != len(times):
        raise ValueError("continuous prediction, target, and time-index lengths must match")
    qualities = []
    local_indices = np.arange(len(times), dtype=np.int64)
    split_points = np.flatnonzero(np.diff(times) > 1) + 1
    for local in np.split(local_indices, split_points):
        item = _quality_for_sequence(pred[local], target[local])
        if item is not None:
            qualities.append(item)
    if not qualities:
        return {"n_contiguous_segments": 0}
    weights = np.asarray([item.pop("weight") for item in qualities], dtype=np.float64)
    result = {"n_contiguous_segments": int(len(qualities))}
    for key in qualities[0]:
        result[key] = float(np.average([item[key] for item in qualities], weights=weights))
    result["velocity_rms_ratio_pred_over_gt"] = float(
        result["velocity_rms_pred"] / (result["velocity_rms_gt"] + 1e-12)
    )
    result["jerk_rms_ratio_pred_over_gt"] = float(
        result["jerk_rms_pred"] / (result["jerk_rms_gt"] + 1e-12)
    )
    return result


def window_overlap_prediction_consistency(pred: np.ndarray,
                                          window_starts: np.ndarray) -> dict:
    """Measure agreement between independently predicted overlap regions.

    This diagnostic uses predictions and their physical timestamps only. It
    detects whether overlap-add is hiding disagreement between short windows
    without using ground-truth angle values.
    """
    pred = np.asarray(pred)
    starts = np.asarray(window_starts, dtype=np.int64)
    if pred.ndim != 3 or len(pred) != len(starts):
        raise ValueError("pred must be (n_windows, time, channels) and align with window_starts")
    if len(pred) < 2:
        return {"overlap_pair_count": 0, "overlap_sample_count": 0, "overlap_rmse": float("nan")}

    order = np.argsort(starts, kind="stable")
    sorted_pred = pred[order]
    sorted_starts = starts[order]
    window_length = int(pred.shape[1])
    squared_error_sum = 0.0
    value_count = 0
    pair_count = 0
    sample_count = 0
    for left_pred, right_pred, left_start, right_start in zip(
        sorted_pred[:-1], sorted_pred[1:], sorted_starts[:-1], sorted_starts[1:]
    ):
        offset = int(right_start - left_start)
        if offset <= 0 or offset >= window_length:
            continue
        difference = left_pred[offset:] - right_pred[:window_length - offset]
        squared_error_sum += float(np.sum(difference ** 2))
        value_count += int(difference.size)
        pair_count += 1
        sample_count += int(len(difference))
    return {
        "overlap_pair_count": int(pair_count),
        "overlap_sample_count": int(sample_count),
        "overlap_rmse": float(np.sqrt(squared_error_sum / value_count)) if value_count else float("nan"),
    }


def postprocess_window_predictions(pred: np.ndarray, target: np.ndarray,
                                   window_starts: np.ndarray,
                                   dynamic_window_indices: np.ndarray | None,
                                   savgol_window: int = 13,
                                   polyorder: int = 2,
                                   fusion: str = "uniform_overlap_add",
                                   filter_kind: str = "savgol",
                                   target_fs: int = 100,
                                   lowpass_hz: float = 8.0) -> dict:
    """Fuse overlapping predictions, then smooth each contiguous predicted sequence.

    Prediction formation uses only model outputs and window positions. Targets are
    averaged solely to return aligned data for metrics after postprocessing.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    starts = np.asarray(window_starts, dtype=np.int64)
    if pred.ndim != 3 or target.shape != pred.shape:
        raise ValueError("pred and target must have shape (n_windows, time, channels)")
    if len(pred) != len(starts):
        raise ValueError("prediction and window-start counts must match")
    if len(pred) == 0:
        return {
            "processed_windows": pred.copy(),
            "continuous_prediction": np.empty((0, pred.shape[-1]), dtype=pred.dtype),
            "continuous_target": np.empty((0, target.shape[-1]), dtype=target.dtype),
            "continuous_time_indices": np.empty(0, dtype=np.int64),
            "continuous_dynamic_mask": np.empty(0, dtype=bool),
            "continuous_overlap_counts": np.empty(0, dtype=np.int16),
            "metadata": {"fusion": fusion, "filter_kind": filter_kind, "savgol_window": int(savgol_window)},
        }

    window_length = int(pred.shape[1])
    first_time = int(starts.min())
    last_time = int(starts.max()) + window_length
    n_time = last_time - first_time
    channels = int(pred.shape[-1])
    pred_sum = np.zeros((n_time, channels), dtype=np.float64)
    target_sum = np.zeros((n_time, channels), dtype=np.float64)
    weights = np.zeros(n_time, dtype=np.float64)
    overlap_counts = np.zeros(n_time, dtype=np.int16)
    dynamic_mask = np.zeros(n_time, dtype=bool)
    sample_weights = _overlap_weights(window_length, fusion)
    dynamic_indices = set(np.asarray(dynamic_window_indices if dynamic_window_indices is not None else [], dtype=np.int64).tolist())

    for window_index, (one_pred, one_target, start) in enumerate(zip(pred, target, starts)):
        left = int(start) - first_time
        right = left + window_length
        pred_sum[left:right] += one_pred * sample_weights[:, None]
        target_sum[left:right] += one_target * sample_weights[:, None]
        weights[left:right] += sample_weights
        overlap_counts[left:right] += 1
        if window_index in dynamic_indices:
            dynamic_mask[left:right] = True

    valid = weights > 0
    fused_pred = np.zeros_like(pred_sum)
    fused_target = np.zeros_like(target_sum)
    fused_pred[valid] = pred_sum[valid] / weights[valid, None]
    fused_target[valid] = target_sum[valid] / weights[valid, None]

    for group in _contiguous_groups(np.flatnonzero(valid)):
        fused_pred[group] = _smooth_segment(
            fused_pred[group], filter_kind, savgol_window, polyorder, target_fs, lowpass_hz
        )

    processed_windows = np.empty_like(pred)
    for window_index, start in enumerate(starts):
        left = int(start) - first_time
        processed_windows[window_index] = fused_pred[left:left + window_length]

    time_indices = np.flatnonzero(valid).astype(np.int64) + first_time
    return {
        "processed_windows": processed_windows,
        "continuous_prediction": fused_pred[valid].astype(pred.dtype, copy=False),
        "continuous_target": fused_target[valid].astype(target.dtype, copy=False),
        "continuous_time_indices": time_indices,
        "continuous_dynamic_mask": dynamic_mask[valid],
        "continuous_overlap_counts": overlap_counts[valid],
        "metadata": {
            "fusion": fusion,
            "filter_kind": filter_kind,
            "savgol_window": int(savgol_window),
            "savgol_polyorder": int(polyorder),
            "target_fs": int(target_fs),
            "lowpass_hz": float(lowpass_hz),
            "n_unique_samples": int(valid.sum()),
            "n_contiguous_segments": int(len(_contiguous_groups(np.flatnonzero(valid)))),
            "mean_overlap": float(weights[valid].mean()),
            "max_overlap": int(weights.max()),
        },
    }
