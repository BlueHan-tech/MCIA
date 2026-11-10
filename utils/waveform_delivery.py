"""Shared signed waveform delivery for the angle and gesture branches."""
from time import perf_counter

import numpy as np
import torch

from data.waveform import STRIDE, WINDOW
from utils.waveform_pipeline import predict_windows


def prediction_starts(length):
    if length <= WINDOW:
        return np.array([0])
    end = length - WINDOW
    starts = np.arange(0, end + 1, STRIDE)
    return starts if starts[-1] == end else np.r_[starts, end]


def stitch_predictions(values, missing, starts, predictions):
    """Keep the nearest centre and blend only inside one continuous missing run."""
    result = values.copy()
    # For a tie, the earlier window owns the sample.
    cuts = (starts[1:] + starts[:-1] + WINDOW - 1) // 2 + 1
    edges = np.r_[0, cuts, len(values)]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        active = missing[lo:hi]
        if active.any():
            np.copyto(result[lo:hi], predictions[i][lo - starts[i]:hi - starts[i]], where=active)
    for i, seam in enumerate(cuts):
        cross = missing[seam - 1] & missing[seam]
        for channel in np.flatnonzero(cross):
            lo, hi = seam - 5, seam + 5
            while lo < seam and not missing[lo:seam, channel].all():
                lo += 1
            while hi > seam and not missing[seam:hi, channel].all():
                hi -= 1
            phase = (np.arange(hi - lo) + 0.5) / (hi - lo)
            weight = 0.5 - 0.5 * np.cos(np.pi * phase)
            left = predictions[i][lo - starts[i]:hi - starts[i], channel]
            right = predictions[i + 1][lo - starts[i + 1]:hi - starts[i + 1], channel]
            result[lo:hi, channel] = (1 - weight) * left + weight * right
    return result


@torch.inference_mode()
def deliver_waveform(recording, missing, model, stats, device, *, domain, batch_size=64, timing=None):
    if missing.shape != recording.filtered.shape or missing.dtype != np.bool_:
        raise ValueError('Expected a Boolean missing mask on the original time grid')
    result = recording.filtered.copy()
    model.eval()
    start_time, count = perf_counter(), 0
    for lo, hi, _ in recording.segments:
        source, damaged = recording.filtered[lo:hi], missing[lo:hi]
        if not damaged.any():
            continue
        length = len(source)
        starts = prediction_starts(length)
        selected = np.flatnonzero(damaged.any(axis=1))
        hits = np.searchsorted(selected, starts + WINDOW) > np.searchsorted(selected, starts)
        predictions = [None] * len(starts)
        standard = stats.standardize(source)
        if length < WINDOW:
            standard = np.pad(standard, ((0, WINDOW - length), (0, 0)))
            damaged = np.pad(damaged, ((0, WINDOW - length), (0, 0)), constant_values=True)
        selected_windows = np.flatnonzero(hits)
        count += len(selected_windows)
        for offset in range(0, len(selected_windows), batch_size):
            indices = selected_windows[offset:offset + batch_size]
            samples = starts[indices, None] + np.arange(WINDOW)
            windows = torch.from_numpy(standard[samples]).to(device)
            keep = torch.from_numpy(~damaged[samples]).to(device=device, dtype=windows.dtype)
            pred = stats.restore(predict_windows(model, windows, keep, domain).cpu().numpy())
            for index, window in zip(indices, pred):
                predictions[index] = window
        result[lo:hi] = stitch_predictions(source, damaged[:length], starts, predictions)
    if timing is not None:
        elapsed = perf_counter() - start_time
        timing.update(windows=count, seconds=elapsed, windows_per_second=count / elapsed)
    return result
