"""Shared original-time-grid preparation for the signed waveform protocol."""
from dataclasses import dataclass

import numpy as np
from scipy import signal


REP_SPLITS = ((1, 3, 4), (6,), (2, 5))
WAVEFORM_FORMAT = 'mcia-waveform-2khz-v1'
FS = 2000
ANGLE_FS = 200
WINDOW = 600
STRIDE = 100
FRONTEND_WARMUP = 120
ANGLE_WARMUP = 500


def runs(values):
    """Half-open bounds for runs of equal values."""
    if not len(values):
        return np.empty((0, 2), dtype=np.int64)
    edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return np.column_stack((edges[:-1], edges[1:]))


def partition_segments(repetitions, *, splits=(0, 1, 2)):
    """Contiguous portions of one partition, preserving tagged rest samples."""
    partition = np.full(len(repetitions), -1, dtype=np.int8)
    for split in splits:
        partition[np.isin(repetitions, REP_SPLITS[split])] = split
    return [(int(lo), int(hi), int(partition[lo])) for lo, hi in runs(partition)
            if partition[lo] >= 0]


@dataclass(frozen=True)
class Windows:
    starts: np.ndarray
    actions: np.ndarray
    repetitions: np.ndarray
    trials: np.ndarray

    def samples(self, step=1):
        return self.starts[:, None] + np.arange(0, WINDOW, step)


def task_windows(labels, repetitions, segments, *, gesture=False):
    """Retain one contiguous action, plus adjacent rest for the angle grid."""
    rows = []
    for lo, hi, _ in segments:
        action_runs = runs(labels[lo:hi].astype(np.int32) * 8 + repetitions[lo:hi]) + lo
        actions = [(start, end) for start, end in action_runs if labels[start] > 0]
        if not actions:
            continue
        anchor = actions[0][0]
        started = gesture
        for i, (start, end) in enumerate(actions):
            if gesture:
                first, last = start, end - WINDOW
            else:
                prev_end = actions[i - 1][1] if i else lo
                next_start = actions[i + 1][0] if i + 1 < len(actions) else hi
                first = max(prev_end, start - WINDOW + 1)
                last = min(next_start - WINDOW, end - 1)
                first = anchor + (first - anchor + STRIDE - 1) // STRIDE * STRIDE
                if not started:
                    if i or first >= start or first > last:
                        first = anchor + (start - anchor + STRIDE - 1) // STRIDE * STRIDE
                        if first + WINDOW > end:
                            continue
                    started = True
            rows.extend((point, labels[start], repetitions[start], start)
                        for point in range(first, last + 1, STRIDE))
    rows.sort()
    starts, actions, reps, trials = np.asarray(rows, dtype=np.int64).reshape(-1, 4).T
    return Windows(starts, actions, reps, trials)


def window_coverage(length, starts, width=WINDOW):
    change = np.zeros(length + 1, dtype=np.int32)
    np.add.at(change, starts, 1)
    np.add.at(change, starts + width, -1)
    return change[:-1].cumsum() > 0


def _sos_filter(values, sos, begins_at_rest):
    def filter_valid(block, has_history):
        zi = signal.sosfilt_zi(sos)[:, :, None] * block[0]
        return signal.sosfilt(sos, block, axis=0, zi=np.zeros_like(zi) if has_history else zi)[0]

    valid = np.isfinite(values)
    if valid.all():
        return filter_valid(values, begins_at_rest)
    result = np.full_like(values, np.nan)
    for ch in range(values.shape[1]):
        for start, end in runs(valid[:, ch]):
            if valid[start, ch]:
                result[start:end, ch:ch + 1] = filter_valid(
                    values[start:end, ch:ch + 1], begins_at_rest and start == 0)
    return result


def filter_segments(emg, labels, segments, stages, *, steady_start=False):
    filtered = np.full_like(emg, np.nan)
    for lo, hi, _ in segments:
        block = emg[lo:hi]
        for sos in stages:
            block = _sos_filter(block, sos, not steady_start and labels[lo] == 0)
        filtered[lo:hi] = block
    return filtered


FRONTEND_SOS = tuple(signal.butter(2, cutoff, btype=kind, fs=FS, output='sos')
                     for cutoff, kind in ((20, 'highpass'), (450, 'lowpass')))
ANGLE_SOS = signal.butter(2, 5, fs=FS, output='sos')


@dataclass(frozen=True)
class Recording:
    raw: np.ndarray
    filtered: np.ndarray
    labels: np.ndarray
    repetitions: np.ndarray
    glove: np.ndarray | None
    segments: list
    transitions: Windows
    gestures: Windows


def prepare_recording(loader, db, subject, exercise, *, with_glove=False, splits=(0, 1, 2),
                      segment_limit=None, action=None):
    if loader.fs != FS:
        raise ValueError(f'{WAVEFORM_FORMAT} requires {FS} Hz input')
    raw, labels, repetitions, glove = loader.load_exercise(db, subject, exercise, with_glove=with_glove)
    segments = partition_segments(repetitions, splits=splits)
    if segment_limit is not None:
        selected = []
        for split in splits:
            eligible = [(lo, hi, part) for lo, hi, part in segments
                        if part == split and (action is None or (labels[lo:hi] == action).any())]
            if eligible:
                indices = np.linspace(0, len(eligible) - 1, min(segment_limit, len(eligible)), dtype=int)
                selected.extend(eligible[index] for index in indices)
        segments = sorted(selected)
    filtered = filter_segments(raw, labels, segments, FRONTEND_SOS)
    return Recording(raw, filtered, labels, repetitions, glove, segments,
                     task_windows(labels, repetitions, segments),
                     task_windows(labels, repetitions, segments, gesture=True))


def angle_inputs(recording, delivered):
    """Envelope on the original grid, then paired 200 Hz EMG and glove samples."""
    envelope = filter_segments(np.abs(delivered), recording.labels, recording.segments, (ANGLE_SOS,))
    samples = recording.transitions.samples(FS // ANGLE_FS)
    return envelope[samples], samples


HAMMING = signal.windows.hamming(128, sym=True).astype(np.float32)


def gesture_log_power(windows, db2_std):
    """D-021/D-022, from (batch, 600, 12) to (batch, 12, 65, 19)."""
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW, 12):
        raise ValueError('Gesture STFT expects (batch, 600, 12)')
    if np.shape(db2_std) != (12,) or np.any(~np.isfinite(db2_std)) or np.any(db2_std <= 0):
        raise ValueError('Gesture STFT requires the fixed DB2 standard deviations')
    padded = np.pad(windows, ((0, 0), (64, 64), (0, 0)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 128, axis=1)[:, ::32]
    spectrum = np.fft.rfft(frames * HAMMING, axis=-1)
    power = (spectrum.real**2 + spectrum.imag**2) / np.dot(HAMMING, HAMMING)
    log_power = 10 * np.log10(power / np.square(db2_std)[None, None, :, None] + 1e-6)
    return log_power.transpose(0, 2, 3, 1).astype(np.float32)


@dataclass(frozen=True)
class ChannelStats:
    mean: np.ndarray
    std: np.ndarray

    def standardize(self, values):
        scale = np.where(self.std > 0, self.std, 1)
        return np.where(self.std > 0, (values - self.mean) / scale, 0).astype(np.float32)

    def restore(self, values):
        return (values * self.std + self.mean).astype(np.float32)


def fit_channel_stats(blocks):
    """Merge training block moments in float64 without retaining the fitting pool."""
    count = 0
    mean = np.zeros(12, dtype=np.float64)
    squares = np.zeros_like(mean)
    for block in blocks:
        values = np.asarray(block, dtype=np.float64).reshape(-1, 12)
        if not len(values):
            continue
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite normalization training input')
        center = values.mean(axis=0)
        delta = center - mean
        total = count + len(values)
        squares += np.square(values - center).sum(axis=0) + delta**2 * (count * len(values) / total)
        mean += delta * (len(values) / total)
        count = total
    if not count:
        raise ValueError('Empty normalization training pool')
    return ChannelStats(mean.astype(np.float32), np.sqrt(squares / count).astype(np.float32))
