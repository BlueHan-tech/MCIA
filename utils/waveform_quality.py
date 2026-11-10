"""D-035/D-036 detection on the shared 2 kHz time grid."""
from dataclasses import dataclass

import numpy as np
from scipy import signal

from data.waveform import FS, STRIDE, filter_segments, window_coverage


SHORT = 400
LOW_SOS = signal.butter(2, 20, 'lowpass', fs=FS, output='sos')
QUANTILES = (0.995, 0.995, 0.95, 0.995)
FEATURES = ('low_rms', 'raw_step', 'band_rms', 'band_peak')


@dataclass(frozen=True)
class Quality:
    missing: np.ndarray
    hard_zero: np.ndarray
    thresholds: np.ndarray
    starts: np.ndarray
    partitions: np.ndarray
    flagged: np.ndarray
    training_counts: np.ndarray

    def metadata(self):
        return {
            'short_window': SHORT, 'stride': STRIDE, 'fs': FS,
            'lowpass_hz': 20, 'lowpass_order': 2,
            'features': FEATURES, 'quantiles': QUANTILES,
            'hard_zero_one_based': (np.flatnonzero(self.hard_zero) + 1).tolist(),
            'thresholds': [[float(value) if np.isfinite(value) else None for value in row]
                           for row in self.thresholds],
            'training_counts': self.training_counts.tolist(),
            'validation_flagged_short_windows': self.flagged[self.partitions == 1].sum(axis=0).tolist(),
        }


def short_features(record, coverage, angle_coverage):
    low = filter_segments(record.raw, record.labels, record.segments, (LOW_SOS,), steady_start=True)
    cumulative = np.r_[0, coverage.cumsum()]
    active = np.r_[0, (record.labels > 0).cumsum()]
    angle = np.r_[0, angle_coverage.cumsum()]
    chunks, ids, feature_chunks = [], [], []
    for lo, hi, partition in record.segments:
        starts = np.arange(lo + (-lo) % STRIDE, hi - SHORT + 1, STRIDE)
        ends = starts + SHORT
        eligible = ((cumulative[ends] > cumulative[starts])
                    & ((active[ends] > active[starts]) | (angle[ends] > angle[starts])))
        starts = starts[eligible]
        if not len(starts):
            continue
        steps = np.abs(np.diff(record.raw[lo:hi].astype(np.float64), axis=0, prepend=record.raw[lo:lo + 1]))
        for start in range(0, len(starts), 128):
            points = starts[start:start + 128]
            rows = points[:, None] + np.arange(SHORT)
            band = record.filtered[rows]
            feature_chunks.append(np.stack((low[rows].std(axis=1, dtype=np.float64),
                                            steps[rows - lo].max(axis=1),
                                            band.std(axis=1, dtype=np.float64),
                                            np.abs(band).max(axis=1)), axis=-1))
            chunks.append(points)
            ids.append(np.full(len(points), partition, dtype=np.int8))
    if not chunks:
        raise ValueError('No eligible short windows for the DB3 detector')
    return np.concatenate(chunks), np.concatenate(ids), np.concatenate(feature_chunks)


def detect_waveform_quality(record, *, angle=True, gesture=True, frozen=None):
    length = len(record.raw)
    transition = window_coverage(length, record.transitions.starts) if angle else np.zeros(length, dtype=bool)
    coverage = transition.copy()
    if gesture:
        coverage |= window_coverage(length, record.gestures.starts)
    starts, partitions, features = short_features(record, coverage, transition)
    valid = np.isfinite(features).all(axis=-1)
    if frozen is None:
        if np.any(partitions == 2):
            raise ValueError('Evaluation samples require frozen DB3 quality thresholds')
        train_active = (record.labels > 0) & np.isin(record.repetitions, (1, 3, 4))
        if not train_active.any():
            raise ValueError('DB3 detector requires training action samples')
        hard_zero = (record.raw[train_active] == 0).all(axis=0)
        thresholds = np.full((12, 4), np.inf)
        counts = np.zeros(12, dtype=np.int64)
        for electrode in np.flatnonzero(~hard_zero):
            training = features[(partitions == 0) & valid[:, electrode], electrode]
            counts[electrode] = len(training)
            if not len(training):
                raise ValueError(f'No eligible training short windows on electrode {electrode + 1}')
            thresholds[electrode] = [np.quantile(training[:, i], q) for i, q in enumerate(QUANTILES)]
    else:
        hard_zero, thresholds, counts = frozen
        if (hard_zero.shape != (12,) or hard_zero.dtype != np.bool_ or thresholds.shape != (12, 4)
                or counts.shape != (12,) or not np.isfinite(thresholds[~hard_zero]).all()
                or (counts[~hard_zero] <= 0).any() or not np.isposinf(thresholds[hard_zero]).all()):
            raise ValueError('Invalid frozen DB3 quality thresholds')
    high = features > thresholds
    flagged = valid & (high[..., 0] | high[..., 1]) & (high[..., 2] | high[..., 3])
    missing = np.broadcast_to(hard_zero, record.raw.shape).copy()
    for electrode in np.flatnonzero(flagged.any(axis=0)):
        covered = window_coverage(length, starts[flagged[:, electrode]], SHORT)
        missing[:, electrode] |= covered & coverage
    return Quality(missing, hard_zero, thresholds, starts, partitions, flagged, counts)
