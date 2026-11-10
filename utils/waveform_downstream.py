"""Downstream-only preparation and paired reporting for the waveform protocol."""
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from data.waveform import REP_SPLITS, WAVEFORM_FORMAT, WINDOW, fit_channel_stats, gesture_log_power


def split_windows(reps, *, final=False):
    split = tuple(np.flatnonzero(np.isin(reps, allowed)) for allowed in REP_SPLITS[:3 if final else 2])
    if any(not len(indices) for indices in split):
        raise ValueError('A required downstream partition has no windows')
    return split[0], split[1], split[-1]


def frozen_development_subject(output, subject, model_ids):
    path = output / 'metrics' / 'waveform_development.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('format') != WAVEFORM_FORMAT or report.get('smoke') is not False:
        raise ValueError(f'Expected a complete non-smoke development run in {path}')
    for row in report['subjects']:
        if row['subject_id'] == subject and row['status'] == 'ok' and row.get('model_sha256') == model_ids:
            return row
    raise ValueError(f'No frozen matching development result for S{subject:02d} in {path}')


def paired_summary(subjects, metric, *, higher_is_better):
    rows = [tuple(metric(subject, arm) for arm in 'ABC') for subject in subjects if subject['status'] == 'ok']
    values = np.asarray(rows, dtype=float).reshape(-1, 3)
    output = {}
    for col, arm in enumerate('ABC'):
        estimates = values[:, col]
        if col:
            delta = (estimates - values[:, 0]) * (1 if higher_is_better else -1)
            output[f'{arm}_improvement_vs_A'] = _summary(delta)
        output[arm] = _summary(estimates)
    return output


def _summary(values):
    values = values[np.isfinite(values)]
    return {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
            'sd': float(values.std(ddof=1)) if len(values) > 1 else None}


class GestureSpectra(Dataset):
    """Generate batched spectra on demand, fitting per-electrode stats on train only."""

    def __init__(self, waves, starts, labels, db2_std, train_idx, *, frozen_stats=None):
        self.waves = waves
        self.starts = np.asarray(starts)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.db2_std = db2_std
        self.stats = frozen_stats if frozen_stats is not None else fit_channel_stats(
            self._spectra(train_idx[i:i + 64]).transpose(0, 2, 3, 1) for i in range(0, len(train_idx), 64))

    def _spectra(self, indices):
        windows = np.stack([self.waves[record][start:start + WINDOW] for record, start in self.starts[indices]])
        result = gesture_log_power(windows, self.db2_std)
        if not np.isfinite(result).all():
            raise ValueError('Gesture spectra contain nonfinite samples')
        return result

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        values = self.stats.standardize(self._spectra(indices).transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        return [{'emg': x, 'label': y} for x, y in zip(torch.from_numpy(values), self.labels[indices])]

    def __len__(self):
        return len(self.labels)
