"""Shared DB3 records, masks, and one delivery per subject/exercise and group."""
import hashlib
import json

import numpy as np

from data.waveform import FS, WAVEFORM_FORMAT, prepare_recording, window_coverage
from utils.paper_pipeline import save_json
from utils.waveform_adaptation import donor_checkpoint
from utils.waveform_delivery import deliver_waveform
from utils.waveform_pipeline import checkpoint_sha256, healthy_checkpoint, load_checkpoint
from utils.waveform_quality import detect_waveform_quality


def targets(cfg):
    return sorted(set(cfg['exp3_regressor']['db3_subjects']) | set(cfg['exp4_gesture']['subjects']))


def prepare_db3_exercise(cfg, loader, subject, exercise, *, glove=False, smoke=False, final=False,
                         smoke_segments=3, run=None):
    angle = (exercise in cfg['exp3_regressor']['exercises'] and
             (subject in cfg['exp3_regressor']['db3_subjects'] or
              subject in cfg['waveform']['donor_adaptation']['candidates']))
    gesture = subject in cfg['exp4_gesture']['subjects'] and exercise in cfg['exp4_gesture']['exercises']
    frozen = None
    if final:
        if smoke or run is None:
            raise ValueError('Final DB3 evaluation requires a frozen development run')
        path = _quality_prefix(run, subject, exercise, smoke=False, final=False).with_suffix('.npz')
        with np.load(path, allow_pickle=False) as data:
            if str(data['format']) != WAVEFORM_FORMAT or np.any(data['partitions'] == 2):
                raise ValueError(f'Invalid development mask: {path}')
            frozen = (data['hard_zero'], data['thresholds'], data['training_counts'])
    record = prepare_recording(loader, 'db3', subject, exercise, with_glove=glove,
                               splits=(0, 1, 2) if final else (0, 1),
                               segment_limit=(1 if exercise == 3 else smoke_segments) if smoke else None,
                               action=49 if smoke and exercise == 3 else None)
    quality = detect_waveform_quality(record, angle=angle, gesture=gesture, frozen=frozen)
    return record, quality


def _scope(smoke, final):
    if smoke and final:
        raise ValueError('A smoke run cannot access the evaluation repetitions')
    return 'final' if final else 'development'


def _quality_prefix(run, subject, exercise, *, smoke, final):
    directory = run / '02_db3_transfer_completion' / 'quality' / _scope(smoke, final)
    return directory / f'S{subject:02d}_E{exercise}'


def write_quality(run, subject, exercise, record, quality, *, smoke=False, final=False):
    prefix = _quality_prefix(run, subject, exercise, smoke=smoke, final=final)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata = quality.metadata()
    validation = np.isin(record.repetitions, (6,))
    for name, windows in (('angle', record.transitions), ('gesture', record.gestures)):
        starts = windows.starts[windows.repetitions == 6]
        coverage = window_coverage(len(record.raw), starts)
        row = {'windows': len(starts), 'covered_samples': int(coverage.sum()),
               'missing_samples_per_electrode': quality.missing[coverage & validation].sum(0).tolist()}
        metadata[name] = row
    save_json(prefix.with_suffix('.json'), metadata)
    np.savez_compressed(prefix.with_suffix('.npz'), format=np.asarray(WAVEFORM_FORMAT),
                        missing=quality.missing, thresholds=quality.thresholds, hard_zero=quality.hard_zero,
                        training_counts=quality.training_counts, starts=quality.starts,
                        partitions=quality.partitions, flagged=quality.flagged)


def load_abc_models(cfg, run, device, subject, *, smoke=False):
    prior = healthy_checkpoint(run)
    b, stats = load_checkpoint(prior, cfg, device, smoke=smoke)
    manifest = json.loads((run / '00_config' / 'waveform_donor_mapping.json').read_text(encoding='utf-8'))
    donors = tuple(manifest['targets'][str(subject)])
    if not donors or subject in donors or manifest['format'] != WAVEFORM_FORMAT:
        raise ValueError(f'Invalid donor mapping for S{subject:02d}')
    c_path = donor_checkpoint(run, donors)
    prior_id = checkpoint_sha256(prior)
    c, c_stats = load_checkpoint(c_path, cfg, device, smoke=smoke, donors=donors, parent=prior_id)
    if not (np.array_equal(stats.mean, c_stats.mean) and np.array_equal(stats.std, c_stats.std)):
        raise ValueError('B and C checkpoint DB2 statistics differ')
    identities = {'B': prior_id, 'C': checkpoint_sha256(c_path)}
    return {'B': (b, 0), 'C': (c, 1)}, stats, identities


def _delivery_identity(record, quality, model_ids):
    result = hashlib.sha256(json.dumps(model_ids, sort_keys=True).encode('ascii'))
    result.update(np.ascontiguousarray(record.filtered))
    result.update(np.ascontiguousarray(quality.missing))
    return result.hexdigest()


def abc_waveforms(run, subject, exercise, record, quality, models, stats, model_ids, device,
                  *, smoke=False, final=False):
    """A uses the observed filtered waveform. B/C share saved deliveries across tasks."""
    directory = run / '02_db3_transfer_completion' / 'augmented_emg' / _scope(smoke, final)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'S{subject:02d}_E{exercise}_waveform.npz'
    identity = _delivery_identity(record, quality, model_ids)
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            if str(payload['identity']) != identity or str(payload['format']) != WAVEFORM_FORMAT:
                raise ValueError(f'Incompatible delivery in this run: {path}')
            print(f'S{subject:02d} E{exercise}: reusing shared B/C waveform', flush=True)
            return {key: payload[key] for key in ('A', 'B', 'C')}
    write_quality(run, subject, exercise, record, quality, smoke=smoke, final=final)
    delivered = {'A': record.filtered}
    timing = {}
    for key, (model, domain) in models.items():
        timing[key] = {}
        wave = deliver_waveform(record, quality.missing, model, stats, device, domain=domain, timing=timing[key])
        observed = ~quality.missing
        if not np.array_equal(wave[observed], record.filtered[observed], equal_nan=True):
            raise RuntimeError(f'{key} changed observed samples for S{subject:02d} E{exercise}')
        delivered[key] = wave
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, format=np.asarray(WAVEFORM_FORMAT), fs=np.asarray(FS),
                        identity=np.asarray(identity), **delivered)
    temp.replace(path)
    save_json(path.with_suffix('.json'), {'identity': identity, 'model_sha256': model_ids, 'delivery': timing})
    return delivered
