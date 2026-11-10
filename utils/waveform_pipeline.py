"""Training and checkpoint contracts for the 2 kHz main experiment."""
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.ninapro_loader import NinaProDataLoader
from data.waveform import (ANGLE_FS, FS, STRIDE, WAVEFORM_FORMAT, WINDOW, ChannelStats,
                           fit_channel_stats, prepare_recording)
from models.completion.mask_generators import WaveformScenarioMix
from models.completion.mcia_core import WaveformMCIA, derive_ch_mask_from_sample_mask
from utils.loss_functions import WaveformImputationLoss, waveform_rms_terms
from utils.paper_pipeline import load_yaml_config, save_json, set_seed, train_mcia_epoch
from utils.run_layout import get_run_dir


PATCH = 40
ARCHITECTURE_KEYS = ('embed_dim', 'n_layers', 'n_heads', 'ffn_dim', 'model_dropout')
SOURCE_FILES = (
    'data/ninapro_loader.py', 'data/waveform.py', 'models/completion/mcia_core.py',
    'models/completion/mask_generators.py', 'models/prediction/dsranet.py',
    'models/prediction/kinematic_regressor.py', 'utils/loss_functions.py', 'utils/paper_pipeline.py',
    'utils/kinematic_target.py', 'utils/kinematic_output_postprocess.py',
    'scripts/01_train_mcia_db2_healthy_prior.py', 'scripts/04_eval_db3_angle_raw_vs_augmented.py',
    'scripts/05_eval_db3_gesture_raw_vs_augmented.py', 'scripts/06_adapt_amputee_donor_prior.py',
    'scripts/run_all_experiments.py', 'docs/EXPERIMENT_PROTOCOL.md',
)


def setup_waveform(root, *, smoke=False):
    cfg = load_yaml_config(root)
    contract = {'format': WAVEFORM_FORMAT, 'fs': FS, 'angle_fs': ANGLE_FS,
                'window_size': WINDOW, 'stride': STRIDE, 'patch_size': PATCH}
    spec = cfg['waveform']
    if any(spec.get(key) != value for key, value in contract.items()):
        raise ValueError(f'Incompatible waveform config: expected {contract}')
    contract['smoke'] = smoke
    requested = os.environ.get('MCIA_RUN_DIR')
    if requested:
        root_dir = Path(requested)
        marker = root_dir / '00_config' / 'waveform.json'
        if not marker.exists() and (root_dir / '00_config' / 'config_snapshot.yaml').exists():
            raise ValueError(f'Cannot write waveform results into an existing legacy run: {root_dir}')
    run = get_run_dir(root, cfg)
    marker = run / '00_config' / 'waveform.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) != contract:
        raise ValueError(f'Incompatible waveform run: {run}')
    from yaml import safe_load
    snapshot = safe_load((run / '00_config' / 'config_snapshot.yaml').read_text(encoding='utf-8'))
    if snapshot != cfg:
        raise ValueError(f'Current config differs from the frozen run snapshot: {run}')
    files = [root / name for name in SOURCE_FILES] + list((root / 'utils').glob('waveform_*.py'))
    hashes = {file.relative_to(root).as_posix(): hashlib.sha256(
        file.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for file in files}
    manifest = run / '00_config' / 'waveform_sources.json'
    if manifest.is_file() and json.loads(manifest.read_text(encoding='utf-8')) != hashes:
        raise ValueError(f'Current source or protocol differs from the frozen run: {run}')
    if not manifest.exists():
        save_json(manifest, hashes)
    save_json(marker, contract)
    paths = cfg['paths']
    loader = NinaProDataLoader(paths['db2'], paths['db3'], fs=FS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if smoke:
        torch.set_num_threads(min(torch.get_num_threads(), 2))
    return cfg, run, loader, device


def model_settings(cfg):
    train = cfg['exp1_mcia']
    settings = {key: train[key] for key in ARCHITECTURE_KEYS}
    settings['dropout'] = settings.pop('model_dropout')
    return dict(settings, window_size=WINDOW, patch_size=PATCH, n_channels=12, num_domains=2)


def mask_generator(cfg, seed):
    return WaveformScenarioMix(group_indices=cfg['exp1_mcia']['group_indices'],
                               min_alive_per_group=cfg['exp1_mcia']['min_alive_per_group'],
                               rng=np.random.default_rng(seed))


class WaveformWindows(Dataset):
    """Slice continuous records on demand to avoid storing overlapping windows."""

    def __init__(self, records, stats):
        self.records = [emg for emg, _ in records]
        self.stats = stats
        self.index = [(record, start) for record, (_, starts) in enumerate(records) for start in starts]
        if not self.index:
            raise ValueError('Empty completion window pool')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        record, start = self.index[index]
        return self.stats.standardize(self.records[record][start:start + WINDOW])


def predict_windows(model, windows, keep, domain):
    masked = torch.where(keep.bool(), windows, 0)
    return model(masked, mask=derive_ch_mask_from_sample_mask(keep), raw_time_mask=keep,
                 domain_id=torch.full((len(windows),), domain, dtype=torch.long, device=windows.device))


def validation_patches(datasets, cfg):
    masks = {}
    for subject, data in datasets.items():
        generator = mask_generator(cfg, cfg['waveform']['seed'] + subject)
        masks[subject] = np.stack([generator.generate_batch_masks(1, scenario=f's{i % 3 + 1}')
                                   .numpy()[0, :, ::PATCH].T.astype(bool) for i in range(len(data))])
    return masks


@torch.inference_mode()
def completion_validation(model, datasets, cfg, device, domain, masks):
    """Evaluate the exact same per-window masks each epoch."""
    model.eval()
    metrics = {}
    for subject, data in datasets.items():
        totals = defaultdict(float)
        for batch_number, windows in enumerate(DataLoader(data, batch_size=cfg['exp1_mcia']['batch_size'])):
            windows = windows.to(device)
            offset = batch_number * cfg['exp1_mcia']['batch_size']
            patches = torch.from_numpy(masks[subject][offset:offset + len(windows)])
            keep = patches.to(device=device, dtype=windows.dtype).repeat_interleave(PATCH, dim=1)
            pred = predict_windows(model, windows, keep, domain)
            completed = torch.where(keep.bool(), windows, pred)
            error = completed - windows
            totals['sse'] += error.square().sum().item()
            totals['sae'] += error.abs().sum().item()
            totals['missing'] += (1 - keep).sum().item()
            rms, count = waveform_rms_terms(completed, windows, 1 - keep)
            totals['rms'] += rms.item()
            totals['rms_weight'] += count.item()
        metrics[subject] = dict(masked_rmse=(totals['sse'] / totals['missing'])**0.5,
                                masked_mae=totals['sae'] / totals['missing'],
                                local_rms_mae=totals['rms'] / totals['rms_weight'])
    return float(np.mean([row['masked_rmse'] for row in metrics.values()])), metrics


def checkpoint_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_checkpoint(path, model, stats, cfg, *, smoke, donors=(), parent=None):
    payload = {'format': WAVEFORM_FORMAT, 'model': model.state_dict(), 'settings': model_settings(cfg),
               'mean': torch.from_numpy(stats.mean), 'std': torch.from_numpy(stats.std),
               'smoke': smoke, 'donors': list(donors), 'parent_sha256': parent}
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def attach_adapters(model):
    from models.completion.mcia_core import Adapter
    for block in model.blocks[-2:]:
        block.adapter = Adapter(model.embed_dim).to(model.domain_embed.weight.device)


def load_checkpoint(path, cfg, device, *, smoke, donors=(), parent=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    settings = model_settings(cfg)
    if (not isinstance(payload, dict) or payload.get('format') != WAVEFORM_FORMAT
            or payload.get('settings') != settings or payload.get('smoke') != smoke
            or payload.get('donors') != list(donors) or payload.get('parent_sha256') != parent):
        raise ValueError(f'Incompatible waveform checkpoint: {path}')
    stats = ChannelStats(payload['mean'].numpy(), payload['std'].numpy())
    if (stats.mean.shape != (12,) or stats.std.shape != (12,) or not np.isfinite(stats.mean).all()
            or not np.isfinite(stats.std).all() or np.any(stats.std <= 0)):
        raise ValueError(f'Invalid DB2 training scale in {path}')
    model = WaveformMCIA(**settings).to(device)
    if donors:
        attach_adapters(model)
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), stats


def healthy_checkpoint(run):
    return run / '01_db2_completion' / 'checkpoints' / 'waveform_best.pt'


def plot_waveform_validation_examples(output):
    from utils.visualization import plot_completion_panel

    path = output / 'metrics' / 'waveform_validation_examples.npz'
    figures = []
    with np.load(path, allow_pickle=False) as data:
        if str(data['format']) != WAVEFORM_FORMAT:
            raise ValueError(f'Incompatible waveform examples: {path}')
        for i, (clean, completed, keep) in enumerate(zip(data['clean'], data['completed'], data['keep'])):
            figure = output / 'figures' / f'waveform_validation_s{i + 1}.png'
            plot_completion_panel(clean, completed, keep, figure, show_envelope=False,
                                  title=f'DB2 validation S{int(data["subject"]):02d} / Scenario S{i + 1}',
                                  header_text='2 kHz signed waveform, fixed DB2 training Z-score units')
            figures.append(figure)
    return figures


@torch.inference_mode()
def save_waveform_validation_examples(run, cfg, device, val, frozen, smoke):
    model, _ = load_checkpoint(healthy_checkpoint(run), cfg, device, smoke=smoke)
    subject = next(iter(val))
    count = min(3, len(val[subject]))
    clean = torch.as_tensor(np.stack([val[subject][i] for i in range(count)]), device=device)
    keep = torch.from_numpy(frozen[subject][:count]).to(device=device, dtype=clean.dtype).repeat_interleave(PATCH, dim=1)
    completed = torch.where(keep.bool(), clean, predict_windows(model, clean, keep, 0))
    output = run / '01_db2_completion'
    np.savez_compressed(output / 'metrics' / 'waveform_validation_examples.npz', format=np.asarray(WAVEFORM_FORMAT),
                        subject=np.asarray(subject), clean=clean.cpu().numpy(),
                        completed=completed.cpu().numpy(), keep=keep.cpu().numpy())
    return plot_waveform_validation_examples(output)


def train_healthy(root, *, smoke=False):
    cfg, run, loader, device = setup_waveform(root, smoke=smoke)
    options = cfg['exp1_mcia']
    seed = cfg['waveform']['seed']
    set_seed(seed)
    train_subjects = options['train_subjects'][:1] if smoke else options['train_subjects']
    val_subjects = options['val_subjects'][:1] if smoke else options['val_subjects']
    if set(train_subjects) & (set(val_subjects) | set(options['test_subjects'])) or set(val_subjects) & set(options['test_subjects']):
        raise ValueError('Overlapping DB2 subject splits')
    train_records = []

    def training_blocks():
        for subject in train_subjects:
            for exercise in cfg['waveform']['db2_exercises']:
                record = prepare_recording(loader, 'db2', subject, exercise, splits=(0,))
                starts = record.transitions.starts[:4] if smoke else record.transitions.starts
                train_records.append((record.filtered, starts))
                for lo, hi, _ in record.segments:
                    yield record.filtered[lo:hi]

    stats = fit_channel_stats(training_blocks())
    if np.any(stats.std <= 0):
        raise ValueError('DB2 training waveform must have nonzero variance on every electrode')
    train = WaveformWindows(train_records, stats)
    train_records.clear()
    val = {}
    for subject in val_subjects:
        records = []
        for exercise in cfg['waveform']['db2_exercises']:
            record = prepare_recording(loader, 'db2', subject, exercise, splits=(1,))
            starts = record.transitions.starts[:3] if smoke else record.transitions.starts
            records.append((record.filtered, starts))
        val[subject] = WaveformWindows(records, stats)
    batches = DataLoader(train, batch_size=options['batch_size'], shuffle=True)
    model = WaveformMCIA(**model_settings(cfg)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=options['lr_factor'],
                                                           patience=options['lr_patience'], min_lr=options['min_lr'])
    masks = mask_generator(cfg, seed)
    frozen = validation_patches(val, cfg)
    criterion = WaveformImputationLoss().to(device)
    best = float('inf')
    best_epoch = 0
    history = []
    for epoch in range(1, (1 if smoke else options['num_epochs']) + 1):
        loss = train_mcia_epoch(model, batches, optimizer, device, masks, criterion, domain_id=0)
        score, metrics = completion_validation(model, val, cfg, device, 0, frozen)
        scheduler.step(score)
        if not np.isfinite(score):
            raise ValueError('Nonfinite DB2 validation score')
        if score < best:
            best, best_epoch = score, epoch
            save_checkpoint(healthy_checkpoint(run), model, stats, cfg, smoke=smoke)
        history.append(dict(epoch=epoch, loss=loss, masked_rmse=score, validation=metrics))
        print(f'DB2 epoch {epoch}: loss={loss:.6f} validation masked RMSE={score:.6f}', flush=True)
        if epoch - best_epoch >= options['patience']:
            break
    save_json(run / '01_db2_completion' / 'metrics' / 'waveform_training.json', {
        'format': WAVEFORM_FORMAT, 'smoke': smoke, 'train_subjects': train_subjects,
        'val_subjects': val_subjects, 'train_windows': len(train),
        'val_windows': {sid: len(data) for sid, data in val.items()}, 'best_epoch': best_epoch,
        'history': history,
    })
    if cfg['viz']['enable']:
        save_waveform_validation_examples(run, cfg, device, val, frozen, smoke)
