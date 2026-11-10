"""C-group training on other donors' fully observed windows."""
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from data.waveform import FS, WAVEFORM_FORMAT, WINDOW, window_coverage
from utils.loss_functions import WaveformImputationLoss
from utils.paper_pipeline import save_json, set_seed, train_mcia_epoch
from utils.waveform_pipeline import (WaveformWindows, attach_adapters, checkpoint_sha256, completion_validation,
                                     healthy_checkpoint, load_checkpoint, mask_generator, save_checkpoint, setup_waveform,
                                     validation_patches)


def clean_donor_windows(record, missing):
    """Return eligible starts per split and the deduplicated effective duration."""
    if missing.dtype != np.bool_ or missing.shape != record.filtered.shape:
        raise ValueError('Expected a Boolean donor mask on the original time grid')
    invalid = (missing | ~np.isfinite(record.filtered)).any(axis=1)
    prefix = np.r_[0, invalid.cumsum()]
    starts = record.transitions.starts
    usable = prefix[starts + WINDOW] == prefix[starts]
    pools = [starts[usable & np.isin(record.transitions.repetitions, reps)]
             for reps in ((1, 3, 4), (6,))]
    return [(pool, int(window_coverage(len(invalid), pool).sum())) for pool in pools]


class BalancedDonorSampler(Sampler):
    """Draw donor IDs uniformly and exhaust each shuffled donor pool before reuse."""

    def __init__(self, counts, batch_size, seed, *, smoke=False):
        self.counts = np.asarray(counts)
        if not len(counts) or (self.counts <= 0).any():
            raise ValueError('A donor must have at least one training window')
        self.offsets = np.r_[0, self.counts.cumsum()[:-1]]
        self.steps = 1 if smoke else max(1, int(np.floor(len(counts) * np.median(counts) / batch_size + 0.5)))
        self.total = self.steps * batch_size
        self.rng = np.random.default_rng(seed)
        self.positions = np.zeros(len(counts), dtype=np.int64)
        self.orders = [self.rng.permutation(count) for count in self.counts]

    def __len__(self):
        return self.total

    def __iter__(self):
        for donor in self.rng.integers(len(self.counts), size=self.total):
            pos = self.positions[donor]
            if pos == self.counts[donor]:
                self.orders[donor] = self.rng.permutation(self.counts[donor])
                pos = 0
            self.positions[donor] = pos + 1
            yield int(self.offsets[donor] + self.orders[donor][pos])


def donor_optimizer(model, lr):
    model.requires_grad_(False)
    attach_adapters(model)
    model.pred_head.requires_grad_(True)
    embedding = model.domain_embed.weight
    embedding.requires_grad_(True)
    trainable_rows = torch.arange(len(embedding), device=embedding.device)[:, None] == 1
    embedding.register_hook(lambda grad: grad * trainable_rows)
    trainable = [param for param in model.parameters() if param.requires_grad and param is not embedding]
    return torch.optim.AdamW([{'params': trainable}, {'params': [embedding], 'weight_decay': 0.0}], lr=lr)


def donor_checkpoint(run, donors):
    key = '_'.join(f'S{donor:02d}' for donor in donors)
    return run / '02_db3_transfer_completion' / 'checkpoints' / f'waveform_donors_{key}.pt'


def adapt_donors(cfg, run, device, donor_pools, *, target, smoke=False):
    """Fit one donor set. Targets can share the checkpoint when the donor set matches."""
    donors = tuple(sorted(set(donor_pools) - {target}))
    if not donors:
        raise ValueError(f'No eligible donors for target S{target:02d}')
    seed = cfg['waveform']['seed']
    set_seed(seed)
    prior = healthy_checkpoint(run)
    parent = checkpoint_sha256(prior)
    model, stats = load_checkpoint(prior, cfg, device, smoke=smoke)
    options = cfg['waveform']['donor_adaptation']
    optimizer = donor_optimizer(model, options['learning_rate'])
    train = [donor_pools[donor][0] for donor in donors]
    val = {donor: donor_pools[donor][1] for donor in donors}
    batch_size = cfg['exp1_mcia']['batch_size'] if not smoke else 4
    sampler = BalancedDonorSampler([len(data) for data in train], batch_size, seed, smoke=smoke)
    batches = DataLoader(ConcatDataset(train), sampler=sampler, batch_size=batch_size)
    mask_gen = mask_generator(cfg, seed)
    loss_fn = WaveformImputationLoss().to(device)
    history = []
    best, best_epoch = float('inf'), 0
    path = donor_checkpoint(run, donors)
    frozen = validation_patches(val, cfg)
    mask_path = run / '00_config' / f'{path.stem}_validation_masks.npz'
    np.savez_compressed(mask_path, **{f'S{sid:02d}': masks for sid, masks in frozen.items()})
    for epoch in range(1, (1 if smoke else options['max_epochs']) + 1):
        loss = train_mcia_epoch(model, batches, optimizer, device, mask_gen, loss_fn, domain_id=1)
        score, metrics = completion_validation(model, val, cfg, device, 1, frozen)
        if not np.isfinite(score):
            raise ValueError(f'Nonfinite donor validation score: {donors}')
        if score < best:
            best, best_epoch = score, epoch
            save_checkpoint(path, model, stats, cfg, smoke=smoke, donors=donors, parent=parent)
        history.append(dict(epoch=epoch, train_loss=loss, masked_rmse=score, validation=metrics))
        print(f'Donors {donors}, epoch {epoch}: loss={loss:.6f}, validation masked RMSE={score:.6f}', flush=True)
        if epoch - best_epoch >= options['patience']:
            break
    save_json(run / '00_config' / f'{path.stem}.json', {
        'donors': donors, 'parent_sha256': parent, 'train_windows': dict(zip(donors, map(len, train))),
        'validation_windows': {donor: len(data) for donor, data in val.items()},
        'validation_mask_order': ['s1', 's2', 's3'],
        'validation_patch_masks': mask_path.name,
        'validation_mask_seeds': {donor: seed + donor for donor in donors},
        'batch_size': batch_size, 'steps_per_epoch': sampler.steps,
        'learning_rate': options['learning_rate'], 'max_epochs': options['max_epochs'],
        'patience': options['patience'], 'best_epoch': best_epoch, 'history': history,
    })
    return donors, path


def train_donor_priors(root, *, smoke=False, audit_only=False):
    from utils.waveform_db3 import prepare_db3_exercise, targets, write_quality

    cfg, run, loader, device = setup_waveform(root, smoke=smoke)
    options = cfg['waveform']['donor_adaptation']
    candidates = options['candidates'][:2] if smoke else options['candidates']
    exercises = cfg['exp3_regressor']['exercises'][:1] if smoke else cfg['exp3_regressor']['exercises']
    stats = None
    if not audit_only:
        _, stats = load_checkpoint(healthy_checkpoint(run), cfg, device, smoke=smoke)
    pools, audit = {}, {}
    for subject in candidates:
        rows = [[], []]
        counts, seconds = np.zeros(2, dtype=int), np.zeros(2)
        for exercise in exercises:
            record, quality = prepare_db3_exercise(cfg, loader, subject, exercise, smoke=smoke, smoke_segments=3)
            write_quality(run, subject, exercise, record, quality, smoke=smoke)
            for split, (starts, unique_samples) in enumerate(clean_donor_windows(record, quality.missing)):
                counts[split] += len(starts)
                seconds[split] += unique_samples / FS
                if not audit_only:
                    rows[split].append((record.filtered, starts[:6] if smoke else starts))
        eligible = bool((counts > 0).all())
        audit[subject] = {'eligible': eligible, 'train_windows': int(counts[0]), 'validation_windows': int(counts[1]),
                          'train_unique_seconds': float(seconds[0]), 'validation_unique_seconds': float(seconds[1])}
        if eligible:
            pools[subject] = None if audit_only else tuple(WaveformWindows(row, stats) for row in rows)
        print(f'Donor S{subject:02d}: {audit[subject]}', flush=True)
    target_ids = [5, 2] if smoke else targets(cfg)
    mapping = {subject: sorted(set(pools) - {subject}) for subject in target_ids}
    budget = {}
    for subject, donors in mapping.items():
        if donors:
            budget[subject] = BalancedDonorSampler([audit[donor]['train_windows'] for donor in donors],
                                                   cfg['exp1_mcia']['batch_size'], 0, smoke=smoke).steps
    save_json(run / '00_config' / 'waveform_donor_eligibility.json', {
        'format': WAVEFORM_FORMAT, 'exercises': exercises, 'candidates': audit,
        'targets': mapping, 'steps_per_epoch': budget,
    })
    if audit_only:
        return
    if any(not donors for donors in mapping.values()):
        raise ValueError('At least one target has no eligible other donor')
    fitted = set()
    for subject, donors in mapping.items():
        key = tuple(donors)
        if key not in fitted:
            adapt_donors(cfg, run, device, pools, target=subject, smoke=smoke)
            fitted.add(key)
    save_json(run / '00_config' / 'waveform_donor_mapping.json', {'format': WAVEFORM_FORMAT, 'targets': mapping})
