"""Time-bounded paired activation screen, train/validation only."""
from __future__ import annotations
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()
import numpy as np
import torch
import yaml
from data.ninapro_loader import NinaProDataLoader
from data.dataset_kinematics import prepare_kinematics_data
from models.completion.mask_generators import ScenarioMixMaskGenerator
from utils.paper_pipeline import build_mcia, build_structural_loss, set_seed


def main():
    started = time.monotonic()
    deadline = started + 360
    run = Path(os.environ['MCIA_RUN_DIR']).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    out = run / '06_diagnostics' / 'softplus_clip_vs_sigmoid_20260908'
    out.mkdir(parents=True, exist_ok=False)
    root = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    cfg = dict(root['signal'], **root['exp1_mcia'])
    device = 'cuda'
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for the bounded comparison')
    torch.set_num_threads(4)
    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loader = NinaProDataLoader(root['paths']['db2'], root['paths']['db3'], fs=cfg['orig_fs'])
    # Reuse the train-repetition-only scaler, not the legacy DB2 scaler.
    x, _, subjects, reps, metadata = prepare_kinematics_data(
        loader, [1, 2], cfg, exercises=(1,), db='db2', return_metadata=True)
    rng = np.random.default_rng(42)
    selected = {}
    for split, allowed, count in [('train', [1, 3, 4], 128), ('validation', [6], 32)]:
        selected[split] = np.concatenate([
            np.sort(rng.choice(np.flatnonzero((subjects == s) & np.isin(reps, allowed)),
                               count, replace=False)) for s in [1, 2]])
    train = torch.tensor(x[selected['train']], device=device)
    val = torch.tensor(x[selected['validation']], device=device)
    val_subjects = subjects[selected['validation']]
    assert not np.isin(reps[np.concatenate(list(selected.values()))], [2, 5]).any()
    np.savez_compressed(out / 'selected_windows.npz', **selected,
                        subjects=subjects, repetitions=reps, starts=metadata['start'])

    def masks(seed):
        return ScenarioMixMaskGenerator(n_channels=cfg['n_channels'], time_steps=cfg['window_size'],
            patch_size=cfg['patch_size'], group_indices=cfg.get('group_indices'),
            min_alive_per_group=cfg.get('min_alive_per_group'), scenario_weights=cfg.get('scenario_weights'),
            scenario_params=cfg.get('scenario_params'), rng=np.random.default_rng(seed))

    scenarios = ['s1', 's2', 's3']
    fixed_masks = {s: masks(7000 + i).generate_mask(val, scenario=s) for i, s in enumerate(scenarios)}
    models = {}
    set_seed(42)
    base = build_mcia(cfg, device)
    for name in ['softplus_clip', 'sigmoid']:
        models[name] = copy.deepcopy(base)
    models['sigmoid'].pred_head[1] = torch.nn.Sigmoid()
    del base
    optimizers = {n: torch.optim.AdamW(m.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
                  for n, m in models.items()}
    criterion = build_structural_loss(cfg, device)
    if criterion is None:
        raise RuntimeError('This screen expects the configured structural loss')
    protocol = dict(database='DB2', subjects=[1, 2], exercise=1, train_repetitions=[1, 3, 4],
                    validation_repetitions=[6], test_evaluated=False, train_windows=256, val_windows=64,
                    seed=42, batch_size=16, planned_epochs=8, learning_rate=cfg['learning_rate'],
                    initialization='identical random weights; only final activation differs',
                    softplus='beta=10, unclipped training loss, clipped evaluation',
                    sigmoid='sigmoid during training and evaluation',
                    comparison='last fully completed paired epoch; no checkpoint selection',
                    preprocessing='project prepare_kinematics_data; train-only scaling; full-record filtering',
                    scope='short-budget reconstruction screen; no downstream or independent test claim',
                    configuration=cfg, gpu=torch.cuda.get_device_name(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report = {'protocol': protocol, 'history': [], 'status': 'running'}

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=True), encoding='utf-8')

    @torch.no_grad()
    def evaluate(model, save_predictions=False):
        model.eval()
        result, predictions = {}, {}
        for scenario, mask in fixed_masks.items():
            chunks = []
            for i in range(0, len(val), 16):
                y, m = val[i:i+16], mask[i:i+16]
                p = model(y*m, raw_time_mask=m).clamp(0, 1)
                completed = p*(1-m)+y*m
                assert torch.equal(completed[m.bool()], y[m.bool()])
                chunks.append(completed)
            p = torch.cat(chunks).cpu().numpy()
            y, missing = val.cpu().numpy(), mask.cpu().numpy() < 0.5
            by_subject = {}
            for subject in [1, 2]:
                rows = val_subjects == subject
                a, b, valid = p[rows], y[rows], missing[rows]
                delta = (a-b)[valid]
                correlations = []
                for j in range(len(a)):
                    for c in range(a.shape[2]):
                        v = valid[j, :, c]
                        if v.sum() >= 4 and np.std(a[j, v, c]) > 1e-8 and np.std(b[j, v, c]) > 1e-8:
                            correlations.append(float(np.corrcoef(a[j, v, c], b[j, v, c])[0, 1]))
                by_subject[str(subject)] = dict(rmse=float(np.sqrt(np.mean(delta**2))),
                    mae=float(np.mean(np.abs(delta))), corr=float(np.mean(correlations)) if correlations else None,
                    valid_correlations=len(correlations), missing_values=int(valid.sum()))
            result[scenario] = {'subjects': by_subject,
                'rmse': float(np.mean([v['rmse'] for v in by_subject.values()])),
                'mae': float(np.mean([v['mae'] for v in by_subject.values()])),
                'corr': float(np.mean([v['corr'] for v in by_subject.values() if v['corr'] is not None]))}
            predictions[scenario] = p
        result['weighted_rmse'] = sum(w*result[s]['rmse'] for w, s in zip([.2, .4, .4], scenarios))
        result['weighted_corr'] = sum(w*result[s]['corr'] for w, s in zip([.2, .4, .4], scenarios))
        return result, predictions

    save()
    for epoch in range(1, 9):
        if time.monotonic() > deadline - 35:
            break
        order = np.random.default_rng(42 + epoch).permutation(len(train))
        generator = masks(4200 + epoch)
        losses = {n: [] for n in models}
        for m in models.values():
            m.train()
        criterion.set_epoch(epoch)
        for step, i in enumerate(range(0, len(order), 16)):
            y = train[order[i:i+16]]
            mask = generator.generate_mask(y)
            for name, model in models.items():
                # Identical dropout/auxiliary-loss RNG on the same batch and mask.
                set_seed(42000 + epoch*100 + step)
                pred = model(y*mask, raw_time_mask=mask, return_aux=True)
                loss, _ = criterion(pred, y, mask)
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                losses[name].append(float(loss))
            if time.monotonic() > deadline:
                report['status'] = 'time_limit_mid_epoch; use preceding complete epoch only'
                save()
                print(report['status'], flush=True)
                return
        metrics = {}
        for name, model in models.items():
            metrics[name], predictions = evaluate(model)
            np.savez_compressed(out / f'{name}_validation.npz', target=val.cpu().numpy(),
                subjects=val_subjects, **predictions,
                **{f'mask_{s}': m.cpu().numpy() for s, m in fixed_masks.items()})
        report['history'].append(dict(epoch=epoch, train_loss={n: float(np.mean(v)) for n, v in losses.items()},
                                      validation=metrics))
        save()
        print(f'epoch={epoch} elapsed={report["elapsed_seconds"]:.1f}s ' +
              ' '.join(f'{n}: RMSE={v["weighted_rmse"]:.5f} corr={v["weighted_corr"]:.4f}' for n, v in metrics.items()), flush=True)
    report['status'] = 'complete' if len(report['history']) == 8 else 'bounded_partial'
    save()
    print(f'results={out / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
