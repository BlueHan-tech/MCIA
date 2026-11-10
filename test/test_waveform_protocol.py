import numpy as np
import torch

from data.waveform import (WINDOW, fit_channel_stats, gesture_log_power,
                           partition_segments, prepare_recording, task_windows, window_coverage)
from models.completion.mask_generators import WaveformScenarioMix
from models.prediction.dsranet import DSRANet
from utils.loss_functions import WaveformImputationLoss
from models.completion.mcia_core import WaveformMCIA
from utils.waveform_adaptation import BalancedDonorSampler, donor_optimizer
from utils.waveform_delivery import prediction_starts, stitch_predictions
from utils.waveform_quality import QUANTILES, SHORT, detect_waveform_quality, short_features


def test_original_time_grid_and_partitions():
    reps = np.repeat([1, 2, 6], [1600, 1600, 1600])
    labels = np.zeros(len(reps), dtype=int)
    labels[527:1330] = 1
    labels[2100:3000] = 2
    labels[3727:4527] = 49
    segments = partition_segments(reps, splits=(0, 1))
    angle = task_windows(labels, reps, segments)
    gesture = task_windows(labels, reps, segments, gesture=True)
    assert set(angle.repetitions) == set(gesture.repetitions) == {1, 6}
    assert angle.starts[0] == 27 and gesture.starts[0] == 527
    assert np.all(reps[angle.samples()] == angle.repetitions[:, None])
    assert np.all(labels[gesture.samples()] == gesture.actions[:, None])
    assert np.all(angle.samples(10)[:, 0] == angle.starts)
    assert np.all(angle.samples(10)[:, -1] == angle.starts + 590)
    assert 49 in gesture.actions

    short_first = np.r_[np.ones(200), np.zeros(800), np.full(800, 2)]
    same_rep = np.ones(len(short_first), dtype=int)
    fallback = task_windows(short_first, same_rep, partition_segments(same_rep))
    assert fallback.starts[0] == 1000 and not (fallback.actions == 1).any()


def test_fixed_stats_and_spectral_model():
    torch.set_num_threads(2)
    raw = np.random.default_rng(3).normal(size=(2, WINDOW, 12)).astype(np.float32)
    raw[0, :, 3] = 7
    stats = fit_channel_stats([raw[0]])
    assert stats.std[3] == 0
    assert not stats.standardize(raw)[..., 3].any()
    ref = np.arange(1, 13, dtype=np.float32)
    x = gesture_log_power(raw, ref)
    assert x.shape == (2, 12, 65, 19)
    np.testing.assert_array_equal(gesture_log_power(np.zeros_like(raw), ref), np.full_like(x, -60))
    segment = np.r_[np.zeros(64), raw[0, :64, 2]]
    window = np.hamming(128)
    expected = 10 * np.log10(np.abs(np.fft.rfft(segment * window))**2 / (window @ window) / ref[2]**2 + 1e-6)
    np.testing.assert_allclose(x[0, 2, :, 0], expected, atol=5e-6)
    model = DSRANet()
    out = model(torch.from_numpy(x))
    assert out.shape == (2, 49)
    torch.nn.functional.cross_entropy(out, torch.tensor([0, 48])).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_mask_scenarios_and_waveform_loss():
    gen = WaveformScenarioMix(rng=np.random.default_rng(17))
    for scenario, dead, partial in (('s1', 0, 4), ('s2', 1, 4), ('s3', 2, 3)):
        mask = gen.generate_batch_masks(12, scenario=scenario).reshape(12, 12, 15, 40)
        assert torch.all(mask == mask[..., :1])
        patch = mask[..., 0]
        counts = (1 - patch).sum(dim=-1)
        assert torch.all((counts == 15).sum(dim=-1) == dead)
        assert torch.all(((counts > 0) & (counts < 15)).sum(dim=-1) == partial)
        assert set(counts.unique().tolist()) <= {0, 4, 6, 8, 15}
        assert torch.all(patch.diff(dim=-1).abs().sum(dim=-1) <= 2)
    keep = torch.ones((1, WINDOW, 12))
    keep[:, 200:280, 0] = 0
    pred = torch.full_like(keep, -2, requires_grad=True)
    target = torch.zeros_like(keep)
    loss, terms = WaveformImputationLoss()(pred, target, keep)
    torch.testing.assert_close(terms['charbonnier'], torch.tensor(2.0))
    torch.testing.assert_close(terms['seam'], torch.tensor(2.0))
    assert terms['stft'] > 0 and terms['rms'] > 0
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad[keep.bool()] == 0).all()


def test_centre_splicing_changes_only_frozen_missing_values():
    signal = np.full((755, 2), -7.0)
    missing = np.zeros_like(signal, dtype=bool)
    missing[:, 0] = True
    missing[350:354, 1] = True
    starts = prediction_starts(len(signal))
    preds = [np.full((WINDOW, 2), i, dtype=float) for i in range(len(starts))]
    output = stitch_predictions(signal, missing, starts, preds)
    assert output[0, 0] == 0 and output[-1, 0] == 2
    assert np.all(np.diff(output[345:355, 0]) > 0)
    assert np.all(output[350:354, 1] == 1)
    np.testing.assert_array_equal(output[~missing], signal[~missing])


def test_donor_budget_and_frozen_weights():
    sampler = BalancedDonorSampler([12, 30, 40], 8, 5)
    assert sampler.steps == 11
    indices = np.asarray(list(sampler))
    assert len(indices) == 88 and indices.max() < 82
    first = indices[indices < 12][:12]
    assert len(first) == len(set(first)) == 12
    model = WaveformMCIA(window_size=600, patch_size=40, embed_dim=24,
                         n_layers=3, n_heads=2, ffn_dim=48, num_domains=2)
    with torch.no_grad():
        model.domain_embed.weight[0].fill_(2)
    opt = donor_optimizer(model, 1e-4)
    prior = {key: value.clone() for key, value in model.state_dict().items()}
    values = torch.randn(2, 600, 12)
    out = model(values, domain_id=torch.ones(2, dtype=torch.long))
    out.square().mean().backward()
    opt.step()
    assert torch.equal(prior['domain_embed.weight'][0], model.domain_embed.weight[0])
    assert not torch.equal(prior['domain_embed.weight'][1], model.domain_embed.weight[1])
    for key, value in model.state_dict().items():
        if key.startswith('pred_head.') or '.adapter.' in key or key == 'domain_embed.weight':
            continue
        assert torch.equal(value, prior[key]), key


def test_short_window_features_and_frozen_train_thresholds():
    from scipy import signal

    class Loader:
        fs = 2000

        def __init__(self):
            self.reps = np.repeat([1, 6, 2], 3500)
            self.labels = np.tile(np.r_[np.zeros(800), np.ones(1800), np.zeros(900)], 3)
            self.raw = np.random.default_rng(35).normal(0, 0.03, (len(self.reps), 12)).astype(np.float32)
            self.raw[:, 1] = 0
            self.raw[4700:4830, 0] += 30

        def load_exercise(self, *args, **kwargs):
            return self.raw.copy(), self.labels, self.reps, None

    loader = Loader()
    record = prepare_recording(loader, 'db3', 1, 1, splits=(0, 1))
    quality = detect_waveform_quality(record)
    transition = window_coverage(len(record.raw), record.transitions.starts)
    union = transition | window_coverage(len(record.raw), record.gestures.starts)
    starts, parts, features = short_features(record, union, transition)
    start = starts[0]
    sos = signal.butter(2, 20, 'lowpass', fs=2000, output='sos')
    x = record.raw[:3500, 0].astype(float)
    low, _ = signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos) * x[0])
    y = record.filtered[start:start + SHORT, 0]
    steps = np.abs(np.diff(x, prepend=x[0]))
    expected = [np.std(low[start:start + SHORT]), steps[start:start + SHORT].max(),
                np.std(y), np.abs(y).max()]
    np.testing.assert_allclose(features[0, 0], expected, rtol=1e-6)
    for col, q in enumerate(QUANTILES):
        np.testing.assert_allclose(quality.thresholds[0, col], np.quantile(features[parts == 0, 0, col], q))
    assert quality.hard_zero[1] and quality.missing[:, 1].all()
    assert quality.flagged[parts == 1, 0].any()
    assert not quality.missing[~union, 0].any()
    loader.raw[7000:, 0] *= 100
    evaluation = prepare_recording(loader, 'db3', 1, 1)
    changed = detect_waveform_quality(evaluation, frozen=(quality.hard_zero, quality.thresholds, quality.training_counts))
    np.testing.assert_array_equal(changed.thresholds, quality.thresholds)
    np.testing.assert_array_equal(changed.missing[:7000], quality.missing[:7000])


def test_reject_legacy_and_wrong_parent_checkpoints():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from utils.paper_pipeline import load_yaml_config
    from utils.waveform_pipeline import load_checkpoint, model_settings
    from data.waveform import WAVEFORM_FORMAT

    root = Path.cwd()
    cfg = load_yaml_config(root)
    with TemporaryDirectory(dir=root / 'outputs') as temp:
        path = Path(temp) / 'checkpoint.pt'
        for payload in ({'model': {}}, {'format': WAVEFORM_FORMAT, 'smoke': True, 'donors': [4],
                                       'settings': model_settings(cfg), 'parent_sha256': 'other'}):
            torch.save(payload, path)
            try:
                load_checkpoint(path, cfg, 'cpu', smoke=True, donors=(4,), parent='expected')
            except ValueError as exc:
                assert 'Incompatible waveform checkpoint' in str(exc)
            else:
                raise AssertionError('A legacy or unrelated checkpoint was accepted')
