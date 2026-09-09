"""Bounded synthetic checks; no datasets, checkpoints or experiment runs."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from models.completion.mcia_core import MCIA, MCIA_Wrapper
from utils.paper_pipeline import complete_with_mask
from utils.evaluation import validate_epoch_mcia


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completion_paths():
    torch.manual_seed(42)
    x = torch.rand(2, 32, 12)
    mask = torch.ones_like(x)
    mask[:, :, 0] = 0
    mask[:, 8:16, 1] = 0
    model = MCIA(window_size=32, n_channels=12, patch_size=8,
                 embed_dim=16, n_layers=1, n_heads=4, ffn_dim=32, dropout=0)
    model.eval()
    with torch.no_grad():
        model.pred_head[0].weight.zero_()
        model.pred_head[0].bias.fill_(2.0)
    # Force overshoot through the real Softplus head, retaining its state schema.
    assert (model(x * mask, raw_time_mask=mask) > 1).all()
    clone = MCIA(window_size=32, n_channels=12, patch_size=8,
                 embed_dim=16, n_layers=1, n_heads=4, ffn_dim=32, dropout=0)
    clone.load_state_dict(model.state_dict(), strict=True)
    expected = torch.where(mask.bool(), x, torch.ones_like(x))
    result = complete_with_mask(model, x, mask)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    wrapper = MCIA_Wrapper(model)
    for guidance in (0.0, 2.0):
        result = wrapper.p_sample(model, x * mask, torch.zeros(2),
                                  torch.ones(2, 12), x * mask,
                                  raw_time_mask=mask, guidance_scale=guidance)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    channel_mask = torch.ones(2, 12)
    channel_mask[:, 0] = 0
    result = wrapper.p_sample(model, x, torch.zeros(2), channel_mask,
                              x * channel_mask[:, None, :])
    torch.testing.assert_close(result, torch.where(channel_mask[:, None, :].bool(), x,
                                                   torch.ones_like(x)), rtol=0, atol=0)

    angle = load_script('04_eval_db3_angle_raw_vs_augmented.py')
    gesture = load_script('05_eval_db3_gesture_raw_vs_augmented.py')
    np.testing.assert_array_equal(angle.apply_mcia(model, x.numpy(), mask.numpy(), 'cpu'), expected.numpy())
    values, _ = gesture.apply_mcia(model, x.numpy(), mask.numpy(), 'cpu', 2)
    np.testing.assert_array_equal(values, expected.numpy())

    class MaskGen:
        mask_ratio = 0.5
        def generate_batch_masks(self, *args, **kwargs):
            return mask.transpose(1, 2)

    loss = validate_epoch_mcia(model, [x], 'cpu', MaskGen(), None)
    assert abs(loss - float(((expected - x) ** 2).mean())) < 1e-7
    from utils.visualization import plot_completion_panel, _run_model_on_batch
    for guidance in (0.0, 2.0):
        result = _run_model_on_batch(model, x, mask, torch.device('cpu'), guidance_scale=guidance)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    output = ROOT / 'outputs' / 'matplotlib_diagnosis' / 'completion_output_range_smoke.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_completion_panel(x[0].numpy(), expected[0].numpy(), mask[0].numpy(),
                          output, title='Synthetic output range contract')
    assert output.stat().st_size > 0
    print(f'[OK] actual completion plot: {output}')
    print('[OK] real Softplus overshoot, strict state load, generation, wrapper, angle, gesture, validation')


if __name__ == '__main__':
    test_completion_paths()
