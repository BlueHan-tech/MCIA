"""ScenarioMixMaskGenerator + train_epoch_mcia + build_report 的冒烟测试。

目标：用合成小数据走通新场景混合 pipeline，验证：
  1) train_epoch_mcia 跑通（含运行期契约断言）
  2) validate_epoch_mcia 跑通
  3) save_training_epoch_snapshot 生成 PNG
  4) build_report 生成 scenario_matrix.png / per_channel_corr.png / REPORT.md
  5) plot_completion_panel writes s1..s3 unified scenario smoke panels
"""
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.completion.mcia_core import MCIA, MCIA_Wrapper, derive_ch_mask_from_sample_mask
from models.completion.mask_generators import ScenarioMixMaskGenerator
from utils.loss_functions import EMGImputationLoss
from utils.evaluation import validate_epoch_mcia, evaluate_subject
from utils.visualization import save_training_epoch_snapshot, build_report, plot_completion_panel

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_mcia_db2", PROJECT_ROOT / "scripts" / "01_train_mcia_db2_healthy_prior.py")
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
train_epoch_mcia = _train_mod.train_epoch_mcia


def save_scenario_panel_smoke(model, dataloader, device, mask_gen, output_dir,
                              scenarios=('s1', 's2', 's3')):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    batch = next(iter(dataloader))
    emg_clean = batch['data'][:1].to(device)
    side = batch.get('side', None)
    age = batch.get('age', None)
    gender = batch.get('gender', None)
    side = side[:1].to(device) if side is not None else None
    age = age[:1].to(device) if age is not None else None
    gender = gender[:1].to(device) if gender is not None else None

    _, T, C = emg_clean.shape
    saved = []
    with torch.no_grad():
        for scenario in scenarios:
            mask_soft = mask_gen.generate_batch_masks(
                1, n_channels=C, time_steps=T, device=device, scenario=scenario,
            )
            mask = (mask_soft.transpose(1, 2) > 0.5).float()
            emg_masked = emg_clean * mask
            mask_1d = derive_ch_mask_from_sample_mask(mask)
            pred = model(
                emg_masked, mask=mask_1d, x_masked=emg_masked,
                drop_condition=False, side=side, age=age, gender=gender,
                raw_time_mask=mask,
            )
            completed = pred * (1 - mask) + emg_clean * mask
            save_path = output_dir / f'{scenario}.png'
            plot_completion_panel(
                emg_clean[0].detach().cpu().numpy(),
                completed[0].detach().cpu().numpy(),
                mask[0].detach().cpu().numpy(),
                save_path=save_path,
                title=f'Scenario {scenario.upper()} Smoke Panel',
                completed_label='MCIA Smoke',
                show_observed_line=False,
                show_mask_background=True,
            )
            saved.append(save_path)
    return saved


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)
    print(f"[smoke] device={device}")

    B_total, T, C = 32, 256, 12
    data = torch.randn(B_total, T, C) * 0.3
    side = torch.zeros(B_total, dtype=torch.long)
    age = torch.zeros(B_total, dtype=torch.float32)
    gender = torch.zeros(B_total, dtype=torch.long)

    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, data, side, age, gender):
            self.data, self.side, self.age, self.gender = data, side, age, gender
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            return {'data': self.data[i], 'side': self.side[i],
                    'age': self.age[i], 'gender': self.gender[i]}

    ds = WrappedDataset(data, side, age, gender)
    train_loader = DataLoader(ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(ds, batch_size=8, shuffle=False)

    mask_gen = ScenarioMixMaskGenerator(
        n_channels=C, time_steps=T, patch_size=8,
    )
    print(f"[smoke] mask_gen ready. weights={mask_gen.weights}")

    model = MCIA(
        window_size=T, n_channels=C, patch_size=8, embed_dim=64,
        n_layers=2, n_heads=4, ffn_dim=128, dropout=0.0,
    ).to(device)
    mcia_wrapper = MCIA_Wrapper(model)
    criterion = EMGImputationLoss(fft_sizes=[16, 32, 64]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("[smoke] running train_epoch_mcia with warmup (s1) ...")
    train_loss = train_epoch_mcia(
        model, train_loader, optimizer, device, mask_gen, criterion,
        scenario='s1', debug_assert_steps=5, patch_size=8,
        cfg_dropout_prob=0.15,
    )
    print(f"[smoke] warmup train_loss={train_loss:.4f}")

    print("[smoke] running train_epoch_mcia with scenario=None (MIX) ...")
    train_loss = train_epoch_mcia(
        model, train_loader, optimizer, device, mask_gen, criterion,
        scenario=None, debug_assert_steps=3, patch_size=8,
        cfg_dropout_prob=0.15,
    )
    print(f"[smoke] mix train_loss={train_loss:.4f}")

    # === 步骤 2：验证 (s1) ===
    val_loss = validate_epoch_mcia(
        model, val_loader, device, mask_gen, criterion, scenario='s1')
    print(f"[smoke] val_loss(s1)={val_loss:.4f}")

    # === 步骤 3：可视化产物 ===
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snap_path = tmp / 'snap.png'
        save_training_epoch_snapshot(
            model, val_loader, device, mask_gen,
            save_path=snap_path, scenario='s1',
            guidance_scale=0.0, epoch=1, val_loss=val_loss,
        )
        assert snap_path.exists(), "snapshot not saved"
        print(f"[smoke] snapshot OK: {snap_path.stat().st_size} bytes")

        # 带多场景的 build_report
        report_dir = tmp / 'report'
        rp = build_report(
            model=model, dataloader=val_loader, device=device, mask_gen=mask_gen,
            output_dir=report_dir, num_samples=8, worst_k=3, stft_n_fft=64,
            scenarios=['s1', 's2', 's3'], main_scenario='s1',
            guidance_scale=0.0, title='Smoke Report',
        )
        assert rp.exists()
        files = sorted(p.name for p in report_dir.iterdir())
        print(f"[smoke] report files: {files}")
        must_have = {'REPORT.md', 'metrics.json', 'metric_bars.png',
                     'scenario_matrix.png', 'per_channel_corr.png',
                     'error_heatmap_worst.png', 'spectrogram_worst.png',
                     'panels', 'failures', 'whole_channel'}
        missing = must_have - set(files)
        assert not missing, f"report missing: {missing}"
        print("[smoke] report contents OK")

        # Unified scenario panel smoke
        scenario_dir = tmp / 'scenario_panels'
        scenario_paths = save_scenario_panel_smoke(
            model, val_loader, device, mask_gen, scenario_dir,
            scenarios=('s1', 's2', 's3'),
        )
        scenario_imgs = sorted(p.name for p in scenario_dir.iterdir())
        print(f"[smoke] scenario panel files: {scenario_imgs}")
        assert {p.name for p in scenario_paths} == {'s1.png', 's2.png', 's3.png'}
        assert all(p.exists() for p in scenario_paths), "missing scenario panel smoke output"

    # === 步骤 4：evaluate_subject ===
    metrics = evaluate_subject(
        model, mcia_wrapper, val_loader, device, mask_gen,
        scenario='s1', guidance_scale=0.0,
    )
    print(f"[smoke] evaluate_subject(s1): MSE={metrics['mse']:.4f} "
          f"Corr={metrics['correlation']:.4f} corr_masked={metrics['corr_masked']:.4f}")

    print("\n[smoke] ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
