"""
单 Batch 过拟合自检（Sanity Overfit Test）
============================================

锁定 1 个 **固定** batch，关闭 dropout，反复训练 ~200 步，
验证模型能否在这批数据上做到极低 MSE（近乎记住）。

为什么这个测试是黄金法则？
  - 若数据流 / mask 逻辑 / 位置编码 / 梯度回传 任何一处底层 bug，
    模型连一个 batch 都吃不下；
  - 通过之后，再跑全量才有意义。

判定标准（默认全部 PASS 视作健康）：
  - 总体 MSE         < 1e-4
  - 缺失区 MSE       < 1e-3
  - 已知区 MSE       < 1e-4
  - final / init     < 1e-3

输出：
  outputs/sanity_overfit/loss_curve.png
  outputs/sanity_overfit/sample0_after_overfit.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.completion.mcia_core import MCIA, derive_ch_mask_from_sample_mask
from models.completion.mask_generators import GroupWiseMaskGenerator
from utils.visualization import plot_completion_panel


# ---------------------- 数据 / mask 工厂 ----------------------

def make_fixed_batch(B=8, T=256, C=12, seed=42, device='cpu'):
    """构造确定性 batch：周期性正弦叠加随机噪声，形态仿 EMG。"""
    g = torch.Generator(device='cpu').manual_seed(seed)
    t = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1)          # (1,T,1)
    freqs = torch.rand(B, 1, C, generator=g) * 10 + 5
    phases = torch.rand(B, 1, C, generator=g) * 2 * 3.14159
    amps = torch.rand(B, 1, C, generator=g) * 0.5 + 0.3
    sig = amps * torch.sin(2 * 3.14159 * freqs * t + phases)
    noise = torch.randn(B, T, C, generator=g) * 0.1
    return (sig + noise).to(device)


def build_fixed_mask(B, T, C, device, difficulty=0.6, seed=123):
    """构造覆盖多场景的确定性 mask（含整通道缺失 / 多段缺失 / 真实 mask 生成器）。"""
    rng = np.random.default_rng(seed)
    mask = torch.ones(B, T, C, device=device)

    if B >= 1:
        mask[0, 80:170, 3] = 0.0                           # 中段大洞
    if B >= 2:
        mask[1, :, 7] = 0.0                                # 整通道缺失
    if B >= 3:
        mask[2, :, 4] = 0.0                                # 多条缺失
        mask[2, 100:200, 10] = 0.0
    if B >= 4:
        for s in range(20, 220, 40):                       # 多段随机
            mask[3, s:s + 20, int(rng.integers(12))] = 0.0

    if B > 4:
        mg = GroupWiseMaskGenerator(n_channels=C, time_steps=T)
        rest = mg.generate_batch_masks(
            B - 4, n_channels=C, time_steps=T,
            device=device, difficulty_level=difficulty,
        )
        rest = (rest.transpose(1, 2) > 0.5).float()
        mask[4:] = rest
    return mask


# ---------------------- 主流程 ----------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg_path = PROJECT_ROOT / 'config.yaml'
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    exp = cfg['exp1_mcia']
    T = cfg['signal']['window_size']
    C = cfg['signal']['n_channels']

    # 允许命令行覆盖 steps / lr
    import argparse
    parser = argparse.ArgumentParser()
    # 健康模型约在 step ~600 后 loss 跌破 1e-3，~1500 后跌破 1e-4；
    # 2000 步保证收敛完备、PASS 稳定。
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--clip', type=float, default=0.0,
                        help='grad-norm clip; 0 = off')
    args, _ = parser.parse_known_args()

    B, STEPS, LR, CLIP = args.batch, args.steps, args.lr, args.clip

    print(f'Device: {device}  | Batch: {B}x{T}x{C}  | steps={STEPS}  lr={LR}  clip={CLIP}')

    emg = make_fixed_batch(B=B, T=T, C=C, device=device)
    raw_mask = build_fixed_mask(B, T, C, device=device)
    ch_mask = derive_ch_mask_from_sample_mask(raw_mask)                 # (B,C) 1=仍有可见时刻
    emg_masked = emg * raw_mask

    total_miss_ratio = float((raw_mask < 0.5).float().mean())
    whole_ch_events = int((ch_mask < 0.5).sum().item())
    print(f'Total mask ratio: {total_miss_ratio*100:.2f}%  | '
          f'whole-channel-all-missing events (ch_mask==0): {whole_ch_events}')

    model = MCIA(
        window_size=T, n_channels=C,
        patch_size=exp['patch_size'], embed_dim=exp['embed_dim'],
        n_layers=exp['n_layers'],
        n_heads=exp['n_heads'],
        ffn_dim=exp['ffn_dim'],
        dropout=0.0,                                                   # 关 dropout → 允许记忆
    ).to(device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    history = []
    print('-' * 70)
    for step in range(STEPS + 1):
        model.train()
        pred = model(
            emg_masked, mask=ch_mask, x_masked=emg_masked,
            drop_condition=False, raw_time_mask=raw_mask,
        )
        loss_full = F.mse_loss(pred, emg)
        diff2 = (pred - emg) ** 2
        miss_sel = (raw_mask < 0.5).float()
        known_sel = 1.0 - miss_sel
        loss_mask = (diff2 * miss_sel).sum() / (miss_sel.sum() + 1e-8)
        loss_known = (diff2 * known_sel).sum() / (known_sel.sum() + 1e-8)

        if step > 0:
            optimizer.zero_grad()
            loss_full.backward()
            if CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

        # 诊断：检查预测头范数与掩码区拟合情况。
        with torch.no_grad():
            head_w = model.pred_head.weight.detach()
            if hasattr(model.film, 'to_gamma'):
                gamma_w = model.film.to_gamma.weight.detach()
                beta_w = model.film.to_beta.weight.detach()
            else:
                beta_w = head_w.new_zeros(())
                gamma_w = model.film.to_h.weight.detach() if hasattr(model.film, 'to_h') else head_w.new_zeros(())

        history.append(dict(
            step=step,
            full=loss_full.item(),
            mask=loss_mask.item(),
            known=loss_known.item(),
            gamma_norm=float(gamma_w.norm().item()),
            beta_norm=float(beta_w.norm().item()),
            head_norm=float(head_w.norm().item()),
        ))
        if step == 0 or (step + 1) % 25 == 0 or step == STEPS:
            print(f'step {step:4d}  full={loss_full.item():.6f}  '
                  f'masked={loss_mask.item():.6f}  known={loss_known.item():.6f}  '
                  f'|γ|={gamma_w.norm().item():.3f}  |β|={beta_w.norm().item():.3f}  '
                  f'|head|={head_w.norm().item():.3f}')

    init, final = history[0], history[-1]
    print('=' * 70)
    print(f'Init  full={init["full"]:.6f}  masked={init["mask"]:.6f}  known={init["known"]:.6f}')
    print(f'Final full={final["full"]:.6f}  masked={final["mask"]:.6f}  known={final["known"]:.6f}')
    ratio = final['full'] / max(init['full'], 1e-12)
    print(f'Final / Init  = {ratio:.2e}')

    criteria = {
        'full  < 1e-4':   final['full']  < 1e-4,
        'mask  < 1e-3':   final['mask']  < 1e-3,
        'known < 1e-4':   final['known'] < 1e-4,
        'ratio < 1e-3':   ratio          < 1e-3,
    }
    for k, v in criteria.items():
        print(f'  [{"PASS" if v else "FAIL"}] {k}')
    print('=' * 70)
    all_pass = all(criteria.values())
    print(f'>>> OVERFIT TEST: {"PASS" if all_pass else "FAIL"} <<<')

    # --------- 绘图 ---------
    out_dir = PROJECT_ROOT / 'outputs' / 'sanity_overfit'
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = [h['step'] for h in history]
    plt.figure(figsize=(8.5, 5))
    plt.semilogy(steps, [h['full'] for h in history],
                 label='loss_full', color='#1f77b4', lw=2.0)
    plt.semilogy(steps, [h['mask'] for h in history],
                 label='loss_masked (missing)', color='#d62728', lw=1.5)
    plt.semilogy(steps, [h['known'] for h in history],
                 label='loss_known (observed)', color='#2ca02c', lw=1.5)
    plt.xlabel('optimizer step')
    plt.ylabel('MSE (log)')
    plt.title(f'Single-Batch Overfit  B={B}, T={T}, C={C}, steps={STEPS}, lr={LR}')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_fp = out_dir / 'loss_curve.png'
    plt.savefig(loss_fp, dpi=140)
    plt.close()
    print(f'Loss curve: {loss_fp}')

    # 过拟合后的单样本 12 通道对比（期望几乎完美重建）
    model.eval()
    with torch.no_grad():
        pred = model(emg_masked, mask=ch_mask, x_masked=emg_masked,
                     drop_condition=False, raw_time_mask=raw_mask)
        completed = pred * (1.0 - raw_mask) + emg * raw_mask

    panel_fp = out_dir / 'sample0_after_overfit.png'
    plot_completion_panel(
        emg[0].detach().cpu().numpy(),
        completed[0].detach().cpu().numpy(),
        raw_mask[0].detach().cpu().numpy(),
        save_path=panel_fp,
        title='Sample 0 reconstruction after single-batch overfit',
    )
    print(f'Panel: {panel_fp}')

    if not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
