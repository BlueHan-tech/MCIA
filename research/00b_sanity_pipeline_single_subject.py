"""
单被试全链路贯通测试（Pipeline Sanity Test）
==============================================

在 1 个健康被试上跑 ~12 个 epoch，端到端体检：

  [1] 数据流    : NinaProDataLoader → EMGCompletionDataset → 课程 mask
  [2] 模型      : MCIA + raw_time_mask + CFG dropout
  [3] 损失      : EMGImputationLoss（仅缺失区域）
  [4] 可视化    : 训练期每 N 轮 save_training_epoch_snapshot
                 训练结束执行 build_report（失败样本画廊 / 频谱对比 / 难度矩阵 / metrics.json / REPORT.md）
  [5] 时间开销  : 每 epoch 耗时 + 总耗时，估算 40 人 × 100 epoch 的工期
  [6] Axial 破局 : 记录 local/head 权重范数，观察 epoch 5 附近是否"开闸"

产物：
  outputs/sanity_pipeline/<timestamp>/
    ├─ epoch_log.json        # 每 epoch 的 loss / 时间 / Axial 范数
    ├─ epoch_log.png         # 时间与 loss 曲线
    ├─ model_norm_curve.png    # local/head 随 epoch 的变化
    ├─ viz/training/         # 每 2 epoch 一张 12 通道对比图
    └─ viz/report/           # build_report 最终体检报告
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader

from data.dataset_db2_emg import EMGCompletionDataset, load_db2_metadata
from models.completion.mcia_core import MCIA, derive_ch_mask_from_sample_mask
from models.completion.mask_generators import (
    GroupWiseMaskGenerator, AdaptiveCurriculumScheduler,
)
from utils.loss_functions import EMGImputationLoss
from utils.evaluation import validate_epoch_mcia
from utils.visualization import save_training_epoch_snapshot, build_report


# ---------------- 工具 ----------------

def _model_stats(model) -> dict:
    with torch.no_grad():
        local = model.local_bypass.pw.weight.detach()
        head = model.pred_head.weight.detach()
    return {
        'local_norm': float(local.norm().item()),
        'head_norm': float(head.norm().item()),
    }


def train_one_epoch(model, loader, optimizer, device, mask_gen, criterion,
                    difficulty, cfg_dropout):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        emg_clean = batch['data'].to(device)
        side = batch['side'].to(device)
        age = batch['age'].to(device)
        gender = batch['gender'].to(device)
        B, T, C = emg_clean.shape

        mask_soft = mask_gen.generate_batch_masks(
            B, n_channels=C, time_steps=T, device=device, difficulty_level=difficulty,
        )
        mask_soft = mask_soft.transpose(1, 2)
        mask_binary = (mask_soft > 0.5).float()
        emg_masked = emg_clean * mask_binary
        mask_1d = derive_ch_mask_from_sample_mask(mask_binary)
        drop = torch.rand(1).item() < cfg_dropout

        pred = model(
            emg_masked, mask=mask_1d, x_masked=emg_masked,
            drop_condition=drop, side=side, age=age, gender=gender,
            raw_time_mask=mask_binary,
        )
        if criterion is not None:
            loss, _ = criterion(pred, emg_clean, mask_binary)
        else:
            loss = torch.nn.functional.mse_loss(pred, emg_clean)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item() * B
        n += B
    return total / max(n, 1)


def main():
    # ---------- 参数 ----------
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', type=int, default=33,
                    help='使用哪个健康被试的缓存（DB2 S33 已缓存在 outputs/exp1_mcia_db2/）')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--viz-every', type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg_path = PROJECT_ROOT / 'config.yaml'
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    exp = cfg['exp1_mcia']
    T = cfg['signal']['window_size']
    C = cfg['signal']['n_channels']

    # ---------- 输出目录 ----------
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = PROJECT_ROOT / 'outputs' / 'sanity_pipeline' / f'S{args.subject:02d}_{ts}'
    (out_dir / 'viz' / 'training').mkdir(parents=True, exist_ok=True)

    # ---------- 数据 ----------
    cache_fp = PROJECT_ROOT / 'outputs' / 'exp1_mcia_db2' / f's{args.subject:02d}_cache.pt'
    if not cache_fp.exists():
        raise FileNotFoundError(
            f'No subject cache at {cache_fp}\n'
            f'  请先用 01_train_mcia_db2_healthy_prior.py 生成，或把该 subject id 加进 test_subjects 跑一次。'
        )
    print(f'[data] load cache: {cache_fp}')
    cache = torch.load(cache_fp, map_location='cpu', weights_only=False)
    segments = cache['data']
    subject_ids = cache['subject_ids']
    reps = cache['repetitions']
    print(f'[data] segments: {segments.shape}  reps: {np.unique(reps)}')

    # 按 NinaPro 通用切分：train reps 1,3,4,6 / val reps 2,5
    train_mask = np.isin(reps, [1, 3, 4, 6])
    val_mask = np.isin(reps, [2, 5])
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        # 兜底：80/20
        idx = np.arange(len(segments))
        np.random.shuffle(idx)
        split = int(len(idx) * 0.8)
        train_idx, val_idx = idx[:split], idx[split:]
        train_mask = np.zeros(len(segments), dtype=bool); train_mask[train_idx] = True
        val_mask = ~train_mask
    print(f'[data] train: {train_mask.sum()} | val: {val_mask.sum()}')

    md = load_db2_metadata(cfg['paths']['metadata_csv'])

    train_set = EMGCompletionDataset(
        segments[train_mask],
        subject_ids=subject_ids[train_mask],
        repetitions=reps[train_mask],
        metadata_dict=md,
    )
    val_set = EMGCompletionDataset(
        segments[val_mask],
        subject_ids=subject_ids[val_mask],
        repetitions=reps[val_mask],
        metadata_dict=md,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)

    # ---------- 模型 / mask gen / 课程 / loss ----------
    mask_gen = GroupWiseMaskGenerator(n_channels=C, time_steps=T)
    curriculum = AdaptiveCurriculumScheduler(
        total_epochs=args.epochs,
        warmup_epochs=max(1, int(args.epochs * 0.2)),
    )
    model = MCIA(
        window_size=T, n_channels=C,
        patch_size=exp['patch_size'], embed_dim=exp['embed_dim'],
        n_layers=exp['n_layers'],
        n_heads=exp['n_heads'],
        ffn_dim=exp['ffn_dim'],
        dropout=exp['model_dropout'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] params: {n_params:,}')

    criterion = EMGImputationLoss(
        w_charbonnier=exp.get('loss_charbonnier', 1.0),
        w_ncc=exp.get('loss_ncc', 0.5),
        w_stft=exp.get('loss_stft', 0.3),
        w_boundary=exp.get('loss_boundary', 0.1),
        w_aux=exp.get('loss_aux', 0.1),
        aux_ratio=exp.get('aux_mask_ratio', 0.10),
        fft_sizes=exp.get('loss_fft_sizes', [16, 32, 64]),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=exp['learning_rate'], weight_decay=1e-5)

    # ---------- 训练 ----------
    epoch_log = []
    total_start = time.time()
    print('=' * 80)
    print(f'Pipeline test: S{args.subject:02d} | epochs={args.epochs} | batch={args.batch} | device={device}')
    print('=' * 80)

    for ep in range(args.epochs):
        t0 = time.time()
        difficulty = curriculum.get_difficulty(ep)
        phase = curriculum.get_phase_name(ep)

        t_train_0 = time.time()
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, device, mask_gen, criterion,
            difficulty=difficulty, cfg_dropout=exp['cfg_dropout_prob'],
        )
        train_time = time.time() - t_train_0

        t_val_0 = time.time()
        val_loss = validate_epoch_mcia(model, val_loader, device, mask_gen, criterion, difficulty=1.0)
        val_time = time.time() - t_val_0

        fs = _model_stats(model)
        ep_time = time.time() - t0
        try:
            cuda_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device == 'cuda' else 0
        except Exception:
            cuda_mem = 0

        record = {
            'epoch': ep + 1, 'difficulty': float(difficulty), 'phase': phase,
            'train_loss': float(tr_loss), 'val_loss': float(val_loss),
            'train_sec': float(train_time), 'val_sec': float(val_time),
            'epoch_sec': float(ep_time),
            'cuda_peak_mb': float(cuda_mem),
            **fs,
        }
        epoch_log.append(record)

        print(f'Ep {ep+1:2d}/{args.epochs} | D={difficulty:.2f} {phase[:22]:22s} | '
              f'tr={tr_loss:.5f} val={val_loss:.5f} | '
              f'{ep_time:5.1f}s (tr {train_time:4.1f}+val {val_time:4.1f}) | '
              f'|local|={fs["local_norm"]:5.2f} |head|={fs["head_norm"]:5.2f} | '
              f'peak {cuda_mem:5.0f}MB')

        # 训练期快照
        if (ep + 1) % args.viz_every == 0 or ep == 0 or ep == args.epochs - 1:
            try:
                save_training_epoch_snapshot(
                    model, val_loader, device, mask_gen,
                    save_path=out_dir / 'viz' / 'training' / f'epoch_{ep+1:03d}.png',
                    difficulty=1.0,
                    guidance_scale=exp.get('guidance_scale', 0.0),
                    epoch=ep + 1, val_loss=val_loss,
                )
            except Exception as e:
                print(f'    [viz] snapshot skipped: {e}')

    total_sec = time.time() - total_start
    print('=' * 80)
    print(f'Total elapsed: {total_sec:.1f}s  ({total_sec/max(1,args.epochs):.1f}s/epoch avg)')

    # ---------- 输出日志 ----------
    with open(out_dir / 'epoch_log.json', 'w', encoding='utf-8') as f:
        json.dump({
            'subject': args.subject,
            'epochs': args.epochs, 'batch_size': args.batch,
            'n_params': int(n_params),
            'train_segments': int(train_mask.sum()),
            'val_segments': int(val_mask.sum()),
            'total_sec': float(total_sec),
            'records': epoch_log,
        }, f, indent=2, ensure_ascii=False)

    # 损失 + 时间曲线
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    eps_axis = [r['epoch'] for r in epoch_log]
    axes[0].plot(eps_axis, [r['train_loss'] for r in epoch_log], '-o', label='train', color='#1f77b4')
    axes[0].plot(eps_axis, [r['val_loss'] for r in epoch_log], '-s', label='val (D=1.0)', color='#d62728')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].set_title('Loss curve')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar(eps_axis, [r['train_sec'] for r in epoch_log],
                color='#1f77b4', alpha=0.7, label='train')
    axes[1].bar(eps_axis, [r['val_sec'] for r in epoch_log],
                bottom=[r['train_sec'] for r in epoch_log],
                color='#d62728', alpha=0.7, label='val')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('seconds')
    axes[1].set_title(f'Epoch time (mean={np.mean([r["epoch_sec"] for r in epoch_log]):.1f}s)')
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    plt.savefig(out_dir / 'epoch_log.png', dpi=140)
    plt.close(fig)

    # 轴向 / 预测头权重范数曲线
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(eps_axis, [r['local_norm'] for r in epoch_log], '-o', color='#d62728', lw=2,
            label='|local_bypass|')
    ax.plot(eps_axis, [r['head_norm'] for r in epoch_log], '-^', color='#1f77b4', lw=2,
            label='|pred_head|')
    ax.set_xlabel('epoch')
    ax.set_ylabel('weight norm')
    ax.set_title('MCIA norm growth')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_dir / 'model_norm_curve.png', dpi=140)
    plt.close(fig)

    # ---------- 报告 ----------
    print('Building report...')
    try:
        report_dir = out_dir / 'viz' / 'report'
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = build_report(
            model=model, dataloader=val_loader, device=device, mask_gen=mask_gen,
            output_dir=report_dir,
            num_samples=12,      # 单被试验证，样本数适度
            worst_k=6,
            stft_n_fft=128,
            difficulty_levels=[0.1, 0.3, 0.5, 0.7, 0.95],
            guidance_scale=exp.get('guidance_scale', 0.0),
            title=f'Pipeline sanity: S{args.subject:02d} ({args.epochs} epochs)',
        )
        print(f'Report: {report_path}')
    except Exception as e:
        print(f'[viz] Report failed: {e}')
        import traceback; traceback.print_exc()

    # ---------- 工期估算 ----------
    mean_ep = float(np.mean([r['epoch_sec'] for r in epoch_log]))
    n_train_per_epoch = int(train_mask.sum())
    # DB2 训练被试 28 人 × 约 1000 段/人 ≈ 28k 段；S33 是 ~1k 段
    ratio_full = 28000 / max(1, n_train_per_epoch)
    print('=' * 80)
    print('Compute estimate:')
    print(f'  This run         : {mean_ep:.1f}s/epoch × {args.epochs}ep = {mean_ep*args.epochs/60:.1f} min')
    print(f'  28 subjects, 100 ep, same batch ≈ {mean_ep*100*ratio_full/3600:.1f} hours')
    print(f'  28 subjects,  50 ep, same batch ≈ {mean_ep*50 *ratio_full/3600:.1f} hours')
    print(f'  (linear scaling w.r.t. train segments; DataLoader overhead ignored)')
    print('=' * 80)
    print(f'All outputs under: {out_dir}')


if __name__ == '__main__':
    main()
