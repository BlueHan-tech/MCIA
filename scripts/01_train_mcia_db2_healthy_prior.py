"""
【实验一】训练 MCIA 补全模型 (DB2 健康人 → DB2 健康人)

MCIA = Multi-Channel Imputation with Axial-attention

训练流程：
1. 预训练：ScenarioMix (S1/S2/S3) + 前 warmup_epochs（config，默认 8）弱化 S1 warmup
2. 测试：逐被试微调评估（Reps 1,3,4,6 校准 + Reps 2,5 测试）
3. 可视化：按场景出 panels / scenario_matrix / per_channel_corr

证明：模型可以还原正常人肌电
"""

import inspect
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# 若服务器 locale 是 C/POSIX，sys.stdout.encoding 会被设成 ascii，
# 导致 print 里的中文、希腊字母 γ/β 被替换成 "?"。强制切到 UTF-8，
# 让日志里真实字符被正确写出（终端是否能渲染取决于 SSH 字体）。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass  # 老版本 Python 或已是 UTF-8，忽略

# 必须在任何触发 matplotlib.pyplot 加载的 import 之前切到非交互后端，
# 否则服务器（无 X server / 无 GUI）下 matplotlib 会默认 Qt，报
# `QXcbConnection: Failed to initialize XRandr` 并导致图件保存失败。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# 纯净 Baseline 的 seed（严格确定性，用于消融对照锚点）
BASELINE_SEED = 42


def set_seed(seed: int):
    """固定所有 RNG 来源，保证 baseline 比特级完全一致可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 将项目根目录加入 sys.path
from data.ninapro_loader import NinaProDataLoader
from data.dataset_db2_emg import EMGCompletionDataset, load_and_cache_data, load_db2_metadata
from models.completion.mcia_core import (
    MCIA,
    MCIA_Wrapper,
    derive_ch_mask_from_sample_mask,
    derive_patch_time_mask,
)
from models.completion.mask_generators import (
    GroupWiseMaskGenerator, AdaptiveCurriculumScheduler,
    ScenarioMixMaskGenerator,
)
from utils.loss_functions import EMGImputationLoss
from utils.evaluation import (
    validate_epoch_mcia, validate_epoch_mcia_masked, evaluate_subject,
    save_training_history, print_finetuning_summary,
    probe_film_norms,
)
from utils.visualization import (
    save_training_epoch_snapshot,
    build_report,
    plot_baseline_comparison,
)
from utils.baselines import evaluate_cubic_spline
from utils.timemae_baseline import train_timemae_baseline, evaluate_timemae
from utils.run_layout import apply_run_paths


def load_config():
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    flat = {
        'db2_path': cfg['paths']['db2'],
        'db3_path': cfg['paths']['db3'],
        'output_dir': cfg['paths']['output'],
        'checkpoints': cfg['paths']['checkpoints'],
        'checkpoints_dir': cfg['paths']['checkpoints'],
        'metadata_csv': cfg['paths']['metadata_csv'],
        'orig_fs': cfg['signal']['orig_fs'],
        'target_fs': cfg['signal']['target_fs'],
        'window_size': cfg['signal']['window_size'],
        'stride': cfg['signal']['stride'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    flat.update(cfg['exp1_mcia'])
    flat['viz'] = cfg.get('viz', {'enable': True, 'every_epochs': 5,
                                   'num_report_samples': 20,
                                   'worst_k': 8, 'stft_n_fft': 128})
    return apply_run_paths(flat, cfg, PROJECT_ROOT)


def _assert_mask_contract(mask_binary_btc: torch.Tensor, patch_size: int, tag: str = '') -> None:
    """训练前若干 step 的运行期契约断言。"""
    assert torch.all((mask_binary_btc == 0) | (mask_binary_btc == 1)), \
        f"{tag} mask not strictly 0/1"
    B, T, C = mask_binary_btc.shape
    # patch 对齐：每个 patch 内要么全 0 要么全 1
    assert T % patch_size == 0, f"{tag} T={T} not divisible by patch_size={patch_size}"
    N = T // patch_size
    grouped = mask_binary_btc.permute(0, 2, 1).reshape(B, C, N, patch_size)
    lo = grouped.amin(dim=-1)
    hi = grouped.amax(dim=-1)
    mixed = int(((lo < 0.5) & (hi > 0.5)).sum().item())
    assert mixed == 0, f"{tag} {mixed} mixed patches detected (patch-align violated)"
    # 最小池化零信息损失：原始掩码与 patch 级 mask_ratio 必须严格一致
    raw_ratio = 1.0 - float(mask_binary_btc.mean().item())
    patch_mask = derive_patch_time_mask(mask_binary_btc, patch_size)  # (B,C,N)
    patch_ratio = 1.0 - float(patch_mask.mean().item())
    assert abs(raw_ratio - patch_ratio) < 1e-6, \
        f"{tag} raw_ratio={raw_ratio:.6f} != patch_ratio={patch_ratio:.6f}"


def train_epoch_mcia(model, dataloader, optimizer, device, mask_gen, criterion,
                    difficulty=0.5, cfg_dropout_prob=0.1, scenario=None,
                    debug_assert_steps: int = 0, patch_size: int = 8,
                    use_personal_condition: bool = False):
    """MCIA 训练一个 epoch

    Args:
        scenario: 强制场景（ScenarioMix 用；None 表示按权重随机 per sample）
        difficulty: 遗留课程学习保留入口
        debug_assert_steps: 训练前 N step 跑运行期契约断言
    """
    model.train()
    total_loss = 0.0
    loss_parts_sum = {}
    step = 0

    for batch in dataloader:
        if isinstance(batch, dict):
            emg_clean = batch['data'].to(device)
            side = batch.get('side', None)
            age = batch.get('age', None)
            gender = batch.get('gender', None)
            if side is not None: side = side.to(device)
            if age is not None: age = age.to(device)
            if gender is not None: gender = gender.to(device)
        else:
            emg_clean = batch.to(device)
            side = age = gender = None
        if not use_personal_condition:
            side = age = gender = None
        B, T, C = emg_clean.shape

        if scenario is not None and hasattr(mask_gen, '_dispatch'):
            mask_soft = mask_gen.generate_batch_masks(
                B, n_channels=C, time_steps=T, device=device, scenario=scenario,
            )
        elif hasattr(mask_gen, '_dispatch'):
            mask_soft = mask_gen.generate_batch_masks(
                B, n_channels=C, time_steps=T, device=device,
            )
        else:
            mask_soft = mask_gen.generate_batch_masks(
                batch_size=B, n_channels=C, time_steps=T,
                device=device, difficulty_level=difficulty,
            )
        mask_soft = mask_soft.transpose(1, 2)          # (B,T,C)
        mask_binary = (mask_soft > 0.5).float()        # (B,T,C) 严格 0/1

        if step < debug_assert_steps:
            _assert_mask_contract(mask_binary, patch_size, tag=f"[step{step}]")

        emg_masked = emg_clean * mask_binary           # 波形级硬零
        mask_1d = derive_ch_mask_from_sample_mask(mask_binary)  # (B,C) 1=通道仍有可见时刻

        drop_condition = torch.rand(1).item() < cfg_dropout_prob

        emg_pred = model(
            emg_masked, mask=mask_1d, x_masked=emg_masked,
            drop_condition=drop_condition,
            side=side, age=age, gender=gender,
            raw_time_mask=mask_binary,
        )

        if criterion is not None:
            loss, loss_parts = criterion(emg_pred, emg_clean, mask_binary)
            for key, value in loss_parts.items():
                loss_parts_sum[key] = loss_parts_sum.get(key, 0.0) + float(value)
        else:
            loss = torch.nn.functional.mse_loss(emg_pred, emg_clean)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

    train_epoch_mcia.last_loss_parts = {
        key: value / max(step, 1) for key, value in loss_parts_sum.items()
    }
    return total_loss / len(dataloader)


def _make_baseline_fns(timemae_model, device):
    """构造 baseline_fns dict，传给 build_report 的 baseline_fns 参数。

    每个 fn 签名：(emg_masked_np: B×T×C, mask_np: B×T×C) -> completed_np: B×T×C
    mask_np: 1=已知，0=缺失
    """
    import numpy as np
    from scipy.interpolate import CubicSpline
    from utils.baselines import _cubic_spline_fill

    fns = {}

    # ── 三次样条 ──
    def cs_fn(emg_masked_np, mask_np):
        B, T, C = emg_masked_np.shape
        # 从 emg_masked_np 和 mask 还原原始观测值（mask=1 的位置有真实信号）
        out = np.zeros_like(emg_masked_np)
        for b in range(B):
            for c in range(C):
                out[b, :, c] = _cubic_spline_fill(
                    emg_masked_np[b, :, c], mask_np[b, :, c]
                )
        return out

    fns['Cubic Spline'] = cs_fn

    # ── TimeMAE ──
    if timemae_model is not None:
        import torch
        tm = timemae_model

        def timemae_fn(emg_masked_np, mask_np):
            x = torch.tensor(emg_masked_np, dtype=torch.float32, device=device)
            m = torch.tensor(mask_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = tm.forward_completion(x, m)
            return out.cpu().numpy()

        fns['TimeMAE'] = timemae_fn

    return fns if fns else None


def main():
    CONFIG = load_config()
    device = CONFIG['device']

    # 固定全部随机源：训练前第一件事
    set_seed(BASELINE_SEED)
    print(f"[repro] seed={BASELINE_SEED} | cudnn.deterministic=True | cudnn.benchmark=False")

    output_dir = Path(CONFIG['exp1_dir'])
    checkpoint_dir = output_dir / 'checkpoints'
    metrics_dir = output_dir / 'metrics'
    figures_dir = output_dir / 'figures'
    cache_dir = output_dir / 'cache'
    for path in (output_dir, checkpoint_dir, metrics_dir, figures_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    run_label = Path(CONFIG.get('run_dir', output_dir)).name

    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    data_loader = NinaProDataLoader(CONFIG['db2_path'], CONFIG['db3_path'], fs=CONFIG['orig_fs'])

    train_subjects = CONFIG['train_subjects']
    val_subjects = CONFIG['val_subjects']
    test_subjects = CONFIG['test_subjects']

    print("="*80)
    print(f"Train: {len(train_subjects)} subjects | Val: {len(val_subjects)} | Test: {len(test_subjects)}")
    print("="*80)

    metadata_dict = load_db2_metadata(CONFIG['metadata_csv'])

    train_data = load_and_cache_data(data_loader, train_subjects, CONFIG, cache_file=cache_dir / 'train_cache.pt')
    val_data = load_and_cache_data(data_loader, val_subjects, CONFIG, cache_file=cache_dir / 'val_cache.pt')

    test_data_dict_all = {}
    print("\n[Loading test data]...")
    for sid in test_subjects:
        test_data_dict_all[sid] = load_and_cache_data(data_loader, [sid], CONFIG, cache_file=cache_dir / f's{sid:02d}_cache.pt')

    train_set = EMGCompletionDataset(train_data['data'], subject_ids=train_data['subject_ids'],
                                      repetitions=train_data['repetitions'], metadata_dict=metadata_dict)
    val_set = EMGCompletionDataset(val_data['data'], subject_ids=val_data['subject_ids'],
                                    repetitions=val_data['repetitions'], metadata_dict=metadata_dict)

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    # ===== 掩码策略：ScenarioMix（默认）或遗留课程学习 =====
    mask_strategy = CONFIG.get('mask_strategy', 'scenario_mix')
    val_scenario = CONFIG.get('val_scenario', 's1')
    debug_mask_contract = bool(CONFIG.get('debug_mask_contract', True))

    if mask_strategy == 'scenario_mix':
        mask_gen = ScenarioMixMaskGenerator(
            n_channels=12,
            time_steps=CONFIG['window_size'],
            patch_size=CONFIG['patch_size'],
            group_indices=CONFIG.get('group_indices', None),
            min_alive_per_group=CONFIG.get('min_alive_per_group', None),
            scenario_weights=CONFIG.get('scenario_weights', None),
            scenario_params=CONFIG.get('scenario_params', None),
        )
        curriculum = None
        print(f"[mask] strategy=scenario_mix fixed_policy "
              f"val_scenario={val_scenario} debug_contract={debug_mask_contract}")
        print(f"[mask] weights={mask_gen.weights}")
        print(f"[mask] min_alive={mask_gen.min_alive}")
    else:
        mask_gen = GroupWiseMaskGenerator(n_channels=12, time_steps=CONFIG['window_size'])
        curriculum = AdaptiveCurriculumScheduler(
            total_epochs=CONFIG['num_epochs'],
            warmup_epochs=int(CONFIG['num_epochs'] * 0.2),
        )
        print(f"[mask] strategy=legacy_curriculum (GroupWiseMaskGenerator + Adaptive)")

    model = MCIA(
        window_size=CONFIG['window_size'],
        n_channels=CONFIG.get('n_channels', 12),
        patch_size=CONFIG['patch_size'],
        embed_dim=CONFIG['embed_dim'],
        n_layers=CONFIG['n_layers'],
        n_heads=CONFIG['n_heads'],
        ffn_dim=CONFIG['ffn_dim'],
        dropout=CONFIG['model_dropout'],
        num_domains=CONFIG.get("num_domains", 2),
        use_synergy_bottleneck=CONFIG.get("use_synergy_bottleneck", False),
        n_synergies=CONFIG.get("n_synergies", 6),
        synergy_gate_scale=CONFIG.get("synergy_gate_scale", 0.5),
        synergy_dropout=CONFIG.get("synergy_dropout", 0.1),
    ).to(device)
    mcia_wrapper = MCIA_Wrapper(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # 保存 epoch-0「出厂」检查点 —— 后续消融实验从此出发，隔绝初始化差异
    init_norms = probe_film_norms(model)
    torch.save({
        'model': model.state_dict(),
        'seed': BASELINE_SEED,
        'init_norms': init_norms,
        'config_snapshot': {k: v for k, v in CONFIG.items() if k != 'viz'},
    }, checkpoint_dir / 'init_model.pth')
    print(
        "[repro] init_model.pth saved | "
        f"|local_bypass|={init_norms.get('local_bypass_norm', 0):.6f} "
        f"|head|={init_norms['head_norm']:.6f}"
    )

    if CONFIG['use_structural_loss']:
        criterion = EMGImputationLoss(
            w_charbonnier=CONFIG.get('loss_charbonnier', 1.0),
            w_ncc=CONFIG.get('loss_ncc', 0.5),
            w_stft=CONFIG.get('loss_stft', 0.3),
            w_boundary=CONFIG.get('loss_boundary', 0.1),
            w_aux=CONFIG.get('loss_aux', 0.1),
            aux_ratio=CONFIG.get('aux_mask_ratio', 0.10),
            fft_sizes=CONFIG.get('loss_fft_sizes', [16, 32, 64]),
        ).to(device)
    else:
        criterion = None

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    # 与早停一致：监控 corr_masked（越大越好），避免 loss 与 corr 方向打架
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=CONFIG['lr_factor'],
        patience=CONFIG['lr_patience'],
        min_lr=float(CONFIG.get('min_lr', 1e-7)),
    )

    best_val_loss = float('inf')
    best_val_corr = -1.0   # Early stopping and checkpoint selection use overall masked correlation.
    no_improve = 0
    train_history, val_history = [], []
    val_masked_mse_history, val_masked_mae_history = [], []
    val_masked_corr_history, val_whole_mse_history = [], []
    val_whole_corr_history, val_mask_ratio_history = [], []
    val_criterion_loss_history = []
    local_bypass_history, head_history = [], []
    train_loss_char_history, train_loss_ncc_history, train_loss_stft_history = [], [], []
    val_corr_s1_history, val_corr_s2_history, val_corr_s3_history = [], [], []

    viz_cfg = CONFIG.get('viz', {})
    viz_enable = viz_cfg.get('enable', True)
    viz_every = int(viz_cfg.get('every_epochs', 5))
    log_every = int(CONFIG.get('log_every_epochs', 20))
    viz_dir = figures_dir / 'training'
    if viz_enable:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining ({mask_strategy} + Fixed Target-Domain Validation)...")
    print(f"[train] log_every_epochs={log_every}  viz_every_epochs={viz_every}")
    print("="*80)
    start_time = time.time()
    use_scenario_mix = (mask_strategy == 'scenario_mix')

    for epoch in range(CONFIG['num_epochs']):
        if use_scenario_mix:
            train_scenario = None
            phase_name = "fixed-scenario-mix"
            difficulty = float('nan')
        else:
            difficulty = curriculum.get_difficulty(epoch)
            phase_name = curriculum.get_phase_name(epoch)
            train_scenario = None

        train_loss = train_epoch_mcia(
            model, train_loader, optimizer, device, mask_gen,
            criterion, difficulty=difficulty,
            cfg_dropout_prob=CONFIG['cfg_dropout_prob'],
            scenario=train_scenario,
            debug_assert_steps=(5 if (debug_mask_contract and use_scenario_mix and epoch == 0) else 0),
            patch_size=CONFIG['patch_size'],
            use_personal_condition=CONFIG.get('use_personal_condition', False),
        )
        train_history.append(train_loss)
        train_loss_parts = getattr(train_epoch_mcia, 'last_loss_parts', {})
        train_loss_char_history.append(train_loss_parts.get('charbonnier_loss', float('nan')))
        train_loss_ncc_history.append(train_loss_parts.get('ncc_loss', float('nan')))
        train_loss_stft_history.append(train_loss_parts.get('stft_loss', float('nan')))

        if use_scenario_mix:
            val_metrics = validate_epoch_mcia_masked(
                model, val_loader, device, mask_gen, criterion,
                scenario=val_scenario,
                use_personal_condition=CONFIG.get('use_personal_condition', False),
            )
        else:
            val_metrics = validate_epoch_mcia_masked(
                model, val_loader, device, mask_gen, criterion,
                difficulty=1.0,
                use_personal_condition=CONFIG.get('use_personal_condition', False),
            )
        if use_scenario_mix:
            val_metrics_per_scenario = {val_scenario: val_metrics}
            for scn in ('s1', 's2', 's3'):
                if scn == val_scenario:
                    continue
                val_metrics_per_scenario[scn] = validate_epoch_mcia_masked(
                    model, val_loader, device, mask_gen, criterion,
                    scenario=scn,
                    use_personal_condition=CONFIG.get('use_personal_condition', False),
                )
            val_corr_s1_history.append(val_metrics_per_scenario['s1']['corr_masked'])
            val_corr_s2_history.append(val_metrics_per_scenario['s2']['corr_masked'])
            val_corr_s3_history.append(val_metrics_per_scenario['s3']['corr_masked'])
        val_loss = val_metrics['loss_for_early_stop']
        val_history.append(val_loss)
        val_masked_mse_history.append(val_metrics['mse_masked'])
        val_masked_mae_history.append(val_metrics['mae_masked'])
        val_masked_corr_history.append(val_metrics['corr_masked_partial'])
        val_whole_mse_history.append(val_metrics['mse_whole'])
        val_whole_corr_history.append(val_metrics['corr_whole'])
        val_mask_ratio_history.append(val_metrics['mask_ratio'])
        val_criterion_loss_history.append(val_metrics['criterion_loss'])

        # 模型范数探针，保留为唯一日志记录路径。
        norms = {
            'local_bypass_norm': val_metrics.get('local_bypass_norm', 0.0),
            'head_norm': val_metrics['head_norm'],
        }
        local_bypass_history.append(norms['local_bypass_norm'])
        head_history.append(norms['head_norm'])

        val_corr = val_metrics['corr_masked']
        if np.isnan(val_corr):
            val_corr = val_metrics['corr_masked_partial']
        if val_corr == val_corr:
            scheduler.step(val_corr)

        improved = ""
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_val_loss = val_loss   # 同步记录对应的 loss（供日志/历史用）
            no_improve = 0
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss, 'val_loss': val_loss,
                'val_metrics': val_metrics,
                'epoch': epoch,
                'train_scenario': train_scenario,
                'val_scenario': val_scenario if use_scenario_mix else None,
                'difficulty': difficulty,
                'seed': BASELINE_SEED,
                'model_norms': norms,
            }, checkpoint_dir / 'best_model.pth')
            improved = " *"
        else:
            no_improve += 1

        elapsed = time.time() - start_time
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == CONFIG['num_epochs'] - 1:
            scn_tag = (train_scenario.upper() if train_scenario else 'MIX') if use_scenario_mix \
                      else f'D={difficulty:.2f}'
            _norms = (
                f"|local|={norms['local_bypass_norm']:.3f} "
            )
            _loss_parts = (
                f"char={train_loss_parts.get('charbonnier_loss', float('nan')):.6f} "
                f"ncc={train_loss_parts.get('ncc_loss', float('nan')):.6f} "
                f"stft={train_loss_parts.get('stft_loss', float('nan')):.6f} "
                f"grad={train_loss_parts.get('boundary_grad_loss', float('nan')):.6f} "
                f"aux={train_loss_parts.get('aux_loss', float('nan')):.6f}"
            )
            print(f"Epoch {epoch+1:3d}/{CONFIG['num_epochs']} | {scn_tag:10s} | {phase_name[:25]:25s} | "
                  f"Train: {train_loss:.6f} | Val_loss: {val_loss:.6f} | "
                  f"Val_criterion: {val_metrics.get('criterion_loss', float('nan')):.6f} | "
                  f"Val_masked_mse: {val_metrics['mse_masked']:.6f} | "
                  f"Val_masked_mae: {val_metrics['mae_masked']:.6f} | "
                  f"Val_corr_partial: {val_metrics['corr_masked_partial']:.4f} | "
                  f"Val_corr_chmiss: {val_metrics['corr_masked_chmiss']:.4f} | "
                  f"Val_corr_all: {val_metrics['corr_masked']:.4f} | "
                  f"Val_whole_mse: {val_metrics['mse_whole']:.6f} | "
                  f"Val_whole_corr: {val_metrics['corr_whole']:.4f} | "
                  f"mask={val_metrics['mask_ratio']:.2%} | Best_corr_masked: {best_val_corr:.4f} | "
                  f"{_norms}|head|={norms['head_norm']:.3f} | {elapsed:.0f}s{improved}")
            print(f"    loss_parts | {_loss_parts}")
            if use_scenario_mix:
                print(f"    mask_weights | {mask_gen.weights}")
                corr_line = " ".join(
                    f"{scn.upper()}={val_metrics_per_scenario[scn]['corr_masked']:.4f}"
                    for scn in ('s1', 's2', 's3')
                )
                print(f"    val_corr_masked_by_scenario | {corr_line}")
                print(f"    val_by_scenario_detail | main={val_scenario}")
                for scn in ('s1', 's2', 's3'):
                    m = val_metrics_per_scenario[scn]
                    print(
                        f"      {scn}: mse_masked={m['mse_masked']:.6f} "
                        f"mse_whole={m['mse_whole']:.6f} "
                        f"corr_masked={m['corr_masked']:.4f} "
                        f"corr_partial={m['corr_masked_partial']:.4f} "
                        f"corr_chmiss={m['corr_masked_chmiss']:.4f} "
                        f"corr_whole={m['corr_whole']:.4f} "
                        f"mask={m['mask_ratio']:.2%}"
                    )

        # 可视化训练期钩子：每 viz_every 轮 + 第 1 轮 + 末轮，保存 12 通道对比图
        if viz_enable and (epoch == 0 or (epoch + 1) % viz_every == 0 or
                           epoch == CONFIG['num_epochs'] - 1):
            try:
                save_training_epoch_snapshot(
                    model, val_loader, device, mask_gen,
                    save_path=viz_dir / f'epoch_{epoch+1:03d}.png',
                    difficulty=1.0,
                    guidance_scale=CONFIG.get('guidance_scale', 0.0),
                    epoch=epoch + 1, val_loss=val_loss,
                    scenario=(val_scenario if use_scenario_mix else None),
                )
            except Exception as viz_err:
                print(f"    [viz] snapshot skipped: {viz_err}")

        if no_improve >= CONFIG['patience']:
            print(f"\nEarly stop: no improvement for {CONFIG['patience']} epochs")
            break

    print(f"\nTraining done in {(time.time()-start_time)/60:.1f} min | Best corr_masked: {best_val_corr:.4f}")

    save_training_history({
        'seed': BASELINE_SEED,
        'init_norms': init_norms,
        'train_loss': train_history, 'val_loss': val_history,
        'val_masked_mse': val_masked_mse_history,
        'val_masked_mae': val_masked_mae_history,
        'val_masked_corr': val_masked_corr_history,
        'val_whole_mse': val_whole_mse_history,
        'val_whole_corr': val_whole_corr_history,
        'val_mask_ratio': val_mask_ratio_history,
        'val_criterion_loss': val_criterion_loss_history,
        'local_bypass_norm': local_bypass_history,
        'head_norm': head_history,
        'train_loss_char': train_loss_char_history,
        'train_loss_ncc': train_loss_ncc_history,
        'train_loss_stft': train_loss_stft_history,
        'val_corr_s1': val_corr_s1_history,
        'val_corr_s2': val_corr_s2_history,
        'val_corr_s3': val_corr_s3_history,
        'best_val_loss': float(best_val_loss),
        'best_val_corr': float(best_val_corr),
        'best_val_corr_partial': float(best_val_corr),
        'total_epochs': len(train_history),
    }, metrics_dir / 'training_history.json')

    # 加载最佳模型
    checkpoint = torch.load(checkpoint_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])

    # ===== 逐被试微调与评估 =====
    print("\n" + "="*80)
    print("Per-subject evaluation (Test on Reps 2,5)")
    print("="*80)

    all_results = []
    run_db2_test_finetune = bool(CONFIG.get('run_db2_test_finetune', True))

    # ── TimeMAE Baseline 预训练（在训练集上，与主模型共享 train_loader）──────
    timemae_model = None
    if viz_enable:
        try:
            window_size = CONFIG['window_size']
            n_channels  = CONFIG.get('n_channels', 12)
            timemae_ckpt = checkpoint_dir / 'timemae_pretrain.pt'
            timemae_model = train_timemae_baseline(
                train_dataloader  = train_loader,
                device            = device,
                data_shape        = (window_size, n_channels),
                wave_length       = CONFIG.get('timemae_wave_length', 8),
                d_model           = CONFIG.get('timemae_d_model', 64),
                attn_heads        = CONFIG.get('timemae_attn_heads', 4),
                layers            = CONFIG.get('timemae_layers', 4),
                reg_layers        = CONFIG.get('timemae_reg_layers', 2),
                epochs            = CONFIG.get('timemae_epochs', 30),
                lr                = CONFIG.get('timemae_lr', 1e-3),
                save_path         = timemae_ckpt,
            )
            print(f"  [TimeMAE] Pre-training done → {timemae_ckpt}")
        except Exception as e:
            print(f"  [TimeMAE] Pre-training skipped: {e}")
            timemae_model = None

    for subject_id in test_subjects:
        print(f"\n--- S{subject_id:02d} ---")
        subject_data_dict = test_data_dict_all[subject_id]
        subject_data = subject_data_dict['data']
        subject_sids = subject_data_dict['subject_ids']
        subject_reps = subject_data_dict.get('repetitions', None)

        if subject_reps is not None:
            calib_mask = np.isin(subject_reps, [1, 3, 4, 6])
            test_mask = np.isin(subject_reps, [2, 5])
            calib_data = subject_data[calib_mask]
            calib_sids = subject_sids[calib_mask] if subject_sids is not None else None
            calib_reps = subject_reps[calib_mask]
            test_data_subj = subject_data[test_mask]
            test_sids_subj = subject_sids[test_mask] if subject_sids is not None else None
            test_reps_subj = subject_reps[test_mask]
        else:
            split_idx = int(len(subject_data) * 0.1)
            calib_data = subject_data[:split_idx]
            calib_sids = subject_sids[:split_idx] if subject_sids is not None else None
            calib_reps = None
            test_data_subj = subject_data[split_idx:]
            test_sids_subj = subject_sids[split_idx:] if subject_sids is not None else None
            test_reps_subj = None

        calib_set = EMGCompletionDataset(calib_data, subject_ids=calib_sids, repetitions=calib_reps, metadata_dict=metadata_dict)
        test_set_subj = EMGCompletionDataset(test_data_subj, subject_ids=test_sids_subj, repetitions=test_reps_subj, metadata_dict=metadata_dict)
        calib_loader = DataLoader(calib_set, batch_size=32, shuffle=True, num_workers=0)
        test_loader_subj = DataLoader(test_set_subj, batch_size=32, shuffle=False, num_workers=0)

        # 零样本（val_scenario 保持与训练 val 同场景，确保可比）
        model.eval()
        eval_kwargs = dict(
            difficulty=1.0,
            guidance_scale=CONFIG['guidance_scale'],
            use_personal_condition=CONFIG.get('use_personal_condition', False),
        )
        if use_scenario_mix:
            eval_kwargs['scenario'] = val_scenario
        zs_metrics = evaluate_subject(model, mcia_wrapper, test_loader_subj, device, mask_gen,
                                      **eval_kwargs)
        print(f"  Zero-shot: corr_masked={zs_metrics.get('corr_masked', float('nan')):.4f} "
              f"mse_masked={zs_metrics.get('mse_masked', float('nan')):.6f} "
              f"mae_masked={zs_metrics.get('mae_masked', float('nan')):.6f} "
              f"corr_partial={zs_metrics.get('corr_masked_partial', float('nan')):.4f}")

        # 零样本逐被试可视化报告（无论是否做微调，都生成）
        if viz_enable:
            try:
                zs_report_dir = figures_dir / 'test' / f'S{subject_id:02d}' / 'zeroshot'
                zs_report_dir.mkdir(parents=True, exist_ok=True)
                # 确保使用预训练权重（finetune 模式下下面会修改权重，这里先出图）
                model.load_state_dict(checkpoint['model'])
                model.eval()
                zs_report_kwargs = dict(
                    model=model, dataloader=test_loader_subj, device=device, mask_gen=mask_gen,
                    output_dir=zs_report_dir,
                    num_samples=int(viz_cfg.get('num_report_samples', 12)),
                    worst_k=int(viz_cfg.get('worst_k', 6)),
                    stft_n_fft=int(viz_cfg.get('stft_n_fft', 128)),
                    guidance_scale=CONFIG.get('guidance_scale', 0.0),
                    title=f'S{subject_id:02d} Zero-shot Report',
                    baseline_fns=_make_baseline_fns(timemae_model, device),
                )
                if use_scenario_mix:
                    zs_report_kwargs['scenarios'] = ['s1', 's2', 's3']
                    zs_report_kwargs['main_scenario'] = val_scenario
                else:
                    zs_report_kwargs['difficulty_levels'] = [0.1, 0.3, 0.5, 0.7, 0.95]
                build_report(**zs_report_kwargs)
                print(f"  [viz] S{subject_id:02d} zero-shot report → {zs_report_dir}")
            except Exception as e:
                print(f"  [viz] S{subject_id:02d} zero-shot report skipped: {e}")

        # 基线对比（与 MCIA 统一：corr_masked / mse_masked / mae_masked）
        cs_metrics = tm_metrics = None
        try:
            cs_scenario = val_scenario if use_scenario_mix else None
            cs_metrics = evaluate_cubic_spline(
                test_loader_subj, mask_gen, device,
                scenario=cs_scenario, difficulty=1.0,
            )
            print(f"  Cubic Spline: corr_masked={cs_metrics['corr_masked']:.4f} "
                  f"mse_masked={cs_metrics['mse_masked']:.6f} "
                  f"mae_masked={cs_metrics.get('mae_masked', float('nan')):.6f}")

            if timemae_model is not None:
                try:
                    tm_metrics = evaluate_timemae(
                        timemae_model, test_loader_subj, mask_gen, device,
                        scenario=cs_scenario, difficulty=1.0,
                    )
                    print(f"  TimeMAE:      corr_masked={tm_metrics['corr_masked']:.4f} "
                          f"mse_masked={tm_metrics['mse_masked']:.6f} "
                          f"mae_masked={tm_metrics.get('mae_masked', float('nan')):.6f}")
                except Exception as e:
                    print(f"  [TimeMAE] Eval skipped: {e}")

            if viz_enable:
                baseline_results = {'MCIA Model': zs_metrics, 'Cubic Spline': cs_metrics}
                if tm_metrics is not None:
                    baseline_results['TimeMAE'] = tm_metrics
                cmp_dir = figures_dir / 'test' / f'S{subject_id:02d}' / 'zeroshot'
                plot_baseline_comparison(
                    baseline_results, cmp_dir,
                    title=f'S{subject_id:02d} Baseline Comparison',
                )
                print(f"  [viz] S{subject_id:02d} baseline comparison → {cmp_dir}")
        except Exception as e:
            print(f"  [baseline] S{subject_id:02d} comparison skipped: {e}")

        if not run_db2_test_finetune:
            all_results.append({
                'subject_id': subject_id,
                'n_calibration': len(calib_data),
                'n_test': len(test_data_subj),
                'zeroshot': zs_metrics,
                'baselines': {
                    'Cubic Spline': cs_metrics,
                    'TimeMAE': tm_metrics,
                },
                'finetuned': None,
                'improvement': None,
            })
            continue

        # 微调：空间主干 + patch 前端共享低 LR（保留跨被试共性）；
        # 模型范数探针，保留为唯一日志记录路径。
        model.load_state_dict(checkpoint['model'])
        for param in model.parameters():
            param.requires_grad = True

        spatial_modules = [
            model.patch_embed, model.local_bypass, model.blocks,
        ]
        spatial_param_ids = set()
        for mod in spatial_modules:
            for p in mod.parameters():
                spatial_param_ids.add(id(p))
        # 位置编码 / mask tokens 视为先验通道；个人信息权重保留兼容但默认不参与主流程。
        extra_params = [model.chan_pos, model.temp_pos, model.mask_token, model.uncond_token]
        for extra in extra_params:
            spatial_param_ids.add(id(extra))
        for p in model.domain_embed.parameters():
            spatial_param_ids.add(id(p))

        spatial_params = [p for p in model.parameters() if id(p) in spatial_param_ids]
        refiner_params = [p for p in model.parameters() if id(p) not in spatial_param_ids]

        lr_spatial = CONFIG.get('finetune_lr_spatial', CONFIG.get('finetune_lr_encoder', 1e-5))
        lr_refiner = CONFIG.get('finetune_lr_refiner', CONFIG.get('finetune_lr_decoder', 2e-4))
        ft_optimizer = optim.AdamW([
            {'params': spatial_params, 'lr': lr_spatial},
            {'params': refiner_params, 'lr': lr_refiner},
        ], weight_decay=1e-5)

        ft_best_loss = float('inf')
        ft_no_improve = 0
        ft_scenario = val_scenario if use_scenario_mix else None  # 微调锁定与 val 同场景
        for ft_epoch in range(CONFIG['finetune_epochs']):
            loss = train_epoch_mcia(model, calib_loader, ft_optimizer, device, mask_gen, criterion,
                                   difficulty=1.0, cfg_dropout_prob=CONFIG['cfg_dropout_prob'],
                                   scenario=ft_scenario, patch_size=CONFIG['patch_size'],
                                   use_personal_condition=CONFIG.get('use_personal_condition', False))
            if loss < ft_best_loss:
                ft_best_loss = loss
                ft_no_improve = 0
                torch.save({'model': model.state_dict()}, checkpoint_dir / f'finetuned_S{subject_id:02d}.pth')
            else:
                ft_no_improve += 1
            if loss > ft_best_loss * 1.2 or ft_no_improve >= CONFIG['finetune_patience']:
                break

        ft_ckpt = torch.load(checkpoint_dir / f'finetuned_S{subject_id:02d}.pth', map_location=device)
        model.load_state_dict(ft_ckpt['model'])

        ft_metrics = evaluate_subject(model, mcia_wrapper, test_loader_subj, device, mask_gen,
                                      **eval_kwargs)
        print(f"  Finetuned: MSE={ft_metrics['mse']:.6f} Corr={ft_metrics['correlation']:.4f}")

        improvement = {
            'mse': (zs_metrics['mse'] - ft_metrics['mse']) / (zs_metrics['mse'] + 1e-8) * 100,
            'mae': (zs_metrics['mae'] - ft_metrics['mae']) / (zs_metrics['mae'] + 1e-8) * 100,
            'correlation': (ft_metrics['correlation'] - zs_metrics['correlation']) / (1.0 - zs_metrics['correlation'] + 1e-8) * 100
        }

        all_results.append({
            'subject_id': subject_id,
            'n_calibration': len(calib_data), 'n_test': len(test_data_subj),
            'zeroshot': zs_metrics, 'finetuned': ft_metrics, 'improvement': improvement
        })

        # 微调后逐被试可视化报告
        if viz_enable:
            try:
                ft_report_dir = figures_dir / 'test' / f'S{subject_id:02d}' / 'finetuned'
                ft_report_dir.mkdir(parents=True, exist_ok=True)
                ft_report_kwargs = dict(
                    model=model, dataloader=test_loader_subj, device=device, mask_gen=mask_gen,
                    output_dir=ft_report_dir,
                    num_samples=int(viz_cfg.get('num_report_samples', 12)),
                    worst_k=int(viz_cfg.get('worst_k', 6)),
                    stft_n_fft=int(viz_cfg.get('stft_n_fft', 128)),
                    guidance_scale=CONFIG.get('guidance_scale', 0.0),
                    title=f'S{subject_id:02d} Finetuned Report',
                    baseline_fns=_make_baseline_fns(timemae_model, device),
                )
                if use_scenario_mix:
                    ft_report_kwargs['scenarios'] = ['s1', 's2', 's3']
                    ft_report_kwargs['main_scenario'] = val_scenario
                else:
                    ft_report_kwargs['difficulty_levels'] = [0.1, 0.3, 0.5, 0.7, 0.95]
                build_report(**ft_report_kwargs)
                print(f"  [viz] S{subject_id:02d} finetuned report → {ft_report_dir}")
            except Exception as e:
                print(f"  [viz] S{subject_id:02d} finetuned report skipped: {e}")


    if run_db2_test_finetune:
        print_finetuning_summary(all_results)
    else:
        def _mean_metric(results, method_key, metric_key):
            vals = []
            for r in results:
                if method_key == 'MCIA':
                    m = r.get('zeroshot', {})
                else:
                    m = (r.get('baselines') or {}).get(method_key) or {}
                v = m.get(metric_key, float('nan'))
                if m and not np.isnan(v):
                    vals.append(v)
            return float(np.mean(vals)) if vals else float('nan')

        print("\nDB2 held-out zero-shot summary (aligned: masked-region metrics)")
        for method in ('MCIA', 'Cubic Spline', 'TimeMAE'):
            print(f"  {method:14s} corr_masked={_mean_metric(all_results, method, 'corr_masked'):.4f} "
                  f"mse_masked={_mean_metric(all_results, method, 'mse_masked'):.6f} "
                  f"mae_masked={_mean_metric(all_results, method, 'mae_masked'):.6f}")

    with open(metrics_dir / 'finetuning_results.json', 'w') as f:
        json.dump({'results': all_results}, f, indent=2)

    # ===== 测试集汇总（零样本模式专用）=====
    if viz_enable and not run_db2_test_finetune and len(all_results) > 0:
        try:
            import csv
            import matplotlib.pyplot as plt

            summary_dir = figures_dir / 'test' / 'summary'
            summary_dir.mkdir(parents=True, exist_ok=True)

            subject_ids  = [r['subject_id']            for r in all_results]
            mse_vals     = [r['zeroshot']['mse_masked']        for r in all_results]
            mae_vals     = [r['zeroshot']['mae_masked']        for r in all_results]
            corr_vals    = [r['zeroshot']['corr_masked'] for r in all_results]
            xlabels      = [f"S{s:02d}" for s in subject_ids]

            # 逐被试指标表 subject_metric_table.csv
            with open(summary_dir / 'subject_metric_table.csv', 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['subject_id', 'corr_masked', 'mse_masked', 'mae_masked'])
                writer.writeheader()
                for sid, mse, mae, corr in zip(subject_ids, mse_vals, mae_vals, corr_vals):
                    writer.writerow({
                        'subject_id': sid, 'corr_masked': corr,
                        'mse_masked': mse, 'mae_masked': mae,
                    })

            fig_w = max(8, len(subject_ids) * 0.9)

            # 逐被试 MSE 柱状图 subject_mse_bar.png
            fig, ax = plt.subplots(figsize=(fig_w, 4))
            bars = ax.bar(xlabels, mse_vals, color='steelblue')
            ax.bar_label(bars, fmt='%.4f', fontsize=7, padding=2)
            avg_mse = float(np.mean(mse_vals))
            ax.axhline(avg_mse, color='red', linestyle='--', linewidth=1, label=f'mean={avg_mse:.4f}')
            ax.set_title('Zero-shot MSE (masked) per Test Subject')
            ax.set_ylabel('mse_masked')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(summary_dir / 'subject_mse_bar.png', dpi=150)
            plt.close(fig)

            # 逐被试 Corr 柱状图 subject_corr_bar.png
            fig, ax = plt.subplots(figsize=(fig_w, 4))
            bars = ax.bar(xlabels, corr_vals, color='darkorange')
            ax.bar_label(bars, fmt='%.4f', fontsize=7, padding=2)
            avg_corr = float(np.mean(corr_vals))
            ax.axhline(avg_corr, color='red', linestyle='--', linewidth=1, label=f'mean={avg_corr:.4f}')
            ax.set_title('Zero-shot Corr (masked) per Test Subject')
            ax.set_ylabel('corr_masked')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(summary_dir / 'subject_corr_bar.png', dpi=150)
            plt.close(fig)

            print(f"\n[viz] Test summary → {summary_dir}")
        except Exception as e:
            print(f"  [viz] Test summary skipped: {e}")

    # ===== 生成补全质量体检报告 =====
    if viz_enable:
        try:
            print("\nBuilding completion-quality report...")
            report_dir = figures_dir / 'report'
            report_dir.mkdir(parents=True, exist_ok=True)

            # 重要：此时 model 是最后一个测试被试（S40）的 finetune 权重，
            # 直接拿去做 val 集体检会语义混乱。这里 reload 预训练最优 ckpt，
            # 让最终报告反映「基础 MCIA 在留出健康被试上的零样本补全能力」。
            model.load_state_dict(checkpoint['model'])
            model.eval()

            # 最终体检：ScenarioMix 走 4 场景矩阵；遗留模式走 5 级难度
            report_kwargs = dict(
                model=model, dataloader=val_loader, device=device, mask_gen=mask_gen,
                output_dir=report_dir,
                num_samples=int(viz_cfg.get('num_report_samples', 20)),
                worst_k=int(viz_cfg.get('worst_k', 8)),
                stft_n_fft=int(viz_cfg.get('stft_n_fft', 128)),
                guidance_scale=CONFIG.get('guidance_scale', 0.0),
                title=f'MCIA Completion Report ({run_label})',
                baseline_fns=_make_baseline_fns(timemae_model, device),
            )
            if use_scenario_mix:
                report_kwargs['scenarios'] = ['s1', 's2', 's3']
                report_kwargs['main_scenario'] = val_scenario
            else:
                report_kwargs['difficulty_levels'] = [0.1, 0.3, 0.5, 0.7, 0.95]
            report_path = build_report(**report_kwargs)
            print(f"  Report: {report_path}")
        except Exception as e:
            print(f"  [viz] Report skipped: {e}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
