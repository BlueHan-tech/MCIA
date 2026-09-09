"""
评估与可视化工具

提供：
- evaluate_subject: 快速评估（返回 MSE/MAE/Correlation）
- evaluate_and_plot: 单难度详细可视化
- evaluate_and_plot_multi_difficulty: 多难度级别可视化
- validate_epoch_mcia: 固定目标域验证
- validate_epoch_mcia_masked: 带 masked 指标；返回值含模型范数字段
- probe_film_norms: pred_head 与 local bypass 的 L2 范数
- save_training_history / print_finetuning_summary: 结果保存与汇总
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.signal import welch

from models.completion.mcia_core import derive_ch_mask_from_sample_mask
from utils.paper_pipeline import safe_pearson_np as _safe_pearson

try:
    from dtaidistance import dtw as _dtw_lib
    _DTW_AVAILABLE = True
except ImportError:
    _DTW_AVAILABLE = False


@torch.no_grad()
def probe_film_norms(model) -> dict:
    '''返回当前 MCIA 核心的模型范数探针。'''
    core = getattr(model, 'model', model)
    local = 0.0
    if hasattr(core, 'local_bypass'):
        local = float(core.local_bypass.pw.weight.detach().norm().item())
    return {
        'head_norm': float(core.pred_head[0].weight.detach().norm().item()),
        'local_bypass_norm': local,
    }


def _make_batch_mask(mask_gen, B, C, T, device, scenario=None, difficulty=1.0):
    """掩码生成的统一入口：支持 ScenarioMix（优先用 scenario）与遗留课程学习（回退到 difficulty）。"""
    if scenario is not None and hasattr(mask_gen, '_dispatch'):
        return mask_gen.generate_batch_masks(B, n_channels=C, time_steps=T,
                                             device=device, scenario=scenario)
    if hasattr(mask_gen, 'mask_ratio'):
        return mask_gen.generate_batch_masks(B, n_channels=C, time_steps=T, device=device)
    return mask_gen.generate_batch_masks(B, n_channels=C, time_steps=T,
                                          device=device, difficulty_level=difficulty)


def validate_epoch_mcia(model, dataloader, device, mask_gen, criterion,
                       difficulty=1.0, cfg_dropout_prob=0.0, scenario=None,
                       use_personal_condition=False):
    """
    固定目标域验证。
    训练集难度动态变化；验证集 scenario 固定（ScenarioMix）或 difficulty 固定（legacy）
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
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
            
            mask_soft = _make_batch_mask(mask_gen, B, C, T, device,
                                          scenario=scenario, difficulty=difficulty)
            mask_soft = mask_soft.transpose(1, 2)
            
            mask_binary = (mask_soft > 0.5).float()
            emg_masked = emg_clean * mask_binary
            mask_1d = derive_ch_mask_from_sample_mask(mask_binary)
            
            drop_condition = torch.rand(1).item() < cfg_dropout_prob
            
            emg_pred = model(
                emg_masked, mask=mask_1d, x_masked=emg_masked,
                drop_condition=drop_condition,
                side=side, age=age, gender=gender,
                raw_time_mask=mask_binary,
            )
            emg_pred = emg_pred.clamp(0.0, 1.0) * (1 - mask_binary) + emg_clean * mask_binary
            
            if criterion is not None:
                loss, _ = criterion(emg_pred, emg_clean, mask_binary)
            else:
                loss = F.mse_loss(emg_pred, emg_clean)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch_mcia_masked(model, dataloader, device, mask_gen, criterion=None,
                              difficulty=1.0, cfg_dropout_prob=0.0, scenario=None,
                              use_personal_condition=False):
    """聚焦缺失区域的验证。

    返回用于模型选择与日志记录的指标。

    ``corr_masked``：所有缺失区通道的 Corr 均值（含 CH-MISS + partial）。
    ``corr_masked_chmiss`` / ``corr_masked_partial``：按整通道缺失 vs 部分时间缺失拆分。
    训练早停与 LR scheduler 应使用 ``corr_masked_partial``，避免少数 CH-MISS 通道拖负指标。

    若提供结构损失，``loss_for_early_stop`` 使用该掩码区目标；全信号 MSE 仅用于日志。

    另含模型范数字段：``head_norm``、``local_bypass_norm``。
    """
    model.eval()
    mse_masked, mae_masked, mse_whole = [], [], []
    corr_masked, corr_masked_chmiss, corr_masked_partial, corr_whole = [], [], [], []
    mask_ratios = []
    criterion_losses = []

    with torch.no_grad():
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
            mask_soft = _make_batch_mask(mask_gen, B, C, T, device,
                                          scenario=scenario, difficulty=difficulty)
            mask_soft = mask_soft.transpose(1, 2)
            mask = (mask_soft > 0.5).float()
            missing = mask < 0.5
            emg_masked = emg_clean * mask
            mask_1d = derive_ch_mask_from_sample_mask(mask)

            # Validation must be deterministic for checkpoint selection.  The
            # caller supplies a separately seeded mask generator; classifier-
            # free condition dropout is a training-only perturbation.
            _ = cfg_dropout_prob
            drop_condition = False
            emg_pred = model(
                emg_masked, mask=mask_1d, x_masked=emg_masked,
                drop_condition=drop_condition,
                side=side, age=age, gender=gender,
                raw_time_mask=mask,
            )
            emg_completed = emg_pred.clamp(0.0, 1.0) * (1.0 - mask) + emg_clean * mask

            diff = emg_completed - emg_clean
            if missing.any():
                masked_diff = diff[missing]
                mse_masked.append(float(torch.mean(masked_diff ** 2).item()))
                mae_masked.append(float(torch.mean(torch.abs(masked_diff)).item()))
            mse_whole.append(float(torch.mean(diff ** 2).item()))
            mask_ratios.append(float((1.0 - mask).mean().item()))

            pred_np = emg_completed.cpu().numpy()
            gt_np = emg_clean.cpu().numpy()
            mask_np = mask.cpu().numpy()
            region_metrics = _masked_region_metrics(pred_np, gt_np, mask_np)
            if not np.isnan(region_metrics['corr_masked']):
                corr_masked.append(region_metrics['corr_masked'])
            if not np.isnan(region_metrics['corr_masked_chmiss']):
                corr_masked_chmiss.append(region_metrics['corr_masked_chmiss'])
            if not np.isnan(region_metrics['corr_masked_partial']):
                corr_masked_partial.append(region_metrics['corr_masked_partial'])
            if not np.isnan(region_metrics['corr_whole']):
                corr_whole.append(region_metrics['corr_whole'])

            if criterion is not None:
                criterion_loss, _ = criterion(emg_completed, emg_clean, mask)
                criterion_losses.append(float(criterion_loss.item()))

    masked_mse = float(np.mean(mse_masked)) if mse_masked else float('inf')
    criterion_mean = float(np.mean(criterion_losses)) if criterion_losses else float('nan')
    corr_mean = float(np.mean(corr_masked)) if corr_masked else float('nan')
    corr_chmiss_mean = float(np.mean(corr_masked_chmiss)) if corr_masked_chmiss else float('nan')
    corr_partial_mean = float(np.mean(corr_masked_partial)) if corr_masked_partial else float('nan')

    # 早停组合指标：结构损失 - 0.3 * corr_bonus
    # corr_masked 越高越好；将其折算为对损失的"奖励"项，确保波形形态进入
    # 检查点选择依据，防止模型靠平滑/均值预测在结构损失上提前收敛。
    if criterion_losses and not np.isnan(corr_mean):
        corr_bonus = max(0.0, corr_mean)   # 只奖励正相关
        loss_for_early_stop = criterion_mean - 0.3 * corr_bonus
    elif criterion_losses:
        loss_for_early_stop = criterion_mean
    else:
        loss_for_early_stop = masked_mse

    film_stats = probe_film_norms(model)
    return {
        'loss_for_early_stop': loss_for_early_stop,
        'mse_masked': masked_mse,
        'mae_masked': float(np.mean(mae_masked)) if mae_masked else float('inf'),
        'corr_masked': corr_mean,
        'corr_masked_chmiss': corr_chmiss_mean,
        'corr_masked_partial': corr_partial_mean,
        'mse_whole': float(np.mean(mse_whole)) if mse_whole else float('inf'),
        'corr_whole': float(np.mean(corr_whole)) if corr_whole else float('nan'),
        'mask_ratio': float(np.mean(mask_ratios)) if mask_ratios else float('nan'),
        'criterion_loss': criterion_mean,
        'head_norm': film_stats['head_norm'],
        'local_bypass_norm': film_stats.get('local_bypass_norm', 0.0),
    }


def _masked_region_metrics(pred: np.ndarray, gt: np.ndarray, mask_btc: np.ndarray):
    """按 masked / known / whole 三区统计 MSE 与 Pearson 相关性。

    mask_btc: (B,T,C) 1=观测保留, 0=缺失（模型需补全）
    """
    missing = (mask_btc < 0.5)
    known = ~missing

    def mse_of(region_bool):
        if region_bool.sum() == 0:
            return float('nan')
        return float(((pred[region_bool] - gt[region_bool]) ** 2).mean())

    def mae_of(region_bool):
        if region_bool.sum() == 0:
            return float('nan')
        return float(np.abs(pred[region_bool] - gt[region_bool]).mean())

    def _corr_per_bc(region_bool):
        """逐 (batch, channel) 在 region_bool 为 True 的时间点上算 Pearson。"""
        corrs = []
        B, T, C = gt.shape
        for b in range(B):
            for c in range(C):
                rb = region_bool[b, :, c]
                if rb.sum() < 4:
                    continue
                p = pred[b, rb, c]
                g = gt[b, rb, c]
                if np.std(p) < 1e-8 or np.std(g) < 1e-8:
                    continue
                r = _safe_pearson(p, g)
                if not np.isnan(r):
                    corrs.append(float(r))
        return corrs

    def corr_of_region(region_bool):
        corrs = _corr_per_bc(region_bool)
        return float(np.mean(corrs)) if corrs else float('nan')

    def corr_masked_split():
        """缺失区 Corr 按通道类型拆分：CH-MISS（整通道无锚点）vs partial（有时间锚点）。"""
        ch_has_known = (mask_btc.max(axis=1) >= 0.5)  # (B, C)，与 derive_ch_mask 一致
        corrs_all, corrs_chmiss, corrs_partial = [], [], []
        B, T, C = gt.shape
        for b in range(B):
            for c in range(C):
                rb = missing[b, :, c]
                if rb.sum() < 4:
                    continue
                p = pred[b, rb, c]
                g = gt[b, rb, c]
                if np.std(p) < 1e-8 or np.std(g) < 1e-8:
                    continue
                r = _safe_pearson(p, g)
                if np.isnan(r):
                    continue
                r = float(r)
                corrs_all.append(r)
                if ch_has_known[b, c]:
                    corrs_partial.append(r)
                else:
                    corrs_chmiss.append(r)
        nan = float('nan')
        return {
            'corr_masked': float(np.mean(corrs_all)) if corrs_all else nan,
            'corr_masked_chmiss': float(np.mean(corrs_chmiss)) if corrs_chmiss else nan,
            'corr_masked_partial': float(np.mean(corrs_partial)) if corrs_partial else nan,
        }

    whole_bool = np.ones_like(missing, dtype=bool)
    corr_split = corr_masked_split()
    return {
        'mse_masked': mse_of(missing),
        'mse_known': mse_of(known),
        'mse_whole': mse_of(whole_bool),
        'mae_masked': mae_of(missing),
        'mae_known': mae_of(known),
        'mae_whole': mae_of(whole_bool),
        'corr_masked': corr_split['corr_masked'],
        'corr_masked_chmiss': corr_split['corr_masked_chmiss'],
        'corr_masked_partial': corr_split['corr_masked_partial'],
        'corr_known': corr_of_region(known),
        'corr_whole': corr_of_region(whole_bool),
    }


def evaluate_subject(model, ddpm, dataloader, device, mask_gen, difficulty=1.0,
                     guidance_scale=0.0, scenario=None, use_personal_condition=False):
    """快速评估：整体 MSE/MAE/Corr + 分区 (masked/known/whole) + 包络PCC/DTW/MDF 细粒度指标。"""
    model.eval()
    mse_list, mae_list, corr_list = [], [], []
    region_accum = {k: [] for k in
                    ['mse_masked', 'mse_known', 'mse_whole',
                     'mae_masked', 'mae_known', 'mae_whole',
                     'corr_masked', 'corr_masked_chmiss', 'corr_masked_partial',
                     'corr_known', 'corr_whole']}
    env_corr_list, dtw_list, mdf_err_list = [], [], []

    with torch.no_grad():
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

            mask_soft = _make_batch_mask(mask_gen, B, C, T, device,
                                          scenario=scenario, difficulty=difficulty)
            mask_soft = mask_soft.transpose(1, 2)
            mask = (mask_soft > 0.5).float()
            emg_masked = emg_clean * mask
            mask_1d = derive_ch_mask_from_sample_mask(mask)

            if hasattr(ddpm, 'p_sample'):
                emg_t = torch.randn_like(emg_clean)
                for i in reversed(range(ddpm.timesteps)):
                    t = torch.full((B,), i, device=device, dtype=torch.long)
                    emg_t = ddpm.p_sample(
                        model, emg_t, t, mask_1d, emg_masked,
                        guidance_scale=guidance_scale,
                        raw_time_mask=mask,
                        side=side, age=age, gender=gender,
                    )
                emg_completed = emg_t.clamp(0.0, 1.0) * (1 - mask) + emg_clean * mask
            else:
                emg_pred = model(
                    emg_masked, mask=mask_1d, x_masked=emg_masked,
                    drop_condition=False, side=side, age=age, gender=gender,
                    raw_time_mask=mask,
                )
                emg_completed = emg_pred.clamp(0.0, 1.0) * (1 - mask) + emg_clean * mask

            mse_list.append(F.mse_loss(emg_completed, emg_clean).item())
            mae_list.append(F.l1_loss(emg_completed, emg_clean).item())

            emg_completed_np = emg_completed.cpu().numpy()
            emg_clean_np = emg_clean.cpu().numpy()
            mask_np = mask.cpu().numpy()

            batch_corr = []
            for b in range(B):
                sample_corr = []
                for c in range(C):
                    pred_signal = emg_completed_np[b, :, c]
                    true_signal = emg_clean_np[b, :, c]
                    if np.std(pred_signal) > 1e-8 and np.std(true_signal) > 1e-8:
                        corr = _safe_pearson(pred_signal, true_signal)
                        if not np.isnan(corr):
                            sample_corr.append(corr)
                if sample_corr:
                    batch_corr.append(np.mean(sample_corr))
            if batch_corr:
                corr_list.extend(batch_corr)

            region_metrics = _masked_region_metrics(emg_completed_np, emg_clean_np, mask_np)
            for k, v in region_metrics.items():
                if not (isinstance(v, float) and np.isnan(v)):
                    region_accum[k].append(v)

            # ── 诊断指标：包络 PCC / DTW / MDF ──────────────────────────────
            _env_win = 16
            _fs = 200
            for b in range(B):
                for c in range(C):
                    pred_s = emg_completed_np[b, :, c].astype(np.float64)
                    true_s = emg_clean_np[b, :, c].astype(np.float64)

                    # 1. 包络 PCC（RMS 包络上的 Pearson）
                    kernel = np.ones(_env_win) / _env_win
                    env_pred = np.sqrt(np.convolve(pred_s ** 2, kernel, 'same') + 1e-12)
                    env_true = np.sqrt(np.convolve(true_s ** 2, kernel, 'same') + 1e-12)
                    if np.std(env_pred) > 1e-8 and np.std(env_true) > 1e-8:
                        r, _ = stats.pearsonr(env_pred, env_true)
                        if not np.isnan(r):
                            env_corr_list.append(float(r))

                    # 2. DTW 距离（允许轻微时间偏移）
                    if _DTW_AVAILABLE:
                        try:
                            d = _dtw_lib.distance_fast(pred_s, true_s)
                            dtw_list.append(float(d))
                        except Exception:
                            pass

                    # 3. MDF 误差（中值频率）
                    try:
                        freqs_p, psd_p = welch(pred_s, fs=_fs, nperseg=min(64, len(pred_s)))
                        freqs_t, psd_t = welch(true_s, fs=_fs, nperseg=min(64, len(true_s)))
                        cum_p = np.cumsum(psd_p)
                        cum_t = np.cumsum(psd_t)
                        mdf_p = freqs_p[np.searchsorted(cum_p, cum_p[-1] / 2)]
                        mdf_t = freqs_t[np.searchsorted(cum_t, cum_t[-1] / 2)]
                        mdf_err_list.append(float(abs(mdf_p - mdf_t)))
                    except Exception:
                        pass

    result = {
        'mse': float(np.mean(mse_list)) if mse_list else 0.0,
        'mae': float(np.mean(mae_list)) if mae_list else 0.0,
        'correlation': float(np.mean(corr_list)) if corr_list else 0.0,
        'envelope_corr': float(np.mean(env_corr_list)) if env_corr_list else float('nan'),
        'dtw_distance':  float(np.mean(dtw_list))      if dtw_list      else float('nan'),
        'mdf_error':     float(np.mean(mdf_err_list))  if mdf_err_list  else float('nan'),
    }
    for k, lst in region_accum.items():
        result[k] = float(np.mean(lst)) if lst else float('nan')
    return result


def evaluate_and_plot_multi_difficulty(model, mcia_wrapper, dataloader, device, output_dir, mask_gen, guidance_scale=2.0):
    """DEPRECATED: legacy plotting helper.

    Current mainline rendering uses utils.visualization.build_report /
    plot_completion_panel. This function is retained for manual backward
    compatibility and its behavior is intentionally unchanged.

    多难度级别可视化：5个难度级别各2个样本（共10个样本）
    """
    from matplotlib.gridspec import GridSpec
    
    model.eval()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    difficulty_levels = [0.1, 0.3, 0.5, 0.7, 0.95]
    samples_per_level = 2
    
    all_samples = []
    for batch in dataloader:
        if isinstance(batch, dict):
            batch_data = batch['data'].to(device)
            side = batch.get('side', None)
            age = batch.get('age', None)
            gender = batch.get('gender', None)
            if side is not None: side = side.to(device)
            if age is not None: age = age.to(device)
            if gender is not None: gender = gender.to(device)
        else:
            batch_data = batch.to(device)
            side = age = gender = None
        
        for idx in range(batch_data.shape[0]):
            all_samples.append({
                'data': batch_data[idx],
                'side': side[idx] if side is not None else None,
                'age': age[idx] if age is not None else None,
                'gender': gender[idx] if gender is not None else None
            })
        if len(all_samples) >= 10:
            break
    
    while len(all_samples) < 10:
        all_samples.extend(all_samples[:10-len(all_samples)])
    all_samples = all_samples[:10]
    
    T, C = all_samples[0]['data'].shape
    emg_clean = torch.stack([s['data'] for s in all_samples])
    
    all_masks, all_emg_completed, all_emg_masked = [], [], []
    
    for level_idx, difficulty in enumerate(difficulty_levels):
        level_samples = all_samples[level_idx * samples_per_level:(level_idx + 1) * samples_per_level]
        B_level = len(level_samples)
        emg_clean_level = emg_clean[level_idx * samples_per_level:(level_idx + 1) * samples_per_level]
        
        mask_soft = mask_gen.generate_batch_masks(B_level, n_channels=C, time_steps=T, device=device, difficulty_level=difficulty)
        mask_soft = mask_soft.transpose(1, 2)
        mask = (mask_soft > 0.5).float()
        emg_masked_level = emg_clean_level * mask
        mask_1d = derive_ch_mask_from_sample_mask(mask)

        with torch.no_grad():
            sides_level = torch.stack([s['side'] for s in level_samples]).to(device) if level_samples[0]['side'] is not None else None
            ages_level = torch.stack([s['age'] for s in level_samples]).to(device) if level_samples[0]['age'] is not None else None
            genders_level = torch.stack([s['gender'] for s in level_samples]).to(device) if level_samples[0]['gender'] is not None else None
            
            emg_pred_level = model(
                emg_masked_level, mask=mask_1d, x_masked=emg_masked_level,
                drop_condition=False,
                side=sides_level, age=ages_level, gender=genders_level,
                raw_time_mask=mask,
            )
            emg_completed_level = emg_pred_level.clamp(0.0, 1.0) * (1 - mask) + emg_clean_level * mask
        
        all_masks.append(mask.cpu().numpy())
        all_emg_completed.append(emg_completed_level.cpu().numpy())
        all_emg_masked.append(emg_masked_level.cpu().numpy())
    
    mask_np = np.concatenate(all_masks, axis=0)
    emg_completed_np = np.concatenate(all_emg_completed, axis=0)
    emg_masked_np = np.concatenate(all_emg_masked, axis=0)
    emg_clean_np = emg_clean.cpu().numpy()
    
    for sample_idx in range(10):
        level_idx = sample_idx // samples_per_level
        difficulty = difficulty_levels[level_idx]
        
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        for ch in range(C):
            row, col = ch // 3, ch % 3
            ax = fig.add_subplot(gs[row, col])
            time_steps = np.arange(T)
            
            ax.plot(time_steps, emg_clean_np[sample_idx, :, ch], 'g-', label='Ground Truth', alpha=0.9, linewidth=1.5)
            ax.plot(time_steps, emg_masked_np[sample_idx, :, ch], 'b--', label='Masked Input', alpha=0.6, linewidth=1.0)
            ax.plot(time_steps, emg_completed_np[sample_idx, :, ch], 'r-', label='Reconstructed', alpha=0.8, linewidth=1.5)
            
            for t_idx in range(T):
                if mask_np[sample_idx, t_idx, ch] < 0.5:
                    ax.axvspan(t_idx-0.5, t_idx+0.5, alpha=0.2, color='yellow', zorder=1)
            
            channel_mask_ratio = 1 - mask_np[sample_idx, :, ch].mean()
            mse_ch = np.mean((emg_completed_np[sample_idx, :, ch] - emg_clean_np[sample_idx, :, ch])**2)
            
            if np.std(emg_completed_np[sample_idx, :, ch]) > 1e-8 and np.std(emg_clean_np[sample_idx, :, ch]) > 1e-8:
                corr = _safe_pearson(emg_completed_np[sample_idx, :, ch], emg_clean_np[sample_idx, :, ch])
                if np.isnan(corr): corr = 0.0
            else:
                corr = 0.0
            
            if channel_mask_ratio > 0.5:
                status, status_color = f"Missing {channel_mask_ratio:.0%}", 'red'
            elif channel_mask_ratio > 0.1:
                status, status_color = f"Partial {channel_mask_ratio:.0%}", 'orange'
            else:
                status, status_color = "Normal", 'green'
            
            ax.set_title(f'Ch{ch+1} [{status}] | MSE: {mse_ch:.4f} | Corr: {corr:.3f}',
                        fontsize=10, fontweight='bold', color=status_color)
            ax.grid(True, alpha=0.3)
            if ch == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        level_name = ['Lv1', 'Lv2', 'Lv3', 'Lv4', 'Lv5'][level_idx]
        fig.suptitle(f'Sample {sample_idx+1} | {level_name} (D={difficulty:.2f})', fontsize=16, fontweight='bold')
        
        plt.savefig(Path(output_dir) / f'sample_{sample_idx+1:02d}_level{level_idx+1}_diff{difficulty:.2f}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Multi-difficulty visualization saved to: {output_dir}")


def evaluate_and_plot_multi_scenario(model, mcia_wrapper, dataloader, device, output_dir,
                                     mask_gen, guidance_scale=2.0,
                                     scenarios=('s1', 's2', 's3'),
                                     samples_per_scenario=2):
    """DEPRECATED: legacy plotting helper.

    Current mainline rendering uses utils.visualization.build_report /
    plot_completion_panel. This function is retained for manual backward
    compatibility and its behavior is intentionally unchanged.

    ScenarioMix 的多场景可视化：每个 scenario 抽 `samples_per_scenario` 张 12 通道对比图。

    与 evaluate_and_plot_multi_difficulty 签名兼容，仅把 difficulty 换成 scenario。
    """
    from matplotlib.gridspec import GridSpec

    model.eval()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 收集样本
    total_needed = len(scenarios) * samples_per_scenario
    all_samples = []
    for batch in dataloader:
        if isinstance(batch, dict):
            batch_data = batch['data'].to(device)
            side = batch.get('side', None)
            age = batch.get('age', None)
            gender = batch.get('gender', None)
            if side is not None: side = side.to(device)
            if age is not None: age = age.to(device)
            if gender is not None: gender = gender.to(device)
        else:
            batch_data = batch.to(device)
            side = age = gender = None
        for idx in range(batch_data.shape[0]):
            all_samples.append({
                'data': batch_data[idx],
                'side': side[idx] if side is not None else None,
                'age': age[idx] if age is not None else None,
                'gender': gender[idx] if gender is not None else None,
            })
        if len(all_samples) >= total_needed:
            break
    while len(all_samples) < total_needed:
        all_samples.extend(all_samples[:total_needed - len(all_samples)])
    all_samples = all_samples[:total_needed]

    T, C = all_samples[0]['data'].shape
    emg_clean = torch.stack([s['data'] for s in all_samples])

    for scn_idx, scenario in enumerate(scenarios):
        lo = scn_idx * samples_per_scenario
        hi = lo + samples_per_scenario
        level_samples = all_samples[lo:hi]
        B_level = len(level_samples)
        emg_clean_level = emg_clean[lo:hi]

        mask_soft = _make_batch_mask(mask_gen, B_level, C, T, device, scenario=scenario)
        mask_soft = mask_soft.transpose(1, 2)
        mask = (mask_soft > 0.5).float()
        emg_masked_level = emg_clean_level * mask
        mask_1d = derive_ch_mask_from_sample_mask(mask)

        with torch.no_grad():
            sides_level = torch.stack([s['side'] for s in level_samples]).to(device) if level_samples[0]['side'] is not None else None
            ages_level = torch.stack([s['age'] for s in level_samples]).to(device) if level_samples[0]['age'] is not None else None
            genders_level = torch.stack([s['gender'] for s in level_samples]).to(device) if level_samples[0]['gender'] is not None else None

            emg_pred_level = model(
                emg_masked_level, mask=mask_1d, x_masked=emg_masked_level,
                drop_condition=False,
                side=sides_level, age=ages_level, gender=genders_level,
                raw_time_mask=mask,
            )
            emg_completed_level = emg_pred_level.clamp(0.0, 1.0) * (1 - mask) + emg_clean_level * mask

        mask_np = mask.cpu().numpy()
        emg_completed_np = emg_completed_level.cpu().numpy()
        emg_masked_np = emg_masked_level.cpu().numpy()
        emg_clean_np = emg_clean_level.cpu().numpy()

        for k in range(B_level):
            mask_ratio = 1.0 - float(mask_np[k].mean())
            fig = plt.figure(figsize=(24, 16))
            gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

            for ch in range(C):
                row, col = ch // 3, ch % 3
                ax = fig.add_subplot(gs[row, col])
                time_steps = np.arange(T)

                ax.plot(time_steps, emg_clean_np[k, :, ch], 'g-', label='Ground Truth', alpha=0.9, linewidth=1.5)
                ax.plot(time_steps, emg_masked_np[k, :, ch], 'b--', label='Masked Input', alpha=0.6, linewidth=1.0)
                ax.plot(time_steps, emg_completed_np[k, :, ch], 'r-', label='Reconstructed', alpha=0.8, linewidth=1.5)

                for t_idx in range(T):
                    if mask_np[k, t_idx, ch] < 0.5:
                        ax.axvspan(t_idx - 0.5, t_idx + 0.5, alpha=0.2, color='yellow', zorder=1)

                ch_mask_ratio = 1 - mask_np[k, :, ch].mean()
                mse_ch = float(np.mean((emg_completed_np[k, :, ch] - emg_clean_np[k, :, ch]) ** 2))
                if np.std(emg_completed_np[k, :, ch]) > 1e-8 and np.std(emg_clean_np[k, :, ch]) > 1e-8:
                    corr = _safe_pearson(emg_completed_np[k, :, ch], emg_clean_np[k, :, ch])
                    if np.isnan(corr): corr = 0.0
                else:
                    corr = 0.0

                if ch_mask_ratio < 1e-6:
                    status, status_color = 'Normal', 'green'
                elif ch_mask_ratio >= 0.99:
                    status, status_color = 'Missing 100%', 'red'
                else:
                    status, status_color = f'Missing {ch_mask_ratio*100:.0f}%', 'orange'
                ax.set_title(f'Ch{ch+1} [{status}] | MSE: {mse_ch:.4f} | Corr: {corr:.3f}',
                             fontsize=10, fontweight='bold', color=status_color)
                ax.grid(True, alpha=0.3)
                if ch == 0:
                    ax.legend(loc='upper right', fontsize=8)

            fig.suptitle(f'Sample {k+1} | {scenario.upper()} (mask_ratio={mask_ratio:.2f})',
                         fontsize=16, fontweight='bold')
            plt.savefig(Path(output_dir) / f'{scenario}_sample_{k+1:02d}_ratio{mask_ratio:.2f}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

    print(f"  Multi-scenario visualization saved to: {output_dir}")


def save_training_history(history, output_path):
    """保存训练历史"""
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved: {output_path}")


def print_finetuning_summary(all_results):
    """打印微调结果摘要（含 Wilcoxon 检验）"""
    print("\n" + "="*100)
    print("===== Fine-tuning Summary =====")
    print("="*100)
    
    print(f"\n{'Subject':<10} {'ZS-MSE':<12} {'FT-MSE':<12} {'delta MSE':<12} {'ZS-Corr':<12} {'FT-Corr':<12} {'delta Corr':<12}")
    print("-" * 100)
    
    zs_mse_list, ft_mse_list = [], []
    zs_corr_list, ft_corr_list = [], []
    
    for res in all_results:
        sid = res['subject_id']
        zs_mse = res['zeroshot']['mse']
        ft_mse = res['finetuned']['mse']
        imp_mse = res['improvement']['mse']
        zs_corr = res['zeroshot']['correlation']
        ft_corr = res['finetuned']['correlation']
        imp_corr = res['improvement']['correlation']
        
        zs_mse_list.append(zs_mse)
        ft_mse_list.append(ft_mse)
        zs_corr_list.append(zs_corr)
        ft_corr_list.append(ft_corr)
        
        print(f"S{sid:02d}       {zs_mse:<12.6f} {ft_mse:<12.6f} {imp_mse:>+10.2f}% "
              f"{zs_corr:<12.4f} {ft_corr:<12.4f} {imp_corr:>+10.2f}%")
    
    avg_improvement = {
        'mse': np.mean([r['improvement']['mse'] for r in all_results]),
        'mae': np.mean([r['improvement']['mae'] for r in all_results]),
        'correlation': np.mean([r['improvement']['correlation'] for r in all_results])
    }
    avg_zeroshot = {
        'mse': np.mean([r['zeroshot']['mse'] for r in all_results]),
        'correlation': np.mean([r['zeroshot']['correlation'] for r in all_results])
    }
    avg_finetuned = {
        'mse': np.mean([r['finetuned']['mse'] for r in all_results]),
        'correlation': np.mean([r['finetuned']['correlation'] for r in all_results])
    }
    
    print("-" * 100)
    print(f"Average  {avg_zeroshot['mse']:<12.6f} {avg_finetuned['mse']:<12.6f} {avg_improvement['mse']:>+10.2f}% "
          f"{avg_zeroshot['correlation']:<12.4f} {avg_finetuned['correlation']:<12.4f} {avg_improvement['correlation']:>+10.2f}%")
    print("="*100)
    
    try:
        stat_mse, p_mse = stats.wilcoxon(zs_mse_list, ft_mse_list)
        stat_corr, p_corr = stats.wilcoxon(zs_corr_list, ft_corr_list)
        print(f"\nWilcoxon Test: MSE p={p_mse:.5f} {'*' if p_mse < 0.05 else 'ns'} | "
              f"Corr p={p_corr:.5f} {'*' if p_corr < 0.05 else 'ns'}")
    except Exception as e:
        print(f"\nStatistical test failed: {e}")
    print("="*100)

