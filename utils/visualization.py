"""补全质量可视化体检（Completion Quality Visualization）

提供一套"不用和旧模型 A/B，也能一眼看出补全好坏"的可视化组件：

- plot_completion_panel     : 单样本 12 通道"clean / masked / reconstructed"对比
                              + RMS 包络 + 分区（masked/known/whole）指标 + mask 高亮
- plot_error_heatmap        : (时间, 通道) 误差热图，直观定位补全失败位置
- plot_metric_bars          : 指标柱状图（masked vs known vs whole）
- plot_spectrogram_compare  : 频谱图对比（clean vs reconstructed）
- plot_failure_gallery      : Top-K 最差样本画廊（按 masked-MSE 排序）
- plot_whole_channel_gallery: 专门挑「整通道缺失」样本展示协同补全效果
- plot_difficulty_matrix    : 5 个难度 × 指标的矩阵图（看「越难越烂」情况）
- save_training_epoch_snapshot : 训练期快照（插入训练循环）
- build_report              : 把以上全部打包到 output_dir + 生成 REPORT.md
- build_db3_inference_report: 对 DB3 补全结果 (.npz) 做 before/after 可视化报告
- plot_training_history_film_norms: 从 ``training_history.json`` 绘制 FiLM/add 范数曲线

所有指标分三区计算：
    masked : 模型必须补全的缺失采样点
    known  : 观测保留的采样点（理论上模型不应擅自改动）
    whole  : 全序列
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import matplotlib.pyplot as plt

import numpy as np
from matplotlib.gridspec import GridSpec



# ---------------------------------------------------------------------
# 指标工具
# ---------------------------------------------------------------------

def _rms_envelope(sig: np.ndarray, win: int = 16) -> np.ndarray:
    """简易滑动 RMS 包络（用于直观对比肌电能量趋势）。"""
    sig = sig.astype(np.float64)
    T = sig.shape[0]
    pad = win // 2
    padded = np.pad(sig ** 2, (pad, pad), mode='edge')
    kernel = np.ones(win) / win
    env_sq = np.convolve(padded, kernel, mode='same')[pad:pad + T]
    return np.sqrt(np.maximum(env_sq, 0))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or len(b) < 4:
        return float('nan')
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-12:
        return float('nan')
    r = np.sum(a * b) / denom
    return float(r) if np.isfinite(r) else float('nan')


def _missing_spans(missing: np.ndarray) -> List[Tuple[int, int]]:
    """Return contiguous [start, end) spans for a 1D missing mask."""
    missing = np.asarray(missing, dtype=bool).reshape(-1)
    if missing.size == 0 or not missing.any():
        return []
    edges = np.diff(np.concatenate(([False], missing, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _shade_missing_background(ax, missing: np.ndarray, color: str, alpha: float) -> None:
    """Shade simple contiguous missing intervals using axvspan."""
    for start, end in _missing_spans(missing):
        ax.axvspan(start - 0.5, end - 0.5, color=color, alpha=alpha,
                   linewidth=0, zorder=0)

def compute_region_metrics(pred: np.ndarray, gt: np.ndarray,
                           mask: np.ndarray) -> Dict[str, float]:
    """(T,C) 数组上的三区指标。mask：1=已知，0=缺失。"""
    missing = mask < 0.5
    known = ~missing
    out = {}

    def mse(region_bool):
        if region_bool.sum() == 0:
            return float('nan')
        return float(((pred[region_bool] - gt[region_bool]) ** 2).mean())

    def mae(region_bool):
        if region_bool.sum() == 0:
            return float('nan')
        return float(np.abs(pred[region_bool] - gt[region_bool]).mean())

    out['mse_masked'] = mse(missing)
    out['mse_known'] = mse(known)
    out['mse_whole'] = float(((pred - gt) ** 2).mean())
    out['mae_masked'] = mae(missing)
    out['mae_known'] = mae(known)
    out['mae_whole'] = float(np.abs(pred - gt).mean())

    # 逐通道 Pearson，仅在 masked 区域至少有 4 个点的通道上计算
    T, C = pred.shape
    corrs_masked, corrs_whole = [], []
    for c in range(C):
        m = missing[:, c]
        if m.sum() >= 4:
            r = _safe_corr(pred[m, c], gt[m, c])
            if not np.isnan(r):
                corrs_masked.append(r)
        r_whole = _safe_corr(pred[:, c], gt[:, c])
        if not np.isnan(r_whole):
            corrs_whole.append(r_whole)
    out['corr_masked'] = float(np.mean(corrs_masked)) if corrs_masked else float('nan')
    out['corr_whole'] = float(np.mean(corrs_whole)) if corrs_whole else float('nan')

    # 掩码占比
    out['mask_ratio'] = float(missing.mean())
    return out


# ---------------------------------------------------------------------
# 模型批量推理工具
# ---------------------------------------------------------------------

def _run_model_on_batch(model, emg_clean: torch.Tensor, mask: torch.Tensor,
                        device: torch.device, side=None, age=None, gender=None,
                        guidance_scale: float = 0.0) -> torch.Tensor:
    """mask：(B,T,C) 1=已知，0=缺失；返回补全后的 (B,T,C)。

    如果 guidance_scale>0，直接在 forward 处做 CFG：
        pred = pred_uncond + s*(pred_cond - pred_uncond)
    然后回填观测区。
    """
    from models.completion.mcia_core import derive_ch_mask_from_sample_mask

    mask_1d = derive_ch_mask_from_sample_mask(mask)  # (B,C) 1=通道仍有任意可见时刻
    emg_masked = emg_clean * mask

    kwargs = dict(
        mask=mask_1d, x_masked=emg_masked,
        side=side, age=age, gender=gender,
        raw_time_mask=mask,
    )
    if guidance_scale and guidance_scale > 0.0:
        pred_cond = model(emg_masked, drop_condition=False, **kwargs)
        pred_uncond = model(emg_masked, drop_condition=True, **kwargs)
        pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    else:
        pred = model(emg_masked, drop_condition=False, **kwargs)

    # 回填观测区
    completed = pred.clamp(0.0, 1.0) * (1.0 - mask) + emg_clean * mask
    return completed


def _collect_dataloader_samples(dataloader, device, max_samples: int) -> Dict:
    """从 dataloader 收集至多 max_samples 个样本（含 side/age/gender 若存在）。"""
    import torch

    data_list, side_list, age_list, gender_list = [], [], [], []
    collected = 0
    for batch in dataloader:
        if isinstance(batch, dict):
            data = batch['data'].to(device)
            side = batch.get('side', None)
            age = batch.get('age', None)
            gender = batch.get('gender', None)
            if side is not None: side = side.to(device)
            if age is not None: age = age.to(device)
            if gender is not None: gender = gender.to(device)
        else:
            data = batch.to(device)
            side = age = gender = None
        n = data.shape[0]
        take = min(n, max_samples - collected)
        data_list.append(data[:take])
        if side is not None: side_list.append(side[:take])
        if age is not None: age_list.append(age[:take])
        if gender is not None: gender_list.append(gender[:take])
        collected += take
        if collected >= max_samples:
            break
    if not data_list:
        return {'data': None, 'side': None, 'age': None, 'gender': None}
    return {
        'data': torch.cat(data_list, dim=0),
        'side': torch.cat(side_list, dim=0) if side_list else None,
        'age': torch.cat(age_list, dim=0) if age_list else None,
        'gender': torch.cat(gender_list, dim=0) if gender_list else None,
    }


# ---------------------------------------------------------------------
# 单样本 12 通道面板
# ---------------------------------------------------------------------

def _visible_signal_ylim(signals: List[np.ndarray], pad_ratio: float = 0.08) -> Optional[tuple]:
    """Return y-limits from signals that are actually drawn on the main axis."""
    finite_parts = []
    for sig in signals:
        arr = np.asarray(sig, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_parts.append(arr)
    if not finite_parts:
        return None

    vals = np.concatenate(finite_parts)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if np.isclose(lo, hi):
        delta = max(abs(lo) * 0.1, 1e-3)
        return lo - delta, hi + delta

    pad = max((hi - lo) * pad_ratio, 1e-6)
    return lo - pad, hi + pad


def plot_completion_panel(
    emg_clean: np.ndarray,
    emg_completed: np.ndarray,
    mask: np.ndarray,
    save_path: Path,
    title: str = '',
    show_envelope: bool = True,
    envelope_win: int = 16,
    baselines: Optional[Dict[str, np.ndarray]] = None,
    clean_label: str = 'Ground Truth',
    completed_label: str = 'MCIA Model',
    observed_label: str = 'Observed',
    clean_color: str = '#2ca02c',
    completed_color: str = '#d62728',
    observed_color: str = '#1f77b4',
    baseline_colors: Optional[List[str]] = None,
    show_baselines: bool = False,
    header_text: Optional[str] = None,
    show_observed_line: bool = False,
    show_mask_background: bool = True,
    mask_background_color: str = '#dceeff',
    mask_background_alpha: float = 0.35,
) -> Dict[str, float]:
    """单样本 12 通道对比图。

    emg_clean / emg_completed / mask : (T, C)；mask：1=已知，0=缺失
    baselines : optional {'method': (T, C) completed signal}. Curves are drawn only
                when show_baselines=True, so hidden baselines do not affect y-limits.
    """
    T, C = emg_clean.shape
    metrics = compute_region_metrics(emg_completed, emg_clean, mask)

    # 基线颜色池（与主线颜色区分）
    _BL_COLORS = baseline_colors or ['#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#bcbd22']
    bl_items = list(baselines.items()) if (show_baselines and baselines) else []

    n_cols = 3
    n_rows = int(np.ceil(C / n_cols))
    fig = plt.figure(figsize=(7.5 * n_cols, 2.6 * n_rows + 1.0))
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, hspace=0.55, wspace=0.22,
                  height_ratios=[0.25] + [1.0] * n_rows)

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis('off')
    hdr = header_text or (
        f"mask ratio: {metrics['mask_ratio']*100:.1f}%    "
        f"MSE(masked/known/whole): {metrics['mse_masked']:.4f} / "
        f"{metrics['mse_known']:.4f} / {metrics['mse_whole']:.4f}    "
        f"Corr(masked/whole): {metrics['corr_masked']:.3f} / {metrics['corr_whole']:.3f}"
    )
    header_ax.text(0.5, 0.5, hdr, ha='center', va='center', fontsize=13,
                   fontweight='bold',
                   bbox=dict(boxstyle='round', fc='#f0f7ff', ec='#4a90e2'))

    time_ax = np.arange(T)
    for ch in range(C):
        row, col = ch // n_cols, ch % n_cols
        ax = fig.add_subplot(gs[row + 1, col])

        missing_ch = mask[:, ch] < 0.5
        if show_mask_background and missing_ch.any():
            _shade_missing_background(
                ax, missing_ch,
                color=mask_background_color,
                alpha=mask_background_alpha,
            )

        ax.plot(time_ax, emg_clean[:, ch], color=clean_color,
                label=clean_label, linewidth=1.2, alpha=0.9, zorder=2)
        ax.plot(time_ax, emg_completed[:, ch], color=completed_color,
                label=completed_label, linewidth=1.3, alpha=0.8, zorder=3)
        visible_main_signals = [emg_clean[:, ch], emg_completed[:, ch]]

        # 基线额外折线
        for i, (bl_name, bl_sig) in enumerate(bl_items):
            c_ = _BL_COLORS[i % len(_BL_COLORS)]
            ax.plot(time_ax, bl_sig[:, ch], color=c_,
                    label=bl_name, linewidth=1.0, alpha=0.75, linestyle='-.', zorder=2)
            visible_main_signals.append(bl_sig[:, ch])

        if show_observed_line:
            observed_sig = emg_clean[:, ch] * mask[:, ch]
            ax.plot(time_ax, observed_sig, color=observed_color,
                    label=observed_label, linewidth=0.9, alpha=0.55,
                    linestyle='--', zorder=2)
            visible_main_signals.append(observed_sig)

        ylim = _visible_signal_ylim(visible_main_signals)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if show_envelope:
            env_g = _rms_envelope(emg_clean[:, ch], envelope_win)
            env_p = _rms_envelope(emg_completed[:, ch], envelope_win)
            ax_rt = ax.twinx()
            ax_rt.plot(time_ax, env_g, color=clean_color, alpha=0.35, linewidth=2.0, linestyle=':', zorder=2)
            ax_rt.plot(time_ax, env_p, color=completed_color, alpha=0.35, linewidth=2.0, linestyle=':', zorder=2)
            ax_rt.set_yticks([])

        mr = (mask[:, ch] < 0.5).mean()
        if mr >= 0.99:
            status, color = f'CH-MISS ({mr*100:.0f}%)', '#b71c1c'
        elif mr > 0.5:
            status, color = f'Heavy {mr*100:.0f}%', '#e65100'
        elif mr > 0.05:
            status, color = f'Partial {mr*100:.0f}%', '#ef6c00'
        else:
            status, color = 'Intact', '#2e7d32'

        m = mask[:, ch] < 0.5
        mse_m = float(((emg_completed[m, ch] - emg_clean[m, ch]) ** 2).mean()) if m.sum() > 0 else float('nan')
        corr_w = _safe_corr(emg_completed[:, ch], emg_clean[:, ch])
        ax.set_title(f'Ch{ch+1} [{status}] | MSE_mask={mse_m:.4f} | Corr={corr_w:.3f}',
                     fontsize=10.5, fontweight='bold', color=color)
        ax.grid(True, alpha=0.25)
        if ch == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2)

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------
# 误差热图
# ---------------------------------------------------------------------

def plot_error_heatmap(emg_clean: np.ndarray, emg_completed: np.ndarray,
                       mask: np.ndarray, save_path: Path, title: str = ''):
    """(T,C) 误差 |pred - gt|，遮盖区半透明高亮。"""
    err = np.abs(emg_completed - emg_clean).T  # (C, T)
    T, C = emg_clean.shape

    fig, ax = plt.subplots(figsize=(max(8, T / 20), 0.4 * C + 2))
    vmax = np.percentile(err, 99) + 1e-6
    im = ax.imshow(err, aspect='auto', origin='lower', cmap='magma',
                   vmin=0, vmax=vmax)
    # 掩码叠加（缺失区加淡色网格）
    miss_overlay = (mask < 0.5).astype(np.float32).T  # (C,T)
    ax.imshow(miss_overlay, aspect='auto', origin='lower',
              cmap='Blues', alpha=0.18)

    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Channel')
    ax.set_yticks(np.arange(C))
    ax.set_yticklabels([f'Ch{c+1}' for c in range(C)])
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('|pred - gt|')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------
# 指标柱状图
# ---------------------------------------------------------------------

def plot_metric_bars(all_metrics: List[Dict[str, float]], save_path: Path,
                     title: str = 'Region-wise Metrics'):
    """聚合 N 个样本的 {mse_masked, mse_known, mse_whole, corr_masked, corr_whole}。"""
    if not all_metrics:
        return
    keys_mse = ['mse_masked', 'mse_known', 'mse_whole']
    keys_corr = ['corr_masked', 'corr_whole']

    def _agg(key):
        vals = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        return (float(np.mean(vals)) if vals else 0.0,
                float(np.std(vals)) if vals else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    mse_means = [_agg(k)[0] for k in keys_mse]
    mse_stds = [_agg(k)[1] for k in keys_mse]
    axes[0].bar(keys_mse, mse_means, yerr=mse_stds,
                color=['#d62728', '#1f77b4', '#7f7f7f'], alpha=0.85,
                capsize=6, edgecolor='black')
    axes[0].set_title('MSE by region', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MSE')
    axes[0].grid(True, axis='y', alpha=0.3)
    for i, (m_val, s_val) in enumerate(zip(mse_means, mse_stds)):
        axes[0].text(i, m_val + s_val, f'{m_val:.4f}', ha='center', va='bottom', fontsize=10)

    corr_means = [_agg(k)[0] for k in keys_corr]
    corr_stds = [_agg(k)[1] for k in keys_corr]
    axes[1].bar(keys_corr, corr_means, yerr=corr_stds,
                color=['#2ca02c', '#ff7f0e'], alpha=0.85,
                capsize=6, edgecolor='black')
    axes[1].set_title('Pearson correlation by region', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Pearson r')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, axis='y', alpha=0.3)
    for i, (m_val, s_val) in enumerate(zip(corr_means, corr_stds)):
        axes[1].text(i, m_val + s_val, f'{m_val:.3f}', ha='center', va='bottom', fontsize=10)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------
# 频谱图对比
# ---------------------------------------------------------------------

def plot_spectrogram_compare(emg_clean: np.ndarray, emg_completed: np.ndarray,
                             mask: np.ndarray, save_path: Path,
                             n_fft: int = 128, channels: Optional[List[int]] = None,
                             title: str = ''):
    """对若干通道画 (clean, reconstructed, |diff|) 三联频谱图。"""
    from scipy.signal import stft
    T, C = emg_clean.shape
    if channels is None:
        # 自动选最有缺失的 3 个通道
        miss_ratio_per_ch = (mask < 0.5).mean(axis=0)
        channels = list(np.argsort(-miss_ratio_per_ch)[:3])

    nperseg = min(n_fft, T)
    noverlap = nperseg // 2

    n_rows = len(channels)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for i, ch in enumerate(channels):
        g = emg_clean[:, ch]
        p = emg_completed[:, ch]
        f_g, t_g, Z_g = stft(g, nperseg=nperseg, noverlap=noverlap)
        f_p, t_p, Z_p = stft(p, nperseg=nperseg, noverlap=noverlap)
        S_g = np.log1p(np.abs(Z_g))
        S_p = np.log1p(np.abs(Z_p))
        S_d = np.abs(S_g - S_p)

        vmax = max(S_g.max(), S_p.max())
        axes[i, 0].pcolormesh(t_g, f_g, S_g, shading='gouraud', vmin=0, vmax=vmax, cmap='viridis')
        axes[i, 0].set_title(f'Ch{ch+1} clean', fontsize=10, fontweight='bold')
        axes[i, 0].set_ylabel('freq (norm)')
        axes[i, 1].pcolormesh(t_p, f_p, S_p, shading='gouraud', vmin=0, vmax=vmax, cmap='viridis')
        axes[i, 1].set_title(f'Ch{ch+1} reconstructed', fontsize=10, fontweight='bold')
        axes[i, 2].pcolormesh(t_g, f_g, S_d, shading='gouraud', cmap='magma')
        axes[i, 2].set_title(f'Ch{ch+1} |log-spec diff|', fontsize=10, fontweight='bold')

    for ax in axes[-1, :]:
        ax.set_xlabel('time (norm)')
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------
# 失败画廊 / 整通道缺失画廊
# ---------------------------------------------------------------------

def plot_failure_gallery(emg_clean: np.ndarray, emg_completed: np.ndarray,
                         mask: np.ndarray, metric_per_sample: np.ndarray,
                         save_dir: Path, k: int = 8, tag: str = 'worst',
                         panel_kwargs: Optional[Dict] = None):
    """emg_*: (N,T,C)；metric_per_sample: (N,) 越大越差。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    panel_kwargs = dict(panel_kwargs or {})
    order = np.argsort(-metric_per_sample)[:k]
    for rank, idx in enumerate(order):
        plot_completion_panel(
            emg_clean[idx], emg_completed[idx], mask[idx],
            save_path=save_dir / f'{tag}_rank{rank+1:02d}_idx{idx}.png',
            title=f'{tag.upper()} rank {rank+1}  metric={metric_per_sample[idx]:.4f}',
            **panel_kwargs,
        )


def plot_whole_channel_gallery(emg_clean: np.ndarray, emg_completed: np.ndarray,
                               mask: np.ndarray, save_dir: Path, k: int = 6,
                               panel_kwargs: Optional[Dict] = None):
    """专挑"存在整通道缺失"的样本展示协同补全效果。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    panel_kwargs = dict(panel_kwargs or {})
    N, T, C = emg_clean.shape
    # 每样本的"被整条抹掉的通道数"
    ch_missing_count = ((mask < 0.5).mean(axis=1) > 0.99).sum(axis=1)  # (N,)
    candidate_idx = np.where(ch_missing_count > 0)[0]
    if len(candidate_idx) == 0:
        return
    order = candidate_idx[np.argsort(-ch_missing_count[candidate_idx])][:k]
    for rank, idx in enumerate(order):
        plot_completion_panel(
            emg_clean[idx], emg_completed[idx], mask[idx],
            save_path=save_dir / f'whole_ch_rank{rank+1:02d}_idx{idx}.png',
            title=f'Whole-channel missing sample  #{rank+1}  '
                  f'(ch_missing={ch_missing_count[idx]})',
            **panel_kwargs,
        )


# ---------------------------------------------------------------------
# 难度矩阵
# ---------------------------------------------------------------------

def plot_difficulty_matrix(level2metrics: Dict[float, Dict[str, float]],
                           save_path: Path, title: str = 'Difficulty x Metric'):
    """level2metrics: {difficulty: {'mse_masked':..., 'corr_masked':...}} → 矩阵图。"""
    if not level2metrics:
        return
    levels = sorted(level2metrics.keys())
    metric_keys = ['mse_masked', 'mse_whole', 'corr_masked', 'corr_whole']
    mat = np.full((len(metric_keys), len(levels)), np.nan)
    for j, lv in enumerate(levels):
        for i, mk in enumerate(metric_keys):
            v = level2metrics[lv].get(mk, float('nan'))
            mat[i, j] = v if not np.isnan(v) else np.nan

    fig, ax = plt.subplots(figsize=(1.6 * len(levels) + 2, 1.1 * len(metric_keys) + 1.5))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([f'D={lv:.2f}' for lv in levels])
    ax.set_yticks(np.arange(len(metric_keys)))
    ax.set_yticklabels(metric_keys)
    for i in range(len(metric_keys)):
        for j in range(len(levels)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        color='white' if 'mse' in metric_keys[i] and v > np.nanmean(mat[i])
                        else 'black', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------
# 训练期快照
# ---------------------------------------------------------------------

def _gen_mask_unified(mask_gen, B, C, T, device,
                      scenario: Optional[str] = None, difficulty: float = 1.0):
    """统一掩码生成入口：scenario 优先（ScenarioMix），回退 difficulty（遗留接口）。"""
    if scenario is not None and hasattr(mask_gen, '_dispatch'):
        return mask_gen.generate_batch_masks(B, n_channels=C, time_steps=T,
                                              device=device, scenario=scenario)
    if hasattr(mask_gen, 'mask_ratio'):
        return mask_gen.generate_batch_masks(B, n_channels=C, time_steps=T, device=device)
    return mask_gen.generate_batch_masks(B, n_channels=C, time_steps=T,
                                          device=device, difficulty_level=difficulty)


def save_training_epoch_snapshot(model, val_loader, device, mask_gen,
                                 save_path: Path, difficulty: float = 1.0,
                                 guidance_scale: float = 0.0,
                                 epoch: int = 0, val_loss: float = 0.0,
                                 scenario: Optional[str] = None):
    """训练中每隔若干 epoch 保存一张对比图（使用 val 集首个样本）。

    若 scenario 指定（ScenarioMix），跨 epoch 固定该场景保证可比；否则退化到 difficulty。
    """
    import torch

    model.eval()
    samples = _collect_dataloader_samples(val_loader, device, max_samples=1)
    if samples['data'] is None:
        return
    emg_clean = samples['data']
    B, T, C = emg_clean.shape
    with torch.no_grad():
        mask_soft = _gen_mask_unified(mask_gen, B, C, T, device,
                                       scenario=scenario, difficulty=difficulty)
        mask = (mask_soft.transpose(1, 2) > 0.5).float()  # (B,T,C)
        completed = _run_model_on_batch(
            model, emg_clean, mask, device,
            side=samples['side'], age=samples['age'], gender=samples['gender'],
            guidance_scale=guidance_scale,
        )
    if scenario is not None:
        cfg_tag = f'scn={scenario.upper()}'
    else:
        cfg_tag = f'D={difficulty:.2f}'
    plot_completion_panel(
        emg_clean[0].cpu().numpy(),
        completed[0].cpu().numpy(),
        mask[0].cpu().numpy(),
        save_path=Path(save_path),
        title=f'Epoch {epoch} | val_loss={val_loss:.5f} | {cfg_tag}',
    )


# ---------------------------------------------------------------------
# 总报告打包
# ---------------------------------------------------------------------

def _difficulty_evaluate(model, samples: Dict, device, mask_gen,
                         difficulty: float, guidance_scale: float = 0.0,
                         scenario: Optional[str] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """用同一批 clean 样本跑一次推理；scenario 优先于 difficulty。"""
    import torch

    emg_clean = samples['data']
    B, T, C = emg_clean.shape
    with torch.no_grad():
        mask_soft = _gen_mask_unified(mask_gen, B, C, T, device,
                                       scenario=scenario, difficulty=difficulty)
        mask = (mask_soft.transpose(1, 2) > 0.5).float()
        completed = _run_model_on_batch(
            model, emg_clean, mask, device,
            side=samples['side'], age=samples['age'], gender=samples['gender'],
            guidance_scale=guidance_scale,
        )
    emg_clean_np = emg_clean.cpu().numpy()
    completed_np = completed.cpu().numpy()
    mask_np = mask.cpu().numpy()
    per_sample = []
    for b in range(B):
        per_sample.append(compute_region_metrics(completed_np[b], emg_clean_np[b], mask_np[b]))
    return emg_clean_np, completed_np, mask_np, per_sample


def _aggregate(per_sample: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_sample:
        return {}
    keys = per_sample[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in per_sample if k in s and not np.isnan(s[k])]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    return out


def _plot_scenario_matrix(scn2metrics: Dict[str, Dict[str, float]],
                          save_path: Path, title: str = 'Scenario x Metric'):
    """scn2metrics: {scenario: {'mse_masked':..., 'corr_masked':...}} → 矩阵图。"""
    if not scn2metrics:
        return
    scenarios = list(scn2metrics.keys())
    metric_keys = ['mse_masked', 'mse_whole', 'corr_masked', 'corr_whole']
    mat = np.full((len(metric_keys), len(scenarios)), np.nan)
    for j, scn in enumerate(scenarios):
        for i, mk in enumerate(metric_keys):
            v = scn2metrics[scn].get(mk, float('nan'))
            mat[i, j] = v if not np.isnan(v) else np.nan

    fig, ax = plt.subplots(figsize=(1.6 * len(scenarios) + 2, 1.1 * len(metric_keys) + 1.5))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([s.upper() for s in scenarios])
    ax.set_yticks(np.arange(len(metric_keys)))
    ax.set_yticklabels(metric_keys)
    for i in range(len(metric_keys)):
        for j in range(len(scenarios)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        color='white' if 'mse' in metric_keys[i] and v > np.nanmean(mat[i])
                        else 'black', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def _plot_per_channel_corr(per_sample: List[Dict[str, float]],
                           clean_arr: np.ndarray, comp_arr: np.ndarray, mask_arr: np.ndarray,
                           save_path: Path, title: str = 'Per-channel corr_masked'):
    """逐通道计算 corr_masked 的分布（均值 + std）并画条形图，快速暴露薄弱通道。"""
    if clean_arr.shape[0] == 0:
        return
    B, T, C = clean_arr.shape
    per_ch_corr: List[List[float]] = [[] for _ in range(C)]
    for b in range(B):
        for c in range(C):
            m = mask_arr[b, :, c] < 0.5
            if m.sum() < 4:
                continue
            p = comp_arr[b, m, c]
            g = clean_arr[b, m, c]
            if np.std(p) < 1e-8 or np.std(g) < 1e-8:
                continue
            r = _safe_corr(p, g)
            if not np.isnan(r):
                per_ch_corr[c].append(float(r))

    means = np.array([np.mean(vs) if vs else np.nan for vs in per_ch_corr])
    stds = np.array([np.std(vs) if vs else 0.0 for vs in per_ch_corr])
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(C)
    ax.bar(xs, means, yerr=stds, color='steelblue', alpha=0.85, capsize=3)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'Ch{i+1}' for i in range(C)], rotation=0)
    ax.set_ylabel('corr_masked')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_training_history_film_norms(
    history_path,
    save_path: Optional[Path] = None,
    title: str = 'MCIA weight norms (training_history.json)',
) -> Optional[Path]:
    '''从 training_history.json 绘制模型范数曲线。'''
    history_path = Path(history_path)
    if not history_path.is_file():
        return None
    with open(history_path, 'r', encoding='utf-8') as f:
        hist = json.load(f)

    series_specs = (
        ('local_bypass_norm', '#1f77b4', 'local_bypass_norm'),
        ('head_norm', '#9467bd', 'head_norm'),
    )
    rows = []
    for key, color, label in series_specs:
        raw = hist.get(key)
        if isinstance(raw, list) and len(raw) > 0:
            rows.append((key, color, label, np.asarray(raw, dtype=np.float64)))
    if not rows:
        return None

    n = min(len(r[3]) for r in rows)
    xs = np.arange(1, n + 1)
    if save_path is None:
        save_path = history_path.parent / 'model_norms_curves.png'
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for _key, color, label, y in rows:
        ax.plot(xs, y[:n], color=color, lw=1.8, label=label)
    ax.set_xlabel('epoch')
    ax.set_ylabel('L2 norm of weights')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return save_path


def build_report(
    model, dataloader, device, mask_gen, output_dir: Path,
    num_samples: int = 20, worst_k: int = 8, stft_n_fft: int = 128,
    difficulty_levels: Optional[List[float]] = None,
    scenarios: Optional[List[str]] = None,
    main_scenario: str = 's1',
    guidance_scale: float = 0.0,
    title: str = 'MCIA Completion Report',
    baseline_fns: Optional[Dict[str, callable]] = None,
    panel_kwargs: Optional[Dict] = None,
) -> Path:
    """生成完整的补全质量体检报告。

    若 `scenarios` 指定（如 ['s1','s2','s3']），按 ScenarioMix 口径出报告；
    否则走遗留 difficulty_levels 路径保持向后兼容。

    baseline_fns : {'方法名': fn}，fn 签名为 fn(emg_masked_np, mask_np) -> completed_np，
                   输入/输出均为 (B, T, C) numpy 数组。与主模型使用相同 mask，
                   结果会叠加在 panel 图的每个通道子图上。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    panel_kwargs = dict(panel_kwargs or {})

    use_scenarios = scenarios is not None and len(scenarios) > 0

    samples = _collect_dataloader_samples(dataloader, device, max_samples=num_samples)
    if samples['data'] is None:
        raise RuntimeError('No samples collected from dataloader')

    # ---- 1. 主口径（scenario=main_scenario 或 最难 D）= 失败画廊 / 热图 / 面板 ----
    if use_scenarios:
        clean_h, comp_h, mask_h, per_sample_h = _difficulty_evaluate(
            model, samples, device, mask_gen, difficulty=1.0,
            scenario=main_scenario, guidance_scale=guidance_scale,
        )
        main_tag = f'scn={main_scenario.upper()}'
    else:
        if difficulty_levels is None:
            difficulty_levels = [0.1, 0.3, 0.5, 0.7, 0.95]
        hardest = max(difficulty_levels)
        clean_h, comp_h, mask_h, per_sample_h = _difficulty_evaluate(
            model, samples, device, mask_gen, hardest, guidance_scale,
        )
        main_tag = f'D={hardest:.2f}'

    plot_metric_bars(per_sample_h, save_path=output_dir / 'metric_bars.png',
                     title=f'{title} ({main_tag})')

    metric_for_rank = np.array([s['mse_masked'] if not np.isnan(s['mse_masked'])
                                else s['mse_whole'] for s in per_sample_h])
    worst_idx = int(np.argmax(metric_for_rank))
    plot_error_heatmap(clean_h[worst_idx], comp_h[worst_idx], mask_h[worst_idx],
                       save_path=output_dir / 'error_heatmap_worst.png',
                       title=f'Error Heatmap (worst sample, idx={worst_idx})')
    plot_spectrogram_compare(clean_h[worst_idx], comp_h[worst_idx], mask_h[worst_idx],
                             save_path=output_dir / 'spectrogram_worst.png',
                             n_fft=stft_n_fft,
                             title=f'Spectrogram compare (idx={worst_idx})')

    plot_failure_gallery(clean_h, comp_h, mask_h, metric_for_rank,
                         save_dir=output_dir / 'failures', k=worst_k,
                         panel_kwargs=panel_kwargs)
    plot_whole_channel_gallery(clean_h, comp_h, mask_h,
                               save_dir=output_dir / 'whole_channel', k=min(worst_k, 6),
                               panel_kwargs=panel_kwargs)

    # 计算 baseline 补全结果（与主模型使用同一组 clean 样本 + 同一 mask）
    bl_completed: Dict[str, np.ndarray] = {}  # 方法名 → (N, T, C)
    if baseline_fns:
        emg_masked_np = clean_h * mask_h       # (N, T, C)  已知区保留，缺失区为0
        for bl_name, bl_fn in baseline_fns.items():
            try:
                bl_out = bl_fn(emg_masked_np, mask_h)   # (N, T, C)
                # 回填已知区（保证公平比较）
                bl_completed[bl_name] = bl_out * (1 - mask_h) + clean_h * mask_h
            except Exception as e:
                print(f"  [viz] Baseline '{bl_name}' failed: {e}")

    panels_dir = output_dir / 'panels'
    panels_dir.mkdir(parents=True, exist_ok=True)
    n_panels = min(6, clean_h.shape[0])
    for i in range(n_panels):
        # 构造该样本的基线字典：方法名 → (T, C)
        sample_baselines = {nm: arr[i] for nm, arr in bl_completed.items()} if bl_completed else None
        sample_panel_kwargs = dict(panel_kwargs)
        if sample_baselines is not None:
            sample_panel_kwargs['baselines'] = sample_baselines
        plot_completion_panel(
            clean_h[i], comp_h[i], mask_h[i],
            save_path=panels_dir / f'sample_{i:02d}.png',
            title=f'Sample {i} | {main_tag}',
            **sample_panel_kwargs,
        )

    # 逐通道相关系数
    _plot_per_channel_corr(per_sample_h, clean_h, comp_h, mask_h,
                           save_path=output_dir / 'per_channel_corr.png',
                           title=f'Per-channel corr_masked ({main_tag})')

    # ---- 2. 矩阵：按 scenario 或 difficulty ----
    if use_scenarios:
        scn2agg: Dict[str, Dict[str, float]] = {}
        for scn in scenarios:
            _, _, _, per_sample_scn = _difficulty_evaluate(
                model, samples, device, mask_gen, difficulty=1.0,
                scenario=scn, guidance_scale=guidance_scale,
            )
            scn2agg[scn] = _aggregate(per_sample_scn)
        _plot_scenario_matrix(scn2agg, save_path=output_dir / 'scenario_matrix.png',
                              title='Metric vs scenario')
        axis_key = 'metrics_per_scenario'
        axis_dump = scn2agg
    else:
        level2agg: Dict[float, Dict[str, float]] = {}
        for lv in difficulty_levels:
            _, _, _, per_sample_lv = _difficulty_evaluate(
                model, samples, device, mask_gen, lv, guidance_scale,
            )
            level2agg[lv] = _aggregate(per_sample_lv)
        plot_difficulty_matrix(level2agg, save_path=output_dir / 'difficulty_matrix.png',
                               title='Metric vs difficulty')
        axis_key = 'metrics_per_difficulty'
        axis_dump = {f'{lv:.2f}': vals for lv, vals in level2agg.items()}

    # ---- 3. 保存 metrics.json + REPORT.md ----
    agg_h = _aggregate(per_sample_h)
    metrics_dump = {
        'mode': 'scenario_mix' if use_scenarios else 'difficulty',
        'main_tag': main_tag,
        'num_samples': int(samples['data'].shape[0]),
        'metrics_main': agg_h,
        axis_key: axis_dump,
    }
    # 兼容历史字段
    if not use_scenarios:
        metrics_dump['hardest_difficulty'] = max(difficulty_levels)
        metrics_dump['metrics_hardest'] = agg_h
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_dump, f, indent=2)

    report_md = _render_report_markdown(title, metrics_dump)
    report_path = output_dir / 'REPORT.md'
    report_path.write_text(report_md, encoding='utf-8')
    return report_path


def _render_report_markdown(title: str, metrics_dump: Dict) -> str:
    lines = [f'# {title}\n']
    mode = metrics_dump.get('mode', 'difficulty')
    main_tag = metrics_dump.get('main_tag', f"D={metrics_dump.get('hardest_difficulty', 1.0):.2f}")

    lines.append(f'## 整体指标 ({main_tag})\n')
    agg = metrics_dump.get('metrics_main') or metrics_dump.get('metrics_hardest', {})
    lines.append('| 指标 | 值 |')
    lines.append('|---|---|')
    for k in ['mse_masked', 'mse_known', 'mse_whole',
              'corr_masked', 'corr_whole', 'mask_ratio']:
        if k in agg:
            lines.append(f'| `{k}` | {agg[k]:.5f} |')

    if mode == 'scenario_mix':
        lines.append('\n## 跨场景表现\n')
        lines.append('| scenario | mse_masked | mse_whole | corr_masked | corr_whole | mask_ratio |')
        lines.append('|---|---|---|---|---|---|')
        for scn, vals in metrics_dump.get('metrics_per_scenario', {}).items():
            lines.append(
                f'| {scn.upper()} | {vals.get("mse_masked", float("nan")):.5f} | '
                f'{vals.get("mse_whole", float("nan")):.5f} | '
                f'{vals.get("corr_masked", float("nan")):.3f} | '
                f'{vals.get("corr_whole", float("nan")):.3f} | '
                f'{vals.get("mask_ratio", float("nan")):.3f} |'
            )
        matrix_img = 'scenario_matrix.png'
    else:
        lines.append('\n## 跨难度表现\n')
        lines.append('| difficulty | mse_masked | mse_whole | corr_masked | corr_whole |')
        lines.append('|---|---|---|---|---|')
        for lv_str, vals in metrics_dump.get('metrics_per_difficulty', {}).items():
            try:
                lv = float(lv_str)
            except Exception:
                lv = lv_str
            lines.append(
                f'| {lv if isinstance(lv, str) else f"{lv:.2f}"} | '
                f'{vals.get("mse_masked", float("nan")):.5f} | '
                f'{vals.get("mse_whole", float("nan")):.5f} | '
                f'{vals.get("corr_masked", float("nan")):.3f} | '
                f'{vals.get("corr_whole", float("nan")):.3f} |'
            )
        matrix_img = 'difficulty_matrix.png'

    lines.append('\n## 图件索引\n')
    for rel in ['metric_bars.png', matrix_img,
                'error_heatmap_worst.png', 'spectrogram_worst.png',
                'per_channel_corr.png']:
        lines.append(f'- `{rel}`')
    lines.append('- `panels/` —— 代表性样本 12 通道对比图')
    lines.append('- `failures/` —— 最差 Top-K 样本画廊')
    lines.append('- `whole_channel/` —— 整通道缺失样本画廊')
    lines.append('- `metrics.json` —— 机读指标')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------
# DB3 推理报告（无真值，可做补全前后差异与频谱查看）
# ---------------------------------------------------------------------

def build_db3_inference_report(completed_dir: Path, report_dir: Path,
                               num_samples: int = 20, worst_k: int = 8,
                               stft_n_fft: int = 128) -> Path:
    """DB3 没有真值，侧重「补全前 vs 补全后 + 掩码诊断」的直观展示。"""
    completed_dir = Path(completed_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(completed_dir.glob('db3_S*.npz'))
    if not files:
        raise RuntimeError(f'No db3_S*.npz found in {completed_dir}')

    panels_dir = report_dir / 'panels'
    panels_dir.mkdir(parents=True, exist_ok=True)

    subject_stats = []
    shown = 0
    for fp in files:
        data = np.load(fp)
        orig = data['original']      # (N,T,C)
        comp = data['completed']
        mask = data['mask']
        N, T, C = orig.shape
        mask_ratio = float((mask < 0.5).mean())
        # 每个 subject 挑 mask 最多的几个样本画
        miss_per_sample = (mask < 0.5).mean(axis=(1, 2))
        order = np.argsort(-miss_per_sample)[:max(1, num_samples // len(files))]
        for rank, idx in enumerate(order):
            clean_like = orig[idx]     # 注意：这里真值位置其实是「原始带缺陷信号」
            completed_like = comp[idx]
            m = mask[idx]
            plot_completion_panel(
                clean_like, completed_like, m,
                save_path=panels_dir / f'{fp.stem}_rank{rank+1:02d}.png',
                title=f'{fp.stem} #{idx}  mask={miss_per_sample[idx]*100:.1f}%',
            )
            shown += 1
            if shown >= num_samples:
                break
        subject_stats.append({
            'subject': fp.stem, 'n_segments': int(N),
            'mean_mask_ratio': float(mask_ratio),
        })
        if shown >= num_samples:
            break

    # 全局 subject 级别 mask_ratio 柱图
    if subject_stats:
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(subject_stats) + 2))
        names = [s['subject'] for s in subject_stats]
        ratios = [s['mean_mask_ratio'] for s in subject_stats]
        ax.barh(names, ratios, color='#ef6c00', alpha=0.85, edgecolor='black')
        ax.set_xlabel('mean mask ratio')
        ax.set_title('DB3 per-subject anomaly mask ratio', fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        for i, r in enumerate(ratios):
            ax.text(r, i, f' {r*100:.1f}%', va='center', fontsize=9)
        fig.tight_layout()
        plt.savefig(report_dir / 'subject_mask_ratio.png', dpi=140, bbox_inches='tight')
        plt.close(fig)

    # 生成 REPORT.md
    lines = ['# DB3 MCIA Inference Report\n',
             '## 被试概览\n',
             '| subject | n_segments | mean_mask_ratio |',
             '|---|---|---|']
    for s in subject_stats:
        lines.append(f'| `{s["subject"]}` | {s["n_segments"]} | {s["mean_mask_ratio"]*100:.2f}% |')
    lines.append('\n## 图件索引\n')
    lines.append('- `subject_mask_ratio.png`')
    lines.append('- `panels/` —— 各被试最严重缺失样本的补全对比图')
    report_path = report_dir / 'REPORT.md'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


# ──────────────────────────────────────────────────────────────────────────────
# 基线对比可视化
# ──────────────────────────────────────────────────────────────────────────────

def plot_baseline_comparison(
    results: Dict[str, dict],
    output_dir: Path,
    title: str = 'Baseline Comparison',
) -> Path:
    """生成雷达图 + 箱线图，对比多个方法的指标。

    Parameters
    ----------
    results : dict
        key = 方法名（如 'MCIA Model', 'Cubic Spline', 'Zero Fill'）
        value = 与 evaluate_subject 格式相同的指标字典
    output_dir : Path
        输出目录
    title : str
        图标题

    Returns
    -------
    Path : 雷达图路径（箱线图同目录下 baseline_boxplot.png）
    """
    import math

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 选择展示的指标（高=好 统一化）────────────────────────────────────────
    # 所有指标转换为"越高越好"，范围大致 [0, 1]
    AXES = [
        ('corr_masked',  'Corr\n(masked)',  True,  1.0),
        ('envelope_corr','Envelope\nCorr',  True,  1.0),
        ('mse_masked',   'MSE\n(masked)',   False, None),  # 需动态归一化
        ('mdf_error',    'MDF\nError',      False, None),
        ('mae',          'MAE\n(whole)',    False, None),
    ]

    methods = list(results.keys())
    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors  = (colors * 4)[:len(methods)]

    # 计算归一化后的值（越高越好，[0,1]）
    def _norm_score(key: str, higher_is_better: bool, fixed_max, all_vals):
        vals = [v.get(key, float('nan')) for v in results.values()]
        valid = [v for v in vals if not math.isnan(v)]
        if not valid:
            return [float('nan')] * len(vals)
        vmin, vmax = min(valid), max(valid)
        if fixed_max is not None:
            vmax = fixed_max
        span = vmax - vmin if vmax > vmin else 1e-8
        normed = [(v - vmin) / span if not math.isnan(v) else float('nan') for v in vals]
        if not higher_is_better:
            normed = [1.0 - n if not math.isnan(n) else float('nan') for n in normed]
        return normed

    axis_labels = [a[1] for a in AXES]
    n_axes = len(AXES)
    scores = {}  # method → list of normed scores per axis
    for i, method in enumerate(methods):
        scores[method] = [
            _norm_score(a[0], a[2], a[3], None)[i]
            for a in AXES
        ]

    # ── 1. 雷达图 ─────────────────────────────────────────────────────────────
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig_radar, ax_r = plt.subplots(figsize=(6, 6),
                                   subplot_kw={'projection': 'polar'})
    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(axis_labels, size=9)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=7, color='grey')
    ax_r.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

    for method, color in zip(methods, colors):
        vals = scores[method]
        # 用 0 替换 nan，避免绘图报错
        plot_vals = [0.0 if math.isnan(v) else v for v in vals]
        plot_vals += plot_vals[:1]
        ax_r.plot(angles, plot_vals, color=color, linewidth=2, label=method)
        ax_r.fill(angles, plot_vals, color=color, alpha=0.08)

    ax_r.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)
    ax_r.set_title(title, size=11, pad=20)
    fig_radar.tight_layout()
    radar_path = output_dir / 'baseline_radar.png'
    fig_radar.savefig(radar_path, dpi=150, bbox_inches='tight')
    plt.close(fig_radar)

    # ── 2. 箱线图（原始值，每个指标一个子图）────────────────────────────────
    # 只展示关键 4 个指标
    BOX_KEYS = [
        ('mse_masked',   'MSE (masked)',   False),
        ('corr_masked',  'Corr (masked)',  True),
        ('envelope_corr','Envelope Corr',  True),
        ('mdf_error',    'MDF Error (Hz)', False),
    ]
    fig_box, axes = plt.subplots(1, len(BOX_KEYS), figsize=(4 * len(BOX_KEYS), 4))
    fig_box.suptitle(title, fontsize=11)

    for ax, (key, label, _) in zip(axes, BOX_KEYS):
        bar_vals = []
        bar_labels = []
        for method in methods:
            v = results[method].get(key, float('nan'))
            bar_vals.append(v if not math.isnan(v) else 0.0)
            bar_labels.append(method)
        bars = ax.bar(bar_labels, bar_vals,
                      color=colors[:len(methods)], alpha=0.85, edgecolor='white')
        ax.bar_label(bars, fmt='%.4f', fontsize=7, padding=2)
        ax.set_title(label, fontsize=9)
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, rotation=15, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig_box.tight_layout()
    box_path = output_dir / 'baseline_bars.png'
    fig_box.savefig(box_path, dpi=150, bbox_inches='tight')
    plt.close(fig_box)

    return radar_path



