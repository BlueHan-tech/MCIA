"""
非ѧϰ型 Baseline 评估

提供与 evaluate_subject 相ͬ输出格ʽ的评估函数，用于对比：
  - evaluate_cubic_spline：三次样条插ֵ补ȫ
  - evaluate_zero_fill     ：ȫ零填充（最简单下界）

返回ֵ均Ϊ dict，key 与 evaluate_subject һ致：
  mse, mae, correlation,
  mse_masked, mse_known, mse_whole,
  corr_masked, corr_known, corr_whole,
  envelope_corr, mdf_error
  (dtw_distance 因计算开销跳过，返回 nan)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Optional

from scipy.interpolate import CubicSpline
from scipy.signal import welch
from scipy import stats

try:
    from dtaidistance import dtw as _dtw_lib
    _DTW_AVAILABLE = True
except ImportError:
    _DTW_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────────────────────────────────────

def _rms_envelope(signal: np.ndarray, win: int = 16) -> np.ndarray:
    kernel = np.ones(win) / win
    return np.sqrt(np.convolve(signal ** 2, kernel, 'same') + 1e-12)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float('nan')
    r, _ = stats.pearsonr(a, b)
    return float(r) if not np.isnan(r) else float('nan')


def _mdf(signal: np.ndarray, fs: int = 200) -> float:
    freqs, psd = welch(signal, fs=fs, nperseg=min(64, len(signal)))
    cum = np.cumsum(psd)
    idx = np.searchsorted(cum, cum[-1] / 2)
    return float(freqs[min(idx, len(freqs) - 1)])


def _cubic_spline_fill(signal: np.ndarray, mask_1d: np.ndarray) -> np.ndarray:
    """
    单ͨ道插ֵ补ȫ。
    signal:  (T,) ԭʼ信号（ȱʧ区ֵ任意）
    mask_1d: (T,) 1=已֪, 0=ȱʧ
    """
    T = len(signal)
    t = np.arange(T, dtype=np.float64)
    known = mask_1d > 0.5

    if known.sum() < 2:
        # 已֪点不足：回退到均ֵ
        fill_val = float(signal[known].mean()) if known.any() else 0.0
        result = signal.copy()
        result[~known] = fill_val
        return result

    cs = CubicSpline(t[known], signal[known], extrapolate=True)
    result = signal.copy()
    result[~known] = cs(t[~known])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 公开 API
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_cubic_spline(dataloader, mask_gen, device,
                          scenario: Optional[str] = None,
                          difficulty: float = 1.0) -> Dict[str, float]:
    """
    用 Cubic Spline 对 dataloader 里ÿ个样本做插ֵ补ȫ，返回与 evaluate_subject
    格ʽ完ȫһ致的ָ标字典。

    Parameters
    ----------
    dataloader : DataLoader
    mask_gen   : ScenarioMaskGenerator 或兼容的 mask 生成器
    device     : torch.device（仅用于生成 mask，Spline 在 CPU 上运行）
    scenario   : mask 场景（None → 随机）
    difficulty : mask 难度
    """
    import torch

    # 延迟导入，避免ѭ环依赖
    from utils.evaluation import _make_batch_mask, _masked_region_metrics

    _fs = 200
    _env_win = 16

    mse_list, mae_list, corr_list = [], [], []
    region_keys = ['mse_masked', 'mse_known', 'mse_whole',
                   'mae_masked', 'mae_known', 'mae_whole',
                   'corr_masked', 'corr_masked_chmiss', 'corr_masked_partial',
                   'corr_known', 'corr_whole']
    region_accum = {k: [] for k in region_keys}
    env_corr_list, mdf_err_list = [], []

    for batch in dataloader:
        if isinstance(batch, dict):
            emg_clean = batch['data'].to(device)
        else:
            emg_clean = batch.to(device)

        B, T, C = emg_clean.shape
        mask_soft = _make_batch_mask(mask_gen, B, C, T, device,
                                     scenario=scenario, difficulty=difficulty)
        mask_soft = mask_soft.transpose(1, 2)        # (B, T, C)
        mask = (mask_soft > 0.5).float()             # 1=已֪, 0=ȱʧ

        emg_np  = emg_clean.cpu().numpy()            # (B, T, C)
        mask_np = mask.cpu().numpy()                 # (B, T, C)

        pred_np = np.zeros_like(emg_np)

        for b in range(B):
            for c in range(C):
                pred_np[b, :, c] = _cubic_spline_fill(
                    emg_np[b, :, c], mask_np[b, :, c]
                )

        # 整体 MSE / MAE / Corr
        mse_list.append(float(np.mean((pred_np - emg_np) ** 2)))
        mae_list.append(float(np.mean(np.abs(pred_np - emg_np))))

        for b in range(B):
            sample_corr = []
            for c in range(C):
                r = _safe_corr(pred_np[b, :, c], emg_np[b, :, c])
                if not np.isnan(r):
                    sample_corr.append(r)
            if sample_corr:
                corr_list.append(float(np.mean(sample_corr)))

        # 分区ָ标（与 evaluate_subject ͬ口径）
        rm = _masked_region_metrics(pred_np, emg_np, mask_np)
        for k, v in rm.items():
            if k in region_accum and not np.isnan(v):
                region_accum[k].append(v)

        # 诊断ָ标
        for b in range(B):
            for c in range(C):
                pred_s = pred_np[b, :, c]
                true_s = emg_np[b, :, c]

                # 包络 PCC
                r = _safe_corr(_rms_envelope(pred_s, _env_win),
                               _rms_envelope(true_s, _env_win))
                if not np.isnan(r):
                    env_corr_list.append(r)

                # MDF 误差
                try:
                    mdf_err_list.append(abs(_mdf(pred_s, _fs) - _mdf(true_s, _fs)))
                except Exception:
                    pass

    result = {
        'mse':          float(np.mean(mse_list))       if mse_list       else float('nan'),
        'mae':          float(np.mean(mae_list))        if mae_list       else float('nan'),
        'correlation':  float(np.mean(corr_list))       if corr_list      else float('nan'),
        'envelope_corr':float(np.mean(env_corr_list))   if env_corr_list  else float('nan'),
        'dtw_distance': float('nan'),                   # 跳过（计算开销大）
        'mdf_error':    float(np.mean(mdf_err_list))    if mdf_err_list   else float('nan'),
    }
    for k in region_keys:
        lst = region_accum[k]
        result[k] = float(np.mean(lst)) if lst else float('nan')
    return result


def evaluate_zero_fill(dataloader, mask_gen, device,
                       scenario: Optional[str] = None,
                       difficulty: float = 1.0) -> Dict[str, float]:
    """零填充 Baseline：ȱʧ区ȫ部填 0，作Ϊ下界参考。"""
    import torch

    from utils.evaluation import _make_batch_mask, _masked_region_metrics

    _fs = 200
    _env_win = 16

    mse_list, mae_list, corr_list = [], [], []
    region_keys = ['mse_masked', 'mse_known', 'mse_whole',
                   'mae_masked', 'mae_known', 'mae_whole',
                   'corr_masked', 'corr_masked_chmiss', 'corr_masked_partial',
                   'corr_known', 'corr_whole']
    region_accum = {k: [] for k in region_keys}
    env_corr_list, mdf_err_list = [], []

    for batch in dataloader:
        if isinstance(batch, dict):
            emg_clean = batch['data'].to(device)
        else:
            emg_clean = batch.to(device)

        B, T, C = emg_clean.shape
        mask_soft = _make_batch_mask(mask_gen, B, C, T, device,
                                     scenario=scenario, difficulty=difficulty)
        mask_soft = mask_soft.transpose(1, 2)
        mask = (mask_soft > 0.5).float()

        emg_np  = emg_clean.cpu().numpy()
        mask_np = mask.cpu().numpy()
        pred_np = emg_np * mask_np          # ȱʧ区Ϊ 0

        mse_list.append(float(np.mean((pred_np - emg_np) ** 2)))
        mae_list.append(float(np.mean(np.abs(pred_np - emg_np))))

        for b in range(B):
            sample_corr = []
            for c in range(C):
                r = _safe_corr(pred_np[b, :, c], emg_np[b, :, c])
                if not np.isnan(r):
                    sample_corr.append(r)
            if sample_corr:
                corr_list.append(float(np.mean(sample_corr)))

        rm = _masked_region_metrics(pred_np, emg_np, mask_np)
        for k, v in rm.items():
            if k in region_accum and not np.isnan(v):
                region_accum[k].append(v)

        for b in range(B):
            for c in range(C):
                pred_s = pred_np[b, :, c]
                true_s = emg_np[b, :, c]
                r = _safe_corr(_rms_envelope(pred_s, _env_win),
                               _rms_envelope(true_s, _env_win))
                if not np.isnan(r):
                    env_corr_list.append(r)
                try:
                    mdf_err_list.append(abs(_mdf(pred_s, _fs) - _mdf(true_s, _fs)))
                except Exception:
                    pass

    result = {
        'mse':          float(np.mean(mse_list))       if mse_list       else float('nan'),
        'mae':          float(np.mean(mae_list))        if mae_list       else float('nan'),
        'correlation':  float(np.mean(corr_list))       if corr_list      else float('nan'),
        'envelope_corr':float(np.mean(env_corr_list))   if env_corr_list  else float('nan'),
        'dtw_distance': float('nan'),
        'mdf_error':    float(np.mean(mdf_err_list))    if mdf_err_list   else float('nan'),
    }
    for k in region_keys:
        lst = region_accum[k]
        result[k] = float(np.mean(lst)) if lst else float('nan')
    return result
