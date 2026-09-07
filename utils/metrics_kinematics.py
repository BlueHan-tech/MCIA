"""
(实验三) 运动学预测评估指标

提供：
- r2_score: 决定系数 R²
- rmse: 均方根误差 RMSE
- mae: 平均绝对误差 MAE
- correlation: 通道级 Pearson 相关系数
- evaluate_kinematics: 综合评估函数
"""

import numpy as np

from utils.paper_pipeline import safe_pearson_np as _safe_pearson


def r2_score(y_true, y_pred, min_variance=1e-6):
    """
    决定系数 R² (Coefficient of Determination)

    R² = 1 - SS_res / SS_tot
    值域 (-inf, 1]，1=完美预测，0=等价于均值预测

    Args:
        y_true: (N, T, C) 或 (N*T, C) 真实关节角度
        y_pred: (N, T, C) 或 (N*T, C) 预测关节角度

    Returns:
        float: 所有通道的平均 R²
    """
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    var_per_channel = np.var(y_true, axis=0)
    valid = var_per_channel >= min_variance

    r2_per_channel = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    r2_per_channel[valid] = 1 - ss_res[valid] / ss_tot[valid]
    r2_avg = float(np.nanmean(r2_per_channel)) if np.any(valid) else float("nan")
    return r2_avg, r2_per_channel, valid


def rmse(y_true, y_pred):
    """
    均方根误差 RMSE

    Returns:
        float: 所有通道的平均 RMSE
    """
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    mse_per_channel = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse_per_channel = np.sqrt(mse_per_channel)
    return float(np.mean(rmse_per_channel)), rmse_per_channel


def mae(y_true, y_pred):
    """平均绝对误差 MAE"""
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    mae_per_channel = np.mean(np.abs(y_true - y_pred), axis=0)
    return float(np.mean(mae_per_channel)), mae_per_channel


def correlation(y_true, y_pred):
    """逐通道 Pearson 相关系数"""
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    n_channels = y_true.shape[1]
    corr_per_channel = np.zeros(n_channels)

    for c in range(n_channels):
        if np.std(y_true[:, c]) > 1e-8 and np.std(y_pred[:, c]) > 1e-8:
            corr_per_channel[c] = _safe_pearson(y_true[:, c], y_pred[:, c])

    return float(np.mean(corr_per_channel)), corr_per_channel


def evaluate_kinematics(y_true, y_pred):
    """
    综合评估函数

    Args:
        y_true: (N, T, C) 真实关节角度
        y_pred: (N, T, C) 预测关节角度

    Returns:
        dict: {'r2': float, 'rmse': float, 'mae': float, 'corr': float, ...}
    """
    r2_avg, r2_ch, r2_valid_ch = r2_score(y_true, y_pred)
    rmse_avg, rmse_ch = rmse(y_true, y_pred)
    mae_avg, mae_ch = mae(y_true, y_pred)
    corr_avg, corr_ch = correlation(y_true, y_pred)

    return {
        'r2': r2_avg,
        'rmse': rmse_avg,
        'mae': mae_avg,
        'corr': corr_avg,
        'r2_per_channel': r2_ch,
        'r2_valid_per_channel': r2_valid_ch,
        'r2_valid_channel_count': int(np.sum(r2_valid_ch)),
        'rmse_per_channel': rmse_ch,
        'mae_per_channel': mae_ch,
        'corr_per_channel': corr_ch,
    }

