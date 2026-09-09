"""
信号处理工具

提供：
- moving_average: 移动ƽ均滤波（包络提ȡ）
- mu_law_compress / mu_law_expand: μ-law ѹ缩/扩չ
- robust_minmax_normalize: ³棒 Min-Max 归һ化
"""

import numpy as np


def moving_average(signal_data, kernel_size):
    """移动ƽ均滤波器，用于包络提ȡ"""
    kernel = np.ones(kernel_size) / kernel_size
    return np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=0, arr=signal_data
    )


def mu_law_compress(x, mu=255.0):
    """
    μ-law ѹ缩变换（放大低幅ֵ信号ϸ节）
    x ӦΪ非负信号 (x >= 0)
    """
    x_max = x.max()
    if x_max <= 0:
        return x
    x_norm = x / x_max
    compressed = np.log1p(mu * x_norm) / np.log1p(mu)
    return compressed * x_max


def mu_law_expand(x, mu=255.0):
    """μ-law 逆变换"""
    x_max = x.max()
    if x_max <= 0:
        return x
    x_norm = x / x_max
    expanded = (np.expm1(x_norm * np.log1p(mu))) / mu
    return expanded * x_max


def robust_minmax_normalize(emg):
    """
    ³棒 Min-Max 归һ化
    ʹ用百分λ数代替绝对最ֵ，避免异常ֵӰ响。
    归һ化到 [0, 1]：0=静Ϣ, 1=最大发力
    """
    q05, q99 = np.percentile(emg, [5, 99])
    scale = (q99 - q05) + 1e-8
    emg_norm = (emg - q05) / scale
    return np.clip(emg_norm, 0.0, 1.0)
