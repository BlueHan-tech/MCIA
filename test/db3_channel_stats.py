"""
DB3 截肢患者 sEMG 逐通道统计体检脚本（仅打印数字，无可视化）。

对 DB3 全部 11 个截肢被试，复用与训练完全一致的预处理流水线
(prepare_data_db3: 带通+陷波+整流+滑动平均包络+下采样+μ-law+robust min-max，
 仅保留动作段)，按 subject × channel 打印以下统计量：

  ① 每个窗口的 RMS               -> 每被试每通道 RMS 分布 (5/25/50/75/95 分位)
  ② 每个 patch(8点) 的 MAD       -> 每被试每通道 MAD 分布 (同上 5 分位)
  ③ 每个窗口的峰值幅值(max|x|)   -> 每被试每通道分布 (同上 5 分位)
  ④ 连续低 MAD patch 的最长连续段长度 (窗口内 MAD<1e-3 patch 的最长连续游程)
                                  -> 每被试每通道分布 (同上 5 分位)

定义说明：
- RMS(窗口,通道)  = sqrt(mean_t x[t]^2)
- MAD(patch,通道) = mean_t |x[t] - mean_t x[t]|   (patch 内 8 点的平均绝对偏差)
- Peak(窗口,通道) = max_t |x[t]|
- 低 MAD 游程       = 单个窗口内、沿时间方向连续 MAD<LOW_MAD_THRESH 的 patch 数的最大值
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_db3_emg import prepare_data_db3
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config

PERCENTILES = [5, 25, 50, 75, 95]
LOW_MAD_THRESH = 1e-3


def compute_patch_mad(segments: np.ndarray, patch_size: int) -> np.ndarray:
    """(N, T, C) -> (N, n_patches, C) 每个 patch 内 8 点的平均绝对偏差。"""
    N, T, C = segments.shape
    n_patches = T // patch_size
    patched = segments[:, : n_patches * patch_size, :].reshape(N, n_patches, patch_size, C)
    patch_mean = patched.mean(axis=2, keepdims=True)
    return np.abs(patched - patch_mean).mean(axis=2)  # (N, n_patches, C)


def longest_low_mad_run_per_window(patch_mad: np.ndarray, thresh: float) -> np.ndarray:
    """(N, n_patches, C) -> (N, C) 每个窗口内沿时间连续低 MAD patch 的最长游程。"""
    is_low = patch_mad < thresh  # (N, n_patches, C)
    N, P, C = is_low.shape
    running = np.zeros((N, C), dtype=np.int32)
    longest = np.zeros((N, C), dtype=np.int32)
    for p in range(P):
        running = (running + 1) * is_low[:, p, :]
        longest = np.maximum(longest, running)
    return longest


def collect_subject_stats(segments: np.ndarray, patch_size: int):
    """返回该被试每个通道的 4 组样本数组 (用于求分位数)。"""
    # ① 每窗口 RMS: (N, C)
    rms = np.sqrt(np.mean(segments ** 2, axis=1))
    # ③ 每窗口峰值幅值: (N, C)
    peak = np.max(np.abs(segments), axis=1)
    # ② 每 patch MAD: (N, n_patches, C) -> 展平为 (N*n_patches, C)
    patch_mad = compute_patch_mad(segments, patch_size)
    mad_flat = patch_mad.reshape(-1, patch_mad.shape[-1])
    # ④ 每窗口最长低 MAD 游程: (N, C)
    low_run = longest_low_mad_run_per_window(patch_mad, LOW_MAD_THRESH)
    return {
        "rms": rms,        # (N, C)
        "mad": mad_flat,   # (M, C)
        "peak": peak,      # (N, C)
        "low_run": low_run,  # (N, C)
    }


def percentile_matrix(per_subject_samples, n_channels: int):
    """per_subject_samples：list[(被试 ID, arr(?, C) 或 None)]
    返回 dict[pct] = matrix(n_subjects, n_channels)。"""
    n_sub = len(per_subject_samples)
    mats = {p: np.full((n_sub, n_channels), np.nan) for p in PERCENTILES}
    for i, (_, arr) in enumerate(per_subject_samples):
        if arr is None or arr.size == 0:
            continue
        qs = np.percentile(arr, PERCENTILES, axis=0)  # (5, C)
        for j, p in enumerate(PERCENTILES):
            mats[p][i, :] = qs[j]
    return mats


def print_matrix(title: str, mat: np.ndarray, subject_ids, n_channels: int, fmt: str):
    print(f"\n{title}")
    header = "  Subj |" + "".join(f"{f'Ch{c + 1}':>10}" for c in range(n_channels))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, sid in enumerate(subject_ids):
        row = f"  S{sid:02d}  |"
        for c in range(n_channels):
            v = mat[i, c]
            row += " " * 0 + (f"{'nan':>10}" if np.isnan(v) else f"{format(v, fmt):>10}")
        print(row)


def main():
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    patch_size = int(config["patch_size"])
    n_channels = int(config.get("n_channels", 12))
    subject_ids = list(cfg.get("exp2_transfer", {}).get("db3_subjects", list(range(1, 12))))

    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])

    print("=" * 88)
    print("DB3 截肢患者 sEMG 逐通道统计 (subject × channel)")
    print(f"被试: {subject_ids}")
    print(f"window_size={config['window_size']}, stride={config['stride']}, "
          f"patch_size={patch_size}, n_channels={n_channels}")
    print(f"低 MAD 阈值 = {LOW_MAD_THRESH}, 分位数 = {PERCENTILES}")
    print("=" * 88)

    stats_keys = ["rms", "mad", "peak", "low_run"]
    per_subject = {k: [] for k in stats_keys}

    for sid in subject_ids:
        try:
            segments, _ = prepare_data_db3(data_loader, [sid], config)
        except Exception as exc:  # noqa: BLE001
            print(f"  S{sid:02d}: FAILED - {exc}")
            for k in stats_keys:
                per_subject[k].append((sid, None))
            continue
        s = collect_subject_stats(segments, patch_size)
        for k in stats_keys:
            per_subject[k].append((sid, s[k]))

    titles = {
        "rms": "① 每窗口 RMS 分布 (sqrt(mean(x^2)))",
        "mad": "② 每 patch(8点) MAD 分布 (mean|x-mean|)",
        "peak": "③ 每窗口峰值幅值分布 (max|x|)",
        "low_run": f"④ 每窗口最长低 MAD 游程分布 (连续 MAD<{LOW_MAD_THRESH} 的 patch 数)",
    }
    fmts = {"rms": ".4f", "mad": ".5f", "peak": ".4f", "low_run": ".1f"}

    for k in stats_keys:
        mats = percentile_matrix(per_subject[k], n_channels)
        print("\n" + "=" * 88)
        print(titles[k])
        print("=" * 88)
        for p in PERCENTILES:
            print_matrix(f"[{titles[k].split(' ')[0]} P{p}]", mats[p],
                         subject_ids, n_channels, fmts[k])

    print("\n完成。")


if __name__ == "__main__":
    main()
