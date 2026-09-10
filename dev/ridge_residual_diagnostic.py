"""开发级诊断：全秩归纳式 Ridge 的剩余误差能否被严格因果的线性时序特征稳定降低。

非正式新模型，不改主链路与协议（EXPERIMENT_PROTOCOL.md §7.5/§7.6；COLLABORATION.md
D-006/E-001/E-002）。只回答 D-006 阶梯的下一问：Ridge 残差是否可被因果线性时序头学习。

臂：
  A  Ridge                     —— dev/ridge_attribution.py 的全局全秩同时刻岭回归（import 复用）。
  B  Ridge + causal residual   —— 线性头 W_L：输入当前及过去 L 个时刻的 ridge 预测、
                                  observed=clean*mask、mask（窗外历史零填充，fit/eval 一致），
                                  仅在 DB2 训练池人工掩码位置以逐通道精确正态方程拟合
                                  residual = clean - ridge（lambda=1e-3 相对正则，先验固定）。
                                  严格因果：时刻 t 只访问 t 及此前帧；无任何 clean 目标作输入。
  L in {0, 8, 16, 32} 全部报告，不按结果挑选，不提升任何臂为正式候选。

口径：S29-S32 仅前向评估；冻结 ScenarioMix seed=20260910+subject_id（与 v1/v2 同掩码）；
统一交付 clip -> patch 边界淡化 -> observed 回填；masked NRMSE（峰值 1）与 K_obs 分层。
掩码生成、masked 指标、crossfade、配置均 import 主模块或既有 dev 脚本，未复制实现。
残差头拟合使用固定种子（20260910）预声明的 8000 训练窗随机子集（全池精确逐通道拟合
计算量不可控；每通道约数十万掩码条目 vs 至多 1189 维特征，统计量充足）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch

from data.dataset_db2_emg import prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import (flatten_pipeline_config, load_yaml_config,
                                  patch_boundary_crossfade)

L_VALUES = (0, 8, 16, 32)
L_MAX = max(L_VALUES)
EVAL_SEED_BASE = 20260910
TRAIN_MASK_SEED = 20260910
SUBSET_SEED = 20260910
N_FIT_WINDOWS = 8000
LAMBDA_HEAD = 1e-3
MASK_BATCH = 1024
EVAL_SUBJECTS = (29, 30, 31, 32)
RIDGE_V1_REFERENCE = {29: 0.1741, 30: 0.1678, 31: 0.1655, 32: 0.1722}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def peak_working_set_mb() -> float:
    if os.name != "nt":
        return float("nan")
    import ctypes
    import ctypes.wintypes as wt

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return float("nan")
    return float(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0)


def build_causal_features(pred: np.ndarray, obs: np.ndarray, msk: np.ndarray,
                          l_max: int) -> np.ndarray:
    """特征列序 = [bias] + lag0(ridge12,obs12,mask12) + ... + lag{l_max}(...)。

    lag ℓ 在时刻 t 取 t-ℓ 帧的值；t-ℓ < 0（窗外历史）三块均零填充，fit/eval 同规则。
    """
    steps = pred.shape[0]
    feats = np.zeros((steps, 1 + (l_max + 1) * 3 * pred.shape[1]), dtype=np.float64)
    feats[:, 0] = 1.0
    for lag in range(l_max + 1):
        col = 1 + lag * 3 * pred.shape[1]
        if lag == 0:
            feats[:, col:col + pred.shape[1]] = pred
            feats[:, col + pred.shape[1]:col + 2 * pred.shape[1]] = obs
            feats[:, col + 2 * pred.shape[1]:col + 3 * pred.shape[1]] = msk
        else:
            feats[lag:, col:col + pred.shape[1]] = pred[:-lag]
            feats[lag:, col + pred.shape[1]:col + 2 * pred.shape[1]] = obs[:-lag]
            feats[lag:, col + 2 * pred.shape[1]:col + 3 * pred.shape[1]] = msk[:-lag]
    return feats


def deliver(pred_np: np.ndarray, clean: np.ndarray, mask: np.ndarray,
            patch_size: int) -> np.ndarray:
    p = torch.from_numpy(np.clip(pred_np, 0.0, 1.0)).float()
    p = patch_boundary_crossfade(p, patch_size)
    return (p * (1 - torch.from_numpy(mask)) + torch.from_numpy(clean * mask)).numpy()


def main() -> None:
    t_start = time.time()
    tm = load("tm", PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py")
    ridge_dev = load("ridge_dev", PROJECT_ROOT / "dev" / "ridge_attribution.py")
    ladder = load("ladder", PROJECT_ROOT / "dev" / "attribution_ladder.py")
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    torch.set_num_threads(8)
    n_channels = int(config["n_channels"])
    patch_size = int(config["patch_size"])
    run_dir = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    out = run_dir / "06_diagnostics" / "ridge_residual_diagnostic_20260910"

    cache_path = run_dir / "07_healthy_completion_benchmark" / "cache" / "db2_train_cache.pt"
    train = torch.load(cache_path, map_location="cpu", weights_only=False)["data"].astype(np.float64)
    print(f"train pool: {len(train)} windows", flush=True)

    flat = train.reshape(-1, n_channels)
    mu = flat.mean(axis=0)
    sigma = np.cov(flat.T)
    print(f"pool sigma cond={np.linalg.cond(sigma):.1f} "
          f"(fit {time.time() - t_start:.0f}s)", flush=True)

    # ---- 固定子集上的逐通道精确残差头拟合（无结果驱动选择）----
    subset_rng = np.random.default_rng(SUBSET_SEED)
    fit_idx = np.sort(subset_rng.choice(len(train), size=N_FIT_WINDOWS, replace=False))
    fit_windows = train[fit_idx]
    gen = tm._scenario_generator(config, TRAIN_MASK_SEED)
    dim = 1 + (L_MAX + 1) * 3 * n_channels
    grams = np.zeros((n_channels, dim, dim), dtype=np.float64)
    cross = np.zeros((dim, n_channels), dtype=np.float64)
    n_rows_channel = np.zeros(n_channels, dtype=np.float64)
    ridge_cache: dict = {}
    t_fit = time.time()
    for b0 in range(0, len(fit_windows), MASK_BATCH):
        b1 = min(b0 + MASK_BATCH, len(fit_windows))
        mask_b = gen.generate_mask(torch.as_tensor(fit_windows[b0:b1])).numpy()
        for i in range(b0, b1):
            clean_w = fit_windows[i]
            mask_w = mask_b[i - b0]
            observed_w = clean_w * mask_w
            pred_w = ridge_dev.predict_window(observed_w, mask_w, mu, sigma,
                                              False, None, ridge_cache)
            missing_w = mask_w < 0.5
            residual_w = clean_w - pred_w
            feats = build_causal_features(pred_w, observed_w, mask_w, L_MAX)
            for c in range(n_channels):
                m_c = missing_w[:, c].astype(np.float64)
                grams[c] += (feats * m_c[:, None]).T @ feats
                cross[:, c] += feats.T @ (residual_w[:, c] * m_c)
                n_rows_channel[c] += float(m_c.sum())
        done = min(b1, len(fit_windows))
        print(f"fit {done}/{len(fit_windows)} windows ({time.time() - t_fit:.0f}s)",
              flush=True)
    print(f"fit rows/channel ~{int(n_rows_channel.min())}-{int(n_rows_channel.max())}",
          flush=True)

    heads = {}
    for L in L_VALUES:
        d_l = 1 + (L + 1) * 3 * n_channels
        w_l = np.zeros((d_l, n_channels), dtype=np.float64)
        for c in range(n_channels):
            a = grams[c, :d_l, :d_l].copy()
            a[np.arange(d_l), np.arange(d_l)] += LAMBDA_HEAD * n_rows_channel[c]
            w_l[:, c] = np.linalg.solve(a, cross[:d_l, c])
        heads[L] = w_l

    # ---- S29-S32 仅前向评估（与 v1/v2 完全同掩码、同交付、同指标）----
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    report = {
        "meta": {
            "script": str(Path(__file__).resolve()),
            "run_dir": str(run_dir),
            "L_values": list(L_VALUES),
            "eval_mask_seed_rule": "20260910 + subject_id (frozen ScenarioMix, same as v1/v2)",
            "train_mask_seed": TRAIN_MASK_SEED,
            "subset_seed": SUBSET_SEED,
            "n_fit_windows": int(N_FIT_WINDOWS),
            "pool_windows": int(len(train)),
            "lambda_head_relative": LAMBDA_HEAD,
            "ridge_lambda": float(ridge_dev.LAMBDA),
            "feature_blocks_per_lag": ["ridge_pred", "observed=clean*mask", "mask"],
            "padding": "zero fill for lags before window start (identical in fit and eval)",
            "causality": "features at t use frames t-L..t only",
            "delivery": "clip [0,1] -> patch boundary crossfade -> observed backfill",
            "features_per_L": {str(L): 1 + (L + 1) * 3 * n_channels for L in L_VALUES},
        },
        "subjects": {},
        "delta_vs_ridge": {},
        "runtime_seconds": None,
        "peak_working_set_mb": None,
    }
    for sid in EVAL_SUBJECTS:
        segments, _, _ = prepare_data_db2(loader, [sid], config, exercises=[1])
        clean = segments.astype(np.float32)
        mask = tm._scenario_generator(config, EVAL_SEED_BASE + sid).generate_mask(
            torch.as_tensor(clean)).numpy().astype(np.float32)
        subj_cache: dict = {}
        ridge_pred = np.stack([
            ridge_dev.predict_window((clean[w] * mask[w]).astype(np.float64),
                                     mask[w], mu, sigma, False, None, subj_cache)
            for w in range(len(clean))])
        subj = {}
        delivered_ridge = deliver(ridge_pred, clean, mask, patch_size)
        subj["ridge"] = ladder.stratified_metrics(delivered_ridge, clean, mask)
        subj["ridge"].update(tm._masked_metrics(delivered_ridge, clean, mask))
        print(f"S{sid} ridge: overall={subj['ridge']['overall']['nrmse']:.4f} "
              f"(v1 reference {RIDGE_V1_REFERENCE[sid]:.4f})", flush=True)
        deltas = {}
        for L in L_VALUES:
            d_l = 1 + (L + 1) * 3 * n_channels
            preds = []
            for w in range(len(clean)):
                feats_w = build_causal_features(
                    ridge_pred[w], (clean[w] * mask[w]).astype(np.float64),
                    mask[w].astype(np.float64), L_MAX)
                preds.append(ridge_pred[w] + feats_w[:, :d_l] @ heads[L])
            pred_arm = np.stack(preds)
            delivered = deliver(pred_arm, clean, mask, patch_size)
            met = ladder.stratified_metrics(delivered, clean, mask)
            met.update(tm._masked_metrics(delivered, clean, mask))
            subj[f"ridge_res_L{L}"] = met
            deltas[f"ridge_res_L{L}"] = {
                key: (None if (met[key]["nrmse"] is None or subj["ridge"][key]["nrmse"] is None)
                      else float(met[key]["nrmse"] - subj["ridge"][key]["nrmse"]))
                for key in ("overall", "kobs_1_5", "kobs_6_11")
            }
            print(f"S{sid} ridge_res_L{L}: overall={met['overall']['nrmse']:.4f} "
                  f"delta={deltas[f'ridge_res_L{L}']['overall']:+.4f}", flush=True)
        report["subjects"][str(sid)] = subj
        report["delta_vs_ridge"][str(sid)] = deltas

    report["runtime_seconds"] = round(time.time() - t_start, 1)
    report["peak_working_set_mb"] = round(peak_working_set_mb(), 1)
    out.mkdir(parents=True, exist_ok=False)
    (out / "ridge_residual_diagnostic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"runtime={report['runtime_seconds']}s peak_ws={report['peak_working_set_mb']}MB",
          flush=True)
    print(f"results={out / 'ridge_residual_diagnostic.json'}", flush=True)


if __name__ == "__main__":
    main()
