"""归因实验：岭回归通道转移（归纳式线性基线）在 §7.5 基准条件下与
MCIA/CP-WOPT 对照，判定 CP 优势来源（线性结构 vs 转导信息）。

方法：从 DB2 训练被试（S01–S28，缓存）估计 12×12 通道协方差；对每个时间点
按当刻观测通道子集 S 用协方差条件闭式解预测缺失通道（w=(Σ_SS+λI)^-1 Σ_Sc），
属归纳式（不接触评估被试数据）、闭式、天然实时。交付与 §7.5 统一
（clip→patch 边界淡化→观测回填）。变体：全局中心化 / 逐窗中心化 ×
全秩 / rank-8 截断。参照（S29 满规模）：MCIA 0.1852 / CP 0.1500。
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

LAMBDA = 1e-3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conditional_weights(sigma: np.ndarray, observed: tuple, lam: float):
    """闭式条件均值权重：预测缺失通道 = μ + w·(x_obs - μ)。"""
    S = list(observed)
    M = [c for c in range(12) if c not in S]
    a = sigma[np.ix_(S, S)] + lam * np.eye(len(S))
    b = sigma[np.ix_(S, M)]
    w = np.linalg.solve(a, b)          # (|S|, |M|)
    return S, M, w


def predict_window(x: np.ndarray, mask: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                   window_centered: bool, rank: int | None, cache: dict) -> np.ndarray:
    pred = x.copy()
    for t in range(x.shape[0]):
        S = tuple(c for c in range(12) if mask[t, c] > 0.5)
        miss = [c for c in range(12) if c not in S]
        if not miss or not S:
            continue
        key = (S, window_centered, rank)
        if key not in cache:
            sig = sigma
            if rank is not None:
                vals, vecs = np.linalg.eigh(sigma)
                idx = np.argsort(-vals)[:rank]
                sig = (vecs[:, idx] * vals[idx]) @ vecs[:, idx].T
            s_list, m_list, w = conditional_weights(sig, S, LAMBDA)
            cache[key] = (s_list, m_list, w)
        s_list, m_list, w = cache[key]
        center = mu[s_list] if not window_centered else x[t, s_list].mean()
        x_s = x[t, s_list] - center
        pred[t, miss] = mu[miss] + w.T @ x_s
    return pred


def main() -> None:
    tm = load("tm", PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py")
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    run_dir = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    out = run_dir / "06_diagnostics" / "ridge_attribution_20260910"
    out.mkdir(parents=True, exist_ok=False)

    cache_path = run_dir / "07_healthy_completion_benchmark" / "cache" / "db2_train_cache.pt"
    train = torch.load(cache_path, map_location="cpu", weights_only=False)["data"].astype(np.float64)
    flat = train.reshape(-1, 12)
    mu = flat.mean(axis=0)
    sigma = np.cov(flat.T)
    print(f"train pool: {len(train)} windows; sigma cond={np.linalg.cond(sigma):.1f}", flush=True)

    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    variants = {"global_fullrank": (False, None), "global_rank8": (False, 8),
                "windowcent_fullrank": (True, None), "windowcent_rank8": (True, 8)}
    report = {"reference_S29": {"MCIA": 0.1852, "CP_WOPT": 0.1500},
              "lambda": LAMBDA, "subjects": {}}
    for sid in (29, 30, 31, 32):
        segments, _, _ = prepare_data_db2(loader, [sid], config, exercises=[1])
        clean = segments.astype(np.float32)
        mask = tm._scenario_generator(config, 20260910 + sid).generate_mask(
            torch.as_tensor(clean)).numpy().astype(np.float32)
        observed = clean * mask
        subj = {}
        for name, (wc, rank) in variants.items():
            t0 = time.time()
            cache = {}
            preds = np.stack([predict_window(observed[w].astype(np.float64), mask[w],
                                             mu, sigma, wc, rank, cache)
                              for w in range(len(clean))])
            p = torch.from_numpy(np.clip(preds, 0.0, 1.0)).float()
            p = patch_boundary_crossfade(p, int(config["patch_size"]))
            delivered = (p * (1 - torch.from_numpy(mask)) + torch.from_numpy(observed)).numpy()
            met = tm._masked_metrics(delivered, clean, mask)
            subj[name] = met
            print(f"S{sid} {name}: NRMSE={met['nrmse_peak_1']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        report["subjects"][str(sid)] = subj
    (out / "ridge_attribution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"results={out / 'ridge_attribution.json'}", flush=True)


if __name__ == "__main__":
    main()
