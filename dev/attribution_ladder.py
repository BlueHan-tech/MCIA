"""四分法归因 v2：岭回归(归纳线性) vs 低秩归纳基线 vs MCIA(归纳深度) vs CP(转导)。

低秩归纳基线（按用户设计）：DB2 训练池学全局低秩时空-通道基（窗口 PCA，
rank K）；测试时仅用当窗观测条目闭式估计潜变量 z=(U_O^T U_O+λI)^-1 U_O^T x_O，
重构缺失条目。归纳式、闭式、可部署（单窗、无跨窗信息）。

分层报告：每个缺失采样点按当刻观测通道数 K_obs 分箱（0 / 1–5 / 6–11），
K_obs=0 表示全通道同时缺失（仅时间上下文可救），6–11 为典型跨通道缺失。
参照：S29 满规模 MCIA 0.1852 / CP 0.1500。
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
from scipy.sparse.linalg import LinearOperator, svds

from data.dataset_db2_emg import prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import (build_mcia, complete_with_mask, flatten_pipeline_config,
                                  load_mcia_state_dict, load_yaml_config,
                                  patch_boundary_crossfade, set_seed)

RANKS = (8, 32)
LAMBDA = 1e-3
K_BINS = {"kobs_0": (0, 0), "kobs_1_5": (1, 5), "kobs_6_11": (6, 11)}
PCA_RANDOM_SEED = 20260910
PCA_BATCH_WINDOWS = 256


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stratified_metrics(delivered: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> dict:
    missing = mask < 0.5
    k_obs = (mask >= 0.5).sum(axis=2)          # (N, T)
    out = {}
    err = (delivered - clean)
    for name, (lo, hi) in K_BINS.items():
        k = k_obs[..., None]                 # (N, T, 1) 广播到 (N, T, C)
        sel = missing & ((k >= lo) & (k <= hi))
        if sel.sum() == 0:
            out[name] = {"n": 0, "nrmse": None}
            continue
        e = err[sel]
        out[name] = {"n": int(sel.sum()),
                     "nrmse": float(np.sqrt(np.mean(e ** 2)))}
    total_e = err[missing]
    out["overall"] = {"n": int(missing.sum()),
                      "nrmse": float(np.sqrt(np.mean(total_e ** 2)))}
    return out


class CenteredWindowOperator(LinearOperator):
    """无物化 N×3072 中心化矩阵的 DB2 训练池线性算子。"""

    def __init__(self, windows: np.ndarray, mean: np.ndarray, batch_size: int) -> None:
        self.windows = windows.reshape(len(windows), -1)
        self.mean = mean
        self.batch_size = batch_size
        super().__init__(dtype=np.float64, shape=self.windows.shape)

    def _matmat(self, right: np.ndarray) -> np.ndarray:
        out = np.empty((self.shape[0], right.shape[1]), dtype=np.float64)
        for start in range(0, self.shape[0], self.batch_size):
            stop = min(start + self.batch_size, self.shape[0])
            out[start:stop] = (self.windows[start:stop] - self.mean) @ right
        return out

    def _matvec(self, right: np.ndarray) -> np.ndarray:
        return self._matmat(np.asarray(right).reshape(-1, 1)).ravel()

    def _rmatmat(self, right: np.ndarray) -> np.ndarray:
        out = np.zeros((self.shape[1], right.shape[1]), dtype=np.float64)
        for start in range(0, self.shape[0], self.batch_size):
            stop = min(start + self.batch_size, self.shape[0])
            out += (self.windows[start:stop] - self.mean).T @ right[start:stop]
        return out

    def _rmatvec(self, right: np.ndarray) -> np.ndarray:
        return self._rmatmat(np.asarray(right).reshape(-1, 1)).ravel()


def fit_window_pca_basis(train: np.ndarray) -> tuple[dict, dict]:
    """在全部 DB2 训练窗上做固定种子的截断 SVD，避免 full SVD 内存爆炸。"""
    flat = train.reshape(len(train), -1)
    mean = flat.mean(axis=0)
    operator = CenteredWindowOperator(train, mean, PCA_BATCH_WINDOWS)
    _, singular_values, vt = svds(
        operator,
        k=max(RANKS),
        which="LM",
        random_state=PCA_RANDOM_SEED,
    )
    order = np.argsort(-singular_values)
    singular_values = singular_values[order]
    vt = vt[order]
    total_ss = 0.0
    for start in range(0, len(flat), PCA_BATCH_WINDOWS):
        centered = flat[start:start + PCA_BATCH_WINDOWS] - mean
        total_ss += float(np.square(centered).sum())
    basis = {"mean": mean}
    shares = {}
    for rank in RANKS:
        basis[rank] = vt[:rank].T
        shares[str(rank)] = float(np.square(singular_values[:rank]).sum() / total_ss)
    metadata = {
        "fit_windows": int(len(train)),
        "fit_features": int(flat.shape[1]),
        "random_seed": PCA_RANDOM_SEED,
        "solver": "scipy.sparse.linalg.svds(arpack)",
        "batch_windows": PCA_BATCH_WINDOWS,
        "variance_share": shares,
    }
    return basis, metadata


def lowrank_deliver(clean: np.ndarray, mask: np.ndarray, basis: dict, rank: int,
                    patch_size: int) -> np.ndarray:
    """PCA 基低秩补全：每窗用观测条目闭式估计潜变量后重构，统一交付。"""
    U = basis[rank]                              # (3072, K)
    flat = clean.reshape(len(clean), -1)         # (N, 3072)
    mflat = mask.reshape(len(clean), -1)
    mu = basis["mean"].reshape(1, -1)
    preds = flat.copy()
    lam = LAMBDA
    for i in range(len(flat)):
        obs = mflat[i] > 0.5
        Uo = U[obs]                              # (n_obs, K)
        z = np.linalg.solve(Uo.T @ Uo + lam * np.eye(U.shape[1]), Uo.T @ (flat[i, obs] - mu[0, obs]))
        preds[i] = (mu[0] + U @ z).reshape(1, -1)
    p = torch.from_numpy(np.clip(preds, 0.0, 1.0)).float().reshape(
        len(clean), clean.shape[1], clean.shape[2])
    p = patch_boundary_crossfade(p, patch_size)
    return (p * (1 - torch.from_numpy(mask)) + torch.from_numpy(clean * mask)).numpy()


def main() -> None:
    tm = load("tm", PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py")
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    device = config["device"]
    torch.set_num_threads(8)
    run_dir = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    out = run_dir / "06_diagnostics" / "attribution_ladder_20260910_v2"
    out.mkdir(parents=True, exist_ok=False)

    cache_path = run_dir / "07_healthy_completion_benchmark" / "cache" / "db2_train_cache.pt"
    train = torch.load(cache_path, map_location="cpu", weights_only=False)["data"].astype(np.float64)
    basis, pca_metadata = fit_window_pca_basis(train)
    for r in RANKS:
        print(f"PCA rank {r}: top variance share={pca_metadata['variance_share'][str(r)]:.3f}", flush=True)

    mcia = build_mcia(config, "cpu")
    ckpt = Path(config["exp1_dir"]) / "checkpoints" / "best_model.pth"
    load_mcia_state_dict(mcia, ckpt, "cpu")
    mcia.eval()

    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    report = {"reference": {"MCIA_S29_preview": 0.1852, "CP_S29_preview": 0.1500},
              "ranks": list(RANKS), "pca": pca_metadata, "subjects": {}}
    for sid in (29, 30, 31, 32):
        segments, _, _ = prepare_data_db2(loader, [sid], config, exercises=[1])
        clean = segments.astype(np.float32)
        mask = tm._scenario_generator(config, 20260910 + sid).generate_mask(
            torch.as_tensor(clean)).numpy().astype(np.float32)
        subj = {}
        for r in RANKS:
            t0 = time.time()
            delivered = lowrank_deliver(clean.astype(np.float64), mask, basis, r,
                                        int(config["patch_size"]))
            subj[f"lowrank_ind_{r}"] = stratified_metrics(delivered, clean, mask)
            print(f"S{sid} lowrank_ind_{r}: overall={subj[f'lowrank_ind_{r}']['overall']['nrmse']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        outs = []
        with torch.no_grad():
            for s in range(0, len(clean), 64):
                outs.append(complete_with_mask(
                    mcia, torch.as_tensor(clean[s:s + 64]), torch.as_tensor(mask[s:s + 64]),
                    patch_size=int(config["patch_size"])))
        mcia_del = torch.cat(outs).numpy()
        subj["mcia_ind"] = stratified_metrics(mcia_del, clean, mask)
        print(f"S{sid} mcia_ind: overall={subj['mcia_ind']['overall']['nrmse']:.4f}", flush=True)
        report["subjects"][str(sid)] = subj
    (out / "attribution_ladder.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"results={out / 'attribution_ladder.json'}", flush=True)


if __name__ == "__main__":
    main()
