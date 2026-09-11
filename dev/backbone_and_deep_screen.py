"""封闭式主干 + 深度模型开发竞赛（E-005）：S1 Kalman / S2 NMF / D2 Transformer 残差 / D3 GRU-D 残差。

开发竞赛，非主方法接入：不改 scripts/、models/、config.yaml、主链路、协议；不读取/运行/生成
S33-S40、DB3、下游任务、SGMD-AAE、CP-WOPT 的任何结果；S29-S32 仅一次最终前向评估。
已有结果不重跑：B1 = Ridge+L32 线性因果残差（E-003，被试等权 0.149606）；D1 = B1+小因果 TCN
（E-004，五 seed 均值 0.148786，SD 0.000394）。唯一例外：D2/D3 需要的 B1 特征按 E-003 冻结
构造（同 8,000 固定训练窗、训练掩码种子 20260910、L32、lambda=1e-3）生成一次，并以逐被试
断言核验评估侧 B1 与 E-003 一致（|delta|<1e-5 否则训练前停止）。

四臂（封闭集合，不再扩展、不调参）：
  S1 因果低维线性 SSM/Kalman：在训练窗干净数据上以 PCA+AR(1) 闭式辨识
     z_t = A z_(t-1)+eps、x_t = mu + C z_t + eta（C 取协方差特征向量，属同一模型族的确定性
     辨识方案）；Q、R 固定对角；推理仅当前可见通道 measurement update 的因果滤波（禁 smoother/
     双向/未来帧）。rank r in {4,6,8}，由训练池内部连续 (subject, repetition) 块 80/20 留出集
     （固定规则：组按 (subject,rep) 升序，每第 5 组为留出）选择，最终在全 8,000 窗重拟合。
  S2 sEMG 肌肉协同 NMF：训练窗干净数据学非负 W（乘法更新，固定 300 iter，每 iter 列归一；
     每 rank 固定 3 个初始化 seed 0/1/2，按拟合目标选 init）；每时刻仅当前可见通道解
     L2 正则 NNLS（小规模活动集精确解）估计 h，用 Wh 恢复缺失。rank {4,6,8} x L2
     {1e-4,1e-3,1e-2} 由同内部留出集选择，最终按固定规则在全 8,000 窗重拟合。
  D2 B1 + 因果 Transformer 残差：输入同 E-004 TCN [B1 base, observed, mask] 36 通道；
     2 层 d_model=16 / 4 heads / FFN=32 / dropout=0.1，因果且局部 64 点注意力 + 固定正弦位置
     编码，零初始化 12 通道残差头。
  D3 B1 + GRU-D 型因果残差：输入为 [B1 base, observed, mask, 每通道距上次观测步数 delta]
     48 通道；单层 GRU hidden=16，零初始化残差头。GRU-D 风格的 delta 特征，非原论文逐字复现。
  D2/D3：masked MSE、AdamW lr=1e-3、batch=64、30 epochs、seeds 20260911-20260915，
     无 early stopping / checkpoint 选择 / 超参搜索；观测位置硬回填，只替换缺失位置。

统一口径：同 B1 的 8,000 训练窗与训练掩码、S29-S32 冻结评估掩码（seed=20260910+sid）、
200 Hz 256x12、clip -> patch crossfade -> observed 回填、masked NRMSE（峰值 1）+ PSNR/RME/
masked Pearson（import eval_healthy_completion_benchmark.metrics_with_pearson）+ K_obs 分层。
内部留出掩码用同族 ScenarioMix、固定种子 20260920（与训练 20260910、评估 20260939-42 不冲突）。

复用（import 不复制）：Ridge predict_window（dev/ridge_attribution.py）；因果特征/交付/常量
（dev/ridge_residual_diagnostic.py）；TCN 结构/堆叠/前向/seed/E-003 断言常量
（dev/ridge_residual_tcn_screen.py）；训练循环 train_arm_seed（dev/ridge_residual_tcn_select.py，
与 E-004 逐位一致）；掩码与 masked 指标（scripts/run_task_matched_literature_baselines.py）；
K_obs 分层（dev/attribution_ladder.py）；含 Pearson 指标
（scripts/eval_healthy_completion_benchmark.py）；配置（utils/paper_pipeline.py）。

判定（预声明）：S1/S2 确定性方法：被试等权均值 < 0.149606 且 >=3/4 被试优于 B1 才是 provisional
winner；D2/D3：五 seed 被试等权均值 < 0.148786 且 >=3/4 被试五 seed 均值优于 B1 才是 provisional
winner。出现 winner 只报告、不叠加；无 winner 则 D1 暂保留为开发集最佳候选，不宣称全局最优。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov

from data.dataset_db2_emg import prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config

INTERNAL_MASK_SEED = 20260920
S1_RANKS = (4, 6, 8)
S2_RANKS = (4, 6, 8)
S2_L2_GRID = (1e-4, 1e-3, 1e-2)
S2_NMF_INIT_SEEDS = (0, 1, 2)
S2_NMF_ITERS = 300
S2_OBJ_CHUNK = 200_000
B1_REF_LITERAL = 0.149606
D1_MEAN_LITERAL = 0.148786
D1_MEAN_EXACT = 0.14878595173358916
DRIFT_TOL = 1e-5
TRANSFORMER_HISTORY = 64


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tm = load("tm", PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py")
ridge_dev = load("ridge_dev", PROJECT_ROOT / "dev" / "ridge_attribution.py")
ladder = load("ladder", PROJECT_ROOT / "dev" / "attribution_ladder.py")
rrd = load("rrd", PROJECT_ROOT / "dev" / "ridge_residual_diagnostic.py")
rts = load("rts", PROJECT_ROOT / "dev" / "ridge_residual_tcn_screen.py")
sel = load("sel", PROJECT_ROOT / "dev" / "ridge_residual_tcn_select.py")
hbench = load("hbench", PROJECT_ROOT / "scripts" / "eval_healthy_completion_benchmark.py")
SEEDS = rts.SEEDS


# ---------------- S1: causal linear-Gaussian SSM (PCA+AR(1) identification + Kalman filter) ----------------

def fit_ssm(windows: np.ndarray, rank: int) -> dict:
    xs = windows.reshape(-1, windows.shape[2])
    mu = xs.mean(axis=0)
    xc = xs - mu
    cov = (xc.T @ xc) / len(xs)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(-evals)
    c_mat = evecs[:, order[:rank]]
    zp = (windows[:, :-1, :] - mu) @ c_mat
    zn = (windows[:, 1:, :] - mu) @ c_mat
    zp2 = zp.reshape(-1, rank)
    zn2 = zn.reshape(-1, rank)
    a_mat = np.linalg.solve(zp2.T @ zp2, zp2.T @ zn2).T
    q_diag = (zn2 - zp2 @ a_mat.T).var(axis=0)
    recon = xc - (xc @ c_mat) @ c_mat.T
    r_diag = recon.var(axis=0)
    rho = float(np.max(np.abs(np.linalg.eigvals(a_mat))))
    stabilized = False
    if rho >= 0.999:
        a_mat = a_mat * (0.999 / rho)
        stabilized = True
    p0 = solve_discrete_lyapunov(a_mat, np.diag(q_diag))
    return {"mu": mu, "C": c_mat, "A": a_mat, "Q": np.diag(q_diag), "r_diag": r_diag,
            "P0": p0, "rho": rho, "stabilized": stabilized, "rank": rank,
            "top_eigvals": [float(v) for v in evals[order[:rank]]]}


def ssm_filter_window(model: dict, observed_w: np.ndarray, mask_w: np.ndarray) -> np.ndarray:
    mu, c_mat, a_mat, q_mat, r_diag, p0 = (model["mu"], model["C"], model["A"],
                                           model["Q"], model["r_diag"], model["P0"])
    rank = c_mat.shape[1]
    eye = np.eye(rank)
    a_t = a_mat.T
    z = np.zeros(rank)
    p_cov = p0.copy()
    out = np.empty((mask_w.shape[0], mu.shape[0]))
    for t in range(mask_w.shape[0]):
        z = a_mat @ z
        p_cov = a_mat @ p_cov @ a_t + q_mat
        obs_idx = np.flatnonzero(mask_w[t] > 0.5)
        if obs_idx.size:
            h_mat = c_mat[obs_idx]
            s_mat = h_mat @ p_cov @ h_mat.T + np.diag(r_diag[obs_idx])
            k_gain = np.linalg.solve(s_mat, h_mat @ p_cov).T
            z = z + k_gain @ (observed_w[t, obs_idx] - mu[obs_idx] - h_mat @ z)
            p_cov = (eye - k_gain @ h_mat) @ p_cov
        out[t] = mu + c_mat @ z
    return out


def ssm_complete_stack(model: dict, clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preds = np.empty(clean.shape)
    for w in range(len(clean)):
        preds[w] = ssm_filter_window(model, clean[w] * mask[w], mask[w])
    return preds


# ---------------- S2: NMF synergy + per-step L2-regularized NNLS ----------------

def fit_nmf(x_mat: np.ndarray, rank: int, seed: int, iters: int = S2_NMF_ITERS):
    rng = np.random.default_rng(seed)
    w_mat = rng.random((x_mat.shape[0], rank)) + 0.1
    h_mat = rng.random((rank, x_mat.shape[1])) + 0.1
    for _ in range(iters):
        h_mat *= (w_mat.T @ x_mat) / (w_mat.T @ w_mat @ h_mat + 1e-12)
        w_mat *= (x_mat @ h_mat.T) / (w_mat @ h_mat @ h_mat.T + 1e-12)
        norms = np.linalg.norm(w_mat, axis=0, keepdims=True)
        w_mat /= norms
        h_mat *= norms.T
    obj = 0.0
    for s0 in range(0, x_mat.shape[1], S2_OBJ_CHUNK):
        s1 = min(s0 + S2_OBJ_CHUNK, x_mat.shape[1])
        err = x_mat[:, s0:s1] - w_mat @ h_mat[:, s0:s1]
        obj += float(np.square(err).sum())
    return w_mat, obj


def nnls_l2(w_mat: np.ndarray, obs_idx: np.ndarray, x_obs: np.ndarray, lam: float,
            cache: dict) -> np.ndarray:
    key = (obs_idx.tobytes(), lam)
    entry = cache.get(key)
    if entry is None:
        w_obs = w_mat[obs_idx]
        gram = w_obs.T @ w_obs + lam * np.eye(w_mat.shape[1])
        entry = (w_obs, gram, np.linalg.inv(gram))
        cache[key] = entry
    w_obs, gram, gram_inv = entry
    b_vec = w_obs.T @ x_obs
    h_vec = gram_inv @ b_vec
    if (h_vec >= 0).all():
        return h_vec
    active = np.ones(h_vec.size, bool)
    h_vec = np.zeros(h_vec.size)
    for _ in range(h_vec.size + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return h_vec
        h_act = np.linalg.solve(gram[np.ix_(idx, idx)], b_vec[idx])
        if (h_act >= -1e-12).all():
            h_vec[idx] = np.clip(h_act, 0.0, None)
            return h_vec
        active[idx[int(np.argmin(h_act))]] = False
    return h_vec


def nmf_complete_stack(w_mat: np.ndarray, lam: float, clean: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    cache: dict = {}
    preds = np.empty(clean.shape)
    for w in range(len(clean)):
        obs_w = clean[w] * mask[w]
        for t in range(clean.shape[1]):
            obs_idx = np.flatnonzero(mask[w, t] > 0.5)
            if obs_idx.size == 0:
                preds[w, t] = 0.0
            else:
                h_vec = nnls_l2(w_mat, obs_idx, obs_w[t, obs_idx], lam, cache)
                preds[w, t] = w_mat @ h_vec
    return preds


# ---------------- D2 / D3 residual models ----------------

def sinusoidal_pe(time_steps: int, d_model: int) -> np.ndarray:
    pos = np.arange(time_steps)[:, None]
    i = np.arange(d_model)[None, :]
    ang = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((time_steps, d_model))
    pe[:, 0::2] = np.sin(ang[:, 0::2])
    pe[:, 1::2] = np.cos(ang[:, 1::2])
    return pe


class CausalLocalTransformerResidual(torch.nn.Module):
    """因果 + 局部 64 点注意力的 2 层 Transformer 残差头，输出层零初始化。"""

    def __init__(self, in_ch: int = 36, d_model: int = 16, nhead: int = 4, ffn: int = 32,
                 dropout: float = 0.1, n_layers: int = 2, history: int = TRANSFORMER_HISTORY,
                 out_ch: int = 12, time_steps: int = 256):
        super().__init__()
        self.in_proj = torch.nn.Conv1d(in_ch, d_model, kernel_size=1)
        self.register_buffer("pe", torch.from_numpy(
            sinusoidal_pe(time_steps, d_model)).float())
        q_pos = torch.arange(time_steps)[:, None]   # 行 = query i
        k_pos = torch.arange(time_steps)[None, :]   # 列 = key j
        # torch 2.5.1 bool attn_mask 语义：True = 禁止注意（探针实证）；行=query、列=key。
        # 允许 j <= i（因果）且 i - j <= history-1（局部 64 点历史）。
        blocked = ~((k_pos <= q_pos) & (q_pos - k_pos <= history - 1))
        self.register_buffer("attn_mask", blocked)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn, dropout=dropout,
            batch_first=True, activation="relu")
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = torch.nn.Linear(d_model, out_ch)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x).transpose(1, 2) + self.pe
        h = self.encoder(h, mask=self.attn_mask)
        return self.head(h).transpose(1, 2)


class GRUDStyleResidual(torch.nn.Module):
    """GRU-D 风格因果残差头（含距上次观测步数特征），输出层零初始化；非原论文复现。"""

    def __init__(self, in_ch: int = 48, hidden: int = 16, out_ch: int = 12):
        super().__init__()
        self.gru = torch.nn.GRU(in_ch, hidden, num_layers=1, batch_first=True)
        self.head = torch.nn.Linear(hidden, out_ch)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.gru(x.transpose(1, 2))
        return self.head(seq).transpose(1, 2)


def time_since_last_obs(mask: np.ndarray) -> np.ndarray:
    n_win, n_time, n_ch = mask.shape
    delta = np.empty(mask.shape, dtype=np.float32)
    counter = np.zeros((n_win, n_ch), dtype=np.float32)
    for t in range(n_time):
        observed = mask[:, t, :] > 0.5
        counter = np.where(observed, 0.0, counter + 1.0)
        delta[:, t, :] = counter
    return delta


def forward_windows(model: torch.nn.Module, x_np: np.ndarray, chunk: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for s0 in range(0, len(x_np), chunk):
            outs.append(model(torch.as_tensor(x_np[s0:s0 + chunk])).numpy())
    return np.concatenate(outs)


def measure_latency_ms(fn, n_runs: int = 20) -> float:
    fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_metrics(delivered, clean, mask):
    met = ladder.stratified_metrics(delivered, clean, mask)
    met.update(hbench.metrics_with_pearson(delivered, clean, mask))
    return met


def main() -> None:
    t_start = time.time()
    script_path = Path(__file__).resolve()
    code_sha = sha256_of(script_path)
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    torch.set_num_threads(8)
    n_channels = int(config["n_channels"])
    patch_size = int(config["patch_size"])
    run_dir = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    out_dir = run_dir / "06_diagnostics" / "backbone_deep_screen_20260911"
    tmp_preds = Path(tempfile.mkdtemp(prefix="backbone_deep_preds_"))
    anomalies: list = []

    # ---- B1 特征生成（E-003 冻结构造，一次性）----
    cache_path = run_dir / "07_healthy_completion_benchmark" / "cache" / "db2_train_cache.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    train_pool = cache["data"].astype(np.float64)
    pool_subjects = cache["subject_ids"]
    pool_reps = cache["repetitions"]
    flat = train_pool.reshape(-1, n_channels)
    mu = flat.mean(axis=0)
    sigma = np.cov(flat.T)
    del flat
    subset_rng = np.random.default_rng(rrd.SUBSET_SEED)
    fit_idx = np.sort(subset_rng.choice(len(train_pool), size=rrd.N_FIT_WINDOWS, replace=False))
    fit_windows = train_pool[fit_idx]

    dim = 1 + (rrd.L_MAX + 1) * 3 * n_channels
    grams = np.zeros((n_channels, dim, dim), dtype=np.float64)
    cross = np.zeros((dim, n_channels), dtype=np.float64)
    n_rows_channel = np.zeros(n_channels, dtype=np.float64)
    ridge_cache: dict = {}
    t_fit = time.time()
    gen = tm._scenario_generator(config, rrd.TRAIN_MASK_SEED)
    for b0 in range(0, len(fit_windows), rrd.MASK_BATCH):
        b1 = min(b0 + rrd.MASK_BATCH, len(fit_windows))
        mask_b = gen.generate_mask(torch.as_tensor(fit_windows[b0:b1])).numpy()
        for i in range(b0, b1):
            clean_w = fit_windows[i]
            mask_w = mask_b[i - b0]
            observed_w = clean_w * mask_w
            pred_w = ridge_dev.predict_window(observed_w, mask_w, mu, sigma,
                                              False, None, ridge_cache)
            missing_w = mask_w < 0.5
            residual_w = clean_w - pred_w
            feats = rrd.build_causal_features(pred_w, observed_w, mask_w, rrd.L_MAX)
            for c in range(n_channels):
                m_c = missing_w[:, c].astype(np.float64)
                grams[c] += (feats * m_c[:, None]).T @ feats
                cross[:, c] += feats.T @ (residual_w[:, c] * m_c)
                n_rows_channel[c] += float(m_c.sum())
        print(f"gram {min(b1, len(fit_windows))}/{len(fit_windows)} ({time.time() - t_fit:.0f}s)",
              flush=True)
    w32 = np.zeros((dim, n_channels), dtype=np.float64)
    for c in range(n_channels):
        a = grams[c].copy()
        a[np.arange(dim), np.arange(dim)] += rrd.LAMBDA_HEAD * n_rows_channel[c]
        w32[:, c] = np.linalg.solve(a, cross[:, c])
    del grams
    gen2 = tm._scenario_generator(config, rrd.TRAIN_MASK_SEED)
    base_rows, obs_rows, msk_rows, cln_rows = [], [], [], []
    for b0 in range(0, len(fit_windows), rrd.MASK_BATCH):
        b1 = min(b0 + rrd.MASK_BATCH, len(fit_windows))
        mask_b = gen2.generate_mask(torch.as_tensor(fit_windows[b0:b1])).numpy()
        for i in range(b0, b1):
            clean_w = fit_windows[i]
            mask_w = mask_b[i - b0]
            observed_w = clean_w * mask_w
            pred_w = ridge_dev.predict_window(observed_w, mask_w, mu, sigma,
                                              False, None, ridge_cache)
            feats = rrd.build_causal_features(pred_w, observed_w, mask_w, rrd.L_MAX)
            base_rows.append(pred_w + feats @ w32)
            obs_rows.append(observed_w)
            msk_rows.append(mask_w)
            cln_rows.append(clean_w)
        print(f"subset base {min(b1, len(fit_windows))}/{len(fit_windows)} "
              f"({time.time() - t_fit:.0f}s)", flush=True)
    base_sub = np.stack(base_rows)
    obs_sub = np.stack(obs_rows)
    msk_sub = np.stack(msk_rows).astype(np.float32)
    clean_sub = np.stack(cln_rows)
    del base_rows, obs_rows, msk_rows, cln_rows, fit_windows, train_pool, cache
    val_clean64 = clean_sub[:64].copy()
    val_mask64 = msk_sub[:64].copy()
    val_base64 = base_sub[:64].copy()
    x_sub36 = rts.stack_inputs(base_sub, obs_sub, msk_sub)
    delta_sub = time_since_last_obs(msk_sub)
    x_sub48 = np.concatenate([x_sub36, delta_sub.transpose(0, 2, 1)], axis=1).copy()
    target_sub = (clean_sub - base_sub).transpose(0, 2, 1).astype(np.float32)
    weight_sub = np.broadcast_to((1.0 - msk_sub).transpose(0, 2, 1),
                                 (len(msk_sub), n_channels, msk_sub.shape[1])).copy().astype(np.float32)

    # ---- S29-S32 冻结评估包 + B1 漂移断言（训练前）----
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    eval_pack = {}
    base_metrics = {}
    for sid in rrd.EVAL_SUBJECTS:
        segments, _, _ = prepare_data_db2(loader, [sid], config, exercises=[1])
        clean = segments.astype(np.float32)
        mask = tm._scenario_generator(config, rrd.EVAL_SEED_BASE + sid).generate_mask(
            torch.as_tensor(clean)).numpy().astype(np.float32)
        subj_cache: dict = {}
        ridge_pred = np.stack([
            ridge_dev.predict_window((clean[w] * mask[w]).astype(np.float64),
                                     mask[w], mu, sigma, False, None, subj_cache)
            for w in range(len(clean))])
        bases = []
        for w in range(len(clean)):
            feats_w = rrd.build_causal_features(
                ridge_pred[w], (clean[w] * mask[w]).astype(np.float64),
                mask[w].astype(np.float64), rrd.L_MAX)
            bases.append(ridge_pred[w] + feats_w @ w32)
        base_subj = np.stack(bases)
        delivered_base = rrd.deliver(base_subj, clean, mask, patch_size)
        met_b = full_metrics(delivered_base, clean, mask)
        base_metrics[str(sid)] = met_b
        eval_pack[sid] = {"clean": clean, "mask": mask,
                          "x36": rts.stack_inputs(base_subj, (clean * mask).astype(np.float64),
                                                  mask.astype(np.float64)),
                          "base": base_subj, "delivered_base": delivered_base}
        eval_pack[sid]["x48"] = np.concatenate(
            [eval_pack[sid]["x36"],
             time_since_last_obs(mask).transpose(0, 2, 1)], axis=1).copy()
        drift = abs(met_b["overall"]["nrmse"] - rts.E003_BASE32_OVERALL[sid])
        print(f"S{sid} B1={met_b['overall']['nrmse']:.6f} drift={drift:.2e}", flush=True)
        if drift >= DRIFT_TOL:
            raise RuntimeError(f"B1 drift tripwire fired for S{sid}: {drift:.3e}; stop before training.")
    base_eq_weight = float(np.mean([base_metrics[str(s)]["overall"]["nrmse"]
                                    for s in rrd.EVAL_SUBJECTS]))
    base_subject_overall = {str(s): float(base_metrics[str(s)]["overall"]["nrmse"])
                            for s in rrd.EVAL_SUBJECTS}
    anomalies.append("B1_base_eq_weight=" + f"{base_eq_weight:.12f}")

    # ---- 内部连续 (subject, repetition) 块 80/20 留出集（固定规则）----
    sub_subjects = pool_subjects[fit_idx]
    sub_reps = pool_reps[fit_idx]
    groups = sorted(set(zip(sub_subjects.tolist(), sub_reps.tolist())))
    holdout_groups = {g for i, g in enumerate(groups) if i % 5 == 4}
    is_hold = np.array([(s, r) in holdout_groups
                        for s, r in zip(sub_subjects.tolist(), sub_reps.tolist())])
    train_block = ~is_hold
    print(f"internal split: {len(groups)} groups, holdout groups={len(holdout_groups)}, "
          f"train windows={int(train_block.sum())}, holdout windows={int(is_hold.sum())}",
          flush=True)
    gen_int = tm._scenario_generator(config, INTERNAL_MASK_SEED)
    holdout_windows = clean_sub[is_hold]
    holdout_masks = []
    for b0 in range(0, len(holdout_windows), rrd.MASK_BATCH):
        b1 = min(b0 + rrd.MASK_BATCH, len(holdout_windows))
        holdout_masks.append(gen_int.generate_mask(
            torch.as_tensor(holdout_windows[b0:b1])).numpy().astype(np.float32))
    holdout_masks = np.concatenate(holdout_masks)

    def pooled_nrmse(preds, clean, mask):
        delivered = rrd.deliver(preds, clean, mask, patch_size)
        return float(tm._masked_metrics(delivered, clean, mask)["nrmse_peak_1"])

    # ---- S1: rank 选择 -> 全子集重拟合 -> 最终评估 ----
    t_arm = time.time()
    s1_table = []
    for rank in S1_RANKS:
        t0 = time.time()
        model = fit_ssm(clean_sub[train_block], rank)
        preds = ssm_complete_stack(model, holdout_windows, holdout_masks)
        score = pooled_nrmse(preds, holdout_windows, holdout_masks)
        s1_table.append({"rank": rank, "holdout_nrmse": score, "rho": model["rho"],
                         "stabilized": model["stabilized"],
                         "fit_seconds": round(time.time() - t0, 1)})
        print(f"S1 rank {rank}: holdout={score:.4f} rho={model['rho']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    s1_best_rank = min(s1_table, key=lambda row: (row["holdout_nrmse"], row["rank"]))["rank"]
    s1_model = fit_ssm(clean_sub, s1_best_rank)
    s1_subjects, s1_preds_pack = {}, {}
    for sid in rrd.EVAL_SUBJECTS:
        pack = eval_pack[sid]
        preds = ssm_complete_stack(s1_model, pack["clean"], pack["mask"])
        delivered = rrd.deliver(preds, pack["clean"], pack["mask"], patch_size)
        s1_subjects[str(sid)] = full_metrics(delivered, pack["clean"], pack["mask"])
        s1_preds_pack[str(sid)] = torch.from_numpy(delivered.astype(np.float32))
    s1_eq = float(np.mean([s1_subjects[str(s)]["overall"]["nrmse"]
                           for s in rrd.EVAL_SUBJECTS]))
    s1_wins = int(sum(s1_subjects[str(s)]["overall"]["nrmse"] < base_subject_overall[str(s)]
                      for s in rrd.EVAL_SUBJECTS))
    s1_latency = measure_latency_ms(
        lambda: ssm_filter_window(s1_model,
                                  eval_pack[29]["clean"][0] * eval_pack[29]["mask"][0],
                                  eval_pack[29]["mask"][0]))
    torch.save(s1_preds_pack, tmp_preds / "predictions_S1_kalman.pt")
    s1_runtime = round(time.time() - t_arm, 1)
    print(f"S1 final: rank={s1_best_rank} eq={s1_eq:.6f} wins_vs_B1={s1_wins}/4 "
          f"latency={s1_latency:.1f}ms ({s1_runtime:.0f}s)", flush=True)

    # ---- S2: (rank, L2) 网格选择 -> 全子集重拟合 -> 最终评估 ----
    t_arm = time.time()
    s2_table = []
    x_train_block = np.ascontiguousarray(
        clean_sub[train_block].transpose(2, 0, 1).reshape(n_channels, -1))
    for rank in S2_RANKS:
        w_best, obj_best, init_objs = None, np.inf, []
        for init_seed in S2_NMF_INIT_SEEDS:
            w_mat, obj = fit_nmf(x_train_block, rank, init_seed)
            init_objs.append(obj)
            if obj < obj_best:
                w_best, obj_best = w_mat, obj
        for lam in S2_L2_GRID:
            preds = nmf_complete_stack(w_best, lam, holdout_windows, holdout_masks)
            score = pooled_nrmse(preds, holdout_windows, holdout_masks)
            s2_table.append({"rank": rank, "l2": lam, "init_objectives": init_objs,
                             "holdout_nrmse": score})
            print(f"S2 rank {rank} l2 {lam:g}: holdout={score:.4f}", flush=True)
    s2_best = min(s2_table, key=lambda row: (row["holdout_nrmse"], row["rank"], row["l2"]))
    x_full = np.ascontiguousarray(clean_sub.transpose(2, 0, 1).reshape(n_channels, -1))
    w_final, obj_final, final_init_objs = None, np.inf, []
    for init_seed in S2_NMF_INIT_SEEDS:
        w_mat, obj = fit_nmf(x_full, s2_best["rank"], init_seed)
        final_init_objs.append(obj)
        if obj < obj_final:
            w_final, obj_final = w_mat, obj
    s2_subjects, s2_preds_pack = {}, {}
    for sid in rrd.EVAL_SUBJECTS:
        pack = eval_pack[sid]
        preds = nmf_complete_stack(w_final, s2_best["l2"], pack["clean"], pack["mask"])
        delivered = rrd.deliver(preds, pack["clean"], pack["mask"], patch_size)
        s2_subjects[str(sid)] = full_metrics(delivered, pack["clean"], pack["mask"])
        s2_preds_pack[str(sid)] = torch.from_numpy(delivered.astype(np.float32))
    s2_eq = float(np.mean([s2_subjects[str(s)]["overall"]["nrmse"]
                           for s in rrd.EVAL_SUBJECTS]))
    s2_wins = int(sum(s2_subjects[str(s)]["overall"]["nrmse"] < base_subject_overall[str(s)]
                      for s in rrd.EVAL_SUBJECTS))
    s2_latency = measure_latency_ms(
        lambda: nmf_complete_stack(w_final, s2_best["l2"],
                                   eval_pack[29]["clean"][:1], eval_pack[29]["mask"][:1]))
    torch.save(s2_preds_pack, tmp_preds / "predictions_S2_nmf.pt")
    s2_runtime = round(time.time() - t_arm, 1)
    print(f"S2 final: rank={s2_best['rank']} l2={s2_best['l2']:g} eq={s2_eq:.6f} "
          f"wins_vs_B1={s2_wins}/4 latency={s2_latency:.1f}ms ({s2_runtime:.0f}s)", flush=True)

    # ---- D2 / D3：每 seed 验证 A -> 训练 -> 验证 B -> 评估 ----
    def run_deep_arm(tag: str, model_ctor, x_train_np, x_eval_key: str):
        t_arm0 = time.time()
        x_t = torch.as_tensor(x_train_np)
        target_t = torch.as_tensor(target_sub)
        weight_t = torch.as_tensor(weight_sub)
        seed_reports = {}
        for seed in SEEDS:
            t_seed = time.time()
            torch.manual_seed(seed)
            model = model_ctor()
            with torch.no_grad():
                zero_out = model(torch.as_tensor(x_train_np[:64])).numpy()
            zero_ok = bool(np.all(zero_out == 0.0))
            d_a = rrd.deliver(val_base64 + zero_out.transpose(0, 2, 1).astype(np.float64),
                              val_clean64, val_mask64, patch_size)
            d_b = rrd.deliver(val_base64, val_clean64, val_mask64, patch_size)
            val_a = bool(zero_ok and np.array_equal(d_a, d_b))
            train_log: list = []
            model = sel.train_arm_seed(seed, model_ctor, x_t, target_t, weight_t, train_log)
            x_probe = eval_pack[rrd.EVAL_SUBJECTS[0]][x_eval_key][:64]
            pert = x_probe.copy()
            t_cut = 100
            pert[:, :, t_cut + 1:] += np.float32(0.37)
            o1 = forward_windows(model, x_probe)
            o2 = forward_windows(model, pert)
            val_b = bool(np.array_equal(o1[:, :, :t_cut + 1], o2[:, :, :t_cut + 1]))
            subj_report, preds_pack = {}, {}
            for sid in rrd.EVAL_SUBJECTS:
                pack = eval_pack[sid]
                res_out = forward_windows(model, pack[x_eval_key])
                completed = pack["base"] + res_out.transpose(0, 2, 1).astype(np.float64)
                delivered = rrd.deliver(completed, pack["clean"], pack["mask"], patch_size)
                subj_report[str(sid)] = full_metrics(delivered, pack["clean"], pack["mask"])
                preds_pack[str(sid)] = torch.from_numpy(delivered.astype(np.float32))
            eq_w = float(np.mean([subj_report[str(s)]["overall"]["nrmse"]
                                  for s in rrd.EVAL_SUBJECTS]))
            torch.save(preds_pack, tmp_preds / f"predictions_{tag}_seed{seed}.pt")
            seed_reports[str(seed)] = {
                "subjects": subj_report,
                "subject_equal_weight_overall": eq_w,
                "validation_A_zero_init_delivers_B1": val_a,
                "validation_B_causal_future_invariance": val_b,
                "final_train_loss": train_log[-1],
                "seed_runtime_seconds": round(time.time() - t_seed, 1),
            }
            print(f"{tag} seed {seed}: eq={eq_w:.4f} valA={val_a} valB={val_b} "
                  f"({time.time() - t_seed:.0f}s)", flush=True)
            latency_model = model
        eq_list = [seed_reports[str(s)]["subject_equal_weight_overall"] for s in SEEDS]
        subject_mean = {str(s): float(np.mean(
            [seed_reports[str(sd)]["subjects"][str(s)]["overall"]["nrmse"] for sd in SEEDS]))
            for s in rrd.EVAL_SUBJECTS}
        wins = int(sum(subject_mean[str(s)] < base_subject_overall[str(s)]
                       for s in rrd.EVAL_SUBJECTS))
        probe_pack = eval_pack[rrd.EVAL_SUBJECTS[0]]
        latency = measure_latency_ms(
            lambda: forward_windows(latency_model, probe_pack[x_eval_key][:1]))
        return {"seeds": seed_reports, "eq_weight_mean": float(np.mean(eq_list)),
                "eq_weight_std_sample": float(np.std(eq_list, ddof=1)),
                "subject_mean_overall": subject_mean, "subject_wins_vs_B1": wins,
                "latency_ms": latency, "runtime_seconds": round(time.time() - t_arm0, 1),
                "all_valA": all(seed_reports[str(s)]["validation_A_zero_init_delivers_B1"]
                                for s in SEEDS),
                "all_valB": all(seed_reports[str(s)]["validation_B_causal_future_invariance"]
                                for s in SEEDS)}

    d2_report = run_deep_arm(
        "D2_transformer",
        lambda: CausalLocalTransformerResidual(
            in_ch=36, d_model=16, nhead=4, ffn=32, dropout=0.1, n_layers=2,
            history=TRANSFORMER_HISTORY, out_ch=n_channels),
        x_sub36, "x36")
    print(f"D2: mean={d2_report['eq_weight_mean']:.6f} sd={d2_report['eq_weight_std_sample']:.6f} "
          f"wins_vs_B1={d2_report['subject_wins_vs_B1']}/4 latency={d2_report['latency_ms']:.1f}ms",
          flush=True)
    d3_report = run_deep_arm(
        "D3_grud",
        lambda: GRUDStyleResidual(in_ch=48, hidden=16, out_ch=n_channels),
        x_sub48, "x48")
    print(f"D3: mean={d3_report['eq_weight_mean']:.6f} sd={d3_report['eq_weight_std_sample']:.6f} "
          f"wins_vs_B1={d3_report['subject_wins_vs_B1']}/4 latency={d3_report['latency_ms']:.1f}ms",
          flush=True)

    # ---- 预声明 winner 判定 ----
    s1_winner = bool(s1_eq < B1_REF_LITERAL and s1_wins >= 3)
    s2_winner = bool(s2_eq < B1_REF_LITERAL and s2_wins >= 3)
    d2_winner = bool(d2_report["eq_weight_mean"] < D1_MEAN_LITERAL
                     and d2_report["subject_wins_vs_B1"] >= 3)
    d3_winner = bool(d3_report["eq_weight_mean"] < D1_MEAN_LITERAL
                     and d3_report["subject_wins_vs_B1"] >= 3)
    winners = [name for name, ok in (("S1", s1_winner), ("S2", s2_winner),
                                     ("D2", d2_winner), ("D3", d3_winner)) if ok]
    verdict = {"provisional_winners": winners,
               "no_winner_conclusion": None if winners else
                   "D1 (B1 + small causal TCN) 暂保留为开发集最佳候选；不宣称全局最优",
               "rule_S": f"eq_weight < {B1_REF_LITERAL} AND >=3/4 subjects beat B1 "
                         f"(B1 exact eq {base_eq_weight:.9f})",
               "rule_D": f"5-seed mean < {D1_MEAN_LITERAL} (D1 exact {D1_MEAN_EXACT}) "
                         f"AND >=3/4 subjects 5-seed mean beat B1"}
    print(f"verdict: winners={winners}", flush=True)

    report = {
        "meta": {
            "script": str(script_path), "code_sha256": code_sha, "run_dir": str(run_dir),
            "frozen_arms": {
                "S1": "causal linear-Gaussian SSM: PCA+AR(1) closed-form identification, "
                      "diag Q/R, causal Kalman filter (no smoother); rank in {4,6,8} "
                      "selected on internal contiguous (subject,repetition)-block 80/20 "
                      "holdout (every 5th group), refit on full 8000 windows",
                "S2": "NMF synergy (multiplicative updates, 300 iters, per-iter column "
                      "normalization, 3 fixed init seeds 0/1/2 by fit objective) + "
                      "per-timestep L2-regularized NNLS on visible channels; "
                      "rank {4,6,8} x L2 {1e-4,1e-3,1e-2} on same internal holdout, "
                      "refit on full 8000 windows",
                "D2": "B1 + causal local(64) Transformer residual: 2 layers, d_model=16, "
                      "4 heads, FFN=32, dropout=0.1, sinusoidal PE, zero-init head",
                "D3": "B1 + GRU-D-style residual: inputs [B1, observed, mask, "
                      "time-since-last-obs], 1-layer GRU hidden=16, zero-init head; "
                      "not a verbatim GRU-D reproduction"},
            "unified_conditions": {
                "training_windows": int(rrd.N_FIT_WINDOWS), "train_mask_seed": rrd.TRAIN_MASK_SEED,
                "lambda_head": rrd.LAMBDA_HEAD, "internal_mask_seed": INTERNAL_MASK_SEED,
                "eval_mask_seed_rule": "20260910 + subject_id",
                "delivery": "clip [0,1] -> patch crossfade -> observed backfill",
                "deep_training": "AdamW lr=1e-3, batch=64, 30 epochs, seeds 20260911-15, "
                                 "no early stopping / checkpoint selection",
                "references": {"B1_eq": B1_REF_LITERAL, "D1_mean": D1_MEAN_EXACT,
                               "D1_sd": 0.0003940080160778956}},
        },
        "B1_tripwire": {"tolerance": DRIFT_TOL,
                        "base32_overall": base_subject_overall,
                        "passed": True},
        "internal_split": {"rule": "groups = unique (subject, repetition) of the 8000 subset "
                                   "sorted ascending; holdout = every 5th group (index%5==4)",
                           "n_groups": len(groups), "n_holdout_groups": len(holdout_groups),
                           "train_windows": int(train_block.sum()),
                           "holdout_windows": int(is_hold.sum())},
        "arms": {
            "S1": {"selection_table": s1_table, "chosen_rank": s1_best_rank,
                   "final_model": {"rho": s1_model["rho"],
                                   "stabilized": s1_model["stabilized"],
                                   "top_eigvals": s1_model["top_eigvals"]},
                   "subjects": s1_subjects, "eq_weight": s1_eq,
                   "wins_vs_B1": s1_wins, "latency_ms": s1_latency,
                   "runtime_seconds": s1_runtime},
            "S2": {"selection_table": s2_table,
                   "chosen": {"rank": s2_best["rank"], "l2": s2_best["l2"]},
                   "final_init_objectives": final_init_objs,
                   "subjects": s2_subjects, "eq_weight": s2_eq,
                   "wins_vs_B1": s2_wins, "latency_ms": s2_latency,
                   "runtime_seconds": s2_runtime},
            "D2": d2_report, "D3": d3_report,
        },
        "verdict": verdict,
        "anomalies": anomalies,
        "runtime_seconds": round(time.time() - t_start, 1),
        "peak_working_set_mb": round(rts.peak_working_set_mb(), 1),
    }

    out_dir.mkdir(parents=True, exist_ok=False)
    for f in sorted(tmp_preds.iterdir()):
        shutil.move(str(f), str(out_dir / f.name))
    tmp_preds.rmdir()
    (out_dir / "backbone_deep_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"runtime={report['runtime_seconds']}s peak_ws={report['peak_working_set_mb']}MB",
          flush=True)
    print(f"results={out_dir / 'backbone_deep_screen.json'}", flush=True)


if __name__ == "__main__":
    main()
