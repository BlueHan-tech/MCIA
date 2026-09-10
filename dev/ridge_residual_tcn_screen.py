"""Linear Dynamic Base + Causal TCN Residual —— 开发筛查（E-004）。

研究问题：小型神经残差能否稳定超过最强归纳式基准
Ridge + L32 causal linear residual（E-003 被试等权 masked NRMSE = 0.1496）。
非主方法接入：不改 scripts/、models/、config.yaml、协议或 MCIA；不读取
S33-S40 / DB3 / glove / 下游 TCN / SGMD-AAE / CP-WOPT；S29-S32 仅前向评估。

复用（import，不复制）：Ridge `predict_window`（dev/ridge_attribution.py）；
因果特征构造、统一交付与 E-003 冻结常量 `build_causal_features`/`deliver`
（dev/ridge_residual_diagnostic.py）；ScenarioMix 与 masked 指标
（scripts/run_task_matched_literature_baselines.py）；K_obs 分层
（dev/attribution_ladder.py）；配置加载（utils/paper_pipeline.py）。

base 构造：E-003 同一固定 8,000 窗子集（种子 20260910）、同一训练掩码种子
20260910、同一逐通道精确正态方程 L32 线性残差头（lambda=1e-3）；对 E-003
已记录数值内置漂移绊线（ridge 与 base 逐被试 overall 须复现，|delta|<1e-5，
否则在训练前中止）。

冻结神经候选：输入 base/observed=clean*mask/mask 共 36 通道；严格因果 TCN
hidden=16、kernel=2、dilation=[1,2,4,8,16]（感受野 32 点），输出 12 通道
非线性残差，输出层零初始化（训练起点严格等于 base）；仅人工掩码位置的
masked MSE；AdamW lr=1e-3、batch=64、固定 30 epochs；5 seeds
20260911-20260915 独立训练，无超参搜索/早停/checkpoint 选择/逐被试适配。
最终输出 = mask*observed + (1-mask)*completed（观测位不被网络改写）。

短验证（内置）：A 零初始化模型训练前交付输出与 base 逐元素一致；
B 扰动 t 之后所有输入，t 之前 TCN 输出逐元素不变（训练后模型）；
C py311 编译（运行前执行）。

判定纪律（预声明）：成功 = 5/5 seed 被试等权 overall NRMSE < 0.1496 且
每 seed 至少 3/4 被试优于 L32 base；否则失败，失败时不再扩网/加图/改损失/
延长训练/换参数。无论成败不进入 S33-S40、下游或主链路。
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
import torch.nn.functional as F

from data.ninapro_loader import NinaProDataLoader
from data.dataset_db2_emg import prepare_data_db2
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config

SEEDS = (20260911, 20260912, 20260913, 20260914, 20260915)
HIDDEN = 16
KERNEL = 2
DILATIONS = (1, 2, 4, 8, 16)
LR = 1e-3
BATCH = 64
EPOCHS = 30
SUCCESS_REF = 0.1496
MIN_SUBJECT_WINS = 3
JUDGEMENT_NOTE = ("success iff all 5 seeds subject-equal-weight overall NRMSE < 0.1496 "
                  "and each seed beats the L32 base on >=3/4 subjects")
# E-003 冻结记录（run_20260908_191852_1 .../ridge_residual_diagnostic_20260910.json）
E003_RIDGE_OVERALL = {29: 0.17413192987442017, 30: 0.16780000925064087,
                      31: 0.16546165943145752, 32: 0.17217318713665009}
E003_BASE32_OVERALL = {29: 0.15521451830863953, 30: 0.1445293426513672,
                       31: 0.14722225069999695, 32: 0.15145745873451233}
DRIFT_TOL = 1e-5


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def peak_working_set_mb() -> float:
    if os.name != "nt":
        return float("nan")
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
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
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    candidates = []
    try:
        candidates.append(ctypes.windll.psapi.GetProcessMemoryInfo)
    except AttributeError:
        pass
    try:
        candidates.append(ctypes.windll.kernel32.K32GetProcessMemoryInfo)
    except AttributeError:
        pass
    for fn in candidates:
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        if fn(handle, ctypes.byref(pmc), pmc.cb):
            return float(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0)
    return float("nan")


class CausalConv1d(torch.nn.Module):
    """左零填充因果卷积：输出时间步 i 只依赖输入 <= i。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.left_pad = dilation * (kernel - 1)
        self.conv = torch.nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class CausalTCNResidual(torch.nn.Module):
    """36 通道输入 -> 5 层因果膨胀卷积（hidden=16, k=2, RF=32）-> 零初始化 12 通道残差头。"""

    def __init__(self, in_ch: int = 36, hidden: int = 16, out_ch: int = 12,
                 kernel: int = 2, dilations: tuple = DILATIONS):
        super().__init__()
        blocks = []
        ch = in_ch
        for d in dilations:
            blocks.append(CausalConv1d(ch, hidden, kernel, d))
            blocks.append(torch.nn.ReLU())
            ch = hidden
        self.body = torch.nn.Sequential(*blocks)
        self.head = torch.nn.Conv1d(hidden, out_ch, kernel_size=1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def stack_inputs(base: np.ndarray, obs: np.ndarray, msk: np.ndarray) -> np.ndarray:
    """(N,T,12)x3 -> (N,36,T) float32，供 Conv1d 使用。"""
    return np.concatenate(
        [base.transpose(0, 2, 1), obs.transpose(0, 2, 1), msk.transpose(0, 2, 1)],
        axis=1).astype(np.float32)


def train_seed(seed: int, x: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
               log: list) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = CausalTCNResidual()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    for epoch in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        for b0 in range(0, n, BATCH):
            idx = perm[b0:b0 + BATCH]
            opt.zero_grad()
            out = model(x[idx])
            loss = ((out - target[idx]) ** 2 * weight[idx]).sum() / weight[idx].sum()
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
        log.append(epoch_loss / (n // BATCH))
    return model


def forward_subject(model: torch.nn.Module, x_subj: np.ndarray) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for s in range(0, len(x_subj), 256):
            outs.append(model(torch.as_tensor(x_subj[s:s + 256])).numpy())
    return np.concatenate(outs)                      # (N, 12, T) float32


def main() -> None:
    t_start = time.time()
    tm = load("tm", PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py")
    ridge_dev = load("ridge_dev", PROJECT_ROOT / "dev" / "ridge_attribution.py")
    ladder = load("ladder", PROJECT_ROOT / "dev" / "attribution_ladder.py")
    rrd = load("rrd", PROJECT_ROOT / "dev" / "ridge_residual_diagnostic.py")
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    torch.set_num_threads(8)
    n_channels = int(config["n_channels"])
    patch_size = int(config["patch_size"])
    run_dir = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    out = run_dir / "06_diagnostics" / "ridge_residual_tcn_screen_20260910"

    # ---- base 冻结构造：E-003 同一子集/掩码种子/拟合方式（漂移断言）----
    cache_path = run_dir / "07_healthy_completion_benchmark" / "cache" / "db2_train_cache.pt"
    train_pool = torch.load(cache_path, map_location="cpu", weights_only=False)["data"].astype(np.float64)
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

    # 第二遍（同种子重建同掩码）：构造训练张量 base/observed/mask/target
    gen2 = tm._scenario_generator(config, rrd.TRAIN_MASK_SEED)
    base_list, obs_list, msk_list, clean_list = [], [], [], []
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
            base_list.append(pred_w + feats @ w32)
            obs_list.append(observed_w)
            msk_list.append(mask_w)
            clean_list.append(clean_w)
        print(f"base {min(b1, len(fit_windows))}/{len(fit_windows)} ({time.time() - t_fit:.0f}s)",
              flush=True)
    base_train = np.stack(base_list)
    obs_train = np.stack(obs_list)
    msk_train = np.stack(msk_list).astype(np.float32)
    clean_train = np.stack(clean_list)
    del base_list, obs_list, msk_list, clean_list, fit_windows, train_pool
    x_train = torch.as_tensor(stack_inputs(base_train, obs_train, msk_train))
    target_train = torch.as_tensor(
        (clean_train - base_train).transpose(0, 2, 1).astype(np.float32))
    weight_train = torch.as_tensor(
        np.broadcast_to((1.0 - msk_train).transpose(0, 2, 1),
                        (len(msk_train), n_channels, msk_train.shape[1])).copy())
    print(f"training tensors ready ({time.time() - t_start:.0f}s)", flush=True)

    # ---- S29-S32 前向评估数据与 base/L32、ridge 两个参照臂 ----
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    eval_pack = {}
    ridge_metrics, base_metrics = {}, {}
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
        obs_subj = (clean * mask).astype(np.float64)
        delivered_ridge = rrd.deliver(ridge_pred, clean, mask, patch_size)
        delivered_base = rrd.deliver(base_subj, clean, mask, patch_size)
        met_r = ladder.stratified_metrics(delivered_ridge, clean, mask)
        met_r.update(tm._masked_metrics(delivered_ridge, clean, mask))
        met_b = ladder.stratified_metrics(delivered_base, clean, mask)
        met_b.update(tm._masked_metrics(delivered_base, clean, mask))
        ridge_metrics[str(sid)] = met_r
        base_metrics[str(sid)] = met_b
        eval_pack[sid] = {"clean": clean, "mask": mask,
                          "x": stack_inputs(base_subj, obs_subj, mask.astype(np.float64)),
                          "base": base_subj, "delivered_base": delivered_base}
        drift_r = abs(met_r["overall"]["nrmse"] - E003_RIDGE_OVERALL[sid])
        drift_b = abs(met_b["overall"]["nrmse"] - E003_BASE32_OVERALL[sid])
        print(f"S{sid} ridge={met_r['overall']['nrmse']:.6f} base32={met_b['overall']['nrmse']:.6f} "
              f"drift r/b={drift_r:.2e}/{drift_b:.2e}", flush=True)
        if drift_r >= DRIFT_TOL or drift_b >= DRIFT_TOL:
            raise RuntimeError(
                f"E-003 drift tripwire fired for S{sid}: ridge {drift_r:.3e}, "
                f"base32 {drift_b:.3e} (tol {DRIFT_TOL}). Abort before training.")
    base_eq_weight = float(np.mean([base_metrics[str(s)]["overall"]["nrmse"]
                                    for s in rrd.EVAL_SUBJECTS]))

    # ---- 5 seeds：验证 A（零初始化）-> 训练 -> 验证 B（因果性）-> 评估 ----
    seed_reports, predictions_store = {}, {}
    val_a_first = None
    for seed in SEEDS:
        t_seed = time.time()
        torch.manual_seed(seed)
        model = CausalTCNResidual()
        # 验证 A：零初始化模型训练前输出 == base（逐元素）
        with torch.no_grad():
            out_zero = model(x_train[:64]).numpy()
        zero_ok = bool(np.all(out_zero == 0.0))
        b64 = base_train[:64]
        d_a = rrd.deliver(b64 + out_zero.transpose(0, 2, 1).astype(np.float64),
                          clean_train[:64], msk_train[:64], patch_size)
        d_b = rrd.deliver(b64, clean_train[:64], msk_train[:64], patch_size)
        val_a = bool(zero_ok and np.array_equal(d_a, d_b))
        if val_a_first is None:
            val_a_first = val_a
        train_log: list = []
        model = train_seed(seed, x_train, target_train, weight_train, train_log)
        # 验证 B：扰动 t 之后所有输入，t 之前输出不变（训练后模型）
        x_probe = eval_pack[rrd.EVAL_SUBJECTS[0]]["x"][:64]
        pert = x_probe.copy()
        t_cut = 100
        pert[:, :, t_cut + 1:] += np.float32(0.37)
        with torch.no_grad():
            o1 = model(torch.as_tensor(x_probe)).numpy()
            o2 = model(torch.as_tensor(pert)).numpy()
        val_b = bool(np.array_equal(o1[:, :, :t_cut + 1], o2[:, :, :t_cut + 1]))
        # 评估：completed = base + tcn 残差，统一交付后仅掩码位置计分
        subj_report, preds = {}, {}
        for sid in rrd.EVAL_SUBJECTS:
            pack = eval_pack[sid]
            res_out = forward_subject(model, pack["x"])
            completed = pack["base"] + res_out.transpose(0, 2, 1).astype(np.float64)
            delivered = rrd.deliver(completed, pack["clean"], pack["mask"], patch_size)
            met = ladder.stratified_metrics(delivered, pack["clean"], pack["mask"])
            met.update(tm._masked_metrics(delivered, pack["clean"], pack["mask"]))
            subj_report[str(sid)] = met
            delta = met["overall"]["nrmse"] - base_metrics[str(sid)]["overall"]["nrmse"]
            print(f"seed {seed} S{sid}: overall={met['overall']['nrmse']:.4f} "
                  f"delta_vs_base={delta:+.4f}", flush=True)
            preds[str(sid)] = torch.from_numpy(delivered.astype(np.float32))
        eq_weight = float(np.mean([subj_report[str(s)]["overall"]["nrmse"]
                                   for s in rrd.EVAL_SUBJECTS]))
        wins = int(sum(subj_report[str(s)]["overall"]["nrmse"]
                       < base_metrics[str(s)]["overall"]["nrmse"]
                       for s in rrd.EVAL_SUBJECTS))
        seed_reports[str(seed)] = {
            "subjects": subj_report,
            "subject_equal_weight_overall": eq_weight,
            "subject_wins_vs_base": wins,
            "deltas_vs_base_overall": {
                str(s): float(subj_report[str(s)]["overall"]["nrmse"]
                              - base_metrics[str(s)]["overall"]["nrmse"])
                for s in rrd.EVAL_SUBJECTS},
            "validation_A_zero_init_delivers_base": val_a,
            "validation_B_causal_future_invariance": val_b,
            "final_train_loss": train_log[-1],
            "seed_runtime_seconds": round(time.time() - t_seed, 1),
        }
        predictions_store[str(seed)] = preds
        print(f"seed {seed}: eq_weight={eq_weight:.4f} wins_vs_base={wins}/4 "
              f"valA={val_a} valB={val_b} ({time.time() - t_seed:.0f}s)", flush=True)

    eq_list = [seed_reports[str(s)]["subject_equal_weight_overall"] for s in SEEDS]
    wins_list = [seed_reports[str(s)]["subject_wins_vs_base"] for s in SEEDS]
    success = bool(all(e < SUCCESS_REF for e in eq_list)
                   and all(w >= MIN_SUBJECT_WINS for w in wins_list))
    print(f"judgement: success={success} eq={ [round(e, 4) for e in eq_list] } "
          f"wins={wins_list} (ref {SUCCESS_REF}, base_exact {base_eq_weight:.6f})",
          flush=True)

    report = {
        "meta": {
            "script": str(Path(__file__).resolve()),
            "run_dir": str(run_dir),
            "frozen_config": {
                "inputs": ["base=ridge+L32_linear_causal_residual", "observed=clean*mask", "mask"],
                "input_channels": 3 * n_channels,
                "hidden": HIDDEN, "kernel": KERNEL, "dilations": list(DILATIONS),
                "receptive_field": 1 + sum(DILATIONS) * (KERNEL - 1),
                "output": "12-channel nonlinear residual, zero-initialized head",
                "loss": "masked MSE on artificial mask positions only",
                "optimizer": "AdamW", "lr": LR, "batch": BATCH, "epochs": EPOCHS,
                "seeds": list(SEEDS),
                "training_data": {
                    "pool_windows": int(rrd.N_FIT_WINDOWS), "subset_seed": rrd.SUBSET_SEED,
                    "mask_seed": rrd.TRAIN_MASK_SEED, "lambda_head": rrd.LAMBDA_HEAD},
                "eval": {"subjects": list(rrd.EVAL_SUBJECTS),
                         "mask_seed_rule": "20260910 + subject_id (frozen ScenarioMix)",
                         "delivery": "clip [0,1] -> patch crossfade -> observed backfill",
                         "output_rule": "mask*observed + (1-mask)*completed"},
            },
            "judgement_rule": JUDGEMENT_NOTE,
            "references": {"ridge_eq_weight": 0.1699, "base_L32_eq_weight": 0.1496,
                           "mcia_eq_weight_historical_only": 0.1773},
            "base_consistency_vs_E003": {
                "tolerance": DRIFT_TOL,
                "ridge_overall": {str(s): float(ridge_metrics[str(s)]["overall"]["nrmse"])
                                  for s in rrd.EVAL_SUBJECTS},
                "base32_overall": {str(s): float(base_metrics[str(s)]["overall"]["nrmse"])
                                   for s in rrd.EVAL_SUBJECTS},
                "passed": True},
        },
        "ridge_arm": ridge_metrics,
        "base_L32_arm": base_metrics,
        "base_L32_eq_weight_this_run": base_eq_weight,
        "seeds": seed_reports,
        "aggregate": {
            "eq_weight_overall_mean": float(np.mean(eq_list)),
            "eq_weight_overall_std_sample": float(np.std(eq_list, ddof=1)),
            "per_seed_eq_weight": {str(s): seed_reports[str(s)]["subject_equal_weight_overall"]
                                   for s in SEEDS},
        },
        "judgement": {"success": success, "ref": SUCCESS_REF,
                      "all_seeds_below_ref": bool(all(e < SUCCESS_REF for e in eq_list)),
                      "min_subject_wins": wins_list},
        "runtime_seconds": round(time.time() - t_start, 1),
        "peak_working_set_mb": round(peak_working_set_mb(), 1),
    }

    out.mkdir(parents=True, exist_ok=False)
    torch.save({str(s): torch.from_numpy(eval_pack[s]["delivered_base"].astype(np.float32))
                for s in rrd.EVAL_SUBJECTS}, out / "predictions_base.pt")
    for seed in SEEDS:
        torch.save(predictions_store[str(seed)], out / f"predictions_seed{seed}.pt")
    (out / "ridge_residual_tcn_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"runtime={report['runtime_seconds']}s peak_ws={report['peak_working_set_mb']}MB",
          flush=True)
    print(f"results={out / 'ridge_residual_tcn_screen.json'}", flush=True)


if __name__ == "__main__":
    main()
