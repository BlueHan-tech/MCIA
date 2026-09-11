"""Ridge + 因果残差 有限开发筛选（E-005）：四个预注册臂，找唯一待冻结配置或保留 E-004。

仅用 DB2 开发被试 S29-S32；不读取/运行/生成 S33-S40、DB3、下游 TCN、SGMD-AAE、
CP-WOPT 的任何结果。非主方法接入：不改 scripts/、models/、config.yaml、协议、MCIA。

预注册四臂（除此之外无任何候选；不以中间结果删 seed/改 epoch/增减层/改损失）：
  A  E-004 基线：8,000 训练窗（E-003 冻结子集），RF=32，hidden=16；
  B  仅扩大训练数据：全部 28,218 个 DB2 训练窗，RF=32，hidden=16；
  C  在 B 上仅扩大因果历史：全池，RF=64（dilation 1..32），hidden=16；
  D  在 C 上仅扩大残差容量：全池，RF=64，hidden=32。

冻结条件（与 E-003/E-004 完全一致）：
  base = Ridge + L32 因果线性残差头（同 8,000 窗子集、训练掩码种子 20260910、
  lambda=1e-3、逐通道精确正态方程；对 E-003 记录内置漂移断言）；交付 clip ->
  patch crossfade -> observed 回填；ScenarioMix 评估掩码 seed=20260910+sid；
  masked NRMSE（峰值 1）与 K_obs 分层同 E-001/002/003/004。TCN 仅预测 base 残差，
  输入 [base, observed, mask]，严格因果，观测位硬回填；AdamW lr=1e-3、batch=64、
  30 epochs、无 early stopping/checkpoint 选择；每臂同 5 seeds 20260911-20260915。

臂 B/C/D 的扩池训练掩码规则（预声明）：E-003/E-004 的 8,000 子集窗沿用同一
生成器流的既有实现（逐位一致）；其余 20,218 窗用同一生成器实例的续流（同一
seed 20260910 的 ScenarioMix 实现的自然扩展），保证共享子集掩码与 E-004 相同。

复用（import，不复制）：Ridge `predict_window`（dev/ridge_attribution.py）；
因果特征/交付/常量（dev/ridge_residual_diagnostic.py）；TCN 结构、训练循环、
输入堆叠、前向、峰值内存、seed 与 E-003 断言常量（dev/ridge_residual_tcn_screen.py）；
掩码与 masked 指标（scripts/run_task_matched_literature_baselines.py）；K_obs 分层
（dev/attribution_ladder.py）；配置（utils/paper_pipeline.py）。

选择规则（预声明）：(2) 取被试等权 5-seed 均值最低的臂；(3) 若最佳臂相对
E-004 的平均改善 <= E-004 五 seed sample SD（0.000394008），视为复杂化证据不足，
保留 E-004；(4) 若最佳臂未在 >=3/4 被试上呈现平均同方向改善，也保留 E-004；
(5) 完成四臂即停止，不因结果接近继续调参。判定输出唯一待冻结候选或保留 E-004。
"""

from __future__ import annotations

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

from data.dataset_db2_emg import prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config

ARMS = (
    {"name": "A_e004_baseline_8k_rf32_h16", "hidden": 16,
     "dilations": (1, 2, 4, 8, 16), "data": "subset"},
    {"name": "B_full_pool_rf32_h16", "hidden": 16,
     "dilations": (1, 2, 4, 8, 16), "data": "pool"},
    {"name": "C_full_pool_rf64_h16", "hidden": 16,
     "dilations": (1, 2, 4, 8, 16, 32), "data": "pool"},
    {"name": "D_full_pool_rf64_h32", "hidden": 32,
     "dilations": (1, 2, 4, 8, 16, 32), "data": "pool"},
)
# E-004 冻结记录（run_20260908_191852_1 .../ridge_residual_tcn_screen_20260910.json）
E004_SEED_EQ = {20260911: 0.1489589773118496, 20260912: 0.14930500835180283,
                20260913: 0.1484975963830948, 20260914: 0.1488642357289791,
                20260915: 0.14830394089221954}
E004_MEAN = 0.14878595173358916
E004_SD = 0.0003940080160778956
ARM_A_TOL = 1e-6
ARM_A_HARD_TOL = 1e-4


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_arm_seed(seed, model_ctor, x, target, weight, log):
    """与 rts.train_seed（E-004 冻结训练循环）逐行一致：RNG 消费顺序、批序、损失
    公式、优化器与超参全部相同，仅把模型构造参数化为 model_ctor 以支持预注册的
    RF/hidden 变体（E-004 helper 硬编码默认构造，无法传入变体）。默认构造路径与
    E-004 bit 级一致，由臂 A 对 E-004 记录的逐 seed 断言把关。
    """
    torch.manual_seed(seed)
    model = model_ctor()
    opt = torch.optim.AdamW(model.parameters(), lr=rts.LR)
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    for epoch in range(rts.EPOCHS):
        perm = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        for b0 in range(0, n, rts.BATCH):
            idx = perm[b0:b0 + rts.BATCH]
            opt.zero_grad()
            out = model(x[idx])
            loss = ((out - target[idx]) ** 2 * weight[idx]).sum() / weight[idx].sum()
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
        log.append(epoch_loss / (n // rts.BATCH))
    return model


tm = load("tm", PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py")
ridge_dev = load("ridge_dev", PROJECT_ROOT / "dev" / "ridge_attribution.py")
ladder = load("ladder", PROJECT_ROOT / "dev" / "attribution_ladder.py")
rrd = load("rrd", PROJECT_ROOT / "dev" / "ridge_residual_diagnostic.py")
rts = load("rts", PROJECT_ROOT / "dev" / "ridge_residual_tcn_screen.py")


def main() -> None:
    t_start = time.time()
    seeds = rts.SEEDS
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    torch.set_num_threads(8)
    n_channels = int(config["n_channels"])
    steps = int(config["window_size"])
    patch_size = int(config["patch_size"])
    run_dir = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    out = run_dir / "06_diagnostics" / "ridge_residual_tcn_select_20260910"
    tmp_preds = Path(tempfile.mkdtemp(prefix="rrtcn_select_preds_"))
    anomalies: list = []

    # ---- base 冻结构造（与 E-003/E-004 逐位一致）+ 全池训练张量 ----
    cache_path = run_dir / "07_healthy_completion_benchmark" / "cache" / "db2_train_cache.pt"
    train_pool = torch.load(cache_path, map_location="cpu", weights_only=False)["data"].astype(np.float64)
    pool_len = len(train_pool)
    flat = train_pool.reshape(-1, n_channels)
    mu = flat.mean(axis=0)
    sigma = np.cov(flat.T)
    del flat
    subset_rng = np.random.default_rng(rrd.SUBSET_SEED)
    fit_idx = np.sort(subset_rng.choice(pool_len, size=rrd.N_FIT_WINDOWS, replace=False))
    comp_idx = np.setdiff1d(np.arange(pool_len), fit_idx)
    fit_windows = train_pool[fit_idx]
    comp_windows = train_pool[comp_idx]

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

    # 子集第二遍（同种子重建同掩码，与 E-004 相同构造路径）
    def build_batch(clean_b, mask_b):
        base_rows, obs_rows, msk_rows, cln_rows = [], [], [], []
        for i in range(len(clean_b)):
            clean_w = clean_b[i]
            mask_w = mask_b[i]
            observed_w = clean_w * mask_w
            pred_w = ridge_dev.predict_window(observed_w, mask_w, mu, sigma,
                                              False, None, ridge_cache)
            feats = rrd.build_causal_features(pred_w, observed_w, mask_w, rrd.L_MAX)
            base_rows.append(pred_w + feats @ w32)
            obs_rows.append(observed_w)
            msk_rows.append(mask_w)
            cln_rows.append(clean_w)
        return (np.stack(base_rows), np.stack(obs_rows),
                np.stack(msk_rows), np.stack(cln_rows))

    gen2 = tm._scenario_generator(config, rrd.TRAIN_MASK_SEED)
    base_sub, obs_sub, msk_sub, clean_sub = [], [], [], []
    for b0 in range(0, len(fit_windows), rrd.MASK_BATCH):
        b1 = min(b0 + rrd.MASK_BATCH, len(fit_windows))
        mask_b = gen2.generate_mask(torch.as_tensor(fit_windows[b0:b1])).numpy()
        bb, oo, mm, cc = build_batch(fit_windows[b0:b1], mask_b)
        base_sub.append(bb); obs_sub.append(oo); msk_sub.append(mm); clean_sub.append(cc)
        print(f"subset base {min(b1, len(fit_windows))}/{len(fit_windows)} "
              f"({time.time() - t_fit:.0f}s)", flush=True)
    base_sub = np.concatenate(base_sub); obs_sub = np.concatenate(obs_sub)
    msk_sub = np.concatenate(msk_sub); clean_sub = np.concatenate(clean_sub)
    x_sub = rts.stack_inputs(base_sub, obs_sub, msk_sub)
    target_sub = (clean_sub - base_sub).transpose(0, 2, 1).astype(np.float32)
    weight_sub = np.broadcast_to((1.0 - msk_sub).transpose(0, 2, 1),
                                 (len(msk_sub), n_channels, steps)).copy().astype(np.float32)
    # 固定验证批（所有臂统一使用，取子集前 64 窗；copy 避免切片持引用阻塞大数组释放）
    val_clean64 = clean_sub[:64].copy()
    val_mask64 = msk_sub[:64].copy()
    val_base64 = base_sub[:64].copy()

    # 全池张量：子集行复制，补集行用同一生成器续流掩码计算
    x_all = np.empty((pool_len, 3 * n_channels, steps), dtype=np.float32)
    target_all = np.empty((pool_len, n_channels, steps), dtype=np.float32)
    weight_all = np.empty((pool_len, n_channels, steps), dtype=np.float32)
    x_all[fit_idx] = x_sub
    target_all[fit_idx] = target_sub
    weight_all[fit_idx] = weight_sub
    del base_sub, obs_sub, msk_sub, clean_sub, x_sub, target_sub, weight_sub
    for b0 in range(0, len(comp_windows), rrd.MASK_BATCH):
        b1 = min(b0 + rrd.MASK_BATCH, len(comp_windows))
        mask_b = gen2.generate_mask(torch.as_tensor(comp_windows[b0:b1])).numpy()
        bb, oo, mm, cc = build_batch(comp_windows[b0:b1], mask_b)
        x_all[comp_idx[b0:b1]] = rts.stack_inputs(bb, oo, mm)
        target_all[comp_idx[b0:b1]] = (cc - bb).transpose(0, 2, 1).astype(np.float32)
        weight_all[comp_idx[b0:b1]] = np.broadcast_to(
            (1.0 - mm).transpose(0, 2, 1), (b1 - b0, n_channels, steps)).copy().astype(np.float32)
        print(f"pool base {min(b1, len(comp_windows))}/{len(comp_windows)} "
              f"({time.time() - t_fit:.0f}s)", flush=True)
    del fit_windows, comp_windows, train_pool
    data_pack = {
        "subset": (torch.as_tensor(x_all[fit_idx]), torch.as_tensor(target_all[fit_idx]),
                   torch.as_tensor(weight_all[fit_idx])),
        "pool": (torch.as_tensor(x_all), torch.as_tensor(target_all),
                 torch.as_tensor(weight_all)),
    }
    print(f"training tensors ready ({time.time() - t_start:.0f}s)", flush=True)

    # ---- S29-S32 冻结评估包 + E-003 漂移断言 ----
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
        obs_subj = (clean * mask).astype(np.float64)
        delivered_base = rrd.deliver(base_subj, clean, mask, patch_size)
        met_b = ladder.stratified_metrics(delivered_base, clean, mask)
        met_b.update(tm._masked_metrics(delivered_base, clean, mask))
        base_metrics[str(sid)] = met_b
        eval_pack[sid] = {"clean": clean, "mask": mask,
                          "x": rts.stack_inputs(base_subj, obs_subj, mask.astype(np.float64)),
                          "base": base_subj, "delivered_base": delivered_base}
        drift = abs(met_b["overall"]["nrmse"] - rts.E003_BASE32_OVERALL[sid])
        print(f"S{sid} base32={met_b['overall']['nrmse']:.6f} drift={drift:.2e}", flush=True)
        if drift >= rts.DRIFT_TOL:
            raise RuntimeError(f"E-003 base drift tripwire fired for S{sid}: {drift:.3e}")
    base_eq_weight = float(np.mean([base_metrics[str(s)]["overall"]["nrmse"]
                                    for s in rrd.EVAL_SUBJECTS]))
    anomalies.append(f"base32_drift_max={max(abs(base_metrics[str(s)]['overall']['nrmse'] - rts.E003_BASE32_OVERALL[s]) for s in rrd.EVAL_SUBJECTS):.2e}")

    # ---- 四臂 x 5 seeds：验证 A -> 训练 -> 验证 B -> 评估 ----
    arm_reports = {}
    for arm in ARMS:
        x_t, target_t, weight_t = data_pack[arm["data"]]
        arm_name = arm["name"]
        seed_reports = {}
        for seed in seeds:
            t_seed = time.time()
            torch.manual_seed(seed)
            model = rts.CausalTCNResidual(hidden=arm["hidden"], dilations=arm["dilations"])
            with torch.no_grad():
                out_zero = model(torch.as_tensor(
                    data_pack["subset"][0][:64])).numpy()
            zero_ok = bool(np.all(out_zero == 0.0))
            d_a = rrd.deliver(val_base64 + out_zero.transpose(0, 2, 1).astype(np.float64),
                              val_clean64, val_mask64, patch_size)
            d_b = rrd.deliver(val_base64, val_clean64, val_mask64, patch_size)
            val_a = bool(zero_ok and np.array_equal(d_a, d_b))
            train_log: list = []
            model = train_arm_seed(
                seed,
                lambda: rts.CausalTCNResidual(hidden=arm["hidden"],
                                              dilations=arm["dilations"]),
                x_t, target_t, weight_t, train_log)
            x_probe = eval_pack[rrd.EVAL_SUBJECTS[0]]["x"][:64]
            pert = x_probe.copy()
            t_cut = 100
            pert[:, :, t_cut + 1:] += np.float32(0.37)
            with torch.no_grad():
                o1 = model(torch.as_tensor(x_probe)).numpy()
                o2 = model(torch.as_tensor(pert)).numpy()
            val_b = bool(np.array_equal(o1[:, :, :t_cut + 1], o2[:, :, :t_cut + 1]))
            subj_report, preds = {}, {}
            for sid in rrd.EVAL_SUBJECTS:
                pack = eval_pack[sid]
                res_out = rts.forward_subject(model, pack["x"])
                completed = pack["base"] + res_out.transpose(0, 2, 1).astype(np.float64)
                delivered = rrd.deliver(completed, pack["clean"], pack["mask"], patch_size)
                met = ladder.stratified_metrics(delivered, pack["clean"], pack["mask"])
                met.update(tm._masked_metrics(delivered, pack["clean"], pack["mask"]))
                subj_report[str(sid)] = met
                preds[str(sid)] = torch.from_numpy(delivered.astype(np.float32))
            eq_weight = float(np.mean([subj_report[str(s)]["overall"]["nrmse"]
                                       for s in rrd.EVAL_SUBJECTS]))
            torch.save(preds, tmp_preds / f"predictions_{arm_name}_seed{seed}.pt")
            seed_reports[str(seed)] = {
                "subjects": subj_report,
                "subject_equal_weight_overall": eq_weight,
                "deltas_vs_base_overall": {
                    str(s): float(subj_report[str(s)]["overall"]["nrmse"]
                                  - base_metrics[str(s)]["overall"]["nrmse"])
                    for s in rrd.EVAL_SUBJECTS},
                "validation_A_zero_init_delivers_base": val_a,
                "validation_B_causal_future_invariance": val_b,
                "final_train_loss": train_log[-1],
                "seed_runtime_seconds": round(time.time() - t_seed, 1),
            }
            print(f"{arm_name} seed {seed}: eq={eq_weight:.4f} valA={val_a} valB={val_b} "
                  f"({time.time() - t_seed:.0f}s)", flush=True)
            if arm_name.startswith("A_"):
                diff = abs(eq_weight - E004_SEED_EQ[seed])
                anomalies.append(f"armA_seed{seed}_vs_E004_diff={diff:.2e}")
                if diff > ARM_A_HARD_TOL:
                    raise RuntimeError(
                        f"Arm A drifted from E-004 for seed {seed}: {diff:.3e}")
        eq_list = [seed_reports[str(s)]["subject_equal_weight_overall"] for s in seeds]
        arm_reports[arm_name] = {
            "config": {"hidden": arm["hidden"], "dilations": list(arm["dilations"]),
                       "receptive_field": 1 + sum(arm["dilations"]),
                       "training_data": arm["data"],
                       "n_train_windows": int(x_t.shape[0])},
            "seeds": seed_reports,
            "eq_weight_mean": float(np.mean(eq_list)),
            "eq_weight_std_sample": float(np.std(eq_list, ddof=1)),
            "subject_mean_overall": {
                str(s): float(np.mean([seed_reports[str(sd)]["subjects"][str(s)]["overall"]["nrmse"]
                                       for sd in seeds])) for s in rrd.EVAL_SUBJECTS},
        }
        print(f"{arm_name}: mean={arm_reports[arm_name]['eq_weight_mean']:.6f} "
              f"sd={arm_reports[arm_name]['eq_weight_std_sample']:.6f}", flush=True)

    # ---- 预声明选择规则 ----
    arm_names = [a["name"] for a in ARMS]
    means = {name: arm_reports[name]["eq_weight_mean"] for name in arm_names}
    best = min(arm_names, key=lambda n: means[n])
    improvement = float(means[[a["name"] for a in ARMS if a["name"].startswith("A_")][0]] - means[best])
    a_name = [a["name"] for a in ARMS if a["name"].startswith("A_")][0]
    if best == a_name:
        selection = {"verdict": "keep_E004", "reason": "arm A (E-004 rerun) is lowest",
                     "best_arm": best}
    else:
        rule3_pass = bool(improvement > E004_SD)
        subj_mean_deltas = {
            s: arm_reports[best]["subject_mean_overall"][s]
            - arm_reports[a_name]["subject_mean_overall"][s]
            for s in arm_reports[a_name]["subject_mean_overall"]}
        wins = int(sum(1 for v in subj_mean_deltas.values() if v < 0))
        rule4_pass = bool(wins >= 3)
        if rule3_pass and rule4_pass:
            selection = {"verdict": "freeze_candidate", "best_arm": best,
                         "improvement_vs_E004": improvement,
                         "rule3_improvement_gt_E004_sd": rule3_pass,
                         "rule4_subject_wins": wins,
                         "subject_mean_deltas_vs_armA": subj_mean_deltas}
        else:
            selection = {"verdict": "keep_E004", "best_arm_by_mean": best,
                         "improvement_vs_E004": improvement,
                         "rule3_improvement_gt_E004_sd": rule3_pass,
                         "rule4_subject_wins": wins,
                         "subject_mean_deltas_vs_armA": subj_mean_deltas,
                         "reason": ("insufficient improvement (rule 3) " if not rule3_pass else "")
                                   + ("direction inconsistent on >=3/4 subjects (rule 4)"
                                      if not rule4_pass else "")}
    print(f"selection: {json.dumps(selection, ensure_ascii=False)}", flush=True)

    report = {
        "meta": {
            "script": str(Path(__file__).resolve()),
            "run_dir": str(run_dir),
            "preregistered_arms": [
                {"name": a["name"], "hidden": a["hidden"], "dilations": list(a["dilations"]),
                 "receptive_field": 1 + sum(a["dilations"]), "training_data": a["data"]}
                for a in ARMS],
            "frozen_conditions": {
                "base": "Ridge + L32 causal linear residual head, identical to E-003/E-004 "
                        "(8000-window subset, mask seed 20260910, lambda=1e-3, per-channel "
                        "normal equations)",
                "extra_pool_mask_rule": "complement windows use continuation of the same "
                                        "seed-20260910 ScenarioMix generator stream",
                "tcn": "input [base, observed, mask]; strictly causal; observed hard backfill; "
                       "masked MSE only; AdamW lr=1e-3, batch=64, 30 epochs, no early stop, "
                       "no checkpoint selection",
                "seeds": list(seeds),
                "eval": "S29-S32, ScenarioMix seed 20260910+sid, clip->crossfade->backfill, "
                        "masked NRMSE (peak 1) + K_obs strata"},
            "selection_rules": {
                "2": "lowest subject-equal-weight 5-seed mean wins",
                "3": f"improvement must exceed E-004 5-seed sample SD ({E004_SD})",
                "4": "best arm must improve on >=3/4 subjects on 5-seed means",
                "5": "stop after these four arms; no further tuning"},
            "references": {"E004_seed_eq": {str(k): v for k, v in E004_SEED_EQ.items()},
                           "E004_mean": E004_MEAN, "E004_sd": E004_SD,
                           "base_L32_eq_weight": base_eq_weight},
            "validations": "A zero-init delivers base / B future-perturbation invariance "
                           "(per arm per seed) / C py311 compile (pre-run)",
        },
        "base_L32_arm": base_metrics,
        "base_L32_eq_weight_this_run": base_eq_weight,
        "arms": arm_reports,
        "arm_eq_weight_means": means,
        "selection": selection,
        "anomalies": anomalies,
        "runtime_seconds": round(time.time() - t_start, 1),
        "peak_working_set_mb": round(rts.peak_working_set_mb(), 1),
    }

    out.mkdir(parents=True, exist_ok=False)
    torch.save({str(s): torch.from_numpy(eval_pack[s]["delivered_base"].astype(np.float32))
                for s in rrd.EVAL_SUBJECTS}, tmp_preds / "predictions_base.pt")
    for f in sorted(tmp_preds.iterdir()):
        shutil.move(str(f), str(out / f.name))
    tmp_preds.rmdir()
    (out / "ridge_residual_tcn_select.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"runtime={report['runtime_seconds']}s peak_ws={report['peak_working_set_mb']}MB",
          flush=True)
    print(f"results={out / 'ridge_residual_tcn_select.json'}", flush=True)


if __name__ == "__main__":
    main()
