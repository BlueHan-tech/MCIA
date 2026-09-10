"""WC-BQD (within-channel baseline quality detector) dev validation.

Standalone diagnostic: implements the bottom-up within-channel detector
(T1 plateau/clipping, T2 low-energy robust-z, T3 low-frequency burst) with
baselines fitted only from train repetitions 1/3/4 active seconds. The main
chain (utils/db3_quality_mask) is untouched.

Pre-declared acceptance criteria (fixed before running; verdicts use
HELD-OUT repetitions only, i.e. deployment-realistic flags from
train-fitted baselines):
  C1  healthy held-out pooled flag rate <= 5%
  C2a healthy |Pearson(per-channel flag rate, per-channel median RMS)| <= 0.3
  C2b healthy Ch11/Ch12 flag rate <= 2x pooled
  C3a DB3 held-out pooled flag rate <= 12% (vs MQP train-rep reference 18.58%)
  C3b DB3 per-channel max rate <= max(3x pooled, pooled+2pp); per-action within [0.5x,2x]
  C3c hard-zero channels: >= 95% of their held-out active seconds flagged
  C4  downstream S05: WCBQD-mask B validation RMSE <= current-mask B RMSE

Detector constants (pre-declared; v0.1 after v0's T1 caught benign
quantization duplicates on DB2 Ch3/Ch10):
  kappa_low=3.5 (Leys 2013), kappa_high=5.0, low_freq=20 Hz, power_frac=0.8,
  plateau: run>=10 at the second's P99.5 amplitude extremes (clipping) OR
  run>=100 samples (50 ms flatline); duplicate-fraction rule removed.

Usage: python test/diagnose_wcbqd_detector.py --stage healthy|db3|downstream
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch
import yaml

from data.ninapro_loader import NinaProDataLoader
# 单一来源：检测器原语与常量全部来自主模块，避免验证脚本与主链路漂移。
from utils.db3_quality_mask import (
    KAPPA_HIGH,
    KAPPA_LOW,
    LOW_FREQ_HZ,
    PLATEAU_DEV_PCTL,
    PLATEAU_HARD,
    PLATEAU_RUN,
    POWER_FRAC,
    TRAIN_REPS,
    _fit_baselines as fit_baselines,
    _runs,
    _wcbqd_flags,
    second_features,
)

DB3_ACTION_IDS = tuple(range(1, 49))


def wcbqd_flags(feats: dict, med: np.ndarray, mad: np.ndarray) -> dict:
    """主模块 _wcbqd_flags 的诊断包装：保留 z 与独立分测试明细用于成分分析。

    t2/t3 按"测试独立触发"口径统计（与 2026-09-09/10 验证记录一致，可与 t1
    重叠）；flag 一律取主模块的并集结果，并断言与本地重算一致——主模块
    实现漂移时该断言立即失败，保证验证始终代表主链路。
    """
    z = (feats["logrms"] - med) / mad
    t1 = feats["t1"]
    t2 = z < -KAPPA_LOW
    t3 = (z > KAPPA_HIGH) & (feats["lowfrac"] >= POWER_FRAC)
    flag = _wcbqd_flags(feats, med, mad)
    assert np.array_equal(flag, t1 | t2 | t3), \
        "main-module WC-BQD verdict diverged from diagnostic recomputation"
    return {"t1": t1, "t2": t2, "t3": t3, "z": z, "flag": flag}


def second_rms(sec: np.ndarray) -> np.ndarray:
    """诊断专用的逐秒 RMS（主模块特征不返回，用于激活相关性分析）。"""
    x = sec - sec.mean(axis=0, keepdims=True)
    return np.sqrt((x ** 2).mean(axis=0))


def iter_seconds(labels: np.ndarray, reps: np.ndarray, action_ids):
    if action_ids is None:
        valid = labels > 0
    else:
        valid = np.isin(labels, action_ids)
    for start, end in _runs(valid):
        for cursor in range(start, end - 2000 + 1, 2000):
            if np.all(reps[cursor:cursor + 2000] == reps[cursor]):
                yield cursor, int(labels[cursor + 1000]), int(reps[cursor])


def collect_subject(loader, subject_id, exercises, db: str):
    if db == "db2":
        data = loader.load_db2_subject(subject_id, exercises)
    else:
        data = loader.load_db3_subject(subject_id, exercises)
    emg = np.asarray(data["emg"], dtype=np.float64)
    labels = np.asarray(data.get("restimulus", data.get("stimulus"))).reshape(-1)
    reps = np.asarray(data["repetition"]).reshape(-1)
    n = min(len(emg), len(labels), len(reps))
    return emg[:n], labels[:n], reps[:n]


def run_detector(loader, subject_id, exercises, db: str, action_ids):
    """Per-second arrays from one subject; baselines from train reps only."""
    emg, labels, reps = collect_subject(loader, subject_id, exercises, db)
    centered = emg - emg.mean(axis=0, keepdims=True)
    rows = list(iter_seconds(labels, reps, action_ids))
    feats = [second_features(centered[c:c + 2000]) for c, _, _ in rows]
    rms_rows = [second_rms(centered[c:c + 2000]) for c, _, _ in rows]
    train_rows = [i for i, (_, _, rep) in enumerate(rows) if rep in TRAIN_REPS]
    if len(train_rows) < 4:
        raise RuntimeError(f"S{subject_id}: only {len(train_rows)} train-repetition seconds")
    med, mad = fit_baselines(np.stack([feats[i]["logrms"] for i in train_rows]))
    decisions = [wcbqd_flags(f, med, mad) for f in feats]
    hard_mask_train = np.isin(reps, TRAIN_REPS)
    if action_ids is None:
        valid_train = hard_mask_train & (labels > 0)
    else:
        valid_train = hard_mask_train & np.isin(labels, action_ids)
    hard = (np.all(emg[valid_train] == 0.0, axis=0)
            if valid_train.any() else np.zeros(12, dtype=bool))
    return {
        "flags": np.stack([d["flag"] for d in decisions]),
        "z": np.stack([d["z"] for d in decisions]),
        "t1": np.stack([d["t1"] for d in decisions]),
        "t2": np.stack([d["t2"] for d in decisions]),
        "t3": np.stack([d["t3"] for d in decisions]),
        "rms": np.stack(rms_rows),
        "action": np.asarray([a for _, a, _ in rows], dtype=np.int32),
        "rep": np.asarray([r for _, _, r in rows], dtype=np.int32),
        "hard": hard,
        "n_seconds": len(rows),
    }


def summarize(res: dict, heldout_only: bool, label: str) -> dict:
    mask = ~np.isin(res["rep"], TRAIN_REPS) if heldout_only else np.ones(len(res["rep"]), bool)
    F, R = res["flags"][mask], res["rms"][mask]
    A = res["action"][mask]
    ch_rate = F.mean(axis=0)
    pooled = float(F.mean())
    med_rms = np.median(R, axis=0)
    pearson = (float(np.corrcoef(ch_rate, med_rms)[0, 1])
               if np.std(ch_rate) > 1e-12 and np.std(med_rms) > 1e-12 else float("nan"))
    pa = np.array([F[A == a].mean() for a in np.unique(A)]) if len(np.unique(A)) else np.array([])
    flagged = F.sum()
    return {
        "label": label,
        "eval_scope": "heldout_reps" if heldout_only else "all_reps",
        "n_channel_seconds": int(F.size),
        "pooled_flag_rate": pooled,
        "per_channel_flag_rate": [f"{v:.2%}" for v in ch_rate],
        "ch_rate_raw": ch_rate.tolist(),
        "per_channel_spread_ratio": float(ch_rate.max() / max(ch_rate.min(), 1e-9)),
        "pearson_flagrate_vs_median_rms": pearson,
        "ch11_ch12_max_rate": float(max(ch_rate[10], ch_rate[11])),
        "per_action_rate_min": float(pa.min()) if len(pa) else None,
        "per_action_rate_max": float(pa.max()) if len(pa) else None,
        "test_composition": {k: float(res[k][mask].sum() / max(1, flagged))
                             for k in ("t1", "t2", "t3")},
        "hard_channels_one_based": (np.flatnonzero(res["hard"]) + 1).tolist(),
        "hard_channel_flag_rate": (float(F[:, res["hard"]].mean())
                                   if res["hard"].any() else None),
        "nonhard_pooled_flag_rate": (float(F[:, ~res["hard"]].mean())
                                     if (~res["hard"]).any() else None),
        "median_rms_per_channel": med_rms.tolist(),
    }


def stage_healthy(out_root: Path, subjects, tag: str = "") -> None:
    root = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = dict(root["signal"], **root["exp1_mcia"])
    loader = NinaProDataLoader(root["paths"]["db2"], root["paths"]["db3"], fs=cfg["orig_fs"])
    out = out_root / f"healthy{tag}"
    out.mkdir(parents=True, exist_ok=False)
    parts, per_subject = [], {}
    for sid in subjects:
        res = run_detector(loader, sid, [1, 2], "db2", None)
        parts.append(res)
        per_subject[f"S{sid:02d}"] = float(res["flags"].mean())
        print(f"S{sid:02d}: {res['n_seconds']}s flag={res['flags'].mean():.2%}", flush=True)
    merged = {k: np.concatenate([p[k] for p in parts])
              for k in ("flags", "z", "t1", "t2", "t3", "rms", "action", "rep")}
    merged["hard"] = np.any(np.stack([p["hard"] for p in parts]), axis=0)
    np.savez_compressed(out / "per_second.npz", **{k: merged[k] for k in
        ("flags", "rms", "action", "rep")})
    heldout = summarize(merged, True, "DB2 S01-S10 E1+E2")
    allsec = summarize(merged, False, "DB2 S01-S10 E1+E2")
    verdicts = {
        "C1_healthy_heldout_pooled_le_5pct": heldout["pooled_flag_rate"] <= 0.05,
        "C2a_abs_pearson_le_0.3": abs(heldout["pearson_flagrate_vs_median_rms"]) <= 0.3,
        "C2b_ch11_12_le_2x_pooled": heldout["ch11_ch12_max_rate"]
                                    <= 2 * heldout["pooled_flag_rate"] + 1e-9,
    }
    payload = {"protocol_note": "baselines fit from train reps 1/3/4 within each subject; "
                                "verdicts on held-out reps (2/5/6)",
               "heldout": heldout, "all_seconds": allsec,
               "per_subject_flag_rate_all_reps": per_subject,
               "predeclared_criteria": verdicts}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"heldout": {k: heldout[k] for k in
        ("pooled_flag_rate", "per_channel_flag_rate", "pearson_flagrate_vs_median_rms",
         "ch11_ch12_max_rate", "test_composition")}, "verdicts": verdicts}, indent=2))
    print(f"results={out / 'results.json'}", flush=True)


def stage_db3(out_root: Path) -> None:
    root = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = dict(root["signal"], **root["exp1_mcia"])
    loader = NinaProDataLoader(root["paths"]["db2"], root["paths"]["db3"], fs=cfg["orig_fs"])
    out = out_root / "db3"
    out.mkdir(parents=True, exist_ok=False)
    parts, per_subject, sids = [], {}, []
    for sid in (2, 3, 4, 5, 6, 7, 8, 9, 11):
        res = run_detector(loader, sid, [1, 2, 3], "db3", DB3_ACTION_IDS)
        parts.append(res)
        sids.append(sid)
        per_subject[f"S{sid:02d}"] = float(res["flags"].mean())
        print(f"S{sid:02d}: {res['n_seconds']}s flag={res['flags'].mean():.2%} "
              f"hard_ch={int(res['hard'].sum())}", flush=True)
    merged = {k: np.concatenate([p[k] for p in parts])
              for k in ("flags", "z", "t1", "t2", "t3", "rms", "action", "rep")}
    merged["hard"] = np.any(np.stack([p["hard"] for p in parts]), axis=0)
    np.savez_compressed(out / "per_second.npz", **{k: merged[k] for k in
        ("flags", "rms", "action", "rep")})
    heldout = summarize(merged, True, "DB3 9 subjects E1-E3")
    allsec = summarize(merged, False, "DB3 9 subjects E1-E3")
    pooled = heldout["pooled_flag_rate"]
    ch = np.asarray(heldout["ch_rate_raw"])
    # C3c 逐被试计算：每个被试自己的硬零通道在自己的 held-out 秒上的一致性。
    # （并集口径会被无硬零被试的同位置活通道稀释，不用于判定。）
    per_subject_hard = {}
    for sid, res in zip(sids, parts):
        if res["hard"].any():
            held = ~np.isin(res["rep"], TRAIN_REPS)
            per_subject_hard[f"S{sid:02d}"] = {
                "hard_channels_one_based": (np.flatnonzero(res["hard"]) + 1).tolist(),
                "heldout_flag_rate_on_own_hard": float(res["flags"][held][:, res["hard"]].mean()),
            }
    hard_min = (min(v["heldout_flag_rate_on_own_hard"] for v in per_subject_hard.values())
                if per_subject_hard else None)
    verdicts = {
        "C3a_db3_heldout_pooled_le_12pct": pooled <= 0.12,
        "C3b_channel_max_le_bound": bool(ch.max() <= max(3 * pooled, pooled + 0.02)),
        "C3b_action_within_band": bool(heldout["per_action_rate_min"] >= 0.5 * pooled
                                       and heldout["per_action_rate_max"] <= 2.0 * pooled),
        "C3c_hard_channels_ge_95pct_per_subject": (hard_min is None or hard_min >= 0.95),
    }
    payload = {"mqp_reference": "18.58% train-rep MQP flag rate (gronlund repro)",
               "heldout": heldout, "all_seconds": allsec,
               "per_subject_flag_rate_all_reps": per_subject,
               "per_subject_hard_consistency": per_subject_hard,
               "hard_min_heldout_consistency": hard_min,
               "union_mask_hard_rate_note": "heldout.hard_channel_flag_rate 使用跨被试并集掩码，"
                                            "被无硬零被试的同位置活通道稀释，不用于判定",
               "predeclared_criteria": verdicts}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"heldout": {k: heldout[k] for k in
        ("pooled_flag_rate", "per_channel_flag_rate", "per_channel_spread_ratio",
         "per_action_rate_min", "per_action_rate_max", "hard_channel_flag_rate",
         "test_composition")}, "verdicts": verdicts}, indent=2))
    print(f"results={out / 'results.json'}", flush=True)


def stage_downstream(out_root: Path) -> None:
    from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
    from utils.paper_pipeline import (build_mcia, complete_with_mask,
                                      flatten_pipeline_config, load_mcia_state_dict,
                                      load_yaml_config, set_seed)
    out = out_root / "downstream"
    out.mkdir(parents=True, exist_ok=False)
    config = flatten_pipeline_config(load_yaml_config(ROOT))
    config = dict(config)
    config["regressor_num_epochs"] = 40
    config["regressor_patience"] = 10
    device = config["device"]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    spec = importlib.util.spec_from_file_location(
        "wcbqd_exp3", ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py")
    exp3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp3)

    ckpt = Path(config["exp1_dir"]) / "checkpoints" / "best_model.pth"
    mcia = build_mcia(config, device)
    load_mcia_state_dict(mcia, ckpt, device)
    mcia.eval()

    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    set_seed(42)
    subject = 5
    raw_emg, angle, _, reps, window_meta = prepare_kinematics_data(
        loader, [subject], config, exercises=config["regressor_exercises"],
        db="db3", return_metadata=True)
    train_idx, val_idx, _test_idx = make_rep_split(reps)
    starts = np.asarray(window_meta["start"], dtype=np.int64)
    current_masks = np.asarray(window_meta["quality_mask"], dtype=np.float32)

    emg, labels, reps_raw = collect_subject(
        loader, subject, [int(e) for e in config["regressor_exercises"]], "db3")
    centered = emg - emg.mean(axis=0, keepdims=True)
    rows = list(iter_seconds(labels, reps_raw, DB3_ACTION_IDS))
    feats = [second_features(centered[c:c + 2000]) for c, _, _ in rows]
    train_rows = [i for i, (_, _, r) in enumerate(rows) if r in TRAIN_REPS]
    med, mad = fit_baselines(np.stack([feats[i]["logrms"] for i in train_rows]))
    valid_train = np.isin(reps_raw, TRAIN_REPS) & np.isin(labels, DB3_ACTION_IDS)
    hard = np.all(emg[valid_train] == 0.0, axis=0)
    keep = np.ones((len(emg) // 10, 12), dtype=np.float32)
    keep[:, hard] = 0.0
    for i, (cursor, _, _) in enumerate(rows):
        fl = wcbqd_flags(feats[i], med, mad)["flag"]
        if fl.any():
            lo = cursor // 10
            keep[lo:lo + 200, fl] = 0.0
    wcbqd_masks = np.empty_like(current_masks)
    for w, s in enumerate(starts):
        seg = keep[s:s + current_masks.shape[1]]
        if len(seg) < current_masks.shape[1]:
            seg = np.vstack([seg, np.ones((current_masks.shape[1] - len(seg), 12),
                                          dtype=np.float32)])
        wcbqd_masks[w] = seg
    print(f"mask missing: current={float((current_masks < 0.5).mean()):.2%} "
          f"wcbqd={float((wcbqd_masks < 0.5).mean()):.2%}", flush=True)

    dev_idx = np.concatenate([train_idx, val_idx])
    dev_target = torch.tensor(raw_emg[dev_idx], dtype=torch.float32, device=device)
    pools = {"raw": raw_emg.copy()}
    for name, masks in (("enhanced_current", current_masks), ("enhanced_wcbqd", wcbqd_masks)):
        mask_t = torch.tensor(masks[dev_idx], dtype=torch.float32, device=device)
        set_seed(42)
        completed = complete_with_mask(mcia, dev_target, mask_t).cpu().numpy()
        pool = raw_emg.copy()
        pool[dev_idx] = completed
        pools[name] = pool
        print(f"completion [{name}] done", flush=True)

    results = {}
    for condition, pool in pools.items():
        set_seed(42)
        model, best_val, epochs = exp3.train_tcn_on_emg(
            pool, angle, train_idx, val_idx, config, device)
        pred, tgt = exp3.predict_on_set(model, pool, angle, val_idx, config, device)
        subsets = exp3.evaluate_subsets(pred, tgt)
        results[condition] = {"best_val_loss": float(best_val), "epochs": int(epochs),
                              "validation_subsets": subsets}
        g = subsets["global"]
        print(f"[{condition}] best_val={best_val:.5f} epochs={epochs} "
              f"val RMSE={g['rmse']:.5f} MAE={g['mae']:.5f}", flush=True)
    verdict = {
        "C4_wcbqd_rmse_le_current": (results["enhanced_wcbqd"]["validation_subsets"]["global"]["rmse"]
                                     <= results["enhanced_current"]["validation_subsets"]["global"]["rmse"]),
    }
    payload = {"subject": subject, "tcn_budget": {"epochs": 40, "patience": 10},
               "results": results, "predeclared_criteria": verdict,
               "mask_missing_ratio": {
                   "current": float((current_masks < 0.5).mean()),
                   "wcbqd": float((wcbqd_masks < 0.5).mean())}}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"results={out / 'results.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("healthy", "db3", "downstream"))
    parser.add_argument("--subjects", default="",
                        help="comma-separated subject ids for the healthy stage")
    parser.add_argument("--tag", default="",
                        help="output subdir suffix (e.g. _ext for the power extension)")
    args = parser.parse_args()
    run = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    out_root = run / "06_diagnostics" / "wcbqd_validation_20260909"
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.stage == "healthy":
        subjects = ([int(v) for v in args.subjects.split(",") if v.strip()]
                    if args.subjects else list(range(1, 11)))
        stage_healthy(out_root, subjects, args.tag)
    elif args.stage == "db3":
        stage_db3(out_root)
    else:
        stage_downstream(out_root)
    print(f"stage={args.stage} script_sha256="
          f"{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} "
          f"elapsed={time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
