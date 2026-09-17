"""协议 §7.5：健康 DB2 受控掩码补全基准。

在 DB2 留出测试被试（S33-S40）上，用有已知真值的合成 ScenarioMix 掩码，
一次性比较冻结的 MCIA、SGMD-AAE 与 CP-WOPT。
本脚本回答“哪种方法更好地重建已知信号”；DB3 任务适配脚本回答下游效用，
不得报告补全指标，因为真实掩码区域没有干净真值。

锁定规则（协议 §7.5）：
  - 基准被试：配置中的 DB2 测试被试（S33-S40），动作集 1，
    与 Exp1 预训练使用相同预处理；
  - 掩码：ScenarioMix 混合模式，每名被试的生成器种子为
    BENCHMARK_SEED + subject_id（预先声明并写入报告）；
  - 所有方法统一交付：裁剪至 [0,1] -> 补丁边界交叉淡化 -> 回填观测值
    （MCIA 调用 complete_with_mask，其余方法在此包装）；
  - SGMD-AAE 与 MCIA 使用相同 DB2 训练池和 ScenarioMix 进行预训练
    （复用 run_task_matched_literature_baselines._train_sgmd）；
  - CP-WOPT 在每名被试的人工遮挡观测上拟合，并标注为转导式处理；
  - 成功判据：被试等权的掩码区域 NRMSE 均值低于两个基线，
    且对每个基线分别至少赢得 4/8 名被试。

写入 <run>/07_healthy_completion_benchmark/。开发迭代仅限 S29-S32；
S33-S40 上的每次运行都须记录掩码种子、检查点和判定。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset_db2_emg import load_and_cache_data, prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from models.baselines.cp_wopt import CPWOPTConfig, complete_cp_wopt
from models.baselines.sgmd_aae import SGMDAAEGenerator
from utils.paper_pipeline import (
    build_mcia,
    complete_with_mask,
    flatten_pipeline_config,
    load_mcia_state_dict,
    load_yaml_config,
    patch_boundary_crossfade,
    safe_pearson_np,
    set_seed,
)
from utils.run_layout import get_run_dir, mark_step


def _load_task_matched_module():
    path = PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py"
    spec = importlib.util.spec_from_file_location("task_matched_shared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED = _load_task_matched_module()

BENCHMARK_SEED = 20260910          # 预先声明的逐被试掩码种子基值


def deliver_uniform(prediction: np.ndarray, observed_values: np.ndarray,
                    observed_mask: np.ndarray, patch_size: int) -> np.ndarray:
    """§7.5 统一交付：裁剪 -> patch 边界淡化 -> 复制回观测值。"""
    p = torch.from_numpy(np.clip(prediction, 0.0, 1.0)).float()
    p = patch_boundary_crossfade(p, patch_size)
    m = torch.from_numpy(observed_mask).float()
    v = torch.from_numpy(observed_values).float()
    return (p * (1.0 - m) + v * m).numpy()


def metrics_with_pearson(delivered: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> dict:
    out = dict(SHARED._masked_metrics(delivered, clean, mask))
    missing = mask < 0.5
    out["masked_pearson"] = safe_pearson_np(delivered[missing], clean[missing])
    return out


@torch.no_grad()
def predict_sgmd_raw(generator: SGMDAAEGenerator, clean: np.ndarray, mask: np.ndarray,
                     device: str, batch_size: int) -> np.ndarray:
    """SGMD 生成器的原始预测（输入已加掩码），尚未应用交付规则。"""
    generator.eval()
    out = np.empty_like(clean)
    for start in range(0, len(clean), batch_size):
        x = torch.as_tensor(clean[start:start + batch_size] * mask[start:start + batch_size],
                            dtype=torch.float32, device=device).unsqueeze(1)
        m = torch.as_tensor(mask[start:start + batch_size], dtype=torch.float32,
                            device=device).unsqueeze(1)
        out[start:start + len(x)] = generator(x, m).squeeze(1).cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol §7.5 healthy completion benchmark.")
    parser.add_argument("--sgmd-epochs", type=int, default=None)
    parser.add_argument("--sgmd-max-train-windows", type=int, default=None)
    parser.add_argument("--skip-sgmd-training", action="store_true")
    parser.add_argument("--mcia-checkpoint", type=Path, default=None)
    parser.add_argument("--subjects", default="",
                        help="comma-separated benchmark subjects; default uses DB2 test subjects S33-S40")
    parser.add_argument("--limit-windows-per-subject", type=int, default=0,
                        help="0 keeps all windows; any cap is an explicitly labelled development run")
    args = parser.parse_args()

    cfg = load_yaml_config(PROJECT_ROOT)
    settings = cfg["literature_baselines"]
    run_dir = get_run_dir(PROJECT_ROOT, cfg, create=True)
    os.environ["MCIA_RUN_DIR"] = str(run_dir)
    config = flatten_pipeline_config(cfg)
    device = config["device"]
    out_dir = run_dir / "07_healthy_completion_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    mark_step(run_dir, "healthy_completion_benchmark", "running")

    seed = int(settings["seed"])
    subjects = ([int(v) for v in args.subjects.split(",") if v.strip()]
                if args.subjects else list(config["test_subjects"]))
    patch_size = int(config["patch_size"])
    dev_run = bool(args.limit_windows_per_subject)

    try:
        data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])

        # --- 冻结的 MCIA 健康先验 ---
        mcia_ckpt = (args.mcia_checkpoint if args.mcia_checkpoint is not None
                     else Path(config["exp1_dir"]) / "checkpoints" / "best_model.pth")
        if not mcia_ckpt.is_file():
            raise FileNotFoundError(f"需要 MCIA 健康先验检查点：{mcia_ckpt}")
        mcia = build_mcia(config, device)
        load_mcia_state_dict(mcia, mcia_ckpt, device)
        mcia.eval()

        # --- 在与 MCIA 相同的 DB2 池/ScenarioMix 上预训练 SGMD-AAE ---
        sgmd_ckpt = out_dir / "checkpoints" / "sgmd_aae_db2_200hz.pth"
        epoch_count = int(args.sgmd_epochs if args.sgmd_epochs is not None
                          else settings["sgmd_aae_epochs"])
        max_windows = int(args.sgmd_max_train_windows if args.sgmd_max_train_windows is not None
                          else settings.get("sgmd_pretrain_max_windows", 0))
        if args.skip_sgmd_training:
            if not sgmd_ckpt.exists():
                raise FileNotFoundError(f"--skip-sgmd-training 要求存在 {sgmd_ckpt}")
            sgmd = SGMDAAEGenerator().to(device)
            sgmd.load_state_dict(torch.load(sgmd_ckpt, map_location=device)["model"])
            sgmd.eval()
        else:
            db2_cache = out_dir / "cache" / "db2_train_cache.pt"
            db2_train = load_and_cache_data(data_loader, config["train_subjects"], config, db2_cache)["data"]
            if max_windows > 0:
                db2_train = db2_train[:max_windows]
            print(f"正在 {len(db2_train)} 个 DB2 窗口上预训练 SGMD-AAE...", flush=True)
            sgmd = SHARED._train_sgmd(db2_train, config, epoch_count,
                                      int(settings["sgmd_aae_batch_size"]), seed, sgmd_ckpt)

        # --- 基准评估 ---
        per_subject = {}
        for subject_id in subjects:
            set_seed(seed)
            segments, _, _ = prepare_data_db2(data_loader, [subject_id], config, exercises=[1])
            clean = segments.astype(np.float32)
            if dev_run and args.limit_windows_per_subject > 0:
                clean = clean[:args.limit_windows_per_subject]
            mask = SHARED._scenario_generator(
                config, BENCHMARK_SEED + int(subject_id)).generate_mask(
                torch.as_tensor(clean)).numpy().astype(np.float32)
            observed_values = clean * mask
            x_t = torch.as_tensor(clean, dtype=torch.float32, device=device)
            m_t = torch.as_tensor(mask, dtype=torch.float32, device=device)

            delivered = {}
            delivered["MCIA"] = complete_with_mask(mcia, x_t, m_t,
                                                   patch_size=patch_size).cpu().numpy()
            sgmd_raw = predict_sgmd_raw(sgmd, clean, mask, device,
                                        int(settings["sgmd_aae_batch_size"]))
            delivered["SGMD_AAE"] = deliver_uniform(sgmd_raw, observed_values, mask, patch_size)
            cp_result = complete_cp_wopt(observed_values, mask,
                                         CPWOPTConfig(rank=int(settings["cp_wopt_rank"]),
                                                      seed=seed + int(subject_id)))
            delivered["CP_WOPT"] = deliver_uniform(cp_result.reconstruction,
                                                   observed_values, mask, patch_size)

            per_subject[int(subject_id)] = {
                "n_windows": int(len(clean)),
                "masked_fraction": float((mask < 0.5).mean()),
                "methods": {name: metrics_with_pearson(d, clean, mask)
                            for name, d in delivered.items()},
            }
            row = " ".join(f"{n}: NRMSE={per_subject[int(subject_id)]['methods'][n]['nrmse_peak_1']:.4f}"
                           for n in delivered)
            print(f"S{subject_id:02d}：{len(clean)} 个窗口 | {row}", flush=True)

        # --- §7.5 判定 ---
        methods = ["MCIA", "SGMD_AAE", "CP_WOPT"]
        mean_nrmse = {m: float(np.mean([per_subject[s]["methods"][m]["nrmse_peak_1"]
                                        for s in per_subject])) for m in methods}
        wins = {m: int(sum(per_subject[s]["methods"]["MCIA"]["nrmse_peak_1"]
                           < per_subject[s]["methods"][m]["nrmse_peak_1"] for s in per_subject))
                for m in ("SGMD_AAE", "CP_WOPT")}
        n_subjects = len(per_subject)
        verdict = {
            "mean_nrmse_lower_than_sgmd": mean_nrmse["MCIA"] < mean_nrmse["SGMD_AAE"],
            "mean_nrmse_lower_than_cp": mean_nrmse["MCIA"] < mean_nrmse["CP_WOPT"],
            "subject_wins_vs_sgmd": wins["SGMD_AAE"],
            "subject_wins_vs_cp": wins["CP_WOPT"],
            "required_wins": int(np.ceil(n_subjects / 2)),
            "pass_7_5": (mean_nrmse["MCIA"] < mean_nrmse["SGMD_AAE"]
                         and mean_nrmse["MCIA"] < mean_nrmse["CP_WOPT"]
                         and wins["SGMD_AAE"] >= int(np.ceil(n_subjects / 2))
                         and wins["CP_WOPT"] >= int(np.ceil(n_subjects / 2))),
        }

        rows = []
        for s in sorted(per_subject):
            for m in methods:
                met = per_subject[s]["methods"][m]
                rows.append({"subject_id": s, "method": m,
                             "nrmse_peak_1": met["nrmse_peak_1"],
                             "psnr_peak_1_db": met["psnr_peak_1_db"],
                             "rme": met["rme"], "masked_pearson": met["masked_pearson"]})
        with (out_dir / "healthy_benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

        report = {
            "protocol": {
                "protocol_section": "EXPERIMENT_PROTOCOL.md §7.5",
                "subjects": sorted(per_subject), "exercise": 1,
                "mask": f"ScenarioMix mixture, per-subject seed {BENCHMARK_SEED}+subject_id",
                "delivery": "uniform clip->patch-boundary crossfade->copy-back for all methods",
                "mcia_checkpoint": str(mcia_ckpt), "sgmd_checkpoint": str(sgmd_ckpt),
                "sgmd_pretraining": {"epochs": epoch_count, "max_windows": max_windows,
                                      "source": "DB2 healthy train subjects, ScenarioMix"},
                "cp_wopt": "fit per subject on artificially masked observations; transductive",
                "dev_run": dev_run,
            },
            "mean_nrmse": mean_nrmse, "subject_wins": wins,
            "per_subject": per_subject, "verdict": verdict,
        }
        (out_dir / "healthy_benchmark_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.bar(methods, [mean_nrmse[m] for m in methods], capsize=4)
        axis.set_title("缺失区 NRMSE（峰值固定为 1，越低越好）")
        axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(out_dir / "healthy_benchmark_summary.png", dpi=160); plt.close(fig)

        mark_step(run_dir, "healthy_completion_benchmark", "completed", {"output": str(out_dir)})
        print(json.dumps({"mean_nrmse": mean_nrmse, "wins": wins, "verdict": verdict}, indent=2))
        print(f"结果={out_dir / 'healthy_benchmark_report.json'}", flush=True)
    except Exception:
        mark_step(run_dir, "healthy_completion_benchmark", "failed", {"output": str(out_dir)})
        raise


if __name__ == "__main__":
    main()
