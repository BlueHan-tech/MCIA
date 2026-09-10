"""Task-matched MCIA / SGMD-AAE / CP-WOPT comparison for DB3 angle estimation.

All methods receive the same 200-Hz envelope windows and the same DB3 quality
mask.  MCIA and SGMD-AAE are pretrained only on the configured healthy DB2
training subjects; CP-WOPT is fit separately within each unlabeled DB3 split.
Every completed representation is then evaluated with a freshly initialized,
otherwise identical Key10 KinematicTCN.  CP-WOPT is therefore explicitly
reported as a split-transductive baseline, not as an inductive model.

This is a task adaptation of the two literature methods.  It is deliberately
separate from their paper-format 2-kHz reproduction entry point.
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
from torch.utils.data import DataLoader, TensorDataset

from data.dataset_db2_emg import load_and_cache_data
from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from models.baselines.cp_wopt import CPWOPTConfig, complete_cp_wopt, relative_mean_error
from models.baselines.sgmd_aae import (
    SGMDAAEConfig, SGMDAAEGenerator, SGMDMultiViewDiscriminator,
    SGMDAAEObjective, sgmd_complete,
)
from models.completion.mask_generators import ScenarioMixMaskGenerator
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config, set_seed
from utils.run_layout import get_run_dir, mark_step


def _load_exp3_module():
    path = PROJECT_ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("mcia_exp3_angle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared Exp3 helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP3 = _load_exp3_module()


def _masked_metrics(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray) -> dict:
    """Masked-region completion metrics; ONLY valid where `target` is clean truth.

    用于有人工遮挡真值的基准（如健康 DB2 受控掩码对比）。DB3 真实质量掩码
    区域的录制信号本身是坏值，对其计算该指标衡量的是"与坏信号的距离"
    而非补全质量（全零预测会在死通道上得满分），因此本脚本不在 DB3 上
    调用它；该函数由健康基准入口复用。
    """
    missing = np.asarray(observed) < 0.5
    if not np.any(missing):
        raise ValueError("Task-matched comparison needs at least one missing entry.")
    error = np.asarray(prediction)[missing] - np.asarray(target)[missing]
    rmse = float(np.sqrt(np.mean(error ** 2)))
    # The shared pipeline emits values clipped to [0, 1]; use that frozen peak
    # rather than a test-window maximum, which would make subjects incomparable.
    nrmse = rmse
    return {
        "masked_count": int(missing.sum()),
        "masked_fraction": float(missing.mean()),
        "rmse": rmse,
        "nrmse_peak_1": nrmse,
        "psnr_peak_1_db": float(20.0 * np.log10(1.0 / max(rmse, 1e-12))),
        "rme": relative_mean_error(np.asarray(prediction)[missing], np.asarray(target)[missing]),
    }


def _scenario_generator(config: dict, seed: int) -> ScenarioMixMaskGenerator:
    return ScenarioMixMaskGenerator(
        n_channels=int(config["n_channels"]), time_steps=int(config["window_size"]),
        patch_size=int(config["patch_size"]), group_indices=config.get("group_indices"),
        min_alive_per_group=config.get("min_alive_per_group"),
        scenario_weights=config.get("scenario_weights"), scenario_params=config.get("scenario_params"),
        rng=np.random.default_rng(seed),
    )


def _train_sgmd(train_values: np.ndarray, config: dict, epochs: int, batch_size: int, seed: int,
                checkpoint_path: Path) -> SGMDAAEGenerator:
    """Pretrain SGMD-AAE on the same healthy DB2 source and mask distribution as MCIA."""
    device = config["device"]
    set_seed(seed)
    generator = SGMDAAEGenerator().to(device)
    discriminator = SGMDMultiViewDiscriminator().to(device)
    objective = SGMDAAEObjective(SGMDAAEConfig())
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=2e-4)
    loader = DataLoader(TensorDataset(torch.as_tensor(train_values, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True, num_workers=0)
    mask_generator = _scenario_generator(config, seed + 1000)
    for epoch in range(epochs):
        generator.train(); discriminator.train()
        losses = []
        for (target_btc,) in loader:
            target = target_btc.to(device).unsqueeze(1)
            mask = mask_generator.generate_batch_masks(
                len(target_btc), device=device).transpose(1, 2).unsqueeze(1)
            # 标准 GAN 单前向模式：生成器只前向一次，D 用 output.detach()。
            output = generator(target * mask, mask)
            fake = output.detach()
            optimizer_d.zero_grad(set_to_none=True)
            objective.discriminator_loss(discriminator(target), discriminator(fake)).backward()
            optimizer_d.step()
            optimizer_g.zero_grad(set_to_none=True)
            loss, _ = objective.generator_loss(output, target, mask, discriminator)
            loss.backward()
            optimizer_g.step()
            losses.append(float(loss.detach().cpu()))
        print(f"  SGMD pretrain epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.6f}", flush=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": generator.state_dict(), "pretraining": "DB2_200Hz_scenario_mix"}, checkpoint_path)
    return generator.eval()


@torch.no_grad()
def _apply_sgmd(model: SGMDAAEGenerator, values: np.ndarray, observed: np.ndarray, device: str,
                batch_size: int) -> np.ndarray:
    delivered = np.empty_like(values)
    model.eval()
    for start in range(0, len(values), batch_size):
        x = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=device).unsqueeze(1)
        m = torch.as_tensor(observed[start:start + batch_size], dtype=torch.float32, device=device).unsqueeze(1)
        delivered[start:start + len(x)] = sgmd_complete(model, x, m).clamp(0.0, 1.0).squeeze(1).cpu().numpy()
    return delivered


def _apply_cp_per_split(values: np.ndarray, observed: np.ndarray, rank: int, seed: int) -> np.ndarray:
    # CP-WOPT has no inductive prediction map.  It fits factors from each split's
    # unlabeled observed EMG only; this is logged as split-transductive behavior.
    result = complete_cp_wopt(values * observed, observed, CPWOPTConfig(rank=rank, seed=seed))
    return np.clip(result.reconstruction, 0.0, 1.0).astype(np.float32)


def _summary_rows(subject_reports: list[dict]) -> list[dict]:
    rows = []
    for subject in subject_reports:
        for method, values in subject["methods"].items():
            angle = values["angle"]["global"]
            rows.append({
                "subject_id": subject["subject_id"], "method": method,
                "angle_rmse": angle["rmse"], "angle_mae": angle["mae"],
                "angle_pearson": angle["pearson"], "angle_r2": angle["r2"],
            })
    return rows


def _write_summary(out_dir: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with (out_dir / "task_matched_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    methods = list(dict.fromkeys(row["method"] for row in rows))
    aggregate = {}
    for method in methods:
        matching = [row for row in rows if row["method"] == method]
        aggregate[method] = {key: {"mean": float(np.nanmean([r[key] for r in matching])),
                                   "std": float(np.nanstd([r[key] for r in matching])), "n": len(matching)}
                             for key in fields[2:]}
    (out_dir / "task_matched_summary.json").write_text(
        json.dumps({"per_subject": rows, "subject_equal_summary": aggregate}, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(7, 4.5))
    means = [aggregate[m]["angle_rmse"]["mean"] for m in methods]
    stds = [aggregate[m]["angle_rmse"]["std"] for m in methods]
    axis.bar(methods, means, yerr=stds, capsize=4)
    axis.set_title("Key10 angle RMSE (lower)"); axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_dir / "task_matched_summary.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 200-Hz task-matched literature-baseline comparison.")
    parser.add_argument("--sgmd-epochs", type=int, default=None)
    parser.add_argument("--sgmd-max-train-windows", type=int, default=None,
                        help="0 keeps every DB2 pretraining window; any cap is an explicitly reduced run.")
    parser.add_argument("--skip-sgmd-training", action="store_true")
    parser.add_argument("--mcia-checkpoint", type=Path, default=None,
                        help="Use this frozen healthy-prior MCIA checkpoint (useful for standalone PyCharm runs).")
    args = parser.parse_args()
    cfg = load_yaml_config(PROJECT_ROOT)
    settings = cfg["literature_baselines"]
    run_dir = get_run_dir(PROJECT_ROOT, cfg, create=True)
    os.environ["MCIA_RUN_DIR"] = str(run_dir)
    config = flatten_pipeline_config(cfg)
    out_dir = run_dir / "06_diagnostics" / "task_matched_literature_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    mark_step(run_dir, "task_matched_literature_baselines", "running")
    seed = int(settings["seed"])
    epoch_count = int(args.sgmd_epochs if args.sgmd_epochs is not None else settings["sgmd_aae_epochs"])
    max_windows = int(args.sgmd_max_train_windows if args.sgmd_max_train_windows is not None
                      else settings.get("sgmd_pretrain_max_windows", 0))
    sgmd_checkpoint = out_dir / "checkpoints" / "sgmd_aae_db2_200hz.pth"
    try:
        data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
        db2_cache = out_dir / "cache" / "db2_train_cache.pt"
        db2_train = load_and_cache_data(data_loader, config["train_subjects"], config, db2_cache)["data"]
        if max_windows > 0:
            db2_train = db2_train[:max_windows]
        if len(db2_train) == 0:
            raise ValueError("No DB2 healthy-pretraining windows available.")
        if args.skip_sgmd_training:
            if not sgmd_checkpoint.exists():
                raise FileNotFoundError(f"--skip-sgmd-training requires {sgmd_checkpoint}")
            sgmd = SGMDAAEGenerator().to(config["device"])
            sgmd.load_state_dict(torch.load(sgmd_checkpoint, map_location=config["device"])["model"])
            sgmd.eval()
        else:
            print(f"Pretraining SGMD-AAE on {len(db2_train)} DB2 200-Hz windows...", flush=True)
            sgmd = _train_sgmd(db2_train, config, epoch_count, int(settings["sgmd_aae_batch_size"]), seed, sgmd_checkpoint)
        if args.mcia_checkpoint is not None:
            if not args.mcia_checkpoint.is_file():
                raise FileNotFoundError(f"MCIA checkpoint not found: {args.mcia_checkpoint}")
            healthy_mcia, mcia_checkpoint = EXP3._load_mcia_from_checkpoint(
                config, config["device"], args.mcia_checkpoint, "healthy_prior_explicit")
        else:
            healthy_mcia, mcia_checkpoint = EXP3.load_healthy_prior_mcia(config, config["device"])
        if healthy_mcia is None:
            raise FileNotFoundError("MCIA healthy-prior checkpoint is required; run Exp1 in this run first.")
        report = {"protocol": {
            "representation": "same DB3 200-Hz envelope, 256x12 windows",
            "mask": "same DB3 quality_mask per window for every method",
            "tcn": "fresh same-config Key10 KinematicTCN with seed reset per method",
            "mcia_checkpoint": str(mcia_checkpoint), "sgmd_checkpoint": str(sgmd_checkpoint),
            "cp_wopt": "fit separately on each unlabeled DB3 split; split-transductive",
            "sgmd_pretraining": {"source": "DB2 healthy train subjects", "epochs": epoch_count,
                                   "windows": int(len(db2_train)), "mask": "MCIA ScenarioMix"},
        }, "subjects": []}
        for subject_id in config["regressor_db3_subjects"]:
            print(f"\n--- Task-matched DB3 S{subject_id:02d} ---", flush=True)
            raw, angle, _, repetitions, metadata = prepare_kinematics_data(
                data_loader, [subject_id], config, exercises=config["regressor_exercises"], db="db3", return_metadata=True)
            train_idx, val_idx, test_idx = make_rep_split(repetitions)
            observed = metadata["quality_mask"].astype(np.float32)
            if not np.any(observed < 0.5):
                raise ValueError(f"S{subject_id:02d} has no quality-mask missing entries.")
            methods = {}
            methods["MCIA"] = EXP3.make_enhanced_pool(healthy_mcia, raw, train_idx, val_idx, test_idx,
                                                        observed, config["device"], patch_size=int(config["patch_size"]))
            methods["SGMD_AAE"] = _apply_sgmd(sgmd, raw, observed, config["device"], int(settings["sgmd_aae_batch_size"]))
            cp = raw.copy()
            for split_number, indices in enumerate((train_idx, val_idx, test_idx)):
                cp[indices] = _apply_cp_per_split(raw[indices], observed[indices], int(settings["cp_wopt_rank"]), seed + split_number)
            methods["CP_WOPT"] = cp
            subject = {"subject_id": int(subject_id), "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}, "methods": {}}
            for method, emg in methods.items():
                set_seed(seed)
                ckpt = out_dir / "checkpoints" / f"S{subject_id:02d}_{method}_tcn.pth"
                result, _, _, _, _ = EXP3.evaluate_group(method, emg, angle, train_idx, val_idx, test_idx,
                                                          config, config["device"], ckpt, subject_id)
                subject["methods"][method] = {"angle": result["subsets"]}
                # DB3 真实质量掩码区域无干净真值：不在此计算补全指标
                # （对坏信号算距离会奖励全零预测，见 _masked_metrics docstring）；
                # 补全质量对比在健康 DB2 受控掩码基准（协议 §7.5）进行。
            report["subjects"].append(subject)
            (out_dir / "task_matched_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        rows = _summary_rows(report["subjects"])
        _write_summary(out_dir, rows)
        mark_step(run_dir, "task_matched_literature_baselines", "completed", {"output": str(out_dir)})
        print(f"\nComparable results: {out_dir / 'task_matched_summary.json'}", flush=True)
    except Exception:
        mark_step(run_dir, "task_matched_literature_baselines", "failed", {"output": str(out_dir)})
        raise


if __name__ == "__main__":
    main()
