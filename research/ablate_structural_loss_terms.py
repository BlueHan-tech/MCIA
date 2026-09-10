"""DB2 小规模结构损失项剪枝消融（开发诊断，不修改主流程）。

对当前主损失中的每个可选项做一次"去掉该项"的短训练对比：
baseline（当前权重）与 drop_one 候选（单项权重置 0）。
Charbonnier 为主保真项，始终保留。仅训练/验证被试参与；不读取测试被试。

用法（先设置 MCIA_RUN_DIR 指向现有 run 目录）：
  python research/ablate_structural_loss_terms.py --epochs 6
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset_db2_emg import EMGCompletionDataset, prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from models.completion.mask_generators import ScenarioMixMaskGenerator
from utils.loss_functions import EMGImputationLoss
from utils.paper_pipeline import build_mcia, complete_with_mask, flatten_pipeline_config, load_yaml_config, set_seed

# 当前主损失的可剪枝项及基准权重（与 build_structural_loss 默认一致）
BASE_WEIGHTS = {
    "ncc": 0.5,
    "stft": 0.3,
    "boundary": 0.1,
    "aux": 0.1,
    "envelope": 0.2,
    "patch_rms": 0.1,
}
SCENARIO = "s3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-windows", type=int, default=1024)
    parser.add_argument("--max-val-windows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--terms", default="",
                        help="comma-separated terms to prune; default prunes all in BASE_WEIGHTS")
    return parser.parse_args()


def subset_windows(windows: np.ndarray, repetitions: np.ndarray, reps: tuple,
                   limit: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(np.isin(repetitions, reps))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return windows[idx[:limit]]


def make_loader(data: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        EMGCompletionDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=g,
    )


def build_mask_generator(config: Dict, seed: int) -> ScenarioMixMaskGenerator:
    return ScenarioMixMaskGenerator(
        n_channels=int(config["n_channels"]),
        time_steps=int(config["window_size"]),
        patch_size=int(config["patch_size"]),
        group_indices=config.get("group_indices"),
        min_alive_per_group=config.get("min_alive_per_group"),
        scenario_weights=config.get("scenario_weights"),
        scenario_params=config.get("scenario_params"),
        rng=np.random.default_rng(seed),
    )


def train_one(pruned_term: str, train_data: np.ndarray, val_data: np.ndarray,
              config: Dict, args: argparse.Namespace, device: torch.device) -> Dict:
    set_seed(args.seed)
    model = build_mcia(config, str(device))
    set_seed(args.seed + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    weights = dict(BASE_WEIGHTS)
    if pruned_term != "baseline":
        weights[pruned_term] = 0.0
    criterion = EMGImputationLoss(
        w_charbonnier=1.0,
        w_ncc=weights["ncc"],
        w_stft=weights["stft"],
        w_boundary=weights["boundary"],
        w_aux=weights["aux"],
        envelope_loss_weight=weights["envelope"],
        patch_rms_loss_weight=weights["patch_rms"],
        envelope_kernel_size=config.get("envelope_kernel_size", 25),
        patch_rms_size=config.get("patch_rms_size", config.get("patch_size", 8)),
    ).to(device)
    mask_generator = build_mask_generator(config, args.seed + 2000)
    train_loader = make_loader(train_data, args.batch_size, True, args.seed + 3000)
    val_loader = make_loader(val_data, args.batch_size, False, args.seed + 4000)

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        criterion.set_epoch(epoch)
        total = 0.0
        for batch in train_loader:
            target = batch["data"].to(device)
            raw_mask = mask_generator.generate_mask(target)
            channel_mask = raw_mask.max(dim=1).values
            masked = target * raw_mask
            pred = model(
                masked, mask=channel_mask, x_masked=masked,
                raw_time_mask=raw_mask, return_aux=True,
            )
            loss, _ = criterion(pred, target, raw_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * target.shape[0]
        print(f"  [{pruned_term}] epoch={epoch}/{args.epochs} loss={total / max(1, len(train_loader)):.5f}")

    validation = evaluate(model, val_loader, config, args.seed + 5000, device)
    validation["elapsed_seconds"] = time.time() - started
    return validation


@torch.no_grad()
def evaluate(model, loader: DataLoader, config: Dict, seed: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    mask_generator = build_mask_generator(config, seed)
    squared = 0.0
    absolute = 0.0
    count = 0
    correlations = []
    for batch in loader:
        target = batch["data"].to(device)
        raw_mask = mask_generator.generate_mask(target, scenario=SCENARIO)
        # 交付走主模块入口（clamp→边界淡化→回填），与主链路单一来源。
        completed = complete_with_mask(model, target, raw_mask,
                                       patch_size=int(config["patch_size"]))
        missing = raw_mask < 0.5
        diff = completed - target
        squared += float(diff[missing].square().sum().item())
        absolute += float(diff[missing].abs().sum().item())
        count += int(missing.sum().item())
        pred_np = completed.cpu().numpy()
        target_np = target.cpu().numpy()
        missing_np = missing.cpu().numpy()
        for b in range(target_np.shape[0]):
            for c in range(target_np.shape[2]):
                region = missing_np[b, :, c]
                if region.sum() < 4:
                    continue
                p, y = pred_np[b, region, c], target_np[b, region, c]
                if np.std(p) > 1e-8 and np.std(y) > 1e-8:
                    correlations.append(float(np.corrcoef(p, y)[0, 1]))
    mse = squared / max(1, count)
    return {
        "rmse_masked": math.sqrt(mse),
        "mae_masked": absolute / max(1, count),
        "corr_masked": float(np.mean(correlations)) if correlations else float("nan"),
        "n_masked_values": count,
    }


def main() -> None:
    args = parse_args()
    run_dir_value = os.environ.get("MCIA_RUN_DIR")
    if not run_dir_value:
        raise RuntimeError("MCIA_RUN_DIR must point to an existing run directory")
    run_dir = Path(run_dir_value).resolve()
    output_dir = run_dir / "06_diagnostics" / "loss_term_prune_small_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = flatten_pipeline_config(load_yaml_config(PROJECT_ROOT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} output={output_dir}")

    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    train_parts = []
    for subject in (1, 2):
        windows, repetitions = prepare_data_db2(data_loader, [subject], config, exercises=[1])
        train_parts.append(subset_windows(
            windows, repetitions, (1, 3, 4, 6),
            args.max_train_windows // 2, args.seed + subject,
        ))
    train_data = np.concatenate(train_parts, axis=0)
    val_windows, val_reps = prepare_data_db2(data_loader, [3], config, exercises=[1])
    val_data = subset_windows(val_windows, val_reps, (2, 5), args.max_val_windows, args.seed + 30)
    print(f"windows train={len(train_data)} val={len(val_data)}")

    terms = [t.strip() for t in args.terms.split(",") if t.strip()] if args.terms else sorted(BASE_WEIGHTS)
    results = {}
    baseline_metrics = train_one("baseline", train_data, val_data, config, args, device)
    results["baseline"] = baseline_metrics
    for term in terms:
        if term not in BASE_WEIGHTS:
            raise ValueError(f"Unknown loss term {term!r}; choose from {sorted(BASE_WEIGHTS)}")
        results[f"drop_{term}"] = train_one(term, train_data, val_data, config, args, device)

    comparison = {}
    for key, metrics in results.items():
        if key == "baseline":
            continue
        comparison[key] = {
            "rmse_relative_change": (
                metrics["rmse_masked"] - baseline_metrics["rmse_masked"]
            ) / baseline_metrics["rmse_masked"],
            "corr_absolute_change": metrics["corr_masked"] - baseline_metrics["corr_masked"],
        }

    payload = {
        "protocol": {
            "purpose": "directional small-scale loss-term pruning dev check only",
            "database": "DB2", "exercises": [1],
            "train_subjects": [1, 2], "validation_subjects": [3],
            "train_repetitions": [1, 3, 4, 6], "validation_repetitions": [2, 5],
            "selection_uses_test": False, "scenario": SCENARIO,
            "epochs": args.epochs, "seed": args.seed,
            "base_weights": BASE_WEIGHTS,
            "window_counts": {"train": len(train_data), "validation": len(val_data)},
        },
        "results": results,
        "comparison_vs_baseline": comparison,
    }
    out_file = output_dir / "loss_term_prune_results.json"
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"saved {out_file}")
    for key, change in comparison.items():
        print(f"  {key}: dRMSE={change['rmse_relative_change']:+.2%} dCorr={change['corr_absolute_change']:+.4f}")


if __name__ == "__main__":
    main()
