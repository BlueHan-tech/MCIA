"""Bounded, inference-only development comparison of MCIA output activations.

Uses DB2 validation subject S29 only.  It neither reads DB2 test subjects nor
writes checkpoints, metrics, figures, or run artifacts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np
import torch
import torch.nn as nn

from models.completion.mask_generators import ScenarioMixMaskGenerator
from models.completion.mcia_core import MCIA, derive_ch_mask_from_sample_mask


RUN = ROOT / "outputs" / "run" / "run_20260907_154902_1" / "01_db2_completion"
CHECKPOINT = RUN / "checkpoints" / "best_model.pth"
VALIDATION_CACHE = RUN / "cache" / "val_cache.pt"


def build_model(device: torch.device, activation: nn.Module) -> MCIA:
    model = MCIA(
        window_size=256,
        n_channels=12,
        patch_size=8,
        embed_dim=192,
        n_layers=6,
        n_heads=6,
        ffn_dim=384,
        dropout=0.1,
        num_domains=2,
    ).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.pred_head[1] = activation
    model.eval()
    return model


def channelwise_masked_metrics(pred: torch.Tensor, target: torch.Tensor, missing: torch.Tensor) -> dict:
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    missing_np = missing.cpu().numpy().astype(bool)
    errors = pred_np[missing_np] - target_np[missing_np]
    correlations = []
    for channel in range(pred_np.shape[-1]):
        p = pred_np[:, :, channel][missing_np[:, :, channel]]
        y = target_np[:, :, channel][missing_np[:, :, channel]]
        if len(p) > 1 and np.std(p) > 1e-12 and np.std(y) > 1e-12:
            correlations.append(float(np.corrcoef(p, y)[0, 1]))
    return {
        "mse_masked": float(np.mean(errors ** 2)),
        "mae_masked": float(np.mean(np.abs(errors))),
        "corr_masked_channel_mean": float(np.mean(correlations)) if correlations else float("nan"),
        "max_completed": float(pred.max().item()),
        "min_completed": float(pred.min().item()),
    }


@torch.no_grad()
def evaluate(model: MCIA, data: torch.Tensor, masks: torch.Tensor, batch_size: int,
             device: torch.device, clamp_predictions: bool) -> tuple[dict, float]:
    completed_all, target_all, missing_all = [], [], []
    started = time.perf_counter()
    for start in range(0, len(data), batch_size):
        target = data[start:start + batch_size].to(device)
        mask = masks[start:start + batch_size].to(device)
        masked = target * mask
        channel_mask = derive_ch_mask_from_sample_mask(mask)
        prediction = model(masked, mask=channel_mask, x_masked=masked, raw_time_mask=mask)
        completed_prediction = prediction.clamp(0.0, 1.0) if clamp_predictions else prediction
        completed = completed_prediction * (1.0 - mask) + target * mask
        completed_all.append(completed.cpu())
        target_all.append(target.cpu())
        missing_all.append((mask < 0.5).cpu())
    elapsed = time.perf_counter() - started
    metrics = channelwise_masked_metrics(
        torch.cat(completed_all), torch.cat(target_all), torch.cat(missing_all)
    )
    observed = ~torch.cat(missing_all)
    metrics["observed_max_abs_error"] = float(
        (torch.cat(completed_all)[observed] - torch.cat(target_all)[observed]).abs().max().item()
    )
    return metrics, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not CHECKPOINT.exists() or not VALIDATION_CACHE.exists():
        raise FileNotFoundError("Required historical DB2 validation cache or checkpoint is absent")

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(VALIDATION_CACHE, map_location="cpu", weights_only=False)
    subject_ids = np.asarray(payload["subject_ids"])
    subject_data = torch.as_tensor(payload["data"], dtype=torch.float32)[subject_ids == 29]
    data = subject_data[:args.windows].contiguous()
    if len(data) != args.windows:
        raise ValueError(f"S29 supplies only {len(data)} windows, requested {args.windows}")

    generator = ScenarioMixMaskGenerator(
        n_channels=12,
        time_steps=256,
        patch_size=8,
        group_indices={"flexor": [0, 1, 2, 3, 8], "extensor": [4, 5, 6, 7, 9], "upper_arm": [10, 11]},
        min_alive_per_group={"flexor": 3, "extensor": 3, "upper_arm": 1},
        rng=np.random.default_rng(42),
    )
    masks = generator.generate_batch_masks(len(data), n_channels=12, time_steps=256, device="cpu", scenario="s3").transpose(1, 2)
    if not torch.all((masks == 0) | (masks == 1)):
        raise AssertionError("Mask is not binary")

    softplus = build_model(device, nn.Softplus(beta=10))
    sigmoid = build_model(device, nn.Sigmoid())
    linear_equal = torch.equal(softplus.pred_head[0].weight.cpu(), sigmoid.pred_head[0].weight.cpu())
    if not linear_equal:
        raise AssertionError("Activation comparison did not preserve output-head linear weights")

    raw_softplus_metrics, raw_softplus_time = evaluate(
        softplus, data, masks, args.batch_size, device, clamp_predictions=False
    )
    softplus_metrics, softplus_time = evaluate(
        softplus, data, masks, args.batch_size, device, clamp_predictions=True
    )
    sigmoid_metrics, sigmoid_time = evaluate(
        sigmoid, data, masks, args.batch_size, device, clamp_predictions=True
    )
    print({
        "scope": "DB2 validation S29, first 512 windows, S3 masks, no DB2 test data",
        "device": str(device),
        "checkpoint": str(CHECKPOINT),
        "same_linear_head_weights": linear_equal,
        "mask_missing_ratio": float((masks < 0.5).float().mean()),
        "softplus_raw": {**raw_softplus_metrics, "seconds": raw_softplus_time},
        "softplus_clip": {**softplus_metrics, "seconds": softplus_time},
        "sigmoid": {**sigmoid_metrics, "seconds": sigmoid_time},
    })


if __name__ == "__main__":
    main()
