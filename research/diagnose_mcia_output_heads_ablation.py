"""Unified 20-epoch DB2 screen of MCIA waveform output heads.

This diagnostic is isolated from Exp1/Exp2/Exp3.  It loads the selected DB2
windows once, resets every candidate to the same seed, and replays identical
ScenarioMix and evaluation-mask streams for every output head.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset_db2_emg import EMGCompletionDataset, prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from models.completion.mask_generators import ScenarioMixMaskGenerator
from models.completion.mcia_core import MCIA
_LEGACY_PATH = PROJECT_ROOT / "research" / "diagnose_mcia_unet_small_ablation.py"
_LEGACY_SPEC = importlib.util.spec_from_file_location("mcia_decoder_ablation", _LEGACY_PATH)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:
    raise ImportError(_LEGACY_PATH)
_LEGACY = importlib.util.module_from_spec(_LEGACY_SPEC)
_LEGACY_SPEC.loader.exec_module(_LEGACY)
MCIAWithNonCausalTCN = _LEGACY.MCIAWithNonCausalTCN
MCIAWithTemporalUNet = _LEGACY.MCIAWithTemporalUNet
load_config = _LEGACY.load_config
make_loss = _LEGACY.make_loss


SCENARIOS = ("s1", "s2", "s3")
HEADS = (
    "linear",
    "residual_mlp",
    "local_conv1d",
    "overlap_add",
    "noncausal_tcn",
    "temporal_unet",
    "partial_conv2d",
    "gated_conv2d",
    "time_channel_conv",
)
SCENARIO_WEIGHTS = {"s1": 0.2, "s2": 0.4, "s3": 0.4}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_kwargs(config: Dict) -> Dict:
    return {
        "window_size": config["window_size"],
        "n_channels": config["n_channels"],
        "patch_size": config["patch_size"],
        "embed_dim": config["embed_dim"],
        "n_layers": config["n_layers"],
        "n_heads": config["n_heads"],
        "ffn_dim": config["ffn_dim"],
        "dropout": config["model_dropout"],
        "num_domains": config.get("num_domains", 2),
        "use_synergy_bottleneck": config.get("use_synergy_bottleneck", False),
        "n_synergies": config.get("n_synergies", 6),
        "synergy_gate_scale": config.get("synergy_gate_scale", 0.5),
        "synergy_dropout": config.get("synergy_dropout", 0.1),
    }


class ResidualMLP(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        hidden = max(32, embed_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, patch_size),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class LocalWaveformRefiner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 5, padding=2, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.out = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, logits: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.pointwise(F.gelu(self.depthwise(logits))))
        return logits + self.out(x)


class PartialConv2DRefiner(nn.Module):
    """Mask-normalized time/channel refinement with fallback to token logits."""

    def __init__(self):
        super().__init__()
        self.feature = nn.Conv2d(1, 24, kernel_size=(5, 3), padding=(2, 1), bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(26, 24, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 1, 3, padding=1),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)
        self.register_buffer("count_kernel", torch.ones(1, 1, 5, 3), persistent=False)

    def forward(self, logits: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        source = observed.transpose(1, 2).unsqueeze(1)
        valid = mask.transpose(1, 2).unsqueeze(1)
        numerator = self.feature(source * valid)
        count = F.conv2d(valid, self.count_kernel, padding=(2, 1))
        normalized = numerator * (15.0 / count.clamp_min(1.0))
        normalized = normalized * (count > 0).to(normalized.dtype)
        base = logits.unsqueeze(1)
        correction = self.fuse(torch.cat((normalized, base, valid), dim=1)).squeeze(1)
        return logits + correction


class GatedConv2DRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Conv2d(3, 24, kernel_size=(5, 3), padding=(2, 1))
        self.gate = nn.Conv2d(3, 24, kernel_size=(5, 3), padding=(2, 1))
        self.out = nn.Conv2d(24, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, logits: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = torch.stack((logits, observed.transpose(1, 2), mask.transpose(1, 2)), dim=1)
        hidden = F.gelu(self.feature(inputs)) * torch.sigmoid(self.gate(inputs))
        return logits + self.out(hidden).squeeze(1)


class TimeChannelRefiner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.temporal = nn.Conv1d(channels, channels, 5, padding=2, groups=channels)
        self.channel = nn.Conv1d(channels, channels, 1)
        self.out = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, logits: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        completed = logits * (1.0 - mask.transpose(1, 2)) + observed.transpose(1, 2)
        hidden = F.gelu(self.temporal(completed))
        hidden = F.gelu(self.channel(hidden))
        return logits + self.out(hidden)


class MCIAWithOutputHead(MCIA):
    def __init__(self, head_name: str, **kwargs):
        super().__init__(**kwargs)
        self.head_name = head_name
        if head_name == "residual_mlp":
            self.extra_head = ResidualMLP(self.embed_dim, self.patch_size)
        elif head_name == "local_conv1d":
            self.extra_head = LocalWaveformRefiner(self.n_channels)
        elif head_name == "partial_conv2d":
            self.extra_head = PartialConv2DRefiner()
        elif head_name == "gated_conv2d":
            self.extra_head = GatedConv2DRefiner()
        elif head_name == "time_channel_conv":
            self.extra_head = TimeChannelRefiner(self.n_channels)
        elif head_name == "overlap_add":
            self.overlap_size = self.patch_size * 2
            self.overlap_proj = nn.Linear(self.embed_dim, self.overlap_size)
            nn.init.normal_(self.overlap_proj.weight, std=0.02)
            nn.init.zeros_(self.overlap_proj.bias)
            ramp = torch.arange(1, self.overlap_size + 1, dtype=torch.float32)
            ramp = torch.minimum(ramp, ramp.flip(0))
            self.register_buffer("overlap_weight", ramp, persistent=False)
        else:
            raise ValueError(head_name)

    def _encode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        drop_condition: bool,
        raw_time_mask: torch.Tensor | None,
        domain_id: torch.Tensor | None,
        chan_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time_steps, channels = x.shape
        if time_steps != self.window_size or channels != self.n_channels:
            raise ValueError(f"Expected (B,{self.window_size},{self.n_channels}), got {tuple(x.shape)}")
        if raw_time_mask is None:
            raw_time_mask = x.new_ones(batch, time_steps, channels)
        observed = x * raw_time_mask
        time_mask = self._derive_time_mask(raw_time_mask).float()
        if chan_valid_mask is None and mask is not None:
            chan_valid_mask = mask.float()
        tokens = self.patch_embed(observed.transpose(1, 2).contiguous())
        if drop_condition:
            tokens = self.uncond_token.expand(batch, channels, self.num_patches, self.embed_dim)
        else:
            token_mask = time_mask.unsqueeze(-1)
            missing = self.mask_token.expand(batch, channels, self.num_patches, self.embed_dim)
            tokens = tokens * token_mask + missing * (1.0 - token_mask)
        tokens = tokens + self.temp_pos + self.chan_pos
        if domain_id is not None:
            tokens = tokens + self.domain_embed(domain_id)[:, None, None, :]
        tokens = self.local_bypass(tokens)
        for block in self.blocks:
            tokens = block(tokens, time_mask, chan_valid_mask=chan_valid_mask)
        tokens = self.norm(tokens)
        if self.synergy_bottleneck is not None:
            tokens = self.synergy_bottleneck(tokens)
        return tokens, observed, raw_time_mask

    def _overlap_decode(self, tokens: torch.Tensor) -> torch.Tensor:
        patches = self.overlap_proj(tokens)
        batch, channels, n_patches, width = patches.shape
        total = (n_patches - 1) * self.patch_size + width
        values = patches.new_zeros(batch, channels, total)
        weights = patches.new_zeros(batch, channels, total)
        weight = self.overlap_weight.to(dtype=patches.dtype).view(1, 1, -1)
        for patch_index in range(n_patches):
            start = patch_index * self.patch_size
            values[:, :, start:start + width] += patches[:, :, patch_index, :] * weight
            weights[:, :, start:start + width] += weight
        crop = self.patch_size // 2
        return (values / weights.clamp_min(1e-6))[:, :, crop:crop + self.window_size]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        x_masked: torch.Tensor | None = None,
        drop_condition: bool = False,
        side: torch.Tensor | None = None,
        age: torch.Tensor | None = None,
        gender: torch.Tensor | None = None,
        raw_time_mask: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
        chan_valid_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        del x_masked, side, age, gender
        tokens, observed, raw_time_mask = self._encode(
            x, mask, drop_condition, raw_time_mask, domain_id, chan_valid_mask
        )
        if self.head_name == "overlap_add":
            logits = self._overlap_decode(tokens)
        else:
            patch_logits = self.pred_head[0](tokens)
            if self.head_name == "residual_mlp":
                patch_logits = patch_logits + self.extra_head(tokens)
            logits = self.patch_recover(patch_logits)
            if self.head_name != "residual_mlp":
                logits = self.extra_head(logits, observed, raw_time_mask)
        pred = F.softplus(logits.transpose(1, 2), beta=10)
        return {"pred": pred, "time": pred} if return_aux else pred


def build_model(head_name: str, config: Dict) -> nn.Module:
    kwargs = model_kwargs(config)
    if head_name == "linear":
        return MCIA(**kwargs)
    if head_name == "noncausal_tcn":
        return MCIAWithNonCausalTCN(**kwargs)
    if head_name == "temporal_unet":
        return MCIAWithTemporalUNet(**kwargs)
    return MCIAWithOutputHead(head_name, **kwargs)


def build_mask_generator(config: Dict, seed: int) -> ScenarioMixMaskGenerator:
    return ScenarioMixMaskGenerator(
        n_channels=config["n_channels"],
        time_steps=config["window_size"],
        patch_size=config["patch_size"],
        group_indices=config.get("group_indices"),
        min_alive_per_group=config.get("min_alive_per_group"),
        scenario_weights=config.get("scenario_weights"),
        scenario_params=config.get("scenario_params"),
        rng=np.random.default_rng(seed),
    )


def subset_windows(data: np.ndarray, repetitions: np.ndarray, reps: Iterable[int], limit: int, seed: int) -> np.ndarray:
    selected = data[np.isin(repetitions, list(reps))]
    if limit > 0 and len(selected) > limit:
        indices = np.random.default_rng(seed).choice(len(selected), limit, replace=False)
        selected = selected[np.sort(indices)]
    return selected


def make_loader(data: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        EMGCompletionDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def train_epoch(model, loader, optimizer, criterion, mask_generator, device) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        target = batch["data"].to(device)
        raw_mask = mask_generator.generate_mask(target)
        masked = target * raw_mask
        pred = model(
            masked,
            mask=raw_mask.max(dim=1).values,
            x_masked=masked,
            raw_time_mask=raw_mask,
            return_aux=True,
        )
        loss, _ = criterion(pred, target, raw_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * len(target)
        count += len(target)
    return total / max(1, count)


@torch.no_grad()
def evaluate(model, loader, config, scenario, seed, device) -> Dict[str, float]:
    model.eval()
    generator = build_mask_generator(config, seed)
    squared = absolute = 0.0
    count = masked_count = total_count = 0
    correlations = []
    for batch in loader:
        target = batch["data"].to(device)
        raw_mask = generator.generate_mask(target, scenario=scenario)
        observed = target * raw_mask
        pred = model(observed, mask=raw_mask.max(dim=1).values, raw_time_mask=raw_mask)
        completed = pred * (1.0 - raw_mask) + target * raw_mask
        missing = raw_mask < 0.5
        diff = completed - target
        squared += float(diff[missing].square().sum())
        absolute += float(diff[missing].abs().sum())
        count += int(missing.sum())
        masked_count += int(missing.sum())
        total_count += raw_mask.numel()
        p = completed.detach().cpu().numpy()
        y = target.detach().cpu().numpy()
        m = missing.detach().cpu().numpy()
        for b in range(len(y)):
            for c in range(y.shape[2]):
                region = m[b, :, c]
                if region.sum() >= 4 and np.std(p[b, region, c]) >= 1e-8 and np.std(y[b, region, c]) >= 1e-8:
                    correlations.append(float(np.corrcoef(p[b, region, c], y[b, region, c])[0, 1]))
    mse = squared / max(1, count)
    rmse = math.sqrt(mse)
    return {
        "mse_masked": mse,
        "rmse_masked": rmse,
        "nrmse_masked": rmse,
        "psnr_masked_db": -10.0 * math.log10(max(mse, 1e-12)),
        "mae_masked": absolute / max(1, count),
        "corr_masked": float(np.mean(correlations)) if correlations else float("nan"),
        "mask_ratio": masked_count / max(1, total_count),
        "n_masked_values": count,
    }


def weighted_nrmse(metrics: Dict[str, Dict[str, float]]) -> float:
    return sum(SCENARIO_WEIGHTS[s] * metrics[s]["nrmse_masked"] for s in SCENARIOS)


def run_head(name, train_data, val_data, test_data, config, args, device) -> Dict:
    set_seed(args.seed)
    model = build_model(name, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = make_loss(config, device)
    train_masks = build_mask_generator(config, args.seed + 2000)
    train_loader = make_loader(train_data, args.batch_size, True, args.seed + 3000)
    val_loader = make_loader(val_data, args.batch_size, False, args.seed + 4000)
    test_loader = make_loader(test_data, args.batch_size, False, args.seed + 5000)
    best_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, train_masks, device)
        val_s3 = evaluate(model, val_loader, config, "s3", args.seed + 6002, device)
        history.append({"epoch": epoch, "train_loss": loss, "validation_s3": val_s3})
        print(f"[{name}] {epoch:02d}/{args.epochs} loss={loss:.5f} val_s3={val_s3['nrmse_masked']:.5f} corr={val_s3['corr_masked']:.4f}")
        if val_s3["mse_masked"] < best_mse:
            best_mse = val_s3["mse_masked"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    validation = {s: evaluate(model, val_loader, config, s, args.seed + 7000 + i, device) for i, s in enumerate(SCENARIOS)}
    test = {s: evaluate(model, test_loader, config, s, args.seed + 8000 + i, device) for i, s in enumerate(SCENARIOS)}
    return {
        "name": name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.time() - started,
        "selection": "lowest validation-subject S3 masked MSE; test unused",
        "best_validation_s3_mse": best_mse,
        "weighted_validation_nrmse": weighted_nrmse(validation),
        "weighted_test_nrmse": weighted_nrmse(test),
        "history": history,
        "validation": validation,
        "test": test,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-windows", type=int, default=1024)
    parser.add_argument("--max-val-windows", type=int, default=256)
    parser.add_argument("--max-test-windows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heads", default=",".join(HEADS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_heads = tuple(x.strip() for x in args.heads.split(",") if x.strip())
    unknown = sorted(set(selected_heads) - set(HEADS))
    if unknown:
        raise ValueError(f"Unknown heads: {unknown}")
    run_dir_value = os.environ.get("MCIA_RUN_DIR")
    if not run_dir_value:
        raise RuntimeError("MCIA_RUN_DIR must point to an existing run directory")
    run_dir = Path(run_dir_value).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    output_dir = run_dir / "06_diagnostics" / "mcia_output_heads_20epoch"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "mcia_output_heads_20epoch_results.json"

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} output={output_dir}")
    set_seed(args.seed)
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    loaded = {}
    for subject in (1, 2, 3, 4):
        windows, _, repetitions = prepare_data_db2(loader, [subject], config, exercises=[1])
        loaded[subject] = (windows, repetitions)
    train_limit = max(1, args.max_train_windows // 2)
    train_data = np.concatenate([
        subset_windows(loaded[s][0], loaded[s][1], (1, 3, 4, 6), train_limit, args.seed + s)
        for s in (1, 2)
    ])
    val_data = subset_windows(loaded[3][0], loaded[3][1], (2, 5), args.max_val_windows, args.seed + 30)
    test_data = subset_windows(loaded[4][0], loaded[4][1], (2, 5), args.max_test_windows, args.seed + 40)
    print(f"windows train={len(train_data)} validation={len(val_data)} test={len(test_data)}")

    results = {}
    for name in selected_heads:
        results[name] = run_head(name, train_data, val_data, test_data, config, args, device)
        partial = {
            "status": "running",
            "completed_heads": list(results),
            "models": results,
        }
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(partial, handle, ensure_ascii=False, indent=2, allow_nan=True)

    ranking = sorted(
        selected_heads,
        key=lambda name: (results[name]["weighted_validation_nrmse"], -results[name]["validation"]["s3"]["corr_masked"]),
    )
    baseline = results["linear"] if "linear" in results else None
    comparisons = {}
    if baseline is not None:
        for name in selected_heads:
            comparisons[name] = {
                "validation_weighted_nrmse_change": results[name]["weighted_validation_nrmse"] / baseline["weighted_validation_nrmse"] - 1.0,
                "test_weighted_nrmse_change": results[name]["weighted_test_nrmse"] / baseline["weighted_test_nrmse"] - 1.0,
                "validation_s3_corr_change": results[name]["validation"]["s3"]["corr_masked"] - baseline["validation"]["s3"]["corr_masked"],
                "parameter_change": results[name]["parameters"] / baseline["parameters"] - 1.0,
            }
    result = {
        "status": "complete",
        "protocol": {
            "purpose": "directional output-head screen; main experiments unchanged",
            "database": "DB2",
            "exercise": [1],
            "train_subjects": [1, 2],
            "validation_subjects": [3],
            "test_subjects": [4],
            "train_repetitions": [1, 3, 4, 6],
            "validation_repetitions": [2, 5],
            "test_repetitions": [2, 5],
            "epochs": args.epochs,
            "seed": args.seed,
            "window_counts": {"train": len(train_data), "validation": len(val_data), "test": len(test_data)},
            "data_loaded_once": True,
            "identical_mask_rng_per_head": True,
            "test_used_for_selection": False,
            "checkpoint_selection": "validation S3 masked MSE",
            "ranking_metric": "0.2*S1 + 0.4*S2 + 0.4*S3 validation masked NRMSE",
        },
        "ranking_by_validation": ranking,
        "models": results,
        "comparison_to_linear": comparisons,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print("\nValidation ranking")
    for rank, name in enumerate(ranking, 1):
        item = results[name]
        delta = comparisons.get(name, {}).get("validation_weighted_nrmse_change", float("nan"))
        print(f"{rank:2d}. {name:18s} val={item['weighted_validation_nrmse']:.5f} test={item['weighted_test_nrmse']:.5f} delta={delta:+.2%}")
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
