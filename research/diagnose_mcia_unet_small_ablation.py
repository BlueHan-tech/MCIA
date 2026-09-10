"""Small DB2 ablation: flat MCIA head versus a temporal U-Net token decoder.

This diagnostic is intentionally isolated from the Exp1/Exp2/Exp3 main path. It
uses fixed subjects, repetitions, masks, and seeds to test one architecture
direction quickly before any full-scale training is considered.
"""

from __future__ import annotations

import argparse
import copy
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
import yaml
from torch.utils.data import DataLoader

from data.dataset_db2_emg import EMGCompletionDataset, prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from models.completion.mask_generators import ScenarioMixMaskGenerator
from models.completion.mcia_core import MCIA
from utils.loss_functions import EMGImputationLoss


SCENARIOS = ("s1", "s2", "s3")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config() -> Dict:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        root = yaml.safe_load(handle)
    config = {
        "db2_path": root["paths"]["db2"],
        "db3_path": root["paths"]["db3"],
        "orig_fs": root["signal"]["orig_fs"],
        "target_fs": root["signal"]["target_fs"],
        "window_size": root["signal"]["window_size"],
        "stride": root["signal"]["stride"],
        "n_channels": root["signal"]["n_channels"],
    }
    config.update(root["exp1_mcia"])
    return config


class TemporalUNetResidual(nn.Module):
    """Identity-initialized two-scale temporal U-Net over MCIA patch tokens."""

    def __init__(self, embed_dim: int):
        super().__init__()

        def block(stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=stride, padding=1),
                nn.GroupNorm(1, embed_dim),
                nn.GELU(),
            )

        self.down1 = block(stride=2)
        self.down2 = block(stride=2)
        self.bottleneck = block()
        self.up1 = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1)
        self.fuse1 = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.GELU(),
        )
        self.up2 = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1)
        self.fuse2 = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.GELU(),
        )
        self.out = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, channels, patches, dim = tokens.shape
        base = tokens.reshape(batch * channels, patches, dim).transpose(1, 2)
        half = self.down1(base)
        quarter = self.bottleneck(self.down2(half))
        up_half = self.up1(quarter)
        up_half = self.fuse1(torch.cat((up_half, half), dim=1))
        up_full = self.up2(up_half)
        up_full = self.fuse2(torch.cat((up_full, base), dim=1))
        refined = base + self.out(up_full)
        return refined.transpose(1, 2).reshape(batch, channels, patches, dim)


class MCIAWithTemporalUNet(MCIA):
    """The current MCIA backbone with one multiscale residual decoder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multiscale_decoder = TemporalUNetResidual(self.embed_dim)

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
        batch, time_steps, channels = x.shape
        if time_steps != self.window_size or channels != self.n_channels:
            raise ValueError(f"Expected (B,{self.window_size},{self.n_channels}), got {tuple(x.shape)}")

        if raw_time_mask is not None:
            x = x * raw_time_mask
            time_mask = self._derive_time_mask(raw_time_mask).float()
        else:
            time_mask = x.new_ones(batch, channels, self.num_patches)
        if chan_valid_mask is None and mask is not None:
            chan_valid_mask = mask.float()

        tokens = self.patch_embed(x.transpose(1, 2).contiguous())
        if drop_condition:
            tokens = self.uncond_token.expand(
                batch, channels, self.num_patches, self.embed_dim
            ).contiguous()
        else:
            token_mask = time_mask.unsqueeze(-1)
            missing_token = self.mask_token.expand(
                batch, channels, self.num_patches, self.embed_dim
            )
            tokens = tokens * token_mask + missing_token * (1.0 - token_mask)

        tokens = tokens + self.temp_pos + self.chan_pos
        if domain_id is not None:
            tokens = tokens + self.domain_embed(domain_id)[:, None, None, :]
        tokens = self.local_bypass(tokens)
        for block_module in self.blocks:
            tokens = block_module(tokens, time_mask, chan_valid_mask=chan_valid_mask)
        tokens = self.norm(tokens)
        if self.synergy_bottleneck is not None:
            tokens = self.synergy_bottleneck(tokens)
        tokens = self.multiscale_decoder(tokens)

        pred_patches = self.pred_head(tokens)
        pred = self.patch_recover(pred_patches).transpose(1, 2).contiguous()
        return {"pred": pred, "time": pred} if return_aux else pred


class DilatedResidualBlock(nn.Module):
    """Lightweight non-causal depthwise-separable TCN block."""

    def __init__(self, embed_dim: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(1, embed_dim)
        self.depthwise = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=embed_dim,
        )
        self.pointwise = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(self.norm(x))
        hidden = self.pointwise(F.gelu(hidden))
        return x + self.dropout(hidden)


class NonCausalTCNResidual(nn.Module):
    """Identity-initialized dilated TCN refinement over MCIA patch tokens."""

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(embed_dim, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.out = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, channels, patches, dim = tokens.shape
        base = tokens.reshape(batch * channels, patches, dim).transpose(1, 2)
        hidden = base
        for block_module in self.blocks:
            hidden = block_module(hidden)
        refined = base + self.out(hidden)
        return refined.transpose(1, 2).reshape(batch, channels, patches, dim)


class MCIAWithNonCausalTCN(MCIAWithTemporalUNet):
    """The current MCIA backbone with a non-causal dilated TCN decoder."""

    def __init__(self, **kwargs):
        dropout = float(kwargs.get("dropout", 0.1))
        super().__init__(**kwargs)
        self.multiscale_decoder = NonCausalTCNResidual(self.embed_dim, dropout)


def build_model(
    config: Dict,
    candidate: bool,
    candidate_decoder: str = "temporal_unet",
) -> nn.Module:
    if not candidate:
        cls = MCIA
    elif candidate_decoder == "noncausal_tcn":
        cls = MCIAWithNonCausalTCN
    elif candidate_decoder == "covariance_loss":
        cls = MCIA
    else:
        cls = MCIAWithTemporalUNet
    return cls(
        window_size=config["window_size"],
        n_channels=config["n_channels"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        ffn_dim=config["ffn_dim"],
        dropout=config["model_dropout"],
        num_domains=config.get("num_domains", 2),
        use_synergy_bottleneck=config.get("use_synergy_bottleneck", False),
        n_synergies=config.get("n_synergies", 6),
        synergy_gate_scale=config.get("synergy_gate_scale", 0.5),
        synergy_dropout=config.get("synergy_dropout", 0.1),
    )


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


def subset_windows(
    data: np.ndarray,
    repetitions: np.ndarray,
    allowed_repetitions: Iterable[int],
    limit: int,
    seed: int,
) -> np.ndarray:
    selected = data[np.isin(repetitions, list(allowed_repetitions))]
    if limit > 0 and len(selected) > limit:
        indices = np.random.default_rng(seed).choice(len(selected), size=limit, replace=False)
        selected = selected[np.sort(indices)]
    return selected


def make_loader(data: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        EMGCompletionDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


class EMGImputationLossWithCovariance(EMGImputationLoss):
    """Add an off-diagonal normalized channel-covariance constraint."""

    def __init__(self, covariance_weight: float, **kwargs):
        super().__init__(**kwargs)
        self.covariance_weight = float(covariance_weight)

    @staticmethod
    def _normalized_covariance(x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()
        normalized = centered / scale
        return torch.einsum("btc,btd->bcd", normalized, normalized) / x.shape[1]

    def forward(
        self,
        pred_x0: torch.Tensor | Dict[str, torch.Tensor],
        target_x0: torch.Tensor,
        mask: torch.Tensor,
    ):
        total, parts = super().forward(pred_x0, target_x0, mask)
        pred_main = (
            pred_x0.get("time", pred_x0.get("pred"))
            if isinstance(pred_x0, dict)
            else pred_x0
        )
        missing = (1.0 - mask).float()
        completed = pred_main * missing + target_x0 * mask
        pred_cov = self._normalized_covariance(completed)
        target_cov = self._normalized_covariance(target_x0)

        missing_fraction = missing.mean(dim=1)
        pair_weight = torch.maximum(
            missing_fraction.unsqueeze(2), missing_fraction.unsqueeze(1)
        )
        channels = target_x0.shape[2]
        off_diagonal = 1.0 - torch.eye(
            channels, device=target_x0.device, dtype=target_x0.dtype
        ).unsqueeze(0)
        pair_weight = pair_weight * off_diagonal
        covariance_loss = (
            (pred_cov - target_cov).abs() * pair_weight
        ).sum() / pair_weight.sum().clamp_min(1.0)
        total = total + self.covariance_weight * covariance_loss
        parts["covariance_loss"] = float(covariance_loss.detach().item())
        return total, parts


def make_loss(
    config: Dict,
    device: torch.device,
    covariance_weight: float = 0.0,
) -> EMGImputationLoss:
    loss_class = (
        EMGImputationLossWithCovariance
        if covariance_weight > 0.0
        else EMGImputationLoss
    )
    kwargs = dict(
        w_charbonnier=config.get("loss_charbonnier", 1.0),
        w_ncc=config.get("loss_ncc", 0.5),
        w_stft=config.get("loss_stft", 0.3),
        w_boundary=config.get("loss_boundary", 0.1),
        w_aux=config.get("loss_aux", 0.1),
        aux_ratio=config.get("aux_mask_ratio", 0.10),
        fft_sizes=config.get("loss_fft_sizes", [16, 32, 64]),
    )
    if covariance_weight > 0.0:
        kwargs["covariance_weight"] = covariance_weight
    return loss_class(**kwargs).to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: EMGImputationLoss,
    mask_generator: ScenarioMixMaskGenerator,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        target = batch["data"].to(device)
        raw_mask = mask_generator.generate_mask(target)
        channel_mask = raw_mask.max(dim=1).values
        masked = target * raw_mask
        pred = model(
            masked,
            mask=channel_mask,
            x_masked=masked,
            raw_time_mask=raw_mask,
            return_aux=True,
        )
        loss, _ = criterion(pred, target, raw_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = target.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    config: Dict,
    scenario: str,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    mask_generator = build_mask_generator(config, seed)
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    correlations = []
    mask_count = 0
    total_count = 0

    for batch in loader:
        target = batch["data"].to(device)
        raw_mask = mask_generator.generate_mask(target, scenario=scenario)
        channel_mask = raw_mask.max(dim=1).values
        masked = target * raw_mask
        pred = model(masked, mask=channel_mask, x_masked=masked, raw_time_mask=raw_mask)
        completed = pred * (1.0 - raw_mask) + target * raw_mask
        missing = raw_mask < 0.5
        diff = completed - target
        squared_error += float(diff[missing].square().sum().item())
        absolute_error += float(diff[missing].abs().sum().item())
        count += int(missing.sum().item())
        mask_count += int(missing.sum().item())
        total_count += int(raw_mask.numel())

        pred_np = completed.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        missing_np = missing.detach().cpu().numpy()
        for batch_index in range(target_np.shape[0]):
            for channel in range(target_np.shape[2]):
                region = missing_np[batch_index, :, channel]
                if region.sum() < 4:
                    continue
                p = pred_np[batch_index, region, channel]
                y = target_np[batch_index, region, channel]
                if np.std(p) < 1e-8 or np.std(y) < 1e-8:
                    continue
                correlations.append(float(np.corrcoef(p, y)[0, 1]))

    mse = squared_error / max(1, count)
    rmse = math.sqrt(mse)
    return {
        "mse_masked": mse,
        "rmse_masked": rmse,
        "nrmse_masked": rmse,
        "psnr_masked_db": -10.0 * math.log10(max(mse, 1e-12)),
        "mae_masked": absolute_error / max(1, count),
        "corr_masked": float(np.mean(correlations)) if correlations else float("nan"),
        "mask_ratio": mask_count / max(1, total_count),
        "n_masked_values": count,
    }


def run_model(
    name: str,
    candidate: bool,
    candidate_decoder: str,
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    config: Dict,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    set_seed(args.seed)
    model = build_model(
        config, candidate=candidate, candidate_decoder=candidate_decoder
    ).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    set_seed(args.seed + 1000)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    covariance_weight = (
        args.lambda_covariance
        if candidate and candidate_decoder == "covariance_loss"
        else 0.0
    )
    criterion = make_loss(config, device, covariance_weight=covariance_weight)
    mask_generator = build_mask_generator(config, args.seed + 2000)
    train_loader = make_loader(train_data, args.batch_size, True, args.seed + 3000)
    val_loader = make_loader(val_data, args.batch_size, False, args.seed + 4000)
    test_loader = make_loader(test_data, args.batch_size, False, args.seed + 5000)

    best_mse = float("inf")
    best_state = None
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, mask_generator, device
        )
        validation = evaluate(
            model, val_loader, config, "s3", args.seed + 6000, device
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_s3": validation})
        print(
            f"[{name}] epoch={epoch}/{args.epochs} loss={train_loss:.5f} "
            f"val_s3_nrmse={validation['nrmse_masked']:.5f} "
            f"corr={validation['corr_masked']:.4f}"
        )
        if validation["mse_masked"] < best_mse:
            best_mse = validation["mse_masked"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        model.load_state_dict(initial_state)
    else:
        model.load_state_dict(best_state)

    validation = {
        scenario: evaluate(
            model, val_loader, config, scenario, args.seed + 7000 + index, device
        )
        for index, scenario in enumerate(SCENARIOS)
    }
    test = {
        scenario: evaluate(
            model, test_loader, config, scenario, args.seed + 8000 + index, device
        )
        for index, scenario in enumerate(SCENARIOS)
    }
    return {
        "name": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selection": "lowest S03 validation S3 masked MSE",
        "best_validation_s3_mse": best_mse,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "validation": validation,
        "test": test,
    }


def relative_change(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / baseline if baseline else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-windows", type=int, default=1024)
    parser.add_argument("--max-val-windows", type=int, default=256)
    parser.add_argument("--max-test-windows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidate-decoder",
        choices=("temporal_unet", "noncausal_tcn", "covariance_loss"),
        default="temporal_unet",
    )
    parser.add_argument("--lambda-covariance", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir_value = os.environ.get("MCIA_RUN_DIR")
    if not run_dir_value:
        raise RuntimeError("MCIA_RUN_DIR must point to an existing run directory")
    run_dir = Path(run_dir_value).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if args.candidate_decoder == "noncausal_tcn":
        output_tag = "mcia_tcn_decoder_small_ablation"
    elif args.candidate_decoder == "covariance_loss":
        output_tag = "mcia_covariance_loss_small_ablation"
    else:
        output_tag = "mcia_unet_small_ablation"
    output_dir = run_dir / "06_diagnostics" / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} output={output_dir}")
    data_loader = NinaProDataLoader(
        config["db2_path"], config["db3_path"], fs=config["orig_fs"]
    )

    loaded = {}
    for subject in (1, 2, 3, 4):
        windows, _, repetitions = prepare_data_db2(
            data_loader, [subject], config, exercises=[1]
        )
        loaded[subject] = (windows, repetitions)

    train_parts = []
    per_subject_limit = max(1, args.max_train_windows // 2)
    for subject in (1, 2):
        windows, repetitions = loaded[subject]
        train_parts.append(
            subset_windows(
                windows, repetitions, (1, 3, 4, 6), per_subject_limit, args.seed + subject
            )
        )
    train_data = np.concatenate(train_parts, axis=0)
    val_data = subset_windows(
        loaded[3][0], loaded[3][1], (2, 5), args.max_val_windows, args.seed + 30
    )
    test_data = subset_windows(
        loaded[4][0], loaded[4][1], (2, 5), args.max_test_windows, args.seed + 40
    )
    print(f"windows train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    baseline = run_model(
        "mcia_flat", False, args.candidate_decoder,
        train_data, val_data, test_data, config, args, device
    )
    if args.candidate_decoder == "noncausal_tcn":
        candidate_name = "mcia_noncausal_tcn"
    elif args.candidate_decoder == "covariance_loss":
        candidate_name = "mcia_covariance_loss"
    else:
        candidate_name = "mcia_temporal_unet"
    candidate = run_model(
        candidate_name, True, args.candidate_decoder,
        train_data, val_data, test_data, config, args, device
    )

    comparisons = {}
    for split in ("validation", "test"):
        comparisons[split] = {}
        for scenario in SCENARIOS:
            base_metrics = baseline[split][scenario]
            candidate_metrics = candidate[split][scenario]
            comparisons[split][scenario] = {
                "nrmse_relative_change": relative_change(
                    candidate_metrics["nrmse_masked"], base_metrics["nrmse_masked"]
                ),
                "corr_absolute_change": (
                    candidate_metrics["corr_masked"] - base_metrics["corr_masked"]
                ),
            }

    result = {
        "protocol": {
            "purpose": "directional small-scale architecture ablation only",
            "database": "DB2",
            "exercises": [1],
            "train_subjects": [1, 2],
            "validation_subjects": [3],
            "test_subjects": [4],
            "train_repetitions": [1, 3, 4, 6],
            "validation_repetitions": [2, 5],
            "test_repetitions": [2, 5],
            "selection_uses_test": False,
            "epochs": args.epochs,
            "seed": args.seed,
            "candidate_decoder": args.candidate_decoder,
            "lambda_covariance": (
                args.lambda_covariance
                if args.candidate_decoder == "covariance_loss"
                else 0.0
            ),
            "window_counts": {
                "train": len(train_data),
                "validation": len(val_data),
                "test": len(test_data),
            },
            "metric_scope": "masked values only; normalized data_range=1",
        },
        "models": {"mcia_flat": baseline, candidate_name: candidate},
        "comparison": comparisons,
        "decision_rule": {
            "promising": "validation NRMSE decreases by >=5% without lower correlation",
            "test_role": "independent confirmation after validation-only selection",
        },
    }
    result_path = output_dir / f"{output_tag}_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=True)

    print("\nCompact comparison (candidate vs baseline)")
    for split in ("validation", "test"):
        for scenario in SCENARIOS:
            item = comparisons[split][scenario]
            print(
                f"{split:10s} {scenario}: nrmse_change={item['nrmse_relative_change']:+.2%} "
                f"corr_change={item['corr_absolute_change']:+.4f}"
            )
    print(f"results={result_path}")


if __name__ == "__main__":
    main()

