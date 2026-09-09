"""面向论文的 sEMG 增强流水线的共享辅助函数。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.completion.mcia_core import MCIA, derive_ch_mask_from_sample_mask
from models.completion.mask_generators import ScenarioMixMaskGenerator
from utils.loss_functions import EMGImputationLoss
from utils.run_layout import apply_run_paths


def load_yaml_config(project_root: Path) -> Dict:
    with open(project_root / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_pipeline_config(cfg: Dict) -> Dict:
    flat = {
        "db2_path": cfg["paths"]["db2"],
        "db3_path": cfg["paths"]["db3"],
        "output_dir": cfg["paths"]["output"],
        "checkpoints_dir": cfg["paths"]["checkpoints"],
        "metadata_csv": cfg["paths"].get("metadata_csv"),
        "orig_fs": cfg["signal"]["orig_fs"],
        "target_fs": cfg["signal"]["target_fs"],
        "n_channels": cfg["signal"]["n_channels"],
        "window_size": cfg["signal"]["window_size"],
        "stride": cfg["signal"]["stride"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    flat.update(cfg.get("exp1_mcia", {}))
    flat.update({f"transfer_{k}": v for k, v in cfg.get("exp2_transfer", {}).items()})
    flat.update({f"regressor_{k}": v for k, v in cfg.get("exp3_regressor", {}).items()})
    flat.update({f"gesture_{k}": v for k, v in cfg.get("exp4_gesture", {}).items()})
    return apply_run_paths(flat, cfg, Path(__file__).resolve().parent.parent)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_mcia(config: Dict, device: str) -> MCIA:
    return MCIA(
        window_size=config["window_size"],
        n_channels=config.get("n_channels", 12),
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
    ).to(device)


def load_mcia_state_dict(
    model: MCIA,
    checkpoint_path,
    device: str,
    required_state_prefixes: Optional[Iterable[str]] = None,
) -> None:
    """Load MCIA weights, optionally requiring checkpoint modules used at inference."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    required_prefixes = tuple(required_state_prefixes or ())
    missing_required = [
        prefix for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in state)
    ]
    if missing_required:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing required inference weights: "
            f"{missing_required}"
        )

    if "domain_embed.weight" in state:
        ckpt_w = state["domain_embed.weight"]
        model_w = model.domain_embed.weight.data
        if ckpt_w.shape != model_w.shape:
            n_copy = min(ckpt_w.shape[0], model_w.shape[0])
            new_w = torch.zeros_like(model_w)
            new_w[:n_copy] = ckpt_w[:n_copy]
            state["domain_embed.weight"] = new_w
            print(f"  [load_mcia] domain_embed expanded "
                  f"{tuple(ckpt_w.shape)} → {tuple(model_w.shape)}, "
                  f"rows 0..{n_copy - 1} preserved")

    model_state = model.state_dict()

    unexpected = [k for k in list(state.keys()) if k not in model_state]
    dropped_required = [
        key for key in unexpected
        if any(key.startswith(prefix) for prefix in required_prefixes)
    ]
    if dropped_required:
        raise ValueError(
            f"Checkpoint {checkpoint_path} contains required weights that do not match "
            f"the instantiated inference model: {dropped_required}"
        )
    for k in unexpected:
        del state[k]
    if unexpected:
        print(f"  [load_mcia] skipped unexpected: {' '.join(unexpected)}")

    reinit = []
    for key in list(state.keys()):
        if state[key].shape != model_state[key].shape:
            state[key] = model_state[key].clone()
            reinit.append(f"{key}(ckpt->model shape fixed)")
    if reinit:
        print(f"  [load_mcia] reinit shape-mismatch: {' '.join(reinit)}")

    missing = [k for k in model_state.keys() if k not in state]
    load_result = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load_mcia] initialized missing: {' '.join(missing)}")
    if load_result.unexpected_keys:
        print(f"  [load_mcia] unexpected after load: {' '.join(load_result.unexpected_keys)}")


def build_mask_generator(config: Dict) -> ScenarioMixMaskGenerator:
    return ScenarioMixMaskGenerator(
        n_channels=config.get("n_channels", 12),
        time_steps=config["window_size"],
        patch_size=config["patch_size"],
        group_indices=config.get("group_indices"),
        min_alive_per_group=config.get("min_alive_per_group"),
        scenario_weights=config.get("scenario_weights"),
        scenario_params=config.get("scenario_params"),
    )


def make_controlled_mask(
    shape: Tuple[int, int, int],
    ratio: float,
    patch_size: int,
    device: str | torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Create a deterministic sample-level observation mask in (B, T, C) format."""
    if len(shape) != 3:
        raise ValueError(f"Expected shape=(B,T,C), got {shape!r}")
    B, T, C = (int(v) for v in shape)
    if B <= 0 or T <= 0 or C <= 0:
        raise ValueError(f"Mask shape must be positive, got {shape!r}")

    ratio = float(np.clip(ratio, 0.0, 1.0))
    patch_size = max(1, int(patch_size))
    n_patches = int(np.ceil(T / patch_size))
    n_missing = int(round(n_patches * C * ratio))
    n_missing = min(max(n_missing, 0), n_patches * C)

    mask_patch = torch.ones((B, n_patches, C), dtype=torch.float32, device=device)
    if n_missing == 0:
        return mask_patch.repeat_interleave(patch_size, dim=1)[:, :T, :]

    gen_device = torch.device(device)
    generator = torch.Generator(device=gen_device)
    if seed is not None:
        generator.manual_seed(int(seed))

    for b in range(B):
        perm = torch.randperm(n_patches * C, generator=generator, device=device)[:n_missing]
        patch_idx = torch.div(perm, C, rounding_mode="floor")
        channel_idx = perm.remainder(C)
        mask_patch[b, patch_idx, channel_idx] = 0.0
    return mask_patch.repeat_interleave(patch_size, dim=1)[:, :T, :]

def build_structural_loss(config: Dict, device: str) -> Optional[EMGImputationLoss]:
    if not config.get("use_structural_loss", True):
        return None
    return EMGImputationLoss(
        w_charbonnier=config.get("loss_charbonnier", 1.0),
        w_ncc=config.get("loss_ncc", 0.5),
        w_stft=config.get("loss_stft", 0.3),
        w_boundary=config.get("loss_boundary", 0.1),
        w_aux=config.get("loss_aux", 0.1),
        aux_ratio=config.get("aux_mask_ratio", 0.10),
        fft_sizes=config.get("loss_fft_sizes", [32, 64, 128]),
        envelope_loss_weight=config.get("envelope_loss_weight", 0.0),
        patch_rms_loss_weight=config.get("patch_rms_loss_weight", 0.0),
        envelope_kernel_size=config.get("envelope_kernel_size", 25),
        patch_rms_size=config.get("patch_rms_size", config.get("patch_size", 8)),
    ).to(device)


def subject_split_indices(
    n_items: int,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(n_items * train_ratio)
    n_val = int(n_items * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    if len(test_idx) == 0 and len(val_idx) > 0:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]
    return train_idx, val_idx, test_idx



def train_mcia_epoch(
    model: MCIA,
    dataloader,
    optimizer,
    device: str,
    mask_gen: ScenarioMixMaskGenerator,
    criterion: Optional[EMGImputationLoss],
    scenario: Optional[str] = None,
    cfg_dropout_prob: float = 0.0,
    domain_id: Optional[int] = None,
    epoch: int = 0,
) -> float:
    model.train()
    if criterion is not None and hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)
    total_loss = 0.0
    for batch in dataloader:
        emg_clean = batch["data"].to(device) if isinstance(batch, dict) else batch.to(device)
        B, T, C = emg_clean.shape
        mask = mask_gen.generate_batch_masks(
            B, n_channels=C, time_steps=T, device=device, scenario=scenario
        ).transpose(1, 2)
        mask = (mask > 0.5).float()
        emg_masked = emg_clean * mask
        mask_1d = derive_ch_mask_from_sample_mask(mask)
        domain_id_t = (
            torch.full((B,), domain_id, dtype=torch.long, device=device)
            if domain_id is not None else None
        )
        pred = model(
            emg_masked,
            mask=mask_1d,
            x_masked=emg_masked,
            drop_condition=(torch.rand(1).item() < cfg_dropout_prob),
            raw_time_mask=mask,
            domain_id=domain_id_t,
            return_aux=(criterion is not None),
        )
        if criterion is None:
            loss = F.mse_loss(pred, emg_clean)
        else:
            loss, _ = criterion(pred, emg_clean, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def complete_with_mask(
    model: MCIA,
    emg: torch.Tensor,
    mask: torch.Tensor,
    domain_id: Optional[int] = None,
) -> torch.Tensor:
    """使用样本级掩码 (B,T,C) 补全 EMG，并保留已观测采样点。"""
    emg_masked = emg * mask
    mask_1d = derive_ch_mask_from_sample_mask(mask)
    B = emg.shape[0]
    domain_id_t = (
        torch.full((B,), domain_id, dtype=torch.long, device=emg.device)
        if domain_id is not None else None
    )
    pred = model(
        emg_masked,
        mask=mask_1d,
        x_masked=emg_masked,
        drop_condition=False,
        raw_time_mask=mask,
        domain_id=domain_id_t,
    )
    return pred.clamp(0.0, 1.0) * (1.0 - mask) + emg * mask


def safe_pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) < 4 or len(b) < 4:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-12:
        return float("nan")
    r = np.sum(a * b) / denom
    return float(r) if np.isfinite(r) else float("nan")

def masked_completion_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    missing = mask < 0.5
    if missing.sum() == 0:
        missing = np.ones_like(mask, dtype=bool)
    diff = pred[missing] - target[missing]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    corrs = []
    corrs_partial = []
    corrs_chmiss = []
    for b in range(target.shape[0]):
        for c in range(target.shape[2]):
            m = missing[b, :, c]
            if m.sum() < 4:
                continue
            x = pred[b, m, c]
            y = target[b, m, c]
            if np.std(x) > 1e-8 and np.std(y) > 1e-8:
                cc = safe_pearson_np(x, y)
                corrs.append(cc)
                if mask[b, :, c].max() >= 0.5:
                    corrs_partial.append(cc)
                else:
                    corrs_chmiss.append(cc)
    return {
        "rmse": rmse,
        "mae": mae,
        "mse_masked": mse,
        "corr": float(np.mean(corrs)) if corrs else float("nan"),
        "corr_masked_partial": float(np.mean(corrs_partial)) if corrs_partial else float("nan"),
        "corr_masked_chmiss": float(np.mean(corrs_chmiss)) if corrs_chmiss else float("nan"),
        "frequency_similarity": frequency_similarity(pred * (1.0 - mask), target * (1.0 - mask)),
    }


def frequency_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    pred_fft = np.abs(np.fft.rfft(pred, axis=1))
    target_fft = np.abs(np.fft.rfft(target, axis=1))
    num = np.sum(pred_fft * target_fft, axis=1)
    den = np.linalg.norm(pred_fft, axis=1) * np.linalg.norm(target_fft, axis=1) + 1e-8
    sim = num / den
    return float(np.nanmean(sim))


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


