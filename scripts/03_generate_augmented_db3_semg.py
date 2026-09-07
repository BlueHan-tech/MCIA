"""
实验 2c：生成 DB3 增强残差 sEMG。

默认论文路径使用受试者特定的预训练/微调 MAE 检查点，
并结合轻量级受控伪缺失掩码，在保留掩码区域之外所有观测样本的同时，
生成增强表征。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.dataset_db3_emg import prepare_data_db3
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import (
    build_mcia,
    complete_with_mask,
    flatten_pipeline_config,
    load_mcia_state_dict,
    load_yaml_config,
    make_controlled_mask,
    save_json,
    set_seed,
)
from utils.rule_anomaly_detector import RuleAnomalyDetector
from utils.visualization import plot_completion_panel


def _load_model_from_path(path: Path, config: dict, device: str):
    if not path.exists():
        raise FileNotFoundError(path)
    model = build_mcia(config, device)
    load_mcia_state_dict(model, path, device)
    model.eval()
    return model, path


def load_direct_model(config, device):
    candidates = [
        Path(config["transfer_checkpoints_dir"]) / "direct_transfer.pth",
        Path(config.get("exp1_dir", "")) / "checkpoints" / "best_model.pth",
        Path(config["checkpoints_dir"]) / "exp1_mcia_db2" / "best_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return _load_model_from_path(path, config, device)
    raise FileNotFoundError("No DB2 healthy-prior MCIA checkpoint found")


def get_config_value(config: dict, *keys: str, default=None):
    for key in keys:
        if key in config:
            return config[key]
    return default


def load_subject_model(subject_id, config, device):
    candidates = [
        Path(config["transfer_checkpoints_dir"]) / f"S{subject_id:02d}" / "pretrained_finetuned.pth",
        Path(config["transfer_checkpoints_dir"]) / f"S{subject_id:02d}" / "direct_transfer.pth",
        Path(config.get("exp1_dir", "")) / "checkpoints" / "best_model.pth",
        Path(config["checkpoints_dir"]) / "exp1_mcia_db2" / "best_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return _load_model_from_path(path, config, device)
    raise FileNotFoundError("No pretrained or subject-specific MCIA checkpoint found")


def main():
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    device = config["device"]
    set_seed(int(config["transfer_random_seed"]))

    output_dir = Path(config["transfer_augmented_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])

    mask_ratio = float(cfg.get("exp2_transfer", {}).get("augmentation_mask_ratio", 0.10))
    mask_mode = str(cfg.get("exp2_transfer", {}).get("augmentation_mask_mode", "rule"))
    if mask_mode not in {"rule", "controlled"}:
        raise ValueError(f"Unsupported augmentation_mask_mode={mask_mode!r}; use 'rule' or 'controlled'.")

    print("=" * 80)
    print("[Exp2c] Generate DB3 augmented residual sEMG")
    print("=" * 80)

    manifest = {"subjects": [], "mask_mode": mask_mode, "augmentation_mask_ratio": mask_ratio}
    for subject_id in config["transfer_db3_subjects"]:
        print(f"\n--- DB3 S{subject_id:02d} ---")
        try:
            segments, _, segment_meta = prepare_data_db3(
                data_loader, [subject_id], config,
                exercises=config["transfer_exercises"], return_metadata=True,
            )
            direct_model, direct_ckpt_path = load_direct_model(config, device)
            model, ckpt_path = load_subject_model(subject_id, config, device)
        except Exception as exc:
            print(f"  skipped: {exc}")
            manifest["subjects"].append({"subject_id": subject_id, "status": "skipped", "reason": str(exc)})
            continue

        rule_outputs = None
        if mask_mode == "rule":
            train_idx = np.flatnonzero(np.isin(
                np.asarray(segment_meta["repetition"], dtype=np.int32),
                config["transfer_train_repetitions"],
            ))
            detector = RuleAnomalyDetector(
                patch_size=config.get("patch_size", 8),
                group_indices=config.get("group_indices"),
            ).fit(segments[train_idx])
            rule_outputs = detector.detect_batch(segments)
            print(
                f"  rule mask: mean={(rule_outputs['mask'] < 0.5).mean():.2%} "
                f"| dead={[int(c + 1) for c in rule_outputs['dead_channels']]}"
            )

        dataset = TensorDataset(torch.FloatTensor(segments))
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        all_enhanced, all_direct_enhanced, all_masks = [], [], []
        offset = 0
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                if mask_mode == "rule":
                    masks_np = rule_outputs["mask"][offset:offset + len(batch)]
                    mask = torch.FloatTensor(masks_np).to(device)
                else:
                    mask = make_controlled_mask(
                        tuple(batch.shape),
                        ratio=mask_ratio,
                        patch_size=config["patch_size"],
                        device=device,
                        seed=int(config["transfer_random_seed"]) + subject_id + offset,
                    )
                direct_enhanced = complete_with_mask(direct_model, batch, mask, domain_id=1)
                enhanced = complete_with_mask(model, batch, mask, domain_id=1)
                all_direct_enhanced.append(direct_enhanced.cpu().numpy())
                all_enhanced.append(enhanced.cpu().numpy())
                all_masks.append(mask.cpu().numpy())
                offset += len(batch)

        enhanced = np.concatenate(all_enhanced, axis=0)
        direct_enhanced = np.concatenate(all_direct_enhanced, axis=0)
        masks = np.concatenate(all_masks, axis=0)
        save_path = output_dir / f"db3_S{subject_id:02d}.npz"
        payload = {
            "original": segments,
            "enhanced": enhanced,
            "direct_enhanced": direct_enhanced,
            "mask": masks,
            "checkpoint": str(ckpt_path),
            "direct_checkpoint": str(direct_ckpt_path),
            "mask_mode": mask_mode,
            "augmentation_mask_ratio": mask_ratio,
        }
        if rule_outputs is not None:
            payload.update({
                "mask_labels": rule_outputs["labels"],
                "patch_mask": rule_outputs["patch_mask"],
                "patch_labels": rule_outputs["patch_labels"],
                "unrecoverable_patches": rule_outputs["unrecoverable_patches"],
                "mask_ratio": rule_outputs["mask_ratio"],
                "unrecoverable_ratio": rule_outputs["unrecoverable_ratio"],
                "dead_channels": rule_outputs["dead_channels"],
            })
        np.savez(save_path, **payload)
        generate_panels = bool(get_config_value(
            config,
            "transfer_generate_db3_completion_panels",
            "generate_db3_completion_panels",
            default=True,
        ))
        if generate_panels:
            viz_files = save_subject_semg_panels(output_dir, subject_id, segments, direct_enhanced, enhanced, masks, config)
        else:
            viz_files = []
        manifest["subjects"].append(
            {
                "subject_id": subject_id,
                "status": "ok",
                "n_segments": int(len(segments)),
                "detector_fit_scope": "train_repetitions_1_3_4_only",
                "checkpoint": str(ckpt_path),
                "direct_checkpoint": str(direct_ckpt_path),
                "file": str(save_path),
                "masked_ratio": float((masks < 0.5).mean()),
                "dead_channels_1based": (
                    [int(c + 1) for c in rule_outputs["dead_channels"]]
                    if rule_outputs is not None else []
                ),
                "viz_files": [str(p) for p in viz_files],
            }
        )
        print(f"  saved {save_path} | masked_ratio={(masks < 0.5).mean():.2%}")

    save_json(output_dir / "manifest.json", manifest)
    print(f"\nSaved augmented DB3 data to: {output_dir}")


def _top_unique(values: np.ndarray, limit: int, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    selected = []
    for idx in np.argsort(-np.asarray(values, dtype=float)):
        idx = int(idx)
        if idx not in exclude and np.isfinite(values[idx]):
            selected.append(idx)
        if len(selected) >= limit:
            break
    return selected


def select_sampled_indices(n_items: int, n_samples: int = 10) -> list[int]:
    if n_items <= 0:
        return []
    n = min(int(n_samples), n_items)
    return [int(i) for i in np.linspace(0, n_items - 1, n, dtype=int)]


def select_high_missing_indices(mask: np.ndarray, n_samples: int = 5,
                                exclude: set[int] | None = None) -> list[int]:
    missing_ratio = (mask < 0.5).mean(axis=(1, 2))
    return _top_unique(missing_ratio, min(int(n_samples), len(mask)), exclude or set())


def save_subject_semg_panels(output_dir: Path, subject_id: int, segments: np.ndarray,
                             direct_enhanced: np.ndarray, finetuned_enhanced: np.ndarray,
                             masks: np.ndarray, config: dict) -> list[Path]:
    run_dir = Path(config.get("run_dir", output_dir.parent.parent))
    subject_root = run_dir / "02_db3_transfer_completion" / "figures" / "12ch_completion" / f"S{subject_id:02d}"
    mode_specs = (
        ("direct_transfer", direct_enhanced, "Direct Transfer MCIA"),
        ("pretrained_finetuned", finetuned_enhanced, "Pretrained + Finetuned MCIA"),
    )

    sampled_n = int(get_config_value(
        config, "transfer_db3_viz_sampled_per_subject", "db3_viz_sampled_per_subject", default=10
    ))
    high_missing_n = int(get_config_value(
        config, "transfer_db3_viz_high_missing_per_subject", "db3_viz_high_missing_per_subject", default=5
    ))
    sampled_indices = select_sampled_indices(len(segments), sampled_n)
    high_indices = select_high_missing_indices(
        masks, high_missing_n, exclude=set(sampled_indices)
    )
    saved: list[Path] = []
    panel_kwargs = {
        "clean_label": "Raw EMG",
        "show_observed_line": False,
        "show_mask_background": True,
        "mask_background_color": "#fff2bf",
        "mask_background_alpha": 0.35,
    }

    def draw(mode_name: str, completed: np.ndarray, completed_label: str,
             idx: int, subset: str, path: Path) -> None:
        plot_completion_panel(
            emg_clean=segments[idx],
            emg_completed=completed[idx],
            mask=masks[idx],
            save_path=path,
            title=f"DB3 S{subject_id:02d} {completed_label} | {subset} segment {idx}",
            completed_label=completed_label,
            **panel_kwargs,
        )

    for mode_name, completed, completed_label in mode_specs:
        sampled_dir = subject_root / mode_name / "sampled"
        high_missing_dir = subject_root / mode_name / "high_missing"
        sampled_dir.mkdir(parents=True, exist_ok=True)
        high_missing_dir.mkdir(parents=True, exist_ok=True)

        for idx in sampled_indices:
            path = sampled_dir / f"S{subject_id:02d}_{mode_name}_sampled_seg{idx:04d}.png"
            draw(mode_name, completed, completed_label, idx, "sampled", path)
            saved.append(path)
        for rank, idx in enumerate(high_indices, start=1):
            path = high_missing_dir / f"S{subject_id:02d}_{mode_name}_high_missing_{rank:02d}_seg{idx:04d}.png"
            draw(mode_name, completed, completed_label, idx, "high-missing", path)
            saved.append(path)
    return saved


if __name__ == "__main__":
    main()



