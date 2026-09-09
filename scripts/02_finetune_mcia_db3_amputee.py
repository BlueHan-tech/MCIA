"""
实验 2a：DB3 截肢者少样本适应生成式 sEMG 先验。

比较两个可训练的受试者特定检查点：
amputee_only（仅截肢者）：随机初始化的 MCIA，仅在 DB3 少样本数据上训练
pretrained_finetuned（预训练+微调）：在相同 DB3 少样本数据上微调的 DB2 健康先验
直接迁移基线将在后续评估中，通过直接加载 DB2 检查点而不进行受试者特定训练来实现。
"""

import sys
import time
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset_db2_emg import EMGCompletionDataset
from data.dataset_db3_emg import prepare_data_db3
from data.ninapro_loader import NinaProDataLoader
from utils.evaluation import validate_epoch_mcia_masked
from utils.paper_pipeline import (
    build_mcia,
    build_mask_generator,
    build_structural_loss,
    flatten_pipeline_config,
    load_mcia_state_dict,
    load_yaml_config,
    save_json,
    set_seed,
    subject_split_indices,
    train_mcia_epoch,
)


def load_pretrained(model, checkpoint_path: Path, device: str) -> None:
    load_mcia_state_dict(model, checkpoint_path, device)


CONDITION_MANIFEST_VERSION = "exp2_validation_selection_v1"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seeded_mask_generator(config: dict, seed: int):
    generator = build_mask_generator(config)
    generator.rng = np.random.default_rng(int(seed))
    return generator


def _split_boundary_audit(repetitions: np.ndarray, train_idx: np.ndarray,
                          val_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    train_set, val_set, test_set = map(set, (train_idx.tolist(), val_idx.tolist(), test_idx.tolist()))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Exp2 repetition split overlap detected")
    expected = {"train": {1, 3, 4}, "validation": {6}, "test": {2, 5}}
    observed = {
        "train": set(np.unique(repetitions[train_idx]).tolist()),
        "validation": set(np.unique(repetitions[val_idx]).tolist()),
        "test": set(np.unique(repetitions[test_idx]).tolist()),
    }
    if observed != expected:
        raise ValueError(f"Unexpected Exp2 repetition boundaries: {observed}")
    return {
        "split_disjoint": True,
        "repetitions": {key: sorted(values) for key, values in expected.items()},
        "window_counts": {
            "train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx)),
        },
        "optimization_data": "train windows only",
        "checkpoint_selection_data": "validation windows only",
        "test_data_usage": "held out from optimization and checkpoint selection",
    }


# ── Adapter（瓶颈残差，零初始化时等价于恒等映射）──

class Adapter(nn.Module):
    def __init__(self, D: int, bottleneck_dim: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.down = nn.Linear(D, bottleneck_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck_dim, D)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


def activate_adapters(model, device: str, n_adapter_blocks: int = 2, bottleneck_dim: int = 32) -> None:
    """
    1. 挂载 Adapter 到最近 n_adapter_blocks 个 block（插槽已存在）。
    2. 冻结全部 backbone；只解冻 adapter + domain_embed + pred_head。
    3. 注册梯度钩子：domain_embed.weight[0]（健康人域）梯度清零，只训练 [1]（截肢域）。
    """
    D = model.embed_dim
    for block in model.blocks[-n_adapter_blocks:]:
        block.adapter = Adapter(D, bottleneck_dim).to(device)

    # 冻结全部
    for p in model.parameters():
        p.requires_grad_(False)

    # 解冻：adapter
    for block in model.blocks[-n_adapter_blocks:]:
        for p in block.adapter.parameters():
            p.requires_grad_(True)

    # 解冻：domain_embed（整个 weight，用 hook 冻结行 0）
    model.domain_embed.weight.requires_grad_(True)

    def _freeze_dom0(grad):
        g = grad.clone()
        g[0].zero_()
        return g
    model.domain_embed.weight.register_hook(_freeze_dom0)

    # 解冻：协同调制 + 时序头
    if getattr(model, "synergy_bottleneck", None) is not None:
        for p in model.synergy_bottleneck.parameters():
            p.requires_grad_(True)
    for p in model.pred_head.parameters():
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [adapter] trainable params = {n_trainable:,} "
          f"(last {n_adapter_blocks} blocks x adapter + domain_embed[1] + synergy + pred_head)")


def make_optimizer(model, config, finetune: bool):
    if not finetune:
        return optim.AdamW(model.parameters(), lr=config["transfer_learning_rate"], weight_decay=1e-5)

    # Adapter 模式：backbone 已冻结，仅对可训练参数使用单一学习率组
    has_frozen = any(not p.requires_grad for p in model.parameters())
    if has_frozen:
        trainable = [p for p in model.parameters() if p.requires_grad]
        lr = float(config.get("transfer_finetune_lr_refiner", config["transfer_learning_rate"]))
        return optim.AdamW(trainable, lr=lr, weight_decay=1e-5)

    # 全量微调：双学习率（空间 backbone vs 解码器/头）
    spatial_modules = [model.patch_embed, model.local_bypass, model.blocks]
    spatial_ids = {id(p) for mod in spatial_modules for p in mod.parameters()}
    for p in [model.chan_pos, model.temp_pos, model.mask_token, model.uncond_token]:
        spatial_ids.add(id(p))
    for p in model.domain_embed.parameters():
        spatial_ids.add(id(p))

    spatial_params = [p for p in model.parameters() if id(p) in spatial_ids]
    refiner_params = [p for p in model.parameters() if id(p) not in spatial_ids]
    return optim.AdamW(
        [
            {"params": spatial_params, "lr": config["transfer_finetune_lr_spatial"]},
            {"params": refiner_params, "lr": config["transfer_finetune_lr_refiner"]},
        ],
        weight_decay=1e-5,
    )




def train_subject_model(model, train_loader, val_loader, config, device, train_mask_gen, criterion,
                        optimizer, epochs, validation_mask_seed: int, domain_id: int = 1):
    selection_metric = str(config["transfer_selection_metric"])
    selection_scenario = str(config["transfer_selection_scenario"])
    best_value = float("inf")
    best_state = None
    best_validation = None
    no_improve = 0
    history = []
    for epoch in range(epochs):
        train_loss = train_mcia_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_mask_gen,
            criterion,
            scenario=config["transfer_training_scenario"],
            cfg_dropout_prob=config.get("cfg_dropout_prob", 0.0),
            domain_id=domain_id,
            epoch=epoch,
        )
        validation_mask_gen = _seeded_mask_generator(config, validation_mask_seed)
        validation = validate_epoch_mcia_masked(
            model, val_loader, device, validation_mask_gen, criterion,
            scenario=selection_scenario, domain_id=domain_id,
        )
        selected_value = float(validation[selection_metric])
        if not np.isfinite(selected_value):
            raise RuntimeError(
                f"Non-finite validation {selection_metric} at epoch {epoch + 1}: {selected_value}"
            )
        history.append({
            "epoch": int(epoch + 1), "train_loss": float(train_loss),
            "selection_metric": selection_metric, "selection_value": selected_value,
            "validation": {key: float(value) for key, value in validation.items()},
        })
        if selected_value < best_value:
            best_value = selected_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_validation = validation
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= config["transfer_patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_value, epoch + 1, best_validation, history


def main():
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    device = config["device"]
    set_seed(int(config["transfer_random_seed"]))

    pretrained_candidates = [
        Path(config.get("exp1_dir", "")) / "checkpoints" / "best_model.pth",
        Path(config["checkpoints_dir"]) / "exp1_mcia_db2" / "best_model.pth",
    ]
    pretrained_ckpt = next((path for path in pretrained_candidates if path.exists()), pretrained_candidates[0])
    if not pretrained_ckpt.exists():
        raise FileNotFoundError(f"Missing DB2 healthy prior checkpoint: {pretrained_ckpt}")

    out_dir = Path(config["transfer_checkpoints_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    criterion = build_structural_loss(config, device)

    summary = {
        "condition_manifest_version": CONDITION_MANIFEST_VERSION,
        "checkpoint_selection": {
            "split": "validation repetition 6 only",
            "metric": str(config["transfer_selection_metric"]),
            "mode": "minimize",
            "scenario": str(config["transfer_selection_scenario"]),
        },
        "subjects": [],
    }
    start_time = time.time()
    print("=" * 80)
    print("[Exp2a] DB3 amputee few-shot MAE adaptation")
    print("=" * 80)

    for subject_id in config["transfer_db3_subjects"]:
        print(f"\n--- DB3 S{subject_id:02d} ---")
        try:
            segments, _, segment_meta = prepare_data_db3(
                data_loader, [subject_id], config,
                exercises=config["transfer_exercises"], return_metadata=True,
            )
        except Exception as exc:
            print(f"  skipped: {exc}")
            summary["subjects"].append({"subject_id": subject_id, "status": "skipped", "reason": str(exc)})
            continue

        repetitions = np.asarray(segment_meta["repetition"], dtype=np.int32)
        train_idx = np.flatnonzero(np.isin(repetitions, config["transfer_train_repetitions"]))
        val_idx = np.flatnonzero(np.isin(repetitions, config["transfer_validation_repetitions"]))
        test_idx = np.flatnonzero(np.isin(repetitions, config["transfer_test_repetitions"]))
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            print("  skipped: incomplete fixed repetition split")
            summary["subjects"].append({"subject_id": subject_id, "status": "skipped", "reason": "incomplete fixed repetition split"})
            continue

        train_set = EMGCompletionDataset(segments[train_idx])
        val_set = EMGCompletionDataset(segments[val_idx])
        train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)
        subj_dir = out_dir / f"S{subject_id:02d}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        split_audit = _split_boundary_audit(repetitions, train_idx, val_idx, test_idx)
        validation_mask_seed = (
            int(config["transfer_random_seed"])
            + int(config["transfer_selection_mask_seed_offset"])
            + int(subject_id)
        )

        results = {
            "subject_id": subject_id,
            "n_total": int(len(segments)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "split": {
                "method": "fixed_repetition_1_3_4__6__2_5",
                "train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist(),
                "source_exercises": [int(value) for value in config["transfer_exercises"]],
                "normalization": segment_meta["normalization"],
            },
            "data_boundary_audit": split_audit,
            "normalization_stats": segment_meta.get("normalization_stats", []),
            "quality_mask_reports": segment_meta.get("quality_mask_reports", []),
            "checkpoint_selection": {
                "metric": str(config["transfer_selection_metric"]),
                "mode": "minimize",
                "scenario": str(config["transfer_selection_scenario"]),
                "mask_seed": validation_mask_seed,
                "validation_repetitions": [int(value) for value in config["transfer_validation_repetitions"]],
            },
        }

        # ── 直接迁移（Exp2b）：不微调，仅复制 DB2 检查点 ──
        import shutil
        dt_path = subj_dir / "direct_transfer.pth"
        shutil.copy2(pretrained_ckpt, dt_path)
        results["direct_transfer"] = {
            "epochs": 0, "note": "DB2 healthy prior, no adaptation",
            "source_checkpoint_sha256": _sha256_file(pretrained_ckpt),
            "checkpoint_sha256": _sha256_file(dt_path),
            "domain_id": None, "adapters_loaded": False,
        }
        print(f"  direct_transfer: copied {pretrained_ckpt.name}")

        # ── 健康预训练 + 截肢者微调（Exp2a pretrained_finetuned）──
        pretrained_seed = int(config["transfer_random_seed"]) + int(subject_id) * 100 + 1
        set_seed(pretrained_seed)
        model = build_mcia(config, device)
        load_pretrained(model, pretrained_ckpt, device)
        activate_adapters(
            model, device,
            n_adapter_blocks=int(config.get("transfer_adapter_blocks", 2)),
            bottleneck_dim=int(config.get("transfer_adapter_bottleneck", 32)),
        )
        optimizer = make_optimizer(model, config, finetune=True)
        best_value, epochs, best_validation, history = train_subject_model(
            model, train_loader, val_loader, config, device,
            _seeded_mask_generator(config, pretrained_seed + int(config["transfer_training_mask_seed_offset"])),
            criterion, optimizer, int(config["transfer_finetune_epochs"]),
            validation_mask_seed, domain_id=1,
        )
        ft_path = subj_dir / "pretrained_finetuned.pth"
        torch.save({
            "model": model.state_dict(), "subject_id": subject_id, "mode": "pretrained_finetuned",
            "selection": {"metric": config["transfer_selection_metric"], "value": best_value,
                          "scenario": config["transfer_selection_scenario"], "best_validation": best_validation},
        }, ft_path)
        results["pretrained_finetuned"] = {
            "best_validation_value": float(best_value), "best_validation": best_validation,
            "epochs": int(epochs), "history": history, "seed": pretrained_seed,
            "checkpoint_sha256": _sha256_file(ft_path), "domain_id": 1, "adapters_loaded": True,
        }
        print(f"  pretrained_finetuned: val_{config['transfer_selection_metric']}={best_value:.6f} epochs={epochs}")

        # ── 仅截肢者训练，相同少样本窗口（Exp2a amputee_only）──
        amputee_seed = int(config["transfer_random_seed"]) + int(subject_id) * 100 + 2
        set_seed(amputee_seed)
        model = build_mcia(config, device)
        optimizer = make_optimizer(model, config, finetune=False)
        best_value, epochs, best_validation, history = train_subject_model(
            model, train_loader, val_loader, config, device,
            _seeded_mask_generator(config, amputee_seed + int(config["transfer_training_mask_seed_offset"])),
            criterion, optimizer, int(config["transfer_amputee_only_epochs"]),
            validation_mask_seed, domain_id=1,
        )
        ao_path = subj_dir / "amputee_only.pth"
        torch.save({
            "model": model.state_dict(), "subject_id": subject_id, "mode": "amputee_only",
            "selection": {"metric": config["transfer_selection_metric"], "value": best_value,
                          "scenario": config["transfer_selection_scenario"], "best_validation": best_validation},
        }, ao_path)
        results["amputee_only"] = {
            "best_validation_value": float(best_value), "best_validation": best_validation,
            "epochs": int(epochs), "history": history, "seed": amputee_seed,
            "checkpoint_sha256": _sha256_file(ao_path), "domain_id": 1, "adapters_loaded": False,
        }
        print(f"  amputee_only: val_{config['transfer_selection_metric']}={best_value:.6f} epochs={epochs}")

        condition_manifest = {
            "version": CONDITION_MANIFEST_VERSION,
            "subject_id": int(subject_id), "healthy_prior_checkpoint": str(pretrained_ckpt),
            "healthy_prior_checkpoint_sha256": _sha256_file(pretrained_ckpt),
            "implementation_sha256": {
                "config": _sha256_file(PROJECT_ROOT / "config.yaml"),
                "exp2_script": _sha256_file(Path(__file__)),
                "db3_dataset": _sha256_file(PROJECT_ROOT / "data" / "dataset_db3_emg.py"),
                "validation": _sha256_file(PROJECT_ROOT / "utils" / "evaluation.py"),
            },
            "data_boundary_audit": split_audit,
            "normalization_stats": results["normalization_stats"],
            "quality_mask_reports": results["quality_mask_reports"],
            "quality_mask_data_boundary": {
                "hard_zero_fit_repetitions": [1, 3, 4],
                "mqp": "fixed per-input-second quality calculation; no optimization or checkpoint selection",
            },
            "checkpoint_selection": results["checkpoint_selection"],
            "modes": {
                mode: {key: value for key, value in results[mode].items() if key != "history"}
                for mode in ("direct_transfer", "pretrained_finetuned", "amputee_only")
            },
        }
        save_json(subj_dir / "conditions.json", condition_manifest)
        results["condition_manifest"] = str(subj_dir / "conditions.json")
        save_json(subj_dir / "split.json", results)
        summary["subjects"].append({"status": "ok", **results})

    summary["elapsed_min"] = (time.time() - start_time) / 60.0
    save_json(out_dir / "finetune_summary.json", summary)
    print(f"\nSaved DB3 adaptation checkpoints to: {out_dir}")


if __name__ == "__main__":
    main()

