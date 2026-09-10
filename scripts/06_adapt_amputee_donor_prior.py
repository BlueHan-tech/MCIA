"""Protocol §5 C-group (amputee-donor prior) development screen.

Adapts the frozen healthy DB2 MCIA prior on the clean donor pool
(S02/S04/S08/S11, E1+E2, repetitions 1/3/4 only; donor repetition 6 only for
checkpoint selection).  Candidate settings are compared on the DEV subjects
S05/S06 using train repetitions to fit each Key10 TCN and validation
repetition 6 only for reporting.  Test repetitions of every subject are never
touched.  Pre-declared selection rule: lowest mean S05/S06 validation global
RMSE across candidates; B (healthy prior, domain_id=None) is the in-table
reference.

Writes checkpoints to <run>/08_amputee_donor_prior/ and the dev report to
<run>/06_diagnostics/c_dev_screen_20260910/.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset_db2_emg import EMGCompletionDataset
from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from data.dataset_db3_emg import prepare_data_db3
from models.completion.mcia_core import derive_ch_mask_from_sample_mask
from utils.paper_pipeline import (
    build_mcia,
    build_mask_generator,
    build_structural_loss,
    flatten_pipeline_config,
    load_mcia_state_dict,
    load_yaml_config,
    save_json,
    set_seed,
    train_mcia_epoch,
)

DONORS = [2, 4, 8, 11]
DEV_SUBJECTS = [5, 6]
CANDIDATES = {
    "full_lr1e-5": {"mode": "full", "lr": 1e-5},
    "full_lr3e-6": {"mode": "full", "lr": 3e-6},
    "adapter_lr1e-4": {"mode": "adapter", "lr": 1e-4},
}
EPOCHS = 8
PATIENCE = 4
TCN_EPOCHS = 40
TCN_PATIENCE = 10
ADAPT_SEED = 4242
DOMAIN = 1


def load_exp3():
    path = PROJECT_ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("c_dev_exp3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


def configure_trainable(model, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad_(True)
    if mode == "adapter":
        for p in model.parameters():
            p.requires_grad_(False)
        for block in model.blocks[-2:]:
            if block.adapter is None:
                block.adapter = Adapter(model.embed_dim).to(next(model.parameters()).device)
            for p in block.adapter.parameters():
                p.requires_grad_(True)
        model.domain_embed.weight.requires_grad_(True)

        def _freeze_row0(grad):
            g = grad.clone()
            g[0].zero_()
            return g
        model.domain_embed.weight.register_hook(_freeze_row0)
        for p in model.pred_head.parameters():
            p.requires_grad_(True)


@torch.no_grad()
def donor_validation(model, loader, config, device) -> float:
    """S3 掩码下的域条件 masked 重建损失（组合主模块构件，仅用于开发筛选）。"""
    model.eval()
    gen = build_mask_generator(config)
    criterion = build_structural_loss(config, device)
    total, count = 0.0, 0
    for batch in loader:
        x = batch["data"].to(device)
        mask = gen.generate_batch_masks(len(x), n_channels=x.shape[2], time_steps=x.shape[1],
                                        device=device, scenario="s3").transpose(1, 2)
        mask = (mask > 0.5).float()
        xm = x * mask
        pred = model(xm, mask=derive_ch_mask_from_sample_mask(mask), x_masked=xm,
                     raw_time_mask=mask,
                     domain_id=torch.full((len(x),), DOMAIN, dtype=torch.long, device=device),
                     return_aux=True)
        loss, _ = criterion(pred, x, mask)
        total += float(loss.item()) * len(x)
        count += len(x)
    model.train()
    return total / max(count, 1)


def adapt_one(name: str, cfg: dict, donor_train, donor_val, base_ckpt, config,
              device, out_dir) -> tuple[nn.Module, dict]:
    set_seed(ADAPT_SEED)
    model = build_mcia(config, device)
    load_mcia_state_dict(model, base_ckpt, device)
    configure_trainable(model, cfg["mode"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=1e-5)
    criterion = build_structural_loss(config, device)
    train_loader = DataLoader(EMGCompletionDataset(donor_train), batch_size=config["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(EMGCompletionDataset(donor_val), batch_size=config["batch_size"],
                            shuffle=False, num_workers=0)
    best, best_state, no_improve = float("inf"), None, 0
    history = []
    for epoch in range(EPOCHS):
        train_loss = train_mcia_epoch(
            model, train_loader, optimizer, device,
            build_mask_generator(config), criterion,
            scenario=None, domain_id=DOMAIN, epoch=epoch,
        )
        val = donor_validation(model, val_loader, config, device)
        value = float(val)
        history.append({"epoch": epoch + 1, "train_loss": float(train_loss), "selection": value})
        if value < best:
            best, no_improve = value, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = out_dir / "checkpoints" / f"C_{name}.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "candidate": name, "mode": cfg["mode"],
                "lr": cfg["lr"], "best_selection": best,
                "adapted_on": f"donors {DONORS} E1+E2 train reps, domain_id={DOMAIN}"}, ckpt)
    print(f"  [adapt {name}] best_selection={best:.6f} ckpt={ckpt.name}", flush=True)
    return model, {"best_selection": best, "history": history, "checkpoint": str(ckpt)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    device = config["device"]
    run_dir = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    out_dir = run_dir / "08_amputee_donor_prior"
    diag = run_dir / "06_diagnostics" / "c_dev_screen_20260910"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag.mkdir(parents=True, exist_ok=True)

    exp3 = load_exp3()
    base_ckpt = Path(config["exp1_dir"]) / "checkpoints" / "best_model.pth"
    if not base_ckpt.is_file():
        raise FileNotFoundError(base_ckpt)

    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])

    print("[C-dev] loading donors...", flush=True)
    train_parts, val_parts = [], []
    for sid in DONORS:
        segments, _, meta = prepare_data_db3(loader, [sid], config,
                                             exercises=[1, 2], return_metadata=True)
        reps = np.asarray(meta["repetition"])
        train_parts.append(segments[np.isin(reps, [1, 3, 4])])
        val_parts.append(segments[reps == 6])
    donor_train = np.concatenate(train_parts)
    donor_val = np.concatenate(val_parts)
    print(f"donors: train={len(donor_train)} val={len(donor_val)} windows", flush=True)

    dev_cache = {}
    for sid in DEV_SUBJECTS:
        raw, angle, _, reps, wmeta = prepare_kinematics_data(
            loader, [sid], config, exercises=config["regressor_exercises"],
            db="db3", return_metadata=True)
        tr, va, _ = make_rep_split(reps)
        dev_cache[sid] = (raw, angle, tr, va, np.asarray(wmeta["quality_mask"], np.float32))

    tcn_cfg = dict(config)
    tcn_cfg["regressor_num_epochs"] = TCN_EPOCHS
    tcn_cfg["regressor_patience"] = TCN_PATIENCE

    def downstream(label, complete_fn) -> dict:
        per = {}
        for sid in DEV_SUBJECTS:
            raw, angle, tr, va, masks = dev_cache[sid]
            pool = raw.copy()
            dev_idx = np.concatenate([tr, va])
            pool[dev_idx] = complete_fn(raw[dev_idx], masks[dev_idx])
            set_seed(ADAPT_SEED)
            model, best_val, _ = exp3.train_tcn_on_emg(pool, angle, tr, va, tcn_cfg, device)
            pred, tgt = exp3.predict_on_set(model, pool, angle, va, tcn_cfg, device)
            per[sid] = exp3.evaluate_subsets(pred, tgt)["global"]
            per[sid]["best_val_loss"] = float(best_val)
            print(f"  [{label} S{sid:02d}] val RMSE={per[sid]['rmse']:.5f}", flush=True)
        mean_rmse = float(np.mean([v["rmse"] for v in per.values()]))
        return {"per_subject": {str(k): v for k, v in per.items()}, "mean_val_rmse": mean_rmse}

    results = {}
    # B reference: frozen healthy prior, domain_id=None
    set_seed(ADAPT_SEED)
    b_model = build_mcia(config, device)
    load_mcia_state_dict(b_model, base_ckpt, device)
    b_model.eval()
    results["B_reference"] = downstream(
        "B", lambda raw, m: exp3.apply_mcia(b_model, raw, m, device, domain_id=None,
                                            patch_size=int(config["patch_size"])))

    for name, cand in CANDIDATES.items():
        model, adapt_info = adapt_one(name, cand, donor_train, donor_val, base_ckpt,
                                      config, device, out_dir)
        model.eval()
        results[f"C_{name}"] = downstream(
            f"C_{name}", lambda raw, m, mm=model: exp3.apply_mcia(
                mm, raw, m, device, domain_id=DOMAIN, patch_size=int(config["patch_size"])))
        results[f"C_{name}"]["adaptation"] = adapt_info

    c_means = {k: v["mean_val_rmse"] for k, v in results.items() if k.startswith("C_")}
    winner = min(c_means, key=c_means.get)
    payload = {
        "protocol": {
            "protocol_section": "EXPERIMENT_PROTOCOL.md §5 dev freeze",
            "donors": DONORS, "dev_subjects": DEV_SUBJECTS,
            "donor_usage": "E1+E2 reps 1/3/4 adapt; rep 6 checkpoint selection only",
            "dev_usage": "train reps fit TCN, rep 6 report; test reps untouched",
            "candidates": CANDIDATES, "epochs": EPOCHS, "patience": PATIENCE,
            "tcn_budget": {"epochs": TCN_EPOCHS, "patience": TCN_PATIENCE},
            "domain_id": DOMAIN,
            "selection_rule_predeclared": "lowest mean S05/S06 validation global RMSE",
            "base_checkpoint": str(base_ckpt),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "results": results,
        "candidate_mean_val_rmse": c_means,
        "winner": winner,
    }
    save_json(diag / "c_dev_screen.json", payload)
    print(json.dumps({"candidate_mean_val_rmse": c_means, "winner": winner}, indent=2))
    print(f"results={diag / 'c_dev_screen.json'}", flush=True)


if __name__ == "__main__":
    main()
