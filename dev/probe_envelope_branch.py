"""包络旁路探针：在 DB2 训练池上短程续训带 EnvelopeBranch 的 MCIA，
然后在 S29（验证被试、冻结基准掩码）上与 CP 的满规模数字对照。

目标：检验包络旁路能否收窄/反超 CP 的 NRMSE 优势（S29 参照：
MCIA 0.1852 / CP 0.1500）。纯开发探针，不改主链路配置。
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

from data.dataset_db2_emg import EMGCompletionDataset, prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from models.completion.mcia_core import derive_ch_mask_from_sample_mask
from utils.paper_pipeline import (build_mcia, build_mask_generator, build_structural_loss,
                                  complete_with_mask, flatten_pipeline_config,
                                  load_mcia_state_dict, load_yaml_config, set_seed)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--subjects", default="29")
    args = parser.parse_args()

    tm = load("tm", PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py")
    cfg = load_yaml_config(PROJECT_ROOT)
    config = dict(flatten_pipeline_config(cfg))
    config["use_envelope_branch"] = True        # 探针内覆写，不写回主配置
    device = config["device"]
    torch.backends.cudnn.benchmark = True
    run_dir = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    out = run_dir / "06_diagnostics" / "envelope_probe_20260910"
    out.mkdir(parents=True, exist_ok=False)

    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    cache = run_dir / "07_healthy_completion_benchmark" / "cache" / "db2_train_cache.pt"
    if cache.is_file():
        db2_train = torch.load(cache, map_location="cpu", weights_only=False)["data"]
        print(f"loaded DB2 train cache: {len(db2_train)} windows", flush=True)
    else:
        from data.dataset_db2_emg import load_and_cache_data
        db2_train = load_and_cache_data(loader, config["train_subjects"], config,
                                        out / "db2_train_cache.pt")["data"]
    if args.max_train_windows > 0:
        db2_train = db2_train[:args.max_train_windows]

    set_seed(42)
    model = build_mcia(config, device)
    base_ckpt = Path(config["exp1_dir"]) / "checkpoints" / "best_model.pth"
    load_mcia_state_dict(model, base_ckpt, device)   # 旁路零初始化=恒等加载
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model ready (envelope branch on), params={trainable:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = build_structural_loss(config, device)
    train_loader = DataLoader(EMGCompletionDataset(db2_train), batch_size=config["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=True)
    mask_gen = build_mask_generator(config)
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in train_loader:
            x = batch["data"].to(device, non_blocking=True)
            mask = mask_gen.generate_batch_masks(len(x), n_channels=x.shape[2],
                                                 time_steps=x.shape[1], device=device).transpose(1, 2)
            mask = (mask > 0.5).float()
            xm = x * mask
            pred = model(xm, mask=derive_ch_mask_from_sample_mask(mask), x_masked=xm,
                         raw_time_mask=mask, return_aux=True)
            loss, _ = criterion(pred, x, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(x)
        print(f"epoch {epoch + 1}/{args.epochs}: loss={total / len(train_loader):.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    torch.save({"model": model.state_dict(), "probe": "envelope_branch finetune",
                "epochs": args.epochs, "lr": args.lr, "base": str(base_ckpt)},
               out / "mcia_envelope_probe.pth")
    model.eval()

    report = {"reference": {"MCIA_frozen": 0.1852, "CP_WOPT": 0.1500,
                             "note": "S29 full-scale preview 2026-09-10"},
              "epochs": args.epochs, "lr": args.lr, "train_windows": int(len(db2_train)),
              "subjects": {}}
    for sid in [int(v) for v in args.subjects.split(",")]:
        segments, _, _ = prepare_data_db2(loader, [sid], config, exercises=[1])
        clean = segments.astype(np.float32)
        mask = tm._scenario_generator(config, 20260910 + sid).generate_mask(
            torch.as_tensor(clean)).numpy().astype(np.float32)
        outs = []
        with torch.no_grad():
            for s in range(0, len(clean), 64):
                outs.append(complete_with_mask(
                    model, torch.as_tensor(clean[s:s + 64]),
                    torch.as_tensor(mask[s:s + 64]), patch_size=int(config["patch_size"])))
        delivered = torch.cat(outs).numpy()
        met = tm._masked_metrics(delivered, clean, mask)
        report["subjects"][str(sid)] = met
        print(f"S{sid}: envelope-MCIA NRMSE={met['nrmse_peak_1']:.4f} "
              f"(frozen 0.1852 / CP 0.1500)", flush=True)
    (out / "envelope_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"results={out / 'envelope_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
