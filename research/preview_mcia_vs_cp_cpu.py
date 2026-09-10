"""CPU-only full-scale preview: MCIA vs CP-WOPT on one validation subject.

Answers "does the smoke-scale CP advantage survive at full window count"
without touching the GPU (which is busy with the S29-S32 rehearsal).
Uses validation subject S29 with the frozen benchmark mask seed. Same uniform
delivery as §7.5. Dev iteration evidence, disclosed.
"""

from __future__ import annotations

import importlib.util
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
import yaml

from data.dataset_db2_emg import prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from models.baselines.cp_wopt import CPWOPTConfig, complete_cp_wopt
from utils.paper_pipeline import (
    build_mcia, complete_with_mask, flatten_pipeline_config,
    load_mcia_state_dict, load_yaml_config, patch_boundary_crossfade, set_seed,
)


def main() -> None:
    subject = 29
    spec = importlib.util.spec_from_file_location(
        "tm", PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    device = "cpu"
    torch.set_num_threads(8)
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    segments, _, _ = prepare_data_db2(loader, [subject], config, exercises=[1])
    clean = segments.astype(np.float32)
    mask = tm._scenario_generator(config, 20260910 + subject).generate_mask(
        torch.as_tensor(clean)).numpy().astype(np.float32)
    observed = clean * mask
    print(f"S{subject}: {len(clean)} windows, masked {float((mask < 0.5).mean()):.1%}", flush=True)

    # MCIA (CPU)
    set_seed(42)
    model = build_mcia(config, device)
    ckpt = Path(config["exp1_dir"]) / "checkpoints" / "best_model.pth"
    load_mcia_state_dict(model, ckpt, device)
    model.eval()
    t0 = time.time()
    outs = []
    with torch.no_grad():
        for s in range(0, len(clean), 64):
            x = torch.as_tensor(clean[s:s + 64], dtype=torch.float32)
            m = torch.as_tensor(mask[s:s + 64], dtype=torch.float32)
            outs.append(complete_with_mask(model, x, m, patch_size=int(config["patch_size"])))
    mcia = torch.cat(outs).numpy()
    print(f"MCIA done in {time.time() - t0:.0f}s", flush=True)

    # CP-WOPT (CPU, per-subject transductive)
    t0 = time.time()
    result = complete_cp_wopt(observed, mask, CPWOPTConfig(rank=8, seed=42 + subject))
    p = torch.from_numpy(np.clip(result.reconstruction, 0.0, 1.0)).float()
    p = patch_boundary_crossfade(p, int(config["patch_size"]))
    cp = (p * (1 - torch.from_numpy(mask)) + torch.from_numpy(observed)).numpy()
    print(f"CP-WOPT done in {time.time() - t0:.0f}s", flush=True)

    for name, delivered in (("MCIA", mcia), ("CP_WOPT", cp)):
        met = tm._masked_metrics(delivered, clean, mask)
        print(f"{name}: NRMSE={met['nrmse_peak_1']:.4f} PSNR={met['psnr_peak_1_db']:.2f}dB "
              f"RME={met['rme']:.4f}", flush=True)


if __name__ == "__main__":
    main()
