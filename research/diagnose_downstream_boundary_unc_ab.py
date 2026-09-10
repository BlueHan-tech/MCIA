"""Small-scale downstream angle A/B for inference-level completion options.

Dev screen only, leak-free by construction:
  - DB3 S05, E1+E2, fixed repetition split (train 1/3/4, validation 6);
  - test repetitions (2/5) are split off but NEVER used for training,
    checkpoint selection, or reporting;
  - four paired conditions share the frozen healthy-prior MCIA checkpoint,
    identical quality masks, TCN architecture, training rules, and seed;
    only the input representation differs:
      raw              A-style raw sEMG
      enhanced_default B-style completion (clip + copy-back)
      enhanced_boundary same + patch-boundary cross-fade
      enhanced_unc015  MC-Dropout uncertainty-gated completion (8 samples)
  - TCN budget compressed to 40 epochs / patience 10 for the dev screen and
    recorded; all conditions share the identical budget.

Reports window-level Key10 subset metrics on VALIDATION windows only.
Writes only to <run>/06_diagnostics/.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import (
    build_mcia,
    complete_with_mask,
    complete_with_mask_uncertainty,
    flatten_pipeline_config,
    load_mcia_state_dict,
    load_yaml_config,
    set_seed,
)

SUBJECT = 5
DEV_EPOCHS = 40
DEV_PATIENCE = 10
CONDITIONS = ["raw", "enhanced_default", "enhanced_boundary", "enhanced_unc015"]


def load_exp3():
    path = ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("downstream_ab_exp3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    started = time.monotonic()
    run = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    out = run / "06_diagnostics" / "downstream_boundary_unc_ab_20260909"
    out.mkdir(parents=True, exist_ok=False)

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    config = dict(config)
    config["regressor_num_epochs"] = DEV_EPOCHS
    config["regressor_patience"] = DEV_PATIENCE
    device = config["device"]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    exp3 = load_exp3()

    ckpt = Path(config["exp1_dir"]) / "checkpoints" / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    mcia = build_mcia(config, device)
    load_mcia_state_dict(mcia, ckpt, device)
    mcia.eval()
    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    set_seed(42)
    raw_emg, angle, _, reps, window_meta = prepare_kinematics_data(
        loader, [SUBJECT], config,
        exercises=config["regressor_exercises"], db="db3", return_metadata=True,
    )
    train_idx, val_idx, test_idx = make_rep_split(reps)
    quality_masks = np.asarray(window_meta["quality_mask"], dtype=np.float32)
    print(f"S{SUBJECT:02d}: windows train={len(train_idx)} val={len(val_idx)} "
          f"(test={len(test_idx)} split off, unused) | "
          f"quality-mask missing={float((quality_masks < 0.5).mean()):.2%}")

    patch_size = int(config["patch_size"])
    dev_idx = np.concatenate([train_idx, val_idx])
    dev_masks = torch.tensor(quality_masks[dev_idx], dtype=torch.float32, device=device)
    dev_target = torch.tensor(raw_emg[dev_idx], dtype=torch.float32, device=device)

    def complete(kind: str) -> np.ndarray:
        set_seed(42)
        if kind == "enhanced_default":
            completed = complete_with_mask(mcia, dev_target, dev_masks)
        elif kind == "enhanced_boundary":
            completed = complete_with_mask(
                mcia, dev_target, dev_masks,
                patch_boundary_smooth=True, patch_size=patch_size,
            )
        elif kind == "enhanced_unc015":
            completed = complete_with_mask_uncertainty(
                mcia, dev_target, dev_masks, n_samples=8, std_gate=0.15,
                patch_size=patch_size,
            )
        else:
            raise ValueError(kind)
        return completed.cpu().numpy()

    pools: dict[str, np.ndarray] = {"raw": raw_emg.copy()}
    dev_masks_np = quality_masks[dev_idx]
    for kind in ("enhanced_default", "enhanced_boundary", "enhanced_unc015"):
        completed = complete(kind)
        observed = dev_masks_np >= 0.5
        np.testing.assert_array_equal(
            completed[observed], raw_emg[dev_idx][observed],
        )
        pool = raw_emg.copy()
        pool[dev_idx] = completed
        pools[kind] = pool
        print(f"  completion [{kind}] done ({time.monotonic() - started:.0f}s)")

    report = {
        "protocol": {
            "purpose": "dev downstream A/B for inference-level completion options",
            "database": "DB3", "subject": SUBJECT,
            "exercises": [int(v) for v in config["regressor_exercises"]],
            "split": {"train": [1, 3, 4], "validation": [6], "test": "split off, never used"},
            "window_counts": {"train": int(len(train_idx)), "validation": int(len(val_idx))},
            "quality_mask_rule": "hard_zero_train_1_3_4_or_gronlund_2005_mqp_p_gt_0_20",
            "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha,
            "tcn_budget_dev_override": {"num_epochs": DEV_EPOCHS, "patience": DEV_PATIENCE},
            "conditions": CONDITIONS,
            "selection_and_reporting": "validation repetition 6 only",
            "test_evaluated": False,
            "paired_seed": 42,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "results": {},
        "status": "running",
    }

    def save() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
        )

    save()
    for condition in CONDITIONS:
        set_seed(42)
        model, best_val, epochs = exp3.train_tcn_on_emg(
            pools[condition], angle, train_idx, val_idx, config, device,
        )
        pred, tgt = exp3.predict_on_set(model, pools[condition], angle, val_idx, config, device)
        subsets = exp3.evaluate_subsets(pred, tgt)
        report["results"][condition] = {
            "best_val_loss": float(best_val),
            "epochs": int(epochs),
            "validation_subsets": subsets,
        }
        save()
        g = subsets["global"]
        print(f"  [{condition}] best_val={best_val:.5f} epochs={epochs} "
              f"val RMSE={g['rmse']:.5f} MAE={g['mae']:.5f} "
              f"R2={g.get('r2', float('nan')):.4f}", flush=True)

    report["status"] = "complete"
    save()
    base = report["results"]["raw"]["validation_subsets"]["global"]
    print("\nvalidation global metrics (vs raw):")
    for condition in CONDITIONS:
        g = report["results"][condition]["validation_subsets"]["global"]
        print(f"  {condition:18s} RMSE={g['rmse']:.5f} MAE={g['mae']:.5f}"
              f" | dRMSE={(g['rmse'] - base['rmse']) / base['rmse']:+.2%}"
              f" dMAE={(g['mae'] - base['mae']) / base['mae']:+.2%}")
    print(f"results={out / 'results.json'}")


if __name__ == "__main__":
    main()
