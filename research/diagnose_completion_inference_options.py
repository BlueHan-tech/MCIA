"""Time-bounded paired inference-option screen on the frozen Exp1 checkpoint.

Conditions (all paired: same checkpoint, same fixed S1/S2/S3 masks, DB2
validation subject 29 only; no test subjects, no training, no main-flow writes):
  baseline           complete_with_mask defaults (clip + copy-back)
  boundary_smooth    + patch-boundary three-point cross-fade
  refine2            + one extra self-refinement forward pass
  unc_g0.10/0.15/0.20 MC-Dropout uncertainty-gated completion (8 samples)

Outputs masked RMSE/MAE/corr per scenario plus the 0.2/0.4/0.4 weighted
summary used by earlier activation screens. Writes only to
<run>/06_diagnostics/.
"""

from __future__ import annotations

import hashlib
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
import yaml

from models.completion.mask_generators import ScenarioMixMaskGenerator
from utils.paper_pipeline import (
    build_mcia,
    complete_with_mask,
    complete_with_mask_uncertainty,
    load_mcia_state_dict,
    set_seed,
)

SUBJECT = 29
N_WINDOWS = 512
SCENARIOS = ["s1", "s2", "s3"]
WEIGHTS = [0.2, 0.4, 0.4]


def corr_over_missing(pred: np.ndarray, target: np.ndarray, missing: np.ndarray) -> float:
    correlations = []
    for j in range(len(pred)):
        for c in range(pred.shape[2]):
            v = missing[j, :, c]
            if v.sum() >= 4 and np.std(pred[j, v, c]) > 1e-8 and np.std(target[j, v, c]) > 1e-8:
                correlations.append(float(np.corrcoef(pred[j, v, c], target[j, v, c])[0, 1]))
    return float(np.mean(correlations)) if correlations else float("nan")


def scenario_metrics(completed: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    missing = mask < 0.5
    delta = (completed - target)[missing]
    return {
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "mae": float(np.mean(np.abs(delta))),
        "corr": corr_over_missing(completed, target, missing),
        "missing_ratio": float(missing.mean()),
        "n_masked_values": int(missing.sum()),
    }


def main() -> None:
    started = time.monotonic()
    run = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    ckpt = run / "01_db2_completion" / "checkpoints" / "best_model.pth"
    val_cache = run / "01_db2_completion" / "cache" / "val_cache.pt"
    for path in (ckpt, val_cache):
        if not path.exists():
            raise FileNotFoundError(path)

    out = run / "06_diagnostics" / "completion_inference_options_20260909"
    out.mkdir(parents=True, exist_ok=False)

    root = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = dict(root["signal"], **root["exp1_mcia"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    set_seed(42)

    cached = torch.load(val_cache, map_location="cpu", weights_only=False)
    data = np.asarray(cached["data"], dtype=np.float32)
    subject_ids = np.asarray(cached["subject_ids"])
    rows = np.flatnonzero(subject_ids == SUBJECT)
    if len(rows) < N_WINDOWS:
        raise RuntimeError(f"S{SUBJECT} has only {len(rows)} cached windows")
    selected = np.sort(np.random.default_rng(42).choice(rows, N_WINDOWS, replace=False))
    target_np = data[selected]
    target = torch.tensor(target_np, dtype=torch.float32, device=device)
    print(f"S{SUBJECT}: {N_WINDOWS} windows | device={device}")

    model = build_mcia(cfg, device)
    load_mcia_state_dict(model, ckpt, device)
    model.eval()
    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    def mask_gen(seed: int) -> ScenarioMixMaskGenerator:
        return ScenarioMixMaskGenerator(
            n_channels=cfg["n_channels"], time_steps=cfg["window_size"],
            patch_size=cfg["patch_size"], group_indices=cfg.get("group_indices"),
            min_alive_per_group=cfg.get("min_alive_per_group"),
            scenario_weights=cfg.get("scenario_weights"),
            scenario_params=cfg.get("scenario_params"),
            rng=np.random.default_rng(seed),
        )

    fixed_masks = {
        s: mask_gen(7000 + i).generate_mask(target, scenario=s)
        for i, s in enumerate(SCENARIOS)
    }

    conditions: dict[str, dict] = {
        "baseline": dict(kind="plain"),
        "boundary_smooth": dict(kind="plain", patch_boundary_smooth=True),
        "refine2": dict(kind="plain", refine_steps=2),
        "unc_g0.10": dict(kind="unc", std_gate=0.10),
        "unc_g0.15": dict(kind="unc", std_gate=0.15),
        "unc_g0.20": dict(kind="unc", std_gate=0.20),
    }

    report = {
        "protocol": {
            "purpose": "paired inference-option screen on frozen Exp1 checkpoint",
            "database": "DB2", "subject": SUBJECT, "windows": N_WINDOWS,
            "scenarios": SCENARIOS, "mask_seeds": {"s1": 7000, "s2": 7001, "s3": 7002},
            "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha,
            "mc_dropout_samples": 8,
            "data_source": "run val_cache.pt (per-subject Q5/Q99 preprocessing)",
            "test_evaluated": False,
            "training_performed": False,
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
    for scenario, mask in fixed_masks.items():
        mask_np = mask.cpu().numpy()
        for name, opts in conditions.items():
            set_seed(42)
            if opts["kind"] == "plain":
                completed = complete_with_mask(
                    model, target, mask,
                    patch_boundary_smooth=opts.get("patch_boundary_smooth", False),
                    patch_size=cfg["patch_size"],
                    refine_steps=opts.get("refine_steps", 1),
                ).cpu().numpy()
            else:
                completed = complete_with_mask_uncertainty(
                    model, target, mask, n_samples=8, std_gate=opts["std_gate"],
                    patch_size=cfg["patch_size"],
                ).cpu().numpy()
            np.testing.assert_array_equal(
                completed[mask_np >= 0.5], target_np[mask_np >= 0.5]
            )
            report["results"].setdefault(name, {})[scenario] = scenario_metrics(
                completed, target_np, mask_np
            )
            save()
        line = " ".join(
            f"{name}: RMSE={report['results'][name][scenario]['rmse']:.5f}"
            for name in conditions
        )
        print(f"[{scenario}] {line}", flush=True)

    for name in conditions:
        per = report["results"][name]
        report["results"][name]["weighted"] = {
            key: float(sum(w * per[s][key] for w, s in zip(WEIGHTS, SCENARIOS)))
            for key in ("rmse", "mae", "corr")
        }
    report["status"] = "complete"
    save()

    base = report["results"]["baseline"]["weighted"]
    print("\nweighted summary (vs baseline):")
    for name in conditions:
        w = report["results"][name]["weighted"]
        print(f"  {name:16s} RMSE={w['rmse']:.5f} MAE={w['mae']:.5f} corr={w['corr']:.4f}"
              f" | dRMSE={(w['rmse']-base['rmse'])/base['rmse']:+.2%}"
              f" dMAE={(w['mae']-base['mae'])/base['mae']:+.2%}"
              f" dCorr={w['corr']-base['corr']:+.4f}")
    print(f"results={out / 'results.json'}")


if __name__ == "__main__":
    main()
