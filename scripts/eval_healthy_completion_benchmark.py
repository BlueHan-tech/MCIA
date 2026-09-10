"""Protocol §7.5: healthy DB2 controlled-mask completion benchmark.

One-shot frozen comparison of MCIA vs SGMD-AAE vs CP-WOPT on DB2 held-out
test subjects (S33-S40), synthetic ScenarioMix masks with known ground truth.
Answers "which method reconstructs known signal better"; the DB3 task-matched
script answers downstream utility and must NOT report completion metrics
(its real mask regions have no clean truth).

Locked rules (protocol §7.5):
  - benchmark subjects: DB2 test subjects from config (S33-S40), exercise 1,
    same preprocessing as Exp1 pretraining;
  - masks: ScenarioMix mixture, per-subject generator seeded
    BENCHMARK_SEED + subject_id (pre-declared, recorded in the report);
  - uniform delivery for ALL methods: clip [0,1] -> patch-boundary cross-fade
    -> copy-back observed (MCIA via complete_with_mask, others wrapped here);
  - SGMD-AAE pretrained on the same DB2 training pool and ScenarioMix as MCIA
    (reuses run_task_matched_literature_baselines._train_sgmd);
  - CP-WOPT fit per subject on the artificially masked observations
    (transductive, labelled as such);
  - success criterion: subject-equalized masked NRMSE mean lower than BOTH
    baselines AND >= 4/8 per-subject wins against each.

Writes to <run>/07_healthy_completion_benchmark/. Dev iteration on S29-S32
only; every run on S33-S40 is recorded (mask seed, checkpoints, verdicts).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset_db2_emg import load_and_cache_data, prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from models.baselines.cp_wopt import CPWOPTConfig, complete_cp_wopt
from models.baselines.sgmd_aae import SGMDAAEGenerator
from utils.paper_pipeline import (
    build_mcia,
    complete_with_mask,
    flatten_pipeline_config,
    load_mcia_state_dict,
    load_yaml_config,
    patch_boundary_crossfade,
    safe_pearson_np,
    set_seed,
)
from utils.run_layout import get_run_dir, mark_step


def _load_task_matched_module():
    path = PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py"
    spec = importlib.util.spec_from_file_location("task_matched_shared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED = _load_task_matched_module()

BENCHMARK_SEED = 20260910          # pre-declared per-subject mask seed base


def deliver_uniform(prediction: np.ndarray, observed_values: np.ndarray,
                    observed_mask: np.ndarray, patch_size: int) -> np.ndarray:
    """Uniform §7.5 delivery: clip -> patch-boundary cross-fade -> copy-back."""
    p = torch.from_numpy(np.clip(prediction, 0.0, 1.0)).float()
    p = patch_boundary_crossfade(p, patch_size)
    m = torch.from_numpy(observed_mask).float()
    v = torch.from_numpy(observed_values).float()
    return (p * (1.0 - m) + v * m).numpy()


def metrics_with_pearson(delivered: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> dict:
    out = dict(SHARED._masked_metrics(delivered, clean, mask))
    missing = mask < 0.5
    out["masked_pearson"] = safe_pearson_np(delivered[missing], clean[missing])
    return out


@torch.no_grad()
def predict_sgmd_raw(generator: SGMDAAEGenerator, clean: np.ndarray, mask: np.ndarray,
                     device: str, batch_size: int) -> np.ndarray:
    """Raw SGMD generator predictions (input-masked), WITHOUT any delivery."""
    generator.eval()
    out = np.empty_like(clean)
    for start in range(0, len(clean), batch_size):
        x = torch.as_tensor(clean[start:start + batch_size] * mask[start:start + batch_size],
                            dtype=torch.float32, device=device).unsqueeze(1)
        m = torch.as_tensor(mask[start:start + batch_size], dtype=torch.float32,
                            device=device).unsqueeze(1)
        out[start:start + len(x)] = generator(x, m).squeeze(1).cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol §7.5 healthy completion benchmark.")
    parser.add_argument("--sgmd-epochs", type=int, default=None)
    parser.add_argument("--sgmd-max-train-windows", type=int, default=None)
    parser.add_argument("--skip-sgmd-training", action="store_true")
    parser.add_argument("--mcia-checkpoint", type=Path, default=None)
    parser.add_argument("--subjects", default="",
                        help="comma-separated benchmark subjects; default uses DB2 test subjects S33-S40")
    parser.add_argument("--limit-windows-per-subject", type=int, default=0,
                        help="0 keeps all windows; any cap is an explicitly labelled development run")
    args = parser.parse_args()

    cfg = load_yaml_config(PROJECT_ROOT)
    settings = cfg["literature_baselines"]
    run_dir = get_run_dir(PROJECT_ROOT, cfg, create=True)
    os.environ["MCIA_RUN_DIR"] = str(run_dir)
    config = flatten_pipeline_config(cfg)
    device = config["device"]
    out_dir = run_dir / "07_healthy_completion_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    mark_step(run_dir, "healthy_completion_benchmark", "running")

    seed = int(settings["seed"])
    subjects = ([int(v) for v in args.subjects.split(",") if v.strip()]
                if args.subjects else list(config["test_subjects"]))
    patch_size = int(config["patch_size"])
    dev_run = bool(args.limit_windows_per_subject)

    try:
        data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])

        # --- frozen MCIA healthy prior ---
        mcia_ckpt = (args.mcia_checkpoint if args.mcia_checkpoint is not None
                     else Path(config["exp1_dir"]) / "checkpoints" / "best_model.pth")
        if not mcia_ckpt.is_file():
            raise FileNotFoundError(f"MCIA healthy-prior checkpoint required: {mcia_ckpt}")
        mcia = build_mcia(config, device)
        load_mcia_state_dict(mcia, mcia_ckpt, device)
        mcia.eval()

        # --- SGMD-AAE pretrained on the same DB2 pool / ScenarioMix as MCIA ---
        sgmd_ckpt = out_dir / "checkpoints" / "sgmd_aae_db2_200hz.pth"
        epoch_count = int(args.sgmd_epochs if args.sgmd_epochs is not None
                          else settings["sgmd_aae_epochs"])
        max_windows = int(args.sgmd_max_train_windows if args.sgmd_max_train_windows is not None
                          else settings.get("sgmd_pretrain_max_windows", 0))
        if args.skip_sgmd_training:
            if not sgmd_ckpt.exists():
                raise FileNotFoundError(f"--skip-sgmd-training requires {sgmd_ckpt}")
            sgmd = SGMDAAEGenerator().to(device)
            sgmd.load_state_dict(torch.load(sgmd_ckpt, map_location=device)["model"])
            sgmd.eval()
        else:
            db2_cache = out_dir / "cache" / "db2_train_cache.pt"
            db2_train = load_and_cache_data(data_loader, config["train_subjects"], config, db2_cache)["data"]
            if max_windows > 0:
                db2_train = db2_train[:max_windows]
            print(f"Pretraining SGMD-AAE on {len(db2_train)} DB2 windows...", flush=True)
            sgmd = SHARED._train_sgmd(db2_train, config, epoch_count,
                                      int(settings["sgmd_aae_batch_size"]), seed, sgmd_ckpt)

        # --- benchmark ---
        per_subject = {}
        for subject_id in subjects:
            set_seed(seed)
            segments, _, _ = prepare_data_db2(data_loader, [subject_id], config, exercises=[1])
            clean = segments.astype(np.float32)
            if dev_run and args.limit_windows_per_subject > 0:
                clean = clean[:args.limit_windows_per_subject]
            mask = SHARED._scenario_generator(
                config, BENCHMARK_SEED + int(subject_id)).generate_mask(
                torch.as_tensor(clean)).numpy().astype(np.float32)
            observed_values = clean * mask
            x_t = torch.as_tensor(clean, dtype=torch.float32, device=device)
            m_t = torch.as_tensor(mask, dtype=torch.float32, device=device)

            delivered = {}
            delivered["MCIA"] = complete_with_mask(mcia, x_t, m_t,
                                                   patch_size=patch_size).cpu().numpy()
            sgmd_raw = predict_sgmd_raw(sgmd, clean, mask, device,
                                        int(settings["sgmd_aae_batch_size"]))
            delivered["SGMD_AAE"] = deliver_uniform(sgmd_raw, observed_values, mask, patch_size)
            cp_result = complete_cp_wopt(observed_values, mask,
                                         CPWOPTConfig(rank=int(settings["cp_wopt_rank"]),
                                                      seed=seed + int(subject_id)))
            delivered["CP_WOPT"] = deliver_uniform(cp_result.reconstruction,
                                                   observed_values, mask, patch_size)

            per_subject[int(subject_id)] = {
                "n_windows": int(len(clean)),
                "masked_fraction": float((mask < 0.5).mean()),
                "methods": {name: metrics_with_pearson(d, clean, mask)
                            for name, d in delivered.items()},
            }
            row = " ".join(f"{n}: NRMSE={per_subject[int(subject_id)]['methods'][n]['nrmse_peak_1']:.4f}"
                           for n in delivered)
            print(f"S{subject_id:02d}: {len(clean)} windows | {row}", flush=True)

        # --- §7.5 verdicts ---
        methods = ["MCIA", "SGMD_AAE", "CP_WOPT"]
        mean_nrmse = {m: float(np.mean([per_subject[s]["methods"][m]["nrmse_peak_1"]
                                        for s in per_subject])) for m in methods}
        wins = {m: int(sum(per_subject[s]["methods"]["MCIA"]["nrmse_peak_1"]
                           < per_subject[s]["methods"][m]["nrmse_peak_1"] for s in per_subject))
                for m in ("SGMD_AAE", "CP_WOPT")}
        n_subjects = len(per_subject)
        verdict = {
            "mean_nrmse_lower_than_sgmd": mean_nrmse["MCIA"] < mean_nrmse["SGMD_AAE"],
            "mean_nrmse_lower_than_cp": mean_nrmse["MCIA"] < mean_nrmse["CP_WOPT"],
            "subject_wins_vs_sgmd": wins["SGMD_AAE"],
            "subject_wins_vs_cp": wins["CP_WOPT"],
            "required_wins": int(np.ceil(n_subjects / 2)),
            "pass_7_5": (mean_nrmse["MCIA"] < mean_nrmse["SGMD_AAE"]
                         and mean_nrmse["MCIA"] < mean_nrmse["CP_WOPT"]
                         and wins["SGMD_AAE"] >= int(np.ceil(n_subjects / 2))
                         and wins["CP_WOPT"] >= int(np.ceil(n_subjects / 2))),
        }

        rows = []
        for s in sorted(per_subject):
            for m in methods:
                met = per_subject[s]["methods"][m]
                rows.append({"subject_id": s, "method": m,
                             "nrmse_peak_1": met["nrmse_peak_1"],
                             "psnr_peak_1_db": met["psnr_peak_1_db"],
                             "rme": met["rme"], "masked_pearson": met["masked_pearson"]})
        with (out_dir / "healthy_benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

        report = {
            "protocol": {
                "protocol_section": "EXPERIMENT_PROTOCOL.md §7.5",
                "subjects": sorted(per_subject), "exercise": 1,
                "mask": f"ScenarioMix mixture, per-subject seed {BENCHMARK_SEED}+subject_id",
                "delivery": "uniform clip->patch-boundary crossfade->copy-back for all methods",
                "mcia_checkpoint": str(mcia_ckpt), "sgmd_checkpoint": str(sgmd_ckpt),
                "sgmd_pretraining": {"epochs": epoch_count, "max_windows": max_windows,
                                      "source": "DB2 healthy train subjects, ScenarioMix"},
                "cp_wopt": "fit per subject on artificially masked observations; transductive",
                "dev_run": dev_run,
            },
            "mean_nrmse": mean_nrmse, "subject_wins": wins,
            "per_subject": per_subject, "verdict": verdict,
        }
        (out_dir / "healthy_benchmark_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.bar(methods, [mean_nrmse[m] for m in methods], capsize=4)
        axis.set_title("Masked NRMSE (peak 1, lower is better)")
        axis.tick_params(axis="x", rotation=25); axis.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(out_dir / "healthy_benchmark_summary.png", dpi=160); plt.close(fig)

        mark_step(run_dir, "healthy_completion_benchmark", "completed", {"output": str(out_dir)})
        print(json.dumps({"mean_nrmse": mean_nrmse, "wins": wins, "verdict": verdict}, indent=2))
        print(f"results={out_dir / 'healthy_benchmark_report.json'}", flush=True)
    except Exception:
        mark_step(run_dir, "healthy_completion_benchmark", "failed", {"output": str(out_dir)})
        raise


if __name__ == "__main__":
    main()
