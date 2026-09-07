"""DB2 MC-dropout calibration for frozen MCIA reconstruction uncertainty."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np
import torch

from data.dataset_db2_emg import prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from models.completion.mask_generators import ScenarioMixMaskGenerator
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config


def load_exp3():
    path = ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("mc_dropout_calibration_exp3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ranks(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return result


def spearman(left, right):
    left_rank, right_rank = ranks(left), ranks(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def parse_subjects(text, default):
    if text:
        return [int(value) for value in text.split(",")]
    return [int(value) for value in default]


def build_mask_generator(config, seed):
    return ScenarioMixMaskGenerator(
        n_channels=int(config["n_channels"]),
        time_steps=int(config["window_size"]),
        patch_size=int(config["patch_size"]),
        group_indices=config.get("group_indices"),
        min_alive_per_group=config.get("min_alive_per_group"),
        scenario_weights=config.get("scenario_weights"),
        scenario_params=config.get("scenario_params"),
        rng=np.random.default_rng(seed),
    )


def collect_batch_records(predictions, clean, mask, subject_ids, patch_size):
    mean_pred = predictions.mean(dim=0).cpu().numpy()
    std_pred = predictions.std(dim=0, unbiased=False).cpu().numpy()
    clean_np = clean.cpu().numpy()
    mask_np = mask.cpu().numpy()
    batch_size, n_steps, n_channels = clean_np.shape
    n_patches = n_steps // patch_size
    missing_patch = (
        mask_np.transpose(0, 2, 1).reshape(
            batch_size, n_channels, n_patches, patch_size
        ).max(axis=-1)
        < 0.5
    )
    uncertainty = (
        std_pred.reshape(batch_size, n_patches, patch_size, n_channels)
        .mean(axis=2)
        .transpose(0, 2, 1)
    )
    error = np.sqrt(
        np.square(
            (mean_pred - clean_np).reshape(
                batch_size, n_patches, patch_size, n_channels
            )
        ).sum(axis=2).transpose(0, 2, 1)
    )
    batch_id, channel_id, patch_id = np.nonzero(missing_patch)
    return np.column_stack(
        [
            np.asarray(subject_ids)[batch_id],
            channel_id,
            patch_id,
            uncertainty[batch_id, channel_id, patch_id],
            error[batch_id, channel_id, patch_id],
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="DB2 MC-dropout calibration of frozen MCIA reconstruction uncertainty"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--subjects",
        default="",
        help="comma-separated DB2 holdout subjects; default uses exp1_mcia.val_subjects",
    )
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mc_samples < 2:
        raise ValueError("mc-samples must be at least 2")
    run_dir = Path(args.run_dir).resolve()
    env_run_dir = os.environ.get("MCIA_RUN_DIR")
    if not env_run_dir or Path(env_run_dir).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    subjects = parse_subjects(args.subjects, config["val_subjects"])
    batch_size = args.batch_size or int(config["batch_size"])
    device = config["device"]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    exp3 = load_exp3()
    model, checkpoint = exp3.load_healthy_prior_mcia(config, device)
    if model is None:
        raise FileNotFoundError("healthy-prior MCIA checkpoint")
    loader = NinaProDataLoader(
        config["db2_path"], config["db3_path"], fs=int(config["orig_fs"])
    )
    windows, window_subjects, _ = prepare_data_db2(
        loader, subjects, config, exercises=[1]
    )
    if args.max_windows:
        windows = windows[:args.max_windows]
        window_subjects = window_subjects[:args.max_windows]
    if len(windows) == 0:
        raise ValueError("No DB2 validation windows")
    mask_gen = build_mask_generator(config, args.seed)

    print(
        f"[mc dropout calibration] DB2 subjects={subjects} windows={len(windows)} "
        f"mc_samples={args.mc_samples} batch={batch_size} "
        f"checkpoint={checkpoint}"
    )
    model.train()
    records = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            stop = min(start + batch_size, len(windows))
            clean = torch.as_tensor(
                windows[start:stop], dtype=torch.float32, device=device
            )
            mask = mask_gen.generate_batch_masks(
                len(clean),
                n_channels=clean.shape[-1],
                time_steps=clean.shape[1],
                device=device,
            ).transpose(1, 2)
            mask = (mask > 0.5).float()
            channel_valid = (mask.mean(dim=1) > 0.5).float()
            predictions = []
            for _ in range(args.mc_samples):
                predictions.append(
                    model(
                        clean * mask,
                        raw_time_mask=mask,
                        chan_valid_mask=channel_valid,
                    )
                )
            stacked = torch.stack(predictions, dim=0)
            records.extend(
                collect_batch_records(
                    stacked,
                    clean,
                    mask,
                    window_subjects[start:stop],
                    int(config["patch_size"]),
                )
            )
    model.eval()

    values = np.asarray(records, dtype=np.float64)
    if len(values) < 4:
        raise RuntimeError("Too few masked patches for calibration")
    uncertainty = values[:, 3]
    error = values[:, 4]
    per_subject = {}
    for subject_id in subjects:
        selected = values[:, 0] == subject_id
        if selected.sum() < 4:
            continue
        per_subject[f"S{subject_id:02d}"] = {
            "masked_patch_count": int(selected.sum()),
            "spearman_uncertainty_vs_l2_error": spearman(
                uncertainty[selected], error[selected]
            ),
            "mean_uncertainty": float(uncertainty[selected].mean()),
            "mean_l2_error": float(error[selected].mean()),
        }
    correlation = spearman(uncertainty, error)
    report = {
        "scope": "DB2 held-out-subject MC-dropout calibration only",
        "angle_policy": "no glove labels or KinematicTCN are loaded or used",
        "checkpoint": str(checkpoint),
        "data": {
            "subjects": subjects,
            "window_count": int(len(windows)),
            "training_mask_strategy": "ScenarioMixMaskGenerator sampled with configured weights",
            "patch_size_samples": int(config["patch_size"]),
        },
        "uncertainty_definition": (
            "mean predictive standard deviation over the samples in a fully masked "
            "channel-time patch across MC-dropout forward passes"
        ),
        "error_definition": (
            "L2 norm of MC predictive mean minus original DB2 EMG within the same "
            "fully masked channel-time patch"
        ),
        "mc_dropout": {
            "samples": args.mc_samples,
            "model_mode": "train for dropout only, inference_mode with no gradients or optimizer",
            "note": "MCIA uses LayerNorm rather than BatchNorm, so no running-statistics update occurs",
        },
        "masked_patch_count": int(len(values)),
        "spearman_uncertainty_vs_l2_error": correlation,
        "calibration_rule": "Spearman correlation greater than 0.2 is a calibration pass",
        "calibration_pass": bool(correlation is not None and correlation > 0.2),
        "per_subject": per_subject,
        "interpretation_limit": (
            "A pass only establishes DB2 reconstruction-error calibration. It does not "
            "establish that uncertainty predicts DB3 angle-estimation gain."
        ),
    }
    out = run_dir / "06_diagnostics" / "mcia_mc_dropout_calibration"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    out_file = out / "metrics" / "db2_mc_dropout_calibration.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "db2_mc_dropout_calibration.npz",
        subject_id=values[:, 0].astype(np.int32),
        channel_id=values[:, 1].astype(np.int32),
        patch_id=values[:, 2].astype(np.int32),
        uncertainty=uncertainty,
        l2_error=error,
    )
    print(
        f"  masked_patches={len(values)} Spearman={correlation:+.4f} "
        f"pass={report['calibration_pass']}"
    )
    for label, item in per_subject.items():
        print(
            f"  {label}: patches={item['masked_patch_count']} "
            f"Spearman={item['spearman_uncertainty_vs_l2_error']:+.4f}"
        )
    print(f"[diagnostic] saved: {out_file}")


if __name__ == "__main__":
    main()
