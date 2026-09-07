"""Read-only audit of Exp3 RuleAnomalyDetector coverage for DB3 subjects."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config
from utils.rule_anomaly_detector import LABEL_NAMES, RuleAnomalyDetector


def parse_subjects(text):
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("subjects must not be empty")
    return values


def label_summary(labels, missing):
    result = {}
    missing_count = int(missing.sum())
    for label, name in LABEL_NAMES.items():
        count = int((labels == label).sum())
        item = {"count": count, "fraction_all_channel_patches": float(count / labels.size)}
        if label != 0:
            item["fraction_of_masked_channel_patches"] = (
                float(count / missing_count) if missing_count else 0.0
            )
        result[name] = item
    return result


def split_summary(detector, windows):
    output = detector.detect_batch(windows)
    patch_mask = output["patch_mask"]
    labels = output["patch_labels"]
    missing = patch_mask < 0.5
    masked_per_window = missing.sum(axis=(1, 2))
    any_time_patch = missing.any(axis=2).sum(axis=1)
    valid_channels = (patch_mask > 0.5).sum(axis=2)
    unrecoverable = output["unrecoverable_patches"]
    return {
        "window_count": int(len(windows)),
        "channel_patch_tokens_per_window": int(np.prod(patch_mask.shape[1:])),
        "mean_mask_ratio": float(missing.mean()),
        "mask_ratio_p50": float(np.quantile(missing.mean(axis=(1, 2)), 0.50)),
        "mask_ratio_p90": float(np.quantile(missing.mean(axis=(1, 2)), 0.90)),
        "mask_ratio_max": float(missing.mean(axis=(1, 2)).max()),
        "mean_masked_channel_patches_per_window": float(masked_per_window.mean()),
        "mean_time_patches_with_any_mask_per_window": float(any_time_patch.mean()),
        "unrecoverable_patch_fraction": float(unrecoverable.mean()),
        "windows_with_any_unrecoverable_fraction": float(unrecoverable.any(axis=1).mean()),
        "time_patches_below_min_valid_channels_fraction": float(
            (valid_channels < detector.cfg.min_valid_channels).mean()
        ),
        "label_distribution": label_summary(labels, missing),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read-only DB3 RuleAnomalyDetector coverage audit"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subjects", default="5,6")
    parser.add_argument("--exercise", type=int, default=1)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    env_run_dir = os.environ.get("MCIA_RUN_DIR")
    if not env_run_dir or Path(env_run_dir).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")
    subjects = parse_subjects(args.subjects)
    config = flatten_pipeline_config(load_yaml_config(ROOT))
    loader = NinaProDataLoader(
        config["db2_path"], config["db3_path"], fs=int(config["orig_fs"])
    )
    subject_reports = []
    for subject_id in subjects:
        emg, _, _, repetitions = prepare_kinematics_data(
            loader, [subject_id], config, exercises=[args.exercise], db="db3"
        )
        train_idx, val_idx, test_idx = make_rep_split(
            repetitions,
            train_reps=(1, 3, 4, 6),
            test_reps=(2, 5),
            val_ratio=float(config["regressor_val_ratio"]),
            seed=int(config["regressor_random_seed"]) + subject_id,
        )
        detector = RuleAnomalyDetector(
            patch_size=int(config["patch_size"]),
            group_indices=config.get("group_indices"),
        ).fit(emg[train_idx])
        split_indices = {"train": train_idx, "validation": val_idx, "test": test_idx}
        splits = {
            name: split_summary(detector, emg[indices])
            for name, indices in split_indices.items()
        }
        report = {
            "subject_id": subject_id,
            "mask_source": (
                "RuleAnomalyDetector fit on this subject's training repetitions only, "
                "then applied unchanged to each split"
            ),
            "detector_config": {
                "patch_size_samples": detector.cfg.patch_size,
                "patch_duration_ms": 1000.0 * detector.cfg.patch_size / float(config["target_fs"]),
                "min_valid_channels": detector.cfg.min_valid_channels,
                "max_mask_ratio": detector.cfg.max_mask_ratio,
            },
            "fit_dead_channels_zero_based": np.where(detector.dead_channels_)[0].astype(int).tolist(),
            "fit_stats": {
                "global_rms": float(detector.fit_stats_["global_rms"]),
                "global_mad": float(detector.fit_stats_["global_mad"]),
            },
            "splits": splits,
        }
        subject_reports.append(report)
        test = splits["test"]
        labels = test["label_distribution"]
        print(
            f"S{subject_id:02d} test: mask={test['mean_mask_ratio']:.3f} "
            f"masked_tokens={test['mean_masked_channel_patches_per_window']:.1f}/"
            f"{test['channel_patch_tokens_per_window']} "
            f"time_patches={test['mean_time_patches_with_any_mask_per_window']:.1f}/32 "
            f"unrecoverable_windows={test['windows_with_any_unrecoverable_fraction']:.3f}"
        )
        print(
            f"  dead={labels['s2_dead_channel']['fraction_of_masked_channel_patches']:.3f} "
            f"s3={labels['s3_group_abnormal']['fraction_of_masked_channel_patches']:.3f} "
            f"s4={labels['s4_time_dropout']['fraction_of_masked_channel_patches']:.3f} "
            f"s1={labels['s1_short_dropout']['fraction_of_masked_channel_patches']:.3f}"
        )
    report = {
        "scope": "read-only audit of the exact RuleAnomalyDetector mask used by Exp3 B/C",
        "angle_policy": "no angle labels, TCN, MCIA inference, or model training is used",
        "subjects": subject_reports,
    }
    out = run_dir / "06_diagnostics" / "db3_rule_mask_coverage"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    out_file = out / "metrics" / "db3_rule_mask_coverage.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diagnostic] saved: {out_file}")


if __name__ == "__main__":
    main()
