"""Read-only reproduction of Gronlund et al. (2005) MQP signal-quality method.

Paper: C. Gronlund et al., "On-line signal quality estimation of multichannel
surface electromyograms", Med. Biol. Eng. Comput. 43, 357-364, 2005,
doi:10.1007/BF02345813.

Sections 2.1 and 2.4 are reproduced without custom detector features or
thresholds. DB3 is sampled at 2000 Hz while the paper used 2048 Hz, so every
independent DB3 one-second segment is resampled to 2048 samples before the
paper's exact 8- and 256-sample descriptor windows are applied.
"""
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
from scipy.signal import resample_poly

from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config


PAPER_FS = 2048
SHORT_SAMPLES = 8                 # tau_1 = 1/256 s at 2048 Hz
LONG_SAMPLES = 256                # tau_2 = 1/8 s at 2048 Hz
OUTER_MULTIPLIER = 1.6            # Section 2.1.2
POOR_QUALITY_THRESHOLD = 0.05     # Section 3.1
EXPECTED_ACTIONS = set(range(1, 49))
TRAIN_REPETITIONS = {1, 3, 4}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _select_segments(raw: np.ndarray, labels: np.ndarray, repetitions: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Select non-overlapping, one-second, action/repetition-pure training segments."""
    source_samples = 2000
    valid = np.isin(repetitions, list(TRAIN_REPETITIONS)) & np.isin(labels, list(EXPECTED_ACTIONS))
    segments: list[np.ndarray] = []
    metadata: list[dict] = []
    for start, end in _runs(valid):
        for cursor in range(start, end - source_samples + 1, source_samples):
            label_window = labels[cursor:cursor + source_samples]
            rep_window = repetitions[cursor:cursor + source_samples]
            if np.all(label_window == label_window[0]) and np.all(rep_window == rep_window[0]):
                segments.append(raw[cursor:cursor + source_samples])
                metadata.append({"start_sample_2000hz": int(cursor), "action": int(label_window[0]), "repetition": int(rep_window[0])})
    if not segments:
        raise ValueError("No action- and repetition-pure one-second training segments.")
    return np.stack(segments).astype(np.float64), metadata


def _descriptors(one_second_2048: np.ndarray) -> np.ndarray:
    """Section 2.1.1: K=256 short SDs and repeated long SDs, shape K,N,2."""
    if one_second_2048.shape[0] != PAPER_FS:
        raise ValueError(f"Expected {PAPER_FS} samples, got {one_second_2048.shape[0]}")
    n_channels = one_second_2048.shape[1]
    short = one_second_2048.reshape(-1, SHORT_SAMPLES, n_channels).std(axis=1, ddof=0)
    long = one_second_2048.reshape(-1, LONG_SAMPLES, n_channels).std(axis=1, ddof=0)
    return np.stack((short, np.repeat(long, LONG_SAMPLES // SHORT_SAMPLES, axis=0)), axis=-1)


def _mqp_flags(observations: np.ndarray) -> np.ndarray:
    """Sections 2.1.2 equations (1)-(4), one MQP outlier flag per channel."""
    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(f"Expected N by 2 observations, got {observations.shape}")
    n_channels = observations.shape[0]
    center = np.median(observations, axis=0)
    distances = np.linalg.norm(observations - center, axis=1)
    subset = observations[np.argsort(distances, kind="stable")[:n_channels // 2]]
    covariance = np.cov(subset, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or np.min(eigenvalues) <= 0.0:
        return np.zeros(n_channels, dtype=bool)

    principal = observations @ eigenvectors
    lower, middle, upper = np.percentile(principal, [11.7, 50.0, 88.3], axis=0)
    upper_radius = OUTER_MULTIPLIER * (upper - middle)
    lower_radius = OUTER_MULTIPLIER * (middle - lower)
    if np.any(upper_radius <= 0.0) or np.any(lower_radius <= 0.0):
        return np.zeros(n_channels, dtype=bool)

    centered = principal - middle
    angle = np.arctan2(centered[:, 1], centered[:, 0])
    a = np.where(np.cos(angle) >= 0.0, upper_radius[0], lower_radius[0])
    b = np.where(np.sin(angle) >= 0.0, upper_radius[1], lower_radius[1])
    boundary = np.sqrt((b * np.sin(angle)) ** 2 + (a * np.cos(angle)) ** 2)
    return np.linalg.norm(centered, axis=1) > boundary


def paper_quality(one_second_2000: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return p_j=D_j/K and K by N flags for one DB3 one-second segment."""
    if one_second_2000.shape[0] != 2000:
        raise ValueError(f"Expected 2000 samples, got {one_second_2000.shape[0]}")
    resampled = resample_poly(one_second_2000, up=128, down=125, axis=0)
    descriptors = _descriptors(resampled)
    flags = np.stack([_mqp_flags(descriptors[index]) for index in range(descriptors.shape[0])])
    return flags.mean(axis=0), flags


def _evaluate(loader: NinaProDataLoader, subject_id: int, exercise: int) -> dict:
    data = loader.load_db3_subject(subject_id, [exercise])
    raw = np.asarray(data["emg"], dtype=np.float64)
    labels = np.asarray(data["restimulus"]).reshape(-1)
    repetitions = np.asarray(data["repetition"]).reshape(-1)
    usable = min(len(raw), len(labels), len(repetitions))
    raw, labels, repetitions = raw[:usable], labels[:usable], repetitions[:usable]
    raw -= raw.mean(axis=0, keepdims=True)  # Section 2.4: subtract DC level
    segments, metadata = _select_segments(raw, labels, repetitions)
    probabilities, flags = zip(*(paper_quality(segment) for segment in segments))
    probabilities_array = np.stack(probabilities)
    flags_array = np.stack(flags)
    return {"probabilities": probabilities_array, "flags": flags_array,
            "poor": probabilities_array > POOR_QUALITY_THRESHOLD, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Gronlund 2005 MQP reproduction on DB3")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subjects", default="2,3,4,5,6,7,8,9,11")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not os.environ.get("MCIA_RUN_DIR") or Path(os.environ["MCIA_RUN_DIR"]).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")

    config = flatten_pipeline_config(load_yaml_config(ROOT))
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=int(config["orig_fs"]))
    out_dir = run_dir / "06_diagnostics" / "gronlund_2005_mqp_db3"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "paper": {"doi": "10.1007/BF02345813", "method": "modified quelplot (MQP)"},
        "scope": "DB3 fixed-48 gesture subjects, E1-E3, training repetitions 1/3/4 only",
        "paper_constants": {"analysis_seconds": 1.0, "paper_sampling_hz": PAPER_FS,
                            "short_sd_window_seconds": "1/256", "long_sd_window_seconds": "1/8",
                            "robust_subset_fraction": 0.5, "percentiles": [11.7, 50.0, 88.3],
                            "outer_multiplier": OUTER_MULTIPLIER, "poor_quality_rule": "p_j > 0.05"},
        "db3_compatibility": "Each 2000-Hz one-second interval is resampled to 2048 samples before exact paper windows are applied.",
        "subjects": [],
    }
    subjects = [int(value) for value in args.subjects.split(",") if value.strip()]
    for subject_id in subjects:
        subject_report = {"subject_id": subject_id, "exercises": []}
        for exercise in (1, 2, 3):
            result = _evaluate(loader, subject_id, exercise)
            np.savez_compressed(out_dir / f"S{subject_id:02d}_E{exercise}_mqp.npz",
                                probabilities=result["probabilities"], outlier_flags=result["flags"],
                                poor_quality=result["poor"],
                                start_sample_2000hz=np.asarray([x["start_sample_2000hz"] for x in result["metadata"]]),
                                action=np.asarray([x["action"] for x in result["metadata"]]),
                                repetition=np.asarray([x["repetition"] for x in result["metadata"]]))
            poor = result["poor"]
            per_channel = []
            for channel in range(poor.shape[1]):
                values = result["probabilities"][:, channel]
                per_channel.append({"channel_one_based": channel + 1,
                                    "one_second_segments_p_gt_0_05": int(poor[:, channel].sum()),
                                    "one_second_segments_fraction_p_gt_0_05": float(poor[:, channel].mean()),
                                    "median_p": float(np.median(values)), "max_p": float(values.max())})
            subject_report["exercises"].append({"exercise": exercise, "one_second_segments": int(len(result["metadata"])),
                                                 "poor_channel_segment_fraction": float(poor.mean()),
                                                 "segments_with_any_poor_channel_fraction": float(poor.any(axis=1).mean()),
                                                 "per_channel": per_channel})
            print(f"S{subject_id:02d} E{exercise}: segments={len(result['metadata'])} poor={poor.mean():.3%}", flush=True)
        report["subjects"].append(subject_report)
    (out_dir / "gronlund_2005_mqp_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
