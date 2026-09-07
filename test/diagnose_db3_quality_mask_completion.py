"""Independent no-leakage test of short low-information EMG patch completion.

This is not part of Exp1/Exp2/Exp3.  The detector learns per-channel patch
thresholds from DB3 training repetitions only, then applies the fixed rule to
train/validation/test EMG without looking at glove or angle targets.  It
compares raw EMG against the same masked-patch MCIA completion path using the
same Key10 KinematicTCN and repetition split.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np
import torch

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config, set_seed


def _load_exp3_module():
    path = PROJECT_ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("exp3_quality_mask_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Exp3 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LowInformationPatchDetector:
    """EMG-only detector for locally weak and locally flat short patches."""

    def __init__(self, patch_size: int, quantile: float):
        self.patch_size = int(patch_size)
        self.quantile = float(quantile)
        self.energy_threshold: np.ndarray | None = None
        self.variation_threshold: np.ndarray | None = None

    def _patch_statistics(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, t, c = windows.shape
        usable = (t // self.patch_size) * self.patch_size
        if usable == 0:
            raise ValueError("patch_size exceeds window length")
        patches = windows[:, :usable].reshape(n, usable // self.patch_size, self.patch_size, c)
        return patches.mean(axis=2), patches.std(axis=2)

    def fit(self, train_windows: np.ndarray) -> "LowInformationPatchDetector":
        energy, variation = self._patch_statistics(train_windows)
        self.energy_threshold = np.quantile(energy, self.quantile, axis=(0, 1))
        self.variation_threshold = np.quantile(variation, self.quantile, axis=(0, 1))
        return self

    def detect(self, window: np.ndarray) -> np.ndarray:
        if self.energy_threshold is None or self.variation_threshold is None:
            raise RuntimeError("Detector must be fit on training EMG before inference")
        energy, variation = self._patch_statistics(window[None])
        invalid = (energy[0] <= self.energy_threshold) & (variation[0] <= self.variation_threshold)
        mask = np.ones_like(window, dtype=np.float32)
        for patch_index in range(invalid.shape[0]):
            start = patch_index * self.patch_size
            stop = start + self.patch_size
            mask[start:stop, invalid[patch_index]] = 0.0
        return mask

    def detect_batch(self, windows: np.ndarray) -> np.ndarray:
        return np.stack([self.detect(window) for window in windows])


def _dynamic_indices(angle: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    scores = np.max(np.ptp(angle, axis=1), axis=1)
    return np.flatnonzero(scores >= threshold), scores


def _mask_summary(mask: np.ndarray) -> dict:
    missing = mask < 0.5
    return {
        "missing_ratio": float(missing.mean()),
        "windows_with_missing": int(np.sum(missing.any(axis=(1, 2)))),
        "per_channel_missing_ratio": [float(value) for value in missing.mean(axis=(0, 1))],
    }


def _train_and_evaluate(label: str, emg: np.ndarray, angle: np.ndarray, train_idx: np.ndarray,
                        val_idx: np.ndarray, test_idx: np.ndarray, exp3, config: dict,
                        device: str, output_dir: Path, subject_id: int, dynamic_threshold: float,
                        seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    set_seed(seed)
    model, best_val, epochs = exp3.train_tcn_on_emg(emg, angle, train_idx, val_idx, config, device)
    checkpoint = output_dir / "checkpoints" / f"S{subject_id:02d}_{label}_best.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    pred, target = exp3.predict_on_set(model, emg, angle, test_idx, config, device)
    dynamic_local, scores = _dynamic_indices(target, dynamic_threshold)
    result = {
        "best_val_loss": float(best_val),
        "epochs": int(epochs),
        "checkpoint": str(checkpoint),
        "full_subsets": exp3.evaluate_subsets(pred, target),
        "dynamic_subsets": exp3.evaluate_subsets(pred[dynamic_local], target[dynamic_local]) if len(dynamic_local) else None,
        "n_test": int(len(target)),
        "n_dynamic_test": int(len(dynamic_local)),
        "dynamic_scores": {
            "median": float(np.median(scores)),
            "p90": float(np.percentile(scores, 90)),
            "max": float(np.max(scores)),
        },
    }
    return result, pred, target


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent DB3 short-patch MCIA completion ablation")
    parser.add_argument("--run-dir", required=True, help="Existing outputs/run/<run_id>")
    parser.add_argument("--subjects", default="6", help="Comma/range list; default S06")
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--patch-quantile", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if not (0.0 < args.patch_quantile < 0.5):
        raise ValueError("patch-quantile must be between 0 and 0.5")

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    config = flatten_pipeline_config(load_yaml_config(PROJECT_ROOT))
    device = config["device"]
    exp3 = _load_exp3_module()
    output_dir = run_dir / "06_diagnostics" / "db3_quality_mask_completion_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics", "predictions", "checkpoints"):
        (output_dir / name).mkdir(exist_ok=True)
    subject_ids = []
    for token in args.subjects.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = (int(value) for value in token.split("-", 1))
            subject_ids.extend(range(lo, hi + 1))
        elif token:
            subject_ids.append(int(token))

    mcia, checkpoint = exp3.load_healthy_prior_mcia(config, device)
    if mcia is None or checkpoint is None:
        raise FileNotFoundError("A healthy-prior MCIA checkpoint is required for this diagnostic")
    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    dynamic_threshold = float(config.get("regressor_dynamic_min_ptp", 0.05))
    report = {
        "scope": "independent DB3 raw-vs-EMG-only-quality-mask-plus-MCIA diagnostic",
        "mcia_checkpoint": str(checkpoint),
        "detector": {
            "fit_scope": "training repetitions only",
            "test_inputs": "EMG only; no glove, angle, or test metrics used for masking",
            "patch_size": int(config["patch_size"]),
            "patch_quantile": float(args.patch_quantile),
            "rule": "patch mean and patch standard deviation both at or below train quantile",
        },
        "subjects": [],
    }

    for subject_id in sorted(set(subject_ids)):
        print(f"\n[quality-mask ablation] S{subject_id:02d}")
        emg, angle, _, repetitions = prepare_kinematics_data(
            data_loader, [subject_id], config, exercises=[args.exercise], db="db3"
        )
        train_idx, val_idx, test_idx = make_rep_split(
            repetitions,
            train_reps=(1, 3, 4, 6),
            test_reps=(2, 5),
            val_ratio=float(config["regressor_val_ratio"]),
            seed=int(config["regressor_random_seed"]) + subject_id,
        )
        detector = LowInformationPatchDetector(config["patch_size"], args.patch_quantile).fit(emg[train_idx])
        masks = detector.detect_batch(emg)
        enhanced = exp3.apply_mcia(mcia, emg, detector, device, batch_size=int(config["regressor_batch_size"]))
        print(f"  mask train={_mask_summary(masks[train_idx])['missing_ratio']:.2%} "
              f"val={_mask_summary(masks[val_idx])['missing_ratio']:.2%} "
              f"test={_mask_summary(masks[test_idx])['missing_ratio']:.2%}")

        raw_result, raw_pred, target = _train_and_evaluate(
            "raw", emg, angle, train_idx, val_idx, test_idx, exp3, config, device,
            output_dir, subject_id, dynamic_threshold, args.seed + subject_id,
        )
        completed_result, completed_pred, _ = _train_and_evaluate(
            "quality_mask_mcia", enhanced, angle, train_idx, val_idx, test_idx, exp3, config, device,
            output_dir, subject_id, dynamic_threshold, args.seed + subject_id,
        )
        delta = {
            subset: {
                "rmse_completed_minus_raw": float(completed_result["full_subsets"][subset]["rmse"] - raw_result["full_subsets"][subset]["rmse"]),
                "dynamic_rmse_completed_minus_raw": (
                    float(completed_result["dynamic_subsets"][subset]["rmse"] - raw_result["dynamic_subsets"][subset]["rmse"])
                    if raw_result["dynamic_subsets"] and completed_result["dynamic_subsets"] else None
                ),
            }
            for subset in raw_result["full_subsets"]
        }
        print(f"  global RMSE raw={raw_result['full_subsets']['global']['rmse']:.4f} "
              f"completed={completed_result['full_subsets']['global']['rmse']:.4f} "
              f"delta={delta['global']['rmse_completed_minus_raw']:+.4f}")
        np.savez(
            output_dir / "predictions" / f"S{subject_id:02d}_quality_mask_ablation.npz",
            target=target, pred_raw=raw_pred, pred_quality_mask_mcia=completed_pred,
            mask=masks[test_idx], test_indices=test_idx,
        )
        report["subjects"].append({
            "subject_id": subject_id,
            "split": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "mask": {"train": _mask_summary(masks[train_idx]), "val": _mask_summary(masks[val_idx]), "test": _mask_summary(masks[test_idx])},
            "methods": {"raw": raw_result, "quality_mask_mcia": completed_result},
            "delta": delta,
        })

    with (output_dir / "metrics" / "quality_mask_completion_ablation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\n[diagnostic] saved: {output_dir}")


if __name__ == "__main__":
    main()
