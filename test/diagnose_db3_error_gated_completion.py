"""Independent validation-trained EMG risk-gate plus MCIA completion diagnostic.

The test-time gate receives EMG-derived features only.  Validation angle errors
are used solely to fit the gate before the final blinded test evaluation.
This script is intentionally outside Exp1/Exp2/Exp3.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config, set_seed


def _load_module(name: str, filename: str):
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subjects(value: str) -> list[int]:
    output: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = (int(item) for item in token.split("-", 1))
            output.extend(range(lo, hi + 1))
        else:
            output.append(int(token))
    return sorted(set(output))


def _window_features(emg: np.ndarray, detector) -> np.ndarray:
    energy, variation = detector._patch_statistics(emg)
    low = (energy <= detector.energy_threshold) & (variation <= detector.variation_threshold)
    channel_energy = emg.mean(axis=1)
    channel_variation = emg.std(axis=1)
    energy_ratio = channel_energy / (detector.energy_threshold[None] + 1e-8)
    variation_ratio = channel_variation / (detector.variation_threshold[None] + 1e-8)
    return np.column_stack((
        low.mean(axis=(1, 2)),
        (emg <= 1e-6).mean(axis=(1, 2)),
        emg.mean(axis=(1, 2)),
        emg.std(axis=(1, 2)),
        np.min(energy_ratio, axis=1),
        np.median(energy_ratio, axis=1),
        np.min(variation_ratio, axis=1),
        np.median(variation_ratio, axis=1),
    ))


def _window_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))


def _dynamic_indices(angle: np.ndarray, threshold: float) -> np.ndarray:
    return np.flatnonzero(np.max(np.ptp(angle, axis=1), axis=1) >= threshold)


@torch.no_grad()
def _complete(mcia, emg: np.ndarray, masks: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    output = np.empty_like(emg)
    for start in range(0, len(emg), batch_size):
        stop = min(start + batch_size, len(emg))
        values = torch.as_tensor(emg[start:stop], dtype=torch.float32, device=device)
        mask = torch.as_tensor(masks[start:stop], dtype=torch.float32, device=device)
        valid = (mask.mean(dim=1) > 0.5).float()
        pred = mcia(values * mask, raw_time_mask=mask, chan_valid_mask=valid)
        output[start:stop] = (pred * (1.0 - mask) + values * mask).cpu().numpy()
    return output


def _mask_summary(masks: np.ndarray, trigger: np.ndarray) -> dict:
    missing = masks < 0.5
    return {
        "triggered_windows": int(trigger.sum()),
        "triggered_ratio": float(trigger.mean()),
        "missing_ratio": float(missing.mean()),
        "per_channel_missing_ratio": [float(value) for value in missing.mean(axis=(0, 1))],
    }


def _train(exp3, emg, angle, train_idx, val_idx, test_idx, config, device, seed):
    set_seed(seed)
    model, best_val, epochs = exp3.train_tcn_on_emg(emg, angle, train_idx, val_idx, config, device)
    val_pred, val_target = exp3.predict_on_set(model, emg, angle, val_idx, config, device)
    test_pred, test_target = exp3.predict_on_set(model, emg, angle, test_idx, config, device)
    return model, float(best_val), int(epochs), val_pred, val_target, test_pred, test_target


def _metrics(exp3, pred, target, dynamic_threshold):
    dynamic = _dynamic_indices(target, dynamic_threshold)
    return {
        "full_subsets": exp3.evaluate_subsets(pred, target),
        "dynamic_subsets": exp3.evaluate_subsets(pred[dynamic], target[dynamic]) if len(dynamic) else None,
        "n_dynamic": int(len(dynamic)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DB3 validation-error-gated MCIA completion diagnostic")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subjects", default="5,6")
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--patch-quantile", type=float, default=0.10)
    parser.add_argument("--high-error-quantile", type=float, default=0.80)
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if not (0.0 < args.patch_quantile < 0.5 and 0.5 < args.high_error_quantile < 1.0):
        raise ValueError("quantiles must be inside their documented ranges")

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    config = flatten_pipeline_config(load_yaml_config(PROJECT_ROOT))
    device = config["device"]
    exp3 = _load_module("exp3_error_gate_helpers", "04_eval_db3_angle_raw_vs_augmented.py")
    quality = _load_module("quality_mask_helpers", "diagnose_db3_quality_mask_completion.py")
    mcia, mcia_checkpoint = exp3.load_healthy_prior_mcia(config, device)
    if mcia is None:
        raise FileNotFoundError("Healthy-prior MCIA checkpoint not found")
    output_dir = run_dir / "06_diagnostics" / "db3_error_gated_completion"
    for name in ("metrics", "predictions", "checkpoints"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    dynamic_threshold = float(config.get("regressor_dynamic_min_ptp", 0.05))
    report = {
        "scope": "independent validation-error-trained, EMG-only test gate plus MCIA completion",
        "mcia_checkpoint": str(mcia_checkpoint),
        "gate": {
            "validation_target": f"top {1.0 - args.high_error_quantile:.0%} raw-TCN validation RMSE",
            "test_inputs": "EMG features only",
            "gate_threshold": args.gate_threshold,
            "patch_quantile": args.patch_quantile,
            "important_limit": "shared subject-wide input normalization follows the current Exp3 loader convention",
        },
        "subjects": [],
    }

    for subject_id in _subjects(args.subjects):
        print(f"\n[error-gated completion] S{subject_id:02d}")
        emg, angle, _, repetitions = prepare_kinematics_data(loader, [subject_id], config, exercises=[args.exercise], db="db3")
        train_idx, val_idx, test_idx = make_rep_split(
            repetitions, train_reps=(1, 3, 4, 6), test_reps=(2, 5),
            val_ratio=float(config["regressor_val_ratio"]), seed=int(config["regressor_random_seed"]) + subject_id,
        )
        raw_model, raw_val, raw_epochs, val_pred, val_target, raw_test_pred, test_target = _train(
            exp3, emg, angle, train_idx, val_idx, test_idx, config, device, args.seed + subject_id
        )
        val_error = _window_rmse(val_pred, val_target)
        error_cutoff = float(np.quantile(val_error, args.high_error_quantile))
        high_error = val_error >= error_cutoff
        detector = quality.LowInformationPatchDetector(config["patch_size"], args.patch_quantile).fit(emg[train_idx])
        gate = make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", random_state=args.seed + subject_id, max_iter=1000
        ))
        gate.fit(_window_features(emg[val_idx], detector), high_error.astype(int))
        val_risk = gate.predict_proba(_window_features(emg[val_idx], detector))[:, 1]
        test_risk = gate.predict_proba(_window_features(emg[test_idx], detector))[:, 1]
        train_trigger = gate.predict_proba(_window_features(emg[train_idx], detector))[:, 1] >= args.gate_threshold
        val_trigger = val_risk >= args.gate_threshold
        test_trigger = test_risk >= args.gate_threshold
        base_masks = detector.detect_batch(emg)
        masks = np.ones_like(base_masks)
        masks[train_idx[train_trigger]] = base_masks[train_idx[train_trigger]]
        masks[val_idx[val_trigger]] = base_masks[val_idx[val_trigger]]
        masks[test_idx[test_trigger]] = base_masks[test_idx[test_trigger]]
        enhanced = _complete(mcia, emg, masks, device, int(config["regressor_batch_size"]))
        _, completed_val, completed_epochs, _, _, completed_test_pred, _ = _train(
            exp3, enhanced, angle, train_idx, val_idx, test_idx, config, device, args.seed + subject_id
        )
        test_error = _window_rmse(raw_test_pred, test_target)
        high_test_error = test_error >= error_cutoff
        raw_metrics = _metrics(exp3, raw_test_pred, test_target, dynamic_threshold)
        completed_metrics = _metrics(exp3, completed_test_pred, test_target, dynamic_threshold)
        val_auc = float(roc_auc_score(high_error, val_risk)) if len(np.unique(high_error)) == 2 else None
        test_auc = float(roc_auc_score(high_test_error, test_risk)) if len(np.unique(high_test_error)) == 2 else None
        delta = {
            subset: {
                "full_rmse_completed_minus_raw": float(completed_metrics["full_subsets"][subset]["rmse"] - raw_metrics["full_subsets"][subset]["rmse"]),
                "dynamic_rmse_completed_minus_raw": (
                    float(completed_metrics["dynamic_subsets"][subset]["rmse"] - raw_metrics["dynamic_subsets"][subset]["rmse"])
                    if raw_metrics["dynamic_subsets"] and completed_metrics["dynamic_subsets"] else None
                ),
            }
            for subset in raw_metrics["full_subsets"]
        }
        print(f"  gate: val_auc={val_auc:.3f} test_auc={test_auc:.3f} test_trigger={test_trigger.mean():.1%} "
              f"mask={_mask_summary(masks[test_idx], test_trigger)['missing_ratio']:.2%}")
        print(f"  global RMSE raw={raw_metrics['full_subsets']['global']['rmse']:.4f} "
              f"completed={completed_metrics['full_subsets']['global']['rmse']:.4f} "
              f"delta={delta['global']['full_rmse_completed_minus_raw']:+.4f}")
        torch.save(raw_model.state_dict(), output_dir / "checkpoints" / f"S{subject_id:02d}_raw_best.pth")
        np.savez(
            output_dir / "predictions" / f"S{subject_id:02d}_error_gated_completion.npz",
            target=test_target, pred_raw=raw_test_pred, pred_completed=completed_test_pred,
            test_gate_risk=test_risk, test_gate_trigger=test_trigger, mask=masks[test_idx], test_indices=test_idx,
        )
        report["subjects"].append({
            "subject_id": subject_id,
            "split": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "raw_tcn": {"best_val_loss": raw_val, "epochs": raw_epochs},
            "completed_tcn": {"best_val_loss": completed_val, "epochs": completed_epochs},
            "error_cutoff_from_validation": error_cutoff,
            "gate": {
                "validation_auc_in_sample": val_auc,
                "test_auc_post_hoc": test_auc,
                "train": _mask_summary(masks[train_idx], train_trigger),
                "validation": _mask_summary(masks[val_idx], val_trigger),
                "test": _mask_summary(masks[test_idx], test_trigger),
            },
            "methods": {"raw": raw_metrics, "error_gated_mcia": completed_metrics},
            "delta": delta,
        })
    with (output_dir / "metrics" / "error_gated_completion_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\n[diagnostic] saved: {output_dir}")


if __name__ == "__main__":
    main()
