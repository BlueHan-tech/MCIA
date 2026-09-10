"""No-leak S05 development screen for conditional MCIA replacement strength."""
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

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config, set_seed
from utils.rule_anomaly_detector import LABEL_S1, LABEL_S2, LABEL_S3, LABEL_S4, RuleAnomalyDetector


def load_exp3():
    path = ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py"
    spec = importlib.util.spec_from_file_location("conditional_alpha_exp3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_alphas(text):
    values = [float(item) for item in text.split(",")]
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise ValueError("s1-s3 alphas must be in (0, 1]")
    return sorted(set(values))


def conditional_alpha_map(labels, s1_s3_alpha):
    alpha = np.zeros(labels.shape, dtype=np.float32)
    alpha[(labels == LABEL_S1) | (labels == LABEL_S3)] = s1_s3_alpha
    alpha[(labels == LABEL_S2) | (labels == LABEL_S4)] = 1.0
    return alpha


@torch.no_grad()
def make_conditional_enhanced(mcia, raw_windows, detector, device, batch_size, s1_s3_alpha):
    metadata = detector.detect_batch(raw_windows)
    masks = metadata["mask"].astype(np.float32)
    alpha = conditional_alpha_map(metadata["labels"], s1_s3_alpha)
    enhanced = np.empty_like(raw_windows)
    for start in range(0, len(raw_windows), batch_size):
        stop = min(start + batch_size, len(raw_windows))
        raw_t = torch.as_tensor(raw_windows[start:stop], dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(masks[start:stop], dtype=torch.float32, device=device)
        alpha_t = torch.as_tensor(alpha[start:stop], dtype=torch.float32, device=device)
        chan_valid = (mask_t.mean(dim=1) > 0.5).float()
        completed = mcia(raw_t * mask_t, raw_time_mask=mask_t, chan_valid_mask=chan_valid)
        enhanced[start:stop] = (raw_t + alpha_t * (completed - raw_t)).cpu().numpy()
    return enhanced, metadata


def summarize_mask_labels(metadata):
    labels = metadata["labels"]
    masked = labels != 0
    return {
        "masked_token_ratio": float(masked.mean()),
        "s1_ratio": float((labels == LABEL_S1).mean()),
        "s2_ratio": float((labels == LABEL_S2).mean()),
        "s3_ratio": float((labels == LABEL_S3).mean()),
        "s4_ratio": float((labels == LABEL_S4).mean()),
    }


def metric_payload(exp3, pred, target, dynamic_threshold, dynamic_score_name):
    full = exp3.evaluate_subsets(pred, target)
    scores = exp3.compute_dynamic_angle_scores(target, dynamic_score_name)
    dynamic_idx = np.flatnonzero(scores >= dynamic_threshold)
    return {
        "subsets": full,
        "dynamic_subsets": exp3.evaluate_subsets(pred[dynamic_idx], target[dynamic_idx]) if len(dynamic_idx) else {},
        "n_dynamic": int(len(dynamic_idx)),
        "n_total": int(len(target)),
        "dynamic_ratio": float(len(dynamic_idx) / max(len(target), 1)),
    }


def global_rmse(metric):
    return float(metric["subsets"]["global"]["rmse"])


def evaluate_condition(exp3, name, emg, angle, train_idx, val_idx, test_idx, config, device, output_dir, seed, threshold, score_name):
    set_seed(seed)
    model, best_val_loss, epochs = exp3.train_tcn_on_emg(emg, angle, train_idx, val_idx, config, device)
    val_pred, val_target = exp3.predict_on_set(model, emg, angle, val_idx, config, device)
    val_metrics = metric_payload(exp3, val_pred, val_target, threshold, score_name)
    checkpoint = output_dir / "checkpoints" / f"{name}_best.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return {
        "model": model,
        "name": name,
        "best_val_loss": float(best_val_loss),
        "epochs": int(epochs),
        "validation": val_metrics,
        "checkpoint": str(checkpoint),
    }


def main():
    parser = argparse.ArgumentParser(description="DB3 conditional-alpha MCIA development diagnostic")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--exercise", type=int, default=1)
    parser.add_argument("--s1-s3-alphas", default="0.05,0.10,0.20")
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    env_run_dir = os.environ.get("MCIA_RUN_DIR")
    if not env_run_dir or Path(env_run_dir).resolve() != run_dir:
        raise RuntimeError("MCIA_RUN_DIR must equal --run-dir")
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    alphas = parse_alphas(args.s1_s3_alphas)
    config = flatten_pipeline_config(load_yaml_config(ROOT))
    device = config["device"]
    batch_size = args.batch_size or int(config["regressor_batch_size"])
    dynamic_threshold = float(config.get("regressor_dynamic_min_ptp", 0.05))
    dynamic_score_name = str(config.get("regressor_dynamic_score", "global_max_ptp"))
    exp3 = load_exp3()
    out = run_dir / "06_diagnostics" / "db3_conditional_alpha_completion"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    raw_emg, angle, _, repetitions = prepare_kinematics_data(
        loader, [args.subject], config, exercises=[args.exercise], db="db3"
    )
    train_idx, val_idx, test_idx = make_rep_split(
        repetitions,
        train_reps=(1, 3, 4, 6),
        test_reps=(2, 5),
        val_ratio=float(config["regressor_val_ratio"]),
        seed=int(config["regressor_random_seed"]) + args.subject,
    )
    detector = RuleAnomalyDetector(
        patch_size=int(config["patch_size"]), group_indices=config.get("group_indices")
    ).fit(raw_emg[train_idx])
    mcia, mcia_checkpoint = exp3.load_healthy_prior_mcia(config, device)
    if mcia is None:
        raise FileNotFoundError("healthy-prior MCIA checkpoint")

    print(
        f"[conditional alpha] S{args.subject:02d} train/val/test="
        f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}; selecting only on validation"
    )
    enhanced_by_name = {"raw": raw_emg}
    label_summary = {}
    for alpha in [1.0] + alphas:
        name = "legacy_full" if alpha == 1.0 else f"conditional_alpha_{alpha:.2f}"
        enhanced, metadata = make_conditional_enhanced(
            mcia, raw_emg, detector, device, batch_size, alpha
        )
        enhanced_by_name[name] = enhanced
        label_summary[name] = summarize_mask_labels(metadata)
        print(f"  prepared {name}")

    seed = int(config["regressor_random_seed"]) + args.subject
    conditions = []
    for name, emg in enhanced_by_name.items():
        result = evaluate_condition(
            exp3, name, emg, angle, train_idx, val_idx, test_idx, config, device,
            out, seed, dynamic_threshold, dynamic_score_name,
        )
        conditions.append(result)
        print(
            f"  {name}: validation global RMSE="
            f"{global_rmse(result['validation']):.6f}, epochs={result['epochs']}"
        )

    winner = min(conditions, key=lambda item: global_rmse(item["validation"]))
    print(f"[locked by validation] {winner['name']}")
    heldout_by_name = {}
    for item in conditions:
        test_pred, test_target = exp3.predict_on_set(
            item["model"], enhanced_by_name[item["name"]], angle, test_idx, config, device
        )
        heldout_by_name[item["name"]] = metric_payload(
            exp3, test_pred, test_target, dynamic_threshold, dynamic_score_name
        )
        np.savez_compressed(
            out / "predictions" / f"S{args.subject:02d}_{item['name']}_heldout_test.npz",
            prediction=test_pred,
            target=test_target,
        )

    report_conditions = []
    for item in conditions:
        report_conditions.append({
            "name": item["name"],
            "s1_s3_alpha": 0.0 if item["name"] == "raw" else (1.0 if item["name"] == "legacy_full" else float(item["name"].rsplit("_", 1)[1])),
            "s2_s4_alpha": 0.0 if item["name"] == "raw" else 1.0,
            "best_val_loss": item["best_val_loss"],
            "epochs": item["epochs"],
            "checkpoint": item["checkpoint"],
            "validation": item["validation"],
            "heldout_test": heldout_by_name[item["name"]],
            "mask_labels": label_summary.get(item["name"], {"masked_token_ratio": 0.0}),
        })
    report = {
        "scope": "S05 development diagnostic; Exp3 and existing outputs are unchanged",
        "leakage_control": {
            "detector_fit": "train repetitions only",
            "model_training": "train repetitions only",
            "policy_selection": "lowest validation global RMSE",
            "heldout_test": "predicted only after policy selection",
            "test_repetitions": [2, 5],
        },
        "subject_id": int(args.subject),
        "exercise": int(args.exercise),
        "mcia_source": "healthy_prior",
        "mcia_checkpoint": str(mcia_checkpoint),
        "split_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "dynamic_definition": {"score": dynamic_score_name, "min_ptp": dynamic_threshold},
        "policy": "S1/S3 use candidate alpha; S2/S4 use alpha=1.0; keep remains raw",
        "conditions": report_conditions,
        "selected_by_validation": winner["name"],
        "heldout_test_selected_policy": heldout_by_name[winner["name"]],
        "heldout_test_all_predeclared_conditions": heldout_by_name,
        "interpretation_limit": "This establishes only the healthy-prior conditional policy on one development subject. S06 must independently confirm it before any Exp3 default changes.",
    }
    out_file = out / "metrics" / f"S{args.subject:02d}_conditional_alpha.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[heldout test] {winner['name']} global RMSE="
        f"{global_rmse(heldout_by_name[winner['name']]):.6f}; dynamic "
        f"{heldout_by_name[winner['name']]['n_dynamic']}/{heldout_by_name[winner['name']]['n_total']}"
    )
    print(f"[diagnostic] saved: {out_file}")


if __name__ == "__main__":
    main()
