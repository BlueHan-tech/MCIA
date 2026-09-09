"""Empirical-null calibration of the MQP mask threshold on healthy DB2 subjects.

Rationale: DB3 amputee recordings are the only place genuine residual-limb
electrode failures occur, but we have no ground truth there. Healthy DB2
subjects (same NinaPro family: 12 channels, 2 kHz) provide an empirical null:
their flag rate approximates the detector's false-positive rate.

Pre-declared decision rule (fixed before looking at results):
  choose the smallest threshold whose pooled healthy flag rate is <= 5%,
  i.e. the healthy 95th percentile of per-channel-second p values.
  If the current 0.20 already satisfies the bound, the calibration validates
  it and no change is proposed.

Hygiene: DB2 train/val subjects only (S33-S40 test subjects untouched);
active intervals only (restimulus != 0); E1+E2; exact same MQP code path
(utils.db3_quality_mask.mqp_probability) as the DB3 mask. Writes only to
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
import yaml

from data.ninapro_loader import NinaProDataLoader
from utils.db3_quality_mask import _runs, mqp_probability

SUBJECTS = list(range(1, 11))          # DB2 S01-S10, train split only
EXERCISES = [1, 2]
TARGET_RATE = 0.05
THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def main() -> None:
    started = time.monotonic()
    run = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    out = run / "06_diagnostics" / "mqp_healthy_null_calibration_20260909"
    out.mkdir(parents=True, exist_ok=False)

    root = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = dict(root["signal"], **root["exp1_mcia"])
    loader = NinaProDataLoader(root["paths"]["db2"], root["paths"]["db3"], fs=cfg["orig_fs"])

    per_subject = {}
    pooled = []
    for subject_id in SUBJECTS:
        try:
            data = loader.load_db2_subject(subject_id, EXERCISES)
        except Exception as exc:
            print(f"S{subject_id:02d} skipped: {exc}", flush=True)
            continue
        probs = []
        emg = np.asarray(data["emg"], dtype=np.float64)
        labels = np.asarray(data.get("restimulus", data.get("stimulus"))).reshape(-1)
        reps = np.asarray(data["repetition"]).reshape(-1)
        n = min(len(emg), len(labels), len(reps))
        emg, labels, reps = emg[:n], labels[:n], reps[:n]
        active = labels > 0
        centered = emg - emg.mean(axis=0, keepdims=True)
        for start, end in _runs(active):
            for cursor in range(start, end - 2000 + 1, 2000):
                if np.all(reps[cursor:cursor + 2000] == reps[cursor]):
                    probs.append(mqp_probability(centered[cursor:cursor + 2000]))
        if not probs:
            print(f"S{subject_id:02d}: no aligned active seconds", flush=True)
            continue
        P = np.stack(probs)                     # (n_seconds, 12)
        per_subject[subject_id] = P
        pooled.append(P)
        np.savez_compressed(out / f"S{subject_id:02d}_mqp.npz", probabilities=P)
        rates = " ".join(f"{t:.2f}:{(P > t).mean():.2%}" for t in (0.10, 0.20, 0.25))
        print(f"S{subject_id:02d}: {len(P)} channel-seconds | {rates}", flush=True)

    A = np.concatenate(pooled)
    p95 = float(np.quantile(A, 0.95))
    grid_rates = {f"{t:.2f}": float((A > t).mean()) for t in THRESHOLD_GRID}
    qualifying = [t for t in THRESHOLD_GRID if (A > t).mean() <= TARGET_RATE]
    proposed = min(qualifying) if qualifying else None
    subj_rates = {
        f"S{sid:02d}": {f"{t:.2f}": float((P > t).mean()) for t in THRESHOLD_GRID}
        for sid, P in per_subject.items()
    }
    worst_at_proposed = None
    if proposed is not None:
        worst_at_proposed = max(subj_rates[s][f"{proposed:.2f}"] for s in subj_rates)

    report = {
        "protocol": {
            "purpose": "empirical-null calibration of MQP threshold on healthy DB2",
            "subjects": SUBJECTS, "exercises": EXERCISES,
            "test_subjects_touched": False,
            "active_interval_rule": "restimulus != 0, 2000-sample aligned seconds within one repetition",
            "mqp_code_path": "utils.db3_quality_mask.mqp_probability (identical to DB3 mask)",
            "decision_rule_predeclared": "smallest grid threshold with pooled healthy flag rate <= 5%",
            "target_false_positive_rate": TARGET_RATE,
            "threshold_grid": THRESHOLD_GRID,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "n_channel_seconds": int(A.size),
        "p_quantiles": {f"p{q}": float(np.quantile(A, q)) for q in (0.50, 0.75, 0.90, 0.95, 0.98)},
        "pooled_p95_continuous": p95,
        "pooled_flag_rates": grid_rates,
        "proposed_threshold": proposed,
        "worst_subject_rate_at_proposed": worst_at_proposed,
        "per_subject_flag_rates": subj_rates,
        "elapsed_seconds": time.monotonic() - started,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\npooled quantiles:", {k: round(v, 4) for k, v in report["p_quantiles"].items()})
    print("pooled flag rates:", {k: f"{v:.2%}" for k, v in grid_rates.items()})
    print(f"proposed threshold (rule: smallest with pooled rate <= {TARGET_RATE:.0%}):", proposed)
    if proposed is not None:
        print(f"  continuous P95 = {p95:.4f}; worst per-subject rate at {proposed:.2f}: "
              f"{worst_at_proposed:.2%}")
    print(f"results={out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
