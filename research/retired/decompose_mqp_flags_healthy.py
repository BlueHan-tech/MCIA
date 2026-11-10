"""结合动作和激活元数据分解健康 DB2 的 MQP 标记。

通过同一 MQP 路径重新加载 DB2 S01-S10（训练侧被试，E1+E2 动作秒），逐秒保存
动作、repetition、12 通道 p 值及 RMS 幅值。在阈值 0.20 下报告：
  - “逐通道 × 逐动作”标记结构（与 DB3 分解对应）；
  - 激活对照：总体及逐通道已标记/未标记“通道-秒”的 RMS 中位数，检验标记
    是否跟踪低/高激活；
  - 多通道共同标记分布。

仅写入 <run>/06_diagnostics/。
"""

from __future__ import annotations

import collections
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

SUBJECTS = list(range(1, 11))
EXERCISES = [1, 2]
THRESHOLD = 0.20


def main() -> None:
    started = time.monotonic()
    run = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    out = run / "06_diagnostics" / "mqp_flag_decomposition_healthy_20260909"
    out.mkdir(parents=True, exist_ok=False)
    root = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = dict(root["signal"], **root["exp1_mcia"])
    loader = NinaProDataLoader(root["paths"]["db2"], root["paths"]["db3"], fs=cfg["orig_fs"])

    all_flags, all_rms, all_actions, all_channels = [], [], [], []
    for subject_id in SUBJECTS:
        try:
            data = loader.load_db2_subject(subject_id, EXERCISES)
        except Exception as exc:
            print(f"S{subject_id:02d} skipped: {exc}", flush=True)
            continue
        emg = np.asarray(data["emg"], dtype=np.float64)
        labels = np.asarray(data.get("restimulus", data.get("stimulus"))).reshape(-1)
        reps = np.asarray(data["repetition"]).reshape(-1)
        n = min(len(emg), len(labels), len(reps))
        emg, labels, reps = emg[:n], labels[:n], reps[:n]
        active = labels > 0
        centered = emg - emg.mean(axis=0, keepdims=True)
        probs, rms, acts = [], [], []
        for start, end in _runs(active):
            for cursor in range(start, end - 2000 + 1, 2000):
                if not np.all(reps[cursor:cursor + 2000] == reps[cursor]):
                    continue
                sec = centered[cursor:cursor + 2000]
                probs.append(mqp_probability(sec))
                rms.append(np.sqrt((sec ** 2).mean(axis=0)))
                acts.append(int(labels[cursor + 1000]))
        if not probs:
            continue
        P, R, A = np.stack(probs), np.stack(rms), np.asarray(acts)
        np.savez_compressed(out / f"S{subject_id:02d}_mqp_meta.npz",
                            probabilities=P, rms=R, action=A, repetition=np.zeros(len(A), dtype=np.int32))
        all_flags.append(P > THRESHOLD)
        all_rms.append(R)
        all_actions.append(A)
        all_channels.extend([subject_id] * len(P))
        print(f"S{subject_id:02d}: {len(P)} seconds", flush=True)

    F = np.concatenate(all_flags)          # (n_sec, 12) bool
    R = np.concatenate(all_rms)
    A = np.concatenate(all_actions)

    ch_rate = F.mean(axis=0)
    per_action = collections.defaultdict(lambda: [0, 0])
    for i, act in enumerate(A):
        per_action[int(act)][0] += int(F[i].sum())
        per_action[int(act)][1] += 12
    act_rate = {a: v[0] / v[1] for a, v in sorted(per_action.items())}
    top5 = sorted(act_rate.items(), key=lambda kv: -kv[1])[:5]

    flagged_rms = R[F]
    unflagged_rms = R[~F]
    per_ch = {
        f"ch{c + 1}": {
            "flag_rate": f"{ch_rate[c]:.2%}",
            "median_rms_flagged": float(np.median(R[:, c][F[:, c]])) if F[:, c].any() else None,
            "median_rms_unflagged": float(np.median(R[:, c][~F[:, c]])) if (~F[:, c]).any() else None,
            "median_rms_all": float(np.median(R[:, c])),
        } for c in range(12)
    }
    corr = float(np.corrcoef(ch_rate, np.median(R, axis=0))[0, 1])
    n_per_sec = F.sum(axis=1)
    hist = collections.Counter(int(v) for v in n_per_sec)
    flagged_secs = int((n_per_sec >= 1).sum())
    multi = int((n_per_sec >= 2).sum())

    report = {
        "protocol": {
            "purpose": "healthy-side MQP flag decomposition with activation metadata",
            "subjects": SUBJECTS, "exercises": EXERCISES, "threshold": THRESHOLD,
            "test_subjects_touched": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "n_channel_seconds": int(F.size),
        "per_channel": per_ch,
        "per_channel_flag_rate": [f"{v:.2%}" for v in ch_rate],
        "per_channel_spread": f"min={ch_rate.min():.2%} max={ch_rate.max():.2%} "
                              f"ratio={ch_rate.max() / max(ch_rate.min(), 1e-9):.1f}x",
        "flag_rate_vs_median_rms_pearson": corr,
        "global_rms_contrast": {
            "median_rms_flagged": float(np.median(flagged_rms)),
            "median_rms_unflagged": float(np.median(unflagged_rms)),
            "ratio": float(np.median(flagged_rms) / max(np.median(unflagged_rms), 1e-12)),
        },
        "per_action_flag_rate_top5": [(a, f"{v:.2%}") for a, v in top5],
        "per_action_flag_rate_range": (f"{min(act_rate.values()):.2%}", f"{max(act_rate.values()):.2%}"),
        "flagged_channels_per_second_hist": {str(k): v for k, v in sorted(hist.items())},
        "multi_channel_flag_seconds": f"{multi}/{flagged_secs} ({multi / max(1, flagged_secs):.1%})",
        "elapsed_seconds": time.monotonic() - started,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("protocol",)}, indent=2)[:1800])
    print(f"results={out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
