"""Decompose DB3 MQP flags by channel x action from existing NPZ artifacts.

Read-only analysis of the frozen Gronlund reproduction NPZs (train reps
1/3/4, E1-E3, 9 subjects). At the current threshold 0.20 it reports:
  - per-exercise per-channel flag rates (12 channels);
  - per-exercise per-action flag rates with concentration stats;
  - the full channel x action flag-rate matrices (JSON);
  - distribution of #flagged channels per flagged second (isolated-electrode
    signature vs multi-channel movement signature).

Writes only to <run>/06_diagnostics/.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

THRESHOLD = 0.20


def main() -> None:
    run = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    src = run.parent / "run_20260907_154902_1" / "06_diagnostics" / "gronlund_2005_mqp_db3"
    if not src.is_dir():
        src = Path(os.environ.get("MCIA_MQP_DIR", src))
    out = run / "06_diagnostics" / "mqp_flag_decomposition_db3_20260909"
    out.mkdir(parents=True, exist_ok=False)

    report = {"threshold": THRESHOLD, "source": str(src), "exercises": {}}
    for exercise in (1, 2, 3):
        per_channel_flags = np.zeros(12)
        per_channel_total = np.zeros(12)
        action_flags = collections.Counter()
        action_total = collections.Counter()
        nflag_dist = collections.Counter()
        matrix = collections.defaultdict(lambda: [0, 0])   # (ch, act) -> [flags, total]
        for path in sorted(src.glob(f"S*_E{exercise}_mqp.npz")):
            data = np.load(path)
            p, actions = data["probabilities"], data["action"]
            flagged = p > THRESHOLD                      # (n_sec, 12)
            per_channel_flags += flagged.sum(axis=0)
            per_channel_total += len(p)
            n_per_sec = flagged.sum(axis=1)
            for k in range(13):
                if (n_per_sec == k).any():
                    nflag_dist[k] += int((n_per_sec == k).sum())
            for i, act in enumerate(actions):
                act = int(act)
                action_flags[act] += int(flagged[i].sum())
                action_total[act] += 12
                for ch in range(12):
                    matrix[(ch, act)][0] += int(flagged[i, ch])
                    matrix[(ch, act)][1] += 1

        ch_rate = per_channel_flags / np.maximum(per_channel_total, 1)
        act_rate = {a: action_flags[a] / action_total[a] for a in sorted(action_total)}
        top5 = sorted(act_rate.items(), key=lambda kv: -kv[1])[:5]
        top5_share = sum(action_flags[a] for a, _ in top5) / max(1, sum(action_flags.values()))
        report["exercises"][f"E{exercise}"] = {
            "n_channel_seconds": int(per_channel_total[0]),
            "per_channel_flag_rate": [f"{v:.2%}" for v in ch_rate],
            "per_channel_spread": f"min={ch_rate.min():.2%} max={ch_rate.max():.2%} "
                                  f"ratio={ch_rate.max() / max(ch_rate.min(), 1e-9):.1f}x",
            "per_action_flag_rate": {str(a): f"{v:.2%}" for a, v in act_rate.items()},
            "top5_actions": [(a, f"{v:.2%}") for a, v in top5],
            "top5_actions_flag_share": f"{top5_share:.2%}",
            "flagged_channels_per_second_hist": {str(k): v for k, v in sorted(nflag_dist.items())},
            "matrix_channel_action": {
                f"ch{ch + 1}_a{act}": {"flags": v[0], "total": v[1], "rate": f"{v[0] / max(v[1], 1):.2%}"}
                for (ch, act), v in sorted(matrix.items())
            },
        }
        print(f"E{exercise}: ch spread {report['exercises'][f'E{exercise}']['per_channel_spread']}")
        print(f"  per-channel: {report['exercises'][f'E{exercise}']['per_channel_flag_rate']}")
        print(f"  top5 actions {top5} share={top5_share:.2%}")
        hist = report["exercises"][f"E{exercise}"]["flagged_channels_per_second_hist"]
        flagged_secs = sum(int(v) for k, v in hist.items() if int(k) >= 1)
        multi = sum(int(v) for k, v in hist.items() if int(k) >= 2)
        print(f"  flagged seconds={flagged_secs}, with>=2 channels={multi} "
              f"({multi / max(1, flagged_secs):.1%})")

    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"results={out / 'results.json'}")


if __name__ == "__main__":
    main()
