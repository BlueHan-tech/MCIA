"""在当前 run 内准备 SGMD-AAE 论文格式的 DB2 输入。

这里不复用实验一的 200 Hz 包络缓存：SGMD-AAE 论文使用经过 49.5--50.5 Hz
带阻滤波的 2 kHz、240 x 12 原始信号窗口。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import scipy.io as sio
from scipy import signal
import yaml


def _paper_notch(emg: np.ndarray, sample_rate: int) -> np.ndarray:
    """三阶 Butterworth 49.5--50.5 Hz 零相位带阻滤波。"""
    sos = signal.butter(3, [49.5, 50.5], btype="bandstop", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, emg, axis=0).astype(np.float32, copy=False)


def _normalize_each_sample(windows: np.ndarray) -> np.ndarray:
    """按照 SGMD-AAE 的说明，将每个 240 x 12 样本映射到 [0,1]。"""
    values = np.asarray(windows, dtype=np.float32)
    lower = values.min(axis=(1, 2), keepdims=True)
    upper = values.max(axis=(1, 2), keepdims=True)
    return (values - lower) / np.maximum(upper - lower, np.finfo(np.float32).eps)


def _windows_from_subject(db2_root: Path, subject_id: int, exercises: list[int],
                          window_size: int, limit: int) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    subject_dir = db2_root / f"DB2_s{subject_id}"
    for exercise in exercises:
        mat_path = subject_dir / f"S{subject_id}_E{exercise}_A1.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"缺少 DB2 记录：{mat_path}")
        raw = np.asarray(sio.loadmat(mat_path)["emg"], dtype=np.float32)
        filtered = _paper_notch(raw, sample_rate=2000)
        usable = (len(filtered) // window_size) * window_size
        windows.extend(filtered[:usable].reshape(-1, window_size, raw.shape[1]))
        if limit > 0 and len(windows) >= limit:
            return windows[:limit]
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, required=True)
    args = parser.parse_args()
    run_dir = os.environ.get("MCIA_RUN_DIR")
    if not run_dir:
        raise RuntimeError("该单步准备过程必须设置 MCIA_RUN_DIR。")
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    baseline_cfg = cfg["literature_baselines"]
    windows: list[np.ndarray] = []
    for subject_id in baseline_cfg["db2_subjects"]:
        remaining = args.max_windows - len(windows) if args.max_windows > 0 else 0
        windows.extend(_windows_from_subject(
            Path(cfg["paths"]["db2"]), int(subject_id),
            [int(value) for value in baseline_cfg["exercises"]], 240, remaining,
        ))
        if args.max_windows > 0 and len(windows) >= args.max_windows:
            break
    if not windows:
        raise RuntimeError("未准备出论文格式的 DB2 窗口。")
    values = _normalize_each_sample(np.stack(windows))
    out_dir = Path(run_dir) / "06_diagnostics" / "literature_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "db2_sgmd_paper_format_windows.npy"
    np.save(out_path, values)
    (out_dir / "input_preprocessing.json").write_text(json.dumps({
        "source": "DB2 raw 2-kHz sEMG",
        "filter": "third-order Butterworth band-stop 49.5-50.5 Hz, zero-phase",
        "window": "240 x 12, non-overlapping",
        "normalization": "per-sample min-max to [0, 1]",
    }, indent=2), encoding="utf-8")
    print(f"已准备形状为 {values.shape} 的归一化 DB2 窗口：{out_path}", flush=True)


if __name__ == "__main__":
    main()
