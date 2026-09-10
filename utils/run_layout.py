"""MCIA 全流程实验的统一输出目录布局。"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict


RUN_ENV = "MCIA_RUN_DIR"


def _resolve_output_root(project_root: Path, cfg: Dict) -> Path:
    output_root = Path(cfg["paths"]["output"])
    return output_root if output_root.is_absolute() else project_root / output_root


def _next_run_dir(output_root: Path) -> Path:
    base = output_root / "run"
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 1
    while True:
        path = base / f"run_{stamp}_{counter}"
        if not path.exists():
            return path
        counter += 1


def get_run_dir(project_root: Path, cfg: Dict, create: bool = True, allow_new: bool = True) -> Path:
    env_value = os.environ.get(RUN_ENV)
    if env_value:
        run_dir = Path(env_value)
    else:
        if not allow_new:
            raise RuntimeError(
                f"{RUN_ENV} is not set. Run scripts/run_all_experiments.py to create a run, "
                f"or set {RUN_ENV} to an existing run directory before running a single step."
            )
        run_dir = _next_run_dir(_resolve_output_root(project_root, cfg))
        os.environ[RUN_ENV] = str(run_dir)
    if create:
        initialize_run_dir(project_root, cfg, run_dir)
    return run_dir


def initialize_run_dir(project_root: Path, cfg: Dict, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "00_config",
        "01_db2_completion/checkpoints",
        "01_db2_completion/metrics",
        "01_db2_completion/figures",
        "01_db2_completion/cache",
        "02_db3_transfer_completion/checkpoints",
        "02_db3_transfer_completion/augmented_emg",
        "02_db3_transfer_completion/figures/12ch_completion",
        "03_angle_prediction/checkpoints",
        "03_angle_prediction/logs",
        "03_angle_prediction/metrics",
        "03_angle_prediction/predictions",
        "03_angle_prediction/figures/comparison",
        "04_gesture_recognition/checkpoints",
        "04_gesture_recognition/metrics",
        "04_gesture_recognition/predictions",
        "04_gesture_recognition/figures",
        "04_gesture_recognition/logs",
        "05_logs",
    ]:
        (run_dir / rel).mkdir(parents=True, exist_ok=True)

    config_src = project_root / "config.yaml"
    config_dst = run_dir / "00_config" / "config_snapshot.yaml"
    if config_src.exists() and not config_dst.exists():
        shutil.copy2(config_src, config_dst)

    env_path = run_dir / "00_config" / "environment.json"
    if not env_path.exists():
        env_path.write_text(
            json.dumps(
                {
                    "python": sys.executable,
                    "cwd": str(project_root),
                    "run_dir": str(run_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = run_dir / "00_config" / "run_manifest.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "steps": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def apply_run_paths(flat: Dict, cfg: Dict, project_root: Path, allow_new: bool = False) -> Dict:
    run_dir = get_run_dir(project_root, cfg, create=True, allow_new=allow_new)
    flat["run_dir"] = str(run_dir)

    exp1 = run_dir / "01_db2_completion"
    exp2 = run_dir / "02_db3_transfer_completion"
    exp3 = run_dir / "03_angle_prediction"

    flat["exp1_dir"] = str(exp1)
    flat["transfer_checkpoints_dir"] = str(exp2 / "checkpoints")
    flat["transfer_augmented_data_dir"] = str(exp2 / "augmented_emg")
    flat["regressor_augmented_data_dir"] = str(exp2 / "augmented_emg")
    flat["regressor_results_path"] = str(exp3 / "metrics" / "db3_angle_raw_vs_augmented_results.json")
    flat["gesture_results_path"] = str(run_dir / "04_gesture_recognition" / "metrics" / "db3_gesture_raw_vs_augmented_results.json")
    return flat


def mark_step(run_dir: Path, step: str, status: str, extra: Dict | None = None) -> None:
    manifest = run_dir / "00_config" / "run_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    data.setdefault("steps", {})[step] = {
        "status": status,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
    }
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")




