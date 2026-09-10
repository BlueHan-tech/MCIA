"""PyCharm-friendly standalone pipeline for the two literature baselines.

Running this file creates a new isolated run, prepares the paper-format DB2
input, and then runs SGMD-AAE followed by CP-WOPT.  It never runs MCIA Exp1,
Exp3, or Exp4, and it does not reuse or modify another run directory.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.run_layout import get_run_dir, mark_step


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _child_environment(run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MCIA_RUN_DIR"] = str(run_dir)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")
    python_prefix = Path(sys.executable).resolve().parent
    env_prefix_paths = [
        python_prefix,
        python_prefix / "Library" / "mingw-w64" / "bin",
        python_prefix / "Library" / "usr" / "bin",
        python_prefix / "Library" / "bin",
        python_prefix / "Scripts",
        python_prefix / "bin",
    ]
    env["PATH"] = os.pathsep.join(
        [str(path) for path in env_prefix_paths if path.exists()] + [env.get("PATH", "")]
    )
    return env


def _run_step(run_dir: Path, step_id: str, name: str, command: list[str], log_name: str) -> None:
    log_path = run_dir / "05_logs" / log_name
    full_command = [sys.executable, "-u", *command]
    command_text = " ".join(full_command)
    start_time = _now()
    start = time.time()
    mark_step(run_dir, step_id, "running", {
        "name": name,
        "command": command_text,
        "log_path": str(log_path),
        "start_time": start_time,
    })
    print(f"\n{'=' * 88}\n{name}\nCommand: {command_text}\nLog: {log_path}\n{'=' * 88}", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            full_command,
            cwd=PROJECT_ROOT,
            env=_child_environment(run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    elapsed_seconds = time.time() - start
    status = "completed" if return_code == 0 else "failed"
    mark_step(run_dir, step_id, status, {
        "name": name,
        "command": command_text,
        "log_path": str(log_path),
        "start_time": start_time,
        "end_time": _now(),
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
    })
    if return_code != 0:
        raise RuntimeError(f"{name} failed (return code {return_code}); see {log_path}")
    print(f"Finished: {name} ({elapsed_seconds:.1f} sec)", flush=True)


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    settings = cfg["literature_baselines"]
    run_dir = get_run_dir(PROJECT_ROOT, cfg, create=True)
    input_path = run_dir / "06_diagnostics" / "literature_baselines" / "db2_sgmd_paper_format_windows.npy"
    os.environ["MCIA_RUN_DIR"] = str(run_dir)
    print(f"Literature-baseline run: {run_dir}", flush=True)
    print("This run contains only SGMD-AAE and CP-WOPT; MCIA main experiments are not run.", flush=True)
    mark_step(run_dir, "literature_pipeline", "running", {"start_time": _now()})
    try:
        _run_step(
            run_dir,
            "literature_prepare_db2",
            "Prepare SGMD-AAE-format DB2 windows",
            ["scripts/prepare_literature_baseline_db2.py", "--max-windows", str(settings["max_windows"])],
            "06_literature_prepare_db2.log",
        )
        common = [
            "scripts/run_literature_completion_baseline.py",
            "--input", str(input_path),
            "--missing-ratio", str(settings["missing_ratio"]),
            "--seed", str(settings["seed"]),
            "--max-samples", str(settings["max_windows"]),
        ]
        _run_step(
            run_dir,
            "literature_sgmd_aae",
            "SGMD-AAE literature baseline",
            [*common, "--method", "sgmd_aae", "--epochs", str(settings["sgmd_aae_epochs"]),
             "--batch-size", str(settings["sgmd_aae_batch_size"])],
            "06_literature_sgmd_aae.log",
        )
        _run_step(
            run_dir,
            "literature_cp_wopt",
            "CP-WOPT literature baseline",
            [*common, "--method", "cp_wopt", "--rank", str(settings["cp_wopt_rank"])],
            "06_literature_cp_wopt.log",
        )
    except Exception:
        mark_step(run_dir, "literature_pipeline", "failed", {"end_time": _now()})
        raise
    mark_step(run_dir, "literature_pipeline", "completed", {"end_time": _now()})
    print(f"\nLiterature baselines completed: {run_dir / '06_diagnostics' / 'literature_baselines'}", flush=True)


if __name__ == "__main__":
    main()
