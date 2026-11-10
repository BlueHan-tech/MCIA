"""Run the 2 kHz prior, donor adaptation, and paired DB3 A/B/C tasks."""

from __future__ import annotations

import os
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.run_layout import mark_step


STEPS = [
    {
        "id": "exp1",
        "name": "Exp1: DB2 healthy-prior completion",
        "command": ["scripts/01_train_mcia_db2_healthy_prior.py"],
        "log_name": "01_exp1_db2_completion.log",
        "artifact": "01_db2_completion/checkpoints/waveform_best.pt",
    },
    {
        "id": "exp2_donor",
        "name": "Exp2: target-excluded DB3 donor adaptation",
        "command": ["scripts/06_adapt_amputee_donor_prior.py"],
        "log_name": "02_exp2_donor_prior.log",
        "artifact": "00_config/waveform_donor_mapping.json",
    },
    {
        "id": "exp3_angle",
        "name": "Exp3: E1+E2 fixed-Key10 continuous angle prediction",
        "command": ["scripts/04_eval_db3_angle_raw_vs_augmented.py"],
        "log_name": "04_exp3_angle_prediction.log",
        "artifact": "03_angle_prediction/metrics/waveform_development.json",
    },
    {
        "id": "exp4_gesture",
        "name": "Exp4: E1+E2+E3 49-class gesture recognition (A/B/C)",
        "command": ["scripts/05_eval_db3_gesture_raw_vs_augmented.py"],
        "log_name": "05_exp4_gesture_recognition.log",
        "artifact": "04_gesture_recognition/metrics/waveform_development.json",
    },
]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _print_header(title: str) -> None:
    line = "=" * 88
    print("\n" + line, flush=True)
    print(title, flush=True)
    print(line, flush=True)


def _python_command(command: list[str]) -> list[str]:
    """让所有阶段都使用启动流水线时的环境。"""
    return [sys.executable, "-u", *command]


def _format_command(command: list[str]) -> str:
    return " ".join(command)


def _print_expected_outputs(run_dir: Path) -> None:
    _print_header("Expected outputs for this run")
    for folder in ('01_db2_completion', '02_db3_transfer_completion', '03_angle_prediction',
                   '04_gesture_recognition', '05_logs'):
        print(f'  {run_dir / folder}', flush=True)


def _run_step(index: int, total: int, step: dict, run_dir: Path) -> None:
    name = step["name"]
    step_id = step["id"]
    full_cmd = _python_command(step['command'])
    command_text = _format_command(full_cmd)
    log_path = run_dir / "05_logs" / step["log_name"]

    _print_header(f"[{index}/{total}] {name}")
    print(f"命令：{command_text}", flush=True)
    print(f"日志：{log_path}", flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONUNBUFFERED"] = "1"
    env["MCIA_RUN_DIR"] = str(run_dir)
    python_prefix = Path(full_cmd[0]).resolve().parent
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

    start_time = _now()
    start = time.time()
    running_meta = {
        "name": name,
        "status": "running",
        "command": command_text,
        "log_path": str(log_path),
        "start_time": start_time,
    }
    mark_step(run_dir, step_id, "running", running_meta)

    return_code = 1
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
        log_f.write(f"Step: {name}\n")
        log_f.write(f"Command: {command_text}\n")
        log_f.write(f"Run dir: {run_dir}\n")
        log_f.write(f"Start time: {start_time}\n")
        log_f.write("\n")
        log_f.flush()

        process = subprocess.Popen(
            full_cmd,
            cwd=PROJECT_ROOT,
            env=env,
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
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

        end_time = _now()
        elapsed_seconds = time.time() - start
        status = "completed" if return_code == 0 else "failed"
        log_f.write("\n")
        log_f.write(f"End time: {end_time}\n")
        log_f.write(f"Elapsed seconds: {elapsed_seconds:.1f}\n")
        log_f.write(f"Return code: {return_code}\n")
        log_f.write(f"Status: {status}\n")
        log_f.flush()

    final_meta = {
        "name": name,
        "status": status,
        "command": command_text,
        "log_path": str(log_path),
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
    }
    mark_step(run_dir, step_id, status, final_meta)

    if return_code != 0:
        print("", flush=True)
        print(f"步骤失败：{name}", flush=True)
        print(f"返回码：{return_code}", flush=True)
        print(f"日志：{log_path}", flush=True)
        raise RuntimeError(f"步骤失败：{name}")

    print(f"\n已完成：{name}（{elapsed_seconds:.1f} 秒）| 日志：{log_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the 2 kHz MCIA development pipeline')
    parser.add_argument('--smoke', action='store_true', help='Use bounded development records and training budgets')
    args = parser.parse_args()
    from utils.windows_conda_path import ensure_current_env_dll_path
    ensure_current_env_dll_path()
    from utils.waveform_pipeline import setup_waveform
    _, run_dir, _, _ = setup_waveform(PROJECT_ROOT, smoke=args.smoke)
    steps = [dict(step, command=step['command'] + (['--smoke'] if args.smoke else [])) for step in STEPS]
    manifest = json.loads((run_dir / '00_config' / 'run_manifest.json').read_text(encoding='utf-8'))

    _print_header("MCIA full experiment pipeline")
    print(f"项目根目录：{PROJECT_ROOT}", flush=True)
    print(f"运行目录：{run_dir}", flush=True)
    print(f"Python：{sys.executable}", flush=True)
    _print_expected_outputs(run_dir)

    total_start_time = _now()
    total_start = time.time()
    mark_step(run_dir, "pipeline", "running", {"start_time": total_start_time})
    try:
        for index, step in enumerate(steps, start=1):
            if (manifest.get('steps', {}).get(step['id'], {}).get('status') == 'completed'
                    and (run_dir / step['artifact']).is_file()):
                print(f"Reusing completed step: {step['id']}", flush=True)
                continue
            _run_step(index, len(steps), step, run_dir)
    except Exception:
        total_elapsed = time.time() - total_start
        mark_step(
            run_dir,
            "pipeline",
            "failed",
            {
                "start_time": total_start_time,
                "end_time": _now(),
                "elapsed_seconds": total_elapsed,
            },
        )
        raise

    total_elapsed = time.time() - total_start
    mark_step(
        run_dir,
        "pipeline",
        "completed",
        {
            "start_time": total_start_time,
            "end_time": _now(),
            "elapsed_seconds": total_elapsed,
        },
    )
    _print_header("MCIA pipeline finished")
    print(f"运行目录：{run_dir}", flush=True)
    print(f"总用时：{total_elapsed:.1f} 秒", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _print_header("Pipeline stopped")
        print(exc, flush=True)
        raise
