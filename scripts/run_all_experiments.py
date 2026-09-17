"""
MCIA 全流程实验的一键入口。

直接运行本文件将按顺序执行：
Exp1：DB2 健康先验补全
Exp3：DB3 连续关节角度预测（A/B）
Exp4：DB3 48 类手势识别（A/B）

可选的文献基线阶段仅在默认流水线成功完成后运行。它是任务匹配的 200 Hz 比较，
与独立的论文格式复现入口不同。
"""

from __future__ import annotations

import os
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.run_layout import get_run_dir, mark_step


STEPS = [
    {
        "id": "exp1",
        "name": "Exp1: DB2 healthy-prior completion",
        "command": ["scripts/01_train_mcia_db2_healthy_prior.py"],
        "log_name": "01_exp1_db2_completion.log",
    },
    {
        "id": "exp3_angle",
        "name": "Exp3: E1+E2 fixed-Key10 continuous angle prediction",
        "command": ["scripts/04_eval_db3_angle_raw_vs_augmented.py", "--groups", "A,B"],
        "log_name": "04_exp3_angle_prediction.log",
    },
    {
        "id": "exp4_gesture",
        "name": "Exp4: E1+E2+E3 48-class gesture recognition (A/B)",
        "command": ["scripts/05_eval_db3_gesture_raw_vs_augmented.py"],
        "log_name": "05_exp4_gesture_recognition.log",
    },
]


def _literature_baseline_steps(cfg: dict) -> list[dict]:
    settings = cfg["literature_baselines"]
    return [
        {
            "id": "task_matched_literature_baselines",
            "name": "Task-matched 200-Hz MCIA / SGMD-AAE / CP-WOPT + Key10 TCN",
            "command": ["scripts/run_task_matched_literature_baselines.py"],
            "log_name": "06_task_matched_literature_baselines.log",
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
    print(f"运行目录：{run_dir}", flush=True)
    print("", flush=True)
    print("实验一输出：", flush=True)
    print(f"  {run_dir / '01_db2_completion'}", flush=True)
    print("    checkpoints/", flush=True)
    print("    metrics/", flush=True)
    print("    figures/", flush=True)
    print("    cache/", flush=True)
    print("", flush=True)
    print("实验三固定 Key10 连续角度预测输出：", flush=True)
    print(f"  {run_dir / '03_angle_prediction'}", flush=True)
    print("    metrics/db3_angle_raw_vs_augmented_results.json", flush=True)
    print("    predictions/Sxx_angle_predictions.npz", flush=True)
    print("", flush=True)
    print("实验四 A/B 手势识别输出：", flush=True)
    print(f"  {run_dir / '04_gesture_recognition'}", flush=True)
    print("    metrics/, predictions/, figures/, checkpoints/", flush=True)
    print("", flush=True)
    print("可选的任务匹配文献基线输出：", flush=True)
    print(f"  {run_dir / '06_diagnostics' / 'task_matched_literature_baselines'}", flush=True)
    print("    相同的 200 Hz 包络/掩码，以及 MCIA/SGMD-AAE/CP-WOPT 补全和 Key10 TCN 结果", flush=True)
    print("", flush=True)
    print("默认流水线只生成 A/B 下游结果。", flush=True)
    print("C 组有独立的供体适配开发入口，不会在此处隐式运行。", flush=True)


def _run_step(index: int, total: int, step: dict, run_dir: Path) -> None:
    name = step["name"]
    step_id = step["id"]
    literature_input = run_dir / "06_diagnostics" / "literature_baselines" / "db2_sgmd_paper_format_windows.npy"
    command = [str(literature_input) if item == "{literature_input}" else item for item in step["command"]]
    full_cmd = _python_command(command)
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
    parser = argparse.ArgumentParser(description="Run the MCIA pipeline.")
    parser.add_argument("--with-literature-baselines", action="store_true",
                        help="After Exp4, run independent SGMD-AAE and CP-WOPT reproductions.")
    args = parser.parse_args()
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    include_literature = args.with_literature_baselines or bool(cfg["literature_baselines"].get("enabled", False))
    steps = [*STEPS, *(_literature_baseline_steps(cfg) if include_literature else [])]
    run_dir = get_run_dir(PROJECT_ROOT, cfg, create=True)
    os.environ["MCIA_RUN_DIR"] = str(run_dir)

    _print_header("MCIA full experiment pipeline")
    print(f"项目根目录：{PROJECT_ROOT}", flush=True)
    print(f"运行目录：{run_dir}", flush=True)
    print(f"Python：{sys.executable}", flush=True)
    print("直接运行本文件以执行固定的完整流水线。", flush=True)
    print(f"文献基线：{'已启用' if include_literature else '已禁用'}", flush=True)
    _print_expected_outputs(run_dir)

    total_start_time = _now()
    total_start = time.time()
    mark_step(run_dir, "pipeline", "running", {"start_time": total_start_time})
    try:
        for index, step in enumerate(steps, start=1):
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
