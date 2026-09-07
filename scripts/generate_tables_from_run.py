"""Regenerate paper tables from an existing run.

Current scope:
- Table V: DB3 controlled artificial-masking completion performance for
  direct_transfer, amputee_only, and pretrained_finetuned checkpoints.
- Table VI: DB3 continuous joint-angle prediction performance reconstructed
  from existing A/B/C angle metrics.

This script is intentionally independent from generate_figures_from_run.py and
legacy generate_paper_figures.py. It never trains models, rewrites checkpoints,
updates caches, or modifies augmented EMG files. Some tables may run
evaluation-only inference from existing checkpoints when the required metrics
were not saved by the original experiment step.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from pathlib import Path
from statistics import stdev
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.kinematic_target import key10_target_metadata

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
TABLE_V_OUTPUTS = (
    "table_v_db3_transfer_completion.csv",
    "table_v_db3_transfer_completion.md",
    "table_v_db3_transfer_completion.html",
)
TABLE_VI_INPUT = Path("03_angle_prediction") / "metrics" / "db3_angle_raw_vs_augmented_results.json"
TABLE_VI_OUTPUTS = (
    "table_vi_angle_prediction.csv",
    "table_vi_angle_prediction.md",
    "table_vi_angle_prediction.html",
)

GROUP_LABELS = {
    "A": "A: raw sEMG",
    "B": "B: healthy-prior MCIA enhanced sEMG",
    "C": "C: subject-finetuned MCIA enhanced sEMG",
}
SUBSET_ORDER = ("global", "mcp", "pip")
METRIC_ORDER = ("rmse", "mae", "pearson", "r2")

TABLE_VI_GROUP_LABELS = {
    "A": "Raw sEMG",
    "B": "Direct-transfer enhanced",
    "C": "Pretrained-finetuned enhanced",
}
TABLE_VI_COLUMNS = (
    ("global_rmse", "global", "rmse", "Global RMSE"),
    ("global_mae", "global", "mae", "Global MAE"),
    ("global_cc", "global", "pearson", "Global CC"),
    ("global_r2", "global", "r2", "Global R2"),
    ("mcp_cc", "mcp", "pearson", "MCP CC"),
    ("pip_cc", "pip", "pearson", "PIP/IP CC"),
)


def resolve_run_dir(run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).expanduser()
    elif os.environ.get("MCIA_RUN_DIR"):
        run_dir = Path(os.environ["MCIA_RUN_DIR"]).expanduser()
    else:
        run_root = DEFAULT_OUTPUT_ROOT / "run"
        candidates = sorted(
            [p for p in run_root.glob("run_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No run directory found under {run_root}. Pass --run-dir or set MCIA_RUN_DIR."
            )
        run_dir = candidates[-1]

    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    return run_dir


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _safe_values(values: list[object]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            cleaned.append(float(value))
    return cleaned


def _format_mean_sd(mean_value: float | None, sd_value: float | None) -> str:
    if mean_value is None or sd_value is None:
        return "N/A"
    return f"{mean_value:.4f} +/- {sd_value:.4f}"


def _compute_angle_table_rows(metrics: dict) -> list[dict[str, object]]:
    subjects = metrics.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("Table V input has no non-empty 'subjects' list.")

    rows: list[dict[str, object]] = []
    for group in ("A", "B", "C"):
        for subset in SUBSET_ORDER:
            subject_values: dict[str, list[float]] = {metric: [] for metric in METRIC_ORDER}
            subject_count = 0
            for subject in subjects:
                if subject.get("status") != "ok":
                    continue
                group_data = subject.get("groups", {}).get(group)
                if not isinstance(group_data, dict):
                    continue
                subset_data = group_data.get("subsets", {}).get(subset)
                if not isinstance(subset_data, dict):
                    continue
                subject_count += 1
                for metric in METRIC_ORDER:
                    value = subset_data.get(metric)
                    if isinstance(value, (int, float)) and not math.isnan(float(value)):
                        subject_values[metric].append(float(value))

            if subject_count == 0:
                raise ValueError(f"No subject-level values found for group={group}, subset={subset}.")

            row: dict[str, object] = {
                "group": group,
                "configuration": GROUP_LABELS.get(group, group),
                "subset": subset,
                "n_subjects": subject_count,
            }
            for metric in METRIC_ORDER:
                values = _safe_values(subject_values[metric])
                mean_value = _mean(values) if values else None
                sd_value = stdev(values) if len(values) > 1 else (0.0 if values else None)
                row[f"{metric}_mean"] = mean_value
                row[f"{metric}_sd"] = sd_value
                row[f"{metric}_n"] = len(values)
                row[metric] = _format_mean_sd(mean_value, sd_value)
            rows.append(row)
    return rows


def _ensure_can_write(outputs: list[Path], force: bool) -> None:
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing table outputs without --force:\n" + joined
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "group",
        "configuration",
        "subset",
        "n_subjects",
        "rmse",
        "mae",
        "pearson",
        "r2",
        "rmse_mean",
        "rmse_sd",
        "rmse_n",
        "mae_mean",
        "mae_sd",
        "mae_n",
        "pearson_mean",
        "pearson_sd",
        "pearson_n",
        "r2_mean",
        "r2_sd",
        "r2_n",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Table V",
        "",
        "Angle prediction performance reconstructed from existing run metrics.",
        "Values are subject-level mean +/- SD.",
        "",
        "| Configuration | Subset | n | RMSE | MAE | Pearson | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {configuration} | {subset} | {n_subjects} | {rmse} | {mae} | {pearson} | {r2} |".format(
                **row
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>Table V</title>",
        "  <style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:6px 10px;}th{background:#f3f3f3;}td.num{text-align:right;}</style>",
        "</head>",
        "<body>",
        "<h1>Table V</h1>",
        "<p>Angle prediction performance reconstructed from existing run metrics. Values are subject-level mean +/- SD.</p>",
        "<table>",
        "<thead><tr><th>Configuration</th><th>Subset</th><th>n</th><th>RMSE</th><th>MAE</th><th>Pearson</th><th>R2</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['configuration']))}</td>"
            f"<td>{html.escape(str(row['subset']))}</td>"
            f"<td class=\"num\">{row['n_subjects']}</td>"
            f"<td class=\"num\">{html.escape(str(row['rmse']))}</td>"
            f"<td class=\"num\">{html.escape(str(row['mae']))}</td>"
            f"<td class=\"num\">{html.escape(str(row['pearson']))}</td>"
            f"<td class=\"num\">{html.escape(str(row['r2']))}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>", "</body>", "</html>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_legacy_angle_table(run_dir: Path, force: bool = False) -> list[Path]:
    input_path = run_dir / (Path("03_angle_prediction") / "metrics" / "db3_angle_raw_vs_augmented_results.json")
    if not input_path.exists():
        raise FileNotFoundError(
            "Missing legacy angle table input metrics file:\n"
            f"  {input_path}\n"
            "Run scripts/04_eval_db3_angle_raw_vs_augmented.py first or choose another --run-dir."
        )

    metrics = json.loads(input_path.read_text(encoding="utf-8"))
    rows = _compute_angle_table_rows(metrics)

    output_dir = run_dir / "tables"
    outputs = [output_dir / name for name in ("legacy_angle_table.csv", "legacy_angle_table.md", "legacy_angle_table.html")]
    _ensure_can_write(outputs, force=force)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(outputs[0], rows)
    _write_md(outputs[1], rows)
    _write_html(outputs[2], rows)
    return outputs



def _compute_table_vi_rows(metrics: dict) -> list[dict[str, object]]:
    subjects = metrics.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("Table VI input has no non-empty 'subjects' list.")

    rows: list[dict[str, object]] = []
    for group in ("A", "B", "C"):
        row: dict[str, object] = {
            "group": group,
            "configuration": TABLE_VI_GROUP_LABELS[group],
        }
        subject_count = 0
        values_by_column: dict[str, list[float]] = {col_id: [] for col_id, _, _, _ in TABLE_VI_COLUMNS}
        for subject in subjects:
            if subject.get("status") != "ok":
                continue
            group_data = subject.get("groups", {}).get(group)
            if not isinstance(group_data, dict):
                continue
            subsets = group_data.get("subsets", {})
            if not isinstance(subsets, dict):
                continue
            subject_count += 1
            for col_id, subset, metric, _ in TABLE_VI_COLUMNS:
                subset_data = subsets.get(subset)
                if not isinstance(subset_data, dict):
                    continue
                value = subset_data.get(metric)
                if isinstance(value, (int, float)) and not math.isnan(float(value)):
                    values_by_column[col_id].append(float(value))
        if subject_count == 0:
            raise ValueError(f"No subject-level values found for Table VI group={group}.")
        row["n_subjects"] = subject_count
        for col_id, _, _, _ in TABLE_VI_COLUMNS:
            vals = _safe_values(values_by_column[col_id])
            mean_value = _mean(vals) if vals else None
            sd_value = stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
            row[col_id] = _format_mean_sd(mean_value, sd_value)
            row[f"{col_id}_mean"] = mean_value
            row[f"{col_id}_sd"] = sd_value
            row[f"{col_id}_n"] = len(vals)
        rows.append(row)
    return rows


def _write_table_vi_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["configuration", "group", "n_subjects"]
    fieldnames.extend(col_id for col_id, _, _, _ in TABLE_VI_COLUMNS)
    for col_id, _, _, _ in TABLE_VI_COLUMNS:
        fieldnames.extend([f"{col_id}_mean", f"{col_id}_sd", f"{col_id}_n"])
    fieldnames.append("note")
    note = (
        "A/B/C correspond to Raw sEMG, Direct-transfer enhanced, and Pretrained-finetuned enhanced; "
        "global/mcp/pip are fixed Key10 evaluation subsets; PIP includes the thumb IP channel; CC is Pearson correlation; "
        "R2 may be negative and high-variance and is reported as originally evaluated."
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {name: row.get(name, "") for name in fieldnames}
            payload["note"] = note
            writer.writerow(payload)


def _table_vi_note_lines() -> list[str]:
    return [
        "Values are subject-level mean +/- SD.",
        "Raw / Direct-transfer enhanced / Pretrained-finetuned enhanced correspond to A/B/C in the saved angle metrics.",
        "global/mcp/pip are fixed Key10 evaluation subsets; PIP includes the thumb IP channel. CC denotes Pearson correlation.",
        "R2 may be negative and have large between-subject variance; values are reported from the original evaluation results.",
    ]


def _write_table_vi_md(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [label for _, _, _, label in TABLE_VI_COLUMNS]
    lines = ["# Table VI", "", "Continuous joint-angle estimation performance."]
    lines.extend(_table_vi_note_lines())
    lines.extend([
        "",
        "| Configuration | n | " + " | ".join(headers) + " |",
        "|---|---:" + "|---:" * len(headers) + "|",
    ])
    for row in rows:
        values = [str(row[col_id]) for col_id, _, _, _ in TABLE_VI_COLUMNS]
        lines.append(f"| {row['configuration']} | {row['n_subjects']} | " + " | ".join(values) + " |")
    lines.extend(["", "## Effective n", ""])
    lines.append("| Configuration | " + " | ".join(f"{label} n" for _, _, _, label in TABLE_VI_COLUMNS) + " |")
    lines.append("|---" + "|---:" * len(TABLE_VI_COLUMNS) + "|")
    for row in rows:
        ns = [str(row[f"{col_id}_n"]) for col_id, _, _, _ in TABLE_VI_COLUMNS]
        lines.append(f"| {row['configuration']} | " + " | ".join(ns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table_vi_html(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [label for _, _, _, label in TABLE_VI_COLUMNS]
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>Table VI angle prediction</title>",
        "  <style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:6px 10px;}th{background:#f3f3f3;}td.num{text-align:right;}p{max-width:980px;}</style>",
        "</head>",
        "<body>",
        "<h1>Table VI</h1>",
        "<p>Continuous joint-angle estimation performance. " + " ".join(html.escape(line) for line in _table_vi_note_lines()) + "</p>",
        "<table>",
        "<thead><tr><th>Configuration</th><th>n</th>" + "".join(f"<th>{html.escape(label)}</th>" for label in headers) + "</tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        cells = "".join(f"<td class=\"num\">{html.escape(str(row[col_id]))}</td>" for col_id, _, _, _ in TABLE_VI_COLUMNS)
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['configuration']))}</td>"
            f"<td class=\"num\">{row['n_subjects']}</td>"
            f"{cells}"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>", "</body>", "</html>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_table_vi(run_dir: Path, force: bool = False) -> list[Path]:
    input_path = run_dir / TABLE_VI_INPUT
    if not input_path.exists():
        raise FileNotFoundError(
            "Missing Table VI input metrics file:\n"
            f"  {input_path}\n"
            "Run scripts/04_eval_db3_angle_raw_vs_augmented.py first or choose another --run-dir."
        )
    legacy = [run_dir / "tables" / name for name in ("table_v.csv", "table_v.md", "table_v.html")]
    existing_legacy = [path for path in legacy if path.exists()]
    if existing_legacy:
        print("Legacy note: existing tables/table_v.* angle-prediction files are historical misnamed remnants and were not deleted:")
        for path in existing_legacy:
            print(f"  {path}")

    metrics = json.loads(input_path.read_text(encoding="utf-8"))
    if metrics.get("angle_target") != key10_target_metadata():
        raise ValueError(
            "Table VI only accepts fixed Key10 Exp3 metrics. The selected report is legacy or incompatible."
        )
    rows = _compute_table_vi_rows(metrics)
    output_dir = run_dir / "tables"
    outputs = [output_dir / name for name in TABLE_VI_OUTPUTS]
    _ensure_can_write(outputs, force=force)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table_vi_csv(outputs[0], rows)
    _write_table_vi_md(outputs[1], rows)
    _write_table_vi_html(outputs[2], rows)
    print("Table VI angle prediction summary:")
    for row in rows:
        print(
            f"  {row['configuration']}: n={row['n_subjects']} "
            f"global_rmse={row['global_rmse']} global_cc={row['global_cc']} mcp_cc={row['mcp_cc']}"
        )
    return outputs


def _ensure_native_import_path() -> None:
    from utils.windows_conda_path import ensure_current_env_dll_path

    ensure_current_env_dll_path()


def _load_run_config(run_dir: Path) -> dict:
    _ensure_native_import_path()
    import yaml
    from utils.paper_pipeline import flatten_pipeline_config

    snapshot = run_dir / "00_config" / "config_snapshot.yaml"
    config_path = snapshot if snapshot.exists() else PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    old_run_dir = os.environ.get("MCIA_RUN_DIR")
    os.environ["MCIA_RUN_DIR"] = str(run_dir)
    try:
        return flatten_pipeline_config(cfg)
    finally:
        if old_run_dir is None:
            os.environ.pop("MCIA_RUN_DIR", None)
        else:
            os.environ["MCIA_RUN_DIR"] = old_run_dir


def _split_test_indices(split_payload: dict, split_path: Path) -> list[int]:
    split = split_payload.get("split")
    if isinstance(split, dict) and "test" in split:
        return [int(x) for x in split["test"]]
    for key in ("test_idx", "test_indices", "test"):
        if key in split_payload:
            return [int(x) for x in split_payload[key]]
    raise KeyError(f"No test split found in {split_path}; expected split.test or test_idx.")


def _mode_specs() -> tuple[tuple[str, str, str], ...]:
    return (
        ("direct_transfer", "Direct transfer", "direct_transfer.pth"),
        ("amputee_only", "Amputee-only", "amputee_only.pth"),
        ("pretrained_finetuned", "Pretrained-finetuned", "pretrained_finetuned.pth"),
    )


def _activate_adapters_for_eval(model, config: dict, device: str) -> None:
    import importlib.util

    module_path = PROJECT_ROOT / "scripts" / "02_finetune_mcia_db3_amputee.py"
    spec = importlib.util.spec_from_file_location("mcia_db3_finetune_eval_helpers", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.activate_adapters(
        model,
        device,
        n_adapter_blocks=int(config.get("transfer_adapter_blocks", 2)),
        bottleneck_dim=int(config.get("transfer_adapter_bottleneck", 32)),
    )


def _evaluate_mode_on_subject(model_path: Path, mode: str, emg, masks, config: dict, device: str) -> dict[str, float]:
    _ensure_native_import_path()
    import numpy as np
    import torch
    from utils.paper_pipeline import build_mcia, complete_with_mask, load_mcia_state_dict, masked_completion_metrics

    model = build_mcia(config, device)
    if mode == "pretrained_finetuned":
        _activate_adapters_for_eval(model, config, device)
    load_mcia_state_dict(model, model_path, device)
    model.eval()

    batch_size = int(config.get("batch_size", 64))
    completed_batches = []
    with torch.no_grad():
        for start in range(0, len(emg), batch_size):
            end = min(start + batch_size, len(emg))
            batch = torch.as_tensor(emg[start:end], dtype=torch.float32, device=device)
            mask = torch.as_tensor(masks[start:end], dtype=torch.float32, device=device)
            completed = complete_with_mask(model, batch, mask, domain_id=1)
            completed_batches.append(completed.detach().cpu().numpy())
    pred = np.concatenate(completed_batches, axis=0)
    metrics = masked_completion_metrics(pred, emg, masks)
    return {
        "corr_masked": float(metrics.get("corr", float("nan"))),
        "mse_masked": float(metrics.get("mse_masked", float("nan"))),
        "mae_masked": float(metrics.get("mae", float("nan"))),
    }


def _generate_subject_test_masks(subject_id: int, n_items: int, config: dict, scenario: str, device: str):
    _ensure_native_import_path()
    import numpy as np
    from utils.paper_pipeline import build_mask_generator

    mask_gen = build_mask_generator(config)
    mask_gen.rng = np.random.default_rng(int(config.get("transfer_random_seed", 42)) + subject_id * 1009)
    batch_size = int(config.get("batch_size", 64))
    chunks = []
    for start in range(0, n_items, batch_size):
        bsz = min(batch_size, n_items - start)
        mask = mask_gen.generate_batch_masks(
            bsz,
            n_channels=int(config.get("n_channels", 12)),
            time_steps=int(config["window_size"]),
            device="cpu",
            scenario=scenario,
        ).transpose(1, 2)
        chunks.append(mask.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _compute_table_v_transfer_rows(run_dir: Path) -> tuple[list[dict[str, object]], list[str], dict[str, object]]:
    _ensure_native_import_path()
    import numpy as np
    from data.dataset_db3_emg import prepare_data_db3
    from data.ninapro_loader import NinaProDataLoader

    config = _load_run_config(run_dir)
    device = str(config.get("device", "cpu"))
    scenario = str(config.get("val_scenario", "s1"))
    subjects = [int(s) for s in config.get("transfer_db3_subjects", [])]
    if not subjects:
        raise ValueError("No transfer_db3_subjects found in run config.")

    ckpt_root = run_dir / "02_db3_transfer_completion" / "checkpoints"
    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    per_mode: dict[str, list[dict[str, float]]] = {mode: [] for mode, _, _ in _mode_specs()}
    warnings: list[str] = []
    evaluated_subjects = 0

    for subject_id in subjects:
        subject_dir = ckpt_root / f"S{subject_id:02d}"
        split_path = subject_dir / "split.json"
        required = [subject_dir / filename for _, _, filename in _mode_specs()]
        missing = [str(path) for path in [split_path, *required] if not path.exists()]
        if missing:
            warnings.append(f"S{subject_id:02d} skipped: missing " + "; ".join(missing))
            continue
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        try:
            test_idx = _split_test_indices(split_payload, split_path)
        except Exception as exc:
            warnings.append(f"S{subject_id:02d} skipped: {exc}")
            continue
        if not test_idx:
            warnings.append(f"S{subject_id:02d} skipped: empty test split in {split_path}")
            continue
        try:
            segments, _ = prepare_data_db3(data_loader, [subject_id], config)
        except Exception as exc:
            warnings.append(f"S{subject_id:02d} skipped: DB3 data load failed: {exc}")
            continue
        max_idx = max(test_idx)
        if max_idx >= len(segments):
            warnings.append(
                f"S{subject_id:02d} skipped: split test index {max_idx} >= n_segments {len(segments)}"
            )
            continue
        test_emg = segments[np.asarray(test_idx, dtype=np.int64)]
        masks = _generate_subject_test_masks(subject_id, len(test_emg), config, scenario, device)
        evaluated_subjects += 1
        for mode, _, filename in _mode_specs():
            metrics = _evaluate_mode_on_subject(subject_dir / filename, mode, test_emg, masks, config, device)
            per_mode[mode].append(metrics)

    rows: list[dict[str, object]] = []
    for mode, label, _ in _mode_specs():
        metrics_list = per_mode[mode]
        row: dict[str, object] = {
            "configuration": label,
            "mode": mode,
            "n_subjects": len(metrics_list),
            "mask_scenario": scenario,
        }
        for metric in ("corr_masked", "mse_masked", "mae_masked"):
            vals = _safe_values([item.get(metric) for item in metrics_list])
            mean_value = _mean(vals) if vals else None
            sd_value = stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_sd"] = sd_value
            row[f"{metric}_n"] = len(vals)
            row[metric] = _format_mean_sd(mean_value, sd_value)
        rows.append(row)

    meta = {
        "scenario": scenario,
        "evaluated_subjects": evaluated_subjects,
        "configured_subjects": len(subjects),
        "note": "DB3 controlled artificial masking completion performance; not real anomaly-region recovery performance.",
    }
    return rows, warnings, meta


def _write_table_v_transfer_csv(path: Path, rows: list[dict[str, object]], warnings: list[str], meta: dict[str, object]) -> None:
    fieldnames = [
        "configuration",
        "mode",
        "n_subjects",
        "mask_scenario",
        "corr_masked",
        "mse_masked",
        "mae_masked",
        "corr_masked_mean",
        "corr_masked_sd",
        "corr_masked_n",
        "mse_masked_mean",
        "mse_masked_sd",
        "mse_masked_n",
        "mae_masked_mean",
        "mae_masked_sd",
        "mae_masked_n",
        "warning_count",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {name: row.get(name, "") for name in fieldnames}
            payload["warning_count"] = len(warnings)
            payload["note"] = meta.get("note", "")
            writer.writerow(payload)


def _write_table_v_transfer_md(path: Path, rows: list[dict[str, object]], warnings: list[str], meta: dict[str, object]) -> None:
    lines = [
        "# Table V",
        "",
        "Comparison of transfer configurations on DB3 controlled artificial masking completion.",
        "Values are subject-level mean +/- SD.",
        f"Mask scenario: `{meta.get('scenario', 'unknown')}`.",
        "Note: this table evaluates controlled artificial masking performance and does not equal real anomaly-region recovery performance.",
        "",
        "| Configuration | n | Masked correlation | Masked MSE | Masked MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {configuration} | {n_subjects} | {corr_masked} | {mse_masked} | {mae_masked} |".format(**row)
        )
    lines.extend(["", "## Effective n", "", "| Configuration | Corr n | MSE n | MAE n |", "|---|---:|---:|---:|"])
    for row in rows:
        lines.append(
            "| {configuration} | {corr_masked_n} | {mse_masked_n} | {mae_masked_n} |".format(**row)
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in warnings)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table_v_transfer_html(path: Path, rows: list[dict[str, object]], warnings: list[str], meta: dict[str, object]) -> None:
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>Table V DB3 transfer completion</title>",
        "  <style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:6px 10px;}th{background:#f3f3f3;}td.num{text-align:right;}li{margin:4px 0;}</style>",
        "</head>",
        "<body>",
        "<h1>Table V</h1>",
        f"<p>Comparison of transfer configurations on DB3 controlled artificial masking completion. Values are subject-level mean +/- SD. Mask scenario: <code>{html.escape(str(meta.get('scenario', 'unknown')))}</code>.</p>",
        "<p>Note: this table evaluates controlled artificial masking performance and does not equal real anomaly-region recovery performance.</p>",
        "<table>",
        "<thead><tr><th>Configuration</th><th>n</th><th>Masked correlation</th><th>Masked MSE</th><th>Masked MAE</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['configuration']))}</td>"
            f"<td class=\"num\">{row['n_subjects']}</td>"
            f"<td class=\"num\">{html.escape(str(row['corr_masked']))}</td>"
            f"<td class=\"num\">{html.escape(str(row['mse_masked']))}</td>"
            f"<td class=\"num\">{html.escape(str(row['mae_masked']))}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    if warnings:
        lines.extend(["<h2>Warnings</h2>", "<ul>"])
        lines.extend(f"<li>{html.escape(item)}</li>" for item in warnings)
        lines.append("</ul>")
    lines.extend(["</body>", "</html>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_table_v(run_dir: Path, force: bool = False) -> list[Path]:
    rows, warnings, meta = _compute_table_v_transfer_rows(run_dir)
    output_dir = run_dir / "tables"
    outputs = [output_dir / name for name in TABLE_V_OUTPUTS]
    _ensure_can_write(outputs, force=force)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table_v_transfer_csv(outputs[0], rows, warnings, meta)
    _write_table_v_transfer_md(outputs[1], rows, warnings, meta)
    _write_table_v_transfer_html(outputs[2], rows, warnings, meta)
    print(
        f"Table V evaluated subjects: {meta.get('evaluated_subjects')}/{meta.get('configured_subjects')} "
        f"| scenario={meta.get('scenario')} | warnings={len(warnings)}"
    )
    for row in rows:
        print(
            f"  {row['configuration']}: n={row['n_subjects']} "
            f"corr_n={row['corr_masked_n']} mse_n={row['mse_masked_n']} mae_n={row['mae_masked_n']}"
        )
    return outputs


def _table_registry() -> dict[str, Callable[[Path, bool], list[Path]]]:
    return {"table_v": export_table_v, "table_vi": export_table_vi}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate paper tables from an existing run without re-training. Table V runs evaluation-only inference from existing DB3 checkpoints; Table VI reads saved angle metrics."
    )
    parser.add_argument("--run-dir", help="Path to outputs/run/<run_id>. Defaults to MCIA_RUN_DIR or latest run.")
    parser.add_argument("--tables", default="all", help="Comma-separated table ids: table_v, table_vi, or all. table_v exports DB3 transfer completion metrics; table_vi exports angle prediction metrics.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing table files.")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    registry = _table_registry()
    requested = [item.strip() for item in args.tables.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        requested = list(registry.keys())

    unknown = [item for item in requested if item not in registry]
    if unknown:
        known = ", ".join(sorted(registry))
        raise ValueError(f"Unknown table id(s): {', '.join(unknown)}. Known: {known}, all")

    print("=" * 80)
    print("Generate tables from existing run")
    print("=" * 80)
    print(f"Run dir: {run_dir}")

    all_outputs: list[Path] = []
    for table_id in requested:
        outputs = registry[table_id](run_dir, args.force)
        all_outputs.extend(outputs)
        print(f"{table_id}: {len(outputs)} files")
        for path in outputs:
            print(f"  {path}")
    print("Done.")


if __name__ == "__main__":
    main()
