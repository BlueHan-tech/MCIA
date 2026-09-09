"""在不重新训练的情况下，从已有 MCIA 运行结果重新生成图表。

点击运行行为：
- 若已设置 MCIA_RUN_DIR，则对该运行出图
- 否则使用最新的 outputs/run/run_* 目录
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import argparse
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

from utils.paper_pipeline import flatten_pipeline_config
import importlib.util


def _load_script_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


db3_figs = _load_script_module("db3_figs", "scripts/03_generate_augmented_db3_semg.py")
exp3_figs = _load_script_module("exp3_figs", "scripts/04_eval_db3_angle_raw_vs_augmented.py")


def latest_run_dir(output_root: Path) -> Path | None:
    run_root = output_root / "run"
    if not run_root.exists():
        return None
    runs = sorted([p for p in run_root.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def load_config_for_run(run_dir: Path) -> dict:
    os.environ["MCIA_RUN_DIR"] = str(run_dir)
    snapshot = run_dir / "00_config" / "config_snapshot.yaml"
    config_path = snapshot if snapshot.exists() else PROJECT_ROOT / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return flatten_pipeline_config(cfg)


def _first_db2_test_subject(config: dict) -> int:
    subjects = config.get("test_subjects") or []
    if not subjects:
        raise ValueError("No DB2 test_subjects found in run config.")
    return int(subjects[0])


def _generate_mask_for_sample(mask_gen, scenario: str, config: dict, seed: int):
    import numpy as _np

    if hasattr(mask_gen, "rng"):
        mask_gen.rng = _np.random.default_rng(seed)
    mask = mask_gen.generate_batch_masks(
        1,
        n_channels=int(config.get("n_channels", 12)),
        time_steps=int(config["window_size"]),
        device="cpu",
        scenario=scenario,
    ).transpose(1, 2)
    return mask.cpu().numpy()[0]


def _complete_db2_sample(model, emg_clean: np.ndarray, mask: np.ndarray, config: dict, device: str) -> np.ndarray:
    import torch
    from utils.paper_pipeline import complete_with_mask

    with torch.no_grad():
        batch = torch.as_tensor(emg_clean[None, :, :], dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask[None, :, :], dtype=torch.float32, device=device)
        completed = complete_with_mask(model, batch, mask_t, domain_id=None)
    return completed.detach().cpu().numpy()[0]


def regenerate_db2_per_channel_error_heatmap(run_dir: Path, config: dict) -> tuple[Path, Path]:
    """Export aggregate S1-S4 per-subject x channel masked MAE heatmap for DB2 held-out subjects."""
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data.dataset_db2_emg import prepare_data_db2
    from data.ninapro_loader import NinaProDataLoader
    from utils.paper_pipeline import build_mask_generator, build_mcia, load_mcia_state_dict, complete_with_mask

    checkpoint = run_dir / "01_db2_completion" / "checkpoints" / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing Exp1 checkpoint for per-channel error heatmap: {checkpoint}")

    subjects = [int(s) for s in (config.get("test_subjects") or [])]
    if not subjects:
        raise ValueError("No DB2 test_subjects found in run config for per-channel heatmap.")

    out_dir = run_dir / "paper_candidates" / "per_channel_error_heatmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "db2_per_channel_masked_mae_heatmap.png"
    csv_path = out_dir / "db2_per_channel_masked_mae_heatmap.csv"

    device = str(config.get("device", "cpu"))
    batch_size = int(config.get("batch_size", 64))
    n_channels = int(config.get("n_channels", 12))
    scenarios = ["s1", "s2", "s3"]
    base_seed = int(config.get("random_seed", 42))

    model = build_mcia(config, device)
    load_mcia_state_dict(model, checkpoint, device)
    model.eval()
    mask_gen = build_mask_generator(config)
    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])

    rows: list[dict] = []
    heat_rows: list[np.ndarray] = []
    subject_labels: list[str] = []

    for subject_id in subjects:
        segments, _, repetitions = prepare_data_db2(data_loader, [subject_id], config)
        test_mask = np.isin(repetitions, [2, 5]) if repetitions is not None else np.ones(len(segments), dtype=bool)
        test_indices = np.flatnonzero(test_mask)
        if len(test_indices) == 0:
            print(f"  DB2 per-channel heatmap S{subject_id:02d}: no Reps 2/5 test windows; skipped.")
            continue

        abs_sum = np.zeros(n_channels, dtype=np.float64)
        count_sum = np.zeros(n_channels, dtype=np.float64)
        subject_windows = segments[test_indices]

        for scenario_rank, scenario in enumerate(scenarios):
            if hasattr(mask_gen, "rng"):
                mask_gen.rng = np.random.default_rng(base_seed + subject_id * 1000 + scenario_rank * 100000)
            for start in range(0, len(subject_windows), batch_size):
                batch_np = np.asarray(subject_windows[start:start + batch_size], dtype=np.float32)
                if batch_np.size == 0:
                    continue
                emg = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
                B, T, C = emg.shape
                mask = mask_gen.generate_batch_masks(
                    B,
                    n_channels=C,
                    time_steps=T,
                    device=device,
                    scenario=scenario,
                ).transpose(1, 2).float()
                completed = complete_with_mask(model, emg, mask, domain_id=None)
                missing = mask < 0.5
                err = torch.abs(completed - emg)
                abs_sum += (err * missing).sum(dim=(0, 1)).detach().cpu().numpy().astype(np.float64)
                count_sum += missing.sum(dim=(0, 1)).detach().cpu().numpy().astype(np.float64)

        with np.errstate(invalid="ignore", divide="ignore"):
            mae = abs_sum / count_sum
        mae[count_sum <= 0] = np.nan
        heat_rows.append(mae)
        subject_label = f"S{subject_id:02d}"
        subject_labels.append(subject_label)
        row = {"subject": subject_label, "scenario": "ALL", "n_test_windows": int(len(test_indices))}
        for ch in range(n_channels):
            row[f"Ch{ch + 1}"] = "" if not np.isfinite(mae[ch]) else float(mae[ch])
            row[f"Ch{ch + 1}_masked_count"] = int(count_sum[ch])
        rows.append(row)
        print(f"  DB2 per-channel heatmap {subject_label}: windows={len(test_indices)} scenarios=S1-S4")

    if not rows:
        raise RuntimeError("No DB2 held-out subject rows were computed for per-channel error heatmap.")

    fieldnames = ["subject", "scenario", "n_test_windows"]
    fieldnames += [f"Ch{i}" for i in range(1, n_channels + 1)]
    fieldnames += [f"Ch{i}_masked_count" for i in range(1, n_channels + 1)]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    heat = np.vstack(heat_rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.25), constrained_layout=True)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f5f5f5")
    im = ax.imshow(heat, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_xticks(np.arange(n_channels))
    ax.set_xticklabels([f"Ch{i}" for i in range(1, n_channels + 1)], fontsize=8)
    ax.set_yticks(np.arange(len(subject_labels)))
    ax.set_yticklabels(subject_labels, fontsize=8.5)
    ax.set_xlabel("EMG channel", fontsize=9)
    ax.set_ylabel("DB2 held-out subject", fontsize=9)

    ax.set_xticks(np.arange(-0.5, n_channels, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subject_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Masked MAE", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"  DB2 per-channel masked MAE heatmap PNG: {png_path}")
    print(f"  DB2 per-channel masked MAE heatmap CSV: {csv_path}")
    return png_path, csv_path


def _load_db2_scenario_metric_rows(run_dir: Path) -> tuple[list[dict], str]:
    """Load per-subject DB2 S1-S4 metrics for paper Fig.1."""
    test_root = run_dir / "01_db2_completion" / "figures" / "test"
    report_metrics = run_dir / "01_db2_completion" / "figures" / "report" / "metrics.json"
    if not test_root.exists():
        raise FileNotFoundError(f"Missing Exp1 per-subject test figures directory: {test_root}")

    rows: list[dict] = []
    metric_files = sorted(test_root.glob("S*/zeroshot/metrics.json"))
    for metrics_path in metric_files:
        subject = metrics_path.parents[1].name
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_scenario = data.get("metrics_per_scenario")
        if not isinstance(per_scenario, dict):
            continue
        for scenario in ("s1", "s2", "s3"):
            metrics = per_scenario.get(scenario)
            if not isinstance(metrics, dict):
                continue
            rows.append({
                "source_level": "subject",
                "subject": subject,
                "scenario": scenario.upper(),
                "corr_masked": metrics.get("corr_masked", np.nan),
                "mse_masked": metrics.get("mse_masked", np.nan),
                "mae_masked": metrics.get("mae_masked", np.nan),
                "mask_ratio": metrics.get("mask_ratio", np.nan),
            })
    if rows:
        return rows, f"per-subject metrics from {test_root}"

    if not report_metrics.exists():
        raise FileNotFoundError(
            "Missing Exp1 Fig.1 inputs. Expected per-subject metrics under "
            f"{test_root} or aggregate report metrics at {report_metrics}."
        )
    data = json.loads(report_metrics.read_text(encoding="utf-8"))
    per_scenario = data.get("metrics_per_scenario")
    if not isinstance(per_scenario, dict):
        raise KeyError(f"Missing metrics_per_scenario in aggregate report metrics: {report_metrics}")
    for scenario in ("s1", "s2", "s3"):
        metrics = per_scenario.get(scenario)
        if not isinstance(metrics, dict):
            continue
        rows.append({
            "source_level": "aggregate",
            "subject": "aggregate",
            "scenario": scenario.upper(),
            "corr_masked": metrics.get("corr_masked", np.nan),
            "mse_masked": metrics.get("mse_masked", np.nan),
            "mae_masked": metrics.get("mae_masked", np.nan),
            "mask_ratio": metrics.get("mask_ratio", np.nan),
        })
    if not rows:
        raise RuntimeError(f"No usable S1-S4 metrics found in {report_metrics}")
    return rows, f"aggregate metrics from {report_metrics}; no subject dots available"


def regenerate_db2_scenario_quant_figure(run_dir: Path) -> tuple[Path, Path]:
    """Export paper Fig.1: DB2 held-out ScenarioMix quantitative performance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, source_note = _load_db2_scenario_metric_rows(run_dir)
    out_dir = run_dir / "paper_candidates" / "fig1_db2_scenario_quantitative"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "figure1_db2_scenario_metrics.csv"
    png_path = out_dir / "figure1_db2_scenario_metrics.png"

    scenarios = ["S1", "S2", "S3", "S4"]
    metrics = [
        ("corr_masked", "Masked correlation", "(a) Masked correlation"),
        ("mse_masked", "Masked MSE", "(b) Masked MSE"),
        ("mae_masked", "Masked MAE", "(c) Masked MAE"),
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_level", "subject", "scenario", "corr_masked", "mse_masked", "mae_masked", "mask_ratio"],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    bar_color = "#d8ecf7"
    edge_color = "#5f9fc4"
    dot_color = "#6f6f6f"
    x = np.arange(len(scenarios), dtype=float)
    rng = np.random.default_rng(20260723)
    source_levels = {str(r.get("source_level", "")) for r in rows}
    has_subject_rows = "subject" in source_levels

    for ax, (key, ylabel, panel_title) in zip(axes, metrics):
        means = []
        sds = []
        values_by_scenario: list[np.ndarray] = []
        for scenario in scenarios:
            vals = np.array([
                float(r[key]) for r in rows
                if r["scenario"] == scenario and np.isfinite(float(r[key]))
            ], dtype=float)
            values_by_scenario.append(vals)
            means.append(float(np.mean(vals)) if vals.size else np.nan)
            sds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)

        ax.bar(x, means, yerr=sds, width=0.52, color=bar_color, edgecolor=edge_color,
               linewidth=0.9, capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
        if has_subject_rows:
            for idx, vals in enumerate(values_by_scenario):
                if vals.size == 0:
                    continue
                jitter = rng.normal(0.0, 0.03, size=vals.size)
                ax.scatter(np.full(vals.size, x[idx]) + jitter, vals, s=12,
                           color=dot_color, alpha=0.68, zorder=3, linewidths=0)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.8)
        ax.set_title(panel_title, fontsize=9.2, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        if key == "corr_masked":
            finite_vals = np.concatenate([v for v in values_by_scenario if v.size]) if any(v.size for v in values_by_scenario) else np.array([])
            if finite_vals.size:
                lo = min(0.0, float(np.nanmin(finite_vals)) - 0.05)
                hi = min(1.0, float(np.nanmax(finite_vals)) + 0.08)
                ax.set_ylim(lo, hi)

    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    n_subjects = len({r["subject"] for r in rows if r.get("source_level") == "subject"})
    print(f"  DB2 Fig.1 source: {source_note}")
    if has_subject_rows:
        print(f"  DB2 Fig.1 subject dots: n={n_subjects}")
    else:
        print("  DB2 Fig.1: per-subject scenario data unavailable; plotted aggregate metrics only.")
    print(f"  DB2 Fig.1 PNG: {png_path}")
    print(f"  DB2 Fig.1 CSV: {csv_path}")
    return png_path, csv_path


def regenerate_db2_fig2_candidates(run_dir: Path, config: dict) -> int:
    """Export paper Fig.2 DB2 completion candidates without baselines or observed-zero lines."""
    import torch
    from data.dataset_db2_emg import prepare_data_db2
    from data.ninapro_loader import NinaProDataLoader
    from utils.paper_pipeline import build_mask_generator, build_mcia, load_mcia_state_dict
    from utils.visualization import plot_completion_panel

    checkpoint = run_dir / "01_db2_completion" / "checkpoints" / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing Exp1 checkpoint for Fig.2 candidates: {checkpoint}")

    subject_id = _first_db2_test_subject(config)
    data_loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    segments, _, repetitions = prepare_data_db2(data_loader, [subject_id], config)
    test_mask = np.isin(repetitions, [2, 5]) if repetitions is not None else np.ones(len(segments), dtype=bool)
    test_indices = np.flatnonzero(test_mask)
    if len(test_indices) == 0:
        raise RuntimeError(f"No DB2 Reps 2/5 test windows found for S{subject_id:02d}.")

    device = str(config.get("device", "cpu"))
    model = build_mcia(config, device)
    load_mcia_state_dict(model, checkpoint, device)
    model.eval()
    mask_gen = build_mask_generator(config)

    out_root = run_dir / "paper_candidates" / "fig2_db2_completion"
    panels_dir = out_root / "panels"
    whole_dir = out_root / "whole_channel"
    panels_dir.mkdir(parents=True, exist_ok=True)
    whole_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    panel_kwargs = {
        "show_observed_line": False,
        "show_mask_background": True,
        "mask_background_color": "#dceeff",
        "mask_background_alpha": 0.35,
    }
    base_seed = int(config.get("random_seed", 42))
    scenarios = ["s1", "s2", "s3"]

    first_idx = int(test_indices[0])
    for rank, scenario in enumerate(scenarios, start=1):
        emg_clean = segments[first_idx]
        mask = _generate_mask_for_sample(mask_gen, scenario, config, seed=base_seed + subject_id * 1000 + rank)
        completed = _complete_db2_sample(model, emg_clean, mask, config, device)
        path = panels_dir / f"S{subject_id:02d}_{scenario}_sample{first_idx:04d}.png"
        plot_completion_panel(
            emg_clean=emg_clean,
            emg_completed=completed,
            mask=mask,
            save_path=path,
            title=f"DB2 S{subject_id:02d} {scenario.upper()} representative sample {first_idx}",
            **panel_kwargs,
        )
        saved.append(path)

    whole_saved = 0
    max_scan = min(len(test_indices), 80)
    for scenario in scenarios:
        if whole_saved >= 2:
            break
        for local_rank, sample_idx in enumerate(test_indices[:max_scan]):
            sample_idx = int(sample_idx)
            mask = _generate_mask_for_sample(
                mask_gen,
                scenario,
                config,
                seed=base_seed + subject_id * 1000 + 10000 + local_rank + scenarios.index(scenario) * 1000,
            )
            whole_missing = (mask < 0.5).all(axis=0)
            if not whole_missing.any():
                continue
            emg_clean = segments[sample_idx]
            completed = _complete_db2_sample(model, emg_clean, mask, config, device)
            path = whole_dir / f"S{subject_id:02d}_{scenario}_whole_channel_sample{sample_idx:04d}.png"
            plot_completion_panel(
                emg_clean=emg_clean,
                emg_completed=completed,
                mask=mask,
                save_path=path,
                title=f"DB2 S{subject_id:02d} {scenario.upper()} whole-channel sample {sample_idx}",
                **panel_kwargs,
            )
            saved.append(path)
            whole_saved += 1
            break

    if whole_saved == 0:
        print("  DB2 Fig.2 whole_channel: no whole-channel missing samples found in deterministic scan; skipped.")
    print(f"  DB2 Fig.2 S{subject_id:02d}: {len(saved)} candidate figures -> {out_root}")
    for path in saved:
        print(f"    {path}")
    return len(saved)


def regenerate_db3_completion_figures(run_dir: Path, config: dict, raw_cfg: dict) -> int:
    aug_dir = run_dir / "02_db3_transfer_completion" / "augmented_emg"
    npz_files = sorted(aug_dir.glob("db3_S*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            "Current run is missing DB3 augmented .npz files under "
            f"{aug_dir}. Run scripts/03_generate_augmented_db3_semg.py first. "
            "Fallback to external augmented directories is not allowed."
        )
    expected_mask_mode = str(raw_cfg.get("exp2_transfer", {}).get(
        "augmentation_mask_mode",
        config.get("transfer_augmentation_mask_mode", "rule"),
    ))
    count = 0
    for npz_path in npz_files:
        subject_id = int(npz_path.stem.split("S")[-1])
        data = np.load(npz_path)
        if "mask" not in data.files:
            raise KeyError(f"Missing required 'mask' in DB3 augmented file: {npz_path}")
        if "mask_mode" not in data.files:
            raise KeyError(f"Missing required 'mask_mode' in DB3 augmented file: {npz_path}")
        mask_mode = str(data["mask_mode"])
        if mask_mode != expected_mask_mode:
            raise ValueError(
                f"DB3 mask_mode mismatch in {npz_path}: "
                f"file has {mask_mode!r}, expected {expected_mask_mode!r}."
            )
        original = data["original"]
        enhanced = data["enhanced"]
        mask = data["mask"]
        direct = data["direct_enhanced"] if "direct_enhanced" in data.files else enhanced
        patch_mask_status = "yes" if "patch_mask" in data.files else "no"
        dead_channels = (
            [int(c + 1) for c in data["dead_channels"].tolist()]
            if "dead_channels" in data.files else []
        )
        saved = db3_figs.save_subject_semg_panels(
            aug_dir, subject_id, original, direct, enhanced, mask, config
        )
        count += len(saved)
        print(
            f"  DB3 S{subject_id:02d}: {len(saved)} figures | "
            f"mask_mode={mask_mode} patch_mask={patch_mask_status} "
            f"dead_channels_1based={dead_channels}"
        )
    return count

def regenerate_exp3_abc_figures(run_dir: Path, config: dict) -> int:
    pred_dir = run_dir / "03_angle_prediction" / "predictions"
    out_dir = run_dir / "03_angle_prediction" / "figures" / "comparison"
    count = 0
    if not pred_dir.exists():
        print(f"  Exp3 skipped: prediction directory not found: {pred_dir}")
        return 0
    plot_groups = getattr(exp3_figs, "ANGLE_PLOT_GROUPS", {})
    expected_groups = ",".join(plot_groups.keys()) if plot_groups else "anatomy"
    n_trials = int(config.get("regressor_viz_trials_per_subject", 2))
    report_path = run_dir / "03_angle_prediction" / "metrics" / "db3_angle_raw_vs_augmented_results.json"
    low_by_subject: dict[int, bool] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for subject in report.get("subjects", []):
                if "subject_id" in subject:
                    low_by_subject[int(subject["subject_id"])] = bool(
                        subject.get("low_dynamic_train_coverage", False)
                    )
        except Exception as exc:
            print(f"  Exp3 warning: could not read dynamic coverage report: {exc}")
    for npz_path in sorted(pred_dir.glob("S*_angle_predictions.npz")):
        subject_id = int(npz_path.stem.split("_")[0].replace("S", ""))
        with np.load(npz_path) as data:
            exp3_figs.assert_key10_prediction_payload(data, str(npz_path))
            required_continuous = {
                "continuous_target", "continuous_time_indices",
                "continuous_overlap_counts", "continuous_dynamic_mask",
                *(f"pred_{grp}_continuous" for grp in ("A", "B")),
            }
            missing_continuous = sorted(required_continuous.difference(data.files))
            if missing_continuous:
                print(
                    f"  Exp3 S{subject_id:02d} skipped: 1280-point continuous payload is missing "
                    f"{missing_continuous}; resume Exp3 predictions to rebuild it."
                )
                continue
            target = np.asarray(data["continuous_target"])
            group_preds = {
                grp: np.asarray(data[f"pred_{grp}_continuous"]) for grp in ("A", "B")
            }
            time_indices = np.asarray(data["continuous_time_indices"], dtype=np.int64)
            overlap_counts = np.asarray(data["continuous_overlap_counts"], dtype=np.int16)
            dynamic_mask = np.asarray(data["continuous_dynamic_mask"], dtype=bool)
        saved, selection = exp3_figs.save_abc_comparison_figures(
            subject_id,
            group_preds,
            target,
            time_indices,
            overlap_counts,
            dynamic_mask,
            out_dir,
            n_trials=n_trials,
            config=config,
            low_dynamic_coverage=low_by_subject.get(subject_id, False),
            output_postprocess=exp3_figs.continuous_output_label(config),
            return_selection=True,
        )
        count += len(saved)
        print(
            f"  Exp3 S{subject_id:02d}: {len(saved)} continuous A/B figures "
            f"({expected_groups}) time_indices={selection.get('selected_time_indices', [])}"
        )
    return count


def _conda_env_args() -> list[str]:
    env_value = os.environ.get("CONDA_DEFAULT_ENV", "apple")
    env_path = Path(env_value)
    if env_path.exists() or env_path.is_absolute() or "/" in env_value or ":" in env_value:
        return ["-p", env_value]
    return ["-n", env_value]


def regenerate_paper_figures(run_dir: Path) -> None:
    """Legacy/manual paper figure rebuild; default flow skips this pending redesign."""
    conda_exe = Path(os.environ.get("CONDA_EXE", r"C:\Users\zouyuhan.pat\miniforge3\Scripts\conda.exe"))
    env = os.environ.copy()
    env["MCIA_RUN_DIR"] = str(run_dir)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, "scripts/generate_paper_figures.py", "--infer"]
    if conda_exe.exists():
        cmd = [str(conda_exe), "run", *_conda_env_args(), "python", "scripts/generate_paper_figures.py", "--infer"]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate current DB2 Fig.1 quantitative metrics, DB2 per-channel error heatmap, DB2 Fig.2 candidates, DB3 completion, and anatomy A/B angle figures from an existing run. "
            "Legacy paper figures are pending redesign and skipped by default."
        )
    )
    parser.add_argument(
        "--include-legacy-paper-figures",
        action="store_true",
        help=(
            "Also run scripts/generate_paper_figures.py --infer to rebuild legacy/pending-redesign "
            "04_paper_figures outputs."
        ),
    )
    args = parser.parse_args()

    raw_cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    if os.environ.get("MCIA_RUN_DIR"):
        run_dir = Path(os.environ["MCIA_RUN_DIR"])
    else:
        output_root = PROJECT_ROOT / "outputs"
        run_dir = latest_run_dir(output_root)
        if run_dir is None:
            raise FileNotFoundError(
                f"No existing current-project run found under {output_root / 'run'}. "
                "Set MCIA_RUN_DIR or run scripts/run_all_experiments.py first."
            )
    os.environ["MCIA_RUN_DIR"] = str(run_dir)
    config = load_config_for_run(run_dir)

    print("=" * 80)
    print("Generate figures from existing run")
    print("=" * 80)
    print(f"Run dir: {run_dir}")

    fig1_png, fig1_csv = regenerate_db2_scenario_quant_figure(run_dir)
    heatmap_png, heatmap_csv = regenerate_db2_per_channel_error_heatmap(run_dir, config)
    db2_count = regenerate_db2_fig2_candidates(run_dir, config)
    db3_count = regenerate_db3_completion_figures(run_dir, config, raw_cfg)
    exp3_count = regenerate_exp3_abc_figures(run_dir, config)
    print(f"DB2 Fig.1 quantitative figure: {fig1_png}")
    print(f"DB2 Fig.1 source CSV: {fig1_csv}")
    print(f"DB2 per-channel masked MAE heatmap: {heatmap_png}")
    print(f"DB2 per-channel masked MAE CSV: {heatmap_csv}")
    print(f"DB2 Fig.2 candidate figures: {db2_count}")
    print(f"DB3 completion figures: {db3_count}")
    print(f"Exp3 anatomy A/B figures: {exp3_count}")

    if args.include_legacy_paper_figures:
        try:
            regenerate_paper_figures(run_dir)
        except Exception as exc:
            print(f"Legacy paper figures skipped: {exc}")
    else:
        print(
            "Legacy paper figures/tables are pending redesign and skipped by default. "
            "Use --include-legacy-paper-figures to rebuild 04_paper_figures manually."
        )

    print("Done.")


if __name__ == "__main__":
    main()

