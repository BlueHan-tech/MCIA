"""
Legacy / pending redesign manual paper-figure entry.

This script generates paper-style figures/tables from real experiment outputs,
but it is no longer part of the default scripts/run_all_experiments.py pipeline
or the default scripts/generate_figures_from_run.py rebuild flow.

Current default mainline figures are:
- 01 DB2 completion report;
- 03 DB3 12ch completion figures;
- 04 anatomy A/B/C angle figures.

Keep this script as a manual legacy entry only. Do not add new mainline figures
here while the paper figure/table system is pending redesign.

Usage:
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --figures fig3,fig4,table1
    python scripts/generate_paper_figures.py --infer

Designed skips:
- Figure 1 architecture diagram and Figure 5 transfer diagram are drawn manually.
- Existing training visualizations are skipped unless --force is used.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset_db2_emg import EMGCompletionDataset
from data.dataset_db3_emg import prepare_data_db3
from data.ninapro_loader import NinaProDataLoader
from utils.baselines import evaluate_cubic_spline
from utils.paper_figures import (
    SCENARIOS,
    collect_exp1_scenario_metrics,
    export_table1_exp1,
    export_table2_exp2a,
    export_table3_exp3,
    plot_figure2_completion_waveform,
    plot_figure3_scenario_bars,
    plot_figure4_training_curves,
    plot_figure7_rule_detector as plot_figure6_rule_detector,
    plot_figure8_angle_traces as plot_figure7_angle_traces,
    plot_figure9_grouped_bars as plot_figure8_grouped_bars,
    plot_figure10_improvement_scatter as plot_figure9_improvement_scatter,
    _load_json,
    _skip,
)
from utils.paper_pipeline import (
    build_mcia,
    build_mask_generator,
    flatten_pipeline_config,
    load_mcia_state_dict,
    load_yaml_config,
)
from utils.rule_anomaly_detector import RuleAnomalyDetector
from utils.kinematic_target import assert_key10_prediction_payload, key10_target_metadata
from utils.timemae_baseline import evaluate_timemae
from utils.visualization import _run_model_on_batch


ALL_ITEMS = (
    "fig2", "fig3", "fig4", "fig6",
    "fig7", "fig8", "fig9",
    "table1", "table2", "table3",
)

def _first_existing(candidates: list[Path]) -> Path:
    return next((path for path in candidates if path.exists()), candidates[0])


def _exp1_checkpoint(exp1_run: Path, name: str) -> Path:
    return _first_existing([exp1_run / "checkpoints" / name, exp1_run / name])


def _exp1_metric(exp1_run: Path, name: str) -> Path:
    return _first_existing([exp1_run / "metrics" / name, exp1_run / name])


def _infer_figure2(exp1_run: Path, config: dict, device: str, out_dir: Path, force: bool):
    """从 DB2 验证被试各取一个 S1/S4 窗口，运行 MCIA 推理并保存四行对比图。"""
    for scn in ("s1", "s3"):
        save = out_dir / f"figure2_completion_{scn}.png"
        if _skip(save, force):
            print(f"  [skip] {save.name} exists")
            continue
        ckpt = _exp1_checkpoint(exp1_run, "best_model.pth")
        if not ckpt.exists():
            print(f"  [fig2] missing checkpoint {ckpt}")
            return
        from data.dataset_db2_emg import prepare_data_db2
        loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
        segments, _, _ = prepare_data_db2(loader, [29], config)  # 验证集受试者
        if len(segments) == 0:
            print("  [fig2] no val segments")
            return
        model = build_mcia(config, device)
        load_mcia_state_dict(model, ckpt, device)
        model.eval()
        mask_gen = build_mask_generator(config)
        emg = torch.FloatTensor(segments[0:1]).to(device)
        B, T, C = emg.shape
        mask_soft = mask_gen.generate_batch_masks(B, C, T, device=device, scenario=scn)
        mask = (mask_soft.transpose(1, 2) > 0.5).float()
        with torch.no_grad():
            completed = _run_model_on_batch(model, emg, mask, device)
        plot_figure2_completion_waveform(
            emg[0].cpu().numpy(), completed[0].cpu().numpy(), mask[0].cpu().numpy(),
            save, channel=0, title=f"Figure 2 - scenario {scn.upper()}",
        )
        print(f"  [fig2] saved {save}")


def _eval_baselines_per_subject(config: dict, device: str, exp1_run: Path) -> dict:
    """为表 1 收集逐被试完整指标（掩码区域对齐口径）。"""
    from data.dataset_db2_emg import prepare_data_db2
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    mask_gen = build_mask_generator(config)
    scenario = config.get("val_scenario", "s1")
    result: dict = {"TimeMAE": {}, "Cubic Spline": {}}

    timemae_ckpt = _exp1_checkpoint(exp1_run, "timemae_pretrain.pt")
    timemae_model = None
    if timemae_ckpt.exists():
        from utils.timemae_baseline import TimeMAECompletion
        timemae_model = TimeMAECompletion(
            data_shape=(config["window_size"], config.get("n_channels", 12)),
            wave_length=config.get("timemae_wave_length", 8),
        ).to(device)
        timemae_model.load_state_dict(torch.load(timemae_ckpt, map_location=device))
        timemae_model.eval()

    for sid in range(33, 41):
        try:
            segments, _, reps = prepare_data_db2(loader, [sid], config)
            test_mask = np.isin(reps, [2, 5]) if reps is not None else np.zeros(len(segments), dtype=bool)
            if not test_mask.any():
                test_mask = np.ones(len(segments), dtype=bool)
            test_loader = DataLoader(
                EMGCompletionDataset(segments[test_mask]), batch_size=32, shuffle=False)
            result["Cubic Spline"][sid] = evaluate_cubic_spline(
                test_loader, mask_gen, device, scenario=scenario)
            if timemae_model is not None:
                result["TimeMAE"][sid] = evaluate_timemae(
                    timemae_model, test_loader, mask_gen, device, scenario=scenario)
        except Exception as exc:
            print(f"  [baseline] S{sid:02d} skipped: {exc}")
    return result


def _eval_baselines_per_scenario(config: dict, device: str, exp1_run: Path) -> dict:
    """在测试被试上按场景评估 TimeMAE 与三次样条基线（较慢，可选）。"""
    from data.dataset_db2_emg import prepare_data_db2
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    mask_gen = build_mask_generator(config)
    result = {"TimeMAE": {s: [] for s in SCENARIOS}, "Cubic Spline": {s: [] for s in SCENARIOS}}

    timemae_ckpt = _exp1_checkpoint(exp1_run, "timemae_pretrain.pt")
    timemae_model = None
    if timemae_ckpt.exists():
        from utils.timemae_baseline import TimeMAECompletion
        timemae_model = TimeMAECompletion(
            data_shape=(config["window_size"], config.get("n_channels", 12)),
            wave_length=config.get("timemae_wave_length", 8),
        ).to(device)
        timemae_model.load_state_dict(torch.load(timemae_ckpt, map_location=device))
        timemae_model.eval()

    for sid in range(33, 41):
        try:
            segments, _, reps = prepare_data_db2(loader, [sid], config)
            test_mask = np.isin(reps, [2, 5]) if reps is not None else np.zeros(len(segments), dtype=bool)
            if not test_mask.any():
                test_mask = np.ones(len(segments), dtype=bool)
            test_set = EMGCompletionDataset(segments[test_mask])
            test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
            for scn in SCENARIOS:
                cs_m = evaluate_cubic_spline(test_loader, mask_gen, device, scenario=scn)
                result["Cubic Spline"][scn].append(cs_m["corr_masked"])
                if timemae_model is not None:
                    tm_m = evaluate_timemae(timemae_model, test_loader, mask_gen, device, scenario=scn)
                    result["TimeMAE"][scn].append(tm_m["corr_masked"])
        except Exception as exc:
            print(f"  [baseline] S{sid:02d} skipped: {exc}")
    return result


def _infer_figure6(config: dict, out_dir: Path, force: bool):
    for sid in (3, 6):
        save = out_dir / f"figure6_rule_detector_S{sid:02d}.png"
        if _skip(save, force):
            print(f"  [skip] {save.name} exists")
            continue
        loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
        try:
            segments, _ = prepare_data_db3(loader, [sid], config)
        except Exception as exc:
            print(f"  [fig6] S{sid:02d} skipped: {exc}")
            continue
        det = RuleAnomalyDetector(patch_size=config.get("patch_size", 8)).fit(segments)
        win = segments[len(segments) // 2]
        plot_figure6_rule_detector(win, det, save, subject_label=f"S{sid:02d}")
        print(f"  [fig6] saved {save}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from experiment outputs")
    parser.add_argument("--figures", default="all",
                        help="Comma-separated list or 'all'")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--exp1-run", default=None, help="Path to exp1 run folder")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--infer", action="store_true",
                        help="Run model inference for fig2 (and baselines for fig3)")
    args = parser.parse_args()

    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    device = config["device"]

    out_dir = Path(args.out_dir) if args.out_dir else Path(config["paper_figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    exp1_run = Path(args.exp1_run) if args.exp1_run else Path(config["exp1_dir"])
    if not exp1_run.exists():
        print(f"  [exp1] current-run directory not found: {exp1_run}")
        exp1_run = None
    items = ALL_ITEMS if args.figures == "all" else [x.strip() for x in args.figures.split(",")]
    print("=" * 70)
    print(f"[Paper Figures] output: {out_dir}")
    if exp1_run:
        print(f"  exp1_run: {exp1_run}")
    print("=" * 70)

    # ── 图 2 ──
    if "fig2" in items and args.infer and exp1_run:
        print("`n[Figure 2] completion waveforms (inference)")
        _infer_figure2(exp1_run, config, device, out_dir, args.force)
    elif "fig2" in items:
        print("\n[Figure 2] skipped — use --infer or existing panels in exp1_run/figures/")

    # ── 图 3 ──
    if "fig3" in items and exp1_run:
        save = out_dir / "figure3_scenario_bars.png"
        if not _skip(save, args.force):
            mcia = collect_exp1_scenario_metrics(exp1_run)
            method_vals = {"MCIA": mcia}
            if args.infer:
                print("`n[Figure 3] evaluating baselines per scenario")
                bl = _eval_baselines_per_scenario(config, device, exp1_run)
                method_vals.update(bl)
            if any(len(v[s]) for v in method_vals.values() for s in SCENARIOS):
                plot_figure3_scenario_bars(method_vals, save)
                print(f"  [fig3] saved {save}")
            else:
                print("  [fig3] no per-scenario data")

    # ── 图 4 ──
    if "fig4" in items and exp1_run:
        save = out_dir / "figure4_training_curves.png"
        if not _skip(save, args.force):
            hist = _exp1_metric(exp1_run, "training_history.json")
            if plot_figure4_training_curves(hist, save):
                print(f"  [fig4] saved {save}")
            else:
                print(f"  [fig4] missing or empty {hist}")


    # ── 图 6 ──
    if "fig6" in items:
        print("`n[Figure 6] rule detector visualization")
        _infer_figure6(config, out_dir, args.force)

    # ── 图 8-10、表 3 — Exp3 ──
    exp3_path = Path(config["regressor_results_path"])
    exp3_report = _load_json(exp3_path) if exp3_path.exists() else None
    exp3_items = {"fig7", "fig8", "fig9", "table3"}
    if not exp3_report and any(item in items for item in exp3_items):
        print(f"  [exp3] missing report: {exp3_path}")
        print("  [exp3] run or resume scripts/04_eval_db3_angle_raw_vs_augmented.py before generating fig7/fig8/fig9/table3")

    if exp3_report and exp3_report.get("angle_target") != key10_target_metadata():
        raise ValueError("Legacy or incompatible Exp3 report: generate new Key10 results before paper figures.")

    if exp3_report and "fig7" in items:
        save = out_dir / "figure7_angle_traces.png"
        if not _skip(save, args.force):
            pred_dir = Path(config.get("run_dir", exp3_path.parent.parent.parent)) / "03_angle_prediction" / "predictions"
            # 使用首个有预测结果的受试者
            plotted = False
            for s in exp3_report.get("subjects", []):
                if s.get("status") != "ok":
                    continue
                sid = s["subject_id"]
                npz = pred_dir / f"S{sid:02d}_angle_predictions.npz"
                if npz.exists():
                    d = np.load(npz)
                    assert_key10_prediction_payload(d, str(npz))
                    preds = {}
                    if "pred_A" in d:
                        preds["A"] = d["pred_A"]
                    if "pred_B" in d:
                        preds["B"] = d["pred_B"]
                    if "pred_C" in d:
                        preds["C"] = d["pred_C"]
                    if preds and "target" in d:
                        plot_figure7_angle_traces(
                            d["target"], preds, save,
                            title=f"S{sid:02d} Key10 thumb MCP (dim 0)",
                        )
                        print(f"  [fig7] saved {save} (S{sid:02d})")
                        plotted = True
                        break
            if not plotted:
                print("  [fig7] no prediction npz found; run exp3 groups B/C first")

    if exp3_report and "fig8" in items:
        save = out_dir / "figure8_grouped_bars.png"
        if not _skip(save, args.force):
            if plot_figure8_grouped_bars(exp3_report, save):
                print(f"  [fig8] saved {save}")

    if exp3_report and "fig9" in items:
        save = out_dir / "figure9_improvement_scatter.png"
        if not _skip(save, args.force):
            if plot_figure9_improvement_scatter(exp3_report, save):
                print(f"  [fig9] saved {save}")
            else:
                print("  [fig9] need groups A+B in exp3 results")

    if exp3_report and "table3" in items:
        tbl = out_dir / "table3_exp3"
        export_table3_exp3(exp3_report, tbl)
        print(f"  [table3] saved {tbl}.csv / .html")

    # ── 表 1 — Exp1 ──
    if "table1" in items and exp1_run:
        fin_path = _exp1_metric(exp1_run, "finetuning_results.json")
        if fin_path.exists():
            fin_data = _load_json(fin_path)
            baseline_ps = _eval_baselines_per_subject(config, device, exp1_run) if args.infer else None
            export_table1_exp1(fin_data, baseline_ps, out_dir / "table1_exp1.csv")
            has_bl = any(r.get("baselines") for r in fin_data.get("results", []))
            note = "aligned masked metrics" if has_bl or baseline_ps else "MCIA only; rerun 01_train or use --infer"
            print(f"  [table1] saved {out_dir / 'table1_exp1.csv'} ({note})")

    # ── 表 2 — Exp2a ──
    if "table2" in items:
        ft_path = Path(config.get("transfer_checkpoints_dir",
                                   Path(config["checkpoints_dir"]) / "exp2_transfer_db3"))
        summary = ft_path / "finetune_summary.json"
        if summary.exists():
            export_table2_exp2a(_load_json(summary), out_dir / "table2_exp2a.csv")
            print(f"  [table2] saved {out_dir / 'table2_exp2a.csv'}")
        else:
            print(f"  [table2] waiting for {summary}")

    print(f"\nDone. Figures in: {out_dir}")


if __name__ == "__main__":
    main()
