"""
Legacy / pending redesign paper figure and table helpers.

This module reads real experiment outputs and renders the old paper-style
figures/tables. It is no longer used by the default run_all pipeline or the
default generate_figures_from_run rebuild flow.

Current default mainline figures are:
- 01 DB2 completion report;
- 03 DB3 12ch completion figures;
- 04 anatomy A/B/C angle figures.

This module intentionally mixes legacy rendering, table export, and metrics
aggregation helpers. Do not delete the whole file before cleanup separates those
responsibilities and verifies any remaining manual paper/table workflows.

Figure 1 architecture and Figure 5 transfer diagrams are manually drawn outside
this module. Existing training visualizations are skipped by paper scripts unless
--force is used.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = ("s1", "s2", "s3")
ANGLE_SUBSETS = ("global", "mcp", "pip")
GROUPS = ("A", "B", "C")
GROUP_LABELS = {"A": "Group A raw", "B": "Group B healthy-prior", "C": "Group C subject-ft"}


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _skip(path: Path, force: bool) -> bool:
    return path.exists() and not force


def _nanmean_std(vals: Sequence[float]) -> Tuple[float, float, int]:
    arr = np.asarray(vals, dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(valid)), float(np.std(valid)), len(valid)


def find_latest_exp1_run(output_dir: Path) -> Optional[Path]:
    root = output_dir / "exp1_mcia_db2"
    if not root.is_dir():
        return None
    runs = sorted(root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        if (run / "best_model.pth").exists() or (run / "training_history.json").exists():
            return run
    return runs[0] if runs else None


def collect_exp1_scenario_metrics(exp1_run: Path, test_subjects: Optional[range] = None) -> Dict[str, List[float]]:
    """从各被试 zero-shot 的 metrics.json 提取逐场景 corr_masked。

    若 test_subjects 为 None，则自动发现 exp1_run/figures/test/ 下全部被试目录，
    以兼容轻量（仅 S31）与完整运行。
    """
    out = {s: [] for s in SCENARIOS}
    test_dir = exp1_run / "figures" / "test"
    if not test_dir.is_dir():
        test_dir = exp1_run / "viz" / "test"
    if test_subjects is not None:
        subject_dirs = [test_dir / f"S{sid:02d}" for sid in test_subjects]
    else:
        subject_dirs = sorted(test_dir.glob("S??")) if test_dir.is_dir() else []

    for sdir in subject_dirs:
        mpath = sdir / "zeroshot" / "metrics.json"
        if not mpath.exists():
            continue
        data = _load_json(mpath)
        per_scn = data.get("metrics_per_scenario", {})
        for scn in SCENARIOS:
            if scn in per_scn and "corr_masked" in per_scn[scn]:
                out[scn].append(per_scn[scn]["corr_masked"])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# ── 图 2 — 四行补全波形（S1 + S4）──
# ──────────────────────────────────────────────────────────────────────────────

def plot_figure2_completion_waveform(
    emg_clean: np.ndarray,
    emg_completed: np.ndarray,
    mask: np.ndarray,
    save_path: Path,
    channel: int = 0,
    title: str = "",
) -> None:
    """四行：原始信号 | 掩码区 | 补全结果 | 真值。"""
    T = emg_clean.shape[0]
    t = np.arange(T)
    emg_masked = emg_clean * mask
    missing = mask[:, channel] < 0.5

    fig, axes = plt.subplots(4, 1, figsize=(10, 5.5), sharex=True)
    row_titles = ["Original", "Mask (observed only)", "MCIA completion", "Ground truth"]
    signals = [emg_clean[:, channel], emg_masked[:, channel],
               emg_completed[:, channel], emg_clean[:, channel]]
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#2ca02c"]

    for ax, sig, rt, col in zip(axes, signals, row_titles, colors):
        if missing.any():
            ax.fill_between(t, sig.min() - 0.02, sig.max() + 0.02,
                            where=missing, color="orange", alpha=0.15, zorder=0)
        ax.plot(t, sig, color=col, linewidth=1.0)
        ax.set_ylabel(rt, fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("sample")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, hspace=0.30)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# ── 图 3 — 场景对比柱状图 ──
# ──────────────────────────────────────────────────────────────────────────────

def plot_figure3_scenario_bars(
    method_scenario_vals: Dict[str, Dict[str, List[float]]],
    save_path: Path,
    metric_key: str = "corr_masked",
    ylabel: str = "corr_masked",
) -> None:
    """
    method_scenario_vals：{方法名: {s1: [逐被试数值], ...}}
    """
    methods = list(method_scenario_vals.keys())
    n_scn = len(SCENARIOS)
    x = np.arange(n_scn)
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for i, method in enumerate(methods):
        means, stds = [], []
        for scn in SCENARIOS:
            m, s, _ = _nanmean_std(method_scenario_vals[method].get(scn, []))
            means.append(m)
            stds.append(s)
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=method, color=colors[i % len(colors)], alpha=0.88, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in SCENARIOS])
    ax.set_ylabel(ylabel)
    ax.set_title("Completion quality by scenario")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, hspace=0.30)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# ── 图 4 — 训练收敛曲线 ──
# ──────────────────────────────────────────────────────────────────────────────

def plot_figure4_training_curves(history_path: Path, save_path: Path) -> bool:
    if not history_path.is_file():
        return False
    hist = _load_json(history_path)
    epochs = np.arange(1, len(hist.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        return False

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    has_parts = all(k in hist for k in ("train_loss_char", "train_loss_ncc", "train_loss_stft"))
    if has_parts:
        ax1.plot(epochs, hist["train_loss_char"], label="char", lw=1.5)
        ax1.plot(epochs, hist["train_loss_ncc"], label="ncc", lw=1.5)
        ax1.plot(epochs, hist["train_loss_stft"], label="stft", lw=1.5)
    else:
        ax1.plot(epochs, hist.get("train_loss", []), label="train_loss", lw=1.5, color="#1f77b4")
        if "val_criterion_loss" in hist:
            ax1.plot(epochs, hist["val_criterion_loss"], label="val_criterion", lw=1.2,
                     color="#ff7f0e", linestyle="--")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    scn_colors = {"s1": "#1f77b4", "s2": "#ff7f0e", "s3": "#2ca02c"}
    has_scn = all(f"val_corr_{s}" in hist for s in SCENARIOS)
    if has_scn:
        for scn in SCENARIOS:
            ax2.plot(epochs, hist[f"val_corr_{scn}"], label=scn.upper(),
                     lw=1.2, color=scn_colors[scn], alpha=0.85)
    elif "val_masked_corr" in hist:
        ax2.plot(epochs, hist["val_masked_corr"], label="val_corr_partial",
                 lw=1.2, color="#9467bd")
    ax2.set_ylabel("val corr_masked")
    ax2.legend(loc="upper right", fontsize=8)

    for tr in hist.get("curriculum_transitions", []):
        ep = tr.get("epoch", 0)
        if ep > 0:
            ax1.axvline(ep, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
            # 使用坐标轴比例变换，使文字始终贴在顶部，不受数据尺度影响
            ax1.text(ep, 1.0, f" S{tr.get('to_stage', '?')}",
                     fontsize=7, color="grey", va="top",
                     transform=ax1.get_xaxis_transform())

    ax1.set_title("Training convergence")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, hspace=0.30)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# ── 图 6 — 伪缺失曲线（corr_masked_partial vs ratio）──
# ──────────────────────────────────────────────────────────────────────────────

def plot_figure6_pseudo_missing(
    report: dict,
    save_path: Path,
    metric: str = "corr_masked_partial",
) -> bool:
    modes = report.get("modes", [])
    ratios = [float(r) for r in report.get("pseudo_missing_ratios", [])]
    if not modes or not ratios:
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"direct_transfer": "#1f77b4", "pretrained_finetuned": "#ff7f0e", "amputee_only": "#2ca02c"}
    for mode in modes:
        ys, ystds = [], []
        for ratio in ratios:
            vals = []
            for subj in report.get("subjects", []):
                m = subj.get("results", {}).get(mode, {}).get("metrics", {}).get(str(ratio), {})
                if m and metric in m:
                    vals.append(m[metric])
            mu, sd, _ = _nanmean_std(vals)
            ys.append(mu)
            ystds.append(sd)
        ax.errorbar(ratios, ys, yerr=ystds, marker="o", capsize=4,
                    label=mode, color=colors.get(mode, None), lw=1.8)

    ax.set_xlabel("pseudo-missing ratio")
    ax.set_ylabel(metric)
    ax.set_title("Exp2b pseudo-missing completion")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, hspace=0.30)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# ── 图 7 — 规则检测器步骤可视化 ──
# ──────────────────────────────────────────────────────────────────────────────

def _rule_masks_by_step(window: np.ndarray, detector) -> Dict[str, np.ndarray]:
    """将 RuleAnomalyDetector 逻辑分解为逐步二值掩码 (T, C)。"""
    T, C = window.shape
    P = T // detector.patch_size
    step0 = np.zeros((T, C), dtype=bool)
    step1 = np.zeros((T, C), dtype=bool)
    step2 = np.zeros((T, C), dtype=bool)

    for c in range(C):
        if detector.theta_dead[c]:
            step0[:, c] = True
            continue
        x = window[:, c]
        patch_mad = np.empty(P)
        for p in range(P):
            seg = x[p * detector.patch_size:(p + 1) * detector.patch_size]
            patch_mad[p] = np.mean(np.abs(seg - seg.mean()))
        is_low = patch_mad < detector.mad_threshold
        in_run = False
        run_start = 0
        for p in range(P + 1):
            if p < P and is_low[p]:
                if not in_run:
                    in_run = True
                    run_start = p
            else:
                if in_run:
                    if p - run_start >= detector.min_run_len:
                        lo, hi = run_start * detector.patch_size, p * detector.patch_size
                        step1[lo:hi, c] = True
                    in_run = False
        rms_w = np.sqrt(np.mean(x ** 2))
        if rms_w < detector.theta_weak[c]:
            step2[:, c] = True
    return {"step0_dead": step0, "step1_dropout": step1, "step2_weak": step2}


def plot_figure7_rule_detector(
    window: np.ndarray,
    detector,
    save_path: Path,
    subject_label: str = "",
    channels: Optional[List[int]] = None,
) -> None:
    """两行：原始 EMG + 分步彩色掩码。"""
    T, C = window.shape
    if channels is None:
        channels = list(range(min(C, 6)))
    steps = _rule_masks_by_step(window, detector)
    step_colors = {"step0_dead": "#d62728", "step1_dropout": "#ff7f0e", "step2_weak": "#bcbd22"}
    t = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    for ch in channels:
        axes[0].plot(t, window[:, ch], lw=0.8, alpha=0.85, label=f"Ch{ch+1}")
    axes[0].set_ylabel("EMG (norm)")
    axes[0].set_title(f"Raw signal {subject_label}")
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[0].grid(True, alpha=0.25)

    for ch in channels:
        axes[1].plot(t, np.zeros(T), color="grey", alpha=0.0)
        base = ch * 1.2
        for step_name, color in step_colors.items():
            m = steps[step_name][:, ch]
            if m.any():
                axes[1].fill_between(t, base, base + 1.0, where=m, color=color, alpha=0.85)
    axes[1].set_yticks([])
    axes[1].set_xlabel("sample")
    axes[1].set_title("Rule mask by step (red=dead, orange=dropout, yellow=weak)")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, hspace=0.30)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# -- Figure 8: single fixed-Key10 trace --
# ------------------------------------------------------------------------------

def plot_figure8_angle_traces(
    target: np.ndarray,
    preds: Dict[str, np.ndarray],
    save_path: Path,
    dim: int = 0,
    title: str = "",
) -> None:
    """Plot the first window for one fixed Key10 channel."""
    t = np.arange(target.shape[1])
    styles = {"A": ("#1f77b4", "Group A raw"),
              "B": ("#ff7f0e", GROUP_LABELS["B"]),
              "C": ("#2ca02c", GROUP_LABELS["C"])}
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, target[0, :, dim], color="black", lw=1.5, label="Ground truth")
    for grp, pred in preds.items():          # pred 为 ndarray (N,T,C)，不是元组
        if pred is None:
            continue
        col, label = styles.get(grp, ("#888", grp))
        ax.plot(t, pred[0, :, dim], color=col, lw=1.2, label=label, alpha=0.85)
    ax.set_xlabel("sample")
    ax.set_ylabel("normalized angle")
    ax.set_title(title or f"Key10 channel {dim}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, hspace=0.30)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# ── 图 9 — 分组柱状图（CC by 子集 × 组）──
# ──────────────────────────────────────────────────────────────────────────────

def plot_figure9_grouped_bars(exp3_report: dict, save_path: Path, metric: str = "pearson") -> bool:
    subjects = [s for s in exp3_report.get("subjects", []) if s.get("status") == "ok"]
    if not subjects:
        return False

    groups_present = [g for g in GROUPS
                      if any(g in s.get("groups", {}) for s in subjects)]
    if not groups_present:
        groups_present = ["A"]

    x = np.arange(len(ANGLE_SUBSETS))
    width = 0.8 / len(groups_present)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}

    for i, grp in enumerate(groups_present):
        means, stds = [], []
        for subset in ANGLE_SUBSETS:
            vals = [s["groups"][grp]["subsets"][subset][metric]
                    for s in subjects if grp in s.get("groups", {})
                    and subset in s["groups"][grp].get("subsets", {})]
            m, sd, _ = _nanmean_std(vals)
            means.append(m)
            stds.append(sd)
        offset = (i - (len(groups_present) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=GROUP_LABELS.get(grp, f"Group {grp}"), color=colors.get(grp, "#888"),
               alpha=0.88, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in ANGLE_SUBSETS])
    ax.set_ylabel(metric)
    ax.set_title("Angle estimation by subset and group")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, hspace=0.30)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# ── 图 10 — 散点图：基线 CC vs 提升（B - A）──
# ──────────────────────────────────────────────────────────────────────────────

def plot_figure10_improvement_scatter(exp3_report: dict, save_path: Path,
                                      subset: str = "global") -> bool:
    xs, ys, labels = [], [], []
    for s in exp3_report.get("subjects", []):
        if s.get("status") != "ok":
            continue
        g = s.get("groups", {})
        if "A" not in g or "B" not in g:
            continue
        cc_a = g["A"]["subsets"].get(subset, {}).get("pearson", float("nan"))
        cc_b = g["B"]["subsets"].get(subset, {}).get("pearson", float("nan"))
        if not (np.isfinite(cc_a) and np.isfinite(cc_b)):
            continue
        xs.append(cc_a)
        ys.append(cc_b - cc_a)
        labels.append(f"S{s['subject_id']:02d}")

    if not xs:
        return False

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, s=60, color="#1f77b4", edgecolors="white", zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_xlabel(f"Group A {subset} CC (baseline)")
    ax.set_ylabel(f"Group B 鈭?A {subset} CC")
    ax.set_title("Healthy-prior enhancement gain vs raw baseline quality")
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, hspace=0.30)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 表格
# ──────────────────────────────────────────────────────────────────────────────

def _baseline_metrics_from_results(
    finetuning_results: dict,
    baseline_per_subject: Optional[Dict[str, Dict[int, dict]]] = None,
) -> Dict[str, Dict[int, dict]]:
    """合并可选推理字典与 finetuning_results 中存储的基线结果。"""
    out: Dict[str, Dict[int, dict]] = {"TimeMAE": {}, "Cubic Spline": {}}
    if baseline_per_subject:
        for method, per_sid in baseline_per_subject.items():
            out.setdefault(method, {}).update(per_sid)
    for r in finetuning_results.get("results", []):
        sid = r.get("subject_id")
        if sid is None:
            continue
        for method, m in (r.get("baselines") or {}).items():
            if m:
                out.setdefault(method, {})[sid] = m
    return out


def export_table1_exp1(
    finetuning_results: dict,
    baseline_per_subject: Optional[Dict[str, Dict[int, dict]]],
    save_path: Path,
    test_subjects: range = range(33, 41),
) -> None:
    """
    行：S33-S40 + 均值±标准差。
    列：MCIA / TimeMAE / 三次样条 × corr_masked、corr_partial、mse_masked、mae_masked。
    所有方法均使用掩码区指标（与 evaluate_subject 一致）。
    """
    rows = []
    merged_baselines = _baseline_metrics_from_results(finetuning_results, baseline_per_subject)

    for sid in test_subjects:
        row = {"subject": f"S{sid:02d}"}
        for r in finetuning_results.get("results", []):
            if r.get("subject_id") == sid:
                zs = r.get("zeroshot", {})
                row["MCIA_corr_masked"] = zs.get("corr_masked", "")
                row["MCIA_corr_partial"] = zs.get("corr_masked_partial", "")
                row["MCIA_mse_masked"] = zs.get("mse_masked", "")
                row["MCIA_mae_masked"] = zs.get("mae_masked", "")
        for method in ("TimeMAE", "Cubic Spline"):
            m = merged_baselines.get(method, {}).get(sid, {})
            row[f"{method}_corr_masked"] = m.get("corr_masked", "")
            row[f"{method}_corr_partial"] = m.get("corr_masked_partial", "")
            row[f"{method}_mse_masked"] = m.get("mse_masked", "")
            row[f"{method}_mae_masked"] = m.get("mae_masked", "")
        rows.append(row)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)


def export_table2_exp2a(finetune_summary: dict, save_path: Path) -> None:
    rows = []
    for s in finetune_summary.get("subjects", []):
        if s.get("status") != "ok":
            continue
        pf = s.get("pretrained_finetuned", {})
        ao = s.get("amputee_only", {})
        pf_loss = pf.get("best_train_loss", float("nan"))
        ao_loss = ao.get("best_train_loss", float("nan"))
        rows.append({
            "subject": f"S{s['subject_id']:02d}",
            "pretrained_finetuned_loss": pf_loss,
            "amputee_only_loss": ao_loss,
            "delta": pf_loss - ao_loss if np.isfinite(pf_loss) and np.isfinite(ao_loss) else "",
        })
    if rows:
        pf_vals = [r["pretrained_finetuned_loss"] for r in rows if r["pretrained_finetuned_loss"] != ""]
        ao_vals = [r["amputee_only_loss"] for r in rows if r["amputee_only_loss"] != ""]
        rows.append({
            "subject": "mean",
            "pretrained_finetuned_loss": np.mean(pf_vals) if pf_vals else "",
            "amputee_only_loss": np.mean(ao_vals) if ao_vals else "",
            "delta": "",
        })
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)


def export_table3_exp3(exp3_report: dict, save_path: Path) -> None:
    """完整定量表；同时输出 HTML，按行高亮最优值。"""
    subjects = [s for s in exp3_report.get("subjects", []) if s.get("status") == "ok"]
    if not subjects:
        return

    metrics = ("pearson", "rmse", "r2")
    cols = []
    for grp in GROUPS:
        for subset in ANGLE_SUBSETS:
            for m in metrics:
                cols.append(f"{grp}_{subset}_{m}")

    rows_html = []
    csv_rows = []
    for s in subjects:
        row = {"subject": f"S{s['subject_id']:02d}"}
        for grp in GROUPS:
            if grp not in s.get("groups", {}):
                continue
            for subset in ANGLE_SUBSETS:
                sub = s["groups"][grp]["subsets"].get(subset, {})
                for m in metrics:
                    row[f"{grp}_{subset}_{m}"] = sub.get(m, "")
        csv_rows.append(row)

        # 在每个子集内，跨可用组高亮最优 Pearson 值
        cells = [f"<td>{row['subject']}</td>"]
        for subset in ANGLE_SUBSETS:
            pearsons = {}
            for grp in GROUPS:
                key = f"{grp}_{subset}_pearson"
                v = row.get(key, "")
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        pearsons[grp] = fv
                except (TypeError, ValueError):
                    pass
            best_grp = max(pearsons, key=pearsons.get) if pearsons else None
            for grp in GROUPS:
                for m in metrics:
                    key = f"{grp}_{subset}_{m}"
                    val = row.get(key, "")
                    style = ""
                    if m == "pearson" and grp == best_grp and val != "":
                        style = ' style="background:#c8e6c9;font-weight:bold"'
                    cells.append(f"<td{style}>{val if val != '' else 'NA'}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_rows:
        # 按字段结构构建完整字段列表，保证每行列数一致
        all_fields = ["subject"] + cols
        with open(save_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore",
                               restval="")
            w.writeheader()
            w.writerows(csv_rows)

    header = "<tr><th>Subject</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    html = (
        "<html><head><meta charset='utf-8'>"
        "<style>table{border-collapse:collapse;font-size:11px}"
        "td,th{border:1px solid #ccc;padding:4px 6px}</style></head>"
        f"<body><h3>表 3 — Exp3 定量结果</h3>"
        f"<table>{header}{''.join(rows_html)}</table></body></html>"
    )
    save_path.with_suffix(".html").write_text(html, encoding="utf-8")

