"""Time-bounded paired range-penalty screen, train/validation only.

Two MCIA models with identical random initialization are trained on the same
DB2 windows and masks; the only difference is the training-loss soft [0,1]
range penalty (baseline w=0 vs w=0.5). Evaluation applies the adopted
delivery rule (clip -> patch-boundary cross-fade -> copy-back) identically to
both arms, and additionally reports the pre-clamp overshoot ratio to verify
the penalty's mechanism. No test subjects, no checkpoint selection; last
fully completed paired epoch decides. Writes only to <run>/06_diagnostics/.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch
import yaml

from data.ninapro_loader import NinaProDataLoader
from data.dataset_kinematics import prepare_kinematics_data
from models.completion.mask_generators import ScenarioMixMaskGenerator
from utils.paper_pipeline import (
    build_mcia,
    build_structural_loss,
    patch_boundary_crossfade,
    set_seed,
)

ARMS = ("baseline_w0", "range_w0.5")
RANGE_WEIGHTS = {"baseline_w0": 0.0, "range_w0.5": 0.5}


def main():
    started = time.monotonic()
    deadline = started + 420
    run = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    out = run / "06_diagnostics" / "range_penalty_screen_20260909"
    out.mkdir(parents=True, exist_ok=False)
    root = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = dict(root["signal"], **root["exp1_mcia"])
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for the bounded comparison")
    torch.set_num_threads(4)
    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loader = NinaProDataLoader(root["paths"]["db2"], root["paths"]["db3"], fs=cfg["orig_fs"])
    x, _, subjects, reps, metadata = prepare_kinematics_data(
        loader, [1, 2], cfg, exercises=(1,), db="db2", return_metadata=True)
    rng = np.random.default_rng(42)
    selected = {}
    for split, allowed, count in [("train", [1, 3, 4], 128), ("validation", [6], 32)]:
        selected[split] = np.concatenate([
            np.sort(rng.choice(np.flatnonzero((subjects == s) & np.isin(reps, allowed)),
                               count, replace=False)) for s in [1, 2]])
    train = torch.tensor(x[selected["train"]], device=device)
    val = torch.tensor(x[selected["validation"]], device=device)
    assert not np.isin(reps[np.concatenate(list(selected.values()))], [2, 5]).any()
    np.savez_compressed(out / "selected_windows.npz", **selected,
                        subjects=subjects, repetitions=reps, starts=metadata["start"])

    def masks(seed):
        return ScenarioMixMaskGenerator(n_channels=cfg["n_channels"], time_steps=cfg["window_size"],
            patch_size=cfg["patch_size"], group_indices=cfg.get("group_indices"),
            min_alive_per_group=cfg.get("min_alive_per_group"), scenario_weights=cfg.get("scenario_weights"),
            scenario_params=cfg.get("scenario_params"), rng=np.random.default_rng(seed))

    scenarios = ["s1", "s2", "s3"]
    fixed_masks = {s: masks(7000 + i).generate_mask(val, scenario=s) for i, s in enumerate(scenarios)}
    set_seed(42)
    base = build_mcia(cfg, device)
    models = {name: copy.deepcopy(base) for name in ARMS}
    del base
    optimizers = {n: torch.optim.AdamW(m.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
                  for n, m in models.items()}
    criteria = {}
    for name in ARMS:
        criterion = build_structural_loss(cfg, device)
        if criterion is None:
            raise RuntimeError("This screen expects the configured structural loss")
        criterion.range_penalty_weight = RANGE_WEIGHTS[name]
        criteria[name] = criterion
    protocol = dict(database="DB2", subjects=[1, 2], exercise=1, train_repetitions=[1, 3, 4],
                    validation_repetitions=[6], test_evaluated=False, train_windows=256, val_windows=64,
                    seed=42, batch_size=16, planned_epochs=8, learning_rate=cfg["learning_rate"],
                    initialization="identical random weights; only range_penalty_weight differs",
                    range_penalty_weights=RANGE_WEIGHTS,
                    evaluation="adopted delivery rule: clamp + patch-boundary crossfade + copy-back",
                    comparison="last fully completed paired epoch; no checkpoint selection",
                    preprocessing="project prepare_kinematics_data; train-only scaling; full-record filtering",
                    scope="short-budget reconstruction screen; no downstream or independent test claim",
                    configuration=cfg, gpu=torch.cuda.get_device_name(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report = {"protocol": protocol, "history": [], "status": "running"}

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")

    @torch.no_grad()
    def evaluate(model):
        model.eval()
        result = {}
        for scenario, mask in fixed_masks.items():
            chunks, overshoot_counts, overshoot_total = [], 0, 0
            for i in range(0, len(val), 16):
                y, m = val[i:i+16], mask[i:i+16]
                raw = model(y * m, raw_time_mask=m)
                overshoot_counts += int((raw > 1.0).sum().item())
                overshoot_total += int(raw.numel())
                pred = patch_boundary_crossfade(raw.clamp(0, 1), cfg["patch_size"])
                completed = pred * (1 - m) + y * m
                assert torch.equal(completed[m.bool()], y[m.bool()])
                chunks.append(completed)
            p = torch.cat(chunks).cpu().numpy()
            y_np, missing = val.cpu().numpy(), mask.cpu().numpy() < 0.5
            delta = (p - y_np)[missing]
            correlations = []
            for j in range(len(p)):
                for c in range(p.shape[2]):
                    v = missing[j, :, c]
                    if v.sum() >= 4 and np.std(p[j, v, c]) > 1e-8 and np.std(y_np[j, v, c]) > 1e-8:
                        correlations.append(float(np.corrcoef(p[j, v, c], y_np[j, v, c])[0, 1]))
            result[scenario] = {
                "rmse": float(np.sqrt(np.mean(delta ** 2))),
                "mae": float(np.mean(np.abs(delta))),
                "corr": float(np.mean(correlations)) if correlations else None,
                "overshoot_ratio_pre_clamp": overshoot_counts / max(1, overshoot_total),
            }
        result["weighted_rmse"] = sum(w * result[s]["rmse"] for w, s in zip([.2, .4, .4], scenarios))
        result["weighted_mae"] = sum(w * result[s]["mae"] for w, s in zip([.2, .4, .4], scenarios))
        result["weighted_corr"] = sum(w * result[s]["corr"] for w, s in zip([.2, .4, .4], scenarios))
        return result

    save()
    for epoch in range(1, 9):
        if time.monotonic() > deadline - 35:
            break
        order = np.random.default_rng(42 + epoch).permutation(len(train))
        generator = masks(4200 + epoch)
        losses = {n: [] for n in ARMS}
        for m in models.values():
            m.train()
        for name in ARMS:
            criteria[name].set_epoch(epoch)
        for step, i in enumerate(range(0, len(order), 16)):
            y = train[order[i:i+16]]
            mask = generator.generate_mask(y)
            for name in ARMS:
                set_seed(42000 + epoch * 100 + step)
                pred = models[name](y * mask, raw_time_mask=mask, return_aux=True)
                loss, _ = criteria[name](pred, y, mask)
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(models[name].parameters(), 1.0)
                optimizers[name].step()
                losses[name].append(float(loss))
            if time.monotonic() > deadline:
                report["status"] = "time_limit_mid_epoch; use preceding complete epoch only"
                save()
                print(report["status"], flush=True)
                return
        metrics = {}
        for name in ARMS:
            metrics[name] = evaluate(models[name])
        report["history"].append(dict(epoch=epoch, train_loss={n: float(np.mean(v)) for n, v in losses.items()},
                                      validation=metrics))
        save()
        elapsed = report["elapsed_seconds"]
        line = " ".join(
            f"{n}: RMSE={v['weighted_rmse']:.5f} corr={v['weighted_corr']:.4f} "
            f"overshoot={v['s3']['overshoot_ratio_pre_clamp']:.2%}"
            for n, v in metrics.items()
        )
        print(f"epoch={epoch} elapsed={elapsed:.1f}s {line}", flush=True)
    report["status"] = "complete" if len(report["history"]) == 8 else "bounded_partial"
    save()
    if report["history"]:
        last = report["history"][-1]["validation"]
        b, r = last["baseline_w0"], last["range_w0.5"]
        print("\nlast paired epoch summary:")
        for key in ("weighted_rmse", "weighted_mae", "weighted_corr"):
            print(f"  {key}: baseline={b[key]:.5f} range={r[key]:.5f} "
                  f"delta={r[key] - b[key]:+.5f}")
        print(f"  pre-clamp overshoot (s3): baseline={b['s3']['overshoot_ratio_pre_clamp']:.2%} "
              f"range={r['s3']['overshoot_ratio_pre_clamp']:.2%}")
    print(f"results={out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
