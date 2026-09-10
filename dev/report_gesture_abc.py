"""报告 harness：S05/S06 开发被试上的手势识别 A/B/C（测试 repetitions，
legacy 已查看状态）。复用 05 的加载/掩码/训练/评估；C 用冻结 adapter 供体
先验（domain_id=1）。产物写入 06_diagnostics。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch

from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import (build_mcia, flatten_pipeline_config,
                                  load_mcia_state_dict, load_yaml_config, set_seed)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="5,6")
    args = parser.parse_args()
    subjects = [int(v) for v in args.subjects.split(",")]

    exp5 = load("exp5", PROJECT_ROOT / "scripts" / "05_eval_db3_gesture_raw_vs_augmented.py")
    tm = load("tm", PROJECT_ROOT / "scripts" / "run_task_matched_literature_baselines.py")
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    device = config["device"]
    run_dir = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    out = run_dir / "06_diagnostics" / "report_gesture_abc_20260910"
    out.mkdir(parents=True, exist_ok=True)

    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])

    base_ckpt = Path(config["exp1_dir"]) / "checkpoints" / "best_model.pth"
    b_model = build_mcia(config, device)
    load_mcia_state_dict(b_model, base_ckpt, device)
    b_model.eval()
    c_ckpt = run_dir / "08_amputee_donor_prior" / "checkpoints" / "C_adapter_lr1e-4.pth"
    if not c_ckpt.is_file():
        raise FileNotFoundError(c_ckpt)
    c_model = build_mcia(config, device)
    c06 = load("c06", PROJECT_ROOT / "scripts" / "06_adapt_amputee_donor_prior.py")
    for index in range(len(c_model.blocks) - 2, len(c_model.blocks)):
        c_model.blocks[index].adapter = c06.Adapter(c_model.embed_dim, 32).to(device)
    load_mcia_state_dict(c_model, c_ckpt, device,
                         required_state_prefixes=tuple(
                             f"blocks.{i}.adapter." for i in range(len(c_model.blocks) - 2, len(c_model.blocks))))
    c_model.eval()

    action_ids = [a for a in range(1, 49)]
    results = {"protocol": {
        "subjects": subjects, "exercises": [1, 2, 3], "actions": "1-48 (49 excluded)",
        "status_label": "development subjects, test repetitions legacy-viewed",
        "selection": "validation macro-F1 (protocol §6)",
        "c_checkpoint": str(c_ckpt),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, "subjects": {}}

    for sid in subjects:
        print(f"[gesture] DB3 S{sid:02d}", flush=True)
        raw_parts, mask_parts, label_parts, rep_parts = [], [], [], []
        for exercise in (1, 2, 3):
            one, _ = exp5.load_db3_windows(loader, sid, exercise, config,
                                           float(config["gesture_min_label_ratio"]),
                                           float(config["gesture_min_repetition_ratio"]))
            keep = np.isin(one.labels, action_ids)
            raw_parts.append(one.emg[keep])
            mask_parts.append(one.quality_mask[keep])
            label_parts.append(one.labels[keep])
            rep_parts.append(one.repetitions[keep])
        raw = np.concatenate(raw_parts)
        masks = np.concatenate(mask_parts)
        labels = np.concatenate(label_parts)
        reps = np.concatenate(rep_parts)
        pools = {"A": raw}
        for name, model, dom in (("B", b_model, None), ("C", c_model, 1)):
            completed, _ = exp5.apply_mcia(model, raw, masks, device,
                                           int(config["regressor_batch_size"]),
                                           patch_size=int(config["patch_size"]))
            if dom == 1:
                # apply_mcia（05 版）无 domain 参数，需要域条件时单独前向
                pass
            pools[name] = completed
        # C 需带 domain_id=1 的专用补全（05 的 apply_mcia 不接受 domain）
        from models.completion.mcia_core import derive_ch_mask_from_sample_mask
        c_completed = np.empty_like(raw)
        with torch.no_grad():
            for s in range(0, len(raw), int(config["regressor_batch_size"])):
                x = torch.as_tensor(raw[s:s + int(config["regressor_batch_size"])],
                                    dtype=torch.float32, device=device)
                m = torch.as_tensor(masks[s:s + int(config["regressor_batch_size"])],
                                    dtype=torch.float32, device=device)
                chan = (m.mean(dim=1) > 0.5).float()
                dom = torch.full((len(x),), 1, dtype=torch.long, device=device)
                pred = c_model(x * m, raw_time_mask=m, chan_valid_mask=chan, domain_id=dom)
                from utils.paper_pipeline import patch_boundary_crossfade
                pred = patch_boundary_crossfade(pred.clamp(0.0, 1.0), int(config["patch_size"]))
                c_completed[s:s + len(x)] = (pred * (1 - m) + x * m).cpu().numpy()
        pools["C"] = c_completed

        tr = np.flatnonzero(np.isin(reps, [1, 3, 4]))
        va = np.flatnonzero(reps == 6)
        te = np.flatnonzero(np.isin(reps, [2, 5]))
        subj = {}
        for name, pool in pools.items():
            set_seed(42)
            model, training = exp5.train_classifier(pool, labels, tr, va, config, device, 42)
            pred = exp5.predict(model, pool, te, int(config["regressor_batch_size"]), device)
            test_metrics = exp5.metrics(labels[te], pred, len(action_ids))
            trial = exp5.trial_majority_metrics(labels[te], pred, reps[te], len(action_ids))
            subj[name] = {"test_window": test_metrics, "test_trial_majority": trial}
            print(f"[S{sid:02d} {name}] macro-F1={test_metrics['macro_f1']:.4f} "
                  f"acc={test_metrics['accuracy']:.4f} trial_acc={trial['accuracy']:.4f}", flush=True)
        results["subjects"][str(sid)] = subj

    (out / "report_gesture_abc.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results={out / 'report_gesture_abc.json'}", flush=True)


if __name__ == "__main__":
    main()
