"""报告 harness：S05/S06 开发被试、测试 repetitions（legacy 已查看状态）上的
角度 A/B/C 全预算结果。复用 04 的训练/评估/连续输出；C 使用已冻结的
adapter 供体先验 checkpoint（domain_id=1）。产物写入 06_diagnostics。
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

from data.dataset_kinematics import make_rep_split, prepare_kinematics_data
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
    parser.add_argument("--c-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    exp3 = load("exp3", PROJECT_ROOT / "scripts" / "04_eval_db3_angle_raw_vs_augmented.py")
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    device = config["device"]
    run_dir = Path(os.environ["MCIA_RUN_DIR"]).resolve()
    out = run_dir / "06_diagnostics" / "report_angle_abc_20260910"
    out.mkdir(parents=True, exist_ok=True)

    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])

    base_ckpt = Path(config["exp1_dir"]) / "checkpoints" / "best_model.pth"
    b_model = build_mcia(config, device)
    load_mcia_state_dict(b_model, base_ckpt, device)
    b_model.eval()

    c_ckpt = args.c_checkpoint or run_dir / "08_amputee_donor_prior" / "checkpoints" / "C_adapter_lr1e-4.pth"
    if not c_ckpt.is_file():
        raise FileNotFoundError(c_ckpt)
    # C checkpoint 含 adapter 槽位，需挂载后加载（Adapter 与 06 脚本同源）
    c_model = build_mcia(config, device)
    c06 = load("c06", PROJECT_ROOT / "scripts" / "06_adapt_amputee_donor_prior.py")
    dim = c_model.embed_dim
    for index in range(len(c_model.blocks) - 2, len(c_model.blocks)):
        c_model.blocks[index].adapter = c06.Adapter(dim, 32).to(device)
    load_mcia_state_dict(c_model, c_ckpt, device,
                         required_state_prefixes=tuple(
                             f"blocks.{i}.adapter." for i in range(len(c_model.blocks) - 2, len(c_model.blocks))))
    c_model.eval()

    results = {"protocol": {
        "subjects": [int(v) for v in args.subjects.split(",")],
        "status_label": "development subjects, test repetitions already viewed (legacy)",
        "budget": "full 100 epochs / patience 20 (protocol §6)",
        "groups": {"A": "raw", "B": "healthy prior, domain_id=None",
                   "C": "frozen adapter donor prior, domain_id=1"},
        "c_checkpoint": str(c_ckpt), "base_checkpoint": str(base_ckpt),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, "subjects": {}}

    for sid in [int(v) for v in args.subjects.split(",")]:
        raw, angle, _, reps, wmeta = prepare_kinematics_data(
            loader, [sid], config, exercises=config["regressor_exercises"],
            db="db3", return_metadata=True)
        tr, va, te = make_rep_split(reps)
        masks = np.asarray(wmeta["quality_mask"], np.float32)
        pools = {"A": raw.copy()}
        for name, model, dom in (("B", b_model, None), ("C", c_model, 1)):
            idx = np.concatenate([tr, va, te])
            pools[name] = raw.copy()
            pools[name][idx] = exp3.apply_mcia(model, raw[idx], masks[idx], device,
                                               domain_id=dom, patch_size=int(config["patch_size"]))
        subj = {}
        for name, pool in pools.items():
            ckpt_dir = out / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            set_seed(42)
            model, best_val, _ = exp3.train_tcn_on_emg(pool, angle, tr, va, config, device)
            torch.save(model.state_dict(), ckpt_dir / f"S{sid:02d}_{name}.pth")
            pred, tgt = exp3.predict_on_set(model, pool, angle, te, config, device)
            subsets = exp3.evaluate_subsets(pred, tgt)
            subj[name] = {"best_val_loss": float(best_val), "test_subsets": subsets}
            print(f"[S{sid:02d} {name}] val={best_val:.5f} test RMSE={subsets['global']['rmse']:.5f} "
                  f"MAE={subsets['global']['mae']:.5f}", flush=True)
        results["subjects"][str(sid)] = subj

    summary = {}
    for sid, subj in results["subjects"].items():
        summary[sid] = {g: round(subj[g]["test_subsets"]["global"]["rmse"], 5) for g in subj}
        deltas = summary[sid]
        print(f"S{sid}: {deltas} | C-A={deltas.get('C', 0) - deltas.get('A', 0):+.5f} "
              f"B-A={deltas.get('B', 0) - deltas.get('A', 0):+.5f}")
    (out / "report_angle_abc.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results={out / 'report_angle_abc.json'}", flush=True)


if __name__ == "__main__":
    main()
