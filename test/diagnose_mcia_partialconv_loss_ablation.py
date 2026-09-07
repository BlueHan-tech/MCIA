"""Validation-only loss grid for the PartialConv MCIA output head.

The four candidates share one DB2 data load and identical ScenarioMix streams.
S04 is evaluated only once, after the winner has been fixed on S03 validation.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np
import torch

from data.dataset_db2_emg import prepare_data_db2
from data.ninapro_loader import NinaProDataLoader
from utils.loss_functions import EMGImputationLoss


_HEAD_PATH = PROJECT_ROOT / "test" / "diagnose_mcia_output_heads_ablation.py"
_HEAD_SPEC = importlib.util.spec_from_file_location("mcia_output_heads", _HEAD_PATH)
if _HEAD_SPEC is None or _HEAD_SPEC.loader is None:
    raise ImportError(_HEAD_PATH)
HEADS = importlib.util.module_from_spec(_HEAD_SPEC)
_HEAD_SPEC.loader.exec_module(HEADS)


LOSS_GRID = {
    "partial_ncc_0p5_char_1p0": {"w_ncc": 0.5, "w_charbonnier": 1.0},
    "partial_ncc_1p0_char_1p0": {"w_ncc": 1.0, "w_charbonnier": 1.0},
    "partial_ncc_1p5_char_1p0": {"w_ncc": 1.5, "w_charbonnier": 1.0},
    "partial_ncc_1p0_char_1p5": {"w_ncc": 1.0, "w_charbonnier": 1.5},
}


def make_criterion(config, weights, device):
    return EMGImputationLoss(
        w_charbonnier=weights["w_charbonnier"],
        w_ncc=weights["w_ncc"],
        w_stft=0.3,
        w_boundary=0.1,
        w_aux=0.1,
        aux_ratio=config.get("aux_mask_ratio", 0.10),
        fft_sizes=(16, 32, 64),
    ).to(device)


def weighted_corr(metrics):
    return sum(HEADS.SCENARIO_WEIGHTS[s] * metrics[s]["corr_masked"] for s in HEADS.SCENARIOS)


def run_candidate(name, weights, train_data, val_data, config, args, device):
    HEADS.set_seed(args.seed)
    model = HEADS.build_model("partial_conv2d", config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    criterion = make_criterion(config, weights, device)
    train_masks = HEADS.build_mask_generator(config, args.seed + 2000)
    train_loader = HEADS.make_loader(
        train_data, args.batch_size, True, args.seed + 3000
    )
    val_loader = HEADS.make_loader(
        val_data, args.batch_size, False, args.seed + 4000
    )
    best_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = HEADS.train_epoch(
            model, train_loader, optimizer, criterion, train_masks, device
        )
        val_s3 = HEADS.evaluate(
            model, val_loader, config, "s3", args.seed + 6002, device
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_s3": val_s3}
        )
        print(
            f"[{name}] {epoch:02d}/{args.epochs} loss={train_loss:.5f} "
            f"val_s3={val_s3['nrmse_masked']:.5f} corr={val_s3['corr_masked']:.4f}"
        )
        if val_s3["mse_masked"] < best_mse:
            best_mse = val_s3["mse_masked"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    validation = {
        scenario: HEADS.evaluate(
            model,
            val_loader,
            config,
            scenario,
            args.seed + 7000 + index,
            device,
        )
        for index, scenario in enumerate(HEADS.SCENARIOS)
    }
    return model, {
        "name": name,
        "loss_weights": weights,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.time() - started,
        "best_validation_s3_mse": best_mse,
        "weighted_validation_nrmse": HEADS.weighted_nrmse(validation),
        "weighted_validation_corr": weighted_corr(validation),
        "validation": validation,
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-windows", type=int, default=1024)
    parser.add_argument("--max-val-windows", type=int, default=256)
    parser.add_argument("--max-test-windows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corr-tolerance", type=float, default=0.005)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir_value = os.environ.get("MCIA_RUN_DIR")
    if not run_dir_value:
        raise RuntimeError("MCIA_RUN_DIR must point to an existing run directory")
    run_dir = Path(run_dir_value).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    output_dir = run_dir / "06_diagnostics" / "mcia_partialconv_loss_grid_20epoch"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "mcia_partialconv_loss_grid_20epoch_results.json"

    config = HEADS.load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} output={output_dir}")
    HEADS.set_seed(args.seed)
    data_loader = NinaProDataLoader(
        config["db2_path"], config["db3_path"], fs=config["orig_fs"]
    )
    loaded = {}
    for subject in (1, 2, 3, 4):
        windows, _, repetitions = prepare_data_db2(
            data_loader, [subject], config, exercises=[1]
        )
        loaded[subject] = (windows, repetitions)
    per_subject = max(1, args.max_train_windows // 2)
    train_data = np.concatenate(
        [
            HEADS.subset_windows(
                loaded[s][0],
                loaded[s][1],
                (1, 3, 4, 6),
                per_subject,
                args.seed + s,
            )
            for s in (1, 2)
        ]
    )
    val_data = HEADS.subset_windows(
        loaded[3][0], loaded[3][1], (2, 5), args.max_val_windows, args.seed + 30
    )
    test_data = HEADS.subset_windows(
        loaded[4][0], loaded[4][1], (2, 5), args.max_test_windows, args.seed + 40
    )
    print(
        f"windows train={len(train_data)} validation={len(val_data)} "
        f"test_held_out={len(test_data)}"
    )

    results = {}
    states = {}
    for name, weights in LOSS_GRID.items():
        model, result = run_candidate(
            name, weights, train_data, val_data, config, args, device
        )
        results[name] = result
        states[name] = copy.deepcopy(model.state_dict())
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"status": "running", "completed": list(results), "models": results},
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=True,
            )

    baseline_name = "partial_ncc_0p5_char_1p0"
    minimum_corr = results[baseline_name]["weighted_validation_corr"] - args.corr_tolerance
    eligible = [
        name
        for name in LOSS_GRID
        if results[name]["weighted_validation_corr"] >= minimum_corr
    ]
    winner = min(eligible, key=lambda name: results[name]["weighted_validation_nrmse"])

    HEADS.set_seed(args.seed)
    winner_model = HEADS.build_model("partial_conv2d", config).to(device)
    winner_model.load_state_dict(states[winner])
    test_loader = HEADS.make_loader(
        test_data, args.batch_size, False, args.seed + 5000
    )
    winner_test = {
        scenario: HEADS.evaluate(
            winner_model,
            test_loader,
            config,
            scenario,
            args.seed + 8000 + index,
            device,
        )
        for index, scenario in enumerate(HEADS.SCENARIOS)
    }
    results[winner]["test"] = winner_test
    results[winner]["weighted_test_nrmse"] = HEADS.weighted_nrmse(winner_test)
    results[winner]["weighted_test_corr"] = weighted_corr(winner_test)

    ranking = sorted(
        LOSS_GRID,
        key=lambda name: (
            results[name]["weighted_validation_nrmse"],
            -results[name]["weighted_validation_corr"],
        ),
    )
    output = {
        "status": "complete",
        "protocol": {
            "database": "DB2",
            "exercise": [1],
            "train_subjects": [1, 2],
            "validation_subjects": [3],
            "test_subjects": [4],
            "train_repetitions": [1, 3, 4, 6],
            "validation_repetitions": [2, 5],
            "test_repetitions": [2, 5],
            "epochs": args.epochs,
            "seed": args.seed,
            "data_loaded_once": True,
            "identical_mask_rng_per_candidate": True,
            "test_evaluated_for": [winner],
            "test_used_for_selection": False,
            "selection_rule": "lowest validation weighted NRMSE with weighted Corr no more than corr_tolerance below the current PartialConv baseline",
            "corr_tolerance": args.corr_tolerance,
            "minimum_eligible_corr": minimum_corr,
            "window_counts": {
                "train": len(train_data),
                "validation": len(val_data),
                "test": len(test_data),
            },
        },
        "winner": winner,
        "eligible": eligible,
        "ranking_by_validation_nrmse": ranking,
        "models": results,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=True)

    print("\nValidation loss-grid ranking")
    for rank, name in enumerate(ranking, 1):
        item = results[name]
        print(
            f"{rank}. {name:27s} nrmse={item['weighted_validation_nrmse']:.5f} "
            f"corr={item['weighted_validation_corr']:.4f} "
            f"eligible={name in eligible}"
        )
    print(
        f"winner={winner} test_nrmse={results[winner]['weighted_test_nrmse']:.5f} "
        f"test_corr={results[winner]['weighted_test_corr']:.4f}"
    )
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
