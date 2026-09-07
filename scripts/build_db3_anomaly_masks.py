"""
使用与场景对齐的规则检测器构建 DB3 异常掩码。

默认点击运行：
    - 加载 config.yaml 中配置的全部 DB3 被试
    - 逐被试拟合规则
    - 将掩码与规则元数据保存至 outputs/db3_anomaly_masks/

本脚本不运行 MCIA 补全，也不修改实验输出。
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_db3_emg import prepare_data_db3
from data.ninapro_loader import NinaProDataLoader
from utils.paper_pipeline import flatten_pipeline_config, load_yaml_config, save_json, set_seed
from utils.rule_anomaly_detector import LABEL_NAMES, RuleAnomalyDetector


def _jsonable_stats(stats: dict) -> dict:
    out = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = float(value)
    return out


def main() -> None:
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    set_seed(int(config.get("transfer_random_seed", 42)))

    out_dir = PROJECT_ROOT / "outputs" / "db3_anomaly_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = list(config.get("transfer_db3_subjects", range(1, 12)))
    loader = NinaProDataLoader(config["db2_path"], config["db3_path"], fs=config["orig_fs"])
    group_indices = config.get("group_indices")
    patch_size = int(config.get("patch_size", 8))

    manifest = {
        "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
        "subjects": [],
    }

    print("=" * 80)
    print("[DB3 anomaly masks] scenario-aligned rule detector")
    print("=" * 80)

    for subject_id in subjects:
        print(f"\n--- DB3 S{int(subject_id):02d} ---")
        try:
            segments, _ = prepare_data_db3(loader, [int(subject_id)], config)
            detector = RuleAnomalyDetector(
                patch_size=patch_size,
                group_indices=group_indices,
            ).fit(segments)
            result = detector.detect_batch(segments)

            save_path = out_dir / f"db3_S{int(subject_id):02d}_anomaly_masks.npz"
            np.savez_compressed(
                save_path,
                original=segments[:, : result["mask"].shape[1], :].astype(np.float32),
                mask=result["mask"],
                labels=result["labels"],
                patch_mask=result["patch_mask"],
                patch_labels=result["patch_labels"],
                unrecoverable_patches=result["unrecoverable_patches"],
                mask_ratio=result["mask_ratio"],
                unrecoverable_ratio=result["unrecoverable_ratio"],
                dead_channels=result["dead_channels"],
                label_codes=np.array(sorted(LABEL_NAMES), dtype=np.int32),
                label_names=np.array([LABEL_NAMES[i] for i in sorted(LABEL_NAMES)], dtype=object),
            )

            stats_path = out_dir / f"db3_S{int(subject_id):02d}_anomaly_stats.json"
            stats = {
                "subject_id": int(subject_id),
                "n_segments": int(len(segments)),
                "mask_ratio_mean": float(result["mask_ratio"].mean()),
                "mask_ratio_max": float(result["mask_ratio"].max()),
                "unrecoverable_ratio_mean": float(result["unrecoverable_ratio"].mean()),
                "dead_channels_1based": [int(c + 1) for c in result["dead_channels"]],
                "fit_stats": _jsonable_stats(detector.fit_stats_),
            }
            save_json(stats_path, stats)
            manifest["subjects"].append(
                {
                    "subject_id": int(subject_id),
                    "status": "ok",
                    "file": str(save_path),
                    "stats_file": str(stats_path),
                    "n_segments": int(len(segments)),
                    "mask_ratio_mean": stats["mask_ratio_mean"],
                    "dead_channels_1based": stats["dead_channels_1based"],
                }
            )
            print(
                f"  saved {save_path} | mask_mean={stats['mask_ratio_mean']:.2%} "
                f"| dead={stats['dead_channels_1based']}"
            )
        except Exception as exc:
            manifest["subjects"].append(
                {"subject_id": int(subject_id), "status": "skipped", "reason": str(exc)}
            )
            print(f"  skipped: {exc}")

    save_json(out_dir / "manifest.json", manifest)
    print(f"\nSaved DB3 anomaly masks to: {out_dir}")


if __name__ == "__main__":
    main()
