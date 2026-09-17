"""连续角度估计使用的固定 Key10 CyberGlove 目标契约。"""

from __future__ import annotations

from typing import Mapping

import numpy as np


KEY10_TARGET_NAME = "key10_cyberglove_mcp_pip"
KEY10_GLOVE_INDICES = [1, 2, 4, 5, 7, 8, 11, 12, 15, 16]
KEY10_CHANNEL_NAMES = (
    "Thumb MCP",
    "Thumb IP",
    "Index MCP",
    "Index PIP",
    "Middle MCP",
    "Middle PIP",
    "Ring MCP",
    "Ring PIP",
    "Little MCP",
    "Little PIP",
)
KEY10_DIM = len(KEY10_GLOVE_INDICES)

KEY10_SUBSETS = {
    "global": list(range(KEY10_DIM)),
    "mcp": [0, 2, 4, 6, 8],
    "pip": [1, 3, 5, 7, 9],
}

KEY10_PLOT_GROUPS = {
    "key10": {
        "shape": (5, 2),
        "figsize": (15.0, 12.0),
        "items": list(zip(KEY10_CHANNEL_NAMES, range(KEY10_DIM))),
    },
}


def key10_target_metadata() -> dict:
    """返回应写入每个新产物的可序列化目标定义。"""
    return {
        "name": KEY10_TARGET_NAME,
        "dimension": KEY10_DIM,
        "source_glove_indices": list(KEY10_GLOVE_INDICES),
        "source_index_base": 0,
        "channel_names": list(KEY10_CHANNEL_NAMES),
        "normalization": "per_subject_per_source_channel_minmax",
    }


def select_key10_angles(angle_values: np.ndarray) -> np.ndarray:
    """从原始或归一化的 22 通道 glove 数组中选择固定 Key10 目标。"""
    values = np.asarray(angle_values)
    if values.ndim < 1 or values.shape[-1] != 22:
        raise ValueError(
            "Key10 目标选择要求最后一维包含 22 个源 glove 通道；"
            f"实际形状为 {values.shape}。"
        )
    return values[..., list(KEY10_GLOVE_INDICES)]


def assert_key10_target(values: np.ndarray, context: str) -> None:
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[-1] != KEY10_DIM:
        raise ValueError(
            f"{context} 必须使用固定的 {KEY10_DIM} 维 {KEY10_TARGET_NAME} 目标；"
            f"实际形状为 {values.shape}。"
        )


def key10_prediction_metadata() -> dict:
    """返回描述固定预测目标的 NPZ 兼容数组。"""
    return {
        "angle_target_name": np.asarray(KEY10_TARGET_NAME),
        "angle_target_dimension": np.asarray(KEY10_DIM, dtype=np.int64),
        "angle_target_source_glove_indices": np.asarray(KEY10_GLOVE_INDICES, dtype=np.int64),
        "angle_target_channel_names": np.asarray(KEY10_CHANNEL_NAMES),
    }


def assert_key10_prediction_payload(payload: Mapping[str, np.ndarray], context: str) -> None:
    """拒绝旧版 22 维 NPZ 产物，而非静默从其继续运行。"""
    required = {
        "target",
        "angle_target_name",
        "angle_target_dimension",
        "angle_target_source_glove_indices",
        "angle_target_channel_names",
        "angle_data_schema",
        "source_exercises",
    }
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ValueError(
            f"{context} 是旧版或不完整的角度预测产物；缺少 {missing}。"
            "请为固定 Key10 目标创建新的 run。"
        )
    assert_key10_target(np.asarray(payload["target"]), f"{context} target")
    if str(np.asarray(payload["angle_data_schema"]).item()) != "e1_e2_exercise_separated_train_only_normalization_v1":
        raise ValueError(f"{context} 未使用当前 E1+E2 无泄漏数据模式。")
    if tuple(np.asarray(payload["source_exercises"], dtype=np.int64).tolist()) != (1, 2):
        raise ValueError(f"{context} 未声明已锁定的 E1+E2 源 exercise。")
    if str(np.asarray(payload["angle_target_name"]).item()) != KEY10_TARGET_NAME:
        raise ValueError(f"{context} 未声明 {KEY10_TARGET_NAME}。")
    if int(np.asarray(payload["angle_target_dimension"]).item()) != KEY10_DIM:
        raise ValueError(f"{context} 未声明 {KEY10_DIM} 维目标。")
    if tuple(np.asarray(payload["angle_target_source_glove_indices"], dtype=np.int64).tolist()) != tuple(KEY10_GLOVE_INDICES):
        raise ValueError(f"{context} 使用了不同的源 glove 通道映射。")
    if tuple(str(value) for value in np.asarray(payload["angle_target_channel_names"]).tolist()) != KEY10_CHANNEL_NAMES:
        raise ValueError(f"{context} 使用了不同的 Key10 通道名称。")
