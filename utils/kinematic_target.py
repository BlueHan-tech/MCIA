"""Fixed Key10 CyberGlove target contract for continuous angle estimation."""

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
    """Return the serialized target definition saved with every new artifact."""
    return {
        "name": KEY10_TARGET_NAME,
        "dimension": KEY10_DIM,
        "source_glove_indices": list(KEY10_GLOVE_INDICES),
        "source_index_base": 0,
        "channel_names": list(KEY10_CHANNEL_NAMES),
        "normalization": "per_subject_per_source_channel_minmax",
    }


def select_key10_angles(angle_values: np.ndarray) -> np.ndarray:
    """Select the fixed Key10 target from a raw or normalized 22-channel glove array."""
    values = np.asarray(angle_values)
    if values.ndim < 1 or values.shape[-1] != 22:
        raise ValueError(
            "Key10 target selection requires a last dimension of 22 source glove channels; "
            f"received shape {values.shape}."
        )
    return values[..., list(KEY10_GLOVE_INDICES)]


def assert_key10_target(values: np.ndarray, context: str) -> None:
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[-1] != KEY10_DIM:
        raise ValueError(
            f"{context} must use the fixed {KEY10_DIM}-D {KEY10_TARGET_NAME} target; "
            f"received shape {values.shape}."
        )


def key10_prediction_metadata() -> dict:
    """Return NPZ-compatible arrays describing the fixed prediction target."""
    return {
        "angle_target_name": np.asarray(KEY10_TARGET_NAME),
        "angle_target_dimension": np.asarray(KEY10_DIM, dtype=np.int64),
        "angle_target_source_glove_indices": np.asarray(KEY10_GLOVE_INDICES, dtype=np.int64),
        "angle_target_channel_names": np.asarray(KEY10_CHANNEL_NAMES),
    }


def assert_key10_prediction_payload(payload: Mapping[str, np.ndarray], context: str) -> None:
    """Reject legacy 22-D NPZ artifacts instead of silently resuming them."""
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
            f"{context} is a legacy or incomplete angle-prediction artifact; missing {missing}. "
            "Create a new run for the fixed Key10 target."
        )
    assert_key10_target(np.asarray(payload["target"]), f"{context} target")
    if str(np.asarray(payload["angle_data_schema"]).item()) != "e1_e2_exercise_separated_train_only_normalization_v1":
        raise ValueError(f"{context} does not use the current E1+E2 no-leakage data schema.")
    if tuple(np.asarray(payload["source_exercises"], dtype=np.int64).tolist()) != (1, 2):
        raise ValueError(f"{context} does not declare the locked E1+E2 source exercises.")
    if str(np.asarray(payload["angle_target_name"]).item()) != KEY10_TARGET_NAME:
        raise ValueError(f"{context} does not declare {KEY10_TARGET_NAME}.")
    if int(np.asarray(payload["angle_target_dimension"]).item()) != KEY10_DIM:
        raise ValueError(f"{context} does not declare a {KEY10_DIM}-D target.")
    if tuple(np.asarray(payload["angle_target_source_glove_indices"], dtype=np.int64).tolist()) != tuple(KEY10_GLOVE_INDICES):
        raise ValueError(f"{context} uses a different source-glove channel mapping.")
    if tuple(str(value) for value in np.asarray(payload["angle_target_channel_names"]).tolist()) != KEY10_CHANNEL_NAMES:
        raise ValueError(f"{context} uses different Key10 channel names.")
