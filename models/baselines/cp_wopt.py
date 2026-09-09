"""CP-WOPT tensor completion from Akmal et al. (IEEE Access, 2019).

The implementation minimizes the paper's observed-entry objective

    1 / 2 || W * (X - [[A, B, C]]) ||_F^2,

where ``W`` is one for observed entries and zero for missing entries.  The
factor update is nonlinear conjugate gradient with the Hestenes--Stiefel
coefficient.  This module deliberately does not inspect targets or labels:
the caller supplies only an EMG tensor and its observation mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CPWOPTConfig:
    """Published solver settings, with rank supplied by the development plan."""

    rank: int
    max_iterations: int = 1000
    max_function_evaluations: int = 10000
    relative_objective_tolerance: float = 1e-8
    line_search_shrink: float = 0.5
    armijo_c1: float = 1e-4
    seed: int = 42


@dataclass
class CPWOPTResult:
    reconstruction: np.ndarray
    factors: tuple[np.ndarray, np.ndarray, np.ndarray]
    objective_history: list[float]
    iterations: int
    function_evaluations: int
    converged: bool


def relative_mean_error(estimate: np.ndarray, target: np.ndarray) -> float:
    """RME (Eq. 14): ||X - X_hat||_F / ||X||_F."""
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if estimate.shape != target.shape:
        raise ValueError(f"Shape mismatch: estimate={estimate.shape}, target={target.shape}")
    denominator = float(np.linalg.norm(target.ravel()))
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.linalg.norm((target - estimate).ravel()) / denominator)


def _validate_inputs(values: np.ndarray, observed_mask: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    observed_mask = np.asarray(observed_mask, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"CP-WOPT requires a third-order tensor, got {values.shape}")
    if values.shape != observed_mask.shape:
        raise ValueError(f"values/mask shape mismatch: {values.shape} vs {observed_mask.shape}")
    if not np.isfinite(values[observed_mask > 0.5]).all():
        raise ValueError("Observed tensor entries must be finite.")
    if not 1 <= int(rank) <= min(values.shape):
        raise ValueError(f"rank must be in [1, {min(values.shape)}], got {rank}")
    return values, (observed_mask > 0.5).astype(np.float64, copy=False)


def _reconstruct(factors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return np.einsum("ir,jr,kr->ijk", *factors, optimize=True)


def _pack(factors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return np.concatenate([factor.ravel() for factor in factors])


def _unpack(vector: np.ndarray, shape: tuple[int, int, int], rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cursor = 0
    factors = []
    for size in shape:
        width = size * rank
        factors.append(vector[cursor:cursor + width].reshape(size, rank))
        cursor += width
    return tuple(factors)  # type: ignore[return-value]


def _objective_and_gradient(
    vector: np.ndarray,
    values: np.ndarray,
    observed_mask: np.ndarray,
    rank: int,
) -> tuple[float, np.ndarray]:
    factors = _unpack(vector, values.shape, rank)
    residual = observed_mask * (_reconstruct(factors) - values)
    objective = 0.5 * float(np.sum(residual * residual))
    a, b, c = factors
    gradient = (
        np.einsum("ijk,jr,kr->ir", residual, b, c, optimize=True),
        np.einsum("ijk,ir,kr->jr", residual, a, c, optimize=True),
        np.einsum("ijk,ir,jr->kr", residual, a, b, optimize=True),
    )
    return objective, _pack(gradient)


def _random_factors(shape: tuple[int, int, int], rank: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    factors = []
    for size in shape:
        factor = rng.standard_normal((size, rank))
        factor /= np.maximum(np.linalg.norm(factor, axis=0, keepdims=True), np.finfo(np.float64).eps)
        factors.append(factor)
    return tuple(factors)  # type: ignore[return-value]


def fit_cp_wopt(values: np.ndarray, observed_mask: np.ndarray, config: CPWOPTConfig) -> CPWOPTResult:
    """Fit weighted CP factors using the paper's Hestenes--Stiefel NCG rule."""
    values, observed_mask = _validate_inputs(values, observed_mask, config.rank)
    vector = _pack(_random_factors(values.shape, config.rank, config.seed))
    objective, gradient = _objective_and_gradient(vector, values, observed_mask, config.rank)
    evaluations = 1
    direction = -gradient
    history = [objective]
    converged = False

    for iteration in range(1, config.max_iterations + 1):
        if evaluations >= config.max_function_evaluations:
            break
        directional_derivative = float(np.dot(gradient, direction))
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            direction = -gradient
            directional_derivative = -float(np.dot(gradient, gradient))
        step = 1.0
        accepted = False
        while evaluations < config.max_function_evaluations:
            candidate = vector + step * direction
            candidate_objective, candidate_gradient = _objective_and_gradient(
                candidate, values, observed_mask, config.rank
            )
            evaluations += 1
            if np.isfinite(candidate_objective) and candidate_objective <= objective + config.armijo_c1 * step * directional_derivative:
                accepted = True
                break
            step *= config.line_search_shrink
            if step < np.finfo(np.float64).eps:
                break
        if not accepted:
            break

        relative_change = abs(objective - candidate_objective) / max(abs(objective), 1.0)
        previous_gradient = gradient
        previous_direction = direction
        vector, objective, gradient = candidate, candidate_objective, candidate_gradient
        history.append(objective)
        if relative_change <= config.relative_objective_tolerance:
            converged = True
            break

        delta_gradient = gradient - previous_gradient
        denominator = float(np.dot(previous_direction, delta_gradient))
        beta_hs = float(np.dot(gradient, delta_gradient) / denominator) if abs(denominator) > 1e-20 else 0.0
        direction = -gradient + max(0.0, beta_hs) * previous_direction

    factors = _unpack(vector, values.shape, config.rank)
    return CPWOPTResult(
        reconstruction=_reconstruct(factors),
        factors=factors,
        objective_history=history,
        iterations=len(history) - 1,
        function_evaluations=evaluations,
        converged=converged,
    )


def complete_cp_wopt(values: np.ndarray, observed_mask: np.ndarray, config: CPWOPTConfig) -> CPWOPTResult:
    """Fit CP-WOPT then copy observed input values back into the delivered tensor."""
    values, observed_mask = _validate_inputs(values, observed_mask, config.rank)
    result = fit_cp_wopt(values, observed_mask, config)
    result.reconstruction = result.reconstruction * (1.0 - observed_mask) + values * observed_mask
    return result
