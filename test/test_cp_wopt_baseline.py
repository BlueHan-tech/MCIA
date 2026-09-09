"""Fast formula-level checks for the CP-WOPT literature baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines.cp_wopt import CPWOPTConfig, complete_cp_wopt, relative_mean_error


def main() -> None:
    rng = np.random.default_rng(7)
    factors = (rng.normal(size=(8, 1)), rng.normal(size=(6, 1)), rng.normal(size=(5, 1)))
    target = np.einsum("ir,jr,kr->ijk", *factors, optimize=True)
    observed = (rng.random(target.shape) > 0.10).astype(np.float64)
    masked = target * observed
    result = complete_cp_wopt(masked, observed, CPWOPTConfig(rank=1, max_iterations=160, seed=11))

    if not np.allclose(result.reconstruction[observed > 0.5], target[observed > 0.5]):
        raise AssertionError("CP-WOPT must preserve observed values in the delivered completion.")
    if len(result.objective_history) < 2 or result.objective_history[-1] >= result.objective_history[0]:
        raise AssertionError("CP-WOPT objective did not decrease on the rank-matched synthetic tensor.")
    missing = observed < 0.5
    missing_rme = relative_mean_error(result.reconstruction[missing], target[missing])
    if not np.isfinite(missing_rme) or missing_rme >= 1e-3:
        raise AssertionError(f"Unexpected CP-WOPT synthetic missing-entry RME: {missing_rme}")
    print(f"ok: iterations={result.iterations} evals={result.function_evaluations} missing_rme={missing_rme:.6f}")


if __name__ == "__main__":
    main()
