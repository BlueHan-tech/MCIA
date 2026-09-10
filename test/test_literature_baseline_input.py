"""Formula-level checks for SGMD-AAE paper-format input preparation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_literature_baseline_db2 import _normalize_each_sample


def main() -> None:
    values = np.array([
        [[-2.0, 0.0], [1.0, 4.0]],
        [[5.0, 5.0], [5.0, 5.0]],
    ], dtype=np.float32)
    normalized = _normalize_each_sample(values)
    assert np.isclose(normalized[0].min(), 0.0)
    assert np.isclose(normalized[0].max(), 1.0)
    assert np.isfinite(normalized).all()
    assert np.allclose(normalized[1], 0.0)
    print("ok: per-sample [0, 1] normalization")


if __name__ == "__main__":
    main()
