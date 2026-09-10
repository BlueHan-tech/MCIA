"""Retired table-generation entry point.

The prior implementation exported DB3 artificial-masking scores from legacy
subject-adaptation checkpoints.  Those scores are not evidence for recovery of
genuinely unreliable residual-limb sEMG and must not be regenerated.
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Table generation is retired pending redesigned, protocol-valid DB3 "
        "adaptation and downstream evaluation results."
    )


if __name__ == "__main__":
    main()
