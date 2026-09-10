"""Retired DB3 subject-adapted augmentation entry point.

It previously consumed checkpoints trained with invalid pseudo-targets from
unmasked DB3 sEMG.  It is intentionally unavailable until the replacement
no-ground-truth DB3 adaptation objective is implemented and validated.
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Exp2c is retired: legacy DB3 subject-adaptation checkpoints must not "
        "generate augmented sEMG."
    )


if __name__ == "__main__":
    main()
