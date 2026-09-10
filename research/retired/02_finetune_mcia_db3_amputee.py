"""Retired DB3 subject-adaptation entry point.

The former implementation synthetically masked an amputee subject's own DB3
sEMG and optimized reconstruction against the unmasked recording.  That is not
a valid target for recovery of genuinely unreliable residual-limb channels, so
the implementation and its checkpoints are retired.  A replacement must first
implement the protocol's no-DB3-ground-truth self-supervised objective.
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Exp2 DB3 subject adaptation is retired: do not synthetically mask DB3 "
        "sEMG and use the original DB3 recording as its reconstruction target. "
        "Existing amputee_only/pretrained_finetuned checkpoints are legacy and "
        "must not be used."
    )


if __name__ == "__main__":
    main()
