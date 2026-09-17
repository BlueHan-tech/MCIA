"""已退役的 DB3 被试适配入口。

旧实现对截肢被试自身的 DB3 sEMG 人工遮挡，并以未遮挡记录为目标优化重建。
该目标不能有效支持真实不可靠残肢通道的恢复，因此实现及其检查点已退役。
替代方案必须先实现协议规定的不依赖 DB3 真值的自监督目标。
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
