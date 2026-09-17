"""已退役的 DB3 被试适配增强入口。

旧实现读取以未遮挡 DB3 sEMG 作为无效伪目标训练的检查点。
在替代的不依赖真值的 DB3 适配目标实现并验证之前，本入口有意保持不可用。
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Exp2c is retired: legacy DB3 subject-adaptation checkpoints must not "
        "generate augmented sEMG."
    )


if __name__ == "__main__":
    main()
