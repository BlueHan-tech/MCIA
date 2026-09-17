"""已退役的表格生成入口。

旧实现从历史被试适配检查点导出 DB3 人工遮挡得分。
这些得分不能证明真实不可靠残肢 sEMG 的恢复效果，不得重新生成。
"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Table generation is retired pending redesigned, protocol-valid DB3 "
        "adaptation and downstream evaluation results."
    )


if __name__ == "__main__":
    main()
