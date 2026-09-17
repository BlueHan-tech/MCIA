"""已退役的论文图像入口，等待 DB3 实验重新设计。"""

from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Paper-figure generation is retired because its legacy DB3 panels and "
        "tables include invalid subject-adaptation results."
    )


if __name__ == "__main__":
    main()
