"""便于在 PyCharm 中从已有 run 重建图像的辅助入口。

在 PyCharm 中打开本文件并点击绿色运行按钮。它为选定 run 设置 MCIA_RUN_DIR，
再调用 scripts/generate_figures_from_run.py。不修改训练、预测、指标、检查点或
缓存数据。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 若要为其他 run 重新生成图像，只需修改这里的 run 编号。
RUN_ID = "run_20260722_153608_1"
INCLUDE_LEGACY_PAPER_FIGURES = False


def main() -> None:
    run_dir = PROJECT_ROOT / "outputs" / "run" / RUN_ID
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    os.chdir(PROJECT_ROOT)
    os.environ["MCIA_RUN_DIR"] = str(run_dir)

    argv_backup = sys.argv[:]
    sys.argv = ["scripts/generate_figures_from_run.py"]
    if INCLUDE_LEGACY_PAPER_FIGURES:
        sys.argv.append("--include-legacy-paper-figures")
    try:
        from scripts.generate_figures_from_run import main as regenerate_main

        regenerate_main()
    finally:
        sys.argv = argv_backup


if __name__ == "__main__":
    main()
