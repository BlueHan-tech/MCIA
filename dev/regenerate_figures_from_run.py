"""PyCharm-run helper for rebuilding figures from an existing run.

Open this file in PyCharm and click the green Run button. It sets MCIA_RUN_DIR
for the selected run, then delegates to scripts/generate_figures_from_run.py.
No training, prediction, metrics, checkpoints, or cached data are modified.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Change only this run id when you want to regenerate figures for another run.
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
