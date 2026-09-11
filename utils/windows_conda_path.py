"""为直接运行 Windows conda 环境 Python 规范化 DLL 搜索路径。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_current_env_dll_path() -> None:
    if os.name != "nt":
        return
    prefix = Path(sys.prefix)
    candidates = [
        prefix,
        prefix / "Library" / "mingw-w64" / "bin",
        prefix / "Library" / "usr" / "bin",
        prefix / "Library" / "bin",
        prefix / "Scripts",
        prefix / "bin",
    ]
    existing = [str(path) for path in candidates if path.exists()]
    if not existing:
        return
    old_parts = os.environ.get("PATH", "").split(os.pathsep)
    old_norm = {part.lower() for part in old_parts}
    prepend = [part for part in existing if part.lower() not in old_norm]
    os.environ["PATH"] = os.pathsep.join(prepend + old_parts)
    for part in existing:
        try:
            os.add_dll_directory(part)
        except (AttributeError, FileNotFoundError, OSError):
            pass
