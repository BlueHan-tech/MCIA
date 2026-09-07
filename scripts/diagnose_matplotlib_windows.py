"""Diagnose Matplotlib native crashes on Windows by isolating each step.

Each probe runs in a fresh child Python process so a native crash in one step
cannot prevent later probes from reporting their own status.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "matplotlib_diagnosis"


PROBES = [
    ("01_import_numpy", """
import numpy
print('numpy imported')
"""),
    ("02_import_matplotlib", """
import matplotlib
print('matplotlib imported')
"""),
    ("03_use_agg", """
import matplotlib
matplotlib.use('Agg')
print('backend', matplotlib.get_backend())
"""),
    ("04_import_pyplot", """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print('pyplot imported', plt.get_backend())
"""),
    ("05_plt_figure", """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()
print('figure created', type(fig).__name__)
"""),
    ("06_fig_add_subplot", """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
print('subplot created', type(ax).__name__)
"""),
    ("07_ax_plot", """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(10), np.arange(10))
print('plot created')
"""),
    ("08_ax_bar", """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(np.arange(5), np.arange(5))
print('bar created')
"""),
    ("09_canvas_draw", """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(10), np.arange(10))
fig.canvas.draw()
print('canvas drawn')
"""),
    ("10_savefig_png", """
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out = Path(r'{out_dir}') / 'test.png'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(10), np.arange(10))
fig.savefig(out)
print('saved', out)
"""),
    ("11_savefig_svg", """
from pathlib import Path
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import numpy as np
out = Path(r'{out_dir}') / 'test.svg'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(10), np.arange(10))
fig.savefig(out)
print('saved', out)
"""),
    ("12_savefig_pdf", """
from pathlib import Path
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
out = Path(r'{out_dir}') / 'test.pdf'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(10), np.arange(10))
fig.savefig(out)
print('saved', out)
"""),
]

BACKEND_PROBES = [
    ("backend_agg_draw", "Agg", "draw"),
    ("backend_agg_png", "Agg", "png"),
    ("backend_svg_save", "svg", "svg"),
    ("backend_pdf_save", "pdf", "pdf"),
]

DLL_NAMES = [
    "python.exe",
    "freetype.dll",
    "libpng16.dll",
    "zlib.dll",
    "mkl_rt.dll",
    "libiomp5md.dll",
]


def _run_child(name: str, code: str, timeout: int = 60) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    code = dedent(code).replace('{out_dir}', str(OUT_DIR))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return {
        "name": name,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _env_info() -> dict:
    info: dict[str, object] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "path_first_20": os.environ.get("PATH", "").split(os.pathsep)[:20],
    }
    code = """
import json
import importlib
info = {}
for name in ['matplotlib', 'numpy', 'PIL']:
    try:
        mod = importlib.import_module(name)
        info[name] = {
            'version': getattr(mod, '__version__', None),
            'file': getattr(mod, '__file__', None),
        }
    except Exception as exc:
        info[name] = {'error': repr(exc)}
try:
    import matplotlib
    info['matplotlib_backend'] = matplotlib.get_backend()
    info['matplotlib_cachedir'] = matplotlib.get_cachedir()
    info['matplotlib_data_path'] = matplotlib.get_data_path()
except Exception as exc:
    info['matplotlib_extra_error'] = repr(exc)
print(json.dumps(info, indent=2))
"""
    child = _run_child("env_packages", code)
    try:
        info["packages"] = json.loads(child["stdout"])
    except Exception:
        info["packages_probe"] = child
    return info


def _where(name: str) -> list[str]:
    exe = shutil.which("where")
    if not exe:
        return []
    proc = subprocess.run(
        [exe, name],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.stderr.strip():
        lines.append("stderr: " + proc.stderr.strip())
    return lines


def _dll_info() -> dict[str, list[str]]:
    return {name: _where(name) for name in DLL_NAMES}


def _font_cache_info(clear: bool) -> dict:
    code = """
from pathlib import Path
import json
import matplotlib
cache = Path(matplotlib.get_cachedir())
files = sorted(str(p) for p in cache.glob('fontlist*.json')) if cache.exists() else []
print(json.dumps({'cachedir': str(cache), 'fontlist_files': files}, indent=2))
"""
    before = _run_child("font_cache_before", code)
    parsed_before = json.loads(before["stdout"]) if before["returncode"] == 0 and before["stdout"] else {}
    removed: list[str] = []
    if clear:
        cache = Path(parsed_before.get("cachedir", ""))
        if cache.exists():
            for path in cache.glob("fontlist*.json"):
                try:
                    path.unlink()
                    removed.append(str(path))
                except Exception as exc:
                    removed.append(f"failed {path}: {exc!r}")
    after = _run_child("font_cache_after", code)
    return {"before": before, "after": after, "removed": removed}


def _backend_probe_code(backend: str, action: str) -> str:
    suffix = {"png": "png", "svg": "svg", "pdf": "pdf", "draw": "png"}[action]
    save_line = {
        "draw": "fig.canvas.draw(); print('draw ok')",
        "png": "fig.savefig(out); print('png save ok', out)",
        "svg": "fig.savefig(out); print('svg save ok', out)",
        "pdf": "fig.savefig(out); print('pdf save ok', out)",
    }[action]
    return f"""
from pathlib import Path
import matplotlib
matplotlib.use({backend!r})
import matplotlib.pyplot as plt
import numpy as np
out = Path(r'{{out_dir}}') / 'backend_{backend}_{action}.{suffix}'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(10), np.arange(10))
{save_line}
"""


def _print_result(result: dict) -> None:
    status = "OK" if result["returncode"] == 0 else "FAIL"
    print(f"[{status}] {result['name']} returncode={result['returncode']}")
    if result.get("stdout"):
        print("  stdout:")
        for line in result["stdout"].splitlines():
            print("    " + line)
    if result.get("stderr"):
        print("  stderr:")
        for line in result["stderr"].splitlines()[:30]:
            print("    " + line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Matplotlib native crashes on Windows")
    parser.add_argument("--clear-font-cache", action="store_true", help="Delete only fontlist*.json before rerunning probes")
    parser.add_argument("--json", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": _env_info(),
        "dll_where": _dll_info(),
        "font_cache": _font_cache_info(args.clear_font_cache),
        "probes": [],
        "backend_probes": [],
    }

    print("=== Environment ===")
    print(json.dumps(report["environment"], indent=2, ensure_ascii=False))
    print("\n=== DLL where ===")
    print(json.dumps(report["dll_where"], indent=2, ensure_ascii=False))
    print("\n=== Font cache ===")
    print(json.dumps(report["font_cache"], indent=2, ensure_ascii=False))

    print("\n=== Step probes ===")
    for name, code in PROBES:
        result = _run_child(name, code)
        report["probes"].append(result)
        _print_result(result)

    print("\n=== Backend probes ===")
    for name, backend, action in BACKEND_PROBES:
        result = _run_child(name, _backend_probe_code(backend, action))
        report["backend_probes"].append(result)
        _print_result(result)

    json_path = Path(args.json) if args.json else OUT_DIR / "diagnosis_report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON report: {json_path}")
    failed = [r for r in report["probes"] + report["backend_probes"] if r["returncode"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
