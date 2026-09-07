# Matplotlib Windows Native Crash Diagnosis

This project has seen Windows native crashes while generating Matplotlib figures. A native crash can terminate the Python interpreter immediately, so normal `try/except` blocks are not enough to locate the failing operation.

Use `scripts/diagnose_matplotlib_windows.py`. It runs every probe in a separate child Python process, so one crash does not hide the result of later probes.

## How to Run

From the project root, in the target conda environment:

```powershell
C:\Users\zouyuhan.pat\miniforge3\envs\apple\python.exe scripts\diagnose_matplotlib_windows.py
```

To clear only Matplotlib font cache files first:

```powershell
C:\Users\zouyuhan.pat\miniforge3\envs\apple\python.exe scripts\diagnose_matplotlib_windows.py --clear-font-cache
```

The script writes `outputs/matplotlib_diagnosis/diagnosis_report.json` and test artifacts under `outputs/matplotlib_diagnosis/`.

## How to Interpret Probe Failures

- `import numpy` fails: NumPy installation, DLL loading, or binary ABI is broken.
- `import matplotlib` fails: Matplotlib package import or dependency loading is broken.
- `matplotlib.use("Agg")` fails: backend selection or Matplotlib configuration is broken.
- `import matplotlib.pyplot` fails: backend initialization or GUI/backend DLL loading is broken.
- `plt.figure()` or `fig.add_subplot()` fails: Matplotlib object creation, font manager, or backend setup is broken.
- `ax.plot()` fails: path creation or transform setup is broken.
- `ax.bar()` fails: patch/path handling is broken. This often points to Matplotlib transform/path renderer or a binary dependency.
- `fig.canvas.draw()` fails: the renderer itself is crashing before file output. Suspect FreeType, font cache, Matplotlib compiled extensions, NumPy ABI, or DLL conflicts.
- `fig.savefig("test.png")` fails but draw succeeds: suspect PNG raster output, Pillow, libpng, zlib, or FreeType.
- SVG/PDF save succeeds while PNG fails: suspect the PNG/Pillow/libpng path rather than the high-level plotting code.
- SVG/PDF also fail: suspect Matplotlib core rendering, fonts, transforms, or shared DLL conflicts.

## Backend Comparison

The script separately tests:

- `Agg` draw
- `Agg` PNG save
- `svg` save
- `pdf` save

If only Agg/PNG fails, focus on raster rendering and image dependencies. If `draw` fails, the problem is lower than PNG output.

## Font Cache Check

Matplotlib font caches can become stale or corrupt. The script prints `matplotlib.get_cachedir()` and lists `fontlist*.json`. With `--clear-font-cache`, it deletes only those font list JSON files, not the conda environment.

If clearing the cache changes the first failing probe, the font cache was likely involved.

## DLL Conflict Check

The script prints `where` results for:

- `python.exe`
- `freetype.dll`
- `libpng16.dll`
- `zlib.dll`
- `mkl_rt.dll`
- `libiomp5md.dll`

Multiple copies from different conda environments, system directories, Git/RTools, CUDA paths, or unrelated application folders are suspicious. The first one on `PATH` usually wins DLL loading.

## Clean Environment Comparison

Create a clean plotting environment and run the same script:

```powershell
conda create -n mcia_plot -c conda-forge python=3.10 numpy matplotlib pillow
conda activate mcia_plot
python scripts\diagnose_matplotlib_windows.py
```

Interpretation:

- `apple` fails but `mcia_plot` succeeds: the `apple` environment is polluted or binary-inconsistent.
- Both fail at the same point: suspect system-level fonts, DLL search path, permissions, graphics stack, or antivirus/security interference.
- Only PNG fails: prefer SVG/PDF for figures temporarily, and inspect Pillow/libpng/zlib.

## Project Fix

The project now fixes the observed crash by forcing the active conda environment's DLL directories to the front of the Windows DLL search path before importing Matplotlib. The full pipeline also prepends the selected Python environment paths to child-process `PATH`.

If the diagnosis script only passes when the environment DLL paths are first, keep the PATH/DLL fix and do not skip figure generation.
