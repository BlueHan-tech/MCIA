# Miniforge environment

This project runs as a Python/PyTorch research codebase. The checked-in conda
environment files are:

- `environment.yml`: CPU environment, safest default for Windows.
- `environment-gpu.yml`: NVIDIA GPU environment with PyTorch CUDA 12.8 wheels.

## One-command setup on Windows

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_miniforge.ps1 -Variant cpu
```

For an NVIDIA GPU machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_miniforge.ps1 -Variant gpu
```

Use the GPU variant for RTX 50-series cards such as RTX 5070.

The script installs Miniforge into `%USERPROFILE%\miniforge3` if `conda.exe` is
not already present, then creates or updates the `apple` environment.

## Manual setup

After installing Miniforge:

```powershell
conda env create -f environment.yml
conda activate apple
```

Or for NVIDIA GPU:

```powershell
conda env create -f environment-gpu.yml
conda activate apple
```

Verify the environment:

```powershell
python -c "import torch, numpy, scipy, pandas, yaml, sklearn, matplotlib; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
