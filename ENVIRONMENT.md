# Miniforge 环境

本项目是 Python/PyTorch 科研代码库。仓库内提供以下 conda 环境文件：

- `environment.yml`：CPU 环境，是 Windows 上最稳妥的默认选择。
- `environment-gpu.yml`：NVIDIA GPU 环境，使用 PyTorch CUDA 12.8 wheel。

## Windows 一键配置

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_miniforge.ps1 -Variant cpu
```

使用 NVIDIA GPU 的机器执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_miniforge.ps1 -Variant gpu
```

RTX 5070 等 RTX 50 系列显卡应使用 GPU 版本。

如果尚未找到 `conda.exe`，脚本会把 Miniforge 安装到
`%USERPROFILE%\miniforge3`，随后创建或更新 `apple` 环境。

## 手动配置

安装 Miniforge 后执行：

```powershell
conda env create -f environment.yml
conda activate apple
```

NVIDIA GPU 环境则执行：

```powershell
conda env create -f environment-gpu.yml
conda activate apple
```

验证环境：

```powershell
python -c "import torch, numpy, scipy, pandas, yaml, sklearn, matplotlib; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 本机数据配置

已验证的数据集根目录（与 F:/A-SCI/B-SCI 共用）配置在 `config.yaml` 中：

- DB2: `F:/A-SCI/Ninapro-DB2`, with `DB2_s1/S1_E1_A1.mat`.
- DB3: `F:/A-SCI/Ninapro-DB3`, with `s5/DB3_s5/S5_E1_A1.mat`.
- 被试元数据：项目内的 `db2_metadata.csv`。
- 实验产物：`outputs/run/`；数据集直接从原位置读取。

本工作站已有可用的 `py311` conda 环境，其中包含 PyTorch 和 CUDA。请从项目
根目录启动；流水线各阶段使用与启动器相同的 Python 解释器。

先运行有界的环境检查（不会执行完整实验，也不会创建 run）：

```powershell
Set-Location E:\paper\MCIA
& D:\work\miniconda\Scripts\conda.exe run --no-capture-output -n py311 python -u scripts/check_data_setup.py
```

该检查会验证已配置的 MAT 文件和元数据，导入全部五个阶段，预处理真实的
DB2/DB3 样本，对每个数据库执行一个批次的 MCIA 训练和一个 Key10 TCN 批次，
检查手势数据划分，并写出一张 PNG 及 `outputs/setup_validation/report.json`。
其中的损失值仅用于验证代码能否执行，不代表实验结果或模型质量。

准备好运行可能持续数小时的完整实验后，执行：

```powershell
Set-Location E:\paper\MCIA
& D:\work\miniconda\Scripts\conda.exe run --no-capture-output -n py311 python -u scripts/run_all_experiments.py
```

新运行应保持 `MCIA_RUN_DIR` 未设置。启动器会创建带时间戳的输出目录，并依次
执行实验一、迁移、增强、角度估计和手势识别。仅在续跑已有 run 时显式设置
`MCIA_RUN_DIR`。全新检出应从实验一开始运行完整流水线；后续阶段依赖前序阶段
生成的检查点。
