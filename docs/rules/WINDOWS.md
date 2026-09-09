# Windows / Conda / DLL / Encoding

本文件是 Windows 环境与编码规则的唯一规范来源。不要把环境故障误判为算法问题。绘图验证见 [PLOTTING.md](PLOTTING.md)。

## Windows / Conda 运行规则

1. 直接调用 `envs\<env>\python.exe` 不等于激活 conda 环境。
2. 在 Windows 下，任何会导入 Matplotlib、NumPy、PyTorch 等 native 包的主脚本，都必须先确保当前 conda env 的 DLL 路径优先。
3. `utils/windows_conda_path.py` 是根因修复，应保留，并且必须在 `import matplotlib` 之前调用。
4. `scripts/run_all_experiments.py` 启动子进程时，必须把所选 Python 环境的以下路径放到 `PATH` 最前：
   - env root
   - `Library\mingw-w64\bin`
   - `Library\usr\bin`
   - `Library\bin`
   - `Scripts`
   - `bin`
5. 不要依赖当前终端里的 base conda、JDK、Git、CUDA、RTools 等路径顺序。

## Windows 中文编码规则

1. Windows 下不要在 PowerShell heredoc、`python -c`、stdin 脚本中直接写中文 literal。
2. 需要在 Python 脚本里写中文时，优先使用 Unicode 转义，或从 UTF-8 文件读取中文内容。
3. 校验中文内容时也必须使用 Unicode 转义，避免校验字符串在进入 Python 前被 PowerShell 编码破坏。
4. 禁止使用 `echo "中文" >> file.md`、`>`、`>>` 这类重定向追加中文内容。
5. 使用 PowerShell 写中文文件时，必须显式指定 `-Encoding utf8`。
6. 写入 Markdown 后，必须用 UTF-8 方式重新读取并校验关键内容。
7. 修改 `CHANGELOG.md`、`AGENTS.md`、中文文档后，必须扫描乱码：`rg -n "\x{FFFD}|\x3F\x3F\x3F" CHANGELOG.md AGENTS.md docs`。
8. 如果发现三个连续问号，不要尝试自动恢复编码；通常原始中文已经丢失，应根据上下文手动恢复原句。
9. `apply_patch` 可用时优先用 `apply_patch` 写 Markdown；如果 Windows 沙箱导致 `apply_patch` 不可用，再使用受限 Python 脚本。
10. 受限 Python 脚本写中文时，脚本源码内中文必须用 Unicode 转义，文件读写必须显式 `encoding="utf-8"`。

## 受限编辑

1. 在 Windows 上避免长 `python -c` 修改源码；PowerShell 引号和换行容易改变实际代码。
2. 如果 `apply_patch` 因沙箱问题失败，可以使用受限 Python 编辑脚本，但必须满足：
   - 只编辑当前工作区内文件。
   - 使用精确匹配或结构化改写。
   - 匹配失败时立即退出，不继续写后续文件。
   - 写完后用 `rg` 和语法检查验证。

## 验证范围

Windows 代码或运行环境修改后，须对修改的 Python 文件执行 `python -m py_compile <changed files>`，运行 `python scripts\diagnose_matplotlib_windows.py`，并完成 [绘图规则](PLOTTING.md) 中最小 Matplotlib smoke 和实际脚本最小真实验证。先按上文激活环境和设置 DLL 路径。

仅移动或修改 Markdown 时，验证 UTF-8、关键内容、规则迁移完整性、链接及乱码扫描；不把文档检查宣称为运行环境或主流程验证。
