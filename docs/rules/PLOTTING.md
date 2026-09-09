# Plotting Rules

训练、metrics 和图像是完整实验产出；不能默认跳过绘图。环境初始化和编码规则见 [WINDOWS.md](WINDOWS.md)。

## Matplotlib 规则

1. 如果画图崩溃，先用父进程逐项拉子进程的诊断脚本定位，不要靠 `try/except`。
2. Native crash 不能被普通 Python 异常捕获，所以必须单 probe 单进程。
3. 优先排查顺序：
   - `import numpy`
   - `import matplotlib`
   - `matplotlib.use("Agg")`
   - `import matplotlib.pyplot`
   - `plt.figure()`
   - `ax.plot()`
   - `ax.bar()`
   - `fig.canvas.draw()`
   - `fig.savefig(".png")`
   - `fig.savefig(".svg")`
   - `fig.savefig(".pdf")`
4. 如果只有把当前 env 的 DLL 路径放到 `PATH` 最前后才通过，结论应优先判断为 DLL 搜索路径污染。
5. 不允许长期保留“静默跳过 savefig”的 monkeypatch。临时护栏必须有明显日志，并在根因修复后删除。

## Windows 绘图 smoke 验证规则

1. 绘图 smoke 必须尽量隔离，只验证目标绘图函数，不应触发训练、模型、torch、数据加载等重依赖。
2. 绘图工具模块不得在顶层导入非绘图必需的重依赖；如 `torch`、模型类、训练工具、数据加载器等，应改为函数内懒加载。
3. 验证 `plot_completion_panel` 这类纯绘图函数时，import 链应只包含 `numpy`、`matplotlib` 和该函数直接需要的轻量工具。
4. Windows 下如果 smoke 在 import 阶段失败，先判断是否为导入链/DLL/OpenMP 问题，不要误判为绘图实现失败。
5. 独立 smoke 脚本如果放在 `outputs/`、`tmp/` 等子目录，必须显式把项目根目录加入 `sys.path`。
6. Windows 下绘图 smoke 优先用真正激活的 conda 环境或 `conda run -n <env>` 执行；不要默认直接调用 `envs\<env>\python.exe`。
7. `ensure_current_env_dll_path()` 必须在导入 `matplotlib.pyplot`、`torch`、`numpy` 之外的 native-heavy 项目前尽早执行。
8. 如果遇到 OpenMP DLL 冲突，不要用 `KMP_DUPLICATE_LIB_OK=TRUE` 作为默认修复；应先清理导入顺序、PATH 顺序和顶层重依赖。
9. smoke 验证应分两级：
   - 纯绘图 smoke：只测目标绘图函数能否保存 PNG。
   - 主流程 smoke：模拟 01/02/03/04 实际脚本导入顺序和调用路径。
10. 只有纯绘图 smoke 和主流程 smoke 都通过，才能说绘图改动已验证。

## Figures

- Exp4 的导入函数名必须与 `utils/paper_figures.py` 一致；使用别名须说明原因。此处 Exp3/Exp4 沿用旧规则称谓，执行时核对实际脚本和依赖，不能仅凭阶段编号判断。
- 图像生成不能默认跳过；临时护栏修复后必须清理。

## 验证规则

Windows 代码或运行环境修改后，先按 [WINDOWS.md](WINDOWS.md) 激活 conda 环境并保证 DLL 路径优先，再至少做以下验证（仅 Markdown 修改的范围见 WINDOWS.md）：

1. 语法检查：

   ```powershell
   python -m py_compile <changed files>
   ```

2. 路径/DLL 诊断：

   ```powershell
   python scripts\diagnose_matplotlib_windows.py
   ```

3. 最小 Matplotlib smoke：

   ```powershell
   python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import numpy as np; fig,ax=plt.subplots(); ax.bar(np.arange(5), np.arange(5)); fig.canvas.draw(); fig.savefig('outputs/matplotlib_smoke.png'); print('ok')"
   ```

4. 对实际脚本做最小真实验证，例如：

   ```powershell
   $env:MCIA_RUN_DIR="E:\paper\MCIA\outputs\run\<run_id>"
   python scripts\generate_paper_figures.py --figures fig7,fig8,fig9,table3 --force
   ```
