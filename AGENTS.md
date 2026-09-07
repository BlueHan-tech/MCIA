# Repository Instructions

## 业务原则

- 下游连续角度估计固定使用单一 KinematicTCN。
- A/B/C 使用相同模型结构、训练规则、数据划分和评估方式。
- 不引入 Transformer、BiLSTM、模型集成或 DB2 到 DB3 迁移作为主实验下游模型。
- 调整仅限于时序对齐、标签处理和轨迹稳定性，目标是避免下游模型成为瓶颈。
- MCIA 的最终有效性以相同被试、数据划分和 KinematicTCN 下，补全后 EMG 的连续关节角度估计优于原始 EMG 为准；不能只以肌电重建误差作为结论。

## 方案锁定后统一替换原则

1. 当一个新方案已经过无泄露开发验证且用户确认采用时，必须将它统一替换到相关的主实验链路、默认配置、评估、可视化和文档中。
2. 不保留新旧方案共存的可选开关、旧默认值或“备用方案”执行路径，除非用户明确要求进行正式比较实验。
3. 旧 run 和旧结果文件可作为历史记录保留；如果必须支持读取旧产物，只能用于 legacy 读取或图像再生成，必须明确标注，且不得影响新主流程的默认行为。


## 实验迭代原则

1. 当前职责是快速、无泄露地判断优化方向，不是在方向未锁定前执行全量主实验。
2. 新方案的默认对比应使用小规模代表性子集（例如 DB2 S01/S02、一个 exercise、固定 repetition split），单次运行目标为 10 分钟内返回结果。
3. 如果一次方案对比超过 10 分钟仍未返回，应停止并缩小范围、复用已有 prediction 或先做纯输出诊断；不应进入数小时的后台监控。
4. 候选方案只能用训练集训练、验证集选择参数；测试集只用于锁定后的报告。需要验证普适性时，先使用少数独立被试者确认，不根据这些测试结果回调参数。
5. 对连续角度估计，优先尝试不改变单一 KinematicTCN 主体的输出稳定化、标签处理和时序对齐方案；优先复用已有 prediction 做快速 A/B 对比，不重跑 MCIA 或其他无关上游训练。
6. 只有当候选方案已在小规模开发集选定、独立小样本确认，且用户明确授权全量复现时，才可运行 run_all_experiments.py 或数小时的主实验。启动前必须说明预计耗时、输出 run 和是否覆盖现有结果。

# Windows Environment Modification Rules

本文档用于约束在 Windows 环境下修改本项目代码时的操作方式。目标是减少 PowerShell、conda、DLL 搜索路径、Matplotlib/native crash、沙箱补丁工具等因素带来的误判。

## 最高优先级原则

1. 不把 Windows 环境问题误判为实验算法问题。
2. 不改模型结构、训练参数、数据处理逻辑，除非任务明确要求。
3. 不用“跳过绘图”作为默认解决方案。训练、metrics、画图属于同一完整实验产出，最终流程必须能生成需要的图。
4. 可以临时加诊断开关，但修复后必须清理临时护栏，避免静默跳过真实产物。
5. 每次修改都要能说明：改了什么、为什么改、验证了什么、还有什么没验证。

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

## Windows 中文编码规则

1. Windows 下不要在 PowerShell heredoc、`python -c`、stdin 脚本中直接写中文 literal。
2. 需要在 Python 脚本里写中文时，优先使用 Unicode 转义，或从 UTF-8 文件读取中文内容。
3. 校验中文内容时也必须使用 Unicode 转义，避免校验字符串在进入 Python 前被 PowerShell 编码破坏。
4. 禁止使用 `echo "中文" >> file.md`、`>`、`>>` 这类重定向追加中文内容。
5. 使用 PowerShell 写中文文件时，必须显式指定 `-Encoding utf8`。
6. 写入 Markdown 后，必须用 UTF-8 方式重新读取并校验关键内容。
7. 修改 `CHANGELOG.md`、`AGENTS.md`、中文文档后，必须扫描乱码：`rg -n "\x{FFFD}|\x3F\x3F\x3F" CHANGELOG.md AGENTS.md WINDOWS_MODIFICATION_RULES.md docs`。
8. 如果发现三个连续问号，不要尝试自动恢复编码；通常原始中文已经丢失，应根据上下文手动恢复原句。
9. `apply_patch` 可用时优先用 `apply_patch` 写 Markdown；如果 Windows 沙箱导致 `apply_patch` 不可用，再使用受限 Python 脚本。
10. 受限 Python 脚本写中文时，脚本源码内中文必须用 Unicode 转义，文件读写必须显式 `encoding="utf-8"`。

## 修改代码规则

1. 优先使用小范围补丁，不做顺手重构。
2. 每次只围绕一个明确问题修改：运行目录、断点续跑、导入错误、DLL 路径、绘图崩溃等不要混成一大坨。
3. 在 Windows 上避免长 `python -c` 修改源码；PowerShell 引号和换行容易改变实际代码。
4. 如果 `apply_patch` 因沙箱问题失败，可以使用受限 Python 编辑脚本，但必须满足：
   - 只编辑当前工作区内文件。
   - 使用精确匹配或结构化改写。
   - 匹配失败时立即退出，不继续写后续文件。
   - 写完后用 `rg` 和语法检查验证。
5. 删除文件前必须先 `rg` 查引用。
6. 删除临时护栏前必须确认没有残留：
   - import
   - 环境变量分支
   - 文档里的旧 workaround 指令
   - run 脚本默认开关

## Run 目录规则

1. 单步脚本不允许偷偷创建新的 run 目录。
2. 单步脚本没有 `MCIA_RUN_DIR` 时必须明确报错。
3. 只有 `scripts/run_all_experiments.py` 可以正常创建新 run。
4. 续跑旧实验必须显式设置 `MCIA_RUN_DIR`。
5. 所有中间产物必须写回当前 run，不允许写到新的空 run。

## Exp3 / Exp4 规则

1. Exp3 必须支持从已有 prediction 重建 metrics。
2. 已完成 subject 不应重复训练。
3. 每个 subject 完成后应尽快落盘 prediction 和 metrics 状态。
4. Exp4 缺少 Exp3 输入时必须明确打印缺失路径和需要先运行的脚本。
5. Exp4 的导入函数名必须和 `utils/paper_figures.py` 保持一致；如果用别名，必须说明原因。
6. 图像生成是最终产出的一部分，不应被默认跳过。

## 验证规则

每次 Windows 相关修改后，至少做以下验证：

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
   $env:MCIA_RUN_DIR="D:\code\A_clean\A_king-main_rescue\outputs\run\<run_id>"
   python scripts\generate_paper_figures.py --figures fig7,fig8,fig9,table3 --force
   ```

## 变更记录规则

1. 每次完成代码或文档修改后，必须在主目录的 changelog 文件中追加一条简短记录。
2. changelog 文件放在项目主目录，命名为 `CHANGELOG.md`。
3. Windows 下写入 `CHANGELOG.md` 必须使用 UTF-8 编码；禁止使用未指定 `-Encoding utf8` 的 PowerShell `Set-Content`、`Add-Content` 或重定向追加中文内容。
4. 记录按日期分组，日期格式使用 `YYYY-MM-DD`。
5. 同一天多次修改时，在同一个日期标题下追加多条记录，不重复创建日期标题。
6. 每条记录应简短说明：
   - 修改了什么
   - 解决了什么问题
   - 涉及的主要模块或脚本
7. 记录应描述事实，不写长篇过程，不记录无效尝试，除非该尝试形成了明确诊断结论。
8. 如果用户明确要求本次不要写 changelog，则以用户要求为准。
9. changelog 示例：

   ```markdown
   ## 2026-07-20

   修复 Windows conda DLL 路径污染导致的 Matplotlib native crash：在主脚本导入 Matplotlib 前规范化当前 conda 环境 DLL 搜索路径，并让全流程子进程优先使用所选 Python 环境路径。

   增强 Exp3 续跑能力：支持从已有 prediction 重建 metrics，跳过已完成 subject，并修复 `_safe_pearson` 未定义导致的指标计算失败。
   ```

## 沟通规则

1. 报告结论时区分：
   - 已定位
   - 已修复
   - 已验证
   - 仍未验证
2. 不说“完整解决”，除非完整主流程已经在目标 Windows 环境跑通。
3. 如果只是让流程不崩但跳过了产物，必须明确说这是临时 workaround。
4. 如果图像是用户需要的结果，不能把“跳过画图”作为最终方案。

## AGENTS.md Reporting Rule

For Windows-related changes, explicitly report which parts of `AGENTS.md` were applied, what was verified, and what remains unverified.
