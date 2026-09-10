# Repository Instructions

## 永久科研诚信规则

1. 不得编造、篡改或选择性隐瞒实验结果；区分探索性诊断、验证集选择与锁定后的测试报告。
2. 训练集用于拟合，验证集用于选择；测试集不得用于选择模型、参数、阈值、后处理或展示方案。已查看的测试结果不得再作为未见测试证据。
3. 对照实验必须保持相同被试、数据划分、模型结构、训练规则和评估方式，只改变协议规定的实验因素。
4. 环境故障不能解释为算法失败；运行成功不能替代有效性证据。结论必须对应实际完成的验证。
5. 具体实验定义和最终有效性判据以 [实验协议](docs/EXPERIMENT_PROTOCOL.md) 为准。
6. 科研方法、实验设计、创新性及论文结论问题，主动检索并核验学术依据，按全局 `academic-research` Skill 综合证据；普通代码维护不强制检索。文献建议不自动授权修改锁定协议或启动实验。

## 最小代码修改规则

1. 优先使用小范围补丁，每次围绕一个明确问题，不做顺手重构。
2. 不改模型结构、训练参数或数据处理逻辑，除非任务明确要求。
3. 删除文件前必须先用 `rg` 查引用。
4. 临时诊断开关必须有明显日志；修复后清理 import、环境变量分支、旧 workaround 文档和 run 默认开关，避免静默跳过真实产物。
5. 新方案经无泄露开发验证且用户确认采用后，统一替换相关主实验链路、默认配置、评估、可视化和文档。
6. 除非用户明确要求正式比较，不保留新旧方案共存开关、旧默认值或备用执行路径。旧产物仅可作明确标注的 legacy 读取或图像再生成，不得影响新主流程默认行为。
7. 验证/诊断/评估脚本凡引用主链路逻辑（模型构建、损失、掩码生成与质量检测、补全交付规则、主实验指标与聚合口径），必须 `import` 主模块，不得复制实现；复制会使验证不再代表主链路。诊断包装可保留分析专用的附加量（描述性统计等），并对主模块判定结果内置一致性断言（漂移绊线，参照 `research/diagnose_wcbqd_detector.py`）。单元测试按规格独立重算期望值不在此限；已冻结留档的历史脚本（sha256 溯源）也不在此限，但须标注不可重跑。

## 固定 Python 环境

1. 本项目的实验与验证环境固定为 Conda `py311`（`D:\\work\\miniconda\\envs\\py311`）；其包含项目所需的 NumPy、PyTorch 与 Matplotlib。
2. Agent 执行任何导入上述依赖的命令、脚本 smoke、训练、评估或绘图验证时，必须优先使用 `conda run -n py311 python ...`；不得把自动打开的 Conda `base` 环境缺少依赖误报为项目环境故障。
3. 在 PyCharm 中运行时，解释器应选择 `D:\\work\\miniconda\\envs\\py311\\python.exe`。Windows DLL 路径初始化和其余约束仍以 [WINDOWS.md](docs/rules/WINDOWS.md) 为准。

## 跨工具协作与决策纪律

1. 每次任务开始前，必须检查 `git status --short`、与当前任务相关的 `git diff`，并阅读 [协作状态与决策记录](docs/COLLABORATION.md)。工作区既有改动默认属于其他工具或用户；不得覆盖、格式化或顺带修改。
2. 涉及模型、数据划分、掩码、评价、baseline、训练规则或计算预算的实质方案，先以“目标—机制—证据—风险—影响范围—最小验证”形式讨论，得到用户确认后才改代码或启动实验。论文或硬结果支撑不足时，只能标为待验证假设。
3. 采用、拒绝、失败、冻结或发现冲突的实质决策，必须更新 `docs/COLLABORATION.md` 的决策记录；每条写明状态、证据、适用边界与日期。`CHANGELOG.md` 只记录已发生的文件事实，不替代决策记录。
4. 每次任务结束前，必须再次检查与任务相关的 diff；若发现其他工具并发改动、代码与协议冲突、或状态文档与实现不一致，记录为待核验项并报告，不得自行选择其中一个版本继续执行。
5. 新增脚本、配置、实验产物或删除路径时，必须在协作记录注明入口、目的、是否主链路、是否已验证；已淘汰方案保留其拒绝原因，避免重复试验。

## source-of-truth 路由

执行相关任务前读取对应文档；跨领域任务读取所有相关规则。

| 内容 | 唯一规范来源 |
| --- | --- |
| 科研诚信、最小修改、路由与报告 | 本文件 |
| Research question、A/B/C、划分、KinematicTCN、终点、聚合、选择、开发与最终测试 | [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) |
| Windows、Conda、DLL、编码和受限编辑 | [docs/rules/WINDOWS.md](docs/rules/WINDOWS.md) |
| Matplotlib 诊断、smoke 与 figures | [docs/rules/PLOTTING.md](docs/rules/PLOTTING.md) |
| run、resume、outputs 与 changelog | [docs/rules/REPRODUCIBILITY.md](docs/rules/REPRODUCIBILITY.md) |
| 跨工具共享状态、方案决策、已验证/已拒绝路径与待核验冲突 | [docs/COLLABORATION.md](docs/COLLABORATION.md) |
| DB3 质量掩码与 WC-BQD 检测器设计、验证与文献支撑 | [docs/WCBQD_DESIGN.md](docs/WCBQD_DESIGN.md) |
| 学术检索与证据综合工作流 | 全局 `academic-research` Skill；安装位置与工具配置见 [docs/ACADEMIC_RESEARCH_SETUP.md](docs/ACADEMIC_RESEARCH_SETUP.md) |

`config.yaml` 和代码是当前实现，run 内产物是该次运行的证据；均不得暗中覆盖协议。历史 changelog、诊断记录和旧 run 不是当前规则来源。发现实现与协议不一致时明确报告，不自行改写协议或扩大实验范围。

## 报告规则

1. 每次修改说明改了什么、为什么改、验证了什么、还有什么没验证；区分“已定位”“已修复”“已验证”“仍未验证”。
2. 只有完整主流程在目标 Windows 环境跑通，才能说“完整解决”。
3. 如果仅避免崩溃而跳过产物，必须标明临时 workaround；用户需要的图像不能以跳过绘图交付。
4. Windows 相关修改必须说明应用了本文件及所路由规则的哪些部分、验证范围与未验证项。
5. 代码或文档修改完成后，按 [可复现规则](docs/rules/REPRODUCIBILITY.md) 更新根目录 `CHANGELOG.md`。
