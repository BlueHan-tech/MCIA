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

## source-of-truth 路由

执行相关任务前读取对应文档；跨领域任务读取所有相关规则。

| 内容 | 唯一规范来源 |
| --- | --- |
| 科研诚信、最小修改、路由与报告 | 本文件 |
| Research question、A/B/C、划分、KinematicTCN、终点、聚合、选择、开发与最终测试 | [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) |
| Windows、Conda、DLL、编码和受限编辑 | [docs/rules/WINDOWS.md](docs/rules/WINDOWS.md) |
| Matplotlib 诊断、smoke 与 figures | [docs/rules/PLOTTING.md](docs/rules/PLOTTING.md) |
| run、resume、outputs 与 changelog | [docs/rules/REPRODUCIBILITY.md](docs/rules/REPRODUCIBILITY.md) |
| 学术检索与证据综合工作流 | 全局 `academic-research` Skill；安装位置与工具配置见 [docs/ACADEMIC_RESEARCH_SETUP.md](docs/ACADEMIC_RESEARCH_SETUP.md) |

`config.yaml` 和代码是当前实现，run 内产物是该次运行的证据；均不得暗中覆盖协议。历史 changelog、诊断记录和旧 run 不是当前规则来源。发现实现与协议不一致时明确报告，不自行改写协议或扩大实验范围。

## 报告规则

1. 每次修改说明改了什么、为什么改、验证了什么、还有什么没验证；区分“已定位”“已修复”“已验证”“仍未验证”。
2. 只有完整主流程在目标 Windows 环境跑通，才能说“完整解决”。
3. 如果仅避免崩溃而跳过产物，必须标明临时 workaround；用户需要的图像不能以跳过绘图交付。
4. Windows 相关修改必须说明应用了本文件及所路由规则的哪些部分、验证范围与未验证项。
5. 代码或文档修改完成后，按 [可复现规则](docs/rules/REPRODUCIBILITY.md) 更新根目录 `CHANGELOG.md`。
