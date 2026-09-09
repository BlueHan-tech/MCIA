# Reproducibility Rules

本文件是 run、resume、outputs 和 changelog 的唯一规范来源。实验启动边界见 [实验协议](../EXPERIMENT_PROTOCOL.md)，编码规则见 [WINDOWS.md](WINDOWS.md)。

## Run 目录规则

1. 单步脚本不允许偷偷创建新的 run 目录。
2. 单步脚本没有 `MCIA_RUN_DIR` 时必须明确报错。
3. 只有 `scripts/run_all_experiments.py` 可以正常创建新 run。
4. 续跑旧实验必须显式设置 `MCIA_RUN_DIR`。
5. 所有中间产物必须写回当前 run，不允许写到新的空 run。

## Resume / outputs

1. Exp3 必须支持从已有 prediction 重建 metrics；已完成 subject 不应重复训练。
2. 每个 subject 完成后尽快落盘 prediction 和 metrics 状态。
3. Exp4 缺少 Exp3 输入时，明确打印缺失路径和应先运行的脚本。阶段称谓沿用旧规则，实际入口与依赖须按脚本核对。
4. 图像是最终产出的一部分，按 [PLOTTING.md](PLOTTING.md) 生成和验证。
5. 旧 run 可作为历史记录保留；legacy 读取或重绘必须明确标注，不影响新主流程默认行为。

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
