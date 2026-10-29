# Reproducibility Rules

本文件是 run、resume、outputs 和 changelog 的唯一规范来源。实验启动边界见 [实验协议](../EXPERIMENT_PROTOCOL.md)，编码规则见 [WINDOWS.md](WINDOWS.md)。

## Run 目录规则

1. 续跑旧实验必须显式设置 `MCIA_RUN_DIR`。
2. 所有中间产物必须写回当前 run。

## Resume / outputs

1. 角度评估支持从已有 prediction 重建 metrics；已完成 subject 不应重复训练。
2. 每个 subject 完成后尽快落盘 prediction 和 metrics 状态。
3. 图像按 [PLOTTING.md](PLOTTING.md) 生成和验证。
4. 旧 run 可作为历史记录保留；legacy 读取或重绘必须明确标注，不影响新主流程默认行为。

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
