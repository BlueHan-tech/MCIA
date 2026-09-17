# 研究诊断归档

本目录包含探索性审计、消融、Oracle 筛查和历史复现。这些脚本不属于主实验
流水线，并被明确排除在 `test/` 之外，因此测试发现只会找到可运行的冒烟测试
或契约检查。

- `audit_*`, `calibrate_*`, `decompose_*`, `diagnose_*`, `level0_*`,
  `preview_*`、`report_*` 和 `reproduce_*` 是研究证据或诊断脚本。它们可能读取
  特定历史 run，或依赖已被替代的实现；复用前必须阅读文件名含义和输出元数据。
- `retired/diagnose_db3_mcia_s05_adaptation_oracle.py` and
  `retired/diagnose_db3_task_aware_mcia.py` 是冻结的历史记录。它们指向已淘汰的
  目标被试伪目标适配入口，不能在当前协议下运行。
- 当前可复用实验应放在 `scripts/`；可运行的自动检查应放在 `test/`。一次性调查
  不应加入这两个目录。

仍在进行中的归因脚本仅位于 `dev/`：`ridge_attribution.py` 和
`attribution_ladder.py`。
