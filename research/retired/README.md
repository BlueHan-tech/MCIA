# 已淘汰的研究代码

这些文件仅为历史追溯而保留。它们不在 `scripts/` 中，不是受支持的入口，也不得
生成新结果。

- `02_finetune_mcia_db3_amputee.py`,
  `03_generate_augmented_db3_semg.py`, `generate_paper_figures.py`, and
  `generate_tables_from_run.py` 依赖无效的目标被试伪目标适配或其衍生产物。
- `diagnose_db3_mcia_s05_adaptation_oracle.py` and
  `diagnose_db3_task_aware_mcia.py` 是基于同一淘汰路径构建的冻结诊断，不能在
  当前协议下运行。
- `paper_figures.py` 是对应的已废弃绘图和制表工具。

历史 run 输出仅能作为各自运行的证据；这些文件均不属于当前主实验或基线比较。
