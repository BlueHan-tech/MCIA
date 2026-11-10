# MCIA
针对截肢患者的肌电补全关节角度估计研究

主实验规格见[实验协议](docs/EXPERIMENT_PROTOCOL.md)。在 `py311` 环境中运行 `python scripts/run_all_experiments.py --smoke` 可完成短时的真实数据入口检查。省略 `--smoke` 会执行完整的训练和验证流程。用 `MCIA_RUN_DIR` 指向同一个 run 可恢复已开始的流程。

正式评估须锁定开发 run，再分别调用角度和手势入口的 `--final`。两个入口使用该 run 的开发 checkpoint 和统计量。旧 200 Hz 结果与新版 run 分开保存。历史绘图及任务内文献比较入口需要显式指定旧 `MCIA_RUN_DIR` 和 `--legacy`。
