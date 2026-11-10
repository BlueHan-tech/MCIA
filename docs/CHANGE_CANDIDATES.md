# 待改事项

记录待讨论问题和实施入口。方案确认后，更新[实验协议](EXPERIMENT_PROTOCOL.md)和[当前决定索引](COLLABORATION.md#当前决定)。

[方案参考资料](references/)集中存放论文原文。

## 已决定

具体方案见[实验协议](EXPERIMENT_PROTOCOL.md)，接入、验收和证据见[当前决定索引](COLLABORATION.md#当前决定)。

## 待讨论

首版主链路暂无待讨论项。供体名单与实际更新步数按 D-027/D-037/D-041 在实施时核验。

## 补充比较（按需讨论）

- 补全质量的文献比较见[协议 §7.5](EXPERIMENT_PROTOCOL.md#75-补全方法学比较与主张边界)和 V-004/V-007。
- 公开基准排名须另行对齐类别、被试、划分、指标和额外训练数据，与主实验配对增益分别报告。

## 实施与验收入口

- **采样率、分窗和滤波**（D-009–D-011、D-014/D-019/D-034）。同步 `config.yaml`、`utils/paper_pipeline.py`、`data/ninapro_loader.py` 和 `data/dataset_db2_emg.py`、`data/dataset_db3_emg.py`、`data/dataset_kinematics.py`。核对分区隔离、EMG 与标签时间索引，以及采样率和窗长是否写入 run 配置。
- **补全训练与适配**（D-016/D-020/D-023/D-027/D-032/D-033/D-037–D-042）。入口为 `scripts/01_train_mcia_db2_healthy_prior.py`、`scripts/06_adapt_amputee_donor_prior.py` 和 `utils/loss_functions.py`。统计各供体合格训练窗、验证窗及去重有效时长，用于核对供体映射和实际更新步数。
- **质量掩码**（D-035/D-036）。在 `utils/db3_quality_mask.py` 及各数据入口核对检测区间、补全采样点与角度时间轴对齐。阈值复核及验证集对照见[协议 §4](EXPERIMENT_PROTOCOL.md#4-缺失定义与补全交付)和 V-008。
- **补全交付与两个下游入口**（D-012/D-013/D-015/D-017/D-018/D-021/D-022/D-025/D-029/D-031）。联查 `scripts/04_eval_db3_angle_raw_vs_augmented.py`、`scripts/05_eval_db3_gesture_raw_vs_augmented.py` 与共享交付逻辑。核对窗口首尾覆盖和等待时间、手势输入输出接口，以及动作 49 在切窗、掩码和评估中的覆盖。测量吞吐是否跟上每秒 20 次窗口更新。
- **旧产物与总入口**。按 V-004/V-014 核对缓存、checkpoint、run 产物和总入口配置，旧 200 Hz 产物标为 legacy。依赖新链路的诊断脚本导入主模块，历史脚本按归档规则保留。
