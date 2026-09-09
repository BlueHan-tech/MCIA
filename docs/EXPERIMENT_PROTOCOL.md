# Experiment Protocol

本文是当前实验设计的规范来源。本次拆分不变更模型、参数或数据处理；下列实现快照来自 `config.yaml`、`data/dataset_kinematics.py`、`scripts/04_eval_db3_angle_raw_vs_augmented.py` 和 `utils/kinematic_target.py`。尚未明确锁定的设计项须在最终测试前记录，不能根据测试结果补定。

## Research question

在相同被试、数据划分和单一 KinematicTCN 下，MCIA 补全后的 EMG 是否比原始 EMG 更有利于连续关节角度估计？最终有效性必须由下游角度估计改善证明，不能只凭 EMG 重建误差下结论。手势识别作为另行报告的任务，不能替代该判据。

## A/B/C

| 组别 | 下游输入 |
| --- | --- |
| A | 原始 EMG |
| B | 健康 DB2 先验 MCIA 补全的 EMG |
| C | 经对应 DB3 被试训练 repetitions 适配的 MCIA 补全 EMG |

各组 KinematicTCN 从头训练，保持相同结构、训练规则、随机种子、划分、目标和评估方式；仅输入表示不同。C 的被试适配指 MCIA，不是迁移下游 TCN。异常检测器只在对应 exercise 的训练 repetitions 上拟合。缺失组必须明确报告，不能作为完整 A/B/C 比较。

## Dataset split

- 当前连续角度主实验：DB3 S05/S06，E1+E2；按 exercise 独立滑窗，禁止窗口跨 exercise 或跨数据划分。
- 固定 repetitions：训练 1/3/4，验证 6，测试 2/5；随机种子 42。EMG 与 glove 归一化统计只从训练 repetitions 计算。
- 当前 DB2 健康先验：训练 S01–S28，验证 S29–S32，测试 S33–S40；不得将先验测试被试用于拟合或选择。
- Key10 使用原始 glove 的零基索引 `[1,2,4,5,7,8,11,12,15,16]`；标签为训练统计归一化后的 glove 通道值，不应未经换算称为角度单位“度”。
- 当前附加手势任务：DB3 S02/S03/S04/S05/S06/S07/S08/S09/S11，E1+E2+E3，动作 1–48（排除 49），同一 repetition 划分；其结果与连续角度任务分开报告。

## KinematicTCN

下游连续角度估计固定使用单一 KinematicTCN。A/B/C 不引入 Transformer、BiLSTM、模型集成或 DB2 到 DB3 下游模型迁移。

当前结构为非因果 same-padding TCN，hidden_dim=64、n_layers=6、kernel_size=3、dropout=0.1，输出 Key10。当前训练配置为 batch_size=64、learning_rate=0.001、最多 100 epochs、patience=20，学习率调度 patience=5、factor=0.5。

优化限于时序对齐、标签处理和轨迹稳定性，目标是避免下游模型成为瓶颈。当前连续输出配置为 triangular overlap-add、6 Hz Butterworth 和验证集拟合的逐通道 affine 校准；A/B/C 使用相同规则，不允许在测试集拟合校准或调整滤波参数。

## Primary endpoint

已锁定的业务判据是补全输入相对 A 的连续角度估计改善。当前实现报告 RMSE、MAE、Pearson 和 R²，以及 global/MCP/PIP 子集。

**待最终测试前锁定：** 唯一主指标、主要比较（C−A、B−A 或两者）、窗口级或连续输出级统计口径，以及成功判定标准。现有规则未明确这些选择，本次整理不擅自指定。不得在看到测试结果后挑选有利指标、组别或子集作为主终点。

## Aggregation

当前实现按被试计算指标，再以被试等权 `nanmean ± nanstd` 汇总并报告有效 n；不同组可能因缺失产物拥有不同 n，正式配对比较必须使用相同被试集合并披露缺失原因。

当前角度主脚本 `evaluate_subsets` 的 RMSE 口径是对该被试所选子集的全部样本与通道误差平方取平均后开根；MAE 同样合并样本与通道，Pearson 和 R² 逐通道计算后取有限值平均，并记录无效通道。不要与 `utils/metrics_kinematics.py` 中逐通道 RMSE 再平均的辅助口径混用。global 为全部 Key10；MCP/PIP 子集映射以 `utils/kinematic_target.py` 为实现依据。窗口级、连续输出级和 dynamic subset 诊断必须分开标注，不混合聚合。

最终主终点采用哪种输出口径、如何报告配对差值与不确定性，必须在测试前与主终点一并锁定。不得把窗口数当作独立被试数，或根据测试表现排除被试。

## Model selection

### EMG completion output range (2026-09-08)

补全交付值采用 `clip(prediction, 0, 1)`，随后按二值掩码复制回原始观测值。该约束匹配当前经 μ-law、分位数归一化及裁剪后的 EMG 目标范围；1 不代表 100% MVC。主流程补全生成、重建评估、下游角度及手势输入统一使用此规则，保存数据、指标与图像沿用同一补全值。引导推理在条件组合后裁剪。

保留 Softplus、模型参数结构、checkpoint 格式及训练前向/损失计算；模型直接前向仍可输出大于 1 的值，范围约束属于补全后处理。后续验证指标使用有界补全，因此不可与历史未裁剪指标不加说明地混用。历史 run 不自动改写，已有 prediction/metrics 的续跑或重绘属于其原有处理版本，不能当作已应用新约束的结果。新规则的下游有效性尚未验证。

### EMG normalization upper reference (2026-09-08)

当前主链路在包络与 μ-law 压缩后，以 Q5 和训练校准数据的 Q99 做鲁棒缩放，再裁剪至 `[0,1]`。Q99 由训练/验证开发比较选定：它将饱和比例降至约 1%，且该开发比较的补全 RMSE/MAE 最低。Q95、Q97.5 仅是已完成的候选比较，不保留为可选执行路径。此选择尚未通过完整下游主实验验证。

- 只用训练集拟合模型及数据统计，验证集选择 checkpoint、参数和候选方案；测试集只用于锁定后的报告。
- 当前 TCN 以最低验证损失选择 checkpoint 和触发早停；当前 MCIA 健康先验以验证集 overall masked correlation 选择 checkpoint。重建选择指标不能替代下游有效性判据。
- affine 输出校准当前使用验证 prediction/target 拟合，必须披露该用途；验证集结果不作独立测试证据。
- 用户确认采用已验证方案后，按根目录 AGENTS.md 的统一替换规则更新主流程，不保留未授权的新旧方案开关。

## Development set

1. 方向未锁定前只做快速、无泄露的开发验证，不执行全量主实验。
2. 默认使用小规模代表性子集，例如 DB2 S01/S02、一个 exercise、固定 repetition split；每次记录实际被试、exercise、划分与候选参数。
3. 单次目标 10 分钟内返回。超过 10 分钟未返回则停止并缩小范围、复用已有 prediction 或做纯输出诊断，不进入数小时后台监控。
4. 优先复用 prediction 比较输出稳定化、标签处理和时序对齐，不重跑 MCIA 或无关上游训练。
5. 候选方案在训练/验证上选定后，使用少量独立被试确认普适性，不根据这些测试结果回调参数；记录其已被查看的状态，避免再将其称为未见最终测试。

## Final test protocol

1. 在测试前冻结被试、划分、A/B/C、模型和训练规则、后处理、主终点、聚合方式及排除规则；留存配置与代码版本。
2. 仅当小规模开发已选定、独立小样本确认且用户明确授权全量复现时，才运行 `scripts/run_all_experiments.py` 或数小时主实验。启动前说明预计耗时、输出 run 和是否覆盖现有结果。
3. 锁定后测试仅用于报告；不得根据测试结果重新选参数。若继续探索，明确披露测试已使用，不把后续结果宣称为同一未见测试。
4. 保存各被试 prediction、metrics、配置、划分和必要图像；失败或缺失必须披露。按相同被试配对报告补全与原始输入的结果，不只报告有利组或肌电重建误差。
5. 产物与续跑执行 [REPRODUCIBILITY.md](rules/REPRODUCIBILITY.md)，图像验证执行 [PLOTTING.md](rules/PLOTTING.md)。
