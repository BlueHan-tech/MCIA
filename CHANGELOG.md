## 2026-09-07

- Configure DB2/DB3 under F:/A-SCI, use project-local metadata/output paths, and keep pipeline subprocesses in the active Python environment. Add a bounded real-data setup check.

- Prepare the initial GitHub source upload; add .gitignore rules for Python caches, IDE settings, local environments, experiment outputs, and model checkpoints.

## 2026-08-26

- 锁定主实验为 DB3 手势识别 S03/S06/S07 的 E1+E2+E3 统一 48 类任务（剔除动作 49），以及 S05/S06 的 E1+E2 Key10 连续关节角度估计。
- 重构 dataset_kinematics.py 与 dataset_db3_emg.py：按 exercise 独立滑窗，固定 repetition 1/3/4、6、2/5 划分，并仅使用训练 repetitions 计算 EMG 和 glove 归一化统计，消除 E1/E2 边界窗口、验证窗口重叠和测试集统计泄露。
- 新增 scripts/05_eval_db3_gesture_raw_vs_augmented.py 为正式 48 类手势识别阶段：A/B/C 使用同一 GestureTCN 和划分，异常检测器仅在对应 exercise 的训练 repetitions 上拟合。
- 更新 run_all_experiments.py、run_layout.py 和配置，默认流程依次输出新掩码 MCIA、无泄露被试适配、E1+E2 角度结果和 48 类手势识别结果。

- 重构 MCIA 默认训练掩码为固定三场景：S1 短时缺失 20%、S2 单整通道缺失 40%、S3 双整通道缺失 40%。同步移除旧 S4 场景与会覆盖配置的 curriculum，并将正式评估、图表与 smoke 同步为 S1/S2/S3。


- 将独立的 DB2/DB3 审计、诊断、预览与早期 baseline 脚本集中移入项目根目录 test/；主实验链路仍保留在 scripts/。


- 为独立 DB3 手势识别诊断加入固定随机种子与 CUDA 确定性设置，使原始与补全输入的快速复测可复现并可分别落盘。


- 新增独立 DB3 手势识别补全诊断：使用固定 repetition 划分比较原始 EMG 与健康先验 MCIA 增强输入，不修改 Exp1、Exp2 或 Exp3 主流程。


- 新增独立 S05 任务感知 MCIA adapter 诊断：冻结 raw KinematicTCN，只在训练 repetition 用规则掩码对适配器联合优化被遮蔽 EMG 重建与角度 MSE，以固定验证窗口对比 raw、健康先验和任务感知补全；测试 repetition 不参与。

- 新增独立 S05 EMG-only Oracle patch 选择器可行性诊断：仅以训练 repetition 生成单个全通道时间 patch 的 Oracle 标签，训练小型完整窗口时序 CNN 选择器，仅在固定验证窗口评估角度收益；测试 repetition 不参与。

- 新增独立 S05 训练集内 MCIA adapter 适配与固定验证窗口 Oracle 诊断：只用 repetition 1/3/4/6 做自监督 ScenarioMix 适配，以固定 64 个验证窗口比较健康先验与适配后的单 patch Oracle 上界；测试 repetition 不参与。

- 新增独立 DB3 条件 alpha 补全诊断：仅在 S05 验证集以 global RMSE 选择 S1/S3 的保守混合系数，S2/S4 保持完全替换；策略锁定后才报告 test repetition，保留 raw 与旧式全替换对照，不修改 Exp3 主流程。

- 新增独立 DB3 补全收益预测可行性诊断：训练集生成收益标签，验证集评估仅看 EMG 的收益排序；测试 repetition 未参与。

- 新增独立 DB3 Oracle 上界诊断：验证集穷举每窗一个固定 40 ms 全通道 patch 的 MCIA 补全位置，量化固定补全预算的理论角度误差收益；测试 repetition 未参与。


- 新增独立 DB3 Oracle 特征筛选诊断：在验证集以预先计算的 Oracle 收益为目标，检验 MCIA 重建差分和训练集归一化后的通道不一致性与收益的 Spearman 相关；测试 repetition 未参与。

- 新增独立 DB3 MCIA 保守 alpha 混合诊断：在 S05 训练划分枚举固定 40 ms 候选 patch，用冻结 MCIA 和 KinematicTCN 比较不同补全幅度；验证、测试 repetition 未参与。

- 新增独立 DB3 无标签 TCN 输出敏感度筛选：以冻结 KinematicTCN 的输出 Jacobian 近似作为全通道 40 ms patch 特征，仅在验证集与 Oracle 收益做关联诊断；测试 repetition 未参与。

- 新增独立 DB3 固定周期稀疏补全诊断：在 S05 验证集对固定 25% patch 掩码的四种相位汇报平均结果，比较 MCIA、零替换和线性插值的同 mask 对照；测试 repetition 未参与。

- 新增独立 DB2 MC-Dropout 重建不确定性校准诊断：在 Exp1 留出被试 S29–S32 和同样的 ScenarioMix 掩码下，检验冻结 MCIA 预测标准差与被遮蔽 patch 真实重建 L2 误差的关联；不使用 glove、TCN 或 DB3。

- 新增独立 DB3 MC-Dropout 不确定性与 Oracle 收益关联诊断：在 S05 验证集严格对齐已有的全通道单个 40 ms patch Oracle 候选，仅以冻结 MCIA 和 EMG 计算 MC-Dropout 不确定性，测试 repetition 未参与。

- 新增独立 DB3 Exp3 规则掩码覆盖审计：以主实验同一的训练集拟合 RuleAnomalyDetector，统计 S05/S06 各 split 的 patch 掩码比例、S1–S4 类型、死通道和上下文不足标记；不运行 MCIA 或 TCN。

## 2026-08-25

- 扩展独立 DB3 验证集高误差窗口诊断：用训练集建立 EMG 正常参考，新增包络调制频带能量、突发/平直持续时间、通道相关性和相对训练分布距离；验证集仅用于比较高误差窗口，测试 repetition 未参与。

## 2026-08-06

- 新增独立 DB3 Key10 拇指上举预览：仅以标准 repetition 划分训练 Raw-EMG KinematicTCN，并从留出测试集中绘制 stimulus=1 的连续轨迹；不运行 MCIA、不写入 Exp3 产物。


- 补充 DB3 glove 审计的全 22 通道与当前 Key10 动态窗口对照：区分手套整体运动记录稀疏与 Key10 未覆盖已存在变化的情况，仅更新诊断结果。


- 新增 DB3 22 通道 CyberGlove 标签质量只读审计：逐被试统计原始与降采样标签更新密度、当前 Key10 动态窗口比例、常量通道归一化风险，并输出 CSV、JSON 和汇总热图，不改变主实验流程。


- 新增独立 DB3 验证集高误差窗口诊断：仅用原始 EMG 训练 KinematicTCN 并保存验证窗口级误差、EMG 特征和预测产物，不预测或评估测试 repetition；不修改主实验。

- 在 AGENTS.md 业务原则中明确 MCIA 的最终有效性以补全后 EMG 在相同被试、划分和 KinematicTCN 下的连续角度优势为准，不以单独的肌电重建误差作为结论。

- 新增独立 DB3 验证误差风险门 MCIA 对照：仅以验证集原始 TCN 误差拟合 EMG 风险门，测试时仅用 EMG 触发短 patch 掩码和 MCIA 补全，输出与原始 EMG 的同一 KinematicTCN 对照；不修改主实验。

- 新增独立 DB3 短片段质量掩码 MCIA 对照：仅用训练 repetition 的 EMG 拟合 patch 能量与波动阈值，测试时不读取 glove 或角度作为掩码输入，与原始 EMG 分别训练同一 KinematicTCN 进行小规模 A/B 评估；不修改主实验。

- 新增独立 DB2/DB3 有效肌电信息审计：以动作相对静息激活、动作区分度和手套速度耦合联合筛选 DB3 低信息通道与动态片段；仅输出后续 MCIA 掩码验证候选，不修改 MCIA、训练或主实验。

## 2026-07-31

- 扩展独立原始 sEMG 审计：新增削顶平台、重复帧和瞬时突发候选检测；确认 S03 Ch7 存在极短保持片段，仅作为后续 MCIA 掩码验证候选。

- 新增独立 DB2/DB3 原始 2 kHz sEMG 同动作质量对照审计：以 40 名 DB2 被试同一 restimulus 动作为参考，检测 DB3 全零通道、低频伪迹、50 Hz 工频和接触突变候选，只读且不运行 MCIA 或 TCN 训练。

- 新增独立 DB3 原始 sEMG 信号质量审计脚本：按被试和通道区分持续平直失效候选、弱活动和伪饱和候选，并与历史 DB3 窗口级角度误差精确时间对齐关联；只读且不运行 MCIA 或 TCN 训练。
- 统一 DB2 sanity 默认下游路径：仅使用 Exp3 同一的 Key10 KinematicTCN 纯 MSE 训练、验证集 affine 校准、三角重叠融合和 6 Hz 低通；删除默认的平滑 TCN、中心点、基线和 ablation 执行路径。
- DB2 sanity 统一输出 1280 点（6.4 秒）Key10 5x2 连续测试图，只使用时间连续、且每点至少双窗口覆盖的片段；新增与 Exp3 一致的 continuous metrics 与 overlap consistency 输出。
- 将 Exp3 ABC 图统一改为 1280 点（6.4 秒）的 Key10 5x2 连续测试轨迹：只选取同一连续时间段、每点至少由两个重叠短窗口支持的片段，A/B/C 使用相同物理时间轴。
- 扩展连续输出诊断：NPZ 保存每点重叠覆盖数，结果 JSON 记录相邻窗口重叠区预测一致性；不改变 TCN 训练、分窗、split 或 full metrics。
- 将正确的 Key10 索引保持为显式的 0-based 列表，并修复 Level0 读取当前 `config.yaml` 的路径。
- 纠正 Key10 CyberGlove 映射：0-based 源通道替换为 `[1, 2, 4, 5, 7, 8, 11, 12, 15, 16]`，对应五指 MCP 与 IP/PIP 关节；评估、绘图、表格和论文汇总同步改为 global/MCP/PIP。
- 补全 Level0 的 Key10 产物可追溯性：结果 JSON、每被试者记录、NPZ 和通道 CSV 均写入固定映射、源 glove 索引和通道名称。
- 修正 DB2 Key10 开发训练脚本的 Windows 导入顺序：在导入 NumPy/Torch 前设置当前 conda DLL 搜索路径。
- 加固 Key10 目标约束：数据集、所有连续角度回归器和 DB2 开发入口只允许 10 维输出，并将论文中旧 22 维结果表替换为 Key10 待新 run 结果。

- 统一主实验为固定 Key10 连续角度目标：数据、模型、Exp3、DB2 诊断、图表和汇总全部改为 10 维，并用元数据拒绝旧 22 维结果被当作新主实验继续使用。

确立“方案锁定后统一替换”原则：当无泄露验证后确认新方案更优时，主实验不保留旧方案的可选开关或默认路径；旧产物仅作 legacy 历史记录。

- 更新 AGENTS.md 实验迭代原则：新方案默认使用 10 分钟内可完成的小规模无泄露对比，全量 run_all 仅在方向锁定并获得明确授权后运行。

- 扩展 DB2 连续角度诊断的输出候选：在固定验证 repetition 上比较三角融合、零相位低通和逐通道仿射校准，校准仅拟合验证集预测，不读取测试标签。
- 将已锁定的输出方案接入 Exp3：保留原始 TCN 训练、分窗、split 和 full metrics，仅替换连续输出的拼接、滤波和验证集校准。

- 修正 Exp3 图像重建时的历史残留图：每个 subject 在写入新 ABC 图前仅清理其 comparison 子目录下同格式的旧 `*_ABC.png`，确保 run_all 产出仅含当前固定连续输出方案的图。
- 将 DB2 验证的固定连续输出方案接入 Exp3：A/B/C 在同一测试窗口位置上执行 uniform overlap-add 和 13 点 Savitzky-Golay 平滑，新增连续指标、连续预测 NPZ 字段及基于连续输出的 ABC 图像，保留旧 `subsets` 和原始预测不覆盖。
## 2026-07-30

- 为独立 DB2 sanity 添加严格整次 repetition 验证集模式和验证集专用的 overlap-add + Savitzky-Golay 候选窗口比较，使平滑参数选择与测试 repetition 分离；不修改 Exp3 。
- 增加独立 DB2 sanity 轨迹稳定性诊断：对测试窗口的 TCN 预测按原始时间位置做 uniform overlap-add 融合，再对连续段做固定 Savitzky-Golay 平滑，分开输出窗口指标、唯一连续时间点指标、图和 NPZ；不修改 Exp3 或主实验。
- 增强独立 DB2 sanity 脚本的多 subject 可复现性：支持 `--output-tag` 隔离诊断产物，并按 `base_seed + subject_id` 为每个 subject 重置随机种子，不修改 Exp3 或主实验数据流程。
- 在 AGENTS.md 新增业务原则：主实验下游连续角度估计固定使用单一 KinematicTCN，A/B/C 保持同一评估规则，调整范围限定为对齐、标签处理和轨迹稳定性。
- 删除 `AGENTS.md` 中提到已合并 `Rules.md` 的过期说明，保留 `AGENTS.md` 作为单一规则入口。
- 将 `Rules.md` 的仓库操作规则合并进 `AGENTS.md`，并删除独立的 `Rules.md` 文件，避免规则入口重复。

## 2026-07-24

- Extended the independent DB2 angle sanity diagnostic with lag-scan, glove downsample, center-point TCN, mean/ridge/random-forest baselines, and trajectory quality outputs under `06_diagnostics/db2_angle_sanity`, without modifying Exp1/Exp2/Exp3 main flows.
- Added independent DB2 continuous-angle regressor sanity diagnostic script `scripts/diagnose_db2_angle_regressor_sanity.py`, training per-subject KinematicTCN models on DB2 kinematics with repetition-based splits and isolated `06_diagnostics/db2_angle_sanity` outputs without touching Exp1/Exp2/Exp3 main artifacts.
- Enhanced Exp3 dynamic subset visualization: ABC angle figures now use configurable dynamic-preferred trial selection, write visualization selection metadata, include dynamic score/rank/coverage in titles and filenames, and from-run figure regeneration reuses or recomputes dynamic test scores without changing training, splits, or full-test metrics.
- Added Exp3 dynamic-angle diagnostics in `scripts/04_eval_db3_angle_raw_vs_augmented.py`: train/val/test splits now report dynamic-window counts, ratios, score summaries, low train-coverage flags, and test dynamic subset metrics without changing windowing, split membership, training, validation, or full-test metrics.
- Refined the DB2 per-channel masked MAE heatmap rendering for the clean paper version by removing the in-figure title and footer note while preserving the EMG channel axis, held-out subject axis, Masked MAE colorbar, output directory, and CSV data.
- Added from-run DB2 per-channel masked MAE heatmap export in `scripts/generate_figures_from_run.py`, computing S1-S4 aggregate held-out-subject x EMG-channel errors by inference-only evaluation from the existing Exp1 checkpoint and writing PNG/CSV paper candidates without retraining or changing training, model, data, metrics, checkpoints, cache, or history.

## 2026-07-23

- Refined the from-run DB2 Fig.1 rendering style for main-text use by removing in-figure title/note text, tightening the 3-panel layout, lightening bars, and reducing subject-dot size without changing data or statistics.
- Added from-run DB2 Fig.1 quantitative ScenarioMix export in `scripts/generate_figures_from_run.py`, producing a 3-panel held-out-subject mean/SD/scatter figure plus source CSV from existing Exp1 per-subject metrics without retraining or touching training, model, data, metrics, checkpoints, cache, or history.
- Updated Exp1 checkpoint selection, LR scheduler, and early stopping in `scripts/01_train_mcia_db2_healthy_prior.py` to use overall `corr_masked` with fallback to `corr_masked_partial`, while leaving model, loss, mask generation, training parameters, data processing, metric definitions, and history schema unchanged.
- Updated `utils.visualization.plot_completion_panel` so completion panels hide baseline curves by default via `show_baselines=False`, preserve explicit baseline rendering when requested, and compute y-limits only from signals drawn on the main axis to prevent extreme Cubic Spline/TimeMAE values from compressing MCIA waveform views.
- Added scripts/generate_tables_from_run.py as the from-run table rebuild entry and implemented Table V export from existing DB3 angle prediction metrics without re-training or re-prediction.
- Enabled DB3 12ch completion panel generation in the default run_all path by honoring exp2_transfer.generate_db3_completion_panels and its visualization count settings in scripts/03_generate_augmented_db3_semg.py.
- Added dev/regenerate_figures_from_run.py as a PyCharm Run-button helper that sets MCIA_RUN_DIR for a selected run and delegates to generate_figures_from_run without re-training.
- Added dev/regenerate_figures_from_run.cmd as a Windows double-click helper for regenerating current DB3 completion and anatomy angle figures from an existing run without re-training.

## 2026-07-22

- 收紧新 run 默认目录初始化：`utils/run_layout.py` 不再默认创建 legacy/manual 或当前未使用的 `04_paper_figures`、`03_angle_prediction/figures/summary`、`01_db2_completion/logs`、`02_db3_transfer_completion/logs` 和 `02_db3_transfer_completion/metrics`，保留当前四步主线必需目录，不移动历史产物。
- 迁移 dev scenario mix smoke 绘图验证：`dev/smoke_scenario_mix.py` 不再调用 `utils.evaluation.evaluate_and_plot_multi_scenario`，改为通过 `plot_completion_panel` 生成 `scenario_panels/s1.png` 到 `s4.png` 统一风格 smoke 图；同时将 `utils.evaluation.py` 中两个旧绘图函数标记为 deprecated，并清理 01 脚本的未使用 legacy 绘图 import。
- 标记 legacy paper figures 体系：在 `scripts/generate_paper_figures.py` 和 `utils/paper_figures.py` 顶部说明其已不属于默认 `run_all_experiments.py` 和默认 `generate_figures_from_run.py` 流程，仅作为手动 legacy 入口保留，并明确 `utils.paper_figures` 混有旧渲染、表格导出和 metrics aggregation，后续清理前不能整文件删除。
- 禁用 `generate_figures_from_run.py` 对 legacy paper figures 的默认重建：新增 `--include-legacy-paper-figures` 显式开关，默认只重建 DB3 12ch completion 图和 anatomy A/B/C angle 图，避免手动重建时重新产出待重设计的 `04_paper_figures` 体系。
收敛 `generate_figures_from_run.py` 的 Exp3 angle 图像重建入口：从已有 `Sxx_angle_predictions.npz` 读取 `target`、`pred_A`、`pred_B`、`pred_C`，显式复用 `04_eval_db3_angle_raw_vs_augmented.py` 中的 `save_abc_comparison_figures` 和 `ANGLE_PLOT_GROUPS` 重建 anatomy A/B/C 图，不再依赖旧 wrist/mcp/dip 输出语义。
对齐 Exp3/04 anatomy angle 诊断图的子图比例：按 01 报告式 panel 的单轴尺寸 7.5x2.6 重算 `ANGLE_PLOT_GROUPS` 各组 figsize，使 abduction 2x2 不再被拉伸，并保持 palm_wrist 3x1 布局和原有颜色、标题、路径不变。
调整 Exp3/04 anatomy angle 诊断图版式：仅修改 `04_eval_db3_angle_raw_vs_augmented.py` 的绘图 layout 元数据和 figure size 计算，使 palm_wrist 改为 3x1，abduction 使用更自然的 2x2 画布比例，保持 prediction、metrics、颜色、标题和输出路径不变。
增强 Exp3/04 角度预测诊断图绘图层：在 `04_eval_db3_angle_raw_vs_augmented.py` 中新增仅用于绘图的 `ANGLE_PLOT_GROUPS`，将 A/B/C angle prediction 诊断图改为 22 维 anatomy 分组展示，并在 prediction 与 metrics 落盘后安全生成少量验收图。
收敛默认实验职责：从 `scripts/run_all_experiments.py` 的默认 STEPS 中移除 Exp4 paper figures/tables 入口，使默认流程停在 `04_eval_db3_angle_raw_vs_augmented.py` 的角度预测与 metrics 阶段，并在 `generate_paper_figures.py` 顶部标注 legacy/pending redesign。
迁移 Exp2 03 DB3 12 通道补全图绘图层：保留 `save_subject_semg_panels` 作为主出口，将 direct_transfer 与 pretrained_finetuned 拆分到独立子目录，复用 `plot_completion_panel` 的 01 风格面板，用浅黄背景标出 DB3 补全区域并默认不显示 Observed 置零线。
回退 Exp2 02 脚本的补全报告生成接线：移除 `02_finetune_mcia_db3_amputee.py` 中的 `build_report` 调用和报告 helper，使 02 默认仅产出 checkpoint、`split.json` 和 `finetune_summary.json`，保留 `utils.visualization` 的 `panel_kwargs` 统一绘图接口供 03/04 复用。
对齐 Exp2 DB3 补全报告绘图层：在 `utils.visualization.build_report` 中增加向后兼容的 `panel_kwargs` 透传，并让 `02_finetune_mcia_db3_amputee.py` 在每个 DB3 subject 训练后为 direct_transfer、pretrained_finetuned、amputee_only 生成 01 风格完整报告，使用浅黄背景标出补全区域且默认不显示 Observed 置零线。
补充 Windows 绘图 smoke 验证规则：在 `Rules.md` 中明确纯绘图 smoke 与主流程 smoke 分级验证、绘图模块顶层重依赖懒加载、conda 环境执行和 OpenMP/DLL 冲突处理原则。

调整 01 DB2 补全报告的 12 通道绘图表达：在 `utils/visualization.py::plot_completion_panel` 中默认不再绘制 Observed 置零退化肌电虚线，改用简单 `axvspan` 浅蓝背景标出 mask 需要补全区域，并将绘图模块的 Windows conda DLL 路径规范化放到 `matplotlib.pyplot` 导入前，未修改训练、模型、数据处理或 metrics 逻辑。

## 2026-07-20

新增 Windows 环境修改规则文档：约束 conda/DLL 路径、Matplotlib native crash 诊断、run 目录、Exp3/Exp4 续跑与验证流程，避免将环境问题误判为算法问题。
新增 `CHANGELOG.md` 写入编码规则：Windows 下追加中文 changelog 必须显式使用 UTF-8，禁止使用未指定编码的 PowerShell 写入或重定向。

新增 Windows 中文编码规则：约束 PowerShell heredoc、`python -c`、stdin 脚本和 Markdown 写入的中文编码方式，要求用 UTF-8 读写和 Unicode 转义校验，并在修改后扫描乱码。

修复 Exp1 验证阶段 `_safe_pearson` 未定义导致全流程在 `validate_epoch_mcia_masked` 停止的问题：在 `utils/evaluation.py` 复用 `safe_pearson_np` 作为皮尔逊相关指标的安全计算函数。

补齐运动学指标模块 `_safe_pearson` 引用：让 `utils/metrics_kinematics.py` 复用 `safe_pearson_np`，避免直接调用 `correlation` 或 `evaluate_kinematics` 时出现同类 NameError。

## 2026-07-21

Added simulated Exp2/Exp3/Exp4 figure-style artifacts under outputs/simulated_experiment_figure_styles: generated reproducible Matplotlib examples for DB3 completion panels, rule-mask steps, Exp3 ABC joint-angle traces, Exp4 paper figures, and a table-style summary without changing experiment logic.

对齐 MCIA 全流程架构：在 `config.yaml` 关闭 `use_synergy_bottleneck`，在 Exp1 初始化 MCIA 时显式传入 `num_domains` 和 synergy 相关配置，并让 Exp4 Figure2 使用 `load_mcia_state_dict` 兼容加载 Exp1 checkpoint。

- 新增 Table V 真实指标补跑入口：`scripts/generate_tables_from_run.py --tables table_v` 现在从已有 DB3 subject checkpoint 和 `split.json` 执行 evaluation-only 受控人工遮挡评估，为 direct_transfer、amputee_only 和 pretrained_finetuned 输出 `table_v_db3_transfer_completion.csv/md/html`，不重训、不修改 checkpoint/cache/augmented_emg，并保留既有 angle prediction 表不覆盖。

- 新增 Table VI angle prediction 正式导出：`scripts/generate_tables_from_run.py --tables table_vi` 现在从 `03_angle_prediction/metrics/db3_angle_raw_vs_augmented_results.json` 读取 A/B/C 受试者级指标，输出 `table_vi_angle_prediction.csv/md/html`，按 Raw sEMG、Direct-transfer enhanced、Pretrained-finetuned enhanced 三行和 Global RMSE/MAE/CC/R2、Wrist/MCP/DIP CC 七列汇总，保留 effective n，不覆盖 `table_v_db3_transfer_completion.*` 或历史误命名 angle `table_v.*` 文件。

- Added Fig.2 DB2 completion paper-candidate export in `scripts/generate_figures_from_run.py`: the from-run rebuild now performs inference-only plotting from the existing Exp1 checkpoint and DB2 test-subject data, writing `paper_candidates/fig2_db2_completion/panels` and `paper_candidates/fig2_db2_completion/whole_channel` PNGs. These candidates show only Ground Truth, MCIA completion, and mask background, with no Observed zero-fill line and no Cubic Spline/TimeMAE baselines; 01 training, metrics, checkpoints, cache/history, and historical report figures are unchanged.
\r\n
## 2026-08-27

重写 research/find.md 的实验设置与结果章节：移除历史 22 维、Exercise 1-only 和随机窗口划分结果，改为引用最新主运行 run_20260826_165143_1 的 DB2 受控补全、DB3 Key10 连续角度和 48 类手势识别结果，并加入待选图目录占位。
\r\n
参考论文的结果章节组织方式调整 research/find.md：补充实验问题导语，按数据集、实验设置、量化结果、计算成本和现有方法比较组织；对尚未完成的成本与 SOTA 对比保留明确占位。

新增独立 DB2 小规模 MCIA 多尺度解码消融脚本：在固定 S01/S02 训练、S03 验证、S04 测试及相同 ScenarioMix 掩码下，对比原始 MCIA 与零初始化 temporal U-Net 残差解码器，输出 masked NRMSE、PSNR、MAE 和相关性到当前 run 的诊断目录，不修改 Exp1/Exp2/Exp3 主流程。

扩展独立 DB2 MCIA 解码器消融：新增膨胀率 1/2/4/8 的轻量非因果 TCN 残差解码候选，并在相同 S01/S02→S03→S04 协议下完成 20/60 epoch 验证；训练损失继续下降但 S2/S3 验证 NRMSE 与相关性未优于原始 MCIA，因此未接入主流程。

新增独立 DB2 MCIA 输出头统一消融脚本：在固定窗口、随机种子和 ScenarioMix 掩码下完成九种输出头的 20 epoch 对比；Temporal U-Net 验证 NRMSE 最低但改善未达 5% 且相关性下降，因此未修改主实验链路。

## 2026-08-28

新增独立 PartialConv 损失权重消融：在固定 DB2 数据划分、窗口和 ScenarioMix 掩码下完成四组 20 epoch 验证；提高 NCC 能增加相关性但显著恶化 NRMSE，当前 NCC=0.5、Charbonnier=1.0 仍为最佳平衡，未修改主实验链路。

