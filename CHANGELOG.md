## 2026-09-11

- 按用户指示新增 `docs/EXPERIMENT_HISTORY_ARCHIVE.md`，冻结旧 200 Hz/256 点主链路之前的已运行开发证据、已淘汰/暂停方案、可复用方法与不可外推边界；同步在 `docs/COLLABORATION.md` 建立索引。补记 S05/S06 作为旧连续 Key10 开发对象的标签可用性依据、动态覆盖、已读取 test 的角色降格，以及旧手势/供体名单的可核验选择边界。未改模型、数据、协议或 run 产物；尚在讨论的新实时受控掉线方案未被写成已采纳。

## 2026-09-10

- 按用户规格新增并运行 `dev/ridge_residual_tcn_screen.py`（E-004 开发筛查，非主链路）：base 严格复用 E-003（同 8,000 窗子集/掩码种子/λ/L32 头，内置漂移断言逐位复现通过），冻结因果 TCN 残差头（36 通道输入、hidden=16、kernel=2、dilation=[1,2,4,8,16]、感受野 32、零初始化头、masked MSE、AdamW lr=1e-3、batch=64、30 epochs、seeds 20260911–15 独立训练，无超参搜索/早停/checkpoint 选择）。验证 A（零初始化交付=base）/B（未来帧不变）/C（编译）全部通过。结果：5 seed 被试等权 overall = 0.1490/0.1493/0.1485/0.1489/0.1483（均值 0.14879、SD 0.00039；base 0.149606），5/5 < 0.1496 且每 seed ≥3/4 被试优于 base（4/3/4/3/4）→ 预声明判定**成功**；效应量小（对 base +0.0008，相对 Ridge 0.1699 累计 −0.021）。首次运行因写盘前局部变量遮蔽 bug 崩溃无产物，仅重命名修复后重跑、数值与崩溃运行打印完全一致（确定性交叉验证）；3,860 s、峰值 4,112 MB、无 OOM。产物 `06_diagnostics/ridge_residual_tcn_screen_20260910/`（JSON + base 与 5 seed 预测 .pt）。不进入 S33–S40/下游/主链路，详见 `docs/COLLABORATION.md` E-004。

- 对 `dev/ridge_residual_diagnostic.py` 做独立静态与短时审计：`py311` 编译通过；未来帧整体扰动不会改变此前时刻的全部因果特征；原始 JSON 复核 L32 在 S29–S32 的 Ridge 差值均为负。记录于协作证据 E-003；未重跑长诊断。

- 按用户规格新增并运行开发诊断 `dev/ridge_residual_diagnostic.py`（非主链路；掩码/指标/淡化/配置 import 主模块，Ridge 臂 import `dev/ridge_attribution.py`，K_obs 分层 import `dev/attribution_ladder.py`）：严格因果线性残差头（输入当前+过去 L 帧的 ridge 预测/observed/mask，窗外零填充；仅训练池人工掩码位置逐通道闭式拟合，固定种子 8,000 窗子集、λ=1e-3），S29–S32 仅前向、同 v1/v2 冻结掩码与交付，L∈{0,8,16,32} 全报告。Ridge 臂精确复现 E-001；L=0 中性（±0.0004），L=8/16/32 被试等权 overall 为 0.1635/0.1567/0.1496（Ridge 0.1699），4/4 被试与两个 K_obs 分层均单调改善且未在 L=32 饱和 → 按预声明三分支判定"可讨论小型因果残差 TCN"；CP-WOPT 未重跑（其 0.150 仅为 S29 旧预览）。数学自检（Gram=lstsq、因果扰动）通过；1,730 s、无 OOM（外部轮询峰值约 4.1 GB）。产物 `06_diagnostics/ridge_residual_diagnostic_20260910/ridge_residual_diagnostic.json`，详见 `docs/COLLABORATION.md` E-003。

- GLM 在用户授权下完成 `dev/attribution_ladder.py` 的真实 DB2 开发运行（S29–S32，207 s，无 OOM）：low-rank rank-8/rank-32/MCIA 被试等权 masked NRMSE 为 0.1904/0.1940/0.1773，既有全秩 Ridge 为 0.1700 且 4/4 优于其余归纳式方法。更新协作记录 E-001/E-002：淘汰训练池单窗低秩主路线，下一候选仅可检验 Ridge 的因果时序残差；CP-WOPT 未同次重跑，不作转导归因。

- 修复开发归因阶梯 `dev/attribution_ladder.py` 的全量 full-SVD 内存/时间失控：改为全 DB2 训练窗、批处理 `svds` 截断 PCA（rank 8/32，固定种子），避免物化中心化大矩阵；中断后的新运行输出写入 `attribution_ladder_20260910_v2/`。仅完成合成数据 smoke 与编译，未启动真实长诊断。

- 经用户确认，将后续补全候选的三层开发门写入 `docs/EXPERIMENT_PROTOCOL.md` §7.6，并在 `docs/COLLABORATION.md` D-006 记录：与岭回归的同口径重建门、clean→damaged→completed 的受控下游恢复率、以及 raw+dropout-augmentation 强对照 D；锁定 0.50 恢复率为工程目标而非文献阈值。未改模型、未实现新脚本、未启动实验。

- 整理代码库边界：`scripts/` 收敛为 14 个当前可运行入口，已退役的 DB3 伪目标适配、旧图表与旧表格入口移至 `research/retired/`；`test/` 收敛为 6 个可直接执行的 smoke/contract test，51 个一次性审计、oracle、ablation 与历史报告移至 `research/`，当前 Ridge v1/v2 保留在 `dev/`。同步移除活图像重建脚本对退役 02/03/paper-figure 路径的调用，更新 setup check、协议、绘图规则与协作索引；`utils/paper_figures.py` 和无消费方的 `paper_figures_dir` 一并归档/移除。语法检查和 6 个直接 smoke 均通过；`py311` 未安装 pytest，未执行 pytest 收集。

- 记录 DB2 开发集 S29–S32 的岭回归 v1 证据：同一时刻全局全秩归纳式 ridge 的被试等权 masked NRMSE 为 0.170，rank-8/逐窗变体退化至 0.226/0.207（逐窗 rank-8 0.264）；若 MCIA 同次同口径参照约为 0.185，则 ridge 暂时优于 MCIA，构成简单高秩线性映射可替代当前 MCIA 的反证，待 v2 输出复核。CP-WOPT 优势仍不能归因。详见 `docs/COLLABORATION.md` E-001。

- 澄清下游缺失增广对照的解释边界：对单一目标被试恒定失效的 Ch9/Ch10，质量掩码没有窗口级动作信息，只标识输入不可用；mask-aware 机制不能被表述为永久通道失效的补救。该约束写入 `docs/COLLABORATION.md` D-005。

- 记录用户确认的研究排序：端到端缺失感知解码暂不替换当前 A/B/C 研究对象；优先准备开发级 `A + 下游缺失增广` 强对照，检验原始输入的鲁棒训练能否追平/超过补全 B。设计、数据隔离与 B+Aug 辅助臂要求写入 `docs/COLLABORATION.md` D-005；本次未实现或运行实验。

- 记录目标被试级自监督适配 B2 的待核验冲突：永久失效 Ch9/Ch10 没有目标真值，零值适配会塌缩为零、排除损失则无法学习个体化恢复；在用户限定适用范围或放弃前，不实现、不启动该候选。详见 `docs/COLLABORATION.md` V-002。

- 记录用户拒绝 CP-WOPT 转导教师蒸馏 MCIA 路线：不新增教师缓存、蒸馏损失或训练入口；拒绝原因与未来若重启所需的防泄漏/公平比较条件写入 `docs/COLLABORATION.md`。

- 按用户决定新增项目部署约束（协议 §1.1）：主方法必须支持在线实时推理，端到端窗口级延迟 < 2 秒（窗口 1.28 s + 计算 < 0.72 s，因果于窗口流）；一次性校准（自监督适配等）允许离线完成不计入在线延迟；依赖整场数据的转导式处理（逐被试张量拟合、全场低秩精修）不得进入 A/B/C 方法定义。当前实测 MCIA 前向约 37 ms/窗，满足约束。§7.5 补充部署约束适用性条款（CP-WOPT 标注为非可部署 baseline）。直接后果：B1（MCIA 输出后全场低秩精修）正式出局；B2（被试级自监督适配，校准阶段完成）与 A（包络旁路）保持为候选，待验证数据决定。

- 新增跨工具协作机制：`docs/COLLABORATION.md` 作为方案地图、决策记录与待核验冲突的共享索引；`AGENTS.md` 与 `.cursor/rules/collaboration.mdc` 要求任务开始/结束检查工作区 diff、先讨论实质科研方案、记录采用/拒绝/冻结决定，并禁止静默覆盖其他工具改动或自行裁决协议冲突。未修改实验实现或启动训练。

- C 组开发冻结（协议 §5）：`scripts/06_adapt_amputee_donor_prior.py` 在供体池（S02/S04/S08/S11，E1+E2 训练 reps 适配、rep 6 仅 checkpoint 选择）上比较三候选（full lr1e-5 / full lr3e-6 / adapter lr1e-4，8 epochs，domain_id=1），按预声明规则（S05/S06 验证 global RMSE 均值最低）冻结 **C = adapter lr1e-4**，写入 `config.yaml` 的 `c_adaptation` 节；checkpoint 存于 `08_amputee_donor_prior/`，筛选报告存于 `06_diagnostics/c_dev_screen_20260910/`。如实记录：三个 C 候选均未超过 B 参照（B 均值 0.16933 vs 最优 C 0.16952，差距噪声量级；S06 上 C 微胜、S05 上 B 微胜），开发集并出现 B 优于 raw 的信号（S05 0.17161 vs raw 0.17353）；C−A 判决留待确认集。开发阶段被试测试 reps 未触碰。

- SGMD-AAE 预训练提速（演练 profiling 后用户确认）：`_train_sgmd` 改为标准 GAN 单前向 detach 模式（生成器由每步两次前向改为一次，`fake = output.detach()` 喂判别器，数学等价、省约 25% 前向开销）；`sgmd_aae_batch_size` 32→64（8GB 显存安全，约 1.5–2× 吞吐；属任务适配预训练的自有配置，非论文原始超参，随 §7.5 报告披露）。中断并清理了慢速版演练的半途产物，全预算演练以新配置重启。

- 实现协议 §7.5 健康基准入口 `scripts/eval_healthy_completion_benchmark.py`（复用 `run_task_matched_literature_baselines` 的 `_train_sgmd`/`_scenario_generator`/`_masked_metrics`，按"验证脚本单一来源"规则 import 而非复制）：DB2 S33–S40、E1、逐被试 ScenarioMix 掩码（预声明种子 20260910+被试号）、**全方法统一交付**（clip→patch 边界淡化→观测回填；MCIA 走 `complete_with_mask`，SGMD 取生成器原始输出、CP 取重建后统一包装），指标含 masked NRMSE/PSNR/RME/Pearson，§7.5 判据自动判定；产物写入 `07_healthy_completion_benchmark/` 并 mark_step。开发级 smoke 通过（单被试 S33、64 窗、SGMD 1 epoch、256 预训练窗，报告标注 dev_run）——smoke 数字无结论意义，但 CP-WOPT 在该缩减设置下 NRMSE 低于 MCIA（0.133 vs 0.181），提示正式基准存在不通过风险。正式一次性冻结报告尚未运行。

- 移除 DB3 任务内比较脚本的误导性补全指标：原 `completion_nrmse/psnr/rme` 拿补全结果与掩码位置**被录制的坏信号**比较（死通道录到精确零值，全零预测会得满分），衡量的是"与坏信号的距离"而非补全质量；从 `run_task_matched_literature_baselines.py` 的被试报告、汇总 CSV/JSON 与图像中删除该列，`_masked_metrics` 保留并加 docstring 限定其仅用于有人工遮挡真值的基准（由 §7.5 入口复用）；协议 §8 同步改述（DB3 表只报下游终点，补全质量仅在有真值基准上报告），§10 更新 §7.5 入口实现状态。下游角度指标不受影响。

- 按用户决定新增协议 §7.5"补全方法学前提判据（必须结果）"：MCIA 在健康 DB2 受控掩码补全（S33–S40、冻结 ScenarioMix 掩码、masked NRMSE 主判据）上必须同时优于 SGMD-AAE 与 CP-WOPT（均值更低且逐被试 ≥4/8 更优）；锁定公平性规则（SGMD 同源同掩码预训练、CP-WOPT 标注 split-transductive、所有方法统一 clamp→边界淡化→回填交付）、防选择纪律（迭代仅限 S29–S32，S33–S40 每次评估披露）与失败语义（未满足禁称"更优补全方法"，下游结论独立但须并置呈现）；§9.6 要求该判据的一次性冻结报告先于确认 run，§10 标注该对比入口尚未实现。未修改实验代码。

- 按用户四项决定修订协议并解除 §4 阻塞项：(1) **T1 差一错误修复**——`_plateau_stats` 此前返回相等对数（L 个相同采样点产生 L−1 对），`PLATEAU_HARD=100` 实际要求 101 个采样点；改为 `ends-starts+1` 采样点语义后，healthy/healthy_ext/db3 三阶段验收全部刷新归档（旧结果存 `*_pre_t1samplefix/`），所有判定与数字不变（0.15%/0.09%/4.82%，硬零一致性 100%），S05/S06 生产 smoke 一致（0.32%/16.67%）；协议 §4 阻塞句改为已解除并记录修复。(2) **连续点入选规则锁入协议 §7.1**——同一 exercise 内时间连续、每点至少双窗口覆盖、跨 exercise 不拼接，实现锚定 `utils/kinematic_output_postprocess`。(3) **手势识别升为并列主终点**——协议 §1 改为双主任务（重要性相等、独立判据、整体结论需两者同时满足），新增 §7.2：确认集 S03/S07/S09 测试窗口 macro-F1 的配对 `C-A`（均值>0 且 ≥2/3 被试>0），次要为 trial-majority/accuracy/balanced/混淆矩阵；§3 补手势数据定义（E1+E2+E3、48 类、标签纯度过滤），§6 补 GestureTCN 训练公平性（验证 macro-F1 选择），§2 注明被试三分同适于两任务，§9.5 与 §10 同步。(4) **移除文献基线脚本 `Observed_only` 臂**（用户确认不需要），零残留。回归：编译、输出范围测试通过。

- 重写 `docs/EXPERIMENT_PROTOCOL.md` 为当前主实验的可执行协议：明确 A/B/C，其中 C 改为健康 DB2 先验经 S02/S04/S08/S11 无标签截肢者清洁供体适配的 `amputee-donor prior`，而非已退役的目标受试者真值微调；将已查看历史 test 的 S05/S06 限定为开发集，并预留 S03/S07/S09 为冻结后确认集；锁定 200 Hz 表示、WC-BQD 掩码、相同 TCN、连续 global Key10 RMSE 主终点、`C-A` 成功判据，以及任务适配 SGMD-AAE/CP-WOPT 的补充比较边界。未修改实验代码或启动训练。

- 新增"验证脚本单一来源"规则并完成全仓审计（用户指示）：AGENTS.md 最小代码修改规则新增第 7 条——验证/诊断/评估脚本引用主链路逻辑（模型构建、损失、掩码生成与质量检测、补全交付规则、主实验指标与聚合口径）必须 import 主模块、不得复制实现，并建议内置漂移绊线（断言主模块判定与本地重算一致）；单元测试按规格独立重算期望值、已冻结留档的历史脚本豁免。审计结果：**合规**——文献基线脚本 MCIA 交付走 `EXP3.make_enhanced_pool`（SGMD/CP 为外部方法自身交付，无主链路逻辑可引用）、`scripts/04`/`05` 为主链路本体且共享 `patch_boundary_crossfade`、消融脚本为 import+子类扩展、`diagnose_wcbqd_detector.py` 本日已单一来源化；**修复一处违规**——`dev/ablate_structural_loss_terms.py` 的 evaluate 内联交付（写于边界淡化采纳前、已过时）改用 `complete_with_mask`（clamp→边界淡化→回填），合成 smoke 通过；**豁免并留档**——7–8 月 DB3 oracle/alpha/gain/sparse/error-gated/task-aware 等一次性诊断与 09-08 的 softplus/range_penalty/output_activation 筛选脚本内联 copy-back 均为已结论冻结记录，不重跑。

- 消除 WC-BQD 验证脚本与主模块的双实现漂移风险（用户指出）：`test/diagnose_wcbqd_detector.py` 删除本地 T1/T2/T3 与常量副本，全部改从 `utils/db3_quality_mask` 导入（`second_features`/`_fit_baselines`/`_wcbqd_flags`/常量）；诊断包装保留 z 与独立分测试明细（成分分析口径与既有验证记录一致），并内置漂移绊线——每秒断言主模块并集判定与本地重算逐位一致，主模块改动导致的语义漂移会在复验收时立即报错。新增诊断专用 `second_rms`（主模块特征不含 rms，激活相关性分析所需）。等价验证：S05/S06 诊断路径重建的 200 Hz 掩码与生产 `db3_quality_mask` 输出**逐位一致**（missing 0.32%/16.67%）。注意：本文件 sha256 因此变更，既往 `06_diagnostics/wcbqd_validation_20260909/` 各 results.json 中记录的运行时 sha 属于重构前版本，其数字仍有效（逐位等价已证），复验收使用新版本脚本。

- 固定项目执行环境为 Conda `py311`：在 `AGENTS.md` 明确要求所有依赖 NumPy、PyTorch 或 Matplotlib 的 Agent 命令、smoke、训练和评估均使用 `conda run -n py311 python ...`，并记录 PyCharm 对应解释器路径；自动打开的 `base` 环境不再作为项目依赖检查依据。

- 清理旧掩码规则残留（用户核查要求）：`scripts/05` 的 quality_mask 元数据描述与 `scripts/04` 的掩码规则标签更新为 WC-BQD v0.1；删除 `data/dataset_db3_emg.py` 中零消费方的 `DB3EMGDataset` 死类（weak/abnormal 阈值辅助退化掩码，删除前经 rg 确认无引用）及 `torch.utils.data.Dataset` 导入；删除 config 中四个死键（`weak_threshold`/`abnormal_threshold`/`channel_anomaly_ratio`/`augmentation_mask_mode`，主链路与手动脚本均不读取）。保留并如实标注：`utils/db3_quality_mask.py` docstring 中的历史说明、CHANGELOG/设计文档/冻结诊断中的 MQP 实验记录；更老的 `utils/anomaly_detect.py` 与 `utils/rule_anomaly_detector.py` 不在主链路但仍被手动掩码构建脚本、legacy paper_figures 与冻结测试引用，连同 `generate_figures_from_run` 对 `transfer_augmentation_mask_mode` 的默认值兜底读取一并保留，待这些工具退役时一并处理。回归：消费方编译、输出范围测试通过；主链路复扫零 MQP/死键残留。

- 新增任务内可比的文献基线入口 `scripts/run_task_matched_literature_baselines.py`，并将全流程可选 literature 阶段切换至该入口：MCIA、SGMD-AAE、CP-WOPT 均在 DB3 上接收相同的 200 Hz 包络、同一质量掩码，随后以固定划分、重新初始化且同配置/同种子的 Key10 KinematicTCN 评估；输出 masked NRMSE/PSNR/RME 与角度指标。SGMD-AAE 以与 MCIA 相同的健康 DB2 训练被试及 ScenarioMix 预训练；CP-WOPT 按训练/验证/测试分区的无标签观测 EMG 拟合，结果明确标记为 split-transductive，避免将其误述为归纳式推理。原 2 kHz、240 点论文格式复现脚本仍为独立入口，不再作为主链路对比结果。

- 采纳 WC-BQD 替换主链路 MQP 层（用户确认，统一替换）：重写 `utils/db3_quality_mask.py` 第二层为通道内基线检测器（T1 幅值条件化平台/死区、T2 低能量稳健 z、T3 低频主导爆发，常量预声明，基线仅用训练 repetitions 1/3/4 活动秒，训练秒不足 4 时降级为仅 T1 并落盘记录），硬零层保留；函数签名与返回契约不变，`dataset_kinematics`/`dataset_db3_emg`/`scripts/05` 零改动；元数据更新为 `mask_rule=hard_zero_train_1_3_4 OR wcbqd_v0.1_...` 并落盘常量与逐秒警报计数。删除 `mqp_probability`/`_mqp_flags`（无主链路消费方）；当日两个冻结校准脚本因此不可重跑，`reproduce_gronlund_db3_quality.py` 自包含不受影响。真实数据 smoke 与验收数字一致（S05 missing 0.32%、S06 16.67% 且 Ch9/Ch10 100% 硬零，t1/t2t3 计数落盘）。新增设计文档 [docs/WCBQD_DESIGN.md](docs/WCBQD_DESIGN.md)（动机、设计、逐组件文献支撑、验证证据、采纳影响）并加入 AGENTS.md 路由表。影响：DB3 补全缺失比例 18.6%→约 4.6%（S05 窗口级 13.1%→0.3%），新旧 run 的 B 组产物不可直接混比；下游全量证据待正式 run 复核（C4 为 S05 单被试开发预算）。

- 更正 WC-BQD DB3 验收的 C3c 判定并撤回一个错误结论：初版汇总把 9 个被试的硬零通道取跨被试并集后套用到全部被试，被无硬零被试的同位置活通道稀释成 26.9%，由此得出"硬零层丢弃 73% 活数据"的结论是**错误的，予以撤回**。逐被试正确计算：硬零通道仅出现在 S06/S07（Ch9/Ch10），二者各自 held-out 秒的 T1 平坦一致性均为 100%——通道为永久性断流，硬零层判断正确，C3c 判定改为**通过**（≥95%）。C3b 通道维度的 26.9% 集中同样完全来自这两个被试的死通道，属正确检出而非检测混淆；硬零层无需重新设计。`stage_db3` 已改为逐被试一致性并保留并集口径仅作标注（"不用于判定"）；含 bug 的旧结果归档为 `db3_v0_unionmask_bug/`，修正结果覆盖 `db3/`。

- 完成 WC-BQD（通道内基线质量检测器，bottom-up 组合：T1 幅值条件化平台/死区、T2 低能量稳健 z、T3 低频主导爆发）全部四条预声明验收（`test/diagnose_wcbqd_detector.py`，纯诊断不动主链路，产物在 run 的 `06_diagnostics/wcbqd_validation_20260909/`）。健康人零校准（DB2 S01–S10 及功效扩展 S11–S20，held-out reps）：池化 flag 率 0.15%/0.09%（C1 通过），通道 Pearson 0.481（S01–S10 形式失败、91 事件功效不足且低于显著性临界）/-0.030（S11–S20 通过），合并 149 事件 Pearson 0.375 统计上不显著，对比 MQP 的 0.952（约 1.8 万事件）；Ch11/12 零 flag（C2b 通过）。DB3 9 被试 held-out：池化 4.82%（C3a 通过，MQP 参照 18.58%）、动作维度 4.0–8.8% 平坦（通过）；通道维度 Ch9/Ch10 26.9%（C3b 形式失败，来源为 S06/S07 永久死通道的正确检出，见上方更正条目）；C3c 经逐被试修正后通过。下游 S05 开发 A/B（C4 通过）：raw 0.17353 / 现行 MQP 掩码 0.18073 / WC-BQD 掩码 0.17406——B 相对 A 的 RMSE 差距从 +4.15% 缩到 +0.30%，证明 MQP 过度掩码是此前 B 劣势的主因之一。检测器经历一次健康零分布驱动的正当校准迭代：v0 的 T1 被 DB2 Ch3/Ch10 量化重复样本触发（dupfrac 规则），v0.1 改为"≥10 样本平台且处于该秒 P99.5 幅值极端（削顶）或 ≥50ms 平坦（死区）"，v0 失败结果归档为 `healthy_v0_t1quantization/`。是否采纳（替换 `db3_quality_mask` 的 MQP 层）留待用户决策。

- 记录 run_20260909_194559_1 失败：19:45:59 启动，19:48:22 死于 Exp1 训练开始阶段（数据准备与 init checkpoint 已完成，无 epoch 产出、无 traceback 落盘）；同一时间窗内一个 GPU 诊断任务（WC-BQD C4，19:46:35 停止）并行在跑，最可能原因为双进程 GPU 争用（OOM/CUDA 错误），未确证。全链路启动前必须确认无其他 GPU 任务；该 run 的 40 被试数据缓存完好，重启可复用。

## 2026-09-09

- 完成 MQP flag 的通道×动作双侧分解（DB3 侧 `test/decompose_mqp_flags_db3.py` 读现有 NPZ；健康人侧 `test/decompose_mqp_flags_healthy.py` 重载 DB2 S01–S10 并补存 action/RMS 元数据；均 0.20 阈值、train 侧数据、只读或仅写 06_diagnostics）。两侧一致结论：**动作维度平坦**（DB3 top5 动作 19–20%≈总体，健康人动作区间 12.9–16.2%），**通道维度强结构化**（DB3 Ch9–12 高达 23–47%，含上臂电极 Ch11/12；健康人 Ch1/7/8/10 达 26–41%，通道间差 20–47 倍），**健康人通道 flag 率与该通道中位 RMS 的 Pearson=0.952，被 flag 通道-秒的中位 RMS 是未 flag 的 5.0 倍**，58–75% 被 flag 秒有 ≥2 通道共现。机制判定：MQP 的跨通道离群框架把"通道间正常幅值差异"（高激活主力通道与近静默上臂通道两个方向）当作异常，健康本底为生理性；检测器混淆激活水平与信号质量。修复方向明确为通道内时序基线检测（每通道与自身历史比较），调全局阈值无法解决该混淆。DB3 侧 RMS 对照因 NPZ 无幅值元数据未计算；产物在 run 的 `06_diagnostics/mqp_flag_decomposition_{db3,healthy}_20260909/`。

- 新增独立文献补全基线实现：`models/baselines/sgmd_aae.py` 按 Zou et al. (2023) 实现 SGMD-AAE 的 240×12 self-mask PartialConv U-Net、多尺度时域/FFT 频域双判别器、self-guided/style/center-alignment/adversarial 损失和论文报告的损失权重/优化器学习率；`scripts/run_literature_completion_baseline.py` 可在显式 `MCIA_RUN_DIR` 下运行 SGMD-AAE 或 CP-WOPT。它们不接入默认主流程，SGMD-AAE 仅接受论文一致的原始 2kHz 120ms 240×12 输入。

- 将本地论文材料目录 `apply/` 加入 `.gitignore`，避免 SGMD-AAE 参考 PDF 被纳入版本控制；该目录仅用于复现时的本地文献核验。

- 新增独立文献基线模块 `models/baselines/cp_wopt.py`：按 Akmal et al. (2019) 的加权 CP/PARAFAC 目标和 Hestenes-Stiefel 非线性共轭梯度实现 CP-WOPT，并提供 RME 及“仅替换缺失值、保留观测值”的交付接口；新增合成低秩张量公式级 smoke，不接入 MCIA 或主实验链路。

- 将 SGMD-AAE 与 CP-WOPT 文献基线接入 `run_all_experiments.py` 的可选末尾阶段：使用 `--with-literature-baselines` 或 `literature_baselines.enabled` 显式启用，顺序运行并将日志、输入与指标隔离在当前 run 的 `06_diagnostics/literature_baselines`。

- 新增 DB2 原始 2 kHz、49.5--50.5 Hz 三阶 Butterworth 带阻、240x12 非重叠窗口准备入口，避免将 Exp1 的 200 Hz、256 点包络缓存误作 SGMD-AAE 论文输入。

- 按用户确认将 `literature_baselines.enabled` 设为 `true`；此后默认 `run_all_experiments.py` 在主链路成功后顺序执行 SGMD-AAE 与 CP-WOPT 独立复现实验。

## 2026-09-10

- 新增 `scripts/run_literature_baselines.py`：PyCharm 可直接运行的独立文献基线管线，每次创建新 run，顺序准备论文格式 DB2 输入并运行 SGMD-AAE、CP-WOPT，不启动或改动 MCIA 主实验链路。

- 修正 SGMD-AAE 文献基线输入尺度：在 2 kHz 带阻滤波及 240x12 分窗后，逐样本 min-max 归一化至 `[0,1]`，并记录 `input_preprocessing.json`；此前直接使用原始伏特量级会使论文定义的 NRMSE 与网络训练尺度失配。

- 修正 SGMD-AAE 生成器的论文结构实现：按 Table 1 固定编码层尺寸 `240x12→24x12→8x4→4x2→2x1→1x1`，对偶数核解码层采用非对称 SAME padding；并按 Eq. (2) 将 PartialConv bias 放在 self-mask 缩放之后，避免 bias 被错误放大。

- 修正 SGMD-AAE 训练/推理输入维度：移除将 `(batch,time,channel)` 错置为 `(batch,channel,time)` 的 permute，确保模型实际接收论文 Table 1 所定义的 `240x12` 时序×通道布局；该错误由严格尺寸修复后的编码层检查暴露。

- 清除同一 SGMD-AAE 训练入口中掩码张量的残留维度转置，使训练目标与 self-mask 均为 `(batch,1,240,12)`。

- 完成 MQP 阈值的健康人经验零校准（`test/calibrate_mqp_threshold_healthy_db2.py`，DB2 S01–S10 训练侧被试、E1+E2 活动段、与 DB3 掩码完全相同的 `mqp_probability` 代码路径、判定规则预先写死为“池化健康 flag 率 ≤5% 的最小网格阈值”、测试被试未触碰）：**预定规则无解**——健康人池化 P95=0.7031，网格内（0.05–0.40）最低 flag 率为 0.40 处的 8.99%；当前 0.20 阈值下健康人 flag 率 14.48%（逐被试 12.14–22.38%），DB3 截肢者为 18.58%，差距仅约 4 个百分点且随阈值升高收窄。结论：“健康≈干净”前提在该检测器上不成立，MQP p 在健康数据上有高本底（疑对正常生理/通道间异质性敏感），5% 目标不可达；若强行取健康 P95=0.70，DB3 掩码将降至 4.76%。不据此改阈值；是否调整留待用户决策，建议后续先做健康 p 本底的逐通道/逐动作分解诊断。产物在 run 的 `06_diagnostics/mqp_healthy_null_calibration_20260909/`。

- 完成 range_penalty 配对筛选（`test/diagnose_range_penalty_screen.py`，与 2026-09-08 Softplus/Sigmoid 筛选同模式：DB2 S01/S02 训练→S03 验证、同初始化、同掩码种子、8 epoch 短预算、两臂仅 `range_penalty_weight` 0 vs 0.5 不同、评估统一用已采纳交付规则、无测试被试、无 checkpoint 选择）：机制生效（clip 前越界比例 0.25%→0.11%）但收益无稳定方向——末 epoch 加权 RMSE +2.2%、corr −0.0027，而第 3/5/6/7 epoch range 臂 RMSE/corr 占优；越界基数本身（<0.3%）过小，无实际可修复空间。判定：训练期软范围惩罚无信号，不采纳；与 Sigmoid 结论合并，"训练期有界输出"方向在重建层面均无证据支持，予以关闭。产物在 run 的 `06_diagnostics/range_penalty_screen_20260909/`。注意事项：短预算下越界比例可能低估全量训练（120 epochs）水平，但按开发规则无信号不进入采纳流程。

- 采纳 patch 边界三点淡化为补全交付规则并统一替换主链路（用户确认）：`complete_with_mask` 移除 `patch_boundary_smooth` 开关、默认执行"clip → patch 边界淡化 → 复制回观测"；`_patch_boundary_crossfade` 更名为公共接口 `patch_boundary_crossfade`；`scripts/04` 的 `apply_mcia`/`make_enhanced_pool` 与 `scripts/05` 的 `apply_mcia` 同步接入并从 config 透传 `patch_size`。共享该函数的表格/图像再生成入口随之使用新交付规则，其产物与采纳前不可直接混比。依据：三个验证被试重建端一致改善（RMSE −1.5~−1.7%、corr 无损）+ 下游开发 A/B 中补全族内三指标全部最优。

- 删除 MC-Dropout 不确定性门控补全路径（用户确认否决）：移除 `complete_with_mask_uncertainty`、`_linear_gap_fill` 及 `paper_pipeline` 仅为它们引入的 `torch.nn as nn` 导入。依据：重建端占优（−2%）但下游开发 A/B 中劣于默认（RMSE +4.49% vs +4.35%、R² 最低），线性插值回退抹除 TCN 依赖的动态信息，不满足下游判据。三个当日诊断脚本（`diagnose_completion_inference_options.py`、`confirm_completion_inference_options.py`、`diagnose_downstream_boundary_unc_ab.py`）保持运行时冻结以维持 results.json 溯源，其中被删路径的条件不可重跑。smoke 测试改写为交付规则断言（非边界点不变、边界点淡化、观测回填、常数不变性、非整除长度原样返回）；py311 环境通过编译、残留引用扫描、候选 smoke、输出范围测试与掩码生成器自检。

- 完成下游角度开发 A/B（`test/diagnose_downstream_boundary_unc_ab.py`，DB3 S05、E1+E2、训练 repetitions 1/3/4、验证 repetition 6 报告、测试 repetitions 划出后全程未用、四条件同种子同掩码同 checkpoint、TCN 预算压缩为 40 epochs/patience 10 并记录、窗口级口径无连续输出后处理）：raw 验证 RMSE 0.17353；补全默认 +4.35%、边界淡化 +4.15%（MAE +1.52%、R² 0.0470，为补全族内三指标全部最优）、不确定性门控 g=0.15 +4.49%（补全族内最差）。与主 run 测试结果（S05/S06 B 相对 A RMSE +0.5%/+0.2%）方向一致：当前 B 组补全未在下游超过 raw。结论：边界淡化是唯一重建与下游两端均不劣化的候选；不确定性门控重建占优但下游劣于默认，按下游判据不采纳。产物在 run 的 `06_diagnostics/downstream_boundary_unc_ab_20260909/`。

- 完成 S30/S31 独立复确认（`test/confirm_completion_inference_options.py`，与初筛同方法学：同冻结 checkpoint、同固定掩码种子、512 窗口/被试、仅验证被试、无训练；差异仅被试换成 S30/S31 并省略已删除的 refine2 条件）：patch 边界淡化 RMSE −1.67%/−1.65%（S30/S31）、corr +0.0004/+0.0012；不确定性门控 g=0.15 RMSE −1.89%/−1.59%、corr −0.0002/+0.0002。与 S29 初筛（边界淡化 −1.49%、g=0.15 −2.06%）方向和量级一致，g=0.10 的 corr 代价在 S29/S30 复现（−0.0040/−0.0029）、S31 消失，0.15 以上三个被试 corr 均无损。产物在 run 的 `06_diagnostics/completion_inference_options_confirm_20260909/`。重建指标复确认通过，下游角度证据仍未验证。

- 删除 `complete_with_mask` 的多步自精炼参数 `refine_steps` 及其实现：配对开发筛选显示两步自精炼使加权 RMSE +61.4%、corr −0.0150（把自身预测当观测再前向导致误差累积），按无效候选清除，不留开关。`test/diagnose_completion_inference_options.py` 保持运行时原样冻结以维持 `results.json` 中脚本 sha256 溯源，其 `refine2` 条件在库中已无对应实现，不可重跑。同步移除 smoke 测试中该分支的断言；py311 环境通过编译、候选选项 smoke 与输出范围测试。

- 完成推理级候选选项的配对开发筛选（`test/diagnose_completion_inference_options.py`，冻结 Exp1 checkpoint、DB2 验证被试 S29、512 窗口、固定 S1/S2/S3 掩码、无训练、未用测试被试）：patch 边界三点淡化三个场景一致改善（加权 RMSE −1.49%、MAE −1.35%、corr +0.0008）；MC-Dropout 不确定性门控在门限 0.10/0.15/0.20 均降低 RMSE/MAE（最多 −2.90%/−2.42%），门限 0.10 伴随 corr −0.0040、0.15 以上 corr 基本不变；两步自精炼严重退化（RMSE +61.4%、corr −0.0150），判定为无效候选。产物在 run 的 `06_diagnostics/completion_inference_options_20260909/`。该筛选仅用重建指标做开发初筛，采纳与否仍需更多验证被试复确认与下游角度证据。

- 新增 MCIA 候选优化能力（默认全部关闭，不改变主流程行为）：`EMGImputationLoss` 可选缺失区软 [0,1] 范围惩罚（`range_penalty_weight`）；`MCIA` 可选零初始化包络旁路 `EnvelopeBranch`（`use_envelope_branch`，加载旧 checkpoint 时恒等，不注入无条件 CFG 分支）；`complete_with_mask` 新增 patch 边界三点淡化和多步自精炼参数；新增 `complete_with_mask_uncertainty` MC-Dropout 不确定性门控补全（低方差采样点用 MC 均值，高方差采样点回退观测锚定线性插值）。删除 `MCIA_Wrapper.q_sample` 无调用方残留。

- 新增 `RuleAlignedMaskGenerator`：从训练 repetitions 质量掩码库重采样 patch 对齐循环平移训练掩码，契约同 ScenarioMix（二值、patch 对齐）；因 Exp2 个体适配链路已退役，当前无主流程消费方，保留为未来无真值 DB3 适配目标的组件。

- 清理 `config.yaml` 死键（`spatial_depth`/`num_heads`/`mlp_ratio`/`pool_*`/`fuse_mode`/`temporal_depth`/`use_mask_swap`/`use_group_bias`/`use_spatial_mixer` 及全部 `alpha_*`/`beta_smooth`，代码无引用）；移除已退役 C 组路径对应的 `adapt_positional`/`training_mask_mode`/`completion_uncertainty`/`completion_patch_boundary_smooth` 键，按约定不恢复任何 C 组（个体适配）改动，`02`/`03` 保持退役占位。

- 新增两个开发/验证入口：`dev/ablate_structural_loss_terms.py`（DB2 S01/S02→S03 小规模结构损失项剪枝消融，不用测试被试）与 `test/test_candidate_upgrade_smoke.py`（合成数据覆盖上述全部新选项：范围惩罚开关、规则掩码契约、包络旁路零初始化恒等、边界淡化非边界保持、多步精炼、不确定性门控回填与训练模式恢复）。py311 环境通过编译、25 项 smoke、掩码生成器自检、输出范围测试及 loss/model 构建与前向；未运行完整主流程、真实数据训练或下游有效性实验，新选项下游收益均未验证。

## 2026-09-08

- 移除 DB3 人工遮挡原始 sEMG、再以原始记录作为重建真值的个体适配训练及其 checkpoint、增强和人工遮挡表格消费链路；Exp3 默认仅运行 A/B，旧 DB3 适配产物保留为历史文件但不能被新主流程读取。`02`、`03` 和旧表格入口改为明确停用提示，等待无真值自监督方案重新定义和验证。

- 恢复 Exp4 的 A/B 主链路：A 使用原始 DB3 sEMG，B 使用健康先验在同一质量掩码下的补全 sEMG；删除 C checkpoint、adapter、`domain_id=1`、C 分类器和 C 指标消费，默认全流程重新包含手势识别 A/B。

- 修复补全输出范围 smoke 的测试调用：将遗留 `Detector` 存根改为与 `apply_mcia` 现有签名一致的预计算 mask 数组，并将实际 PNG 写入可写的 `outputs/matplotlib_diagnosis`；在 `py311` Conda 环境通过模型输出范围、观测值回填、严格 state load、`apply_mcia` 与实际 Matplotlib 保存 smoke。

- 手势识别主链路改用 DB3 两层质量掩码：训练 repetitions 的严格零值硬缺失与 Gronlund（2005）MQP 固定 p>0.20 的逐通道一秒异常段取并集；补全、保存与下游输入共享该掩码，并将固定 48 类被试及其受试者适配范围扩展至 S02/S03/S04/S05/S06/S07/S08/S09/S11。

- 将相同 DB3 两层质量掩码同步到 Exp2c 增强 EMG 生成和 Exp3 连续角度 A/B/C 的补全输入，移除这两条主链路对旧 RuleAnomalyDetector 的依赖。

- 新增独立只读诊断脚本，逐公式复现 Gronlund 等（2005）的 MQP 多通道 sEMG 质量估计，并限定用于 DB3 固定 48 类被试的训练 repetitions；不修改主异常掩码、MCIA 或下游实验链路。

- 配置全局 GitHub MCP 为官方 repos 只读入口及七个仓库/代码读取工具，新增认证说明；TOML 和 Codex 配置读取验证通过，缺少 PAT，尚未验证远程访问。

- 在 DB2 S01/S02、E1、训练 repetitions 1/3/4 和验证 repetition 6 上完成 Q95/Q97.5/Q99 的固定初始化、掩码和训练预算补全对比。Q99 的饱和比例及验证 RMSE/MAE 最低，用户确认采用；主链路的 DB2/DB3 补全、角度/手势输入及共同鲁棒归一化工具统一改为 Q5/Q99，移除候选分位数诊断路径。未使用测试 repetitions，未改写历史产物。

- 在 DB2 验证被试 S29 的 512 个窗口上，以相同 checkpoint、S3 掩码和输出头线性参数完成 Softplus 原始输出、Softplus+clip 与直接替换 Sigmoid 的开发性推理对比；clip 降低 masked MSE/MAE 并提高 masked correlation，Sigmoid 的 masked MSE/MAE 更高。未读取 DB2 测试被试，未写入 run 或 checkpoint。新增可复现的限定诊断脚本。

- 统一 MCIA 补全后处理：预测裁剪至 [0,1] 后复制回观测值，覆盖补全生成、重建评估/绘图、角度与手势输入及引导推理；保留 Softplus、checkpoint 和训练计算，不改写历史产物。新增合成超界输入的调用路径检查，在 Windows py311 环境通过语法检查、输出范围/观测复制/参数严格加载检查、Matplotlib 诊断与实际补全绘图 smoke；未运行完整主流程或下游有效性实验，本变更仅修正输出范围。

- 接入全局科研证据规则、academic-research Skill 和独立 Semantic Scholar MCP，在项目 AGENTS.md 增加路由及安装说明；验证全局发现、配置读取与 MCP 握手，记录公共搜索 429 限流及网页原文备用路径通过，未改动实验代码或 conda 环境。

- 将 AGENTS.md 拆分为通用规则与 source-of-truth 路由，新增实验协议及 Windows、绘图、可复现专项规则；整理当前实现并标明待锁定的主终点，未修改实验代码或配置。

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
