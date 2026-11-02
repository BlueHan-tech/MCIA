# 待改事项

记录待讨论的方案想法，避免点子直接变成代码修改。状态为“待讨论”或“已决定”；确认后更新[协议](EXPERIMENT_PROTOCOL.md)，并将决定记入[项目决定与实验记录](COLLABORATION.md)。

## 已决定，待实施

### 全链路统一窗口设置

- 已决定：生成式补全、手势识别和连续关节角度估计共用 300 ms 窗口、250 ms 重叠、50 ms 步长；按用户要求，等用户明确要求修改项目时再实施。
- 样本数按采样率换算：2 kHz 下为 600 点窗口、100 点步长；200 Hz 下为 60 点窗口、10 点步长。统一的是时间范围和更新间隔，不是所有模型都必须使用同一采样率或输入表示。
- 启发与边界：DSRANet 在 DB3 手势识别中报告 300 ms 外层窗口、250 ms 重叠；窗口内 STFT 另用 64 ms Hamming 窗、16 ms 帧步长。该论文支持手势链路上的可行性，但不能证明同一设置对补全和连续角度估计最优，也不能把精度差异归因于窗口重叠。详见 [DSRANet](https://doi.org/10.1016/j.bspc.2025.108845) 及[对应方法专利](https://patents.google.com/patent/CN120256824A/en)。

### 全链路采样率

- 已决定：补全输入/输出为 2 kHz 原始肌电；手势识别使用补全后的 2 kHz 信号；角度估计将同一补全结果整流、平滑并降采样为 200 Hz 包络。目标窗口分别为 2 kHz 下 600 点/步长 100 点，200 Hz 下 60 点/步长 10 点。
- 依据与边界：DB3 的 DSRANet 以 2 kHz 肌电生成 STFT 表示用于手势识别；DB2 角度估计论文将 2 kHz 肌电整流、均值滤波并降采样十倍至 200 Hz。文献支持分支选择，但不证明 2 kHz 原始波形补全会优于 200 Hz 包络补全。[DSRANet](https://doi.org/10.1016/j.bspc.2025.108845)；[DB2 角度估计](https://doi.org/10.1002/aisy.202500590)。
- 风险：当前补全模型处理 200 Hz 包络；改为补全 2 kHz 原始波形会改变重建目标并增加计算量。50 ms 步长要求持续处理每秒 20 个窗口；需在开发数据上验证补全质量、两个下游任务及吞吐。代码尚未修改，按用户此前要求等明确实施指令。

## 待讨论

### Butterworth 低通滤波

- 当前：协议列出带通、50 Hz 陷波、整流、10 点包络和降采样；未定义在原始肌电或 STFT 前额外增加独立 Butterworth 低通。角度预测输出端的 6 Hz Butterworth 是另一处理步骤。
- 待讨论：是否增加低通、加在原始肌电还是整流/包络之后、是否服务于降采样抗混叠，以及截止频率、阶数和因果实现。
- 证据边界：DSRANet 的已核对方法段给出 STFT 参数，但没有明确说明 Butterworth 低通；不能据此把低通视为该高精度结果的必要条件。

### D-009/D-010 实施所需代码改动范围

- 说明：窗口与采样率方案已经决定；本项只讨论落实时必须同步的代码入口和接口，不重新讨论方案本身。代码当前未改。
- 配置与分窗：`config.yaml` 和 `utils/paper_pipeline.py` 目前用一组 `target_fs/window_size/stride` 控制全链路。需改为按时间定义 300 ms/50 ms，再分别换算：2 kHz 补全和手势为 600/100 点，200 Hz 角度为 60/10 点；同步传给训练、适配、推理和 run 配置记录。
- 数据与质量掩码：`data/dataset_db2_emg.py`、`data/dataset_db3_emg.py`、`data/dataset_kinematics.py` 和 `scripts/05_eval_db3_gesture_raw_vs_augmented.py` 当前会先生成 200 Hz 包络。需让补全和手势分支使用 2 kHz 信号；角度分支从补全结果按既定预处理生成 200 Hz 包络。`utils/db3_quality_mask.py` 当前只返回按目标采样率降采样后的掩码，需提供与 2 kHz 补全/手势对齐的掩码，并保证角度分支降采样后的掩码、标签和 EMG 时间对齐。
- 补全模型与训练目标：`models/completion/mcia_core.py`、`models/completion/mask_generators.py` 和 `utils/loss_functions.py` 中，窗口长度、patch、包络/RMS 核和 STFT 尺寸均以采样点表示；`MCIA` 输出头为非负 Softplus，交付路径还会裁剪至 `[0,1]`。新目标是有正负值的 2 kHz 原始肌电，需一并明确归一化、输出反归一化/范围、损失时标、掩码 patch 时长和模型形状，不能只把 `target_fs` 改成 2000。`scripts/01_train_mcia_db2_healthy_prior.py` 与 `scripts/06_adapt_amputee_donor_prior.py` 是相应训练/适配入口。
- 下游评估与交付：`scripts/04_eval_db3_angle_raw_vs_augmented.py` 和 `scripts/05_eval_db3_gesture_raw_vs_augmented.py` 需分别使用 200 Hz 和 2 kHz 的窗口、标签与掩码对齐方式，并在 A/B/C 间保持同一解码器及数据划分。连续补全窗口之间还需定义融合/边界处理，再将同一 2 kHz 补全结果送入手势与角度分支；50 ms 更新意味着每秒处理 20 个窗口，需记录是否能持续跟上。`research/` 中引用主链路预处理的诊断入口也应改为导入主模块或明确标为历史脚本。
- 兼容边界：现有 200 Hz 包络 checkpoint、预测和 run 产物不能直接当作 2 kHz 原始波形版本；新旧结果需按采样率和表示标记，旧产物保留为 legacy。实现时还需处理 §8 中旧 200 Hz 文献 baseline 的匹配口径（见 V-004）。

### 截肢者肌电的质量判定

- 想法：区分相对正常与异常肌电；正常段用于有参照地评估补全，异常段在补全时按缺失处理。
- 待讨论：如何定义判定规则，以及“正常”是否足以作为可靠真值。当前尚未形成可采用的规则。
