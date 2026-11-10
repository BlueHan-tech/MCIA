# 研究诊断归档

本目录包含探索性审计、消融、Oracle 筛查和历史复现。这些脚本不属于主实验
流水线，并被明确排除在 `test/` 之外，因此测试发现只会找到可运行的冒烟测试
或契约检查。

## 保留的核查工具

- `audit_db2_db3_raw_emg_quality.py`、`audit_db3_glove_label_quality.py` 和 `plot_db3_s01_raw_semg_by_task.py` 用于原始肌电、手套标签和波形核查。
- `smoke_scenario_mix.py` 和 `print_mcia_params.py` 用于冒烟检查和参数量核对。ScenarioMix 检查随 2 kHz 主链路迁移时更新。

## 首批历史归档

归档前的源码快照为 Git 提交 `f0c3f8f167a9da36c1a4f989bfe1501dc820ba65`。复核源码及其加载关系时使用相应历史版本和原路径。下列脚本是历史记录，不在当前协议下重跑。源码快照用于追溯清理前的文件位置，各实验的实际版本以 run 元数据为准。

- 旧质量检测移入 [retired/](retired/README.md)，包括 `calibrate_mqp_threshold_healthy_db2.py`、`decompose_mqp_flags_db3.py`、`decompose_mqp_flags_healthy.py` 和 `reproduce_gronlund_db3_quality.py`。Grönlund 对照产物位于 `run_20260907_154902_1/06_diagnostics/gronlund_2005_mqp_db3`。MQP 校准和分解产物位于 `run_20260908_191852_1/06_diagnostics/` 下的 `mqp_healthy_null_calibration_20260909`、`mqp_flag_decomposition_db3_20260909` 和 `mqp_flag_decomposition_healthy_20260909`。
- [WC-BQD 验收脚本](diagnose_wcbqd_detector.py) 原地归档，保留项目规则和[旧验收记录](../docs/COLLABORATION.md#待核验问题)的链接。产物位于 `run_20260908_191852_1/06_diagnostics/wcbqd_validation_20260909`。
- 旧补全消融移入 [retired/](retired/README.md)，包括 `diagnose_range_penalty_screen.py`、`diagnose_softplus_clip_vs_sigmoid.py`，以及成组保留的 `diagnose_mcia_partialconv_loss_ablation.py`、`diagnose_mcia_output_heads_ablation.py`、`diagnose_mcia_unet_small_ablation.py`。范围惩罚和输出激活产物分别位于 `run_20260908_191852_1/06_diagnostics/range_penalty_screen_20260909` 和 `run_20260907_154902_1/06_diagnostics/softplus_clip_vs_sigmoid_20260908`。
- 旧 Oracle 筛查和专项报告原地归档。Oracle 组包含 `diagnose_db3_completion_oracle.py`、`diagnose_db3_oracle_feature_screen.py`、`diagnose_db3_oracle_patch_selector.py`、`diagnose_db3_oracle_tcn_sensitivity_screen.py` 和 `diagnose_db3_mc_dropout_oracle_gain.py`，各自写入所选 run 的 `06_diagnostics/`，子目录名为脚本名去掉 `diagnose_` 和 `.py`。`report_angle_abc.py` 和 `report_gesture_abc.py` 分别写入 `06_diagnostics/report_angle_abc_20260910` 和 `06_diagnostics/report_gesture_abc_20260910`。当前工作区未找到这批 Oracle 和专项报告的对应产物，以上子目录名来自脚本。

## 从已有 run 重建图表

在 PyCharm 中运行 [正式绘图入口](../scripts/generate_figures_from_run.py)，在运行配置的环境变量中设置 `MCIA_RUN_DIR` 为所选 run 目录的绝对路径。

## 其他诊断

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
