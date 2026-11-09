# Mig-PPO V2 多环境筛选与拟合报告

Run `mig_ppo_v2_multienv_20261105T130307Z`；选择阶段：`complete`；CPU 仿真验收：`complete_with_explicit_pending_checks`。

单个配对通过仅表示该冻结 trace 上 `gain_C > 0` 且有成功迁移证据，不代表统计显著或跨 seed 稳健。

| 场景 | 热点位置 / 实际占比 | 路由 | QPS | Stay C | Balanced C | gain_C | 迁移成功 / 请求 | 选择状态 | 拟合 | 独立验证 / 仿真 |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
| HOT1-L | 1 / 50.000% | fixed | 1.000 | 224097.177 | 221836.577 | 0.010 | 16 / 15 | selected | fit_available | complete / pending or failed |
| HOT1-M | 1 / 70.833% | fixed | 0.600 | 235264.777 | 219657.056 | 0.066 | 22 / 20 | selected | fit_available | incomplete / pending or failed |
| HOT1-H | 1 / — | fixed | — | — | — | — | 0 / 0 | skipped_failed_candidate | not_in_fit_set | not_run_selection_ineligible / pending or failed |
| HOT2-L | 2 / 50.000% | fixed | 0.600 | 227427.830 | 210303.616 | 0.075 | 12 / 12 | selected | fit_available | complete / pending or failed |
| HOT2-M | 2 / 68.750% | fixed | 0.600 | 233192.310 | 207263.824 | 0.111 | 16 / 15 | selected | fit_available | complete / pending or failed |
| HOT2-H | 2 / 89.583% | fixed | 1.000 | 287642.047 | 235859.493 | 0.180 | 34 / 34 | selected | fit_available | complete / pending or failed |
| HOT3-L | 3 / 50.000% | fixed | 0.600 | 218796.433 | 213179.126 | 0.026 | 12 / 12 | selected | fit_available | complete / pending or failed |
| HOT3-M | 3 / 70.833% | fixed | 0.600 | 233207.026 | 216353.257 | 0.072 | 22 / 21 | selected | fit_available | complete / pending or failed |
| HOT3-H | 3 / 89.583% | fixed | 1.000 | 259969.329 | 229150.846 | 0.119 | 32 / 31 | selected | fit_available | complete / pending or failed |
| HOT4-L | 4 / 50.000% | fixed | 0.600 | 221317.936 | 217145.901 | 0.019 | 14 / 13 | selected | fit_available | complete / pending or failed |
| HOT4-M | 4 / 70.833% | fixed | 0.600 | 238225.330 | 219728.465 | 0.078 | 21 / 21 | selected | fit_available | complete / pending or failed |
| HOT4-H | 4 / 89.583% | fixed | 1.000 | 260068.190 | 229470.534 | 0.118 | 29 / 28 | selected | fit_available | complete / pending or failed |
| NATIVE | — / — | native | 0.600 | 214089.639 | 214600.973 | -0.002 | 3 / 3 | test_only_no_positive_gain | fit_available | complete / pending or failed |

## 适用范围与限制

- `NATIVE` 保持 `test_only`；新请求路由通过本任务模拟分发器重新决策，不能把它作为 PPO 训练环境。
- 物理拟合来源是一对开发 trace；重复决策样本并非独立实验。迁移暂停时间与失败概率没有足够样本时保持 unavailable。
- 只有 `sim_pretraining_ready=true` 的人工热点环境可进入后续 PPO 仿真采样；本报告不会启动训练。
- 完整状态曲线、奖励压力、接口容差及迁移尾部任一项未审计通过时，仿真 ready 保持 false。

详见 `model_registry.json`、`reward_frozen.json`、`reward_reconstruction.json`、`sim_validation_summary.json`、`cleanup_verification.json`。

## 独立验证与完整验收判定

validation_gain_C 来自模型冻结后新 seed 的独立真实配对；simulated_gain_C 来自同一 validation trace 的 CPU 闭环；selection_gain_C 是开发筛选配对。完整任务书验收还要求所有逐项曲线、接口、奖励与迁移合同。

| 场景 | 候选 | 筛选 gain_C | 独立验证状态 / gain_C | 仿真 gain_C | gain 方向一致 | 聚合 C/gain 门 | 完整物理验收 | PPO ready |
|---|---|---:|---:|---:|---|---|---|---|
| HOT1-L | HOT1-L_backup_n48_q1p0_s26110523 | +1.01% | complete / -62.32% | -0.41% | 是 | 失败 | failed_D_pressure_curve | 否 |
| HOT1-M | HOT1-M_primary_s26110521 | +6.63% | incomplete / — | — | — | — | not_run_or_incomplete_pair | 否 |
| HOT1-H | — | — | not_run / — | — | — | — | not_run_or_incomplete_pair | 否 |
| HOT2-L | HOT2-L_primary_s26110521 | +7.53% | complete / +3.96% | -0.38% | 否 | 通过 | failed_D_pressure_curve | 否 |
| HOT2-M | HOT2-M_backup_n32_q0p6_s26110522 | +11.12% | complete / +7.39% | -0.58% | 否 | 通过 | failed_D_pressure_curve | 否 |
| HOT2-H | HOT2-H_backup_n48_q1p0_s26110523 | +18.00% | complete / +12.58% | -0.50% | 否 | 失败 | failed_D_pressure_curve | 否 |
| HOT3-L | HOT3-L_primary_s26110521 | +2.57% | complete / +3.11% | -0.29% | 否 | 通过 | failed_D_pressure_curve | 否 |
| HOT3-M | HOT3-M_primary_s26110521 | +7.23% | complete / +6.96% | -0.36% | 否 | 通过 | failed_D_pressure_curve | 否 |
| HOT3-H | HOT3-H_backup_n48_q1p0_s26110523 | +11.85% | complete / +12.24% | -0.66% | 否 | 失败 | failed_D_pressure_curve | 否 |
| HOT4-L | HOT4-L_primary_s26110521 | +1.89% | complete / +0.15% | -0.28% | 否 | 通过 | failed_D_pressure_curve | 否 |
| HOT4-M | HOT4-M_primary_s26110521 | +7.76% | complete / +9.41% | -0.40% | 否 | 通过 | failed_D_pressure_curve | 否 |
| HOT4-H | HOT4-H_backup_n48_q1p0_s26110523 | +11.77% | complete / +13.55% | -0.61% | 否 | 失败 | failed_D_pressure_curve | 否 |
| NATIVE | NATIVE_primary_s26110521 | -0.24% | complete / -0.32% | +0.00% | 否 | 通过 | failed_D_pressure_curve | 否 |

聚合门通过 7/11 个完整仿真配对；完整验收仍全部未通过：所有 11 个完整配对的 D pressure 曲线未达误差线，且正收益真实场景的仿真 gain 方向相反。HOT1-L 独立收益反向；HOT1-M 超过 200 秒未完成；HOT2-H、HOT3-H、HOT4-H 的独立真实收益为正，但模型误差超线。

全量逐行误差见 `acceptance_detail.csv`；原始 CPU 结果、验收复核和验证索引修正记录见 `sim_validation_summary_before_acceptance_review.json`、`simulation_validation_review.json`、`simulation_validation_amendment.json`。本轮无环境达到 PPO ready，不启动训练。

## 逐场景曲线与事件审计

基于各自首个请求发送点对齐，并按 1.5 秒左保持采样；完整逐点数值见 `simulation_curve_audit.json`。请求数、路由与执行审计在 11 个完整配对中均通过，13 动作合法性均通过。
配对门通过数：outstanding 10/11，waiting 11/11，per-instance KV 8/11，D pressure 0/11，成功迁移计数 11/11。独立真实迁移服务暂停时间没有直接测量，copy/cost 时长不替代暂停。

所有完整配对的 D pressure 曲线均未达到任务书误差线；因此没有环境通过完整物理拟合验收。状态新鲜度、同输入特征/掩码数值等价、token-gap/5 秒速率及服务暂停仍为 pending。
