# Mig-PPO 环境建模与校准结果摘要

## 结论

使用已有 20 组实验日志完成了数据审计、服务/迁移候选模型比较、独立 seed 404 验证，以及 V0–V3 的系统回放；没有重新采集真实数据或新增 GPU 实验。校准只使用 seeds 101/202/303，seed 404 的响应标签没有进入拟合。

**服务迭代模型有可量化但有限的拟合能力；系统级仿真仍与真实队列和 TTFT 有显著差距。** 校准了 Prefill/Decode 批次服务模型、请求实际输出长度分布，以及 executor iteration 到请求 TPOT 的经验 cadence 系数。加入 cadence 后 TPOT 更接近真实值，但 TTFT、吞吐量和队列仍未通过系统级验证。闭环控制器是贪心基线，不是 PPO checkpoint。

## 数据审计

- 数据目录共有 20 个 run：15 个校准（QPS 9/12/15/18/21 × seeds 101/202/303），5 个 seed-404 验证 run。
- 校准服务 CSV 49,058 行，去重后 37,815 个有效 profile 样本；验证 CSV 16,462 行，去重后 12,855 个有效 profile 样本。重复轮询约 22.3%，没有发现 profile 标签冲突。
- `profiling_data` 的毫秒标签是 `executor_iteration_wall_ms`：Ray executor dispatch 到 worker 返回结果的主机端 wall time。它不是纯 GPU 时间，也不是 TTFT。实例状态字段按前一调度 step 对齐，避免把异步快照错配成当前 Batch 状态。
- 迁移终态成本样本为校准 74 条、验证 31 条；包括成功及中止事件。日志没有迁移暂停服务时长的标签。
- 采集配置写的是测量 60 秒、drain 10 秒，但真实 seed-404 响应记录延长到 80.32 秒。系统回放因此使用统一 90 秒响应截止；队列和 KV 仅比较相同的前 60 秒负载窗口。
- trace 的 output token 是生成预算。成功请求的实际输出均值约为预算均值的 9%–22%。仿真从校准 seeds 的 10,347 条成功请求中拟合输出长度分布，seed 404 的输出长度只作验证标签；被 drain 删失的输出长度不可用，经验分布可能有选择偏差。
- 17 个 run 在迁移槽位保护修复前采集，3 个 QPS 21 补充 run 在修复后采集。延迟采集口径相同，迁移行为有代码版本差异。当前数据只覆盖 Migration ON。

## 拟合和留出验证

服务候选选择使用按 run 分组交叉验证；seed 404 不参与候选选择。下表列出全阶段模型比较和正式测量窗口诊断。

| 模型 | 选中候选 | 校准分组 CV MAE / RMSE | CV R² | seed 404 测量窗口 MAE / RMSE |
|:---|:---:|---:|---:|---:|
| Prefill 批次耗时 | P1：批次 token 数 + Prefill Batch 数 | 41.44 / 93.31 ms | 0.038 | 42.2 / 89.4 ms |
| Decode 批次耗时 | D3：batch + 平均 context + 交互项 | 26.50 / 101.83 ms | 0.029 | 30.1 / 71.8 ms |
| 迁移成本 | M2：Huber 全局回归 + 收缩实例对修正 | 342.29 / 698.54 ms | -0.035 | 335.0 / 566.9 ms |

只看 60 秒测量窗口，P1/D3 的校准分组 CV MAE/RMSE 分别为 45.87/102.38 ms、30.43/82.99 ms，R² 为 0.026、0.051。延迟特征只解释了少量变化。Decode 的 batch 和 context 高度相关，不能把低 context 系数解释成 context 没有成本。

迁移 M2 在 seed 404 的 R² 为 -0.025；样本仍不足以可靠估计每一对实例。6 个样本数足够的有向实例对使用收缩修正，其余使用全局模型。残差尺度约 264 ms。迁移阻塞时长仍是无直接标签支持的环境近似。

系统级回放另用校准 run 拟合了 decode cadence 系数：中位数 **5.828**（15 个校准 run，P05/P95 为 4.66/13.19）。这是每个请求发出 token 的经验间隔与预测 executor iteration 的比例，不替代迭代时长标签，也不使用 seed 404。

## seed 404 系统验证

以下使用 V3：P1/D3 服务模型、动态 KV、输出长度分布、cadence 系数和迁移残差不确定性；闭环动作为贪心规则。吞吐量按 90 秒内完成的 HTTP 200 数除以 60 秒到达窗口计算。实际 HTTP 499 作为 drain/客户端取消删失单独统计，不等同于引擎故障；仿真没有客户端取消模型。

| QPS | 到达 | 真实 HTTP 200 / 499 | V3 完成 / 未完成 | 真实 / V3 TTFT 均值 ms | 真实 / V3 TTFT P95 ms | 真实 / V3 TPOT 均值 ms | 真实 / V3 吞吐 req/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 549 | 400 / 149 | 329 / 220 | 6,836 / 403 | 25,032 / 644 | 542 / 583 | 6.67 / 5.48 |
| 12 | 712 | 608 / 104 | 380 / 332 | 12,898 / 464 | 28,786 / 788 | 491 / 688 | 10.13 / 6.33 |
| 15 | 888 | 827 / 61 | 416 / 472 | 9,242 / 1,619 | 26,487 / 9,453 | 604 / 759 | 13.78 / 6.93 |
| 18 | 1,069 | 579 / 490 | 428 / 641 | 6,708 / 3,042 | 19,928 / 17,233 | 559 / 777 | 9.65 / 7.13 |
| 21 | 1,239 | 1,056 / 183 | 444 / 795 | 11,878 / 4,073 | 27,164 / 22,862 | 977 / 791 | 17.60 / 7.40 |

cadence 校准明显缩小了 TPOT 差距，但 V3 吞吐仍比 HTTP 200 观测低约 18%–58%。V3 TTFT 均值在各 QPS 仍低约 2–28 倍，说明请求排队/首 token 等待还有未建模的系统机制。V0–V3 的 V3 完成量与 V2 相近或略低，尚不能说明迁移不确定性带来一致改进。

| QPS | 真实队列均值 / P95 | V3 队列均值 / P95 | 真实 KV 均值 / P95 | V3 KV 均值 / P95 |
|---:|---:|---:|---:|---:|
| 9 | 1.11 / 0 | 0.75 / 1.5 | 0.079 / 0.162 | 0.092 / 0.153 |
| 12 | 2.52 / 20 | 1.15 / 2.2 | 0.137 / 0.301 | 0.126 / 0.212 |
| 15 | 3.00 / 32 | 6.13 / 30.2 | 0.130 / 0.268 | 0.152 / 0.232 |
| 18 | 1.76 / 8 | 18.82 / 65.2 | 0.133 / 0.245 | 0.164 / 0.233 |
| 21 | 12.96 / 92 | 34.0 / 104.3 | 0.159 / 0.278 | 0.174 / 0.233 |

高负载下队列误差仍不稳定：QPS 18/21 仿真队列高于观测，QPS 12 又低估队列 P95。真实 HTTP 499 和 10 秒 nominal drain 对计数/队列解释有删失影响，故吞吐与队列差异不应全部归因于服务曲线。

固定 Migration ON 事件回放与闭环仿真分开报告。固定回放使用日志成本/终态并以 watcher 时间减成本近似事件开始时间，不用于验证成本预测器；部分事件无法映射到有效请求。真实初始路由、调度决策、KV 块生命周期和 mixed-batch overlap 仍未被完整采集。

## 交付物

- 本摘要：`refinement_summary.md`
- 逐字段数据审计及统计：[data_audit.md](../ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505/data_audit.md)、[data_audit.json](../ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505/data_audit.json)
- 候选模型指标：[model_comparison.json](../ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505/model_comparison.json)、[model_error_attribution.md](model_error_attribution.md)
- 原训练配置的服务/迁移拟合：[mig_ppo_calibration_refined.yaml](../ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505/mig_ppo_calibration_refined.yaml)
- 加入输出长度和 cadence 系数的系统验证配置：[mig_ppo_calibration_system_validation.yaml](mig_ppo_calibration_system_validation.yaml)
- 完整 QPS、V0–V3、固定/闭环、服务残差和队列/KV 指标：[system_validation.md](system_validation.md)、[system_validation.json](system_validation.json)
- Prefill/Decode/迁移残差诊断图：校准目录 `refined_plots/`

测试命令 `/venv/main/bin/python -m pytest -q rl_scheduler/tests`：**29 passed**；有一个 pytest 配置警告，当前 pytest 不识别 `asyncio_fixture_loop_scope`。

保留 P1/D3 和 M2 作为可通过配置切换的校准候选及回退旁路；当前独立系统验证不足以宣称仿真器已准确复现真实系统，正式 PPO 策略效果仍需用更完整的队列、请求路由、迁移暂停和输出长度信息验证。
