# 给接手模型的 P0/P1 交接指令

你接手的是已经自动停止的真实四卡快速筛选实验。先读本文件和原任务书，再查看原始证据。不要只根据对话中的早期进度判断状态。

## 当前状态

- 主批目录：`benchmark/results/p0p1_fast_20261104T104200Z/`。
- 状态：`incomplete_budget`，`recommended=false`。本批已结束，当前没有实验进程；不是仍在等待最后一个 Stay。
- 本批用户明确授权重新启动；本批独立预算从北京时间 2026-11-04 18:42:31 起算，19:20:53 清理结束，用时 38.37 分钟。不得现在接着启动缺失回合并沿用旧批通过标签。
- 正式回合共 7 轮，全部 128/128 成功且生命周期、首次 engine admission、输出长度、实际迁移审计通过。独立预热也通过。
- 缺失：Q15-B / seed 141104 的 Stay。启动前已剩不足 8 分钟，脚本按原规则未启动。没有把慢请求截掉，没有 PPO 训练。

## 配置和已知结论

Qwen2-1.5B，本地 model/Qwen/Qwen2-1.5B；四张 RTX 3060，各 TP=1；half、eager；max_model_len=2112，max_num_seqs=128，gpu_memory_utilization=.65，max_num_batched_tokens=2112；rayrpc / SR / rayqueue，Ray CPU=16。

每轮 128 请求，精确输入 128 / 输出 1536 token，temperature=0，ignore_eos=true，n=best_of=1。固定 115/128 到热点，约 90%；Q15-A=1.0 req/s，Q15-B=.6 req/s。四实例实际 total_blocks 都为 8565；阈值冻结为 8565/8=1070.625。5 秒冷却、2 秒新鲜度、全局一个迁移不变。

种子 141103、141104 在本批计划前按结构化 seed 字段检查，未用过；trace 在 traces/ 冻结，同一 seed/场景严格复用哈希。旧种子 141101、141102 属于失败/中断尝试，绝不能混为本批独立确认种子。

P0 通过（工程门槛，不是统计等价）：Stay cost 漂移 3.402%，mean 漂移 3.486%；PPO-STAY 相对两次 Stay 平均 cost 偏差 3.999%，mean 偏差 4.696%；P95 没有恶化。PPO-STAY 是固定随机权重的当前 CPU 网络路径开销检查，并非历史策略复现。全部零真实迁移；torch 线程数为 1，measurement_only=true，不保存伪造行为 log-prob，不可成为 rollout。

| 场景/种子 | cost 改善 | mean 改善 | 成功/started | 不同请求 | 覆盖率 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Q15-A / 141103 | -16.09% | -15.98% | 38/38 | 38 | 29.69% | 服务变慢，失败，未进入其第二种子 |
| Q15-B / 141103 | +15.09% | +17.82% | 44/44 | 44 | 34.38% | 单种子完整配对通过 |
| Q15-B / 141104 | N/A | N/A | 46/46，另 rejected=1 | 46 | 35.94% | 缺 Stay，不能判断收益 |

Q15-B / 141103 的 P95 E2E、P95 TTFT、P95 TPOT 分别改善 12.67%、10.66%、12.67%。它相对另一个请求率只有限复用 P0；必须查看发包延迟和控制开销，不能宣称 Q15-B 已完成独立 P0。

机会/只能 Stay（全部请求结束前，按当时原始 mask）：

- Q15-A Stay-A：197/198 有机会，1 次只能 Stay；Stay-B 同为 197/198、只能 Stay 1。
- Q15-A Balanced：53/204 有机会，只能 Stay 151，自愿 Stay 15，迁移意图 38。
- Q15-B Stay：284/286 有机会，只能 Stay 2。
- Q15-B seed1 Balanced：80/255 有机会，只能 Stay 175，自愿 Stay 36，迁移意图 44。
- Q15-B seed2 Balanced：73/258 有机会，只能 Stay 185，自愿 Stay 26，迁移意图 47。
- 三个 Balanced 成功频率分别为 8.90、8.36、8.66 次/分钟；没有重复/反向成功迁移。

## 文件索引

1. 原始授权任务书：`rl_scheduler/docs/p0p1_fast_track_prompt.md`，含完整门槛、范围、停止条件。
2. 执行器：`benchmark/run_p0p1_fast.py`（plan/run/analyze），复用 `benchmark/run_full_completion_balance.py` 和 `rl_scheduler/full_completion.py`。
3. 关键代码：full_completion_service.py、real_bridge.py、real_observation.py、real_opportunity.py、cohort_ppo.py；精确执行版本在主批 source_snapshot/，完整源哈希在 plan.json。
4. 最小验证：`rl_scheduler/tests/test_p0p1_fast.py`；主批 cpu_validation.log：42 passed，5 subtests passed。
5. 计划和配置：主批 plan.json、runtime_config.yaml、real_llumnix.yaml、traces/。
6. 原始证据：主批 runs/*/{requests,decisions,migrations,events}.jsonl、episode.json、execution_audit.json、opportunity.json；causal_events/ 是服务原始流，runtime/ray/ 是 Ray 日志。
7. 结论：results.csv、comparison.json、training_environment_handoff.json、report.md；准确信息以原始账本和逐项门槛为准，不使用旧数据凑两个种子。
8. 运行日志：runner.log、api_server.log；环境、实际参数、worker PID→GPU UUID 在 environment_audit.json；正常清理见 service_stopped.json（cleanup_error=null）。
9. 总控托管：controller_supervisor.json 和 services/ 下 shell/conf；autorestart=false，当前已退出，不会自动重启。

## 必须知道的实现与分析注意事项

- 首次尝试 p0p1_fast_20261104T101500Z 因把五秒在线窗口增量当累计产出而误中止；已修为累计 token_high，并有窗口淘汰回归测试。
- 唯一修复重试 p0p1_fast_20261104T101818Z 因用户意外中断聊天，未托管的采样进程退出；该回合只有 29/128 客户端完整成功，明确 incomplete，不能比较性能。
- 本批总控已经交给 supervisor，与聊天进程分开。所有失败、中断原始结果均在包中保留。
- 计划阶段曾用文本子串查 seed，误把延迟数值当种子；已改为结构化 seed/seeds 字段并有回归测试。p0p1_fast_20261104T104130Z 仅为计划失败/未启动 GPU 的诊断目录，不能算正式实验。
- Balanced 可选阈值确实传到执行分支；旧调用默认 9000 不变。每个决策记录的 balanced_threshold 可核对。
- **无机会原因时间戳缺陷**：opportunity_stats 使用决策循环起始 monotonic_ns（早于 snapshot）调用 eligible_pair，可能把“快照在该时间点尚未采集”错误记为 stale_state。不要把现有 stale_state 原因计数当真实瓶颈。原始 action_mask、N_opp/N_forced、迁移与请求审计不受该事后原因标注影响。原始日志未在全部策略单独保存 raw.decision_ns，不能伪造精确时间修复历史原因；后续执行版本应记录实际 builder 决策时间，再计算原因。busy/cooldown 可从原始快照描述，但避免声称原因穷尽、互斥或精确归因。
- balanced_filter_stay_despite_opportunity 同时包含阈值、源/目标排序和迁后检查，不能全部称为阈值过滤。
- 神经网络耗时、特征转换、control_work、完整间隔必须分开。PPO-STAY forward P95≈5.04 ms；实际控制间隔 P95≈1.29 s。不要把全部周期归为神经网络。
- Q15-B Stay / Balanced 的 control_work P95 为 280.46/301.69 ms，发包 P95 为 3.09/3.65 ms；本批没有显示相对 Q15-A 明显增大的开销，但这不是独立请求率 P0。

## 接手后先做什么

先读结果，不启动服务：

```bash
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
cat benchmark/results/p0p1_fast_20261104T104200Z/run_state.json
cat benchmark/results/p0p1_fast_20261104T104200Z/comparison.json
cat benchmark/results/p0p1_fast_20261104T104200Z/results.csv
.venv/bin/python benchmark/results/p0p1_fast_20261104T104200Z/services/monitor_once.py
```

`analyze` 可从当前原始账本重建派生报告，不启动 GPU：

```bash
.venv/bin/python benchmark/run_p0p1_fast.py analyze --run-dir benchmark/results/p0p1_fast_20261104T104200Z
```

本批结论必须保持 incomplete_budget / recommended=false。没有第二个完整配对，不能写 fast_candidate_passed，不能开始 PPO，不能继续启动缺失 Stay 当作同一预算内实验。

下一次最值得测试的是 Q15-B 两个完整新种子配对，保持模型、负载、阈值规则、冷却等不变，并按任务书处理 P0 的有限外推；这是建议，**当前交接不自动授权新一批 GPU 实验或重置预算**。用户明确授权下一批后再创建新目录、新计划和未用过的种子，先修复实际决策时间戳记录，使用 supervisor 托管总控。不要扩大候选矩阵，不要回退 0.5B，不要 PPO 训练。

所有当前原始日志在本机原路径保留。归档包不包含模型权重或 Python 虚拟环境；新机器需同版本依赖和本地 Qwen2-1.5B 权重。包内保留 runtime_config 的原绝对路径，迁移机器时必须新建计划并调整路径，不能伪称原配置复现。
