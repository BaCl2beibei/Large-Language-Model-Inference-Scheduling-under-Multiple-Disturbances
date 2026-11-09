# P0/P1 快速筛选

状态：execution_failed；推荐 PPO 环境：False。
模型 Qwen2-1.5B，输入 128 / 实际输出 1536 token，128 请求，90% 热点。
PPO-STAY 为固定随机权重的 CPU 网络路径开销检查；没有训练。
P0：{"passed": false, "reason": "missing_or_invalid_P0"}

| 场景 | 种子 | 模式 | 状态 | mean ms | P95 ms | cost ms | 成功搬迁 | 不同请求 | 机会/决策 | 只能 Stay |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q15-A | 141101 | Stay-A | measurement_incomplete | None | None | None | 0 | 0 | 167/169 | 2 |

运行状态及真实时间：`{"status": "execution_failed", "started_wall": 1793786941.7434099, "reason": "RuntimeError: first episode/lifecycle/routing/execution audit failure; stop formal comparisons", "traceback": "Traceback (most recent call last):\n  File \"/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/run_p0p1_fast.py\", line 423, in run\n    for mode in ('Stay-A','PPO-STAY','Stay-B'): await episode('Q15-A',s1,mode,trace('Q15-A',s1))\n  File \"/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/run_p0p1_fast.py\", line 401, in episode\n    if e['status']!='passed' or audit['status']!='passed': raise RuntimeError('first episode/lifecycle/routing/execution audit failure; stop formal comparisons')\nRuntimeError: first episode/lifecycle/routing/execution audit failure; stop formal comparisons\n", "elapsed_sec": 492.0067419349216, "formal_rounds": 1, "ended_wall": 1793787433.7501526, "episode_durations_sec": [238.93464008602314]}`
逐种子收益及逐项门槛见 comparison.json。缺失回合均不判通过。完整原始证据在 runs/、causal_events/ 和 runtime/。
未达标时不进入 PPO；不能由大量迁移或 GPU 均衡推出服务改善。
原因统计可重叠。Balanced 过滤计数同时含阈值、配对排序和迁后检查，不能全部归因于阈值。
Q15-B 复用 P0 仅属不同请求率的有限外推；控制及发包开销另存每轮 opportunity.json。
下一项建议由失败证据决定；未完成两种子比较前不调整当前冻结配置。
