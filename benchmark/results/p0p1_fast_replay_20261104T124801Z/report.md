# Q15-B 同 trace 配对重放

本目录是独立 attempt；旧批次保持不变。Stay 与 Balanced 使用同一冻结 trace，来源为旧批次 Q15-B / seed 141104。
这是一组重放配对，不是新 seed，也不单独构成 PPO 推荐。P0 仅有限复用 Q15-A / 1.0 req/s 的检查。

| 模式 | 状态 | mean ms | P95 ms | cost ms | 成功迁移 | 不同请求 | 机会/决策 | 只能 Stay | 控制耗时 P95 ms | 发包延迟 P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced | passed | 121063.9003317422 | 143819.60028745 | 265037.90136383596 | 43 | 43 | 80/253 | 173 | 311.85079959999956 | 3.928534449999998 |
| Stay | passed | 150425.16094588296 | 168543.519249 | 319111.2216762267 | 0 | 0 | 283/287 | 4 | 260.53641159999995 | 2.7047943499999993 |

配对门槛：`{"passed": true, "checks": {"positive_cost": true, "positive_mean": true, "cost_lower_than_each_stay": true, "p95": true, "ttft": true, "tpot": true, "succeeded": true, "coverage": true, "frequency": true, "success_rate": true, "stay_opportunities": true}, "relative_improvement": {"cost": 0.16944976120975794, "mean": 0.1951884939295746, "p95": 0.1466916026894145, "cvar95": 0.14649734300069953, "ttft": 0.06647759535597486, "tpot": 0.14670445511881428}, "versus_each_stay": [{"cost": 0.16944976120975794, "mean": 0.1951884939295746, "p95": 0.1466916026894145, "cvar95": 0.14649734300069953, "ttft": 0.06647759535597486, "tpot": 0.14670445511881428}]}`

没有启动 PPO 训练。完整原始账本在 runs/；阈值、配置、GPU 映射与清理审计见 plan.json 和 environment_audit.json。

最终清理核查通过：所属进程、Ray 进程及 GPU 计算进程残留均为 0，服务 `cleanup_error=null`。核查详情见 `cleanup_verification.json`；实例 worker PID 到 GPU UUID 映射见 `worker_pid_gpu_uuid_mapping.json`。本 attempt 用时约 15.97 分钟。
