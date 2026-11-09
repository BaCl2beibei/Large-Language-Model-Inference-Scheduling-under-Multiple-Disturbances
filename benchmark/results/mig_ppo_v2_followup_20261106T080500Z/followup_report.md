# HOT1-H / NATIVE 实验补测报告

Run `mig_ppo_v2_followup_20261106T080500Z`；结果状态：`complete`。
采用规则：同一 trace 的 Stay/Balanced 完整配对，Balanced 的 C=mean(E2E)+CVaR95(E2E) 更低，且至少一笔成功迁移通过审计。
未采集独立真实验证数据。HOT1-H 的 trace 与 HOT2-H/HOT3-H/HOT4-H 的 QPS1 候选共享基础请求轨迹，仅热点位置改为实例 1。

| 场景 | QPS | 状态 | Stay C (ms) | Balanced C (ms) | gain_C | 成功迁移数 | 通过 |
|---|---:|---|---:|---:|---:|---:|---|
| HOT1-H | 1.000 | complete | 160901.320 | 225393.085 | -0.401 | 26 | False |
| NATIVE | 1.250 | complete | 188367.612 | 143788.141 | 0.237 | 1 | True |

逐请求路由、迁移回执、执行审计、trace hash 与每轮 200 秒窗口详见 `comparisons/` 和 `runs/`。
