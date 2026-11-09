# HOT1-H / NATIVE 实验补测报告

Run `mig_ppo_v2_hot1h_repeat_20261106T083209Z`；结果状态：`complete`。
采用规则：同一 trace 的 Stay/Balanced 完整配对，Balanced 的 C=mean(E2E)+CVaR95(E2E) 更低，且至少一笔成功迁移通过审计。
未采集独立真实验证数据。HOT1-H 使用固定热点路由至逻辑实例 1。

| 场景 | QPS | 状态 | Stay C (ms) | Balanced C (ms) | gain_C | 成功迁移数 | 通过 |
|---|---:|---|---:|---:|---:|---:|---|
| HOT1-H | 1.000 | complete | 158940.236 | 218411.135 | -0.374 | 26 | False |

逐请求路由、迁移回执、执行审计、trace hash 与每轮 200 秒窗口详见 `comparisons/` 和 `runs/`。
