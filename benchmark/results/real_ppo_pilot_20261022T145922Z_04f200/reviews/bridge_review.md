# 04 Ray 端口修复复审：passed（仅 CPU/配置）

结论：**passed**。本次复核重新绑定了 pilot、Llumnix、smoke spec、05 命令和 P1/P2 offline gates 的当前 SHA256。R1–R7 的针对性回归与原迁移链复核通过。该结论只允许在预算充足时进行冻结 smoke；它不代表真机验收通过。

## 配置修复

先前 telemetry 启动证据明确显示 GCS 端口 16379 与 Ray 默认 worker 端口区间 10002–19999 重叠。当前 `real_pilot.yaml:service.ray_port` 和 `real_llumnix.yaml:SERVER.RAY_CLUSTER_PORT` 都为 **20000**；20000 不在该区间，预检时没有监听者。旧失败证据保留于 `handoff/05.json` 的归档副本；没有修改 trace、请求分布、seed、SLO 或迁移门槛。

当前配置绑定：pilot `{sha(pilot_path)}`；Llumnix `{sha(ll_path)}`；smoke spec `{sha(RUN/'smoke_spec.yaml')}`；05 command list `{sha(RUN/'commands/05_smoke.json')}`。

## R1–R7 复核与测试

| 问题 | 本次确认 |
|---|---|
| R1 | `rl_scheduler/real_runner.py`：每角色先验证，再合并同源水位；跨轮 checkpoint 回退被检测。 |
| R2 | `rl_scheduler/real_env.py; rl_scheduler/real_bridge.py`：未决 decision ID 在新非 Stay 提交前查询；仍未知则拒绝新提交并保留原 action/logp。 |
| R3 | `rl_scheduler/real_bridge.py`：controller acquire 在第一次 await 前建立 fence，返回后复查 native idle/intervention。 |
| R4 | `llumnix/real_control.py; llumnix/entrypoints/vllm/api_server.py; llumnix/entrypoints/vllm/real_metrics_stream.py`：取消 tombstone 覆盖迁移/投递队列；无法证明 absence 时保持 unknown。 |
| R5 | `rl_scheduler/real_runner.py; rl_scheduler/real_env.py`：reset 要求 API、Manager、四实例 local/worker durable、passed 且 pending_delivery=0。 |
| R6 | `benchmark/run_real_ppo.py`：CLI 在预算记账和服务启动前校验 review、offline gate 和前序 live gate；cleanup 后发布 live gate。 |
| R7 | `rl_scheduler/real_runner.py`：固定拓扑每个必需 worker 都需通过 flush，失败 worker 不能被 local fsync 替代。 |


本次重跑 **30 个 targeted CPU tests，全部通过**：独立审查 9 项、R1–R7 修复回归 15 项、runner/cleanup 集成 6 项。原失败 Ray 启动日志、原 04 文件、旧 offline gates 和旧命令清单均按原字节保存在 `repairs/ray_port_collision/prior_20261022T1740/`，其 SHA256 见本次 review JSON。新 P1/P2 gates 对当前源码、配置和证据重新绑定。

## 05 前置预算

05 冻结清单最大预算为 **6060 秒**。当前持续预算账本的 P0–P3 剩余值低于 6060 秒，因此不启动 GPU/Ray 服务，不执行 telemetry/shadow/active/overhead 真机命令。`handoff/04` 与本次 offline gates 只证明配置/CPU 检查通过；05 live 仍未完成。新 review/gates 没有重置 `experiment_clock.json`。

本次只修正与 Ray 端口有关的配置绑定和审查/门控材料。未更改业务源码、权重、trace、seed、阈值或命令业务参数；未启动 GPU 服务，没有真实请求或真实迁移结果。
