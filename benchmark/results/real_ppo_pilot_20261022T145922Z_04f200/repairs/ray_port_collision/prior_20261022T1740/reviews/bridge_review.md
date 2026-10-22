# 04 修复复审：passed

结论：**passed，原 R1–R7 已关闭；本轮未发现新的阻断问题。** 仅允许下一步骤按冻结协议执行 05 smoke；不表示真机验收通过，不自动启动 GPU、P3 或 PPO。

本轮依据 04 prompt 进行独立源码审查，核验 active_run、历史 02/03 passed、repair_01 交接和两个 repair_01 重验 gate。三个当前材料共 243 个源码/配置/证据绑定均匹配（85 + 79 + 79）。旧 failed 审查和交接逐字节归档至 `reviews/rereview_01/prior_04/`，原反例代码、失败日志和迁移链证据保留。

## R1–R7 复核

| 问题 | 当前源码 | 结论与实现依据 |
|---|---|---|
| R1 | `rl_scheduler/real_runner.py:24`<br>`rl_scheduler/real_runner.py:55` | 已关闭。每个角色先验证，随后按 source 合并同轮可比较水位；跨轮 checkpoint 回退检测保留。 |
| R2 | `rl_scheduler/real_env.py:121`<br>`rl_scheduler/real_bridge.py:335`<br>`rl_scheduler/real_bridge.py:356` | 已关闭。Env 与 bridge 跟踪未决 ID；新非 Stay 提交前查询原 ID，仍未知则拒绝新提交；保留原 action/logp。 |
| R3 | `rl_scheduler/real_bridge.py:96`<br>`rl_scheduler/real_bridge.py:130` | 已关闭。第一次 await 前设置 acquiring fence；返回后复查 native idle 和 intervention，异常回收 acquiring。 |
| R4 | `llumnix/real_control.py:14`<br>`llumnix/real_control.py:235`<br>`llumnix/entrypoints/vllm/api_server.py:69`<br>`llumnix/entrypoints/vllm/real_metrics_stream.py:82` | 已关闭。Manager/四实例设置取消 tombstone，核查 dispatch、迁移、运行/等待/暂停/预分配队列、epoch 与投递；API 清空队列后关闭流；未证明的取消保持 unknown。 |
| R5 | `rl_scheduler/real_runner.py:55`<br>`rl_scheduler/real_runner.py:99`<br>`rl_scheduler/real_env.py:72` | 已关闭。reset 显式要求 API/Manager/四实例 local/worker 的 durable、passed、pending_delivery=0；prepare 有界。 |
| R6 | `benchmark/run_real_ppo.py:66`<br>`benchmark/run_real_ppo.py:99`<br>`benchmark/run_real_ppo.py:147`<br>`benchmark/run_real_ppo.py:219` | 已关闭。live plan 在预算写入和服务启动前验证当前独立审查及冻结顺序的前序 live gate；清理本任务服务之后才发布结果 gate。 |
| R7 | `rl_scheduler/real_runner.py:24`<br>`rl_scheduler/real_runner.py:55` | 已关闭。固定拓扑的必需角色先校验，失败 worker 即使缺少 source/durable 也被阻断，不能被 local fsync 替代。 |

## 独立链路与验证

本轮实际运行 **30 个针对性 CPU 测试，全部通过**：原独立审查 9 项、修复回归 15 项、runner 集成及清理超时 6 项。修复报告中的 78 项全量测试只作为经 hash 核验的前置证据，没有冒称本轮又运行 78 项。

- 请求生命周期：客户端 send/receive 时钟用于 TTFT/TPOT 和评分；内部 engine/API/queue 时间仅用于归因。测试覆盖截止前/后的完成与 cleanup 对账，迟到完成不回填历史成功。完整输出流与 sampled causal 事件仍分离；缺失终态/producer 不得通过。
- 迁移链：陈旧候选在原仲裁入口被拒绝；有效候选沿 `ControlArbiter -> controlled_migrate -> MigrationCoordinator._migrate_out_one_request -> destination commit -> source free/unpause` 返回同一 decision/migration ID 的 succeeded。实际执行原 coordinator 方法体，RPC、KV 与引擎为 CPU mock，证据为本轮 `migration_chain.json`。
- 取消链：补审 tombstone 对晚到 generate/preallocate/receive/commit 的拦截，四实例 absence proof 的 epoch/角色稳定性，engine step/output barrier，API 队列和流结束顺序。未决 dispatch/迁移/投递/RPC 不被记作取消成功。已运行相关修复测试；并非实际 GPU 迁移取消验收。
- 观测与动作：采样时 observation/mask/action/logp 不回写，normalizer 在回合内冻结，UUID/epoch mapping 变化使回合失效。接管期间普通控制器受 acquiring fence 限制；故障介入使回合无效。
- 清理与资源：finish 共用 60 秒截止，未知终态、清理超时或未确认释放使回合 invalid；共享预算不因进程切换重置。CLI 的 finally 保留本任务服务清理，所有权限制未被移除；watchdog CPU 测试通过。
- 四条冻结命令的实际 `--dry-run` 均退出 0，写入本次审查前全部明确被旧 failed review 阻断，预算未写入。新的 passed 文件落盘后将另存一次只读门控核验至交接证据，不执行 live 命令。

证据：`reviews/rereview_01/original_9.log`、`repair_regressions.log`、`integration.log`、`migration_chain.json`、`commands.json`、`prerequisites.json`、`preservation.json`。源码与 smoke 配置逐文件及聚合 SHA256 见 `bridge_review.json`。

## 范围、预算与交接

仅写 reviews/handoff 和审查脚本；业务源码、配置、前序 gate、active pointer、实验时钟和既有 Git 状态均保留。没有加载模型、启动 GPU 服务、修改测试 seed/阈值、派生代理、提交或推送。本轮 GPU 实验时间为 0；实验时钟尚未启动。剩余总预算 28800 秒，P0–P3 7200 秒、P4 7200 秒、P5 14400 秒。

后续可执行范围仅为 05：依次 telemetry → shadow → active → overhead；后序阶段仍依赖真实前序 live gate。本步停止，实际性能、成功迁移次数、overhead 和 GPU 清理效果仍须 05 验证。active pointer 内的 failed_pending_rerun 是修复交接时的旧描述，本步依照只写 reviews/handoff 的范围保留；本次结论见 `handoff/04.json`，CLI 直接校验本次 `reviews/bridge_review.json`。
