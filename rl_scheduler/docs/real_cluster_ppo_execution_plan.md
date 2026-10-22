# Mig-PPO 真实集群训练执行任务书

本文可直接交给执行模型。按阶段执行、按证据验收。本文是实施规格，文中的新文件、命令和接口尚未实现，不得宣称已可用。

## 0. 任务、范围与执行纪律

项目目录：`/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream`。

目标：在现有四卡 Llumnix/vLLM v0 集群中实现可审计的 on-policy PPO 迁移训练，取消高精度仿真拟合作为训练前置条件。先证明真实交互闭环可靠，再进行有界训练和独立评估。成功运行 PPO 不等于证明 PPO 优于基线。

这是调度器训练，不训练或微调 Qwen 参数。第一版仅支持当前单机四卡、四个独立推理实例、非 PD 模式；多节点泛化留待后续。不要顺带更换推理模型、vLLM 版本、CUDA、驱动或请求调度算法。

收到用户明确要求执行本文后，可自动逐阶段推进；不需要每阶段重复请求确认。以下运行上限是建议的默认执行边界，用户另有预算时以用户为准。仅阅读本文不构成启动实验的指令。

- 阅读 `/workspace/AGENTS.md`。服务按实例规范由 supervisor 管理；内部控制接口绑定 localhost，不新增公开入口。
- 保留已有暂停目标，不擅自恢复旧的仿真拟合任务；不调用目标管理工具创建新目标，除非用户明确要求。
- 保留所有已有改动、未跟踪文件和删除状态。禁止 `git reset --hard`、`git clean`、批量恢复删除文件和 `git add .`。
- 本任务只使用现有实例；不得租新机器、升级配置、停止或销毁 Vast 实例。结束时可清理本任务创建的推理/训练进程，不停止管理服务。
- 默认不提交、不推送、不发布；交付准确 diff 和复现材料。用户另有 Git 指令则遵循。
- 不启动默认 2,000 回合任务。不要使用 seed 1213、1319 作为最终独立测试。
- 不以降低验收阈值、删掉失败请求、把截尾当成功或修改测试集来通过验收。
- 不安排子代理，除非用户另行要求。

默认预算：从首次启动本任务 GPU 服务起累计实验墙钟时间最多 8 小时，其中 PPO 训练阶段最多 2 小时；包括启动、预热、排空、失败尝试和恢复。四卡总预算上限相当于约 32 GPU 小时。CPU 编码时间单独记录。使用持久化计时器，恢复执行不得重置预算。到限后保存证据并停止实验；预算不足以完成评估时明确标记未完成，不削减测试后宣称通过。

建议分配：P0–P3 的 GPU 检查及机会实验不超过 2 小时，P4 训练不超过 2 小时，P5 评估预留 4 小时。仅是上限，不保证这些时长内可以收敛。

## 1. 必须先读的已有材料

源码：

- `rl_scheduler/train_ppo.py`、`policy.py`、`observation.py`、`normalization.py`、`action_mask.py`、`reward.py`、`env.py`。
- `rl_scheduler/docs/architecture_audit.md`。
- `llumnix/manager.py` 的 `_poll_instance_info_loop()`、`_migrate()`。
- `llumnix/global_scheduler/migration_scheduler.py`、`migration_policy.py`、`migration_filter.py`。
- `llumnix/llumlet/local_migration_scheduler.py`、`migration_coordinator.py`。
- `llumnix/metrics/causal_events.py`。
- `llumnix/backends/output_forwarder.py`、`llumnix/queue/zmq_server.py`。
- `benchmark/run_baseline_4gpu.py` 的 `execute_trace()` 和 abort barrier。
- `benchmark/run_ppo_batch_reuse.py`，尤其是 worker startup gate、服务复用和 trace manifest。

报告根目录：`benchmark/results/llumnix_causal_hifi_20261022_r1/`。阅读：

- `root_cause_analysis.md`、`model_selection.md`。
- `validation_seed1213/system_validation.md`。
- `queue_stage_seed1319_comparison.md`。

已知事实必须保留在设计中：

1. 现有训练器直接创建模拟环境，尚无真机 PPO 闭环。
2. 现有观测包含拟合迁移时间和启发式净收益，不能直接把默认值当作真实信息。
3. seed 1319 的 OFF 807 个服务端完成事件全部落在声明的 90 秒测量加排空窗口之后。
4. seed 1319 的 send→enqueue 延迟 P95 约 5–6.5 秒；这是跨阶段间隔，不能简称为输出队列驻留时间。
5. 现有输出阶段诊断存在采样事件；不能把抽样记录间隔当成每个 token 的延迟。
6. 当前奖励有裁剪饱和，且仿真中未完成请求主要在回合结束记账。
7. 真实系统不会在优化器更新时暂停。

## 2. 总体实现选择：先做有限回合的同步 PPO

固定一次行为策略及归一化参数，采集完整的一个或多个真实回合；所有回合结束并完成事件对账后，计算奖励与 GAE，再更新 PPO。更新时不发送测量请求，确认集群为空，模型权重保持驻留。

第一版不实现持续流量下异步 learner、不实现跨策略版本混合 rollout、不实现离线 PPO。这样先解决可审计性，避免真实时间在更新期间偷偷流逝但轨迹漏记。

每个回合：

1. `reset`：确认四实例健康、无本任务未清请求、无在途迁移、投递待处理已清理；KV 回到已记录的空闲基线，不能要求固定预分配缓存为零。
2. 预热：只在服务首次启动或确有必要时执行。预热请求单独命名并完全清理，不进入训练奖励。之后回合复用驻留模型。
3. 到达阶段：默认 120 秒，按预生成 trace 的绝对时间发送请求；每 1 秒做一次动作决策。
4. 排空阶段：停止新到达，PPO 继续控制并持续计奖，最长 120 秒。所有请求和迁移结束后允许提前终止。
5. 到达硬截止仍未完成：先冻结评分边界及未完成集合，再进行取消清理。清理期间出现的完成不能回填成截止前成功。
6. 奖励及动作对账通过后，回合才可以加入 PPO buffer。

取消清理必须有独立的 60 秒期限并计入预算，不能照搬旧压测可等待 900 秒的 abort barrier。清理失败时停止后续回合；必要时仅重启本任务拥有的推理服务并重新验证空闲状态，旧回合保留为 invalid。接近总预算时预留至少 60 秒清理时间，不继续启动新回合。未能确认取消成功的请求不得计为已取消。

有限回合目标包含截止时未完成惩罚，并在观测中加入当前阶段、剩余到达时间及剩余总时间。完整排空或预定义截止是本任务的终态，末端价值为零。不得把任意采样 buffer 边界当终态。进程故障、人工中断、预算紧急中断的回合标记 invalid，不默认当作正常零价值终态训练；若以后支持非终态截断，必须另外实现正确 bootstrap 并测试。

相邻真实决策记录实际 `dt_sec`。不得因一次回调延迟补发多个虚构的 1 秒 transition。折扣使用 `gamma_t = gamma_per_second ** dt_sec`；GAE 衰减用 `lambda_per_second ** dt_sec`，并正确处理终态。

## 3. 建议新增文件与最小接口

尽量独立新增文件，避免破坏现有模拟器流程：

| 文件 | 职责 |
|---|---|
| `rl_scheduler/real_env.py` | 真实回合状态机、reset/step/finish、真实时钟与清理 |
| `rl_scheduler/real_bridge.py` | Llumnix 只读快照、控制权、动作提交及回执 |
| `rl_scheduler/real_observation.py` | 纯实测特征、最近 5 秒统计、稳定实例映射 |
| `rl_scheduler/real_metrics.py` | 请求台账、时间窗口、事件增量、终态对账 |
| `rl_scheduler/real_reward.py` | 新奖励与逐分量审计 |
| `rl_scheduler/real_rollout.py` | 冻结策略版本、完整回合 rollout、变时长 GAE |
| `rl_scheduler/train_real_ppo.py` | CPU 上 PPO 更新、预算控制和 checkpoint |
| `rl_scheduler/evaluate_real.py` | 同一真实环境中的固定策略评估 |
| `rl_scheduler/configs/real_pilot.yaml` | 独立真实实验配置，不继承仿真默认参数 |
| `benchmark/run_real_ppo.py` | 阶段编排、工作负载、服务管理与门控 |
| `rl_scheduler/tests/test_real_*.py` | 实质性不变量测试 |

允许按实际依赖调整文件划分，但必须在 implementation.md 给出对应关系，不实现第二套 KV 传输机制。

建议桥接接口如下，属于待实现规格：

```python
snapshot() -> Snapshot
acquire_controller(run_id, controller_version) -> Lease
submit_migration(decision_id, src_id, dst_id,
                 expected_candidate_id, snapshot_version) -> ActionReceipt
poll_events(cursor) -> EventsWithCursor
release_controller(lease) -> None
```

reset/step 应保持 Gym 风格语义，但真实回合奖励可在结束对账后重建；训练器只能使用 finalised rollout。step 的暂定奖励不能与最终重建奖励重复相加。

## 4. P0：基线快照与实验预注册

操作：

- 检查 Git 状态、现有 Python 环境、GPU 与 Ray 进程；只输出必要硬件信息，不输出凭证。
- 记录当前 HEAD、tracked diff、将要修改的相关未跟踪源码副本及 SHA-256。不要复制整个模型目录或数据集。
- 新建独立结果目录 `benchmark/results/real_ppo_pilot_<unique_run_id>/`，禁止覆盖旧结果。
- 保存 `preflight.json`、`source_manifest_before.json`、`implementation.md`、`budget.json`。
- 冻结初始工作负载、SLO、时钟/窗口定义、seed 分组和实验顺序规则到 `experiment_spec.yaml`。

seed 建议：机会实验 2101/2102/2103；训练 3100–3199；开发评估 4101/4102/4103；最终评估 5101/5102/5103。先搜索已有结果是否使用这些种子；若已用，选择新的不重叠集合并记录，之后不得随结果更换。

验收：能说明现有推理配置与服务所属；源代码快照完整；预算和 seed 分组已经落盘。此阶段不加载模型。

## 5. P1：建立真实请求台账与一致的评分边界

必须区分：planned/sent/accepted、engine_first、api_first、client_first、server_terminal、client_terminal、cancel_requested、cancel_ack、score_deadline。

客户端到达、首输出、最后输出和终态用于用户可见主指标；API/引擎时间线作为内部诊断，不能冒充客户端时间。若现有客户端不能正确流式解析，先修复采集并通过测试，不能用服务端时间代替后继续训练。

要求：

- 所有请求带 `run_id/episode_id/request_id`，保存客户端与服务端 ID 的映射；重试不能变成新成功请求。
- 同机使用既有 monotonic 时钟机制；记录事件发生时间和采集到达时间。观测只能用决策时已收到的数据，禁止事后把迟到事件回填进过去观测。
- 奖励允许在回合结束后按真实事件发生时间重建，但不修改保存的 obs/action/logp。
- 首输出和终态必须完整采集；吞吐用实际累计 token 数的正增量去重，不把流式 chunk 当成一个 token，不把全文累计 token 重复累加。
- TPOT 定义为每个已完成请求 `(client_last-client_first)/(actual_output_tokens-1)`，输出少于 2 token 记不适用，不记零。
- chunk 间隔及最长无输出时间单独报告；没有逐 token 时间戳时不能报告精确 token 级 ITL/P95 ITL。
- 到达阶段窗口吞吐、全固定截止窗口吞吐、最终清理完成数分别报告。迟到 HTTP 200 只进入清理附表，不回填窗口吞吐。
- 人工 drain 取消、真实超时、服务失败、截止未完成分别计数。一次请求只计一个最终业务结果。
- 所有请求必须对应终态或在评分截止时被明确标为 outstanding/censored。清理后还必须有清理结果；unknown 不等于 censored。
- 输出采样事件只做诊断，不能用每 64 token 的诊断采样推算精确停顿。
- 迁移总耗时、源端暂停、目标提交、迁移后首次输出分别保留实测诊断。某项缺少可配对时间戳就标记缺失，不由总耗时推算。最终奖励首版不需要拟合这些量。
- 增加可确认事件刷盘进度的机制；不能仅 sleep 一段时间便宣称事件已齐全。

必须测试：边界前后完成、迟到记录、重复终态、累计输出计数回退/重复、单 token 请求、HTTP 499 人工取消、未知终态、仅空队列但仍有投递、事件序列缺口。

验收：构造事件与手算主指标完全一致；100% 请求归属明确；清理开始后完成不会改变冻结后的分数；在线可见观测与离线奖励对账严格分开。

## 6. P2：真实动作执行、控制权与影子验证

动作：0=Stay，1–12=有向实例对。回合内固定 instance UUID 到索引的映射；实例重建使映射失效时终止该回合，不能静默换位。

初版普通迁移采用 SR，最多一个全局在途迁移；从迁移终态开始冷却 5 秒。必须在动作接受时原子占用控制槽，不能依赖下一次轮询才发现 busy。目标 KV 保留量复用现有 mask 的 5%/16 blocks 规则，并由执行端做实际容量复核。

PPO 接管时禁用另一套普通负载均衡决策；保留 failover/pre-stop。如果这些路径介入，记录原因，该回合不作为初版正常训练数据。不要把真实故障掩盖成 PPO 成功或失败。

候选语义采用“决策时的 SR 候选请求”：执行端复核 `expected_candidate_id`、状态、容量和版本，不一致则明确拒绝，不替换成另一个请求。必要时沿现有 migrate_out 调用增加可选参数，默认值保持旧调用兼容。

每个动作必须记录：

```text
run_id, episode_id, policy_version, decision_id, monotonic_time,
snapshot_version, instance_index_map, obs_received_at, feature_schema,
raw_obs, normalized_obs, normalizer_version, action_mask,
sampled_action, old_logp, old_value, src_id, dst_id, candidate_id,
receipt_status, rejection_reason, migration_id, terminal_status,
actual_request_id, copied_blocks, migration_start/end, dt_sec
```

记录 sampled action 与最终执行结果；被拒绝的迁移动作仍按原采样动作和 logp 入轨迹，环境反馈为拒绝，不能改写成 Stay。使用执行时复核消除危险状态，但不要用事后 mask 替换采样时 mask。

控制器退出时按租约释放控制权；遗留迁移清理完毕再切换策略。接口超时没有回执时不能立即重发新动作；先按 decision_id 查询并保证幂等。迁移未确认结束时保持槽位，不因超时擅自解锁。

观测：复用真实队列/KV/候选长度、实例吞吐、首输出未达标请求数、无输出停顿数、状态年龄及近 5 秒趋势；加入 episode 阶段/剩余时间。缺失值显式 mask。不得包含实际未来输出长度、未来 trace、未来完成时间。删除预测迁移时间/预测净收益，或改为显式禁用且不输入网络；不能用零值伪装已知成本。

不要直接复用按固定列索引实现的原 normalizer 处理新特征。实现有版本的真实特征归一化，训练和评估严格共享冻结的特征顺序及参数。

先完成无 GPU 的 mock bridge 测试；再在真机上 shadow（只建议动作）检查状态和延迟。随后用低负载、倾斜请求场景做受约束的主动迁移验证。

验收：

- 影子模式零实际迁移；索引一致；缺失/陈旧状态不能产生可执行迁移。
- 主动 smoke 至少 10 次有明确终态的尝试、其中至少 5 次成功，用于证明接口可用，不能据此估计可靠恢复分布。
- 全部动作有回执或明确可解释的拒绝；全部接受动作可关联 migration_id 和终态；无重复执行、无未知在途任务。
- CPU 决策路径 P95 小于 100 ms；如未达到，先检查读取/序列化/线程争用，不擅自改大允许值。
- 比较相同 trace 下桥接监控开启/关闭的开销，至少做一组重复配对；吞吐退化超过 5% 或 TTFT P95 退化超过 10% 时先定位。此项是工程检查，少量重复不能宣称严格统计等价。

若普通场景没有合法候选，不直接放松容量或状态检查；先构造有解释的倾斜负载。P2 不通过，不进入 PPO。

## 7. P3：确认迁移有收益空间，冻结奖励

先用机会实验 seed 对 Stay 与受同样并发/冷却限制的 Balanced 做配对实验。两个场景、每场景 3 个 seed、每个策略一遍，共 12 个回合；交替/随机化策略顺序，并保留完全相同外生 trace。

场景选取规则：从 QPS 3、6、9 的短探测开始，找出可稳定完成请求的档位和有持续队列但不全面失效的档位；不自动跳到 QPS 18。探测也计预算。若三档均不健康，先调查服务链路，不训练。

- 场景 A：正常 dispatcher、稳定中等负载。
- 场景 B：同一模型与长度分布，对一个固定实例施加预注册的到达倾斜。可用 70%/10%/10%/10% 路由作为初始设置；必须验证实际初始路由，所有策略共享同一路由规则。

修改初始路由仅作为显式实验扰动，不让 PPO 控制路由；不得把路由改善误算成迁移收益。请求输出预算不等于实际长度，保持相同采样参数，不要求旧链路强制精确输出长度。

指标：固定窗口完成率、SLO goodput、客户端 token 吞吐、条件 TTFT/TPOT、首 token 截尾比例、截止未完成数、请求驻留时间积分、迁移尝试及终态、输出链路延迟。

主指标定义：SLO goodput = 在固定总截止内正常完成且 TTFT≤2,000 ms、请求级 TPOT≤150 ms 的请求数 / 固定总窗口秒数。单 token 请求只考核 TTFT，单独报告数量。所有 offered 请求作为完成率分母。若大部分场景两策略 goodput 都为零，不进入 PPO；先说明工作负载/SLO 是否合理，在训练前形成有记录的新协议，不能看最终测试修改 SLO。

探索性机会门槛：至少一个场景中，Balanced 在 3 个 seed 中至少 2 个比 Stay 的主指标相对提升≥10%，且非人工取消的错误率不升高超过 1 个百分点。Stay goodput=0 时不用相对百分比，以绝对 goodput 和完成率说明，不自动通过。此门槛是是否继续投入的小样本工程判据，不是论文显著性证明。

若未通过，输出 `opportunity_gate.json` 和建议：继续修复系统、调整预先声明的扰动，或没有足够证据支持投入 PPO。不要宣称 PPO 在理论上不可行，也不要继续长训碰运气。

奖励初版采用可审计的计数和时间积分，取消原先的逐项 `[0,2]` 硬裁剪：

```text
r_good * 本窗口首次满足终态 SLO 的完成请求数 / C_ref
-w_hold * ∫所有已发送但客户端未终态请求数 dt / (N_ref * 1s)
-w_first * ∫等待首输出且已超过 TTFT SLO 的请求数 dt / (N_ref * 1s)
-w_gap * ∫已首输出但超过 gap_limit 无新可见输出的请求数 dt / (N_ref * 1s)
-w_attempt * 本窗口迁移尝试数
-w_bad * 本窗口迁移失败/中止/明确拒绝数
在最终截止额外减去 w_tail * 截止未完成请求数 / N_ref
```

`gap_limit=max(1s, 5*tpot_slo)` 是无输出停顿代理，不称为精确 TPOT。请求在客户端真正终态前都持续产生驻留代价，避免仅统计引擎 waiting queue。失败请求不能被移出完整结果统计。

初始权重建议 `r_good=1, w_hold=0.2, w_first=0.5, w_gap=0.2, w_attempt=0.02, w_bad=0.2, w_tail=2`。这是待审计起点，不是已验证最优值。`C_ref` 使用健康机会实验 Stay 的平均每秒完成数且下限为 1；`N_ref` 使用同一运行的时间平均 outstanding 数且下限为 1。冻结后写入配置；最终测试不参与权重或尺度选择。

保留所有分量和未归一化量。先在固定策略轨迹上检查尺度：是否某分量长期淹没其余分量、是否奖励改善却主指标下降。最多进行一次有记录的训练前修订，之后冻结。不要声称代理奖励与最终主指标严格等价。

P3 验收：机会门槛通过；请求/动作审计仍通过；观测归一化、奖励尺度、SLO、训练/测试场景已经冻结。

## 8. P4：小规模 on-policy PPO 训练

建议独立配置：

```yaml
environment:
  num_instances: 4
  decision_interval_sec: 1.0
  arrival_horizon_sec: 120
  drain_horizon_sec: 120
  max_inflight_migrations: 1
  migration_cooldown_sec: 5.0
  max_state_age_sec: 2.0
  ttft_slo_ms: 2000
  tpot_slo_ms: 150
ppo:
  device: cpu
  torch_num_threads: 1
  hidden_size: 64
  learning_rate: 0.0001
  rollout_min_transitions: 512
  update_epochs: 4
  minibatch_size: 64
  gamma_per_second: 0.99
  lambda_per_second: 0.97
  clip_epsilon: 0.1
  entropy_coefficient: 0.01
  value_coefficient: 0.5
  max_grad_norm: 0.5
  target_kl: 0.02
training:
  max_wall_time_sec: 7200
  max_transitions: 10000
  max_updates: 20
  evaluation_interval_updates: 5
```

三项训练上限先到即停，训练内开发评估计入墙钟预算。不得宣称必然能跑满 20 updates。优先使用完整回合组成至少 512 transitions 的 batch，允许超过 512，不能截断回合来凑整。保留按回合的终态屏蔽，不能跨 reset 传播 GAE。

每批采样时模型、normalizer、奖励参数完全冻结；策略版本必须唯一。更新后清空旧 rollout，不能用历史 Balanced/Stay 数据伪装为 on-policy 样本。

复用现有分层网络前验证：全 mask 时 Stay 概率精确为 1、logp=0、entropy=0；非法动作概率为零；采样与重新计算 logp 一致。现有数值夹紧行为可能需要修正。全 mask 的强制 Stay 不贡献 actor/entropy 梯度，但可训练 critic。记录“可选择迁移的决策数”，不要把大量强制 Stay 当成充分探索。

实际拒绝动作保留原 logp；终态失败在实际观察窗口入奖励。不要把迁移全部成本与全部收益都塞进发起时刻。

GAE 使用真实 dt 的折扣和冻结的 finalised reward。加入梯度/损失有限性检查、approx KL 超阈值时提前结束本轮 epoch。只对 advantage 做批内归一化，不在线改奖励尺度。

保存 best、last、optimizer、随机数状态、normalizer、schema、预算、策略版本。恢复时从干净回合开始，不继续使用崩溃前未完成 rollout。

训练前必须通过测试：完整分层概率、mask 全禁用、动作拒绝、重复提交、跨窗口迁移、变 dt GAE 手算、有限终态与无效回合区分、策略版本不可混合、归一化冻结、断点恢复预算。

每 5 updates 用开发 seed、相同固定评价规则评估；以主指标选择 best，不以训练 reward 选择。必要时同时保留最后 checkpoint 作诊断。最终测试不得介入选择。

止损规则：出现事件缺口、unknown 动作终态、OOM、请求错误突然增多或清理不完整，停止采样并诊断；每种同一基础设施故障最多一次自动修复重试，保留两次证据。连续 3 次开发评估比配对 Stay 主指标低 20% 以上，则停止本轮并报告，不无上限调参。

P4 交付不强求性能胜出：至少有有效采样、实际成功迁移、有限数值的 PPO 更新、可恢复 checkpoint。若只是 Stay 或全 mask，不能宣称已验证迁移学习。

## 9. P5：冻结策略后的独立真实评估

运行前保存最终 checkpoint hash、源代码 manifest、协议 hash。第一次查看最终测试结果后，不再更新策略或调参；若必须修改，该测试组变开发数据，另选全新最终组并说明预算限制。

初版比较四种方法：Stay、受约束 Balanced、受约束 Defrag、Mig-PPO。它们共享请求、初始路由规则、SR 候选语义、并发上限、冷却和容量规则；如另测原生不受这些额外限制的启发式，单独标为 native，不能混用名称。

场景：A 正常稳定负载、B 已声明的倾斜负载、C 未用于训练的突发负载。突发 trace 在 P0/P3 冻结，不能根据 PPO 表现挑选。每场景 3 个最终 seed，四种方法共 36 回合；预算不足时标记未完成，并给出已跑矩阵，不缩小比较后宣称正式通过。

默认每回合到达 120 秒、最多排空 120 秒，服务复用。公平性固定 trace、模型、采样参数、外生扰动和评分边界，不要求迁移后状态轨迹相同。实际发送时间偏离计划也要报告，防止负载发生器改变到达过程。

PPO 使用开发阶段预先选定的确定性或随机策略执行方式；默认确定性评估。随机评估如需另做，独立列出策略采样 seed，不能与确定性结果择优拼接。

报告：逐回合原始指标、每个 seed 的配对差值、均值和范围、探索性置信区间。仅 3 个 seed 不足以支持强显著性结论；bootstrap 的重采样单位是完整运行/配对 seed，不是数万 token。

工程成功与研究成功分开写：

- 工程成功：闭环、统计、回放、checkpoint 和预算审计全部通过。
- 初步效果通过：B 场景主指标相对最强受约束启发式平均提升≥5%，且至少 2/3 seed 方向一致；A 场景平均退化不超过 5%，完成率不下降超过 1 个百分点，错误率不增加超过 1 个百分点。基线为零时单独报告绝对量，不做无穷大提升。
- C 场景完整报告泛化结果，不隐瞒退化。
- 上述效果门槛是预注册的初步判断，不是论文充分证据。未通过就写“真实 PPO 可运行，但未证明优于基线”。

## 10. 阶段命令与交付格式

以下命令是待实现的 CLI 验收合同，不是现成命令。编排器必须检查阶段依赖与 gate 文件对应的源码/配置 hash，不接受手工只写 `passed: true`。

```bash
.venv/bin/python benchmark/run_real_ppo.py --config rl_scheduler/configs/real_pilot.yaml --stage preflight --output-dir <RUN_DIR>
.venv/bin/python benchmark/run_real_ppo.py --config rl_scheduler/configs/real_pilot.yaml --stage telemetry --output-dir <RUN_DIR>
.venv/bin/python benchmark/run_real_ppo.py --config rl_scheduler/configs/real_pilot.yaml --stage bridge --output-dir <RUN_DIR>
.venv/bin/python benchmark/run_real_ppo.py --config rl_scheduler/configs/real_pilot.yaml --stage opportunity --output-dir <RUN_DIR>
.venv/bin/python benchmark/run_real_ppo.py --config rl_scheduler/configs/real_pilot.yaml --stage train --output-dir <RUN_DIR>
.venv/bin/python benchmark/run_real_ppo.py --config rl_scheduler/configs/real_pilot.yaml --stage evaluate --output-dir <RUN_DIR>
```

先验证 `.venv/bin/python` 确为现有 Llumnix 运行环境；若不同，统一替换为探测到的路径，不新建一套环境掩盖依赖问题。实现 `--dry-run` 输出即将运行的配置、预计回合数和剩余预算。

每阶段 gate 至少包括：`stage/status/source_hash/config_hash/checks/artifacts/reason/remaining_budget`，status 为 passed/failed/incomplete，不仅一个布尔值。

结果目录至少包含：

```text
preflight.json
experiment_spec.yaml
source_manifest_before.json
source_manifest_after.json
implementation.md
budget.json
gates/
traces/manifest.json
runs/<episode_id>/requests.jsonl
runs/<episode_id>/decisions.jsonl
runs/<episode_id>/migrations.jsonl
runs/<episode_id>/reward_windows.jsonl
runs/<episode_id>/audit.json
training_metrics.jsonl
checkpoints/
evaluation/paired_results.csv
evaluation/report.md
reproduction.md
final_status.md
```

压缩事件日志前验证解压字节数与 SHA-256；不要删除唯一原始证据。当前 `/workspace` 并非持久化挂载卷，报告中指出备份需求，但未经指示不向外部服务上传数据。

最终汇报必须回答：完成到哪阶段、哪些 gate 通过、实际运行时长和 GPU 小时、真实迁移成功数、PPO 更新数、最终对基线结果、仍未解决的问题。列出改动文件与已跑测试；不要用“训练成功”替代真实性能结果。

## 11. 执行模型不得自行作出的简化

- 不能把 `env.step()` 替换成 sleep 后读取旧 CSV 就宣称是真实环境。
- 不能把全局 ordinary migration 关掉却保留一个实际上不可执行的 PPO 动作入口。
- 不能给所有缺失字段填 0 而不提供 missing mask。
- 不能把仅有内部 API 时间的指标标为客户端延迟。
- 不能用相同旧测试 seed 反复修代码后继续称独立验收。
- 不能在 GPU 推理期间让训练器默认占用 cuda:0。
- 不能把 4 GPU 当作 4 个独立训练环境。
- 不能只报告成功请求的延迟而省略未完成与截尾比例。
- 不能删除失败回合而不在总实验清单和预算中记录。
- 不能跳过机会实验，直接以更多 PPO 步数替代迁移收益证据。

执行顺序固定为 P0 → P1 → P2 → P3 → P4 → P5。某阶段失败，先在本阶段诊断；不能进入依赖它的阶段。预算耗尽或证据不足时，交付可复现的失败结论，也是本任务的有效结果。
