# 测试监督 Agent 任务书：Mig-PPO V2 多热点环境筛选与真实轨迹拟合

日期：2026-11-05（Asia/Shanghai）。本文是待执行的任务书，不是实验通过报告。将本文完整交给测试监督 Agent。

## 1. 本轮目标、授权范围与优先级

你接手 Llumnix 四实例迁移调度项目。请落实必要实现、CPU 验证、真实环境测试、拟合与交付，不能只提出建议。先读适用的 `AGENTS.md` 和当前代码，保留用户未提交修改及原始实验记录。

本轮用户要求：

1. 设置 **4 个热点位置（实例 1、2、3、4）×3 档热点强度，共 12 个场景**。
2. 另外设置 **1 个采用原生 Llumnix 新请求到达/初始路由路径的环境**，共 13 个目标场景。
3. **每个候选只测试一次：同一 trace 的 Stay 一轮 + Balanced 一轮。首选失败允许一次预注册备选；备选也失败，直接跳过并进入下一个环境。** 不反复重试同一配置。单轮服务总时限为 **200 s**。迁移策略相比 Stay 有主服务成本收益即可采用。
4. **先完成全部环境的评估和参数选择，再统一进行逐环境物理模型拟合，最后用独立真实数据验证。** 每个环境可复用最终选定候选的一对 Stay/Balanced 评估数据作开发拟合；另用新 seed 的一对数据作独立验证，不能把同一对数据同时称作拟合与独立验证。审计并复用旧 Q15-B 中仍有效的部分，其余重新拟合。
5. 交付可供后续仿真 PPO 使用的环境注册表、模型、原始证据和复现命令。本轮不开展正式 PPO 训练、真实 PPO 更新或论文最终评测。

默认角色划分：12 个人工热点场景是后续仿真训练/开发的候选；`NATIVE` 是独立测试环境，即使其收益为正，也不自动加入 PPO 训练采样。对原生环境做物理模型校准属于模型开发，不等于已经完成原生策略盲测；以后评估 PPO 需另外锁定未参与拟合、阈值或 checkpoint 选择的原生测试 trace。

本任务书覆盖旧文档中的“找到一个就停止”“只允许 Q15-B”“不新增 Stay/Balanced 配对”等阶段限制。旧文档是方法和历史证据来源，不是本轮新的通过条件。不要在找到第一个赢家后结束剩余矩阵，也不要为了凑齐 13 个通过结果篡改失败场景。

执行本任务书时，实施和运行本轮限定范围内的真实四卡实验属于任务范围；实现、审计、拟合和预算内运行不必逐轮询问用户。若现场存在本任务无权停止的其他服务，记录冲突并等待资源，不终止它们。不要启动下一阶段训练或扩大本文的候选矩阵。

## 2. 先核对的文档、代码与旧证据

方法来源：

- `rl_scheduler/docs/research_implementation_plan_v2.md`：同路径对照、完整 cohort、服务收益与均衡收益的区别、校准分层。
- `rl_scheduler/docs/p0p1_fast_track_prompt.md`、`p0p1_fast_usage.md`：冻结 trace、实际 engine admission、受管服务、预算与失败账本。
- `rl_scheduler/docs/p2p3_q15b_execution_prompt.md`：数据来源/划分、物理模型、给定动作回放与策略闭环分别验收。
- `rl_scheduler/docs/ppo_entrypoints.md`：当前 Mig-PPO V2 状态、奖励和最新执行设置。

当前公共实现：

```text
benchmark/run_ppo_sim.py
benchmark/run_ppo_real.py
benchmark/run_p2p3_q15b.py
benchmark/run_p0p1_fast.py
benchmark/run_p0p1_q15b_pair_replay.py
benchmark/run_full_completion_balance.py
rl_scheduler/ppo_runtime.py
rl_scheduler/cohort_ppo.py
rl_scheduler/cohort_reward.py
rl_scheduler/real_observation.py
rl_scheduler/full_completion.py
rl_scheduler/real_bridge.py
rl_scheduler/real_runner.py
rl_scheduler/real_opportunity.py
rl_scheduler/q15b_data.py
rl_scheduler/q15b_sim.py
rl_scheduler/env.py
rl_scheduler/service_model.py
rl_scheduler/causal_calibration.py
llumnix/manager.py
llumnix/global_scheduler/{global_scheduler,dispatch_scheduler,dispatch_policy}.py
llumnix/entrypoints/vllm/real_metrics_stream.py
```

优先审计这些旧目录，按索引流式读取，包含压缩文件；不能假定目录或已清理的旧脚本仍存在：

```text
benchmark/results/p0p1_fast_20261104T104200Z/
benchmark/results/p0p1_fast_replay_20261104T124801Z/
benchmark/results/p2p3_q15b_20261104T135200Z/
```

旧 Q15-B：Qwen2-1.5B、128 请求、128 输入/1536 输出 token、0.6 req/s、115/128 热点路由。旧 seed 141104 同次 replay 的 Balanced/Stay 成本改善约 16.94%，只作为历史开发证据，必须由原始请求日志复算后引用。旧校准只覆盖有限 Q15-B Stay/Balanced；Q15-A 的负收益符号曾未复现，不能把旧模型称为通用模拟器。

## 3. 冻结当前 PPO 合约，不能沿用旧设置

本轮所有用于新环境筛选和拟合的 Stay、Constrained-Balanced，以及测量专用 PPO-STAY，都采用相同的 **1.5 s 名义决策间隔、0 s 迁移冷却**。仍保持全局单个在途迁移、SR 候选、真实容量/新鲜度/健康检查和未确认回执占用。

特别检查：当前公共代码可能只在有 PPO controller 时自动采用新设置；直接调用旧 Stay/Balanced 分支仍可能得到 1 s/5 s。必须显式打通所有本轮模式的配置，核对观测 builder、仿真 `_start_migration`、真实 lease/Manager 仲裁器三处，再从运行日志验证。仅改 YAML、常量或报告字段不算完成。保留旧实验复现模式，不能把旧配置或原始结果改成新版本。

状态仍按当前 V2：实例 9 项、全局 8 项、每个有向对 2 项，缺失位及顺序与当前 normalizer 一致。`cooldown_remaining_sec` 在本轮恒为 0，不能擅自删字段改变 checkpoint 形状。Stay+12 对动作顺序不变。

奖励调用共享 `MigPPOV2Reward`：

```text
r = -dt * (.40 D_mean + .30 Q_mean + .20 H_mean + .10 B_star_mean)
    - xi * failure_count / failure_ref
    + .10 * (gamma_t * Phi_next - Phi)
    - invalid_migration_cost * invalid_migration_count
Phi = -(.45 Q + .35 H + .20 B_star)
gamma_t = .99 ** dt_seconds
lambda_t = .97 ** dt_seconds
```

默认固定参考尺度、服务时间阈值和权重取当前 `rl_scheduler/configs/mig_ppo_v2_reward.json` 与实现。热点占用阈值 `hotspot_threshold` 必须依据人工热点 fit 数据中的 KV—拥塞关系校准，并冻结选择规则、数值、来源和不确定性；不能默认 80%，也不能用验证集、原生测试环境的性能或哪条奖励更好来选择阈值。该阈值不是下文的“热点请求占比”。尽量采用所有注册环境共享的奖励尺度/阈值；确需分环境阈值时显式版本化，禁止跨环境直接比较未经对齐的 return。

处理好依赖顺序：原始 Stay/Balanced 采集和按服务成本筛选不依赖尚未校准的热点阈值，PPO-STAY 开销检查也不需要在线计算奖励。允许计划中先记 `reward_threshold_pending`，先完成全部场景的评估采集与选择，再统一用选定配对拟合/校准并冻结模型和奖励参数，最后采集独立验证配对；不能边评估环境边拟合调参。不为迎合旧 `prepare` 的必填参数临时填 80% 或其他无依据阈值；奖励未冻结时不启动 PPO 训练。阈值确实不可识别时明确阻塞对应奖励校准项，保留物理拟合和真实筛选结果。

服务时间压力使用真实客户端输出接收时间线，保留每个新 token/输出帧的时间和累计计数。多 token 帧使用实际共同接收时间，不插值伪造 token 间隔；零增量最终帧不重置停顿。只有 TPOT 汇总的日志不足以重建 V2 时间奖励。

以后 PPO 更新仍遵守：主动 Stay 参与 Actor/Critic，强制 Stay 不参与 Actor 损失、优势归一化、entropy/KL/clip 统计；全部有效窗口参与 Critic 和完整真实时间 GAE。不能压缩或删除强制 Stay 窗口。本轮只做 CPU 合约检查，不把真实 Stay/Balanced 日志作为 on-policy PPO 更新数据。

## 4. 13 个目标场景及参数选择

### 4.1 共用服务和请求设置

以已接受 Q15-B 的实际服务配置为起点：Qwen2-1.5B、四个 TP=1 副本、half/eager、Ray CPU=16、rayrpc/SR/rayqueue；`max_model_len=2112`、`max_num_seqs=128`、`max_num_batched_tokens=2112`、`gpu_memory_utilization=0.65`。现场只读核验 GPU/依赖/模型版本和实际生效参数，禁止静默切为 0.5B 或压低并发制造热点。

**首选每轮减少为 48 个请求**，输入仍恰为 128 tokenizer token，实际输出仍恰为 1536 token；temperature=0、ignore_eos=true、n=best_of=1。首选请求率统一 0.6 req/s；减少的是请求总数，不降低服务并发上限、不擅自缩短输出。

一轮指一种方法的一次 cohort；一对/一组指同一 candidate 和 trace 的 Stay、Balanced 各一轮。**单轮从计划 episode 起点计时，总服务窗口最多 200 s（3 分 20 秒）**。首选请求到达时间域为 `48/0.6=80 s`，余下最多约 120 s 留给排空；实际到达跨度以冻结 trace 为准。请求全部完成且迁移已对账就提前结束，不固定空转到 200 s，也不在 80 s 后新增请求。

这是为 200 s 短回合选择的新请求规模，不是已实测保证能完成；强热点若仍超时，按 §4.4 改用更小请求量的备选，而不是延长窗口。原先 128 个请求在 0.6 req/s 下仅到达域就约 213 s，不能继续保留再在 200 s 截掉后半段。旧 Q15-B 的请求数和轮长只作历史参考，不复用为本轮耗时承诺。

执行 Agent 须将真实运行、仿真、`remaining_total_sec` 的 horizon、窗口审计及 watchdog 同步显式配置为 **200 s**，不能只改报告字段或沿用公共代码中的 3600 s 默认值。200 s 内未完成则记 incomplete，尚未结束的请求不得从收益分母删除；按既有确认取消/清理协议收尾，不继续性能采样、不伪造 terminal、也不无限等待排空。启动/最终清理不属于 200 s 性能窗口，但计入总预算并单列实际耗时。

继续使用现有固定 N 轨迹生成方式：在 `N/qps` 区间内生成并排序 N 个时间点，记录具体生成器/版本和结果。若代码是 uniform order statistics，应如实命名，不能改写成未经条件化的指数间隔 Poisson 过程。原生路由场景也复用同一外生到达时间和请求内容，不因为名字“原生到达”而另换发包过程。

请求量 N 必须成为显式配置：修改只允许 128 请求的 trace/审计/仿真断言、请求覆盖分母和输出总量核对，不能只缩 trace 而断言仍要求 128/128。奖励中的固定 `n_ref/q_ref/q_max_ref/failure_ref` 不随 N 自动缩放；normalizer 和权重仍按共享冻结定义。

### 4.2 人工热点矩阵

本轮把“强度”明确为 **初始发往热点实例的请求占比**，不同时改变 QPS、请求长度或 GPU 速度：

| 强度 ID | 目标热点比例 | 热点请求数 | 三个非热点实例的请求数 |
|---|---:|---:|---|
| L | 50% | 24 | 8、8、8 |
| M | 70% | 34 | 5、5、4 |
| H | 90% | 43 | 2、2、1 |

表内为首选 N=48 的整数分配。通用规则为 `floor(N * fraction + .5)`；首选实际比例为 50%、70.8333%、89.5833%，不能把目标比例当实测比例。非热点余数按预注册稳定顺序分配。

固定场景 ID：

```text
HOT1-L HOT1-M HOT1-H
HOT2-L HOT2-M HOT2-H
HOT3-L HOT3-M HOT3-H
HOT4-L HOT4-M HOT4-H
NATIVE
```

实例 1–4 是显式且稳定的逻辑位置，记录与 instance_id、epoch、worker PID、GPU UUID/PCI bus 的对应。配对期间 epoch 和卡映射不能漂移。轮换热点位置不能仅改 seed：旧 `make_trace` 由 `seed % 4` 决定热点，必须增加显式 hotspot 参数或对初始路由做可审计的实例排列。不同位置共享相同到达/请求基础 trace；首选使用同一随机排列的前 24/34/43 个请求组成三档热点集合；更小 N 的备选重新按同一整数规则生成并记录实际比例，避免把强度变化混入另一组到达时间或 prompt。

同一位置/强度的两种策略必须使用完全相同的输入、到达时刻、路由和生成参数。保存不含路由的 `base_trace_hash` 和包含路由的 `full_trace_hash`。

### 4.3 原生 Llumnix 环境的定义与实现缺口

`NATIVE` 表示 **不注入 initial_instance_id，由现有 Llumnix 原生新请求分发器决定首次实例**。以实际服务配置中的 `DISPATCH_POLICY=load`、`DISPATCH_LOAD_METRIC=remaining_steps` 为核对起点，冻结当前算法、top-k/随机并列选择、状态更新频率、随机状态和完整配置。禁止用四实例 round-robin、均匀固定路由或上一轮记录的路由冒充原生路径。

这不等于启用另一套原生迁移器：Stay 与 Constrained-Balanced 仍通过同一控制 lease 隔离 routine migration，只保留原生**新请求分发**。Stay 必须实证为零迁移，Balanced 只能有本任务的执行意图；不能让两套迁移器同时动作。

同轨迹指 **相同外生请求、计划到达时间、内容和生成参数**。原生路由依赖策略影响后的当前系统状态，因此 Stay 与 Balanced 的实际首次实例可以不同；不能要求两次实际路由逐请求相同，也不能把 Balanced 的路由强行重放给 Stay。控制相同的分发算法及随机机制，记录各自真实决策输入和首次 admission。

必须先补齐并用 CPU HTTP fixture 验证：

- `FullCompletionIO` 发送链路不注入 `initial_instance_id`，trace 明确标为 `routing_mode=native`。
- Manager 原生 dispatch 分支记录可审计的初始路由事件及选择信息；目前显式路由分支有 `real_initial_route`，不能假定原生分支已有等价证据。
- `summarize_full_cohort`、`run_episode`、执行审计和 trace 校验能处理未预设的实例；不能通过预期路由字典为空来让原生路由审计“空对空通过”。
- 原生模式要求每个已接纳请求的 Manager 首次分发与首次 engine admission 一致，属于四个健康 epoch；迁移后的 owner 不能覆盖首次实例。
- 仿真原生模式根据当时模拟状态执行同一分发逻辑/可验证适配，含请求计数、状态陈旧、top-k 和并列规则。回放真实路由仅可称给定路由诊断，不能用于原生闭环验收或向 PPO 提供未来实例选择。

原生模式固定 `role=test_only`。若没有正收益，保留它作为测试/负对照环境并明确 `selected_for_positive_gain=false`；不得为让它通过偷偷改为人工热点。即使正收益也仍是测试环境，不能自动混入训练。它须交付已有真实轨迹和拟合/验证状态，不得从第 13 行消失。

### 4.4 首选失败后，只允许一个条件式备选

逐环境执行，不先遍历全部首选再回头补跑：**首选一次配对 → 通过则锁定；未通过则一次备选配对 → 通过则锁定备选；仍未通过则跳过，立即进入下一个环境。** 每个 candidate 只运行一次，不因小负收益增加 seed 或重复到通过。

首选统一为 `N=48, qps=0.6`。计划阶段预注册以下条件式备选，每个环境至多选择其中一项：

| 首选未通过的主要原因 | 唯一执行的备选 | 到达时间域 | 窗口上限 |
|---|---|---:|---:|
| 任一轮到 200 s 仍未完成、积压无法排空 | `N=32, qps=0.6`，先降低请求总量 | 约 53.33 s | 200 s |
| 完整配对但无收益、热点/机会不足 | `N=48, qps=1.0`，测试更集中的到达 | 48 s | 200 s |

不同时执行两个备选，不再增加第三个候选。模型、输入/输出长度、热点位置和目标路由比例不变；备选新建 candidate_id、trace/hash，Stay/Balanced 内部必须使用完全同一配置/轨迹，不能与首选的某一轮拼成配对。原生备选也只改 N/qps，仍保持原生分发，不能改成人工热点。

候选中某一轮已 incomplete 时，该候选不能构成有效收益配对，可省去另一轮，记录已消耗尝试后转备选。环境本身的超时、负收益或无机会按此规则推进；若是全局服务失效、生产者/路由审计失败，则先停整批定位故障，不能带着不可信数据继续矩阵。预算不足则明确 not_run，不放宽 200 s。

备选仍未通过的人工热点环境记录 skipped，不进入正式模型拟合清单；其日志可保留作诊断。NATIVE 始终保留测试环境身份，有完整审计配对时可作对应测试环境拟合，即使无正收益也不能抹去负结果。不同候选/场景不能共用一轮 Stay 冒充各自配对基线。若最终 N/qps 因位置而不同，不把跨位置差异解释为纯硬件位置效应。

## 5. 相比 Stay 有收益：唯一性能采用条件

比较方法为 **Constrained-Balanced**：复用 `balanced_action`，四实例共享 SR/容量/单在途保护，1.5 s 控制周期、0 冷却。预热排空后读取实际 KV 容量并冻结 `F_threshold=min(total_blocks_i)/8`；8565/1070.625 仅为旧运行核对值，不能硬写为现场容量。不把 Balanced 选择阈值加入 PPO 合法性 mask，不把该启发式称为“完整原生 Llumnix 迁移策略”。

主服务成本在任何新筛选数据出现前冻结：

```text
C = mean(E2E_seconds) + CVaR95(E2E_seconds)
gain_C = (C_Stay - C_Balanced) / C_Stay
```

沿用现有 CVaR95 精确分数尾部实现。**一条完整、审计有效的同轨迹配对中 `gain_C > 0` 即满足性能采用条件**。无需额外 seed，不要求最低 5%，不要求 mean、P95、TTFT 和 TPOT 同时改善。它们和 V2 奖励四项都要完整报告，不能换成恰好获胜的指标或只看 GPU/KV 均衡。

基础证据有效性不是新的收益门槛：两轮必须完整服务同一冻结的 N 个请求（首选 48，超时备选 32）、实际长度匹配、时钟/路由/日志/迁移审计通过。Stay 为零迁移；要把收益归为迁移环境证据，Balanced 至少应有一笔请求 ID/源目标/复制 blocks/终态齐全的真实成功迁移。若两边都零迁移，微小耗时差只能是同策略运行差，不能当迁移收益。成功数、覆盖率、重复回迁、机会比例仅作诊断，不恢复旧的次数或比例门槛。小样本 CVaR95 仍按分数尾部精确计算，并报告 N 与尾部有效样本量，不能宣称尾部估计稳定。

有错误、失败、未知终态或不完整 cohort 的配对不参与正收益选择，保留为失败证据；不得只统计已完成请求从而得到虚假收益。V2 本身保留有审计证据的失败惩罚，该合约仍应测试，本轮干净环境筛选不靠失败数据通过。

明确写入报告：这只接受一个特定 trace 上的开发环境，不证明统计显著、位置对称、PPO 有收益、所有迁移有益或原生路径必然有改善。小正收益照样按用户规则接受，但必须显示具体值、运行顺序和不确定性，不包装成稳健收益。

## 6. 执行顺序、同路径控制与预算

先 CPU 后 GPU，冻结 `plan.json`、13 行环境矩阵、候选分支、种子、数据划分、判定函数、阈值规则、配置/代码 hash 和预算，再启动。

### 6.1 阶段 A：全部环境评估完再拟合

1. 审计旧数据/代码；完成 1.5 s/0 s/200 s、显式 N、原生路由、奖励时间线的 CPU 检查。预先登记所有评估候选 trace 和独立验证 seed，禁止与旧账本冲突。
2. 独占四卡服务，预热一次，核验容量、GPU UUID、实际引擎参数和生产者/输出转发。没有必要审计证据不开始正式轮次。
3. 按矩阵顺序逐环境执行 §4.4：首选失败后立即试唯一备选，备选仍失败就跳过、进入下一个环境；相邻环境交替 Stay/Balanced 顺序。只做完整率、执行审计和第 5 节收益判断，不在中途拟合物理模型或改变奖励参数。
4. 不加 Stay-A/Stay-B/PPO-STAY 完整重复回合。P0 引用可核验旧证据并结合本轮 CPU/控制耗时/发包延迟检查；缺独立开销对照则明确未独立验证，不宣称 5% 等价。全局测量/服务故障按 §6.5 停止。
5. 全部 13 行完成评估或明确跳过/未执行后，封存 `selection_summary.json` 和最终环境参数。完整保存所有原始快照、时间线和迁移回执，后续拟合复用这些数据；不能边拟合边回头重选环境。

### 6.2 阶段 B：统一进行物理模型拟合

阶段 A 结束后，再统一审计旧模型复用性、提取逐环境数据、校准奖励阈值、拟合物理模型和进行 CPU 开发检查。每个拟合环境原则上使用**最终选定候选的一对评估数据**，失败首选不能冒充备选的同环境标签。兼容的旧数据/其他环境物理记录可明确引用为先验或共享物理核心，但逐项记录来源和适用范围。

拟合结束冻结每环境模型、奖励/归一化、执行配置和代码 hash，再进入阶段 C。通常先停止本阶段任务服务，再做 CPU 拟合，避免租用四卡空等；重新启动验证服务时按 GPU UUID 恢复相同逻辑位置，重新记录 epoch，核对模型/容量/配置，不能沿用旧进程身份或把重启后的容量变化隐瞒。

### 6.3 阶段 C：新 seed 的独立真实验证

每个需要交付模拟模型的环境，**再用一个新 seed 的 Stay/Balanced 配对**验证，选定 N/qps、热点、原生分发设置和 200 s 上限都保持不变。验证 trace 与全部候选筛选/拟合 trace 按 base trace 分组分离，不能只换 run_id、路由标签或切请求行。共享 base trace 的各位置/强度不能跨 fit/validation。

采集验证数据前模型和阈值已经冻结；验证数据不用于调参、选模型、调奖励阈值或重选候选。验证两种方法的实际效果都要报告，验证中 Balanced 不再获胜不撤销阶段 A 的单 trace 采用事实，但要如实标出收益不稳定；模型是否能重现验证轨迹/策略差异按 §9 判定。

每个环境独立验证只做一对，不因结果差再采新 seed。若用验证数据修正模型，该数据立即降为 development-validation，`independent_validation_passed=false`，本轮不为重新取得盲测自动加轮；交付修正模型和验证待补状态。

### 6.4 时间与次数预算

- 评估：13 个环境首选至多 13 对，必要备选再至多 13 对，最多 52 个 cohort；通过首选就不跑备选。
- 独立验证：只针对最终要交付模型的已选人工热点环境和 NATIVE 测试环境，至多 13 对/26 个 cohort。
- 合计正式采样最多 **39 对、78 个 cohort**；不是固定要跑满。拟合本身复用评估配对，不新增专门的拟合采样。失败尝试计数，不加同配置重试或 P0 重复。

**单轮最多 200 s，一对两轮最多 400 s（6 分 40 秒）**，完成即结束。首选到达域 80 s；备选域见 §4.4。最多 78 轮的服务窗口合计上界为 15,600 s（4 小时 20 分），启动/排空/清理另计并公开。

本任务 GPU 集群墙钟目标按实测轮长尽量压缩、默认硬上限 **5 小时**（含两阶段启动、预热与清理，约 20 GPU-hours 上界）；CPU 提取/统一拟合/验证软上限 2 小时、硬上限 4 小时。每开启下一完整配对，必须预计能跑完并预留至少 8 分钟收尾余量；不能因为总预算不足截短单轮后声称完成，剩余行记 not_run/incomplete。不得把 CPU 拟合时闲置服务的墙钟从 GPU 占用预算中扣掉。

### 6.5 超时、审计故障与服务清理

Stay/Balanced 保存相同强度的状态/日志，轮间核对无遗留请求和 pending 迁移。服务 readiness 总等待不超过 8 分钟；120 s 无输出进展或 200 s 单轮硬截止触发本任务 watchdog，按不完整回合处理，再依据 §4.4 判断是否运行唯一备选。

独立 supervisor/watchdog 只管理本任务拥有的进程。第一项全局数据/路由/生命周期审计失败时停止后续正式比较，保留证据；必要修复先做 CPU 验证，不在同一候选上无限自动重试、不跨 attempt 拼接结果。普通环境未通过不等于全局故障，按首选—备选—跳过流程继续下一环境。所有服务/取消/flush 必须有可核验终态；无法安全清理时停止，不把未确认状态留给下一环境。

## 7. 旧拟合是否复用：逐层判断，逐环境登记

**修改状态和奖励不自动使物理服务/迁移模型失效，也不自动证明旧模型仍适用。** 至少输出以下审计：

| 层次 | 可以复用的条件 | 不能直接复用的情况 |
|---|---|---|
| 服务速度、KV/请求生命周期 | 同模型/硬件/引擎、单位一致、负载与上下文覆盖、新真实验证误差合格 | 新强度/实例进入覆盖外，服务路径/容量变化，物理误差不合格 |
| 迁移成本与干扰 | copied blocks、方向/设备、暂停语义可追溯，零冷却的新迁移频率仍在有效域 | 旧 5 s 冷却造成的稀疏迁移无法解释连续迁移、方向异质性或源目标干扰 |
| 状态构造/掩码 | 原始事件和快照足以按决策接收时间重建 V2 全部字段 | 只有旧 tensor/汇总，缺客户端 token 时间线、真实采样时刻或候选 blocks |
| 奖励、GAE、checkpoint | 新奖励只从原始时间线重算；新配置重新冻结 | 旧 E2E+CVaR reward、gamma=lambda=1、1 s/5 s 的轨迹不能改标为当前 PPO 合约 |
| 新环境覆盖 | 该环境有独立索引、验证证据及适用域 | 仅把 Q15-B 模型复制/改名 13 份便称逐环境拟合通过 |

从旧 fit/validation 的真实文件确定热点位置，不按报告概括猜测。旧生成器 `seed % 4` 会让不同 seed 对应不同热点；不能把旧单源数据等同于四位置均覆盖。旧 141104 数据已用于模型选择，归为开发验证，不重新称盲测。

先将旧物理模型的只读副本、适用域和 hash 冻结，再用新参数/当前执行设置的真实配对验证。若逐环境证据支持，共享同一个物理系数文件是允许的，但每个环境必须有独立环境配置、真实数据索引、覆盖分析、残差和验收结果，并明确 `reused_validated`。需要新估计时记 `refitted`；资料不足记 `unavailable`，不能填默认数后通过。

逐环境数据使用规则：

1. **一对能作初步拟合，但不是“所有模型参数都已充分辨识”的保证。** Stay 提供未迁移服务轨迹，Balanced 额外提供迁移及其后果；一对可以包含多条服务/迁移观测，但它们来自同一 trace，不能把相关记录当成多个独立实验。
2. 每个环境有自己的最终选定评估配对索引，作为 fit 数据；不能用一对数据替代全部 13 环境的实测覆盖。物理条件和覆盖支持时可共享低复杂度系数或经过审计的旧模型，并逐环境验证。无迁移、块大小/方向覆盖不足时，不能从零样本拟合独立迁移成本或宣称全部方向可靠。
3. 按阶段 A→B→C 执行。新 seed 的验证配对在模型冻结后采集，所有完整 base trace 分组与拟合/筛选隔离；不得按每秒或请求随机切分。旧已参与选择的数据不重新命名为独立验证。
4. 每个采用环境交付模型/复用来源、真实 fit 与独立 validation 配对的 hash、覆盖和误差。模型必要部分无法由这一对辨识时，优先用兼容旧数据/共享物理核心并声明限制；仍不足则 `fit_insufficient`，不能补任意参数使其通过。
5. NATIVE 不论是否正收益都保留测试环境身份，具备完整审计评估配对时进行拟合和独立验证，不用于 PPO 策略训练。首选、备选均无完整配对时登记数据不足，不从部分请求制造完整性能/奖励标签。
6. 旧 Q15-B 可作为物理模型和历史数据来源，但当前 N=48/32、200 s 已是新版本环境，不能声称复现旧 128 请求运行。由相应矩阵数据做兼容性检查/重拟合，不额外添加第 14 个筛选场景。

本轮的独立验证是环境/物理模型验证，不是训练后 PPO 的最终性能盲测。若用于调参则降为开发数据；预算不足/验证不完整/误差不合格分别标 pending/incomplete/failed，不能宣称仿真已获独立验收，也不自动追加第三对。

## 8. 真实轨迹提取与模型要求

每条记录带 source run、candidate/environment_id、trace_group、实例/方向、策略、配置 hash、时间单位、是否完整及缺失原因。保存原始客户端和服务端事件、原始快照/决策、首次路由/engine admission、迁移回执与生命周期；压缩后校验可读性和 hash。

按每个环境提取：

- 服务：running/waiting、KV used/free/独立 reservation、candidate blocks、last_step、实际 output token 进展、上下文/batch 信息。区分引擎 step 和端到端有效 cadence，不重复叠加通信/输出延迟。
- 客户端：实际 sent/first/每次正 token 增量/terminal 时间、计划到达与发包偏差、outstanding/overdue/stalled、5 s token 速率。
- 迁移：实际 request_id、source/destination、copied blocks/可核验 bytes、start/copy/commit/terminal、成功/拒绝/失败、源目标负载；未知暂停单列，不能把整个复制时长称为实测暂停。
- 控制：状态 measured/received/decision 时间及年龄、回执确认延迟、busy 区间、完整动作 mask、SR 变化、三类决策统计。0 冷却后重复/反向迁移必须真实建模和报告，不额外加隐藏冷却。
- 原生路由：当前分发所用的负载/请求计数、候选/top-k/并列、选中实例、信息更新时刻。没有记录的输入不能用未来日志填补。

先在 fit 数据上比较低复杂度分桶模型与现有回归，以开发误差、覆盖和复杂度选择，并在阶段 C 前冻结；不得查看独立验证误差后再挑模型。可引用匹配的旧数据作分组开发检查；只有一对新 fit 数据时不能伪造多个独立实验。不引入 GNN/GRU/收益分类器。允许共享物理核心加逐实例/逐方向残差，只有实测支持时才启用额外参数；缺样本方向标覆盖不足，不能靠热点重命名假定四卡完全对称。

以 `q15b_sim.py` / `env.py` 为基础扩展显式环境配置和 native routing，不能继续硬编码 8565 blocks、141103/141104、固定 hotspot、1 s/5 s 或只允许 Q15-B 的输入。不得任意设置 1% 随机失败；无失败样本只说明本批未观测到，仿真关闭随机失败时标明该适用限制。

仿真推进实际请求/Prefill/Decode、KV 分配释放、抢占重计算、迁移预留/归属/追加 KV、复制与服务暂停；不能用真实未来日志在时间点强行校正状态。最终长度可用于模拟事件推进，不能作为 actor 在线特征。

若 NATIVE 的真实样本没有迁移，只能验证其原生分发、服务推进及已执行动作的后果。可在物理条件和覆盖允许时引用人工热点数据的迁移成本模型，但必须标明原生场景迁移动作的实测覆盖不足，不能用零迁移的 Stay/Balanced 闭环宣称已验证 PPO 在原生环境中的迁移后果；该策略覆盖验证留给后续独立任务。

输出 `reward_reconstruction.json`：同一请求时间线、快照、动作/回执和边界下，新奖励四项、失败计费、终态势函数、真实时间 gamma/lambda 必须可重算。物理模型复用不能跳过新奖励/状态合约验证。不能把整轮 Balanced 的收益分给每次迁移，也不能给模拟器加固定迁移收益系数。

## 9. 逐环境仿真验收：与环境采用分别判定

阶段 B 先在 fit 配对上做 CPU 开发检查；模型冻结并完成阶段 C 的独立真实配对后，以独立配对作为主要验收数据。分别报告 fit/validation 误差，不得只报拟合优度。冻结规则后执行：

1. **给定动作/路由回放**：排查事件/归属/单位和成本。标明使用了真实动作或路由，不能称闭环。
2. **Stay 与 Constrained-Balanced 闭环**：在模拟自身状态上重新选择动作；native 模式还须重新选择新请求初始实例。对比相应真实完整配对。
3. **PPO CPU 合约 smoke**：共享当前网络、状态、normalizer、13 动作、1.5 s/0 s、V2 reward、choice_only、GAE，完成 rollout/update/checkpoint round-trip；合成更新不算真实 on-policy 学习。本轮不训练正式策略，也不以随机初始 PPO 必须获胜验收。

逐行报告以下结果，不能只给全矩阵平均误差：

| 检查 | 本轮预注册工程验收线 |
|---|---|
| 请求/事件/KV | 请求身份和长度正确，时间因果/源目标归属守恒；KV 无无解释负值/超容量；零冷却和单在途执行真实生效 |
| 接口 | 当前特征顺序/单位/缺失位、动作编号、同逻辑输入的 mask 一致；数值转换预设 atol=2e-6、rtol=1e-6 |
| 奖励代码 | 相同原始输入的四项与总奖励重建误差≤1e-10；正确 terminal、真实 dt；零增量输出不重置停顿；强制 Stay 轨迹不丢失 |
| mean/P95 E2E、C | Stay、Balanced 各自相对误差≤20% |
| outstanding 曲线 | 对齐 episode 起点后，绝对误差积分/真实面积≤20% |
| waiting / KV | 每实例 waiting 的曲线归一化绝对面积误差≤20%；真实平均 waiting<1 时改用 MAE≤1 请求；每实例有效 KV fraction 的时间 MAE≤0.05 |
| 新奖励压力 | D 的曲线归一化绝对面积误差≤20%，真实均值<0.05 时用 MAE≤0.05；Q、H、B_star 的时间 MAE 各≤0.05；另报逐项积分和去 shaping 的实际压力成本误差 |
| 成功迁移数 | 误差≤max(5, 真实成功数×20%)，同时报告不同请求覆盖、重复和反向迁移 |
| 迁移耗时 | 有样本时中位数相对误差≤20%、P95≤30%；不足以估计的尾部标 unavailable，不伪造已通过 |
| 配对收益 | 报告模拟/真实 gain_C 与 mean 改善，gain_C 差≤10 个百分点；明显正/负收益的方向须一致。近零收益单列不确定性，不改用户的正收益采用规则 |

逐 Token 接收停顿、5 s 速率、队列/KV slope、状态年龄和回执延迟另做分布/曲线诊断，避免只拟合整轮 E2E 却生成错误 PPO 反馈。缺时间线时不能把时间奖励校准标为 passed。新 reward 的总 return 还含塑形与边界项，不能用“return 更高”代替第 5 节的真实主成本选择。

至少用一个已观察到的失败候选或历史 Q15-A 作负收益诊断；只读复用已有证据即可，不强制新增负例真实轮次。若仿真把所有迁移都算成正收益，必须报告缺少的成本/干扰，不能宣布通用校准完成。

分开记录：`selection_passed`（当前同轨迹有收益）、`physical_fit_passed`、`interface_reward_passed`、`closed_loop_passed`、`sim_pretraining_ready` 和 `sim_evaluation_ready`。仅当当前环境必要项均有证据、覆盖范围明确时才设对应 ready，并附 `validation_scope=independent_trace`、对应冻结 hash 和实际验证状态；若仅有 fit 重建则只能 `readiness_scope=development_only`，不能写独立验收通过；物理拟合未过不撤销真实配对结果，而是交付“已选但拟合待补”。原生场景用 `sim_evaluation_ready` 表达可测试，不启用 `sim_pretraining_ready`，不能强行把两种角色合并。

## 10. 入口、交付文件与监督 Agent 最终回答

优先扩展现有 **仿真入口 `benchmark/run_ppo_sim.py`、真实入口 `benchmark/run_ppo_real.py`**，新增明确的环境配置/阶段参数；原有单 Q15-B 流程保持显式兼容。不再制造若干含硬编码目录/seed 的平行运行器。现有入口还不具备本任务完整多环境筛选/原生路由能力，不要未经实现就直接调用旧 prepare/run 全流程。

必须有实际可执行的 CPU plan/validate、真实受管 collect/screen、CPU extract/fit/validate/report 能力。上面是所需能力，不是声称这些子命令已经存在。完成后提供经过真实 `--help` 核对的命令与参数，plan/report/fit 不启动 GPU。真实运行不能绕过所有权、freeze hash、日志审计和 watchdog。

创建独立目录，例如：

```text
benchmark/results/mig_ppo_v2_multienv_<UTC>/
  plan.json / protocol.md / budget.json
  environment_matrix.json              # 13 行全部保留，含候选、角色、状态
  environment_audit.json
  data_split.json / source_index.json
  reuse_audit.json                      # 旧物理模型、状态/奖励和每个环境分别判定
  base_traces/ / traces/                # 明确外生 hash 与带路由 hash
  runtime_config.yaml / real_llumnix.yaml
  runs/ / attempts.jsonl                # 包含失败尝试，append-only
  comparisons/                         # 首选/备选的同轨迹评估，含跳过原因
  selection_summary.json               # 所有环境评估结束后统一封存
  validation_runs/                     # 模型冻结后新 seed 的真实独立配对
  calibration/<environment_id>/        # 模型/共享模型引用、覆盖、误差、来源
  sim_validation/<environment_id>/     # 回放、闭环、PPO 合约分别交付
  model_registry.json                  # 每环境模型/配置/数据/schema/hash/适用域
  reward_frozen.json / normalizer_frozen.json
  reward_reconstruction.json
  results.csv / report.md
  cleanup_verification.json
  handoff_sim_ppo.json / reproduce.md / next_step.md
```

`handoff_sim_ppo.json` 至少包含：13 场景状态，实际采用参数，accepted positive-gain 环境列表，NATIVE 的测试角色，旧 Q15-B alias/重拟合情况，逐环境模型和证据路径/哈希，筛选/拟合与独立验证分组、先评估后统一拟合的阶段记录，schema 和输入形状，采样/冷却/GAE/奖励参数，实际容量/映射规则，原生分发配置，支持域/未覆盖方向，未完成项、模拟加载/rollout 命令及后续 CPU 训练所需依赖。只封装环境，不自动开始训练。

监督 Agent 最终用中文给出 **13 行完整表**：场景、热点位置/实际占比、路由模式、实际 QPS、Stay/Balanced 成本与 gain_C、首选/备选及跳过状态、采用状态、迁移数/不同请求数、复用或重拟合、独立验证/仿真验收状态、证据路径。没有运行的行写 not_run，预算耗尽写 incomplete，原生无收益如实写 rejected_for_positive_gain 或 test_only，不用全局 passed 掩盖缺口。

最后回答：旧拟合哪些仍有效；哪些新采/重拟合；哪些环境已可供仿真 PPO；原生路由是否真正闭环验证；实际 CPU/GPU 墙钟和 GPU-hours；是否还有本任务残留服务；下一条经过核对的命令。禁止把任务书、代码已实现、单次正收益、接口 smoke 或拟合误差合格互相代替。
