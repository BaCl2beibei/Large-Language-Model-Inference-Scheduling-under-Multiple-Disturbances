# 完整请求完成下的平均与长尾时延实验：Balance vs Stay v2

版本日期：2026-10-29（UTC）。当前状态：**implementation_updated / calibration_in_progress / candidate_search_pending**。

本文件依据用户最新要求重新设计实验：优化平均及长尾请求时延；停止到达后等待所有请求完成，再计算指标；允许搜索有迁移收益的场景。设计定稿后，用户明确要求实现并进行真实 GPU 搜索；后续实测记录见本文第12节。实验没有请求完成或实验总墙钟时间上限。只把完整、合规的成对结果视为探索证据，不从无效或未完成运行推断策略胜负。

本设计取代后续研究中旧主任务书第 2、5、7、8、9 节涉及固定评分截止、常规取消排空、条件时延和固定机会矩阵的规定。旧方案、06/07 交接和既有 gate 保留为历史记录；它们不证明本设计已经实现或验收。本设计不是旧 `run_real_opportunity.py` 可以直接执行的配置。

相关交付位于 `rl_scheduler/docs/full_completion_balance_v2/`：`experiment_spec.yaml`、`candidate_matrix.json`、`prior_diagnostics_session_snapshot.json`、`design_manifest.json`。

证据可用性说明：本轮先成功读取并CPU分析了原 run `benchmark/results/real_ppo_pilot_20261022T145922Z_04f200/` 的06/07交接和请求记录。随后原run目录及active_run指针在工作区消失，搜索未找到，本轮没有执行删除。附件保存工具返回的统计摘要，明确标为session snapshot；原始文件目前不可重新核验，不编造其hash或补造历史gate。该情况不影响设计交付，但实现/实测前需要重新确认源码、运行配置和可用历史证据。

## 1. 07 究竟说明了什么

07 完成的是若干 Stay 探测，**配对回合为 0**，没有 Balance 相对 Stay 的性能证据。其 incomplete 不能解释为迁移方法无效，也不能解释为 Balance 已优于 Stay。

本次在原目录消失前从导出记录重新统计，以下是当时可见结果，不把审计失败行当有效实验：

| recovery6 探测 | 计划 / 已发 | 旧期限内成功 | 导出记录中的最终成功 | 说明 |
|---|---:|---:|---:|---|
| A/qps3 | 99 / 89 | 88 | 89 | 有 1 个迟到成功；另 10 个未发送 |
| B/qps3 | 99 / 55 | 52 | 52 | 另 3 个已发请求被人工取消；44 个未发送 |
| A/qps6 | 185 / 121 | 83 | 121 | 38 个成功被旧期限截掉；64 个未发送 |
| B/qps6 | 185 / 161 | 52 | 64 | 审计失败；导出中另有 97 个已发请求无终态，只作诊断 |

这些数据不能填造缺失请求的完成时延，不能将 A/qps6 的最终成功数换回旧固定窗口指标，不能拿不同探测比较策略。旧阶段的 passed 是其旧口径的审计状态，并非新完整队列实验合格。

B/qps3 的 35 个快照与决策逐项对齐后，34 个有合法非 Stay mask，现有 `balanced_action(threshold=9000)` 在 32 个快照上选择非 Stay。**这是历史 Stay 状态上的动作重放**，不包含迁移后的状态变化、冷却或成本，绝不能把 32 当作可成功迁移次数或速度收益。

另一个需要测量的机制是阈值：记录中的每实例 total_blocks=31612。对于非空目标，`F=(free_blocks-waiting_blocks)/(running+waiting)`；当目标请求数≥4，即使空余全部 blocks，也有 F≤7903<9000，不能成为旧 Balanced 的目标。这解释了为什么“增大 QPS”可能反而关闭迁移机会。但 B/qps3 已有大量候选，因此不应先改阈值再宣称找到原策略的收益。

发送端也必须处理：旧 `_issue()` 到 arrival_end 会直接返回，`stop_arrivals()` 会取消未发 task；`OpportunityEpisodeIO` 的 HTTP total timeout 为 360 秒。只把主循环排空时间延长不能满足新要求。事件重放、每次 poll 的 fsync、sender 线程的 GIL/CPU 竞争与 API 排队需要区分；07 的日志不能单独证明哪一个是唯一瓶颈。

## 2. 研究问题与不可混淆的结论

先回答：在同一模型、硬件、请求集合、到达过程和初始路由下，**仅增加运行中迁移**，现有受约束 Balanced 能否同时改善全体请求的平均 E2E 与 P99 E2E？在哪些负载/请求异质性条件下有效，哪些条件下无效或有害？

按三层结论交付：

1. 机制成立：有真实迁移成功，并观察到负载重新分配；不等于尾时延下降。
2. 探索场景成立：开发种子上的完整配对有平均/尾部收益；不能作为独立验证。
3. 独立验证成立：事前冻结的场景和策略在新种子上重复显示收益，并报告不确定性及负对照。场景经过搜索，应称“搜索所得的迁移有利场景”，不能泛称所有真实业务都受益。

Balance 不是 PPO。证明 Balance 在某些场景优于 Stay 只证明可利用的调度空间，不证明 PPO 能学到它，更不证明 PPO 优于 Balance。

## 3. 新终止语义：有限请求集合，完成驱动结束

一次回合预先冻结 N 个请求 ID、prompt/token 长度、输出规则、计划到达时间、初始路由/外生扰动和随机种子。

状态机：

`idle → warmup_complete → admitting_frozen_cohort → finishing_cohort → reconciling → complete`

- 到达窗口 T_arrival 只决定何时生成需求，不是评分截止，也不是终止请求的理由。
- 到达窗口之后不增加新的计划请求；因发生器延迟而尚未发送的既定请求仍属于本次 cohort，必须发送并保留迟发时间。按用户现行要求，send lag 无论多高都只记录、不作为排除门槛；不得删除迟发请求换取更好分数。
- finishing 阶段保持同一策略：Stay 继续 Stay，Balanced 继续合法迁移。不能让 Balanced 在到达结束后失去控制权，也不能只对一方禁用排空阶段动作。
- 对一次有效时延实验，冻结集合中的 N 个请求必须恰好发送一次、被接受一次、正常完成一次，输出完整；最后一个客户端终态到达后才计算全体指标。
- 流自然 EOF、HTTP 200、引擎空队列不能单独证明完整完成。还需客户端明确成功终态、请求 ID/输出计数一致、所有迁移确定终态、投递归零、生产者 checkpoint/序列审计及 KV 基线恢复。
- 分开记录 `last_token`、`client_terminal`、`data_complete`、`audit_complete`、`service_stopped`。审计耗时和停服耗时不加入请求延迟或 cohort makespan。
- 为控制高频采集开销，客户端与引擎输出事件固定保留首个正 token、每 64 token 的累计计数和最终累计计数；服务端 terminal 另行核验精确输出长度。首 token、最终 token 和终态时间戳保留，不能据这些采样事件声称逐 token ITL。
- 每个引擎迭代对应的 batch/executor/GPU 诊断按进程和事件类型保留首条及每第 256 条；它们只作抽样诊断，不参与完整性审计、token 计数或主指标。实验进程关闭 core dump，避免服务退出时把大型 core 文件写入实例管理数据区。
- 不设 arrival+60/120/360 秒完成截断，不因“超时未完成”常规 abort，不用取消代替排空。自然排空阶段没有固定时长上限。

故障例外：进程死亡、断链且不能恢复同一请求流、输出计数损坏、未知迁移、磁盘错误等是真实失败。保存全体请求的当前状态、失败原因和已知时间；整次标为 `service_failed` / `measurement_incomplete`，平均及 P99 主结果为 null，不能删掉失败请求后声称完整 cohort 获胜。已完成子集可单独标为“条件诊断”，不进入策略胜负。

长时间无进展只触发诊断告警，不以固定秒数自动取消慢请求。明确不可恢复的故障或用户中止才进入 owned-service 故障清理；此时不是正常完成。控制 RPC 超时/租约 TTL 是控制安全机制，与请求生存期不同：未知动作停止新增迁移、继续收集请求，回合控制合规性单独标记。独立心跳不依赖大日志扫描。

## 4. 完整、可比较的时延口径

每个请求 i 记录客户端同一 monotonic 时钟上的：计划释放 a_i，HTTP 调用开始 s_i（连接池等待包含在其后），首 token f_i，最后 token l_i，成功终态 c_i。

- **主 E2E：L_i=c_i-s_i**。包含传输、接入、排队、生成及终态传递，不用 server accept 作起点。
- **需求相对 E2E：L_i^schedule=c_i-a_i**。和 send lag=s_i-a_i 同时报告，避免发生器滞后把服务压力隐藏到统计之外。
- TTFT=f_i-s_i；decode span=l_i-f_i；TPOT=(l_i-f_i)/(实际输出 token 数−1)。单 token 的 TPOT 为 N/A。
- 独立报告终态收尾 c_i-l_i；实际 token/chunk 跨度可与 terminal E2E 不同。没有精确逐 token 时间戳时，只报告 chunk gap，不能伪称 token P99 ITL。
- 每条 trace 报告 L 的 mean、median、P90、P95、P99、max 和 CVaR95；同时报告 TTFT/TPOT 同类统计、最后 1%/5% 请求清单与 ECDF/CCDF。
- 所有分位数使用同一预注册算法：线性插值 Type 7，h=(N−1)p。CVaR95 使用经验分布的 `min_eta [eta + sum(max(L_i−eta,0))/(0.05N)]`，边界分数权重精确处理，不任意凑整尾部数量。
- 固定全体的成功 SLO 比例可以保留；cohort throughput=N/(max c_i−min a_i)，token throughput 同理。这些是完整 cohort 指标，不能与旧 240 秒 goodput 混称。
- 固定时间点的 backlog/完成数只作为曲线诊断。慢请求稍后完成仍进入最终主指标，不再设置“超过旧期限则永久排除”。

主研究目标是 **mean 与 P99 同时下降**。P95/CVaR95 用于小规模搜索阶段较平滑的排序，不可在看到验证失败后将 P99 偷换成获胜的分位数。

统计样本单位是完整 paired trace/seed。请求共享排队和迁移影响，不按单请求或 token 做独立 bootstrap；不能平均若干 P99 后称为“合并样本的 P99”。报告各 seed P99、其配对比值，以及另行标注的 pooled 分布。

## 5. 比较的策略必须明确

首轮保持目前实现：

- `Stay-control-v1`：无迁移，但保留与 Balanced 相同的观测、控制租约、客户端和事件采集成本。
- `Balanced-9000-SR-v1`：直接复用现有 `real_opportunity.balanced_action`；阈值 9000，原始 SR 候选、一次全局迁移、终态后 5 秒冷却、2 秒状态新鲜度、原容量 reserve。

不把“更激进的新 heuristic”悄悄叫作旧 Balanced。以下只在独立诊断支线中使用：

- 若存在负载差但阈值排除所有目标，预注册 `Balanced-calibrated-SR-v2`，在开发集选 threshold，再冻结独立验证。原 9000 版本必须保留对照。
- 若迁移速度瓶颈是 5 秒冷却，可单独做 {1,2,5} 秒敏感性实验；这是新的控制版本，不与原版本结果拼接。没有放宽容量、幂等、unknown 保护的空间。
- SR 是当前序列长度候选，不是“真实剩余时长最短”的 oracle。离线已知的输出长度、未来扰动和未来尾部身份不得输入 Balanced/PPO 观测。

两策略共享冻结的路由规则。synthetic 固定路由用于隔离迁移因果作用，必须明确标注；之后另在相同 native dispatcher 下验证鲁棒性。native dispatcher 的实际逐请求目标可随策略状态改变，这是共享路由算法的闭环总效应，不能要求两个策略的实际目标仍完全相同，也不能与固定路由的直接迁移效应混淆。

## 6. 优先搜索什么场景

迁移可能有效的条件是：局部拥塞、其他实例有可用能力、请求尚有足够工作、迁移成本小于缓解拥塞带来的节省。四卡同时重载、没有可接收目标或请求将立即完成时，迁移可能无效或更差。

推荐首先运行 **短输入、长输出、持续到达倾斜**，然后测试 **请求数相等但长请求集中**。前者检验迁移机制，后者更接近“请求计数均衡但工作量不均衡”的问题。不能只制造一个毫无意义的极端 Stay 基线，然后声称普适提升。

| 场景 | 构造 | 要检验的机制 | 负面可能性 |
|---|---|---|---|
| S1 decode 倾斜 | 输入128，输出1536；热点路由份额0.70或0.90；其余均分 | 初始 KV 较小、剩余生成较长，迁移后减轻热点 decode 干扰 | 大批次本来效率高，搬走请求未必降低群体延迟 |
| S2 工作量倾斜 | 每实例请求数严格相等；输入128；输出80%/90%为128，20%/10%为1536；尽可能把80%的长请求放到一个实例 | 计数均衡掩盖长请求集中，短请求结束后出现空闲接收端 | SR 未选中真正影响尾部的请求；长请求的 KV 已变大 |
| S3 迁移热点 | 输入128，输出512；到达中段出现热点/突发，随后热点转向另一实例 | 静态路由无法处理既有存量与热点变化 | 冷却过长、迁移落后于热点、产生往返搬迁 |
| C0 对称负对照 | 均匀固定路由，同长度、同负载 | 没有不均衡时不应凭迁移总能获利 | 迁移/控制开销使结果更差，应照实报告 |
| C1 小收益窗口 | 输入128，输出64，热点0.90 | 请求很快结束，迁移难摊销成本 | 预期无收益或负收益，不能事后删除 |

当前 max_model_len=2112，以上 token 组合都低于限制；prompt **实测 tokenizer 长度**必须计入模板/special tokens。主实验使用冻结的目标输出长度和 ignore_eos/min/max token 控制能力，经实际计数验证；若接口不能强制精确长度，先修复或另立自然输出协议，不能把 max_tokens 当实际长度。两策略输出 token 总量或逐请求长度不一致时不能把少生成当调度加速。

S2 路由生成：先冻结请求数四等分，再用受约束排列使热点承接约80%的长请求，不改变每实例请求总数；不可行的配额提前报错。其他实例分配剩余长请求。热点物理卡在 seed 间轮换，避免 GPU/主机亲和性偏差。

S3 使用总请求数固定的分段强度：前25%和后25%时间倍率1，中间50%倍率 b∈{2,4}；整体归一化保持平均 λ。前半段热点为 h，后半段为 (h+1)%4，热点份额0.80，其他均分。扰动时间和路由在运行前冻结，不由哪个策略更慢来触发。

S4（后续泛化支线，不属于首轮16候选）：native dispatcher 下混合输入1536/输出128的 prefill 请求与输入128/输出1536的 decode 请求，研究资源竞争；若要增加算力/带宽扰动，先独立测量并版本化，不能临时启动未知 GPU 争用程序。

## 7. 从机制筛查到独立验证的搜索程序

### 7.1 发请求与测量能力先过关

先运行只用 CPU 的慢流/迟发/大日志回放测试，再进行小规模真实完整完成试验，确认旧 arrival cutoff、360 秒 HTTP 截断和 routine cancellation 全部退出新路径。

主 sender 应独立进程或足够隔离的事件循环，避免日志排序/fsync阻塞到达；预生成/tokenize请求，journal顺序写且检查点异步持久化，绝不靠丢必需事件减负。完整生命周期/输出累计计数必留；高频非必需诊断可分层记录并固定两策略的采样设置。sender 的迟发数据按两策略相同口径记录，不作为停止或排除条件。

旧版曾建议 send lag P99≤100ms 且 max≤500ms 作为发生器质量判据。用户现已明确要求取消所有时延门槛：从本次修订起，无论 send lag 多高，完整 cohort 都继续进入 calibration 和 candidate 搜索，并在结果中报告 send lag。仍须 100% 发出并完成计划 cohort、精确输出长度匹配、route/event/cleanup 审计通过；不删除迟发请求，也不因时延高而过滤运行。

07 的 flush/cleanup失败必须以流未结束、输出投递、生产者 fsync、事件游标、RPC等待分别定位；旧报错本身不足以宣称增加 timeout 就修好了。请求完成等待与审计等待分离；审计阻塞告警保留原始数据，不强行输出 passed。

### 7.2 用本机实测尺度选负载，不固定 QPS 3/6/9

在固定模型/四卡/TP1下，对各 profile 先测：单请求服务时间、并发 C={1,4,16,32} 时的步时与完整请求时延、一个迁移在不同已生成长度下的停顿/总耗时。都是有限请求、全部完成，迁移参数测量使用原 coordinator。

为每个 profile 用均匀固定路由、Stay、N=256 的同步释放批次，独立做3次完整排空；令 `lambda_ref=median(256/(last_client_terminal-first_scheduled))`。它只是本机有限批次的吞吐参考，不是理论稳定容量或各 QPS 的健康证书。再用 λ=ρ·lambda_ref 构造开环有限 cohort；记录实际 λ，跨 profile 不共用未经测量的参考。

直觉上若单卡服务能力约 μ，则局部热点 pλ>μ 而总 λ<4μ 时可能出现可转移负载。但 continuous batching/长度/内存令 μ 随负载变化，该不等式只指导候选，不是验证或仿真结果。

### 7.3 初始16个候选与逐级增加样本

机器可读矩阵见 `candidate_matrix.json`：S1 六组（ρ={0.35,0.55,0.75}×热点={0.70,0.90}）；S2 四组（ρ={0.55,0.75}×长请求比例={0.10,0.20}）；S3 四组（ρ={0.55,0.75}×突发倍率={2,4}）；C0、C1各一组。λ_ref 未测前候选只含符号负载，不编造 QPS。

1. **粗筛**：每候选 N=128、2个开发 seed、Stay/Balanced 配对，共64回合。先做S1，接着S2/S3，负对照必保留。每个回合必须完整结束；不能在发现一方暂时落后时提前终止该回合。
2. **确认**：最多选3个非支配候选（mean/P95/CVaR95联合考虑），每候选 N=2048、5个新开发 seed。样本量增加改变负载持续时间，必须在此阶段重新确认效果；小样本获胜不自动晋级为研究结论。
3. **冻结**：选出一个主场景，固定所有参数、策略版本、token规则、N=2048、seed生成规则、比较顺序、指标与分析脚本 hash。保留筛掉/失败/负收益的全部候选及原因。
4. **独立验证**：主场景、C0、主场景的ρ×0.8和ρ×1.2邻域，各10个未查看 seed，N=2048，两个策略。邻域不根据验证表现再挑；C0使用与主场景相同profile、基准ρ和到达强度曲线，仅将路由/长请求分配改为均匀，保留全局突发。C1开发负对照结果也保留。

粗筛排序：任一预定pair控制/数据无效的候选标为 feasibility_failed，不能用剩余pair的均值替代完整结果；有效候选按 `min(median(1-mean_B/mean_S), median(1-P95_B/P95_S))` 降序，CVaR95改善作为首个平局决胜，candidate_id作为最终决胜。若只有P95改善而mean显著恶化，标为权衡，不能称“双改善”。无效回合不得当作最差性能点喂给代理模型；另建 feasibility 标签并保留失败率。

初始网格后没有阳性结果时，按预声明诊断扩展开发搜索：无目标→调低全局负载/增强偏斜；有合法mask但无动作→检查阈值/配对谓词；动作成功但无收益→增加可摊销的剩余工作、减小prompt，或接受负结果；迁移太少→单列控制参数敏感性。每次扩展生成新 spec版本并保留原失败，不能调整 holdout。允许用随机搜索、Sobol/贝叶斯优化做后续提案，所有代理模型仅辅助选下一个真实实验，不提供性能结论。

**优先推荐机制引导的逐级筛选，而非先训练 PPO 或大规模盲扫。** 参数维度少、单次真实实验昂贵且可能失败；先确认候选能迁移且可能受益，信息量更高。没有任何算法能保证既定硬件/策略必然存在正收益场景；找不到时交付有界参数区域内的负结果，而不是强行造出获胜。

这不是时长上限：64回合、候选数、seed数是实验设计与统计样本量。没有总墙钟或请求等待秒数限制。搜索阶段可按新增设计继续扩展；每次完整执行、保留结果，不自动无限重复相同失败。

## 8. 配对、公平性与独立验证判据

每一 paired seed 使用完全相同的请求ID集合、token负载、计划到达、固定路由、prompt和外生事件；只切换迁移策略。采用预先平衡的 AB/BA 顺序。每个回合从相同空闲KV基线开始，warmup另命名且全部完成。prefix cache、GPU温度/频率、CPU使用、模型常驻、telemetry配置和服务重启都记录；必要的重启以整个pair为块，失败尝试保留。

隔离发送与收集资源；记录host CPU与实际路由，防止把负载发生器变慢当作性能提升。相同实际 token 工作量是有效性条件之一。不同策略不能分别选最有利初始mapping、参数或服务状态。

有效完整pair对指标 m 定义 `g_m=1-m_B/m_S`，同时报告秒数差。独立验证的主场景预注册成功标准：

- 全部预定10个pair完成、审计有效，所有请求正常结束，无常规取消，无未来信息输入；若出现故障不声称已完成预定验证，不悄悄剔除seed。
- mean E2E 和 P99 E2E 的配对几何平均比值均≤0.95（各至少5%的点估计改善）。
- 以完整paired seed为重采样块，固定10000次bootstrap，计算 log(B/S) 的均值；两个主指标分别使用单侧97.5%上界，经Bonferroni得到同时95%的保守判据，两个比值上界均<1。
- schedule-relative E2E不出现相反方向的点估计；send lag 始终报告但不设时延门槛；C0及邻域结果完整公布，区分主场景成功与跨场景稳健性。

N=2048 时顶端1%约20个请求，仍不是精确P99的保证；10个seed的区间若很宽，报告证据不足，不只增加seed直到p值过线。若需要追加，形成新的固定样本量验证批次，原批次完整保留；或事前制定有效序贯方法。

单独报告 max/最差请求、CVaR95、各请求长度组的时延、迁移与非迁移请求分布；“迁移请求更快”不是随机化因果证据，因为迁移对象被选择过。全cohort配对是主因果比较。

主场景的5%是新的实际效应目标，与旧goodput提升10%没有换算关系。不能因结果不佳再降低；若只证明tail改善但mean损失，明确报告Pareto权衡。

## 9. 与后续 PPO 目标的衔接

完整排空后，`∫ N_outstanding(t) dt = Σ_i (c_i-s_i)`，因此全回合驻留积分直接对应完整 cohort 的总 E2E；积分不能在旧 deadline截断。建议首先保留这一准确的平均时延分量。

训练用较平滑的尾部风险代理：`J = mean(L)/T_ref + beta * CVaR95(L)/T_ref`，初始 beta=1，T_ref由开发Stay数据冻结。CVaR是训练代理，不等于P99；独立评价仍检验mean/P99。不再用固定截止时“每个未完成请求罚常数”代替长尾损失。

可实现为每步 `-∫outstanding dt/(N*T_ref)`，全请求完成时加 `-beta*CVaR95(L)/T_ref`；原始与归一化分量都记录。最初不额外加入可淹没时延的完成奖励；迁移成本已进入真实延迟，若额外惩罚迁移次数需作为单独正则实验说明。以 gamma=1 的有限完整回合作为这一目标的直接实现；使用 gamma<1 会折扣较晚的尾部代价，不能再声称与上式严格等价。

收集期间策略与normalizer冻结；变长回合的dt按实际时间记录；只有全部完成才自然terminal且末端价值0。训练批次边界不是请求终点，不能取消所有请求来凑固定rollout长度。异常中断不当作正常零价值terminal；如要利用截断轨迹须单独实现正确bootstrap和缺失终态处理。

本次不训练PPO，也不冻结未经测量的T_ref或宣称上述权重最优。

## 10. 实现交接：哪些地方必须变，哪些已可复用

以下均为待实现清单，不是本次已做的业务改动。

| 部分 | 读取时的实现落点（实施前重新核实） | 新要求 |
|---|---|---|
| 状态机 | `real_env.py: RealEnvConfig/step/finish` | 新建或显式版本化 full_completion模式；去固定score_deadline/常规取消，全部客户端终态后再finalize |
| sender | `real_opportunity_runner.py: OpportunityEpisodeIO._issue/stop_arrivals/_open_sender_session` | 有限cohort任务队列；不跳过迟发；HTTP total/sock_read不以固定时长截慢请求；连接建立失败仍按故障处理 |
| 台账 | `real_metrics.py: RequestLedger.freeze_score/export` | 新全cohort聚合入口，required IDs固定，核验集合与全终态；不得用 infinity deadline骗过旧评分逻辑 |
| 统计 | `real_opportunity.py: metric_summary/healthy/paired_gate` | 实现mean/P99/CVaR/完整cohort及配对分析；删除旧99%期限健康筛选与240秒goodput主gate |
| 控制 | `real_bridge.py`、`real_control.py` | 复用原coordinator与保护机制；finish全程心跳；unknown不能重发新动作 |
| 采集 | `real_opportunity_runner.py: poll/heartbeat` | 增量读取/索引，sender与高频日志处理隔离；完整计数不丢，审计终态与性能终态分开 |
| CLI | `benchmark/run_real_opportunity.py` | 新schema独立验证；不得沿用旧脚本声称支持；新source/config/trace/analysis绑定 |
| 资源计时 | `real_opportunity_budget_override.py`、service/watchdog | 不继承已关闭invocation lease，不篡改旧时钟；新的协议只累计计时、进度与成本，不设置总实验/请求完成墙钟截止；保留服务所有权与故障清理 |
| 输出 | 新versioned experiment目录 | 写request表、配对表、ECDF/CCDF、置信区间、完整候选账本和失败记录；旧07证据不覆盖 |

必须先有的CPU验证：超过旧90/240/360秒后成功仍计入；最后一个极慢请求决定P99/mean；计划窗口结束后的迟发/迟接受仍入同一cohort；人为取消/无终态不能生成完整统计；全部完成才结束且仍等待投递审计；输出长度不等、重复ID、token回退拒绝；CVaR分数权重/Type7手算；pair缺失/seed重复/策略间负载变化拒绝；客户端延迟不会从schedule-relative指标中消失；变长尾部奖励积分恒等式。

**不接受的快捷方式**：只把120秒改成900秒；只对完成请求计算P99；把max_tokens当实际输出；只减少指标分母；删租约/幂等/flush检查；拿同一批search seeds再宣称独立验证；看完结果后只留下Balance赢的场景。

## 11. 文献依据及适用边界

Llumnix原论文将异质请求、排队与运行中迁移作为核心问题，并报告平均与P99请求/阶段时延；它的freeness指标及源/目标阈值也支持先检查迁移触发机制。本文只是借鉴研究问题与指标，不能把原文不同模型、GPU规模和传输条件下的收益搬到当前四卡Qwen2-0.5B环境。[OSDI 2024原论文](https://www.usenix.org/system/files/osdi24-sun-biao.pdf)

vLLM现行基准文档区分request-rate/burstiness/concurrency，并说明客户端排队与schedule-relative E2E的统计。这支持同时记录计划释放、发送与完成时间。该网页对应当前版本，不代表本仓库的旧vLLM已具备相同CLI或字段；本设计要求在现有collector中明确实现。[vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)

上面的候选矩阵、搜索顺序、完整cohort语义及统计门槛是本项目的新设计，不是上述论文已经验证的结论。

## 12. 2026-10-29 实测进度

| 运行 | 状态 | 证据 |
|---|---|---|
| `benchmark/results/full_completion_balance_v2_ray_screen_20261029T093554Z/` | `candidate_search_in_progress` | manifest 确认 send-lag gate 关闭；4 sender loops；keep-alive 3s；控制 RPC 无总时长超时；只读 snapshot 最多三次传输重试；16 个候选。5 个 profile 的 Stay 校准均 3/3 完整通过（N=256、cohort/route/event audit passed）；`lambda_ref` 为 decode1536 `1.205279`、bimodal_0.1 `4.384736`、bimodal_0.2 `4.038420`、decode512 `7.081839`、decode64 `33.148785` rps。decode1536 三次基线 client/schedule E2E mean/P99 为 `142,598.899/184,793.755`、`142,755.711/184,876.353` ms；`162,895.595/224,001.834`、`163,022.004/224,069.879` ms；`158,306.388/212,321.621`、`158,423.726/212,391.860` ms。全程未按时延筛除。S1_01 两个 seed 的 Stay/Balanced 配对均完整通过 cohort/route/event/cleanup 审计（每回合128/128、196,608 tokens）。seed 82001：Stay/Balanced client E2E mean `56,100.427/100,651.382` ms、P99 `67,867.062/118,222.821` ms，Balance mean/P99 gain `-79.41%/-74.20%`，吞吐 `0.3273/0.3649` rps，22 次迁移尝试。seed 82002：Stay/Balanced mean `104,990.879/98,013.155` ms、P99 `117,003.055/106,833.386` ms，Balance mean/P99 gain `+6.65%/+8.69%`，吞吐 `0.3226/0.3150` rps，27 次迁移尝试。按双 seed 一致性规则，candidate summary `exploratory_win_both_metrics=false`，中位 mean/P99 gain `-36.38%/-32.75%`，不算胜出。S1_02 seed 82001 配对完整通过审计，Balance 首个 seed 暂现改善：Stay/Balanced client E2E mean `118,865.523/104,293.215` ms、P99 `133,518.637/122,739.684` ms，mean/P99 gain `+12.26%/+8.07%`，吞吐 `0.3254/0.3160` rps，Balance 有40次迁移尝试。第二个 seed 尚未配对完成；其 Stay 回合正在运行，完成后仍需 Balance 回合验证，因此不能把该候选记为胜出。全程无时延门槛。归档监控此前两次因 flush 回执层级解析不全跳过，原始全局 causal 流未截断；解析已按实际 Llumlet/worker 回执结构修正，等待下个 episode 边界重试。当前双 seed 一致胜出场景数仍为0。 |
| `benchmark/results/full_completion_balance_v2_ray_smoke_20261029T073751Z/` | `passed` | Ray queue warmup 与 32×256 queue probe 完整结束；精确输出 8,192 tokens；probe send-lag P99 21.73ms、max 21.86ms；owned-service cleanup 通过。它只验证测量/队列通路，不是策略比较。 |
| `20261029T061900Z` | `measurement_incomplete`；原始目录已清理 | N=256 校准已发出 256 个请求，但输出转发器在最大 174 token 时失联（ZMQ 报 `Socket operation on non-socket`）；客户端/服务端终态均为 0，控制租约到期，不能计入策略排名。owned-service cleanup 通过。 |
| `20261029T062729Z` / `063340Z` | 两次 Ray smoke 未完成；原始目录已清理 | 都未进入策略比较；前者控制 RPC 状态不确定，后者另暴露 `RayQueueServer.qsize` 不存在。队列实现与探测随后修正。 |
| `20261029T092227Z` | `control_invalid`；原始目录待清理 | 首个 calibration 通过（N=256、393,216 tokens、审计通过，send-lag P99 307.581ms）。第二个 repeat 也完成全部 256/256、393,216 tokens 并通过 cohort/event audit；send-lag P99 184.969ms、max 190.717ms，schedule-E2E mean 151,241.364ms、P99 202,340.774ms。但一次 15 秒控制 RPC 总超时将 episode 标为 control-invalid，无候选。新修订移除控制 RPC 总超时，并对只读 snapshot 传输错误最多重试三次。service cleanup passed；随后复核确认无残留进程。 |
| `20261029T091340Z` | `control_transport_failed`；原始目录待清理 | 无时延门槛且 heartbeat 重试已生效。首个 N=256、decode1536 Stay calibration 完整通过：256/256、393,216 tokens、route/event audit 通过；schedule-E2E mean 138,183.722ms、P99 175,707.602ms，send-lag P99 352.543ms、max 355.027ms，`lambda_ref` repeat=1.45695 rps。下一 calibration 开始前发生 `ServerDisconnectedError`，无候选比较；结合断连发生在 episode 间且服务端 cleanup 正常，推测是客户端复用了服务端已关闭的空闲 HTTP 连接。新一轮将让客户端 keep-alive 提前过期。 |
| `20261029T090407Z` | `control_invalid`；原始目录已清理 | 无时延门槛的首轮：N=256、decode1536 Stay 全部完成，393,216 tokens、route/cohort/event audit 通过；schedule-E2E mean 129,291.868ms、P99 158,775.564ms，send-lag P99 338.891ms、max 343.113ms。episode 因两次短暂 heartbeat `ServerDisconnectedError` 标为 control-invalid，没有 candidate；随后 release/final snapshot 均返回 200，service cleanup passed，且未发现遗留进程。heartbeat 改为在租约到期前重试传输错误。 |
| `20261029T064259Z` | 完整校准；原始目录按用户要求清理 | N=256、decode1536 Stay：256/256 终态、393,216 tokens，route/event audit 通过；send-lag 均值 130.608ms、P99 246.170ms、max 247.214ms。旧版因时延门槛停止，没有 candidate。 |
| `20261029T0744Z` | 完整校准；原始目录按用户要求清理 | N=256、decode1536 Stay：256/256 终态、393,216 tokens，完整审计通过；send-lag P99 171.182ms、max 172.265ms。旧版因时延门槛停止，没有 candidate。 |
| `20261029T0800Z` | `measurement_incomplete`；原始目录按用户要求清理 | 256/256 请求发出，但客户端输出停在 128–320 token、终态为 0；服务端计数推进到 1,152 token。约 5 分钟无客户端进展；取消未确认，随后停止自有进程并释放 GPU。无完整 cohort、无 candidate。 |
| `20261029T0820Z` | 未启动实验；原始目录按用户要求清理 | 启动包装脚本误放在输出目录，runner 未启动，也未占用 GPU；无实验数据。 |
| `20261029T0821Z` | `control_invalid`；原始目录按用户要求清理 | warmup 4/4、精确 32 tokens；曾因 server ID 前缀导致初始路由审计失败，没有 calibration 或 candidate。前缀已恢复。 |
| `20261029T0827Z` | 完整校准；原始目录按用户要求清理 | N=256、decode1536 Stay：256/256 终态、393,216 tokens，完整审计通过；send-lag P99 314.805ms、max 317.580ms。旧版因时延门槛停止，没有 candidate。|
| `20261029T0840Z` | 完整校准；原始目录按用户要求清理 | N=256、decode1536 Stay：256/256 终态、393,216 tokens，完整审计通过；send-lag P99 237.737ms、max 239.291ms。旧版因时延门槛停止，没有 candidate。|
| `20261029T0852Z` | 完整校准；原始目录按用户要求清理 | 四个 sender loop 版本；warmup 4/4 通过。N=256、decode1536 Stay：256/256 终态、393,216 tokens，完整审计通过；send-lag 均值 119.345ms、P99 303.662ms、max 319.379ms。该进程仍载入旧门槛代码，因而没有 candidate。owned-service cleanup 通过。|

第二条记录发现，同时释放的 256 个请求经同一个同步客户端事件日志锁写入；collector 的 checkpoint/fsync 会与发送协程竞争。客户端事件写入已改为有序后台队列，checkpoint 作为队列 barrier，仍等待 JSONL 和 checkpoint 文件完成 fsync；有限 cohort 的请求 payload/JSON 也在释放前预编码，`sent` 时间戳记录在 HTTP 请求 context 建立后。08:00Z 复跑显示客户端流在 320 token 后停滞。因内部 metrics endpoint 只读取首个、每 64 token 和最终累计计数，队列转发已改为只发送这些诊断样本；普通推理请求仍逐次转发。0827Z 四个发送事件环消除了客户端流停滞，但完整校准的 send-lag P99 为 314.81ms。随后测试了单事件环（0840Z，P99 237.74ms）和四个事件环、每环批量启动（0852Z，P99 303.66ms）。这些轮次在当时的代码中仍被旧门槛挡住；现在已移除时延筛选，之后继续执行 calibration 与所有候选。08:21Z 的一次前缀命名改动触发 route audit 失败，现已恢复兼容命名。以上实现调整都不是策略收益证据；写入该段时 Balance 胜出场景数为 **0 个已证实**（仅为该段历史快照，后续续跑结果见本文末尾更新）。

所有真实服务进程均设置 `RLIMIT_CORE=0`。不得清理 Vast 管理的数据区；按用户授权，本轮将 061900Z 失败校准及早期、已被 073751Z 最新 smoke 覆盖的 smoke 原始目录清理，以上保留失败原因和有效摘要；同时清理已无关联进程的旧 Ray 临时日志目录和 38 个已退出实验服务日志，共回收约 67.6 MB。为降低当前 screen 的存储占用，每个 episode 完整性审计通过且 trace、逐请求结果、决策记录和 episode 摘要已落盘后，自动移除重复的压缩 `runs/<episode>/events.jsonl.gz` 与 `client_events.jsonl.gz`；`artifact_prune_manifest.jsonl` 逐文件保留压缩/未压缩 SHA-256、大小及审计状态，全局原始 `causal_events/` 流保留到最终完整审计。后续运行记录并报告 send-lag，但不设时延门槛；完整 cohort、精确 token 长度、route/event/cleanup 审计与配对 seed 要求继续执行。

### 2026-10-29 正式筛选续跑状态

复核发现上表对 S1_02 seed 82002 的描述已过时：Stay 回合已完整通过审计，匹配的 Balanced 回合在初始 `/real_control/flush` 写请求时遇到 `ClientOSError`，episode 尚未开始发送请求。整次 screen 随后以 `run_failure.json` 标记 incomplete，owned service cleanup 通过；不能将这次断连解释为策略负结果，也不能将该 seed 计为有效 pair。之前 S1_02 seed 82001 的 pair 有效且 mean、P99 分别改善 12.26%、8.07%，但单 seed 只算待复核信号。截至本节记录时，全 screen 仍没有双 seed 均改善 mean 与 P99 的确认候选；该状态随后被本文末尾续跑更新取代。

失败发生在幂等 producer flush barrier。客户端已增加最多三次短间隔传输重试；续跑 runner 增加了按冻结 candidate ID 选择候选、复用原 3/3 通过的 `lambda_ref.json` 的能力。正式续跑仍从同一冻结矩阵生成完整配对 trace 和两个 seed，不设时延筛除；复用校准只省去重复校准，不改变候选目标 QPS。新 run 的 manifest 会记录所选候选和校准来源，旧 failed run 与其有效配对证据继续保留。

### 2026-10-29 12:47 UTC 续跑更新

权威结果目录：`benchmark/results/full_completion_balance_v2_ray_cont_20261029T105608Z/`。该续跑在上一正式 run 已完成的 S1_01 之外，按冻结矩阵运行 S1_02–S1_06、S2_07–S2_10、S3_11–S3_14、C0_15、C1_16；没有 send-lag 或完成时延门槛。Profile 的 3/3 Stay 校准复用来源仍是 `full_completion_balance_v2_ray_screen_20261029T093554Z/lambda_ref.json`，decode1536 λ_ref=1.205278936 rps。每个候选 N=128、seed 82001/82002、两策略配对。下列 5 个候选均有 2/2 个有效完整 pair；每个 Balance/Stay episode 均 128/128 正常终态，decode1536 输出各为 196,608 tokens，候选摘要的 route/event/cohort 审计通过。

`gain = 1 - metric_Balanced / metric_Stay`；正值代表 Balance 更低。括号中按 seed 82001、82002 顺序。

| 候选 | ρ / 目标 QPS / 热点 | Balance 迁移尝试（两个 seed） | mean E2E gain | P99 E2E gain |
|---|---|---:|---:|---:|
| S1_02 | 0.35 / 0.421848 / 0.90 | 42、43 | +16.34%、+13.66% | +11.71%、+10.43% |
| S1_03 | 0.55 / 0.662903 / 0.70 | 17、15 | +4.34%、+1.89% | +2.17%、+1.31% |
| S1_04 | 0.55 / 0.662903 / 0.90 | 33、31 | +12.82%、+11.41% | +8.11%、+6.68% |
| S1_05 | 0.75 / 0.903959 / 0.70 | 15、14 | +3.87%、+5.48% | +2.56%、+8.71% |
| S1_06 | 0.75 / 0.903959 / 0.90 | 28、28 | +10.20%、+9.61% | +5.01%、+4.85% |

以上为搜索开发 seed 上的**探索性双指标胜出**，不是独立验证结论。S1_01 在前一 run 的两个 seed 方向不一致，未计为胜出。续跑快照时 S2_07 的 Stay seed 82001 正在完整排空（128 个请求已发送，115 个客户端终态）；之后仍需其 Balance 配对及其余候选。该时刻 runner 与 causal rotation monitor 均存活；可用磁盘约 255 MiB，继续按每 episode 完成后轮转归档。

按用户授权，已删除旧 run `full_completion_balance_v2_ray_screen_20261029T093554Z/runtime/ray/` 中已结束服务的 Ray 临时日志（42,472,769 bytes）；`service_cleanup.json` 为 passed，所有被删文件的 SHA-256、路径和字节数记在该旧 run 的 `manual_cleanup_manifest.jsonl`。配对结果、trace、ledger 和请求级审计材料均保留。截至 12:47 UTC，当前续跑没有 `screening_summary.json` 或 `service_cleanup.json`，因为候选搜索仍在执行。12:49 UTC 再检查时，S2_07 seed 82001 的 Stay 已 128/128 完成且 episode audit passed；配对的 Balance 已发送 50/128 个请求、收到 29 个客户端终态，仍在排空。

### 2026-10-29 12:56 UTC 续跑更新

S2_07（bimodal_0.1，ρ=0.55，目标 2.411605 QPS，长请求热点承接份额 0.80）两个 seed 的 Stay/Balance 配对均已完整通过审计，每回合 128/128 客户端终态。Balance 迁移尝试数为 seed 82001/82002 的 15/17。以 `1 - metric_Balanced / metric_Stay` 计算，mean E2E gain 分别为 +2.91%、+1.19%，P99 E2E gain 分别为 +3.66%、+2.01%；candidate summary 标记 `valid_pairs=2`、`exploratory_win_both_metrics=true`。因此续跑中已确认 6 个开发 seed 上的探索性双指标胜出候选（S1_02–S1_06、S2_07），均尚非独立验证结论。

当前仍运行同一正式续跑，S2_08 后续候选继续按冻结矩阵执行，不设 send-lag 或延迟门槛。12:56 UTC 检查时 runner、推理服务及 causal rotation monitor 均存活；可用磁盘约 246 MiB，episode 完成后继续归档释放原始流占用。完整逐回合证据见 `benchmark/results/full_completion_balance_v2_ray_cont_20261029T105608Z/candidate_ledger.jsonl` 与 `episodes.jsonl`。

### 2026-10-29 13:05 UTC 续跑更新

S2_08（bimodal_0.2，ρ=0.55，目标 2.221131 QPS，长请求热点承接份额 0.80）两个 seed 的 Stay/Balance 配对均完整通过审计，每回合 128/128 客户端终态。Balance 迁移尝试数为 seed 82001/82002 的 15/21。mean E2E gain 分别为 +1.46%、+4.95%，P99 E2E gain 分别为 +1.84%、+5.27%；候选摘要 `valid_pairs=2`、`exploratory_win_both_metrics=true`。当前续跑已有 7 个开发 seed 上的探索性双指标胜出候选（S1_02–S1_06、S2_07、S2_08），尚非独立验证结论。

S2_09（bimodal_0.1，ρ=0.75）已开始；13:05 UTC 检查时 seed 82001 的 Stay 正在排空。runner、推理服务与 causal rotation monitor 均存活，没有时延门槛；当前可用磁盘约 238 MiB，按 episode 边界继续归档。

### 2026-10-29 13:13 UTC 续跑更新

S2_09（bimodal_0.1，ρ=0.75，目标 3.288552 QPS，长请求热点承接份额 0.80）两个 seed 的 Stay/Balance 配对均完整通过审计、每回合 128/128 终态；Balance 迁移尝试数为 14/15。mean E2E gain 为 +2.98%/−0.14%，P99 gain 为 +4.07%/+1.53%（按 seed 82001/82002）。由于第二个 seed 的 mean 略有回退，候选摘要 `valid_pairs=2`、`exploratory_win_both_metrics=false`，不计入双指标胜出数。当前双指标胜出仍为 7 个（S1_02–S1_06、S2_07、S2_08）。

S2_10（bimodal_0.2，ρ=0.75）已开始；13:13 UTC 检查时 seed 82001 的 Balance 正在运行。runner、推理服务及归档监控均存活；不设时延门槛。完整候选回合与审计证据仍在当前续跑的 `candidate_ledger.jsonl`、`episodes.jsonl` 中。

### 2026-10-29 13:22 UTC 续跑更新

S2_10（bimodal_0.2，ρ=0.75，目标 3.028815 QPS，长请求热点承接份额 0.80）两个 seed 的 Stay/Balance 配对均完整通过审计，每回合 128/128 客户端终态。Balance 迁移尝试数为 15/18；mean E2E gain 分别为 +4.95%、+4.12%，P99 E2E gain 分别为 +6.73%、+4.28%。候选摘要 `valid_pairs=2`、`exploratory_win_both_metrics=true`。当前续跑共 8 个双指标探索性胜出候选（S1_02–S1_06、S2_07、S2_08、S2_10）；S2_09 两个 seed 均完整但 mean 方向不一致，保留为非胜出结果。

S3_11（decode512，ρ=0.55，突发倍率2）已开始；13:22 UTC 检查时 seed 82001 的 Stay 正在排空。runner、推理服务及 causal rotation monitor 均存活，无时延门槛；当前可用磁盘约237 MiB，继续按完整 episode 审计边界轮转归档。

### 2026-10-29 13:27 UTC 续跑更新

S3_11（decode512，ρ=0.55，目标 3.895011 QPS，热点份额0.80、突发倍率2、热点轮换）两个 seed 的 Stay/Balance 配对均完整通过审计，每回合128/128终态。Balance 迁移尝试均为7次；mean E2E gain 为 +1.88%/+5.34%，P99 E2E gain 为 +0.76%/+4.50%。候选摘要 `valid_pairs=2`、`exploratory_win_both_metrics=true`，当前共9个双指标探索性胜出（S1_02–S1_06、S2_07、S2_08、S2_10、S3_11）。

S3_12（decode512，ρ=0.55，突发倍率4）已启动；13:27 UTC 检查时 seed 82001 的 Balance 正在运行。runner、推理服务和 causal rotation monitor 均存活，仍不设任何时延门槛。

### 2026-10-29 13:32 UTC 续跑更新

S3_12 两个 seed 的 Stay/Balance 配对均完整通过审计，每回合128/128终态；Balance 迁移尝试为6/8。seed 82001 的 mean/P99 gain 为 +0.40%/+3.46%，seed 82002 为 −1.34%/−3.04%，因此 `exploratory_win_both_metrics=false`，不计入胜出。当前双指标探索性胜出仍为9个（S1_02–S1_06、S2_07、S2_08、S2_10、S3_11）。

S3_13（decode512，ρ=0.75，突发倍率2）已开始；13:32 UTC 检查时 seed 82001 的 Stay 正在运行。正式 runner、推理服务及 causal rotation monitor 均存活；不设置 send-lag 或完成时延门槛。

### 2026-10-29 13:36 UTC 续跑更新

S3_13（decode512，ρ=0.75，目标 5.311379 QPS，热点份额0.80、突发倍率2、热点轮换）两个 seed 的配对均完整通过审计、每回合128/128终态，Balance 迁移尝试均为6次。mean E2E gain 为 +0.78%/+4.18%，P99 gain 为 −1.58%/+5.51%；seed 82001 的 P99 回退，故 `exploratory_win_both_metrics=false`。当前双指标胜出数仍为9个。

S3_14（decode512，ρ=0.75，突发倍率4）已开始；13:36 UTC 检查时 seed 82001 的 Balance 正在运行。runner、推理服务和 causal rotation monitor 均存活，不设时延门槛；后续仍有 S3_14、C0、C1 待完成。

### 2026-10-29 13:41 UTC 续跑更新

S3_14（decode512，ρ=0.75，目标5.311379 QPS，热点份额0.80、突发倍率4、热点轮换）两个 seed 的配对均完整通过审计、每回合128/128终态，Balance 迁移尝试均为6次。mean E2E gain 为 +2.20%/−3.07%，P99 gain 为 +2.91%/−4.73%；seed 82002 两项回退，`exploratory_win_both_metrics=false`。S3 四组候选已全部完成；双指标探索性胜出共9个，负向和不一致结果均保留。

C0_15 均匀固定路由负对照已启动；13:41 UTC 检查时 seed 82001 的 Stay 正在运行。C0、C1 对照仍需按完整双 seed 配对完成；runner、推理服务与归档监控存活，无时延门槛。

### 2026-10-29 13:56 UTC 续跑更新

C0_15（decode1536，ρ=0.55，目标0.662903 QPS，均匀固定路由）seed 82001 的 Stay/Balance 配对已完整通过审计、每回合128/128终态；Balance mean gain +0.38%，P99 gain −3.09%，迁移尝试2次，不能算双指标胜出。seed 82002 的 Balance 也已通过128/128审计，配对 Stay 正在运行；C0 暂无双 seed 结论。当前9个探索性双指标胜出保持不变。

### 2026-10-29 14:00 UTC 续跑更新

C0_15 两个 seed 的 Stay/Balance 配对均完整通过审计、每回合128/128终态；迁移尝试数为2/0。mean gain 为 +0.38%/+4.69%，P99 gain 为 −3.09%/+13.17%。均匀负对照的 P99 方向随 seed 改变，故 `exploratory_win_both_metrics=false`，按冻结规则保留但不算胜出。

C1_16 短输出负对照已启动；14:00 UTC 检查时 seed 82001 的 Balance 已通过128/128审计（3次迁移尝试），配对 Stay 正在运行。当前9个双指标探索性胜出不变；正式 runner、推理服务及归档监控存活，仍不设时延门槛。

### 2026-10-29 14:06 UTC 首轮16候选粗筛完成

权威总表为 `benchmark/results/full_completion_balance_v2_ray_cont_20261029T105608Z/screening_summary.json`，状态 `screened`。续跑覆盖冻结矩阵中除已在前一 run 完成的 S1_01 外的其余15个候选；与前一 run 的 S1_01 汇总合并后，16/16 候选均有2/2有效完整配对。续跑的61个 episode（单独 warmup 加60个策略回合）全部 `passed`；60个候选回合都为128/128客户端终态且审计无 issue。send-lag 与完成时延均未设筛选门槛。

以下9个候选在两个开发 seed 上 mean 与 P99 都为正 gain，符合本轮“Balance 优于 Stay”的探索性双指标条件。`gain=1−Balanced/Stay`；百分比为两个 seed 的中位 gain。

| 候选 | 场景 | ρ / 目标 QPS | median mean gain | median P99 gain |
|---|---|---:|---:|---:|
| S1_02 | decode1536，热点90% | 0.35 / 0.421848 | +15.00% | +11.07% |
| S1_03 | decode1536，热点70% | 0.55 / 0.662903 | +3.12% | +1.74% |
| S1_04 | decode1536，热点90% | 0.55 / 0.662903 | +12.12% | +7.39% |
| S1_05 | decode1536，热点70% | 0.75 / 0.903959 | +4.68% | +5.64% |
| S1_06 | decode1536，热点90% | 0.75 / 0.903959 | +9.91% | +4.93% |
| S2_07 | bimodal_0.1，长请求热点承接80% | 0.55 / 2.411605 | +2.05% | +2.84% |
| S2_08 | bimodal_0.2，长请求热点承接80% | 0.55 / 2.221131 | +3.20% | +3.55% |
| S2_10 | bimodal_0.2，长请求热点承接80% | 0.75 / 3.028815 | +4.54% | +5.50% |
| S3_11 | decode512，热点轮换、突发倍率2 | 0.55 / 3.895011 | +3.61% | +2.63% |

其余7个候选及 S1_01 的混合/负结果均保留在 ledger 和总表中；C0_15、C1_16 负对照不计胜出。C1_16 seed 82001/82002 mean gain 为 −13.81%/+8.48%，P99 gain 为 −14.50%/+13.06%，候选汇总为 `exploratory_win_both_metrics=false`。

这是 N=128、两个开发 seed 的搜索筛选结果，属于探索性证据，不是独立验证或普适收益结论。本轮目标“找到几组实测 Balance 优于 Stay 的场景”已达成；文档中 N=2048 五个新开发 seed 的候选确认，以及冻结主场景后的独立验证，仍是后续研究阶段。

筛选服务最终 `service_cleanup.json` 为 passed。`service_stopped.json` 同时保留了 `owned process cleanup not confirmed` 的原始告警；14:06 UTC 的 `post_run_cleanup_verification.json` 复核登记的52个 PID 全部不存在、Supervisor 配置已移除。按用户授权，已删除已完成 run 的 `runtime/ray/` 临时日志（391个文件、109,803,823 bytes）；逐文件 SHA-256 与大小保存在 `runtime_ray_cleanup_manifest.jsonl`。候选 ledger、summary、trace、episode 审计和压缩 causal archives 均保留。
