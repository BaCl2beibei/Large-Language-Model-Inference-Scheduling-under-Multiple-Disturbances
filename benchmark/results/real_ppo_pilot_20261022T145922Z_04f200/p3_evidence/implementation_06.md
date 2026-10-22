# Step 06 implementation and frozen protocol

用户明确要求跳过 smoke，直接执行 06。本步因此完成 CPU 准备；不因 05 缺失而拒绝编码，也不把 05、p1_live 或 p2_live 改成 passed。例外原文、run_id、范围在 p3_evidence/user_override_06.json；07 的独立入口会保留此例外，而其他审计、预算、SLO、容量和冷却规则继续生效。当前无 GPU 执行授权。

新增文件对应关系：

- rl_scheduler/real_reward.py：按完整客户端事件重建好请求计数、outstanding/first/gap 三项时间积分、原 coordinator 尝试、失败/中止/明确拒绝、截止 tail。窗口 [lo,hi)，最后包含截止点一次；不裁剪。只有 scoring+cleanup 审计通过、回合 valid、动作终态明确、C_ref/N_ref 已测得时才输出 finalised reward。原 obs/action/logp 不回填。
- rl_scheduler/real_opportunity.py：6 个短探测、健康阈值、固定选档、12 回合配对顺序、constrained Balanced、主指标/错误/截尾/零分母、机会 gate、trace 与预算只读检查。
- rl_scheduler/real_opportunity_runner.py：使用原 LiveEpisodeIO、RealEnv、RealBridge 和原 KV 执行链；全程一个受监管服务，顺序执行，不派生代理。按 engine admission 审计实际初始实例，倾斜还要求 Manager 声明与预注册路由一致。首次审计失败保存 interrupted/gate 并停止，无自动重试。
- benchmark/run_real_opportunity.py：独立 P3 CLI；主任务书的 run_real_ppo.py --stage opportunity 仍明确拒绝。保留旧入口源码避免破坏当前 hash 绑定的 04/P1/P2，07 清单调用新入口。dry-run 无启动、无预算写入；live 需当前 P3/bridge/config 绑定及完整最大时长。服务 finally 只清理本任务 owner。

冻结规则：

- 探测 QPS 3/6/9，每档 A/B 各一次 Stay，共 6 回合，30s 到达+60s 排空，分别用 7101/7102/7103。健康要求 offered 全 sent、completion >=99%、nonmanual error <=1%、first censor <=1%、send lateness P95 <=1s，全部请求/动作/清理/路由审计通过。
- A 选健康正常档位的低侧中位数；B 选健康倾斜档位中，到达最后三分之一至少半数样本存在等待队列的最高档位。不存在则停止 incomplete，不挑新负载。
- 机会 A/B 各 2101/2102/2103，Stay/Balanced 配对共 12 回合，120+120s 固定窗口，60s 清理。按 scene_index+seed_index 交替政策顺序，同一 scene/seed 对共享完整 trace、sampling、外生路由规则、窗口与实例 epoch 映射。
- 输入/输出预算 256/512/1048 的 9 个组合等概率，保存本地 tokenizer 生成的 synthetic prompt pool，输入预算与实际编码数分别保存。输出预算不是实现长度。12 个候选/探测 trace 文件覆盖全部预注册负载，不截短 offered，记录 hash。
- Balanced 使用实测 remaining-capacity/request、native 剩余容量阈值方向及 Balanced 排序/改善且不过冲的判据；迁移 delta 用实际 SR blocks。没有沿用不可见的 native killed-request override，版本及差异写在协议里。它不是预测剩余输出长度，也没有迁移时间/净收益拟合。一个槽、5s 终态冷却、2s 状态年龄、5%/16-block reserve 仍由原 bridge/env/executor 执行。
- 主指标正常完成且 TTFT<=2000ms、请求 TPOT<=150ms 的数量/240s；单 token 只考核 TTFT。错误按固定截止客户端 timeout/service_error/all offered；手动取消另列；unknown 使审计 invalid。另列首输出截尾、终态截尾、未发送比例、完成率及绝对差值。
- 机会 gate 要求至少一个场景 2/3 seed 提升 >=10%，且该场景所有 seed 非手动错误率增加均 <=1pp。使用整数计数比较门槛，不加 epsilon 放宽。Stay goodput=0 仅报绝对量，不自动通过；两场景都双方零则 incomplete。完整矩阵/trace/路由规则/窗口/映射不符一律 incomplete。
- 奖励初始 good/hold/first/gap/attempt/bad/tail 权重 1/.2/.5/.2/.02/.2/2；C_ref、N_ref 当前 null。仅未来健康 Stay 机会回合按 pooled completed/seconds、outstanding-request-seconds/seconds 求值并分别 floor 1，保留原值、数据 ID 和不健康排除项。probe/Balanced/dev/final 不用于尺度拟合。记录分量占比与 reward 增但主指标降的配对，不自动调权；训练前最多一次有记录修订。
- final burst 5101/5102/5103 规则现在冻结：基准 QPS 取探测选定 A，0–30/45–90/105–120s 为 1x，30–45/90–105s 为 3x，Poisson、同样九组合/采样/正常路由。机会选档后、任何最终策略表现可见前生成并 hash。现在没有查看或运行最终策略。

客户端是否足够：现有 NUL JSON parser、累计正增量与 first/last/terminal 事件可支持请求级 TTFT/TPOT 和驻留/停顿代理积分；28 个原 P1 CPU 测试及新增 A/B HTTP 端到端验证通过。chunk 不是 token，不报告精确 ITL。没有真机样本，因此不能声称真机输出/迁移/取消链已验收；07 遇到缺事件、路由不符或 unknown 即停止，不能把未验证当成功。

本步不改任何既有业务源码、配置、门槛、seed、04 review、P1/P2 offline gate、05 失败/预检材料或预算。阶段哈希独立，source_manifest_p3.json 引用旧桥接输入，原 source_manifest_after.json 保留。

包装核验首轮因 3 个本步新建初始副本后来成为已跟踪 clean 文件，严格 Git status 条目检查退出 1；失败材料保存在 p3_evidence/packaging_attempt_01/。修正仅针对本步副本，并逐字节与原文件比对，所有原有用户状态与源码均保留；58 项业务测试没有失败或被重跑掩盖。
