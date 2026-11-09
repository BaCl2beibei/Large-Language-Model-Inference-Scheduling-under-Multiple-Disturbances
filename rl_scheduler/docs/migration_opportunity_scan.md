# Stay / constrained Balanced 迁移机会扫描

本轮交付代码和 CPU 自动化测试，不启动真实 GPU 实验。真实结果和最终推荐由后续实验与独立分析产生；无实测数据不填充预测数值。没有启动 PPO，未改网络或奖励。

## 环境审计与复用

2026-11-02（Asia/Shanghai）只读检查：4 × RTX 3060 12 GiB，未发现 GPU 计算进程；`.venv` 中 vLLM 0.6.3.post1、Ray 2.47.1、Qwen2 tokenizer 和模型路径存在。GPU 时间预留无法从本地确认。实验入口在启动前再次检查版本、外来计算进程；复用 supervisor 服务所有权、外来 Ray 拒绝及清理机制。

复用 `RealEnv` 的有限窗口、`OpportunityLiveEpisodeIO` 独立线程 Poisson 发包及尾部事件游标、`RealBridge` 的 SR/容量余量/健康/全局单并发/5 秒冷却、已有 Balanced 迁后不反转和差值缩小判断。这里是受约束 Balanced；关闭原生 routine/pre-stop migration，不与原生 Balanced 混算。

新增可关闭 `LLUMNIX_SCAN_TELEMETRY=1`，旁观原调度器实际 `can_allocate` / `can_schedule` 返回值，不改调度决定，异常时恢复原方法。记录 KV LATER/NEVER、sequence/token budget 拒绝、抢占模式及调度 prefill/decode token 数。计数允许重叠，不代表互斥阻塞时间；未知抢占原因不能自动标为 KV。worker flush 暴露 PID，使用实时 NVIDIA PID→GPU UUID 确认实例映射，无法确认即停止，不假设实例列表顺序等于物理卡号。

## 运行

在仓库根目录使用 `.venv/bin/python`。以下每条实验命令自行完成该批工作并退出，不需要聊天工具轮询。stdout 可重定向到自己的日志；完成后才独立分析。

```bash
# 只生成计划、代码指纹与只读环境审计，无 GPU 服务启动。
.venv/bin/python benchmark/run_migration_opportunity_scan.py plan --run-dir benchmark/results/migration_opportunity_scan_my_run

# 阶段一粗扫描，墙钟预算 2 小时。
.venv/bin/python benchmark/run_migration_opportunity_scan.py scan --run-dir benchmark/results/migration_opportunity_scan_my_run --max-seconds 7200

# 后续独立分析：产生粗扫描表、诊断、候选 shortlist。
.venv/bin/python benchmark/run_migration_opportunity_scan.py analyze --run-dir benchmark/results/migration_opportunity_scan_my_run

# 在 shortlist 周围做独立 scheduler-limit 控制，墙钟预算 2 小时。
.venv/bin/python benchmark/run_migration_opportunity_scan.py refine --run-dir benchmark/results/migration_opportunity_scan_my_run --candidates benchmark/results/migration_opportunity_scan_my_run/candidate_shortlist.json --max-seconds 7200

.venv/bin/python benchmark/run_migration_opportunity_scan.py analyze --run-dir benchmark/results/migration_opportunity_scan_my_run

# 选定 2–3 个候选（shortlist 是 case 对象数组，可复制为人工审阅的 candidates.json）。
# 阈值从当前运行 Stay 到达阶段的有限 F 值经验 Q25/Q50/Q75 冻结；分布退化则停止。
.venv/bin/python benchmark/run_migration_opportunity_scan.py freeze --run-dir benchmark/results/migration_opportunity_scan_my_run --candidates benchmark/results/migration_opportunity_scan_my_run/candidate_shortlist.json

# 阶段二，一次命令执行全部已冻结候选的三 seed 四策略配对。
.venv/bin/python benchmark/run_migration_opportunity_scan.py paired --run-dir benchmark/results/migration_opportunity_scan_my_run --max-seconds 14400

# 后续独立统计与绘图。
.venv/bin/python benchmark/run_migration_opportunity_scan.py analyze --run-dir benchmark/results/migration_opportunity_scan_my_run
```

同一目录再次执行 `scan/refine/paired` 会跳过完整记录的 case/seed/policy/threshold；未完成尝试保留并新建服务子目录。配对中断后，恢复命令跳过已完整完成的候选；对未完成候选整体重跑12组，确保新实例 epoch 一致。账本保留旧证据，分析取每组最新完成记录，拒绝跨 epoch 比较。代码指纹改变也要求新目录。`freeze` 拒绝覆盖既有阈值。

## 预计矩阵与时间

| 批次 | 配置 | 组数 |
|---|---|---:|
| Stay 粗扫 | 512/512、512/1024、1024/1024 × QPS 3/6/9；seq=192、batch tokens=4096 | 9 |
| 候选局部控制 | 每候选 seq=128/256（batch=4096）；seq=192/batch=2112 | 6–9 |
| 配对 | 2–3 候选 × 3 seeds（91001/2/3）× Stay、L、M、H | 24–36 |

固定 memory utilization=.65、上下文2112、70/10/10/10 路由、temperature=0、ignore_eos=true。每组120秒到达+120秒测量排空+最多60秒清理。每次服务启动后另做10秒、四实例各一次的短预热，不纳入候选统计。按后端参数分组保持模型驻留。模型加载/服务重启另计，启动最多1200秒，故扫描约36分钟测量+启动/清理；局部控制24–36分钟测量+3类后端启动；配对96–144分钟测量+启动/清理。并非所有参数全排列。

未形成真实 KV 压力时，先增加实际 KV 需求/逐级扩展 QPS；不自动生成全排列或降低并发。扩展需要在新运行计划中明确注册配置。

## 数据口径与保守判定

- `resources.jsonl` 保留每卡 KV%、GPU compute%、显存MiB/显存%、running/waiting/head wait、free/used blocks、prefill/decode 调度 token 速率及实际 output token 速率。实际完成 output token 总数来自请求账本，不能用生成预算代替。
- 固定240秒窗口。资源采用最多2.5秒前向保持、时间加权均值和加权 P95，长缺口不填0；GPU 数据使用 NVIDIA 查询完成时刻。四卡标准差/极差仅在四卡都有值时计算，另报覆盖率。保留各卡而非只报集群均值。
- `remaining_steps=(free-waiting_blocks)/(running+waiting)`，闲置实例不进入阈值分布；原 Balanced 仍可将其视为空闲目标。冻结数据指纹必须通过核验。
- 合法机会是 PPO 原始 mask 中存在合法 pair；阈值来源/目标数、阈值后 pair 数、实际 Balanced 排序/迁后谓词拒绝、容量/状态/冷却/并发分别保存于 decision telemetry。只将真实协调器 start + terminal succeeded 计为迁移成功，成功请求必须属于已接入 cohort。
- 独立报告 request coverage、opportunity ratio、action given opportunity、成功率、每请求迁移次数、间隔、耗时及各实例迁入/迁出；不合并为单一“迁移率”。迁移终态在清理中完成时保留终态时间，可从事件表识别跨窗口收尾。
- 延迟从窗口内成功完成请求计算，同时报告 offered/sent/accepted/completed/错误/timeout/未完成，避免只比较完成请求造成偏差。SLO goodput 以固定窗口为分母。零基线提升率为 N/A，仍给出绝对差异。
- 初始路由必须通过服务端路由及首次 engine admission 核验；模型初始热态、轨迹、sampling、SLO、窗口一致，策略顺序按 seed 轮换。
- 自动 shortlist 是后续人工审阅候选，不是最终推荐。KV 判定要求足够覆盖率、持续热点 Waiting、原调度器真实 KV allocation 拒绝及其他卡余量，另用提高并发/改变 token budget 控制排除人工瓶颈。当前探索规则：覆盖率≥80%、到达阶段 Waiting 占比≥20%，不把 KV 50–85%设为硬门槛。
- `recommended` 要求有 seq256 与 batch2112 控制证据、每组多次成功迁移且成功率≥80%、合法机会与均衡/热点排队改善，以及某阈值至少2/3 seed goodput提高≥10%、错误率绝对增加≤1个百分点。工程筛选规则均在输出中显式记录，不是统计显著性结论；缺配对为 conditional，完整配对仍未通过为 not_recommended。

## 输出与验收边界

输出目录含完整 configs/traces/raw_runs、逐卡资源表、两个最终汇总表、跨 seed 均值/标准差、迁移事件/统计、性能改善、瓶颈报告、阈值冻结、推荐报告及8张要求的图。只读取本次紧凑 summary 和 resource streams，后处理不重复扫描旧 events。无真实数据时表为空、报告明确未执行，不生成虚构图。

CPU fixture 模拟 HTTP 协议用于自动化验证，不是 GPU 实测，也不写入正式结果目录。仍待后续真实验收：worker PID/UUID 映射、Ray 服务清理、GPU 遥测开销与长时间真实迁移稳定性。本轮没有声称两个阶段实测完成或找到可推荐环境。

```bash
.venv/bin/python -m pytest -q rl_scheduler/tests/test_migration_scan.py rl_scheduler/tests/test_real_env.py rl_scheduler/tests/test_real_bridge.py rl_scheduler/tests/test_real_opportunity.py rl_scheduler/tests/test_real_metrics.py rl_scheduler/tests/test_real_runner.py
```
