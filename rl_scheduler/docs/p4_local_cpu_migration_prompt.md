# P4：原生 Windows / Linux CPU 十三环境 PPO 训练与搬迁

更新：2026-11-06。取代旧版必须 WSL2 的计划。已实现本地训练器、可运行辅助脚本及便携 ZIP 导出；Windows 不需要安装 WSL。Windows 尚须在用户主机实际验收，不能把 Linux 测试写成 Windows 测试。

## 1. 方案和入口

**Windows 原生 Python 与 Linux 使用同一份 CPU 模拟器、PPO、奖励和 checkpoint。** 训练约3.2万参数的调度网络，不运行四个大模型。跨系统搬迁整个同版本包和 run 目录，重新安装本机依赖后恢复；不要搬运 `.venv`。浮点结果允许数值容差，不承诺跨平台逐位相同。

公共入口仍只有仿真 `benchmark/run_ppo_sim.py` 与真实 `benchmark/run_ppo_real.py`。新增本地实现为 `benchmark/ppo_local.py`，不替换真实集群流程。Linux 专用真实预算锁保留在真实路径，纯 CPU 路径使用跨平台原子写入和排他文件锁；按需导入真实审计依赖。

| 已实现子命令 | 功能 |
| --- | --- |
| `portable-export --output-dir DIR` | 从当前冻结拟合模型生成完整 Windows / Linux CPU ZIP及SHA256 |
| `local-doctor --bundle-root DIR` | 核对文件hash、13项配置、V2权重和固定观测参考输出 |
| `local-smoke --bundle-root DIR --run-dir DIR` | 13项完整数值trace、一次可丢弃PPO更新、保存重载及200秒超时失败检查 |
| `local-train --bundle-root DIR --run-dir DIR` | 有限预算、多环境、同策略版本采样和更新 |
| `local-train ... --resume` | 恢复 latest 的权重、优化器、随机数、回合及更新进度 |
| `local-evaluate ... --split development` | 同轨迹 Stay / Balanced / PPO 开发评估，可重用基线缓存 |
| `local-evaluate ... --split heldout` | 冻结已选策略，独立模拟轨迹测试，不用测试结果选策略 |
| `local-export-policy ... --output-dir DIR` | 导出CPU策略、归一化、奖励、环境清单和验证标记 |

打包模板位于 `benchmark/windows_bundle/`，包内 `README_WINDOWS.md` 给出具体操作；`README_WINDOWS_WSL.md` 仅保留兼容说明。运行依赖为固定 CPU PyTorch、NumPy、PyYAML，无 Ray、vLLM、Llumnix、tokenizer 或 Qwen 权重。

## 2. 模型依据与训练范围

冻结拟合目录 `benchmark/results/mig_ppo_v2_refit_20261106T100025Z/` 包含27个已见fit/development配对、54次闭环。C宏平均误差12.08%→6.84%，D误差89.02%→8.19%。HOT1-H负收益、HOT1-L异常快Stay、NATIVE1.25正收益仍未完整复现；部分迁移暂停/尾部验收仍未达标。旧真实数据都已参与模型开发，独立真实验证仍缺失。

因此本地训练标记 `scope=local_simulation_development`、`allow_unvalidated_physics=true`、`formal_ppo_ready=false`。这允许开始仿真开发，不表示物理精度或真实迁移收益已通过；不重写历史注册表、误差和源码hash。初始策略为新建 V2 网络；旧 `real_updated.pt` 的状态维度不兼容，不能直接恢复。

## 3. 十三环境清单

每回合从下列13项**有放回等概率抽取一项，每项1/13**，回合内环境不变；每次更新4个新回合使用同一个冻结策略版本。环境ID不进入策略输入。NATIVE本次按用户要求纳入训练，之后不能称为未见环境；开发/留出测试使用独立轨迹。

| env_id |   热点位置 | 目标强度 / 实际热点请求数        | 请求数 N | QPS | 对应拟合来源 pair_id                      |
| ------ | ---------: | -------------------------------- | -------: | --: | ----------------------------------------- |
| HOT1-L |          1 | 50% / 24                         |       48 | 1.0 | HOT1-L_backup_n48_q1p0_s26110523          |
| HOT1-M |          1 | 70% / 34                         |       48 | 0.6 | HOT1-M_primary_s26110521                  |
| HOT1-H |          1 | 90% / 43                         |       48 | 1.0 | HOT1-H_backup_n48_q1p0_s26110523_followup |
| HOT2-L |          2 | 50% / 24                         |       48 | 0.6 | HOT2-L_primary_s26110521                  |
| HOT2-M |          2 | 70% / 22（68.75%）               |       32 | 0.6 | HOT2-M_backup_n32_q0p6_s26110522          |
| HOT2-H |          2 | 90% / 43                         |       48 | 1.0 | HOT2-H_backup_n48_q1p0_s26110523          |
| HOT3-L |          3 | 50% / 24                         |       48 | 0.6 | HOT3-L_primary_s26110521                  |
| HOT3-M |          3 | 70% / 34                         |       48 | 0.6 | HOT3-M_primary_s26110521                  |
| HOT3-H |          3 | 90% / 43                         |       48 | 1.0 | HOT3-H_backup_n48_q1p0_s26110523          |
| HOT4-L |          4 | 50% / 24                         |       48 | 0.6 | HOT4-L_primary_s26110521                  |
| HOT4-M |          4 | 70% / 34                         |       48 | 0.6 | HOT4-M_primary_s26110521                  |
| HOT4-H |          4 | 90% / 43                         |       48 | 1.0 | HOT4-H_backup_n48_q1p0_s26110523          |
| NATIVE | 无固定热点 | 模拟原生分发器依据当前状态选实例 |       48 | 0.6 | NATIVE_primary_s26110521                  |

共同约束：四实例；输入128、输出1536 token；单回合**最多200秒仿真时间**，所有请求完成后可提前结束；到达窗口为 `N/QPS`，始终在200秒内。HOT2-M保留已测的32请求，不擅自把所有环境统一成48或128请求。

到达时刻复用已测生成器：在 `[0,N/QPS)` 内抽取 N 个均匀时刻并排序（`uniform_order_statistics_fixed_n`）。人工热点先按 `floor(N*f+0.5)` 固定热点请求数，再随机排列，其余请求轮流分到其他三个实例。NATIVE 的“原生”指路由/分发方式；不要把原始到达过程暗改成 Poisson。每次抽样后按实际 trace 计算 arrival_end。

HOT1-L QPS=0.6、NATIVE QPS=1.0/1.25作为额外诊断配置导出，**不加入主13项抽样池**。尤其保留已知异常/反向收益证据，不因其不利于模型而删除。

## 4. 冻结的训练合同

- 同一 Actor/Critic：原始68值，归一化实例/全局/迁移对维度18/16/4；Stay加12有向迁移对。
- 决策间隔1.5秒；迁移冷却0秒。保留单迁移并发、容量、SR候选等合法性Mask。
- 主动Stay参与Actor/Critic/GAE；全部迁移动作被Mask的强制Stay不参与Actor，但保留Critic与连续GAE轨迹。
- V2四项共享奖励及势函数塑形，冻结 `calibration/reward_definition.json`；`gamma=.99**dt`、`lambda=.97**dt`。不另加固定迁移成功奖。
- 单回合至多200秒仿真时间，完成可提前结束，无实时sleep。到期未完成请求显式timeout，保留部分客户端输出并计失败惩罚；势函数终值0。墙钟watchdog超时属于执行错误，不制造伪终止、也不跳过困难环境。
- 训练只关闭大体量引擎诊断日志，逐回合决策、奖励、客户端输出帧、动作和压缩rollout保留。原拟合路径默认仍采用原有完整日志/完成行为。
- 默认Adam lr3e-4、epochs2、minibatch64、clip.2、value.5、entropy.01、grad.5、KL.03，choice_only。
- 默认16次更新/64回合，训练与开发评估共1800秒软预算，Torch2线程，单进程；用户可用CLI提高总预算。

## 5. 人工步骤（Windows）

**M1【人工】** 选择本机可写目录，例如 `D:\MigPPO`；准备64位Python3.10/3.11/3.12（已有Conda可创建独立环境）。无需WSL、虚拟化、重启或管理员权限。

**M2【人工】** 将实际生成的ZIP与旁边SHA256下载到该目录，核对整体hash后解压，进入 `mig_ppo_v2_cpu`；不要在 `C:\Windows\System32` 工作。连接服务器时，SSH地址/端口/用户名由用户填写，不写入包内。

```powershell
Get-FileHash .\mig_ppo_v2_cpu_windows_linux.zip -Algorithm SHA256
cd D:\MigPPO\mig_ppo_v2_cpu
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_cpu.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\smoke.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\train.ps1
# 等旧进程退出后恢复；总预算可通过参数调整：
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\train.ps1 -Resume -MaxUpdates 32 -WallBudgetSec 3600
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\evaluate.ps1 -Split heldout
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\export_policy.ps1
```

初次安装需要联网。Python路径选错时用 `install_cpu.ps1 -PythonExe '实际python.exe路径'`。脚本执行策略只对这次PowerShell进程设置，不修改系统全局策略；也能直接调用 `.venv\Scripts\python.exe benchmark\run_ppo_sim.py ...`。

**M3【人工】** 审阅本机doctor/smoke报告与一次更新结果；若要延长训练，选择时间/更新预算。CPU、内存、磁盘是否足够以本机实际测量为准。

**M4【人工】** 服务器完整研究归档需另行备份，当前ZIP仅为训练包；备份确认后，租赁停止/退租由用户自行处理，本任务不操作服务或平台。

**M5【人工，未来】** 在独立真实轨迹上安排策略验证后再考虑真实接管。当前本地收益只说明模拟环境中的表现。

## 6. Linux 上继续训练

同一ZIP解压后安装Linux依赖即可；若从Windows迁移，复制同版本训练包和 `runs/`，重新创建Linux虚拟环境。

```bash
cd mig_ppo_v2_cpu
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python benchmark/run_ppo_sim.py local-doctor --bundle-root .
python benchmark/run_ppo_sim.py local-train --bundle-root . --run-dir runs/local_001
# 已有同一run时使用：
python benchmark/run_ppo_sim.py local-train --bundle-root . --run-dir runs/local_001 --resume
```

最新 checkpoint 保存权重、优化器、环境抽样 RNG、Python/NumPy/Torch RNG、待更新回合和in-flight回合。中断采样重做相同环境/seed；更新中断从更新前的完整checkpoint恢复；不保存半更新权重。锁文件残留时先确认旧进程退出，再使用 `--recover-lock`。恢复只允许更改总更新/时间预算，默认继承原预算；实际已用墙钟会记录，包括可捕获中断/失败。

## 7. 开发选择与留出测试

训练seed从80,000,000递增；开发seed900,000,000起；留出seed900,010,000起，PPO使用独立动作seed。每个split每环境2条trace，每条PPO2个动作seed，与Stay/Balanced比较。相同N/QPS的环境共享到达基础轨迹，按忽略请求ID/路由的基础工作负载hash检查split不交叠。

第0、8、16次更新开发评估，按13环境等权平均V2回报选择，回报相同参考C，同时不允许失败率高于初始策略开发结果；只有完整评估才能产生selected。Stay/Balanced按输入与参数hash缓存。评估会保存并恢复训练采样RNG。

训练用 `latest.pt` 恢复，测试默认用 `selected.pt`。留出测试一旦冻结策略hash，不允许用另一策略反复挑测试结果；同一策略可恢复中断测试。开发预算耗尽可提高总预算继续；`--skip-evaluation` 仅诊断训练链路，不能据此宣称候选有效。

## 8. 给本机测试监督 agent 的执行 prompt

在用户解压的 `mig_ppo_v2_cpu` 内工作，先读取 `README_WINDOWS.md` 和manifest；记录平台、Python、CPU、内存、导入源码路径。不连接真实服务，不下载模型权重，不修改拟合模型、奖励、状态、动作Mask或环境强度。

Windows执行install/doctor/smoke；Linux用README中的Python命令。doctor、13项smoke、一次更新和保存重载通过后，再执行有限预算标准训练与开发评估。可安装requirements-test并执行 `python -m pytest -q tests`，验证实际模拟超时、迁移中断清理、连续训练与恢复一致、随机数恢复和留出冻结规则。不得将平台未执行项写为已通过。

逐项记录13环境的C、平均/p95 E2E、失败率、V2四项压力与回报、迁移总数/独立请求/重复迁移/反向迁移、合法机会、主动/强制Stay。报告覆盖次数及分环境结果，不仅报告宏平均。失败回合和困难环境保留；watchdog失败需修复或保留报错等待恢复，不用换seed丢弃。

用开发结果选择策略；冻结后仅一次留出模拟测试；最后导出候选策略及完整标记。交付本机报告、checkpoint、评估、SHA256与尚缺独立真实验证的说明。当前源码下不重跑27对历史拟合或启动GPU长训。

## 9. 验收证据位置

本次实际交付目录：`benchmark/results/mig_ppo_v2_native_cpu_20261106T113030Z/`。训练包为 `mig_ppo_v2_cpu_windows_linux.zip`，旁边提供同名 `.sha256` 文件；解压后从 `README_WINDOWS.md` 开始。

服务端交付目录中的 `export_report.json` 和 `verification_report.json` 记录实际包路径、SHA256、Linux测试与Windows未执行状态。包内 `validation/` 保留上游误差摘要、注册表和固定策略参考输出；`tests/` 是可在两系统执行的便携合同测试。默认运行结果为 `runs/doctor_report.json`、`runs/smoke_001/smoke_report.json`、`runs/local_001/local_report.json`、`checkpoints/` 和 `evaluation/`。

依赖安装方案参考[PyTorch官方CPU历史版本说明](https://pytorch.org/get-started/previous-versions/)与[Python官方venv说明](https://docs.python.org/3/library/venv.html)。
