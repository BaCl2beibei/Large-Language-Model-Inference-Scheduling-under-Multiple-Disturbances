# Mig-PPO V2 原生 Windows / Linux CPU 训练包

本包不需要 WSL、Linux 虚拟机、CUDA、Ray、vLLM、Llumnix 或 Qwen 权重。训练的是调度网络（约3.2万参数）。相同 Python 源码也可在 Linux CPU 上训练，两边共用策略格式。

支持目标：Windows 10/11 **x64**，64位 Python **3.10/3.11/3.12**，CPU版 PyTorch。服务端能验证的是 Linux 和隔离导入；Windows 的实际兼容性以你本机 `doctor`、`smoke` 和一次更新为准。若已安装 Conda，可以用现有兼容版本Python创建本包专用venv，不在base里覆盖依赖。

## 你需要人工完成的步骤

1. 下载ZIP及其SHA256，选择可写的目录（例如 `D:\MigPPO`），核对hash后解压。**不要在 `C:\Windows\System32` 中训练。** 打开解压后的 `mig_ppo_v2_cpu` 文件夹。
2. 准备64位Python 3.10–3.12；若当前Python是3.13或更新，可在Conda中建 `conda create -n migppo310 python=3.10` 并激活，再运行安装脚本。初次依赖安装需要联网；之后训练不使用网络。
3. 在这个文件夹打开 PowerShell，执行下面的安装和短测。普通用户权限即可；不需要安装WSL或重启电脑。
4. 若要继续增加训练轮数/时间，自行选择预算；旧服务器停止/退租与未来真实测试另行处理，本包不操作它们。

## Windows：从安装到训练

```powershell
# 先进入解压后的实际目录：
cd D:\MigPPO\mig_ppo_v2_cpu
# 仅为本次脚本启动指定执行策略，不修改系统全局设置：
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_cpu.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\smoke.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\train.ps1
```

若脚本找错Python，明确指定已有解释器，例如：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_cpu.ps1 -PythonExe 'D:\Miniconda3\envs\migppo310\python.exe'
```

无需激活venv，脚本始终使用 `.venv\Scripts\python.exe`。也可完全不用PS脚本，直接运行：

```powershell
.\.venv\Scripts\python.exe benchmark\run_ppo_sim.py local-doctor --bundle-root .
.\.venv\Scripts\python.exe benchmark\run_ppo_sim.py local-smoke --bundle-root . --run-dir runs\smoke_002
.\.venv\Scripts\python.exe benchmark\run_ppo_sim.py local-train --bundle-root . --run-dir runs\local_002
```

现有目录不覆盖；重做smoke/新训练换RunName。配置/源码/参数受SHA256保护，先不要编辑包内冻结文件；更新次数与时间可通过命令参数调整。

## 断点恢复、评估和导出

Ctrl-C后等待进程返回，再执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\train.ps1 -Resume
# 总预算提高到32次更新、3600秒（不是再追加3600秒）：
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\train.ps1 -Resume -MaxUpdates 32 -WallBudgetSec 3600
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\evaluate.ps1 -Split heldout
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\export_policy.ps1
```

恢复默认使用 `runs/local_001/checkpoints/latest.pt`，不是selected.pt。正在采样的回合会使用原环境、原trace seed及原采样RNG重做；更新中断则回到完整更新之前，不使用半更新权重。若强制杀进程留下锁，先确认旧进程已退出，再在恢复命令加入 `-RecoverLock`；不要同时启动两个进程写同一run。

`-SkipEvaluation` 仅作快速训练链路诊断，跳过后没有候选选择结论；正常训练不加它。默认开发评估选出的 `selected.pt` 才能进入留出模拟测试；测试对象一旦冻结不可换策略挑结果。同一策略可恢复中断的测试或复核缓存。导出包记录开发选择和真实验证缺失，导出本身不代表真实有效。

## Linux：同一训练包直接运行

```bash
cd mig_ppo_v2_cpu
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python benchmark/run_ppo_sim.py local-doctor --bundle-root .
python benchmark/run_ppo_sim.py local-smoke --bundle-root . --run-dir runs/smoke_001
python benchmark/run_ppo_sim.py local-train --bundle-root . --run-dir runs/local_001
# 继续同一个run；可以把整个run目录和同版本包一起搬到另一系统：
python benchmark/run_ppo_sim.py local-train --bundle-root . --run-dir runs/local_001 --resume
python benchmark/run_ppo_sim.py local-evaluate --bundle-root . --run-dir runs/local_001 --split heldout
python benchmark/run_ppo_sim.py local-export-policy --bundle-root . --run-dir runs/local_001 --output-dir exports/local_001
```

跨系统恢复使用相同包hash、Python/PyTorch兼容版本；权重/优化器/RNG格式可搬迁，但不承诺浮点计算逐位一致。包内参考输出用于数值容差检查。

可选的合同测试（两系统均可；Windows用 `.venv\\Scripts\\python.exe` 替换python）：

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```

此测试覆盖超时、部分输出、迁移中断清理、训练/恢复一致、采样或更新中断恢复、评估缓存和留出策略冻结。便携包来源说明可参阅[PyTorch官方CPU安装说明](https://pytorch.org/get-started/previous-versions/)和[Python官方venv说明](https://docs.python.org/3/library/venv.html)。

## 默认训练合同

- 一个共享V2 Actor/Critic；每回合从12个人工热点和NATIVE中**有放回等概率抽样，概率各1/13**，回合内不换环境。NATIVE主配置为QPS0.6。env_id不进入策略状态。
- 每回合最多**200秒仿真时间**，并非墙钟等待200秒；决策间隔1.5秒，迁移冷却0秒，单迁移并发及动作Mask保留。
- 每次更新用同一冻结策略生成4个新回合；默认16次更新/64回合、训练与评估合计1800秒软预算，CPU 2线程，单进程。
- 状态原始68值，实例/全局/迁移对归一化维度18/16/4。奖励来自 `calibration/reward_definition.json`；折扣`.99**dt`、GAE`.97**dt`。
- 主动Stay参与Actor；所有迁移Mask导致的强制Stay只参与Critic与GAE。失败回合不丢弃；200秒仍未结束的请求明确记timeout、部分输出如实保留，计入失败惩罚。
- 从 `initial_v2.pt` 新策略开始；历史旧状态维度的 `real_updated.pt` 不兼容。一次smoke的 `smoke_only.pt` 不是训练起点。
- 训练数值trace不加载tokenizer/大模型。除HOT2-M为32请求外其余48请求，输入128/输出1536；QPS/热点强度按已测配置固定。
- 第0、8、16次更新后开发评估；13环境各2条trace，PPO各2个动作seed，Stay/Balanced基线缓存。按13项等权平均V2回报选择策略，并报告C、失败率等。只有完整开发评估才选checkpoint。
- 开发与留出模拟trace分开，旧真实数据全部是已见开发证据；**尚缺独立真实验证**。NATIVE本次按用户要求进入开发训练，不再声称是未见训练分布的环境。

## 结果位置与已知限制

`runs/doctor_report.json`、`runs/smoke_001/smoke_report.json`、`runs/local_001/local_report.json`、`checkpoints/latest.pt`、`checkpoints/selected.pt`、`updates/`、`evaluation/`、`episodes/`。原始逐引擎事件不写入训练日志；每回合决策、Token接收帧、奖励、动作和压缩rollout保留。

拟合误差在 `validation/upstream_report.md`。HOT1-H负收益、HOT1-L异常快Stay、NATIVE1.25正收益仍未复现，部分迁移暂停/尾部也未达标；当前训练仅为仿真开发。`formal_ppo_ready=false`，不得把模拟留出测试称为独立真实验证，不得用环境ID查表补偿迁移收益。

`SHA256SUMS`覆盖导出时的包文件，不含它自身、用户安装的venv和新产生的runs。压缩包整体hash由ZIP旁的 `.sha256` 提供。训练包不包含完整研究归档；原始真实日志应另外备份后再处理服务器租赁。
