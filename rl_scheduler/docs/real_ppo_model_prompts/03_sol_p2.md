# P2：动作桥接与真实环境实现

建议选择：GPT-6.1 Sol，推理强度 high。模型选择在界面/API 中设置，prompt 本身不会切换模型。

以下内容可整体交给执行模型。

你在同一工作区接手 Mig-PPO 真机训练项目。不要假定前一个模型的总结正确，先检查实际文件和证据。

项目：/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
主任务书：/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/rl_scheduler/docs/real_cluster_ppo_execution_plan.md
实例规范：/workspace/AGENTS.md

完整阅读实例规范；阅读主任务书的总体约束、阶段依赖及本次相关章节。主任务书中新增的文件和 CLI 是待实现规格，必须核实是否存在。本 prompt 将范围限定为当前交接步骤，优先于任务书中“自动推进全部阶段”的安排。完成当前步骤后停止，不自动进入下一步骤。

共同边界：
- 保留现有修改、未跟踪文件和删除状态。不 reset/clean，不批量 git add，不提交或推送。
- 仅使用当前实例；不租新机器，不更换模型/CUDA/驱动，不停止管理服务或 Vast 实例。
- 不启动旧的仿真拟合目标，不启动默认 2000 回合训练，不创建或恢复自动目标。
- 不派生子代理，不并发启动集群实验。
- 主任务书建议的总实验上限 8 小时、PPO 阶段 2 小时适用于整条流程，切换模型不重置。GPU 服务启动、运行、等待交接和清理均计时；CPU 编码时间另计。
- 只有本 prompt 明确允许的步骤可以使用 GPU。每个 GPU 步骤结束清理本任务服务，释放 GPU，避免等待下一个模型期间继续消耗实验预算；不得清理他人服务。
- 不放宽门槛、不隐去失败、不改最终测试集求通过。未知事件不能记作成功、失败或已取消。
- 重复失败时保留最小复现和证据；不得无上限试错。运行型任务禁止修改代码、权重、配置或 seed。
- 不读取或输出凭证。源代码快照不包括 .env、密钥、模型权重和整个数据集。

共享交接位置：
/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/rl_scheduler/runs/real_ppo_handoff/active_run.json
除第一步外，先从此文件读取 run_dir，再读取 run_dir 下的 handoff 和 gates。
若指针缺失、目录不一致或前置验收不成立，停止依赖执行，输出具体缺失项；不要另建一个实验绕过失败。
gate 必须基于实际检查生成，并绑定该阶段相关源码、配置和证据的 hash。无关文档或新模块增加不自动使所有 gate 失效，但实际影响已验收行为的改动必须重验。

每一步写：
run_dir/handoff/<本步编号>.md
run_dir/handoff/<本步编号>.json
内容包括 status（passed/failed/incomplete）、范围、改动文件、相关 hash、实际运行命令及退出码、测试/实验证据、遗留问题、剩余预算、下一步是否可执行。
不要把“代码检查通过”写成“真机验收通过”，不要把“训练能运行”写成“优于基线”。
最终回复简短给出状态、关键证据、交接文件路径；不要求用户重复确认已授权的本步操作。

当前任务：
本步编号：03。你是真实集群控制接口实现者。
前置：02 passed，p1_offline passed。
允许：代码及 CPU/mock 测试。禁止 GPU 压测和 PPO 训练。

实现主任务书 P2 及后续 smoke 必需的最小真实环境：
- real_bridge.py、real_env.py、real_observation.py、对应有版本的 normalizer。
- Manager/迁移调度器与 migrate_out 最小兼容改动；复用现有 KV 迁移机制。
- Stay+12 对有向实例，稳定索引；SR 候选执行前复核，不一致明确拒绝。
- 控制租约、一个全局在途迁移、原子占槽、完成后冷却、容量复核。
- decision_id 幂等、回执、migration_id 关联、终态反馈；超时不能当未执行后盲目重发。
- PPO 控制与普通 Balanced 互斥，failover/pre-stop 单独处理。
- snapshot 状态年龄、当前可见数据、缺失 mask、最近历史和有限回合剩余时间。
- 删除拟合迁移时间与预测净收益的策略输入；不假装未来输出长度已知。
- reset、有限到达/排空、截止冻结、60 秒清理期限和进程归属；空闲 KV 使用基线语义。
- 持久化跨进程预算、服务监管、退出清理。

实现 run_real_ppo.py 的 --dry-run、telemetry、bridge 阶段，尚未实现的阶段明确拒绝。
为下一模型生成 commands/05_smoke.json，使用结构化 argv/env、配置路径、预计时长和 hash；仅包含经检查的命令，不写凭证。包含低负载 telemetry、shadow、受约束主动迁移、开销配对检查，按任务书预算执行。
未实现的参数不能出现在所谓可运行命令中。

测试覆盖竞态、重复请求、候选变化、容量不足、状态陈旧、断线、索引失效、清理超时和控制器退出。
输出 gates/p2_offline.json、smoke_spec.yaml。区分离线通过与真机未验证。
准备好审查材料后停止，交给 Astra；不要自行开始 smoke。

