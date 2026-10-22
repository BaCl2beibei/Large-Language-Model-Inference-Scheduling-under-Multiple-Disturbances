# P3：判断收益空间并冻结奖励

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
本步编号：08。你是机会实验分析与训练协议负责人。
前置：07 的实验清单、原始数据和审计完整；此步无需假定机会 gate 已通过。
允许：读取并重算结果、修复纯报告错误、保存协议与尺度。禁止新 GPU 实验，禁止更改已观察实验的入组规则。

独立复算：
- 固定截止的 goodput/完成率与清理附表是否分开；
- 配对 trace 与路由是否一致，实际到达是否偏离；
- 机会门槛是否真实满足，差异能否由迁移而不是采集/启动/清理差异解释；
- 合法动作频率、真实迁移样本量和奖励各分量是否支持开展小规模训练。

机会 gate 未通过：写失败/证据不足结论及最小后续建议，停止，不进入 P4。
通过时：
1. 依据预注册健康 Stay 数据计算 C_ref/N_ref 和观测归一化参数。
2. 按任务书允许范围做至多一次训练前奖励尺度修订，记录理由；不能用最终 seed。
3. 冻结 real_pilot.yaml、奖励、normalizer、schema、SLO、场景分布、开发/最终协议和评价方式。
4. 检查剩余预算是否足以开展有意义的训练及最终评估；不足明确 incomplete。
5. 输出 gates/p3.json、reward_audit.md、frozen_protocol_manifest.json。

不能把小样本机会门槛写成统计显著性证明。
通过后交给 Sol 实现 P4；若出现复杂的机制归因争议，整理证据交 Astra，而非强行通过。

