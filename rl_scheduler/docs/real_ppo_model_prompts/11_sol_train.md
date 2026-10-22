# P4：有界真机训练与开发评估

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
本步编号：11。你是真机训练执行与诊断负责人。
前置：10 passed，当前源码/配置与审查一致，P1–P3 live gate 有效。
允许：按 commands/11_train.json 运行真实训练、开发评估、保存 checkpoint 和诊断。禁止改冻结的奖励、SLO、场景、seed 或最终评价规则。

先 dry-run 核实 CPU 策略训练、单环境四 GPU、冻结协议、剩余总预算和训练最多 2 小时；不得启动默认 2000 回合。
按任务书采完整真实回合、对账、更新。持续检查真实成功迁移数、可选择动作数、拒绝率、数据完整性、KL/损失和实际收益。
记录所有失败回合及时间，不用训练 reward 上升代替开发主指标。
异常时先停止本任务采样并安全清理。若需修改算法、桥接或指标代码，输出缺陷证据并停止；经修复和相应复审后再恢复，不能边改边继续使用旧 rollout。
连续开发评估退化或任一预算触发时，按任务书止损，不无限增加步数。

训练结束：
- 固定 best/last 及选择理由，记录真实 transitions、updates、迁移终态数、墙钟和 GPU 小时。
- 检查 checkpoint 加载及 schema 一致性，不新增 GPU 试验凑通过。
- 输出 gates/p4_live.json；无有效更新/无真实迁移时不能标为迁移学习已验证。
- 在未查看最终结果前生成具体 commands/12_evaluate.json 和 final_eval_manifest.json，锁定 checkpoint hash、四策略、三场景、三 seed、顺序与窗口。
- 剩余预算不足以完成评估矩阵时明确说明，不自动删基线。

释放本任务 GPU 服务，交给 Luna 执行最终评估。本步不读取最终 seed 的策略表现。

