# 修复交接模板

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

你接手的是当前阶段的修复任务，使用 GPT-6.1 Sol high。
先读取 active_run.json 对应 run_dir 的最新 failed/incomplete handoff 和 reviews，确认阻断发生在哪一步；若缺少明确问题单，先定位，不猜测重跑。

目标：仅修复已被证据确认、阻止当前阶段完成的问题。
- 修改前记录当前相关 source/config hash，保留失败证据。
- 对每项阻断给出最小根因、最小修改和能复现的测试。
- 不顺带重构，不更换奖励/门槛/seed 来掩盖失败。
- 若确需调整实验设计，写出新协议草案和影响，标记旧结果的适用范围，不沿用旧 gate。
- 修复影响 P1/P2 的行为，回到 04 审查，再重做受影响的 05 live 检查。
- 修复影响 P4 的算法或时序，回到 10 审查。旧策略/未完成 rollout 不可继续复用。
- 仅报告/绘图修复可 CPU 重算，不必重跑 GPU，但记录与旧结果的差异。
- 本修复步骤只运行 CPU 测试，不能自行启动新的 GPU 尝试。
- 同一问题两轮仍失败，整理最小复现、尝试记录和证据，交 Astra 定点诊断。

输出 run_dir/handoff/repair_<序号>.md/json，列出必须重验的 gate。不要自行把审查状态改成 passed。

