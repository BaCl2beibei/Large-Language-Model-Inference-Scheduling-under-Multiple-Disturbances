# 首次审查：指标与迁移控制

建议选择：GPT-6 Astra，推理强度 high。模型选择在界面/API 中设置，prompt 本身不会切换模型。

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
本步编号：04。你是 P1/P2 的独立审查者。
前置：02、03 passed。允许读源码、diff、事件和运行针对性的 CPU 测试；只写 reviews/handoff 和审查测试，禁止修改业务实现、加载模型或启动 GPU。

目标不是复述前模型总结，而是找出会让真机数据无效或动作失控的缺陷。
检查：
1. 客户端与内部时间是否混用，迟到完成是否污染评分，未完成请求是否漏计。
2. 采样 token 事件是否被当作完整输出；事件/终态缺失是否被隐去。
3. 动作是否真的进入原有迁移路径，普通控制器是否仍在竞争发动作。
4. 候选复核、槽位原子性、超时查询、decision_id 幂等、终态关联是否成立。
5. 观测是否只包含当时可见信息，UUID 映射及 normalizer 是否一致。
6. reset 是否可能残留请求、投递或迁移；60 秒清理与总预算是否真正执行。
7. smoke 命令与现有 CLI 是否匹配，是否具备运行后自动释放本任务资源的能力。

至少对一条请求生命周期和一条拒绝/成功迁移链做独立追踪；必要时补最小反例测试。无新疑点不要反复跑全部测试。
问题给出严重性、源码位置、触发条件、后果及最小修复要求。
写 reviews/bridge_review.md/json，结论只允许 passed/failed/incomplete，绑定相关实现与 smoke 配置 hash。
passed 只表示允许按原协议进行 smoke，不代表真机已通过。
有阻断问题交回 Sol 修复；本步不要亲自扩大实现。

