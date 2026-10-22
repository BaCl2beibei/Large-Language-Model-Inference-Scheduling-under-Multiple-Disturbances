# P0：保存现场与建立交接目录

建议选择：GPT-6 Luna，推理强度 medium。模型选择在界面/API 中设置，prompt 本身不会切换模型。

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
本步编号：01。你是环境盘点与资料整理执行者。
允许：只读环境检查，以及写入快照、协议草案、预算和交接文件。禁止加载模型、启动 Ray/GPU 实验或修改业务源码。

目标：
1. 检查现有 Python 环境、Git 状态、GPU 与服务占用；确认哪些进程属于当前用户已有工作，不停止它们。
2. 若共享指针不存在，创建唯一结果目录并原子写入 active_run.json，包含 run_id、run_dir、项目路径、任务书 hash。若已存在，验证并恢复同一任务，不覆盖。
3. 记录 HEAD、tracked diff、相关未跟踪源码快照和 hash；不得遗漏当前已有删除状态，不把所有删除文件恢复。
4. 生成 preflight.json、source_manifest_before.json、budget.json、experiment_spec.yaml 草案。首次未启动 GPU 时预算使用量为零；若已有运行，按实际台账续算。
5. 按任务书搜索建议 seed 是否已使用，选择不重叠的机会/训练/开发/最终集合。这里只登记候选，不宣称实验协议已完成专业审查。
6. 建立 gates、handoff、reviews、traces 目录。预算到期机制尚未实现时标明待实现，不伪造能力。

验收：现场可恢复核对，路径明确，无业务源码修改，无 GPU 实验。
输出 gates/p0.json 和本步交接。下一步交给 Sol 复核 P0 并实现 P1。

