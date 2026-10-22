# P1：请求台账与评分语义

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
本步编号：02。你是指标与事件链实现者。
前置：01 passed。先复核 P0，发现记录不完整就在本步补齐且记录。
允许：相关代码和实质性 CPU 测试。禁止启动 GPU 实验。

目标：实现主任务书 P1 的真实请求台账、评分边界、指标计算及事件完整性检查。
重点代码：real_metrics.py、必要客户端/事件采集修改和 test_real_metrics.py。保留旧接口默认兼容。
逐项实现：
- planned/sent/accepted、engine/API/client 首输出、客户端/服务端终态、取消请求与确认、评分截止明确区分。
- 累计 token 增量去重；chunk 不等于 token；采样阶段事件不冒充完整输出序列。
- 有限窗口内完成与清理阶段迟到完成隔离；HTTP 499 人工截尾与真实超时分开。
- 所有已发送请求直到客户端终态都计入 outstanding 积分，不仅统计引擎等待队列。
- 保存事件发生时间和接收时间；历史观测禁止回填未来才收到的事件。
- 增量游标、序列缺口、刷盘/完整性确认机制；没有完整证据时返回 incomplete。
- 能导出请求台账、窗口指标和审计结果，供后续 runner 调用。

写具有明确手算结果的测试：窗口边界、迟到、重复、token 回退、单 token、取消、缺终态、投递未完成、事件缺口。
可以读取已有少量日志检验兼容性，但它们不是新独立测试，也不用于训练。

输出 gates/p1_offline.json，不生成 p1_live 的通过结论；implementation.md 记录接口、指标定义、未验证点。
本步不实现 PPO，不运行集群。

