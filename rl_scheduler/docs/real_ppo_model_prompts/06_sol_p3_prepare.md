# P3：机会实验设计与脚本准备

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
本步编号：06。你是实验协议与奖励设计者。
前置：05 的 p1_live、p2_live passed；先独立核对样本和审计。若不通过，停止并开修复交接，不进入 P3。
允许：代码、CPU 测试、协议冻结。禁止运行新的 GPU 实验。

实现机会实验编排和可审计的 real_reward.py：
- 依据任务书从 QPS 3/6/9 探测健康档位；把探测选择规则写成固定规则，不事后凭结果挑场景。
- Stay 与受相同并发/冷却约束的 Balanced，正常与倾斜两个场景，各 3 个机会 seed。
- 相同 trace、可核对的真实初始路由、运行顺序随机化/交替、相同窗口。
- 主指标、错误率、截尾比例、机会 gate 和零分母处理明确实现。
- 奖励计数/时间积分/终态未完成惩罚有手算测试，不沿用会饱和的旧奖励。
- 明确客户端流式实现是否足以支撑奖励；不足时先修复并重新经过受影响审查及 live gate。
- 提前登记最终突发测试的生成规则与 seed，不读取最终策略表现来挑选测试。
- 阶段相关 hash 分离：训练文件新增不应错误使旧 bridge gate 失效；影响桥接逻辑的修改必须重验。

输出 opportunity_spec.yaml、commands/07_opportunity.json、初始奖励配置及待从机会数据计算的尺度规则。
本步只能冻结选择规则，不能把尚未测得的 C_ref/N_ref 编造为实测值。
CPU 测试与 dry-run 通过后，写 gates/p3_ready.json，停止等待运行步骤。

