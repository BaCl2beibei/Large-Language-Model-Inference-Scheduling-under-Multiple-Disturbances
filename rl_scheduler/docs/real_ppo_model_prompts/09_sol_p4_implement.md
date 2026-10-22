# P4：实现真实 PPO 训练器

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
本步编号：09。你是 PPO 实现者。
前置：08 passed、p3 passed、冻结协议完整。
允许：实现/测试代码及 dry-run。禁止 GPU 训练或新压测。

实现 real_rollout.py、train_real_ppo.py、evaluate_real.py 和 runner train/evaluate 阶段。
采用完整真实回合采样后更新：
- 同一 rollout 固定策略、normalizer、奖励；达到至少 512 transitions 时只在完整回合边界停止采样。
- 更新时确认集群空闲，不漏记优化期间的真实请求。
- 保存完整分层动作 old_logp 和采样时 mask；拒绝动作不改写为 Stay。
- 全 mask 时 Stay 概率=1、logp=0、entropy=0；强制 Stay 不贡献 actor/entropy 梯度。
- 使用 dt 的 gamma/lambda，完整有限终态与 invalid 回合分开，GAE 不跨 reset。
- finalised reward 对账后才能优化，暂定奖励不重复累计。
- CPU 单线程、小网络、KL 提前停止、数值检查、三项预算先到即停。
- checkpoint 保存优化器、随机状态、策略版本、normalizer、schema、预算及训练步数。
- 开发评估选 best；固定最终测试不参与选择。
- 断点恢复从干净回合开始，不能复用未完成/旧策略 rollout。
- 最终评估四策略共享约束，原生启发式若另测必须另命名。

先跑具有手算基准的 mock/CPU 测试和最小优化 smoke；不要用仿真训练结果声称真机有效。
输出 gates/p4_offline.json、commands/11_train.json、commands/12_evaluate_template.json、reproduction.md。
训练与评估的命令、预估回合数、预算、终止行为必须 dry-run 可见。
准备审查材料后停止，交给 Astra 审查，不开始训练。

