# Qwen2-1.5B / Q15-B：P2 与 P3 交付

| 项目 | 状态 | 证据 |
|---|---|---|
| p2_cpu_contract | passed | ppo_contract_report.json、cpu_validation.log、audit_recovery_tests.log |
| p2_real_collection | passed | real_collection_report.json、runs/ppo_151101/execution_audit_rechecked.json |
| p2_update | passed | real_update_report.json、final_verify.json |
| p3_data_model | passed | calibration/fit_report.json、calibration/extraction_index.json |
| p3_closed_loop | passed | sim_validation/closed_loop_report.json；仅限 Q15-B |
| ready_for_real_training | passed | 支持下一轮小预算链路开发，不代表已学到有效策略 |
| ready_for_sim_pretraining | passed | 仅限当前 Q15-B；Q15-A 负收益诊断失败，不支持推广 |

## 改了什么

保留现有13动作和共享实例编码 actor/critic。新增显式 actor_sample_mode=choice_only：全部281个时间步先计算奖励、完整return和优势，actor优势归一化、损失、entropy、KL、clip只用122个有选择样本，保留全部83个 voluntary Stay；critic仍用全部时间步。旧模式默认all。无选择minibatch跳过actor，全无选择报告no_actor_update。共享编码器仍可能因critic更新间接改变策略。损失权重为value=0.5、entropy=0.01，二者各按自身样本数归一化；这改变相对权重，已显式记录。
观测、mask、action、old_logp、old_value、policy_version和动作分类在动作前保存，执行拒绝不改写为Stay。时间戳使用observation_decision_ns；旧日志缺少该字段的无机会原因记unknown。
复用原离散事件环境的请求、KV、引擎事件和迁移生命周期，新增完整cohort适配层，模拟与真实共用因果观测、掩码、13动作、归一化、奖励和完整回报。未使用旧512 blocks/60秒截止/0.05秒步长/1%随机失败默认值。

## 真实结果

只启动1个正式PPO cohort，seed151101，128/128完整成功。共281个决策：只能Stay 159、有选择但Stay 83、迁移意图39；actor_eligible=122。条件于有机会的平均迁移概率32.24%，实际迁移选择比例31.97%。
真实成功迁移39/39，不同请求20个（15.62%），拒绝0，失败0。其中重复成功迁移19次、反向6次，不能把39次迁移称为39个不同请求或有效收益。
回合本身及服务端生命周期审计通过，但首次CPU执行审计因对generator调用len()失败。按规则停止后续采样并清理；没有启动第二轮、没有自动补跑。原失败审计和初始失败集合报告均保留。仅修复小规模intent流的列表化，再对相同原始日志核验成功。原始请求、动作、mask、logp完全保留，未改标历史Stay/Balanced或测量数据。
以此唯一完整同版本真实rollout做了1次PPO更新（冻结设置2个epoch、10个minibatch优化步），没有轮中更新或更新后真实评测。old_logp重算最大绝对误差2.38e-07，奖励恒等式误差0。参数变化、概率/value/梯度有限、checkpoint重载输出一致。
T_ref直接从接受的replay Stay请求重算并冻结为150.425160945883秒；beta=gamma=lambda=1，真实终态V=0，所有等待/冷却均进入积分。
真实前向P95=5.867ms，特征转换P95=1.129ms，控制工作P95=290.618ms，发包延迟P95=3.171ms。接受的replay Balanced/Stay控制工作P95为311.85/260.54ms，发包P95为3.93/2.70ms；未见明显新开销，但不能据单轮证明所有负载无开销。

## P3：限定环境的开发校准

按完整trace seed分组：141103 Q15-B拟合；所有141104归同一开发验证组，配对仅用同次replay。T_ref与简单模型选择利用了已接受开发数据，不是盲测。拟合1793条有效cadence记录，开发验证1782条；分桶模型MAE=6.230ms，低复杂度回归MAE=6.503ms，选择分桶。

| 闭环模式 | mean误差 | P95误差 | cost误差 | 在途曲线误差 | 成功迁移（模拟/真实） |
|---|---:|---:|---:|---:|---:|
| Q15-B_Balanced | 1.07% | 0.11% | 0.38% | 2.61% | 45/43 |
| Q15-B_Stay | 1.10% | 2.07% | 1.65% | 1.37% | 0/0 |

模拟mean/cost收益为17.75%/15.23%；真实为19.52%/16.94%，差约1.77/1.72个百分点。迁移耗时中位数/P95误差为7.66%/12.42%，记录43条，尾部估计仅限此样本。KV占用额外诊断保存在sim_validation/kv_diagnostics.json。
给定动作回放43次全部可启动，另存replay_report.json；它不是闭环验证。闭环Stay/Balanced均从模拟自身当前状态重新决策；没有用真实未来状态校正。模拟PPO完成一次rollout/更新/保存/加载，文件simulation_only.pt明确不是真实on-policy更新。
**限制：Q15-A负收益方向未复现，模拟仍预测迁移有益，说明当前有效cadence不能充分表达不同到达率下的服务/目标干扰。P3通过仅针对Q15-B，不能称为通用迁移模拟器。** 未训练逐迁移因果收益分类器，未使用写死收益加成；未进行数值模型修正。迁移暂停不可分离，缺失率100%；采用全传输时长保守上界，不冒充实测暂停，也未猜测blocks→bytes常数。

## 验证、预算与下一步

主CPU测试46项通过；补充HTTP/桥接22项及5个子测试通过；审计修复后相关回归42项及5个子测试通过。历史48个源文件hash复核未改变。执行前版本在source_snapshot/，修复后交付版本在final_source/，差异记录在postprocessing_versions.json。
前置CPU墙钟阶段（含实现）约16.7分钟；提取/校准/模拟流水线及其两次接口修复约157.4秒。GPU含启动、预热、采样及清理共10.57分钟，低于30分钟目标和35分钟硬限；CPU未超90分钟软预算。只用1/2次真实尝试、1/1次真实更新。所属进程、Ray、GPU已清理，见cleanup_verification.json。
下一步先运行下面的只读CPU核验命令，不启动服务、采样或更新：

```bash
.venv/bin/python benchmark/run_p2p3_q15b.py verify --run-dir benchmark/results/p2p3_q15b_20261104T135200Z
```

P4建议另行冻结最多2个Q15-B cohort、最多1次更新的小预算方案，继承已更新checkpoint、不重置bias，并优先检查同请求重复/反向迁移。该方案需要下一轮明确预算授权，本轮不自动开展。
