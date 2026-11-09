# Mig-PPO V2 多环境筛选与拟合协议

Run: `mig_ppo_v2_multienv_20261105T130307Z`。正式顺序为 13 场景完整筛选、统一 CPU 拟合、冻结后新 seed 独立验证。

主成本 `C=mean(E2E)+CVaR95(E2E)`，尾部分数按精确分数权重计算；采用条件为完整同轨迹配对、`gain_C>0` 且 Balanced 至少一笔有请求/源目标/blocks/终态的成功迁移。

首选 N=48、0.6 req/s；仅超时走 N=32/0.6，完整但无合格收益走 N=48/1.0。单轮硬窗 200 秒，所有策略统一 1.5 秒决策、0 秒冷却。备选失败则跳过。

人工热点强度只改变初始路由占比；NATIVE 不发送 `initial_instance_id`，并核对 Manager 原生 dispatch 与首次 engine admission。NATIVE 固定 test_only。

奖励阈值只在最终选定的人工热点拟合快照中按预注册 Youden 规则校准；数据只有一个标签时保持 pending。验证 trace 不能用于改模型或阈值。正式 PPO 训练、真实 PPO 更新和论文最终评测均不在本轮。
