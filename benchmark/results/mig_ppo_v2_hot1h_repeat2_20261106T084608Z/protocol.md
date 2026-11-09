# HOT1-H 新 trace 复测

Run `mig_ppo_v2_hot1h_repeat2_20261106T084608Z`。沿用相同环境设置：Qwen2-1.5B 四卡、48 请求、1 req/s、128 输入 token、1536 输出 token、1.5 s 决策、0 s 冷却、200 s 单轮窗口。热点为逻辑实例 1，43/48 请求（89.583%）。

本次使用新 arrival seed `26110681` 和 prompt seed `26110682`；Stay/Balanced 共用同一新 trace。若完整配对成本收益为正且有成功迁移审计，即判通过。独立真实验证、拟合及 PPO 更新均不在范围内。
