# 下一步（默认不启动GPU）

先只读复核：

```bash
.venv/bin/python benchmark/run_p2p3_q15b.py verify --run-dir benchmark/results/p2p3_q15b_20261104T135200Z
```

核验通过后，P2支持下一轮小预算真实链路开发；P3只支持本轮固定Q15-B的受限模拟预训练开发。不得把Q15-A负收益诊断失败省略，不声称PPO优于Balanced。下一项优先分析39次成功迁移为何只覆盖20个请求（19次重复、6次反向），不要加迁移奖励或强制均匀动作。

P4建议独立新目录、检查未使用的新seed、继承real_updated.pt，最多2个128请求Q15-B cohort、最多1次choice_only更新，GPU30分钟目标/35分钟硬限和8分钟启动预留不变。先形成配置与预算，再取得下一轮授权；当前没有自动扩训命令。不要在已封账目录运行launch/run/resume-cpu。

精确执行版本、初始失败审计、重新审计与恢复脚本均保留；recover_cpu_audit_and_update.py包含排他更新guard，不可删除guard再次更新。最终verify无副作用。
