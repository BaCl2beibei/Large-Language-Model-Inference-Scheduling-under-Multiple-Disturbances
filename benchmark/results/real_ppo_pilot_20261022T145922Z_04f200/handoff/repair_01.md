# Repair 01：CPU passed，独立审查待重验

已修复 R1–R7；78 项 CPU 回归及原 9 项独立审查案例通过，编译和 diff 检查通过。详细根因、修改、测试与边界见 `repairs/bridge_r1_r7/report.md`，准确源码差异见 `repair_01.patch`。

原 failed 审查、04 交接、旧 gate 和失败反例均保留。当前 CPU gate 为 `gates/p1_offline_revalidated_repair_01.json`、`gates/p2_offline_revalidated_repair_01.json`，共享指针已引用。命令仅更新源码 hash，冻结 argv/env/seed/trace/时长及所有验收阈值不变。

四条真实 CLI dry-run 均退出 0、GPU_will_start=false，唯一阻塞均为 failed 独立审查；四个 live plan 均在预算/服务操作之前拒绝。未启动模型、Ray、GPU 服务、PPO 或下一阶段，实验时钟字节未变，总剩余 28800 秒（P0–P3 7200、P4 7200、P5 14400）。

一次核验包装脚本误将既有 EXITED 服务导致的 supervisorctl status 退出 3 当作失败；失败记录已保存，修正为只读清单后复用既有成功测试结果完成核验。未触碰管理服务。

下一步交给独立 04 CPU 审查。审查状态仍 failed；05 GPU smoke 不可执行。交接详情、源码/配置/证据 hash 与实际命令/退出码见 `repair_01.json`。完成本次限定修复并停止。
