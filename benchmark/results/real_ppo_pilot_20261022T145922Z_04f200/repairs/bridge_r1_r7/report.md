# Repair 01：R1–R7 CPU 修复

本次只修复 04 failed 审查确认的七项问题；没有启动 GPU/Ray/模型、PPO、机会实验或下一阶段。原 reviews/*、handoff/04.* 及旧 gates 保持原字节。

| 问题 | 最小根因与修改 | 复现与新增覆盖 |
| --- | --- | --- |
| R1 | 同进程多角色回执遍历顺序产生水位倒退。逐角色验证后，同轮按 producer identity 合并可比较的最大 sequence/through 水位；台账跨轮回退检测不变。 | 原 R1、独立 worker / 同进程、跨轮 regression。 |
| R2 | unknown 后环境仍提交新 ID。env 与 transport 保存 unresolved fence；新非 Stay 前按旧 ID 查询，未确定即停止采样并有界清理。原 action/logp/unknown 保留，决策日志逐条 fsync。 | 原 R2；直接 transport、环境退出清理、已知终态后才允许新提交。 |
| R3 | 获取租约时 RPC await 窗口允许 routine 入场。首次 await 前 acquisition fence，授予前复核 native task/fault/intervention；异常 finally 释放 acquisition fence。 | 原 R3；pre-stop intervention、provider 异常。 |
| R4 | API 最近输出路由不能证明迁移后 owner。Manager 在 dispatch 端及全部四实例安装永久每 epoch tombstone；晚到 generate / reserve / prealloc / receive / commit 不能创建请求。跟踪 dispatch、入站传输、generating 和受控迁移；在原 engine step/output-processing barrier 后 abort 并检查 running/waiting/swapped、paused、preallocated、migration sets / pending、request_ids 和 delivery。Manager 再核对四实例 ID / epoch / 健康；API 等队列清空、HTTP stream done 后才 confirmed。失联、重建、未结束操作或未知迁移保持 unknown。real-* 请求不旁路 Manager 重派。保留实际 engine 生产事件，fence 后不再投递该请求的新输出。 | 原 R4；目标已运行而源无请求、暂停/预分配、晚到 dispatch/commit、in-flight engine barrier、RPC 不确定、epoch 变化、投递未空、unproven server outcome。 |
| R5 | reset 只检查引擎快照。固定四实例 TP1 角色契约要求 API / Manager / local / worker 的 durable passed 及显式 pending=0；prepare 有 cleanup_sec 超时，失败立即阻断 begin。Manager pending 也包含 dispatch task。 | 原 R5；各角色 pending 缺失/非零。 |
| R6 | CLI 未强制独立审查与 live 顺序。live plan 在服务/预算操作前检查 hash 绑定的 04 passed；telemetry → shadow → active → overhead 的 gate、结果与传递依赖 hash 逐项核验。真实结果和 owned cleanup 后由程序产出 p1_live / p2_shadow_live / p2_active_live / p2_live，已有尝试不覆盖。dry-run 只读时钟并列出阻塞。 | 原 R6；review 缺失/failed/incomplete/stale、前序缺失、结果 hash 改变及顺序。 |
| R7 | 失败 worker 无 source/durable 被递归丢弃。先依据固定四实例 TP1 角色数量建立必需角色，再验证完整结构及每个 barrier，最后才合并 producer；失败回执落盘保留。 | 原 R7；pending_cuda_telemetry、writer_closed、tracing_disabled、缺失/多余 worker。 |

CPU：78 项回归 + 原 9 项独立审查案例通过（含原 7 项反例及请求/原 coordinator 成功链）。原审查测试源码未改；唯一会写 reviews/migration_chain.json 的测试输出通过 Path.write_text 包装重定向到 repairs/bridge_r1_r7/review_migration_chain_after.json，原失败证据逐字节校验。

verify_cpu.py 首次包装执行退出 1：将 supervisorctl status 的退出 3 当成必需成功检查。原输出显示既有 jupyter/pyworker EXITED；78+9 测试、编译及 diff 均通过。该包装失败单独保存 verification_attempt_01.json / commands_attempt_01.json；修正为只读清单后复用已完成测试证据完成核验，没有操作服务、重跑实验或隐藏失败。

源码差异以 repair_01.patch 为准（相对独立审查输入，保留既有修改）。before/after 保存相关源码；source_manifest_after.json 已更新并保留先前副本。commands/05_smoke.json 仅更新源码绑定和说明，四个 argv/env/config/seed/trace/时长不变；原版在 before/commands_05_smoke.json，旧审查对应该版。

CPU gate 使用 gates/p1_offline_revalidated_repair_01.json、gates/p2_offline_revalidated_repair_01.json；旧 gates 不改，因行为源码改变只作为历史材料。下一步是 04 独立重审。reviews/bridge_review.json 仍 failed，05 不可执行；尚无任何真机验收，p1_live / p2_live 尚未生成。
