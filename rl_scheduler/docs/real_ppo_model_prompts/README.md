# Mig-PPO 模型交接 Prompt 索引

本目录只提供任务 prompt，目前未执行其中任何代码实施、GPU 实验或训练。

每个文件都包含项目路径、共享约束、目标、验收产物和停止位置，可以独立交给指定模型。先在界面/API 选择模型及推理强度，再粘贴对应文件；不要同时启动多个步骤。

与主任务书的关系：本交接流程把阶段拆成“实现 → 审查/执行 → 判定”。本次只执行交付的一个步骤，不再沿用主任务书的全流程自动推进指令。

| 顺序 | 模型 | 强度 | Prompt |
|---|---|---|---|
| 01 | GPT-6 Luna | medium | [P0：保存现场与建立交接目录](01_luna_p0.md) |
| 02 | GPT-6.1 Sol | high | [P1：请求台账与评分语义](02_sol_p1.md) |
| 03 | GPT-6.1 Sol | high | [P2：动作桥接与真实环境实现](03_sol_p2.md) |
| 04 | GPT-6 Astra | high | [首次审查：指标与迁移控制](04_astra_review_bridge.md) |
| 05 | GPT-6 Luna | medium | [P1/P2 真机 smoke 与证据采集](05_luna_live_smoke.md) |
| 06 | GPT-6.1 Sol | high | [P3：机会实验设计与脚本准备](06_sol_p3_prepare.md) |
| 07 | GPT-6 Luna | medium | [P3：运行固定机会实验](07_luna_opportunity.md) |
| 08 | GPT-6.1 Sol | high | [P3：判断收益空间并冻结奖励](08_sol_p3_decision.md) |
| 09 | GPT-6.1 Sol | high | [P4：实现真实 PPO 训练器](09_sol_p4_implement.md) |
| 10 | GPT-6 Astra | high | [第二次审查：PPO 数学与训练时序](10_astra_review_ppo.md) |
| 11 | GPT-6.1 Sol | high | [P4：有界真机训练与开发评估](11_sol_train.md) |
| 12 | GPT-6 Luna | medium | [P5：运行冻结的独立评估](12_luna_final_eval.md) |
| 13 | GPT-6.1 Sol | high | [P5：配对分析与最终报告草稿](13_sol_analysis.md) |
| 14 | GPT-6 Astra | high | [最终审查：结论与证据一致性](14_astra_final_review.md) |

使用建议：Sol 主线尽量保留同一会话；Astra 审查用独立会话并读取实际证据；Luna 每次只拿当前运行步骤。不是要求 14 个模型并行，而是按顺序进行 14 次有范围的交接。

共享指针由第 01 步创建在 `rl_scheduler/runs/real_ppo_handoff/active_run.json`。每步结果通过 run_dir 的 handoff/gates/reviews 交接，不能只依赖聊天总结。

若前一步 failed/incomplete，不继续执行后面的正常 prompt。先交给 [Sol 修复模板](90_sol_repair.md)；两轮仍无可靠根因时用 [Astra 定点诊断模板](91_astra_diagnosis.md)。修改后回到受影响的审查/验收步骤，不重跑整套无关测试或全部 GPU 实验。

GPU 步骤只有 05、07、11、12；每步结束释放本任务 GPU 服务，防止交接期间耗尽预算。预算跨步骤累计，禁止重置。

简短发送方式示例（模型已在界面选好）：

> 请完整读取并执行 /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/rl_scheduler/docs/real_ppo_model_prompts/01_luna_p0.md。仅执行该文件限定步骤，完成后写入交接文件并停止。

如使用不能访问同一文件系统的工具，必须提供对应 prompt、主任务书、所需源码与前一步证据；仅复制绝对路径不能提供访问权限。

这些 prompt 采用明确任务边界、相关上下文和可验证产物的写法，参考 [OpenAI 官方提示指南](https://developers.openai.com/api/docs/guides/prompt-engineering)。具体阶段设计来自本项目任务书，并非官方对 Mig-PPO 的验证结论。

