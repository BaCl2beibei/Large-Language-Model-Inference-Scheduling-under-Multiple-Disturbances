# Mig-PPO V2 CPU 物理模型优化

共 27 个完整配对（16 fit、11 development），另有未配对物理样本。全部为已见开发数据；independent_validation_pending。

| 模块 | 最终方法 | 输入 |
|---|---|---|
| Prefill | admission→engine-first 固定中位数 | 128-token prompt；排队由事件引擎计算 |
| Decode | 稳健 NNLS：截距 + batch + 总 context | 当前 batch、上下文；64-token 客户端 flush |
| KV/队列 | 离散事件、逐 Token 分配与 FCFS | prompt+已生成 tokens、容量、预留/释放 |
| 迁移 | NNLS 总时长/暂停；有样本的物理方向修正 | copied blocks、GPU→GPU；不加固定收益 |

共享 cadence 系数：[61.44618491536418, 0.47636095553433305, 0.1594258920594865]；物理 GPU 速率修正未通过开发选择，未采用。所有迁移增益来自状态推进。

误差为同一批实际 CPU 闭环的宏平均，回放不计入总览。D 使用完整 200 秒精确积分；状态按各自冻结实验起点对齐，1.5 秒保持采样，结束后补零。

| 指标 | 优化前 | 优化后 | 优化后通过 / n |
|---|---:|---:|---:|
| Mean E2E | 10.67% | 6.94% | 49/54 |
| P95 E2E | 13.89% | 7.30% | 50/54 |
| C | 12.08% | 6.84% | 50/54 |
| D pressure | 89.02% | 8.19% | 49/54 |
| Outstanding | 11.13% | 7.49% | 49/54 |
| Waiting MAE/请求 | 0.0036 | 0.0035 | 54/54 |
| KV MAE | 0.0085 | 0.0050 | 52/54 |
| Q MAE | 0.0001 | 0.0001 | 54/54 |
| H MAE | 0.0211 | 0.0088 | 52/54 |
| B* MAE | 0.0054 | 0.0019 | 54/54 |
| 迁移中位耗时 | 7.93% | 5.32% | 21/23 |
| 迁移P95耗时 | 12.65% | 14.01% | 22/23 |
| 暂停中位耗时 | 22.02% | 6.26% | 22/23 |
| 暂停P95耗时 | 129.78% | 19.74% | 18/23 |
| gain_C / pp | 13.36 | 9.94 | 23/27 |

| 环境/trace | 数据用途 | Stay C误差 | Balanced C误差 | 真gain | 仿真gain | D最差误差 | 全门 |
|---|---|---:|---:|---:|---:|---:|---|
| HOT1-L_backup_n48_q1p0_s26110523 | fit | 0.49% | 3.66% | 1.01% | 4.16% | 3.31% | 未通过 |
| HOT1-L_primary_s26110521 | fit | 75.78% | 0.52% | -68.35% | 4.73% | 83.68% | 未通过 |
| HOT1-M_primary_s26110521 | fit | 0.74% | 4.15% | 6.63% | 9.84% | 3.33% | 通过 |
| HOT2-H_backup_n48_q1p0_s26110523 | fit | 9.92% | 6.72% | 18.00% | 15.09% | 10.64% | 未通过 |
| HOT2-L_primary_s26110521 | fit | 3.93% | 1.01% | 7.53% | 4.73% | 5.72% | 未通过 |
| HOT2-M_backup_n32_q0p6_s26110522 | fit | 5.67% | 1.97% | 11.12% | 7.64% | 5.30% | 通过 |
| HOT3-H_backup_n48_q1p0_s26110523 | fit | 0.33% | 3.99% | 11.85% | 15.09% | 3.26% | 未通过 |
| HOT3-L_primary_s26110521 | fit | 0.14% | 2.35% | 2.57% | 4.73% | 2.00% | 通过 |
| HOT3-M_primary_s26110521 | fit | 0.14% | 2.69% | 7.23% | 9.85% | 2.70% | 通过 |
| HOT4-H_backup_n48_q1p0_s26110523 | fit | 0.37% | 4.12% | 11.77% | 15.09% | 3.82% | 通过 |
| HOT4-L_primary_s26110521 | fit | 1.27% | 4.13% | 1.89% | 4.73% | 2.62% | 通过 |
| HOT4-M_primary_s26110521 | fit | 1.97% | 4.18% | 7.76% | 9.85% | 2.91% | 通过 |
| NATIVE_backup_n48_q1p0_s26110523 | fit | 3.28% | 4.58% | -1.36% | 0.00% | 2.75% | 通过 |
| NATIVE_primary_s26110521 | fit | 3.50% | 3.73% | -0.24% | 0.00% | 2.07% | 通过 |
| HOT1-H_backup_n48_q1p0_s26110523_followup | fit | 61.03% | 2.39% | -40.08% | 15.09% | 65.61% | 未通过 |
| NATIVE_n48_q1p25_s26110661 | fit | 11.32% | 45.84% | 23.67% | 0.00% | 65.04% | 未通过 |
| DEVELOPMENT_HOT1-L | development_validation | 65.84% | 2.57% | -62.32% | 4.64% | 74.65% | 未通过 |
| DEVELOPMENT_HOT2-H | development_validation | 0.38% | 2.31% | 12.58% | 14.92% | 1.89% | 通过 |
| DEVELOPMENT_HOT2-L | development_validation | 0.47% | 1.07% | 3.96% | 4.54% | 1.65% | 通过 |
| DEVELOPMENT_HOT2-M | development_validation | 0.13% | 0.44% | 7.39% | 7.67% | 1.60% | 通过 |
| DEVELOPMENT_HOT3-H | development_validation | 0.19% | 2.87% | 12.24% | 14.93% | 2.38% | 通过 |
| DEVELOPMENT_HOT3-L | development_validation | 0.65% | 0.84% | 3.11% | 4.54% | 1.75% | 通过 |
| DEVELOPMENT_HOT3-M | development_validation | 0.04% | 2.78% | 6.96% | 9.52% | 2.02% | 通过 |
| DEVELOPMENT_HOT4-H | development_validation | 1.44% | 0.17% | 13.55% | 14.92% | 2.14% | 通过 |
| DEVELOPMENT_HOT4-L | development_validation | 1.70% | 2.77% | 0.15% | 4.54% | 2.11% | 未通过 |
| DEVELOPMENT_HOT4-M | development_validation | 1.02% | 0.90% | 9.41% | 9.52% | 1.95% | 通过 |
| DEVELOPMENT_NATIVE | development_validation | 2.20% | 2.51% | -0.32% | 0.00% | 1.78% | 通过 |

收益绝对值小于 2% 的配对按接近零处理，单次实验无法判断稳定符号。真实动作回放（预测成本与观测成本两种）分列于 validation_results.json，不冒充闭环结果。

开发数据全门通过配对：['HOT1-M_primary_s26110521', 'HOT2-M_backup_n32_q0p6_s26110522', 'HOT3-L_primary_s26110521', 'HOT3-M_primary_s26110521', 'HOT4-H_backup_n48_q1p0_s26110523', 'HOT4-L_primary_s26110521', 'HOT4-M_primary_s26110521', 'DEVELOPMENT_HOT2-H', 'DEVELOPMENT_HOT2-L', 'DEVELOPMENT_HOT2-M', 'DEVELOPMENT_HOT3-H', 'DEVELOPMENT_HOT3-L', 'DEVELOPMENT_HOT3-M', 'DEVELOPMENT_HOT4-H', 'DEVELOPMENT_HOT4-M']。正式 PPO ready 列表为空；尚缺未参与开发的新真实独立验证。NATIVE 始终仅作测试。

43 项相关单元测试通过（测试明细另存）；未启动 GPU 实验或 PPO 训练。
