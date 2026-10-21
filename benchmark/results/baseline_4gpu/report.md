# Llumnix 四 GPU 无扰动基准报告

## 实验状态

- 状态：`complete`
- 开始：2026-10-20T10:05:00.590269+00:00
- 完成：2026-10-20T10:59:16.833334+00:00
- 已完成正式重复：18 / 18
- 实验采用真实本地 vLLM/Llumnix 推理，无 CPU/GPU/网络/故障注入。
- 工作目录：`/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream`；结果：`/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/baseline_4gpu`。

## 环境和配置

- 四 GPU：NVIDIA GeForce RTX 3060, NVIDIA GeForce RTX 3060, NVIDIA GeForce RTX 3060, NVIDIA GeForce RTX 3060
- 模型：`/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/model/Qwen/Qwen2-0.5B`
- 每种模式保持 4 个独立实例、TP=1、rayrpc 后端、`load` 派发、`defrag` 迁移配对策略、同一权重和显存配置。
- OFF/ON 配置文件除 `MANAGER.ENABLE_ROUTINE_MIGRATION` 外逐项相同；运行日志验证有效参数，并由 Ray 状态核验 4 GPU。
- 到达过程：Poisson。每个 QPS/seed 生成一次 trace，在 OFF 和 ON 间原样复用；轨迹含固定的短/中/长输入输出长度混合。
- SLO 阈值：TTFT 2000 ms，TPOT 150 ms，沿用仓库 disturbance_exp 默认值。缺失服务端时间线从有效样本分母剔除并单列，不视为达标。
- 建议从头复现命令：`cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream && source .venv/bin/activate && python benchmark/run_baseline_4gpu.py --execute --fresh`

## 容量预实验与选定 QPS

- 稳定容量观测：≥6.550 requests/s。
- 低/中/高目标 QPS：low=1.310, medium=3.275, high=5.567。
- 预实验逐档数据见 `capacity_probe.csv`。稳定性由测量窗内完成速率与实际发送速率比较。最高探测档仍稳定，真实饱和点未测到；三档 QPS 按已观测稳定速率下界的 20%/50%/85% 设置。

## 汇总结果

每个 ON/OFF × QPS 组合包含 3 个种子；表格是三次重复的均值，重复级原始指标与标准差在 `summary.csv` 中。

| 模式 | QPS 档 | 成功请求均值 | P95 TTFT ms | P95 TPOT ms | 完成请求/s | 输出 token/s | TTFT SLO 违约率 | TPOT SLO 违约率 | 迁移操作均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | high | 677.0 | 125.265 | 10.745 | 5.642 | 542.2 | 0.000 | 0.000 | 0.0 |
| OFF | low | 158.7 | 136.971 | 11.473 | 1.322 | 126.8 | 0.000 | 0.000 | 0.0 |
| OFF | medium | 394.7 | 117.496 | 9.263 | 3.289 | 315.9 | 0.000 | 0.000 | 0.0 |
| ON | high | 677.0 | 162.103 | 17.724 | 5.642 | 542.2 | 0.000 | 0.000 | 1137.7 |
| ON | low | 158.7 | 158.647 | 11.573 | 1.322 | 126.8 | 0.000 | 0.000 | 46.0 |
| ON | medium | 394.7 | 162.765 | 17.684 | 3.289 | 315.9 | 0.000 | 0.000 | 582.7 |

## 服务质量和数据完整性

- 每条请求记录包含目标到达、实际发送、完成、状态码、成功/失败/超时和服务端 per-token 时间线（可用时）。
- TTFT/TPOT 优先来自 `/generate_benchmark` 的服务端 `per_token_latency`；没有完整时间线时保留为 N/A。
- GPU 遥测每秒覆盖四卡的利用率、显存、温度和功耗；四卡均值、峰值显存及负载离散度汇总到 `summary.csv`。
- 迁移日志：总事件 5299，成功 5148，非成功 151。可观测到实际迁移事件。
- 迁移日志不暴露实例到物理 GPU 的可靠映射时，`source_gpu` / `target_gpu` 留空，不推测填充。

## 图表

- `p95_latency.png`
- `throughput.png`
- `slo_violations.png`
- `gpu_resources.png`
- `migration.png`

## 可追溯文件

- `requests.csv`：逐请求记录，含预热和正式测量阶段。
- `gpu_metrics.csv`：四卡每秒采样。
- `migration_events.csv`：可观测的迁移完成事件及原始日志行。
- `summary.csv`：重复行以及跨种子均值/标准差。
- `config.json`、`state.json`、`configs/`、`traces/`、`server_logs/`：配置、状态、配对轨迹和运行日志。

## 限制

- 工作区当前不是挂载持久卷；Vast recycle/destroy 会删除这些本地结果。
- GPU 物理互联存在 NUMA/SYS 与 PHB 路径差异，日志中若无法可靠映射实例至 GPU，迁移事件中的 GPU 字段保留空值。
