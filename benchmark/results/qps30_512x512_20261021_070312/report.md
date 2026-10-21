# QPS 30、512 输入 / 512 输出：Migration OFF/ON

## 结果

单种子 `101`；延迟使用服务端逐 Token 时间线计算，无法取得的值保留为 N/A。`summary.csv` 保留各重复层级的汇总列，逐请求数据在 `requests.csv`。

| 模式 | 发送请求 | 成功请求 | 失败/超时 | 实际发送 QPS | 测量窗完成请求/s | 测量窗输出 token/s | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) | TTFT/TPOT 样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | 500.0 | 500.0 | 0.0 | 30.44 | 4.17 | N/A | 642.33 | 70460.00 | 40.39 | 63.49 | 32902.33 | 85833.63 | 500/500 |
| ON | 500.0 | 500.0 | 0.0 | 30.44 | 4.17 | N/A | 7425.90 | 61102.24 | 58.09 | 77.73 | 38170.52 | 90702.24 | 500/500 |

## 配置与校验

- 状态：`complete`；目标 QPS 30.0；正式请求 500 条/模式；测量 120.0 秒；随机种子 101。
- 每条请求的正式轨迹输入长度 `[512]`、输出长度 `[512]`；两种模式复用同一条轨迹，SHA-256 `39208b6c30822227dbaf509518f1158912dc0ca4264dc86f368e0247fb878fc3`。
- 服务器返回的输入/输出 token 长度按模式记录在报告内嵌数据中：`{"off": {"expected_input": [512], "expected_output": [512], "server_input": [], "server_output": [512]}, "on": {"expected_input": [512], "expected_output": [512], "server_input": [], "server_output": [512]}}`。请求使用 `ignore_eos`，目标输出长度为 512；实际返回长度以 CSV 记录为准。
- 服务端返回的 `server_input_tokens` 字段缺失，记为 N/A；输入文本在本地 trace 生成时按 tokenizer 截到 512 tokens。实际输出 token 数由响应记录。
- 迁移 ON 配置：阈值 `9000.0`、`PAIR_MIGRATION_FREQUENCY=1`、Balanced、`rayrpc`、负载指标 `remaining_steps`。OFF/ON 的启动日志确认迁移开关和参数；Ray 状态确认四卡。
- 测量阶段记录迁移事件 101 次（成功 100、非成功 1）；成功迁移平均日志耗时 `598.34` ms，耗时样本 100。这些事件来自迁移完成日志，不代表未产生日志的所有潜在尝试。
- 请求长度使 Offered output load 达到约 QPS × 512 输出 token/s；是否达到稳态取决于实测完成速率，见 CSV 和 GPU 采样。
- 每个正式请求的客户端 E2E 从实际发送计至完整响应或传输错误返回；TTFT/TPOT 使用服务端逐 Token 时间线。延迟统计覆盖正式轨迹中的全部请求，包括测量窗结束后完成的请求。
- TTFT、TPOT、E2E 百分位以正式测量阶段发送且最终成功的请求计算；HTTP/传输错误单独记录。客户端没有总请求、连接或读取超时，停止发送后无期限等待所有请求任务返回再汇总。
- 预热 30.0 秒，预热请求单独记录、不计入每种模式的 500 条正式请求；停止正式发送后持续排空，直到所有正式请求任务结束。

## 图表

- `plots/p95_latency.png`
- `plots/mean_latency.png`
- `plots/throughput.png`
- `plots/slo_violations.png`
- `plots/gpu_resources.png`
- `plots/latency_percentiles.png`

## 文件

- `requests.csv`：逐请求数据，包括期望与服务端 token 长度、TTFT/TPOT、E2E、成功/超时状态。
- `gpu_metrics.csv`：每 GPU 利用率和显存采样。
- `migration_events.csv`：可解析的迁移事件。
- `summary.csv`：OFF/ON 运行指标及统计汇总。
- `config.json`、`state.json`、`configs/`、`traces/`、`server_logs/`：配置、状态、共用请求轨迹及日志。

## 复现

```bash
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
.venv/bin/python benchmark/run_qps5_512x512_off_on.py --seed 101 --qps 30.0 --request-count 500 --measure-sec 120.0 --warmup-sec 30.0
```

## 实例级 KV Cache 与排队指标

以下指标从 Llumnix 的 `*_instance_instance.csv` 中按请求测量窗截取（每种模式 120 秒）；样本数和采样间隔见 `instance_metrics_summary.csv`。

| 模式 | 实例样本 | KV Cache 均值 | KV Cache P95 | 平均 total blocks | 平均 used blocks | 平均 free blocks | 平均运行请求/实例 | 平均等待请求/实例 | 等待请求最大值 | `num_killed_requests` 最大值 | 非零样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | 2542 | 7.382% | 10.228% | 32988 | 2435.2 | 30552.8 | 50.32 | 50.157 | 238 | 0 | 0 |
| ON | 5057 | 3.573% | 10.168% | 32988 | 1178.7 | 31809.3 | 24.47 | 21.406 | 211 | 0 | 0 |

各实例 KV Cache 峰值和排队情况：

| 模式 | 实例 | GPU | KV 均值 | KV P95 | KV 最大值 | 平均运行请求 | 平均等待请求 | 最大等待请求 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OFF | Instance 1 | N/A | 6.100% | 9.768% | 10.137% | 41.05 | 0.59 | 3 |
| OFF | Instance 2 | N/A | 6.385% | 10.283% | 10.413% | 43.10 | 0.54 | 2 |
| OFF | Instance 3 | N/A | 4.411% | 8.004% | 9.118% | 32.22 | 0.06 | 2 |
| OFF | Instance 4 | N/A | 8.483% | 10.329% | 10.725% | 57.79 | 86.57 | 238 |
| ON | Instance 1 | N/A | 2.263% | 9.052% | 10.716% | 15.22 | 0.28 | 6 |
| ON | Instance 2 | N/A | 8.602% | 10.518% | 10.789% | 58.66 | 83.14 | 211 |
| ON | Instance 3 | N/A | 1.939% | 9.384% | 10.583% | 13.57 | 1.57 | 26 |
| ON | Instance 4 | N/A | 1.456% | 8.589% | 10.555% | 10.20 | 0.26 | 12 |

GPU 计算活动均值来自精确墙钟测量窗内的 `gpu_metrics.csv` 样本；峰值显存取四卡各自峰值中的最大值：

| 模式 | 四卡平均 GPU 利用率 | 四卡最高显存占用 |
|---|---:|---:|
| OFF | 11.98% | 8415 MiB |
| ON | 28.19% | 8329 MiB |

迁移指标只计入正式测量阶段有完成日志的迁移操作：

| 模式 | 记录迁移数 | 成功 | 非成功 | 成功迁移平均耗时 (ms) | P95 耗时 (ms) | 耗时样本数 |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 0 | 0 | 0 | N/A | N/A | 0 |
| ON | 101 | 100 | 1 | 598.34 | 891.75 | 100 |

说明：逐实例样本保留了 `gpu_cache_usage`、total/used/free GPU blocks、运行/等待请求数及 `num_killed_requests`。块数是否来自原始字段或由缓存比例推导，见逐样本 CSV 的 `block_counts_source`。

`num_killed_requests` 在当前 vLLM scheduler 中是采样时的抢占相关请求 gauge，不是累计抢占事件计数；因此这里只报告采样最大值和非零样本数。等待队列、KV 利用率均按实例快照统计。

样本时间是管理器轮询得到的实例快照，不是固定频率重采样；图中保留原始点。`Instance 1–4` 是各模式内按首次出现时间命名，不代表 GPU 编号。GPU 利用率和显存图来自独立的 `nvidia-smi` 四卡采样。GPU 和迁移原始 CSV 的 `phase` 已按精确测量窗校正，采集器原始阶段标签保存在 `logged_phase`。
迁移耗时取迁移完成日志中的端到端操作耗时，不等于被迁移请求的精确停顿时间；事件数也只覆盖可解析的迁移完成日志。

新增图表：
- `plots/instance_kv_cache.png`
- `plots/instance_request_pressure.png`
- `plots/instance_kv_cache_per_instance.png`
- `plots/instance_waiting_queue.png`
- `plots/gpu_utilization_memory_summary.png`
- `plots/migration_metrics.png`
- `migration_metrics_summary.csv` / `migration_measurement_events.csv`：正式测量窗迁移事件统计及原始筛选记录。
