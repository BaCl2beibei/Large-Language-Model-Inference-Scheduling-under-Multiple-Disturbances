# QPS 30、512 输入 / 512 输出：Migration OFF/ON

## 结果

单种子 `101`；延迟使用服务端逐 Token 时间线计算，无法取得的值保留为 N/A。`summary.csv` 保留各重复层级的汇总列，逐请求数据在 `requests.csv`。

| 模式 | 发送请求 | 成功请求 | 失败/超时 | 实际发送 QPS | 测量窗完成请求/s | 测量窗输出 token/s | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) | TTFT/TPOT 样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | 2983.0 | 1253.0 | 1730.0 | 29.83 | 7.97 | 4080.64 | 14488.36 | 152187.20 | 14.81 | 17.00 | 30484.78 | 170382.52 | 1253/1253 |
| ON | 2983.0 | 1318.0 | 1665.0 | 29.83 | 7.42 | 3799.04 | 25980.79 | 159068.30 | 15.92 | 19.89 | 41505.11 | 171025.16 | 1318/1318 |

## 配置与校验

- 状态：`complete`；目标 QPS 30.0；测量 100.0 秒；随机种子 101。
- 每条请求的正式轨迹输入长度 `[512]`、输出长度 `[512]`；两种模式复用同一条轨迹，SHA-256 `a497e437419a10b76350759f88906a459f86ce08fa0122ef79654f4428468cdb`。
- 服务器返回的输入/输出 token 长度按模式记录在报告内嵌数据中：`{"off": {"expected_input": [512], "expected_output": [512], "server_input": [], "server_output": [512.0]}, "on": {"expected_input": [512], "expected_output": [512], "server_input": [], "server_output": [512.0]}}`。请求使用 `ignore_eos`，目标输出长度为 512；实际返回长度以 CSV 记录为准。
- 服务端返回的 `server_input_tokens` 字段缺失，记为 N/A；输入文本在本地 trace 生成时按 tokenizer 截到 512 tokens。实际输出 token 数由响应记录。
- 迁移 ON 配置：阈值 `9000.0`、`PAIR_MIGRATION_FREQUENCY=1`、Balanced、`rayrpc`、负载指标 `remaining_steps`。OFF/ON 的启动日志确认迁移开关和参数；Ray 状态确认四卡。
- 测量阶段记录迁移事件 1 次（成功 1、非成功 0）；成功迁移平均日志耗时 `246.31` ms，耗时样本 1。这些事件来自迁移完成日志，不代表未产生日志的所有潜在尝试。
- 请求长度使 Offered output load 达到约 QPS × 512 输出 token/s；是否达到稳态取决于实测完成速率，见 CSV 和 GPU 采样。
- 测量窗完成请求/s 只计请求发送后 100 秒内完成的请求；`summary.csv` 另有排空后完成请求数/100 秒，不能与测量窗吞吐混为一谈。
- TTFT、TPOT、E2E 百分位以正式测量阶段发送且最终成功的请求计算；表中 TTFT/TPOT 样本数分别给出有效样本数。高超时率意味着延迟百分位只描述成功请求，需与失败/超时数一起解读。
- 预热 5.0 秒；停止发送后最多排空 600.0 秒。未排空请求计为超时/失败，不估算时延。

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
.venv/bin/python benchmark/run_qps5_512x512_off_on.py --seed 101 --qps 30.0 --measure-sec 100.0 --warmup-sec 5.0 --drain-sec 600.0
```

## 实例级 KV Cache 与排队指标

以下指标从 Llumnix 的 `*_instance_instance.csv` 中按请求测量窗截取（每种模式 100 秒）；样本数和采样间隔见 `instance_metrics_summary.csv`。

| 模式 | 实例样本 | KV Cache 均值 | KV Cache P95 | 平均 total blocks | 平均 used blocks | 平均 free blocks | 平均运行请求/实例 | 平均等待请求/实例 | 等待请求最大值 | `num_killed_requests` 最大值 | 非零样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | 6766 | 2.189% | 2.737% | 34960 | 765.2 | 34195.1 | 15.86 | 99.395 | 414 | 0 | 0 |
| ON | 6691 | 2.179% | 2.784% | 34960 | 761.8 | 34198.5 | 15.81 | 99.821 | 386 | 0 | 0 |

各实例 KV Cache 峰值和排队情况：

| 模式 | 实例 | GPU | KV 均值 | KV P95 | KV 最大值 | 平均运行请求 | 平均等待请求 | 最大等待请求 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OFF | Instance 1 | 0 | 2.188% | 2.761% | 2.804% | 15.83 | 31.78 | 54 |
| OFF | Instance 2 | 2 | 2.186% | 2.709% | 2.789% | 15.86 | 11.21 | 22 |
| OFF | Instance 3 | 3 | 2.189% | 2.720% | 2.803% | 15.84 | 15.25 | 51 |
| OFF | Instance 4 | 1 | 2.192% | 2.760% | 2.843% | 15.89 | 339.61 | 414 |
| ON | Instance 1 | 3 | 2.166% | 2.735% | 2.832% | 15.71 | 2.85 | 10 |
| ON | Instance 2 | 0 | 2.181% | 2.781% | 2.858% | 15.83 | 14.31 | 29 |
| ON | Instance 3 | 2 | 2.190% | 2.783% | 2.837% | 15.88 | 332.44 | 386 |
| ON | Instance 4 | 1 | 2.178% | 2.840% | 2.895% | 15.80 | 49.88 | 74 |

GPU 计算活动均值来自精确墙钟测量窗内的 `gpu_metrics.csv` 样本；峰值显存取四卡各自峰值中的最大值：

| 模式 | 四卡平均 GPU 利用率 | 四卡最高显存占用 |
|---|---:|---:|
| OFF | 42.34% | 7953 MiB |
| ON | 38.97% | 7961 MiB |

实例与 GPU 的对应关系来自每个 Llumnix actor 的 Ray accelerator ID，并与 `nvidia-smi` 的 GPU index 对齐：

| 模式 | 实例 | GPU | 平均 GPU 利用率 | P95 GPU 利用率 | 峰值显存 (MiB) |
|---|---|---:|---:|---:|---:|
| OFF | Instance 1 | 0 | 43.12% | 57.65% | 7953 |
| OFF | Instance 4 | 1 | 39.65% | 53.90% | 7903 |
| OFF | Instance 2 | 2 | 43.33% | 53.65% | 7953 |
| OFF | Instance 3 | 3 | 43.24% | 53.65% | 7953 |
| ON | Instance 1 | 3 | 40.27% | 56.00% | 7961 |
| ON | Instance 3 | 2 | 37.12% | 43.65% | 7961 |
| ON | Instance 2 | 0 | 39.69% | 47.65% | 7941 |
| ON | Instance 4 | 1 | 38.81% | 47.65% | 7931 |

迁移指标只计入正式测量阶段有完成日志的迁移操作：

| 模式 | 记录迁移数 | 成功 | 非成功 | 成功迁移平均耗时 (ms) | P95 耗时 (ms) | 耗时样本数 |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 0 | 0 | 0 | N/A | N/A | 0 |
| ON | 1 | 1 | 0 | 246.31 | 246.31 | 1 |

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
- `plots/instance_gpu_utilization.png`
- `plots/migration_metrics.png`
- `instance_gpu_metrics.csv` / `instance_gpu_summary.csv`：Ray 实例 GPU 映射及对应利用率样本/汇总。
- `migration_metrics_summary.csv` / `migration_measurement_events.csv`：正式测量窗迁移事件统计及原始筛选记录。
