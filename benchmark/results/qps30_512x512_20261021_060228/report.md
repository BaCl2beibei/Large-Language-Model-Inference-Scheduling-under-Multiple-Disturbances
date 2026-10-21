# QPS 30、512 输入 / 512 输出：Migration OFF/ON

## 结果

单种子 `101`；延迟使用服务端逐 Token 时间线计算，无法取得的值保留为 N/A。`summary.csv` 保留各重复层级的汇总列，逐请求数据在 `requests.csv`。

| 模式 | 发送请求 | 成功请求 | 失败/超时 | 实际发送 QPS | 测量窗完成请求/s | 测量窗输出 token/s | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) | TTFT/TPOT 样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | 2983.0 | 1135.0 | 1848.0 | 29.83 | 6.46 | N/A | 29430.98 | 153046.72 | 16.89 | 22.95 | 41576.31 | 175630.14 | 1135/1135 |
| ON | 2983.0 | 1038.0 | 1945.0 | 29.83 | 5.41 | N/A | 34457.61 | 153274.66 | 17.54 | 23.85 | 46049.47 | 168862.15 | 1038/1038 |

## 配置与校验

- 状态：`complete`；目标 QPS 30.0；测量 100.0 秒；随机种子 101。
- 每条请求的正式轨迹输入长度 `[512]`、输出长度 `[512]`；两种模式复用同一条轨迹，SHA-256 `a497e437419a10b76350759f88906a459f86ce08fa0122ef79654f4428468cdb`。
- 服务器返回的输入/输出 token 长度按模式记录在报告内嵌数据中：`{"off": {"expected_input": [512], "expected_output": [512], "server_input": [], "server_output": [512.0]}, "on": {"expected_input": [512], "expected_output": [512], "server_input": [], "server_output": [512.0]}}`。请求使用 `ignore_eos`，目标输出长度为 512；实际返回长度以 CSV 记录为准。
- 服务端返回的 `server_input_tokens` 字段缺失，记为 N/A；输入文本在本地 trace 生成时按 tokenizer 截到 512 tokens。实际输出 token 数由响应记录。
- 迁移 ON 配置：阈值 `9000.0`、`PAIR_MIGRATION_FREQUENCY=1`、Balanced、`rayrpc`、负载指标 `remaining_steps`。OFF/ON 的启动日志确认迁移开关和参数；Ray 状态确认四卡。
- 测量阶段记录迁移事件 5 次（成功 5、非成功 0）；成功迁移平均日志耗时 `570.32` ms，耗时样本 5。这些事件来自迁移完成日志，不代表未产生日志的所有潜在尝试。
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
| OFF | 4363 | 2.111% | 2.697% | 34962 | 738.1 | 34223.9 | 15.51 | 91.400 | 339 | 0 | 0 |
| ON | 5034 | 2.147% | 2.792% | 34962 | 750.8 | 34211.2 | 15.61 | 96.823 | 301 | 0 | 0 |

各实例 KV Cache 峰值和排队情况：

| 模式 | 实例 | GPU | KV 均值 | KV P95 | KV 最大值 | 平均运行请求 | 平均等待请求 | 最大等待请求 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OFF | Instance 1 | N/A | 2.140% | 2.651% | 2.766% | 15.71 | 82.25 | 126 |
| OFF | Instance 2 | N/A | 2.153% | 2.697% | 2.772% | 15.73 | 242.06 | 339 |
| OFF | Instance 3 | N/A | 2.098% | 2.712% | 2.794% | 15.49 | 18.91 | 57 |
| OFF | Instance 4 | N/A | 2.053% | 2.746% | 2.929% | 15.09 | 20.16 | 65 |
| ON | Instance 1 | N/A | 2.142% | 2.837% | 2.929% | 15.72 | 81.00 | 136 |
| ON | Instance 2 | N/A | 2.203% | 2.777% | 2.849% | 15.81 | 249.68 | 301 |
| ON | Instance 3 | N/A | 2.130% | 2.769% | 2.852% | 15.51 | 29.49 | 70 |
| ON | Instance 4 | N/A | 2.114% | 2.606% | 2.886% | 15.40 | 26.97 | 53 |

GPU 计算活动均值来自精确墙钟测量窗内的 `gpu_metrics.csv` 样本；峰值显存取四卡各自峰值中的最大值：

| 模式 | 四卡平均 GPU 利用率 | 四卡最高显存占用 |
|---|---:|---:|
| OFF | 34.81% | 7953 MiB |
| ON | 32.81% | 7961 MiB |

迁移指标只计入正式测量阶段有完成日志的迁移操作：

| 模式 | 记录迁移数 | 成功 | 非成功 | 成功迁移平均耗时 (ms) | P95 耗时 (ms) | 耗时样本数 |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 0 | 0 | 0 | N/A | N/A | 0 |
| ON | 5 | 5 | 0 | 570.32 | 918.06 | 5 |

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
