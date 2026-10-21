> **已被后续修正版实验取代，不用于正式对照。** 该试运行的客户端连接池上限为 512，使部分请求在本机连接池排队，没有按目标 QPS 到达 Llumnix。

# QPS 15、1024 输入 / 1024 输出：Migration OFF/ON，ENABLE_DEFRAG=False

## 结果

使用同一条 Poisson 到达轨迹配对比较 Migration OFF/ON。服务端逐 Token 时间线用于 TTFT/TPOT；客户端 E2E 从实际发送计至完整响应或传输错误。百分位只使用成功请求；完整请求状态见 `requests.csv`。

| 模式 | 发送请求 | 成功 | 失败 | 实际发送 QPS | 测量窗完成请求/s | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | 2684 | 629 | 2055 | 14.91 | 0.57 | 464.32 | 3553.77 | 14.20 | 94.01 | 254048.45 | 393861.36 |
| ON | 2684 | 616 | 2068 | 14.92 | 0.67 | 385.96 | 1474.01 | 111.42 | 432.57 | 287335.42 | 443107.02 |

## 请求完整性

| 模式 | 轨迹请求数 | 已发送 | 成功响应 | 失败/超时 | 排空上限触发超时 | 无完成时间 |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 2684 | 2684 | 629 | 2055 | 2055 | 2055 |
| ON | 2684 | 2684 | 616 | 2068 | 2068 | 2068 |

- 本次状态：`partial`。目标 QPS 15.0；实际轨迹在 180.0 秒测量窗生成，seed `101`。两种模式的轨迹 SHA-256：`8e576d61aadc5b76eb029ced01f5f89eb2a5485e11b994f3fc2b58714de1f8d2`。
- 输入/输出目标长度均为 1024 token。模型上下文长度为 2048；请求使用 `ignore_eos`，返回的实际输出长度按请求记录。
- 启动配置：Qwen2-0.5B、4 个 Llumnix 实例（每卡一个，TP=1）、`MAX_NUM_SEQS=128`、`GPU_MEMORY_UTIL=0.5`，Balanced 配对策略，`ENABLE_DEFRAG=False`。
- 预热 60.0 秒；正式测量 180.0 秒；每个阶段请求无 HTTP 总时限。正式测量结束后最多排空 300.0 秒；达到上限仍未结束的任务在 CSV 中标记，不纳入成功延迟分位数。
- OFF/ON 配置文件除迁移开关外相同；运行时同时核对四卡 Ray 资源、4 个实例 ready 日志及 Manager 实际登记的实例数。整组核验结果：`True`；未通过的模式不会作为有效四实例测量纳入汇总。

## 资源、排队、KV cache 与迁移遥测

沿用 qps30 的采集流程：`gpu_metrics.csv` 逐秒记录四张 GPU 的利用率、显存、温度和功耗；Llumnix 实例日志汇总运行/等待请求数、KV cache 使用率及 GPU blocks；迁移日志记录可解析的迁移完成事件。`analyze_instance_metrics.py` 会生成相同口径的汇总 CSV 和图表。

## 图表

- `plots/p95_latency.png`
- `plots/mean_latency.png`
- `plots/throughput.png`
- `plots/slo_violations.png`
- `plots/gpu_resources.png`
- `plots/latency_percentiles.png`

## 文件

- `requests.csv`：预热与正式测量的逐请求时间戳、状态、E2E、TTFT/TPOT 和 token 数。
- `gpu_metrics.csv`：四卡 GPU 遥测；`instance_metrics.csv`：实例级队列和 KV cache 原始采样。
- `summary.csv`、`instance_metrics_summary.csv`、`gpu_metrics_summary.csv`、`migration_metrics_summary.csv`：汇总数据。
- `instance_gpu_metrics.csv`、`instance_gpu_summary.csv`：实例到 GPU 的可确认映射和统计。
- `migration_events.csv`、`migration_measurement_events.csv`：原始迁移日志与测量窗事件。
- `config.json`、`state.json`、`configs/`、`traces/`、`server_logs/`：实验配置、状态、共用轨迹和服务日志。

## 复现

```bash
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
.venv/bin/python benchmark/run_qps15_1024x1024_no_defrag.py
```

## 实例级 KV Cache 与排队指标

以下指标从 Llumnix 的 `*_instance_instance.csv` 中按请求测量窗截取（每种模式 180 秒）；样本数和采样间隔见 `instance_metrics_summary.csv`。

| 模式 | 实例样本 | KV Cache 均值 | KV Cache P95 | 平均 total blocks | 平均 used blocks | 平均 free blocks | 平均运行请求/实例 | 平均等待请求/实例 | 等待请求最大值 | `num_killed_requests` 最大值 | 非零样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | 3681 | 17.351% | 38.018% | 22095 | 3833.7 | 18261.3 | 43.86 | 4.668 | 78 | 0 | 0 |
| ON | 4703 | 40.904% | 55.207% | 22095 | 9037.8 | 13057.2 | 103.50 | 2.456 | 16 | 0 | 0 |

各实例 KV Cache 峰值和排队情况：

| 模式 | 实例 | GPU | KV 均值 | KV P95 | KV 最大值 | 平均运行请求 | 平均等待请求 | 最大等待请求 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OFF | Instance 1 | N/A | 10.718% | 27.520% | 29.332% | 27.33 | 0.15 | 10 |
| OFF | Instance 2 | N/A | 26.206% | 41.801% | 43.974% | 65.85 | 11.29 | 78 |
| OFF | Instance 3 | N/A | 13.096% | 26.771% | 32.487% | 33.89 | 1.28 | 37 |
| OFF | Instance 4 | N/A | 11.921% | 27.661% | 29.871% | 29.73 | 0.07 | 7 |
| ON | Instance 1 | N/A | 40.197% | 54.107% | 54.863% | 95.90 | 0.10 | 9 |
| ON | Instance 2 | N/A | 41.397% | 55.800% | 56.488% | 106.27 | 3.29 | 16 |
| ON | Instance 3 | N/A | 40.991% | 55.198% | 57.687% | 105.89 | 4.25 | 10 |
| ON | Instance 4 | N/A | 41.031% | 55.207% | 55.850% | 105.93 | 2.19 | 15 |

GPU 计算活动均值来自精确墙钟测量窗内的 `gpu_metrics.csv` 样本；峰值显存取四卡各自峰值中的最大值：

| 模式 | 四卡平均 GPU 利用率 | 四卡最高显存占用 |
|---|---:|---:|
| OFF | 17.34% | 6745 MiB |
| ON | 13.92% | 9393 MiB |

迁移指标只计入正式测量阶段有完成日志的迁移操作：

| 模式 | 记录迁移数 | 成功 | 非成功 | 成功迁移平均耗时 (ms) | P95 耗时 (ms) | 耗时样本数 |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 0 | 0 | 0 | N/A | N/A | 0 |
| ON | 3 | 2 | 1 | 766.51 | 800.62 | 2 |

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
