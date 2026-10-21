# QPS 5、512 输入 / 512 输出：Migration OFF/ON

## 结果

单种子 `101`；延迟使用服务端逐 Token 时间线计算，无法取得的值保留为 N/A。`summary.csv` 保留各重复层级的汇总列，逐请求数据在 `requests.csv`。

| 模式 | 发送请求数 | 成功请求 | 超时 | 实际发送 QPS | 测量窗完成 QPS | P95 E2E (ms) | Mean TTFT (ms) | P95 TTFT (ms) | Mean TPOT (ms) | P95 TPOT (ms) | 成功迁移数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | 507.0 | 507.0 | 0.0 | 5.07 | 4.81 | 5420.62 | 68.43 | 100.64 | 9.75 | 10.45 | 0.0 |
| ON | 507.0 | 507.0 | 0.0 | 5.07 | 4.70 | 7427.95 | 107.97 | 161.13 | 12.88 | 14.29 | 9.0 |

## 配置与校验

- 状态：`complete`；QPS 5.0；测量 100.0 秒；随机种子 101。
- 每条请求的正式轨迹输入长度 `[512]`、输出长度 `[512]`；两种模式复用同一条轨迹，SHA-256 `c61257f81e889170f1338f004438fffde00bbddf6bedf15ccab3986f43144f54`。
- 服务器返回的输入/输出 token 长度按模式记录在报告内嵌数据中：`{"off": {"expected_input": [512], "expected_output": [512], "server_input": [], "server_output": [512]}, "on": {"expected_input": [512], "expected_output": [512], "server_input": [], "server_output": [512]}}`。请求使用 `ignore_eos`，目标输出长度为 512；实际返回长度以 CSV 记录为准。
- 服务端返回的 `server_input_tokens` 字段缺失，记为 N/A；输入文本在本地 trace 生成时按 tokenizer 截到 512 tokens。实际输出 token 数由响应记录。
- 迁移 ON 配置：阈值 `9000.0`、`PAIR_MIGRATION_FREQUENCY=1`、Balanced、`rayrpc`、负载指标 `remaining_steps`。OFF/ON 的启动日志确认迁移开关和参数；Ray 状态确认四卡。
- 记录到迁移事件 9 次（成功 9、失败 0）。未发生迁移时不推定迁移耗时。
- 请求长度使 Offered output load 达到约 QPS × 512 输出 token/s；是否达到稳态取决于实测完成速率，见 CSV 和 GPU 采样。
- 表中的测量窗完成 QPS 只计 100 秒发送窗口内完成的请求。`summary.csv` 的 `request_throughput_rps` 和 `output_tokens_per_sec` 将排空期间完成的请求计入分子，因此不应当作测量窗内吞吐量。
- 预热 5.0 秒；停止发送后最多排空 300.0 秒。未排空请求计为超时/失败，不估算时延。

## 图表

- `plots/gpu_resources.png`
- `plots/mean_latency.png`
- `plots/migration.png`
- `plots/p95_latency.png`
- `plots/slo_violations.png`
- `plots/throughput.png`

## 文件

- `requests.csv`：逐请求数据，包括期望与服务端 token 长度、TTFT/TPOT、E2E、成功/超时状态。
- `gpu_metrics.csv`：每 GPU 利用率和显存采样。
- `migration_events.csv`：可解析的迁移事件。
- `summary.csv`：OFF/ON 运行指标及统计汇总。
- `config.json`、`state.json`、`configs/`、`traces/`、`server_logs/`：配置、状态、共用请求轨迹及日志。

## 复现

```bash
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
.venv/bin/python benchmark/run_qps5_512x512_off_on.py --seed 101 --qps 5.0 --measure-sec 100.0 --warmup-sec 5.0 --drain-sec 300.0
```

## 实例级 KV Cache 与排队指标

以下指标从 Llumnix 的 `*_instance_instance.csv` 中按请求测量窗截取（每种模式 100 秒）；样本数和采样间隔见 `instance_metrics_summary.csv`。

| 模式 | 实例样本 | KV Cache 均值 | KV Cache P95 | 平均 total blocks | 平均 used blocks | 平均 free blocks | 平均运行请求/实例 | 平均等待请求/实例 | 等待请求最大值 | `num_killed_requests` 最大值 | 非零样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OFF | 7023 | 0.865% | 1.216% | 34960 | 302.3 | 34657.9 | 6.27 | 0.000 | 0 | 0 | 0 |
| ON | 6882 | 1.132% | 1.565% | 34960 | 395.7 | 34564.5 | 8.22 | 0.000 | 0 | 0 | 0 |

GPU 计算活动均值来自已有 `gpu_metrics.csv` 测量阶段样本；峰值显存取四卡各自峰值中的最大值：

| 模式 | 四卡平均 GPU 利用率 | 四卡最高显存占用 |
|---|---:|---:|
| OFF | 53.04% | 7895 MiB |
| ON | 42.63% | 7923 MiB |

说明：历史实例 CSV 直接记录了 `gpu_cache_usage`、`num_available_gpu_blocks`、运行/等待请求数和 `num_killed_requests`，没有单独写出 total/used/free 块数。本分析按源码关系 `available = total - used`、`gpu_cache_usage = used / total` 推导历史块数，并在逐样本 CSV 的 `block_counts_source` 标明。后续日志已扩展为直接写 total/used/free。

`num_killed_requests` 在当前 vLLM scheduler 中是采样时的抢占相关请求 gauge（swapped 请求加上已被抢占且仍等待的请求），不是累计的抢占事件计数；本次两种模式所有样本均为 0，因此只能报告‘采样时未观察到非零值’，不能据此生成累计事件数。Waiting Requests 在全部采样中为 0。

本次观测中，ON 的 KV Cache 平均利用率为 1.132%（P95 1.565%），OFF 为 0.865%（P95 1.216%）；ON 平均每实例运行 8.22 个请求，OFF 为 6.27 个。四实例平均等待数均为 0，抢占相关 gauge 均未出现非零样本。

样本时间是管理器轮询得到的实例快照，不是固定频率重采样；图中保留原始点。`Instance 1–4` 是各模式内按首次出现时间命名，不代表 GPU 编号。GPU 利用率和显存图来自独立的 `nvidia-smi` 四卡采样。

新增图表：
- `plots/instance_kv_cache.png`
- `plots/instance_request_pressure.png`
- `plots/gpu_utilization_memory_summary.png`
