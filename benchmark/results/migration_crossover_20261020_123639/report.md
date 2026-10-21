# Llumnix Migration ON/OFF 快速交叉点实验

## 结论

在本次测试范围（0–5 QPS）内，没有测到 ON 的 P95 端到端延迟低于 OFF。

## 实验设置

- 状态：`complete`；种子：`101`；迁移策略：`balanced`。
- 四个独立实例，TP=1，模型 `/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/model/Qwen/Qwen2-0.5B`；GPU 显存利用率配置 `0.65`。
- OFF/ON 配置仅 `ENABLE_ROUTINE_MIGRATION` 不同；调度策略、rayrpc、`remaining_steps`、模型及其他参数相同。启动日志确认了开关和 Balanced 策略，Ray 状态确认四张 GPU。
- 每个 QPS 的 Poisson 请求轨迹只生成一次，OFF/ON 共用相同轨迹文件；实际发送 QPS、请求完成情况和逐请求延迟保存在 `requests.csv`。
- QPS 0 是空闲控制：两个模式下均不发送请求，延迟及成功率记为 N/A；GPU 指标仍在测量窗口采集。
- 预热 5s、测量 20s，停止发送后最多排空 180s。单种子结果用于快速筛查，不能代表正式重复实验或统计显著性。
- GPU 每秒采样；迁移次数按 Llumnix server log 中可解析的迁移完成日志计数。未解析到事件不代表没有尝试迁移。

## 配对结果

延迟列按 `OFF / ON` 排列，单位 ms；相对变化按 ON 相对 OFF 计算，负数代表 ON 更快。

| 目标 QPS | 实际发送 QPS OFF/ON | 测量窗完成 QPS OFF/ON | 发送数 OFF/ON | 排空后成功数 OFF/ON | P95 E2E OFF/ON | P95 TTFT OFF/ON | P95 TPOT OFF/ON | P95 E2E 变化 | ON 成功迁移数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0/0.0 | 0.0/0.0 | 0/0 | 0.0/0.0 | N/A | N/A | N/A | N/A | 0 |
| 1 | 1.55/1.55 | 1.55/1.55 | 31/31 | 31/31 | 1045.6/1074.2 | 120.5/113.9 | 8.7/9.6 | +2.7% | 2 |
| 2 | 2.45/2.45 | 2.35/2.35 | 49/49 | 49/49 | 1111.6/1330.3 | 126.4/158.3 | 8.9/10.5 | +19.7% | 4 |
| 3 | 3.70/3.70 | 3.60/3.55 | 74/74 | 74/74 | 1059.8/1409.4 | 130.2/165.1 | 8.6/11.7 | +33.0% | 11 |
| 4 | 4.80/4.80 | 4.60/4.55 | 96/96 | 96/96 | 1091.9/1445.4 | 113.2/165.5 | 8.7/12.6 | +32.4% | 13 |
| 5 | 5.35/5.35 | 5.30/5.20 | 107/107 | 107/107 | 1114.1/1521.9 | 119.6/171.9 | 9.1/12.8 | +36.6% | 15 |

TTFT/TPOT 优先来自服务端逐 Token 延迟时间线；缺失值不会估算。SLO 阈值沿用基准脚本默认值（TTFT 2000ms，TPOT 150ms），有效样本和违约率见 `summary.csv`。

“测量窗完成 QPS”只统计 20 秒发送窗口内完成的请求；“排空后成功数”还包含窗口结束后完成的请求，不能单独作为稳态吞吐量。

## 图表

- `plots/p95_latency.png`
- `plots/throughput.png`
- `plots/slo_violations.png`
- `plots/gpu_resources.png`
- `plots/migration.png`

## 文件

- `requests.csv`：逐请求测量和预热记录。
- `gpu_metrics.csv`：逐卡 GPU 利用率、显存等采样。
- `migration_events.csv`：可解析的迁移完成事件。
- `summary.csv`：每个 QPS × 模式的单种子指标。
- `config.json`、`state.json`、`configs/`、`traces/`、`server_logs/`：真实配置、配对轨迹、日志和运行状态。

## 复现命令

```bash
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
source .venv/bin/activate
python benchmark/run_migration_crossover.py --seed 101 --qps 0.0 1.0 2.0 3.0 4.0 5.0 --warmup-sec 5 --measure-sec 20 --drain-sec 180 --policy balanced
```
