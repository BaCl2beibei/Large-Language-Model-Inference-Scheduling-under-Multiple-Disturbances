# Llumnix Migration ON/OFF 快速交叉点实验

## 结论

在本次测试范围（5.5675–20 QPS）内，没有测到 ON 的 P95 端到端延迟低于 OFF。

## 实验设置

- 状态：`complete`；种子：`101`；迁移策略：`balanced`。
- 四个独立实例，TP=1，模型 `/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/model/Qwen/Qwen2-0.5B`；GPU 显存利用率配置 `0.65`。
- OFF/ON 配置仅 `ENABLE_ROUTINE_MIGRATION` 不同；调度策略、rayrpc、`remaining_steps`、模型及其他参数相同。启动日志确认了开关和 Balanced 策略，Ray 状态确认四张 GPU。
- 每个 QPS 的 Poisson 请求轨迹只生成一次，OFF/ON 共用相同轨迹文件；实际发送 QPS、请求完成情况和逐请求延迟保存在 `requests.csv`。
- 预热 5s、测量 20s，停止发送后最多排空 180s。单种子结果用于快速筛查，不能代表正式重复实验或统计显著性。
- GPU 每秒采样；迁移次数按 Llumnix server log 中可解析的迁移完成日志计数，表中计数包含对应档位的预热与测量阶段。未解析到事件不代表没有尝试迁移。

## 配对结果

延迟列按 `OFF / ON` 排列，单位 ms；相对变化按 ON 相对 OFF 计算，负数代表 ON 更快。

| 目标 QPS | 实际发送 QPS OFF/ON | 测量窗完成 QPS OFF/ON | 发送数 OFF/ON | 排空后成功数 OFF/ON | P95 E2E OFF/ON | P95 TTFT OFF/ON | P95 TPOT OFF/ON | P95 E2E 变化 | ON 成功迁移数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.5675 | 5.85/5.85 | 5.60/5.55 | 117/117 | 117/117 | 1140.6/1554.3 | 127.9/166.4 | 8.7/13.0 | +36.3% | 21 |
| 8 | 8.20/8.20 | 8.05/7.90 | 164/164 | 164/164 | 1124.7/1644.7 | 102.6/159.7 | 9.2/13.6 | +46.2% | 17 |
| 10 | 10.15/10.15 | 9.80/9.75 | 203/203 | 203/203 | 1184.0/1696.5 | 110.3/165.7 | 9.3/13.3 | +43.3% | 16 |
| 12 | 12.55/12.55 | 11.70/11.50 | 251/251 | 251/251 | 1187.0/1732.9 | 91.4/165.3 | 9.7/13.9 | +46.0% | 9 |
| 16 | 16.00/16.00 | 15.75/15.60 | 320/320 | 320/320 | 1285.9/1857.8 | 96.8/168.2 | 10.5/15.0 | +44.5% | 5 |
| 20 | 20.30/20.30 | 19.05/18.55 | 406/406 | 406/406 | 1332.0/2120.2 | 97.8/192.7 | 10.6/16.7 | +59.2% | 6 |

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
python benchmark/run_migration_crossover.py --seed 101 --qps 5.5675 8.0 10.0 12.0 16.0 20.0 --warmup-sec 5 --measure-sec 20 --drain-sec 180 --policy balanced
```
