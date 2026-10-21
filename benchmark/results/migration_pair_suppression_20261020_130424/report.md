# Llumnix Migration Pair-Suppression 消融实验

## 结论

本次是迁移配对抑制消融：ON 保留迁移初始化、过滤、Balanced 配对计算和后台轮询，但不执行任何迁移。该组与 OFF 的性能差异对应这套启用路径在无实际迁移时的观测差异。

## 实验设置

- 状态：`complete`；种子：`101`；迁移策略：`balanced`。
- 四个独立实例，TP=1，模型 `/workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/model/Qwen/Qwen2-0.5B`；GPU 显存利用率配置 `0.65`。
- OFF/ON 配置仅 `ENABLE_ROUTINE_MIGRATION` 不同；调度策略、rayrpc、`remaining_steps`、模型及其他参数相同。启动日志确认了开关和 Balanced 策略，Ray 状态确认四张 GPU。
- 比较组为 OFF / ON（配对丢弃）。
- ON 保持 `ENABLE_ROUTINE_MIGRATION=true` 和 `balanced` 策略；过滤与 Balanced 配对计算照常运行，但配对结果在提交迁移任务前被实验开关丢弃。启动日志确认了该开关。
- 启动、预热、测量和排空日志中记录的被丢弃候选配对数：117；ON 实际迁移事件数：0。
- 每个 QPS 的 Poisson 请求轨迹只生成一次，OFF/ON 共用相同轨迹文件；实际发送 QPS、请求完成情况和逐请求延迟保存在 `requests.csv`。

- 预热 5s、测量 20s，停止发送后最多排空 180s。单种子结果用于快速筛查，不能代表正式重复实验或统计显著性。
- GPU 每秒采样；迁移次数按 Llumnix server log 中可解析的迁移完成日志计数。未解析到事件不代表没有尝试迁移。

## 配对结果

延迟列按 `OFF / ON` 排列，单位 ms；相对变化按 ON 相对 OFF 计算，负数代表 ON 更快。

| 目标 QPS | 实际发送 QPS OFF/ON | 测量窗完成 QPS OFF/ON | 发送数 OFF/ON | 排空后成功数 OFF/ON | P95 E2E OFF/ON | Mean TTFT OFF/ON | P95 TTFT OFF/ON | Mean TPOT OFF/ON | P95 TPOT OFF/ON | P95 E2E 变化 | ON 成功迁移数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5.35/5.35 | 5.30/5.30 | 107/107 | 107/107 | 1125.3/1117.1 | 76.3/73.0 | 116.8/111.6 | 8.1/8.0 | 9.0/9.0 | -0.7% | 0 |
| 20 | 20.30/20.30 | 19.10/18.90 | 406/406 | 406/406 | 1344.4/1911.9 | 69.4/94.4 | 101.5/147.2 | 9.5/12.4 | 10.5/16.1 | +42.2% | 0 |
| 28 | 28.55/28.55 | 27.00/27.00 | 571/571 | 571/571 | 1559.2/1612.0 | 76.4/89.2 | 113.5/122.7 | 10.8/11.1 | 12.4/12.6 | +3.4% | 0 |

TTFT/TPOT 优先来自服务端逐 Token 延迟时间线；缺失值不会估算。SLO 阈值沿用基准脚本默认值（TTFT 2000ms，TPOT 150ms），有效样本和违约率见 `summary.csv`。

“测量窗完成 QPS”只统计 20 秒发送窗口内完成的请求；“排空后成功数”还包含窗口结束后完成的请求，不能单独作为稳态吞吐量。

## 图表

- `plots/p95_latency.png`
- `plots/mean_latency.png`
- `plots/throughput.png`
- `plots/slo_violations.png`
- `plots/gpu_resources.png`
- `plots/migration.png`

## 文件

- `requests.csv`：逐请求测量和预热记录。
- `gpu_metrics.csv`：逐卡 GPU 利用率、显存等采样。
- `migration_events.csv`：可解析的迁移完成事件。
- `pairing_ablation_events.csv`：Balanced 计算出配对、随后被消融开关丢弃的记录（如有）。
- `summary.csv`：每个 QPS × 模式的单种子指标。
- `config.json`、`state.json`、`configs/`、`traces/`、`server_logs/`：真实配置、配对轨迹、日志和运行状态。

## 复现命令

```bash
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
source .venv/bin/activate
python benchmark/run_migration_crossover.py --seed 101 --qps 5.0 20.0 28.0 --warmup-sec 5 --measure-sec 20 --drain-sec 180 --policy balanced --suppress-pairs
```
