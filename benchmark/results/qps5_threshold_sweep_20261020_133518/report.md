# QPS 5 Migration Threshold Sweep

## Comparison

Values are mean ± sample standard deviation across seeds 101, 202, and 303. TTFT and TPOT use the service's per-token timing data.

| Mode | P95 E2E (ms) | Mean TTFT (ms) | Mean TPOT (ms) | Completed throughput (req/s) | Successful migrations/run (all phases) | Mean migration cost (ms, all phases) |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 1122.88 ± 25.47 | 74.04 ± 5.49 | 7.92 ± 0.10 | 4.92 ± 0.48 | 0.0 ± 0.0 | N/A ± N/A |
| Threshold 2000 | 1093.97 ± 36.78 | 71.49 ± 3.72 | 7.92 ± 0.05 | 4.92 ± 0.48 | 0.0 ± 0.0 | N/A ± N/A |
| Threshold 5000 | 1344.86 ± 31.77 | 89.92 ± 2.22 | 9.58 ± 0.08 | 4.92 ± 0.48 | 0.0 ± 0.0 | N/A ± N/A |
| Threshold 8000 | 1465.85 ± 191.42 | 96.47 ± 14.59 | 10.04 ± 1.31 | 4.92 ± 0.48 | 0.7 ± 0.6 | 129.0 ± 21.5 |
| Threshold 20000 | 1909.45 ± 367.90 | 110.41 ± 6.56 | 11.39 ± 0.87 | 4.92 ± 0.48 | 16.0 ± 7.0 | 104.5 ± 14.8 |

| Paired P95 difference vs OFF | Mean change | Across-seed SD | Seed pairs |
|---|---:|---:|---:|
| Threshold 2000 vs OFF | N/A% | N/A% | 0 |
| Threshold 5000 vs OFF | N/A% | N/A% | 0 |
| Threshold 8000 vs OFF | N/A% | N/A% | 0 |
| Threshold 20000 vs OFF | N/A% | N/A% | 0 |

## Migration events during measurement windows

Warmup and drain events are recorded in `migration_events.csv`; this table counts only events attributed to the 20-second measurement windows.

| Mode | Events | Successful | Failed |
|---|---:|---:|---:|
| OFF | 0 | 0 | 0 |
| Threshold 2000 | 0 | 0 | 0 |
| Threshold 5000 | 0 | 0 | 0 |
| Threshold 8000 | 2 | 2 | 0 |
| Threshold 20000 | 42 | 40 | 2 |

## Method and validation

- Status: `complete`; target QPS 5.0; seeds 101, 202, 303; 4 independent Llumnix instances, TP=1.
- All migration runs use Balanced, `rayrpc`, `remaining_steps`, the same dispatch policy and model. The only ON-mode setting varied is `MIGRATE_OUT_THRESHOLD`; OFF disables routine migration.
- Each seed has one Poisson request trace reused unchanged by all five modes. Trace SHA-256 by seed: `{"101": "0ea4da0013ca32762c3104901d38c73052ba3c317a3ff0cb16f50f688286c307", "202": "4c175a8a1aeb16ac76bf4a528ab6c5fb62351a2bb7afe04b975c045aef80782f", "303": "c38d5b68f2f1050a125ce8ff30f441278a6322d9a1727b340b8615216b0d41f3"}`.
- Startup logs verify the migration enable flag, threshold value, Balanced policy, and Ray's four-GPU resources for every mode. No CPU/GPU load, network, or fault injection was used.
- Warmup 5.0s, measurement 20.0s, drain deadline 180.0s. Request success, timing sample counts, and GPU metrics are in the CSV files.
- Migration count and average migration cost in the comparison table include all observed phases (warmup, measurement, drain). The separate event table isolates the measurement windows.
- This is a short, three-seed comparison; it is exploratory and does not establish statistical significance.

## Plots

- `plots/latency_comparison.png`
- `plots/migration_comparison.png`

## Files

- `requests.csv`, `gpu_metrics.csv`, `migration_events.csv`, `summary.csv`
- `config.json`, `state.json`, `environment_audit.md`, `configs/`, `traces/`, `server_logs/`

## Reproduce

```bash
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
.venv/bin/python benchmark/run_qps5_threshold_sweep.py --seeds 101 202 303 --qps 5.0 --warmup-sec 5.0 --measure-sec 20.0 --drain-sec 180.0 --policy balanced
```
