# QPS 5 Migration Pair Frequency Comparison

## Results

Entries are mean ± sample standard deviation across seeds 101, 202, and 303. TTFT and TPOT use service-side per-token timing.

| Mode | P95 E2E (ms) | Mean TTFT (ms) | Mean TPOT (ms) | Completed throughput (req/s) | Successful migrations/run (all phases) | Mean migration cost (ms, all phases) |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 1118.17 ± 9.18 | 70.42 ± 3.61 | 7.97 ± 0.08 | 4.92 ± 0.48 | 0.0 ± 0.0 | N/A ± N/A |
| Frequency 1 | 1300.84 ± 43.98 | 83.25 ± 0.72 | 8.80 ± 0.04 | 4.92 ± 0.48 | 0.3 ± 0.6 | 86.9 ± 0.0 |
| Frequency 5 | 1116.27 ± 23.56 | 70.67 ± 2.84 | 7.95 ± 0.03 | 4.92 ± 0.48 | 0.0 ± 0.0 | N/A ± N/A |
| Frequency 10 | 1242.67 ± 47.80 | 83.80 ± 2.09 | 8.66 ± 0.20 | 4.92 ± 0.48 | 0.3 ± 0.6 | 97.8 ± 0.0 |

| Paired P95 difference vs OFF | Mean change | Across-seed SD | Seed pairs |
|---|---:|---:|---:|
| Frequency 1 vs OFF | 16.4% | 4.9% | 3 |
| Frequency 5 vs OFF | -0.2% | 1.3% | 3 |
| Frequency 10 vs OFF | 11.2% | 5.1% | 3 |

## Migration events in measurement windows

The comparison-table migration counts and cost include warmup, measurement, and drain phases. This table isolates the 20-second measurement windows.

| Mode | Events | Successful | Failed |
|---|---:|---:|---:|
| OFF | 0 | 0 | 0 |
| Frequency 1 | 1 | 1 | 0 |
| Frequency 5 | 0 | 0 | 0 |
| Frequency 10 | 1 | 1 | 0 |

## Method and validation

- Status: `complete`; QPS 5.0; seeds 101, 202, 303; four independent Llumnix instances, TP=1.
- `MIGRATE_OUT_THRESHOLD=8000.0` and Balanced policy are fixed. ON conditions vary `PAIR_MIGRATION_FREQUENCY` across 1, 5, 10; OFF disables routine migration.
- All conditions use `rayrpc`, `remaining_steps`, the same model, dispatch configuration, GPU settings, and per-seed Poisson request trace. SHA-256 by seed: `{"101": "0ea4da0013ca32762c3104901d38c73052ba3c317a3ff0cb16f50f688286c307", "202": "4c175a8a1aeb16ac76bf4a528ab6c5fb62351a2bb7afe04b975c045aef80782f", "303": "c38d5b68f2f1050a125ce8ff30f441278a6322d9a1727b340b8615216b0d41f3"}`.
- Startup logs verify migration enablement, threshold 8000, each configured frequency, Balanced policy, and Ray's four-GPU resources. No load, network, or fault injection was used.
- Warmup 5.0s, measurement 20.0s, drain deadline 180.0s. Request success, token timing sample counts, and GPU metrics are recorded in the CSV files.
- This short three-seed comparison is exploratory; it does not establish statistical significance.

## Plots

- `plots/latency_comparison.png`
- `plots/migration_comparison.png`

## Files

- `requests.csv`, `gpu_metrics.csv`, `migration_events.csv`, `summary.csv`
- `config.json`, `state.json`, `environment_audit.md`, `configs/`, `traces/`, `server_logs/`

## Reproduce

```bash
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
.venv/bin/python benchmark/run_qps5_frequency_sweep.py --seeds 101 202 303 --qps 5.0 --warmup-sec 5.0 --measure-sec 20.0 --drain-sec 180.0 --policy balanced
```
