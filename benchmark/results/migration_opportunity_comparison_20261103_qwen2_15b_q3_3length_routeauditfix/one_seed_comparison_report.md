# One-seed paired comparison: QPS 3

Generated from the append-only run ledger and episode summaries. The resource CSV maps episode-local instance IDs to persistent GPU UUIDs.
All values use the fixed 600 s scoring window (120 s arrival + 480 s drain). Means and quantiles for sampled resources use time-weighted summaries.
Model/config: Qwen2-1.5B, max_model_len=2112, gpu_memory_utilization=0.65, max_num_seqs=192, max_num_batched_tokens=4096; QPS=3; seed=91000.
Balanced thresholds are frozen per scenario from the Stay run's finite, non-idle arrival-phase F samples at empirical 25th/50th/75th percentiles. The exact Stay evidence paths and hashes are in `threshold_selection.json`.

Complete scenario/seed groups: 2; missing groups: 1.

## Paired performance and outcomes

SLO goodput is completed requests meeting the configured TTFT/TPOT bounds divided by 600 s. Goodput/throughput improvements are versus the same-seed Stay trace; latency reductions use (Stay − Balanced) / Stay.

| Scenario | Seed | Policy | F threshold | SLO goodput req/s | Goodput Δ% | P95 TTFT ms | TTFT ↓% | P95 TPOT ms | TPOT ↓% | Completed/accepted | Completion rate | Errors | Deadline unfinished | E2E mean ms | E2E P95 ms | Client output tok/s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| i512_o512_q3_s192_b4096 | 91000 | Stay | N/A | 0.3967 | N/A | 545.76 | N/A | 163.85 | N/A | 348/348 | 1.000 | 0 | 0 | 55446.66 | 83976.73 | 296.96 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-L | 597.75 | 0.2450 | -38.24 | 5187.99 | -850.60 | 195.75 | -19.47 | 348/348 | 1.000 | 0 | 0 | 69713.04 | 100313.73 | 296.96 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-M | 1065.86 | 0.2117 | -46.64 | 12240.82 | -2142.88 | 198.71 | -21.28 | 348/348 | 1.000 | 0 | 0 | 72707.00 | 101788.24 | 296.96 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-H | 2544.33 | 0.2083 | -47.48 | 14153.44 | -2493.33 | 200.75 | -22.52 | 348/348 | 1.000 | 0 | 0 | 74566.02 | 102824.46 | 296.96 |
| i512_o1024_q3_s192_b4096 | 91000 | Stay | N/A | 0.2167 | N/A | 112035.51 | N/A | 250.10 | N/A | 348/348 | 1.000 | 0 | 0 | 168053.18 | 273110.58 | 593.92 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-L | 315.59 | 0.1933 | -10.77 | 133656.75 | -19.30 | 281.29 | -12.47 | 348/348 | 1.000 | 0 | 0 | 181461.34 | 301990.40 | 593.92 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-M | 523.62 | 0.1833 | -15.38 | 140782.53 | -25.66 | 310.02 | -23.96 | 348/348 | 1.000 | 0 | 0 | 200575.86 | 332985.11 | 593.92 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-H | 901.88 | 0.1750 | -19.23 | 139119.47 | -24.17 | 303.69 | -21.43 | 348/348 | 1.000 | 0 | 0 | 196155.40 | 317742.84 | 593.92 |

## Secondary performance and queue changes vs Stay

Throughput changes use the client completion ledger; resource throughput by GPU is reported in the resource table below. Waiting deltas are Balanced − Stay; negative values mean less waiting.

| Scenario | Seed | Policy | Completion throughput req/s | Throughput Δ% | Client output tok/s | Output Δ% | E2E mean reduction % | E2E P95 reduction % | Waiting mean Δ | Worst-GPU Waiting P95 Δ |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| i512_o512_q3_s192_b4096 | 91000 | Stay | 0.5800 | N/A | 296.96 | N/A | N/A | N/A | N/A | N/A |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-L | 0.5800 | 0.00 | 296.96 | 0.00 | -25.73 | -19.45 | 0.12 | 3 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-M | 0.5800 | 0.00 | 296.96 | 0.00 | -31.13 | -21.21 | 0.23 | 10 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-H | 0.5800 | 0.00 | 296.96 | 0.00 | -34.48 | -22.44 | 0.29 | 15 |
| i512_o1024_q3_s192_b4096 | 91000 | Stay | 0.5800 | N/A | 593.92 | N/A | N/A | N/A | N/A | N/A |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-L | 0.5800 | 0.00 | 593.92 | 0.00 | -7.98 | -10.57 | -0.24 | -2 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-M | 0.5800 | 0.00 | 593.92 | 0.00 | -19.35 | -21.92 | 0.61 | 9 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-H | 0.5800 | 0.00 | 593.92 | 0.00 | -16.72 | -16.34 | 0.63 | 12 |

## Balance dispersion improvement

σ is the time-weighted average of each sampled four-GPU population standard deviation. Improvement = (σStay − σBalanced) / σStay. A zero or missing Stay denominator is N/A; the absolute difference remains in the CSV.

| Scenario | Seed | Policy | Metric | σ Stay | σ Balanced | Absolute Δ (Stay − Balanced) | Improvement % |
|---|---:|---|---|---:|---:|---:|---:|
| i512_o512_q3_s192_b4096 | 91000 | Balanced-L | kv_pct | 7.33 | 8.20 | -0.86 | -11.79 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-L | gpu_pct | 2.52 | 2.76 | -0.24 | -9.36 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-L | running | 11.85 | 13.26 | -1.41 | -11.89 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-L | waiting | 0.00 | 0.21 | -0.21 | -5880.62 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-M | kv_pct | 7.33 | 8.44 | -1.10 | -15.04 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-M | gpu_pct | 2.52 | 2.67 | -0.15 | -6.08 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-M | running | 11.85 | 13.62 | -1.77 | -14.90 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-M | waiting | 0.00 | 0.38 | -0.38 | -10650.33 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-H | kv_pct | 7.33 | 8.97 | -1.63 | -22.29 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-H | gpu_pct | 2.52 | 3.18 | -0.66 | -26.08 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-H | running | 11.85 | 14.48 | -2.63 | -22.17 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-H | waiting | 0.00 | 0.51 | -0.51 | -14304.96 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-L | kv_pct | 22.32 | 20.88 | 1.44 | 6.46 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-L | gpu_pct | 7.09 | 4.62 | 2.46 | 34.77 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-L | running | 27.04 | 25.63 | 1.42 | 5.24 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-L | waiting | 10.56 | 10.05 | 0.52 | 4.90 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-M | kv_pct | 22.32 | 23.34 | -1.02 | -4.57 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-M | gpu_pct | 7.09 | 5.30 | 1.79 | 25.29 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-M | running | 27.04 | 28.74 | -1.69 | -6.26 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-M | waiting | 10.56 | 11.54 | -0.98 | -9.27 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-H | kv_pct | 22.32 | 22.63 | -0.32 | -1.41 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-H | gpu_pct | 7.09 | 5.50 | 1.58 | 22.34 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-H | running | 27.04 | 27.81 | -0.76 | -2.82 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-H | waiting | 10.56 | 11.58 | -1.01 | -9.58 |

## Migration behavior

Action share is selected migration actions / all decisions. Legal-opportunity action share is chosen migration actions when at least one legal pair existed / decisions with at least one legal pair. Requested migration success rate follows the requested denominator: succeeded / started (accepted and executed). Rejected requests are separate.

| Scenario | Seed | Policy | Decisions | Selected | Action % all | Legal decisions | Selected with legal opportunity | Action % legal | Submitted | Started | Succeeded | Success % started | Success/min | Migrated request coverage % | Rejected | Failed | Mean cost s | P95 cost s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| i512_o512_q3_s192_b4096 | 91000 | Stay | 493 | 0 | 0.00 | 130 | 0 | 0.00 | 0 | 0 | 0 | N/A | 0.00 | 0.00 | 0 | 0 | N/A | N/A |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-L | 497 | 35 | 7.04 | 42 | 35 | 83.33 | 35 | 27 | 26 | 96.30 | 2.60 | 7.47 | 8 | 1 | 0.68 | 0.87 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-M | 491 | 22 | 4.48 | 72 | 22 | 30.56 | 22 | 20 | 19 | 95.00 | 1.90 | 5.46 | 2 | 1 | 0.69 | 0.94 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-H | 496 | 15 | 3.02 | 104 | 15 | 14.42 | 15 | 14 | 14 | 100.00 | 1.40 | 4.02 | 1 | 0 | 0.68 | 0.96 |
| i512_o1024_q3_s192_b4096 | 91000 | Stay | 477 | 0 | 0.00 | 301 | 0 | 0.00 | 0 | 0 | 0 | N/A | 0.00 | 0.00 | 0 | 0 | N/A | N/A |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-L | 460 | 66 | 14.35 | 73 | 66 | 90.41 | 66 | 54 | 52 | 96.30 | 5.20 | 14.94 | 12 | 2 | 0.89 | 1.31 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-M | 461 | 56 | 12.15 | 110 | 56 | 50.91 | 56 | 50 | 49 | 98.00 | 4.90 | 14.08 | 6 | 1 | 0.93 | 1.45 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-H | 478 | 47 | 9.83 | 130 | 47 | 36.15 | 47 | 42 | 40 | 95.24 | 4.00 | 11.49 | 5 | 2 | 1.03 | 1.57 |

## Four-GPU resource summary

For each metric, four-GPU mean is the arithmetic mean of the four per-GPU time-weighted means. P95 columns report the mean and worst of the four per-GPU time-weighted P95s. Output token throughput is summed over the four GPUs.

| Scenario | Seed | Policy | KV mean % | KV mean GPU P95 % | KV worst GPU P95 % | KV peak % | GPU util mean % | GPU mean GPU P95 % | GPU worst GPU P95 % | GPU peak % | Memory mean MiB | Memory peak MiB | Memory util mean % | Running mean | Running worst GPU P95 | Waiting mean | Waiting worst GPU P95 | Output tok/s total |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| i512_o512_q3_s192_b4096 | 91000 | Stay | 4.90 | 25.92 | 90.13 | 98.99 | 8.23 | 38.75 | 47.00 | 87.00 | 7229.42 | 11789.00 | 58.83 | 7.92 | 145 | 0.00 | 0 | 302.98 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-L | 6.15 | 30.87 | 98.83 | 99.99 | 7.49 | 30.00 | 45.00 | 100.00 | 7518.24 | 11627.00 | 61.18 | 9.93 | 155 | 0.12 | 3 | 303.97 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-M | 6.27 | 30.55 | 98.87 | 99.94 | 7.14 | 29.00 | 48.00 | 100.00 | 7647.21 | 10143.00 | 62.23 | 10.10 | 157 | 0.23 | 10 | 300.16 |
| i512_o512_q3_s192_b4096 | 91000 | Balanced-H | 6.48 | 30.56 | 99.11 | 99.97 | 7.63 | 29.50 | 50.00 | 100.00 | 7691.30 | 10283.00 | 62.59 | 10.45 | 158 | 0.29 | 15 | 301.37 |
| i512_o1024_q3_s192_b4096 | 91000 | Stay | 14.87 | 34.12 | 99.85 | 99.99 | 11.27 | 38.75 | 47.00 | 100.00 | 7268.78 | 8661.00 | 59.15 | 18.02 | 143 | 6.14 | 128 | 603.69 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-L | 16.23 | 37.15 | 99.73 | 99.99 | 14.60 | 35.00 | 43.00 | 100.00 | 7407.17 | 11815.00 | 60.28 | 19.66 | 140 | 5.91 | 126 | 591.56 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-M | 18.03 | 39.55 | 99.78 | 99.99 | 14.59 | 30.50 | 47.00 | 100.00 | 7461.77 | 9221.00 | 60.72 | 21.96 | 142 | 6.76 | 137 | 599.84 |
| i512_o1024_q3_s192_b4096 | 91000 | Balanced-H | 17.35 | 39.18 | 99.86 | 99.96 | 14.53 | 31.50 | 51.00 | 100.00 | 7508.07 | 9413.00 | 61.10 | 21.13 | 143 | 6.78 | 140 | 598.45 |

Per-GPU time-weighted means/P95/peaks, including GPU UUIDs and coverage, are in `one_seed_gpu_resources.csv`. Full episode audit paths and SHA-256 digests are in `one_seed_pair_audit.json`.

## Interpretation

- **i512_o512_q3_s192_b4096, seed 91000:** Stay SLO goodput 0.3967 req/s. Balanced-L: SLO goodput -38.24% vs Stay; KV dispersion improvement -11.79%; migrated requests 7.47%. Balanced-M: SLO goodput -46.64% vs Stay; KV dispersion improvement -15.04%; migrated requests 5.46%. Balanced-H: SLO goodput -47.48% vs Stay; KV dispersion improvement -22.29%; migrated requests 4.02%.
- **i512_o1024_q3_s192_b4096, seed 91000:** Stay SLO goodput 0.2167 req/s. Balanced-L: SLO goodput -10.77% vs Stay; KV dispersion improvement 6.46%; migrated requests 14.94%. Balanced-M: SLO goodput -15.38% vs Stay; KV dispersion improvement -4.57%; migrated requests 14.08%. Balanced-H: SLO goodput -19.23% vs Stay; KV dispersion improvement -1.41%; migrated requests 11.49%.
