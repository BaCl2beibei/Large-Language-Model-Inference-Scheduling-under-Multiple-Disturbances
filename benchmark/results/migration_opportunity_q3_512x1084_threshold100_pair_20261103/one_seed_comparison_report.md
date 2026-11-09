# One-seed paired comparison: QPS 3, input 512 / output 1084

Generated from the append-only run ledger and episode summaries. The resource CSV maps episode-local instance IDs to persistent GPU UUIDs.
All values use the fixed 600 s scoring window (120 s arrival + 480 s drain). Means and quantiles for sampled resources use time-weighted summaries.
Model/config: Qwen2-1.5B, max_model_len=2112, gpu_memory_utilization=0.65, max_num_seqs=192, max_num_batched_tokens=4096; QPS=3; seed=91000.
Balanced-L threshold F=100 was explicitly set by the user for this test; it is an override, not a Stay-derived quantile. Stay and Balanced-L use the same seed and generated request trace.

Complete scenario/seed groups: 1; missing groups: 0.

## Paired performance and outcomes

SLO goodput is completed requests meeting the configured TTFT/TPOT bounds divided by 600 s. Goodput/throughput improvements are versus the same-seed Stay trace; latency reductions use (Stay − Balanced) / Stay.

| Scenario | Seed | Policy | F threshold | SLO goodput req/s | Goodput Δ% | P95 TTFT ms | TTFT ↓% | P95 TPOT ms | TPOT ↓% | Completed/accepted | Completion rate | Errors | Deadline unfinished | E2E mean ms | E2E P95 ms | Client output tok/s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| i512_o1084_q3_s192_b4096 | 91000 | Stay | N/A | 0.2283 | N/A | 122271.04 | N/A | 235.38 | N/A | 348/348 | 1.000 | 0 | 0 | 174391.14 | 281342.43 | 628.72 |
| i512_o1084_q3_s192_b4096 | 91000 | Balanced-L | 100 | 0.1950 | -14.60 | 150927.32 | -23.44 | 283.88 | -20.61 | 348/348 | 1.000 | 0 | 0 | 199715.90 | 332927.41 | 628.72 |

## Secondary performance and queue changes vs Stay

Throughput changes use the client completion ledger; resource throughput by GPU is reported in the resource table below. Waiting deltas are Balanced − Stay; negative values mean less waiting.

| Scenario | Seed | Policy | Completion throughput req/s | Throughput Δ% | Client output tok/s | Output Δ% | E2E mean reduction % | E2E P95 reduction % | Waiting mean Δ | Worst-GPU Waiting P95 Δ |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| i512_o1084_q3_s192_b4096 | 91000 | Stay | 0.5800 | N/A | 628.72 | N/A | N/A | N/A | N/A | N/A |
| i512_o1084_q3_s192_b4096 | 91000 | Balanced-L | 0.5800 | 0.00 | 628.72 | 0.00 | -14.52 | -18.34 | 0.55 | 2 |

## Balance dispersion improvement

σ is the time-weighted average of each sampled four-GPU population standard deviation. Improvement = (σStay − σBalanced) / σStay. A zero or missing Stay denominator is N/A; the absolute difference remains in the CSV.

| Scenario | Seed | Policy | Metric | σ Stay | σ Balanced | Absolute Δ (Stay − Balanced) | Improvement % |
|---|---:|---|---|---:|---:|---:|---:|
| i512_o1084_q3_s192_b4096 | 91000 | Balanced-L | kv_pct | 22.59 | 23.15 | -0.56 | -2.48 |
| i512_o1084_q3_s192_b4096 | 91000 | Balanced-L | gpu_pct | 8.19 | 5.56 | 2.63 | 32.11 |
| i512_o1084_q3_s192_b4096 | 91000 | Balanced-L | running | 26.59 | 27.66 | -1.07 | -4.04 |
| i512_o1084_q3_s192_b4096 | 91000 | Balanced-L | waiting | 10.85 | 11.88 | -1.04 | -9.57 |

## Migration behavior

Action share is selected migration actions / all decisions. Legal-opportunity action share is chosen migration actions when at least one legal pair existed / decisions with at least one legal pair. Requested migration success rate follows the requested denominator: succeeded / started (accepted and executed). Rejected requests are separate.

| Scenario | Seed | Policy | Decisions | Selected | Action % all | Legal decisions | Selected with legal opportunity | Action % legal | Submitted | Started | Succeeded | Success % started | Success/min | Migrated request coverage % | Rejected | Failed | Mean cost s | P95 cost s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| i512_o1084_q3_s192_b4096 | 91000 | Stay | 474 | 0 | 0.00 | 304 | 0 | 0.00 | 0 | 0 | 0 | N/A | 0.00 | 0.00 | 0 | 0 | N/A | N/A |
| i512_o1084_q3_s192_b4096 | 91000 | Balanced-L | 479 | 60 | 12.53 | 82 | 60 | 73.17 | 60 | 53 | 52 | 98.11 | 5.20 | 14.94 | 7 | 1 | 1.08 | 1.56 |

## Four-GPU resource summary

For each metric, four-GPU mean is the arithmetic mean of the four per-GPU time-weighted means. P95 columns report the mean and worst of the four per-GPU time-weighted P95s. Output token throughput is summed over the four GPUs.

| Scenario | Seed | Policy | KV mean % | KV mean GPU P95 % | KV worst GPU P95 % | KV peak % | GPU util mean % | GPU mean GPU P95 % | GPU worst GPU P95 % | GPU peak % | Memory mean MiB | Memory peak MiB | Memory util mean % | Running mean | Running worst GPU P95 | Waiting mean | Waiting worst GPU P95 | Output tok/s total |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| i512_o1084_q3_s192_b4096 | 91000 | Stay | 15.22 | 35.10 | 99.86 | 99.99 | 11.88 | 41.00 | 58.00 | 100.00 | 7300.16 | 11703.00 | 59.41 | 17.94 | 141 | 6.32 | 127 | 634.57 |
| i512_o1084_q3_s192_b4096 | 91000 | Balanced-L | 18.42 | 39.47 | 99.83 | 99.99 | 15.16 | 37.00 | 51.00 | 100.00 | 7378.68 | 8973.00 | 60.05 | 21.65 | 141 | 6.87 | 129 | 635.36 |

Per-GPU time-weighted means/P95/peaks, including GPU UUIDs and coverage, are in `one_seed_gpu_resources.csv`. Full episode audit paths and SHA-256 digests are in `one_seed_pair_audit.json`.

## Interpretation

- **i512_o1084_q3_s192_b4096, seed 91000:** Stay SLO goodput 0.2283 req/s. Balanced-L: SLO goodput -14.60% vs Stay; KV dispersion improvement -2.48%; migrated requests 14.94%.
