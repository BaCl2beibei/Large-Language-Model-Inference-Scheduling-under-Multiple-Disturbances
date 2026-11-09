# Paired comparison pilot: feasibility checkpoint

> Scope is one workload scene, seed 91001, four policies (4 of the planned 36 episodes). This checkpoint does not claim completion of the remaining scenes or seeds. The old pre-fix L run and the SIGTERM-interrupted M sample are excluded.

## Configuration and pairing

- Model: `model/Qwen/Qwen2-1.5B`; `max_model_len=2112`, `gpu_memory_utilization=0.65`, `max_num_seqs=192`, `max_num_batched_tokens=4096`.
- Workload: 512 input / 512 output tokens, 3 QPS, seed `91001`, fixed scoring window 240 s; 365/365 offered requests sent and accepted in every valid episode.
- Same request trace SHA-256: `e0f2ee32de5aafbb962915d9a008491bfff4cce5468a2cce8ebd775b780a8ad7`. Initial fixed routing sent 259/365 requests (71.0%) to the hot instance; the other routes received 41, 34, and 31.
- Frozen thresholds use the scheduler score `F`, not KV percentage: Balanced-L `549.0`, Balanced-M `1057.0`, Balanced-H `1898.5`.
- All four selected summaries report `status=passed`; M/H ran after a service restart following the interrupted M attempt. GPU instance assignment changed across that restart, so per-card comparisons below are aligned by route role and include each episode’s GPU UUID.

## KV load and performance

| Policy | F threshold | Four-instance mean KV | Per-instance mean KV% (hot / other 1 / other 2 / other 3) | Per-instance sampled max KV% (same order) | SLO goodput req/s | P95 TTFT ms | P95 TPOT ms | Completed | Output tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| Stay | N/A | 9.60% | 32.86 / 2.23 / 1.65 / 1.64 | 72.23 / 8.69 / 6.06 / 5.35 | 1.521 | 275.95 | 108.47 | 365/365 | 778.67 |
| Balanced-L | 549.00 | 13.73% | 43.26 / 4.50 / 3.61 / 3.57 | 95.79 / 12.07 / 9.64 / 9.12 | 1.204 | 354.14 | 155.26 | 365/365 | 778.67 |
| Balanced-M | 1057.00 | 13.98% | 45.58 / 3.81 / 3.42 / 3.13 | 99.78 / 10.39 / 9.91 / 8.05 | 1.142 | 607.28 | 157.96 | 365/365 | 778.67 |
| Balanced-H | 1898.50 | 14.47% | 47.49 / 4.05 / 3.39 / 2.96 | 99.67 / 10.99 / 9.23 / 7.96 | 0.967 | 2946.07 | 162.45 | 365/365 | 778.67 |

The sampled maximum is the largest per-instance KV sample within that episode window, not the P95. Output token throughput is identical across the policies because all 365 requests completed with the same 512 output tokens. Latency and SLO goodput still degraded under Balanced.

### Change versus Stay

| Policy | SLO goodput change | P95 TTFT change | P95 TPOT change | E2E mean change | E2E P95 change |
|---|---:|---:|---:|---:|---:|
| Balanced-L | -0.317 req/s (-20.8%) | +78.19 ms (+28.3%) | +46.78 ms (+43.1%) | +17715.11 ms (+43.5%) | +23856.35 ms (+42.8%) |
| Balanced-M | -0.379 req/s (-24.9%) | +331.33 ms (+120.1%) | +49.49 ms (+45.6%) | +18773.54 ms (+46.1%) | +25290.58 ms (+45.4%) |
| Balanced-H | -0.554 req/s (-36.4%) | +2670.12 ms (+967.6%) | +53.97 ms (+49.8%) | +21058.86 ms (+51.7%) | +27601.60 ms (+49.6%) |

All episodes completed 365/365 requests with zero non-manual errors. Balanced-L/M/H did not raise fixed-window completion or output-token throughput. They reduced SLO goodput by 20.8% / 24.9% / 36.4% and increased P95 TTFT by 28.3% / 120.1% / 967.6% versus Stay.

## Per-instance resource readings

Means are time-weighted across the 240 s scoring window. P95 and max are sample distributions within that window. `other 1/2/3` are ordered by fixed initial route share (41, 34, 31 requests).

| Policy | Route role | GPU UUID | KV mean / P95 / max % | GPU util mean / P95 % | GPU memory mean / max MiB | GPU memory util mean % | Running mean / P95 | Waiting mean / P95 | Output tok/s mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Stay | hot | `GPU-de41c419-b360-1592-a753-f718bdd5b0ee` | 32.86 / 69.35 / 72.23 | 25.19 / 68.00 | 10263 / 11237 | 83.52 | 52.96 / 113 | 0.08 / 1 | 555.65 |
| Stay | other 1 | `GPU-8ec028dd-cd3d-55c3-daf2-066e44491ddd` | 2.23 / 7.02 / 8.69 | 18.34 / 36.00 | 6777 / 6777 | 55.15 | 3.58 / 12 | 0.00 / 0 | 88.01 |
| Stay | other 2 | `GPU-f3cc4337-82f1-115b-9058-58cde88a4124` | 1.65 / 5.05 / 6.06 | 19.60 / 39.00 | 6806 / 6813 | 55.39 | 2.66 / 8 | 0.00 / 0 | 73.08 |
| Stay | other 3 | `GPU-55fbd150-b302-9b67-3a0d-425855d9f47d` | 1.64 / 4.38 / 5.35 | 18.04 / 36.00 | 6777 / 6777 | 55.15 | 2.64 / 7 | 0.00 / 0 | 66.59 |
| Balanced-L | hot | `GPU-de41c419-b360-1592-a753-f718bdd5b0ee` | 43.26 / 92.00 / 95.79 | 25.62 / 70.00 | 9943 / 11829 | 80.92 | 69.92 / 150 | 0.03 / 0 | 512.22 |
| Balanced-L | other 1 | `GPU-8ec028dd-cd3d-55c3-daf2-066e44491ddd` | 4.50 / 11.15 / 12.07 | 13.82 / 26.00 | 6828 / 6841 | 55.57 | 7.22 / 17 | 0.00 / 0 | 96.98 |
| Balanced-L | other 2 | `GPU-f3cc4337-82f1-115b-9058-58cde88a4124` | 3.61 / 8.68 / 9.64 | 15.72 / 36.00 | 6820 / 6821 | 55.50 | 5.76 / 14 | 0.01 / 0 | 91.87 |
| Balanced-L | other 3 | `GPU-55fbd150-b302-9b67-3a0d-425855d9f47d` | 3.57 / 8.49 / 9.12 | 15.08 / 34.00 | 6829 / 6841 | 55.58 | 5.69 / 13 | 0.01 / 0 | 82.17 |
| Balanced-M | hot | `GPU-8ec028dd-cd3d-55c3-daf2-066e44491ddd` | 45.58 / 97.69 / 99.78 | 24.88 / 71.00 | 9496 / 11761 | 77.28 | 73.70 / 159 | 0.13 / 1 | 527.95 |
| Balanced-M | other 1 | `GPU-f3cc4337-82f1-115b-9058-58cde88a4124` | 3.81 / 9.55 / 10.39 | 16.78 / 43.00 | 6828 / 6841 | 55.56 | 6.10 / 15 | 0.00 / 0 | 92.54 |
| Balanced-M | other 2 | `GPU-de41c419-b360-1592-a753-f718bdd5b0ee` | 3.42 / 8.68 / 9.91 | 15.79 / 37.00 | 6813 / 6821 | 55.45 | 5.46 / 14 | 0.01 / 0 | 85.74 |
| Balanced-M | other 3 | `GPU-55fbd150-b302-9b67-3a0d-425855d9f47d` | 3.13 / 7.40 / 8.05 | 15.46 / 30.00 | 6801 / 6805 | 55.35 | 4.99 / 12 | 0.01 / 0 | 77.71 |
| Balanced-H | hot | `GPU-8ec028dd-cd3d-55c3-daf2-066e44491ddd` | 47.49 / 99.02 / 99.67 | 23.88 / 61.00 | 10488 / 10697 | 85.35 | 76.81 / 160 | 0.48 / 5 | 537.83 |
| Balanced-H | other 1 | `GPU-f3cc4337-82f1-115b-9058-58cde88a4124` | 4.05 / 9.73 / 10.99 | 16.19 / 28.00 | 6841 / 6841 | 55.67 | 6.48 / 16 | 0.01 / 0 | 95.23 |
| Balanced-H | other 2 | `GPU-de41c419-b360-1592-a753-f718bdd5b0ee` | 3.39 / 8.22 / 9.23 | 14.43 / 24.00 | 6821 / 6821 | 55.51 | 5.43 / 13 | 0.01 / 0 | 77.41 |
| Balanced-H | other 3 | `GPU-55fbd150-b302-9b67-3a0d-425855d9f47d` | 2.96 / 6.75 / 7.96 | 15.20 / 25.00 | 6805 / 6805 | 55.38 | 4.74 / 11 | 0.00 / 0 | 73.38 |

## Migration activity and balance dispersion

| Policy | Selected actions / decisions | Legal-opportunity action rate | Successful / started | Rejected | Failed | Successes/min | Migrated-request coverage | Mean / P95 cost ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stay | 0/197 (0.00%) | 0.00% | 0/0 | 0 | 0 | 0.00 | 0.00% | N/A / N/A |
| Balanced-L | 30/191 (15.71%) | 83.33% | 23/23 | 7 | 0 | 5.75 | 6.30% | 498.21 / 665.96 |
| Balanced-M | 22/199 (11.06%) | 30.99% | 17/17 | 5 | 0 | 4.25 | 4.66% | 564.97 / 747.28 |
| Balanced-H | 12/193 (6.22%) | 12.77% | 12/12 | 0 | 0 | 3.00 | 3.29% | 559.01 / 694.50 |

### Balance improvement relative to Stay

Improvement is `(sigma_Stay - sigma_policy) / sigma_Stay`; negative values mean dispersion increased. Each sigma is the time average of the across-four-instance standard deviation.

| Policy | KV dispersion | GPU utilization dispersion | Running dispersion | Waiting dispersion |
|---|---:|---:|---:|---:|
| Balanced-L | -26.9% | -24.2% | -27.4% | +58.7% |
| Balanced-M | -35.8% | -21.7% | -36.3% | -76.8% |
| Balanced-H | -41.9% | -12.1% | -42.5% | -534.0% |

## Interpretation and scope

- Migration is functioning after the backend initialization fix: L/M/H succeeded 23/23, 17/17, and 12/12 times after 30, 22, and 12 selections respectively. Rejections were 7, 5, and 0; there were no terminal migration failures in these valid runs.
- Despite successful moves, only 6.3%, 4.7%, and 3.3% of accepted requests were migrated. Migration moves eligible in-flight requests; it does not redirect subsequent arrivals. The trace continues to send about 71% of arrivals to the hot route.
- The hot instance mean KV rose from 32.86% under Stay to 43.26%, 45.58%, and 47.49%; its sampled peak rose from 72.23% to 95.79%, 99.78%, and 99.67%. Balanced did not reduce this hotspot in this pilot.
- Mean request residency/queuing may contribute to higher KV use because latency worsened, but this is an interpretation of these observations rather than a separately isolated causal measurement.
- This is one scene and one seed. It establishes feasibility of the repaired migration path, but does not establish multi-seed stability or results for the other two workload scenes. Remaining planned episodes: 32 of 36.

### Why a higher threshold yielded fewer migration actions

The policy score `F=(free_blocks-waiting_blocks)/(running+waiting)` is available blocks per in-flight request. A candidate source must satisfy `F <= T`; a destination must satisfy `F >= T`. Increasing `T` therefore admits more instances on the source side but demands more headroom from the destination side. The policy then zips the sorted source and destination lists and still requires a legal action, an observed migration candidate, and a strict improvement in the post-migration score.

| Policy | T | Sum of source-eligible instance counts across decisions | Sum of destination-eligible instance counts across decisions | Decisions with ≥1 threshold-qualified legal pair | Qualified legal pairs summed across decisions | Successful migrations |
|---|---:|---:|---:|---:|---:|---:|
| Balanced-L | 549.0 | 213 | 551 | 30 | 99 | 23 |
| Balanced-M | 1057.0 | 390 | 406 | 22 | 68 | 17 |
| Balanced-H | 1898.5 | 435 | 338 | 12 | 36 | 12 |

These counts show the source side broadening while the destination side narrows; the resulting threshold-qualified pairs fell from 99 to 68 to 36. The `opportunity_ratio` in the migration summary uses a broader denominator: it counts decisions with any legal `action_mask` pair, regardless of whether that pair meets the threshold. It should not be read as the count of threshold-qualified migration opportunities.

### Difference from the earlier coarse-scan Stay

| Evidence | Coarse scan Stay | Paired-pilot Stay |
|---|---:|---:|
| Seed / trace SHA-256 | 91000 / `75f90ebd...bbaa4003` | 91001 / `e0f2ee32...780a8ad7` |
| Offered and completed | 348 / 348 | 365 / 365 |
| Hot-route requests | 249 (71.6%) | 259 (71.0%) |
| Scoring window | 240 s (120 s arrival + 120 s drain) | 240 s (120 s arrival + 120 s drain) |
| Hot-instance waiting mean / P95 / max | 7.49 / 39 / 52 | 0.078 / 1 / 3 |
| P95 TTFT | 39.72 s | 276 ms |

The model path, inference limits, GPU UUID set, and fixed-skew routing were the same. The traces were different, and the coarse-scan service ran before the recorded migration-backend initialization patch; the pilot ran after it. The patch addressed scan-mode migration backend initialization, but this comparison does not isolate whether that code change contributed to the queue/latency difference. The trace arrival burst counts were similar at the 10 s scale, so the available evidence does not support explaining the gap by request count or route share alone. Treat the coarse scan as threshold-selection evidence, not as a directly comparable performance control. The full paired matrix uses the repaired code and a shared trace within each Stay/Balanced pair.

## Audit paths

- Stay: `benchmark/results/migration_opportunity_comparison_20261103_qwen2_15b_q3_3length/raw_runs/paired_1793697602882432072_0/runs/5f3ab4f9f9b7391ec1bd/scan_summary.json`
- Balanced-L (repaired run): `benchmark/results/migration_opportunity_comparison_20261103_qwen2_15b_q3_3length/raw_runs/paired_1793697602882432072_0/runs/38cc64bd6ca036c41ec0/scan_summary.json`
- Balanced-M: `benchmark/results/migration_opportunity_comparison_20261103_qwen2_15b_q3_3length/raw_runs/paired_1793698652965009587_0/runs/25757b9382d32385e772/scan_summary.json`
- Balanced-H: `benchmark/results/migration_opportunity_comparison_20261103_qwen2_15b_q3_3length/raw_runs/paired_1793698652965009587_0/runs/e38304d33aad11256e50/scan_summary.json`
- Frozen thresholds: `benchmark/results/migration_opportunity_comparison_20261103_qwen2_15b_q3_3length/threshold_selection.json`
- Patch and excluded pre-fix evidence: `benchmark/results/migration_opportunity_comparison_20261103_qwen2_15b_q3_3length/configs/paired_scan_migration_backend_init_patch.json`
