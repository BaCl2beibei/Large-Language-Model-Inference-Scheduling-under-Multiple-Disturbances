# Seed 404 system validation

Held-out response times are used only as comparison labels. Simulation inputs are the measured request arrival trace, input/output token counts, and (for fixed replay only) logged migration events.

## Comparisons

V0 uses the existing scalar service coefficients and disables KV pressure. V1 replaces service times with promoted calibration models. V2 adds dynamic KV allocation/growth and a corrected re-prefill path after preemption. V3 adds fitted residual uncertainty to migration duration. The closed-loop controller is a transparent greedy baseline, not a trained PPO policy.

| QPS | Observed completed | V0 closed-loop completed | V1 | V2 | V3 | Observed avg TTFT ms | V3 avg TTFT ms | Observed avg TPOT ms | V3 avg TPOT ms | Observed completions/s | V3 completions/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.0 | 608 | 159 | 539 | 582 | 556 | 12898.04 | 158.19 | 490.53 | 53.83 | 0.00 | 7.62 |
| 15.0 | 827 | 179 | 697 | 720 | 746 | 9242.18 | 480.94 | 603.98 | 56.18 | 0.00 | 7.22 |
| 18.0 | 579 | 191 | 734 | 793 | 702 | 6708.02 | 2091.28 | 558.70 | 60.73 | 0.00 | 7.25 |
| 21.0 | 1056 | 187 | 638 | 655 | 682 | 11877.84 | 4289.39 | 976.52 | 67.69 | 0.00 | 7.15 |
| 9.0 | 400 | 157 | 391 | 387 | 418 | 6835.81 | 130.14 | 542.25 | 50.46 | 0.03 | 5.35 |

## Closed-loop details

| QPS | Version | Completed / failed / unfinished | TTFT avg / P95 ms | TPOT request avg / P95 ms | Throughput req/s | Queue avg / P95 | KV avg / P95 | Migrations success / fail |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 12.0 | V0 | 159 / 0 / 553 | 154.78 / 204.71 | 118.45 / 144.43 | 1.83 | 19.34 / 77.00 | 0.00 / 0.00 | 77 / 1 |
| 12.0 | V1 | 539 / 0 / 173 | 151.51 / 271.75 | 55.04 / 94.28 | 6.28 | 0.60 / 2.00 | 0.00 / 0.00 | 112 / 2 |
| 12.0 | V2 | 582 / 0 / 130 | 154.59 / 321.48 | 52.48 / 89.77 | 7.83 | 0.65 / 2.00 | 0.15 / 0.26 | 115 / 3 |
| 12.0 | V3 | 556 / 0 / 156 | 158.19 / 353.56 | 53.83 / 91.04 | 7.62 | 0.69 / 2.00 | 0.16 / 0.26 | 116 / 2 |
| 15.0 | V0 | 179 / 0 / 709 | 159.44 / 227.03 | 129.33 / 149.93 | 1.97 | 79.56 / 252.00 | 0.00 / 0.00 | 74 / 1 |
| 15.0 | V1 | 697 / 0 / 191 | 211.08 / 625.70 | 58.14 / 104.77 | 6.60 | 1.51 / 4.00 | 0.00 / 0.00 | 99 / 1 |
| 15.0 | V2 | 720 / 0 / 168 | 183.42 / 510.46 | 57.42 / 104.28 | 8.32 | 1.21 / 4.00 | 0.20 / 0.30 | 116 / 0 |
| 15.0 | V3 | 746 / 0 / 142 | 480.94 / 1745.08 | 56.18 / 100.35 | 7.22 | 5.74 / 25.00 | 0.21 / 0.34 | 90 / 2 |
| 18.0 | V0 | 191 / 0 / 878 | 184.44 / 247.82 | 134.77 / 150.63 | 2.18 | 153.45 / 413.00 | 0.00 / 0.00 | 50 / 1 |
| 18.0 | V1 | 734 / 0 / 335 | 2047.10 / 7494.81 | 58.12 / 109.18 | 7.75 | 47.86 / 145.00 | 0.00 / 0.00 | 69 / 0 |
| 18.0 | V2 | 793 / 0 / 276 | 2254.57 / 8058.08 | 56.91 / 115.66 | 7.50 | 47.75 / 144.00 | 0.24 / 0.36 | 62 / 0 |
| 18.0 | V3 | 702 / 0 / 367 | 2091.28 / 7177.97 | 60.73 / 109.15 | 7.25 | 50.15 / 148.05 | 0.24 / 0.35 | 63 / 0 |
| 21.0 | V0 | 187 / 0 / 1052 | 197.74 / 246.85 | 137.63 / 150.68 | 2.57 | 234.23 / 560.00 | 0.00 / 0.00 | 33 / 0 |
| 21.0 | V1 | 638 / 0 / 601 | 3628.76 / 15531.18 | 68.90 / 111.76 | 7.53 | 126.00 / 265.00 | 0.00 / 0.00 | 35 / 0 |
| 21.0 | V2 | 655 / 0 / 584 | 3243.80 / 13839.90 | 64.94 / 117.90 | 7.43 | 114.88 / 253.05 | 0.25 / 0.35 | 48 / 1 |
| 21.0 | V3 | 682 / 0 / 557 | 4289.39 / 16603.84 | 67.69 / 108.52 | 7.15 | 132.07 / 301.00 | 0.26 / 0.37 | 50 / 1 |
| 9.0 | V0 | 157 / 0 / 392 | 144.30 / 196.42 | 105.14 / 125.04 | 1.83 | 0.39 / 2.00 | 0.00 / 0.00 | 87 / 1 |
| 9.0 | V1 | 391 / 0 / 158 | 125.72 / 171.26 | 51.87 / 83.82 | 4.95 | 0.30 / 1.00 | 0.00 / 0.00 | 119 / 2 |
| 9.0 | V2 | 387 / 0 / 162 | 125.05 / 166.29 | 51.26 / 84.73 | 5.02 | 0.28 / 1.00 | 0.11 / 0.16 | 115 / 1 |
| 9.0 | V3 | 418 / 0 / 131 | 130.14 / 172.31 | 50.46 / 82.46 | 5.35 | 0.33 / 1.05 | 0.12 / 0.18 | 99 / 2 |

## Fixed event replay coverage

| QPS | Logged measurement attempts | V2 attempts applied | Source/request/capacity mismatches | Median start delay ms |
|---:|---:|---:|---:|---:|
| 12.0 | 2 | 2 | 0 | -29.46 |
| 15.0 | 3 | 3 | 0 | -12.54 |
| 18.0 | 5 | 3 | 0 | -5.22 |
| 21.0 | 6 | 2 | 0 | 212.33 |
| 9.0 | 3 | 3 | 0 | -8.31 |

## Fixed ON event replay

This replays the observed measurement-phase migration schedule and terminal outcomes. The action time is approximated as watcher observation time minus logged migration cost, and the source is forced for requests that later appear in the event log. Events with no request ID, or requests not yet running at their logged source, cannot be applied; per-QPS counts and delays are in the JSON. Fixed replay consumes the observed duration and therefore does not evaluate migration-cost prediction uncertainty.

## Validation scope and limitations

- Service iteration labels are executor wall times and are compared by phase and QPS; they are not isolated GPU time or request latency.
- The logs do not expose exact initial request-to-worker routing, full scheduler decisions, block allocation/release events, or migration service-pause labels. Initial routing uses the simulator's least-work rule, except requests with a migration event are routed to the logged first source.
- The simulator combines prefill and decode service durations using the maximum of the phase predictions when both are active. Logs do not identify mixed-batch overlap semantics; this is a declared approximation.
- Fixed migration replay uses the logged event outcome. Closed-loop runs use the robust global migration duration estimate and residual sampling; migration failure probability remains the configured uncalibrated default.
- Queue and KV comparisons are approximate because logged queue/KV snapshots are polling samples, while simulated state is event-driven.

Detailed QPS, P95, queue/KV, migration counts, replay coverage, and iteration distributions are in `system_validation.json`.
