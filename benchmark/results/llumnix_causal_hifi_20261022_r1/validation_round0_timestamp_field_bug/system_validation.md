# Causal system validation

Training seed: 606; held-out seed: 707. 
Measurement/drain: 60s / 30s. 
The predictive simulator receives held-out measurement arrivals, prompt lengths, and budgets only.

## Held-out error summary

| Version | Metric | Macro MAPE | Macro MAE | Signed bias |
|:---|:---|---:|---:|---:|
| round2_v3 | request_throughput_rps | 100.00% | 0.160 | -0.160 |
| round2_v3 | output_tokens_per_sec | 100.00% | 504.333 | -504.333 |
| round2_v3 | queue_mean_requests | 448.09% | 7.032 | -6.885 |
| round2_v3 | queue_p95_requests | 38.16% | 21.374 | -21.231 |
| round2_v3 | kv_mean_fraction | 21.50% | 0.032 | -0.032 |
| round2_v3 | kv_p95_fraction | 24.86% | 0.057 | -0.057 |
| causal_rebuild | request_throughput_rps | 100.00% | 0.160 | -0.160 |
| causal_rebuild | output_tokens_per_sec | 100.00% | 504.333 | -504.333 |
| causal_rebuild | queue_mean_requests | 105.95% | 25.799 | -25.796 |
| causal_rebuild | queue_p95_requests | 97.54% | 78.056 | -77.985 |
| causal_rebuild | kv_mean_fraction | 52.61% | 0.090 | -0.090 |
| causal_rebuild | kv_p95_fraction | 60.58% | 0.157 | -0.157 |
| causal_rebuild | migration_total_mean_ms | 62.12% | 403.787 | -403.787 |
| causal_rebuild | migration_pause_stage_mean_ms | 79.40% | 286.717 | -286.717 |
| causal_rebuild | migration_recovery_mean_ms | 2325.77% | 7581.781 | 6277.558 |
| causal_fixed_replay | request_throughput_rps | 100.00% | 0.160 | -0.160 |
| causal_fixed_replay | output_tokens_per_sec | 100.00% | 504.333 | -504.333 |
| causal_fixed_replay | queue_mean_requests | 109.28% | 25.610 | -25.606 |
| causal_fixed_replay | queue_p95_requests | 95.10% | 77.128 | -77.056 |
| causal_fixed_replay | kv_mean_fraction | 58.00% | 0.097 | -0.097 |
| causal_fixed_replay | kv_p95_fraction | 64.15% | 0.164 | -0.164 |
| causal_fixed_replay | migration_total_mean_ms | 14.78% | 103.259 | -103.259 |
| causal_fixed_replay | migration_pause_stage_mean_ms | 6.35% | 33.465 | 33.465 |
| causal_fixed_replay | migration_recovery_mean_ms | 3.86% | 120.974 | 120.974 |

## Warmup phase boundary

| Mode | QPS | HTTP 200 | HTTP 499 | Migrations started | In flight at measurement start |
|:---:|---:|---:|---:|---:|---:|
| off | 12 | 77 | 89 | 0 | 0 |
| off | 15 | 123 | 123 | 0 | 0 |
| off | 18 | 244 | 0 | 0 | 0 |
| off | 21 | 213 | 70 | 0 | 0 |
| off | 24 | 183 | 190 | 0 | 0 |
| off | 3 | 52 | 0 | 0 | 0 |
| off | 9 | 150 | 13 | 0 | 0 |
| on | 12 | 74 | 92 | 1 | 0 |
| on | 15 | 235 | 11 | 2 | 0 |
| on | 18 | 124 | 120 | 2 | 0 |
| on | 21 | 128 | 155 | 1 | 0 |
| on | 24 | 279 | 94 | 2 | 0 |
| on | 3 | 52 | 0 | 3 | 0 |
| on | 9 | 82 | 81 | 3 | 0 |

## Per-run values

| Mode | QPS | Version | Request throughput | Token throughput | Mean/P95 TTFT ms | Mean/P95 TPOT ms | Queue mean/P95 | KV mean/P95 | Migrations (attempt/success/fail/abort) | Total/pause/recovery ms |
|:---:|---:|:---|---:|---:|---:|---:|---:|---:|:---:|---:|
| off | 12 | observed | 0.000 | 457.4 | 4836.2/10252.5 | 323.9/754.1 | 6.03/32.75 | 17.42%/28.73% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 12 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 1.39/3.00 | 13.13%/22.82% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 12 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.15/0.50 | 6.09%/7.55% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 12 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.14/0.50 | 4.85%/6.32% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 15 | observed | 0.000 | 305.0 | 6015.1/24299.2 | 453.0/2266.7 | 18.99/63.75 | 18.62%/29.02% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 15 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 8.38/37.50 | 15.63%/23.51% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 15 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.21/0.50 | 9.23%/11.43% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 15 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.25/0.75 | 8.52%/10.71% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 18 | observed | 0.000 | 575.6 | 6766.9/21602.6 | 533.9/2666.6 | 33.08/109.25 | 20.22%/28.87% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 18 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 22.23/79.75 | 17.04%/23.71% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 18 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.27/0.75 | 4.48%/5.41% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 18 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.33/1.00 | 3.80%/5.24% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 21 | observed | 0.000 | 369.1 | 15499.6/32952.7 | 451.9/2504.4 | 52.21/154.00 | 20.77%/28.95% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 21 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 39.86/118.00 | 17.85%/24.01% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 21 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.31/0.75 | 11.85%/15.65% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 21 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.51/1.75 | 11.42%/14.71% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 24 | observed | 0.000 | 273.0 | 12421.7/22112.6 | 505.8/2893.7 | 69.59/165.76 | 20.85%/29.19% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 24 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 60.07/159.26 | 18.11%/23.90% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 24 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.40/1.00 | 13.86%/16.84% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 24 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 2.17/9.25 | 13.69%/17.96% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 3 | observed | 0.767 | 798.7 | 342.6/949.7 | 106.3/342.2 | 0.01/0.00 | 4.79%/8.12% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 3 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.19/0.50 | 3.08%/4.53% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 3 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.02/0.25 | 3.35%/4.72% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 3 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.03/0.25 | 2.39%/3.21% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 9 | observed | 0.467 | 792.8 | 5163.8/22231.6 | 233.2/562.3 | 0.23/1.76 | 13.94%/25.22% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 9 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.77/1.50 | 9.67%/15.76% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 9 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.10/0.25 | 5.69%/7.37% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 9 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.09/0.50 | 4.42%/5.78% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 12 | observed | 0.000 | 427.5 | 9916.7/57881.4 | 356.8/945.1 | 5.51/27.75 | 17.30%/28.32% | 1/1/0/0 | 655.1/320.1/250.2 |
| on | 12 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 1.64/6.00 | 13.08%/22.46% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 12 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.14/0.50 | 5.87%/7.32% | 68/67/0/1 | 208.1/60.7/12850.0 |
| on | 12 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.13/0.50 | 4.82%/6.19% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 15 | observed | 0.000 | 566.2 | 4978.9/32275.0 | 451.2/1773.8 | 19.21/80.00 | 18.76%/28.51% | 1/1/0/0 | 403.5/399.5/5799.1 |
| on | 15 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 8.30/38.76 | 15.62%/23.53% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 15 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.19/0.50 | 8.99%/10.99% | 70/70/0/0 | 201.0/58.7/12411.1 |
| on | 15 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.25/0.75 | 8.38%/10.71% | 1/1/0/0 | 403.5/399.5/5799.1 |
| on | 18 | observed | 0.000 | 248.6 | 7614.5/15831.7 | 574.4/2874.8 | 33.94/104.50 | 20.05%/28.72% | 2/1/0/1 | 651.0/306.0/5907.7 |
| on | 18 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 23.12/79.50 | 16.85%/23.69% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 18 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.26/0.75 | 4.34%/5.42% | 65/61/0/4 | 200.7/58.6/12394.6 |
| on | 18 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.30/1.00 | 3.55%/4.76% | 1/0/0/1 | 312.5/310.1/5907.7 |
| on | 21 | observed | 0.000 | 189.7 | 17467.5/32289.3 | 590.0/2840.0 | 51.92/159.75 | 20.71%/28.63% | 1/1/0/0 | 696.8/284.6/16724.9 |
| on | 21 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 39.93/117.75 | 17.80%/24.06% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 21 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.28/0.75 | 11.78%/15.21% | 70/70/0/0 | 201.1/58.7/12421.4 |
| on | 21 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.37/1.00 | 11.35%/15.06% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 24 | observed | 0.000 | 455.0 | 22226.4/45023.1 | 456.8/2599.3 | 72.60/197.00 | 21.05%/29.04% | 1/1/0/0 | 444.4/439.9/12578.4 |
| on | 24 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 60.75/159.02 | 18.01%/23.84% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 24 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.38/1.00 | 13.98%/17.33% | 71/71/0/0 | 199.4/58.2/12317.2 |
| on | 24 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.77/3.00 | 13.51%/16.37% | 1/1/0/0 | 444.4/439.9/12578.4 |
| on | 3 | observed | 0.983 | 949.7 | 405.8/2164.8 | 84.3/152.0 | 0.00/0.00 | 4.74%/8.20% | 1/1/0/0 | 363.4/135.5/123.2 |
| on | 3 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.19/0.50 | 3.09%/4.60% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 3 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.02/0.25 | 2.75%/3.71% | 55/55/0/0 | 215.2/62.8/13291.6 |
| on | 3 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.02/0.25 | 2.42%/3.39% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 9 | observed | 0.017 | 652.3 | 6199.6/18585.0 | 296.3/741.8 | 0.63/3.51 | 13.86%/26.32% | 2/2/0/0 | 1044.6/539.4/3131.5 |
| on | 9 | round2_v3 | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.76/1.50 | 9.63%/15.72% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 9 | causal_rebuild | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.09/0.25 | 5.22%/7.09% | 66/65/1/0 | 206.8/60.4/12772.0 |
| on | 9 | causal_fixed_replay | 0.000 | 0.0 | 0.0/0.0 | 0.0/0.0 | 0.11/0.50 | 4.58%/6.20% | 1/1/0/0 | 970.0/669.2/3615.4 |

## Fixed batch schedule replay

Observed held-out batches are fixed inputs for this local service model check.

| Mode | QPS | Batches | Round 2 MAE ms / MAPE | Causal MAE ms / MAPE |
|:---:|---:|---:|---:|---:|
| off | 12 | 1798 | 378.07 / 567.80% | 26.42 / 27.30% |
| off | 15 | 1821 | 381.73 / 617.10% | 23.88 / 29.32% |
| off | 18 | 1662 | 392.78 / 571.70% | 21.43 / 25.72% |
| off | 21 | 1507 | 417.39 / 585.29% | 27.92 / 30.24% |
| off | 24 | 1593 | 410.53 / 605.68% | 27.72 / 32.84% |
| off | 3 | 2652 | 286.47 / 597.56% | 26.56 / 23.53% |
| off | 9 | 2424 | 333.01 / 629.39% | 19.26 / 28.78% |
| on | 12 | 1607 | 370.15 / 443.74% | 25.89 / 22.90% |
| on | 15 | 1588 | 384.09 / 470.67% | 21.55 / 20.49% |
| on | 18 | 1407 | 402.78 / 456.51% | 29.05 / 23.75% |
| on | 21 | 1398 | 408.00 / 470.65% | 29.54 / 24.75% |
| on | 24 | 1403 | 424.99 / 484.32% | 26.04 / 20.38% |
| on | 3 | 3259 | 282.78 / 522.43% | 19.40 / 29.53% |
| on | 9 | 1740 | 345.82 / 431.60% | 25.89 / 23.76% |

## Fixed route and migration replay

| Mode | QPS | Observed migration starts | Replayed starts | Rejected starts | First-batch route coverage |
|:---:|---:|---:|---:|---:|---:|
| off | 12 | 0 | 0 | 0 | 75.4% |
| off | 15 | 0 | 0 | 0 | 60.8% |
| off | 18 | 0 | 0 | 0 | 52.5% |
| off | 21 | 0 | 0 | 0 | 46.7% |
| off | 24 | 0 | 0 | 0 | 40.0% |
| off | 3 | 0 | 0 | 0 | 98.5% |
| off | 9 | 0 | 0 | 0 | 95.5% |
| on | 12 | 1 | 1 | 1 | 75.0% |
| on | 15 | 1 | 1 | 0 | 60.0% |
| on | 18 | 2 | 2 | 1 | 51.1% |
| on | 21 | 1 | 1 | 1 | 44.8% |
| on | 24 | 1 | 1 | 0 | 39.3% |
| on | 3 | 1 | 1 | 1 | 100.0% |
| on | 9 | 2 | 2 | 1 | 94.5% |

## Accuracy gate

Independent accuracy gate passed: **False**.

| Metric | Passed | Worst held-out run | Error | Limit |
|:---|:---:|:---|---:|---:|
| request_throughput_rps | False | off QPS 3 | 1.000 | 0.200 |
| ttft_mean_ms | False | n/a | n/a | 0.25 |
| ttft_p95_ms | False | n/a | n/a | 0.3 |
| tpot_mean_ms | False | n/a | n/a | 0.2 |
| tpot_p95_ms | False | n/a | n/a | 0.25 |
| queue_mean_requests | False | on QPS 24 | 72.219 | 14.520 |
| queue_p95_requests | False | on QPS 24 | 196.000 | 39.400 |
| kv_mean_fraction | False | off QPS 18 | 15.741 | 5.000 |
| kv_p95_fraction | False | off QPS 18 | 23.462 | 10.000 |
| request_lifecycle_and_collection | True | n/a | n/a | None |

## Figures

![OFF held-out metrics](plots/system_validation_off.png)

![ON held-out metrics](plots/system_validation_on.png)

![Round 2 versus causal MAPE](plots/round2_vs_causal_mape.png)

## Definitions and limitations

Queue and KV are resampled at 50 ms per-instance grids. Request output and server completion use the API/server event chain; client E2E is retained as a separate metric. HTTP 499 tails are counted as censored.
Warmup requests are drained or aborted before the measurement phase. The simulator starts with an empty request/KV state at that boundary; it does not model GPU cache or thermal effects from warmup.
Migration ON is evaluated with the simulator's greedy controller, while the real run uses the unchanged Llumnix routine. This tests the simulator's migration mechanism and trigger approximation; it is not an exact policy replay.
The fixed replay diagnostic uses held-out first-batch instance routes and observed ON migration starts, durations, pause, and recovery labels. It is a mechanism diagnostic and is excluded from the independent prediction gate.
A gate passes only when every OFF and ON QPS run meets the per-run threshold and every measurement request is joined to a terminal event.
