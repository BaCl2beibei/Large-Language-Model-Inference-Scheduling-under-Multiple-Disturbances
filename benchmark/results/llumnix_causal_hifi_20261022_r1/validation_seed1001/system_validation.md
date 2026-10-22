# Causal system validation

Training seed(s): [606, 909]; held-out seed: 1001. 
Measurement/drain: 60s / 30s. 
The predictive simulator receives held-out measurement arrivals, prompt lengths, and budgets only.

## Held-out error summary

| Version | Metric | Macro MAPE | Macro MAE | Signed bias |
|:---|:---|---:|---:|---:|
| round2_v3 | request_throughput_rps | 431.88% | 3.387 | 3.387 |
| round2_v3 | output_tokens_per_sec | 84.27% | 809.270 | -683.735 |
| round2_v3 | ttft_mean_ms | 85.90% | 3589.466 | -2264.125 |
| round2_v3 | ttft_p95_ms | 37.88% | 7168.331 | -5402.479 |
| round2_v3 | tpot_mean_ms | 244.38% | 429.162 | 429.162 |
| round2_v3 | tpot_p95_ms | 95.57% | 432.986 | 63.547 |
| round2_v3 | queue_mean_requests | 996.24% | 9.035 | -8.884 |
| round2_v3 | queue_p95_requests | 68.55% | 31.125 | -30.804 |
| round2_v3 | kv_mean_fraction | 23.91% | 0.037 | -0.037 |
| round2_v3 | kv_p95_fraction | 25.08% | 0.058 | -0.058 |
| causal_rebuild | request_throughput_rps | 336.14% | 3.032 | 2.954 |
| causal_rebuild | output_tokens_per_sec | 162.92% | 596.446 | 93.394 |
| causal_rebuild | ttft_mean_ms | 63.42% | 3715.988 | -2742.666 |
| causal_rebuild | ttft_p95_ms | 60.48% | 7692.269 | -4706.283 |
| causal_rebuild | tpot_mean_ms | 35.03% | 79.485 | 10.210 |
| causal_rebuild | tpot_p95_ms | 36.10% | 313.148 | -251.209 |
| causal_rebuild | queue_mean_requests | 458.05% | 8.734 | -8.478 |
| causal_rebuild | queue_p95_requests | 123.94% | 29.569 | -28.819 |
| causal_rebuild | kv_mean_fraction | 13.40% | 0.022 | -0.017 |
| causal_rebuild | kv_p95_fraction | 13.66% | 0.034 | -0.025 |
| causal_rebuild | migration_total_mean_ms | 50.69% | 324.576 | -324.576 |
| causal_rebuild | migration_pause_stage_mean_ms | 72.35% | 200.224 | -200.224 |
| causal_rebuild | migration_recovery_mean_ms | 11645.91% | 13497.465 | 13497.465 |
| causal_fixed_replay | request_throughput_rps | 328.04% | 2.918 | 2.913 |
| causal_fixed_replay | output_tokens_per_sec | 156.69% | 577.890 | 97.317 |
| causal_fixed_replay | ttft_mean_ms | 89.18% | 3597.087 | -2045.494 |
| causal_fixed_replay | ttft_p95_ms | 117.77% | 7635.047 | -3446.163 |
| causal_fixed_replay | tpot_mean_ms | 31.20% | 77.210 | 4.761 |
| causal_fixed_replay | tpot_p95_ms | 31.65% | 310.916 | -258.684 |
| causal_fixed_replay | queue_mean_requests | 765.61% | 8.361 | -5.943 |
| causal_fixed_replay | queue_p95_requests | 223.89% | 29.252 | -25.966 |
| causal_fixed_replay | kv_mean_fraction | 13.10% | 0.023 | -0.021 |
| causal_fixed_replay | kv_p95_fraction | 12.07% | 0.031 | -0.030 |
| causal_fixed_replay | migration_total_mean_ms | 12.36% | 82.710 | -74.433 |
| causal_fixed_replay | migration_pause_stage_mean_ms | 9.05% | 18.745 | -13.368 |
| causal_fixed_replay | migration_recovery_mean_ms | 7.57% | 8.550 | -8.550 |

## Warmup phase boundary

| Mode | QPS | HTTP 200 | HTTP 499 | Migrations started | In flight at measurement start |
|:---:|---:|---:|---:|---:|---:|
| off | 12 | 163 | 25 | 0 | 0 |
| off | 15 | 37 | 192 | 0 | 0 |
| off | 18 | 64 | 217 | 0 | 0 |
| off | 21 | 40 | 267 | 0 | 0 |
| off | 24 | 75 | 263 | 0 | 0 |
| off | 3 | 43 | 0 | 0 | 0 |
| off | 9 | 128 | 0 | 0 | 0 |
| on | 12 | 188 | 0 | 1 | 0 |
| on | 15 | 229 | 0 | 2 | 0 |
| on | 18 | 281 | 0 | 3 | 0 |
| on | 21 | 307 | 0 | 1 | 0 |
| on | 24 | 338 | 0 | 1 | 0 |
| on | 3 | 43 | 0 | 1 | 0 |
| on | 9 | 128 | 0 | 1 | 0 |

## Per-run values

| Mode | QPS | Version | Request throughput | Token throughput | Mean/P95 TTFT ms | Mean/P95 TPOT ms | Queue mean/P95 | KV mean/P95 | Migrations (attempt/success/fail/abort) | Total/pause/recovery ms |
|:---:|---:|:---|---:|---:|---:|---:|---:|---:|:---:|---:|
| off | 12 | observed | 0.417 | 890.2 | 4713.5/14839.3 | 260.5/711.1 | 4.17/20.76 | 17.32%/29.41% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 12 | round2_v3 | 4.100 | 419.2 | 1034.6/1632.0 | 649.7/872.4 | 1.39/3.00 | 12.28%/21.47% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 12 | causal_rebuild | 3.283 | 1073.1 | 538.5/1744.1 | 262.5/552.7 | 1.32/4.00 | 14.16%/25.40% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 12 | causal_fixed_replay | 3.233 | 1062.6 | 932.2/3538.7 | 270.5/562.1 | 2.36/5.75 | 14.10%/25.33% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 15 | observed | 0.000 | 404.9 | 28445.2/55291.3 | 166.2/360.1 | 19.28/72.75 | 19.27%/29.78% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 15 | round2_v3 | 4.550 | 457.6 | 1759.0/6312.1 | 722.0/899.8 | 7.02/31.00 | 14.93%/23.26% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 15 | causal_rebuild | 3.117 | 1407.8 | 1872.4/12197.3 | 256.5/515.3 | 12.54/53.75 | 17.43%/25.96% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 15 | causal_fixed_replay | 2.850 | 1372.4 | 2369.5/10845.1 | 259.1/516.9 | 14.85/55.75 | 17.18%/25.94% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 18 | observed | 0.000 | 315.6 | 6834.7/12119.8 | 264.8/928.8 | 33.31/105.75 | 20.42%/29.77% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 18 | round2_v3 | 4.650 | 484.2 | 3284.1/14814.4 | 748.5/894.4 | 20.65/67.51 | 16.24%/23.23% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 18 | causal_rebuild | 8.050 | 1183.7 | 1121.2/5694.0 | 288.9/568.9 | 5.13/22.25 | 16.23%/25.31% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 18 | causal_fixed_replay | 8.017 | 1165.3 | 1898.0/6556.8 | 287.7/568.9 | 8.19/22.00 | 15.85%/25.12% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 21 | observed | 0.000 | 166.5 | 9758.6/17674.3 | 488.9/1984.6 | 49.87/146.00 | 21.19%/29.54% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 21 | round2_v3 | 4.683 | 494.9 | 4755.1/20312.2 | 765.5/907.5 | 37.69/106.52 | 17.01%/23.24% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 21 | causal_rebuild | 4.283 | 1414.4 | 4070.4/21494.0 | 284.2/623.5 | 38.71/114.02 | 19.45%/26.26% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 21 | causal_fixed_replay | 4.067 | 1356.1 | 5607.5/22049.4 | 285.8/625.2 | 44.50/118.50 | 18.68%/25.84% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 24 | observed | 0.000 | 175.1 | 5656.2/18645.7 | 549.6/2370.3 | 65.77/183.02 | 21.92%/29.88% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 24 | round2_v3 | 4.750 | 504.2 | 6006.9/22905.4 | 772.7/895.7 | 56.16/147.50 | 17.45%/23.31% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 24 | causal_rebuild | 3.367 | 1484.8 | 6210.1/28307.4 | 280.8/607.7 | 65.96/167.25 | 20.30%/25.87% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 24 | causal_fixed_replay | 3.067 | 1425.6 | 6846.2/25014.7 | 281.3/607.8 | 70.66/171.75 | 19.47%/25.84% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 3 | observed | 1.433 | 1192.6 | 206.3/711.2 | 55.6/81.8 | 0.00/0.00 | 4.20%/6.74% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 3 | round2_v3 | 1.450 | 138.3 | 655.5/997.3 | 440.0/551.2 | 0.18/0.50 | 2.80%/4.19% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 3 | causal_rebuild | 0.883 | 761.0 | 253.0/418.9 | 107.0/208.0 | 0.06/0.25 | 4.48%/7.32% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 3 | causal_fixed_replay | 1.400 | 987.2 | 299.0/765.9 | 70.6/145.2 | 0.08/0.50 | 4.16%/6.45% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 9 | observed | 0.417 | 838.1 | 2907.2/14971.2 | 184.2/479.7 | 1.95/11.75 | 9.21%/16.76% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 9 | round2_v3 | 3.583 | 347.0 | 851.2/1270.4 | 562.4/678.5 | 0.78/1.75 | 8.90%/15.18% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 9 | causal_rebuild | 1.667 | 996.0 | 755.9/3151.9 | 239.6/415.2 | 1.33/3.00 | 12.14%/22.25% | 0/0/0/0 | 0.0/0.0/0.0 |
| off | 9 | causal_fixed_replay | 1.567 | 889.3 | 2705.6/10710.2 | 229.4/397.2 | 9.42/23.00 | 10.32%/17.67% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 12 | observed | 0.917 | 1743.9 | 747.4/2189.0 | 186.0/476.3 | 6.48/36.50 | 17.44%/28.27% | 2/2/0/0 | 331.6/174.6/152.1 |
| on | 12 | round2_v3 | 4.083 | 415.3 | 1008.9/1578.1 | 650.7/838.5 | 1.30/2.50 | 12.26%/21.44% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 12 | causal_rebuild | 3.367 | 1061.8 | 553.6/1821.6 | 274.5/562.1 | 1.30/3.51 | 14.23%/25.02% | 83/82/1/0 | 203.1/59.3/12541.7 |
| on | 12 | causal_fixed_replay | 3.233 | 1068.4 | 832.2/3351.2 | 262.6/555.5 | 2.07/5.25 | 14.11%/25.14% | 1/1/0/0 | 352.3/188.0/148.7 |
| on | 15 | observed | 0.733 | 1724.4 | 1516.1/10986.2 | 212.8/594.0 | 22.26/83.50 | 18.95%/28.42% | 2/2/0/0 | 301.6/169.1/172.3 |
| on | 15 | round2_v3 | 4.450 | 453.9 | 1743.1/6808.8 | 724.2/861.2 | 7.04/31.00 | 14.93%/23.23% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 15 | causal_rebuild | 3.033 | 1380.8 | 2249.0/12840.7 | 258.7/515.3 | 13.99/56.01 | 17.21%/26.08% | 41/39/2/0 | 202.3/59.0/12493.4 |
| on | 15 | causal_fixed_replay | 2.867 | 1356.0 | 2503.7/12895.5 | 263.3/516.9 | 15.26/57.00 | 17.30%/26.21% | 2/2/0/0 | 301.6/169.1/172.3 |
| on | 18 | observed | 0.783 | 1853.9 | 1900.2/17361.8 | 210.5/537.6 | 37.87/121.50 | 20.45%/28.96% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 18 | round2_v3 | 4.600 | 482.0 | 3140.4/14715.6 | 753.6/911.2 | 20.47/67.50 | 16.29%/23.24% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 18 | causal_rebuild | 8.017 | 1154.3 | 1205.3/6157.0 | 294.9/569.1 | 5.60/24.25 | 16.12%/25.10% | 70/70/0/0 | 217.5/63.5/13436.0 |
| on | 18 | causal_fixed_replay | 8.067 | 1168.4 | 1548.6/6674.5 | 288.4/568.9 | 6.80/24.00 | 16.04%/25.24% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 21 | observed | 0.817 | 1798.5 | 2048.0/17718.7 | 225.0/588.5 | 57.02/161.50 | 20.74%/28.80% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 21 | round2_v3 | 4.733 | 494.1 | 4929.6/19760.7 | 758.6/907.5 | 38.18/106.75 | 16.92%/23.47% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 21 | causal_rebuild | 4.267 | 1400.8 | 4279.6/20014.1 | 286.0/623.3 | 39.35/114.52 | 19.31%/25.80% | 28/27/1/0 | 204.5/59.7/12631.5 |
| on | 21 | causal_fixed_replay | 4.133 | 1384.9 | 4995.9/23487.3 | 284.4/623.5 | 42.35/117.51 | 19.03%/26.06% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 24 | observed | 0.933 | 1819.0 | 3014.1/27948.1 | 232.2/607.7 | 74.38/201.25 | 21.77%/28.87% | 1/1/0/0 | 339.7/338.1/94.5 |
| on | 24 | round2_v3 | 4.700 | 503.0 | 6003.8/23274.2 | 769.7/936.2 | 56.23/145.75 | 17.45%/23.39% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 24 | causal_rebuild | 3.100 | 1477.2 | 5779.9/28971.2 | 281.5/607.8 | 66.81/173.00 | 20.32%/26.34% | 22/22/0/0 | 198.0/57.8/12228.5 |
| on | 24 | causal_fixed_replay | 3.250 | 1453.2 | 6742.0/26958.3 | 282.2/607.7 | 68.13/168.50 | 19.92%/25.92% | 1/1/0/0 | 339.7/338.1/94.5 |
| on | 3 | observed | 0.683 | 743.4 | 280.3/935.2 | 106.8/290.7 | 0.00/0.00 | 4.63%/7.87% | 2/2/0/0 | 707.2/213.9/110.5 |
| on | 3 | round2_v3 | 1.450 | 138.4 | 652.3/997.8 | 440.0/551.2 | 0.18/0.50 | 2.80%/4.19% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 3 | causal_rebuild | 1.050 | 803.0 | 259.2/444.1 | 95.3/196.2 | 0.07/0.25 | 4.29%/7.00% | 85/84/1/0 | 266.5/77.8/16458.2 |
| on | 3 | causal_fixed_replay | 1.383 | 989.0 | 296.7/614.6 | 71.1/145.2 | 0.08/0.50 | 4.15%/6.36% | 1/1/0/0 | 314.3/133.6/71.1 |
| on | 9 | observed | 0.817 | 1586.0 | 348.4/842.8 | 169.0/468.0 | 0.10/0.50 | 14.09%/26.78% | 1/1/0/0 | 1045.6/427.3/93.5 |
| on | 9 | round2_v3 | 3.583 | 347.6 | 853.7/1220.8 | 562.8/663.5 | 0.79/1.75 | 8.91%/15.21% | 0/0/0/0 | 0.0/0.0/0.0 |
| on | 9 | causal_rebuild | 1.817 | 961.0 | 830.8/3090.2 | 244.6/397.2 | 1.58/5.25 | 11.89%/21.28% | 89/87/2/0 | 233.0/68.0/14388.4 |
| on | 9 | causal_fixed_replay | 1.600 | 936.1 | 2162.2/10526.0 | 242.6/416.8 | 4.51/11.25 | 11.38%/21.36% | 1/1/0/0 | 1045.6/427.3/93.5 |

## Fixed batch schedule replay

Observed held-out batches are fixed inputs for this local service model check.

| Mode | QPS | Batches | Round 2 MAE ms / MAPE | Causal MAE ms / MAPE |
|:---:|---:|---:|---:|---:|
| off | 12 | 2539 | 333.21 / 467.45% | 21.57 / 24.82% |
| off | 15 | 1895 | 348.49 / 387.63% | 32.12 / 26.46% |
| off | 18 | 1898 | 360.93 / 393.12% | 33.83 / 27.88% |
| off | 21 | 1786 | 365.29 / 378.97% | 38.77 / 28.59% |
| off | 24 | 1774 | 370.31 / 370.86% | 38.11 / 26.10% |
| off | 3 | 4445 | 280.81 / 602.16% | 17.72 / 31.88% |
| off | 9 | 2600 | 320.52 / 547.27% | 27.71 / 29.68% |
| on | 12 | 1923 | 327.85 / 326.81% | 29.84 / 23.38% |
| on | 15 | 1680 | 335.49 / 298.49% | 39.26 / 23.82% |
| on | 18 | 1613 | 337.78 / 280.88% | 34.43 / 22.20% |
| on | 21 | 1495 | 345.13 / 273.86% | 42.28 / 21.14% |
| on | 24 | 1420 | 354.36 / 264.62% | 46.14 / 21.90% |
| on | 3 | 2728 | 275.47 / 458.72% | 43.70 / 39.90% |
| on | 9 | 1941 | 307.04 / 323.91% | 36.72 / 25.97% |

## Fixed route and migration replay

| Mode | QPS | Observed migration starts | Replayed starts | Rejected starts | First-batch route coverage |
|:---:|---:|---:|---:|---:|---:|
| off | 12 | 0 | 0 | 0 | 82.8% |
| off | 15 | 0 | 0 | 0 | 61.9% |
| off | 18 | 0 | 0 | 0 | 54.6% |
| off | 21 | 0 | 0 | 0 | 48.2% |
| off | 24 | 0 | 0 | 0 | 42.9% |
| off | 3 | 0 | 0 | 0 | 99.5% |
| off | 9 | 0 | 0 | 0 | 86.2% |
| on | 12 | 2 | 2 | 1 | 75.8% |
| on | 15 | 2 | 2 | 0 | 59.7% |
| on | 18 | 0 | 0 | 0 | 50.9% |
| on | 21 | 0 | 0 | 0 | 44.6% |
| on | 24 | 1 | 1 | 0 | 39.6% |
| on | 3 | 2 | 2 | 1 | 99.5% |
| on | 9 | 1 | 1 | 0 | 98.2% |

## Accuracy gate

Independent accuracy gate passed: **False**.

| Metric | Passed | Worst held-out run | Error | Limit |
|:---|:---:|:---|---:|---:|
| request_throughput_rps | False | off QPS 15 | inf | 0.200 |
| ttft_mean_ms | False | on QPS 9 | 1.384 | 0.250 |
| ttft_p95_ms | False | on QPS 9 | 2.667 | 0.300 |
| tpot_mean_ms | False | off QPS 3 | 0.923 | 0.200 |
| tpot_p95_ms | False | off QPS 3 | 1.541 | 0.250 |
| queue_mean_requests | False | on QPS 18 | 32.262 | 7.573 |
| queue_p95_requests | False | on QPS 18 | 97.250 | 24.300 |
| kv_mean_fraction | True | on QPS 18 | 4.327 | 5.000 |
| kv_p95_fraction | True | on QPS 9 | 5.506 | 10.000 |
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
