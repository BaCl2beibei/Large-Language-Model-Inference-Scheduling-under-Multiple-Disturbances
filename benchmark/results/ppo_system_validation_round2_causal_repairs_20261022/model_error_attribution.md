# Model and system error attribution

This report separates service iteration fit, request-level system behavior, migration costs, and collection censoring. Calibration uses seeds 101/202/303; seed 404 is held out.

## Prefill and decode service models

| Stage | Model | Grouped CV all phases MAE / RMSE ms | Grouped CV measurement-only MAE / RMSE ms | Seed 404 measurement MAE / RMSE ms | Bias ms | Residual P05 / P50 / P95 ms | Labels >500 ms | Abs residual >250 ms |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| prefill | P1 | 41.4 / 93.3 | 45.9 / 102.4 | 42.2 / 89.4 | -0.8 | -106.0 / 19.9 / 49.1 | 7 | 13 |
| decode | D3 | 26.5 / 101.8 | 30.4 / 83.0 | 30.1 / 71.8 | 4.7 | -67.6 / 14.0 / 50.2 | 16 | 30 |

The measurement-only grouped CV is slightly worse in MAE for both stages than the all-phase report (P1 45.9 vs 41.4 ms; D3 30.4 vs 26.5 ms). Measurement-only RMSE falls for D3 (83.0 vs 101.8 ms), while the seed-404 measurement-window P1/D3 errors are 42.2/89.4 and 30.1/71.8 ms. Residual tails remain, but this does not justify a more complex service regressor. Keep P1/D3.

Service labels are executor iteration wall times. They are not request TTFT/TPOT. Phase filtering reduces P1 holdout RMSE from about 217 ms to 89 ms by excluding out-of-window long stalls, while D3 measurement-only RMSE is 72 ms. Large executor stalls are sparse and the logs do not identify a unique cause; they cannot be safely assigned to batch/context features.

## Migration duration

| Model | Split | N | MAE / RMSE ms | Bias ms | Residual P05 / P50 / P95 ms |
|:---|:---|---:|---:|---:|---:|
| M2_huber_pair_shrunk | grouped CV | 74 | 342.3 / 698.5 | -125.5 | -1004.2 / 48.5 / 327.2 |
| M2_huber_pair_shrunk | seed 404 | 31 | 335.0 / 566.9 | -65.0 | -684.8 / 113.1 / 466.2 |
| M2_huber_pair_shrunk (ABORTED_DST) | seed 404 | 5 | 272.8 / 329.9 | 184.4 | -144.4 / 153.4 / 500.1 |
| M2_huber_pair_shrunk (FINISHED) | seed 404 | 26 | 346.9 / 601.9 | -112.9 | -757.2 / 92.9 / 335.2 |

M2 remains high-error and status-dependent; seed 404 has only 31 migration events. Block count alone has weak association with cost, and directed-pair shrinkage rests on few observations. `cost_ms` is end-to-end migration wall time. No measured pause-time label exists, so the environment's request blocking during migration remains an approximation.

## System-level request and resource errors

Closed-loop rows below use the same one-second observation refresh and share the train-only output-length/cadence models. Relative errors are signed against observed request response throughput (`HTTP 200 / 60s`); queue and KV compare per-instance units.

| QPS | Version | Throughput sim / observed (error %) | TTFT avg sim / obs ms | TPOT avg sim / obs ms (observed N) | Queue mean sim / obs | KV mean sim / obs | Sim completed / observed HTTP 200 |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 12.0 | V0 | 6.1 / 10.1 (-40.0%) | 952.7 / 12898.0 | 671.6 / 490.5 (349) | 1.3 / 3.4 | 0.000 / 0.168 | 365 / 608 |
| 12.0 | V1 | 6.3 / 10.1 (-37.5%) | 919.5 / 12898.0 | 650.7 / 490.5 (349) | 1.2 / 3.4 | 0.000 / 0.168 | 380 / 608 |
| 12.0 | V2 | 6.4 / 10.1 (-37.0%) | 930.4 / 12898.0 | 656.3 / 490.5 (349) | 1.2 / 3.4 | 0.125 / 0.168 | 383 / 608 |
| 12.0 | V3 | 6.3 / 10.1 (-38.0%) | 934.2 / 12898.0 | 651.0 / 490.5 (349) | 1.2 / 3.4 | 0.126 / 0.168 | 377 / 608 |
| 15.0 | V0 | 6.6 / 13.8 (-52.4%) | 2319.0 / 9242.2 | 729.4 / 604.0 (456) | 7.0 / 3.6 | 0.000 / 0.144 | 394 / 827 |
| 15.0 | V1 | 6.9 / 13.8 (-49.8%) | 2052.9 / 9242.2 | 713.5 / 604.0 (456) | 6.2 / 3.6 | 0.000 / 0.144 | 415 / 827 |
| 15.0 | V2 | 7.0 / 13.8 (-49.6%) | 2044.0 / 9242.2 | 715.3 / 604.0 (456) | 6.0 / 3.6 | 0.152 / 0.144 | 417 / 827 |
| 15.0 | V3 | 6.9 / 13.8 (-49.8%) | 2047.5 / 9242.2 | 710.2 / 604.0 (456) | 6.1 / 3.6 | 0.152 / 0.144 | 415 / 827 |
| 18.0 | V0 | 6.8 / 9.7 (-29.2%) | 3435.3 / 6708.0 | 747.4 / 558.7 (411) | 19.3 / 1.5 | 0.000 / 0.143 | 410 / 579 |
| 18.0 | V1 | 7.2 / 9.7 (-25.2%) | 3617.8 / 6708.0 | 734.2 / 558.7 (411) | 18.6 / 1.5 | 0.000 / 0.143 | 433 / 579 |
| 18.0 | V2 | 7.2 / 9.7 (-24.9%) | 3747.2 / 6708.0 | 734.4 / 558.7 (411) | 18.9 / 1.5 | 0.164 / 0.143 | 435 / 579 |
| 18.0 | V3 | 7.1 / 9.7 (-26.3%) | 3592.3 / 6708.0 | 729.4 / 558.7 (411) | 18.7 / 1.5 | 0.164 / 0.143 | 427 / 579 |
| 21.0 | V0 | 7.0 / 17.6 (-60.2%) | 4669.4 / 11877.8 | 762.4 / 976.5 (677) | 35.8 / 21.5 | 0.000 / 0.175 | 420 / 1056 |
| 21.0 | V1 | 7.5 / 17.6 (-57.7%) | 4695.8 / 11877.8 | 750.0 / 976.5 (677) | 34.5 / 21.5 | 0.000 / 0.175 | 447 / 1056 |
| 21.0 | V2 | 7.4 / 17.6 (-57.8%) | 4863.1 / 11877.8 | 751.1 / 976.5 (677) | 34.4 / 21.5 | 0.174 / 0.175 | 446 / 1056 |
| 21.0 | V3 | 7.4 / 17.6 (-57.9%) | 4865.8 / 11877.8 | 752.2 / 976.5 (677) | 34.3 / 21.5 | 0.174 / 0.175 | 445 / 1056 |
| 9.0 | V0 | 5.2 / 6.7 (-21.5%) | 820.4 / 6835.8 | 587.2 / 542.3 (321) | 0.8 / 2.1 | 0.000 / 0.107 | 314 / 400 |
| 9.0 | V1 | 5.4 / 6.7 (-19.0%) | 817.5 / 6835.8 | 562.9 / 542.3 (321) | 0.7 / 2.1 | 0.000 / 0.107 | 324 / 400 |
| 9.0 | V2 | 5.4 / 6.7 (-19.0%) | 811.3 / 6835.8 | 563.8 / 542.3 (321) | 0.8 / 2.1 | 0.092 / 0.107 | 324 / 400 |
| 9.0 | V3 | 5.4 / 6.7 (-18.8%) | 814.5 / 6835.8 | 561.6 / 542.3 (321) | 0.8 / 2.1 | 0.092 / 0.107 | 325 / 400 |

Observed HTTP 499s were drain-triggered censored requests, not server failures. The held-out runs contain no true HTTP failures or request timeouts. The calibrated cadence layer is shared across all versions; use the per-QPS errors below to judge whether it closes the request-level timing gap.

### Client versus server request duration

| QPS | Complete server timelines / HTTP 200 | Server duration mean ms | Client minus server duration mean / P50 / P95 ms |
|---:|---:|---:|---:|
| 12.0 | 349 / 608 | 40092.0 | 6163.7 / 7166.0 / 7810.2 |
| 15.0 | 456 / 827 | 29598.4 | 12574.3 / 6642.6 / 30395.2 |
| 18.0 | 414 / 579 | 30909.8 | 23415.5 / 42320.4 / 42788.4 |
| 21.0 | 784 / 1056 | 31208.1 | 14749.8 / 5684.6 / 37890.0 |
| 9.0 | 321 / 400 | 45291.7 | 7678.0 / 1396.4 / 30180.7 |

Client completion timestamps are distinct from the server's per-token timeline. The client-minus-server duration gap varies by QPS on the subset with complete per-token timelines; the table above gives those sample counts and residual distributions. Completion-window throughput therefore remains a separate collection metric.

Decode cadence scale fit from calibration runs only: 5.828 across 15 runs. The scale audit and per-QPS fixed/closed-loop metrics are in `system_validation.json`; the final combined model/configuration is `mig_ppo_calibration_system_validation.yaml`.

## Data coverage and capacity

The empirical output-length distribution uses 10,347 successful calibration responses; censored output lengths are unavailable. The per-instance KV capacity is set to 22,084 blocks and supported by the log inference in the system validation report.
The collection mixes 17 runs before and 3 runs after the migration-slot guard change. Service instrumentation was kept the same, but migration behavior has a code-version split.

No additional real-data collection was run. The current ON-only dataset and the weak system prediction results should be addressed before formal Mig-PPO policy training.
