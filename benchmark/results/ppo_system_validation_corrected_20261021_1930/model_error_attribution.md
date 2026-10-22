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
| 12.0 | V0 | 11.9 / 10.1 (17.1%) | 134.8 / 12898.0 | 76.3 / 490.5 (349) | 0.1 / 2.5 | 0.000 / 0.137 | 712 / 608 |
| 12.0 | V1 | 11.9 / 10.1 (17.1%) | 129.3 / 12898.0 | 76.9 / 490.5 (349) | 0.1 / 2.5 | 0.000 / 0.137 | 712 / 608 |
| 12.0 | V2 | 11.9 / 10.1 (17.1%) | 130.3 / 12898.0 | 76.6 / 490.5 (349) | 0.1 / 2.5 | 0.039 / 0.137 | 712 / 608 |
| 12.0 | V3 | 11.9 / 10.1 (17.1%) | 130.9 / 12898.0 | 77.4 / 490.5 (349) | 0.1 / 2.5 | 0.039 / 0.137 | 712 / 608 |
| 15.0 | V0 | 14.8 / 13.8 (7.4%) | 140.4 / 9242.2 | 81.1 / 604.0 (456) | 0.2 / 3.0 | 0.000 / 0.130 | 888 / 827 |
| 15.0 | V1 | 14.8 / 13.8 (7.4%) | 131.8 / 9242.2 | 79.4 / 604.0 (456) | 0.2 / 3.0 | 0.000 / 0.130 | 888 / 827 |
| 15.0 | V2 | 14.8 / 13.8 (7.4%) | 133.7 / 9242.2 | 80.0 / 604.0 (456) | 0.2 / 3.0 | 0.049 / 0.130 | 888 / 827 |
| 15.0 | V3 | 14.8 / 13.8 (7.4%) | 131.4 / 9242.2 | 80.0 / 604.0 (456) | 0.2 / 3.0 | 0.049 / 0.130 | 888 / 827 |
| 18.0 | V0 | 17.8 / 9.7 (84.6%) | 163.6 / 6708.0 | 86.1 / 558.7 (411) | 0.3 / 1.8 | 0.000 / 0.133 | 1069 / 579 |
| 18.0 | V1 | 17.8 / 9.7 (84.6%) | 152.6 / 6708.0 | 82.9 / 558.7 (411) | 0.3 / 1.8 | 0.000 / 0.133 | 1069 / 579 |
| 18.0 | V2 | 17.8 / 9.7 (84.6%) | 150.4 / 6708.0 | 83.8 / 558.7 (411) | 0.3 / 1.8 | 0.060 / 0.133 | 1069 / 579 |
| 18.0 | V3 | 17.8 / 9.7 (84.6%) | 150.1 / 6708.0 | 83.4 / 558.7 (411) | 0.3 / 1.8 | 0.060 / 0.133 | 1069 / 579 |
| 21.0 | V0 | 20.6 / 17.6 (17.3%) | 149.5 / 11877.8 | 90.7 / 976.5 (677) | 0.3 / 13.0 | 0.000 / 0.159 | 1239 / 1056 |
| 21.0 | V1 | 20.6 / 17.6 (17.3%) | 137.9 / 11877.8 | 85.6 / 976.5 (677) | 0.3 / 13.0 | 0.000 / 0.159 | 1239 / 1056 |
| 21.0 | V2 | 20.6 / 17.6 (17.3%) | 137.8 / 11877.8 | 86.0 / 976.5 (677) | 0.3 / 13.0 | 0.071 / 0.159 | 1239 / 1056 |
| 21.0 | V3 | 20.6 / 17.6 (17.3%) | 137.2 / 11877.8 | 85.5 / 976.5 (677) | 0.3 / 13.0 | 0.071 / 0.159 | 1239 / 1056 |
| 9.0 | V0 | 9.2 / 6.7 (37.2%) | 131.4 / 6835.8 | 71.0 / 542.3 (321) | 0.1 / 1.1 | 0.000 / 0.079 | 549 / 400 |
| 9.0 | V1 | 9.2 / 6.7 (37.2%) | 125.2 / 6835.8 | 73.4 / 542.3 (321) | 0.1 / 1.1 | 0.000 / 0.079 | 549 / 400 |
| 9.0 | V2 | 9.2 / 6.7 (37.2%) | 125.6 / 6835.8 | 73.4 / 542.3 (321) | 0.1 / 1.1 | 0.028 / 0.079 | 549 / 400 |
| 9.0 | V3 | 9.2 / 6.7 (37.2%) | 126.1 / 6835.8 | 74.6 / 542.3 (321) | 0.1 / 1.1 | 0.028 / 0.079 | 549 / 400 |

Observed HTTP 499s were drain-triggered censored requests, not server failures. The held-out runs contain no true HTTP failures or request timeouts. Across QPS, the unscaled simulator completes too many requests and substantially underpredicts request-level TTFT/TPOT and queue length even when executor iteration residuals are small. This is a system timing/queue transition gap, not evidence that P1/D3 per-iteration regressors need more terms.

### Client versus server request duration

| QPS | Complete server timelines / HTTP 200 | Server duration mean ms | Client minus server duration mean / P50 / P95 ms |
|---:|---:|---:|---:|
| 12.0 | 349 / 608 | 40092.0 | 6163.7 / 7166.0 / 7810.2 |
| 15.0 | 456 / 827 | 29598.4 | 12574.3 / 6642.6 / 30395.2 |
| 18.0 | 414 / 579 | 30909.8 | 23415.5 / 42320.4 / 42788.4 |
| 21.0 | 784 / 1056 | 31208.1 | 14749.8 / 5684.6 / 37890.0 |
| 9.0 | 321 / 400 | 45291.7 | 7678.0 / 1396.4 / 30180.7 |

Client completion timestamps are distinct from the server's per-token timeline. The baseline showed client response completion 2.7–28.6 seconds later at the median on the timeline-complete subset, depending on QPS; completion-window throughput therefore remains a separate collection metric.

This baseline reports the unscaled per-iteration simulator and predates the cadence correction. The scale audit and per-QPS fixed/closed-loop metrics are in `system_validation.json`; the final combined model/configuration is `mig_ppo_calibration_system_validation.yaml`.

## Data coverage and capacity

The empirical output-length distribution uses 10,347 successful calibration responses; censored output lengths are unavailable. The per-instance KV capacity is set to 22,084 blocks and supported by the log inference in the system validation report.
The collection mixes 17 runs before and 3 runs after the migration-slot guard change. Service instrumentation was kept the same, but migration behavior has a code-version split.

No additional real-data collection was run. The current ON-only dataset and the weak system prediction results should be addressed before formal Mig-PPO policy training.
