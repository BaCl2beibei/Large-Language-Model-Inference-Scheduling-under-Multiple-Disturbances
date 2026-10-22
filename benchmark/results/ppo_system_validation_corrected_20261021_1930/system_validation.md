# Seed 404 system validation

Data root: `benchmark/results/ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505`. Held-out seed: 404. Measurement window: 60.0s; replay horizon: 90.0s (30.0s simulated drain).

The collection nominal drain was 10.0s, but seed-404 responses were recorded as late as 80.32s. All request outcome and latency comparisons therefore use the common 90s cutoff; queue and KV snapshots use the shared 60s workload window. Request arrivals use actual client send times joined to the trace, with scheduled times as fallback. Realized response lengths are sampled from successful calibration responses only; seed-404 output lengths are held out for comparison. Drain-censored requests have no realized-length labels, so the empirical length model may underrepresent long outputs.

V0 uses legacy scalar service coefficients and disables KV dynamics. V1 uses calibrated service models and disables KV dynamics. V2 adds dynamic KV allocation and preemption recovery. V3 adds migration-duration residual sampling. All four versions use the same run-level decode output cadence scale, fitted as the median calibration-run ratio of request TPOT to predicted executor iteration time; raw executor predictions remain separately reported. The fixed-event replay and closed-loop greedy baseline are reported separately. Closed-loop action observations refresh once per second; the decision interval remains 50 ms. The greedy baseline is not a trained PPO policy.

## Observed request outcomes

| QPS | Arrivals | HTTP 200 by horizon | Failed by horizon | Timed out | Drain-censored 499 | Unfinished at horizon | HTTP 200 total | TTFT N | TPOT N / HTTP 200 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.0 | 712 | 608 | 0 | 0 | 104 | 0 | 608 | 608 | 349 / 608 |
| 15.0 | 888 | 827 | 0 | 0 | 61 | 0 | 827 | 827 | 456 / 827 |
| 18.0 | 1069 | 579 | 0 | 0 | 490 | 0 | 579 | 579 | 411 / 579 |
| 21.0 | 1239 | 1056 | 0 | 0 | 183 | 0 | 1056 | 1056 | 677 / 1056 |
| 9.0 | 549 | 400 | 0 | 0 | 149 | 0 | 400 | 400 | 321 / 400 |

HTTP 499 drain-censored responses are separated from actual failures and timeouts; they do not by themselves establish an engine failure. `HTTP 200 total` includes recorded responses at any time, while `HTTP 200 by horizon` is the comparable count. The trace output token count is a generation budget: observed realized output tokens average less than that budget. TPOT is available for only a subset of successful responses; sample counts and coverage are retained in JSON.

## Arrival trace timing audit

| QPS | Arrivals using actual send time | Mean actual minus scheduled ms | P95 ms | Maximum ms |
|---:|---:|---:|---:|---:|
| 12.0 | 712 | 2.1 | 4.8 | 88.2 |
| 15.0 | 888 | 2.1 | 5.8 | 57.2 |
| 18.0 | 1069 | 79.2 | 158.0 | 2909.7 |
| 21.0 | 1239 | 2.4 | 3.0 | 285.2 |
| 9.0 | 549 | 1.9 | 4.8 | 13.5 |

## Client and server completion timing

For requests with a complete server per-token timeline, the sum of server token intervals gives an independent server-side request duration. Client response completion can be later because it also includes client/event-loop and response handling time.

| QPS | Full server timelines / HTTP 200 | Server timeline mean / P95 ms | Client minus server timeline mean / P50 / P95 ms |
|---:|---:|---:|---:|
| 12.0 | 349 / 608 | 40092.0 / 62045.2 | 6163.7 / 7166.0 / 7810.2 |
| 15.0 | 456 / 827 | 29598.4 / 44502.1 | 12574.3 / 6642.6 / 30395.2 |
| 18.0 | 414 / 579 | 30909.8 / 41905.5 | 23415.5 / 42320.4 / 42788.4 |
| 21.0 | 784 / 1056 | 31208.1 / 50008.3 | 14749.8 / 5684.6 / 37890.0 |
| 9.0 | 321 / 400 | 45291.7 / 72801.7 | 7678.0 / 1396.4 / 30180.7 |

## System metrics by QPS, replay mode, and version

Latency cells show `simulated / observed`; errors are signed percentage differences against the observed value. Queue and KV values are mean/p95 per-instance measurements over the shared workload window. Successful response throughput is the project summary metric `HTTP 200 responses / 60s`; first-window completion throughput (`completion_sec <= 60`) is also shown separately. The common 90-second horizon bounds simulated completed counts.

| QPS | Replay | Version | Sim complete / failed / unfinished | TTFT avg / P95 ms (sim / obs) | TPOT avg / P95 ms (sim / obs) | TPOT observed N | HTTP 200 /60 req/s (sim / obs; error) | Completed by 60s req/s (sim / obs) | Queue mean / P95 | KV mean / P95 | Mig started / success / fail / abort | Mig cost mean / P95 ms |
|---:|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.0 | fixed events | V0 | 712 / 0 / 0 | 135.2 / 12898.0; 176.8 / 28786.3 | 75.9 / 490.5; 79.9 / 1146.1 | 349 / 608 | 11.9 / 10.1 (17.1%) | 10.6 / 0.0 | 0.1 / 0.5 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V0 | 712 / 0 / 0 | 134.8 / 12898.0; 177.6 / 28786.3 | 76.3 / 490.5; 80.3 / 1146.1 | 349 / 608 | 11.9 / 10.1 (17.1%) | 10.5 / 0.0 | 0.1 / 0.5 | 0.000 / 0.000 | 30 / 29 / 1 / 0 | 564.1 / 805.6 |
| 12.0 | fixed events | V1 | 712 / 0 / 0 | 130.5 / 12898.0; 175.2 / 28786.3 | 82.1 / 490.5; 97.8 / 1146.1 | 349 / 608 | 11.9 / 10.1 (17.1%) | 10.3 / 0.0 | 0.1 / 0.5 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V1 | 712 / 0 / 0 | 129.3 / 12898.0; 171.4 / 28786.3 | 76.9 / 490.5; 85.2 / 1146.1 | 349 / 608 | 11.9 / 10.1 (17.1%) | 10.6 / 0.0 | 0.1 / 0.5 | 0.000 / 0.000 | 63 / 61 / 2 / 0 | 491.9 / 500.8 |
| 12.0 | fixed events | V2 | 712 / 0 / 0 | 130.5 / 12898.0; 175.2 / 28786.3 | 82.1 / 490.5; 97.8 / 1146.1 | 349 / 608 | 11.9 / 10.1 (17.1%) | 10.3 / 0.0 | 0.1 / 0.5 | 0.041 / 0.052 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V2 | 712 / 0 / 0 | 130.3 / 12898.0; 172.8 / 28786.3 | 76.6 / 490.5; 87.6 / 1146.1 | 349 / 608 | 11.9 / 10.1 (17.1%) | 10.5 / 0.0 | 0.1 / 0.5 | 0.039 / 0.049 | 63 / 63 / 0 / 0 | 526.7 / 557.2 |
| 12.0 | fixed events | V3 | 712 / 0 / 0 | 130.5 / 12898.0; 175.2 / 28786.3 | 82.1 / 490.5; 97.8 / 1146.1 | 349 / 608 | 11.9 / 10.1 (17.1%) | 10.3 / 0.0 | 0.1 / 0.5 | 0.041 / 0.052 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V3 | 712 / 0 / 0 | 130.9 / 12898.0; 169.9 / 28786.3 | 77.4 / 490.5; 86.4 / 1146.1 | 349 / 608 | 11.9 / 10.1 (17.1%) | 10.4 / 0.0 | 0.1 / 0.5 | 0.039 / 0.049 | 59 / 57 / 2 / 0 | 584.3 / 1141.0 |
| 15.0 | fixed events | V0 | 888 / 0 / 0 | 140.1 / 9242.2; 184.3 / 26486.6 | 80.7 / 604.0; 85.2 / 1043.8 | 456 / 827 | 14.8 / 13.8 (7.4%) | 13.2 / 0.0 | 0.2 / 0.5 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 931.3 / 1133.5 |
| 15.0 | closed-loop | V0 | 888 / 0 / 0 | 140.4 / 9242.2; 186.4 / 26486.6 | 81.1 / 604.0; 86.7 / 1043.8 | 456 / 827 | 14.8 / 13.8 (7.4%) | 13.2 / 0.0 | 0.2 / 0.5 | 0.000 / 0.000 | 41 / 40 / 1 / 0 | 484.1 / 665.9 |
| 15.0 | fixed events | V1 | 888 / 0 / 0 | 135.7 / 9242.2; 178.9 / 26486.6 | 85.4 / 604.0; 101.6 / 1043.8 | 456 / 827 | 14.8 / 13.8 (7.4%) | 13.1 / 0.0 | 0.2 / 0.5 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 931.3 / 1133.5 |
| 15.0 | closed-loop | V1 | 888 / 0 / 0 | 131.8 / 9242.2; 170.8 / 26486.6 | 79.4 / 604.0; 89.6 / 1043.8 | 456 / 827 | 14.8 / 13.8 (7.4%) | 13.3 / 0.0 | 0.2 / 0.5 | 0.000 / 0.000 | 65 / 64 / 1 / 0 | 495.0 / 523.1 |
| 15.0 | fixed events | V2 | 888 / 0 / 0 | 135.7 / 9242.2; 178.9 / 26486.6 | 85.4 / 604.0; 101.6 / 1043.8 | 456 / 827 | 14.8 / 13.8 (7.4%) | 13.1 / 0.0 | 0.2 / 0.5 | 0.052 / 0.064 | 2 / 2 / 0 / 0 | 931.3 / 1133.5 |
| 15.0 | closed-loop | V2 | 888 / 0 / 0 | 133.7 / 9242.2; 173.2 / 26486.6 | 80.0 / 604.0; 91.8 / 1043.8 | 456 / 827 | 14.8 / 13.8 (7.4%) | 13.3 / 0.0 | 0.2 / 0.5 | 0.049 / 0.061 | 65 / 64 / 1 / 0 | 523.2 / 556.5 |
| 15.0 | fixed events | V3 | 888 / 0 / 0 | 135.7 / 9242.2; 178.9 / 26486.6 | 85.4 / 604.0; 101.6 / 1043.8 | 456 / 827 | 14.8 / 13.8 (7.4%) | 13.1 / 0.0 | 0.2 / 0.5 | 0.052 / 0.064 | 2 / 2 / 0 / 0 | 931.3 / 1133.5 |
| 15.0 | closed-loop | V3 | 888 / 0 / 0 | 131.4 / 9242.2; 172.4 / 26486.6 | 80.0 / 604.0; 88.4 / 1043.8 | 456 / 827 | 14.8 / 13.8 (7.4%) | 13.3 / 0.0 | 0.2 / 0.5 | 0.049 / 0.060 | 62 / 61 / 1 / 0 | 494.0 / 959.6 |
| 18.0 | fixed events | V0 | 1069 / 0 / 0 | 161.2 / 6708.0; 237.9 / 19928.1 | 85.5 / 558.7; 91.4 / 891.2 | 411 / 579 | 17.8 / 9.7 (84.6%) | 15.4 / 0.0 | 0.3 / 0.8 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 814.4 / 1130.7 |
| 18.0 | closed-loop | V0 | 1069 / 0 / 0 | 163.6 / 6708.0; 274.5 / 19928.1 | 86.1 / 558.7; 92.2 / 891.2 | 411 / 579 | 17.8 / 9.7 (84.6%) | 15.4 / 0.0 | 0.3 / 0.8 | 0.000 / 0.000 | 45 / 45 / 0 / 0 | 536.9 / 764.8 |
| 18.0 | fixed events | V1 | 1069 / 0 / 0 | 151.0 / 6708.0; 216.5 / 19928.1 | 87.0 / 558.7; 106.1 / 891.2 | 411 / 579 | 17.8 / 9.7 (84.6%) | 15.4 / 0.0 | 0.3 / 0.8 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 814.4 / 1130.7 |
| 18.0 | closed-loop | V1 | 1069 / 0 / 0 | 152.6 / 6708.0; 238.6 / 19928.1 | 82.9 / 558.7; 93.5 / 891.2 | 411 / 579 | 17.8 / 9.7 (84.6%) | 15.5 / 0.0 | 0.3 / 0.8 | 0.000 / 0.000 | 65 / 65 / 0 / 0 | 498.0 / 513.5 |
| 18.0 | fixed events | V2 | 1069 / 0 / 0 | 151.0 / 6708.0; 216.5 / 19928.1 | 87.0 / 558.7; 106.1 / 891.2 | 411 / 579 | 17.8 / 9.7 (84.6%) | 15.4 / 0.0 | 0.3 / 0.8 | 0.062 / 0.077 | 2 / 2 / 0 / 0 | 814.4 / 1130.7 |
| 18.0 | closed-loop | V2 | 1069 / 0 / 0 | 150.4 / 6708.0; 214.2 / 19928.1 | 83.8 / 558.7; 95.3 / 891.2 | 411 / 579 | 17.8 / 9.7 (84.6%) | 15.5 / 0.0 | 0.3 / 0.8 | 0.060 / 0.074 | 66 / 66 / 0 / 0 | 518.9 / 551.4 |
| 18.0 | fixed events | V3 | 1069 / 0 / 0 | 151.0 / 6708.0; 216.5 / 19928.1 | 87.0 / 558.7; 106.1 / 891.2 | 411 / 579 | 17.8 / 9.7 (84.6%) | 15.4 / 0.0 | 0.3 / 0.8 | 0.062 / 0.077 | 2 / 2 / 0 / 0 | 814.4 / 1130.7 |
| 18.0 | closed-loop | V3 | 1069 / 0 / 0 | 150.1 / 6708.0; 213.5 / 19928.1 | 83.4 / 558.7; 98.6 / 891.2 | 411 / 579 | 17.8 / 9.7 (84.6%) | 15.4 / 0.0 | 0.3 / 0.8 | 0.060 / 0.074 | 63 / 63 / 0 / 0 | 549.1 / 965.2 |
| 21.0 | fixed events | V0 | 1239 / 0 / 0 | 148.4 / 11877.8; 195.0 / 27163.8 | 89.9 / 976.5; 94.7 / 1676.4 | 677 / 1056 | 20.6 / 17.6 (17.3%) | 17.8 / 0.0 | 0.3 / 0.8 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V0 | 1239 / 0 / 0 | 149.5 / 11877.8; 195.4 / 27163.8 | 90.7 / 976.5; 96.6 / 1676.4 | 677 / 1056 | 20.6 / 17.6 (17.3%) | 17.8 / 0.0 | 0.3 / 0.8 | 0.000 / 0.000 | 57 / 57 / 0 / 0 | 662.5 / 662.8 |
| 21.0 | fixed events | V1 | 1239 / 0 / 0 | 141.6 / 11877.8; 189.7 / 27163.8 | 91.1 / 976.5; 114.5 / 1676.4 | 677 / 1056 | 20.6 / 17.6 (17.3%) | 17.8 / 0.0 | 0.3 / 0.8 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V1 | 1239 / 0 / 0 | 137.9 / 11877.8; 179.4 / 27163.8 | 85.6 / 976.5; 95.9 / 1676.4 | 677 / 1056 | 20.6 / 17.6 (17.3%) | 18.1 / 0.0 | 0.3 / 0.8 | 0.000 / 0.000 | 65 / 63 / 2 / 0 | 662.9 / 669.1 |
| 21.0 | fixed events | V2 | 1239 / 0 / 0 | 141.6 / 11877.8; 189.7 / 27163.8 | 91.1 / 976.5; 114.5 / 1676.4 | 677 / 1056 | 20.6 / 17.6 (17.3%) | 17.8 / 0.0 | 0.3 / 0.8 | 0.075 / 0.091 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V2 | 1239 / 0 / 0 | 137.8 / 11877.8; 181.3 / 27163.8 | 86.0 / 976.5; 96.4 / 1676.4 | 677 / 1056 | 20.6 / 17.6 (17.3%) | 18.0 / 0.0 | 0.3 / 0.8 | 0.071 / 0.086 | 65 / 65 / 0 / 0 | 504.4 / 514.2 |
| 21.0 | fixed events | V3 | 1239 / 0 / 0 | 141.6 / 11877.8; 189.7 / 27163.8 | 91.1 / 976.5; 114.5 / 1676.4 | 677 / 1056 | 20.6 / 17.6 (17.3%) | 17.8 / 0.0 | 0.3 / 0.8 | 0.075 / 0.091 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V3 | 1239 / 0 / 0 | 137.2 / 11877.8; 179.2 / 27163.8 | 85.5 / 976.5; 94.7 / 1676.4 | 677 / 1056 | 20.6 / 17.6 (17.3%) | 18.0 / 0.0 | 0.3 / 0.8 | 0.071 / 0.086 | 64 / 64 / 0 / 0 | 508.6 / 898.6 |
| 9.0 | fixed events | V0 | 549 / 0 / 0 | 132.2 / 6835.8; 170.1 / 25031.9 | 70.7 / 542.3; 74.3 / 939.8 | 321 / 400 | 9.2 / 6.7 (37.2%) | 8.0 / 0.0 | 0.1 / 0.2 | 0.000 / 0.000 | 1 / 1 / 0 / 0 | 319.1 / 319.1 |
| 9.0 | closed-loop | V0 | 549 / 0 / 0 | 131.4 / 6835.8; 172.6 / 25031.9 | 71.0 / 542.3; 75.3 / 939.8 | 321 / 400 | 9.2 / 6.7 (37.2%) | 8.1 / 0.0 | 0.1 / 0.2 | 0.000 / 0.000 | 22 / 22 / 0 / 0 | 474.0 / 669.7 |
| 9.0 | fixed events | V1 | 549 / 0 / 0 | 130.1 / 6835.8; 171.7 / 25031.9 | 79.3 / 542.3; 93.2 / 939.8 | 321 / 400 | 9.2 / 6.7 (37.2%) | 7.9 / 0.0 | 0.1 / 0.5 | 0.000 / 0.000 | 1 / 1 / 0 / 0 | 319.1 / 319.1 |
| 9.0 | closed-loop | V1 | 549 / 0 / 0 | 125.2 / 6835.8; 166.9 / 25031.9 | 73.4 / 542.3; 82.4 / 939.8 | 321 / 400 | 9.2 / 6.7 (37.2%) | 8.0 / 0.0 | 0.1 / 0.2 | 0.000 / 0.000 | 63 / 63 / 0 / 0 | 489.6 / 509.6 |
| 9.0 | fixed events | V2 | 549 / 0 / 0 | 130.1 / 6835.8; 171.7 / 25031.9 | 79.3 / 542.3; 93.2 / 939.8 | 321 / 400 | 9.2 / 6.7 (37.2%) | 7.9 / 0.0 | 0.1 / 0.5 | 0.030 / 0.037 | 1 / 1 / 0 / 0 | 319.1 / 319.1 |
| 9.0 | closed-loop | V2 | 549 / 0 / 0 | 125.6 / 6835.8; 165.8 / 25031.9 | 73.4 / 542.3; 82.4 / 939.8 | 321 / 400 | 9.2 / 6.7 (37.2%) | 8.0 / 0.0 | 0.1 / 0.2 | 0.028 / 0.036 | 63 / 61 / 2 / 0 | 529.2 / 551.4 |
| 9.0 | fixed events | V3 | 549 / 0 / 0 | 130.1 / 6835.8; 171.7 / 25031.9 | 79.3 / 542.3; 93.2 / 939.8 | 321 / 400 | 9.2 / 6.7 (37.2%) | 7.9 / 0.0 | 0.1 / 0.5 | 0.030 / 0.037 | 1 / 1 / 0 / 0 | 319.1 / 319.1 |
| 9.0 | closed-loop | V3 | 549 / 0 / 0 | 126.1 / 6835.8; 170.7 / 25031.9 | 74.6 / 542.3; 83.1 / 939.8 | 321 / 400 | 9.2 / 6.7 (37.2%) | 8.0 / 0.0 | 0.1 / 0.2 | 0.028 / 0.036 | 58 / 57 / 1 / 0 | 607.0 / 1238.5 |

## Fixed migration replay coverage

The event replay consumes logged event time, source, destination, blocks, duration, and terminal status. Event start time is approximated by watcher observation time minus the recorded duration. Requests linked to a logged migration are routed to the first logged source only in this fixed replay. The closed-loop run receives no logged route map or event schedule.

| QPS | Logged migration rows | Linked replay requests | Logged status counts | V2 started / nonstarted / still queued | Median start delay ms |
|---:|---:|---:|:---|---:|---:|
| 12.0 | 2 | 2 | {'MigrationStatus.FINISHED': 2} | 2 / 0 / 0 | -29.5 |
| 15.0 | 4 | 3 | {'MigrationStatus.ABORTED_DST': 1, 'MigrationStatus.FINISHED': 3} | 2 / 1 / 0 | 329.8 |
| 18.0 | 8 | 5 | {'MigrationStatus.ABORTED_DST': 3, 'MigrationStatus.FINISHED': 5} | 2 / 3 / 0 | -27.5 |
| 21.0 | 7 | 6 | {'MigrationStatus.ABORTED_DST': 1, 'MigrationStatus.FINISHED': 6} | 2 / 4 / 0 | 212.3 |
| 9.0 | 3 | 3 | {'MigrationStatus.FINISHED': 3} | 1 / 2 / 0 | -9.7 |

## Engine iteration validation

Service intervals are classified against each run's measurement epoch. A sample is in-window only when its estimated start and end both fall within the 60-second measurement period. Labels are executor iteration wall times, not isolated GPU time or request latency.

| QPS | Phase | N | Observed mean / P95 ms | Model MAE / RMSE ms | Bias ms | Residual P05 / P50 / P95 ms | Labels >500 ms | Abs residual >250 ms |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 9.0 | prefill | 192 | 87.9 / 152.7 | 29.7 / 43.9 | 12.9 | -42.8 / 21.6 / 46.4 | 1 | 1 |
| 9.0 | decode | 1491 | 74.9 / 143.7 | 21.3 / 34.7 | 0.1 | -60.1 / 9.1 / 29.5 | 1 | 2 |
| 12.0 | prefill | 231 | 95.5 / 171.1 | 31.9 / 38.7 | 10.5 | -68.7 / 21.2 / 48.5 | 0 | 0 |
| 12.0 | decode | 1610 | 79.6 / 154.7 | 23.0 / 43.3 | 2.9 | -64.4 / 10.0 / 38.7 | 1 | 2 |
| 15.0 | prefill | 301 | 122.2 / 204.7 | 50.0 / 155.5 | -9.7 | -90.9 / 17.7 / 48.3 | 4 | 7 |
| 15.0 | decode | 1326 | 90.7 / 164.3 | 36.2 / 122.5 | 2.3 | -68.8 / 15.0 / 54.2 | 7 | 7 |
| 18.0 | prefill | 339 | 119.8 / 246.3 | 45.4 / 65.4 | -4.9 | -140.4 / 17.0 / 49.8 | 1 | 1 |
| 18.0 | decode | 1282 | 84.6 / 169.6 | 34.5 / 60.0 | 10.4 | -65.5 / 21.4 / 55.0 | 5 | 9 |
| 21.0 | prefill | 280 | 120.2 / 257.8 | 46.9 / 67.6 | -4.7 | -139.0 / 21.4 / 51.1 | 1 | 4 |
| 21.0 | decode | 854 | 94.0 / 211.6 | 42.9 / 72.5 | 11.5 | -92.0 / 26.6 / 64.2 | 2 | 10 |

### Phase classification counts

`service interval [latest deduplicated snapshot timestamp - executor wall time, snapshot timestamp] relative to min(actual_send_wall_time - actual_send_sec); only intervals wholly inside the 60s workload window are measurement samples`

| Kind and phase | Samples |
|:---|---:|
| decode:measurement | 6563 |
| decode:warmup | 4408 |
| prefill:measurement | 1343 |
| prefill:warmup | 541 |

### Grouped calibration CV after phase filtering

The selected P1/D3 structures are evaluated on calibration runs with service intervals wholly inside measurement. This is a diagnostic refit by held-out run; seed 404 is excluded.

| Stage | Model | CV phase | N | Groups | MAE / RMSE ms | R² | Macro run MAE ms |
|:---|:---:|:---|---:|---:|---:|---:|---:|
| prefill | P1 | all phases (prior report) | 4893 | n/a | 41.4 / 93.3 | 0.038 | n/a |
| prefill | P1 | measurement only | 3900 | 15 | 45.9 / 102.4 | 0.026 | 45.0 |
| decode | D3 | all phases (prior report) | 32347 | n/a | 26.5 / 101.8 | 0.029 | n/a |
| decode | D3 | measurement only | 19742 | 15 | 30.4 / 83.0 | 0.051 | 31.7 |

### Model residuals by phase

| Kind | Phase | N | MAE / RMSE ms | Bias ms | Residual P05 / P50 / P95 ms | Labels >500 ms | Abs residual >250 ms |
|:---|:---|---:|---:|---:|---:|---:|---:|
| prefill | all | 1884 | 42.5 / 206.6 | 0.7 | -88.2 / 21.9 / 48.3 | 8 | 14 |
| prefill | measurement | 1343 | 42.2 / 89.4 | -0.8 | -106.0 / 19.9 / 49.1 | 7 | 13 |
| prefill | warmup | 541 | 43.3 / 358.8 | 4.3 | -24.1 / 25.4 / 44.7 | 1 | 1 |
| prefill | drain | 0 | n/a / n/a | n/a | n/a / n/a / n/a | 0 | 0 |
| prefill | boundary | 0 | n/a / n/a | n/a | n/a / n/a / n/a | 0 | 0 |
| decode | all | 10971 | 24.0 / 57.7 | 3.9 | -56.4 / 9.9 / 44.4 | 17 | 33 |
| decode | measurement | 6563 | 30.1 / 71.8 | 4.7 | -67.6 / 14.0 / 50.2 | 16 | 30 |
| decode | warmup | 4408 | 14.8 / 24.6 | 2.7 | -39.4 / 6.6 / 26.7 | 1 | 3 |
| decode | drain | 0 | n/a / n/a | n/a | n/a / n/a / n/a | 0 | 0 |
| decode | boundary | 0 | n/a / n/a | n/a | n/a / n/a / n/a | 0 | 0 |

## Calibration and system limitations

- The output-length model uses 10347 successful seed-101/202/303 measurement responses. It is conditioned on input length and max-token budget. Unobserved drain-censored response lengths create selection bias; this model is an empirical approximation.
- Current collection is Migration ON only. Fixed-event replay evaluates those logged event conditions, while closed-loop results use the simulator's migration cost model and configured failure probability.
- Migration cost labels are end-to-end migration wall times. No measured service-pause label exists; simulated blocking during migration is an explicit approximation.
- Logs do not provide exact initial worker routing, complete scheduler decisions, per-request block allocation transitions, or mixed-batch overlap semantics. Initial routing uses least projected work with the calibrated expected response length; logged source routing is restricted to fixed replay.
- Queue and KV comparisons use per-instance units. Observed data are polling snapshots; simulated values are sampled on the event-driven replay clock.
- Fixed replay and greedy closed-loop results test system behavior under different controls. Closed-loop greedy results do not establish PPO policy performance.

- The configured 22,084 KV blocks per instance are supported by 44,111 log-derived free-block/usage samples (p05/median/p95 22,084/22,084/22,084); this remains a log inference, not an independent hardware capacity measurement.
The calibrated service models remain P1/D3 with the previously selected migration model. The report preserves all QPS/version/mode metrics and error calculations in `system_validation.json`; the combined final calibration is `mig_ppo_calibration_system_validation.yaml`.
