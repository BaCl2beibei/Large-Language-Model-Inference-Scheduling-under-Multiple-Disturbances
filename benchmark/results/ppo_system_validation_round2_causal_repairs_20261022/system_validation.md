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

Latency cells show `simulated / observed`; errors are signed percentage differences against the observed value. Queue and KV values are per-sample means across instances on a shared 50 ms grid, with p95 across the 60-second sample series. Observed engine snapshots are carried forward to that grid. Successful response throughput is the project summary metric `HTTP 200 responses / 60s`; first-window completion throughput (`completion_sec <= 60`) is also shown separately. The common 90-second horizon bounds simulated completed counts.

| QPS | Replay | Version | Sim complete / failed / unfinished | TTFT avg / P95 ms (sim / obs) | TPOT avg / P95 ms (sim / obs) | TPOT observed N | HTTP 200 /60 req/s (sim / obs; error) | Completed by 60s req/s (sim / obs) | Queue mean / P95 | KV mean / P95 | Mig started / success / fail / abort | Mig cost mean / P95 ms |
|---:|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.0 | fixed events | V0 | 365 / 0 / 347 | 940.8 / 12898.0; 1532.7 / 28786.3 | 669.0 / 490.5; 822.5 / 1146.1 | 349 / 608 | 6.1 / 10.1 (-40.0%) | 3.6 / 0.0 | 1.2 / 2.2 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V0 | 365 / 0 / 347 | 952.7 / 12898.0; 1557.3 / 28786.3 | 671.6 / 490.5; 823.2 / 1146.1 | 349 / 608 | 6.1 / 10.1 (-40.0%) | 3.6 / 0.0 | 1.3 / 2.5 | 0.000 / 0.000 | 74 / 71 / 3 / 0 | 623.9 / 839.0 |
| 12.0 | fixed events | V1 | 380 / 0 / 332 | 939.3 / 12898.0; 1439.7 / 28786.3 | 649.9 / 490.5; 809.4 / 1146.1 | 349 / 608 | 6.3 / 10.1 (-37.5%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V1 | 380 / 0 / 332 | 919.5 / 12898.0; 1428.2 / 28786.3 | 650.7 / 490.5; 821.0 / 1146.1 | 349 / 608 | 6.3 / 10.1 (-37.5%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.000 / 0.000 | 84 / 82 / 2 / 0 | 642.6 / 839.0 |
| 12.0 | fixed events | V2 | 380 / 0 / 332 | 939.3 / 12898.0; 1439.7 / 28786.3 | 649.9 / 490.5; 809.4 / 1146.1 | 349 / 608 | 6.3 / 10.1 (-37.5%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.126 / 0.211 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V2 | 383 / 0 / 329 | 930.4 / 12898.0; 1467.9 / 28786.3 | 656.3 / 490.5; 840.5 / 1146.1 | 349 / 608 | 6.4 / 10.1 (-37.0%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.125 / 0.209 | 87 / 84 / 3 / 0 | 511.4 / 624.2 |
| 12.0 | fixed events | V3 | 380 / 0 / 332 | 939.3 / 12898.0; 1439.7 / 28786.3 | 649.9 / 490.5; 809.4 / 1146.1 | 349 / 608 | 6.3 / 10.1 (-37.5%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.126 / 0.211 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V3 | 377 / 0 / 335 | 934.2 / 12898.0; 1471.3 / 28786.3 | 651.0 / 490.5; 789.5 / 1146.1 | 349 / 608 | 6.3 / 10.1 (-38.0%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.126 / 0.214 | 81 / 80 / 1 / 0 | 533.2 / 1205.5 |
| 15.0 | fixed events | V0 | 398 / 0 / 490 | 2213.7 / 9242.2; 12131.1 / 26486.6 | 727.9 / 604.0; 872.3 / 1043.8 | 456 / 827 | 6.6 / 13.8 (-51.9%) | 4.0 / 0.0 | 6.6 / 31.8 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 931.3 / 1133.5 |
| 15.0 | closed-loop | V0 | 394 / 0 / 494 | 2319.0 / 9242.2; 11583.8 / 26486.6 | 729.4 / 604.0; 872.9 / 1043.8 | 456 / 827 | 6.6 / 13.8 (-52.4%) | 3.9 / 0.0 | 7.0 / 32.3 | 0.000 / 0.000 | 51 / 50 / 1 / 0 | 619.8 / 764.8 |
| 15.0 | fixed events | V1 | 417 / 0 / 471 | 2189.0 / 9242.2; 8789.9 / 26486.6 | 712.8 / 604.0; 933.3 / 1043.8 | 456 / 827 | 7.0 / 13.8 (-49.6%) | 4.2 / 0.0 | 6.1 / 28.5 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 931.3 / 1133.5 |
| 15.0 | closed-loop | V1 | 415 / 0 / 473 | 2052.9 / 9242.2; 9722.3 / 26486.6 | 713.5 / 604.0; 872.2 / 1043.8 | 456 / 827 | 6.9 / 13.8 (-49.8%) | 4.0 / 0.0 | 6.2 / 31.2 | 0.000 / 0.000 | 57 / 57 / 0 / 0 | 612.3 / 764.8 |
| 15.0 | fixed events | V2 | 417 / 0 / 471 | 2189.0 / 9242.2; 8789.9 / 26486.6 | 712.8 / 604.0; 933.3 / 1043.8 | 456 / 827 | 7.0 / 13.8 (-49.6%) | 4.2 / 0.0 | 6.1 / 28.5 | 0.151 / 0.230 | 2 / 2 / 0 / 0 | 931.3 / 1133.5 |
| 15.0 | closed-loop | V2 | 417 / 0 / 471 | 2044.0 / 9242.2; 9502.9 / 26486.6 | 715.3 / 604.0; 855.6 / 1043.8 | 456 / 827 | 7.0 / 13.8 (-49.6%) | 4.1 / 0.0 | 6.0 / 30.5 | 0.152 / 0.231 | 62 / 61 / 1 / 0 | 520.0 / 636.3 |
| 15.0 | fixed events | V3 | 417 / 0 / 471 | 2189.0 / 9242.2; 8789.9 / 26486.6 | 712.8 / 604.0; 933.3 / 1043.8 | 456 / 827 | 7.0 / 13.8 (-49.6%) | 4.2 / 0.0 | 6.1 / 28.5 | 0.151 / 0.230 | 2 / 2 / 0 / 0 | 931.3 / 1133.5 |
| 15.0 | closed-loop | V3 | 415 / 0 / 473 | 2047.5 / 9242.2; 9899.1 / 26486.6 | 710.2 / 604.0; 866.4 / 1043.8 | 456 / 827 | 6.9 / 13.8 (-49.8%) | 4.0 / 0.0 | 6.1 / 30.5 | 0.152 / 0.231 | 53 / 53 / 0 / 0 | 582.6 / 1138.0 |
| 18.0 | fixed events | V0 | 413 / 0 / 656 | 3547.5 / 6708.0; 18451.6 / 19928.1 | 748.2 / 558.7; 873.1 / 891.2 | 411 / 579 | 6.9 / 9.7 (-28.7%) | 4.2 / 0.0 | 19.0 / 66.8 | 0.000 / 0.000 | 3 / 3 / 0 / 0 | 660.5 / 1095.5 |
| 18.0 | closed-loop | V0 | 410 / 0 / 659 | 3435.3 / 6708.0; 17747.5 / 19928.1 | 747.4 / 558.7; 872.4 / 891.2 | 411 / 579 | 6.8 / 9.7 (-29.2%) | 4.2 / 0.0 | 19.3 / 66.8 | 0.000 / 0.000 | 42 / 42 / 0 / 0 | 664.0 / 839.0 |
| 18.0 | fixed events | V1 | 438 / 0 / 631 | 3663.3 / 6708.0; 17959.7 / 19928.1 | 730.4 / 558.7; 868.3 / 891.2 | 411 / 579 | 7.3 / 9.7 (-24.4%) | 4.4 / 0.0 | 18.5 / 63.2 | 0.000 / 0.000 | 3 / 3 / 0 / 0 | 660.5 / 1095.5 |
| 18.0 | closed-loop | V1 | 433 / 0 / 636 | 3617.8 / 6708.0; 19356.2 / 19928.1 | 734.2 / 558.7; 880.3 / 891.2 | 411 / 579 | 7.2 / 9.7 (-25.2%) | 4.3 / 0.0 | 18.6 / 64.2 | 0.000 / 0.000 | 48 / 48 / 0 / 0 | 626.2 / 813.0 |
| 18.0 | fixed events | V2 | 438 / 0 / 631 | 3663.3 / 6708.0; 17959.7 / 19928.1 | 730.4 / 558.7; 868.3 / 891.2 | 411 / 579 | 7.3 / 9.7 (-24.4%) | 4.4 / 0.0 | 18.5 / 63.2 | 0.164 / 0.232 | 3 / 3 / 0 / 0 | 660.5 / 1095.5 |
| 18.0 | closed-loop | V2 | 435 / 0 / 634 | 3747.2 / 6708.0; 19403.4 / 19928.1 | 734.4 / 558.7; 897.3 / 891.2 | 411 / 579 | 7.2 / 9.7 (-24.9%) | 4.3 / 0.0 | 18.9 / 65.3 | 0.164 / 0.232 | 51 / 51 / 0 / 0 | 509.8 / 636.3 |
| 18.0 | fixed events | V3 | 438 / 0 / 631 | 3663.3 / 6708.0; 17959.7 / 19928.1 | 730.4 / 558.7; 868.3 / 891.2 | 411 / 579 | 7.3 / 9.7 (-24.4%) | 4.4 / 0.0 | 18.5 / 63.2 | 0.164 / 0.232 | 3 / 3 / 0 / 0 | 660.5 / 1095.5 |
| 18.0 | closed-loop | V3 | 427 / 0 / 642 | 3592.3 / 6708.0; 16401.2 / 19928.1 | 729.4 / 558.7; 872.3 / 891.2 | 411 / 579 | 7.1 / 9.7 (-26.3%) | 4.3 / 0.0 | 18.7 / 64.0 | 0.164 / 0.232 | 46 / 46 / 0 / 0 | 497.1 / 929.7 |
| 21.0 | fixed events | V0 | 425 / 0 / 814 | 4720.3 / 11877.8; 22677.9 / 27163.8 | 762.1 / 976.5; 873.4 / 1676.4 | 677 / 1056 | 7.1 / 17.6 (-59.8%) | 4.3 / 0.0 | 35.4 / 104.8 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V0 | 420 / 0 / 819 | 4669.4 / 11877.8; 21519.9 / 27163.8 | 762.4 / 976.5; 872.8 / 1676.4 | 677 / 1056 | 7.0 / 17.6 (-60.2%) | 4.4 / 0.0 | 35.8 / 104.8 | 0.000 / 0.000 | 46 / 46 / 0 / 0 | 662.1 / 662.0 |
| 21.0 | fixed events | V1 | 452 / 0 / 787 | 4758.5 / 11877.8; 23385.6 / 27163.8 | 748.5 / 976.5; 872.5 / 1676.4 | 677 / 1056 | 7.5 / 17.6 (-57.2%) | 4.5 / 0.0 | 33.7 / 101.0 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V1 | 447 / 0 / 792 | 4695.8 / 11877.8; 21749.3 / 27163.8 | 750.0 / 976.5; 858.3 / 1676.4 | 677 / 1056 | 7.5 / 17.6 (-57.7%) | 4.4 / 0.0 | 34.5 / 103.3 | 0.000 / 0.000 | 47 / 46 / 1 / 0 | 662.0 / 662.0 |
| 21.0 | fixed events | V2 | 452 / 0 / 787 | 4758.5 / 11877.8; 23385.6 / 27163.8 | 748.5 / 976.5; 872.5 / 1676.4 | 677 / 1056 | 7.5 / 17.6 (-57.2%) | 4.5 / 0.0 | 33.7 / 101.0 | 0.174 / 0.234 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V2 | 446 / 0 / 793 | 4863.1 / 11877.8; 23840.7 / 27163.8 | 751.1 / 976.5; 858.9 / 1676.4 | 677 / 1056 | 7.4 / 17.6 (-57.8%) | 4.4 / 0.0 | 34.4 / 104.5 | 0.174 / 0.233 | 43 / 43 / 0 / 0 | 502.7 / 502.7 |
| 21.0 | fixed events | V3 | 452 / 0 / 787 | 4758.5 / 11877.8; 23385.6 / 27163.8 | 748.5 / 976.5; 872.5 / 1676.4 | 677 / 1056 | 7.5 / 17.6 (-57.2%) | 4.5 / 0.0 | 33.7 / 101.0 | 0.174 / 0.234 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V3 | 445 / 0 / 794 | 4865.8 / 11877.8; 24586.7 / 27163.8 | 752.2 / 976.5; 869.4 / 1676.4 | 677 / 1056 | 7.4 / 17.6 (-57.9%) | 4.4 / 0.0 | 34.3 / 102.8 | 0.174 / 0.233 | 32 / 32 / 0 / 0 | 563.1 / 947.3 |
| 9.0 | fixed events | V0 | 317 / 0 / 232 | 819.9 / 6835.8; 1230.0 / 25031.9 | 580.8 / 542.3; 680.9 / 939.8 | 321 / 400 | 5.3 / 6.7 (-20.8%) | 3.2 / 0.0 | 0.8 / 1.8 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 750.8 / 1139.4 |
| 9.0 | closed-loop | V0 | 314 / 0 / 235 | 820.4 / 6835.8; 1200.8 / 25031.9 | 587.2 / 542.3; 688.2 / 939.8 | 321 / 400 | 5.2 / 6.7 (-21.5%) | 3.2 / 0.0 | 0.8 / 1.5 | 0.000 / 0.000 | 76 / 76 / 0 / 0 | 614.6 / 839.0 |
| 9.0 | fixed events | V1 | 325 / 0 / 224 | 803.4 / 6835.8; 1171.7 / 25031.9 | 562.3 / 542.3; 667.6 / 939.8 | 321 / 400 | 5.4 / 6.7 (-18.8%) | 3.2 / 0.0 | 0.8 / 1.5 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 750.8 / 1139.4 |
| 9.0 | closed-loop | V1 | 324 / 0 / 225 | 817.5 / 6835.8; 1182.1 / 25031.9 | 562.9 / 542.3; 660.5 / 939.8 | 321 / 400 | 5.4 / 6.7 (-19.0%) | 3.2 / 0.0 | 0.7 / 1.5 | 0.000 / 0.000 | 86 / 84 / 2 / 0 | 610.7 / 839.0 |
| 9.0 | fixed events | V2 | 325 / 0 / 224 | 803.4 / 6835.8; 1171.7 / 25031.9 | 562.3 / 542.3; 667.6 / 939.8 | 321 / 400 | 5.4 / 6.7 (-18.8%) | 3.2 / 0.0 | 0.8 / 1.5 | 0.092 / 0.154 | 2 / 2 / 0 / 0 | 750.8 / 1139.4 |
| 9.0 | closed-loop | V2 | 324 / 0 / 225 | 811.3 / 6835.8; 1189.7 / 25031.9 | 563.8 / 542.3; 664.2 / 939.8 | 321 / 400 | 5.4 / 6.7 (-19.0%) | 3.2 / 0.0 | 0.8 / 1.5 | 0.092 / 0.153 | 88 / 88 / 0 / 0 | 520.1 / 636.3 |
| 9.0 | fixed events | V3 | 325 / 0 / 224 | 803.4 / 6835.8; 1171.7 / 25031.9 | 562.3 / 542.3; 667.6 / 939.8 | 321 / 400 | 5.4 / 6.7 (-18.8%) | 3.2 / 0.0 | 0.8 / 1.5 | 0.092 / 0.154 | 2 / 2 / 0 / 0 | 750.8 / 1139.4 |
| 9.0 | closed-loop | V3 | 325 / 0 / 224 | 814.5 / 6835.8; 1179.9 / 25031.9 | 561.6 / 542.3; 661.1 / 939.8 | 321 / 400 | 5.4 / 6.7 (-18.8%) | 3.2 / 0.0 | 0.8 / 1.5 | 0.092 / 0.152 | 80 / 78 / 2 / 0 | 524.8 / 1127.2 |

## Fixed migration replay coverage

The event replay consumes logged event time, source, destination, blocks, duration, and terminal status. Event start time is approximated by watcher observation time minus the recorded duration. Requests linked to a logged migration are routed to the first logged source only in this fixed replay. The closed-loop run receives no logged route map or event schedule.

| QPS | Logged migration rows | Linked replay requests | Logged status counts | V2 started / nonstarted / still queued | Median start delay ms |
|---:|---:|---:|:---|---:|---:|
| 12.0 | 2 | 2 | {'MigrationStatus.FINISHED': 2} | 2 / 0 / 0 | -29.5 |
| 15.0 | 4 | 3 | {'MigrationStatus.ABORTED_DST': 1, 'MigrationStatus.FINISHED': 3} | 2 / 1 / 0 | 329.8 |
| 18.0 | 8 | 5 | {'MigrationStatus.ABORTED_DST': 3, 'MigrationStatus.FINISHED': 5} | 3 / 2 / 0 | -5.2 |
| 21.0 | 7 | 6 | {'MigrationStatus.ABORTED_DST': 1, 'MigrationStatus.FINISHED': 6} | 2 / 2 / 2 | 212.3 |
| 9.0 | 3 | 3 | {'MigrationStatus.FINISHED': 3} | 2 / 1 / 0 | -9.0 |

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
- Queue and KV comparisons use a common 50 ms sample grid. Observed per-instance engine snapshots are carried forward to each sample; simulator state is sampled after each 50 ms step. Each sample is averaged across instances before mean and p95 aggregation.
- Fixed replay and greedy closed-loop results test system behavior under different controls. Closed-loop greedy results do not establish PPO policy performance.

- The configured 22,084 KV blocks per instance are supported by 44,111 log-derived free-block/usage samples (p05/median/p95 22,084/22,084/22,084); this remains a log inference, not an independent hardware capacity measurement.
The calibrated service models remain P1/D3 with the previously selected migration model. The report preserves all QPS/version/mode metrics and error calculations in `system_validation.json`; the combined final calibration is `mig_ppo_calibration_system_validation.yaml`.

### Collection timing addendum

The statement above that no additional real-data collection was run was accurate when this system-validation report was generated. A later paired QPS 15, seed 505 Migration OFF/ON pilot was collected and is reported separately in `benchmark/results/ppo_migration_pilot_qps15_seed505_20261021/paired_pilot_analysis.md` and `.json`. That single-QPS pilot was not used to refit this validation and does not change the five-QPS admission results.
