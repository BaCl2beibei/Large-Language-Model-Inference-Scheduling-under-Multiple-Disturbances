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
| 12.0 | fixed events | V0 | 365 / 0 / 347 | 473.5 / 12898.0; 821.8 / 28786.3 | 701.8 / 490.5; 893.9 / 1146.1 | 349 / 608 | 6.1 / 10.1 (-40.0%) | 3.6 / 0.0 | 1.1 / 2.2 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V0 | 364 / 0 / 348 | 493.1 / 12898.0; 910.6 / 28786.3 | 704.8 / 490.5; 896.9 / 1146.1 | 349 / 608 | 6.1 / 10.1 (-40.1%) | 3.6 / 0.0 | 1.2 / 2.2 | 0.000 / 0.000 | 76 / 73 / 3 / 0 | 554.0 / 1091.6 |
| 12.0 | fixed events | V1 | 384 / 0 / 328 | 461.9 / 12898.0; 771.6 / 28786.3 | 680.0 / 490.5; 866.1 / 1146.1 | 349 / 608 | 6.4 / 10.1 (-36.8%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V1 | 381 / 0 / 331 | 468.6 / 12898.0; 784.7 / 28786.3 | 686.0 / 490.5; 890.3 / 1146.1 | 349 / 608 | 6.3 / 10.1 (-37.3%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.000 / 0.000 | 86 / 84 / 2 / 0 | 606.8 / 839.0 |
| 12.0 | fixed events | V2 | 384 / 0 / 328 | 461.9 / 12898.0; 771.6 / 28786.3 | 680.0 / 490.5; 866.1 / 1146.1 | 349 / 608 | 6.4 / 10.1 (-36.8%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.126 / 0.211 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V2 | 381 / 0 / 331 | 474.9 / 12898.0; 777.4 / 28786.3 | 683.8 / 490.5; 893.1 / 1146.1 | 349 / 608 | 6.3 / 10.1 (-37.3%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.125 / 0.212 | 88 / 87 / 1 / 0 | 509.0 / 636.3 |
| 12.0 | fixed events | V3 | 384 / 0 / 328 | 461.9 / 12898.0; 771.6 / 28786.3 | 680.0 / 490.5; 866.1 / 1146.1 | 349 / 608 | 6.4 / 10.1 (-36.8%) | 3.7 / 0.0 | 1.2 / 2.2 | 0.126 / 0.211 | 2 / 2 / 0 / 0 | 333.2 / 360.5 |
| 12.0 | closed-loop | V3 | 380 / 0 / 332 | 464.2 / 12898.0; 787.9 / 28786.3 | 688.3 / 490.5; 880.5 / 1146.1 | 349 / 608 | 6.3 / 10.1 (-37.5%) | 3.7 / 0.0 | 1.1 / 2.2 | 0.126 / 0.212 | 86 / 85 / 1 / 0 | 501.9 / 976.8 |
| 15.0 | fixed events | V0 | 400 / 0 / 488 | 1692.3 / 9242.2; 9672.1 / 26486.6 | 772.8 / 604.0; 1027.0 / 1043.8 | 456 / 827 | 6.7 / 13.8 (-51.6%) | 4.0 / 0.0 | 6.5 / 30.8 | 0.000 / 0.000 | 3 / 3 / 0 / 0 | 1643.9 / 2877.9 |
| 15.0 | closed-loop | V0 | 397 / 0 / 491 | 1692.4 / 9242.2; 9699.7 / 26486.6 | 774.7 / 604.0; 1016.7 / 1043.8 | 456 / 827 | 6.6 / 13.8 (-52.0%) | 4.0 / 0.0 | 6.4 / 31.8 | 0.000 / 0.000 | 51 / 51 / 0 / 0 | 576.7 / 839.0 |
| 15.0 | fixed events | V1 | 417 / 0 / 471 | 1648.4 / 9242.2; 8795.4 / 26486.6 | 754.1 / 604.0; 966.4 / 1043.8 | 456 / 827 | 7.0 / 13.8 (-49.6%) | 4.2 / 0.0 | 6.1 / 29.8 | 0.000 / 0.000 | 3 / 3 / 0 / 0 | 1643.9 / 2877.9 |
| 15.0 | closed-loop | V1 | 419 / 0 / 469 | 1656.5 / 9242.2; 9595.6 / 26486.6 | 758.3 / 604.0; 984.5 / 1043.8 | 456 / 827 | 7.0 / 13.8 (-49.3%) | 4.0 / 0.0 | 6.1 / 29.8 | 0.000 / 0.000 | 66 / 65 / 1 / 0 | 635.9 / 839.0 |
| 15.0 | fixed events | V2 | 417 / 0 / 471 | 1648.4 / 9242.2; 8795.4 / 26486.6 | 754.1 / 604.0; 966.4 / 1043.8 | 456 / 827 | 7.0 / 13.8 (-49.6%) | 4.2 / 0.0 | 6.1 / 29.8 | 0.151 / 0.230 | 3 / 3 / 0 / 0 | 1643.9 / 2877.9 |
| 15.0 | closed-loop | V2 | 416 / 0 / 472 | 1646.8 / 9242.2; 9909.0 / 26486.6 | 757.6 / 604.0; 983.7 / 1043.8 | 456 / 827 | 6.9 / 13.8 (-49.7%) | 4.1 / 0.0 | 6.1 / 29.3 | 0.152 / 0.232 | 58 / 56 / 2 / 0 | 515.1 / 636.3 |
| 15.0 | fixed events | V3 | 417 / 0 / 471 | 1648.4 / 9242.2; 8795.4 / 26486.6 | 754.1 / 604.0; 966.4 / 1043.8 | 456 / 827 | 7.0 / 13.8 (-49.6%) | 4.2 / 0.0 | 6.1 / 29.8 | 0.151 / 0.230 | 3 / 3 / 0 / 0 | 1643.9 / 2877.9 |
| 15.0 | closed-loop | V3 | 416 / 0 / 472 | 1619.3 / 9242.2; 9452.8 / 26486.6 | 759.1 / 604.0; 1010.1 / 1043.8 | 456 / 827 | 6.9 / 13.8 (-49.7%) | 4.0 / 0.0 | 6.1 / 30.2 | 0.152 / 0.232 | 53 / 53 / 0 / 0 | 525.0 / 1117.4 |
| 18.0 | fixed events | V0 | 412 / 0 / 657 | 3004.2 / 6708.0; 16407.5 / 19928.1 | 793.2 / 558.7; 1035.4 / 891.2 | 411 / 579 | 6.9 / 9.7 (-28.8%) | 4.2 / 0.0 | 19.0 / 66.8 | 0.000 / 0.000 | 3 / 3 / 0 / 0 | 660.5 / 1095.5 |
| 18.0 | closed-loop | V0 | 407 / 0 / 662 | 3196.8 / 6708.0; 17613.4 / 19928.1 | 790.2 / 558.7; 1013.0 / 891.2 | 411 / 579 | 6.8 / 9.7 (-29.7%) | 4.3 / 0.0 | 19.5 / 66.5 | 0.000 / 0.000 | 42 / 42 / 0 / 0 | 611.5 / 1079.0 |
| 18.0 | fixed events | V1 | 434 / 0 / 635 | 3078.9 / 6708.0; 15424.1 / 19928.1 | 774.1 / 558.7; 960.4 / 891.2 | 411 / 579 | 7.2 / 9.7 (-25.0%) | 4.4 / 0.0 | 18.5 / 64.3 | 0.000 / 0.000 | 3 / 3 / 0 / 0 | 660.5 / 1095.5 |
| 18.0 | closed-loop | V1 | 435 / 0 / 634 | 3079.7 / 6708.0; 18181.0 / 19928.1 | 775.3 / 558.7; 1029.2 / 891.2 | 411 / 579 | 7.2 / 9.7 (-24.9%) | 4.4 / 0.0 | 18.5 / 64.2 | 0.000 / 0.000 | 53 / 53 / 0 / 0 | 633.1 / 839.0 |
| 18.0 | fixed events | V2 | 434 / 0 / 635 | 3078.9 / 6708.0; 15424.1 / 19928.1 | 774.1 / 558.7; 960.4 / 891.2 | 411 / 579 | 7.2 / 9.7 (-25.0%) | 4.4 / 0.0 | 18.5 / 64.3 | 0.164 / 0.232 | 3 / 3 / 0 / 0 | 660.5 / 1095.5 |
| 18.0 | closed-loop | V2 | 431 / 0 / 638 | 3062.4 / 6708.0; 17049.2 / 19928.1 | 780.6 / 558.7; 994.9 / 891.2 | 411 / 579 | 7.2 / 9.7 (-25.6%) | 4.3 / 0.0 | 18.8 / 64.5 | 0.164 / 0.232 | 50 / 50 / 0 / 0 | 503.9 / 583.1 |
| 18.0 | fixed events | V3 | 434 / 0 / 635 | 3078.9 / 6708.0; 15424.1 / 19928.1 | 774.1 / 558.7; 960.4 / 891.2 | 411 / 579 | 7.2 / 9.7 (-25.0%) | 4.4 / 0.0 | 18.5 / 64.3 | 0.164 / 0.232 | 3 / 3 / 0 / 0 | 660.5 / 1095.5 |
| 18.0 | closed-loop | V3 | 428 / 0 / 641 | 3042.0 / 6708.0; 17232.9 / 19928.1 | 777.1 / 558.7; 989.3 / 891.2 | 411 / 579 | 7.1 / 9.7 (-26.1%) | 4.3 / 0.0 | 18.8 / 65.2 | 0.164 / 0.233 | 47 / 47 / 0 / 0 | 555.6 / 1215.8 |
| 21.0 | fixed events | V0 | 425 / 0 / 814 | 4186.1 / 11877.8; 22892.3 / 27163.8 | 808.7 / 976.5; 1017.1 / 1676.4 | 677 / 1056 | 7.1 / 17.6 (-59.8%) | 4.3 / 0.0 | 34.8 / 103.8 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V0 | 422 / 0 / 817 | 4238.0 / 11877.8; 21079.9 / 27163.8 | 805.9 / 976.5; 1045.1 / 1676.4 | 677 / 1056 | 7.0 / 17.6 (-60.0%) | 4.3 / 0.0 | 35.6 / 105.3 | 0.000 / 0.000 | 34 / 34 / 0 / 0 | 662.0 / 662.0 |
| 21.0 | fixed events | V1 | 445 / 0 / 794 | 4401.3 / 11877.8; 23650.6 / 27163.8 | 787.9 / 976.5; 987.9 / 1676.4 | 677 / 1056 | 7.4 / 17.6 (-57.9%) | 4.5 / 0.0 | 34.1 / 103.0 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V1 | 448 / 0 / 791 | 4461.5 / 11877.8; 23605.6 / 27163.8 | 794.5 / 976.5; 991.5 / 1676.4 | 677 / 1056 | 7.5 / 17.6 (-57.6%) | 4.5 / 0.0 | 34.4 / 103.3 | 0.000 / 0.000 | 40 / 40 / 0 / 0 | 662.0 / 662.0 |
| 21.0 | fixed events | V2 | 445 / 0 / 794 | 4401.3 / 11877.8; 23650.6 / 27163.8 | 787.9 / 976.5; 987.9 / 1676.4 | 677 / 1056 | 7.4 / 17.6 (-57.9%) | 4.5 / 0.0 | 34.1 / 103.0 | 0.173 / 0.232 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V2 | 447 / 0 / 792 | 4440.2 / 11877.8; 21491.9 / 27163.8 | 792.2 / 976.5; 1050.6 / 1676.4 | 677 / 1056 | 7.5 / 17.6 (-57.7%) | 4.5 / 0.0 | 34.6 / 102.8 | 0.173 / 0.233 | 47 / 47 / 0 / 0 | 502.7 / 502.7 |
| 21.0 | fixed events | V3 | 445 / 0 / 794 | 4401.3 / 11877.8; 23650.6 / 27163.8 | 787.9 / 976.5; 987.9 / 1676.4 | 677 / 1056 | 7.4 / 17.6 (-57.9%) | 4.5 / 0.0 | 34.1 / 103.0 | 0.173 / 0.232 | 2 / 2 / 0 / 0 | 1143.5 / 1307.8 |
| 21.0 | closed-loop | V3 | 444 / 0 / 795 | 4073.2 / 11877.8; 22861.5 / 27163.8 | 790.9 / 976.5; 999.7 / 1676.4 | 677 / 1056 | 7.4 / 17.6 (-58.0%) | 4.4 / 0.0 | 34.0 / 104.3 | 0.174 / 0.233 | 41 / 41 / 0 / 0 | 486.6 / 837.7 |
| 9.0 | fixed events | V0 | 315 / 0 / 234 | 404.2 / 6835.8; 692.7 / 25031.9 | 605.6 / 542.3; 728.9 / 939.8 | 321 / 400 | 5.2 / 6.7 (-21.3%) | 3.2 / 0.0 | 0.7 / 1.5 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 750.8 / 1139.4 |
| 9.0 | closed-loop | V0 | 316 / 0 / 233 | 433.0 / 6835.8; 701.6 / 25031.9 | 608.7 / 542.3; 731.8 / 939.8 | 321 / 400 | 5.3 / 6.7 (-21.0%) | 3.2 / 0.0 | 0.8 / 1.5 | 0.000 / 0.000 | 74 / 73 / 1 / 0 | 612.5 / 839.0 |
| 9.0 | fixed events | V1 | 331 / 0 / 218 | 413.3 / 6835.8; 690.4 / 25031.9 | 585.9 / 542.3; 734.2 / 939.8 | 321 / 400 | 5.5 / 6.7 (-17.3%) | 3.3 / 0.0 | 0.8 / 1.5 | 0.000 / 0.000 | 2 / 2 / 0 / 0 | 750.8 / 1139.4 |
| 9.0 | closed-loop | V1 | 325 / 0 / 224 | 409.5 / 6835.8; 668.7 / 25031.9 | 583.4 / 542.3; 705.0 / 939.8 | 321 / 400 | 5.4 / 6.7 (-18.8%) | 3.2 / 0.0 | 0.7 / 1.5 | 0.000 / 0.000 | 83 / 82 / 1 / 0 | 621.8 / 1066.3 |
| 9.0 | fixed events | V2 | 331 / 0 / 218 | 413.3 / 6835.8; 690.4 / 25031.9 | 585.9 / 542.3; 734.2 / 939.8 | 321 / 400 | 5.5 / 6.7 (-17.3%) | 3.3 / 0.0 | 0.8 / 1.5 | 0.092 / 0.153 | 2 / 2 / 0 / 0 | 750.8 / 1139.4 |
| 9.0 | closed-loop | V2 | 329 / 0 / 220 | 415.7 / 6835.8; 660.4 / 25031.9 | 585.9 / 542.3; 710.1 / 939.8 | 321 / 400 | 5.5 / 6.7 (-17.8%) | 3.3 / 0.0 | 0.8 / 1.5 | 0.092 / 0.151 | 88 / 87 / 1 / 0 | 505.3 / 576.5 |
| 9.0 | fixed events | V3 | 331 / 0 / 218 | 413.3 / 6835.8; 690.4 / 25031.9 | 585.9 / 542.3; 734.2 / 939.8 | 321 / 400 | 5.5 / 6.7 (-17.3%) | 3.3 / 0.0 | 0.8 / 1.5 | 0.092 / 0.153 | 2 / 2 / 0 / 0 | 750.8 / 1139.4 |
| 9.0 | closed-loop | V3 | 329 / 0 / 220 | 403.0 / 6835.8; 643.9 / 25031.9 | 583.4 / 542.3; 723.1 / 939.8 | 321 / 400 | 5.5 / 6.7 (-17.8%) | 3.3 / 0.0 | 0.7 / 1.5 | 0.092 / 0.151 | 85 / 85 / 0 / 0 | 500.3 / 949.2 |

## Fixed migration replay coverage

The event replay consumes logged event time, source, destination, blocks, duration, and terminal status. Event start time is approximated by watcher observation time minus the recorded duration. Requests linked to a logged migration are routed to the first logged source only in this fixed replay. The closed-loop run receives no logged route map or event schedule.

| QPS | Logged migration rows | Linked replay requests | Logged status counts | V2 started / nonstarted / still queued | Median start delay ms |
|---:|---:|---:|:---|---:|---:|
| 12.0 | 2 | 2 | {'MigrationStatus.FINISHED': 2} | 2 / 0 / 0 | -29.5 |
| 15.0 | 4 | 3 | {'MigrationStatus.ABORTED_DST': 1, 'MigrationStatus.FINISHED': 3} | 3 / 0 / 0 | -12.5 |
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
- Queue and KV comparisons use per-instance units. Observed data are polling snapshots; simulated values are sampled on the event-driven replay clock.
- Fixed replay and greedy closed-loop results test system behavior under different controls. Closed-loop greedy results do not establish PPO policy performance.

- The configured 22,084 KV blocks per instance are supported by 44,111 log-derived free-block/usage samples (p05/median/p95 22,084/22,084/22,084); this remains a log inference, not an independent hardware capacity measurement.
The calibrated service models remain P1/D3 with the previously selected migration model. The report preserves all QPS/version/mode metrics and error calculations in `system_validation.json`; the combined final calibration is `mig_ppo_calibration_system_validation.yaml`.
