# Accuracy comparison

## Same held-out seed 1213: Round 2 versus causal rebuild

The table compares Round 2 V3 and the causal rebuild on the same paired OFF/ON
seed 1213 traces and metric definitions. MAPE values are the macro percentage
error across the 14 mode/QPS rows. Queue values are mean absolute error; KV
values are mean absolute percentage-point error.

| Metric | Round 2 V3 | Causal rebuild | Change |
|:---|---:|---:|---:|
| Request throughput MAPE | 2,985.1% | 2,637.2% | 347.8 percentage points lower; still fails badly. The macro excludes zero denominators, while the per-run gate scores nonzero prediction at zero throughput as infinity. |
| Output tokens/s MAPE | 126.2% | 265.1% | 138.8 points worse. |
| Mean TTFT MAPE | 157.8% | 130.7% | 27.2 points better. |
| P95 TTFT MAPE | 163.6% | 232.5% | 69.0 points worse. |
| Mean TPOT MAPE | 224.8% | 34.2% | 190.6 points better. |
| P95 TPOT MAPE | 128.8% | 51.2% | 77.6 points better. |
| Mean queue MAE | 7.141 | 5.355 | 1.786 requests lower. |
| P95 queue MAE | 25.500 | 24.283 | 1.217 requests lower. |
| Mean KV error | 3.52 pp | 1.88 pp | 1.64 pp lower. |
| P95 KV error | 5.48 pp | 2.58 pp | 2.90 pp lower. |

The local fixed-batch replay also improves batch execution-time prediction over
Round 2, but that does not translate into a passing system-level gate.

## Independent accuracy gate

**The gate did not pass.** Throughput, both TTFT metrics, both TPOT metrics,
and both queue metrics fail at least one per-run threshold. KV mean/P95 and
request-lifecycle/collection integrity pass.

| Gate | Worst run | Observed | Predicted | Error / limit |
|:---|:---|---:|---:|---:|
| Throughput MAPE ≤20% | OFF, QPS 15 | 0.000 req/s | 2.767 req/s | Infinite error |
| Mean TTFT MAPE ≤25% | OFF, QPS 24 | 1,176.7 ms | 5,989.3 ms | 4.09 / 0.25 |
| P95 TTFT MAPE ≤30% | OFF, QPS 24 | 2,679.0 ms | 26,217 ms | 8.79 / 0.30 |
| Mean TPOT MAPE ≤20% | OFF, QPS 3 | 52.78 ms | 100.22 ms | 0.899 / 0.20 |
| P95 TPOT MAPE ≤25% | OFF, QPS 3 | 74.16 ms | 205.14 ms | 1.766 / 0.25 |
| Mean queue MAE limit | ON, QPS 18 | 29.32 | 8.01 | 21.31 / 5.86 |
| P95 queue MAE limit | ON, QPS 18 | 122.25 | 30.76 | 91.49 / 24.45 |
| Mean KV error ≤5 pp | OFF, QPS 18 | — | — | Pass |
| P95 KV error ≤10 pp | OFF, QPS 18 | — | — | Pass |

## Error attribution added after the frozen-model validation

The diagnostic overlay reports first-token delivery coverage and engine-to-API
callback delay for all 14 held-out rows. For OFF/QPS18, only 98/1,095 sent
measurement requests produced an API-visible first token within the 60 s window;
997 are right-censored there. First-output delay is 6.55 s median and 35.20 s
P95. ON/QPS18 covers 547/1,095 requests, with 19 ms median and 3.16 s P95.
These diagnostics were not used to fit or select the model.

The historical Round 2 report lists throughput MAPE 38.13%, mean/P95 TTFT
72.83%/56.00%, mean/P95 TPOT 21.48%/25.60%, queue mean/P95 MAE 7.199/31.543,
and mean KV error 1.77 pp. That baseline used its own validation group; the
seed-1213 table above is the apples-to-apples comparison on the new trace.

Formal PPO migration-scheduling training remains blocked by the accuracy gate.
