# Matched Migration OFF/ON pilot: QPS 15, seed 505

This is a targeted pilot on a new seed. Both modes used the same saved 942-request trace, four eager vLLM instances, max model length 2112, 128 sequences per instance, and the calibrated nine-profile workload. The two YAML configs differ only in `ENABLE_ROUTINE_MIGRATION`.

Trace SHA-256: `09a0dc7c657eab19985e65847a773d6fdc9a85416a2eb8b189432c638e284cc6`. Arrival window: 60 seconds. Shared result horizon: 90 seconds.

## Paired results

| Measure | Migration OFF | Migration ON |
|---|---:|---:|
| Requests sent | 942 | 942 |
| Server final tokens by 60s | 407 | 223 |
| Server final tokens by 90s | 650 | 226 |
| Requests unfinished at server horizon 90s | 292 | 716 |
| HTTP 200 bodies received | 767 | 588 |
| HTTP 499 abort-censored | 175 | 354 |
| Client bodies received by 90s | 0 | 7 |
| Server TTFT mean / P95 by 90s (ms) | 11550.277 / 28068.080 | 8138.910 / 28597.965 |
| Server TPOT mean / P95 by 90s (ms; timeline matched) | 935.123 / 1269.560 | 437.039 / 846.644 |
| Queue mean / P95 per instance snapshot | 2.568 / 18.000 | 0.876 / 6.000 |
| Queue mean / P95 of five-second instance means | 2.937 / 11.370 | 25.008 / 132.000 |
| KV usage mean / P95 from GPU cache metric (%) | 12.266 / 24.218 | 12.130 / 26.232 |
| KV usage mean / P95 of five-second instance means (%) | 12.256 / 21.822 | 16.377 / 26.982 |
| Measurement migrations: attempts, successes, failures/aborts | 0, 0, 0 | 6, 6, 0 |

## Timing and censoring

For OFF, 767 complete HTTP 200 responses arrived after 90 seconds. Their server final-token timestamps show 650 had completed by 90 seconds, with 117 finishing afterward. The remaining 175 requests were aborted at the drain barrier. ON returned 588 HTTP 200 responses; server token timestamps place 226 by 90 seconds and 362 after 90 seconds, with 354 abort-censored 499s.

The mean final-token-to-client-body delay for time-lined HTTP 200 requests was 28184.304 ms OFF and 16342.367 ms ON. These client completion timestamps cannot stand in for server token completion timestamps.

TTFT and TPOT summaries by 90 seconds use only successful HTTP 200 responses with server final-token timestamps within the horizon. HTTP 499 requests have no returned token timeline. TPOT additionally requires the timeline count to equal the response output-token count. The actual generated output count is recorded separately from each request’s output-token budget.

## Queue, KV, engine, and migration observations

The server startup logs report 22,084 GPU KV blocks per instance in both modes; each instance CSV also records available blocks. This directly confirms the runtime-reported capacity for this deployment, though it remains a vLLM-reported rather than independent hardware measurement.

OFF measurement engine iteration samples: prefill n=402, decode n=1457. ON: prefill n=281, decode n=1468. These are descriptive executor iteration wall times, not full Engine Step or request latency labels, and were not used to refit the simulator.

ON had 6 terminal measurement migration records: 6 successful and 0 unsuccessful. Logged cost mean/P95 was 582.468/791.392 ms. No migration Pause Time ground truth was collected.

## Limits

This is one paired seed and one server startup per mode, so it identifies a useful causal contrast but does not quantify run-to-run variance. It adds a Migration OFF service baseline and a matched ON comparison; it does not satisfy the five-QPS system admission gate or validate migration pause time. The simulator remains ineligible for formal PPO training based on the separate five-QPS system validation.

The JSON retains all per-request server timeline conversions, queue/KV samples, engine iteration summaries, and migration events. Clock alignment uses the shared Linux `CLOCK_MONOTONIC` implementation: API token timestamps are `time.perf_counter()`, while the benchmark records `time.monotonic()` offsets and wall-clock send/response times.

Queue and KV fields are shown both per recorded instance snapshot and as means within five-second bins. The latter gives each time bin equal weight; the raw snapshot distribution is included for tails and per-instance inspection. These pilot summaries use five-second bins and should not be substituted for the 50 ms system-validation queue gate.
