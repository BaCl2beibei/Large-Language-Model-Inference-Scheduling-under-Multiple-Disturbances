# Seed 1319 queue-stage paired diagnostic

## Runs

- OFF: [`queue_stage_seed1319_off_pg60_rpc270_concurrent`](queue_stage_seed1319_off_pg60_rpc270_concurrent/)
- ON: [`queue_stage_seed1319_on_pg60_rpc270`](queue_stage_seed1319_on_pg60_rpc270/)
- Both used QPS 18, seed 1319, 1,076 measurement arrivals, the same four-GPU model and an identical trace SHA-256: `45d242898c037953b35a9d391ef57b0f80103330378ea5eda1d71a564e1a6ee1`.
- Both passed the four-worker startup gate on GPUs 0–3. The source manifests differ only in `llumnix/entrypoints/setup.py` and `benchmark/run_ppo_batch_reuse.py`, which contain the parallel readiness/startup-timeout diagnostic changes. The inference and event instrumentation paths were unchanged.

OFF command:

```bash
LLUMNIX_WAIT_PLACEMENT_GROUP_TIMEOUT=60 \
LLUMNIX_INIT_INSTANCES_TIMEOUT=270 \
LLUMNIX_INSTANCE_READY_TIMEOUT=1200 \
LLUMNIX_SERVER_STARTUP_TIMEOUT=1200 \
.venv/bin/python benchmark/run_ppo_batch_reuse.py \
  --qps 18 --calibration-seeds 606 909 --validation-seed 1319 \
  --warmup-sec 15 --measure-sec 60 --drain-sec 30 --max-num-seqs 128 \
  --output-dir benchmark/results/llumnix_causal_hifi_20261022_r1/queue_stage_seed1319_off_pg60_rpc270_concurrent \
  --mode off --validation-only
```

## Event integrity

| Mode | Source streams | Events | JSON errors | Missing sequence values | Missing request terminals | Unmatched batches |
|:--|--:|--:|--:|--:|--:|--:|
| OFF | 5 | 270,536 | 0 | 0 | 0 | 0 |
| ON | 5 | 381,755 | 0 | 0 | 0 | 0 |

The OFF audit is in [`event_integrity_audit.md`](queue_stage_seed1319_off_pg60_rpc270_concurrent/event_integrity_audit.md); the ON audit is in [`event_integrity_audit.md`](queue_stage_seed1319_on_pg60_rpc270/event_integrity_audit.md). All measurement requests were joined to server terminal events in both runs.

## Request lifecycle and declared drain horizon

Each run sent and the server received all 1,076 measurement requests, with no client timeout. OFF returned 807 HTTP 200 responses and 269 HTTP 499 responses; ON returned 598 HTTP 200 and 478 HTTP 499.

The terminal-event timestamps show that the 30-second nominal drain horizon does not include all server completions. In OFF, all 807 `server_completed` events occurred after the 90-second measurement-plus-drain horizon, during the benchmark abort barrier. In ON, 2 completed during measurement, 82 during the nominal drain, and 514 after that horizon. The remaining 269 OFF and 478 ON requests were aborted without final output. This explains why client HTTP 200 counts differ from server completion counts inside the declared horizon; use the latter for windowed throughput. The event audit found no lost terminal events.

This is a measurement-window limitation to carry into system-level scoring. It does not invalidate the event stream, but the client’s abort-barrier time must remain separate from the fixed measurement-plus-drain interval.

## Output and queue stages

Times are milliseconds. The percentile samples are matched event pairs from measurement requests through the 30-second nominal drain horizon.

| Stage | OFF n | OFF p50 | OFF p95 | ON n | ON p50 | ON p95 |
|:--|--:|--:|--:|--:|--:|--:|
| Engine output → forwarder submit | 2,660 | 1.85 | 14.12 | 2,088 | 6.17 | 29.01 |
| Submit → forwarder loop start | 2,660 | 8.57 | 27.25 | 2,088 | 13.74 | 56.96 |
| Loop start → queue send | 2,660 | 0.83 | 18.40 | 2,088 | 2.16 | 29.89 |
| Queue send → server enqueue | 833 | 1,190.35 | 6,460.72 | 2,059 | 1,599.29 | 5,053.46 |
| Server queue residence | 833 | 7.67 | 227.60 | 2,059 | 7.44 | 35.19 |
| Dequeue → API callback | 833 | 7.62 | 132.74 | 2,059 | 32.45 | 104.34 |

The internal forwarder steps have p95 values below 57 ms, while queue-send-to-enqueue is about 5–6.5 seconds at p95. Queue residence is much shorter at the median, although its OFF p95 is higher than ON. The large send-to-enqueue interval is not pure queue residence; it spans producer send through server-side enqueue and needs that boundary semantics kept explicit in subsequent attribution.

Descriptive request metrics were OFF TTFT mean/p95 3,043/10,874 ms (607 samples) and TPOT mean/p95 235/936 ms (24,584 token samples); ON was 2,699/6,127 ms (514 samples) and 381/1,272 ms (57,359 samples). OFF had no migrations; ON logged 2 attempts, 1 successful. These are single sequential runs with different migration settings and different censoring, so they are diagnostic comparisons, not an accuracy-gate result.

## Outcome

The repeated OFF initialization passed with the parallel readiness fix, and the complete paired event streams confirm queue-stage timestamps are joinable. The evidence narrows the delay to before enqueue rather than time spent resident in the output queue. It does not establish that the simulator meets the independent accuracy thresholds. The earlier seed 1213 independent accuracy gate remains failed ([`system_validation.json`](validation_seed1213/system_validation.json)); formal PPO was not started.

The OFF runner compressed five source streams from about 341 MB to 18 MB after verifying each restored stream’s byte count and SHA-256. The Ray cluster shut down and GPU memory returned to approximately 1 MiB per GPU.
