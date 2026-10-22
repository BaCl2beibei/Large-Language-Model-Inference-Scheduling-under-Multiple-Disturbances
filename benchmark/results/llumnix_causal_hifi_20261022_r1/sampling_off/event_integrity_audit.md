# Causal event integrity audit

Source streams: 5; events: 5009200; JSON parse errors: 0.

## Process streams

| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |
|:---|---:|---:|---:|---:|---:|
| 5d430c52a833:320237 | 1181951 | 0 | 0 | 0 | 0.000000 |
| 5d430c52a833:323536 | 1288900 | 0 | 0 | 2598 | 1683.313474 |
| 5d430c52a833:323597 | 1279757 | 0 | 0 | 2456 | 27.158870 |
| 5d430c52a833:323658 | 631319 | 0 | 0 | 580 | 38.117517 |
| 5d430c52a833:323659 | 627273 | 0 | 0 | 612 | 6195.776650 |

## Joins

Server requests received/terminal: 15735/15735; missing terminals: 0.
Scheduled/executor batches: 66358/66358; unmatched schedules: 0; failed batches: 0.

## Event counts

| Event | Count |
|:---|---:|
| batch_scheduled | 66358 |
| clock_sync | 5 |
| engine_iteration_completed | 66358 |
| engine_request_output | 3236703 |
| executor_iteration_started | 66358 |
| gpu_event_capture_initialized | 4 |
| gpu_execution_completed | 66358 |
| kv_request_blocks_changed | 228587 |
| llm_engine_step_completed | 66358 |
| request_abort_requested | 15297 |
| request_engine_admitted | 15735 |
| request_internal_timeline | 8752 |
| request_output_at_api | 1118145 |
| request_server_completed | 8286 |
| request_server_received | 15735 |
| request_server_terminal | 15735 |
| scheduler_idle | 14426 |

Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.
