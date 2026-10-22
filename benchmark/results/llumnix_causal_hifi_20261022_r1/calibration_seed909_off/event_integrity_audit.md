# Causal event integrity audit

Source streams: 5; events: 4221593; JSON parse errors: 0.

## Process streams

| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |
|:---|---:|---:|---:|---:|---:|
| 5d430c52a833:371250 | 1908242 | 0 | 0 | 0 | 0.000000 |
| 5d430c52a833:374507 | 561542 | 0 | 0 | 835 | 36.484774 |
| 5d430c52a833:374508 | 598654 | 0 | 0 | 1033 | 52.352173 |
| 5d430c52a833:374509 | 557202 | 0 | 0 | 784 | 24.731179 |
| 5d430c52a833:374690 | 595953 | 0 | 0 | 1072 | 35.394283 |

## Joins

Server requests received/terminal: 7992/7992; missing terminals: 0.
Scheduled/executor batches: 42111/42111; unmatched schedules: 0; failed batches: 0.

## Event counts

| Event | Count |
|:---|---:|
| batch_scheduled | 42111 |
| clock_sync | 5 |
| engine_iteration_completed | 42111 |
| engine_request_output | 1953812 |
| executor_iteration_started | 42111 |
| gpu_event_capture_initialized | 4 |
| gpu_execution_completed | 42111 |
| kv_request_blocks_changed | 134750 |
| llm_engine_step_completed | 42111 |
| request_abort_requested | 6114 |
| request_engine_admitted | 7992 |
| request_internal_timeline | 7740 |
| request_output_at_api | 1872572 |
| request_server_completed | 5831 |
| request_server_received | 7992 |
| request_server_terminal | 7992 |
| scheduler_idle | 6234 |

Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.
