# Causal event integrity audit

Source streams: 5; events: 2846552; JSON parse errors: 0.

## Process streams

| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |
|:---|---:|---:|---:|---:|---:|
| 5d430c52a833:396064 | 639162 | 0 | 0 | 0 | 0.000000 |
| 5d430c52a833:399340 | 617145 | 0 | 0 | 1091 | 37.461413 |
| 5d430c52a833:399403 | 622138 | 0 | 0 | 977 | 811.852503 |
| 5d430c52a833:399464 | 348799 | 0 | 0 | 431 | 18.694005 |
| 5d430c52a833:399526 | 619308 | 0 | 0 | 1066 | 33.622110 |

## Joins

Server requests received/terminal: 7542/7542; missing terminals: 0.
Scheduled/executor batches: 41699/41699; unmatched schedules: 0; failed batches: 0.

## Event counts

| Event | Count |
|:---|---:|
| batch_scheduled | 41699 |
| clock_sync | 5 |
| engine_iteration_completed | 41699 |
| engine_request_output | 1854625 |
| executor_iteration_started | 41699 |
| gpu_event_capture_initialized | 4 |
| gpu_execution_completed | 41699 |
| kv_request_blocks_changed | 127945 |
| llm_engine_step_completed | 41699 |
| request_abort_requested | 6976 |
| request_engine_admitted | 7542 |
| request_internal_timeline | 2880 |
| request_output_at_api | 611915 |
| request_server_completed | 2306 |
| request_server_received | 7542 |
| request_server_terminal | 7542 |
| scheduler_idle | 8775 |

Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.
