# Causal event integrity audit

Source streams: 5; events: 2645651; JSON parse errors: 0.

## Process streams

| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |
|:---|---:|---:|---:|---:|---:|
| 5d430c52a833:379261 | 543392 | 0 | 0 | 0 | 0.000000 |
| 5d430c52a833:382526 | 338291 | 0 | 0 | 475 | 27.246899 |
| 5d430c52a833:382587 | 593383 | 0 | 0 | 1044 | 506.234977 |
| 5d430c52a833:382588 | 586250 | 0 | 0 | 927 | 33.857667 |
| 5d430c52a833:382709 | 584335 | 0 | 0 | 928 | 35.722563 |

## Joins

Server requests received/terminal: 7742/7742; missing terminals: 0.
Scheduled/executor batches: 38654/38654; unmatched schedules: 0; failed batches: 0.

## Event counts

| Event | Count |
|:---|---:|
| batch_scheduled | 38654 |
| clock_sync | 5 |
| engine_iteration_completed | 38654 |
| engine_request_output | 1768768 |
| executor_iteration_started | 38654 |
| gpu_event_capture_initialized | 4 |
| gpu_execution_completed | 38654 |
| kv_request_blocks_changed | 122786 |
| llm_engine_step_completed | 38654 |
| request_abort_requested | 7372 |
| request_engine_admitted | 7742 |
| request_internal_timeline | 2902 |
| request_output_at_api | 515117 |
| request_server_completed | 2516 |
| request_server_received | 7742 |
| request_server_terminal | 7742 |
| scheduler_idle | 9685 |

Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.
