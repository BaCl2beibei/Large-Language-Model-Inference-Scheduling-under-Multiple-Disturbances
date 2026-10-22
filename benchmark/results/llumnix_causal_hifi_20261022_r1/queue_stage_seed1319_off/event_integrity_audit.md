# Causal event integrity audit

Source streams: 5; events: 213982; JSON parse errors: 0.

## Process streams

| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |
|:---|---:|---:|---:|---:|---:|
| 5d430c52a833:414126 | 79384 | 0 | 0 | 0 | 0.000000 |
| 5d430c52a833:417454 | 28280 | 0 | 0 | 55 | 13.308515 |
| 5d430c52a833:417516 | 28110 | 0 | 0 | 47 | 18.594060 |
| 5d430c52a833:417579 | 49813 | 0 | 0 | 161 | 17.085900 |
| 5d430c52a833:417640 | 28395 | 0 | 0 | 65 | 30.768737 |

## Joins

Server requests received/terminal: 1334/1334; missing terminals: 0.
Scheduled/executor batches: 2524/2524; unmatched schedules: 0; failed batches: 0.

## Event counts

| Event | Count |
|:---|---:|
| batch_scheduled | 2524 |
| clock_sync | 5 |
| engine_iteration_completed | 2524 |
| engine_request_output | 106327 |
| executor_iteration_started | 2524 |
| gpu_event_capture_initialized | 4 |
| gpu_execution_completed | 2524 |
| kv_request_blocks_changed | 9295 |
| llm_engine_step_completed | 2524 |
| output_queue_dequeue | 2054 |
| output_queue_enqueue | 2054 |
| request_abort_requested | 1334 |
| request_engine_admitted | 1334 |
| request_internal_timeline | 1207 |
| request_output_at_api | 68859 |
| request_server_completed | 1207 |
| request_server_received | 1334 |
| request_server_terminal | 1334 |
| scheduler_idle | 5014 |

Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.
