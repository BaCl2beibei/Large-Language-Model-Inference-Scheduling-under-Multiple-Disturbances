# Causal event integrity audit

Source streams: 5; events: 270536; JSON parse errors: 0.

## Process streams

| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |
|:---|---:|---:|---:|---:|---:|
| 5d430c52a833:442876 | 43633 | 0 | 0 | 0 | 0.000000 |
| 5d430c52a833:446798 | 54275 | 0 | 0 | 237 | 34.072086 |
| 5d430c52a833:446859 | 58957 | 0 | 0 | 307 | 13.960994 |
| 5d430c52a833:446927 | 54769 | 0 | 0 | 307 | 23.411319 |
| 5d430c52a833:447008 | 58902 | 0 | 0 | 148 | 25.786145 |

## Joins

Server requests received/terminal: 1334/1334; missing terminals: 0.
Scheduled/executor batches: 5165/5165; unmatched schedules: 0; failed batches: 0.

## Event counts

| Event | Count |
|:---|---:|
| batch_scheduled | 5165 |
| clock_sync | 5 |
| engine_iteration_completed | 5165 |
| engine_request_output | 168377 |
| executor_iteration_started | 5165 |
| gpu_event_capture_initialized | 4 |
| gpu_execution_completed | 5165 |
| kv_request_blocks_changed | 13436 |
| llm_engine_step_completed | 5165 |
| output_forwarder_loop_start | 3354 |
| output_forwarder_queue_client_send | 3354 |
| output_forwarder_submit | 3354 |
| output_queue_dequeue | 1162 |
| output_queue_enqueue | 1162 |
| request_abort_requested | 1331 |
| request_engine_admitted | 1334 |
| request_internal_timeline | 984 |
| request_output_at_api | 35352 |
| request_server_completed | 973 |
| request_server_received | 1334 |
| request_server_terminal | 1334 |
| scheduler_idle | 7861 |

Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.
