# Causal event integrity audit

Source streams: 5; events: 381755; JSON parse errors: 0.

## Process streams

| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |
|:---|---:|---:|---:|---:|---:|
| 5d430c52a833:429407 | 172241 | 0 | 0 | 0 | 0.000000 |
| 5d430c52a833:432806 | 81047 | 0 | 0 | 314 | 35.726453 |
| 5d430c52a833:432807 | 84567 | 0 | 0 | 189 | 16.678456 |
| 5d430c52a833:432930 | 21896 | 0 | 0 | 27 | 9.221060 |
| 5d430c52a833:432993 | 22004 | 0 | 0 | 47 | 14.166283 |

## Joins

Server requests received/terminal: 1334/1334; missing terminals: 0.
Scheduled/executor batches: 2456/2456; unmatched schedules: 0; failed batches: 0.

## Event counts

| Event | Count |
|:---|---:|
| batch_scheduled | 2456 |
| clock_sync | 5 |
| engine_iteration_completed | 2456 |
| engine_request_output | 160970 |
| executor_iteration_started | 2456 |
| gpu_event_capture_initialized | 4 |
| gpu_execution_completed | 2456 |
| kv_request_blocks_changed | 11502 |
| llm_engine_step_completed | 2456 |
| migration_attempt_completed | 3 |
| migration_attempt_started | 3 |
| migration_destination_committed | 2 |
| migration_preallocation_completed | 3 |
| migration_request_pause_started | 3 |
| migration_request_resumed | 3 |
| migration_stage_completed | 3 |
| migration_stage_started | 4 |
| migration_target_commit_started | 2 |
| migration_transfer_completed | 3 |
| migration_transfer_started | 3 |
| output_forwarder_loop_start | 3033 |
| output_forwarder_queue_client_send | 3033 |
| output_forwarder_submit | 3033 |
| output_queue_dequeue | 3033 |
| output_queue_enqueue | 3033 |
| request_abort_requested | 1214 |
| request_engine_admitted | 1334 |
| request_internal_timeline | 978 |
| request_output_at_api | 160458 |
| request_server_completed | 856 |
| request_server_received | 1334 |
| request_server_terminal | 1334 |
| scheduler_idle | 14289 |

Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.
