# Causal event integrity audit

Source streams: 5; events: 3007938; JSON parse errors: 0.

## Process streams

| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |
|:---|---:|---:|---:|---:|---:|
| 5d430c52a833:403370 | 1379358 | 0 | 0 | 0 | 0.000000 |
| 5d430c52a833:406631 | 512966 | 0 | 0 | 1040 | 33.499831 |
| 5d430c52a833:406633 | 307140 | 0 | 0 | 479 | 77.856862 |
| 5d430c52a833:406754 | 508670 | 0 | 0 | 756 | 29.236990 |
| 5d430c52a833:406815 | 299804 | 0 | 0 | 416 | 23.481790 |

## Joins

Server requests received/terminal: 7542/7542; missing terminals: 0.
Scheduled/executor batches: 31175/31175; unmatched schedules: 0; failed batches: 0.

## Event counts

| Event | Count |
|:---|---:|
| batch_scheduled | 31175 |
| clock_sync | 5 |
| engine_iteration_completed | 31175 |
| engine_request_output | 1362260 |
| executor_iteration_started | 31175 |
| gpu_event_capture_initialized | 4 |
| gpu_execution_completed | 31175 |
| kv_request_blocks_changed | 96512 |
| llm_engine_step_completed | 31175 |
| migration_attempt_completed | 26 |
| migration_attempt_started | 26 |
| migration_destination_committed | 26 |
| migration_preallocation_completed | 46 |
| migration_request_pause_started | 26 |
| migration_request_resumed | 26 |
| migration_stage_completed | 46 |
| migration_stage_started | 46 |
| migration_target_commit_started | 26 |
| migration_transfer_completed | 46 |
| migration_transfer_started | 46 |
| request_abort_requested | 6486 |
| request_engine_admitted | 7542 |
| request_internal_timeline | 6360 |
| request_output_at_api | 1346140 |
| request_server_completed | 5287 |
| request_server_received | 7542 |
| request_server_terminal | 7542 |
| scheduler_idle | 5997 |

Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.
