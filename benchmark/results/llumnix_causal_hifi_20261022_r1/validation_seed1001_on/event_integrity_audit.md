# Causal event integrity audit

Source streams: 5; events: 3323393; JSON parse errors: 0.

## Process streams

| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |
|:---|---:|---:|---:|---:|---:|
| 5d430c52a833:386324 | 1541719 | 0 | 0 | 0 | 0.000000 |
| 5d430c52a833:389641 | 567910 | 0 | 0 | 1254 | 460.243718 |
| 5d430c52a833:389642 | 344790 | 0 | 0 | 324 | 16.521484 |
| 5d430c52a833:389643 | 335602 | 0 | 0 | 404 | 22.216679 |
| 5d430c52a833:389824 | 533372 | 0 | 0 | 1133 | 603.319873 |

## Joins

Server requests received/terminal: 7742/7742; missing terminals: 0.
Scheduled/executor batches: 30298/30298; unmatched schedules: 0; failed batches: 0.

## Event counts

| Event | Count |
|:---|---:|
| batch_scheduled | 30298 |
| clock_sync | 5 |
| engine_iteration_completed | 30298 |
| engine_request_output | 1508309 |
| executor_iteration_started | 30298 |
| gpu_event_capture_initialized | 4 |
| gpu_execution_completed | 30298 |
| kv_request_blocks_changed | 106015 |
| llm_engine_step_completed | 30298 |
| migration_attempt_completed | 18 |
| migration_attempt_started | 18 |
| migration_destination_committed | 18 |
| migration_preallocation_completed | 33 |
| migration_request_pause_started | 18 |
| migration_request_resumed | 18 |
| migration_stage_completed | 33 |
| migration_stage_started | 33 |
| migration_target_commit_started | 18 |
| migration_transfer_completed | 33 |
| migration_transfer_started | 33 |
| request_abort_requested | 6540 |
| request_engine_admitted | 7742 |
| request_internal_timeline | 6724 |
| request_output_at_api | 1507480 |
| request_server_completed | 5490 |
| request_server_received | 7742 |
| request_server_terminal | 7742 |
| scheduler_idle | 7837 |

Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.
