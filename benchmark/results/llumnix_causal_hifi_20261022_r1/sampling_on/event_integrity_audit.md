# Causal event integrity audit

Source streams: 5; events: 4518301; JSON parse errors: 0.

## Process streams

| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |
|:---|---:|---:|---:|---:|---:|
| 5d430c52a833:343336 | 956119 | 0 | 0 | 0 | 0.000000 |
| 5d430c52a833:346599 | 614001 | 0 | 0 | 492 | 37.324689 |
| 5d430c52a833:346661 | 1158568 | 0 | 0 | 1944 | 37.030140 |
| 5d430c52a833:346723 | 626214 | 0 | 0 | 530 | 30.153632 |
| 5d430c52a833:346784 | 1163399 | 0 | 0 | 1733 | 885.946321 |

## Joins

Server requests received/terminal: 15735/15735; missing terminals: 0.
Scheduled/executor batches: 59963/59963; unmatched schedules: 0; failed batches: 0.

## Event counts

| Event | Count |
|:---|---:|
| batch_scheduled | 59963 |
| clock_sync | 5 |
| engine_iteration_completed | 59963 |
| engine_request_output | 3019026 |
| executor_iteration_started | 59963 |
| gpu_event_capture_initialized | 4 |
| gpu_execution_completed | 59963 |
| kv_request_blocks_changed | 214385 |
| llm_engine_step_completed | 59963 |
| migration_attempt_completed | 47 |
| migration_attempt_started | 47 |
| migration_destination_committed | 42 |
| migration_preallocation_completed | 77 |
| migration_request_pause_started | 45 |
| migration_request_resumed | 45 |
| migration_stage_completed | 76 |
| migration_stage_started | 81 |
| migration_target_commit_started | 42 |
| migration_transfer_completed | 77 |
| migration_transfer_started | 77 |
| request_abort_requested | 15363 |
| request_engine_admitted | 15735 |
| request_internal_timeline | 8195 |
| request_output_at_api | 893285 |
| request_server_completed | 7805 |
| request_server_received | 15735 |
| request_server_terminal | 15735 |
| scheduler_idle | 12557 |

Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.
