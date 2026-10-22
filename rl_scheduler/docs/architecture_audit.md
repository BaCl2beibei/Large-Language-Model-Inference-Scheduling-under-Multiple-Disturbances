# Llumnix architecture audit for Mig-PPO

This audit records the existing v0 path before adding the independent Mig-PPO
research package. It deliberately distinguishes fields already present in
Llumnix from fields added for Mig-PPO observation.

## State source, units, and freshness

`InstanceInfo` is assembled by the backend scheduler and returned by the
Llumlet. For the vLLM v0 backend, `SchedulerLlumnix._get_instance_info()` reads
the running and waiting queues and block manager after each engine scheduling
step. It reports running/waiting/killed request counts, total/used/free GPU KV
blocks, cache usage, waiting-head block estimate and wall-clock wait, aggregate
waiting-block estimate, decode batch size, running sequence lengths, batched
tokens, and inference type. `LLMEngineLlumnix._process_request_outputs()` stamps
`timestamp` with `time.time()` (Unix wall-clock seconds), increments `step_id`,
and records profiling data `(inference_type, sequence_count, total_sequence_len,
last_inference_latency)`. The last latency is milliseconds; this phase also
copies it into the existing optional `InstanceInfo.last_inference_latency`
field so the feature is directly readable. The Manager polls
Llumlet info at `MANAGER.POLLING_INTERVAL` (default 0.05 seconds); each backend
state itself advances on vLLM engine steps, so a polled record may repeat the
same step/timestamp. The observation builder therefore checks timestamps rather
than assuming every poll contains new measurements.

`num_batched_tokens` has stage-dependent meaning in vLLM v0: for a scheduled
prefill step it is the sum of scheduled request lengths, while for a decode
step it is the count of running sequences. Fields are integers or floats
except IDs/enums and the `running_seq_lens` list.
`num_blocks_first_waiting_request` and `num_blocks_all_waiting_requests` in
vLLM v0 are prompt-token/block-size estimates and can be fractional; they are
not a precise reservation count. `num_free_gpu_blocks` and the actual SR
candidate KV block count are integer block counts. `gpu_cache_usage` is a
fraction in [0, 1] when total blocks are nonzero. For vLLM v1, the current code
does not expose an accurate queue-head wait time or preemption count, and its
candidate SR data is unavailable; these are missing values, not zero.

`kv_blocks_ratio` is computed as `(used GPU blocks + estimated blocks needed
by all waiting requests) / total GPU blocks`. It can exceed 1 and is `Inf`
when total blocks are zero. `remaining_steps` is `(available blocks -
waiting-request block estimate) / effective request count`; with defrag enabled
the request count and head-of-line block subtraction differ. It means remaining
KV-block capacity per counted request. It is not a request's remaining token
count, decode steps, or execution time. A no-request instance reports infinity.
Normalization maps non-finite values to finite numbers and carries a separate
missing/invalid mask.

## Migration decision and execution path

`GlobalScheduler.push_migrations()` delegates to `MigrationScheduler` and
returns a list of `(MigrationType, list[(source_instance_id,
target_instance_id)])` tasks. In non-PD mode, routine migration is
`NEUTRAL_LOAD_BALANCE`. The base filter checks migration-backend readiness,
valid computed load metrics, and available migration slots; routine load
filters select overloaded sources and eligible destinations. Existing
Balanced, Defrag, and Failover policies remain unchanged.

`Manager._migrate()` iterates those tasks and creates asynchronous Llumlet
`migrate_out()` calls. Dispatching that remote call is not proof of success.
`MigrationCoordinator._migrate_out_one_request()` performs the transfer and
destination commit, then logs the terminal status, request IDs, total copied KV
blocks, and elapsed milliseconds (`migrate done` line). A returned request ID
means the transfer committed; empty results can also mean rejection or abort.
The coordinator log is the existing source for success/failure/cost telemetry.

With `INSTANCE.REQUEST_MIGRATION_POLICY: SR`, the local scheduler filters
running, non-migrating Decode requests whose current request length is strictly
between the configured min/max bounds, then selects the shortest by
`request_len` (prompt plus generated tokens in the vLLM v0 wrapper). The
previous `num_blocks_last_running_request` is sampled from the last running
queue element in engine output processing. It is not the selected SR request's
KV size and must not be used for Mig-PPO cost or capacity checks.

For vLLM v0, `InstanceInfo` now has optional, read-only SR candidate fields. The
scheduler populates those from the exact current SR selection and its block
table; absent candidates are explicitly invalid. These fields are additive and
do not change the existing migration API or request selection. The candidate
may still change between observation and migration; execution-time selection
must be revalidated before any future live Mig-PPO integration.

## Existing simulator capability and limits

`docs/Simulator.md` documents collection of real instance profiling CSV,
fitting latency models into a profiling database, and starting the existing
simulator backend. `BackendSimVLLM` reuses the Llumnix scheduler and normal
multi-instance request/migration path. `SimGPUExecutor` schedules simulated
prefill/decode latency from fitted profiling curves, returns dummy token IDs,
and models cache transfer latency from cache-block size and migration
bandwidth. Thus it can exercise queue and KV movement through the existing
backend when run as a Llumnix/Ray service.

It is not a standalone Gymnasium-style environment: it has no `reset(seed)`,
`step(action)`, action/reward collection, or seeded request-trajectory API.
Its latency curves are fits to a specific benchmark profile and extrapolation
is not evidence of real-GPU performance. The new independent environment uses
explicit service-time and migration-cost calibration files, advances request
queues/KV allocation/completion after each action, and marks default calibration
as uncalibrated. Results are simulation evidence only until calibrated and
validated against the target four-GPU runs.

## Mig-PPO boundary

The research package consumes `InstanceInfo`-shaped read-only state, derives a
masked Stay-or-directed-pair action, and supplies independent training and
reward tooling. The existing Manager, GlobalScheduler return structure,
RayRPC transfer path, and local KV-copy mechanics are not replaced by this
phase. A later live integration must preserve failover/pre-stop paths and use
the coordinator's terminal migration event for execution telemetry.
