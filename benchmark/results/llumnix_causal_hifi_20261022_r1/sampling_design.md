# Causal Llumnix sampling design

## Purpose and status

This is the sampling contract used for the high fidelity simulator rebuild. It
adds events at the client, API server, Llumnix scheduler, engine executor,
individual GPU workers, KV allocator, and migration coordinator. The
implementation is opt in through `LLUMNIX_CAUSAL_TRACE_DIR`; ordinary serving
runs keep the telemetry path disabled. Four GPU calibration, paired-mode, and
held-out runs are recorded below. A separate output-queue residence event is
still needed: current traces identify engine-to-API delay but do not split each
queue stage.

## Event clocks and identity

Every JSONL event has a schema version, process and thread identity, a host name,
`CLOCK_MONOTONIC` nanoseconds, and a Unix wall clock nanosecond timestamp. A
`clock_sync` event records a wall/monotonic pair and a bound on the sampling
uncertainty. The client and four Llumnix workers are on one host, so monotonic
timestamps share one clock. Cross-host results must retain host identity and be
offset-calibrated before joining; wall-clock time alone is not used for latency
subtraction.

The request ID sent by the client is retained through the API, engine queue,
scheduler batches, token outputs, and migration operations. Batch IDs are local
to the serving process and join schedule events with executor events. GPU worker
records carry request IDs plus sequence-length signatures so they can be joined
to the engine batch without forcing a CUDA synchronization.

Prompts, generated text, and token contents are never written to telemetry. The
sampler records lengths, IDs, state transitions, and timing only.

## Request lifecycle

The client request row records scheduled arrival, actual send monotonic time,
HTTP completion monotonic time, input length, output budget, actual output length,
HTTP status, timeout, and error. The server event stream records:

- API request receipt and output budget;
- engine admission and the selected instance;
- each output update, the cumulative and newly emitted token counts, and its
  instance;
- completion, abort request, cancellation, error, or censor status.

When the trace header is enabled, the API records the raw `RequestTimestamps`
timeline at the first and final output updates: API, manager, Llumnix instance,
engine admission, output processing, queue send/receive, and API delivery. The
compact run still records every API output update separately. This preserves
server token completion and client HTTP response completion as separate
measurements without serializing the full timeline on every token. HTTP 499 is
recorded as cancellation/censoring when it came from the drain barrier; it is
not recoded as an engine failure.

Each generated engine output and API callback is joined by request ID and
cumulative output-token count. The validation overlay reports this
engine-to-API callback delay and the fraction of sent measurement requests
whose first token is visible within the measurement window. It also keeps
first-token requests arriving after the window as right-censored observations.
New instrumentation records engine output, output-forwarder submission,
forwarder-thread start, queue-client send, ZMQ enqueue, dequeue, queue depth,
and API callback times for the first, every 64th, and final output updates. The
queue events are sampled in both single-item and batch ZMQ paths and in both
vLLM client paths. Joins use request ID, instance ID, and cumulative token
count so migration cannot cross-match output updates from different instances. Seed
1213 and earlier calibration runs predate these fields; they remain useful for
end-to-end output-delivery diagnosis but cannot estimate per-stage queue time.
The repository's initial Ray placement-group wait timed out at 5 seconds in an
ON startup before serving began. The first retry set
`LLUMNIX_WAIT_PLACEMENT_GROUP_TIMEOUT=60`, but exposed a second startup bug: the
API caller waited a hard-coded 60 seconds for the scaler's serial
`init_instances` RPC and timed out before the placement-group wait returned.
The caller now budgets `max(60, initial_instances * placement_group_timeout +
30)` seconds, overridable with `LLUMNIX_INIT_INSTANCES_TIMEOUT`; both timeout
values are saved in each run config. This changes startup waiting only, not the
migration policy or inference model. Failed startup directories are retained
and contain no measurement requests.

A same-trace OFF retry then reached the 900 s outer startup deadline while the
API setup code was checking each instance sequentially. One instance became
ready; three others had loaded model weights and allocated KV cache but their
Llumlet constructors had not returned. Readiness futures are now submitted and
waited together, with the per-instance deadline applied to the full group.
`LLUMNIX_INSTANCE_READY_TIMEOUT` and `LLUMNIX_SERVER_STARTUP_TIMEOUT` control
that deadline and the runner's health gate; both values are written to run
configuration. The timed-out OFF attempt remains preserved and is excluded
from the comparison.

## Scheduler batch and GPU events

At each nonempty scheduler decision, the sampler records the scheduled request
IDs and, per request, prefill/decode state, sequence count, context length, and
scheduled token count. Batch-level fields include:

- batch type (`prefill`, `decode`, or `mixed`), prefill and decode counts/tokens;
- vLLM `num_batched_tokens`, total context tokens, sequence lengths, and batch
  size;
- running and waiting request/sequence counts before and after scheduling;
- preempted, ignored, swapped, and copied block counts;
- used, free, total, and fractional KV blocks at the scheduler boundary.

The executor records wall time from Ray dispatch through returned worker output.
Each request output records the server-visible token count and finish reason.
GPU workers use paired `torch.cuda.Event` measurements on the model execution
stream. A background poller reads completed CUDA events; no per-iteration device
synchronization is added. CUDA elapsed time is therefore explicitly distinct
from executor iteration wall time and request token latency. The instrumentation
also records request-local KV block-table deltas at scheduler boundaries.

## Migration events

Each migration has source and target instance IDs and a request ID. The event
stream records attempt start/end, stage boundaries, destination preallocation,
cache transfer start/end with block count, destination commit, and source/target
request resume. The final source removal is the direct request-pause start. The
first subsequent API-visible token event identifies actual generation recovery.

The resulting labels are kept separate:

- **Migration total duration:** attempt start through terminal result.
- **Request pause:** final removal from the source through the first token after
  the migration (with an intermediate target-commit timestamp retained).
- **Recovery cost:** target commit through the first token after migration.

Failed and aborted migrations retain their measured partial stages and status.

## Run matrix and split discipline

The run matrix covers QPS 3, 9, 12, 15, 18, 21, and 24, under Migration OFF
and ON. Each mode pair reuses a byte-identical saved arrival trace and the same
prompt/budget profiles. Seeds 606 and 909 are calibration groups; seeds 808,
1001, and 1213 are held-out run groups. The seed 1213 system gate is the
current full seven-QPS OFF/ON acceptance result. No request, token, batch, or
time sample from a run is split across sides. The 60 second arrival window and
30 second drain produce an explicitly censored/unfinished tail at the shared
90 second horizon. The runner completes a separate 15 second warmup phase
first, waits through its drain, and aborts any remaining warmup requests before
starting measurement. Validation therefore starts the measured queue and KV
state empty at this phase boundary; GPU cache and thermal state from warmup are
outside the simulator.

Output budgets and realized lengths are distinct. The collector records the
actual server token count and finish reason, including censored token prefixes
at the abort boundary. It does not use `min_tokens=max_tokens` as a length label.
Any conditional length model will account for completed EOS outputs, budget
stops, and right-censored prefixes.

Every run stores its trace checksum, serving config, code commit and dirty diff
hash, hardware/software manifest, model ID, random seed, measurement windows,
startup gate, request count, and telemetry file inventory. Existing Round 2 and
Round 3 directories are read-only inputs and are never overwritten.

## Overhead and completeness checks

The JSONL writer queues events from inference threads to a background writer,
flushes every 128 events or 250 ms, and flushes on shutdown without an `fsync`.
Queue/KV state is captured with every nonempty scheduler batch and at 50 ms
intervals while the scheduler is idle; validation resamples these observations
to a 50 ms carry-forward grid. CUDA timing never synchronizes the serving stream.
The pilot will report event counts, JSONL parse failures,
process-local sequence gaps, GPU timing coverage, and comparison of throughput
and batch wall time against an otherwise identical telemetry-disabled run. A
dataset is not eligible for fitting if event drops, unmatched batches, missing
request terminals, or clock-order violations exceed the declared audit limits.

Requests use their ordinary EOS behavior (`ignore_eos=false`); the requested
output length is only the maximum token budget. This lets the collector measure
EOS completion, budget stops, and drain-censored prefixes separately.
