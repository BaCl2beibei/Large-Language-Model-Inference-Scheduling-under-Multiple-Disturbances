# P0 implementation and environment snapshot

Run: real_ppo_pilot_20261022T145922Z_04f200
Captured: 2026-10-22T14:59:22.113725+00:00
P0 status: environment inventory and evidence packaging complete. This is not real-cluster acceptance.

## Runtime ownership

The project virtual environment .venv/bin/python exists and reports Python 3.10.12, Ray 2.47.1, vLLM 0.6.3.post1, torch 2.4.0, and Llumnix 0.1.dev230+g09e0a10.d20261020. It is the existing Llumnix runtime. /venv/main/bin/python is Python 3.12.15 with CPU-only torch 2.6.0 and no Ray/vLLM. System Python is 3.12.3.

Four RTX 3060 GPUs (12 GiB each) were visible and showed 0% utilization / 1 MiB reported use. nvidia-smi reported no compute PIDs. supervisorctl status showed caddy, cron, instance_portal, syncthing, tensorboard, and tunnel_manager running; jupyter and pyworker were exited. No Ray or vLLM process was observed. Long-lived Python children were associated by parent process with managed services; PID 779 (user) was under the syncthing wrapper. Nothing was stopped.

## Saved inference configuration

No live inference server was detected, so the current live inference configuration is not available. The latest saved validation config is benchmark/results/ppo_system_validation_round3_fixed_lengths_20261022/qps_15/pilot_spec.json (QPS 15.0, seed 505, 60 s arrival + 30 s drain, four instances, model length 2112, max sequences 128, GPU memory fraction 0.5, eager mode True). It records a forced exact output-length validation policy and is historical evidence only; the new draft does not reuse that policy. The earlier calibration config names /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/model/Qwen/Qwen2-0.5B with tensor parallel size 1.

## Source and change preservation

The baseline was HEAD 92f05f545943b2f984bde1ddd30c7bf8e413da56 with 18 tracked modifications, 310 tracked deletions, and 1266 untracked paths. All pre-existing status entries were recorded. The complete tracked diff, including deletions, is saved compressed at source_snapshots/tracked_worktree_diff.patch.gz; 6 related untracked Python source files were copied byte-for-byte under source_snapshots/untracked_worktree/ and hashed. Model weights, datasets, and old result trees were not copied. P0 did not edit business source.

## New real-PPO implementation spec verification

The taskbook proposed real environment, bridge, observation, metrics, reward, rollout, trainer, evaluator, config, runner CLI, and test_real_*.py files are absent in the current worktree. .venv/bin/python is present and confirmed as the existing Llumnix environment. The CLI/API remains a pending implementation contract.

## Budget and protocol

No task-owned inference/Ray/GPU process was running; no model was loaded and no GPU experiment was started in P0. The task-specific 8-hour experiment clock therefore remains at 0 seconds under the P0 rule. Historical benchmark result directories are not linked to an active real-PPO run ledger. The workspace volume flag is false; artifacts here will not survive recycle/destroy and need an off-box backup if irreplaceable.

experiment_spec.yaml is a candidate draft: QPS probes 3/6/9; 120 s arrival, 120 s drain, fixed 240 s score window, 60 s cleanup deadline; TTFT 2,000 ms and TPOT 150 ms; seed groups recorded after a metadata/path scan. It is not a professional review. Exact sampling parameters must be captured before P3 opportunity runs.

Automatic budget expiry enforcement is not implemented. budget.json records that gap; any future GPU step must wait until a persistent timer/enforcer is implemented. No automatic cleanup capability is claimed.

## Handoff

P0 artifacts are stored in this run directory. The shared active pointer is /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/rl_scheduler/runs/real_ppo_handoff/active_run.json. The next step is Sol's P0 review and P1 CPU implementation. No later stage was started.

## P1 / step 02 — CPU implementation and offline acceptance

This section supersedes the P0 statement that all proposed real metrics files are absent. P1 is an offline gate only. No cluster, GPU experiment, PPO, or independent live evaluation was run. Existing non-streaming `/generate`, `/generate_benchmark`, and `/abort_benchmark` behavior is retained. New code is opt-in through the following interfaces.

### Modules and caller contract

- `rl_scheduler/real_metrics.py`: `Event`, `Checkpoint`, `EventJournal`, `EventCursor`, `RequestLedger`, `CausalEventAdapter`, and `migration_diagnostics`.
- `rl_scheduler/real_client.py`: `StreamingRequestCollector.issue/cancel` and `NulJSONDecoder`.
- `llumnix/entrypoints/vllm/real_metrics_stream.py`: installs `/generate_real_metrics`, `/cancel_real_metrics`, and `/real_metrics_flush` on the existing API. No listener/service/port was created. Output frames carry cumulative actual token counts and cumulative generated text; text is never copied into request telemetry.
- `llumnix/entrypoints/vllm/api_server.py`: additive registration and an explicit abort RPC barrier. The legacy client's `abort()` catches RPC errors, so its return cannot be used as a cancellation acknowledgement. The new barrier awaits the existing instance abort RPCs and only then clears the existing client stream state. No KV transfer or scheduling mechanism was added.
- `llumnix/metrics/causal_events.py`: additive stream epoch IDs and a bounded `flush(timeout_sec)` barrier. Sequence assignment and enqueue are serialized together. Durable acknowledgement requires writer flush + fsync; disabled, closed, failed or timed-out writers return incomplete. Existing output sampling remains unchanged.

The future runner must construct the complete offered request ID manifest from its frozen trace **before issuing traffic**, and inventory all relevant producer stream IDs. Pass both into `RequestLedger(... expected_request_ids=..., expected_sources=...)`. Missing manifests, missing producers, missing sequence positions, conflicting events, token regressions, or unconfirmed watermarks return incomplete. A zero-event source needs an explicit zero-sequence acknowledgement too. Do not omit an engine/API producer to make the audit pass.

Register every planned request as an event, including requests that never send. Use one logical `(run_id, episode_id, request_id)`; server IDs are mapped explicitly and namespaced with a deterministic digest of those three IDs. A second dispatch under the same ID is rejected (409); the client never retries automatically. A reused server ID across logical requests or conflicting send events is audited as incomplete.

`EventJournal.record()` is for events in this host's CLOCK_MONOTONIC nanosecond domain. Producer-local sequence/watermark files are durable. `EventCursor.poll()` retains partial JSONL lines and offsets, detects rotation/truncation/malformed complete lines, and stamps actual collector receipt. `RequestLedger.poll_events()` propagates cursor faults into the ledger. For causal files pass a `CausalEventAdapter`, configured with the saved server-to-logical mapping; poll **whole** streams rather than request-filtered shards so sequence continuity remains meaningful. Raw legacy records are readable, but absent producer acknowledgements prevent a completeness pass.

Collect checkpoints from every inventoried process through the relevant boundary and feed `acknowledge(Checkpoint(...))`. API `/real_metrics_flush` confirms only its own process; it does not acknowledge engine actors or pending output delivery. Engine actors need to invoke the same process-local `flush()` via future bridge integration. Acknowledged `through_ns` means no producer may subsequently append an occurrence at or before that watermark. A violation invalidates the audit.

`observe(decision_ns)` uses only events received by that decision, as well as occurred by it. Delayed collection never changes an earlier observation. Offline reconstruction may use event occurrence time after collection completes. `freeze_score()` freezes only after the complete producer acknowledgements and offered manifest pass. Once frozen it returns the same score; contradictory late scoring evidence still invalidates the separate audit. The caller must never train on score_status alone: **both score and cleanup audit must pass**.

`export(directory, cleanup_through_ns=...)` writes `requests.jsonl`, `events.jsonl`, `window_metrics.json`, `cleanup.json`, `audit.json`, `checkpoints.json`, and `producer_manifest.json`. Request rows preserve every occurrence/receipt pair, server mapping, score outcome, final cleanup outcome, cancellation confirmation, and measured internal/client first times. The runner must additionally retain the original causal diagnostic records and the freeze-time checkpoint history. `migration_diagnostics(raw_events)` can be serialized alongside the request exports; it only pairs explicit total, pause-start, commit-start, commit-ack/resume and subsequent destination engine-output timestamps. Missing pairs and client recovery remain null, with no inferred pause or token ITL.

### Metric definitions and boundary convention

The arrival and fixed score windows are nested closed timestamp windows `[start, arrival_end]` and `[start, score_deadline]`; event exactly on either right boundary is included. Residence integrals use the continuous interval and are unaffected by point inclusion. These are not successive reward windows; a future reward implementation must use a consistent non-overlapping convention for adjacent windows.

- All offered requests, including never-sent ones, are the completion-rate denominator. `sent`, `accepted`, engine/API/client first, server/client terminal, cancellation requested/acknowledged and score deadline remain separate fields/events.
- Completion and goodput count only normal **client** success at/before the window end. TTFT is client first positive output minus actual send, not API or engine latency. TPOT is `(client_last-client_first)/(actual_output_tokens-1)` in milliseconds; one token is N/A and is counted separately. Goodput is SLO-qualified completion count / fixed window seconds (TTFT <= 2000 ms; TPOT <= 150 ms; one-token requests use TTFT only).
- `client_output_tokens` sums only positive increments of the cumulative **client received** high-water count inside the window. Identical counts do not extend last output or add tokens; regression is preserved, does not double count recovery, and blocks acceptance. HTTP packet/chunk counts never count as tokens. `completed_output_tokens` is a separate diagnostic for normal completions. Visible partial output before the deadline may contribute token throughput even if its request finishes later; the later output and HTTP success cannot be added to the frozen window.
- Conditional TTFT samples include requests whose first client output is in-window, including later unfinished requests. Conditional TPOT includes in-window normal completions. Report first-output censored count, outstanding IDs and terminal outcomes alongside these conditional distributions. P95 uses linear interpolation at `(n-1)*0.95`.
- `outstanding_request_seconds` integrates **all sent** requests until actual client terminal, clipped to the window. Engine completion or an empty engine queue cannot end this integral. Overdue-first integral begins at send + TTFT SLO. Output-gap integral begins after each positive client output + `max(1s, 5*tpot_slo)` and stops at the next output/client terminal/window end. `longest_no_output_ms` uses the same positive output timeline including its observed tail. Raw packet chunk intervals are reported separately. Exact per-token ITL/P95 ITL is explicitly unavailable.
- After deadline a missing client terminal has score outcome `censored`; cleanup outcome remains unresolved until a known terminal arrives. An unknown terminal is **unknown**, never changed into censoring/success/failure/cancellation. Manual HTTP 499/local cleanup cancellation becomes `manual_cancel` only with a matching requested cancellation and positive acknowledgement. Unconfirmed cancellation stays unknown and makes the cleanup audit incomplete. Genuine client timeout is a distinct outcome.
- Cleanup uses separate client 60-second/server 55-second bounds for the whole cancellation batch, not the old 900-second abort barrier. A request absent from the server lookup is unknown. A backend acknowledgement does not prove all pending output delivery is drained: each producer needs a separately justified `pending_delivery=0` before cleanup audit passes.

### Offline evidence, P0 supplement and limitations

P0 was rechecked against the actual pointer, HEAD, all 172 source inventory entries, six untracked source copies, raw/decompressed tracked diff, pre-existing Git statuses, configuration and captured evidence hashes. All 14 checks passed; see `p1_evidence/p0_review.json`. P0 did not specify the derivation of its aggregate `source_hash`; P1 supplements it with independently checked file bindings and a documented canonical source-map hash, without rewriting the historical P0 gate. P0's budget enforcer and exact live sampling/protocol freeze remain unresolved; those are still blockers before GPU experiments.

Reproduce with:

```sh
.venv/bin/python benchmark/results/real_ppo_pilot_20261022T145922Z_04f200/p1_evidence/verify_p1.py
```

The existing project Python environment is used; no package/model/CUDA/driver was installed or changed. 28 substantive unittest cases cover manual calculations, initial/right boundaries, delayed receipt, duplicated tokens/terminals, regression, one token, all-sent residence, manual vs timeout vs unknown, missing terminal, missing offered requests, pending delivery, sequence gaps/missing tail, immutable score, durable concurrent writer flush, partial incremental reads, sampled diagnostics, fragmented real localhost HTTP/UTF-8, incomplete EOF, client timeout, duplicate request rejection and an ASGI cancellation barrier with an injected backend.

The saved hand-calculated example offers four requests (one never sent), normally completes two within 8 seconds and one during cleanup. Fixed completion rate is 2/4=0.5; SLO goodput is 2/8=0.25 rps; visible token throughput is 7/8=0.875 tps; outstanding integral is 2+3+6=11 request-seconds; output-gap integral is 0+1+4=5 request-seconds. The late completion never changes the frozen score. Raw events, request rows, metrics, cleanup and audit are in `p1_evidence/hand_calculated/`.

No existing benchmark logs were treated as new independent tests or training input. Full Llumnix/vLLM/Ray loading, actual GPU output delivery, migrated-request abort behavior, cluster-wide producer discovery/flush RPCs, pending-delivery counters, control ownership, timing/overhead and live metric agreement remain **unverified**. The optional routes retain per-request identity/state for the serving process lifetime; a future bounded service lifecycle must rotate/release finished episode state safely. Existing Llumnix internal failover/retry behavior is not replaced here; failover/faulted episodes must be invalidated by the future runner. These limitations prohibit a p1_live pass. No `gates/p1_live.json` is generated.

P1 stops after saving `gates/p1_offline.json` and `handoff/02.{md,json}`. P2 CPU work can consume this offline gate only after checking its hashes. Any GPU phase still requires the missing persistent budget enforcement, its own authorization/dependencies and live acceptance; no subsequent phase is automatically started.
