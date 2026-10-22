# Mig-PPO simulator causal audit and admission decision

**Decision: not suitable for formal PPO training.** The lifecycle and accounting repairs pass deterministic checks, but the final five-QPS system prediction gates fail by wide margins. The configured 2,000-episode CUDA training and post-training policy simulation were therefore not started. This follows the conditional training rule in the goal objective.

## Scope and preserved evidence

The original calibration collection remains intact at `benchmark/results/ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505/`. It contains 20 Migration ON runs at QPS 9/12/15/18/21, with seeds 101/202/303 used for fitting and seed 404 used as the holdout. Seventeen runs predate and three follow the migration-slot guard change. The new OFF/ON pilot is isolated under `benchmark/results/ppo_migration_pilot_qps15_seed505_20261021/`; it does not overwrite or refit any earlier run. Existing unrelated modified and untracked repository files were preserved.

The simulator's code changes are under `rl_scheduler/`. The detailed per-cell system results and calibration remain in `system_validation.md`, `system_validation.json`, `model_error_attribution.md`, `mig_ppo_calibration_system_validation.yaml`, and `mig_ppo_calibration_scale1_fallback.yaml` in this output directory.

## Causal-chain audit and repairs

| Area | Evidence and repair | Remaining limit |
|---|---|---|
| Reset and deterministic inputs | Fixed stale trace RNG use when switching from trace arrivals to Poisson arrivals. A deterministic reset regression test covers the transition. | Randomized production traces still depend on their configured seed, as intended. |
| Batch time and cadence | A mixed GPU batch now shares one completion timestamp and advances the simulation clock once. Raw P1/D3 `predicted_ms` is kept separately from cadence-scaled effective decode interval; a mixed batch uses the modeled maximum prefill/decode interval rather than summing per-request times. Optional lifecycle tracing records the raw and effective values and the one clock advance. A directed mixed-batch test checks shared completion and clock behavior. | The `max(prefill, scaled decode)` mixed-batch service rule remains an approximation. Executor iteration wall labels do not independently establish real mixed-batch overlap or whole-engine-step semantics. The cadence factor is empirical and run-dependent, not universal. |
| Admission, sequence cap, and KV | The invariant checker compares registered requests with queue ownership, enforces the active sequence limit including pending incoming migrations, and checks used plus reserved KV blocks against capacity. Incoming migrations reserve a destination slot, with capacity rechecked at migration start and replay. Targeted migration/capacity tests pass. Pilot worker logs report 22,084 KV blocks per instance in both modes, confirming the runtime-reported value for this deployment; the older 44,111-row log inference agrees. | Capacity is vLLM-reported, not an independent hardware measurement. Old production logs lack exact per-request block allocation/free transitions. |
| Release and failure paths | Completion is idempotent and releases resources once. Invalid frees are no longer hidden by `max(0, ...)`. Migration replay records observed versus reserved blocks, rejects identical source/target, and terminal migration wakes idle source/destination queues. Tests cover timeout during migration, resource release, and queue wakeup. | Real abort/censor logs do not give a full independent request-by-request KV ledger. |
| Queue ownership and state transitions | Optional lifecycle events cover arrival, wait/admission, batch schedule/complete, first token, allocation/release, preemption, completion/failure, and migration. Invariants reject duplicate/missing queue ownership and illegal source/target migration states. | Historical logs do not capture every scheduler decision or exact initial worker routing. |
| Observation timing and sampling | System-validation queue/KV data uses a shared fixed 50 ms grid with last observation carried forward, averages instances at each time sample, and reports p95 over the common 60 s work window. Tests cover the fixed-grid carry-forward behavior. Client response completion remains separate from server token timestamps. | The pilot's descriptive queue/KV summaries use five-second bins; they are not substituted for the 50 ms admission gate. |
| Future-information leakage | Closed-loop observations are built from current state, recent history, and current candidate actions. The closed-loop runner receives no logged route map or future migration schedule. Fixed-event replay receives logged events by design and is reported separately. The report records the closed-loop leakage check. | Closed-loop validation uses a greedy controller, not PPO. It does not demonstrate learned-policy behavior. |

The simulator supports configurable service models and cadence. The final experimental cadence candidate is 5.827594; the fallback YAML sets cadence to 1.0. P1/D3/M2 and the candidate/fallback are kept in YAML rather than being hard-coded into the transition logic.

## Fixed-trace cadence experiment

`paired_cadence_trace_qps15_seed404.json.gz` records both cadence settings on the same QPS 15 arrival trace and realized output-length samples drawn from training seeds 101/202/303. Seed 404 was not used to fit cadence. It runs without migration or future route/event injection.

| Measure | Scale 1.0 | Candidate 5.827594 |
|---|---:|---:|
| Completed in 60 s work window | 786 | 243 |
| Completed by common 90 s horizon | 888 | 414 |
| Failed / unfinished by 90 s | 0 / 0 | 0 / 474 |
| Shared 90 s completed cohort TTFT mean / P95 | 137.56 / 186.75 ms | 2,012.18 / 9,871.07 ms |
| Shared cohort TPOT mean (n=389) | 86.05 ms | 708.75 ms |
| Token intervals (n=15,873) | 85.41 ms mean | 713.92 ms mean |

The cadence candidate slows completion substantially and does not close the request-level gap. Queue mean/p95 were 0.179/0.5 at scale 1 and 5.984/30.012 at the candidate. KV mean/p95 were 1.29%/1.63% and 3.79%/5.78%, respectively. This is a mechanism comparison, not a universal estimate: only one held-out arrival trace was replayed, while output lengths came from training data. It supports retaining scale 1 as an explicit rollback and treating 5.827594 as an experimental candidate.

## Five-QPS system validation

The system report contains 40 cells: five QPS values × four versions (V0–V3) × fixed-event and closed-loop replay. The modes are separate because fixed replay injects logged migration events while closed-loop greedy chooses migrations without those future logs. Neither closed-loop mode is PPO.

Observed requests over the shared horizon had no true failures or client timeouts. All 3,970 historical HTTP 499s are drain-censored. Per-QPS arrival and observed HTTP 200 / 499 counts were:

| QPS | Arrivals | HTTP 200 | Drain-censored 499 | True failures / timeouts | V3 closed-loop complete / unfinished by 90 s |
|---:|---:|---:|---:|---:|---:|
| 9 | 549 | 400 | 149 | 0 / 0 | 325 / 224 |
| 12 | 712 | 608 | 104 | 0 / 0 | 377 / 335 |
| 15 | 888 | 827 | 61 | 0 / 0 | 415 / 473 |
| 18 | 1,069 | 579 | 490 | 0 / 0 | 427 / 642 |
| 21 | 1,239 | 1,056 | 183 | 0 / 0 | 445 / 794 |

The validation compares the common 90 s outcome horizon and 60 s arrival/work window. Observed client HTTP response counts, server completion timestamps, and first-window completions are retained as separate fields; the report separately gives client/server timeline gaps and the 60 s completion metric. Queue and KV use the common 60 s grid.

| V3 error summary | Fixed-event replay | Closed-loop greedy | Admission limit |
|---|---:|---:|---:|
| Throughput MAPE | 37.48% | 38.13% | ≤20% |
| Mean TTFT MAPE | 72.52% | 72.83% | ≤25% |
| P95 TTFT MAPE | 56.18% | 56.00% | ≤30% |
| Mean TPOT MAPE | 21.66% | 21.48% | ≤20% |
| P95 TPOT MAPE | 23.89% | 25.60% | ≤25% |
| Queue mean MAE | 7.064 requests/instance | 7.199 requests/instance | ≤max(0.5, 20% of observed mean) |
| Queue P95 MAE | 30.64 requests/instance | 31.543 requests/instance | ≤max(1, 20% of observed P95) |
| KV mean absolute error | 1.73 pp | 1.77 pp | ≤5 pp |
| KV P95 absolute error | 3.65 pp | 3.66 pp | ≤10 pp |

KV gates pass. Throughput, TTFT, mean TPOT, closed-loop P95 TPOT, and both queue gates fail. Per-QPS main-metric errors also exceed 40% without a validated explanation: worst closed-loop relative errors are 57.86% for throughput, 92.76% for mean TTFT, 95.29% for P95 TTFT, 32.72% for mean TPOT, and 48.14% for P95 TPOT. The fixed-event per-QPS table and all V0–V3 values are in `system_validation.md` and its JSON. Seed 404 was already used for diagnosis and is not a new independent final holdout. The new seed 505 pilot covers QPS 15 only and cannot replace five-QPS confirmation.

## Paired Migration OFF/ON pilot

The bounded pilot used QPS 15, seed 505, the same saved trace in both modes, 942 arrivals, a 60 s arrival window, and a shared 90 s server horizon. The mode configs differ only in `ENABLE_ROUTINE_MIGRATION`; the startup gate confirmed four ready eager workers and four Ray workers. Each command exited 0. `pilot_run_manifest.json` records commands and gate evidence; raw logs/configuration and request/token records remain in `off/` and `on/`.

| Measure | OFF | ON |
|---|---:|---:|
| Server final tokens by 60 s / 90 s | 407 / 650 | 223 / 226 |
| Requests without a server final token by 90 s | 292 | 716 |
| HTTP 200 / drain-abort 499 | 767 / 175 | 588 / 354 |
| Client bodies received by 90 s | 0 | 7 |
| Server TTFT mean / P95 by 90 s | 11,550.28 / 28,068.08 ms | 8,138.91 / 28,597.96 ms |
| Server TPOT mean / P95 by 90 s, full timelines | 935.12 / 1,269.56 ms (n=255) | 437.04 / 846.64 ms (n=85) |
| Queue mean / P95, equal-weight five-second bin means | 2.937 / 11.370 | 25.008 / 132.000 |
| KV mean / P95, equal-weight five-second bin means | 12.256% / 21.822% | 16.377% / 26.982% |
| Measurement migrations, finished / unsuccessful | 0 / 0 | 6 / 0 |

The apparent client HTTP success count is not the server completion count at the 90 s horizon. All 767 OFF HTTP 200 bodies arrived after 90 s; ON had 7 bodies by 90 s and 581 afterward. Server final-token timestamps show 117 OFF and 362 ON completed later than 90 s; the other 175 and 354 requests were abort-censored at the drain barrier. There were no client timeouts or other statuses. The final-token-to-client-body lag for time-lined successes averaged 28.18 s OFF and 16.34 s ON. The API token clock (`perf_counter`) and benchmark (`monotonic`) share the same `CLOCK_MONOTONIC` implementation; raw conversions and request timelines are retained in the pilot JSON.

The five-second means expose a queue spike that row-weighted per-instance snapshots hide: OFF raw snapshot queue mean/P95 was 2.568/18, ON was 0.876/6, while maximum ON snapshot queue was 260. The five-second mean is descriptive and is not the five-QPS 50 ms gate. Both server logs report 22,084 KV blocks per worker. ON's six migration terminal records all finished, with cost mean/P95 582.468/791.392 ms and block count mean/P95 20.167/29.25. No Pause Time label was measured. ZMQ output-forwarder errors (“Server is dead” / “Socket operation on non-socket”) appeared in both modes during abort/load; the single run per mode cannot assign cause. The 424-completion ON-minus-OFF difference is a single paired contrast, not a replicated causal effect or run-to-run estimate.

## Service and migration model residuals

| Model | Grouped CV MAE / RMSE | Seed 404 MAE / RMSE | Bias (seed 404) | Residual P05 / P50 / P95 (seed 404) |
|---|---:|---:|---:|---:|
| P1 prefill, all phases | 41.44 / 93.31 ms | — | — | — |
| P1 prefill, measurement only | 45.87 / 102.38 ms | 42.17 / 89.37 ms | −0.75 ms | −106.0 / 19.9 / 49.1 ms |
| D3 decode, all phases | 26.50 / 101.83 ms | — | — | — |
| D3 decode, measurement only | 30.43 / 82.99 ms | 30.12 / 71.79 ms | +4.73 ms | −67.6 / 14.0 / 50.2 ms |
| M2 migration | 342.29 / 698.54 ms (74 CV events) | 334.98 / 566.92 ms (31 events) | −65.0 ms | −684.8 / 113.1 / 466.2 ms |

P1 and D3 labels are executor iteration wall time, not kernel time, request TTFT/TPOT, or a complete engine step. Their extreme iteration stalls are sparse and not attributable to one available feature. M2 residuals are status-dependent and heavy-tailed; seed 404 has only 5 `ABORTED_DST` and 26 `FINISHED` events. Migration `cost_ms` is end-to-end duration, not service pause. The pilot adds six successful ON events but no pause truth, so it does not validate migration blocking time.

## Reproducibility, smoke, and commands

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /venv/main/bin/python -m rl_scheduler.system_validation --data-root benchmark/results/ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505 --output-dir benchmark/results/ppo_system_validation_round2_causal_repairs_20261022` — exit 0; 40 result cells.
- `/venv/main/bin/python -m compileall -q rl_scheduler` — exit 0.
- `/venv/main/bin/python -m pytest -q rl_scheduler` — exit 0; 35 passed, one pre-existing warning about unknown `asyncio_fixture_loop_scope` pytest configuration.
- Targeted `env.py` and `system_validation.py` tests — exit 0; 14 passed.
- Paired pilot commands are recorded in its run manifest; OFF and ON both exited 0.
- Final bounded CPU smoke command and output are recorded in `ppo_smoke_run_metadata.json`: `--timesteps 8`, seed 322, one episode, one PPO update and one evaluation; exit 0. It saved `mig_ppo_best.pt` and `mig_ppo_final.pt`. Two earlier bounded attempts were stopped at exit 130 before metrics/checkpoints and have individual `INCOMPLETE.md` notes.

The formal configuration is 2,000 episodes on `cuda:0`; the actual CLI priority was audited. It was not run. No training checkpoint from a formal budget and no post-training policy comparison exist. The calibrated fallback and candidate YAMLs are present, but passing functional smoke does not compensate for failed system accuracy gates.

## Rounds and stop rule

- **Round 1:** retained the training-only cadence fit (5.827594) as an experimental candidate after the earlier system comparison showed lower TPOT error but worse throughput error. The same-trace scale 1 comparison and explicit fallback are retained.
- **Round 2:** repaired reset determinism, lifecycle/queue/KV invariants, migration reservations/releases and wakeups, mixed-batch time accounting, fixed-grid queue/KV sampling, and trace diagnostics. Targeted tests and the complete simulator test suite pass. A new QPS 15 paired OFF/ON pilot was then run to address the missing service baseline.
- **Stop:** the five-QPS system gates still fail, the pilot is only one QPS/one paired seed, and measured pause-time truth is unavailable. No third model-tuning round is justified from this evidence; continuing against the already inspected seed 404 or fitting the one-run pilot would risk tuning to insufficient validation data. Preserve the best current implementation and report it as **not suitable for formal PPO training**.

A bounded CPU PPO smoke demonstrates that reset, stepping, action sampling, one update, evaluation, and checkpoint save execute. It is not formal training. No 2,000-episode job or post-training simulation was started because the mandatory admission criteria did not pass.
