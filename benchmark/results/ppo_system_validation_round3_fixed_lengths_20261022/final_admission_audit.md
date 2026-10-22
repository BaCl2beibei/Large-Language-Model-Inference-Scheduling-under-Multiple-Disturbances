# Final Mig-PPO admission audit

## Decision

**不适合正式 PPO 训练** — the simulator does not meet the formal-training admission criteria. The three-round limit has been reached. Preserve the current best validated code and artifacts; stop tuning and do not start formal training.

The lifecycle and accounting repairs pass the recorded deterministic checks. System-level predictive accuracy does not pass, and the independent seed 505 confirmation could not be completed with matched realized output lengths. The configured 2,000-episode run and post-training policy evaluation were not started, as required by the admission gate.

## Admission results

| Gate | Evidence | Result |
|---|---|---|
| Lifecycle, queue ownership, batch time, KV accounting, and migration reservations | Directed regression tests and invariants summarized in `../ppo_system_validation_round2_causal_repairs_20261022/causal_audit.md`; full test command recorded below | Pass for implemented checks |
| Grouped service-model validation | P1 measurement-only grouped CV MAE/RMSE 45.87/102.38 ms; D3 30.43/82.99 ms. Seed 404 P1 42.17/89.37 ms and D3 30.12/71.79 ms. M2 seed 404 has 31 events, MAE/RMSE 334.98/566.92 ms; no measured migration Pause Time label | Recorded; migration pause remains uncalibrated |
| Five-QPS V0–V3 system replay, fixed events and closed loop | 40 result cells across QPS 9/12/15/18/21, in `../ppo_system_validation_round2_causal_repairs_20261022/system_validation.md` and `.json` | Completed on seed 404 for diagnosis; seed 404 is not a new independent final holdout |
| Closed-loop V3 aggregate error | Throughput MAPE 38.13% (limit 20%); mean TTFT 72.83% (25%); P95 TTFT 56.00% (30%); mean TPOT 21.48% (20%); P95 TPOT 25.60% (25%) | Fail |
| Queue error | Mean MAE 7.199 and P95 MAE 31.543 requests/instance; both exceed their observed-value thresholds | Fail |
| KV error | Mean absolute error 1.77 percentage points (limit 5); P95 3.66 pp (limit 10) | Pass |
| Per-QPS main metrics | The worst closed-loop relative errors reach 57.86% throughput, 92.76% mean TTFT, 95.29% P95 TTFT, 32.72% mean TPOT, and 48.14% P95 TPOT | Fail; >40% errors lack a validated explanation |
| Independent matched-length confirmation | Seed 505 QPS 15 OFF probe applied `min_tokens=max_tokens` to all 942 requests, but 147/775 HTTP 200 outputs (19.0%) missed target; mean absolute length error 15.16 tokens. ON and the other QPS levels were not run | Incomplete; cannot serve as final confirmation |
| Reproducible code checks | `python -m compileall -q rl_scheduler` exit 0; `python -m pytest -q rl_scheduler` exit 0, 37 passed, one existing pytest configuration warning | Pass |
| Bounded PPO smoke | CPU, seed 322, 8 timesteps; one episode, one update, one evaluation, best/final checkpoints saved; exit 0 | Pass; smoke only |
| Formal PPO and post-training evaluation | `rl_scheduler/configs/train.yaml` specifies 2,000 episodes; only smoke checkpoints exist in the validation result directories | Not run because admission failed |

## Functional evidence coverage

The deterministic suite includes directed checks for trace/Poisson reset determinism, mixed-batch single completion timing, preemption without a generated token, destination sequence/KV reservation, timeout cleanup during migration, fixed-grid queue/KV sampling, and realized-output trace bounds. Representative tests are `test_reset_from_trace_to_poisson_preserves_seeded_arrival_sequence`, `test_mixed_gpu_batch_emits_prefill_and_decode_at_one_batch_completion`, `test_preempted_request_repeats_context_prefill_without_emitting_a_token`, `test_incoming_migration_reserves_destination_sequence_slot_and_kv_once`, `test_timeout_during_migration_releases_source_then_target_reservation_once`, and `test_observed_queue_and_kv_are_sampled_on_simulator_fixed_grid`.

The no-future-information check is a structural code-path audit: `LlumnixMigrationEnv._observation()` builds the current observation, the PPO rollout passes normalized observation tensors to the policy, and the policy accepts those tensors and a current legal-action mask. Training does not load fixed-event logs or route maps. In `system_validation.py`, logged routes and migration events are supplied to fixed-event replay; the separate closed-loop runner receives neither. This supports the input boundary, but no trained PPO checkpoint was evaluated because admission failed.

The candidate YAML and scale 1 fallback YAML retain configurable P1/D3 models, migration cost settings, and decode cadence. The fallback therefore remains available; migration service-pause behavior remains an explicit uncalibrated approximation because no pause-time labels exist.

## Interpretation and limitations

The scale 5.827594 candidate remains experimental. The paired seed 404 trace comparison showed substantially slower completion under the candidate than scale 1; the scale 1 YAML is retained as an explicit fallback. P1/D3 labels are executor iteration wall times, not request TTFT/TPOT or isolated GPU time. The 22,084 KV blocks per instance are supported by runtime/log evidence but are not an independent hardware measurement. Migration wall time is available, but measured service-pause truth is not.

Fixed-event replay and closed-loop greedy replay have different control inputs and remain separately reported. Closed-loop results are not PPO results. Seed 404 was used for diagnosis, and the seed 505 probe failed the matched-output-length condition, so no fresh independent five-QPS confirmation is claimed. The Round 2 causal audit records its then-current recommendation to stop before another model-tuning round; the later seed 505 fixed-length investigation was documented as Round 3 and is the terminal decision here.

## Authoritative artifacts

- Causal-chain audit, gate analysis, and cadence comparison: `../ppo_system_validation_round2_causal_repairs_20261022/causal_audit.md`
- Full five-QPS report and machine-readable results: `../ppo_system_validation_round2_causal_repairs_20261022/system_validation.md` and `system_validation.json`
- Service and migration error attribution: `../ppo_system_validation_round2_causal_repairs_20261022/model_error_attribution.md`
- Candidate and scale 1 fallback configurations: `../ppo_system_validation_round2_causal_repairs_20261022/mig_ppo_calibration_system_validation.yaml` and `mig_ppo_calibration_scale1_fallback.yaml`
- CPU smoke command, exit status, and checkpoints: `../ppo_system_validation_round2_causal_repairs_20261022/ppo_smoke_run_metadata.json`
- Round 3 matched-length probe decision and per-request mismatch evidence: `round3_decision.md`, `qps_15/off/output_length_enforcement_audit_summary.json`, and `qps_15/off/output_length_enforcement_audit.json`

## Verification commands

```text
python -m compileall -q rl_scheduler    exit 0
python -m pytest -q rl_scheduler        exit 0 (37 passed; one existing config warning)
```

No further real runs were started after the Round 3 stop decision. The last recorded GPU check showed the device idle. Existing user changes and result files remain preserved.

## Targeted real-run accounting

Five targeted real runs were attempted against the objective's 12-run cap: two in the initial seed 505 Migration OFF/ON pilot, two in the later seed-controlled OFF/ON pair, and the single failed matched-length OFF probe documented here. The initial pilot is summarized in `../ppo_migration_pilot_qps15_seed505_20261021/paired_pilot_analysis.md`; the seed-controlled pair remains under `../ppo_system_validation_round3_seed505_20261022/qps_15/`. The final probe stopped before its ON run and the remaining four QPS levels.
