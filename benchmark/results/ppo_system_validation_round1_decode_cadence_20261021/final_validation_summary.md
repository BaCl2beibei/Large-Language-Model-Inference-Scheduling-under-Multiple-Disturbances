# Mig-PPO simulator final validation summary

Status: system validation and code checks completed; system accuracy acceptance **not met**; formal Mig-PPO training is **not recommended yet**.

## Dataset and evaluation scope

- Calibration data: `benchmark/results/ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505`; 20 Llumnix runs across QPS 9/12/15/18/21, seeds 101/202/303 for fitting and seed 404 held out.
- Four instances, 128 sequences per instance, max model length 2112. Collection is Migration ON only; it mixes 17 runs before and 3 after the migration-slot guard change.
- Two complete seed-404 replay sets were used: the unscaled corrected baseline and Round 1 with train-only decode cadence factor. Each reports all 5 QPS × V0–V3 × fixed-event / closed-loop modes.
- Comparison uses a 60s workload-arrival window and 90s outcome horizon because successful client completions were recorded as late as 80.32s. Queue/KV state comparison remains on the shared 60s window.
- Trace `output_tokens` values are budgets. Simulated realized lengths come from 10,347 successful seed 101/202/303 responses only; held-out output lengths were not used. Censored requests have no realized-length labels.

## Round 1 model comparison

The unscaled baseline uses the same corrected trace joins, successful-output length model, horizon, and censor classification. Round 1 adds a single decode token interval scale fitted as the median of per-run mean `server_tpot_ms / predicted D3 iteration time` across the 15 calibration runs (5.828; held-out seed excluded). It preserves raw `predicted_ms` for executor-iteration validation. Values below are mean absolute relative errors across five QPS.

| Replay mode | Model | Mean TTFT | P95 TTFT | Mean TPOT | P95 TPOT | HTTP 200 / 60s throughput |
|:--|:--|--:|--:|--:|--:|--:|
| Fixed event | Unscaled baseline V0 | 98.4% | 99.2% | 86.7% | 92.2% | 32.7% |
| Fixed event | Round 1 V0 | 78.4% | 58.3% | 28.4% | 20.3% | 40.3% |
| Fixed event | Round 1 V1 | 77.9% | 59.4% | 25.9% | 20.5% | 37.3% |
| Fixed event | Round 1 V2 | 77.9% | 59.4% | 25.9% | 20.5% | 37.3% |
| Fixed event | Round 1 V3 | 77.9% | 59.4% | 25.9% | 20.5% | 37.3% |
| Closed loop | Unscaled baseline V0 | 98.4% | 99.2% | 86.6% | 92.1% | 32.7% |
| Closed loop | Round 1 V0 | 77.6% | 58.3% | 28.6% | 19.6% | 40.6% |
| Closed loop | Round 1 V1 | 77.8% | 56.0% | 26.1% | 21.9% | 37.6% |
| Closed loop | Round 1 V2 | 77.9% | 58.5% | 26.3% | 20.2% | 37.6% |
| Closed loop | Round 1 V3 | 78.7% | 57.7% | 26.3% | 20.2% | 37.8% |

Cadence correction substantially reduces TPOT error and also improves TTFT MAPE relative to the unscaled baseline. It worsens completion-throughput MAPE by about 5–8 percentage points. Absolute TTFT error remains large, and Round 1 underpredicts completed throughput at every QPS. V1–V3 add little request-level improvement over V0 under this run; they remain valuable for their distinct service/KV/migration mechanisms, but this replay does not establish that they reproduce those dynamics accurately.

## Observed outcomes and Round 1 closed-loop V2

The following request counts are observed HTTP 200 outcomes and V2 simulated completions by the 90s horizon. Throughput rates divide these outcome counts by the 60s arrival window. All per-QPS details for each version and fixed replay are in `system_validation.md` and `system_validation.json`.

| QPS | Arrivals | Observed HTTP 200 | Drain-censored 499 | Sim V2 completed | Sim failed / unfinished | Observed / simulated throughput req/s | Mean queue per instance observed / sim | Mean KV fraction observed / sim |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 549 | 400 | 149 | 329 | 0 / 220 | 6.67 / 5.48 | 1.11 / 0.78 | 0.079 / 0.092 |
| 12 | 712 | 608 | 104 | 381 | 0 / 331 | 10.13 / 6.35 | 2.52 / 1.16 | 0.137 / 0.125 |
| 15 | 888 | 827 | 61 | 416 | 0 / 472 | 13.78 / 6.93 | 3.00 / 6.08 | 0.130 / 0.152 |
| 18 | 1069 | 579 | 490 | 431 | 0 / 638 | 9.65 / 7.18 | 1.76 / 18.78 | 0.133 / 0.164 |
| 21 | 1239 | 1056 | 183 | 447 | 0 / 792 | 17.60 / 7.45 | 12.96 / 34.63 | 0.159 / 0.173 |

There were no genuine observed HTTP failures or client timeouts; all 3,970 HTTP 499 responses in the 20-run dataset are drain-censored. Simulated failures are zero in these runs; unfinished counts are horizon-censored simulator requests.

### Fixed-event replay versus closed-loop greedy

- Fixed replay injects the logged seed-404 migration event times, pairs, blocks, costs and terminal statuses; it is a conditional state-replay check. It does not represent autonomous policy performance.
- Closed loop receives no logged route map or future event schedule. It selects legal actions using a greedy net-gain rule, not PPO. In Round 1 V2 it started 88/88/58/50/47 migration attempts at QPS 9/12/15/18/21 respectively, far more than the 3/2/4/8/7 logged events. The greedy baseline is therefore not a faithful proxy for production migration behavior.
- The full report lists migration success/failure/abort, duration mean/P95, fixed-event start delays, queue/KV P95 and both replay modes for every V0–V3 cell. Simulated blocking during migration remains an explicit approximation because the data contain no measured service-pause labels.

## Model error attribution

- **Prefill P1:** all-phase grouped CV MAE/RMSE 41.44/93.31ms; measurement-only grouped CV 45.87/102.38ms; seed-404 measurement-only 42.17/89.37ms, bias −0.75ms. Keep P1; the rare long stalls are not identifiable from available features.
- **Decode D3:** all-phase grouped CV MAE/RMSE 26.50/101.83ms; measurement-only grouped CV 30.43/82.99ms; seed-404 measurement-only 30.12/71.79ms, bias +4.73ms. Keep D3 for executor iteration prediction; do not conflate this label with request-level token cadence.
- **Decode cadence candidate:** factor 5.828 is train-only and median-robust; per-run factors range from 4.08 to 13.88, showing substantial run variation. This candidate improves request TPOT fit but trades away throughput accuracy; it is not accepted as a universal scale.
- **Migration M2:** grouped CV MAE/RMSE 342.29/698.54ms; seed-404 334.98/566.92ms (31 events), bias −64.96ms. Seed-404 FINISHED n=26 MAE 346.93ms/bias −112.91ms; ABORTED_DST n=5 MAE 272.82ms/bias +184.39ms. Pair data are sparse and block count weakly predicts cost.
- Complete server per-token timelines differ from client completion timestamps by mean 6.2–23.4s depending on QPS (subset coverage 53–74%); client completion includes additional delay. This timing gap and queue/state transition mismatch need resolution before tuning further against request-level labels.
- KV capacity 22,084 blocks per instance is inferred from 44,111 log samples (p05/median/p95 all 22,084), not independently measured hardware capacity.

See [model error attribution](model_error_attribution.md) for grouped residuals, per-QPS iteration distributions, and migration residuals; see [full system validation](system_validation.md) for all QPS/version/mode metrics and replay audits.

## Implemented and verified

- Request trace uses measured client send time when present; closed-loop validation receives no future migration routing/events.
- Trace output budgets are sampled through a deterministic, train-only realized-length model; raw service iteration predictions remain separate from optional effective decode token intervals.
- Drain-censored 499, genuine failures, timeouts and horizon unfinished requests are separated; per-instance queue and KV metrics are reported.
- Minimal CLI fix: explicit `--timesteps` now takes precedence over `training.total_episodes`; explicit `--episodes` remains higher priority. Three unit tests cover this precedence.

## Verification commands

| Command / check | Result |
|:--|:--|
| `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /venv/main/bin/python -m rl_scheduler.system_validation --data-root benchmark/results/ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505 --output-dir benchmark/results/ppo_system_validation_corrected_20261021_1930` | exit 0; unscaled baseline, all QPS/version/mode rows |
| `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /venv/main/bin/python -m rl_scheduler.system_validation --data-root benchmark/results/ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505 --output-dir benchmark/results/ppo_system_validation_round1_decode_cadence_20261021` | exit 0; Round 1 |
| `/venv/main/bin/python -m compileall -q rl_scheduler` | exit 0 |
| `/venv/main/bin/python -m pytest -q rl_scheduler` | exit 0; 32 passed; one existing pytest warning for unrecognized `asyncio_fixture_loop_scope` |
| `/venv/main/bin/python -m rl_scheduler.calibrate --instance-csv benchmark/results/ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505/calibration_inputs/service_instance.csv --migration-csv benchmark/results/ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505/calibration_inputs/migration_events.csv --output benchmark/results/ppo_system_validation_round1_decode_cadence_20261021/mig_ppo_calibration_cli_raw.yaml` | exit 0; raw CLI fit 7,311 prefill / 41,370 decode samples; retained as a baseline sanity artifact, not used in place of refined P1/D3 YAML |
| `/venv/main/bin/python -m rl_scheduler.calibrate --help` and `/venv/main/bin/python -m rl_scheduler.train_ppo --help` | both exit 0; actual CLI flags audited |
| `/venv/main/bin/python -m rl_scheduler.train_ppo --config benchmark/results/ppo_system_validation_round1_decode_cadence_20261021/ppo_smoke_config.yaml --output-dir benchmark/results/ppo_system_validation_round1_decode_cadence_20261021/ppo_smoke_cli_timesteps8 --timesteps 8 --seed 322 --calibration benchmark/results/ppo_system_validation_round1_decode_cadence_20261021/ppo_smoke_calibration.yaml` | exit 0; global_step 8, one PPO update and one evaluation; checkpoint saved at `ppo_smoke_cli_timesteps8/mig_ppo_best.pt` |
| Initial smoke command: `/venv/main/bin/python -m rl_scheduler.train_ppo --config benchmark/results/ppo_system_validation_round1_decode_cadence_20261021/ppo_smoke_config.yaml --output-dir benchmark/results/ppo_system_validation_round1_decode_cadence_20261021/ppo_smoke --timesteps 8 --seed 321 --calibration benchmark/results/ppo_system_validation_round1_decode_cadence_20261021/mig_ppo_calibration_system_validation.yaml` | exit 130; configured 2,000-episode target overrode `--timesteps 8`; interrupted at global_step 4,888, then fixed and covered by regression tests |

## Artifacts, limitations and decision

- Final Round 1 candidate YAML: `mig_ppo_calibration_system_validation.yaml` (train-only cadence scale 5.828); unscaled corrected baseline YAML is in `../ppo_system_validation_corrected_20261021_1930/mig_ppo_calibration_system_validation.yaml`. The standalone raw `calibrate` CLI output is `mig_ppo_calibration_cli_raw.yaml`.
- The 8-step smoke-only YAML and checkpoint are software-compatibility artifacts, not a trained or evaluated policy.
- No additional real GPU data collection was performed. Existing data are broad enough to identify the present mismatch, while ON-only migrations, no pause labels, incomplete route/scheduler state, version mixture and client/server timing gaps call for a targeted pilot after the timing semantics and queue transition are audited.
- **Decision:** keep P1/D3/M2 for their iteration and migration cost roles, treat the decode cadence scale as an experimental candidate, and defer formal Mig-PPO training. System-level accuracy does not meet the target: TTFT remains severely underpredicted, completed throughput is materially low, and queue dynamics diverge at QPS 15–21. Use a small targeted follow-up study (including Migration OFF/Stay and aligned server-side token/queue events) before another model round.
