# PPO entrypoints

Use these two commands as the public entrypoints for the current Q15-B workflow:

| Purpose | Entrypoint | What it does |
|---|---|---|
| Simulator and CPU checks | `benchmark/run_ppo_sim.py` | Calibrates the frozen Q15-B simulator, runs the P3 simulator checks, and checks the synthetic PPO update contract. It does not start a GPU service. |
| Real Q15-B run | `benchmark/run_ppo_real.py` | Starts the bounded, audited real collection and PPO update after simulator preparation passes. |

Create and prepare a fresh run directory:

First copy `rl_scheduler/configs/mig_ppo_v2_reward.json` to an experiment-specific
JSON file and replace `hotspot_threshold: null` with a threshold measured from
the real KV occupancy/congestion relationship. The template deliberately has
no assumed 80% threshold. All reference scales and weights are frozen into
the new run's `manifest.json` and shared by simulation and real collection.

```bash
.venv/bin/python benchmark/run_ppo_sim.py prepare \
  --run-dir benchmark/results/mig_ppo_v2_<UTC> \
  --reward-config /path/to/calibrated_reward.json
```

V2 defaults to trace seeds `161101,161102` and policy seed `161001`. Use
`--trace-seeds SEED1 SEED2 --policy-seed SEED3` for subsequent fresh runs; the
existing seed reuse check remains active.

Review the simulator and CPU contract report:

```bash
.venv/bin/python benchmark/run_ppo_sim.py report --run-dir benchmark/results/mig_ppo_v2_<UTC>
```

Only after preparation passes, launch the bounded real run:

```bash
.venv/bin/python benchmark/run_ppo_real.py launch --run-dir benchmark/results/mig_ppo_v2_<UTC>
```

Read the real-run status or verify the delivered checkpoints:

```bash
.venv/bin/python benchmark/run_ppo_real.py report --run-dir benchmark/results/mig_ppo_v2_<UTC>
.venv/bin/python benchmark/run_ppo_real.py verify --run-dir benchmark/results/mig_ppo_v2_<UTC>
```

`rl_scheduler/cohort_ppo.py` contains the shared PPO policy/update implementation. The existing P3 preparation path validates/calibrates the simulator and performs a synthetic contract update. P4 multi-update offline training is now implemented by the `local-*` subcommands below. Do not treat `checkpoints/simulation_only.pt` as a real-run checkpoint; the accepted real pipeline starts from `checkpoints/initial.pt` and writes its real update to `checkpoints/real_updated.pt`.

## Native Windows and Linux CPU training

Build a standalone CPU ZIP from the latest frozen multi-environment models:

```bash
.venv/bin/python benchmark/run_ppo_sim.py portable-export --output-dir benchmark/results/portable_cpu_<UTC>
```

Extract the ZIP and follow `README_WINDOWS.md`. Windows uses native Python
and PowerShell; WSL is unnecessary. Linux uses the same source and CLI:

```bash
python benchmark/run_ppo_sim.py local-doctor --bundle-root .
python benchmark/run_ppo_sim.py local-smoke --bundle-root . --run-dir runs/smoke_001
python benchmark/run_ppo_sim.py local-train --bundle-root . --run-dir runs/local_001
python benchmark/run_ppo_sim.py local-train --bundle-root . --run-dir runs/local_001 --resume
python benchmark/run_ppo_sim.py local-evaluate --bundle-root . --run-dir runs/local_001 --split heldout
python benchmark/run_ppo_sim.py local-export-policy --bundle-root . --run-dir runs/local_001 --output-dir exports/local_001
```

Each episode samples one of thirteen environments uniformly with replacement;
four episodes use the same policy version before each PPO update. Episodes
end within 200 simulated seconds; unfinished requests explicitly time out and
remain in the reward and GAE. This finite-horizon behavior is enabled only for
local development, preserving historical full-cohort fitting defaults.
Local checkpoints include optimizer and RNG state and are portable between
Windows and Linux with the same bundle and compatible dependencies; recreate
the virtual environment on each system. Native Windows execution still
requires doctor/smoke on the user's machine. Models remain development-only
with `formal_ppo_ready=false` and independent real validation pending. See
[the P4 operating plan](p4_local_cpu_migration_prompt.md) for manual steps.

`benchmark/run_p2p3_q15b.py` is now the internal implementation module. The retired Mig-PPO and full-completion PPO trainers, the old Qwen2-0.5B calibration collector/evaluator, their trainer-specific tests and preparation files, and the 0.5B run artifacts have been removed. The old full-cohort result remains summarized in `research_implementation_plan_v2.md` so cleanup does not erase the research conclusion.

## Mig-PPO V2 state and reward

`rl_scheduler/real_observation.py` supplies exactly these raw policy inputs:

| Group | Features, in order |
|---|---|
| Each of four instances | `running`, `waiting`, `head_wait_sec`, `used_fraction`, `candidate_blocks`, `last_step_ms`, `tokens_per_sec_5s`, `waiting_slope_5s`, `used_fraction_slope_5s` |
| Global | `remaining_arrival_sec`, `remaining_total_sec`, `outstanding`, `overdue_first`, `output_stalled`, `client_tokens_per_sec_5s`, `busy`, `cooldown_remaining_sec` |
| Each of twelve directed pairs | `waiting_difference`, `remaining_kv_blocks_difference` |

Both pair differences are **source minus destination**. Remaining KV blocks
are `total_blocks - used_blocks - reserved_blocks_not_in_used`; waiting demand
does not count as allocated KV. Effective `used_fraction` uses the same
reservation rule. vLLM's allocator already includes migration preallocation
in `used_blocks`, so the real adapter reports independent extra reservations
as zero. Raw inputs comprise `4×9 + 8 + 12×2 = 68` values; normalization appends
missing-value bits, producing instance/global/pair dimensions `18/16/4`.
The action space remains Stay plus twelve directed migrations, with the
existing legality mask and one migration at a time. PPO has **zero migration
cooldown** in the observation mask, simulator, and real Manager arbiter.
The `cooldown_remaining_sec` input is retained for schema compatibility and
is always zero for PPO. Busy migrations and other feasibility checks can
still mask all twelve migrations.

PPO samples every **1.5 seconds** in both simulation and real collection.
Real collection skips missed ticks after a control overrun; saved actual
timestamps determine `dt`, discounting, and GAE. The shared constants live in
`rl_scheduler/ppo_runtime.py` and are frozen in `manifest.settings`.

The default PPO update mode is `choice_only`:

| Decision | Actor loss / normalization / entropy / KL / clipping | Critic | GAE trajectory |
|---|---|---|---|
| At least one legal migration; policy chooses Stay | Included | Included | Retained |
| All migrations masked; forced Stay | Excluded | Included | Retained |

No forced Stay window is removed or merged before GAE. A batch containing
only forced Stay decisions updates the critic without an actor loss.

Remaining arrival time uses the frozen arrival horizon. Remaining total time
uses the frozen episode safety horizon (3,600 seconds for the existing real
preparation; 200 seconds for portable local training), rather than the last
arrival time or the eventual completion time. Earlier real service or batch budget
aborts remain incomplete evidence. TTFT overdue and output stalled counts use
2 seconds and 0.15 seconds, respectively.

`MigPPOV2Reward` in `rl_scheduler/cohort_reward.py` implements:

```text
r = -dt * (.40 D_mean + .30 Q_mean + .20 H_mean + .10 B_star_mean)
    - xi * failure_count / failure_ref
    + .10 * (gamma_t * Phi_next - Phi)
    - invalid_migration_cost * invalid_migration_count
Phi = -(.45 Q + .35 H + .20 B_star)
B_star = H * 2 * std(KV utilization across four instances)
gamma_t = .99 ** dt_seconds
lambda_t = .97 ** dt_seconds
```

`Q` combines clipped mean and maximum queue pressure with equal weights.
`H` clips `(max_utilization - hotspot_threshold)/(1-hotspot_threshold)` to
`[0,1]`. `D` divides the sum of capped first-token waiting ages plus `kappa=.5`
times capped decode silence ages by fixed `n_ref`. Ages use 2-second and
0.15-second references, and each normalized age is capped at 2.

Time pressure is integrated analytically between actual client output receipt
timestamps, including the waiting interval before the first token. A frame
containing several new tokens gives them the same measured client receipt
time; no token spacing is inferred inside that frame. Repeated final frames
with zero new tokens do not reset the silence timer. Aggregate TPOT is never
substituted for the timeline. Request summaries now preserve the entire
timeline rather than only the online five-second buffer.

Queue, hotspot and gated balance integrals use a zero-order hold of each saved
decision snapshot over its actual window; they approximate continuous cluster
pressure at the control sampling cadence. Every window records `D_integral`,
pressures, four reward components, failure counts, migration cost, actual `dt`,
`gamma_t` and `lambda_t` in `reward_audit`. Only the true audited episode terminal
sets the next potential and value to zero. GAE uses the same per-window discount
as shaping, including forced Stay intervals.

Defaults for the otherwise unspecified scales are `n_ref=q_ref=q_max_ref=64`,
`failure_ref=64`, `xi=1`, and a `0.01` cost per rejected or failed migration.
These are experiment starting parameters; they do not depend on the current
request count. There is no fixed successful-migration reward. Audited service
errors, timeouts and confirmed cancellations remain in V2 complete cohorts
and incur the failure penalty. Unknown outcomes, missing terminal evidence,
or missing token timelines cannot train.

The raw observation schema is `real-observation-v2`, and the policy/reward
schema is `mig-ppo-v2`. Old run directories and checkpoints require a fresh
preparation and policy initialization. `CohortReward` and undiscounted returns
remain solely for historical full-completion audits and their older trainer;
both canonical entrypoints use the V2 reward.

## Multi-environment physical refit (CPU only)

Refit the audited November 5 multi-environment and November 6 followup ledgers:

```bash
.venv/bin/python benchmark/run_ppo_sim.py physical-refit \
  --run-dir benchmark/results/mig_ppo_v2_refit_<UTC> --workers 4
```

The command reads existing real data, compares small physical models, and runs
Stay/Balanced closed loops plus observed-action replays with predicted and
observed migration durations. Native closed loops recompute dispatch from
simulated state; replay alone fixes measured initial routes. It starts no GPU
service or PPO training. `--stage audit|fit|validate|report` supports individual
CPU stages; frozen fits and original source directories cannot be overwritten.

The shared physical V2 model separates executor time, effective engine progress,
and client output flushes (first token, then cumulative 64-token boundaries).
KV allocation follows generated tokens between client receipts. Migration
transfer and pause are fitted from their own logged events, with measured
GPU-direction corrections for total duration when supported. Final costs,
policy labels and environment IDs are not predictor inputs.

Results include `physical_model.json`, directly loadable per-environment models
under `frozen_environments/`, `model_registry.json`, `acceptance_detail.csv`,
and `report.md`. The currently collected traces provide fit and development
validation only: `independent_validation_pending` remains true. Native stays
test-only, and the registry does not authorize formal PPO training.
