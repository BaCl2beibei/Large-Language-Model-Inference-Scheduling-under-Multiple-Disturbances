# Mig-PPO design

## Scope and boundary

This phase implements sections 1–6 of the requested work: an audit, reusable
observation and mask code, hierarchical actor-critic, observed-window reward,
and a seeded multi-instance simulator/training loop. The live Manager scheduling
path is intentionally not wired to Mig-PPO in this phase. Existing policy names,
`GlobalScheduler.push_migrations()` return shape, RayRPC calls, and KV transfer
implementation are preserved.

## Observation

The shared observation builder accepts `InstanceInfo`-shaped objects. Each
instance row follows the ten requested load fields `(running, waiting, head
wait, KV usage, free KV ratio, preemptions, decode batch, last step latency,
batched tokens, remaining_steps)` plus SR-candidate KV ratio, stage indicators,
and candidate validity. It separately carries a missing mask. The candidate
request ID and exact candidate block count are metadata for validation and are
not used as an opaque model feature.

The global row contains instance count, a past-window arrival rate, running-count
variance, KV-use variance, total waiting requests, active migration count, and
time since the last migration. Arrival rate is computed from observed arrival
timestamps only. Per-pair features are source/destination waiting-count and KV
deltas, exact SR candidate KV ratio, predicted migration milliseconds, and a
simple current-state net-gain heuristic. The gain is a model feature, never a
counterfactual reward. The observation also carries predicted cost uncertainty
and a one-hot marker for pair-fitted, global-fallback, or uncalibrated-default
cost estimates, so the policy can distinguish estimates with different support.

`remaining_steps` is normalized as KV blocks per counted request; non-finite
values become finite and set a missing bit. It is never interpreted as tokens
or execution time. The normalizer applies capacity/SLO scales and fits mean/std
for time/computation features from training observations. Its state is stored in
the checkpoint and reused by evaluation.

For vLLM v0, `InstanceInfo` now carries an additive snapshot of the candidate
that current SR rules would select. It includes request ID, block-table size,
stage, validity, and timestamp. This snapshot can be stale by execution time;
any later live integration must reselect SR and repeat capacity checks just
before submission.

## Action law and safety

Action 0 means Stay; following entries are all ordered `(source, target)` pairs.
The actor emits a Bernoulli migration gate and a conditional masked categorical
pair policy. For Stay, log-probability is `log(1-p_mig)`. For pair `k`, it is
`log(p_mig) + log(q_k)`. Entropy is gate entropy plus migration probability
times conditional pair entropy. PPO rollout and update use these same joint
probabilities. Deterministic evaluation compares Stay's joint probability with
every legal pair's joint probability.

The mask is fail-closed and checks distinct IDs, fresh state/candidate,
candidate validity, healthy units, migration slots, in-flight status, neutral
type compatibility, destination acceptance, SR candidate KV blocks, waiting
KV demand, safety reserve, cooldown, and one-migration-at-a-time behavior. Stay
is always legal. If no pair is legal, the gate probability is forced to zero.

## Cost estimation and reward

The migration model fits `alpha + beta * blocks` from terminal migration CSV
events. It uses per-directed-pair coefficients after enough examples and a
global model otherwise. With no samples it returns explicitly marked
uncalibrated defaults and uncertainty. The observation carries predicted cost;
reward uses only actual simulated migration completion costs (and future live
integration must consume coordinator terminal records).

Reward components follow the requested throughput, waiting, TPOT, SLO, KV,
migration-cost, migration-count, and failed/aborted terms. Raw measurements,
normalized terms, weighted terms, timeout counts, and terminal unfinished
counts are emitted separately. Waiting pressure integrates queued request
seconds during the window. TTFT and TPOT enter when first-token and subsequent
token timestamps are observed; future latency is never read. At episode end,
unfinished requests are reported and counted in terminal SLO pressure.

## Simulation and calibration

The existing Llumnix simulator reuses the full vLLM queue and migration path,
but it does not expose a reset/step API or PPO rewards. `LlumnixMigrationEnv`
therefore provides a Gymnasium-style API without requiring Gymnasium. Its
discrete-event core schedules dynamic arrivals, per-instance queues, prefill
and batched decode service, KV allocation/growth, capacity stalls, approximate
KV-pressure preemption, completion, timeout, in-flight KV transfer, and
post-migration queue evolution. A transfer reserves destination blocks and
pauses the selected SR request until its terminal event. This is a calibrated
simulation model, not a substitute for real-GPU evaluation.

Service-time defaults are placeholders and every episode reports
`calibration_status: uncalibrated` unless a fit is supplied. `calibrate.py`
fits prefill/decode curves from Llumnix `*_instance.csv` profiling data and the
migration setup/bytes model from terminal migration CSV rows. Use data captured
on the target four-GPU setup. The fit does not establish that the simulator
predicts real workload outcomes; compare held-out runs before using it for
claims.

The calibration workload uses QPS `{9, 12, 15, 18, 21}` and a balanced cycle
through all nine combinations of prompt/response lengths `{256, 512, 1048}`.
It matches the target run's four RTX 3060 GPUs, four Qwen2-0.5B Llumnix
instances, 64 sequences per instance, 16-token KV blocks, eager-mode vLLM, and
2112-token model limit. `collect_ppo_calibration.py` holds seed 404 out and
fits only from seeds 101, 202, and 303. PPO training uses the CUDA-enabled
checkout environment.

## Training and evaluation outputs

`train_ppo.py` uses GAE and the clipped PPO objective, logs actor/critic loss,
entropy, KL, Stay rate, valid migrations, migration completion, queue pressure,
observed TTFT/TPOT, KV usage, and imbalance, and saves the best fixed-validation
checkpoint. `evaluate.py` runs fixed low/medium/high seeds using deterministic
actions and writes full per-request and migration episode traces, with a Stay
baseline by default. Reward weights are overrideable through YAML for ablation.

## Limitations to retain in reports

- The environment's service model and transfer jitter are only as good as the
  calibration inputs; default values are not RTX 3060 measurements.
- The simulator's per-instance capacity, dispatch skew, and preemption are
  abstractions of Llumnix/vLLM behavior, not an implementation of their GPU
  execution kernels.
- Observation of a real SR request does not guarantee it remains the request
  selected later. No live active migration is enabled by this phase.
- Shadow decisions cannot be assigned a counterfactual reward from ordinary
  service outcomes.
- A trained checkpoint is a candidate for subsequent evaluation, not evidence
  that Mig-PPO improves on Stay, Balanced, or Defrag.

