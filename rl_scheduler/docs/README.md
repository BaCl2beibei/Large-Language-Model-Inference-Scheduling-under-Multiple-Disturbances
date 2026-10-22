# Mig-PPO research module

This module implements the first development phase (audit, state/action design,
policy, reward, and simulator/training environment). It is independent of the
live Llumnix migration decision path.

## Install

Install the research extras into the checkout environment so training uses the
same CUDA-enabled PyTorch as the Llumnix benchmark:

```bash
.venv/bin/python -m pip install -r requirements-rl.txt
```

## Training workload

[`configs/train.yaml`](../configs/train.yaml) matches the real collection run:
four Llumnix instances on four RTX 3060 GPUs, Qwen2-0.5B, one worker per GPU,
`max_num_seqs=128` per instance (up to 512 aggregate sequence slots), 16-token
blocks, `max_model_len=2112`, and nine equally
weighted prompt/response combinations from lengths 256, 512, and 1048 tokens.
Both real traces and simulator episodes use Poisson QPS levels 9, 12, 15, 18,
and 21. Llumnix vLLM v0/v1 entrypoints force eager mode even if a caller omits
the flag, and the benchmark launcher also passes `--enforce-eager`; CUDA graph
capture is disabled for these runs. The simulator uses the same nine-profile
cycle. PPO runs on `cuda:0` in the checkout's `.venv`.

The supplied `editable_environment_config.yaml` is an exported reference and
the benchmark runner does not load it. Mig-PPO reads its training workload from
`rl_scheduler/configs/train.yaml`.

## Collect and calibrate target-cluster logs

Run the collection script once. It performs five QPS levels × four seeds; seeds
101/202/303 are used for calibration and seed 404 is held out for validation.
Each run uses a 30-second warmup and a 200-second measurement window.
Migration OFF is not run. One eager Llumnix server with four instances stays up
for the full batch, avoiding 19 repeated Ray and model startups. The startup gate
waits for four workers to be ready and registered in the Manager, with one worker
assigned to each GPU; no requests start until it passes. After each warmup and
measurement drain window, unfinished requests are explicitly aborted by request
ID, grouped into one abort RPC per worker, and the server waits for their API
handlers to exit before moving to the next window. The abort barrier allows up
to 15 minutes for a heavily loaded phase to clear. Each consecutive block of nine generated requests covers all nine
prompt/output length combinations once. GPU and instance telemetry continue
during the experiment.

```bash
.venv/bin/python benchmark/collect_ppo_calibration.py
```

If collection exits before all windows finish, resume the same output directory
to keep completed windows and retry only incomplete ones:

```bash
.venv/bin/python benchmark/collect_ppo_calibration.py \
  --output-dir benchmark/results/ppo_calibration_mixed_2112_eager_<run> \
  --resume
```

Results are written to a new `benchmark/results/ppo_calibration_mixed_2112_eager_*`
directory. The collection manifest lists the combined CSVs. Only calibration
seeds feed the fit; held-out seed 404 stays in `validation_inputs/`.

The client drain window defaults to 15 seconds after each phase. Requests that
remain unfinished are retained as censored rows, then cancelled before the next
phase so they cannot leak into another QPS/seed window. Instance profiling rows
and terminal migration records are the fit inputs.

Fit the service and transfer curves from the combined calibration CSVs:

```bash
.venv/bin/python -m rl_scheduler.calibrate \
  --instance-csv benchmark/results/ppo_calibration_mixed_2112_eager_<run>/calibration_inputs/service_instance.csv \
  --migration-csv benchmark/results/ppo_calibration_mixed_2112_eager_<run>/calibration_inputs/migration_events.csv \
  --output rl_scheduler/configs/mig_ppo_calibration.yaml
```

The calibration YAML can be passed to training. Fit samples should come from
the same model, engine settings, GPU type, and transfer backend planned for
evaluation. The collection script runs eager-mode Migration ON only at QPS
9/12/15/18/21 with 128 sequences per instance, seeds 101/202/303 for calibration and independent seed 404
for validation, using 30 seconds of warmup and 200 seconds of measurement per
run. The batch waits until four Llumnix workers are ready, the Manager has
registered all four, and there is exactly one worker on each GPU before it
sends any workload. The service profile is therefore fitted from Migration ON
instance rows; migration cost is fitted from the corresponding terminal
migration events. The held-out seed is kept out of the calibration CSVs.

These Llumnix traces should calibrate and validate the simulator. Standard PPO
updates need fresh rollouts from the current policy interacting with an
environment, because a fixed trace from Balanced/Defrag/Stay does not contain
the counterfactual queue and completion outcomes for Mig-PPO's other actions.
The calibrated simulator supplies those policy-dependent rollouts. Hold out
separate Llumnix seeds/runs for final comparisons against Stay, Balanced, and
Defrag.

## Train and evaluate

```bash
.venv/bin/python -m rl_scheduler.train_ppo \
  --config rl_scheduler/configs/train.yaml \
  --calibration rl_scheduler/configs/mig_ppo_calibration.yaml \
  --episodes 2000 \
  --output-dir rl_scheduler/runs/mig_ppo

.venv/bin/python -m rl_scheduler.evaluate \
  --checkpoint rl_scheduler/runs/mig_ppo/mig_ppo_best.pt \
  --scenarios qps_9 qps_12 qps_15 qps_18 qps_21 \
  --seeds 404 --device cuda:0 \
  --output rl_scheduler/runs/mig_ppo/evaluation.json
```

The default environment values are explicitly marked uncalibrated. Use them to
exercise the code only; do not present resulting returns or latency as target
GPU performance. The evaluator includes an always-Stay baseline and stores
episode request/migration traces plus the calibration marker.

## Interface notes

`LlumnixMigrationEnv.reset(seed, options)` returns `(observation, info)` and
`step(action)` returns the Gymnasium five-tuple. Observation arrays include
instance/global/pair features and a joint action mask; action zero is Stay.
The four-instance action space contains 12 directed pairs plus Stay. The
policy's PPO log-probability and entropy use the complete hierarchical action
distribution.

This phase preserves all existing serving policies and migration APIs. Live
shadow/active integration, scheduler cadence/cooldown configuration, and real
coordinator telemetry are later work; see
[`architecture_audit.md`](architecture_audit.md) and [`design.md`](design.md).

