# Reproduction

Run all commands from the repository root. They use new output directories and
retain the existing Round 2/3 results. The recorded collection used four RTX
3060 GPUs, QPS 3/9/12/15/18/21/24, 15 s warmup, a 60 s measurement window, a
30 s drain, and `max-num-seqs=128`.

## Collect calibration groups

Seed 606 OFF and ON were collected as paired groups; seed 909 was an additional
OFF calibration group. The two mode runs reuse saved arrival traces by QPS.

```bash
.venv/bin/python benchmark/run_ppo_batch_reuse.py \
  --qps 3 9 12 15 18 21 24 --calibration-seeds 606 --validation-seed 707 \
  --warmup-sec 15 --measure-sec 60 --drain-sec 30 --max-num-seqs 128 \
  --output-dir benchmark/results/llumnix_causal_hifi_20261022_r1/sampling_off \
  --mode off --calibration-only

.venv/bin/python benchmark/run_ppo_batch_reuse.py \
  --qps 3 9 12 15 18 21 24 --calibration-seeds 606 --validation-seed 707 \
  --warmup-sec 15 --measure-sec 60 --drain-sec 30 --max-num-seqs 128 \
  --output-dir benchmark/results/llumnix_causal_hifi_20261022_r1/sampling_on \
  --mode on --calibration-only

.venv/bin/python benchmark/run_ppo_batch_reuse.py \
  --qps 3 9 12 15 18 21 24 --calibration-seeds 909 --validation-seed 1001 \
  --warmup-sec 15 --measure-sec 60 --drain-sec 30 --max-num-seqs 128 \
  --output-dir benchmark/results/llumnix_causal_hifi_20261022_r1/calibration_seed909_off \
  --mode off --calibration-only
```

## Fit the selected models

```bash
.venv/bin/python -m rl_scheduler.causal_calibration \
  --data-dir benchmark/results/llumnix_causal_hifi_20261022_r1/sampling_off \
            benchmark/results/llumnix_causal_hifi_20261022_r1/sampling_on \
            benchmark/results/llumnix_causal_hifi_20261022_r1/calibration_seed909_off \
  --training-seed 606 --training-seed 909 \
  --output-dir benchmark/results/llumnix_causal_hifi_20261022_r1/models_seed606_909
```

## Collect the independent validation seed

Seed 1213 is not included in calibration. For a fresh reproduction, choose a
new unused seed and a new output directory; do not refit on that seed.

For the paired seed 1319 queue diagnostic, allow enough time for placement
group creation and check all four instance readiness calls concurrently. This
seed is diagnostic only and is not used for fitting or the acceptance gate:

```bash
LLUMNIX_WAIT_PLACEMENT_GROUP_TIMEOUT=60 \
LLUMNIX_INIT_INSTANCES_TIMEOUT=270 \
LLUMNIX_INSTANCE_READY_TIMEOUT=1200 \
LLUMNIX_SERVER_STARTUP_TIMEOUT=1200 \
.venv/bin/python benchmark/run_ppo_batch_reuse.py \
  --qps 18 --calibration-seeds 606 909 --validation-seed 1319 \
  --warmup-sec 15 --measure-sec 60 --drain-sec 30 --max-num-seqs 128 \
  --output-dir benchmark/results/llumnix_causal_hifi_20261022_r1/queue_stage_seed1319_off_pg60_rpc270_concurrent \
  --mode off --validation-only
```

```bash
.venv/bin/python benchmark/run_ppo_batch_reuse.py \
  --qps 3 9 12 15 18 21 24 --calibration-seeds 606 909 --validation-seed 1213 \
  --warmup-sec 15 --measure-sec 60 --drain-sec 30 --max-num-seqs 128 \
  --output-dir benchmark/results/llumnix_causal_hifi_20261022_r1/validation_seed1213_off \
  --mode off --validation-only

.venv/bin/python benchmark/run_ppo_batch_reuse.py \
  --qps 3 9 12 15 18 21 24 --calibration-seeds 606 909 --validation-seed 1213 \
  --warmup-sec 15 --measure-sec 60 --drain-sec 30 --max-num-seqs 128 \
  --output-dir benchmark/results/llumnix_causal_hifi_20261022_r1/validation_seed1213_on \
  --mode on --validation-only
```

## Score and audit

```bash
.venv/bin/python -m rl_scheduler.causal_system_validation \
  --off-dir benchmark/results/llumnix_causal_hifi_20261022_r1/validation_seed1213_off \
  --on-dir benchmark/results/llumnix_causal_hifi_20261022_r1/validation_seed1213_on \
  --model-dir benchmark/results/llumnix_causal_hifi_20261022_r1/models_seed606_909 \
  --output-dir benchmark/results/llumnix_causal_hifi_20261022_r1/validation_seed1213 \
  --round2-config benchmark/results/ppo_system_validation_round2_causal_repairs_20261022/mig_ppo_calibration_system_validation.yaml \
  --validation-seed 1213 --warmup-sec 15 --measure-sec 60 --drain-sec 30

```

The collection and validation summaries include event sequence, request
terminal, and batch-join audits. An optional standalone audit can be run on one
experiment root with
`.venv/bin/python -m rl_scheduler.causal_event_audit --data-dir <experiment-root>`;
it writes derived audit files into that root.

Completed event streams are losslessly gzipped by the collection pipeline. The
archiver verifies the decoded byte count and SHA-256 before deleting the
uncompressed duplicate; each run records an archive manifest.

## Code checks

```bash
python3 -m py_compile rl_scheduler/causal_system_validation.py \
  rl_scheduler/causal_event_io.py benchmark/run_ppo_batch_reuse.py
/venv/main/bin/python -m pytest -q rl_scheduler/tests/test_causal_models.py
git diff --check
```

The full repository test suite additionally requires Ray in the test runner;
the available `/venv/main` environment does not include it. Formal PPO training
must remain disabled until a fresh independent run passes every system-level
acceptance threshold.
