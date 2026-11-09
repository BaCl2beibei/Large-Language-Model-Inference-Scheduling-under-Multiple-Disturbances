#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
exec /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/.venv/bin/python -m rl_scheduler.full_completion_service --root /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/migration_opportunity_q3_512x1084_threshold100_pair_20261103/raw_runs/paired_1793708219348755167_0 --config /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/migration_opportunity_q3_512x1084_threshold100_pair_20261103/raw_runs/paired_1793708219348755167_0/runtime_config.yaml
