#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
exec /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/.venv/bin/python -m rl_scheduler.full_completion_service --root /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/mig_ppo_v2_hot1h_q1p25_20261106T085546Z --config /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/mig_ppo_v2_hot1h_q1p25_20261106T085546Z/runtime_config.yaml
