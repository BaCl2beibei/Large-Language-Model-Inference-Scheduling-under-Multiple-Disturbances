#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
exec /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/.venv/bin/python -u /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/run_p0p1_q15b_pair_replay.py --source-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/p0p1_fast_20261104T104200Z --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/p0p1_fast_replay_20261104T124801Z run >> /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/p0p1_fast_replay_20261104T124801Z/runner.log 2>&1
