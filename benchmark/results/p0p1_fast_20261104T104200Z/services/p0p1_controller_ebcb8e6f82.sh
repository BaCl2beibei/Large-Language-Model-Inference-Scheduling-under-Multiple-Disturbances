#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
exec /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/.venv/bin/python -u /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/run_p0p1_fast.py run --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/p0p1_fast_20261104T104200Z >> /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/p0p1_fast_20261104T104200Z/runner.log 2>&1
