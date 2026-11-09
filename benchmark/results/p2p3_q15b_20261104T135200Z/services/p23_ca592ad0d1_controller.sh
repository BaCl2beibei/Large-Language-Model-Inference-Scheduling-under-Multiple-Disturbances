#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream
exec /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/.venv/bin/python -u /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/run_p2p3_q15b.py run --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/p2p3_q15b_20261104T135200Z >> /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/p2p3_q15b_20261104T135200Z/controller.log 2>&1
