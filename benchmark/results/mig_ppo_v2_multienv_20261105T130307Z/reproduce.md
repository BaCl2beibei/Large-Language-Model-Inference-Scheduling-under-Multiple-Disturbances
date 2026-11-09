# 复现命令

```bash
python benchmark/run_ppo_real.py multienv-plan --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/mig_ppo_v2_multienv_20261105T130307Z
python benchmark/run_ppo_real.py multienv-collect --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/mig_ppo_v2_multienv_20261105T130307Z --stage selection
python benchmark/run_ppo_sim.py multienv-fit --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/mig_ppo_v2_multienv_20261105T130307Z
python benchmark/run_ppo_real.py multienv-collect --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/mig_ppo_v2_multienv_20261105T130307Z --stage validation
python benchmark/run_ppo_sim.py multienv-validate --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/mig_ppo_v2_multienv_20261105T130307Z
python benchmark/run_ppo_sim.py multienv-report --run-dir /workspace/Large-Language-Model-Inference-Scheduling-under-Multiple-Disturbances-upstream/benchmark/results/mig_ppo_v2_multienv_20261105T130307Z
```
