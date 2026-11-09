# PPO workflow

The old Qwen2-0.5B calibration, training, and evaluation instructions were
retired with their entrypoints. The current Qwen2-1.5B Q15-B workflow is
documented in [`ppo_entrypoints.md`](ppo_entrypoints.md).

The shared policy, rollout, PPO update, and checkpoint code is in
[`../cohort_ppo.py`](../cohort_ppo.py). The Q15-B simulator still reuses the
discrete-event environment in `../env.py`; the deleted legacy actor and trainer
are not part of the current path.
