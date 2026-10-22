# Incomplete smoke attempt

Command used `--timesteps 8`, but the smoke YAML retained a 512-step PPO rollout. The trainer rounds the timestep budget up to a full rollout, so this attempt was interrupted with exit code 130 during evaluation before it wrote metrics or checkpoints.

This was a smoke-configuration error. The corrected CPU smoke uses an 8-step rollout and a 2-step normalizer and is stored in `../ppo_smoke_cpu8_corrected/`.
