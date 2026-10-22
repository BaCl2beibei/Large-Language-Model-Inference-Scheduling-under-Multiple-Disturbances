# Interrupted initial PPO smoke attempt

Command:

```text
/venv/main/bin/python -m rl_scheduler.train_ppo --config benchmark/results/ppo_system_validation_round1_decode_cadence_20261021/ppo_smoke_config.yaml --output-dir benchmark/results/ppo_system_validation_round1_decode_cadence_20261021/ppo_smoke --timesteps 8 --seed 321 --calibration benchmark/results/ppo_system_validation_round1_decode_cadence_20261021/mig_ppo_calibration_system_validation.yaml
```

Exit code: 130. The config still had `training.total_episodes: 2000`; before the CLI fix that episode target took precedence, so training continued past the requested 8 steps. The process was interrupted at global step 4,888. Partial metrics are preserved in `training_metrics.jsonl`. This was a user-space cancellation, not a model or environment exception.

The CLI precedence was corrected and covered by unit tests. A subsequent 8-step CPU run using the same 2,000-episode config and a smoke-only 8-decision calibration completed at global step 8 and saved a checkpoint under `../ppo_smoke_cli_timesteps8/`.
