# Incomplete smoke attempt

The rollout was correctly limited to 8 steps, but the full calibration YAML overrode the smoke config's `max_decisions=8` with its formal value of 4,000. Its evaluation therefore began 15 full-length episodes. This attempt was interrupted with exit code 130 before metrics or checkpoints were written.

The bounded smoke calibration keeps the final fitted models and cadence value while setting only the smoke horizon to 8 decisions; it is used in `ppo_smoke_cpu8_bounded/`.
