# HOT1-H / NATIVE follow-up

Run `mig_ppo_v2_followup_20261106T080500Z`. The serving stack, Qwen2-1.5B model, four RTX 3060 workers, request sizes (128 input / 1536 output tokens), 200 s episode horizon, 1.5 s decision interval, 0 s migration cooldown, SR migration and Balanced threshold rule match the prior run.

HOT1-H uses N=48, QPS=1.0 and the same base trace seed/prompt seed as the selected HOT2-H/HOT3-H/HOT4-H QPS1 backup candidates; only the 90% hotspot route changes to logical instance 1.

NATIVE tries N=48 at QPS 1.25, 1.5, 2.0 in ascending order. Each tier is paired on its own trace. Stop this sweep after the first complete positive cost gain with at least one audited successful Balanced migration. The adoption rule is the frozen main cost C=mean(E2E)+CVaR95(E2E). No independent real validation, model fitting, PPO training or PPO update is in scope.
