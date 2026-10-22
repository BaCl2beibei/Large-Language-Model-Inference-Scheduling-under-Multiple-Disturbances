"""Train Mig-PPO with clipped PPO, hierarchical action log-probabilities and GAE."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rl_scheduler.cost_estimator import MigPPOMigrationCostEstimator
from rl_scheduler.env import LlumnixMigrationEnv, MigPPOEnvConfig
from rl_scheduler.normalization import MigPPONormalizer
from rl_scheduler.policy import MigPPOActorCritic
from rl_scheduler.reward import MigPPORewardCalculator, MigPPORewardScales, MigPPORewardWeights


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "train.yaml"


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _resolve_episode_target(episodes: int | None, total_timesteps: int | None,
                           training: dict[str, Any]) -> int | None:
    if episodes is not None:
        return int(episodes)
    # An explicit timestep limit is commonly used for smoke runs and takes
    # precedence over a longer episode count saved in the training config.
    if total_timesteps is not None:
        return None
    configured = training.get("total_episodes")
    return int(configured) if configured is not None else None


def _tensors(normalized, device: torch.device):
    return (
        torch.as_tensor(normalized.instance_features, dtype=torch.float32, device=device),
        torch.as_tensor(normalized.global_features, dtype=torch.float32, device=device),
        torch.as_tensor(normalized.pair_features, dtype=torch.float32, device=device),
        torch.as_tensor(normalized.pair_mask, dtype=torch.bool, device=device),
    )


def _checkpoint_payload(policy, normalizer, env_config, cost_estimator, ppo_config,
                        step, eval_result, episodes_completed=0):
    return {
        "algorithm": "Mig-PPO",
        "policy_state_dict": policy.state_dict(),
        "model_spec": {
            "instance_feature_dim": policy.instance_encoder[0].in_features,
            "global_feature_dim": policy.global_encoder[0].in_features,
            "pair_feature_dim": policy.pair_head[0].in_features - policy.instance_encoder[-2].out_features * 3,
            "num_instances": policy.num_instances,
            "hidden_size": policy.instance_encoder[0].out_features,
        },
        "normalizer": normalizer.state_dict(),
        "cost_estimator": cost_estimator.state_dict(),
        "environment_config": asdict(env_config),
        "ppo_config": ppo_config,
        "training_step": int(step),
        "training_episodes": int(episodes_completed),
        "validation": eval_result,
    }


def _evaluate(policy, normalizer, env_config, cost_estimator, device,
              scenarios=None, seeds=(9001, 9002, 9003)) -> dict[str, Any]:
    if scenarios is None:
        scenarios = tuple(f"qps_{float(qps):g}" for qps in env_config.target_qps_levels) \
            or ("low", "medium", "high")
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            env = LlumnixMigrationEnv(
                env_config, cost_estimator=cost_estimator,
                reward_calculator=MigPPORewardCalculator(
                    weights=MigPPORewardWeights(), scales=MigPPORewardScales(),
                    ttft_slo_ms=env_config.ttft_slo_ms),
            )
            obs, _ = env.reset(seed=seed, options={"scenario": scenario})
            episode_return = 0.0
            stays = actions = valid_migrations = successful = failed = 0
            completed_tokens = completed_requests = 0
            queue_length = 0.0
            decisions = 0
            while True:
                normalized = normalizer.transform(obs)
                tensors = _tensors(normalized, device)
                with torch.no_grad():
                    output = policy.act(*tensors, deterministic=True)
                action = int(output.action.item())
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                decisions += 1
                actions += 1
                stays += int(action == 0)
                valid_migrations += int(action > 0 and info["invalid_action_reason"] is None)
                successful += int(info["reward"]["migration_succeeded"])
                failed += int(info["reward"]["migration_failed"] + info["reward"]["migration_aborted"])
                completed_tokens += int(info["metrics"]["completed_output_tokens"])
                completed_requests += int(info["metrics"]["completed_requests"])
                queue_length += float(info["metrics"]["average_queue_length"])
                if terminated or truncated:
                    break
            rows.append({
                "scenario": scenario, "seed": seed, "episode_return": episode_return,
                "stay_rate": stays / max(1, actions), "valid_migrations": valid_migrations,
                "migration_successes": successful, "migration_failures_or_aborts": failed,
                "output_tokens_per_decision": completed_tokens / max(1, decisions),
                "completed_requests": completed_requests,
                "mean_queue_length": queue_length / max(1, decisions),
                "decisions": decisions,
            })
    return {
        "mean_episode_return": float(np.mean([row["episode_return"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }


def _fit_normalizer(env, seed: int, sample_steps: int) -> MigPPONormalizer:
    observations = []
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(max(1, sample_steps)):
        observations.append(obs)
        legal = np.flatnonzero(obs.action_mask.action_mask)
        action = int(rng.choice(legal)) if len(legal) else 0
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return MigPPONormalizer().fit(observations)


def train(config_path: str | Path = DEFAULT_CONFIG, *, output_dir: str | Path | None = None,
          total_timesteps: int | None = None, seed: int | None = None,
          calibration_path: str | Path | None = None,
          episodes: int | None = None) -> Path:
    payload = load_config(config_path)
    environment_payload = dict(payload.get("environment", {}))
    if calibration_path is not None:
        calibration_payload = load_config(calibration_path)
        environment_payload.update(calibration_payload.get("environment", calibration_payload))
    env_config = MigPPOEnvConfig(**environment_payload)
    ppo = dict(payload.get("ppo", {}))
    training = dict(payload.get("training", {}))
    seed = int(seed if seed is not None else training.get("seed", 42))
    total_timesteps = int(total_timesteps if total_timesteps is not None
                          else training.get("total_timesteps", 20000))
    episode_target = _resolve_episode_target(episodes, total_timesteps, training)
    if episode_target is not None and episode_target < 1:
        raise ValueError("episodes must be a positive integer")
    out = Path(output_dir or training.get("output_dir", "rl_scheduler/runs/mig_ppo"))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(int(training.get("torch_num_threads", 1)))
    device = torch.device(ppo.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Configured training device {device} is unavailable in this PyTorch environment")

    estimator = MigPPOMigrationCostEstimator(
        env_config.migration_alpha_ms, env_config.migration_beta_ms_per_block)
    if env_config.migration_cost_estimator:
        estimator = MigPPOMigrationCostEstimator.from_state_dict(env_config.migration_cost_estimator)
    elif env_config.migration_cost_csv:
        estimator.fit_csv(env_config.migration_cost_csv)
    reward_cfg = payload.get("reward", {})
    weights = MigPPORewardWeights(**reward_cfg.get("weights", {}))
    scales = MigPPORewardScales(**reward_cfg.get("scales", {}))
    reward_calculator = MigPPORewardCalculator(weights, scales, env_config.ttft_slo_ms)
    env = LlumnixMigrationEnv(env_config, cost_estimator=estimator, reward_calculator=reward_calculator)
    normalizer = _fit_normalizer(env, seed, int(training.get("normalization_steps", 256)))
    obs, _ = env.reset(seed=seed + 1)
    normalized = normalizer.transform(obs)
    policy = MigPPOActorCritic(
        instance_feature_dim=normalized.instance_features.shape[-1],
        global_feature_dim=normalized.global_features.shape[-1],
        pair_feature_dim=normalized.pair_features.shape[-1],
        num_instances=env_config.num_instances,
        hidden_size=int(ppo.get("hidden_size", 64)),
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(ppo.get("learning_rate", 3e-4)))

    steps_per_update = int(ppo.get("rollout_steps", 512))
    minibatch_size = int(ppo.get("minibatch_size", 64))
    update_epochs = int(ppo.get("update_epochs", 10))
    gamma = float(ppo.get("gamma", 0.99))
    gae_lambda = float(ppo.get("gae_lambda", 0.95))
    clip_epsilon = float(ppo.get("clip_epsilon", 0.2))
    value_coefficient = float(ppo.get("value_coefficient", 0.5))
    entropy_coefficient = float(ppo.get("entropy_coefficient", 0.01))
    max_grad_norm = float(ppo.get("max_grad_norm", 0.5))
    eval_interval = int(training.get("evaluation_interval_updates", 10))
    maximum_steps = episode_target * env_config.max_decisions if episode_target is not None else total_timesteps
    updates = max(1, math.ceil(maximum_steps / steps_per_update))
    rng = np.random.default_rng(seed + 2)
    global_step = 0
    best_return = -math.inf
    episode_returns: list[float] = []
    current_episode_return = 0.0
    episodes_completed = 0
    reset_seed = seed + 100
    metrics_file = (out / "training_metrics.jsonl").open("a", encoding="utf-8")
    best_path = out / "mig_ppo_best.pt"

    try:
        for update in range(1, updates + 1):
            if episode_target is not None and episodes_completed >= episode_target:
                break
            obs_buffer, action_buffer, logp_buffer, value_buffer = [], [], [], []
            reward_buffer, done_buffer = [], []
            stay_count = valid_migration_count = invalid_action_count = 0
            migration_success_count = migration_terminal_count = 0
            queue_values = []
            ttft_values = []
            tpot_values = []
            kv_values = []
            imbalance_values = []
            for _ in range(steps_per_update):
                normalized = normalizer.transform(obs)
                tensor_obs = _tensors(normalized, device)
                with torch.no_grad():
                    action_output = policy.act(*tensor_obs, deterministic=False)
                action = int(action_output.action.item())
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                obs_buffer.append(normalized)
                action_buffer.append(action)
                logp_buffer.append(float(action_output.log_probability.item()))
                value_buffer.append(float(action_output.value.item()))
                reward_buffer.append(float(reward))
                done_buffer.append(float(done))
                stay_count += int(action == 0)
                valid_migration_count += int(action > 0 and info["invalid_action_reason"] is None)
                invalid_action_count += int(info["invalid_action_reason"] is not None)
                queue_values.append(float(info["metrics"]["average_queue_length"]))
                migration_success_count += int(info["reward"]["migration_succeeded"])
                migration_terminal_count += int(info["reward"]["migration_succeeded"]
                                                 + info["reward"]["migration_failed"]
                                                 + info["reward"]["migration_aborted"])
                if info["metrics"]["mean_observed_ttft_ms"] > 0:
                    ttft_values.append(float(info["metrics"]["mean_observed_ttft_ms"]))
                if info["metrics"]["mean_observed_tpot_ms"] > 0:
                    tpot_values.append(float(info["metrics"]["mean_observed_tpot_ms"]))
                kv_values.append(float(info["metrics"]["mean_kv_usage"]))
                imbalance_values.append(float(info["metrics"]["load_imbalance"]))
                current_episode_return += float(reward)
                global_step += 1
                obs = next_obs
                if done:
                    episode_returns.append(current_episode_return)
                    current_episode_return = 0.0
                    episodes_completed += 1
                    reset_seed += 1
                    obs, _ = env.reset(seed=reset_seed)
                    if episode_target is not None and episodes_completed >= episode_target:
                        break

            with torch.no_grad():
                last_normalized = normalizer.transform(obs)
                last_value = float(policy.act(*_tensors(last_normalized, device), deterministic=True).value.item())
            advantages = np.zeros(len(reward_buffer), dtype=np.float32)
            last_advantage = 0.0
            for idx in reversed(range(len(reward_buffer))):
                next_nonterminal = 1.0 - done_buffer[idx]
                next_val = last_value if idx == len(reward_buffer) - 1 else value_buffer[idx + 1]
                delta = reward_buffer[idx] + gamma * next_val * next_nonterminal - value_buffer[idx]
                last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
                advantages[idx] = last_advantage
            returns = advantages + np.asarray(value_buffer, dtype=np.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            inst_batch = torch.as_tensor(np.stack([x.instance_features for x in obs_buffer]), dtype=torch.float32, device=device)
            global_batch = torch.as_tensor(np.stack([x.global_features for x in obs_buffer]), dtype=torch.float32, device=device)
            pair_batch = torch.as_tensor(np.stack([x.pair_features for x in obs_buffer]), dtype=torch.float32, device=device)
            mask_batch = torch.as_tensor(np.stack([x.pair_mask for x in obs_buffer]), dtype=torch.bool, device=device)
            action_batch = torch.as_tensor(action_buffer, dtype=torch.long, device=device)
            old_logp_batch = torch.as_tensor(logp_buffer, dtype=torch.float32, device=device)
            advantage_batch = torch.as_tensor(advantages, dtype=torch.float32, device=device)
            return_batch = torch.as_tensor(returns, dtype=torch.float32, device=device)
            indices = np.arange(len(reward_buffer))
            actor_losses, critic_losses, entropies, kls = [], [], [], []
            for _epoch in range(update_epochs):
                rng.shuffle(indices)
                for start in range(0, len(indices), minibatch_size):
                    mb = torch.as_tensor(indices[start:start + minibatch_size], dtype=torch.long, device=device)
                    new_logp, entropy, value, _ = policy.evaluate_actions(
                        inst_batch[mb], global_batch[mb], pair_batch[mb], mask_batch[mb], action_batch[mb])
                    ratio = torch.exp(new_logp - old_logp_batch[mb])
                    surrogate = torch.minimum(
                        ratio * advantage_batch[mb],
                        torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage_batch[mb],
                    )
                    actor_loss = -surrogate.mean()
                    critic_loss = 0.5 * torch.square(return_batch[mb] - value).mean()
                    entropy_mean = entropy.mean()
                    loss = actor_loss + value_coefficient * critic_loss - entropy_coefficient * entropy_mean
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - torch.log(ratio.clamp_min(1e-8))).mean()
                    actor_losses.append(float(actor_loss.item()))
                    critic_losses.append(float(critic_loss.item()))
                    entropies.append(float(entropy_mean.item()))
                    kls.append(float(approx_kl.item()))

            row = {
                "algorithm": "Mig-PPO", "update": update, "global_step": global_step,
                "training_episodes": episodes_completed,
                "mean_episode_return": float(np.mean(episode_returns[-10:])) if episode_returns else None,
                "actor_loss": float(np.mean(actor_losses)), "critic_loss": float(np.mean(critic_losses)),
                "entropy": float(np.mean(entropies)), "approx_kl": float(np.mean(kls)),
                "stay_rate": stay_count / max(1, len(reward_buffer)),
                "valid_migrations": valid_migration_count,
                "invalid_actions": invalid_action_count,
                "migration_terminal_events": migration_terminal_count,
                "migration_completion_rate": migration_success_count / max(1, valid_migration_count),
                "mean_queue_length": float(np.mean(queue_values)),
                "mean_observed_ttft_ms": float(np.mean(ttft_values)) if ttft_values else 0.0,
                "mean_observed_tpot_ms": float(np.mean(tpot_values)) if tpot_values else 0.0,
                "mean_kv_usage": float(np.mean(kv_values)) if kv_values else 0.0,
                "load_imbalance": float(np.mean(imbalance_values)) if imbalance_values else 0.0,
                "calibration_status": env_config.calibration_status,
            }
            eval_result = None
            if update % eval_interval == 0 or update == updates:
                policy.eval()
                eval_result = _evaluate(policy, normalizer, env_config, estimator, device)
                row["validation_mean_episode_return"] = eval_result["mean_episode_return"]
                policy.train()
                if eval_result["mean_episode_return"] > best_return:
                    best_return = eval_result["mean_episode_return"]
                    torch.save(_checkpoint_payload(policy, normalizer, env_config, estimator, ppo,
                                                    global_step, eval_result, episodes_completed), best_path)
            metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
            metrics_file.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
    finally:
        metrics_file.close()
    if not best_path.exists():
        torch.save(_checkpoint_payload(policy, normalizer, env_config, estimator, ppo,
                                        global_step, {}, episodes_completed), best_path)
    torch.save(_checkpoint_payload(policy, normalizer, env_config, estimator, ppo,
                                   global_step, {}, episodes_completed), out / "mig_ppo_final.pt")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mig-PPO migration policy")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--episodes", type=int, help="number of complete training episodes")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--calibration", help="fitted YAML from rl_scheduler.calibrate")
    args = parser.parse_args()
    path = train(args.config, output_dir=args.output_dir, total_timesteps=args.timesteps, seed=args.seed,
                 calibration_path=args.calibration, episodes=args.episodes)
    print(f"best_checkpoint={path}")


if __name__ == "__main__":
    main()

