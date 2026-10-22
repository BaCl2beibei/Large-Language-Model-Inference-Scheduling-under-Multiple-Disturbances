"""Deterministic Mig-PPO and Stay-baseline evaluation in fixed workloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_scheduler.cost_estimator import MigPPOMigrationCostEstimator
from rl_scheduler.env import LlumnixMigrationEnv, MigPPOEnvConfig
from rl_scheduler.normalization import MigPPONormalizer
from rl_scheduler.policy import MigPPOActorCritic
from rl_scheduler.train_ppo import _tensors


def load_policy(checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("algorithm") != "Mig-PPO":
        raise ValueError("checkpoint is not a Mig-PPO checkpoint")
    spec = checkpoint["model_spec"]
    policy = MigPPOActorCritic(**spec).to(torch.device(device))
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy, MigPPONormalizer.from_state_dict(checkpoint["normalizer"]), checkpoint


def evaluate(checkpoint_path: str | Path, scenarios=None,
             seeds=(101, 202, 303), device: str = "cpu", stay_baseline: bool = True) -> dict[str, Any]:
    policy, normalizer, checkpoint = load_policy(checkpoint_path, device)
    env_config = MigPPOEnvConfig(**checkpoint["environment_config"])
    if scenarios is None:
        scenarios = tuple(f"qps_{float(qps):g}" for qps in env_config.target_qps_levels) \
            or ("low", "medium", "high")
    estimator = MigPPOMigrationCostEstimator.from_state_dict(checkpoint["cost_estimator"])
    results = []
    for scenario in scenarios:
        for seed in seeds:
            env = LlumnixMigrationEnv(env_config, cost_estimator=estimator)
            obs, reset_info = env.reset(seed=seed, options={"scenario": scenario})
            total_reward = 0.0
            tokens = requests = stays = migrations = successful = failures = 0
            queue, ttft, tpot = [], [], []
            while True:
                normalized = normalizer.transform(obs)
                with torch.no_grad():
                    output = policy.act(*_tensors(normalized, torch.device(device)), deterministic=True)
                action = int(output.action.item())
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                tokens += int(info["metrics"]["completed_output_tokens"])
                requests += int(info["metrics"]["completed_requests"])
                stays += int(action == 0)
                migrations += int(action > 0 and info["invalid_action_reason"] is None)
                successful += int(info["reward"]["migration_succeeded"])
                failures += int(info["reward"]["migration_failed"] + info["reward"]["migration_aborted"])
                queue.append(float(info["metrics"]["average_queue_length"]))
                if info["metrics"]["mean_observed_ttft_ms"] > 0:
                    ttft.append(float(info["metrics"]["mean_observed_ttft_ms"]))
                if info["metrics"]["mean_observed_tpot_ms"] > 0:
                    tpot.append(float(info["metrics"]["mean_observed_tpot_ms"]))
                if terminated or truncated:
                    break
            results.append({
                "algorithm": "Mig-PPO", "scenario": scenario, "seed": seed,
                "episode_return": total_reward, "completed_requests": requests,
                "output_tokens": tokens, "stay_rate": stays / max(1, env.decision_count),
                "migration_attempts": migrations, "migration_successes": successful,
                "migration_failures_or_aborts": failures,
                "mean_queue_length": float(np.mean(queue)) if queue else 0.0,
                "mean_observed_ttft_ms": float(np.mean(ttft)) if ttft else 0.0,
                "mean_observed_tpot_ms": float(np.mean(tpot)) if tpot else 0.0,
                "calibration_status": reset_info["calibration_status"],
                "episode": env.export_episode(),
            })
            if stay_baseline:
                base_env = LlumnixMigrationEnv(env_config, cost_estimator=estimator)
                base_obs, _ = base_env.reset(seed=seed, options={"scenario": scenario})
                base_return = base_tokens = base_requests = 0.0
                base_queue, base_ttft, base_tpot = [], [], []
                while True:
                    base_obs, reward, terminated, truncated, _info = base_env.step(0)
                    base_return += float(reward)
                    base_tokens += float(_info["metrics"]["completed_output_tokens"])
                    base_requests += float(_info["metrics"]["completed_requests"])
                    base_queue.append(float(_info["metrics"]["average_queue_length"]))
                    if _info["metrics"]["mean_observed_ttft_ms"] > 0:
                        base_ttft.append(float(_info["metrics"]["mean_observed_ttft_ms"]))
                    if _info["metrics"]["mean_observed_tpot_ms"] > 0:
                        base_tpot.append(float(_info["metrics"]["mean_observed_tpot_ms"]))
                    if terminated or truncated:
                        break
                results.append({
                    "algorithm": "Stay", "scenario": scenario, "seed": seed,
                    "episode_return": base_return, "completed_requests": int(base_requests),
                    "output_tokens": int(base_tokens), "stay_rate": 1.0,
                    "migration_attempts": 0, "migration_successes": 0,
                    "migration_failures_or_aborts": 0,
                    "mean_queue_length": float(np.mean(base_queue)) if base_queue else 0.0,
                    "mean_observed_ttft_ms": float(np.mean(base_ttft)) if base_ttft else 0.0,
                    "mean_observed_tpot_ms": float(np.mean(base_tpot)) if base_tpot else 0.0,
                    "calibration_status": env_config.calibration_status,
                    "episode": base_env.export_episode(),
                })
    return {
        "algorithm": "Mig-PPO", "checkpoint": str(checkpoint_path),
        "training_step": checkpoint.get("training_step"),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mig-PPO on deterministic seeds")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="workload scenario names; defaults to the checkpoint's configured QPS levels")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-stay-baseline", action="store_true")
    parser.add_argument("--output", default="mig_ppo_evaluation.json")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, tuple(args.scenarios) if args.scenarios else None,
                      tuple(args.seeds), args.device,
                      stay_baseline=not args.no_stay_baseline)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": args.output, "result_count": len(result["results"])}, indent=2))


if __name__ == "__main__":
    main()

