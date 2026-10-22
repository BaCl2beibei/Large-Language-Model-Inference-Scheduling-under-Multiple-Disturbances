"""Finite, shared train/inference normalization for Mig-PPO observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from rl_scheduler.observation import MigPPOObservation


@dataclass(frozen=True)
class NormalizedMigPPOObservation:
    instance_features: np.ndarray
    global_features: np.ndarray
    pair_features: np.ndarray
    pair_mask: np.ndarray


class MigPPONormalizer:
    """Apply capacity/SLO scaling and optional training-set z-score statistics."""

    def __init__(
        self,
        request_capacity: float = 64.0,
        wait_slo_ms: float = 2000.0,
        step_ms_scale: float = 100.0,
        token_batch_scale: float = 2048.0,
        remaining_steps_scale: float = 64.0,
        arrival_rate_scale: float = 4.0,
        migration_time_scale_ms: float = 1000.0,
        migration_gain_scale_ms: float = 2000.0,
        migration_cooldown_sec: float = 5.0,
    ) -> None:
        self.request_capacity = max(1.0, float(request_capacity))
        self.wait_slo_ms = max(1.0, float(wait_slo_ms))
        self.step_ms_scale = max(1.0, float(step_ms_scale))
        self.token_batch_scale = max(1.0, float(token_batch_scale))
        self.remaining_steps_scale = max(1.0, float(remaining_steps_scale))
        self.arrival_rate_scale = max(1e-3, float(arrival_rate_scale))
        self.migration_time_scale_ms = max(1.0, float(migration_time_scale_ms))
        self.migration_gain_scale_ms = max(1.0, float(migration_gain_scale_ms))
        self.migration_cooldown_sec = max(1e-3, float(migration_cooldown_sec))
        self.statistics: dict[str, tuple[float, float]] = {}

    def fit(self, observations: Iterable[MigPPOObservation]) -> "MigPPONormalizer":
        samples: dict[str, list[np.ndarray]] = {
            "instance_last_step_ms": [], "instance_batched_tokens": [], "instance_remaining_steps": [],
            "global_arrival_rate": [], "global_since_migration": [],
            "pair_migration_ms": [], "pair_gain_ms": [], "pair_cost_uncertainty_ms": [],
        }
        for obs in observations:
            if len(obs.instance_features):
                samples["instance_last_step_ms"].append(obs.instance_features[:, 7].astype(np.float64))
                samples["instance_batched_tokens"].append(obs.instance_features[:, 8].astype(np.float64))
                samples["instance_remaining_steps"].append(obs.instance_features[:, 9].astype(np.float64))
            samples["global_arrival_rate"].append(np.asarray([obs.global_features[1]], dtype=np.float64))
            samples["global_since_migration"].append(np.asarray([obs.global_features[6]], dtype=np.float64))
            if len(obs.pair_features):
                samples["pair_migration_ms"].append(obs.pair_features[:, 3].astype(np.float64))
                samples["pair_gain_ms"].append(obs.pair_features[:, 4].astype(np.float64))
                samples["pair_cost_uncertainty_ms"].append(
                    obs.pair_cost_uncertainty_ms[:, 0].astype(np.float64))
        self.statistics = {}
        for name, arrays in samples.items():
            if not arrays:
                continue
            values = np.concatenate(arrays)
            values = values[np.isfinite(values)]
            if len(values):
                mean = float(np.mean(values))
                std = float(np.std(values))
                self.statistics[name] = (mean if np.isfinite(mean) else 0.0,
                                         std if np.isfinite(std) and std > 1e-6 else 1.0)
        return self

    def _zscore(self, values: np.ndarray, key: str) -> np.ndarray:
        mean, std = self.statistics.get(key, (0.0, 1.0))
        return np.clip((values - mean) / std, -10.0, 10.0)

    def transform(self, obs: MigPPOObservation) -> NormalizedMigPPOObservation:
        x = np.nan_to_num(obs.instance_features.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
        x[:, 0] /= self.request_capacity
        x[:, 1] /= self.request_capacity
        x[:, 2] /= self.wait_slo_ms
        x[:, 3] = np.clip(x[:, 3], 0.0, 2.0)
        x[:, 4] = np.clip(x[:, 4], 0.0, 1.0)
        x[:, 5] /= self.request_capacity
        x[:, 6] /= self.request_capacity
        x[:, 7] = self._zscore(x[:, 7], "instance_last_step_ms")
        x[:, 8] = self._zscore(x[:, 8], "instance_batched_tokens")
        x[:, 9] = self._zscore(x[:, 9], "instance_remaining_steps")
        x[:, 10] = np.clip(x[:, 10], 0.0, 1.0)
        x[:, 11:14] = np.clip(x[:, 11:14], 0.0, 1.0)
        x = np.concatenate((x, obs.instance_missing.astype(np.float32)), axis=-1)

        g = np.nan_to_num(obs.global_features.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
        n = max(1, int(g[0]))
        g[0] /= n
        g[1] = self._zscore(g[1:2], "global_arrival_rate")[0]
        g[2] /= self.request_capacity ** 2
        g[3] = np.clip(g[3], 0.0, 1.0)
        g[4] /= (n * self.request_capacity)
        g[5] /= n
        g[6] = self._zscore(g[6:7], "global_since_migration")[0]
        g = np.concatenate((g, obs.global_missing.astype(np.float32)), axis=-1)

        p = np.nan_to_num(obs.pair_features.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
        if len(p):
            p[:, 0] /= self.request_capacity
            p[:, 1] = np.clip(p[:, 1], -1.0, 1.0)
            p[:, 2] = np.clip(p[:, 2], 0.0, 1.0)
            p[:, 3] = self._zscore(p[:, 3], "pair_migration_ms")
            p[:, 4] = self._zscore(p[:, 4], "pair_gain_ms")
        uncertainty = self._zscore(
            np.nan_to_num(obs.pair_cost_uncertainty_ms.astype(np.float32, copy=True),
                          nan=0.0, posinf=0.0, neginf=0.0),
            "pair_cost_uncertainty_ms",
        )
        source = np.nan_to_num(obs.pair_cost_source.astype(np.float32, copy=True),
                               nan=0.0, posinf=0.0, neginf=0.0)
        p = np.concatenate((p, obs.pair_missing.astype(np.float32), uncertainty, source), axis=-1)
        return NormalizedMigPPOObservation(
            instance_features=np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
            global_features=np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
            pair_features=np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0),
            pair_mask=obs.pair_mask.astype(np.bool_, copy=True),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "scales": {
                "request_capacity": self.request_capacity,
                "wait_slo_ms": self.wait_slo_ms,
                "step_ms_scale": self.step_ms_scale,
                "token_batch_scale": self.token_batch_scale,
                "remaining_steps_scale": self.remaining_steps_scale,
                "arrival_rate_scale": self.arrival_rate_scale,
                "migration_time_scale_ms": self.migration_time_scale_ms,
                "migration_gain_scale_ms": self.migration_gain_scale_ms,
                "migration_cooldown_sec": self.migration_cooldown_sec,
            },
            "statistics": {key: list(value) for key, value in self.statistics.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "MigPPONormalizer":
        normalizer = cls(**dict(state.get("scales", {})))  # type: ignore[arg-type]
        statistics = state.get("statistics", {})
        if isinstance(statistics, Mapping):
            normalizer.statistics = {
                str(key): (float(value[0]), max(1e-6, float(value[1])))
                for key, value in statistics.items()
            }
        return normalizer

