"""Observed-window reward accounting for Mig-PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class MigPPORewardWeights:
    throughput: float = 0.20
    waiting_pressure: float = 0.25
    tpot_pressure: float = 0.20
    slo_violations: float = 0.25
    kv_pressure: float = 0.10
    migration_cost: float = 0.10
    migration_count: float = 0.05
    migration_failure: float = 0.20


@dataclass(frozen=True)
class MigPPORewardScales:
    throughput_tokens_per_sec: float = 100.0
    mean_waiting_requests: float = 4.0
    tpot_ms: float = 150.0
    slo_violation_rate: float = 0.10
    kv_usage: float = 0.90
    migration_cost_ms: float = 1000.0
    migration_operations: float = 1.0
    migration_failures: float = 1.0


@dataclass
class MigPPORewardMetrics:
    window_sec: float
    instance_count: int
    completed_requests: int = 0
    completed_output_tokens: int = 0
    accumulated_waiting_request_seconds: float = 0.0
    observed_tpot_ms: list[float] = field(default_factory=list)
    observed_ttft_ms: list[float] = field(default_factory=list)
    ttft_slo_violations: int = 0
    tpot_slo_violations: int = 0
    timeout_requests: int = 0
    unfinished_requests: int = 0
    kv_usage_samples: list[float] = field(default_factory=list)
    migration_cost_ms: float = 0.0
    migration_operations: int = 0
    migration_succeeded: int = 0
    migration_failed: int = 0
    migration_aborted: int = 0
    average_queue_length: float = 0.0
    load_imbalance: float = 0.0


@dataclass(frozen=True)
class MigPPORewardBreakdown:
    reward: float
    raw: dict[str, float]
    normalized: dict[str, float]
    weighted: dict[str, float]
    migration_succeeded: int
    migration_failed: int
    migration_aborted: int
    timeout_requests: int
    unfinished_requests: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MigPPORewardCalculator:
    """Calculate reward only from metrics observed in the completed window.

    The environment contributes TTFT/TPOT samples when their timestamp occurs,
    so no future request latency is read by the reward function. Requests still
    waiting accrue pressure continuously and timeout/unfinished counts remain
    explicit in the returned breakdown.
    """

    def __init__(self, weights: MigPPORewardWeights = MigPPORewardWeights(),
                 scales: MigPPORewardScales = MigPPORewardScales(),
                 ttft_slo_ms: float = 2000.0) -> None:
        self.weights = weights
        self.scales = scales
        self.ttft_slo_ms = max(1.0, float(ttft_slo_ms))

    @staticmethod
    def _clip(value: float) -> float:
        if not value == value or value == float("inf") or value == float("-inf"):
            return 0.0
        return min(2.0, max(0.0, float(value)))

    def compute(self, metrics: MigPPORewardMetrics) -> MigPPORewardBreakdown:
        window = max(1e-6, float(metrics.window_sec))
        instances = max(1, int(metrics.instance_count))
        throughput = max(0.0, metrics.completed_output_tokens / window)
        mean_waiting_requests = max(0.0, metrics.accumulated_waiting_request_seconds / (window * instances))
        observed_tpot = [max(0.0, x) for x in metrics.observed_tpot_ms if x == x and x != float("inf")]
        mean_tpot = sum(observed_tpot) / len(observed_tpot) if observed_tpot else 0.0
        observed_ttft = [max(0.0, x) for x in metrics.observed_ttft_ms if x == x and x != float("inf")]
        mean_ttft = sum(observed_ttft) / len(observed_ttft) if observed_ttft else 0.0
        observed_slo_events = len(observed_tpot) + len(observed_ttft) \
            + metrics.timeout_requests + metrics.unfinished_requests
        violation_events = metrics.ttft_slo_violations + metrics.tpot_slo_violations \
            + metrics.timeout_requests + metrics.unfinished_requests
        violation_rate = violation_events / max(1, observed_slo_events)
        kv_values = [min(1.0, max(0.0, float(x))) for x in metrics.kv_usage_samples if x == x]
        mean_kv_usage = sum(kv_values) / len(kv_values) if kv_values else 0.0
        failure_events = metrics.migration_failed + metrics.migration_aborted
        raw = {
            "completed_requests": float(metrics.completed_requests),
            "completed_output_tokens": float(metrics.completed_output_tokens),
            "throughput_tokens_per_sec": throughput,
            "accumulated_waiting_request_seconds": float(metrics.accumulated_waiting_request_seconds),
            "mean_waiting_requests": mean_waiting_requests,
            "mean_observed_tpot_ms": mean_tpot,
            "mean_observed_ttft_ms": mean_ttft,
            "ttft_slo_violations": float(metrics.ttft_slo_violations),
            "tpot_slo_violations": float(metrics.tpot_slo_violations),
            "timeout_requests": float(metrics.timeout_requests),
            "unfinished_requests": float(metrics.unfinished_requests),
            "slo_violation_rate": violation_rate,
            "mean_kv_usage": mean_kv_usage,
            "actual_migration_cost_ms": max(0.0, float(metrics.migration_cost_ms)),
            "migration_operations": float(metrics.migration_operations),
            "migration_succeeded": float(metrics.migration_succeeded),
            "migration_failed": float(metrics.migration_failed),
            "migration_aborted": float(metrics.migration_aborted),
            "average_queue_length": float(metrics.average_queue_length),
            "load_imbalance": float(metrics.load_imbalance),
        }
        normalized = {
            "throughput": self._clip(throughput / max(1e-6, self.scales.throughput_tokens_per_sec)),
            "waiting_pressure": self._clip(mean_waiting_requests / max(1e-6, self.scales.mean_waiting_requests)),
            "tpot_pressure": self._clip(mean_tpot / max(1e-6, self.scales.tpot_ms)),
            "slo_violations": self._clip(violation_rate / max(1e-6, self.scales.slo_violation_rate)),
            "kv_pressure": self._clip(mean_kv_usage / max(1e-6, self.scales.kv_usage)),
            "migration_cost": self._clip(metrics.migration_cost_ms / max(1e-6, self.scales.migration_cost_ms)),
            "migration_count": self._clip(metrics.migration_operations / max(1e-6, self.scales.migration_operations)),
            "migration_failure": self._clip(failure_events / max(1e-6, self.scales.migration_failures)),
        }
        weighted = {
            "throughput": self.weights.throughput * normalized["throughput"],
            "waiting_pressure": -self.weights.waiting_pressure * normalized["waiting_pressure"],
            "tpot_pressure": -self.weights.tpot_pressure * normalized["tpot_pressure"],
            "slo_violations": -self.weights.slo_violations * normalized["slo_violations"],
            "kv_pressure": -self.weights.kv_pressure * normalized["kv_pressure"],
            "migration_cost": -self.weights.migration_cost * normalized["migration_cost"],
            "migration_count": -self.weights.migration_count * normalized["migration_count"],
            "migration_failure": -self.weights.migration_failure * normalized["migration_failure"],
        }
        reward = float(sum(weighted.values()))
        return MigPPORewardBreakdown(
            reward=reward, raw=raw, normalized=normalized, weighted=weighted,
            migration_succeeded=metrics.migration_succeeded,
            migration_failed=metrics.migration_failed,
            migration_aborted=metrics.migration_aborted,
            timeout_requests=metrics.timeout_requests,
            unfinished_requests=metrics.unfinished_requests,
        )

