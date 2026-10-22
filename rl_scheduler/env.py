"""Seeded, discrete-event multi-instance environment for Mig-PPO training.

The environment advances request queues, KV allocation, prefill/decode service,
timeouts, preemption, and in-flight transfers. It is intentionally a simulation
model: calibration metadata is surfaced in every reset/step result.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import yaml

from rl_scheduler.action_mask import MigPPOActionMaskConfig
from rl_scheduler.cost_estimator import MigPPOMigrationCostEstimator
from rl_scheduler.observation import MigPPOObservation, RollingArrivalRate, build_mig_ppo_observation
from rl_scheduler.reward import MigPPORewardCalculator, MigPPORewardMetrics, MigPPORewardWeights, MigPPORewardScales
from rl_scheduler.service_model import predict_service_model


@dataclass
class MigPPOEnvConfig:
    num_instances: int = 4
    decision_interval_sec: float = 1.0
    max_decisions: int = 300
    max_running_requests: Optional[int] = 32  # compatibility alias for older configs
    max_num_seqs_per_instance: Optional[int] = None
    total_kv_blocks: int = 512
    block_size_tokens: int = 16
    max_model_len: int = 2048
    max_num_batched_tokens: int = 4096
    ttft_slo_ms: float = 2000.0
    tpot_slo_ms: float = 150.0
    request_timeout_sec: Optional[float] = 60.0
    migration_cooldown_sec: float = 5.0
    max_state_age_sec: float = 2.0
    safety_margin_blocks: int = 16
    safety_margin_fraction: float = 0.05
    migration_failure_probability: float = 0.01
    workload_scenario: str = "medium"
    arrival_model: str = "poisson"
    target_qps_levels: tuple[float, ...] = ()
    arrival_rates_rps: dict[str, float] = field(default_factory=lambda: {
        "low": 0.50, "medium": 6.00, "high": 16.00, "bursty": 9.00,
    })
    prompt_profiles: tuple[int, ...] = (64, 128, 256)
    output_profiles: tuple[int, ...] = (64, 96, 128)
    # Optional empirical distribution for converting traced max-token budgets
    # into realized response lengths. Synthetic training profiles retain their
    # configured exact lengths unless they arrive through an explicit trace.
    output_length_model: Optional[dict[str, object]] = None
    # Initial values are deliberately marked uncalibrated. Replace them with
    # fits from target-cluster profiling logs before interpreting results.
    prefill_ms_per_token: float = 0.025
    prefill_intercept_ms: float = 4.0
    decode_ms_per_request: float = 2.5
    decode_ms_per_context_token: float = 0.004
    decode_intercept_ms: float = 1.0
    service_time_model: Optional[dict[str, object]] = None
    service_model_mode: str = "legacy"
    # A calibrated multiplier converts executor iteration wall time into the
    # observed per-request token interval when requests are not emitted on
    # every engine iteration. The default preserves legacy behavior.
    decode_token_interval_scale: float = 1.0
    trace_events_enabled: bool = False
    trace_token_events: bool = False
    validate_state_invariants: bool = False
    instance_capacity_scale: float = 1.0
    kv_dynamics_enabled: bool = True
    observation_refresh_interval_sec: Optional[float] = None
    min_engine_step_ms: float = 1.0
    migration_alpha_ms: float = 50.0
    migration_beta_ms_per_block: float = 0.1
    migration_cost_csv: Optional[str] = None
    migration_cost_estimator: Optional[dict[str, object]] = None
    migration_uncertainty_scale: float = 0.0
    calibration_status: str = "uncalibrated"
    routing_skew_probability: float = 0.12

    def __post_init__(self) -> None:
        if not self.prompt_profiles or not self.output_profiles:
            raise ValueError("prompt_profiles and output_profiles must be non-empty")
        longest_request = max(self.prompt_profiles) + max(self.output_profiles)
        if self.max_model_len < longest_request:
            raise ValueError(
                f"max_model_len={self.max_model_len} cannot fit the longest prompt/response pair "
                f"({longest_request} tokens)"
            )
        if self.arrival_model != "poisson":
            raise ValueError("Mig-PPO currently supports Poisson request arrivals")
        if self.service_model_mode not in {"legacy", "calibrated"}:
            raise ValueError("service_model_mode must be 'legacy' or 'calibrated'")
        if self.decode_token_interval_scale <= 0:
            raise ValueError("decode_token_interval_scale must be positive")
        if any(float(qps) <= 0 for qps in self.target_qps_levels):
            raise ValueError("target_qps_levels must contain positive values")

    @property
    def active_sequence_capacity(self) -> int:
        configured = self.max_num_seqs_per_instance
        if configured is None:
            configured = self.max_running_requests
        return max(1, int(configured if configured is not None else 32))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MigPPOEnvConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "environment" in payload:
            payload = payload["environment"]
        return cls(**payload)


@dataclass(eq=False)
class _Request:
    request_id: str
    arrival_time: float
    prompt_tokens: int
    target_output_tokens: int
    requested_output_tokens: Optional[int] = None
    output_tokens: int = 0
    stage: str = "waiting"
    allocated_blocks: int = 0
    waiting_since: float = 0.0
    last_token_time: Optional[float] = None
    first_token_time: Optional[float] = None
    migrating: bool = False
    preemptions: int = 0
    tpot_samples_ms: list[float] = field(default_factory=list)
    token_timestamps_sec: list[float] = field(default_factory=list)

    @property
    def request_len(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass
class _SimInstance:
    instance_id: str
    total_blocks: int
    capacity_scale: float
    running: list[_Request] = field(default_factory=list)
    waiting: list[_Request] = field(default_factory=list)
    used_blocks: int = 0
    reserved_blocks: int = 0
    next_step_time: float = math.inf
    last_step_ms: float = 0.0
    last_batch_tokens: int = 0
    preempted_requests: int = 0
    is_migrating: bool = False
    step_in_flight: bool = False
    pending_prefill: list[_Request] = field(default_factory=list)
    pending_decode: list[_Request] = field(default_factory=list)
    pending_prefill_time: float = 0.0
    pending_decode_time: float = 0.0
    service_step_count: int = 0
    pending_step_id: int = 0
    pending_step_start_time: float = 0.0
    pending_step_end_time: float = 0.0

    @property
    def free_blocks(self) -> int:
        return max(0, self.total_blocks - self.used_blocks - self.reserved_blocks)


@dataclass
class _PendingMigration:
    action_id: str
    source_id: str
    target_id: str
    request_id: str
    blocks: int
    start_time: float
    end_time: float
    predicted_ms: float
    actual_ms: float
    replay_status: Optional[str] = None
    observed_blocks: Optional[int] = None


@dataclass
class _RemainingSteps:
    remaining_steps: float


class _InstanceView:
    """InstanceInfo-shaped, read-only view consumed by the shared builder."""

    def __init__(self, env: "LlumnixMigrationEnv", instance: _SimInstance) -> None:
        now = env._wall_time()
        self.instance_id = instance.instance_id
        self.instance_type = "neutral"
        self.unit_status = "healthy"
        self.timestamp = now
        self.num_running_requests = len(instance.running)
        self.num_waiting_requests = len(instance.waiting)
        self.waiting_time_first_waiting_request = (
            max(0.0, env.now - instance.waiting[0].arrival_time) if instance.waiting else 0.0
        )
        self.num_total_gpu_blocks = instance.total_blocks
        self.num_free_gpu_blocks = instance.free_blocks
        self.num_used_gpu_blocks = instance.used_blocks + instance.reserved_blocks
        self.num_blocks_all_waiting_requests = sum(
            env._blocks_for_tokens(request.request_len) for request in instance.waiting
        )
        self.gpu_cache_usage = min(1.0, self.num_used_gpu_blocks / max(1, self.num_total_gpu_blocks))
        self.num_killed_requests = instance.preempted_requests
        self.decode_batch_size = sum(request.stage == "decode" for request in instance.running)
        self.last_inference_latency = instance.last_step_ms
        self.num_batched_tokens = instance.last_batch_tokens
        free_after_waiting = self.num_free_gpu_blocks - self.num_blocks_all_waiting_requests
        counted_requests = self.num_running_requests + self.num_waiting_requests
        self.remaining_steps = _RemainingSteps(
            float("inf") if counted_requests == 0 else free_after_waiting / counted_requests
        )
        self.migration_candidate_request_id = None
        self.migration_candidate_kv_blocks = None
        self.migration_candidate_stage = None
        self.migration_candidate_valid = False
        self.migration_candidate_timestamp = now
        candidate = env._sr_candidate(instance)
        if candidate is not None:
            self.migration_candidate_request_id = candidate.request_id
            self.migration_candidate_kv_blocks = (candidate.allocated_blocks if env.config.kv_dynamics_enabled
                                                  else env._blocks_for_tokens(candidate.request_len))
            self.migration_candidate_stage = candidate.stage
            self.migration_candidate_valid = self.migration_candidate_kv_blocks > 0 and not candidate.migrating
        self.has_migration_slot = not instance.is_migrating
        self.is_migrating = instance.is_migrating
        self.can_accept_migration = not instance.is_migrating \
            and len(instance.running) < env.config.active_sequence_capacity

    def is_unit_healthy(self) -> bool:
        return True


class LlumnixMigrationEnv:
    """Gymnasium-style seeded environment with an explicit Mig-PPO action mask."""

    def __init__(
        self,
        config: Optional[MigPPOEnvConfig] = None,
        *,
        cost_estimator: Optional[MigPPOMigrationCostEstimator] = None,
        reward_calculator: Optional[MigPPORewardCalculator] = None,
        reward_weights: MigPPORewardWeights = MigPPORewardWeights(),
        reward_scales: MigPPORewardScales = MigPPORewardScales(),
    ) -> None:
        self.config = config or MigPPOEnvConfig()
        if self.config.num_instances < 2:
            raise ValueError("Mig-PPO requires at least two instances")
        if self.config.decision_interval_sec <= 0:
            raise ValueError("decision_interval_sec must be positive")
        self.cost_estimator = cost_estimator or (
            MigPPOMigrationCostEstimator.from_state_dict(self.config.migration_cost_estimator)
            if self.config.migration_cost_estimator else
            MigPPOMigrationCostEstimator(self.config.migration_alpha_ms, self.config.migration_beta_ms_per_block)
        )
        if self.config.migration_cost_csv:
            self.cost_estimator.fit_csv(self.config.migration_cost_csv)
            if self.cost_estimator.global_model is not None:
                self.config.calibration_status = "migration_cost_calibrated"
        self.reward_calculator = reward_calculator or MigPPORewardCalculator(
            reward_weights, reward_scales, self.config.ttft_slo_ms)
        self.mask_config = MigPPOActionMaskConfig(
            max_state_age_sec=self.config.max_state_age_sec,
            migration_cooldown_sec=self.config.migration_cooldown_sec,
            safety_margin_blocks=self.config.safety_margin_blocks,
            safety_margin_fraction=self.config.safety_margin_fraction,
            neutral_only=True,
        )
        self.rng = np.random.default_rng()
        self.output_rng = np.random.default_rng()
        self.py_rng = random.Random()
        self.scenario = self.config.workload_scenario
        self.now = 0.0
        self.wall_epoch = 1_800_000_000.0
        self.decision_count = 0
        self.request_count = 0
        self.action_count = 0
        self.arrival_rate = 0.0
        self.next_arrival = math.inf
        self.instances: list[_SimInstance] = []
        self.requests: dict[str, _Request] = {}
        self.arrival_rate_history = RollingArrivalRate()
        self.last_migration_time: Optional[float] = None
        self.pending_migration: Optional[_PendingMigration] = None
        self.migration_events: list[dict[str, object]] = []
        self.request_events: list[dict[str, object]] = []
        self.request_results: list[dict[str, object]] = []
        self.service_iterations: list[dict[str, object]] = []
        self.observed_ttft_samples_ms: list[float] = []
        self.observed_tpot_samples_ms: list[float] = []
        self._output_distribution_cache: dict[
            tuple[int, int], tuple[list[int], list[float]] | None
        ] = {}
        self._expected_output_token_cache: dict[tuple[int, int], int] = {}
        self._arrival_trace: list[dict[str, object]] = []
        self._arrival_trace_index = 0
        self._trace_routes: dict[str, str] = {}
        self._profile_cycle: list[tuple[int, int]] = []
        self._last_obs: Optional[MigPPOObservation] = None
        self._last_observation_time = -math.inf

    def _wall_time(self) -> float:
        return self.wall_epoch + self.now

    def _record_lifecycle_event(self, event: str, request: Optional[_Request] = None,
                                instance: Optional[_SimInstance] = None, *,
                                timestamp: Optional[float] = None,
                                timestamp_source: str = "simulator_state",
                                **details: object) -> None:
        if not self.config.trace_events_enabled:
            return
        event_time = self.now if timestamp is None else float(timestamp)
        row: dict[str, object] = {
            "timestamp_sec": event_time,
            "processed_at_sec": self.now,
            "timestamp_wall_sec": self.wall_epoch + event_time,
            "timestamp_source": timestamp_source,
            "event": event,
        }
        if request is not None:
            row.update({
                "request_id": request.request_id,
                "stage": request.stage,
                "output_tokens": request.output_tokens,
                "target_output_tokens": request.target_output_tokens,
                "output_token_budget": request.requested_output_tokens or request.target_output_tokens,
                "allocated_blocks": request.allocated_blocks,
                "migrating": request.migrating,
            })
        if instance is not None:
            row.update({
                "instance_id": instance.instance_id,
                "waiting_requests": len(instance.waiting),
                "running_requests": len(instance.running),
                "active_prefill_requests": sum(
                    item.stage in {"prefill", "resume_prefill"} for item in instance.running
                ),
                "active_decode_requests": sum(item.stage == "decode" for item in instance.running),
                "used_blocks": instance.used_blocks,
                "reserved_blocks": instance.reserved_blocks,
                "free_blocks": instance.free_blocks,
            })
        row.update(details)
        self.lifecycle_events.append(row)

    def _incoming_migration_slots(self, instance: _SimInstance) -> int:
        pending = self.pending_migration
        return int(pending is not None and pending.target_id == instance.instance_id)

    def _check_state_invariants(self) -> None:
        """Raise when request ownership, sequence slots, or KV accounting diverge."""
        locations: dict[str, tuple[str, str]] = {}
        for instance in self.instances:
            for request in instance.waiting:
                if request.request_id in locations:
                    raise AssertionError(f"request {request.request_id} appears in multiple queues")
                locations[request.request_id] = (instance.instance_id, "waiting")
                if self.requests.get(request.request_id) is not request or request.stage != "waiting":
                    raise AssertionError(f"waiting state mismatch for request {request.request_id}")
                if request.migrating:
                    raise AssertionError(f"waiting request {request.request_id} cannot be migrating")
            for request in instance.running:
                if request.request_id in locations:
                    raise AssertionError(f"request {request.request_id} appears in multiple queues")
                locations[request.request_id] = (instance.instance_id, "running")
                if self.requests.get(request.request_id) is not request:
                    raise AssertionError(f"running request {request.request_id} is not registered")
                if request.stage not in {"prefill", "resume_prefill", "decode"}:
                    raise AssertionError(f"running request {request.request_id} has stage {request.stage!r}")
            if len(instance.running) + self._incoming_migration_slots(instance) > self.config.active_sequence_capacity:
                raise AssertionError(f"sequence capacity exceeded on {instance.instance_id}")
            if min(instance.used_blocks, instance.reserved_blocks) < 0:
                raise AssertionError(f"negative KV accounting on {instance.instance_id}")
            if instance.used_blocks + instance.reserved_blocks > instance.total_blocks:
                raise AssertionError(f"KV capacity exceeded on {instance.instance_id}")
            if self.config.kv_dynamics_enabled:
                expected_used = sum(request.allocated_blocks for request in instance.running)
                if instance.used_blocks != expected_used:
                    raise AssertionError(
                        f"used KV mismatch on {instance.instance_id}: {instance.used_blocks} != {expected_used}"
                    )
                expected_reserved = (
                    self.pending_migration.blocks
                    if self.pending_migration is not None
                    and self.pending_migration.target_id == instance.instance_id else 0
                )
                if instance.reserved_blocks != expected_reserved:
                    raise AssertionError(
                        f"reserved KV mismatch on {instance.instance_id}: "
                        f"{instance.reserved_blocks} != {expected_reserved}"
                    )
            elif instance.used_blocks or instance.reserved_blocks:
                raise AssertionError(f"KV must remain disabled on {instance.instance_id}")
        if set(locations) != set(self.requests):
            missing = sorted(set(self.requests) - set(locations))
            extra = sorted(set(locations) - set(self.requests))
            raise AssertionError(f"request registry/queue mismatch; unqueued={missing}, unregistered={extra}")
        if self.pending_migration is not None:
            pending = self.pending_migration
            source = next((item for item in self.instances if item.instance_id == pending.source_id), None)
            target = next((item for item in self.instances if item.instance_id == pending.target_id), None)
            if source is None or target is None:
                raise AssertionError("pending migration references an unknown instance")
            request = self.requests.get(pending.request_id)
            if request is not None and (request not in source.running or not request.migrating):
                raise AssertionError(f"pending migration ownership mismatch for {pending.request_id}")
            if not source.is_migrating or not target.is_migrating:
                raise AssertionError("pending migration is missing an instance migration flag")
        elif any(instance.is_migrating for instance in self.instances):
            raise AssertionError("instance migration flag is set without a pending migration")

    def _blocks_for_tokens(self, tokens: int) -> int:
        return max(1, int(math.ceil(max(1, tokens) / max(1, self.config.block_size_tokens))))

    def _choose_scenario(self) -> str:
        if self.config.workload_scenario != "mix":
            return self.config.workload_scenario
        if self.config.target_qps_levels:
            target_qps = float(self.rng.choice(self.config.target_qps_levels))
            return f"qps_{target_qps:g}"
        return str(self.rng.choice(("low", "medium", "high"), p=(0.34, 0.33, 0.33)))

    def _scenario_arrival_rate(self, scenario: str) -> float:
        if scenario in self.config.arrival_rates_rps:
            return float(self.config.arrival_rates_rps[scenario])
        if scenario.startswith("qps_"):
            try:
                return float(scenario.removeprefix("qps_"))
            except ValueError:
                pass
        raise ValueError(f"unknown Mig-PPO workload scenario: {scenario}")

    def _sample_prompt_response(self) -> tuple[int, int]:
        # Shuffle a Cartesian-product cycle so every 9 consecutive arrivals
        # contain each of the configured prompt/response pairs exactly once.
        if not self._profile_cycle:
            self._profile_cycle = list(product(self.config.prompt_profiles, self.config.output_profiles))
            self.py_rng.shuffle(self._profile_cycle)
        return self._profile_cycle.pop()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.output_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4D4947]))
            self.py_rng = random.Random(seed)
        if options and options.get("target_qps") is not None:
            target_qps = float(options["target_qps"])
            self.scenario = f"qps_{target_qps:g}"
        elif options and options.get("scenario"):
            self.scenario = str(options["scenario"])
        else:
            self.scenario = self._choose_scenario()
        self.arrival_rate = self._scenario_arrival_rate(self.scenario)
        self.now = 0.0
        self.decision_count = 0
        self.request_count = 0
        self.action_count = 0
        self.wall_epoch = 1_800_000_000.0 + float(seed or 0) * 100_000.0
        self.instances = [
            _SimInstance(f"instance_{idx}", self.config.total_kv_blocks,
                         float(self.config.instance_capacity_scale))
            for idx in range(self.config.num_instances)
        ]
        self.requests = {}
        self.arrival_rate_history = RollingArrivalRate(window_sec=60.0)
        self.arrival_rate_history.start(self._wall_time())
        self.last_migration_time = None
        self.pending_migration = None
        self.migration_events = []
        self.request_events = []
        self.lifecycle_events = []
        self.request_results = []
        self.service_iterations = []
        self.observed_ttft_samples_ms = []
        self.observed_tpot_samples_ms = []
        self._profile_cycle = []
        self._arrival_trace = [dict(item) for item in (options or {}).get("arrival_trace", [])]
        self._arrival_trace.sort(key=lambda item: float(item["arrival_time"]))
        self._arrival_trace_index = 0
        self._trace_routes = {str(key): str(value) for key, value in
                              (options or {}).get("request_routes", {}).items()}
        self.next_arrival = self._sample_next_arrival()
        instance_ids = (options or {}).get("instance_ids")
        if instance_ids:
            if len(instance_ids) != len(self.instances):
                raise ValueError("instance_ids length must equal num_instances")
            for instance, instance_id in zip(self.instances, instance_ids):
                instance.instance_id = str(instance_id)
        self._last_obs = self._observation()
        self._last_observation_time = self.now
        return self._last_obs, {
            "algorithm": "Mig-PPO", "scenario": self.scenario,
            "arrival_rate_rps": self.arrival_rate, "target_qps": self.arrival_rate,
            "calibration_status": self.config.calibration_status,
            "cost_model": self.cost_estimator.state_dict(),
        }

    def _sample_next_arrival(self) -> float:
        if self._arrival_trace:
            if self._arrival_trace_index >= len(self._arrival_trace):
                return math.inf
            return float(self._arrival_trace[self._arrival_trace_index]["arrival_time"])
        if self.arrival_rate <= 0:
            return math.inf
        return self.now + float(self.rng.exponential(1.0 / self.arrival_rate))

    def _instance_views(self) -> dict[str, _InstanceView]:
        return {instance.instance_id: _InstanceView(self, instance) for instance in self.instances}

    def _observation(self) -> MigPPOObservation:
        migration_count = int(self.pending_migration is not None)
        obs = build_mig_ppo_observation(
            self._instance_views(), now=self._wall_time(),
            arrival_rate_rps=self.arrival_rate_history.rate(self._wall_time()),
            migrations_in_progress=migration_count,
            last_migration_time=(self.wall_epoch + self.last_migration_time)
            if self.last_migration_time is not None else None,
            cost_estimator=self.cost_estimator,
            mask_config=self.mask_config,
            migration_in_progress=migration_count > 0,
        )
        return obs

    def _sr_candidate(self, instance: _SimInstance) -> Optional[_Request]:
        candidates = [request for request in instance.running
                      if request.stage == "decode" and not request.migrating]
        return min(candidates, key=lambda request: request.request_len, default=None)

    def _dispatch_arrival(self, request: _Request) -> None:
        candidates = self.instances
        forced_instance = self._trace_routes.get(request.request_id)
        if forced_instance is not None:
            dst = next((instance for instance in candidates if instance.instance_id == forced_instance), None)
            if dst is None:
                raise ValueError(f"arrival trace route references unknown instance {forced_instance!r}")
            request.waiting_since = self.now
            dst.waiting.append(request)
            self.requests[request.request_id] = request
            if not math.isfinite(dst.next_step_time):
                dst.next_step_time = self.now
            self._record_lifecycle_event("waiting", request, dst, timestamp_source="arrival_event",
                                         queue_position=len(dst.waiting), forced_route=True)
            return
        skew = 0.0 if self.scenario == "low" else self.config.routing_skew_probability
        if self.rng.random() < skew:
            # A small fraction of direct assignments creates realistic local
            # arrival bursts instead of forcing every state to be balanced.
            dst = candidates[int(self.rng.integers(0, len(candidates)))]
        else:
            def projected_work(instance: _SimInstance) -> float:
                running_work = sum(
                    max(1, self._expected_output_tokens(request) - request.output_tokens)
                    * self.config.decode_ms_per_request * instance.capacity_scale
                    for request in instance.running if not request.migrating
                )
                waiting_work = sum(
                    self.config.prefill_intercept_ms + self.config.prefill_ms_per_token * req.prompt_tokens
                    for req in instance.waiting
                )
                return (running_work + waiting_work) * max(instance.capacity_scale, 1e-3)
            dst = min(candidates, key=projected_work)
        request.waiting_since = self.now
        dst.waiting.append(request)
        self.requests[request.request_id] = request
        if not math.isfinite(dst.next_step_time):
            dst.next_step_time = self.now
        self._record_lifecycle_event("waiting", request, dst, timestamp_source="arrival_event",
                                     queue_position=len(dst.waiting), forced_route=False)

    @staticmethod
    def _output_profile_key(prompt_tokens: int, output_budget: int) -> str:
        return f"{int(prompt_tokens)}:{int(output_budget)}"

    def _output_distribution(self, prompt_tokens: int, output_budget: int) -> tuple[list[int], list[float]] | None:
        cache_key = (int(prompt_tokens), int(output_budget))
        if cache_key in self._output_distribution_cache:
            return self._output_distribution_cache[cache_key]
        model = self.config.output_length_model or {}
        profiles = model.get("profiles", {})
        if not isinstance(profiles, Mapping):
            self._output_distribution_cache[cache_key] = None
            return None
        profile = profiles.get(self._output_profile_key(prompt_tokens, output_budget))
        if not isinstance(profile, Mapping):
            profile = profiles.get(f"*:{int(output_budget)}")
        if not isinstance(profile, Mapping):
            self._output_distribution_cache[cache_key] = None
            return None
        try:
            lengths = [max(1, min(int(output_budget), int(value)))
                       for value in profile.get("lengths", [])]
            counts = [max(0, int(value)) for value in profile.get("counts", [])]
        except (TypeError, ValueError, OverflowError):
            self._output_distribution_cache[cache_key] = None
            return None
        if not lengths or len(lengths) != len(counts) or sum(counts) <= 0:
            self._output_distribution_cache[cache_key] = None
            return None
        total = float(sum(counts))
        result = lengths, [count / total for count in counts]
        self._output_distribution_cache[cache_key] = result
        return result

    def _expected_output_tokens(self, request: _Request) -> int:
        budget = int(request.requested_output_tokens or request.target_output_tokens)
        cache_key = (int(request.prompt_tokens), budget)
        if cache_key in self._expected_output_token_cache:
            return self._expected_output_token_cache[cache_key]
        distribution = self._output_distribution(request.prompt_tokens, budget)
        if distribution is None:
            self._expected_output_token_cache[cache_key] = budget
            return budget
        lengths, probabilities = distribution
        expected = max(1, int(round(sum(length * probability
                                        for length, probability in zip(lengths, probabilities)))))
        self._expected_output_token_cache[cache_key] = expected
        return expected

    def _sample_traced_output_length(self, prompt_tokens: int, output_budget: int) -> int:
        distribution = self._output_distribution(prompt_tokens, output_budget)
        if distribution is None:
            return max(1, int(output_budget))
        lengths, probabilities = distribution
        return int(self.output_rng.choice(lengths, p=probabilities))

    def _process_arrival(self) -> None:
        trace_entry = None
        if self._arrival_trace_index < len(self._arrival_trace):
            trace_entry = self._arrival_trace[self._arrival_trace_index]
            self._arrival_trace_index += 1
            prompt = int(trace_entry["prompt_tokens"])
            output_budget = int(trace_entry.get("output_budget_tokens", trace_entry["output_tokens"]))
            fixed_output = trace_entry.get("realized_output_tokens")
            if fixed_output is None:
                output = self._sample_traced_output_length(prompt, output_budget)
                output_source = "calibration_length_sample"
            else:
                output = int(fixed_output)
                if output < 1 or output > output_budget:
                    raise ValueError(
                        f"trace realized output length {output} is outside request budget 1..{output_budget}"
                    )
                output_source = str(trace_entry.get("realized_output_source", "fixed_trace_sample"))
        else:
            prompt, output = self._sample_prompt_response()
            output_budget = output
            output_source = "configured_profile"
        self.request_count += 1
        request = _Request(
            request_id=(str(trace_entry.get("request_id", f"req-{self.request_count:08d}"))
                        if trace_entry else f"req-{self.request_count:08d}"), arrival_time=self.now,
            prompt_tokens=prompt, target_output_tokens=output,
            requested_output_tokens=output_budget,
        )
        self._record_lifecycle_event(
            "arrival", request, timestamp=self.now, timestamp_source="request_arrival_trace",
            scheduled_arrival_sec=(trace_entry.get("scheduled_arrival_sec") if trace_entry else None),
            actual_send_sec=(trace_entry.get("actual_send_sec") if trace_entry else None),
            output_length_source=output_source,
        )
        self._dispatch_arrival(request)
        self.arrival_rate_history.record(self._wall_time())
        self.request_events.append({
            "timestamp": self._wall_time(), "event": "arrival", "request_id": request.request_id,
            "prompt_tokens": prompt, "output_token_budget": output_budget,
            "target_output_tokens": output, "output_length_source": output_source,
        })
        self.next_arrival = self._sample_next_arrival()

    def _prefill_latency_ms(self, token_count: int, instance: _SimInstance) -> float:
        model = self.config.service_time_model or {}
        if self.config.service_model_mode == "calibrated" and model.get("prefill"):
            batch = max(1, int(getattr(self, "_active_prefill_batch", 1)))
            features = {"batch_tokens": float(token_count), "total_context_tokens": float(token_count),
                        "batch_size": float(batch), "running_requests": float(getattr(self, "_active_running_count", 0)),
                        "avg_context_tokens": float(token_count) / batch}
            value = predict_service_model(model["prefill"], features)  # type: ignore[arg-type]
            return max(self.config.min_engine_step_ms, value * instance.capacity_scale)
        return max(self.config.min_engine_step_ms,
                   (self.config.prefill_intercept_ms + self.config.prefill_ms_per_token * token_count)
                   * instance.capacity_scale)

    def _decode_latency_ms(self, batch_size: int, total_context_tokens: int, instance: _SimInstance) -> float:
        model = self.config.service_time_model or {}
        if self.config.service_model_mode == "calibrated" and model.get("decode"):
            features = {"batch_size": float(batch_size), "total_context_tokens": float(total_context_tokens),
                        "avg_context_tokens": float(total_context_tokens) / max(1, batch_size),
                        "batch_tokens": float(batch_size), "running_requests": float(len(instance.running))}
            value = predict_service_model(model["decode"], features)  # type: ignore[arg-type]
            return max(self.config.min_engine_step_ms, value * instance.capacity_scale)
        return max(self.config.min_engine_step_ms,
                   (self.config.decode_intercept_ms
                    + self.config.decode_ms_per_request * batch_size
                    + self.config.decode_ms_per_context_token * total_context_tokens)
                   * instance.capacity_scale)

    def _schedule_waiting(self, instance: _SimInstance) -> list[_Request]:
        admitted: list[_Request] = []
        batched_prompt_tokens = 0
        sequence_limit = max(0, self.config.active_sequence_capacity - self._incoming_migration_slots(instance))
        while instance.waiting and len(instance.running) < sequence_limit:
            request = instance.waiting[0]
            blocks = self._blocks_for_tokens(request.request_len)
            if self.config.kv_dynamics_enabled and blocks > instance.free_blocks:
                break
            if admitted and batched_prompt_tokens + request.request_len > self.config.max_num_batched_tokens:
                break
            instance.waiting.pop(0)
            request.stage = "prefill" if request.output_tokens == 0 else "resume_prefill"
            request.allocated_blocks = blocks
            request.waiting_since = 0.0
            if self.config.kv_dynamics_enabled:
                instance.used_blocks += request.allocated_blocks
            instance.running.append(request)
            admitted.append(request)
            batched_prompt_tokens += request.request_len
            self._record_lifecycle_event(
                "admitted", request, instance, timestamp_source="scheduler_admission",
                queue_wait_ms=max(0.0, (self.now - request.arrival_time) * 1000.0),
                admission_stage=request.stage, batch_position=len(admitted),
            )
        return admitted

    def _run_engine_step(self, instance: _SimInstance, metrics: MigPPORewardMetrics,
                         window_start: float, window_end: float) -> None:
        if instance.step_in_flight:
            self._record_lifecycle_event(
                "engine_step_completed", instance=instance,
                timestamp=instance.pending_step_end_time, timestamp_source="simulated_gpu_batch_completion",
                step_id=instance.pending_step_id,
                batch_start_sec=instance.pending_step_start_time,
                batch_duration_ms=max(0.0, (self.now - instance.pending_step_start_time) * 1000.0),
                clock_advance_ms=max(0.0, (instance.pending_step_end_time - instance.pending_step_start_time) * 1000.0),
            )
            for request in instance.pending_prefill:
                if request in instance.running and not request.migrating:
                    if request.stage == "resume_prefill":
                        request.stage = "decode"
                        self._record_lifecycle_event(
                            "resume_prefill_completed", request, instance,
                            timestamp=instance.pending_step_end_time,
                            timestamp_source="simulated_gpu_batch_completion",
                            step_id=instance.pending_step_id,
                        )
                    else:
                        self._record_token(request, instance.pending_prefill_time, metrics, window_start, window_end)
                elif request in instance.running and request.migrating:
                    self._record_lifecycle_event(
                        "prefill_output_suppressed_during_migration", request, instance,
                        timestamp=instance.pending_step_end_time,
                        timestamp_source="simulated_gpu_batch_completion",
                        step_id=instance.pending_step_id,
                    )
            for request in instance.pending_decode:
                if request in instance.running and not request.migrating:
                    self._record_token(request, instance.pending_decode_time, metrics, window_start, window_end)
                elif request in instance.running and request.migrating:
                    self._record_lifecycle_event(
                        "decode_output_suppressed_during_migration", request, instance,
                        timestamp=instance.pending_step_end_time,
                        timestamp_source="simulated_gpu_batch_completion",
                        step_id=instance.pending_step_id,
                    )
            instance.pending_prefill = []
            instance.pending_decode = []
            instance.step_in_flight = False

        admitted = self._schedule_waiting(instance)
        prefill_tokens = sum(request.request_len for request in admitted)
        decoders = [request for request in instance.running
                    if request.stage == "decode" and not request.migrating]
        self._active_prefill_batch = len(admitted)
        self._active_running_count = len(instance.running)
        prefill_ms = self._prefill_latency_ms(prefill_tokens, instance) if admitted else 0.0
        decode_iteration_ms = self._decode_latency_ms(
            len(decoders), sum(request.request_len for request in decoders), instance
        ) if decoders else 0.0
        decode_token_interval_ms = decode_iteration_ms * self.config.decode_token_interval_scale
        step_ms = max(self.config.min_engine_step_ms, prefill_ms, decode_token_interval_ms)
        instance.service_step_count += 1
        step_id = instance.service_step_count
        step_start = self.now
        step_end = self.now + step_ms / 1000.0
        instance.last_step_ms = step_ms
        instance.last_batch_tokens = prefill_tokens + len(decoders)
        self._record_lifecycle_event(
            "engine_step_scheduled", instance=instance,
            timestamp=step_start, timestamp_source="simulated_batch_start",
            step_id=step_id, batch_end_sec=step_end, batch_duration_ms=step_ms,
            prefill_request_ids=[request.request_id for request in admitted],
            decode_request_ids=[request.request_id for request in decoders],
            prefill_iteration_ms=prefill_ms, decode_iteration_ms=decode_iteration_ms,
            decode_token_interval_ms=decode_token_interval_ms,
            decode_token_interval_scale=self.config.decode_token_interval_scale,
            waiting_after_admission=len(instance.waiting),
            kv_used_blocks=instance.used_blocks, kv_reserved_blocks=instance.reserved_blocks,
        )
        if admitted:
            self.service_iterations.append({
                "timestamp_sec": step_start, "batch_start_sec": step_start, "batch_end_sec": step_end,
                "instance_id": instance.instance_id, "step_id": step_id, "phase": "prefill",
                "batch_size": len(admitted), "request_ids": [request.request_id for request in admitted],
                "running_requests": len(instance.running), "waiting_requests": len(instance.waiting),
                "batch_tokens": prefill_tokens, "predicted_ms": prefill_ms,
                "step_duration_ms": step_ms, "clock_advanced_once": True,
                "kv_used_blocks": instance.used_blocks, "kv_reserved_blocks": instance.reserved_blocks,
            })
        if decoders:
            self.service_iterations.append({
                "timestamp_sec": step_start, "batch_start_sec": step_start, "batch_end_sec": step_end,
                "instance_id": instance.instance_id, "step_id": step_id, "phase": "decode",
                "batch_size": len(decoders), "request_ids": [request.request_id for request in decoders],
                "running_requests": len(instance.running), "waiting_requests": len(instance.waiting),
                "batch_tokens": len(decoders),
                "total_context_tokens": sum(req.request_len for req in decoders),
                "predicted_ms": decode_iteration_ms,
                "effective_token_interval_ms": decode_token_interval_ms,
                "step_duration_ms": step_ms, "clock_advanced_once": True,
                "kv_used_blocks": instance.used_blocks, "kv_reserved_blocks": instance.reserved_blocks,
            })
        if not admitted and not decoders:
            instance.next_step_time = math.inf
            instance.step_in_flight = False
            return
        instance.pending_prefill = list(admitted)
        instance.pending_decode = list(decoders)
        # One GPU forward batch emits all of its request outputs at the batch
        # completion event, whose wall time is the maximum concurrent phase
        # prediction. Do not timestamp a faster phase before that event.
        instance.pending_prefill_time = step_end
        instance.pending_decode_time = step_end
        instance.step_in_flight = True
        instance.pending_step_id = step_id
        instance.pending_step_start_time = step_start
        instance.pending_step_end_time = step_end
        instance.next_step_time = step_end

    def _record_token(self, request: _Request, timestamp: float, metrics: MigPPORewardMetrics,
                      window_start: float, window_end: float) -> None:
        request.output_tokens += 1
        if self.config.trace_events_enabled:
            request.token_timestamps_sec.append(float(timestamp))
        if request.first_token_time is None:
            request.first_token_time = timestamp
            ttft_ms = max(0.0, (timestamp - request.arrival_time) * 1000.0)
            request.last_token_time = timestamp
            self.observed_ttft_samples_ms.append(ttft_ms)
            request.stage = "decode"
            if window_start <= timestamp <= window_end:
                metrics.observed_ttft_ms.append(ttft_ms)
                if ttft_ms > self.config.ttft_slo_ms:
                    metrics.ttft_slo_violations += 1
            self._record_lifecycle_event(
                "first_token", request, self._request_instance(request), timestamp=timestamp,
                timestamp_source="simulated_gpu_batch_completion", ttft_ms=ttft_ms,
                generated_token_index=request.output_tokens,
            )
        else:
            tpot_ms = max(0.0, (timestamp - float(request.last_token_time)) * 1000.0)
            request.tpot_samples_ms.append(tpot_ms)
            self.observed_tpot_samples_ms.append(tpot_ms)
            request.last_token_time = timestamp
            if window_start <= timestamp <= window_end:
                metrics.observed_tpot_ms.append(tpot_ms)
                if tpot_ms > self.config.tpot_slo_ms:
                    metrics.tpot_slo_violations += 1
            if self.config.trace_token_events:
                self._record_lifecycle_event(
                    "decode_token", request, self._request_instance(request), timestamp=timestamp,
                    timestamp_source="simulated_gpu_batch_completion", tpot_ms=tpot_ms,
                    generated_token_index=request.output_tokens,
                )
        if window_start <= timestamp <= window_end:
            metrics.completed_output_tokens += 1
        if request.output_tokens >= request.target_output_tokens:
            self._complete_request(request, timestamp, metrics, window_start, window_end)
        else:
            self._grow_kv_or_preempt(self._request_instance(request), request)

    def _request_instance(self, request: _Request) -> _SimInstance:
        for instance in self.instances:
            if request in instance.running:
                return instance
        raise RuntimeError(f"request {request.request_id} is not running")

    def _complete_request(self, request: _Request, timestamp: float, metrics: MigPPORewardMetrics,
                          window_start: float, window_end: float) -> None:
        if self.requests.get(request.request_id) is not request:
            return
        if window_start <= timestamp <= window_end:
            metrics.completed_requests += 1
        owner = next((instance for instance in self.instances if request in instance.running), None)
        if owner is None:
            raise AssertionError(f"completing request {request.request_id} is not running")
        owner.running.remove(request)
        released_blocks = request.allocated_blocks
        if self.config.kv_dynamics_enabled:
            owner.used_blocks -= released_blocks
        request.allocated_blocks = 0
        self._record_lifecycle_event(
            "kv_released", request, owner, timestamp=timestamp,
            timestamp_source="request_completion", released_blocks=released_blocks,
            release_reason="completion",
        )
        if not owner.running and not owner.waiting and not owner.step_in_flight:
            owner.next_step_time = math.inf
        elif not owner.step_in_flight:
            owner.next_step_time = min(owner.next_step_time, timestamp)
        request.stage = "finished"
        self._record_lifecycle_event("completed", request, timestamp=timestamp,
                                     timestamp_source="request_completion",
                                     completed_at_sec=timestamp)
        self.request_results.append(self._request_result(request, "completed", timestamp))
        self.request_events.append({
            "timestamp": self.wall_epoch + timestamp, "event": "completed", "request_id": request.request_id,
            "output_tokens": request.output_tokens,
        })
        self.requests.pop(request.request_id, None)

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        return float(np.quantile(values, quantile)) if values else 0.0

    def _request_result(self, request: _Request, status: str, timestamp: float) -> dict[str, object]:
        ttft = ((request.first_token_time - request.arrival_time) * 1000.0
                if request.first_token_time is not None else None)
        result: dict[str, object] = {"request_id": request.request_id, "status": status,
                "arrival_sec": request.arrival_time, "completion_sec": timestamp,
                "arrival_wall_sec": self.wall_epoch + request.arrival_time,
                "completion_wall_sec": self.wall_epoch + timestamp,
                "e2e_ms": max(0.0, (timestamp - request.arrival_time) * 1000.0),
                "ttft_ms": max(0.0, float(ttft)) if ttft is not None else None,
                "tpot_mean_ms": float(np.mean(request.tpot_samples_ms)) if request.tpot_samples_ms else None,
                "tpot_p95_ms": self._percentile(request.tpot_samples_ms, .95),
                "ttft_start_sec": request.arrival_time,
                "ttft_end_sec": request.first_token_time,
                "ttft_timestamp_source": "request_arrival_trace_and_simulated_gpu_batch_completion",
                "tpot_start_sec": request.first_token_time,
                "tpot_end_sec": request.last_token_time,
                "tpot_timestamp_source": "simulated_gpu_batch_completion",
                "tpot_sample_count": len(request.tpot_samples_ms),
                "input_tokens": request.prompt_tokens, "target_output_tokens": request.target_output_tokens,
                "output_token_budget": request.requested_output_tokens or request.target_output_tokens,
                "generated_tokens": request.output_tokens, "preemptions": request.preemptions}
        if self.config.trace_events_enabled:
            result["token_timestamps_sec"] = list(request.token_timestamps_sec)
            result["token_tpot_ms"] = list(request.tpot_samples_ms)
        return result

    def _grow_kv_or_preempt(self, instance: _SimInstance, request: _Request) -> None:
        wanted_blocks = self._blocks_for_tokens(request.request_len)
        if not self.config.kv_dynamics_enabled or wanted_blocks <= request.allocated_blocks:
            return
        delta = wanted_blocks - request.allocated_blocks
        if delta <= instance.free_blocks:
            request.allocated_blocks = wanted_blocks
            instance.used_blocks += delta
            self._record_lifecycle_event("kv_grown", request, instance, timestamp_source="token_allocation",
                                         added_blocks=delta, wanted_blocks=wanted_blocks)
            return
        # Approximate vLLM KV-pressure preemption: pause the least-recently
        # admitted, non-migrating decode request and return it to the queue.
        victim = next((item for item in reversed(instance.running)
                       if item is not request and item.stage == "decode" and not item.migrating), None)
        if victim is not None:
            instance.running.remove(victim)
            released_blocks = victim.allocated_blocks
            instance.used_blocks -= released_blocks
            victim.allocated_blocks = 0
            victim.stage = "waiting"
            victim.waiting_since = self.now
            victim.preemptions += 1
            instance.waiting.insert(0, victim)
            instance.preempted_requests += 1
            self._record_lifecycle_event("preempted", victim, instance,
                                         timestamp_source="kv_pressure_preemption",
                                         released_blocks=released_blocks,
                                         preemptions=victim.preemptions,
                                         resume_stage="resume_prefill")
            if delta <= instance.free_blocks:
                request.allocated_blocks = wanted_blocks
                instance.used_blocks += delta
                self._record_lifecycle_event("kv_grown", request, instance,
                                             timestamp_source="token_allocation",
                                             added_blocks=delta, wanted_blocks=wanted_blocks)
                return
        if request in instance.running and not request.migrating:
            instance.running.remove(request)
            released_blocks = request.allocated_blocks
            instance.used_blocks -= released_blocks
            request.allocated_blocks = 0
            request.stage = "waiting"
            request.waiting_since = self.now
            request.preemptions += 1
            instance.waiting.insert(0, request)
            instance.preempted_requests += 1
            self._record_lifecycle_event("preempted", request, instance,
                                         timestamp_source="kv_pressure_preemption",
                                         released_blocks=released_blocks,
                                         preemptions=request.preemptions,
                                         resume_stage="resume_prefill")

    def _expire_requests(self, metrics: MigPPORewardMetrics, window_start: float, window_end: float) -> None:
        if self.config.request_timeout_sec is None:
            return
        expired = [request for request in self.requests.values()
                   if self.now - request.arrival_time >= self.config.request_timeout_sec]
        for request in expired:
            for instance in self.instances:
                if request in instance.waiting:
                    instance.waiting.remove(request)
                if request in instance.running:
                    instance.running.remove(request)
                    released_blocks = request.allocated_blocks
                    if self.config.kv_dynamics_enabled:
                        instance.used_blocks -= released_blocks
                    request.allocated_blocks = 0
                    self._record_lifecycle_event(
                        "kv_released", request, instance, timestamp_source="request_timeout",
                        released_blocks=released_blocks, release_reason="timeout",
                    )
                if not instance.step_in_flight:
                    if instance.running or instance.waiting:
                        if not math.isfinite(instance.next_step_time):
                            instance.next_step_time = self.now
                    else:
                        instance.next_step_time = math.inf
            request.stage = "failed"
            self.request_results.append(self._request_result(request, "timeout", self.now))
            self._record_lifecycle_event("failed", request, timestamp_source="request_timeout",
                                         failure_reason="timeout")
            self.requests.pop(request.request_id, None)
            metrics.timeout_requests += 1
            self.request_events.append({
                "timestamp": self._wall_time(), "event": "timeout", "request_id": request.request_id,
                "preemptions": request.preemptions,
            })

    def _start_migration(self, action: int, obs: MigPPOObservation, metrics: MigPPORewardMetrics) -> Optional[str]:
        if action == 0:
            return None
        pair_idx = action - 1
        if pair_idx < 0 or pair_idx >= len(obs.action_mask.pairs) or not obs.pair_mask[pair_idx]:
            return "action_mask_rejected"
        if self.pending_migration is not None:
            return "migration_in_progress"
        src_id, dst_id = obs.action_mask.pairs[pair_idx]
        src = next(instance for instance in self.instances if instance.instance_id == src_id)
        dst = next(instance for instance in self.instances if instance.instance_id == dst_id)
        candidate = self._sr_candidate(src)
        if candidate is None or candidate.migrating:
            return "candidate_changed"
        if len(dst.running) + self._incoming_migration_slots(dst) >= self.config.active_sequence_capacity:
            return "destination_sequence_capacity"
        blocks = (candidate.allocated_blocks if self.config.kv_dynamics_enabled
                  else self._blocks_for_tokens(candidate.request_len))
        safe_free = dst.free_blocks - sum(self._blocks_for_tokens(req.request_len) for req in dst.waiting)
        reserve = max(self.config.safety_margin_blocks,
                      int(math.ceil(dst.total_blocks * self.config.safety_margin_fraction)))
        if safe_free - blocks < reserve:
            return "capacity_recheck_failed"
        if self.last_migration_time is not None \
                and self.now - self.last_migration_time < self.config.migration_cooldown_sec:
            return "cooldown"

        prediction = self.cost_estimator.predict(src_id, dst_id, blocks)
        uncertainty = prediction.uncertainty_ms * max(0.0, self.config.migration_uncertainty_scale)
        actual_ms = max(1.0, float(self.rng.normal(prediction.duration_ms, uncertainty))
                        if uncertainty > 0 else prediction.duration_ms)
        action_id = f"mig-{self.decision_count:06d}-{self.action_count:06d}"
        self.action_count += 1
        candidate.migrating = True
        src.is_migrating = True
        dst.is_migrating = True
        if self.config.kv_dynamics_enabled:
            dst.reserved_blocks += blocks
        self.pending_migration = _PendingMigration(
            action_id=action_id, source_id=src_id, target_id=dst_id,
            request_id=candidate.request_id, blocks=blocks,
            start_time=self.now, end_time=self.now + actual_ms / 1000.0,
            predicted_ms=prediction.duration_ms, actual_ms=actual_ms,
        )
        metrics.migration_operations += 1
        self._record_lifecycle_event(
            "migration_started", candidate, src, timestamp_source="migration_action",
            action_id=action_id, target_instance_id=dst.instance_id, blocks_reserved=blocks,
            migration_end_sec=self.pending_migration.end_time,
            source_used_blocks=src.used_blocks, target_used_blocks=dst.used_blocks,
            target_reserved_blocks=dst.reserved_blocks,
        )
        return None

    def start_replay_migration(self, event: Mapping[str, object]) -> str:
        """Inject one logged migration attempt for fixed-trajectory validation.

        This consumes event timing, source, destination, request, duration and
        terminal status as exogenous replay inputs. It must not be used to
        evaluate the fitted migration predictor.
        """
        if self.pending_migration is not None:
            return "migration_in_progress"
        request = self.requests.get(str(event.get("request_id", "")))
        src_id, dst_id = str(event.get("source_instance", "")), str(event.get("target_instance", ""))
        src = next((instance for instance in self.instances if instance.instance_id == src_id), None)
        dst = next((instance for instance in self.instances if instance.instance_id == dst_id), None)
        if src is None or dst is None or request is None:
            return "missing_request_or_instance"
        if src is dst:
            return "source_equals_target"
        if request not in src.running or request.stage != "decode" or request.migrating:
            return "request_not_running_at_logged_source"
        if len(dst.running) + self._incoming_migration_slots(dst) >= self.config.active_sequence_capacity:
            return "destination_sequence_capacity"
        observed_blocks = max(0, int(event.get("blocks", request.allocated_blocks)))
        blocks = max(observed_blocks, request.allocated_blocks) if self.config.kv_dynamics_enabled else observed_blocks
        if blocks > dst.free_blocks:
            return "destination_capacity_mismatch"
        duration_ms = max(1.0, float(event.get("cost_ms", 1.0)))
        self.action_count += 1
        request.migrating = True
        src.is_migrating = True
        dst.is_migrating = True
        if self.config.kv_dynamics_enabled:
            dst.reserved_blocks += blocks
        status = str(event.get("status", "FINISHED")).split(".")[-1].lower()
        self.pending_migration = _PendingMigration(
            action_id=f"replay-{self.action_count:06d}", source_id=src_id, target_id=dst_id,
            request_id=request.request_id, blocks=blocks, start_time=self.now,
            end_time=self.now + duration_ms / 1000.0, predicted_ms=duration_ms,
            actual_ms=duration_ms, replay_status=status, observed_blocks=observed_blocks,
        )
        self._record_lifecycle_event(
            "migration_started", request, src, timestamp_source="fixed_event_replay",
            action_id=self.pending_migration.action_id, target_instance_id=dst.instance_id,
            blocks_reserved=blocks, observed_blocks=observed_blocks,
            migration_end_sec=self.pending_migration.end_time,
            source_used_blocks=src.used_blocks, target_used_blocks=dst.used_blocks,
            target_reserved_blocks=dst.reserved_blocks,
        )
        return "started"

    def _finish_migration(self, metrics: MigPPORewardMetrics, window_start: float, window_end: float) -> None:
        pending = self.pending_migration
        if pending is None or self.now + 1e-9 < pending.end_time:
            return
        src = next(instance for instance in self.instances if instance.instance_id == pending.source_id)
        dst = next(instance for instance in self.instances if instance.instance_id == pending.target_id)
        request = self.requests.get(pending.request_id)
        status = "success"
        if request is None or request not in src.running or not request.migrating:
            status = "aborted"
        elif pending.replay_status is not None and pending.replay_status not in {"finished", "success"}:
            status = "failed" if "fail" in pending.replay_status else "aborted"
        elif pending.replay_status is None and self.rng.random() < self.config.migration_failure_probability:
            status = "failed"
        if status == "success":
            src.running.remove(request)
            if self.config.kv_dynamics_enabled:
                src.used_blocks -= request.allocated_blocks
                dst.reserved_blocks -= pending.blocks
                dst.used_blocks += request.allocated_blocks
            request.migrating = False
            dst.running.append(request)
            request.stage = "decode"
            if not dst.step_in_flight:
                dst.next_step_time = min(dst.next_step_time, self.now)
            metrics.migration_succeeded += 1
        else:
            if self.config.kv_dynamics_enabled:
                dst.reserved_blocks -= pending.blocks
            if request is not None:
                request.migrating = False
            if status == "failed":
                metrics.migration_failed += 1
            else:
                metrics.migration_aborted += 1
        src.is_migrating = False
        dst.is_migrating = False
        for instance in (src, dst):
            if instance.step_in_flight:
                continue
            if instance.running or instance.waiting:
                if not math.isfinite(instance.next_step_time):
                    instance.next_step_time = self.now
            else:
                instance.next_step_time = math.inf
        metrics.migration_cost_ms += pending.actual_ms
        event = {
            "algorithm": "Mig-PPO", "action_id": pending.action_id,
            "source_instance": pending.source_id, "target_instance": pending.target_id,
            "request_id": pending.request_id,
            "blocks": pending.observed_blocks if pending.observed_blocks is not None else pending.blocks,
            "reserved_blocks": pending.blocks,
            "start_time": self.wall_epoch + pending.start_time,
            "end_time": self.wall_epoch + pending.end_time,
            "predicted_cost_ms": pending.predicted_ms, "cost_ms": pending.actual_ms,
            "status": status, "fixed_trace_replay": pending.replay_status is not None,
        }
        self.migration_events.append(event)
        self._record_lifecycle_event(
            f"migration_{status}", request, src if request is not None else None,
            timestamp=self.now, timestamp_source="migration_terminal_event",
            action_id=pending.action_id, request_id=pending.request_id, source_instance_id=src.instance_id,
            target_instance_id=dst.instance_id, blocks_reserved=pending.blocks,
            observed_blocks=pending.observed_blocks,
            source_used_blocks=src.used_blocks, source_reserved_blocks=src.reserved_blocks,
            target_used_blocks=dst.used_blocks, target_reserved_blocks=dst.reserved_blocks,
        )
        self.last_migration_time = self.now
        self.pending_migration = None

    def _advance(self, metrics: MigPPORewardMetrics, window_start: float, window_end: float) -> None:
        queue_integral = 0.0
        kv_integral = 0.0
        imbalance_integral = 0.0
        elapsed = 0.0
        while self.now < window_end - 1e-9:
            next_deadline = (
                min((request.arrival_time + self.config.request_timeout_sec
                     for request in self.requests.values()), default=math.inf)
                if self.config.request_timeout_sec is not None else math.inf
            )
            migration_end = self.pending_migration.end_time if self.pending_migration else math.inf
            service_event = min((instance.next_step_time for instance in self.instances), default=math.inf)
            event_time = min(window_end, self.next_arrival, migration_end, service_event, next_deadline)
            if event_time < self.now:
                event_time = self.now
            dt = event_time - self.now
            if dt > 0:
                waiting = sum(len(instance.waiting) for instance in self.instances)
                kv = sum((instance.used_blocks + instance.reserved_blocks) / instance.total_blocks
                         for instance in self.instances) / len(self.instances)
                counts = np.asarray([len(instance.running) + len(instance.waiting) for instance in self.instances])
                queue_integral += waiting * dt
                kv_integral += min(1.0, kv) * dt
                imbalance_integral += float(np.var(counts)) * dt
                elapsed += dt
                self.now = event_time

            progressed = False
            if self.pending_migration and self.pending_migration.end_time <= self.now + 1e-9:
                self._finish_migration(metrics, window_start, window_end)
                progressed = True
            if self.next_arrival <= self.now + 1e-9:
                self._process_arrival()
                progressed = True
            for instance in self.instances:
                if instance.next_step_time <= self.now + 1e-9:
                    self._run_engine_step(instance, metrics, window_start, window_end)
                    progressed = True
            before = len(self.requests)
            self._expire_requests(metrics, window_start, window_end)
            progressed = progressed or len(self.requests) != before
            if self.config.validate_state_invariants:
                self._check_state_invariants()
            if not progressed and dt <= 1e-12:
                # Avoid a pathological zero-time loop from a stale event.
                self.now = min(window_end, self.now + 1e-6)

        metrics.accumulated_waiting_request_seconds = queue_integral
        metrics.average_queue_length = queue_integral / max(elapsed, 1e-6)
        metrics.kv_usage_samples = [kv_integral / max(elapsed, 1e-6)]
        metrics.load_imbalance = imbalance_integral / max(elapsed, 1e-6)
        if self.config.validate_state_invariants:
            self._check_state_invariants()

    def step(self, action: int):
        if self._last_obs is None:
            raise RuntimeError("call reset() before step()")
        window_start = self.now
        window_end = self.now + self.config.decision_interval_sec
        metrics = MigPPORewardMetrics(
            window_sec=self.config.decision_interval_sec,
            instance_count=self.config.num_instances,
        )
        action = int(action)
        invalid_reason = None
        if action < 0 or action >= self._last_obs.action_count:
            action = 0
            invalid_reason = "action_out_of_range"
        elif action > 0 and not self._last_obs.action_mask.action_mask[action]:
            action = 0
            invalid_reason = "action_mask_rejected"
        if invalid_reason is None:
            invalid_reason = self._start_migration(action, self._last_obs, metrics)
            if invalid_reason is not None:
                action = 0
        self._advance(metrics, window_start, window_end)
        self.decision_count += 1
        terminated = False
        truncated = self.decision_count >= self.config.max_decisions
        if truncated:
            metrics.unfinished_requests = len(self.requests)
        breakdown = self.reward_calculator.compute(metrics)
        refresh_interval = self.config.observation_refresh_interval_sec
        if refresh_interval is None or refresh_interval <= 0 \
                or self.now - self._last_observation_time >= refresh_interval - 1e-9:
            self._last_obs = self._observation()
            self._last_observation_time = self.now
        info = {
            "algorithm": "Mig-PPO", "scenario": self.scenario,
            "decision": self.decision_count, "action": action,
            "action_mask": self._last_obs.action_mask.action_mask.tolist(),
            "invalid_action_reason": invalid_reason,
            "reward": breakdown.to_dict(),
            "metrics": breakdown.raw,
            "calibration_status": self.config.calibration_status,
            "migration_events": self.migration_events[-2:],
        }
        return self._last_obs, breakdown.reward, terminated, truncated, info

    def export_episode(self) -> dict[str, object]:
        """Return reproducibility artifacts from the current episode."""
        return {
            "algorithm": "Mig-PPO", "scenario": self.scenario,
            "decisions": self.decision_count, "request_events": list(self.request_events),
            "lifecycle_events": list(self.lifecycle_events),
            "migration_events": list(self.migration_events),
            "request_results": list(self.request_results),
            "unfinished_requests": [self._request_result(request, "unfinished", self.now)
                                    for request in self.requests.values()],
            "service_iterations": list(self.service_iterations),
            "observed_ttft_samples_ms": list(self.observed_ttft_samples_ms),
            "observed_tpot_samples_ms": list(self.observed_tpot_samples_ms),
            "calibration_status": self.config.calibration_status,
            "cost_model": self.cost_estimator.state_dict(),
        }

