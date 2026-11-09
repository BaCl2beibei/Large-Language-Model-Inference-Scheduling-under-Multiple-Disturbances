"""Read-only Mig-PPO observation construction from Llumnix-like state."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from rl_scheduler.action_mask import MigPPOActionMask, MigPPOActionMaskConfig, build_mig_ppo_action_mask
from rl_scheduler.cost_estimator import MigPPOMigrationCostEstimator


INSTANCE_FEATURE_NAMES = (
    "running_requests", "waiting_requests", "head_wait_ms", "kv_usage",
    "free_kv_ratio", "preempted_requests", "decode_batch_size",
    "last_step_ms", "batched_tokens", "remaining_steps",
    "candidate_kv_ratio", "candidate_prefill", "candidate_decode", "candidate_valid",
)
GLOBAL_FEATURE_NAMES = (
    "instance_count", "arrival_rate_rps", "running_count_variance",
    "kv_usage_variance", "waiting_request_count", "migrations_in_progress",
    "seconds_since_migration",
)
PAIR_FEATURE_NAMES = (
    "waiting_count_delta", "kv_usage_delta", "candidate_kv_ratio",
    "predicted_migration_ms", "predicted_net_gain_ms",
)
COST_MODEL_SOURCE_NAMES = ("pair", "global", "default_uncalibrated")


@dataclass(frozen=True)
class MigPPOObservation:
    instance_features: np.ndarray
    instance_missing: np.ndarray
    global_features: np.ndarray
    global_missing: np.ndarray
    pair_features: np.ndarray
    pair_missing: np.ndarray
    pair_cost_uncertainty_ms: np.ndarray
    pair_cost_source: np.ndarray
    action_mask: MigPPOActionMask
    candidate_request_ids: tuple[Optional[str], ...]
    candidate_blocks: tuple[Optional[int], ...]
    timestamp: float

    @property
    def pair_mask(self) -> np.ndarray:
        return self.action_mask.pair_mask

    @property
    def action_count(self) -> int:
        return 1 + len(self.action_mask.pairs)


class RollingArrivalRate:
    """Estimate arrivals from past event times only (Unix seconds)."""

    def __init__(self, window_sec: float = 60.0) -> None:
        self.window_sec = max(1e-3, float(window_sec))
        self._events: deque[float] = deque()
        self._start_time: Optional[float] = None

    def start(self, timestamp: float) -> None:
        self._start_time = float(timestamp)

    def record(self, timestamp: float) -> None:
        timestamp = float(timestamp)
        if math.isfinite(timestamp):
            self._events.append(timestamp)

    def rate(self, now: float) -> float:
        cutoff = float(now) - self.window_sec
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        start_time = self._start_time
        if start_time is None:
            start_time = self._events[0] if self._events else float(now) - self.window_sec
        exposure = min(self.window_sec, max(1e-3, float(now) - start_time))
        return len(self._events) / exposure


def _safe_number(obj: object, name: str, *, scale: float = 1.0) -> tuple[float, bool]:
    value = getattr(obj, name, None)
    try:
        value = float(value) * scale
    except (TypeError, ValueError, OverflowError):
        return 0.0, True
    if not math.isfinite(value):
        return 0.0, True
    return value, False


def _load_value(obj: object, name: str, nested_name: str) -> tuple[float, bool]:
    metric = getattr(obj, name, None)
    value = getattr(metric, nested_name, None) if metric is not None else None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0, True
    if not math.isfinite(value):
        return 0.0, True
    return value, False


def build_mig_ppo_observation(
    instance_infos: Mapping[str, object] | Sequence[object],
    *,
    now: Optional[float] = None,
    arrival_rate_rps: Optional[float] = None,
    migrations_in_progress: int = 0,
    last_migration_time: Optional[float] = None,
    cost_estimator: Optional[MigPPOMigrationCostEstimator] = None,
    mask_config: MigPPOActionMaskConfig = MigPPOActionMaskConfig(),
    migration_in_progress: bool = False,
) -> MigPPOObservation:
    now = time.time() if now is None else float(now)
    if isinstance(instance_infos, Mapping):
        by_id = {str(key): value for key, value in instance_infos.items()}
    else:
        by_id = {str(getattr(info, "instance_id", i)): info for i, info in enumerate(instance_infos)}
    ids = tuple(sorted(by_id))
    if not ids:
        raise ValueError("Mig-PPO needs at least one instance state")
    infos = [by_id[instance_id] for instance_id in ids]
    estimator = cost_estimator or MigPPOMigrationCostEstimator()

    rows: list[list[float]] = []
    missing_rows: list[list[float]] = []
    candidate_ids: list[Optional[str]] = []
    candidate_blocks: list[Optional[int]] = []
    for info in infos:
        running, mr = _safe_number(info, "num_running_requests")
        waiting, mq = _safe_number(info, "num_waiting_requests")
        head_wait, mw = _safe_number(info, "waiting_time_first_waiting_request", scale=1000.0)
        kv_usage, mk = _safe_number(info, "gpu_cache_usage")
        total_blocks, mt = _safe_number(info, "num_total_gpu_blocks")
        free_blocks, mf = _safe_number(info, "num_free_gpu_blocks")
        if mt or total_blocks <= 0 or mf:
            free_ratio, mfree = 0.0, True
        else:
            free_ratio, mfree = free_blocks / total_blocks, False
        preempted, mp = _safe_number(info, "num_killed_requests")
        decode_batch, md = _safe_number(info, "decode_batch_size")
        last_step_ms, ml = _safe_number(info, "last_inference_latency")
        batched_tokens, mb = _safe_number(info, "num_batched_tokens")
        remaining_steps, ms = _load_value(info, "remaining_steps", "remaining_steps")
        blocks_value = getattr(info, "migration_candidate_kv_blocks", None)
        try:
            blocks = int(blocks_value) if blocks_value is not None else None
        except (TypeError, ValueError, OverflowError):
            blocks = None
        state_timestamp = getattr(info, "timestamp", None)
        try:
            state_age = now - float(state_timestamp)
            state_fresh = math.isfinite(state_age) and -0.25 <= state_age <= mask_config.max_state_age_sec
        except (TypeError, ValueError, OverflowError):
            state_fresh = False
        candidate_timestamp = getattr(info, "migration_candidate_timestamp", None)
        try:
            candidate_age = now - float(candidate_timestamp)
            candidate_fresh = math.isfinite(candidate_age) and -0.25 <= candidate_age <= mask_config.max_state_age_sec
        except (TypeError, ValueError, OverflowError):
            candidate_fresh = False
        candidate_valid = bool(getattr(info, "migration_candidate_valid", False)) \
            and blocks is not None and blocks > 0 and state_fresh and candidate_fresh
        candidate_ratio = (blocks / total_blocks) if candidate_valid and not mt and total_blocks > 0 else 0.0
        candidate_stage = str(getattr(info, "migration_candidate_stage", "") or "").lower()
        candidate_prefill = float(candidate_valid and candidate_stage == "prefill")
        candidate_decode = float(candidate_valid and candidate_stage == "decode")
        row = [
            running, waiting, head_wait, kv_usage, free_ratio, preempted,
            decode_batch, last_step_ms, batched_tokens, remaining_steps,
            candidate_ratio, candidate_prefill, candidate_decode, float(candidate_valid),
        ]
        missing = [
            mr, mq, mw, mk, mfree, mp, md, ml, mb, ms,
            float(not candidate_valid), float(not candidate_valid), float(not candidate_valid), 0.0,
        ]
        if not state_fresh:
            row = [0.0] * len(INSTANCE_FEATURE_NAMES)
            missing = [1.0] * len(INSTANCE_FEATURE_NAMES)
        rows.append(row)
        missing_rows.append(missing)
        raw_candidate_id = getattr(info, "migration_candidate_request_id", None)
        candidate_ids.append(str(raw_candidate_id) if candidate_valid and raw_candidate_id is not None else None)
        candidate_blocks.append(blocks if candidate_valid else None)

    instance_features = np.asarray(rows, dtype=np.float32)
    instance_missing = np.asarray(missing_rows, dtype=np.float32)
    running_values = instance_features[:, 0]
    waiting_values = instance_features[:, 1]
    valid_running = instance_missing[:, 0] == 0
    valid_waiting = instance_missing[:, 1] == 0
    kv_values = instance_features[:, 3]
    valid_kv = np.asarray([not row[3] for row in missing_rows], dtype=bool)
    if arrival_rate_rps is None:
        arrival_rate, arrival_missing = 0.0, True
    else:
        try:
            raw_arrival_rate = float(arrival_rate_rps)
            arrival_missing = not math.isfinite(raw_arrival_rate)
            arrival_rate = max(0.0, raw_arrival_rate) if not arrival_missing else 0.0
        except (TypeError, ValueError, OverflowError):
            arrival_rate, arrival_missing = 0.0, True
    if last_migration_time is None:
        since_migration, since_missing = 0.0, True
    else:
        try:
            raw_last_migration_time = float(last_migration_time)
            since_migration = now - raw_last_migration_time
            since_missing = not math.isfinite(since_migration)
        except (TypeError, ValueError, OverflowError):
            since_migration, since_missing = 0.0, True
        if since_missing:
            since_migration = 0.0
        else:
            since_migration = max(0.0, since_migration)
    global_features = np.asarray([
        len(ids), arrival_rate, float(np.var(running_values[valid_running])) if np.any(valid_running) else 0.0,
        float(np.var(kv_values[valid_kv])) if np.any(valid_kv) else 0.0,
        float(np.sum(waiting_values[valid_waiting])), max(0, int(migrations_in_progress)), since_migration,
    ], dtype=np.float32)
    global_missing = np.asarray([
        0, arrival_missing, not bool(np.any(valid_running)), not bool(np.any(valid_kv)),
        not bool(np.any(valid_waiting)), 0, since_missing,
    ], dtype=np.float32)

    action_mask = build_mig_ppo_action_mask(
        by_id, now=now, config=mask_config, last_migration_time=last_migration_time,
        migration_in_progress=migration_in_progress,
    )
    pair_rows: list[list[float]] = []
    pair_missing_rows: list[list[float]] = []
    pair_uncertainties: list[float] = []
    pair_cost_sources: list[list[float]] = []
    index = {instance_id: idx for idx, instance_id in enumerate(ids)}
    for src_id, dst_id in action_mask.pairs:
        si, di = index[src_id], index[dst_id]
        src, dst = infos[si], infos[di]
        blocks = candidate_blocks[si]
        total = float(getattr(src, "num_total_gpu_blocks", 0) or 0)
        block_ratio = (blocks / total) if blocks is not None and total > 0 else 0.0
        prediction = estimator.predict(src_id, dst_id, blocks or 0)
        source_index = COST_MODEL_SOURCE_NAMES.index(prediction.source) \
            if prediction.source in COST_MODEL_SOURCE_NAMES else len(COST_MODEL_SOURCE_NAMES) - 1
        # This is a current-state heuristic used only as a feature. It is not
        # treated as observed counterfactual reward.
        src_pressure = instance_features[si, 1] * max(instance_features[si, 2], 1.0) \
            + instance_features[si, 0] * max(instance_features[si, 7], 1.0)
        dst_pressure = instance_features[di, 1] * max(instance_features[di, 2], 1.0) \
            + instance_features[di, 0] * max(instance_features[di, 7], 1.0)
        net_gain_ms = float(src_pressure - dst_pressure - prediction.duration_ms)
        pair_rows.append([
            float(instance_features[si, 1] - instance_features[di, 1]),
            float(instance_features[si, 3] - instance_features[di, 3]),
            float(block_ratio), prediction.duration_ms, net_gain_ms,
        ])
        pair_missing_rows.append([
            float(instance_missing[si, 1] or instance_missing[di, 1]),
            float(instance_missing[si, 3] or instance_missing[di, 3]),
            float(blocks is None or total <= 0), 0.0, 0.0,
        ])
        pair_uncertainties.append(float(prediction.uncertainty_ms))
        source_features = [0.0] * len(COST_MODEL_SOURCE_NAMES)
        source_features[source_index] = 1.0
        pair_cost_sources.append(source_features)

    pair_features = np.asarray(pair_rows, dtype=np.float32).reshape((-1, len(PAIR_FEATURE_NAMES)))
    pair_missing = np.asarray(pair_missing_rows, dtype=np.float32).reshape((-1, len(PAIR_FEATURE_NAMES)))
    pair_cost_uncertainty_ms = np.asarray(pair_uncertainties, dtype=np.float32).reshape((-1, 1))
    pair_cost_source = np.asarray(pair_cost_sources, dtype=np.float32).reshape(
        (-1, len(COST_MODEL_SOURCE_NAMES)))
    for array in (instance_features, instance_missing, global_features, global_missing,
                  pair_features, pair_missing, pair_cost_uncertainty_ms, pair_cost_source):
        np.nan_to_num(array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return MigPPOObservation(
        instance_features=instance_features,
        instance_missing=instance_missing,
        global_features=global_features,
        global_missing=global_missing,
        pair_features=pair_features,
        pair_missing=pair_missing,
        pair_cost_uncertainty_ms=pair_cost_uncertainty_ms,
        pair_cost_source=pair_cost_source,
        action_mask=action_mask,
        candidate_request_ids=tuple(candidate_ids),
        candidate_blocks=tuple(candidate_blocks),
        timestamp=now,
    )

