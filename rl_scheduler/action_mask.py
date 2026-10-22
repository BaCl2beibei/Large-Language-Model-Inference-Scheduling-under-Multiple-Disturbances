"""Safety and validity masks for Stay and directed migration-pair actions."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MigPPOActionMask:
    instance_ids: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    pair_mask: np.ndarray
    stay_valid: bool = True

    @property
    def action_mask(self) -> np.ndarray:
        return np.concatenate((np.asarray([self.stay_valid], dtype=np.bool_), self.pair_mask))


@dataclass(frozen=True)
class MigPPOActionMaskConfig:
    max_state_age_sec: float = 2.0
    migration_cooldown_sec: float = 5.0
    safety_margin_blocks: int = 16
    safety_margin_fraction: float = 0.05
    neutral_only: bool = True


def _value(info: object, name: str, default=None):
    return getattr(info, name, default)


def _enum_name(value: object) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value)).lower()


def _healthy(info: object) -> bool:
    method = getattr(info, "is_unit_healthy", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            return False
    status = _enum_name(_value(info, "unit_status", "healthy"))
    return status in {"healthy", "unitstatus.healthy", "ready", "true"}


def _fresh(info: object, now: float, max_age: float) -> bool:
    timestamp = _value(info, "timestamp")
    try:
        timestamp = float(timestamp)
    except (TypeError, ValueError):
        return False
    return math.isfinite(timestamp) and -0.25 <= now - timestamp <= max_age


def _candidate_is_fresh(info: object, now: float, max_age: float) -> bool:
    timestamp = _value(info, "migration_candidate_timestamp")
    try:
        timestamp = float(timestamp)
    except (TypeError, ValueError):
        return False
    return math.isfinite(timestamp) and -0.25 <= now - timestamp <= max_age


def _candidate_blocks(info: object) -> Optional[int]:
    if not bool(_value(info, "migration_candidate_valid", False)):
        return None
    value = _value(info, "migration_candidate_kv_blocks")
    try:
        blocks = int(value)
    except (TypeError, ValueError):
        return None
    return blocks if blocks > 0 else None


def _compatible(src: object, dst: object, neutral_only: bool) -> bool:
    src_type = _enum_name(_value(src, "instance_type"))
    dst_type = _enum_name(_value(dst, "instance_type"))
    if neutral_only:
        return src_type.endswith("neutral") and dst_type.endswith("neutral")
    return not src_type or not dst_type or src_type == dst_type


def build_mig_ppo_action_mask(
    instance_infos: Mapping[str, object] | Sequence[object],
    *,
    now: Optional[float] = None,
    config: MigPPOActionMaskConfig = MigPPOActionMaskConfig(),
    last_migration_time: Optional[float] = None,
    migration_in_progress: bool = False,
) -> MigPPOActionMask:
    """Build a fail-closed mask; action 0 (Stay) always remains valid."""
    now = time.time() if now is None else float(now)
    if isinstance(instance_infos, Mapping):
        infos = dict(instance_infos)
    else:
        infos = {
            str(_value(info, "instance_id", index)): info
            for index, info in enumerate(instance_infos)
        }
    ids = tuple(sorted(str(key) for key in infos))
    pair_list = tuple((src, dst) for src in ids for dst in ids if src != dst)
    pair_mask = np.zeros(len(pair_list), dtype=np.bool_)

    cooldown_active = migration_in_progress
    if last_migration_time is not None:
        try:
            cooldown_active = cooldown_active or now - float(last_migration_time) < config.migration_cooldown_sec
        except (TypeError, ValueError):
            cooldown_active = True

    by_id = {str(key): value for key, value in infos.items()}
    for idx, (src_id, dst_id) in enumerate(pair_list):
        src, dst = by_id[src_id], by_id[dst_id]
        if cooldown_active:
            continue
        if not _fresh(src, now, config.max_state_age_sec) or not _fresh(dst, now, config.max_state_age_sec):
            continue
        if not _candidate_is_fresh(src, now, config.max_state_age_sec):
            continue
        blocks = _candidate_blocks(src)
        if blocks is None:
            continue
        candidate_stage = _enum_name(_value(src, "migration_candidate_stage"))
        if candidate_stage not in {"decode", "requestinferencetype.decode"}:
            continue
        if not _healthy(src) or not _healthy(dst):
            continue
        if bool(_value(src, "is_migrating", False)) or bool(_value(dst, "is_migrating", False)):
            continue
        if not bool(_value(src, "has_migration_slot", True)) or not bool(_value(dst, "has_migration_slot", True)):
            continue
        if not bool(_value(dst, "can_accept_migration", True)):
            continue
        if not _compatible(src, dst, config.neutral_only):
            continue
        free_blocks = _value(dst, "num_free_gpu_blocks")
        total_blocks = _value(dst, "num_total_gpu_blocks")
        waiting_blocks = _value(dst, "num_blocks_all_waiting_requests", 0)
        try:
            free_blocks = int(free_blocks)
            total_blocks = int(total_blocks)
            waiting_blocks = max(0, int(math.ceil(float(waiting_blocks or 0))))
        except (TypeError, ValueError, OverflowError):
            continue
        if total_blocks <= 0 or free_blocks < 0:
            continue
        reserve = max(config.safety_margin_blocks, int(math.ceil(total_blocks * config.safety_margin_fraction)))
        available_after_waiting = free_blocks - waiting_blocks
        if available_after_waiting - blocks < reserve:
            continue
        pair_mask[idx] = True

    return MigPPOActionMask(ids, pair_list, pair_mask, stay_valid=True)

