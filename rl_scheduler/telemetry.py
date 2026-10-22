"""Append-only Mig-PPO decision and simulation event export."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


class MigPPOTelemetryWriter:
    """Write observations, masks, actions, reward terms and migration events as JSONL."""

    def __init__(self, log_dir: str | Path, run_id: str, seed: int) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.seed = int(seed)
        self.decision_path = self.log_dir / f"{self.run_id}_decisions.jsonl"
        self.migration_path = self.log_dir / f"{self.run_id}_migrations.jsonl"
        self.request_path = self.log_dir / f"{self.run_id}_requests.jsonl"

    def _append(self, path: Path, record: Mapping[str, object]) -> None:
        payload = {
            "algorithm": "Mig-PPO", "run_id": self.run_id, "seed": self.seed,
            **dict(record),
        }
        line = json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_decision(self, *, timestamp: float, source_ids: tuple[str, ...],
                     candidate_request_ids: tuple[Optional[str], ...], action: int,
                     action_mask: object, reward_breakdown: object,
                     fallback_reason: Optional[str] = None) -> None:
        mask = getattr(action_mask, "action_mask", action_mask)
        self._append(self.decision_path, {
            "timestamp": timestamp, "instance_ids": source_ids,
            "candidate_request_ids": candidate_request_ids,
            "action": action, "action_mask": mask,
            "reward": reward_breakdown, "fallback_reason": fallback_reason,
        })

    def log_migration(self, event: Mapping[str, object]) -> None:
        self._append(self.migration_path, event)

    def log_request_event(self, event: Mapping[str, object]) -> None:
        self._append(self.request_path, event)

