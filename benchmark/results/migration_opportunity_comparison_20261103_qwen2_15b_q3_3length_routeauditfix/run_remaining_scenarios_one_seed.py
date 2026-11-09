#!/usr/bin/env python3
"""Run remaining QPS=3 scenes at seed 91000 against frozen Stay traces.

The 1024/1024 L attempt exposed that the traffic client had a 360 s total
timeout, shorter than the fixed 600 s scoring window. This wrapper disables
that total timeout while retaining 10 s connection timeouts, so client streams
remain observable through scoring and bounded cleanup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rl_scheduler.real_env import RealEnvConfig
import rl_scheduler.migration_scan_runner as scan_runner
import benchmark.run_migration_opportunity_scan as runner
from rl_scheduler.real_budget import atomic_json

_original_client_timeout = aiohttp.ClientTimeout


def _window_aligned_client_timeout(*args, **kwargs):
    if kwargs.get("total") == 360.0:
        kwargs = dict(kwargs)
        kwargs.update(total=None, connect=10.0, sock_connect=10.0, sock_read=None)
    return _original_client_timeout(*args, **kwargs)


aiohttp.ClientTimeout = _window_aligned_client_timeout

SCENARIOS = (
    "i512_o512_q3_s192_b4096",
    "i1024_o1024_q3_s192_b4096",
)
SEED = 91000
POLICIES = ("Balanced-L", "Balanced-M", "Balanced-H")

_original_episode = scan_runner.episode


async def _episode_with_full_drain(*, episode_id, **kwargs):
    if episode_id != "warmup":
        kwargs["horizons"] = RealEnvConfig(arrival_sec=120.0, drain_sec=480.0)
    return await _original_episode(episode_id=episode_id, **kwargs)


scan_runner.episode = _episode_with_full_drain


def _one_seed_schedule(cases, seeds, thresholds):
    selected = [case for case in cases if case["scenario"] in SCENARIOS]
    by_name = {case["scenario"]: case for case in selected}
    if set(by_name) != set(SCENARIOS):
        raise RuntimeError(f"expected exactly scenarios {SCENARIOS}; got {tuple(by_name)}")
    return [
        dict(case=by_name[scenario], seed=SEED, policy=policy,
             threshold=thresholds[scenario][policy])
        for scenario in SCENARIOS for policy in POLICIES
    ]


runner.paired_schedule = _one_seed_schedule

parser = argparse.ArgumentParser()
parser.add_argument("--max-seconds", type=float, default=86400.0)
args = parser.parse_args()
sys.argv = [
    "run_migration_opportunity_scan.py", "paired",
    "--run-dir", str(RUN_DIR), "--max-seconds", str(args.max_seconds),
]
print(
    f"single_seed_remaining_scenes scenarios={','.join(SCENARIOS)} "
    f"seed={SEED} policies={','.join(POLICIES)} measured_horizons=120+480 "
    "client_total_timeout=None (was 360s, shorter than 600s window)",
    flush=True,
)
runner.main()

status_path = RUN_DIR / "paired_status.json"
if status_path.exists():
    status = json.loads(status_path.read_text())
    if status.get("status") == "completed":
        atomic_json(status_path, dict(
            status="single_seed_three_scenario_slice_completed",
            scenarios=list(SCENARIOS),
            seed=SEED,
            policies=list(POLICIES),
            full_36_episode_matrix_complete=False,
        ))
