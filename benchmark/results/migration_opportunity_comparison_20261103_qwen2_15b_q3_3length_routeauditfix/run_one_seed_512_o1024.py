#!/usr/bin/env python3
"""Run one-seed Balanced policies for the 512/1024 QPS3 scan baseline.

The already audited seed-91000 Stay scan is the baseline. This runs only the
three Balanced policies against its frozen trace, with the same 120+480 s
measurement window used by the scan. The wrapper marks the paired status as a
slice so it cannot be mistaken for completion of the full paired matrix.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rl_scheduler.real_env import RealEnvConfig
import rl_scheduler.migration_scan_runner as scan_runner
import benchmark.run_migration_opportunity_scan as runner
from rl_scheduler.real_budget import atomic_json

SCENARIO = "i512_o1024_q3_s192_b4096"
SEED = 91000
POLICIES = ("Balanced-L", "Balanced-M", "Balanced-H")

_original_episode = scan_runner.episode


async def _episode_with_full_drain(*, episode_id, **kwargs):
    if episode_id != "warmup":
        kwargs["horizons"] = RealEnvConfig(arrival_sec=120.0, drain_sec=480.0)
    return await _original_episode(episode_id=episode_id, **kwargs)


scan_runner.episode = _episode_with_full_drain


def _one_seed_schedule(cases, seeds, thresholds):
    selected = [case for case in cases if case["scenario"] == SCENARIO]
    if len(selected) != 1:
        raise RuntimeError(f"expected exactly one {SCENARIO} case; got {len(selected)}")
    case = selected[0]
    return [
        dict(case=case, seed=SEED, policy=policy,
             threshold=thresholds[SCENARIO][policy])
        for policy in POLICIES
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
    f"single_seed_slice scenario={SCENARIO} seed={SEED} "
    f"policies={','.join(POLICIES)} measured_horizons=120+480",
    flush=True,
)
runner.main()

status_path = RUN_DIR / "paired_status.json"
if status_path.exists():
    status = json.loads(status_path.read_text())
    if status.get("status") == "completed":
        atomic_json(status_path, dict(
            status="single_seed_slice_completed",
            scenario=SCENARIO,
            seed=SEED,
            policies=list(POLICIES),
            full_matrix_complete=False,
        ))
