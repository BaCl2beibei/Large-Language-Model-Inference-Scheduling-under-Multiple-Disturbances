#!/usr/bin/env python3
"""Run one audited Stay vs Balanced-L pair at explicit threshold F=100."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import aiohttp
ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rl_scheduler.real_env import RealEnvConfig
import rl_scheduler.migration_scan_runner as scan_runner
import benchmark.run_migration_opportunity_scan as runner
from rl_scheduler.real_budget import atomic_json

_original_timeout = aiohttp.ClientTimeout
def _window_aligned_timeout(*args, **kwargs):
    if kwargs.get("total") == 360.0:
        kwargs = dict(kwargs)
        kwargs.update(total=None, connect=10.0, sock_connect=10.0, sock_read=None)
    return _original_timeout(*args, **kwargs)
aiohttp.ClientTimeout = _window_aligned_timeout

SCENARIO = "i512_o1084_q3_s192_b4096"
SEED = 91000
_original_episode = scan_runner.episode
async def _full_window_episode(*, episode_id, **kwargs):
    if episode_id != "warmup":
        kwargs["horizons"] = RealEnvConfig(arrival_sec=120.0, drain_sec=480.0)
    return await _original_episode(episode_id=episode_id, **kwargs)
scan_runner.episode = _full_window_episode

def _schedule(cases, seeds, thresholds):
    matches = [case for case in cases if case["scenario"] == SCENARIO]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one case for {SCENARIO}; found {len(matches)}")
    case = matches[0]
    return [
        dict(case=case, seed=SEED, policy="Stay", threshold=None),
        dict(case=case, seed=SEED, policy="Balanced-L", threshold=100),
    ]
runner.paired_schedule = _schedule
parser = argparse.ArgumentParser()
parser.add_argument("--max-seconds", type=float, default=86400.0)
args = parser.parse_args()
sys.argv = ["run_migration_opportunity_scan.py", "paired", "--run-dir", str(RUN_DIR), "--max-seconds", str(args.max_seconds)]
print(f"paired_test scenario={SCENARIO} seed={SEED} policies=Stay,Balanced-L threshold=100 measured_window=120+480 client_total_timeout=None", flush=True)
runner.main()
status_path = RUN_DIR / "paired_status.json"
if status_path.exists():
    status = json.loads(status_path.read_text())
    if status.get("status") == "completed":
        atomic_json(status_path, dict(status="single_seed_pair_completed", scenario=SCENARIO, seed=SEED, policies=["Stay", "Balanced-L"], threshold=100, full_matrix_complete=False))
