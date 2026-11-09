#!/usr/bin/env python3
"""Run audited scan/paired stages with a longer, common full-completion drain.

The existing 120 s arrival trace and all request seeds remain unchanged. Every
measured episode uses 120 s arrival + 480 s drain so the over-capacity 1024-token
workloads have time to reach terminal outcomes without masking cancel ambiguity.
Warmup keeps its separately defined 2 s + 8 s horizon.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rl_scheduler.real_env import RealEnvConfig
import rl_scheduler.migration_scan_runner as scan_runner
import benchmark.run_migration_opportunity_scan as runner

_original_episode = scan_runner.episode

async def _episode_with_longer_drain(*, episode_id, **kwargs):
    if episode_id != 'warmup':
        kwargs['horizons'] = RealEnvConfig(arrival_sec=120.0, drain_sec=480.0)
    return await _original_episode(episode_id=episode_id, **kwargs)

scan_runner.episode = _episode_with_longer_drain

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=('scan', 'paired'))
parser.add_argument('--max-seconds', type=float, default=86400.0)
args = parser.parse_args()
print('measured_horizons arrival_sec=120 drain_sec=480; warmup=2+8; stage='+args.stage,
      flush=True)
sys.argv = ['run_migration_opportunity_scan.py', args.stage, '--run-dir', str(RUN_DIR),
            '--max-seconds', str(args.max_seconds)]
runner.main()
