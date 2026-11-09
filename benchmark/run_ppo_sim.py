#!/usr/bin/env python3
"""Canonical entrypoint for Q15-B simulator validation and CPU PPO contract checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "resume-cpu", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--reward-config", type=Path, required=True,
                                 help="Mig-PPO V2 JSON with calibrated hotspot_threshold and fixed reference scales")
            command.add_argument("--trace-seeds", type=int, nargs=2, default=[161101, 161102])
            command.add_argument("--policy-seed", type=int, default=161001)
    for name, help_text in (
            ("multienv-fit", "Fit and freeze per-environment CPU physical models and V2 reward"),
            ("multienv-validate", "Run CPU closed-loop validation on frozen models and independent real traces"),
            ("multienv-report", "Write the multi-environment handoff and reports")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--run-dir", type=Path, required=True)
    command = commands.add_parser('physical-refit', help='CPU-only V2 physical audit, refit and development validation')
    command.add_argument('--run-dir', type=Path, required=True)
    command.add_argument('--stage', choices=('all','audit','fit','validate','report'), default='all')
    command.add_argument('--workers', type=int, default=4)
    command = commands.add_parser('portable-export', help='Build native Windows/Linux CPU ZIP')
    command.add_argument('--output-dir', type=Path, required=True)
    command.add_argument('--calibration-dir', type=Path)
    for name in ('local-doctor','local-smoke','local-train','local-evaluate','local-export-policy'):
        command=commands.add_parser(name,help='Portable native CPU '+name.removeprefix('local-'))
        command.add_argument('--bundle-root',type=Path,default=Path('.'))
        if name=='local-doctor':
            command.add_argument('--output-file',type=Path)
        else:command.add_argument('--run-dir',type=Path,required=True)
        if name in ('local-export-policy',):command.add_argument('--output-dir',type=Path,required=True)
        if name=='local-train':
            command.add_argument('--config',type=Path)
            command.add_argument('--resume',nargs='?',const='latest')
            command.add_argument('--max-updates',type=int)
            command.add_argument('--wall-budget-sec',type=float)
            command.add_argument('--skip-evaluation',action='store_true',help='Diagnostic updates only; no candidate selection')
            command.add_argument('--recover-lock',action='store_true',help='Use only after the previous training process has exited')
        if name=='local-evaluate':
            command.add_argument('--split',choices=('development','heldout'),default='development')
            command.add_argument('--checkpoint',type=Path)
    args = parser.parse_args(argv)
    if args.command=='portable-export' or args.command.startswith('local-'):
        from benchmark.ppo_local import dispatch
        print(json.dumps(dispatch(args),indent=2,ensure_ascii=False))
        return 0
    run_dir = args.run_dir.resolve()

    if args.command == 'physical-refit':
        from benchmark.mig_ppo_v2_refit import run
        print(json.dumps(run(run_dir, stage=args.stage, workers=args.workers), indent=2, ensure_ascii=False))
    elif args.command.startswith("multienv-"):
        from benchmark import mig_ppo_v2_multienv as multienv
        if args.command == "multienv-fit":
            result = multienv.fit(run_dir)
        elif args.command == "multienv-validate":
            result = multienv.validate_simulation(run_dir)
        else:
            result = multienv.write_delivery(run_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "prepare":
        from benchmark import run_p2p3_q15b as pipeline
        pipeline.prepare(run_dir, reward_config=json.loads(args.reward_config.read_text()),
                         trace_seeds=args.trace_seeds, policy_seed=args.policy_seed)
    elif args.command == "resume-cpu":
        from benchmark import run_p2p3_q15b as pipeline
        if (run_dir / "gpu_budget.json").exists():
            parser.error("CPU resume is unavailable after a real run has started")
        pipeline.finish_prepare(run_dir)
    else:
        from benchmark import run_p2p3_q15b as pipeline
        print(json.dumps(pipeline.final_report(run_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
