#!/usr/bin/env python3
"""Canonical entrypoint for the accepted Q15-B real PPO run and its reports."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark import run_p2p3_q15b as pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "report", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
    # Used only by the owned supervisor created by `launch`.
    for name in ("run", "watchdog"):
        command = commands.add_parser(name, help=argparse.SUPPRESS)
        command.add_argument("--run-dir", type=Path, required=True)
    plan = commands.add_parser("multienv-plan", help="CPU-only frozen 13-environment preflight and trace registration")
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--evaluation-seed", type=int, default=26110521)
    plan.add_argument("--validation-seed", type=int, default=26110524)
    plan.add_argument("--gpu-budget-sec", type=float, default=18000.)
    plan.add_argument("--prior-gpu-reservation-sec", type=float, default=0.)
    plan.add_argument("--prior-run-dir", type=Path,
                      help="Record a failed preflight run whose scored candidate collection never began")
    collect = commands.add_parser("multienv-collect", help="Run one owned real GPU collection stage")
    collect.add_argument("--run-dir", type=Path, required=True)
    collect.add_argument("--stage", choices=("selection", "validation"), required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()

    if args.command == "multienv-plan":
        from benchmark import mig_ppo_v2_multienv as multienv
        print(json.dumps(multienv.build_plan(run_dir, evaluation_seed=args.evaluation_seed,
            validation_seed=args.validation_seed, gpu_budget_sec=args.gpu_budget_sec,
            prior_gpu_reservation_sec=args.prior_gpu_reservation_sec,
            prior_run_dir=args.prior_run_dir), indent=2, ensure_ascii=False))
    elif args.command == "multienv-collect":
        from benchmark import mig_ppo_v2_multienv as multienv
        print(json.dumps(multienv.collect(run_dir, stage=args.stage), indent=2, ensure_ascii=False))
    elif args.command == "launch":
        pipeline.launch(run_dir, entry_script=Path(__file__).resolve())
    elif args.command == "run":
        asyncio.run(pipeline.real_run(run_dir))
    elif args.command == "watchdog":
        pipeline.watchdog(run_dir)
    elif args.command == "report":
        print(json.dumps(pipeline.final_report(run_dir), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(pipeline.verify_delivery(run_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
