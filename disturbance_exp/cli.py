"""Command line entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .controller import report_result, run_experiment
from .safety import doctor_sync
from .scenario import ScenarioError, load_scenario, validate_scenario


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m disturbance_exp")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="read-only environment and service check")
    doctor.add_argument("--scenario", help="scenario used for endpoint locations")
    run = commands.add_parser("run", help="validate, plan, or execute one bounded scenario")
    run.add_argument("--scenario", required=True)
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="send real requests and run enabled injectors")
    mode.add_argument("--dry-run", action="store_true", help="default; write a plan without sending requests")
    run.add_argument("--allow-gpu-load", action="store_true", help="permit the memory-gated GPU injector")
    run.add_argument("--result-root")
    report = commands.add_parser("report", help="rebuild plots and Markdown from an existing result")
    report.add_argument("--result-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            cfg = load_scenario(args.scenario) if args.scenario else validate_scenario({})
            report = doctor_sync(cfg)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("service_ok") else 1
        if args.command == "run":
            result_dir = asyncio.run(run_experiment(
                args.scenario, execute=args.execute,
                allow_gpu_load=args.allow_gpu_load, result_root=args.result_root,
            ))
            print(result_dir)
            return 0
        if args.command == "report":
            report_path = asyncio.run(report_result(args.result_dir))
            print(report_path)
            return 0
    except (ScenarioError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2
