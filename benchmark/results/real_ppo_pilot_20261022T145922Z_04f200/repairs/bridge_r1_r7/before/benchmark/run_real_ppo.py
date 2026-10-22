#!/usr/bin/env python3
"""Audited real telemetry and bridge smoke CLI. No PPO stages are implemented."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from rl_scheduler.real_budget import BudgetLedger,atomic_json
from rl_scheduler.real_bridge import digest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_gate(path,run_dir,*,source_root=ROOT):
    value=json.loads(Path(path).read_text())
    if value.get('status')!='passed':raise RuntimeError('gate not passed: '+str(path))
    if not value.get('checks') or any(check.get('status')!='passed' for check in value['checks']):
        raise RuntimeError('gate lacks successful evidence checks')
    if not value.get('evidence_hashes'):raise RuntimeError('gate lacks evidence hashes')
    source_map=value.get('source_hashes')
    if not source_map or value.get('source_hash')!=digest(source_map):raise RuntimeError('invalid source binding')
    for name,expected in source_map.items():
        if sha(source_root/name)!=expected:raise RuntimeError('gate source changed: '+name)
    config_bindings=value.get('config_hashes',{'experiment_spec.yaml':value.get('config_hash')})
    for name,expected in config_bindings.items():
        candidate=Path(run_dir)/name if (Path(run_dir)/name).exists() else source_root/name
        if sha(candidate)!=expected:raise RuntimeError('gate config changed: '+name)
    for name,expected in value.get('evidence_hashes',{}).items():
        if sha(Path(run_dir)/name)!=expected:raise RuntimeError('gate evidence changed: '+name)
    return value


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=['telemetry','bridge','preflight','opportunity','train','evaluate'],required=True)
    parser.add_argument('--mode',choices=['shadow','active','overhead'],default='shadow')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--manage-services',action='store_true')
    parser.add_argument('--invocation-id')
    return parser.parse_args(argv)


def plan(args,cfg):
    if args.stage not in ('telemetry','bridge'):raise ValueError('stage not implemented: '+args.stage)
    run_dir=args.output_dir.resolve()
    pointer=json.loads((ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json').read_text())
    if Path(pointer['run_dir']).resolve()!=run_dir:raise RuntimeError('active run pointer mismatch')
    if cfg.get('schema_version')!='real-pilot-v1':raise ValueError('wrong real config schema')
    if args.config.resolve()!=ROOT/'rl_scheduler/configs/real_pilot.yaml':raise ValueError('unregistered configuration path')
    from rl_scheduler.real_runner import verify_trace_manifest
    verify_trace_manifest(cfg)
    # P1's API registration changed in P2 and is explicitly revalidated, not silently ignored.
    p1=run_dir/'gates/p1_offline_revalidated_03.json'
    if p1.exists():validate_gate(p1,run_dir)
    else:validate_gate(run_dir/'gates/p1_offline.json',run_dir)
    if not args.dry_run:validate_gate(run_dir/'gates/p2_offline.json',run_dir)
    key='telemetry' if args.stage=='telemetry' else args.mode
    repeats=4 if key=='overhead' else 1
    case=cfg['smoke'][key]
    duration=repeats*(case['arrival_sec']+case['drain_sec']+60)+60
    if args.manage_services:duration+=cfg['service']['startup_timeout_sec']
    budget=BudgetLedger(run_dir).require(duration)
    return {'stage':args.stage,'mode':key,'config':str(args.config.resolve()),'config_hash':sha(args.config),
            'episodes':repeats,'case':case,'estimated_max_sec':duration,
            'remaining_budget':budget['remaining'],'manage_services':args.manage_services,
            'server_argv':cfg['service']['server_argv'],'env':cfg['service']['env'],
            'GPU_will_start':bool(args.manage_services and not args.dry_run),
            'dry_run':args.dry_run,'training_implemented':False}


async def wait_ready(cfg,budget):
    import aiohttp
    timeout=cfg['service']['startup_timeout_sec'];started=time.monotonic()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3.)) as session:
        while time.monotonic()-started<timeout:
            budget.require(60.)
            try:
                async with session.get(cfg['service']['base_url']+'/is_ready') as response:
                    if response.status==200 and await response.json() is True:return
            except (aiohttp.ClientError,asyncio.TimeoutError):pass
            await asyncio.sleep(1.)
    raise TimeoutError('four-instance readiness not established within startup bound')


def main(argv=None):
    args=parse_args(argv)
    try:
        import yaml
        cfg=yaml.safe_load(args.config.read_text())
        intent=plan(args,cfg)
        if args.dry_run:
            print(json.dumps(intent,indent=2));return 0
        from rl_scheduler.real_service import ManagedService
        from rl_scheduler.real_runner import run_live
        service=ManagedService(ROOT,args.output_dir,args.config.resolve())
        invocation=args.invocation_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+args.stage+'_'+args.mode
        folder=args.output_dir/'smoke_runs'/invocation
        folder.mkdir(parents=True,exist_ok=False)
        atomic_json(folder/'intent.json',intent)
        started_service=False
        try:
            if args.manage_services:
                started_service=True  # start failure also needs owned-only cleanup
                service.start()
            else:
                service.assert_running_owned()
            budget=BudgetLedger(args.output_dir)
            asyncio.run(wait_ready(cfg,budget))
            result=asyncio.run(run_live(cfg,run_dir=args.output_dir,stage=args.stage,mode=args.mode,invocation_id=invocation))
            print(json.dumps({'status':result['status'],'result':str(folder/'result.json')}))
            return 0 if result['status']=='passed' else 1
        except BaseException as exc:
            atomic_json(folder/'failure.json',{'status':'incomplete','reason':type(exc).__name__+':'+str(exc),
                                              'remaining_budget':BudgetLedger(args.output_dir).status()['remaining']})
            raise
        finally:
            if started_service:service.stop()
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print(json.dumps({'status':'incomplete','reason':str(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
