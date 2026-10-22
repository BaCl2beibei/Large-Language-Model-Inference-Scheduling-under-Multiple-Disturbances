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


LIVE_GATES = {'telemetry':'p1_live.json', 'shadow':'p2_shadow_live.json',
              'active':'p2_active_live.json', 'overhead':'p2_live.json'}
LIVE_ORDER = tuple(LIVE_GATES)
LIVE_CHECKS = {
    'telemetry': {'episodes_reconciled','low_load_normal_completion'},
    'shadow': {'episodes_reconciled','CPU_decision_path_P95_below_100ms','zero_actual_migration','zero_coordinator_migration_attempts'},
    'active': {'episodes_reconciled','CPU_decision_path_P95_below_100ms','at_least_10_terminals','at_least_5_successes'},
    'overhead': {'episodes_reconciled','paired_overhead_within_thresholds'},
}


def live_dependencies(run_dir, key, offline):
    """Every live stage requires the independent review and all earlier live gates."""
    review_path=run_dir/'reviews/bridge_review.json'
    review=validate_gate(review_path,run_dir)
    if review.get('stage')!='04_bridge_review':raise RuntimeError('independent step-04 review required')
    for gate in offline:
        for name,value in gate['source_hashes'].items():
            if review['source_hashes'].get(name)!=value:
                raise RuntimeError('review does not bind current offline source: '+name)
        for name,value in gate['config_hashes'].items():
            if review['config_hashes'].get(name)!=value:
                raise RuntimeError('review does not bind current offline config: '+name)
    for name,value in review.get('dependency_hashes',{}).items():
        if sha(ROOT/name)!=value:raise RuntimeError('review dependency changed: '+name)
    dependencies={'reviews/bridge_review.json':sha(review_path)}
    for previous in LIVE_ORDER[:LIVE_ORDER.index(key)]:
        name='gates/'+LIVE_GATES[previous]
        gate=validate_gate(run_dir/name,run_dir)
        if gate.get('live_case')!=previous or gate.get('dependency_gate_hashes')!=dependencies:
            raise RuntimeError('live gate dependency chain changed: '+name)
        if gate['source_hashes']!=review['source_hashes'] or gate['config_hashes']!=review['config_hashes']:
            raise RuntimeError('live gate review binding changed: '+name)
        if gate['result_path'] not in gate['evidence_hashes']:
            raise RuntimeError('unbound live result: '+name)
        result=json.loads((run_dir/gate['result_path']).read_text())
        if (result.get('status')!='passed' or not result.get('checks')
                or not LIVE_CHECKS[previous].issubset(result['checks'])
                or not all(value is True for value in result['checks'].values())):
            raise RuntimeError('live result not passed: '+name)
        dependencies[name]=sha(run_dir/name)
    return dependencies


def publish_live_gate(run_dir,key,invocation,intent):
    """Called only after owned service cleanup; status is derived from actual results."""
    run_dir=Path(run_dir)
    path=run_dir/'smoke_runs'/invocation/'result.json'
    result=json.loads(path.read_text())
    review=validate_gate(run_dir/'reviews/bridge_review.json',run_dir)
    if live_dependencies(run_dir,key,[])!=intent['dependency_gate_hashes']:
        raise RuntimeError('live dependency hashes changed during execution')
    result_path=str(path.relative_to(run_dir))
    evidence={str(p.relative_to(run_dir)):sha(p) for p in path.parent.rglob('*') if p.is_file()}
    for row in result.get('results',[]):
        for p in (run_dir/'runs'/row['episode_id']).rglob('*'):
            if p.is_file():evidence[str(p.relative_to(run_dir))]=sha(p)
    checks=[{'name':name,'status':'passed' if passed is True else 'failed'}
            for name,passed in result.get('checks',{}).items()]
    checks.append({'name':'owned_cleanup_completed','status':'passed'})
    gate={'stage':'05_live','live_case':key,'status':result['status'], 'checks':checks,
          'source_hashes':review['source_hashes'],'source_hash':review['source_hash'],
          'config_hashes':review['config_hashes'],'evidence_hashes':evidence,
          'dependency_gate_hashes':intent['dependency_gate_hashes'],'result_path':result_path}
    if (not LIVE_CHECKS[key].issubset(result.get('checks',{}))
            or any(c['status']!='passed' for c in checks)):gate['status']='incomplete'
    destination=run_dir/'gates'/LIVE_GATES[key]
    if destination.exists():raise RuntimeError('live gate already exists; preserve prior attempt')
    atomic_json(destination,gate)


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
    key='telemetry' if args.stage=='telemetry' else args.mode
    paths=pointer.get('offline_gates',{
        'P1':'gates/p1_offline_revalidated_03.json' if (run_dir/'gates/p1_offline_revalidated_03.json').exists() else 'gates/p1_offline.json',
        'P2':'gates/p2_offline.json'})
    blockers=[];offline=[];dependencies={}
    for name in ('P1','P2'):
        try:offline.append(validate_gate(run_dir/paths[name],run_dir))
        except (RuntimeError,OSError,KeyError,ValueError) as exc:blockers.append(name+': '+str(exc))
    try:dependencies=live_dependencies(run_dir,key,offline)
    except (RuntimeError,OSError,KeyError,ValueError) as exc:blockers.append('live prerequisites: '+str(exc))
    if (run_dir/'gates'/LIVE_GATES[key]).exists():blockers.append('live stage already attempted')
    if blockers and not args.dry_run:raise RuntimeError('; '.join(blockers))
    repeats=4 if key=='overhead' else 1
    case=cfg['smoke'][key]
    duration=repeats*(case['arrival_sec']+case['drain_sec']+60)+60
    if args.manage_services:duration+=cfg['service']['startup_timeout_sec']
    ledger=BudgetLedger(run_dir)
    if args.dry_run:
        state=json.loads((run_dir/'experiment_clock.json').read_text())
        if state['first_started_wall'] is not None:
            delta=max(0.,time.time()-state['last_wall'])
            if state['boot_id']==ledger.boot_id:delta=max(delta,time.monotonic()-state['last_mono'])
            state['elapsed_sec']+=delta
            if state['phase']:state['phase_usage'][state['phase']]+=delta
        budget={'remaining':ledger._remaining(state)}
        if state['exhausted'] or min(budget['remaining']['total_experiment_wall_time_sec'],budget['remaining']['P0_P3_sec'])<duration:
            blockers.append('budget reserve insufficient')
    else:budget=ledger.require(duration)
    return {'stage':args.stage,'mode':key,'config':str(args.config.resolve()),'config_hash':sha(args.config),
            'episodes':repeats,'case':case,'estimated_max_sec':duration,
            'remaining_budget':budget['remaining'],'manage_services':args.manage_services,
            'server_argv':cfg['service']['server_argv'],'env':cfg['service']['env'],
            'GPU_will_start':bool(args.manage_services and not args.dry_run),
            'dry_run':args.dry_run,'training_implemented':False,
            'blocked_reasons':blockers,'ready_for_live':not blockers,
            'dependency_gate_hashes':dependencies}


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
                started_service=True
            budget=BudgetLedger(args.output_dir)
            asyncio.run(wait_ready(cfg,budget))
            result=asyncio.run(run_live(cfg,run_dir=args.output_dir,stage=args.stage,mode=args.mode,invocation_id=invocation))
        except BaseException as exc:
            atomic_json(folder/'failure.json',{'status':'incomplete','reason':type(exc).__name__+':'+str(exc),
                                              'remaining_budget':BudgetLedger(args.output_dir).status()['remaining']})
            raise
        finally:
            if started_service:service.stop()
        publish_live_gate(args.output_dir,intent['mode'],invocation,intent)
        print(json.dumps({'status':result['status'],'result':str(folder/'result.json')}))
        return 0 if result['status']=='passed' else 1
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print(json.dumps({'status':'incomplete','reason':str(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
