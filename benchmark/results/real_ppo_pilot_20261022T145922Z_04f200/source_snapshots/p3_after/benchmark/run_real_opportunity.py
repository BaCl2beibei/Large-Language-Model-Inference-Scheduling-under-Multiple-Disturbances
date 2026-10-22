#!/usr/bin/env python3
"""P3-only CLI, isolated from hash-bound P1/P2 smoke implementation."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from rl_scheduler.real_budget import BudgetLedger,atomic_json
from rl_scheduler.real_bridge import digest
from rl_scheduler.real_opportunity import maximum_duration,readonly_budget,load_frozen_trace


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke_cli():
    spec=importlib.util.spec_from_file_location('real_smoke_gate_validation',ROOT/'benchmark/run_real_ppo.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--opportunity-spec',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stage',choices=['opportunity'],required=True)
    p.add_argument('--invocation-id',required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--manage-services',action='store_true')
    return p.parse_args(argv)


def validate_inputs(args,spec,runtime,*,require_ready=True):
    run=args.output_dir.resolve();pointer=json.loads((ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json').read_text())
    if Path(pointer['run_dir']).resolve()!=run or Path(spec['run_dir']).resolve()!=run:raise ValueError('active run/spec directory mismatch')
    if args.opportunity_spec.resolve()!=run/'opportunity_spec.yaml':raise ValueError('unregistered opportunity protocol')
    if args.config.resolve()!=Path(spec['runtime_config']).resolve() or args.config.resolve()!=ROOT/'rl_scheduler/configs/real_pilot.yaml':
        raise ValueError('unregistered runtime configuration')
    if spec['schema']!='real-opportunity-v1' or runtime['schema_version']!='real-pilot-v1':raise ValueError('wrong protocol schema')
    validator=smoke_cli();dependencies={}
    for name in [*pointer['offline_gates'].values(),'reviews/bridge_review.json']:
        validator.validate_gate(run/name,run);dependencies[name]=sha(run/name)
    review=json.loads((run/'reviews/bridge_review.json').read_text())
    for name,value in review.get('dependency_hashes',{}).items():
        if sha(ROOT/name)!=value:raise RuntimeError('review runtime dependency changed: '+name)
    waiver=spec['prerequisite_exception']
    if waiver.get('05_smoke')=='skipped_by_user':
        decision=json.loads((run/'p3_evidence/user_override_06.json').read_text())
        if (decision.get('user_instruction')!='跳过smoke，直接执行06_sol_p3_prepare.md'
                or decision.get('smoke_skipped') is not True or decision.get('run_id')!=pointer['run_id']):
            raise RuntimeError('missing explicit user prerequisite exception')
        dependencies['p3_evidence/user_override_06.json']=sha(run/'p3_evidence/user_override_06.json')
    else:
        for name in ('gates/p1_live.json','gates/p2_live.json'):
            validator.validate_gate(run/name,run);dependencies[name]=sha(run/name)
    for qps,seed in zip(spec['probe']['qps'],spec['probe']['seeds']):load_frozen_trace(run,spec,seed=seed,qps=qps,phase='probe')
    for qps in spec['probe']['qps']:
        for seed in spec['opportunity']['seeds']:load_frozen_trace(run,spec,seed=seed,qps=qps,phase='opportunity')
    if require_ready:
        gate=validator.validate_gate(run/'gates/p3_ready.json',run)
        if gate.get('stage')!='P3_CPU_preparation' or gate.get('dependency_hashes')!=dependencies:
            raise RuntimeError('P3 ready dependency hash mismatch')
    return dependencies


def plan(args,spec,runtime):
    dependencies=validate_inputs(args,spec,runtime,require_ready=False)
    reasons=[];validator=smoke_cli();gate=args.output_dir/'gates/p3_ready.json'
    try:
        value=validator.validate_gate(gate,args.output_dir)
        if value.get('stage')!='P3_CPU_preparation' or value.get('dependency_hashes')!=dependencies:
            raise RuntimeError('P3 ready dependency hash mismatch')
    except (OSError,ValueError,RuntimeError,KeyError) as exc:reasons.append('P3 preparation gate: '+str(exc))
    if not args.manage_services:reasons.append('opportunity requires owned managed service and automatic teardown')
    maximum=maximum_duration(spec,runtime);budget=readonly_budget(args.output_dir)
    if budget['exhausted'] or min(budget['remaining']['total_experiment_wall_time_sec'],budget['remaining']['P0_P3_sec'])<maximum:
        reasons.append('persistent P0-P3/total budget insufficient for complete '+str(maximum)+' second maximum')
    if (args.output_dir/'opportunity_runs'/args.invocation_id).exists():reasons.append('invocation already attempted; no automatic retry or overwrite')
    if reasons and not args.dry_run:raise RuntimeError('; '.join(reasons))
    return {'stage':'opportunity','dry_run':args.dry_run,'GPU_will_start':bool(not args.dry_run and not reasons),
        'ready_for_live':not reasons,'blocked_reasons':reasons,'estimated_max_sec':maximum,
        'episodes':{'probes':6,'paired_opportunity':12},'protocol_hash':digest(spec),'runtime_config_hash':sha(args.config),
        'dependency_hashes':dependencies,'remaining_budget':budget['remaining'],
        'prior_smoke':'skipped_by_user_not_verified','C_ref':None,'N_ref':None,'PPO_implemented':False}


def main(argv=None):
    args=parse_args(argv)
    try:
        import yaml
        spec=yaml.safe_load(args.opportunity_spec.read_text());runtime=yaml.safe_load(args.config.read_text())
        intent=plan(args,spec,runtime)
        if args.dry_run:print(json.dumps(intent,indent=2));return 0
        from rl_scheduler.real_service import ManagedService
        from rl_scheduler.real_opportunity_runner import run_opportunity
        folder=args.output_dir/'opportunity_runs'/args.invocation_id;folder.mkdir(parents=True,exist_ok=False)
        atomic_json(folder/'intent.json',intent)
        service=ManagedService(ROOT,args.output_dir,args.config.resolve());started=False
        try:
            # Existing guard and owner ledger enforce the original cumulative budget.
            BudgetLedger(args.output_dir).require(intent['estimated_max_sec'])
            started=True;service.start()
            asyncio.run(smoke_cli().wait_ready(runtime,BudgetLedger(args.output_dir)))
            result=asyncio.run(run_opportunity(spec,runtime,run_dir=args.output_dir,invocation_id=args.invocation_id))
        except BaseException as exc:
            atomic_json(folder/'failure.json',{'status':'incomplete','reason':type(exc).__name__+':'+str(exc)})
            raise
        finally:
            if started:service.stop()
        # Revalidate code/protocol/dependencies before publishing an aggregate gate.
        dependencies=validate_inputs(args,spec,runtime)
        if dependencies!=intent['dependency_hashes']:raise RuntimeError('P3 dependencies changed during run')
        ready=json.loads((args.output_dir/'gates/p3_ready.json').read_text())
        report={**result,'stage':'P3_opportunity','scope':'exploratory Stay/Balanced comparison, not PPO proof',
            'source_hashes':ready['source_hashes'],'source_hash':ready['source_hash'],
            'config_hashes':ready['config_hashes'],'config_hash':ready['config_hash'],
            'dependency_hashes':dependencies,'checks':[{'name':'paired_opportunity','status':result['status']},
                {'name':'owned_cleanup','status':'passed'}],
            'evidence_hashes':{str(p.relative_to(args.output_dir)):sha(p) for p in folder.rglob('*') if p.is_file()},
            'remaining_budget':readonly_budget(args.output_dir)['remaining'],'prior_smoke':'skipped_by_user_unverified'}
        destination=args.output_dir/'opportunity_gate.json'
        if destination.exists():raise RuntimeError('existing opportunity result; preserve attempt')
        atomic_json(destination,report)
        print(json.dumps({'status':report['status'],'gate':str(destination)}))
        return 0 if report['status']=='passed' else 1
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print(json.dumps({'status':'incomplete','reason':str(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
