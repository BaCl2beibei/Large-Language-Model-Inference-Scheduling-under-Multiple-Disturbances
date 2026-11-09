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
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from rl_scheduler.real_budget import BudgetLedger,atomic_json
from rl_scheduler.real_bridge import digest
from rl_scheduler.real_opportunity import maximum_duration,readonly_budget,load_frozen_trace
from rl_scheduler.real_opportunity_budget_override import (OpportunityManagedService, OpportunityBudgetLedger,
                                                          close_lease, continue_lease, create_lease)


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
    p.add_argument('--continuation-id',help='one explicit manual recovery continuation under the original invocation lease')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--manage-services',action='store_true')
    p.add_argument('--user-authorized-budget-override',action='store_true',
                   help='bypass cumulative budget checks for this single fixed P3 invocation; retain clock accounting and enforce its invocation lease')
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
    if getattr(args,'user_authorized_budget_override',False):
        try:
            authorization=json.loads((args.output_dir/'budget_override_authorization.json').read_text())
            expected_run=json.loads((args.output_dir/'budget.json').read_text())['run_id']
            if authorization.get('run_id')!=expected_run or authorization.get('invocation_id')!=args.invocation_id:
                reasons.append('single-invocation budget override authorization missing or mismatched')
        except (OSError,ValueError,KeyError,TypeError):
            reasons.append('single-invocation budget override authorization missing or invalid')
    if (not getattr(args,'user_authorized_budget_override',False) and
            (budget['exhausted'] or min(budget['remaining']['total_experiment_wall_time_sec'],budget['remaining']['P0_P3_sec'])<maximum)):
        reasons.append('persistent P0-P3/total budget insufficient for complete '+str(maximum)+' second maximum')
    invocation_dir=args.output_dir/'opportunity_runs'/args.invocation_id
    continuation_id=getattr(args,'continuation_id',None)
    prior_probes=None;prior_probe_disposition=None;continuation_parent_lease=None;continuation_runtime_sec=None;remaining=None
    continuation_deadline_extension_sec=None
    if continuation_id:
        if continuation_id not in ('recovery2','recovery3','recovery4','recovery5','recovery6') or not re.fullmatch(r'[a-z0-9_]+',continuation_id):
            reasons.append('only the named recovery2/recovery3/recovery4/recovery5/recovery6 continuations are allowed')
        continuation_dir=invocation_dir/'attempts'/continuation_id
        if continuation_dir.exists():reasons.append('manual recovery already attempted; preserve prior evidence')
        previous=invocation_dir/'opportunity_gate.json'
        try:
            old=json.loads(previous.read_text())
            if old.get('status')!='incomplete' or 'ServerDisconnectedError' not in old.get('reason',''):
                reasons.append('recovery2 requires the retained original API disconnect attempt')
        except (OSError,ValueError):reasons.append('prior incomplete invocation evidence missing')
        recovery1_dir=invocation_dir/'attempts'/'recovery1'
        recovery2_dir=invocation_dir/'attempts'/'recovery2'
        recovery1_probe_file=recovery1_dir/'probes.json'
        if continuation_id=='recovery2':
            try:
                prior_gate=json.loads((recovery1_dir/'opportunity_gate.json').read_text())
                if (prior_gate.get('status')!='incomplete'
                        or 'controller expired or external intervention' not in prior_gate.get('reason','')
                        or prior_gate.get('probes_completed')!=2 or prior_gate.get('episodes_completed')!=0):
                    reasons.append('recovery2 requires recovery1 controller-expiry evidence after two probes and no pairs')
                prior_probes=json.loads(recovery1_probe_file.read_text())
                probe_keys=[(r.get('scene'),r.get('qps'),r.get('seed')) for r in prior_probes]
                expected={('A',3,7101),('B',3,7101)}
                if (set(probe_keys)!=expected or len(probe_keys)!=len(expected)
                        or any(r.get('phase')!='probe' or r.get('policy')!='Stay' or r.get('status')!='passed'
                               for r in prior_probes)
                        or any(r.get('mapping')!=prior_probes[0].get('mapping') for r in prior_probes)):
                    reasons.append('recovery2 requires the two audited qps3 Stay probes from recovery1 with one stable mapping')
                else:
                    invalid_submission=[f'{r["scene"]}/qps{r["qps"]}/seed{r["seed"]}' for r in prior_probes
                        if r.get('request_submission_audit',{}).get('status')!='passed']
                    if invalid_submission:
                        prior_probe_disposition={'status':'excluded_incomplete_submission_audit',
                            'source':'recovery1/probes.json','excluded_keys':invalid_submission,
                            'reason':'pre-fix episodes did not prove every frozen request was sent and accepted; rerun all six probes'}
                        prior_probes=[]
                    else:
                        prior_probe_disposition={'status':'reused_after_submission_audit','source':'recovery1/probes.json',
                            'reused_keys':[f'{r["scene"]}/qps{r["qps"]}/seed{r["seed"]}' for r in prior_probes]}
            except (OSError,ValueError,KeyError,TypeError):
                reasons.append('recovery1 probe and gate evidence missing or invalid')
            interrupted=args.output_dir/'runs'/(args.invocation_id+'_recovery1_probe_A_6')/'interrupted.json'
            try:
                row=json.loads(interrupted.read_text())
                if (row.get('status')!='incomplete' or row.get('phase')!='probe' or row.get('scene')!='A'
                        or row.get('qps')!=6 or row.get('seed')!=7102):
                    reasons.append('recovery2 requires the retained interrupted A/qps6/seed7102 probe')
            except (OSError,ValueError,KeyError,TypeError):reasons.append('recovery1 interrupted-probe evidence missing or invalid')
            expected_latest='recovery1'
            prior_status={'probes_completed':2,'paired_runs_started':0}
            continuation_parent_lease=recovery1_dir/'budget_override_lease.json'
        elif continuation_id=='recovery3':
            try:
                failed=json.loads((recovery2_dir/'opportunity_gate.json').read_text())
                if (failed.get('status')!='incomplete'
                        or 'controller expired or external intervention' not in failed.get('reason','')
                        or failed.get('probes_completed')!=1 or failed.get('episodes_completed')!=0):
                    reasons.append('recovery3 requires the retained invalid recovery2 after one probe and no pairs')
                interrupted=args.output_dir/'runs'/(args.invocation_id+'_recovery2_probe_B_3')/'interrupted.json'
                row=json.loads(interrupted.read_text())
                if (row.get('status')!='incomplete' or row.get('phase')!='probe' or row.get('scene')!='B'
                        or row.get('qps')!=3 or row.get('seed')!=7101):
                    reasons.append('recovery3 requires the retained interrupted B/qps3/seed7101 workload')
            except (OSError,ValueError,KeyError,TypeError):
                reasons.append('recovery2 incomplete probe and gate evidence missing or invalid')
            prior_probes=[]
            prior_probe_disposition={'status':'excluded_invalid_controller_attempt',
                'source':'recovery2/probes.json','excluded_keys':['A/qps3/seed7101'],
                'reason':'recovery2 stopped in B before fixed arrivals/deadline completed; all six probes rerun on the new service epoch'}
            expected_latest='recovery2'
            prior_status={'probes_completed':1,'paired_runs_started':0}
            continuation_parent_lease=recovery2_dir/'budget_override_lease.json'
        elif continuation_id=='recovery4':
            recovery3_dir=invocation_dir/'attempts'/'recovery3'
            try:
                failed=json.loads((recovery3_dir/'failure.json').read_text())
                if (failed.get('status')!='incomplete'
                        or 'supervisorctl' not in failed.get('reason','')
                        or 'start' not in failed.get('reason','')):
                    reasons.append('recovery4 requires the retained recovery3 supervisor-start failure')
                if (recovery3_dir/'opportunity_gate.json').exists():
                    reasons.append('recovery4 requires recovery3 to have stopped before any episode score')
                if (args.output_dir/'runs'/(args.invocation_id+'_recovery3_probe_A_3')).exists():
                    reasons.append('recovery4 requires recovery3 to have started no request workload')
            except (OSError,ValueError,KeyError,TypeError):
                reasons.append('recovery3 pre-traffic supervisor failure evidence missing or invalid')
            prior_probes=[]
            prior_probe_disposition={'status':'no_prior_probe_rows','source':'recovery3/failure.json',
                'excluded_keys':[],'reason':'recovery3 failed before starting its owned service; recovery4 reruns all six probes'}
            expected_latest='recovery3'
            expected_summary_status='incomplete_before_service'
            prior_status={'probes_completed':0,'paired_runs_started':0}
            continuation_parent_lease=recovery3_dir/'budget_override_lease.json'
        elif continuation_id=='recovery5':
            recovery4_dir=invocation_dir/'attempts'/'recovery4'
            try:
                failed=json.loads((recovery4_dir/'failure.json').read_text())
                gate=json.loads((recovery4_dir/'opportunity_gate.json').read_text())
                interrupted=args.output_dir/'runs'/(args.invocation_id+'_recovery4_probe_A_3')/'interrupted.json'
                row=json.loads(interrupted.read_text())
                scoring=row.get('workload_scoring',{})
                if (failed.get('status')!='incomplete' or 'frozen trace submission invalid' not in failed.get('reason','')
                        or gate.get('status')!='incomplete' or gate.get('probes_completed')!=0
                        or gate.get('episodes_completed')!=0
                        or row.get('phase')!='probe' or row.get('scene')!='A' or row.get('qps')!=3
                        or row.get('seed')!=7101 or scoring.get('offered')!=99
                        or scoring.get('sent_by_deadline')!=97 or scoring.get('completed_by_deadline')!=97):
                    reasons.append('recovery5 requires the retained recovery4 A/qps3 attempt with 97 accepted and completed tasks and no pairs')
            except (OSError,ValueError,KeyError,TypeError):
                reasons.append('recovery4 strict-offered-denominator attempt evidence missing or invalid')
            prior_probes=[]
            prior_probe_disposition={'status':'excluded_old_completion_denominator_attempt',
                'source':'recovery4/interrupted.json','excluded_keys':['A/qps3/seed7101'],
                'reason':'user clarified that scheduler completion rate uses service-accepted tasks; recovery5 reruns all six probes under the amended metric'}
            expected_latest='recovery4'
            expected_summary_status='incomplete_after_start'
            prior_status={'probes_completed':0,'paired_runs_started':0}
            continuation_parent_lease=recovery4_dir/'budget_override_lease.json'
        else:
            recovery5_dir=invocation_dir/'attempts'/'recovery5'
            try:
                failed=json.loads((recovery5_dir/'failure.json').read_text())
                lease=json.loads((recovery5_dir/'budget_override_lease.json').read_text())
                run_artifacts=list((args.output_dir/'runs').glob(args.invocation_id+'_recovery5_*'))
                if (failed.get('status')!='incomplete' or 'KeyboardInterrupt' not in failed.get('reason','')
                        or lease.get('closed_wall') is None or run_artifacts):
                    reasons.append('recovery6 requires the retained recovery5 startup interruption before any request traffic')
            except (OSError,ValueError,KeyError,TypeError):
                reasons.append('recovery5 stopped-service/no-traffic evidence missing or invalid')
            prior_probes=[]
            prior_probe_disposition={'status':'no_probe_rows_user_steered_startup_stop',
                'source':'recovery5/failure.json','excluded_keys':[],
                'reason':'user clarified to retain controller ownership and remove only all-offered submission gating; recovery5 was interrupted before request traffic'}
            expected_latest='recovery5'
            expected_summary_status='incomplete_before_request_traffic'
            prior_status={'probes_completed':0,'paired_runs_started':0}
            continuation_parent_lease=recovery5_dir/'budget_override_lease.json'
        try:
            summary=json.loads((args.output_dir/'p3_evidence/07_live_attempt_summary.json').read_text())
            prior=[x for x in summary.get('attempts',[]) if x.get('invocation_id')==args.invocation_id
                   and x.get('attempt_id')==expected_latest]
            if (summary.get('latest_attempt_id')!=expected_latest or len(prior)!=1
                    or prior[0].get('status', 'incomplete_after_start')!=locals().get('expected_summary_status','incomplete_after_start')
                    or any(prior[0].get(key)!=value for key,value in prior_status.items())
                    or prior[0].get('owned_services_stopped') is not True or prior[0].get('gpu_released') is not True):
                reasons.append(continuation_id+' requires the retained stopped-service/no-pairs summary for '+expected_latest)
        except (OSError,ValueError,KeyError,TypeError):reasons.append('prior live attempt summary missing or invalid')
        original_lease=invocation_dir/'budget_override_lease.json'
        try:
            lease=json.loads(original_lease.read_text())
            remaining=(int(lease['deadline_monotonic_ns'])-__import__('time').monotonic_ns())/1e9
            if lease.get('run_id')!=json.loads((args.output_dir/'budget.json').read_text())['run_id']:
                reasons.append('original invocation lease run ID mismatch')
            if lease.get('invocation_id')!=args.invocation_id or lease.get('closed_wall') is None:
                reasons.append('original invocation lease is not cleanly closed')
            parent=json.loads(continuation_parent_lease.read_text())
            if (parent.get('closed_wall') is None or parent.get('invocation_id')!=args.invocation_id
                    or parent.get('run_id')!=lease.get('run_id')):
                reasons.append(continuation_id+' parent lease is not a clean child of the original invocation lease')
            else:
                # Follow immutable continuation parents back to the original
                # lease. Earlier recovery code wrote an immediate-parent
                # deadline into original_deadline_monotonic_ns; validate each
                # link instead of assuming every child stores the root value.
                lineage=parent;seen=set();lineage_ok=True
                while lineage.get('continuation_of'):
                    lineage_path=Path(lineage['continuation_of']).resolve()
                    if lineage_path in seen or not lineage_path.is_file():
                        lineage_ok=False;break
                    seen.add(lineage_path)
                    ancestor=json.loads(lineage_path.read_text())
                    child_deadline=lineage.get('original_deadline_monotonic_ns',lineage.get('deadline_monotonic_ns'))
                    if (ancestor.get('closed_wall') is None or ancestor.get('run_id')!=lease.get('run_id')
                            or ancestor.get('invocation_id')!=args.invocation_id
                            or child_deadline!=ancestor.get('deadline_monotonic_ns')):
                        lineage_ok=False;break
                    lineage=ancestor
                if (not lineage_ok or lineage.get('run_id')!=lease.get('run_id')
                        or lineage.get('invocation_id')!=args.invocation_id
                        or lineage.get('deadline_monotonic_ns')!=lease.get('deadline_monotonic_ns')):
                    reasons.append(continuation_id+' parent lease lineage does not end at the original invocation lease')
        except (OSError,ValueError,KeyError,TypeError):reasons.append('original invocation lease missing or invalid')
        if continuation_parent_lease and prior_probes is not None and remaining is not None:
            probe=spec['probe'];h=spec['horizons']
            remaining_probe_runs=probe['runs']-len(prior_probes)
            continuation_runtime_sec=(runtime['service']['startup_timeout_sec']+60
                +remaining_probe_runs*(probe['arrival_sec']+probe['drain_sec']+h['cleanup_sec'])
                +spec['opportunity']['runs']*(h['arrival_sec']+h['drain_sec']+h['cleanup_sec'])+120.)
            continuation_deadline_extension_sec=max(0.,continuation_runtime_sec-remaining)
        if (args.output_dir/'opportunity_gate.json').exists():reasons.append('aggregate opportunity gate already exists; preserve it')
    elif invocation_dir.exists():
        reasons.append('invocation already attempted; no automatic retry or overwrite')
    if reasons and not args.dry_run:raise RuntimeError('; '.join(reasons))
    return {'stage':'opportunity','dry_run':args.dry_run,'GPU_will_start':bool(not args.dry_run and not reasons),
        'ready_for_live':not reasons,'blocked_reasons':reasons,'estimated_max_sec':maximum,
        'episodes':{'probes':6,'paired_opportunity':12},'protocol_hash':digest(spec),'runtime_config_hash':sha(args.config),
        'dependency_hashes':dependencies,'remaining_budget':budget['remaining'],
        'user_authorized_budget_override':getattr(args,'user_authorized_budget_override',False),
        'continuation_id':continuation_id,
        'continuation_prior_probe_count':len(prior_probes) if prior_probes is not None else 0,
        'continuation_prior_probe_disposition':prior_probe_disposition,
        'continuation_runtime_sec':continuation_runtime_sec,
        'continuation_parent_lease_path':(str(continuation_parent_lease)
            if continuation_parent_lease is not None else None),
        'continuation_prior_probes_path':(str(recovery1_probe_file)
            if continuation_id and prior_probes else None),
        'continuation_prior_probes_sha256':(sha(recovery1_probe_file)
            if continuation_id and prior_probes else None),
        'original_invocation_lease_remaining_sec':(remaining if continuation_id and 'remaining' in locals() else None),
        'continuation_deadline_extension_sec':continuation_deadline_extension_sec,
        'prior_smoke':'skipped_by_user_not_verified','C_ref':None,'N_ref':None,'PPO_implemented':False}


def main(argv=None):
    args=parse_args(argv)
    # Supervisor resolves its program command independently of this CLI's cwd.
    # Canonicalise paths before writing managed-service wrappers so a relative
    # --output-dir cannot produce a non-runnable command in supervisor.conf.
    args.output_dir=args.output_dir.resolve()
    args.config=args.config.resolve()
    args.opportunity_spec=args.opportunity_spec.resolve()
    try:
        import yaml
        spec=yaml.safe_load(args.opportunity_spec.read_text());runtime=yaml.safe_load(args.config.read_text())
        intent=plan(args,spec,runtime)
        if args.dry_run:print(json.dumps(intent,indent=2));return 0
        from rl_scheduler.real_opportunity_runner import run_opportunity
        authorization=json.loads((args.output_dir/'budget_override_authorization.json').read_text())
        if (not args.user_authorized_budget_override or authorization.get('invocation_id')!=args.invocation_id
                or authorization.get('run_id')!=json.loads((args.output_dir/'budget.json').read_text())['run_id']):
            raise RuntimeError('single-invocation budget override authorization missing or mismatched')
        prior_probes=None
        if args.continuation_id and intent.get('continuation_prior_probes_path'):
            prior_probes_path=Path(intent['continuation_prior_probes_path'])
            if sha(prior_probes_path)!=intent['continuation_prior_probes_sha256']:
                raise RuntimeError('validated prior probe file changed before execution')
            prior_probes=json.loads(prior_probes_path.read_text())
        elif args.continuation_id:
            prior_probes=[]
        invocation_dir=args.output_dir/'opportunity_runs'/args.invocation_id
        if args.continuation_id:
            continuation_parent_lease=Path(intent['continuation_parent_lease_path'])
            if not continuation_parent_lease.is_file():
                raise RuntimeError('continuation parent lease path is not a file')
            folder=invocation_dir/'attempts'/args.continuation_id
            folder.mkdir(parents=True,exist_ok=False)
            lease_path=folder/'budget_override_lease.json'
            lease=continue_lease(continuation_parent_lease,lease_path,
                run_id=authorization['run_id'],invocation_id=args.invocation_id,
                max_runtime_sec=intent['continuation_runtime_sec'],allow_extend=True)
            intent['continuation_of']=str(continuation_parent_lease.relative_to(args.output_dir))
            intent['original_lease_deadline_preserved']=lease['deadline_extended_sec']==0
            intent['lease_deadline_extension_sec']=lease['deadline_extended_sec']
        else:
            folder=invocation_dir
            folder.mkdir(parents=True,exist_ok=False)
            lease_path=folder/'budget_override_lease.json'
            create_lease(lease_path,run_id=authorization['run_id'],invocation_id=args.invocation_id,
                         max_runtime_sec=intent['estimated_max_sec']+120.)
        intent['attempt_artifact_dir']=str(folder.relative_to(args.output_dir))
        intent['attempt_id']=args.continuation_id or 'initial'
        intent['invocation_lease_path']=str(lease_path.relative_to(args.output_dir))
        started=False
        service=None
        try:
            service=OpportunityManagedService(ROOT,args.output_dir,args.config.resolve(),
                invocation_id=args.invocation_id,lease_path=lease_path,artifact_dir=folder,
                service_tag=(args.invocation_id+'_'+args.continuation_id) if args.continuation_id else args.invocation_id)
            intent['effective_service_runtime_override']=service.runtime_override
            intent['effective_service_runtime_sha256']=sha(service.config_path)
            atomic_json(folder/'intent.json',intent)
            budget=OpportunityBudgetLedger(args.output_dir,lease_path)
            budget.activate('P0_P3',required_sec=0.)
            started=True;service.start()
            asyncio.run(smoke_cli().wait_ready(runtime,budget))
            result=asyncio.run(run_opportunity(spec,runtime,run_dir=args.output_dir,invocation_id=args.invocation_id,
                budget_override_lease=lease_path,artifact_dir=folder,
                episode_prefix=(args.invocation_id+'_'+args.continuation_id) if args.continuation_id else args.invocation_id,
                prior_probes=prior_probes))
        except BaseException as exc:
            atomic_json(folder/'failure.json',{'status':'incomplete','reason':type(exc).__name__+':'+str(exc)})
            raise
        finally:
            try:
                if started and service is not None:service.stop()
            finally:
                close_lease(lease_path, 'opportunity_cli_cleanup')
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
            'remaining_budget':readonly_budget(args.output_dir)['remaining'],'prior_smoke':'skipped_by_user_unverified',
            'user_authorized_budget_override':True,'budget_override_lease':str(lease_path.relative_to(args.output_dir)),
            'budget_override_lease_hash':sha(lease_path),
            'attempt_id':args.continuation_id or 'initial',
            'attempt_artifact_dir':str(folder.relative_to(args.output_dir)),
            'continuation_of':intent.get('continuation_of'),
            'original_lease_deadline_preserved':bool(intent.get('original_lease_deadline_preserved',False)),
            'lease_deadline_extension_sec':intent.get('lease_deadline_extension_sec',0.),
            'effective_service_runtime_override':intent['effective_service_runtime_override'],
            'effective_service_runtime_sha256':intent['effective_service_runtime_sha256']}
        destination=args.output_dir/'opportunity_gate.json'
        if destination.exists():raise RuntimeError('existing opportunity result; preserve attempt')
        atomic_json(destination,report)
        print(json.dumps({'status':report['status'],'gate':str(destination)}))
        return 0 if report['status']=='passed' else 1
    except Exception as exc:
        print(json.dumps({'status':'incomplete','reason':str(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
