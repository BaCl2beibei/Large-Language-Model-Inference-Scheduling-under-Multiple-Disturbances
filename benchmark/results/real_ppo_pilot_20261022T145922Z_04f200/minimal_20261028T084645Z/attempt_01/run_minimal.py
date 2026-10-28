"""One-start, two-episode real feasibility check under protocol.json amendment."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

CASE=Path(__file__).resolve().parent
PARENT=CASE.parent
ROOT=CASE.parents[3]
# CASE -> original run -> results -> benchmark -> repository.
if not (ROOT/'rl_scheduler').is_dir():ROOT=CASE.parents[3]
sys.path.insert(0,str(ROOT))
from rl_scheduler.real_bridge import RealBridge,digest
from rl_scheduler.real_budget import BudgetLedger,atomic_json
from rl_scheduler.real_env import RealEnv,RealEnvConfig
from rl_scheduler.real_runner import LiveEpisodeIO
from rl_scheduler.real_opportunity_runner import reconcile_migrations,route_audit
from rl_scheduler.real_reward import RewardScales,reconstruct_rewards
from rl_scheduler.real_service import ManagedService,identity_alive


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validator():
    spec=importlib.util.spec_from_file_location('mini_original_cli',ROOT/'benchmark/run_real_ppo.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def preflight():
    protocol=json.loads((CASE/'protocol.json').read_text());baseline=json.loads((CASE/'baseline.json').read_text())
    assert str(PARENT)==protocol['parent_run_dir']
    assert sha(__file__)==protocol['code_sha256'],'minimal runner changed after freeze'
    assert sha(CASE/'runtime.yaml')==protocol['runtime_sha256'],'runtime changed after freeze'
    for name,value in protocol['trace_hashes'].items():assert sha(CASE/'traces'/name)==value,name
    pointer=json.loads((ROOT/'rl_scheduler/runs/real_ppo_handoff/active_run.json').read_text())
    assert Path(pointer['run_dir'])==PARENT
    for name,value in baseline['source_hashes'].items():assert sha(ROOT/name)==value,name
    for name,value in baseline['root_config_hashes'].items():assert sha(ROOT/name)==value,name
    assert sha(PARENT/'experiment_clock.json')==baseline['parent_clock_sha256']
    for name in [*pointer['offline_gates'].values(),'reviews/bridge_review.json']:validator().validate_gate(PARENT/name,PARENT)
    assert not (CASE/'result.json').exists(),'attempt already exists; preserve it'
    # Two bounded real cohorts, frozen before any GPU service is started.
    return protocol


def save_jsonl(path,rows):
    path=Path(path)
    with path.open('w') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


async def episode(protocol,case,bridge,traffic,budget):
    label=case['label'];episode_id='mini_'+label
    trace=json.loads((CASE/'traces'/('mini_'+label+'.json')).read_text())
    io=LiveEpisodeIO(bridge,traffic,run_id=budget.input['run_id'],episode_id=episode_id,run_dir=CASE,
        trace=trace,sampling=protocol['sampling'],arrival_sec=case['arrival_sec'],
        route_shares=case.get('initial_route'))
    records=[];accepted=None;accepted_payload=None;duplicate_proof=None;rejection=None;outcome=None
    def save(record):
        records.append(record)
        with (io.directory/'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    env=RealEnv(bridge,io,budget,run_id=budget.input['run_id'],episode_id=episode_id,ledger_factory=io.ledger,
        config=RealEnvConfig(arrival_sec=case['arrival_sec'],drain_sec=case['drain_sec'],cleanup_sec=60.),decision_sink=save)
    try:
        obs,_=await env.reset(mode=case['mode'],controller_version='minimal-feasibility-v1')
        if label=='migration':
            # Intentional known rejection: wrong candidate, no real migration.
            rejection=await bridge.call('submit_migration',lease_id=bridge.lease['lease_id'],
                decision_id=episode_id+':intentional-rejection',src_id=env.snapshot['mapping'][0][0],
                dst_id=env.snapshot['mapping'][1][0],expected_candidate_id='__deliberately_stale_candidate__',
                snapshot_version=env.snapshot['version'])
            atomic_json(io.directory/'explicit_rejection.json',rejection)
            if rejection.get('terminal_status')!='rejected':raise RuntimeError('intentional stale candidate was not explicitly rejected')
        attempts=0
        while env.state in ('arrival','drain'):
            if accepted is not None:
                receipt=await bridge.query_receipt(accepted)
                env.receipts[accepted]=receipt
                if receipt.get('terminal_status')=='succeeded' and duplicate_proof is None:
                    duplicate_proof=await bridge.call('submit_migration',**accepted_payload)
                    atomic_json(io.directory/'duplicate_receipt.json',duplicate_proof)
                    if (duplicate_proof.get('migration_id')!=receipt.get('migration_id')
                            or duplicate_proof.get('terminal_status')!='succeeded'):
                        raise RuntimeError('duplicate decision did not return identical known migration')
                if receipt.get('terminal_status')=='unknown' or receipt.get('receipt_status')=='unknown':
                    raise RuntimeError('uncertain execution; stop new actions and clean up')
            action=0
            if label=='migration' and accepted is None and attempts<case['maximum_nonstay_decisions']:
                legal=[a for a in range(1,13) if obs['raw']['action_mask'][a]]
                if legal:action=legal[0]
            obs,_,done,_,info=await env.step(action,policy_version='minimal-probe-v1')
            if action:
                attempts+=1;record=records[-1];receipt=info['receipt']
                if receipt.get('receipt_status')=='accepted':
                    accepted=record['decision_id'];accepted_payload=dict(bridge.pending[accepted])
                elif receipt.get('receipt_status')=='unknown':raise RuntimeError('unknown submit; no new decision issued')
            if done:break
        result=await env.finish()
        through=min(cp.through_ns for cp in env.ledger.checkpoints.values())
        env.ledger.export(io.directory,cleanup_through_ns=through)
        atomic_json(io.directory/'episode.json',result)
        events=[e for e in env.ledger.events.values() if env.start_ns<=e.occurred_ns<=through]
        migrations=reconcile_migrations(records,env.receipts,events)
        routing=route_audit(io,env.ledger,events,io.mapping)
        atomic_json(io.directory/'route_audit.json',routing)
        save_jsonl(io.directory/'migrations.jsonl',migrations)
        if result['status']!='passed':raise RuntimeError('episode audit incomplete: '+str(result['invalid_reasons']))
        if routing['status']!='passed':raise RuntimeError('initial engine admission routing incomplete')
        if result['score']['fixed']['completed']!=len(trace):raise RuntimeError('not all minimal requests completed normally by deadline')
        if label=='migration':
            succeeded=[r for r in result['final_receipts'].values() if r.get('terminal_status')=='succeeded']
            if len(succeeded)!=1 or len([m for m in migrations if m['attempt_ns'] is not None])!=1:
                raise RuntimeError('minimum actual successful migration not established')
            receipt=succeeded[0]
            if receipt.get('actual_request_id')!=receipt.get('expected_candidate_id') or not receipt.get('copied_blocks',0)>0:
                raise RuntimeError('actual migrated request/blocks not established')
            if duplicate_proof is None:raise RuntimeError('duplicate idempotency proof missing')
        edges=[env.start_ns]+[r['monotonic_time'] for r in records[1:] if r['monotonic_time']<env.deadline_ns]+[env.deadline_ns]
        atomic_json(io.directory/'reward_inputs.json',{'edges_ns':edges,'cleanup_through_ns':through,'migrations':migrations})
        await bridge.call('release_episode',run_id=budget.input['run_id'],episode_id=episode_id)
        summary={'label':label,'status':'passed','score':result['score']['fixed'],'receipts':result['final_receipts'],
            'route_audit':routing,'initial_rejection':rejection,'duplicate_proof':duplicate_proof,'migrations':migrations,
            'decision_latency_ms':[r['cpu_decision_latency_ms'] for r in records],
            'episode_directory':str(io.directory.relative_to(CASE))}
        outcome=(summary,env.ledger,edges,migrations,through)
        atomic_json(io.directory/'minimal_summary.json',summary)
        return outcome
    finally:
        if outcome is None:
            try:atomic_json(io.directory/'interrupted.json',await env.close())
            except BaseException as exc:atomic_json(io.directory/'interrupted.json',{'status':'incomplete','reason':type(exc).__name__+':'+str(exc)})
        await io.close()


async def traffic_validation(protocol,cfg,budget):
    import aiohttp
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.)) as control, \
               aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=75.)) as traffic:
        bridge=RealBridge(cfg['service']['base_url'],control);results=[]
        for case in protocol['cases']:
            item=await episode(protocol,case,bridge,traffic,budget);results.append(item)
            atomic_json(CASE/'progress.json',[x[0] for x in results])
        baseline=results[0][0]['score'];duration=baseline['duration_sec']
        raw_c=baseline['completed']/duration;raw_n=baseline['outstanding_request_seconds']/duration
        scales=RewardScales(max(1.,raw_c),max(1.,raw_n),'minimal baseline only; diagnostic, not production opportunity scales')
        atomic_json(CASE/'mini_reward_scales.json',{'scope':'mini-only diagnostic reward path check, not formal C_ref/N_ref',
            'raw_C_ref':raw_c,'raw_N_ref':raw_n,'used':asdict(scales),'original_reward_initial_yaml_modified':False})
        for summary,ledger,edges,migrations,through in results:
            rewards=reconstruct_rewards(ledger,edges,migrations,scales=scales,cleanup_through_ns=through,valid_episode=True)
            save_jsonl(CASE/summary['episode_directory']/'reward_windows.jsonl',rewards)
            if not all(math.isfinite(r['reward']) for r in rewards):raise RuntimeError('nonfinite live reward reconstruction')
            summary['diagnostic_reward_sum']=sum(r['reward'] for r in rewards)
            summary['reward_scope']='measured mini-only diagnostic scales; original P3 scales remain unmeasured'
        return [item[0] for item in results]


def stop_owned(service):
    record={}
    try:
        r=service._supervisor('stop',service.server_name,check=False,timeout_sec=30.)
        record['supervisor_server_exit_code']=r.returncode
    except subprocess.TimeoutExpired:record['supervisor_server_timeout']=True
    service.owner.stop(timeout=10.)
    try:
        r=service._supervisor('stop',service.guard_name,check=False,timeout_sec=10.)
        record['supervisor_guard_exit_code']=r.returncode
    except subprocess.TimeoutExpired:record['supervisor_guard_timeout']=True
    if service.owner.path.exists():
        remaining=[x for x in json.loads(service.owner.path.read_text())['processes'] if identity_alive(x['identity'])]
        if remaining:raise RuntimeError('owned process cleanup unconfirmed')
    record['owned_processes_remaining']=0
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    protocol=preflight()
    import yaml
    cfg=yaml.safe_load((CASE/'runtime.yaml').read_text())
    if args.dry_run:
        print(json.dumps({'status':'ready_for_amended_minimal_only','GPU_will_start':False,'maximum_wall_sec':900,
            'case_count':2,'request_count':3,'original_parent_budget_clock_reset':False},indent=2));return 0
    lock=(PARENT.parent/'.real_ppo_cluster.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    service=ManagedService(ROOT,CASE,CASE/'runtime.yaml');budget=BudgetLedger(CASE)
    started=False;began=time.time();report={'status':'incomplete','started_wall':began,'protocol_sha256':sha(CASE/'protocol.json')}
    try:
        budget.activate('P0_P3',required_sec=900.)
        started=True;service.start()
        asyncio.run(validator().wait_ready(cfg,budget))
        atomic_json(CASE/'ready.json',{'status':'four-instance ready','seconds_since_start':time.time()-began})
        report['episodes']=asyncio.run(asyncio.wait_for(traffic_validation(protocol,cfg,budget),timeout=180.))
        report['status']='passed'
    except BaseException as exc:
        report.update(status='incomplete',reason=type(exc).__name__+':'+str(exc))
    finally:
        if started:
            try:report['cleanup']=stop_owned(service)
            except BaseException as exc:report.update(status='incomplete',cleanup_reason=type(exc).__name__+':'+str(exc))
        report['experiment_wall_sec']=time.time()-began
        report['GPU_hour_equivalent_upper_bound']=report['experiment_wall_sec']*4/3600
        report['budget_status_at_stop']=budget.status()
        report['parent_clock_unchanged']=sha(PARENT/'experiment_clock.json')==protocol['budget_amendment']['parent_clock_sha256']
        report['full_original_05_06_accepted']=False;report['PPO_updates']=0
        atomic_json(CASE/'result.json',report)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    print(json.dumps({'status':report['status'],'reason':report.get('reason'),'wall_sec':report['experiment_wall_sec'],
        'result':str(CASE/'result.json')},indent=2))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
