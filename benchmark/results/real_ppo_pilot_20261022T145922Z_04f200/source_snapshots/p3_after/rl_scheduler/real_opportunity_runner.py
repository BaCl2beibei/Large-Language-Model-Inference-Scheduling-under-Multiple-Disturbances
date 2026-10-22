"""Sequential real P3 episodes over existing audited bridge/IO. No PPO or simulation.

Importing this module starts nothing. Only run_opportunity() sends traffic; the
CLI verifies the frozen protocol/gates/budget and owns service cleanup first.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from .real_bridge import RealBridge, digest
from .real_budget import BudgetLedger, atomic_json
from .real_env import RealEnv, RealEnvConfig
from .real_metrics import _percentile
from .real_runner import LiveEpisodeIO
from .real_opportunity import (balanced_action, episode_schedule, healthy, load_frozen_trace,
                               metric_summary, paired_gate, select_loads)
from .real_reward import RewardWeights, fit_reference_scales, reconstruct_rewards, reward_scale_diagnostics


def reconcile_migrations(decisions,receipts,events):
    starts={}
    for event in events:
        if event.data.get('event')=='migration_attempt_started':
            mid=event.data.get('migration_id')
            if not mid or mid in starts:raise ValueError('uncorrelated or duplicate original coordinator start')
            starts[mid]=event.occurred_ns
    records=[];claimed=set()
    for decision in decisions:
        if decision['sampled_action']==0:continue
        receipt=receipts[decision['decision_id']]
        mid=receipt.get('migration_id');terminal=receipt.get('terminal_status')
        if receipt.get('receipt_status') in ('shadow','stay'):
            raise ValueError('opportunity Balanced must execute under active control')
        if terminal not in ('succeeded','failed','rejected'):
            raise ValueError('unknown non-Stay terminal')
        if mid in starts:claimed.add(mid)
        if terminal=='succeeded' and mid not in starts:raise ValueError('success without measured coordinator attempt')
        terminal_ns=receipt.get('terminal_ns')
        if terminal_ns is None and receipt.get('receipt_status')=='rejected' and mid is None:
            terminal_ns=decision['monotonic_time']  # explicit local saved-mask rejection
        if type(terminal_ns) is not int:raise ValueError('missing actual migration terminal timestamp')
        records.append({'decision_id':decision['decision_id'],'migration_id':mid,
            'attempt_ns':starts.get(mid),'terminal_ns':terminal_ns,'terminal_status':terminal,
            'actual_request_id':receipt.get('actual_request_id'),'rejection_reason':receipt.get('rejection_reason')})
    if set(starts)!=claimed:raise ValueError('unclaimed migration, including unexpected native migration')
    return records


def route_audit(io,ledger,events,mapping):
    admitted={};declared={};valid_ids={row[0] for row in mapping}
    for event in events:
        if event.data.get('event')=='request_engine_admitted' and event.request_id:
            admitted.setdefault(event.request_id,[]).append(event.data['instance_id'])
        if event.data.get('event')=='real_initial_route' and event.request_id:
            declared.setdefault(event.request_id,[]).append(event.data['instance_id'])
    rows,_=ledger._rows();sent={r['request_id'] for r in rows if r['sent_ns'] is not None}
    correct=set(admitted)==sent and all(len(values)==1 and values[0] in valid_ids for values in admitted.values())
    observed={key:values[0] for key,values in admitted.items() if len(values)==1}
    if io.route_shares:
        correct=(correct and observed==io.expected_routes
                 and declared=={key:[value] for key,value in io.expected_routes.items()})
    return {'status':'passed' if correct else 'incomplete','rule':'fixed skew' if io.route_shares else 'native dispatcher',
            'observed_initial_routes':observed,'expected_fixed_routes':dict(io.expected_routes),
            'missing_admission_ids':sorted(sent-set(admitted)),'ambiguous_admission_ids':sorted(k for k,v in admitted.items() if len(v)!=1)}


async def run_episode(spec,runtime,case,*,bridge,session,run_dir,episode_id,budget,session_mapping=None):
    run_dir=Path(run_dir);run_id=json.loads((run_dir/'budget.json').read_text())['run_id']
    h=spec['probe'] if case['phase']=='probe' else spec['horizons']
    trace=load_frozen_trace(run_dir,spec,seed=case['seed'],qps=case['qps'],phase=case['phase'])
    shares=spec['scenes'][case['scene']].get('route_shares')
    io=LiveEpisodeIO(bridge,session,run_id=run_id,episode_id=episode_id,run_dir=run_dir,trace=trace,
                     sampling=runtime['sampling'],arrival_sec=h['arrival_sec'],route_shares=shares)
    samples=[];decisions=[];payload=None
    def decision_sink(record):
        import os
        decisions.append(record)
        with (io.directory/'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    env=RealEnv(bridge,io,budget,run_id=run_id,episode_id=episode_id,ledger_factory=io.ledger,
        config=RealEnvConfig(arrival_sec=h['arrival_sec'],drain_sec=h['drain_sec'],cleanup_sec=spec['horizons']['cleanup_sec'],
            decision_interval_sec=spec['horizons']['decision_interval_sec'],ttft_slo_ms=spec['slos']['ttft_ms'],tpot_slo_ms=spec['slos']['tpot_ms']),
        decision_sink=decision_sink)
    try:
        observation,_=await env.reset(mode='stay' if case['policy']=='Stay' else 'active',controller_version='opportunity-fixed-v1')
        if session_mapping is not None and io.mapping!=session_mapping:raise RuntimeError('instance epoch mapping changed across paired runs')
        while env.state in ('arrival','drain'):
            samples.append({'observed_ns':time.monotonic_ns(),'measured':env.snapshot['instances']})
            action=0 if case['policy']=='Stay' else balanced_action(observation,env.snapshot,threshold=spec['balanced']['threshold'])
            observation,_,done,_,_=await env.step(action,policy_version=case['policy']+'-fixed-v1')
            if done:break
        result=await env.finish()
        through=min((cp.through_ns for cp in env.ledger.checkpoints.values()),default=0)
        evidence=list(e for e in env.ledger.events.values() if env.start_ns<=e.occurred_ns<=through)
        migrations=reconcile_migrations(decisions,env.receipts,evidence)
        routes=route_audit(io,env.ledger,evidence,io.mapping)
        if routes['status']!='passed':raise RuntimeError('actual initial routing not reconciled')
        exported=env.ledger.export(io.directory,cleanup_through_ns=through)
        if result['status']!='passed' or set(exported.values())!={'passed'}:raise RuntimeError('episode score or cleanup not reconciled')
        if case['policy']=='Stay' and migrations:raise RuntimeError('Stay executed migration')
        metrics=metric_summary(env.ledger)
        rows,_=env.ledger._rows()
        late=[r['send_lateness_ms'] for r in rows if r['send_lateness_ms'] is not None]
        last_third=[s for s in samples if env.start_ns+int(h['arrival_sec']*2/3*1e9)<=s['observed_ns']<env.arrival_end_ns]
        edges=[env.start_ns]+[r['monotonic_time'] for r in decisions[1:] if r['monotonic_time']<env.deadline_ns]+[env.deadline_ns]
        row={**case,'episode_id':episode_id,'status':'passed','mapping':io.mapping,'metrics':metrics,
            'trace_hash':digest(trace),'external_case_hash':digest({'trace':trace,'route_shares':shares,
                'sampling':runtime['sampling'],'arrival_sec':h['arrival_sec'],'drain_sec':h['drain_sec']}),
            'send_lateness_p95_ms':_percentile(late,.95),'queue_last_third_samples':len(last_third),
            'queue_last_third_positive':sum(any(state.get('waiting',0)>0 for state in s['measured']) for s in last_third),
            'migrations':migrations,'route_audit':routes,'reward_scale_status':'awaiting_healthy_Stay_opportunity_data',
            'live_smoke_prior_status':'skipped_by_user_unverified'}
        atomic_json(io.directory/'episode.json',result)
        atomic_json(io.directory/'opportunity_episode.json',row)
        atomic_json(io.directory/'queue_samples.json',samples)
        atomic_json(io.directory/'reward_inputs.json',{'edges_ns':edges,'migrations':migrations,'cleanup_through_ns':through})
        (io.directory/'migrations.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in migrations))
        payload=(row,env.ledger,edges,migrations,through)
        await bridge.call('release_episode',run_id=run_id,episode_id=episode_id)
        return payload
    except BaseException as exc:
        if env.state not in ('new','finished'):
            try:await env.close()
            except BaseException:pass
        invalid={**case,'episode_id':episode_id,'status':'incomplete','reason':type(exc).__name__+':'+str(exc)}
        atomic_json(io.directory/'interrupted.json',invalid)
        raise
    finally:
        await io.close()


async def run_opportunity(spec,runtime,*,run_dir,invocation_id):
    import aiohttp
    run_dir=Path(run_dir);folder=run_dir/'opportunity_runs'/invocation_id
    budget=BudgetLedger(run_dir);probes=[];episodes=[];saved=[];mapping=None
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.)) as control, \
                   aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=360.),connector=aiohttp.TCPConnector(limit=4096)) as traffic:
            bridge=RealBridge(runtime['service']['base_url'],control)
            for qps,seed in zip(spec['probe']['qps'],spec['probe']['seeds']):
                for scene in ('A','B'):
                    case={'phase':'probe','scene':scene,'seed':seed,'qps':qps,'policy':'Stay'}
                    item=await run_episode(spec,runtime,case,bridge=bridge,session=traffic,run_dir=run_dir,
                        episode_id=f'{invocation_id}_probe_{scene}_{qps}',budget=budget,session_mapping=mapping)
                    row=item[0];mapping=row['mapping'];probes.append(row)
                    atomic_json(folder/'probes.json',probes)
            selection=select_loads(probes,spec);atomic_json(folder/'selection.json',selection)
            if selection['status']!='passed':
                atomic_json(folder/'opportunity_gate.json',selection);return selection
            for index,case in enumerate(episode_schedule(spec,selection['selected'])):
                item=await run_episode(spec,runtime,case,bridge=bridge,session=traffic,run_dir=run_dir,
                    episode_id=f'{invocation_id}_{index:02}_{case["scene"]}_{case["seed"]}_{case["policy"]}',
                    budget=budget,session_mapping=mapping)
                saved.append(item);episodes.append(item[0]);atomic_json(folder/'episodes.json',episodes)
            gate=paired_gate(episodes,spec)
            # Always report the full paired result even when reference fitting is impossible.
            atomic_json(folder/'opportunity_gate.json',gate)
            try:
                scales,provenance=fit_reference_scales(episodes,healthy=lambda row:healthy(row,spec))
                atomic_json(folder/'reward_scales.json',{'status':'measured','scales':asdict(scales),'provenance':provenance})
                for row,ledger,edges,migrations,through in saved:
                    rewards=reconstruct_rewards(ledger,edges,migrations,scales=scales,cleanup_through_ns=through,valid_episode=True,
                                                weights=RewardWeights(**spec['reward']['weights']))
                    row['reward_windows']=rewards
                    path=run_dir/'runs'/row['episode_id']/'reward_windows.jsonl'
                    path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rewards))
                atomic_json(folder/'reward_scale_diagnostics.json',reward_scale_diagnostics(episodes))
            except ValueError as exc:
                gate.update(status='incomplete',reward_scale_reason=str(exc))
                atomic_json(folder/'reward_scales.json',{'status':'unmeasured','C_ref':None,'N_ref':None,'reason':str(exc)})
            # Materialise final C arrivals by preregistered rule, before any policy/final result.
            from .real_opportunity import make_trace
            burst={str(seed):make_trace(spec,seed=seed,qps=selection['selected']['A'],duration=120,bursty=True)
                   for seed in spec['final_burst']['seeds']}
            atomic_json(folder/'final_burst_traces.json',{'rule':spec['final_burst'],'base_qps':selection['selected']['A'],
                                                       'traces':burst,'trace_hashes':{k:digest(v) for k,v in burst.items()}})
            atomic_json(folder/'paired_results.json',gate)
            atomic_json(folder/'episodes.json',episodes)
            atomic_json(folder/'opportunity_gate.json',gate)
            return gate
    except BaseException as exc:
        gate={'status':'incomplete','reason':type(exc).__name__+':'+str(exc),'probes_completed':len(probes),
              'episodes_completed':len(episodes),'failed_attempts_retained':True,'automatic_retry':False}
        atomic_json(folder/'opportunity_gate.json',gate)
        raise
