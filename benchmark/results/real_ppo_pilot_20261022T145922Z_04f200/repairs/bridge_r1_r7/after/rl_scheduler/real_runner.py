"""Live telemetry/bridge smoke orchestration; imported without starting services.

Only run_live() performs real traffic; callers must verify gates and start the
owned supervisor service. No opportunity experiment or training exists here.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from .real_bridge import RealBridge
from .real_budget import BudgetLedger, atomic_json
from .real_client import StreamingRequestCollector
from .real_env import RealEnv, RealEnvConfig, assert_idle
from .real_metrics import CausalEventAdapter, Checkpoint, EventCursor, EventJournal, RequestLedger, _percentile
from .real_bridge import digest


def flatten_acks(value):
    """Validate every role barrier before coalescing process-shared producers.

    A worker's queue barrier cannot be replaced by the later local fsync, even
    when both roles share a writer. Cross-round regression remains a ledger error.
    """
    producers = {}
    def visit(node):
        if isinstance(node, dict):
            if any(key in node for key in ('status', 'source', 'durable')):
                if (node.get('status') != 'passed' or node.get('durable') is not True
                        or node.get('pending_delivery') != 0 or not node.get('source')
                        or type(node.get('last_sequence')) is not int or node['last_sequence'] < 0
                        or type(node.get('through_ns')) is not int or node['through_ns'] <= 0):
                    raise RuntimeError('unconfirmed role flush: '+str(node))
                old = producers.get(node['source'])
                if old:
                    a = (old['last_sequence'], old['through_ns'])
                    b = (node['last_sequence'], node['through_ns'])
                    if not (all(x <= y for x,y in zip(a,b)) or all(x >= y for x,y in zip(a,b))):
                        raise RuntimeError('inconsistent same-producer watermarks')
                    if b < a:return
                producers[node['source']] = node
            else:
                for nested in node.values():visit(nested)
        elif isinstance(node,list):
            for nested in node:visit(nested)
    visit(value)
    yield from producers.values()


def role_inventory(raw, worker_count=1):
    """Required roles come from the fixed four-instance TP topology, not successes."""
    roles = {'api': raw['api'], 'manager': raw['cluster']['manager']}
    instances = raw['cluster']['instances']
    if not isinstance(instances,list) or len(instances)!=4:
        raise RuntimeError('incomplete cluster producer inventory')
    for i,instance in enumerate(instances):
        roles[f'instance:{i}:local'] = instance['local']
        workers = instance['workers']
        if not isinstance(workers,list) or len(workers)!=worker_count:
            raise RuntimeError('incomplete worker role inventory')
        for j,ack in enumerate(workers):roles[f'instance:{i}:worker:{j}']=ack
    for role,ack in roles.items():
        if not isinstance(ack,dict) or 'source' not in ack:
            raise RuntimeError('missing role acknowledgement: '+role+': '+str(ack))
        list(flatten_acks(ack))
    return {role:ack['source'] for role,ack in roles.items()}


def checkpoint(raw):
    return Checkpoint(source=raw['source'],last_sequence=raw['last_sequence'],through_ns=raw['through_ns'],
                      received_ns=time.monotonic_ns(),pending_delivery=raw.get('pending_delivery'),durable=raw['durable'])


def server_request_id(run_id,episode_id,request_id):
    return 'real-'+hashlib.sha256(json.dumps((run_id,episode_id,request_id)).encode()).hexdigest()


class LiveEpisodeIO:
    def __init__(self,bridge,session,*,run_id,episode_id,run_dir,trace,sampling,arrival_sec,route_shares=None):
        self.bridge,self.session=bridge,session
        self.run_id,self.episode_id=run_id,episode_id
        self.run_dir=Path(run_dir);self.directory=self.run_dir/'runs'/episode_id
        self.directory.mkdir(parents=True,exist_ok=False)
        self.trace,self.sampling,self.arrival_sec=trace,sampling,arrival_sec
        self.route_shares=route_shares
        self.journal=EventJournal(self.directory/'client_events.jsonl',run_id,episode_id)
        self.client=StreamingRequestCollector(self.journal)
        self.cursor=EventCursor();self.tasks={};self.stop_new=False
        self.expected_routes={}
        self.adapter=CausalEventAdapter(run_id,episode_id,
                    {server_request_id(run_id,episode_id,row['request_id']):row['request_id'] for row in trace})
        self.checkpoint_history=[]

    async def prepare(self):
        snapshot=await self.bridge.snapshot()
        assert_idle(snapshot)
        self.mapping=snapshot['mapping']
        raw=await self.bridge.call('flush')
        atomic_json(self.directory/'initial_flush.json',raw)
        self.roles=role_inventory(raw)
        self.acks=list(flatten_acks(raw))
        if any(x.get('status')!='passed' or not x.get('durable') for x in self.acks):
            raise RuntimeError('initial producer flush unconfirmed')
        self.sources=sorted({x['source'] for x in self.acks}|{self.journal.source})
        self.causal_dir=Path(json.loads((self.run_dir/'service_session.json').read_text())['causal_dir'])
        atomic_json(self.directory/'initial_flush.json',raw)
        atomic_json(self.directory/'trace.json',self.trace)

    def ledger(self,start,arrival,deadline):
        return RequestLedger(self.run_id,self.episode_id,start_ns=start,arrival_end_ns=arrival,
                             score_deadline_ns=deadline,expected_sources=self.sources,
                             expected_request_ids=[r['request_id'] for r in self.trace])

    async def begin(self,start_ns,arrival_end_ns,deadline_ns,ledger):
        self.start_ns,self.arrival_end_ns,self.deadline_ns=start_ns,arrival_end_ns,deadline_ns
        for row in self.trace:
            self.journal.record('planned',row['request_id'],scheduled_ns=start_ns+int(row['scheduled_arrival_sec']*1e9))
        for row in self.trace:
            self.tasks[row['request_id']]=asyncio.create_task(self._issue(row))
        await self.poll(ledger)

    async def _issue(self,row):
        delay=self.start_ns+int(row['scheduled_arrival_sec']*1e9)-time.monotonic_ns()
        if delay>0:await asyncio.sleep(delay/1e9)
        if self.stop_new or time.monotonic_ns()>=self.arrival_end_ns:return
        body={**self.sampling,'prompt':row['prompt'],'max_tokens':row['output_tokens']}
        if self.route_shares:
            # Fixed trace-index routing, shared across policies; no future policy input.
            draw=random.Random(row['route_seed']).random()
            cumulative=0.;index=3
            for i,share in enumerate(self.route_shares):
                cumulative+=share
                if draw<cumulative:index=i;break
            body['initial_instance_id']=self.mapping[index][0]
            self.expected_routes[row['request_id']]=body['initial_instance_id']
        await self.client.issue(self.session,self.bridge.base_url+'/generate_real_metrics',row['request_id'],body)

    async def poll(self,ledger):
        cp=self.journal.checkpoint()
        ledger.poll_events(self.cursor,self.journal.path)
        ledger.acknowledge(cp)
        for path in sorted(self.causal_dir.glob('events*.jsonl')):
            ledger.poll_events(self.cursor,path,adapter=self.adapter)
        if self.cursor.issues:raise RuntimeError('event cursor integrity failure')

    async def stop_arrivals(self):
        self.stop_new=True
        tasks=[]
        for rid,task in self.tasks.items():
            if rid not in self.client.sent and not task.done():task.cancel();tasks.append(task)
        if tasks:await asyncio.gather(*tasks,return_exceptions=True)

    async def cleanup(self,deadline_ns,ledger):
        await self.stop_arrivals()
        await self.poll(ledger)
        unfinished=[rid for rid,task in self.tasks.items() if rid in self.client.sent and not task.done()]
        # Also abort timed-out/disconnected clients whose backend result is unknown.
        rows=ledger.observe(time.monotonic_ns())['requests']
        cleanup_ids=sorted(set(unfinished)|{r['request_id'] for r in rows if r['sent_ns'] is not None and r['outcome'] in ('unknown','timeout',None)})
        if cleanup_ids:
            result=await asyncio.wait_for(self.client.cancel(self.session,self.bridge.base_url+'/cancel_real_metrics',cleanup_ids),
                                          timeout=max(.001,(deadline_ns-time.monotonic_ns())/1e9))
            atomic_json(self.directory/'cancellation.json',result)
        pending=[task for task in self.tasks.values() if not task.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending,return_exceptions=True),timeout=max(.001,(deadline_ns-time.monotonic_ns())/1e9))
        while True:
            await self.poll(ledger)
            snapshot=await self.bridge.snapshot()
            if snapshot.get('invalid_reasons'):raise RuntimeError('external intervention in episode')
            try:
                assert_idle(snapshot)
                break
            except RuntimeError:
                if time.monotonic_ns()>=deadline_ns:raise
                await asyncio.sleep(min(.1,max(0.,(deadline_ns-time.monotonic_ns())/1e9)))

    async def flush(self,ledger):
        raw=await self.bridge.call('flush')
        self.checkpoint_history.append(raw)
        atomic_json(self.directory/'flush_history.json',self.checkpoint_history)
        if role_inventory(raw)!=self.roles:
            raise RuntimeError('role producer identity changed inside episode')
        await self.poll(ledger)
        client_cp=self.journal.checkpoint(pending_delivery=0 if all(task.done() for task in self.tasks.values()) else 1)
        ledger.poll_events(self.cursor,self.journal.path);ledger.acknowledge(client_cp)
        acks=list(flatten_acks(raw))
        if {x['source'] for x in acks}|{client_cp.source}!=set(self.sources):
            raise RuntimeError('producer set changed inside episode')
        if any(x.get('status')!='passed' for x in acks):raise RuntimeError('producer flush incomplete')
        for ack in acks:ledger.acknowledge(checkpoint(ack))
        atomic_json(self.directory/'flush_history.json',self.checkpoint_history)
        atomic_json(self.directory/'event_cursor.json',{'offsets':self.cursor.offsets,'identities':self.cursor.identities,'issues':self.cursor.issues})

    async def sleep(self,seconds):await asyncio.sleep(seconds)

    async def close(self):
        for task in self.tasks.values():
            if not task.done():task.cancel()
        if self.tasks:await asyncio.gather(*self.tasks.values(),return_exceptions=True)
        self.journal.close()


def trace_recipe(cfg):
    return {'smoke':cfg['smoke'],'sampling':cfg['sampling'],'model_path':cfg['service']['model_path'],
            'max_model_len':cfg['service']['max_model_len']}


def verify_trace_manifest(cfg):
    manifest_path=Path(cfg['smoke_trace_manifest'])
    manifest=json.loads(manifest_path.read_text())
    if manifest['recipe_hash']!=digest(trace_recipe(cfg)):raise RuntimeError('frozen smoke recipe changed')
    for item in manifest['traces'].values():
        path=manifest_path.parent/item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('frozen smoke trace changed')
    return manifest


def make_trace(cfg,case,seed):
    if cfg.get('smoke_trace_manifest'):
        manifest=verify_trace_manifest(cfg)
        item=manifest['traces'][str(seed)]
        return json.loads((Path(cfg['smoke_trace_manifest']).parent/item['path']).read_text())
    from disturbance_exp.workload import generate_trace,load_local_tokenizer
    tokenizer=load_local_tokenizer(cfg['service']['model_path'])
    if tokenizer is None:raise RuntimeError('existing local tokenizer unavailable')
    trace_cfg={'experiment':{'seed':seed,'duration_sec':case['arrival_sec']},
               'workload':{'arrival':'poisson','base_qps':case['qps'],'input_tokens':case['input_tokens'],
                           'output_tokens':case['output_tokens'],'max_model_len':cfg['service']['max_model_len']},
               'limits':{'max_requests':case['max_requests']}}
    return [{**asdict(request),'route_seed':seed*1000000+index} for index,request in enumerate(generate_trace(trace_cfg,tokenizer))]


def decision_action(observation,mode):
    if mode=='stay' or observation is None:return 0
    raw=observation['raw'];mask=raw['action_mask'];features=raw['pair_features']
    legal=[i for i in range(1,13) if mask[i]]
    if not legal:return 0
    # Smoke probe selects a legal high-to-low measured running-pressure pair.
    return max(legal,key=lambda i:(features[i-1][0],features[i-1][1],-i))


async def run_episode(cfg,case,*,bridge,session,run_dir,episode_id,mode,seed,monitor=True):
    run_dir=Path(run_dir);run_id=json.loads((run_dir/'budget.json').read_text())['run_id']
    trace=make_trace(cfg,case,seed)
    io=LiveEpisodeIO(bridge,session,run_id=run_id,episode_id=episode_id,run_dir=run_dir,trace=trace,
                     sampling=cfg['sampling'],arrival_sec=case['arrival_sec'],route_shares=case.get('route_shares'))
    records=[]
    def save_decision(record):
        # The submitted action and uncertain receipt survive an interrupted run.
        import os
        records.append(record)
        with (io.directory/'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    env=RealEnv(bridge,io,BudgetLedger(run_dir),run_id=run_id,episode_id=episode_id,ledger_factory=io.ledger,
                config=RealEnvConfig(arrival_sec=case['arrival_sec'],drain_sec=case['drain_sec']),decision_sink=save_decision)
    result=None
    try:
        observation,_=await env.reset(mode=mode)
        while env.state in ('arrival','drain'):
            if monitor:
                action=decision_action(observation,mode)
                observation,_,done,_,_=await env.step(action)
                if done:break
            else:
                # Overhead OFF retains only lifecycle/lease health and the same client metrics.
                await bridge.renew_controller()
                await io.sleep(min(10.,max(0.,(env.deadline_ns-time.monotonic_ns())/1e9)))
                await io.poll(env.ledger)
                if time.monotonic_ns()>=env.arrival_end_ns:await io.stop_arrivals();env.state='drain'
                if time.monotonic_ns()>=env.deadline_ns:env.state='deadline'
        result=await env.finish()
        raw_events=[event for event in env.ledger.events.values() if env.start_ns<=event.occurred_ns<=env.deadline_ns]
        result['actual_migration_attempts']=sum(e.data.get('event')=='migration_attempt_started' for e in raw_events)
        observed_routes={e.request_id:e.data.get('instance_id') for e in raw_events if e.data.get('event')=='real_initial_route'}
        if io.route_shares and observed_routes!=io.expected_routes:
            result['status']='incomplete';result['valid_episode']=False
            result['invalid_reasons'].append('explicit_initial_route_not_verified')
        result['route_audit']={'expected':io.expected_routes,'observed':observed_routes}
        through=min((cp.through_ns for cp in env.ledger.checkpoints.values()),default=0)
        env.ledger.export(io.directory,cleanup_through_ns=through)
        atomic_json(io.directory/'episode.json',result)
        (io.directory/'decisions.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in records))
        (io.directory/'migrations.jsonl').write_text(''.join(json.dumps({'decision_id':key,**value},allow_nan=False)+'\n'
                                                                 for key,value in env.receipts.items() if value.get('migration_id')))
        return result
    finally:
        if result is None:
            try:
                invalid=await env.close();atomic_json(io.directory/'interrupted.json',invalid)
            except BaseException as exc:atomic_json(io.directory/'interrupted.json',{'status':'incomplete','reason':type(exc).__name__})
        await io.close()


async def run_live(cfg,*,run_dir,stage,mode,invocation_id):
    import aiohttp
    run_dir=Path(run_dir)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.),connector=aiohttp.TCPConnector(limit=4096)) as control_session, \
               aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=360.),connector=aiohttp.TCPConnector(limit=4096)) as traffic_session:
        bridge=RealBridge(cfg['service']['base_url'],control_session)
        if stage=='telemetry':cases=[('telemetry',cfg['smoke']['telemetry'],'stay',True)]
        elif mode=='overhead':cases=[(f'overhead_{i}',cfg['smoke']['overhead'],'stay' if not enabled else 'shadow',enabled)
                                    for i,enabled in enumerate((False,True,True,False))]
        else:cases=[(mode,cfg['smoke'][mode],mode,True)]
        results=[]
        for label,case,control_mode,monitor in cases:
            seed=case['seed']
            result=await run_episode(cfg,case,bridge=bridge,session=traffic_session,run_dir=run_dir,
                                     episode_id=invocation_id+'_'+label,mode=control_mode,seed=seed,monitor=monitor)
            results.append({'label':label,'episode_id':invocation_id+'_'+label,'result':result})
            if result['status']!='passed':break
        status=all(row['result']['status']=='passed' for row in results) and len(results)==len(cases)
        checks={'episodes_reconciled':bool(status)}
        if stage=='telemetry':
            checks['low_load_normal_completion']=all(row['result']['score']['fixed']['completed']>0
                and row['result']['score']['fixed']['completed']==row['result']['score']['fixed']['offered']
                for row in results)
        if stage=='bridge' and mode in ('shadow','active'):
            latencies=sorted(d['cpu_decision_latency_ms'] for row in results for d in row['result']['decisions'])
            p95=_percentile(latencies,.95)
            checks['CPU_decision_path_P95_below_100ms']=p95 is not None and p95<100.
        else:p95=None
        if stage=='bridge' and mode=='shadow':
            checks['zero_actual_migration']=all(not receipt.get('migration_id') for row in results
                for receipt in row['result']['final_receipts'].values())
            checks['zero_coordinator_migration_attempts']=all(row['result']['actual_migration_attempts']==0 for row in results)
        if stage=='bridge' and mode=='active':
            receipts=[receipt for row in results for receipt in row['result']['final_receipts'].values() if receipt.get('migration_id')]
            checks['at_least_10_terminals']=sum(r.get('terminal_status') in ('succeeded','failed','rejected') for r in receipts)>=10
            checks['at_least_5_successes']=sum(r.get('terminal_status')=='succeeded' for r in receipts)>=5
        if stage=='bridge' and mode=='overhead' and len(results)==4:
            fixed=[row['result']['score']['fixed'] for row in results]
            pairs=[(fixed[0],fixed[1]),(fixed[3],fixed[2])]
            deltas=[]
            for off,on in pairs:
                tps=off['client_token_throughput_tps'];ttft=off['ttft_p95_ms']
                deltas.append({'throughput_degradation':1-on['client_token_throughput_tps']/tps if tps>0 else None,
                               'TTFT_P95_degradation':on['ttft_p95_ms']/ttft-1 if ttft and on['ttft_p95_ms'] is not None else None})
            checks['paired_overhead_within_thresholds']=all(d['throughput_degradation'] is not None and d['TTFT_P95_degradation'] is not None
                and d['throughput_degradation']<=.05 and d['TTFT_P95_degradation']<=.10 for d in deltas)
        else:deltas=[]
        report={'stage':stage,'mode':mode,'status':'passed' if all(checks.values()) else 'incomplete',
                'checks':checks,'results':results,'cpu_decision_path_p95_ms':p95,
                'overhead_pairs':deltas,'independent_research_evaluation':False}
        atomic_json(run_dir/'smoke_runs'/invocation_id/'result.json',report)
        return report
