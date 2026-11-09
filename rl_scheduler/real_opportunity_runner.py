"""Sequential real P3 episodes over existing audited bridge/IO. No PPO or simulation.

Importing this module starts nothing. Only run_opportunity() sends traffic; the
CLI verifies the frozen protocol/gates/budget and owns service cleanup first.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import threading
import time

from .real_bridge import RealBridge, digest
from .real_budget import atomic_json
from .real_opportunity_budget_override import OpportunityBudgetLedger
from .real_env import RealEnv, RealEnvConfig, assert_idle
from .real_metrics import RequestLedger, _percentile
from .real_runner import LiveEpisodeIO
from .real_opportunity import (balanced_action, episode_schedule, healthy, load_frozen_trace,
                               metric_summary, paired_gate, select_loads)
from .real_reward import RewardWeights, fit_reference_scales, reconstruct_rewards, reward_scale_diagnostics


def prime_causal_history(cursor, paths):
    """Start each episode at the durable pre-traffic tail of service logs.

    A new P3 episode has a fresh client journal, but reuses the same long-lived
    service event files.  Scanning those files from byte zero on every episode
    is both quadratic and can hold the initial controller lease past expiry.
    Call only after the service flush barrier and before starting this episode's
    traffic; all earlier records belong to already reconciled episodes.
    """
    for path in sorted(paths):
        path = Path(path)
        stat = path.stat()
        key = str(path.resolve())
        if stat.st_size:
            with path.open('rb') as stream:
                stream.seek(-1, 2)
                if stream.read(1) != b'\n':
                    raise RuntimeError('causal event log has an incomplete baseline record: '+str(path))
        cursor.offsets[key] = stat.st_size
        cursor.identities[key] = (stat.st_dev, stat.st_ino)


class OpportunityRequestLedger(RequestLedger):
    """Audit only sequence numbers produced after this episode's flush fence.

    P3 reuses long-lived service producers across episodes, while each ledger is
    episode-local.  Its durable initial flush supplies the producer sequence
    watermark; events before that watermark belong to already reconciled work.
    Keep the shared P1/P2 ledger unchanged and retain strict gap checks after
    each P3 watermark.
    """
    def __init__(self, *args, sequence_baselines, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_baselines = {source: int(sequence)
                                   for source, sequence in sequence_baselines.items()}
        if any(sequence < 0 for sequence in self.sequence_baselines.values()):
            raise ValueError('negative producer sequence baseline')
        # The full event history remains authoritative for scoring.  The online
        # observation needs only a tiny per-request projection; rebuilding and
        # sorting every output/chunk event at every decision is quadratic under
        # an intentionally skewed, high-output trace.
        self._online = {}

    def ingest(self, events):
        events = list(events)
        new_events = []
        batch_keys = set()
        for event in events:
            key = (event.source, event.sequence)
            if key not in self.events and key not in batch_keys:
                new_events.append(event)
                batch_keys.add(key)
        super().ingest(events)
        for event in new_events:
            if (event.run_id, event.episode_id) != (self.run_id, self.episode_id):
                continue
            if event.source not in self.sources or not event.request_id:
                continue
            state = self._online.setdefault(event.request_id, {
                'sent': None, 'first': None, 'last': None, 'terminal': None,
                'terminal_data': None, 'token_high': 0,
                'token_increments': deque(),
            })
            stamp = (event.occurred_ns, event.received_ns)
            if event.kind == 'sent':
                if state['sent'] is None or stamp[0] < state['sent'][0]:
                    state['sent'] = stamp
            elif event.kind == 'client_first':
                if state['first'] is None or stamp[0] < state['first'][0]:
                    state['first'] = stamp
            elif event.kind == 'client_output':
                count = event.data.get('cumulative_tokens')
                if type(count) is int and count > state['token_high']:
                    delta = count - state['token_high']
                    state['token_high'] = count
                    state['token_increments'].append((event.occurred_ns, delta, event.received_ns))
                    if state['first'] is None or stamp[0] < state['first'][0]:
                        state['first'] = stamp
                    if state['last'] is None or stamp[0] > state['last'][0]:
                        state['last'] = stamp
            elif event.kind == 'client_terminal':
                if state['terminal'] is None or stamp[0] < state['terminal'][0]:
                    state['terminal'] = stamp
                    state['terminal_data'] = dict(event.data)

    def observe(self, decision_ns):
        rows = []
        for request_id, state in self._online.items():
            sent = state['sent']
            if sent is None or sent[0] > decision_ns or sent[1] > decision_ns:
                continue
            first = state['first']
            terminal = state['terminal']
            terminal_data = state['terminal_data'] or {}
            outcome = terminal_data.get('outcome', 'unknown') if terminal else None
            http_status = terminal_data.get('http_status') if terminal else None
            if outcome == 'cancelled' or http_status == 499:
                outcome = 'unknown'
            elif outcome == 'success' and http_status in (None, 200):
                outcome = 'success'
            elif outcome not in ('timeout', 'service_error'):
                outcome = 'unknown' if terminal else None
            increments = state['token_increments']
            cutoff = decision_ns - 5_000_000_000
            while increments and increments[0][0] <= cutoff:
                increments.popleft()
            visible_increments = [(at, count) for at, count, received in increments
                                  if at <= decision_ns and received <= decision_ns]
            rows.append({
                'request_id': request_id,
                'sent_ns': sent[0],
                'client_first_ns': first[0] if first and first[0] <= decision_ns and first[1] <= decision_ns else None,
                'client_last_ns': state['last'][0] if state['last'] and state['last'][0] <= decision_ns
                    and state['last'][1] <= decision_ns else None,
                'client_terminal_ns': terminal[0] if terminal and terminal[0] <= decision_ns and terminal[1] <= decision_ns else None,
                'outcome': outcome if terminal and terminal[0] <= decision_ns and terminal[1] <= decision_ns else None,
                'token_increments': visible_increments,
            })
        return {'as_of_ns': decision_ns, 'requests': rows, 'issues': [],
                'outstanding': sum(row['client_terminal_ns'] is None for row in rows)}

    def audit(self, *, through_ns=None, require_cleanup=False):
        result = super().audit(through_ns=through_ns, require_cleanup=require_cleanup)
        gaps = {source: [sequence for sequence in missing
                         if sequence > self.sequence_baselines.get(source, 0)]
                for source, missing in result['sequence_gaps'].items()}
        gaps = {source: missing for source, missing in gaps.items() if missing}
        issues = set(result['issues'])
        for source in result['sequence_gaps']:
            issue = f'sequence_gap:{source}'
            if source in gaps:
                issues.add(issue)
            else:
                issues.discard(issue)
        result['sequence_gaps'] = gaps
        result['issues'] = sorted(issues)
        result['status'] = 'incomplete' if issues else 'passed'
        return result


class OpportunityLiveEpisodeIO(LiveEpisodeIO):
    """Keep cumulative causal-log parsing off the request/control event loop.

    A fresh P3 episode starts with a new EventCursor, so its first poll may have
    to decode the service's growing per-process logs. Running that synchronous
    JSONL scan on the asyncio loop delayed scheduled requests and control RPCs.
    The inherited cursor, journal, and ledger are still single-writer; the poll
    is awaited before any other episode code touches them.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._sender_loop=None
        self._sender_thread=None
        self._sender_session=None

    async def _open_sender_session(self):
        import aiohttp
        self._sender_session=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=360.),
            connector=aiohttp.TCPConnector(limit=4096,force_close=True))

    def _start_sender_loop(self):
        if self._sender_loop is not None:return
        self._sender_loop=asyncio.new_event_loop()
        def run_loop():
            asyncio.set_event_loop(self._sender_loop)
            self._sender_loop.run_forever()
        self._sender_thread=threading.Thread(target=run_loop,name='p3-fixed-arrival-sender',daemon=True)
        self._sender_thread.start()
        try:
            asyncio.run_coroutine_threadsafe(self._open_sender_session(),self._sender_loop).result(timeout=10.)
        except BaseException:
            self._sender_loop.call_soon_threadsafe(self._sender_loop.stop)
            self._sender_thread.join(timeout=5.)
            self._sender_loop.close()
            self._sender_loop=None;self._sender_thread=None
            raise

    @staticmethod
    def _wrap_futures(futures):
        return [asyncio.wrap_future(future,loop=asyncio.get_running_loop()) for future in futures]

    async def begin(self,start_ns,arrival_end_ns,deadline_ns,ledger):
        self.start_ns,self.arrival_end_ns,self.deadline_ns=start_ns,arrival_end_ns,deadline_ns
        for row in self.trace:
            self.journal.record('planned',row['request_id'],scheduled_ns=start_ns+int(row['scheduled_arrival_sec']*1e9))
        await asyncio.to_thread(self._start_sender_loop)
        for row in self.trace:
            self.tasks[row['request_id']]=asyncio.run_coroutine_threadsafe(self._issue(row),self._sender_loop)
        await self.poll(ledger)

    async def _issue(self,row):
        delay=self.start_ns+int(row['scheduled_arrival_sec']*1e9)-time.monotonic_ns()
        if delay>0:await asyncio.sleep(delay/1e9)
        if self.stop_new or time.monotonic_ns()>=self.arrival_end_ns:return
        body={**self.sampling,'prompt':row['prompt'],'max_tokens':row['output_tokens']}
        if self.route_shares:
            draw=random.Random(row['route_seed']).random()
            cumulative=0.;index=3
            for i,share in enumerate(self.route_shares):
                cumulative+=share
                if draw<cumulative:index=i;break
            body['initial_instance_id']=self.mapping[index][0]
            self.expected_routes[row['request_id']]=body['initial_instance_id']
        await self.client.issue(self._sender_session,self.bridge.base_url+'/generate_real_metrics',row['request_id'],body)

    async def stop_arrivals(self):
        self.stop_new=True
        to_cancel=[]
        for request_id,future in self.tasks.items():
            if request_id not in self.client.sent and not future.done():
                future.cancel();to_cancel.append(future)
        if to_cancel:
            await asyncio.gather(*self._wrap_futures(to_cancel),return_exceptions=True)

    async def cleanup(self,deadline_ns,ledger):
        await self.stop_arrivals()
        await self.poll(ledger)
        unfinished=[request_id for request_id,future in self.tasks.items()
                    if request_id in self.client.sent and not future.done()]
        rows=ledger.observe(time.monotonic_ns())['requests']
        cleanup_ids=sorted(set(unfinished)|{row['request_id'] for row in rows
            if row['sent_ns'] is not None and row['outcome'] in ('unknown','timeout',None)})
        if cleanup_ids:
            result=await asyncio.wait_for(self.client.cancel(self.session,
                self.bridge.base_url+'/cancel_real_metrics',cleanup_ids),
                timeout=max(.001,(deadline_ns-time.monotonic_ns())/1e9))
            atomic_json(self.directory/'cancellation.json',result)
        pending=[future for future in self.tasks.values() if not future.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*self._wrap_futures(pending),return_exceptions=True),
                timeout=max(.001,(deadline_ns-time.monotonic_ns())/1e9))
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

    def _poll_sync(self, ledger):
        cp=self.journal.checkpoint()
        ledger.poll_events(self.cursor,self.journal.path)
        ledger.acknowledge(cp)
        for path in sorted(self.causal_dir.glob('events*.jsonl')):
            ledger.poll_events(self.cursor,path,adapter=self.adapter)
        if self.cursor.issues:raise RuntimeError('event cursor integrity failure')

    async def prepare(self):
        await super().prepare()
        self.sequence_baselines = {ack['source']: int(ack['last_sequence']) for ack in self.acks}
        self.sequence_baselines[self.journal.source] = 0
        prime_causal_history(self.cursor, self.causal_dir.glob('events*.jsonl'))

    def ledger(self, start, arrival, deadline):
        return OpportunityRequestLedger(self.run_id, self.episode_id, start_ns=start,
            arrival_end_ns=arrival, score_deadline_ns=deadline, expected_sources=self.sources,
            expected_request_ids=[row['request_id'] for row in self.trace],
            sequence_baselines=self.sequence_baselines)

    async def poll(self,ledger):
        await asyncio.to_thread(self._poll_sync,ledger)

    async def preserve_fixed_workload(self,ledger,arrival_end_ns,deadline_ns):
        """Keep the frozen arrival stream alive after a control-side failure.

        Invalid control episodes remain invalid, but do not cancel offered work
        early.  The client keeps issuing only inside the registered arrival
        window; completion is still scored at the original finite deadline.
        """
        poll_errors=[]
        while time.monotonic_ns()<deadline_ns:
            now=time.monotonic_ns()
            if now>=arrival_end_ns:
                await self.stop_arrivals()
            await asyncio.sleep(min(.5,max(0.,(deadline_ns-now)/1e9)))
            try:
                await self.poll(ledger)
            except Exception as exc:
                poll_errors.append(type(exc).__name__+':'+str(exc))
        await self.stop_arrivals()
        try:
            await self.poll(ledger)
        except Exception as exc:
            poll_errors.append(type(exc).__name__+':'+str(exc))
        return poll_errors

    def _stop_sender_loop(self):
        if self._sender_loop is None:return
        try:
            if self._sender_session is not None:
                asyncio.run_coroutine_threadsafe(self._sender_session.close(),self._sender_loop).result(timeout=10.)
        finally:
            self._sender_loop.call_soon_threadsafe(self._sender_loop.stop)
            self._sender_thread.join(timeout=10.)
            self._sender_loop.close()
            self._sender_loop=None;self._sender_thread=None;self._sender_session=None

    async def close(self):
        pending=[future for future in self.tasks.values() if not future.done()]
        for future in pending:future.cancel()
        if pending:
            try:
                await asyncio.wait_for(asyncio.gather(*self._wrap_futures(pending),return_exceptions=True),
                    timeout=10.)
            except asyncio.TimeoutError:pass
        await asyncio.to_thread(self._stop_sender_loop)
        self.journal.close()


async def _nonblocking_controller_heartbeat(bridge,stop,errors):
    """Keep P3's controller lease alive while Manager snapshots await actors."""
    import aiohttp
    try:
        while True:
            try:
                await asyncio.wait_for(stop.wait(),timeout=5.)
                return
            except asyncio.TimeoutError:
                # Lease renewal is idempotent. Retry brief loopback disconnects
                # (including a stale keep-alive socket) before invalidating a
                # finite episode; a lost response does not imply a lost lease.
                for attempt in range(3):
                    try:
                        result=await bridge.call('renew_controller_nonblocking',lease_id=bridge.lease['lease_id'])
                        bridge.lease=result
                        break
                    except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, OSError):
                        if attempt==2:
                            raise
                        await asyncio.sleep(.05*(attempt+1))
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        errors.append(type(exc).__name__+':'+str(exc))


async def _stop_heartbeat(task,stop):
    if stop is not None:stop.set()
    if task is not None:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)


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
    # Route integrity covers every request actually accepted by the service,
    # including a request whose acceptance lands just after the arrival cutoff.
    # Such late acceptances remain outside the scoring cohort and are reported
    # separately by request_submission_audit.
    rows,_=ledger._rows();accepted={r['request_id'] for r in rows
        if r['accepted_ns'] is not None}
    correct=set(admitted)==accepted and all(len(values)==1 and values[0] in valid_ids for values in admitted.values())
    observed={key:values[0] for key,values in admitted.items() if len(values)==1}
    expected_routes={key:value for key,value in io.expected_routes.items() if key in accepted}
    if io.route_shares:
        correct=(correct and observed==expected_routes
                 and declared=={key:[value] for key,value in expected_routes.items()})
    return {'status':'passed' if correct else 'incomplete','rule':'fixed skew' if io.route_shares else 'native dispatcher',
            'observed_initial_routes':observed,'expected_fixed_routes':expected_routes,
            'accepted_request_ids':sorted(accepted),
            'missing_admission_ids':sorted(accepted-set(admitted)),
            'unexpected_admission_ids':sorted(set(admitted)-accepted),
            'ambiguous_admission_ids':sorted(k for k,v in admitted.items() if len(v)!=1)}


def request_submission_audit(rows,expected_ids,*,arrival_end_ns=None):
    expected=sorted(expected_ids);by_id={row['request_id']:row for row in rows}
    if arrival_end_ns is None:arrival_end_ns=max((row.get('accepted_ns') or 0 for row in rows),default=0)
    missing_records=sorted(set(expected)-set(by_id))
    not_sent=sorted(request_id for request_id in expected
                        if request_id in by_id and by_id[request_id].get('sent_ns') is None)
    late_acceptance=sorted(request_id for request_id in expected if request_id in by_id
        and by_id[request_id].get('accepted_ns') is not None
        and by_id[request_id]['accepted_ns']>arrival_end_ns)
    accepted_ids={request_id for request_id in expected if request_id in by_id
        and by_id[request_id].get('accepted_ns') is not None
        and by_id[request_id]['accepted_ns']<=arrival_end_ns}
    sent_without_acceptance=sorted(request_id for request_id in expected if request_id in by_id
        and by_id[request_id].get('sent_ns') is not None
        and by_id[request_id].get('accepted_ns') is None)
    delays=[(by_id[request_id]['accepted_ns']-by_id[request_id]['sent_ns'])/1e6
            for request_id in accepted_ids if request_id in by_id
            and by_id[request_id].get('sent_ns') is not None
            and by_id[request_id].get('accepted_ns') is not None]
    accepted=len(accepted_ids)
    return {'status':'passed' if not missing_records and accepted>0 else 'incomplete',
        'offered':len(expected),'recorded':len(expected)-len(missing_records),
        'sent':len(expected)-len(missing_records)-len(not_sent),
        'accepted':accepted,'missing_record_ids':missing_records,'not_sent_ids':not_sent,
        'sent_without_acceptance_ids':sent_without_acceptance,'late_acceptance_ids':late_acceptance,
        'acceptance_delay_p95_ms':_percentile(delays,.95),'acceptance_delay_max_ms':max(delays) if delays else None,
        'acceptance_clock':'service accepted event monotonic time',
        'completion_denominator':'accepted within arrival window'}


async def run_episode(spec,runtime,case,*,bridge,session,run_dir,episode_id,budget,session_mapping=None):
    run_dir=Path(run_dir);run_id=json.loads((run_dir/'budget.json').read_text())['run_id']
    h=spec['probe'] if case['phase']=='probe' else spec['horizons']
    trace=load_frozen_trace(run_dir,spec,seed=case['seed'],qps=case['qps'],phase=case['phase'])
    shares=spec['scenes'][case['scene']].get('route_shares')
    io=OpportunityLiveEpisodeIO(bridge,session,run_id=run_id,episode_id=episode_id,run_dir=run_dir,trace=trace,
                     sampling=runtime['sampling'],arrival_sec=h['arrival_sec'],route_shares=shares)
    samples=[];decisions=[];payload=None;submission=None
    def decision_sink(record):
        import os
        decisions.append(record)
        with (io.directory/'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    env=RealEnv(bridge,io,budget,run_id=run_id,episode_id=episode_id,ledger_factory=io.ledger,
        config=RealEnvConfig(arrival_sec=h['arrival_sec'],drain_sec=h['drain_sec'],cleanup_sec=spec['horizons']['cleanup_sec'],
            decision_interval_sec=spec['horizons']['decision_interval_sec'],ttft_slo_ms=spec['slos']['ttft_ms'],tpot_slo_ms=spec['slos']['tpot_ms']),
        decision_sink=decision_sink,external_lease_heartbeat=True)
    heartbeat_stop=asyncio.Event();heartbeat_errors=[];heartbeat_task=None
    def start_heartbeat():
        nonlocal heartbeat_task
        if heartbeat_task is not None:raise RuntimeError('duplicate opportunity controller lease acquisition')
        heartbeat_task=asyncio.create_task(_nonblocking_controller_heartbeat(bridge,heartbeat_stop,heartbeat_errors))
    async def stop_heartbeat():
        await _stop_heartbeat(heartbeat_task,heartbeat_stop)
    bridge.on_controller_acquired=start_heartbeat
    bridge.before_controller_release=stop_heartbeat
    try:
        observation,_=await env.reset(mode='stay' if case['policy']=='Stay' else 'active',controller_version='opportunity-fixed-v1')
        if heartbeat_errors:raise RuntimeError('controller heartbeat failed: '+heartbeat_errors[0])
        if session_mapping is not None and io.mapping!=session_mapping:raise RuntimeError('instance epoch mapping changed across paired runs')
        while env.state in ('arrival','drain'):
            samples.append({'observed_ns':time.monotonic_ns(),'measured':env.snapshot['instances']})
            action=0 if case['policy']=='Stay' else balanced_action(observation,env.snapshot,threshold=spec['balanced']['threshold'])
            observation,_,done,_,_=await env.step(action,policy_version=case['policy']+'-fixed-v1')
            if heartbeat_errors:raise RuntimeError('controller heartbeat failed: '+heartbeat_errors[0])
            if done:break
        result=await env.finish()
        through=min((cp.through_ns for cp in env.ledger.checkpoints.values()),default=0)
        evidence=list(e for e in env.ledger.events.values() if env.start_ns<=e.occurred_ns<=through)
        migrations=reconcile_migrations(decisions,env.receipts,evidence)
        routes=route_audit(io,env.ledger,evidence,io.mapping)
        exported=env.ledger.export(io.directory,cleanup_through_ns=through)
        rows,_=env.ledger._rows(occurred_by_ns=through)
        submission=request_submission_audit(rows,[r['request_id'] for r in trace],arrival_end_ns=env.arrival_end_ns)
        if result['status']!='passed' or set(exported.values())!={'passed'}:
            raise RuntimeError('episode score or cleanup not reconciled: score='+str(result.get('status'))
                +' score_audit='+str(result.get('score',{}).get('audit'))
                +' invalid='+str(result.get('invalid_reasons'))+' cleanup='+str(exported)
                +' submission='+json.dumps(submission,sort_keys=True))
        if submission['status']!='passed':
            raise RuntimeError('frozen trace submission invalid: '+json.dumps(submission,sort_keys=True))
        if routes['status']!='passed':raise RuntimeError('actual initial routing not reconciled: '+json.dumps(routes,sort_keys=True))
        if case['policy']=='Stay' and migrations:raise RuntimeError('Stay executed migration')
        metrics=metric_summary(env.ledger)
        late=[r['send_lateness_ms'] for r in rows if r['send_lateness_ms'] is not None]
        last_third=[s for s in samples if env.start_ns+int(h['arrival_sec']*2/3*1e9)<=s['observed_ns']<env.arrival_end_ns]
        edges=[env.start_ns]+[r['monotonic_time'] for r in decisions[1:] if r['monotonic_time']<env.deadline_ns]+[env.deadline_ns]
        row={**case,'episode_id':episode_id,'status':'passed','mapping':io.mapping,'metrics':metrics,
            'trace_hash':digest(trace),'external_case_hash':digest({'trace':trace,'route_shares':shares,
                'sampling':runtime['sampling'],'arrival_sec':h['arrival_sec'],'drain_sec':h['drain_sec']}),
            'send_lateness_p95_ms':_percentile(late,.95),'queue_last_third_samples':len(last_third),
            'queue_last_third_positive':sum(any(state.get('waiting',0)>0 for state in s['measured']) for s in last_third),
            'migrations':migrations,'route_audit':routes,'request_submission_audit':submission,
            'reward_scale_status':'awaiting_healthy_Stay_opportunity_data',
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
        fixed_workload_errors=[]
        # A controller/decision failure invalidates the episode, but it must not
        # cancel the preregistered workload or shorten its scoring horizon.
        # Requests remain scheduled only inside the arrival window; after that
        # window there are no catch-up sends.
        if (not isinstance(exc,asyncio.CancelledError) and hasattr(env,'ledger')
                and hasattr(env,'arrival_end_ns') and hasattr(env,'deadline_ns')
                and time.monotonic_ns()<env.deadline_ns):
            try:
                fixed_workload_errors=await io.preserve_fixed_workload(
                    env.ledger,env.arrival_end_ns,env.deadline_ns)
            except BaseException as preserve_exc:
                fixed_workload_errors.append(type(preserve_exc).__name__+':'+str(preserve_exc))
        await stop_heartbeat()
        result=None
        if env.state not in ('new','finished'):
            try:result=await env.close()
            except BaseException as close_exc:
                fixed_workload_errors.append('cleanup:'+type(close_exc).__name__+':'+str(close_exc))
        invalid={**case,'episode_id':episode_id,'status':'incomplete','reason':type(exc).__name__+':'+str(exc)}
        if result is not None:
            invalid['episode_cleanup_status']=result.get('status')
            invalid['episode_invalid_reasons']=result.get('invalid_reasons',[])
        if hasattr(env,'ledger'):
            score=env.ledger.freeze_score()
            rows,_=env.ledger._rows(occurred_by_ns=env.deadline_ns)
            submission=request_submission_audit(rows,[r['request_id'] for r in trace],arrival_end_ns=env.arrival_end_ns)
            by_id={row['request_id']:row for row in rows}
            accepted_ids=sorted(request_id for request_id in (r['request_id'] for r in trace)
                if request_id in by_id and by_id[request_id].get('accepted_ns') is not None
                and by_id[request_id]['accepted_ns']<=env.arrival_end_ns)
            completed_ids=sorted(request_id for request_id in accepted_ids
                if request_id in by_id and by_id[request_id].get('outcome')=='success'
                and by_id[request_id].get('client_terminal_ns') is not None
                and by_id[request_id]['client_terminal_ns']<=env.deadline_ns)
            expected_ids=sorted(r['request_id'] for r in trace)
            unfinished_ids=sorted(set(accepted_ids)-set(completed_ids))
            sent_ids=sorted(request_id for request_id in expected_ids
                if request_id in by_id and by_id[request_id].get('sent_ns') is not None
                and by_id[request_id]['sent_ns']<=env.deadline_ns)
            not_sent_ids=sorted(set(expected_ids)-set(sent_ids))
            invalid['workload_scoring']={
                'valid_for_analysis':False,
                'validity':'descriptive_only_control_or_audit_failure',
                'arrival_window_sec':h['arrival_sec'],
                'post_arrival_requests':'none',
                'score_deadline_ns':env.deadline_ns,
                'offered':len(expected_ids),
                'sent_by_deadline':len(sent_ids),
                'accepted_in_arrival_window':len(accepted_ids),
                'completed_by_deadline':len(completed_ids),
                'completion_rate_accepted':len(completed_ids)/len(accepted_ids) if accepted_ids else None,
                'completion_denominator':'accepted within arrival window',
                'unfinished_ids':unfinished_ids,
                'not_sent_ids':not_sent_ids,
                'excluded_unaccepted_ids':sorted(set(expected_ids)-set(accepted_ids)),
                'fixed_window_metrics':score.get('fixed'),
                'score_audit':score.get('audit'),
                'request_submission_audit':submission,
                'preserve_fixed_workload_errors':fixed_workload_errors,
            }
        elif submission is not None:invalid['request_submission_audit']=submission
        atomic_json(io.directory/'interrupted.json',invalid)
        raise
    finally:
        await stop_heartbeat()
        bridge.on_controller_acquired=None
        bridge.before_controller_release=None
        await io.close()


async def run_opportunity(spec,runtime,*,run_dir,invocation_id,budget_override_lease,
                          artifact_dir=None,episode_prefix=None,prior_probes=None):
    import aiohttp
    run_dir=Path(run_dir);folder=Path(artifact_dir) if artifact_dir is not None else run_dir/'opportunity_runs'/invocation_id
    prefix=episode_prefix or invocation_id
    budget=OpportunityBudgetLedger(run_dir,budget_override_lease);probes=list(prior_probes or []);episodes=[];saved=[]
    probe_keys=[(row.get('scene'),row.get('qps'),row.get('seed')) for row in probes]
    if (len(probe_keys)!=len(set(probe_keys))
            or any(row.get('phase')!='probe' or row.get('policy')!='Stay' or row.get('status')!='passed'
                   for row in probes)):
        raise ValueError('manual continuation requires unique audited Stay probe rows')
    if any(row.get('request_submission_audit',{}).get('status')!='passed' for row in probes):
        raise ValueError('manual continuation cannot reuse probes without at least one service-accepted request and complete trace records')
    mapping=probes[0]['mapping'] if probes else None
    if mapping is not None and any(row.get('mapping')!=mapping for row in probes):
        raise ValueError('prior probes do not share the fixed instance epoch mapping')
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.),
                                         connector=aiohttp.TCPConnector(limit=128,force_close=True)) as control, \
                   aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=360.),connector=aiohttp.TCPConnector(limit=4096)) as traffic:
            bridge=RealBridge(runtime['service']['base_url'],control)
            for qps,seed in zip(spec['probe']['qps'],spec['probe']['seeds']):
                for scene in ('A','B'):
                    key=(scene,qps,seed)
                    if key in probe_keys:
                        continue
                    case={'phase':'probe','scene':scene,'seed':seed,'qps':qps,'policy':'Stay'}
                    item=await run_episode(spec,runtime,case,bridge=bridge,session=traffic,run_dir=run_dir,
                        episode_id=f'{prefix}_probe_{scene}_{qps}',budget=budget,session_mapping=mapping)
                    row=item[0];mapping=row['mapping'];probes.append(row)
                    probe_keys.append(key)
                    atomic_json(folder/'probes.json',probes)
            atomic_json(folder/'probes.json',probes)
            selection=select_loads(probes,spec);atomic_json(folder/'selection.json',selection)
            if selection['status']!='passed':
                atomic_json(folder/'opportunity_gate.json',selection);return selection
            for index,case in enumerate(episode_schedule(spec,selection['selected'])):
                item=await run_episode(spec,runtime,case,bridge=bridge,session=traffic,run_dir=run_dir,
                    episode_id=f'{prefix}_{index:02}_{case["scene"]}_{case["seed"]}_{case["policy"]}',
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
