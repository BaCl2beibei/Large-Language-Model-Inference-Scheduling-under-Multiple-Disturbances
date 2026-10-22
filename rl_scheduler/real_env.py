"""Finite real-time episode state machine. No PPO or synthetic transitions here."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import time

from .real_bridge import PAIRS
from .real_observation import RealObservationBuilder, RealNormalizer, SCHEMA_VERSION


@dataclass(frozen=True)
class RealEnvConfig:
    arrival_sec: float=120.
    drain_sec: float=120.
    decision_interval_sec: float=1.
    cleanup_sec: float=60.
    ttft_slo_ms: float=2000.
    tpot_slo_ms: float=150.

    def __post_init__(self):
        if self.arrival_sec<=0 or self.drain_sec<0 or self.decision_interval_sec<=0 or not 0<self.cleanup_sec<=60:
            raise ValueError('invalid finite episode/cleanup horizons')


def assert_idle(snapshot,baseline=None):
    if snapshot.get('busy') or snapshot.get('invalid_reasons'):
        raise RuntimeError('control slot busy or intervention invalidated episode')
    current={}
    for state in snapshot['instances']:
        if state.get('healthy') is not True or state.get('type')!='neutral':
            raise RuntimeError('instance unhealthy or unsupported')
        for name in ('running','waiting','pending_delivery','preallocated_requests','paused_requests'):
            if state.get(name)!=0:raise RuntimeError('cluster not idle: '+name)
        if state.get('is_migrating') is not False:
            raise RuntimeError('migration is not confirmed idle')
        key=(state['instance_id'],state['epoch'])
        used=state.get('used_blocks')
        if not isinstance(used,int) or used<0:raise RuntimeError('idle KV baseline unknown')
        if baseline is not None and baseline.get(key)!=used:
            raise RuntimeError('idle KV baseline changed')
        current[key]=used
    if len(current)!=4:raise RuntimeError('four idle instances required')
    return current


class RealEnv:
    """io owns HTTP traffic, event collection, and task ownership; bridge owns control.

    io contract: prepare, begin(start,arrival,deadline), poll, stop_arrivals,
    cleanup(deadline_ns), flush, sleep. Real events are collected even while a
    decision RPC runs. finish reconciles before allowing any rollout finalisation.
    """
    def __init__(self,bridge,io,budget,*,run_id,episode_id,ledger_factory,
                 config=RealEnvConfig(),normalizer=None,clock=time.monotonic_ns,
                 decision_sink=lambda record:None):
        self.bridge,self.io,self.budget=bridge,io,budget
        self.run_id,self.episode_id=run_id,episode_id
        self.ledger_factory,self.config,self.clock=ledger_factory,config,clock
        self.normalizer=normalizer or RealNormalizer()
        self.decision_sink=decision_sink
        self.state='new';self.invalid_reasons=[];self.decisions=[];self.receipts={}
        self.baseline=None;self.lease=None;self.previous_decision_ns=None
        self.finalised=False
        self.unresolved_decision = None

    async def reset(self,*,mode='active',controller_version='smoke-v1'):
        if self.state not in ('new','finished'):
            raise RuntimeError('previous episode has not finished cleanup')
        if self.invalid_reasons:raise RuntimeError('invalid environment cannot be reused')
        self.budget.require(self.config.arrival_sec+self.config.drain_sec+self.config.cleanup_sec)
        await asyncio.wait_for(self.io.prepare(),timeout=self.config.cleanup_sec)
        snapshot=await self.bridge.snapshot()
        self.baseline=assert_idle(snapshot,self.baseline)
        self.lease=await self.bridge.acquire_controller(self.run_id,controller_version,self.episode_id,mode)
        self.state='resetting'
        snapshot=await self.bridge.snapshot()
        if snapshot['mapping']!=self.lease['mapping']:raise RuntimeError('mapping changed during reset')
        self.start_ns=self.clock()
        self.arrival_end_ns=self.start_ns+int(self.config.arrival_sec*1e9)
        self.deadline_ns=self.arrival_end_ns+int(self.config.drain_sec*1e9)
        self.ledger=self.ledger_factory(self.start_ns,self.arrival_end_ns,self.deadline_ns)
        self.builder=RealObservationBuilder(snapshot['mapping'],ttft_slo_ms=self.config.ttft_slo_ms,
                                            tpot_slo_ms=self.config.tpot_slo_ms)
        self.normalizer_hash=self.normalizer.hash
        self.state='arrival'
        await self.io.begin(self.start_ns,self.arrival_end_ns,self.deadline_ns,self.ledger)
        self.current_observation=await self._observation()
        return self.current_observation,{'phase':self.state,'baseline_KV_blocks':dict(self.baseline),
                                          'finalised':False}

    async def _observation(self):
        path_started=self.clock()
        await self.io.poll(self.ledger)
        snapshot=await self.bridge.snapshot()
        now=self.clock()
        if snapshot.get('lease_expired') or snapshot.get('invalid_reasons'):
            raise RuntimeError('controller expired or external intervention')
        if self.normalizer.hash!=self.normalizer_hash:raise RuntimeError('normalizer changed inside episode')
        self.snapshot=snapshot
        raw=self.builder.build(snapshot,now_ns=now,ledger=self.ledger,episode_start_ns=self.start_ns,
                               arrival_end_ns=self.arrival_end_ns,deadline_ns=self.deadline_ns)
        raw['decision_path_started_ns']=path_started
        return {'raw':raw,'normalized':self.normalizer.transform(raw)}

    async def step(self,action,*,policy_version='smoke-v1',old_logp=None,old_value=None):
        if self.state not in ('arrival','drain'):raise RuntimeError('episode is not stepping')
        try:
            self.budget.require(self.config.cleanup_sec)
            now=self.clock()
            if now>=self.arrival_end_ns and self.state=='arrival':
                await self.io.stop_arrivals();self.state='drain'
            if now>=self.deadline_ns:
                self.state='deadline'
                return None,0.,True,False,{'phase':self.state,'finalised':False}
            await self.bridge.renew_controller()
            observation=self.current_observation
            now=self.clock();raw=observation['raw']
            if type(action) is not int or not 0<=action<=12:raise ValueError('action must be 0..12')
            if action and self.unresolved_decision is not None:
                previous = self.unresolved_decision
                final = await self.bridge.query_receipt(previous)
                self.receipts[previous] = deepcopy(final)
                if final.get('terminal_status') not in ('succeeded','failed','rejected'):
                    raise RuntimeError('unresolved decision '+previous+'; stop sampling and clean up')
                self.unresolved_decision = None
            decision_id=f'{self.run_id}:{self.episode_id}:{len(self.decisions)}'
            dt=(now-self.previous_decision_ns)/1e9 if self.previous_decision_ns is not None else (now-self.start_ns)/1e9
            self.previous_decision_ns=now
            record={'run_id':self.run_id,'episode_id':self.episode_id,'policy_version':policy_version,
                    'decision_id':decision_id,'monotonic_time':now,'snapshot_version':self.snapshot['version'],
                    'instance_index_map':deepcopy(self.snapshot['mapping']),'obs_received_at':raw['obs_received_at'],
                    'feature_schema':SCHEMA_VERSION,'raw_obs':raw,'normalized_obs':observation['normalized'],
                    'normalizer_version':self.normalizer.version,'normalizer_hash':self.normalizer_hash,
                    'action_mask':list(raw['action_mask']),'sampled_action':action,'old_logp':old_logp,
                    'old_value':old_value,'dt_sec':dt,'src_id':None,'dst_id':None,'candidate_id':None}
            if action==0:
                receipt={'receipt_status':'stay','terminal_status':None,'migration_id':None,'rejection_reason':None}
            else:
                src,dst=PAIRS[action-1]
                src_id,dst_id=self.snapshot['mapping'][src][0],self.snapshot['mapping'][dst][0]
                by_id={s['instance_id']:s for s in self.snapshot['instances']}
                candidate=by_id[src_id].get('candidate_id')
                record.update(src_id=src_id,dst_id=dst_id,candidate_id=candidate)
                if not raw['action_mask'][action]:
                    receipt={'receipt_status':'rejected','terminal_status':'rejected',
                             'rejection_reason':'saved_action_mask_invalid','migration_id':None}
                else:
                    receipt=await self.bridge.submit_migration(decision_id,src_id,dst_id,candidate,self.snapshot['version'])
            record.update(receipt)
            record['cpu_decision_latency_ms']=(self.clock()-raw['decision_path_started_ns'])/1e6
            self.decisions.append(deepcopy(record));self.decision_sink(deepcopy(record))
            self.receipts[decision_id]=deepcopy(receipt)
            if receipt.get('receipt_status') in ('accepted','unknown') and receipt.get('terminal_status') not in ('succeeded','failed','rejected'):
                self.unresolved_decision = decision_id
            # Return the next observation after one real interval. The action above
            # retains exactly the observation/mask from which it was selected.
            await self.io.sleep(max(0.,self.config.decision_interval_sec-(self.clock()-now)/1e9))
            if self.clock()>=self.arrival_end_ns and self.state=='arrival':
                await self.io.stop_arrivals();self.state='drain'
            if self.clock()>=self.deadline_ns:
                self.state='deadline'
                return None,0.,True,False,{'phase':self.state,'receipt':receipt,'finalised':False}
            self.current_observation=await self._observation()
            return self.current_observation,0.,False,False,{'phase':self.state,'receipt':receipt,'finalised':False,'reward_provisional':True}
        except BaseException as exc:
            self.invalid_reasons.append(type(exc).__name__+':'+str(exc));self.state='invalid'
            raise

    async def finish(self):
        if self.state in ('new','finished'):raise RuntimeError('no active episode to finish')
        if self.clock()<self.deadline_ns and self.state!='invalid':
            self.invalid_reasons.append('interrupted_before_finite_deadline');self.state='invalid'
        await self.io.stop_arrivals()
        boundary={'score_deadline_ns':self.deadline_ns,'observed_at_ns':self.clock(),
                  'observed_outstanding_ids':[r['request_id'] for r in self.ledger.observe(self.clock())['requests']
                                              if r['sent_ns'] is not None and r['client_terminal_ns'] is None]}
        # Freeze the scoring timestamp before cleanup; final score waits for evidence.
        cleanup_deadline=self.clock()+int(self.config.cleanup_sec*1e9)
        try:
            await asyncio.wait_for(self.io.cleanup(cleanup_deadline,self.ledger),timeout=self.config.cleanup_sec)
            for decision_id,receipt in list(self.receipts.items()):
                if receipt.get('receipt_status') in ('accepted','unknown'):
                    final=await asyncio.wait_for(self.bridge.query_receipt(decision_id),timeout=max(.001,(cleanup_deadline-self.clock())/1e9))
                    self.receipts[decision_id]=final
                    if final.get('terminal_status') not in ('succeeded','failed','rejected'):
                        raise RuntimeError('migration terminal remains unknown')
            await asyncio.wait_for(self.io.flush(self.ledger),timeout=max(.001,(cleanup_deadline-self.clock())/1e9))
            snapshot=await asyncio.wait_for(self.bridge.snapshot(),timeout=max(.001,(cleanup_deadline-self.clock())/1e9))
            assert_idle(snapshot,self.baseline)
            cleanup_through=min((cp.through_ns for cp in self.ledger.checkpoints.values()),default=0)
            cleanup_audit=self.ledger.audit(through_ns=cleanup_through,require_cleanup=True)
            if cleanup_audit['status']!='passed':
                self.invalid_reasons.extend(cleanup_audit['issues'])
        except BaseException as exc:
            self.invalid_reasons.append('cleanup:'+type(exc).__name__+':'+str(exc))
        finally:
            try:
                released=await asyncio.wait_for(self.bridge.release_controller(),timeout=max(.001,(cleanup_deadline-self.clock())/1e9))
                if released.get('status')!='released':self.invalid_reasons.append('controller_release_unconfirmed')
            except BaseException as exc:
                self.invalid_reasons.append('release:'+type(exc).__name__)
        score=self.ledger.freeze_score()
        if score['status']!='passed':self.invalid_reasons.extend(score['audit']['issues'])
        self.finalised=not self.invalid_reasons
        self.state='finished'
        return {'status':'passed' if self.finalised else 'incomplete','valid_episode':self.finalised,
                'finite_terminal':self.finalised,'terminal_value_zero_allowed':self.finalised,
                'boundary':boundary,'score':score,'decisions':deepcopy(self.decisions),
                'final_receipts':deepcopy(self.receipts),'invalid_reasons':list(self.invalid_reasons)}

    async def close(self):
        if self.state=='resetting' and not hasattr(self,'ledger'):
            self.invalid_reasons.append('reset_interrupted')
            try:await asyncio.wait_for(self.bridge.release_controller(),timeout=5.)
            except BaseException:self.invalid_reasons.append('reset_release_unconfirmed')
            self.state='finished'
            return {'status':'incomplete','valid_episode':False,'invalid_reasons':self.invalid_reasons}
        if self.state not in ('new','finished'):
            self.state='invalid';self.invalid_reasons.append('controller_exit')
            return await self.finish()
        return {'status':'closed','valid_episode':self.finalised}
