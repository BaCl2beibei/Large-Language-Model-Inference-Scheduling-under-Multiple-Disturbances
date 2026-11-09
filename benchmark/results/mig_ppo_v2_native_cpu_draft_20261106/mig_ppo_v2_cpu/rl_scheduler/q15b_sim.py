"""Complete-cohort adapter over the existing discrete-event engine.

Uses the real observation/mask/actor/reward interfaces. No future log corrections.
"""
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import math
import os
import time
import numpy as np
from .env import LlumnixMigrationEnv,MigPPOEnvConfig
from .reward import MigPPORewardMetrics
from .real_bridge import PAIRS
from .real_observation import RealObservationBuilder
from .cohort_statistics import summarize_values
from .cohort_reward import CohortReward
from .real_opportunity import balanced_action
from .portable_io import atomic_json
from .ppo_runtime import PPO_DECISION_INTERVAL_SEC, PPO_MIGRATION_COOLDOWN_SEC


class _DiscardList(list):
    """Bound memory when diagnostic engine rows are disabled for training."""
    def append(self, value):
        pass


class Q15BCohortSim(LlumnixMigrationEnv):
    def __init__(self,model,trace,seed,output_dir,*,minimal_logging=False,
                 finalize_deadline=False,wall_time_limit_sec=None):
        self.minimal_logging=minimal_logging;self.finalize_deadline=finalize_deadline
        self.wall_time_limit_sec=wall_time_limit_sec
        self.model=model;self.trace=trace;self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=False)
        self.n_request=len(trace)
        self.physical_v2=model.get('schema')=='mig-ppo-v2-shared-physical-v2'
        self.routing_mode=(trace[0].get('routing_mode','fixed') if trace else 'fixed')
        cfg=MigPPOEnvConfig(num_instances=4,total_kv_blocks=model['capacity'],block_size_tokens=16,max_model_len=2112,
            max_num_seqs_per_instance=128,max_num_batched_tokens=2112,request_timeout_sec=None,
            max_decisions=10000,decision_interval_sec=model.get('decision_interval_sec',1.),migration_failure_probability=0.,
            prompt_profiles=(128,),output_profiles=(1536,),migration_cooldown_sec=model.get('migration_cooldown_sec',5.),max_state_age_sec=2.,
            service_model_mode='calibrated',service_time_model={'batch_execution':dict(terms=[],centers=[],scales=[],coefficients=[],intercept_ms=1.)},
            migration_alpha_ms=model['migration_intercept_ms'],migration_beta_ms_per_block=model['migration_ms_per_block'],
            migration_pause_model=model.get('migration_pause_components',{}),trace_events_enabled=not minimal_logging,trace_token_events=False,validate_state_invariants=True,
            state_sample_interval_sec=1.,calibration_status='Q15-B development-only')
        super().__init__(cfg)
        self.sim_tokens={};self.token_history={};self.token_counts=[0]*4;self.completed_projection={};self.output_high={}
        self.client_buffer={}
        self.engine_predictions={}
        self.arrivals_by_id={r['request_id']:r for r in trace}
        self.event_file=open(os.devnull,'w') if minimal_logging else (self.output_dir/'events.jsonl').open('w',encoding='utf-8')
        instance_ids=model.get('instance_ids',[f'instance_{i}' for i in range(4)])
        route_rows={r['request_id']:instance_ids[r['initial_instance_index']] for r in trace
                    if r.get('initial_instance_index') is not None}
        opts=dict(arrival_trace=[dict(request_id=r['request_id'],arrival_time=r['scheduled_arrival_sec'],prompt_tokens=r['input_tokens'],
                       output_tokens=r['output_tokens'],realized_output_tokens=r['output_tokens']) for r in trace],
                  request_routes=route_rows,target_qps=float(trace[0].get('target_qps',.6)) if trace else .6,
                  native_dispatch=self.routing_mode=='native',instance_ids=instance_ids,
                  native_dispatch_update_interval_sec=float(model.get('native_dispatch_update_interval_sec',.05)))
        super().reset(seed=seed,options=opts)
        if minimal_logging:
            self.service_iterations=_DiscardList();self.state_samples=_DiscardList()
        for idx,instance in enumerate(self.instances):
            instance.total_blocks=int(model.get('capacity_by_logical_instance',[model['capacity']]*4)[idx])
        if self.physical_v2:
            for key,co in model.get('logical_migration_directions',{}).items():
                src,dst=key.split('>')
                self.cost_estimator.pair_models[(instance_ids[int(src)],instance_ids[int(dst)])]=(
                    co['intercept_ms'],co['ms_per_block'],0.,co['n'])
        self.mapping=[[i.instance_id,'sim-v1'] for i in self.instances]
        self.builder=RealObservationBuilder(self.mapping);self.control_records=[];self.curves=[]

    def _observation(self):return None # legacy observation never reaches the actor

    def _record_lifecycle_event(self,event,request=None,instance=None,**kw):
        if not self.minimal_logging and hasattr(self,'event_file'):
            rid=kw.pop('request_id',None)
            row=dict(event=event,time_sec=kw.pop('timestamp',self.now),processed_sec=self.now,
                     request_id=request.request_id if request else rid,instance_id=instance.instance_id if instance else None,**kw)
            self.event_file.write(json.dumps(row)+'\n')

    def _run_engine_step(self,instance,metrics,lo,hi):
        if self.physical_v2:
            previous=instance.last_step_ms
            super()._run_engine_step(instance,metrics,lo,hi)
            instance.last_step_ms=self.engine_predictions.get(instance.instance_id,previous) if instance.step_in_flight else previous
            return
        n=max(1,len(instance.running)+min(len(instance.waiting),16))
        ctx=sum(r.request_len for r in instance.running)/max(1,len(instance.running))
        if self.model['selected']=='bucket':
            k=min(6,max(0,int(np.searchsorted(self.model['bucket_edges'],n,side='right')-1)))
            ms=self.model['bucket_medians_ms'][k] or self.model['global_median_ms']
        else:ms=float(np.dot([1.,n,ctx/1024],self.model['ridge_coefficients']))
        self.config.service_time_model['batch_execution']['intercept_ms']=max(1.,ms)
        super()._run_engine_step(instance,metrics,lo,hi)

    def _physical_features(self,instance,features):
        idx=self.instances.index(instance)
        return dict(n=features['batch_size'],context=features['total_context_tokens'],
                     active=sum(bool(i.running) for i in self.instances),
                     global_n=sum(len(i.running) for i in self.instances),
                     busy=float(self.pending_migration is not None),
                     hardware=self.model['hardware_by_logical_instance'][idx])
    def _predict_batch_execution_ms(self,instance,features):
        if not self.physical_v2:
            return super()._predict_batch_execution_ms(instance,features)
        from .service_model import predict_physical_cadence
        engine=predict_physical_cadence(self.model.get('engine',self.model['service']),self._physical_features(instance,features))
        prediction=max(engine,self.model['prefill_ms']) if features['num_prefill_requests'] else engine
        self.engine_predictions[instance.instance_id]=prediction
        return prediction

    def _predict_output_cadence_ms(self,instance,features):
        if not self.physical_v2:
            return super()._predict_output_cadence_ms(instance,features)
        from .service_model import predict_physical_cadence
        return predict_physical_cadence(self.model['service'],self._physical_features(instance,features))

    def _record_token(self,request,at,metrics,lo,hi):
        owner=self._request_instance(request);idx=self.instances.index(owner)
        self.token_counts[idx]+=1;self.output_high[request.request_id]=request.output_tokens+1
        count=self.client_buffer.get(request.request_id,0)+1
        generated=request.output_tokens+1
        frame=int(self.model.get('client_frame_tokens',1)) if self.physical_v2 else 1
        # Match first-token streaming followed by multi-step output flushes.
        # GPU progress and KV growth continue between client receipts.
        if generated==1 or generated%frame==0 or generated>=request.target_output_tokens:
            q=self.sim_tokens.setdefault(request.request_id,deque());q.append((round(at*1e9),count))
            self.token_history.setdefault(request.request_id,[]).append((round(at*1e9),count))
            count=0
        self.client_buffer[request.request_id]=count
        super()._record_token(request,at,metrics,lo,hi)

    def _complete_request(self,request,at,metrics,lo,hi):
        self.completed_projection[request.request_id]=dict(first=request.first_token_time,last=at,sent=request.arrival_time,end=at)
        super()._complete_request(request,at,metrics,lo,hi)

    def observe(self,now_ns):
        result=[]
        for rid in self.output_high.keys() | self.requests.keys():
            active=self.requests.get(rid);done=self.completed_projection.get(rid)
            if active:sent=active.arrival_time;first=active.first_token_time;last=active.last_token_time;end=None
            elif done:sent=done['sent'];first=done['first'];last=done['last'];end=done['end']
            else:continue
            if self.physical_v2:
                history=self.token_history.get(rid,[])
                first=history[0][0]/1e9 if history else None
                last=history[-1][0]/1e9 if history else None
            q=self.sim_tokens.setdefault(rid,deque())
            while q and q[0][0]<now_ns-5_000_000_000:q.popleft()
            ns=lambda x:None if x is None else round(x*1e9)
            result.append(dict(request_id=rid,sent_ns=ns(sent),client_first_ns=ns(first),client_last_ns=ns(last),
                               client_terminal_ns=ns(end),token_increments=list(q)))
        return {'requests':result}

    def snapshot(self):
        ns=round(self.now*1e9);states=[]
        for idx,i in enumerate(self.instances):
            candidate=self._sr_candidate(i)
            states.append(dict(instance_id=i.instance_id,epoch='sim-v1',measured_ns=ns,healthy=True,type='neutral',
                is_migrating=i.is_migrating,has_slot=not i.is_migrating,total_blocks=i.total_blocks,free_blocks=i.free_blocks,
                used_blocks=i.used_blocks,reserved_blocks_not_in_used=i.reserved_blocks if self.physical_v2 else 0,waiting_blocks=sum(self._blocks_for_tokens(r.request_len) for r in i.waiting),
                running=len(i.running),waiting=len(i.waiting),head_wait_sec=self.now-i.waiting[0].arrival_time if i.waiting else 0.,
                candidate_id=candidate.request_id if candidate else None,candidate_length=candidate.request_len if candidate else None,
                candidate_blocks=candidate.allocated_blocks if candidate else None,candidate_stage='decode' if candidate else None,
                candidate_policy='SR',last_step_ms=i.last_step_ms,output_tokens_total=self.token_counts[idx]))
        return dict(version=str(len(self.control_records)),received_ns=ns,instances=states,mapping=self.mapping,busy=self.pending_migration is not None,
                    last_terminal_ns=round(self.last_migration_time*1e9) if self.last_migration_time is not None else None,invalid_reasons=[])

    def raw(self):
        snap=self.snapshot();now=round(self.now*1e9)
        raw=self.builder.build(snap,now_ns=now,ledger=self,episode_start_ns=0,
            arrival_end_ns=round(max(r['scheduled_arrival_sec'] for r in self.trace)*1e9),deadline_ns=round(self.episode_horizon_sec*1e9))
        return raw,snap

    def _finalize_at_deadline(self):
        """Finite-horizon local simulation: explicit failures, never fake tokens."""
        metrics=MigPPORewardMetrics(window_sec=0.,instance_count=4)
        if self.pending_migration is not None:
            pending=self.pending_migration
            pending.replay_status='aborted';pending.end_time=self.now
            pending.actual_ms=max(0.,(self.now-pending.start_time)*1000.)
            pending.pause_start_time=min(pending.pause_start_time,self.now)
            pending.pause_stage_ms=min(pending.pause_stage_ms,pending.actual_ms)
            pending.recovery_ms=0.
            self._finish_migration(metrics,self.now,self.now)
        for request in list(self.requests.values()):
            rid=request.request_id
            buffered=self.client_buffer.get(rid,0)
            if buffered:
                at=round(self.now*1e9)
                self.token_history.setdefault(rid,[]).append((at,buffered))
                self.sim_tokens.setdefault(rid,deque()).append((at,buffered))
                self.client_buffer[rid]=0
            for instance in self.instances:
                if request in instance.running:
                    instance.running.remove(request)
                    instance.used_blocks-=request.allocated_blocks
                if request in instance.waiting:instance.waiting.remove(request)
            request.allocated_blocks=0;request.migrating=False;request.stage='failed'
            self.request_results.append(self._request_result(request,'timeout',self.now))
            self.completed_projection[rid]=dict(first=request.first_token_time,last=request.last_token_time,
                                               sent=request.arrival_time,end=self.now)
            self.output_high[rid]=request.output_tokens
            self.requests.pop(rid)
        for instance in self.instances:
            instance.pending_prefill=[];instance.pending_decode=[]
            instance.step_in_flight=False;instance.next_step_time=math.inf
        self._check_state_invariants()

    def run(self,policy,T_ref,controller=None,replay_actions=None,reward_definition=None,episode_horizon_sec=None):
        if episode_horizon_sec is None:
            episode_horizon_sec=float(self.model.get('episode_horizon_sec',3600.))
        self.episode_horizon_sec=episode_horizon_sec
        if self.finalize_deadline and (reward_definition is None or not self.trace or
                not 0<=min(r['scheduled_arrival_sec'] for r in self.trace)<=max(r['scheduled_arrival_sec'] for r in self.trace)<episode_horizon_sec):
            raise ValueError('deadline simulation requires V2 reward and all arrivals before the horizon')
        started=time.monotonic()
        self.config.decision_interval_sec=float(self.model.get('decision_interval_sec',
            PPO_DECISION_INTERVAL_SEC if controller is not None else self.config.decision_interval_sec))
        self.config.migration_cooldown_sec=float(self.model.get('migration_cooldown_sec',
            PPO_MIGRATION_COOLDOWN_SEC if controller is not None else self.config.migration_cooldown_sec))
        self.builder=RealObservationBuilder(self.mapping, cooldown_sec=self.config.migration_cooldown_sec,
            **({'ttft_slo_ms': reward_definition.ttft_ref_sec*1000,
                'tpot_slo_ms': reward_definition.token_gap_ref_sec*1000} if reward_definition is not None else {}))
        # Replay input is an exogenous action schedule only, never a state correction.
        if replay_actions is not None:self._migration_replay_trace=sorted(replay_actions,key=lambda r:r['migration_time_sec'])
        while len(self.request_results)<len(self.trace) or self.pending_migration is not None:
            if self.wall_time_limit_sec is not None and time.monotonic()-started>self.wall_time_limit_sec:
                raise TimeoutError('wall-clock watchdog interrupted sampling; no rollout terminal is manufactured')
            if self.finalize_deadline and self.now>=episode_horizon_sec-1e-9:
                self._finalize_at_deadline();break
            if self.now>episode_horizon_sec:raise RuntimeError('sim incomplete at episode horizon; never used for PPO')
            raw,snap=self.raw();record=None
            action=balanced_action({'raw':raw},snap,threshold=self.model['capacity']/8) if policy=='Balanced' else 0
            if controller is not None:record=controller(raw,0);action=record['action']
            if not raw['action_mask'][action]:raise AssertionError('masked simulated action')
            metrics=MigPPORewardMetrics(window_sec=self.config.decision_interval_sec,instance_count=4)
            reason=None
            if action:
                ob=SimpleNamespace(action_mask=SimpleNamespace(pairs=[(self.mapping[a][0],self.mapping[b][0]) for a,b in PAIRS]),pair_mask=raw['action_mask'][1:])
                reason=self._start_migration(action,ob,metrics)
            row=dict(monotonic_ns=raw['decision_ns'],observation_decision_ns=raw['decision_ns'],action_mask=raw['action_mask'],sampled_action=action,
                     executed_action=action,rejection=reason,snapshot=snap,ppo=record)
            self.control_records.append(row)
            self.curves.append(dict(time_sec=self.now,outstanding=len(self.requests),kv=[i.used_blocks for i in self.instances]))
            lo=self.now
            hi=min(lo+self.config.decision_interval_sec,episode_horizon_sec) if self.finalize_deadline else lo+self.config.decision_interval_sec
            self._advance(metrics,lo,hi);self.decision_count+=1
        self.event_file.close();self._check_state_invariants()
        if len({r['request_id'] for r in self.request_results})!=self.n_request or any(
                r['status'] not in ('completed','timeout') or (r['status']=='timeout' and not self.finalize_deadline)
                or (r['status']=='completed' and r['generated_tokens']!=self.arrivals_by_id[r['request_id']]['output_tokens'])
                for r in self.request_results):
            raise AssertionError('invalid simulated cohort')
        requests=[dict(request_id=r['request_id'],sent_ns=round(r['arrival_sec']*1e9),client_terminal_ns=round(r['completion_sec']*1e9),
                       client_first_ns=self.token_history[r['request_id']][0][0] if self.token_history.get(r['request_id']) else None,
                       token_increments=self.token_history.get(r['request_id'],[]),outcome='success' if r['status']=='completed' else 'timeout',
                       actual_output_tokens=r['generated_tokens'],expected_output_tokens=self.arrivals_by_id[r['request_id']]['output_tokens'],
                       client_e2e_ms=r['e2e_ms']) for r in self.request_results]
        summary=dict(status='passed',cohort_complete=True,route_audit_passed=True,audit={'status':'passed'},planned_request_count=self.n_request,
                     decision_interval_sec=self.config.decision_interval_sec,migration_cooldown_sec=self.config.migration_cooldown_sec,
                     routing_mode=self.routing_mode,episode_horizon_sec=self.episode_horizon_sec,
                     finite_horizon_deadline=self.finalize_deadline,
                     episode_id=self.output_dir.name,requests=requests,
                     primary_metrics={'client_e2e_ms':summarize_values([r['client_e2e_ms'] for r in requests])})
        end=max(r['client_terminal_ns'] for r in requests);decisions=[r for r in self.control_records if r['monotonic_ns']<end]
        if reward_definition is None:
            reward=CohortReward(T_ref).reconstruct(summary,[r['monotonic_ns'] for r in decisions])
        else:
            reward=reward_definition.reconstruct(summary,[r['monotonic_ns'] for r in decisions],decisions)
        for name,rs in [('decisions',decisions),('requests',requests),('migrations',self.migration_events),('curve',self.curves)]:
            (self.output_dir/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rs),encoding='utf-8')
        atomic_json(self.output_dir/'episode.json',summary);atomic_json(self.output_dir/'reward.json',reward)
        atomic_json(self.output_dir/'replay_results.json',self.migration_replay_results)
        return summary,decisions,self.migration_events
