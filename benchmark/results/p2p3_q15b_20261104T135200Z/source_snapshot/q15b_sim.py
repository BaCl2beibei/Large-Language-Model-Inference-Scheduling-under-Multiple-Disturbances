"""Complete-cohort adapter over the existing discrete-event engine.

Uses the real observation/mask/actor/reward interfaces. No future log corrections.
"""
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import math
import numpy as np
from .env import LlumnixMigrationEnv,MigPPOEnvConfig
from .reward import MigPPORewardMetrics
from .real_bridge import PAIRS
from .real_observation import RealObservationBuilder
from .full_completion import summarize_values
from .cohort_reward import CohortReward
from .real_opportunity import balanced_action
from .real_budget import atomic_json


class Q15BCohortSim(LlumnixMigrationEnv):
    def __init__(self,model,trace,seed,output_dir):
        self.model=model;self.trace=trace;self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=False)
        cfg=MigPPOEnvConfig(num_instances=4,total_kv_blocks=model['capacity'],block_size_tokens=16,max_model_len=2112,
            max_num_seqs_per_instance=128,max_num_batched_tokens=2112,request_timeout_sec=None,
            max_decisions=10000,decision_interval_sec=model['decision_interval_sec'],migration_failure_probability=0.,
            prompt_profiles=(128,),output_profiles=(1536,),migration_cooldown_sec=5.,max_state_age_sec=2.,
            service_model_mode='calibrated',service_time_model={'batch_execution':dict(terms=[],centers=[],scales=[],coefficients=[],intercept_ms=1.)},
            migration_alpha_ms=model['migration_intercept_ms'],migration_beta_ms_per_block=model['migration_ms_per_block'],
            migration_pause_model={},trace_events_enabled=True,trace_token_events=False,validate_state_invariants=True,
            state_sample_interval_sec=1.,calibration_status='Q15-B development-only')
        super().__init__(cfg)
        self.sim_tokens={};self.token_counts=[0]*4;self.completed_projection={};self.output_high={}
        self.arrivals_by_id={r['request_id']:r for r in trace}
        self.event_file=(self.output_dir/'events.jsonl').open('w')
        opts=dict(arrival_trace=[dict(request_id=r['request_id'],arrival_time=r['scheduled_arrival_sec'],prompt_tokens=r['input_tokens'],
                       output_tokens=r['output_tokens'],realized_output_tokens=r['output_tokens']) for r in trace],
                  request_routes={r['request_id']:f'instance_{r["initial_instance_index"]}' for r in trace},target_qps=.6)
        super().reset(seed=seed,options=opts)
        self.mapping=[[i.instance_id,'sim-v1'] for i in self.instances]
        self.builder=RealObservationBuilder(self.mapping);self.control_records=[];self.curves=[]

    def _observation(self):return None # legacy observation never reaches the actor

    def _record_lifecycle_event(self,event,request=None,instance=None,**kw):
        if hasattr(self,'event_file'):
            rid=kw.pop('request_id',None)
            row=dict(event=event,time_sec=kw.pop('timestamp',self.now),processed_sec=self.now,
                     request_id=request.request_id if request else rid,instance_id=instance.instance_id if instance else None,**kw)
            self.event_file.write(json.dumps(row)+'\n')

    def _run_engine_step(self,instance,metrics,lo,hi):
        n=max(1,len(instance.running)+min(len(instance.waiting),16))
        ctx=sum(r.request_len for r in instance.running)/max(1,len(instance.running))
        if self.model['selected']=='bucket':
            k=min(6,max(0,int(np.searchsorted(self.model['bucket_edges'],n,side='right')-1)))
            ms=self.model['bucket_medians_ms'][k] or self.model['global_median_ms']
        else:ms=float(np.dot([1.,n,ctx/1024],self.model['ridge_coefficients']))
        self.config.service_time_model['batch_execution']['intercept_ms']=max(1.,ms)
        super()._run_engine_step(instance,metrics,lo,hi)

    def _record_token(self,request,at,metrics,lo,hi):
        owner=self._request_instance(request);idx=self.instances.index(owner)
        self.token_counts[idx]+=1;self.output_high[request.request_id]=request.output_tokens+1
        q=self.sim_tokens.setdefault(request.request_id,deque());q.append((int(at*1e9),1))
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
                used_blocks=i.used_blocks,waiting_blocks=sum(self._blocks_for_tokens(r.request_len) for r in i.waiting),
                running=len(i.running),waiting=len(i.waiting),head_wait_sec=self.now-i.waiting[0].arrival_time if i.waiting else 0.,
                candidate_id=candidate.request_id if candidate else None,candidate_length=candidate.request_len if candidate else None,
                candidate_blocks=candidate.allocated_blocks if candidate else None,candidate_stage='decode' if candidate else None,
                candidate_policy='SR',last_step_ms=i.last_step_ms,output_tokens_total=self.token_counts[idx]))
        return dict(version=str(len(self.control_records)),received_ns=ns,instances=states,mapping=self.mapping,busy=self.pending_migration is not None,
                    last_terminal_ns=round(self.last_migration_time*1e9) if self.last_migration_time is not None else None,invalid_reasons=[])

    def raw(self):
        snap=self.snapshot();now=round(self.now*1e9)
        raw=self.builder.build(snap,now_ns=now,ledger=self,episode_start_ns=0,
            arrival_end_ns=round(max(r['scheduled_arrival_sec'] for r in self.trace)*1e9),deadline_ns=round(max(r['scheduled_arrival_sec'] for r in self.trace)*1e9))
        return raw,snap

    def run(self,policy,T_ref,controller=None,replay_actions=None):
        # Replay input is an exogenous action schedule only, never a state correction.
        if replay_actions is not None:self._migration_replay_trace=sorted(replay_actions,key=lambda r:r['migration_time_sec'])
        while len(self.request_results)<len(self.trace) or self.pending_migration is not None:
            if self.now>3600:raise RuntimeError('sim incomplete after 3600s; never used for PPO')
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
            lo=self.now;self._advance(metrics,lo,lo+self.config.decision_interval_sec);self.decision_count+=1
        self.event_file.close();self._check_state_invariants()
        if len({r['request_id'] for r in self.request_results})!=128 or any(r['generated_tokens']!=1536 or r['status']!='completed' for r in self.request_results):
            raise AssertionError('invalid simulated cohort')
        requests=[dict(request_id=r['request_id'],sent_ns=round(r['arrival_sec']*1e9),client_terminal_ns=round(r['completion_sec']*1e9),
                       actual_output_tokens=r['generated_tokens'],expected_output_tokens=1536,client_e2e_ms=r['e2e_ms']) for r in self.request_results]
        summary=dict(status='passed',cohort_complete=True,route_audit_passed=True,audit={'status':'passed'},planned_request_count=128,
                     episode_id=self.output_dir.name,requests=requests,primary_metrics={'client_e2e_ms':summarize_values([r['client_e2e_ms'] for r in requests])})
        end=max(r['client_terminal_ns'] for r in requests);decisions=[r for r in self.control_records if r['monotonic_ns']<end]
        reward=CohortReward(T_ref).reconstruct(summary,[r['monotonic_ns'] for r in decisions])
        for name,rs in [('decisions',decisions),('requests',requests),('migrations',self.migration_events),('curve',self.curves)]:
            (self.output_dir/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rs))
        atomic_json(self.output_dir/'episode.json',summary);atomic_json(self.output_dir/'reward.json',reward)
        atomic_json(self.output_dir/'replay_results.json',self.migration_replay_results)
        return summary,decisions,self.migration_events
