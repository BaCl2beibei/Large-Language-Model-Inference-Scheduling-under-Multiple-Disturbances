"""Versioned, measured-only real-cluster observation and frozen normalization."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from types import MappingProxyType

from .real_bridge import PAIRS, digest, eligible_pair

SCHEMA_VERSION='real-observation-v1'
INSTANCE_FEATURES=('running','waiting','head_wait_sec','used_fraction','free_fraction',
                   'candidate_length','candidate_blocks','last_step_ms','state_age_sec',
                   'tokens_per_sec_5s','waiting_slope_5s','used_fraction_slope_5s')
GLOBAL_FEATURES=('arrival_phase','drain_phase','remaining_arrival_sec','remaining_total_sec',
                 'outstanding','overdue_first','output_stalled','client_tokens_per_sec_5s',
                 'busy','cooldown_remaining_sec','healthy_instances')
PAIR_FEATURES=('running_difference','used_fraction_difference','destination_available_blocks')
SCALES={'running':64.,'waiting':64.,'head_wait_sec':2.,'used_fraction':1.,'free_fraction':1.,
        'candidate_length':2048.,'candidate_blocks':128.,'last_step_ms':100.,'state_age_sec':2.,
        'tokens_per_sec_5s':128.,'waiting_slope_5s':16.,'used_fraction_slope_5s':1.,
        'arrival_phase':1.,'drain_phase':1.,'remaining_arrival_sec':120.,'remaining_total_sec':240.,
        'outstanding':64.,'overdue_first':64.,'output_stalled':64.,'client_tokens_per_sec_5s':512.,
        'busy':1.,'cooldown_remaining_sec':5.,'healthy_instances':4.,
        'running_difference':64.,'used_fraction_difference':1.,'destination_available_blocks':128.}


def _finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def _pack(values):
    missing=[not _finite(x) for x in values]
    return [0. if absent else float(x) for x,absent in zip(values,missing)],missing


class RealObservationBuilder:
    def __init__(self, instance_mapping, *, ttft_slo_ms=2000., tpot_slo_ms=150., max_age_sec=2., cooldown_sec=5.):
        self.mapping=tuple(tuple(x) for x in instance_mapping)
        if len(self.mapping)!=4 or len(set(self.mapping))!=4:
            raise ValueError('four stable instance epochs required')
        self.ttft_sec=ttft_slo_ms/1000
        self.gap_sec=max(1.,5*tpot_slo_ms/1000)
        self.max_age_sec,self.cooldown_sec=max_age_sec,cooldown_sec
        self.history=deque()

    def build(self,snapshot,*,now_ns,ledger,episode_start_ns,arrival_end_ns,deadline_ns):
        if tuple(tuple(x) for x in snapshot['mapping'])!=self.mapping:
            raise RuntimeError('instance mapping invalidated')
        if snapshot['received_ns']>now_ns:
            raise ValueError('snapshot was not received by decision time')
        # Historical decisions are immutable outputs, never recomputed with late data.
        states={x['instance_id']:x for x in snapshot['instances']}
        visible=ledger.observe(now_ns)['requests']
        self.history.append((now_ns,snapshot))
        while len(self.history)>1 and self.history[1][0]<=now_ns-5_000_000_000:
            self.history.popleft()
        old_ns,old_snapshot=self.history[0]
        old={x['instance_id']:x for x in old_snapshot['instances']}
        dt=(now_ns-old_ns)/1e9
        features,missing=[],[]
        for instance_id,epoch in self.mapping:
            s=states[instance_id]; total=s.get('total_blocks'); used=s.get('used_blocks'); free=s.get('free_blocks')
            ratio=(used/total if _finite(used) and _finite(total) and total>0 else None)
            free_ratio=(free/total if _finite(free) and _finite(total) and total>0 else None)
            prior=old.get(instance_id,{})
            def delta(name):
                a,b=s.get(name),prior.get(name)
                return (a-b)/dt if dt>0 and _finite(a) and _finite(b) and a>=0 and b>=0 else None
            throughput=delta('output_tokens_total')
            if throughput is not None and throughput<0:throughput=None
            old_ratio=(prior.get('used_blocks')/prior['total_blocks'] if prior.get('total_blocks') and _finite(prior.get('used_blocks')) else None)
            age=(now_ns-s['measured_ns'])/1e9 if s.get('measured_ns') is not None else None
            vals=[s.get('running'),s.get('waiting'),s.get('head_wait_sec'),ratio,free_ratio,
                  s.get('candidate_length'),s.get('candidate_blocks'),s.get('last_step_ms'),age,
                  throughput,delta('waiting'),(ratio-old_ratio)/dt if dt>0 and ratio is not None and old_ratio is not None else None]
            row,mask=_pack(vals);features.append(row);missing.append(mask)
        outstanding=[r for r in visible if r['sent_ns'] is not None and r['client_terminal_ns'] is None]
        overdue=sum(r['client_first_ns'] is None and now_ns-r['sent_ns']>self.ttft_sec*1e9 for r in outstanding)
        stalled=sum(r['client_last_ns'] is not None and now_ns-r['client_last_ns']>self.gap_sec*1e9 for r in outstanding)
        window_sec=min(5.,max(0.,(now_ns-episode_start_ns)/1e9))
        tokens=sum(n for r in visible for t,n in r['token_increments'] if now_ns-5e9<t<=now_ns)
        cooldown=max(0.,self.cooldown_sec-(now_ns-snapshot['last_terminal_ns'])/1e9) if snapshot.get('last_terminal_ns') is not None else 0.
        global_values=[float(now_ns<arrival_end_ns),float(arrival_end_ns<=now_ns<deadline_ns),
                       max(0.,(arrival_end_ns-now_ns)/1e9),max(0.,(deadline_ns-now_ns)/1e9),
                       len(outstanding),overdue,stalled,tokens/window_sec if window_sec>0 else None,
                       float(snapshot['busy']),cooldown,sum(s.get('healthy') is True for s in states.values())]
        global_features,global_missing=_pack(global_values)
        pair_features,pair_missing,mask=[],[],[True]
        for src,dst in PAIRS:
            a,b=states[self.mapping[src][0]],states[self.mapping[dst][0]]
            valid,_=eligible_pair(a,b,now_ns,self.max_age_sec)
            valid=valid and not snapshot['busy'] and cooldown==0 and not snapshot.get('lease_expired') and not snapshot.get('invalid_reasons')
            mask.append(bool(valid))
            def difference(name):
                x,y=a.get(name),b.get(name)
                return x-y if _finite(x) and _finite(y) else None
            def used_ratio(s):
                return s['used_blocks']/s['total_blocks'] if _finite(s.get('used_blocks')) and s.get('total_blocks',0)>0 else None
            x,y=used_ratio(a),used_ratio(b)
            avail=(b['free_blocks']-b['waiting_blocks'] if _finite(b.get('free_blocks')) and _finite(b.get('waiting_blocks')) else None)
            row,missing_row=_pack([difference('running'),x-y if x is not None and y is not None else None,avail])
            pair_features.append(row);pair_missing.append(missing_row)
        return {'schema':SCHEMA_VERSION,'instance_feature_names':list(INSTANCE_FEATURES),
                'global_feature_names':list(GLOBAL_FEATURES),'pair_feature_names':list(PAIR_FEATURES),
                'instance_mapping':[list(x) for x in self.mapping],
                'instance_features':features,'instance_missing':missing,
                'global_features':global_features,'global_missing':global_missing,
                'pair_features':pair_features,'pair_missing':pair_missing,'action_mask':mask,
                'snapshot_version':snapshot['version'],'obs_received_at':snapshot['received_ns'],'decision_ns':now_ns}


class RealNormalizer:
    """Named schema-bound scales; immutable within and across rollout/evaluation."""
    version='real-normalizer-v1'
    def __setattr__(self,name,value):
        if getattr(self,'_frozen',False):raise AttributeError('real normalizer is frozen')
        object.__setattr__(self,name,value)
    def __init__(self,scales=None):
        values=dict(SCALES if scales is None else scales)
        if set(values)!=set(SCALES) or any(not _finite(v) or v<=0 for v in values.values()):
            raise ValueError('exact feature scale schema with finite positive scales required')
        self.scales=MappingProxyType(values)
        self.hash=digest(self.state_dict())
        self._frozen=True

    def state_dict(self):
        return {'version':self.version,'schema':SCHEMA_VERSION,'instance_features':list(INSTANCE_FEATURES),
                'global_features':list(GLOBAL_FEATURES),'pair_features':list(PAIR_FEATURES),'scales':dict(self.scales)}

    @classmethod
    def from_state_dict(cls,state):
        candidate=cls(state.get('scales'))
        if candidate.state_dict()!=state:raise ValueError('normalizer version/schema mismatch')
        return candidate

    def transform(self,observation):
        if digest(self.state_dict())!=self.hash:raise ValueError('normalizer state mutated')
        if observation['schema']!=SCHEMA_VERSION:raise ValueError('wrong observation schema')
        result={'version':self.version,'schema':SCHEMA_VERSION,'normalizer_hash':self.hash,
                'action_mask':list(observation['action_mask'])}
        for group,names in [('instance',INSTANCE_FEATURES),('global',GLOBAL_FEATURES),('pair',PAIR_FEATURES)]:
            if observation[group+'_feature_names']!=list(names):raise ValueError('feature order mismatch')
            values=observation[group+'_features'];masks=observation[group+'_missing']
            def normalize(row,mask):
                return [0. if absent else value/self.scales[name] for name,value,absent in zip(names,row,mask)]+[float(x) for x in mask]
            result[group+'_features']=(normalize(values,masks) if group=='global' else
                                       [normalize(row,mask) for row,mask in zip(values,masks)])
        return result
