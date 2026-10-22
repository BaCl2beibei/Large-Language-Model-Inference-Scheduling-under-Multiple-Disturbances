"""Auditable finite-episode rewards from complete client events, without clipping.

Reconstruction is offline after score + cleanup audits. It never mutates saved
observations/actions/logp or treats a chunk as a token. Scales must be measured
from preregistered healthy Stay opportunity runs before rewards can be used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .real_metrics import NS, _overlap


@dataclass(frozen=True)
class RewardWeights:
    good: float = 1.
    hold: float = .2
    first: float = .5
    gap: float = .2
    attempt: float = .02
    bad: float = .2
    tail: float = 2.

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('finite nonnegative reward weights required')


@dataclass(frozen=True)
class RewardScales:
    C_ref: float
    N_ref: float
    provenance: str

    def __post_init__(self):
        if not self.provenance or any(not math.isfinite(v) or v < 1 for v in (self.C_ref, self.N_ref)):
            raise ValueError('measured scales with floor 1 and provenance required')


def terminal_good(row, ttft_ms=2000., tpot_ms=150.):
    return (row['outcome']=='success' and row['actual_output_tokens']>0
            and row['ttft_ms'] is not None and row['ttft_ms']<=ttft_ms
            and (row['actual_output_tokens']==1 or row['tpot_ms'] is not None and row['tpot_ms']<=tpot_ms))


def reconstruct_rewards(ledger, edges_ns, migrations, *, scales, cleanup_through_ns,
                        valid_episode, weights=RewardWeights()):
    """Partition events into [lo,hi), with the final deadline included once.

    migrations contains one reconciled record per non-Stay decision. attempt_ns
    is the original coordinator's start, never a submit/receipt timestamp.
    A known terminal after the scoring deadline stays outside scoring.
    """
    if not valid_episode:
        raise ValueError('invalid episodes cannot receive finalised training rewards')
    if scales is None:
        raise ValueError('C_ref/N_ref unmeasured; retain raw events until opportunity scales exist')
    if ledger.audit()['status']!='passed' or ledger.audit(through_ns=cleanup_through_ns,require_cleanup=True)['status']!='passed':
        raise ValueError('complete scoring and cleanup audits required')
    edges=list(edges_ns)
    if (len(edges)<2 or edges[0]!=ledger.start_ns or edges[-1]!=ledger.deadline_ns
            or any(type(t) is not int for t in edges) or any(a>=b for a,b in zip(edges,edges[1:]))):
        raise ValueError('strict partition of the finite scoring interval required')
    ids=set();migration_ids=set()
    for record in migrations:
        if not record.get('decision_id') or record['decision_id'] in ids:
            raise ValueError('duplicate or missing decision identity')
        ids.add(record['decision_id'])
        if record.get('terminal_status') not in ('succeeded','failed','rejected','aborted') or type(record.get('terminal_ns')) is not int:
            raise ValueError('unknown migration terminal cannot be scored')
        if record['terminal_ns']<ledger.start_ns:raise ValueError('foreign pre-episode terminal')
        stamp=record.get('attempt_ns');mid=record.get('migration_id')
        if stamp is not None:
            if type(stamp) is not int or stamp<ledger.start_ns or not mid or mid in migration_ids or stamp>record['terminal_ns']:
                raise ValueError('invalid or duplicate original migration attempt')
            migration_ids.add(mid)
        if record['terminal_status']=='succeeded' and stamp is None:
            raise ValueError('success without original coordinator start')
    rows,problems=ledger._rows(occurred_by_ns=ledger.deadline_ns)
    if problems:raise ValueError('request ledger inconsistent')
    gap_ns=round(max(1.,5*ledger.tpot_slo_ms/1000)*NS)
    result=[]
    for index,(lo,hi) in enumerate(zip(edges,edges[1:])):
        final=index==len(edges)-2
        def inside(t):return t is not None and lo<=t and (t<=hi if final else t<hi)
        raw=dict(good=0,hold=0.,first=0.,gap=0.,attempt=0,bad=0,tail=0)
        for row in rows:
            sent,terminal=row['sent_ns'],row['client_terminal_ns']
            if sent is None:continue
            end=min(terminal if terminal is not None else ledger.deadline_ns,ledger.deadline_ns)
            raw['hold']+=_overlap(sent,end,lo,hi)
            first=row['client_first_ns']
            raw['first']+=_overlap(sent+round(ledger.ttft_slo_ms*1e6),min(first if first is not None else end,end),lo,hi)
            times=[t for t,n in row['token_increments'] if t<=end and n>0]
            for a,b in zip(times,times[1:]+[end]):raw['gap']+=_overlap(a+gap_ns,b,lo,hi)
            raw['good']+=int(inside(terminal) and terminal_good(row,ledger.ttft_slo_ms,ledger.tpot_slo_ms))
            if final:raw['tail']+=int(sent<=ledger.deadline_ns and (terminal is None or terminal>ledger.deadline_ns))
        for record in migrations:
            raw['attempt']+=int(inside(record.get('attempt_ns')))
            raw['bad']+=int(record['terminal_status'] in ('failed','rejected','aborted') and inside(record['terminal_ns']))
        normalized={key:value/(scales.C_ref if key=='good' else scales.N_ref if key in ('hold','first','gap','tail') else 1.)
                    for key,value in raw.items()}
        weighted={key:normalized[key]*getattr(weights,key)*(1 if key=='good' else -1) for key in raw}
        reward=math.fsum(weighted.values())
        if not math.isfinite(reward):raise ValueError('nonfinite reward')
        result.append({'start_ns':lo,'end_ns':hi,'dt_sec':(hi-lo)/NS,'final_finite_terminal':final,
                       'raw':raw,'normalized':normalized,'weighted':weighted,'reward':reward,
                       'scales':asdict(scales),'weights':asdict(weights),'clipped':False,
                       'gap_is_stall_proxy_not_exact_ITL':True})
    return result


def fit_reference_scales(episodes, *, healthy):
    """Pooled means over healthy Stay opportunity runs only, never probe/dev/final."""
    eligible=[]
    for episode in episodes:
        if episode.get('phase')!='opportunity' or episode.get('policy')!='Stay':continue
        if episode.get('status')!='passed':raise ValueError('invalid Stay evidence cannot be silently omitted')
        if healthy(episode):eligible.append(episode)
    if not eligible:raise ValueError('no measured healthy Stay opportunity data for C_ref/N_ref')
    for episode in eligible:
        m=episode['metrics']
        if (any(not math.isfinite(m[key]) or m[key]<0 for key in ('completed','outstanding_request_seconds','duration_sec'))
                or m['duration_sec']<=0):raise ValueError('invalid measured reference metrics')
    duration=sum(e['metrics']['duration_sec'] for e in eligible)
    if duration<=0:raise ValueError('nonpositive reference duration')
    c=sum(e['metrics']['completed'] for e in eligible)/duration
    n=sum(e['metrics']['outstanding_request_seconds'] for e in eligible)/duration
    provenance=','.join(e['episode_id'] for e in eligible)
    return RewardScales(max(1.,c),max(1.,n),provenance),{
        'status':'measured_from_healthy_Stay_opportunity','raw_C_ref_rps':c,'raw_N_ref':n,
        'episode_ids':[e['episode_id'] for e in eligible],'duration_sec':duration,
        'excluded_unhealthy_Stay_ids':[e['episode_id'] for e in episodes
            if e.get('phase')=='opportunity' and e.get('policy')=='Stay' and not healthy(e)]}


def reward_scale_diagnostics(episodes):
    """Report component concentration and reward/primary disagreements; never retune."""
    totals={};pairs={}
    for episode in episodes:
        rows=episode.get('reward_windows',[])
        sums={key:math.fsum(abs(row['weighted'][key]) for row in rows) for key in asdict(RewardWeights())}
        overall=math.fsum(sums.values())
        totals[episode['episode_id']]={'absolute_component_sums':sums,
            'largest_fraction':max(sums.values())/overall if overall else None}
        pairs.setdefault((episode['scene'],episode['seed']),{})[episode['policy']]=episode
    disagreements=[]
    for (scene,seed),pair in pairs.items():
        if set(pair)!= {'Stay','Balanced'}:continue
        def reward(e):return math.fsum(w['reward'] for w in e.get('reward_windows',[]))
        dr=reward(pair['Balanced'])-reward(pair['Stay'])
        dg=pair['Balanced']['metrics']['slo_goodput_rps']-pair['Stay']['metrics']['slo_goodput_rps']
        if dr>0 and dg<0:disagreements.append({'scene':scene,'seed':seed,'reward_delta':dr,'goodput_delta':dg})
    return {'component_audit':totals,'reward_improves_while_primary_declines':disagreements,
            'automatic_weight_revision':False}
