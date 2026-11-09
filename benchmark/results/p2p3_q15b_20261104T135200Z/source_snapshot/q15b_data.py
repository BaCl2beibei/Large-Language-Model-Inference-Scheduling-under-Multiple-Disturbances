"""Streaming, source-bound Q15-B development evidence. Never modifies input logs."""
from pathlib import Path
import hashlib
import json
import math
from collections import Counter
import numpy as np
from .real_budget import atomic_json
from .full_completion import summarize_values
from .cohort_reward import CohortReward

ROOT=Path(__file__).resolve().parents[1]
DEVELOPMENT=ROOT/'benchmark/results/p0p1_fast_20261104T104200Z'
REPLAY=ROOT/'benchmark/results/p0p1_fast_replay_20261104T124801Z'


def read(p): return json.loads(Path(p).read_text())

def stream(p):
    with Path(p).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def sources():
    specs=[(DEVELOPMENT,'05_Q15-B_141103_Stay','fit','Q15-B',141103,'Stay'),
           (DEVELOPMENT,'06_Q15-B_141103_Balanced','fit','Q15-B',141103,'Balanced'),
           (DEVELOPMENT,'07_Q15-B_141104_Balanced','development_validation_unpaired','Q15-B',141104,'Balanced'),
           (REPLAY,'01_Q15-B_141104_Balanced','development_validation','Q15-B',141104,'Balanced'),
           (REPLAY,'02_Q15-B_141104_Stay','development_validation','Q15-B',141104,'Stay'),
           (DEVELOPMENT,'01_Q15-A_141103_Stay-A','negative_diagnostic','Q15-A',141103,'Stay'),
           (DEVELOPMENT,'03_Q15-A_141103_Stay-B','negative_diagnostic','Q15-A',141103,'Stay'),
           (DEVELOPMENT,'04_Q15-A_141103_Balanced','negative_diagnostic','Q15-A',141103,'Balanced')]
    return [dict(path=str(run/'runs'/name),run=str(run),episode_id=name,split=split,candidate=c,seed=seed,policy=p) for run,name,split,c,seed,p in specs]


def episode_origin(path, trace):
    first=next(stream(Path(path)/'client_events.jsonl'))
    return first['data']['scheduled_ns']-round(trace[0]['scheduled_arrival_sec']*1e9)


def extract(out, T_ref):
    out=Path(out);cal=out/'calibration';cal.mkdir(exist_ok=True)
    summaries=[];service=[];moves=[];reward_reports=[]
    with (cal/'service_records.jsonl').open('w') as sf,(cal/'migration_records.jsonl').open('w') as mf:
        for spec in sources():
            path=Path(spec['path']);e=read(path/'episode.json');trace=read(path/'trace.json');origin=episode_origin(path,trace)
            assert e['status']=='passed' and e['cohort_complete'] and read(path/'execution_audit.json')['status']=='passed'
            count=Counter();reason=Counter();last={};ts=[];missing=Counter();n=0
            for d in stream(path/'decisions.jsonl'):
                at=d.get('observation_decision_ns');mask=d['action_mask'];action=d.get('executed_action',d['sampled_action'])
                if d['monotonic_ns']>=max(r['client_terminal_ns'] for r in e['requests']):continue
                ts.append(at if at is not None else d['monotonic_ns']);n+=1
                legal=sum(mask[1:]);count['all']+=1;count['eligible']+=legal>0
                count['forced_stay' if not legal else 'voluntary_stay' if not action else 'migration_attempt']+=1
                if not legal:
                    if at is None:reason['unknown_missing_observation_time']+=1
                    else:
                        s=d['snapshot'];found=set()
                        if s['busy']:found.add('busy')
                        if s.get('last_terminal_ns') is not None and at-s['last_terminal_ns']<5e9:found.add('cooldown')
                        from .real_bridge import PAIRS,eligible_pair
                        states={s['instance_id']:s for s in s['instances']}
                        for a,b in PAIRS:
                            _,r=eligible_pair(states[s['mapping'][a][0]],states[s['mapping'][b][0]],at)
                            if r:found.add(r)
                        reason.update(found)
                for s in d['snapshot']['instances']:
                    prev=last.get(s['instance_id']);last[s['instance_id']]=(d['snapshot']['received_ns'],s)
                    if prev is None:continue
                    stamp,p=prev;dt=(d['snapshot']['received_ns']-stamp)/1e9
                    a,b=p.get('output_tokens_total'),s.get('output_tokens_total')
                    running=(s.get('running',0)+p.get('running',0))/2
                    if a is None or b is None:missing['output_tokens_total']+=1;continue
                    delta=b-a
                    if not (0<dt<5 and delta>0 and running>0):continue
                    row=dict(source=spec['episode_id'],split=spec['split'],seed=spec['seed'],instance=s['instance_id'],dt_sec=dt,
                             running=running,waiting=s.get('waiting'),kv_fraction=s['used_blocks']/s['total_blocks'],
                             mean_context=(s['used_blocks']+p['used_blocks'])*8/running,
                             token_progress=delta,token_interval_ms=dt*running/delta*1000,
                             engine_executor_step_ms=s.get('last_step_ms'),busy=d['snapshot']['busy'])
                    sf.write(json.dumps(row)+'\n');service.append(row)
            unique={r['migration_id']:r for r in e['migration_receipts'].values() if r.get('migration_id')}
            success=[]
            for mid,r in unique.items():
                if r.get('migration_start_ns') is None:continue
                assert r.get('terminal_status') in ('succeeded','failed','rejected')
                duration=(r['terminal_ns']-r['migration_start_ns'])/1e6
                row=dict(source=spec['episode_id'],split=spec['split'],seed=spec['seed'],migration_id=mid,
                         blocks=r.get('copied_blocks'),duration_ms=duration,status=r['terminal_status'],
                         request_id=r.get('actual_request_id'),source_id=r.get('src_id'),target_id=r.get('dst_id'),
                         pause_ms=None,pause_unknown=True)
                mf.write(json.dumps(row)+'\n');moves.append(row)
                if r['terminal_status']=='succeeded':success.append(r.get('actual_request_id'))
            reward=CohortReward(T_ref).reconstruct(e,ts)
            reward_reports.append(dict(source=spec['episode_id'],return_value=reward['return'],objective=reward['objective'],identity_error=abs(reward['return']+reward['objective'])))
            summaries.append(dict(**spec,trace_sha256=e['trace_sha256'],planned=128,completed=e['measured_request_count'],counts=dict(count),
                                  no_opportunity_reasons=dict(reason),succeeded=len(success),distinct_requests=len(set(success)),
                                  missing=dict(missing),decisions=n,origin_ns=origin))
    atomic_json(out/'opportunity_summary.json',summaries);atomic_json(out/'historical_reward_identity.json',reward_reports)
    atomic_json(cal/'extraction_index.json',dict(sources=summaries,service_records=len(service),migration_records=len(moves),
                definition='effective per-request token interval from cumulative engine counters, separate from executor wall time; includes batch overlap/prefill overhead, excludes long idle intervals',
                pause='unavailable: total coordinator transfer time observed; pause not inferred as measured',
                blocks_to_bytes='not performed; blocks retained as native units; no guessed byte size'))
    return service,moves


def fit_models(out, rows, moves):
    """Compare coarse bucket medians and low-complexity ridge on declared dev groups."""
    from .causal_calibration import _metric
    fit=[r for r in rows if r['split']=='fit' and .5<r['token_interval_ms']<2000]
    valid=[r for r in rows if r['split']=='development_validation' and .5<r['token_interval_ms']<2000]
    edges=[0,2,4,8,16,32,64,128]
    def bucket(n):return min(6,max(0,int(np.searchsorted(edges,n,side='right')-1)))
    medians=[float(np.median([r['token_interval_ms'] for r in fit if bucket(r['running'])==k])) if any(bucket(r['running'])==k for r in fit) else None for k in range(7)]
    globalmedian=float(np.median([r['token_interval_ms'] for r in fit]))
    def bp(r):return medians[bucket(r['running'])] or globalmedian
    def feature(r):return [1.,r['running'],r['mean_context']/1024]
    X=np.array([feature(r) for r in fit]);y=np.array([r['token_interval_ms'] for r in fit])
    coef=np.linalg.solve(X.T@X+np.diag([0.,1.,1.]),X.T@y)
    def rp(r):return max(1.,float(np.dot(feature(r),coef)))
    truth=[r['token_interval_ms'] for r in valid]
    bm=_metric(truth,[bp(r) for r in valid]);rm=_metric(truth,[rp(r) for r in valid])
    # Explicit development validation selection; never described as held-out blind testing.
    bmae=float(np.mean([abs(bp(r)-r['token_interval_ms']) for r in valid]));rmae=float(np.mean([abs(rp(r)-r['token_interval_ms']) for r in valid]))
    chosen='bucket' if bmae<=rmae else 'ridge'
    trainingmoves=[m for m in moves if m['split']=='fit' and m['status']=='succeeded' and m['blocks']]
    A=np.array([[1.,m['blocks']] for m in trainingmoves]);target=np.array([m['duration_ms'] for m in trainingmoves])
    mc=np.linalg.lstsq(A,target,rcond=None)[0];mc[0]=max(1.,mc[0]);mc[1]=max(0.,mc[1])
    vm=[m for m in moves if m['split']=='development_validation' and m['status']=='succeeded' and m['blocks']]
    migration_errors={}
    for q,tol in [(.5,.2),(.95,.3)]:
        actual=float(np.quantile([m['duration_ms'] for m in vm],q));pred=float(np.quantile([mc[0]+mc[1]*m['blocks'] for m in vm],q))
        migration_errors[str(q)]={'actual_ms':actual,'predicted_ms':pred,'relative_error':abs(pred-actual)/actual,'passed':abs(pred-actual)/actual<=tol,'n':len(vm)}
    model=dict(schema='q15b-effective-cadence-v1',selected=chosen,bucket_edges=edges,bucket_medians_ms=medians,global_median_ms=globalmedian,
               ridge_coefficients=coef.tolist(),migration_intercept_ms=float(mc[0]),migration_ms_per_block=float(mc[1]),
               capacity=8565,block_size_tokens=16,decision_interval_sec=1.,cooldown_sec=5.,pause_assumption='full transfer duration as conservative upper bound, not a measured pause',
               no_random_failures=True,fit_group='141103 Q15-B only',validation_selection='141104 development validation, used for simple model choice',
               coverage={'running':[min(r['running'] for r in fit),max(r['running'] for r in fit)],'mean_context':[min(r['mean_context'] for r in fit),max(r['mean_context'] for r in fit)]})
    atomic_json(Path(out)/'calibration/model.json',model)
    atomic_json(Path(out)/'calibration/fit_report.json',dict(status='passed',bucket_error=bm,ridge_error=rm,bucket_mae_ms=bmae,ridge_mae_ms=rmae,
                selected=chosen,migration_errors=migration_errors,fit_samples=len(fit),validation_samples=len(valid),missing_pause_rate=1.,
                limitations=['coarse token-window model, not GPU kernel timing','no bandwidth/length extrapolation','migration source/target interference only supported by observed cadence; not a causal gain label']))
    return model
