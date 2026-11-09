"""CPU-only physical refit from immutable, audited Mig-PPO V2 ledgers.

No policy labels, environment IDs, rewards or final costs are service-model inputs.
Old validation traces are development data; no independent validation is claimed.
"""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'benchmark/results/mig_ppo_v2_multienv_20261105T130307Z'
FOLLOWUP = ROOT / 'benchmark/results/mig_ppo_v2_followup_20261106T080500Z'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def numeric(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError('non-serializable audit value: '+type(v).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,default=numeric)+'\n')


def rows(path):
    path = Path(path)
    if not path.exists():
        path = Path(str(path)+'.gz')
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_hash(trace):
    # Match groups across hotspot placements, routes and candidate names.
    base = [{k: r[k] for k in ('request_id', 'scheduled_arrival_sec', 'input_tokens', 'output_tokens', 'prompt')}
            for r in trace]
    return hashlib.sha256(json.dumps(base, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sources():
    pairs = []
    for root in (OLD, FOLLOWUP):
        for path in sorted((root/'comparisons').glob('*.json')):
            c = read(path)
            if c.get('status') != 'complete':
                continue
            pairs.append(dict(pair_id=c['candidate_id'], environment=c['environment_id'],
                              root=str(root), role='fit', episodes={p: c['episodes'][p]['episode_id'] for p in ('Stay','Balanced')}))
    validation = defaultdict(dict)
    for path in sorted((OLD/'runs').glob('VALIDATION_*/episode.json')):
        e = read(path)
        if e.get('status')=='passed' and e.get('cohort_complete'):
            env = path.parent.name.split('_')[1]
            validation[env][e['policy']] = path.parent.name
    for env, episodes in sorted(validation.items()):
        if len(episodes)==2:
            pairs.append(dict(pair_id='DEVELOPMENT_'+env, environment=env,
                              root=str(OLD), role='development_validation',episodes=episodes))
    return pairs


def hardware(root):
    audit = read(root/'environment_audit.json')
    mapping = {}
    def visit(x):
        if isinstance(x,dict):
            if x.get('instance_id') and x.get('gpu_uuid'):
                mapping[x['instance_id']] = x['gpu_uuid']
            for v in x.values(): visit(v)
        elif isinstance(x,list):
            for v in x: visit(v)
    visit(audit)
    return mapping


def extract(run_dir):
    """Retain raw identities and exact frame times; reduce correlated frame copies."""
    run_dir = Path(run_dir)
    pairs = sources()
    samples, engines, migrations, prefill, inventory, excluded = [], [], [], [], [], []
    all_episodes = []
    for pair in pairs:
        for policy, eid in pair['episodes'].items():
            all_episodes.append((Path(pair['root']), eid, pair, policy))
    # Additional completed lone Stay runs support physics, not paired gain counts.
    for root in sorted((ROOT/'benchmark/results').glob('mig_ppo_v2_hot1h_repeat*')):
        for path in sorted((root/'runs').glob('*/episode.json')):
            if 'warmup' not in path.parent.name:
                all_episodes.append((root,path.parent.name,dict(role='fit',pair_id=None,environment='HOT1-H_repeat'), read(path)['policy']))
    paired_ids={eid for _,eid,_,_ in all_episodes}
    for path in sorted((OLD/'runs').glob('*/episode.json')):
        if path.parent.name not in paired_ids and 'warmup' not in path.parent.name:
            role='development_validation' if path.parent.name.startswith('VALIDATION_') else 'fit'
            all_episodes.append((OLD,path.parent.name,dict(role=role,pair_id=None,environment='unpaired'),read(path).get('policy')))
    mappings = {}
    for root,eid,pair,policy in all_episodes:
        d = root/'runs'/eid
        ep = read(d/'episode.json')
        audit = read(d/'execution_audit.json') if (d/'execution_audit.json').exists() else {}
        if not (ep.get('status')=='passed' and ep.get('cohort_complete') and ep.get('route_audit_passed')
                and ep.get('audit',{}).get('status')=='passed' and audit.get('status')=='passed'):
            excluded.append(dict(episode_id=eid,reason='incomplete or failed request/route/execution audit'))
            continue
        trace = read(d/'trace.json'); reqs=rows(d/'requests.jsonl'); decs=rows(d/'decisions.jsonl')
        events=rows(d/'events.jsonl')
        origin = int(round(np.median([r['scheduled_ns']-next(t['scheduled_arrival_sec'] for t in trace if t['request_id']==r['request_id'])*1e9 for r in reqs])))
        group = base_hash(trace)
        hw = mappings.setdefault(str(root), hardware(root))
        ids = [r[0] for r in ep['frozen_mapping']]
        keys = [hw.get(i, 'unmapped:'+i) for i in ids]
        times = np.array([x.get('observation_decision_ns',x['monotonic_ns']) for x in decs],dtype=np.int64)
        common=dict(episode_id=eid, group=group, role=pair['role'])
        diag=[e for e in events if e.get('kind')=='diagnostic']
        step_wall={e['data'].get('batch_id'):e['data'].get('engine_step_wall_ms') for e in diag
                   if e.get('data',{}).get('event')=='llm_engine_step_completed'}
        engine_outputs=defaultdict(list); admissions={}; firsts={}; migdiag=defaultdict(list)
        for e in events:
            if e.get('kind')=='engine_first': firsts[e.get('request_id')]=e['occurred_ns']
        for e in diag:
            v=e.get('data') or {};k=v.get('event');iid=v.get('instance_id');at=v.get('monotonic_ns')
            if k=='request_engine_admitted': admissions.setdefault(e.get('request_id'),at)
            if v.get('migration_id'): migdiag[v['migration_id']].append(v)
            if k=='engine_iteration_completed' and v.get('succeeded') and iid in ids:
                engines.append(dict(**common, hardware=keys[ids.index(iid)],
                    n=float(v.get('batch_size') or 0), context=float(v.get('total_context_tokens') or 0),
                    prefill_tokens=float(v.get('prefill_tokens') or 0),prefills=int(v.get('num_prefill_requests') or 0),
                    ms=float(v['executor_wall_ms']),
                    engine_step_wall_ms=float(step_wall.get(v.get('batch_id')) or v['executor_wall_ms'])))
            if k=='engine_request_output' and at and e.get('request_id') and iid in ids:
                engine_outputs[e['request_id']].append((at,v,iid))
        aggregated=defaultdict(list)
        for rid, rr in engine_outputs.items():
            rr.sort(key=lambda v:v[0]); previous=None
            for at,v,iid in rr:
                output=int(v.get('output_tokens') or 0)
                if previous and iid==previous[2] and output>previous[1]:
                    dt=(at-previous[0])/1e6;delta=output-previous[1]
                    n=float(v.get('running_requests') or 0)
                    idx=max(0,min(len(times)-1,int(np.searchsorted(times,at,side='right')-1)))
                    states=decs[idx]['snapshot']['instances']
                    state=next(s for s in states if s['instance_id']==iid)
                    kv=float(v.get('kv_utilization') or 0)*float(state['total_blocks'])
                    ctx=max(0,kv*16-32*n)  # interval midpoint of a 64-step output bundle
                    feat=dict(n=n,context=ctx,active=sum(s.get('running',0)>0 for s in states),
                              global_n=sum(s.get('running',0) for s in states),
                              busy=float(decs[idx]['snapshot'].get('busy',False)))
                    if delta>=8 and 1<=dt/delta<=1000 and n>0:
                        row=dict(**common,hardware=keys[ids.index(iid)],time_sec=(at-origin)/1e9,
                                 ms=dt/delta,delta=delta,**feat)
                        aggregated[(iid,round((at-origin)/1e9,1))].append(row)
                previous=(at,output,iid)
        for rr in aggregated.values():
            r=rr[0].copy()
            for k in ('ms','n','context','active','global_n','busy'):r[k]=float(np.median([v[k] for v in rr]))
            r['correlated_request_copies']=len(rr);samples.append(r)
        for r in reqs:
            if r['request_id'] in admissions and r['request_id'] in firsts:
                elapsed=(firsts[r['request_id']]-admissions[r['request_id']])/1e6
                if 0<elapsed<5000:prefill.append(dict(**common,ms=elapsed,prompt_tokens=r.get('expected_input_tokens',128)))
        server_to_trace={e.get('server_request_id'):e.get('request_id') for e in events if e.get('server_request_id') and e.get('request_id')}
        for v in ep.get('migration_receipts',{}).values():
            if v.get('terminal_status')!='succeeded' or not v.get('migration_start_ns') or not v.get('copied_blocks'):continue
            ds=migdiag[v['migration_id']]
            bykind=defaultdict(list)
            for x in ds:bykind[x['event']].append(x)
            pause=min((x['monotonic_ns'] for x in bykind['migration_request_pause_started']),default=None)
            resume=max((x['monotonic_ns'] for x in bykind['migration_request_resumed']),default=None)
            transfer=sum(float(x.get('transfer_ms') or 0) for x in bykind['migration_transfer_completed'])
            duration=(v['terminal_ns']-v['migration_start_ns'])/1e6
            migrations.append(dict(**common,hardware_src=hw.get(v['src_id'],'unmapped:'+v['src_id']),
                hardware_dst=hw.get(v['dst_id'],'unmapped:'+v['dst_id']),blocks=v['copied_blocks'],ms=duration,
                pause_ms=(resume-pause)/1e6 if pause and resume and resume>=pause else None,
                transfer_ms=transfer or None,migration_id=v['migration_id'],request_id=server_to_trace.get(v.get('actual_request_id')),
                source_index=ids.index(v['src_id']),target_index=ids.index(v['dst_id']),time_sec=(v['migration_start_ns']-origin)/1e9))
        increments=Counter(n for r in reqs for _,n in r['token_increments'])
        item=dict(**common, root=str(root),path=str(d),policy=policy,pair_id=pair['pair_id'],environment=pair['environment'],
            origin_ns=origin,hardware=keys,request_count=len(reqs),output_tokens=sum(r['actual_output_tokens'] for r in reqs),
            frame_delta_counts=dict(increments),trace_sha256=sha(d/'trace.json'),episode_sha256=sha(d/'episode.json'),
            valid=True,source_files={p.name:sha(p) for p in d.iterdir() if p.name in ('requests.jsonl.gz','decisions.jsonl.gz','events.jsonl.gz','execution_audit.json')})
        inventory.append(item)
        print('extracted',eid,len(samples),flush=True)
    valid={r['episode_id'] for r in inventory}
    pairs=[p for p in pairs if all(e in valid for e in p['episodes'].values())]
    fit_groups={r['group'] for r in inventory if r['role']=='fit'}
    dev_groups={r['group'] for r in inventory if r['role']=='development_validation'}
    if fit_groups & dev_groups: raise ValueError('fit/development base-trace group overlap')
    data=dict(pairs=pairs,episodes=inventory,samples=samples,engines=engines,prefill=prefill,migrations=migrations,excluded=excluded,
              fit_groups=sorted(fit_groups),development_groups=sorted(dev_groups),independent_validation_pending=True,
              validation_note='Old independent validation was previously reviewed; now development validation. Followups are fit data. Repeated Stay runs are unpaired physics support.')
    save(run_dir/'data_audit.json',data)
    return data


def design(r,method):
    from rl_scheduler.service_model import physical_cadence_features
    return physical_cadence_features(r,method)


def fit_service(samples,method):
    fit=[s for s in samples if s['role']=='fit']
    x=np.array([design(r,method) for r in fit]);y=np.array([r['ms'] for r in fit])
    # Huber-like IRLS with nonnegative terms; downweight extreme host stalls.
    weights=np.ones(len(y));coef=None
    for _ in range(6):
        coef=nnls(x*weights[:,None]**.5,y*weights**.5)[0]
        err=y-x@coef;scale=max(1.,float(np.median(np.abs(err)))*1.4826)
        weights=np.minimum(1.,1.5*scale/np.maximum(np.abs(err),1e-9))
    factors={}
    for key in sorted({r['hardware'] for r in fit}):
        rr=[r for r in fit if r['hardware']==key]
        ratio=np.median([r['ms']/max(1.,float(np.dot(design(r,method),coef))) for r in rr])
        factors[key]=float(np.clip(1+(ratio-1)*len(rr)/(len(rr)+50),.4,2.5))
    return dict(method=method,coefficients=coef.tolist(),hardware_factors=factors,min_ms=5.)


def predict(model,r):
    return max(model['min_ms'],float(np.dot(design(r,model['method']),model['coefficients'])))*model.get('hardware_factors',{}).get(r.get('hardware'),1.)


def fit_models(run_dir,data):
    comparisons=[];models=[]
    for method in ('constant','linear','hinge_interactions'):
        for correction in (False,True):
            m=fit_service(data['samples'],method)
            if not correction:m['hardware_factors']={}
            errors={}
            for role in ('fit','development_validation'):
                rr=[r for r in data['samples'] if r['role']==role]
                # Macro averaging by episode prevents large batches from dominating.
                episode_errors=defaultdict(list)
                for r in rr:episode_errors[r['episode_id']].append(abs(predict(m,r)-r['ms'])/r['ms'])
                errors[role]=float(np.mean([np.mean(v) for v in episode_errors.values()])) if rr else None
            comparisons.append(dict(method=method,hardware_correction=correction,errors=errors))
            models.append(m)
    selected=int(np.argmin([r['errors']['development_validation'] for r in comparisons]))
    service=models[selected]
    mf=[r for r in data['migrations'] if r['role']=='fit']
    def fit_component(rr,key):
        usable=[r for r in rr if r.get(key) is not None]
        if not usable:return None
        co=nnls(np.array([[1.,r['blocks']] for r in usable]),np.array([r[key] for r in usable]))[0]
        return dict(intercept_ms=float(co[0]),ms_per_block=float(co[1]),n=len(usable))
    migration={key:fit_component(mf,key) for key in ('ms','pause_ms','transfer_ms')}
    migration['directions']={}
    for src,dst in sorted({(r['hardware_src'],r['hardware_dst']) for r in mf}):
        rr=[r for r in mf if (r['hardware_src'],r['hardware_dst'])==(src,dst)]
        if len(rr)>=8:migration['directions'][src+'>'+dst]=fit_component(rr,'ms')
    pre=[r['ms'] for r in data['prefill'] if r['role']=='fit']
    frame=Counter()
    for r in data['episodes']:
        if r['role']=='fit':frame.update({int(k):v for k,v in r['frame_delta_counts'].items() if int(k)>1})
    engine_candidates=[];engine_models=[]
    for method in ('constant','linear'):
        fit=[r for r in data['engines'] if r['role']=='fit' and not r['prefills']]
        co=nnls(np.array([design(r,method) for r in fit]),np.array([r['ms'] for r in fit]))[0]
        errors=defaultdict(list)
        for r in data['engines']:
            if r['role']=='development_validation' and not r['prefills']:
                y=r['ms']
                errors[r['episode_id']].append(abs(float(np.dot(design(r,method),co))-y)/y)
        engine_candidates.append(dict(method=method,development_macro_relative_mae=float(np.mean([np.mean(v) for v in errors.values()]))))
        engine_models.append(dict(method=method,coefficients=co.tolist(),hardware_factors={},min_ms=5.))
    engine_selected=int(np.argmin([r['development_macro_relative_mae'] for r in engine_candidates]))
    frozen=dict(schema='mig-ppo-v2-shared-physical-v2',service=service,engine=engine_models[engine_selected],engine_target='executor_wall_ms; sampled flush-boundary engine_step_wall_ms retained for audit, not charged on every token',prefill_ms=float(np.median(pre)),
        client_frame_tokens=frame.most_common(1)[0][0],migration=migration,capacity=8565,
        decision_interval_sec=1.5,migration_cooldown_sec=0.,episode_horizon_sec=200.,
        fit_groups=data['fit_groups'],development_groups=data['development_groups'],independent_validation_pending=True,
        no_policy_or_environment_features=True,
        support=dict(service_samples=len(data['samples']),engine_samples=len(data['engines']),prefill_samples=len(pre),
                     migration_fit_samples=len(mf),measured_pause_samples=sum(r['pause_ms'] is not None for r in mf)))
    save(run_dir/'method_comparison.json',dict(candidates=comparisons,selected=selected,engine_candidates=engine_candidates,engine_selected=engine_selected,
         selection_metric='macro episode frame-level relative MAE on previously reviewed development traces; not independent'))
    save(run_dir/'physical_model.json',frozen)
    save(run_dir/'reward_definition.json',read(OLD/'reward_frozen.json')['parameters'])
    return frozen


def adapter_model(model, episode):
    m=copy.deepcopy(model)
    m['hardware_by_logical_instance']=episode['hardware']
    m['instance_ids']=[r[0] for r in read(Path(episode['path'])/'episode.json')['frozen_mapping']]
    m['capacity_by_logical_instance']=[s['total_blocks'] for s in rows(Path(episode['path'])/'decisions.jsonl')[0]['snapshot']['instances']]
    component=m['migration']['ms']
    m['migration_intercept_ms']=component['intercept_ms'];m['migration_ms_per_block']=component['ms_per_block']
    if m['migration'].get('pause_ms'):
        m['migration_pause_components']={'pause_stage_ms':m['migration']['pause_ms']}
    m['logical_migration_directions']={}
    for i,src in enumerate(episode['hardware']):
        for j,dst in enumerate(episode['hardware']):
            co=m['migration']['directions'].get(src+'>'+dst)
            if co:m['logical_migration_directions'][f'{i}>{j}']=co
    return m


def baseline_model(environment):
    p=OLD/'calibration'/environment/'model.json'
    if p.exists():return read(p)
    # HOT1-H was never fitted: use the same shared high-load prior as HOT2-H,
    # explicitly marked in every baseline record; no invented "before" result.
    m=read(OLD/'calibration/HOT2-H/model.json')
    m['baseline_note']='HOT1-H had no old model; unmodified old HOT2-H prior'
    return m


def request_metrics(ep):
    from rl_scheduler.full_completion import summarize_values
    vals=summarize_values([r['client_e2e_ms'] for r in ep['requests']])
    return {'mean':vals['mean'],'p95':vals['p95'],'C':vals['mean']+vals['cvar95']}


def d_curve(requests,origin,edges,reward):
    total=np.zeros(len(edges)-1)
    def primitive(age,ref):
        age=np.maximum(0,age)
        return np.where(age<=2*ref,age*age/(2*ref),2*age-2*ref)
    def add(a,b,ref,weight):
        lo=np.maximum(edges[:-1],a);hi=np.minimum(edges[1:],b)
        valid=hi>lo
        total[valid]+=weight*(primitive(hi[valid]-a,ref)-primitive(lo[valid]-a,ref))/reward.n_ref
    for r in requests:
        sent=(r['sent_ns']-origin)/1e9;end=(r['client_terminal_ns']-origin)/1e9
        first=r.get('client_first_ns')
        add(sent,end if first is None else (first-origin)/1e9,reward.ttft_ref_sec,1.)
        ts=[(at-origin)/1e9 for at,n in r['token_increments'] if n>0]
        for a,b in zip(ts,[*ts[1:],end]):add(a,b,reward.token_gap_ref_sec,reward.kappa)
    return total/np.diff(edges)


def curves(ep,decs,origin,edges,reward):
    dt=np.diff(edges);mid=(edges[:-1]+edges[1:])/2
    times=np.array([(d.get('observation_decision_ns',d['monotonic_ns'])-origin)/1e9 for d in decs])
    end=max(r['client_terminal_ns'] for r in ep['requests'])/1e9-origin/1e9
    result={k:[] for k in ('waiting','kv','Q','H','B_star')}
    for t in mid:
        idx=int(np.searchsorted(times,t,side='right')-1)
        if idx<0 or t>=end:
            result['waiting'].append([0.]*4);result['kv'].append([0.]*4)
            for k in ('Q','H','B_star'):result[k].append(0.)
            continue
        snap=decs[idx]['snapshot'];states=snap['instances']
        result['waiting'].append([s['waiting'] for s in states])
        result['kv'].append([(s['used_blocks']+s.get('reserved_blocks_not_in_used',0))/s['total_blocks'] for s in states])
        ps=reward.pressures(snap)
        for k in ('Q','H','B_star'):result[k].append(ps[k])
    result={k:np.array(v) for k,v in result.items()}
    result['D']=d_curve(ep['requests'],origin,edges,reward)
    result['outstanding']=np.array([sum(r['sent_ns']<=origin+t*1e9<r['client_terminal_ns'] for r in ep['requests']) for t in mid])
    return result


def compare_curves(real,sim,dt):
    result={}
    for key,a in real.items():
        b=sim[key]
        if a.ndim==1:
            area=float(np.dot(a,dt));err=float(np.dot(np.abs(a-b),dt));mae=err/sum(dt)
            result[key]=dict(real_area=area,absolute_error_area=err,relative_area_error=err/area if area>1e-9 else None,mae=mae)
        else:
            area=np.sum(a*dt[:,None],axis=0);err=np.sum(np.abs(a-b)*dt[:,None],axis=0)
            result[key]=dict(mae=float(np.mean(err/sum(dt))),max_instance_mae=float(max(err/sum(dt))),
                             per_instance_mae=(err/sum(dt)).tolist(),
                             relative_area_error=float(np.sum(err)/np.sum(area)) if np.sum(area)>1e-9 else None)
    result['D']['passed']=result['D']['relative_area_error']<=.2 if result['D']['real_area']/sum(dt)>.05 else result['D']['mae']<=.05
    result['outstanding']['passed']=(result['outstanding']['relative_area_error'] or 0)<=.2
    result['waiting']['passed']=all((e<=1.) if a<=sum(dt) else (e*sum(dt)/a<=.2)
                                  for e,a in zip(result['waiting']['per_instance_mae'],np.sum(real['waiting']*dt[:,None],axis=0)))
    result['kv']['passed']=result['kv']['max_instance_mae']<=.05
    for k in ('Q','H','B_star'):result[k]['passed']=result[k]['mae']<=.05
    return result


def migration_metrics(data,eid,sim_events):
    real=[r for r in data['migrations'] if r['episode_id']==eid]
    simulated=[r for r in sim_events if r['status']=='success']
    result=dict(real_count=len(real),sim_count=len(simulated),count_passed=abs(len(real)-len(simulated))<=max(5,.2*len(real)))
    for label,key in [('duration','ms'),('pause','pause_ms')]:
        rv=[r[key] for r in real if r.get(key) is not None]
        sv=[r['cost_ms' if label=='duration' else 'pause_stage_ms'] for r in simulated]
        if not rv or not sv:
            result[label]=dict(median_error=None,p95_error=None,passed=None,n_real=len(rv),n_sim=len(sv));continue
        med=abs(float(np.median(sv))-float(np.median(rv)))/float(np.median(rv))
        p95=abs(float(np.percentile(sv,95))-float(np.percentile(rv,95)))/float(np.percentile(rv,95))
        result[label]=dict(median_error=med,p95_error=p95,passed=med<=.2 and (len(rv)<5 or p95<=.3),n_real=len(rv),n_sim=len(sv))
    return result


def replay_schedule(data,episode,model,predict_cost):
    events=[]
    for r in data['migrations']:
        if r['episode_id']!=episode['episode_id']:continue
        duration=r['ms'];pause=r['pause_ms']
        if predict_cost:
            co=model.get('logical_migration_directions',{}).get(f"{r['source_index']}>{r['target_index']}")
            co=co or model['migration']['ms']
            duration=co['intercept_ms']+co['ms_per_block']*r['blocks']
            co=model['migration'].get('pause_ms')
            pause=co['intercept_ms']+co['ms_per_block']*r['blocks'] if co else duration
        if r['request_id'] is None:continue
        instance_ids=model.get('instance_ids',[f'instance_{i}' for i in range(4)])
        events.append(dict(request_id=r['request_id'],source_instance=instance_ids[r['source_index']],
            target_instance=instance_ids[r['target_index']],migration_time_sec=r['time_sec'],blocks=r['blocks'],cost_ms=duration,
            pause_stage_ms=min(duration,pause if pause is not None else duration),status='success',migration_id=r['migration_id']))
    return events


def validate(run_dir,data,model,only=None):
    from rl_scheduler.q15b_sim import Q15BCohortSim
    from rl_scheduler.cohort_reward import MigPPOV2Reward
    reward=MigPPOV2Reward.from_state_dict(read(Path(run_dir)/'reward_definition.json'))
    episodes={r['episode_id']:r for r in data['episodes']}
    results=[];edges=np.r_[np.arange(0,200,1.5),200.];dt=np.diff(edges)
    for pair in data['pairs']:
        if only and pair['pair_id'] not in only:continue
        record={k:v for k,v in pair.items() if k!='episodes'};record['versions']={}
        for version in ('before','after'):
            policies={}
            for policy,eid in pair['episodes'].items():
                episode=episodes[eid];d=Path(episode['path']);real_ep=read(d/'episode.json');real_decs=rows(d/'decisions.jsonl')
                real_curves=curves(real_ep,real_decs,episode['origin_ns'],edges,reward)
                m=baseline_model(pair['environment']) if version=='before' else adapter_model(model,episode)
                modes=[policy]
                if version=='after' and policy=='Balanced':modes+=['replay_predicted_cost','replay_observed_cost']
                for mode in modes:
                    out=Path(run_dir)/'simulations'/pair['pair_id']/version/mode.lower()
                    marker=out/'validation.json'
                    if marker.exists():
                        cached=read(marker)
                        if version=='after' and cached.get('frozen_physical_model_sha256')!=sha(Path(run_dir)/'physical_model.json'):
                            raise ValueError('cached validation uses another physical model; choose a new run directory')
                        if (out/'model_input.json').exists() and read(out/'model_input.json')!=m:
                            raise ValueError('cached simulation initial conditions differ; choose a new run directory')
                        if read(out/'reward.json')['definition']!=reward.state_dict():
                            raise ValueError('cached simulation uses another reward definition')
                        policies[mode]=cached;continue
                    try:
                        trace=read(d/'trace.json')
                        if 'replay' in mode:
                            # Native replay fixes observed initial routes to separate routing errors.
                            actual={r['request_id']:r for r in real_ep['requests']}
                            trace=copy.deepcopy(trace)
                            for t in trace:
                                route=actual[t['request_id']]['initial_instance_index']
                                if route is None:
                                    instance=real_ep['observed_initial_routes'][t['request_id']]
                                    route=[r[0] for r in real_ep['frozen_mapping']].index(instance)
                                t['initial_instance_index']=route
                                t['routing_mode']='fixed'
                        schedule=replay_schedule(data,episode,m,mode=='replay_predicted_cost') if 'replay' in mode else None
                        if (out/'episode.json').exists():
                            summary=read(out/'episode.json');decs=rows(out/'decisions.jsonl');events=rows(out/'migrations.jsonl')
                        else:
                            sim=Q15BCohortSim(m,trace,seed=26110699,output_dir=out)
                            save(out/'model_input.json',m)
                            summary,decs,events=sim.run(policy if 'replay' not in mode else 'Stay',T_ref=1.,
                                replay_actions=schedule,reward_definition=reward,episode_horizon_sec=200.)
                        sm=request_metrics(summary);rm=request_metrics(real_ep)
                        errors={k:abs(sm[k]-rm[k])/rm[k] for k in rm}
                        cs=compare_curves(real_curves,curves(summary,decs,0,edges,reward),dt)
                        mig=migration_metrics(data,eid,events)
                        # Reconstruct both rewards through the shared V2 contract.
                        times=[x.get('observation_decision_ns',x['monotonic_ns']) for x in real_decs]
                        end=max(r['client_terminal_ns'] for r in real_ep['requests'])
                        eligible=[(t,d) for t,d in zip(times,real_decs) if t<end]
                        rr=reward.reconstruct(real_ep,[t for t,_ in eligible],[d for _,d in eligible])
                        sr=read(out/'reward.json')
                        reward_ok=all(abs(r['return']-math.fsum(w['reward'] for w in r['windows']))<1e-8 and r['terminal_potential']==0 for r in (rr,sr))
                        badmask=sum(len(d['action_mask'])!=13 or not d['action_mask'][d['executed_action']] for d in decs)
                        val=dict(status='complete',mode=mode,real=rm,simulated=sm,errors=errors,curves=cs,migration=mig,
                            reward_identity_passed=reward_ok,mask_invalid_count=badmask,
                            request_identity_passed={r['request_id'] for r in summary['requests']}=={r['request_id'] for r in real_ep['requests']},
                            output_token_identity_passed=sum(r['actual_output_tokens'] for r in summary['requests'])==episode['output_tokens'],
                            simulated_routing=summary['routing_mode'],real_episode=eid,
                            replay_results=read(out/'replay_results.json') if schedule is not None else None,
                            frozen_physical_model_sha256=sha(Path(run_dir)/'physical_model.json') if version=='after' else None)
                        val['passed']=all(v<=.2 for v in errors.values()) and all(c['passed'] for c in cs.values()) and mig['count_passed'] and (mig['duration']['passed'] is not False) and (mig['pause']['passed'] is not False) and reward_ok and badmask==0
                        if schedule is not None:val['passed']=val['passed'] and len(val['replay_results'])==len(schedule) and all(x['result']=='started' for x in val['replay_results'])
                        save(out/'state_curves.json',dict(edges_sec=edges.tolist(),real={k:v.tolist() for k,v in real_curves.items()},simulated={k:v.tolist() for k,v in curves(summary,decs,0,edges,reward).items()}))
                        save(marker,val)
                    except Exception as exc:
                        val=dict(status='failed',error=type(exc).__name__+': '+str(exc),passed=False)
                        import traceback
                        save(out/'failure.json',dict(**val,traceback=traceback.format_exc()))
                    policies[mode]=val
                    # The full event stream remains reproducible but compressed.
                    if out.exists():
                        for file in out.glob('*.jsonl'):
                            with file.open('rb') as src,gzip.open(str(file)+'.gz','wb') as dest:
                                import shutil
                                shutil.copyfileobj(src,dest)
                            file.unlink()
                    print('validated',pair['pair_id'],version,mode,val['status'],val.get('errors'),flush=True)
            stay=policies.get('Stay',{});bal=policies.get('Balanced',{})
            rec=dict(policies=policies)
            if stay.get('status')=='complete' and bal.get('status')=='complete':
                real_gain=1-bal['real']['C']/stay['real']['C'];sim_gain=1-bal['simulated']['C']/stay['simulated']['C']
                sign=abs(real_gain)<.02 or sim_gain*real_gain>0
                rec.update(real_gain=real_gain,simulated_gain=sim_gain,gain_error_pp=100*abs(sim_gain-real_gain),
                           direction_consistent=sign,passed=stay['passed'] and bal['passed'] and sign and abs(sim_gain-real_gain)<=.1)
            else:rec['passed']=False
            if version=='after':
                rec['closed_loop_contract_passed']=rec['passed']
                replay_ok=all(policies.get(mode,{}).get('passed',False)
                              for mode in ('replay_predicted_cost','replay_observed_cost'))
                rec['all_modes_passed']=rec['passed'] and replay_ok
            record['versions'][version]=rec
        results.append(record)
        result_path=Path(run_dir)/'validation_results.json' if only is None else Path(run_dir)/'pair_results'/((only[0] if len(only)==1 else 'subset')+'.json')
        save(result_path,dict(pairs=results,independent_validation_pending=True,
            alignment='Own frozen scheduled experiment origin; full fixed 200 s, terminal tails padded zero. 1.5 s ZOH states; exact D integrals.',
            closed_loop_only_aggregate=True))
    return results


def _worker(args):
    run_dir,pair_id=args
    run_dir=Path(run_dir)
    return validate(run_dir,read(run_dir/'data_audit.json'),read(run_dir/'physical_model.json'),only=[pair_id])[0]


def run(run_dir,stage='all',workers=4):
    """No external/GPU execution, even in the default all stage."""
    run_dir=Path(run_dir).resolve()
    if any(run_dir==source or source in run_dir.parents for source in (OLD,FOLLOWUP)):
        raise ValueError('source experiment directories are immutable; use a new refit directory')
    if (run_dir/'model_registry.json').exists() and stage in ('all','audit','fit'):
        raise FileExistsError('frozen refit is write-once; use a new run directory')
    if workers<1:raise ValueError('positive CPU worker count required')
    run_dir.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    if stage in ('all','audit'):
        extract(run_dir)
        write_physical_audit(run_dir)
    if stage in ('all','fit'):fit_models(run_dir,read(run_dir/'data_audit.json'))
    if stage in ('all','validate'):
        from concurrent.futures import ProcessPoolExecutor
        data=read(run_dir/'data_audit.json');runtime_hashes=physical_runtime_hashes()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results=list(pool.map(_worker,[(str(run_dir),p['pair_id']) for p in data['pairs']]))
        if physical_runtime_hashes()!=runtime_hashes:raise RuntimeError('physical runtime changed during validation')
        save(run_dir/'validation_results.json',dict(pairs=results,independent_validation_pending=True,physical_runtime_hashes=runtime_hashes,
             alignment='Frozen scheduled origin; full fixed 200 s; exact client D integrals; 1.5 s ZOH state samples',
             closed_loop_only_aggregate=True))
    if stage in ('all','report'):report(run_dir)
    result=dict(stage=stage,cpu_wall_sec=time.monotonic()-start,run_dir=str(run_dir),gpu_runs=0,ppo_training_runs=0)
    save(run_dir/('execution_'+stage+'.json'),result)
    return result


def physical_runtime_hashes():
    return {p:sha(ROOT/p) for p in ('rl_scheduler/env.py','rl_scheduler/q15b_sim.py',
        'rl_scheduler/service_model.py','rl_scheduler/cohort_reward.py','rl_scheduler/real_observation.py','rl_scheduler/ppo_runtime.py')}


def report(run_dir):
    run_dir=Path(run_dir);data=read(run_dir/'data_audit.json');model=read(run_dir/'physical_model.json')
    validation=read(run_dir/'validation_results.json')
    if validation.get('physical_runtime_hashes')!=physical_runtime_hashes():
        raise ValueError('validation runtime hashes missing or changed; rerun CPU validation')
    pairs=validation['pairs']
    for pair in pairs:
        aft=pair['versions']['after']
        aft['closed_loop_contract_passed']=aft['passed']
        aft['all_modes_passed']=aft['passed'] and all(aft['policies'].get(mode,{}).get('passed',False)
            for mode in ('replay_predicted_cost','replay_observed_cost'))
    validation=read(run_dir/'validation_results.json');validation['pairs']=pairs
    save(run_dir/'validation_results.json',validation)
    aggregates={};details=[]
    for version in ('before','after'):
        valid=[];pair_valid=[]
        for pair in pairs:
            v=pair['versions'][version]
            if v.get('gain_error_pp') is not None:pair_valid.append(v)
            for policy in ('Stay','Balanced'):
                x=v['policies'][policy]
                if x.get('status')=='complete':valid.append(x)
        result={}
        def stat(values,passed=None,unit='fraction'):
            vv=[v for v in values if v is not None]
            return dict(mean=float(np.mean(vv)) if vv else None,n=len(vv),passed=sum(passed) if passed is not None else None,unit=unit)
        for k in ('mean','p95','C'):
            result[k]=stat([x['errors'][k] for x in valid],[x['errors'][k]<=.2 for x in valid])
        result['D']=stat([x['curves']['D']['relative_area_error'] for x in valid],[x['curves']['D']['passed'] for x in valid])
        for k in ('outstanding','waiting','kv','Q','H','B_star'):
            metric='relative_area_error' if k=='outstanding' else 'mae'
            result[k]=stat([x['curves'][k][metric] for x in valid],[x['curves'][k]['passed'] for x in valid],
                           'requests' if k=='waiting' else 'fraction')
        for component in ('duration','pause'):
            for q in ('median','p95'):
                xx=[x['migration'][component] for x in valid if x['mode']=='Balanced' and x['migration'][component][q+'_error'] is not None]
                if q=='p95':xx=[x for x in xx if x['n_real']>=5 and x['n_sim']>=5]
                result['migration_'+component+'_'+q]=stat([x[q+'_error'] for x in xx],
                    [x[q+'_error']<=(.2 if q=='median' else .3) for x in xx])
        result['gain']=stat([v['gain_error_pp'] for v in pair_valid],[v['gain_error_pp']<=10 and v['direction_consistent'] for v in pair_valid],unit='percentage_points')
        strong=[v for v in pair_valid if abs(v['real_gain'])>=.02]
        result['direction']=dict(passed=sum(v['direction_consistent'] for v in strong),n=len(strong),near_zero_count=len(pair_valid)-len(strong),near_zero_band=.02)
        result['closed_loop_pair_passed']=sum(v['passed'] for v in pair_valid)
        result['full_pair_passed']=sum(v.get('all_modes_passed',v['passed']) for v in pair_valid);result['pair_count']=len(pair_valid)
        aggregates[version]=result
    qualified=[]
    for pair in pairs:
        aft=pair['versions']['after'];entry=dict(pair_id=pair['pair_id'],environment=pair['environment'],role=pair['role'],
            real_gain=aft.get('real_gain'),simulated_gain=aft.get('simulated_gain'),gain_error_pp=aft.get('gain_error_pp'),
            direction_consistent=aft.get('direction_consistent'),closed_loop_contract_passed=aft['passed'],
            development_contract_passed=aft.get('all_modes_passed',False),
            sim_pretraining_ready=False,independent_validation_pending=True)
        for policy in ('Stay','Balanced'):
            x=aft['policies'][policy]
            entry[policy]=dict(status=x['status'],C_error=x.get('errors',{}).get('C'),
                D_error=x.get('curves',{}).get('D',{}).get('relative_area_error'))
        if aft.get('all_modes_passed',False) and pair['environment']!='NATIVE':qualified.append(pair['pair_id'])
        details.append(entry)
    save(run_dir/'accuracy_summary.json',dict(statistic='Macro mean over complete closed-loop policy episodes; gain over paired closed loops. Replay excluded. Full fixed 200 s. D=integral absolute curve difference / real integral; KV/Waiting=time MAE.',
        aggregates=aggregates,episodes=len(data['episodes']),pairs=len(pairs),
        fit_pairs=sum(p['role']=='fit' for p in pairs),development_pairs=sum(p['role']=='development_validation' for p in pairs),
        independent_validation_pending=True,development_qualified_pairs=qualified,per_pair=details))
    from datetime import datetime,timezone
    frozen_environments=[]
    for pair in data['pairs']:
        initial=next(e for e in data['episodes'] if e['episode_id']==pair['episodes']['Stay'])
        path=run_dir/'frozen_environments'/pair['pair_id']/'model.json'
        save(path,adapter_model(model,initial))
        aft=next(p for p in pairs if p['pair_id']==pair['pair_id'])['versions']['after']
        frozen_environments.append(dict(environment=pair['environment'],pair_id=pair['pair_id'],
            model_path=str(path.relative_to(run_dir)),sha256=sha(path),data_role=pair['role'],
            native_test_only=pair['environment']=='NATIVE',
            closed_loop_contract_passed=aft['closed_loop_contract_passed'],all_modes_passed=aft['all_modes_passed'],
            sim_pretraining_ready=False))
    registry=dict(schema='mig-ppo-v2-physical-refit-freeze-v2',frozen_utc=datetime.now(timezone.utc).isoformat(),physical_model_sha256=sha(run_dir/'physical_model.json'),
        reward_definition_sha256=sha(run_dir/'reward_definition.json'),
        data_audit_sha256=sha(run_dir/'data_audit.json'),validation_sha256=sha(run_dir/'validation_results.json'),
        code_hashes={str(p):sha(ROOT/p) for p in map(Path, ['rl_scheduler/env.py','rl_scheduler/q15b_sim.py',
                    'rl_scheduler/service_model.py','rl_scheduler/real_observation.py','rl_scheduler/cohort_reward.py',
                    'rl_scheduler/ppo_runtime.py','benchmark/mig_ppo_v2_refit.py','benchmark/run_ppo_sim.py'])},
        environment_models=frozen_environments,
        native_role='test_only; never PPO pretraining',independent_validation_pending=True,formal_ppo_ready_environments=[],
        development_qualified_pairs=qualified,models_frozen=True,
        limitations=['No unseen independent real validation; all validation here is development.',
            'Fixed prompt 128/output 1536; prefill prompt-length slope not identified.',
            'Low queue and KV pressure; capacity exhaustion/preemption extrapolation is unvalidated.',
            'Observed whole-episode speed residuals have no identified causal attribution; no final-cost or gain compensation.',
            'Scalar directed link NNLS and pooled pause NNLS do not reproduce every transfer tail.',
            'No observed failures is not a failure probability estimate.'])
    save(run_dir/'model_registry.json',registry)
    refs={}
    for root,names in [(OLD,['report.md','model_registry.json','source_index.json','sim_validation_summary.json','acceptance_detail.csv','simulation_curve_audit.json','reward_reconstruction.json']),
                       (FOLLOWUP,['followup_report.md'])]:
        for name in names:
            p=root/name
            # Parse original acceptance artifacts, not only filesystem names.
            content=read(p) if p.suffix=='.json' else list(csv.DictReader(p.open())) if p.suffix=='.csv' else p.read_text()
            refs[str(p.relative_to(ROOT))]=dict(sha256=sha(p),records=len(content))
    save(run_dir/'source_index.json',dict(sources=data['episodes'],excluded=data['excluded'],reference_artifacts=refs,
         fit_groups=data['fit_groups'],development_groups=data['development_groups'],
         group_hash_rule='Canonical request id, prompt, arrival, input and output; excludes routes/hotspot/candidate.',
         all_complete_gain_signs_included=True,repeated_stay_is_not_independent_pair=True,
         independent_validation_pending=True,previous_validation_reclassified=True))
    residuals=[]
    for e in data['episodes']:
        ss=[r for r in data['samples'] if r['episode_id']==e['episode_id']]
        residuals.append(dict(episode_id=e['episode_id'],environment=e['environment'],policy=e['policy'],role=e['role'],
            n=len(ss),median_effective_step_ms=float(np.median([r['ms'] for r in ss])) if ss else None,
            median_observed_over_predicted=float(np.median([r['ms']/predict(model['service'],r) for r in ss])) if ss else None))
    save(run_dir/'service_residual_audit.json',dict(episodes=residuals,
        interpretation='HOT1-H Stay and NATIVE1.25 Balanced have large whole-episode speed residuals. Current load/context features cannot identify their cause. Correlation from one paired run cannot establish a counterfactual migration gain; no per-environment or policy speed adjustment is applied.'))
    methods=[('Prefill','admission→engine-first 固定中位数','128-token prompt；排队由事件引擎计算'),
             ('Decode','引擎 executor-wall NNLS + 稳健有效 cadence NNLS','当前 batch、上下文；取 max 推进一次；64-token flush'),
             ('KV/队列','离散事件、逐 Token 分配与 FCFS','prompt+已生成 tokens、容量、预留/释放'),
             ('迁移','NNLS 总时长/暂停；有样本的物理方向修正','copied blocks、GPU→GPU；不加固定收益')]
    def pc(x):return 'N/A' if x is None else f'{100*x:.2f}%'
    lines=['# Mig-PPO V2 CPU 物理模型优化', '',
        f'共 {len(pairs)} 个完整配对（{sum(p["role"]=="fit" for p in pairs)} fit、{sum(p["role"]=="development_validation" for p in pairs)} development），另有未配对物理样本。全部为已见开发数据；independent_validation_pending。',
        '','| 模块 | 最终方法 | 输入 |','|---|---|---|']
    lines += ['| '+' | '.join(row)+' |' for row in methods]
    lines += ['',f'共享 cadence 系数：{model["service"]["coefficients"]}；物理 GPU 速率修正未通过开发选择，未采用。所有迁移增益来自状态推进。',
        '','误差为同一批实际 CPU 闭环的宏平均，回放不计入总览。D 使用完整 200 秒精确积分；状态按各自冻结实验起点对齐，1.5 秒保持采样，结束后补零。','',
        '| 指标 | 优化前 | 优化后 | 优化后通过 / n |','|---|---:|---:|---:|']
    labels=[('Mean E2E','mean'),('P95 E2E','p95'),('C','C'),('D pressure','D'),('Outstanding','outstanding'),
            ('Waiting MAE/请求','waiting'),('KV MAE','kv'),('Q MAE','Q'),('H MAE','H'),('B* MAE','B_star'),
            ('迁移中位耗时','migration_duration_median'),('迁移P95耗时','migration_duration_p95'),
            ('暂停中位耗时','migration_pause_median'),('暂停P95耗时','migration_pause_p95'),('gain_C / pp','gain')]
    for label,key in labels:
        b=aggregates['before'][key];a=aggregates['after'][key]
        fmt=(lambda x:'N/A' if x is None else f'{x:.4f}') if key in ['waiting','kv','Q','H','B_star'] else (lambda x:'N/A' if x is None else f'{x:.2f}') if key=='gain' else pc
        lines.append(f'| {label} | {fmt(b["mean"])} | {fmt(a["mean"])} | {a["passed"]}/{a["n"]} |')
    lines += ['','| 环境/trace | 数据用途 | Stay C误差 | Balanced C误差 | 真gain | 仿真gain | D最差误差 | 全门 |',
              '|---|---|---:|---:|---:|---:|---:|---|']
    for p in pairs:
        a=p['versions']['after'];st=a['policies']['Stay'];bl=a['policies']['Balanced']
        if st['status']=='complete' and bl['status']=='complete':
            lines.append(f'| {p["pair_id"]} | {p["role"]} | {pc(st["errors"]["C"])} | {pc(bl["errors"]["C"])} | {pc(a["real_gain"])} | {pc(a["simulated_gain"])} | {pc(max(st["curves"]["D"]["relative_area_error"],bl["curves"]["D"]["relative_area_error"]))} | {"通过" if a["all_modes_passed"] else "未通过"} |')
        else:lines.append(f'| {p["pair_id"]} | {p["role"]} | N/A | N/A | N/A | N/A | N/A | 仿真未完成 |')
    lines += ['','收益绝对值小于 2% 的配对按接近零处理，单次实验无法判断稳定符号。真实动作回放（预测成本与观测成本两种）分列于 validation_results.json，不冒充闭环结果。',
              '',f'开发数据全门通过配对：{qualified}。正式 PPO ready 列表为空；尚缺未参与开发的新真实独立验证。NATIVE 始终仅作测试。',
              '', '相关单元测试见 test_results.txt；未启动 GPU 实验或 PPO 训练。']
    (run_dir/'report.md').write_text('\n'.join(lines)+'\n')
    flat=[]
    for p in pairs:
        for version,v in p['versions'].items():
            for mode,x in v['policies'].items():
                row=dict(pair_id=p['pair_id'],environment=p['environment'],data_role=p['role'],
                         version=version,mode=mode,status=x['status'],passed=x['passed'])
                if x['status']=='complete':
                    row.update({k+'_error':v for k,v in x['errors'].items()})
                    row.update({k+'_mae':v['mae'] for k,v in x['curves'].items()})
                    row.update(D_area_error=x['curves']['D']['relative_area_error'],
                        migrations_real=x['migration']['real_count'],migrations_sim=x['migration']['sim_count'])
                flat.append(row)
    with (run_dir/'acceptance_detail.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in flat))))
        writer.writeheader();writer.writerows(flat)
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['svg.hashsalt']='mig-ppo-v2-physical-refit-v2'
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        labels=['Mean E2E','P95 E2E','C','D pressure'];keys=['mean','p95','C','D'];idx=np.arange(4)
        for i,version in enumerate(('before','after')):
            ax[0].bar(idx+(i-.5)*.35,[100*aggregates[version][k]['mean'] for k in keys],width=.35,label=version)
        ax[0].set_xticks(idx,labels);ax[0].set_ylabel('Macro relative error (%)');ax[0].legend()
        for version in ('before','after'):
            rr=[p['versions'][version] for p in pairs if 'real_gain' in p['versions'][version]]
            ax[1].scatter([100*r['real_gain'] for r in rr],[100*r['simulated_gain'] for r in rr],label=version,alpha=.75)
        ax[1].plot([-75,30],[-75,30],'k--',linewidth=1);ax[1].axhline(0,color='gray',linewidth=.5);ax[1].axvline(0,color='gray',linewidth=.5)
        ax[1].set_xlabel('Real gain_C (%)');ax[1].set_ylabel('Closed-loop simulated gain_C (%)');ax[1].legend()
        fig.savefig(run_dir/'accuracy_overview.png',dpi=160)
        svg=run_dir/'accuracy_overview.svg';fig.savefig(svg,metadata={'Date':None});plt.close(fig)
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    except ImportError:
        pass
    return aggregates


def write_physical_audit(run_dir):
    """Request lifecycle and local migration diagnostics, separate from fit targets."""
    run_dir=Path(run_dir);data=read(run_dir/'data_audit.json');requests_out=[];neighborhoods=[]
    engine_counters=Counter();max_kv=0.;frame_counts=Counter()
    for e in data['episodes']:
        directory=Path(e['path']);ep=read(directory/'episode.json');decs=rows(directory/'decisions.jsonl')
        times=[d.get('observation_decision_ns',d['monotonic_ns']) for d in decs]
        events=rows(directory/'events.jsonl');admission={};engine_first={}
        for event in events:
            v=event.get('data') or {};kind=v.get('event')
            if kind=='request_engine_admitted':admission.setdefault(event['request_id'],v['monotonic_ns'])
            if event.get('kind')=='engine_first':engine_first.setdefault(event['request_id'],event['occurred_ns'])
            if kind=='engine_iteration_completed':engine_counters['sampled_preemptions']+=int(v.get('num_preempted') or 0)
            if 'preempt' in str(kind).lower():engine_counters['preemption_related_events']+=1
        for d in decs:
            for s in d['snapshot']['instances']:
                max_kv=max(max_kv,(s['used_blocks']+s.get('reserved_blocks_not_in_used',0))/s['total_blocks'])
        for r in ep['requests']:
            frame_counts.update(n for _,n in r['token_increments'])
            rid=r['request_id'];route=ep.get('observed_initial_routes',{}).get(rid)
            rel=lambda at:None if at is None else (at-e['origin_ns'])/1e9
            requests_out.append(dict(episode_id=e['episode_id'],data_role=e['role'],request_id=rid,
                arrival_sec=rel(r['scheduled_ns']),sent_sec=rel(r['sent_ns']),admission_sec=rel(admission.get(rid)),
                engine_first_sec=rel(engine_first.get(rid)),client_first_sec=rel(r['client_first_ns']),
                client_terminal_sec=rel(r['client_terminal_ns']),client_e2e_ms=r['client_e2e_ms'],
                output_tokens=r['actual_output_tokens'],client_frame_count=len(r['token_increments']),observed_initial_instance=route))
        for m in data['migrations']:
            if m['episode_id']!=e['episode_id']:continue
            ns=e['origin_ns']+int(m['time_sec']*1e9)
            pre=decs[max(0,np.searchsorted(times,ns,side='right')-1)]['snapshot']['instances']
            post=decs[min(len(decs)-1,np.searchsorted(times,ns+3_000_000_000))]['snapshot']['instances']
            row=dict(episode_id=e['episode_id'],migration_id=m['migration_id'],time_sec=m['time_sec'],
                duration_ms=m['ms'],pause_ms=m['pause_ms'],copy_transfer_ms=m['transfer_ms'],
                copy_excluded_control_setup_ack_ms=m['ms']-m['transfer_ms'] if m['transfer_ms'] is not None else None)
            for label,i in [('source',m['source_index']),('target',m['target_index'])]:
                a,b=pre[i],post[i]
                for key in ('running','waiting','used_blocks'):
                    row[f'{label}_{key}_before']=a[key];row[f'{label}_{key}_after_3s']=b[key]
                dt=(post[i]['measured_ns']-pre[i]['measured_ns'])/1e9
                row[f'{label}_server_tokens_per_sec']=(b['output_tokens_total']-a['output_tokens_total'])/dt if dt>0 else None
            row['cohort_completions_next_3s']=sum(ns<=r['client_terminal_ns']<ns+3_000_000_000 for r in ep['requests'])
            neighborhoods.append(row)
    for name,rr in [('request_lifecycles',requests_out),('migration_neighborhoods',neighborhoods)]:
        with (run_dir/(name+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rr))))
            writer.writeheader();writer.writerows(rr)
    save(run_dir/'physical_audit_summary.json',dict(requests=len(requests_out),successful_migration_events=len(neighborhoods),
        client_frame_delta_counts=dict(frame_counts),sampled_engine_counters=dict(engine_counters),
        max_observed_kv_fraction=max_kv,
        caveats=['Queue/KV exhaustion and recomputation have little or no measured support; passing low-load MAE does not validate overload.',
                 '3-second neighborhoods include new arrivals and overlapping workload changes; they are diagnostic, not causal migration gain estimates.']))
