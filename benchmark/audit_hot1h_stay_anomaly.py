import csv, gzip, json, math, re, statistics
from collections import Counter
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'benchmark/results/hot1h_stay_anomaly_audit_assets'
ASSET.mkdir(parents=True, exist_ok=True)
SPECS={
 'HOT1-H': ('benchmark/results/mig_ppo_v2_followup_20261106T080500Z','HOT1-H_backup_n48_q1p0_s26110523_followup'),
 'HOT2-H': ('benchmark/results/mig_ppo_v2_multienv_20261105T130307Z','HOT2-H_backup_n48_q1p0_s26110523'),
 'HOT3-H': ('benchmark/results/mig_ppo_v2_multienv_20261105T130307Z','HOT3-H_backup_n48_q1p0_s26110523'),
 'HOT4-H': ('benchmark/results/mig_ppo_v2_multienv_20261105T130307Z','HOT4-H_backup_n48_q1p0_s26110523'),
}

def read_json(p): return json.loads(Path(p).read_text())
def read_gz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
def pct(xs,p):
    if not xs: return None
    xs=sorted(float(x) for x in xs)
    h=(len(xs)-1)*p; lo=math.floor(h); hi=math.ceil(h)
    return xs[lo] if lo==hi else xs[lo]*(hi-h)+xs[hi]*(h-lo)
def mean(xs): return statistics.fmean(xs) if xs else None

records={}
for env,(relroot,suffix) in SPECS.items():
    root=ROOT/relroot
    comp=read_json(root/'comparisons'/f'{suffix}.json')
    stay=comp['episodes']['Stay']
    epdir=root/'runs'/stay['episode_id']
    episode=read_json(epdir/'episode.json')
    reqs=read_gz(epdir/'requests.jsonl.gz')
    events=read_gz(epdir/'events.jsonl.gz')
    decisions=read_gz(epdir/'decisions.jsonl.gz')
    trace=read_json(epdir/'trace.json')
    execution=read_json(epdir/'execution_audit.json')
    opportunity=read_json(epdir/'opportunity.json')
    runroot=root
    start=min(r['scheduled_ns'] for r in reqs)
    sends=[r['sent_ns'] for r in reqs if r.get('sent_ns') is not None]
    terminals=[r['client_terminal_ns'] for r in reqs if r.get('client_terminal_ns') is not None]
    admissions={}; manager_routes={}
    # The episode events.gz contains the candidate-linked diagnostic stream,
    # including admission and manager-route timestamps in its nested data.
    for ev in events:
        if ev.get('kind')!='diagnostic': continue
        d=ev.get('data') or {};kind=d.get('event');rid=ev.get('request_id') or ev.get('request_id')
        outer=ev.get('request_id')
        if kind=='request_engine_admitted' and outer and outer not in admissions:
            admissions[outer]={'ns':d.get('monotonic_ns'),'instance_id':d.get('instance_id'),'pid':d.get('pid')}
        elif kind=='real_initial_route' and outer and outer not in manager_routes:
            manager_routes[outer]={'ns':d.get('monotonic_ns'),'instance_id':d.get('instance_id'),'route_source':d.get('route_source')}
    engine_first={}
    for ev in events:
        if ev.get('kind')=='engine_first' and ev.get('request_id'):
            engine_first[ev['request_id']]=ev.get('occurred_ns')
    accepted_ns={}
    for ev in events:
        if ev.get('kind')=='accepted' and ev.get('request_id'):
            accepted_ns[ev['request_id']]=ev.get('occurred_ns')
    # Map logical instance index to Ray GPU ID from frozen route map and startup logs.
    mapping=episode.get('frozen_mapping') or []
    mapping_ids=[entry[0] for entry in mapping]
    pat=re.compile(r"Llumlet pid=(\d+).*?Llumlet\(actor_name=instance_([0-9a-f]+),.*?gpu_ids=\['([^']+)'\]")
    gpu_by_instance={}
    api_log=root/'api_server.log'
    for line in api_log.open(encoding='utf-8',errors='replace'):
        m=pat.search(line)
        if m: gpu_by_instance[m.group(2)]=m.group(3)
    route_counts=Counter(r.get('initial_instance_index') for r in reqs)
    output_counts=Counter(r.get('actual_output_tokens') for r in reqs)
    primary=episode['primary_metrics']
    e2e=primary['client_e2e_ms']; ttft=primary['ttft_ms']; tpot=primary['tpot_ms']
    # Event-linked state samples are binned by second; they are not a uniform polling series.
    diag=[]
    diag_kinds=Counter()
    for ev in events:
        if ev.get('kind')!='diagnostic': continue
        d=ev.get('data') or {}; diag_kinds[d.get('event')]+=1
        iid=d.get('instance_id')
        if iid not in mapping_ids or d.get('monotonic_ns') is None: continue
        i=mapping_ids.index(iid)
        used=d.get('kv_used_blocks'); free=d.get('kv_free_blocks'); total=(used+free) if used is not None and free is not None else None
        kv=(used/total) if total else d.get('kv_utilization')
        t=(d['monotonic_ns']-start)/1e9
        diag.append({'t':t,'logical_index':i,'running':d.get('num_running_requests'),
                     'waiting':d.get('num_waiting_requests'),'kv_utilization':kv,
                     'kv_used_blocks':used,'kv_total_blocks':total,'event':d.get('event')})
    # per-second diagnostic means and request load curve
    max_t=max([(x['client_terminal_ns']-start)/1e9 for x in reqs if x.get('client_terminal_ns')]+[0])
    bins=list(range(0,math.ceil(max_t)+1))
    state_rows=[]
    state_lookup={}
    for i in range(4):
        for b in bins:
            rows=[d for d in diag if d['logical_index']==i and b<=d['t']<b+1]
            row={'environment':env,'second':b,'logical_instance':i,
                 'gpu_id':gpu_by_instance.get(mapping_ids[i]) if i<len(mapping_ids) else None,
                 'running_mean':mean([x['running'] for x in rows if x['running'] is not None]),
                 'waiting_mean':mean([x['waiting'] for x in rows if x['waiting'] is not None]),
                 'kv_mean':mean([x['kv_utilization'] for x in rows if x['kv_utilization'] is not None]),
                 'diagnostic_samples':len(rows)}
            state_rows.append(row);state_lookup[(i,b)]=row
    pressure=[]
    for b in bins:
        t=(b+.5)
        sent_before=sum((r['sent_ns']-start)/1e9<=t for r in reqs if r.get('sent_ns') is not None)
        done_before=sum((r['client_terminal_ns']-start)/1e9<=t for r in reqs if r.get('client_terminal_ns') is not None)
        planned=sum((r['scheduled_ns']-start)/1e9<=t for r in reqs if r.get('scheduled_ns') is not None)
        running=sum((state_lookup[(i,b)]['running_mean'] or 0) for i in range(4))
        waiting=sum((state_lookup[(i,b)]['waiting_mean'] or 0) for i in range(4))
        pressure.append({'environment':env,'second':b,'planned_cumulative':planned,
                         'sent_cumulative':sent_before,'completed_cumulative':done_before,
                         'outstanding':sent_before-done_before,'running_total':running,'waiting_total':waiting})
    # top tail request lifecycle timeline
    top=[]
    for r in sorted(reqs,key=lambda x:x['client_e2e_ms'],reverse=True)[:5]:
        rid=r['request_id'];idx=r.get('initial_instance_index');adm=admissions.get(rid,{});route=manager_routes.get(rid,{})
        first= r.get('client_first_ns');terminal=r.get('client_terminal_ns')
        top.append({'environment':env,'request_id':rid,'initial_index':idx,
            'initial_gpu':gpu_by_instance.get(mapping_ids[idx]) if idx is not None and idx<len(mapping_ids) else None,
            'output_tokens':r.get('actual_output_tokens'),'E2E_ms':r.get('client_e2e_ms'),
            'TTFT_ms':r.get('ttft_ms'),'TPOT_ms':r.get('tpot_ms'),
            'send_lag_ms':r.get('send_lag_ms'),
            'send_s':(r['sent_ns']-start)/1e9 if r.get('sent_ns') else None,
            'accepted_ms_after_send':((accepted_ns[rid]-r['sent_ns'])/1e6 if rid in accepted_ns else None),
            'admission_ms_after_send':((adm['ns']-r['sent_ns'])/1e6 if adm.get('ns') else None),
            'engine_first_ms_after_send':((engine_first[rid]-r['sent_ns'])/1e6 if rid in engine_first else None),
            'client_first_ms_after_send':((first-r['sent_ns'])/1e6 if first else None),
            'completion_s':(terminal-start)/1e9 if terminal else None,
            'initial_route_audit_match': episode.get('route_audit',{}).get('manager_first_routes',{}).get(rid)==mapping_ids[idx] and episode.get('route_audit',{}).get('engine_first_admissions',{}).get(rid)==mapping_ids[idx] if idx is not None and idx<len(mapping_ids) else None,
            'admitted_instance_id':adm.get('instance_id')})
    hot_index=comp['candidate_metadata']['hotspot_instance_index']
    hot_reqs=[r for r in reqs if r.get('initial_instance_index')==hot_index]
    nonhot_reqs=[r for r in reqs if r.get('initial_instance_index')!=hot_index]
    def route_group_stats(group):
        if not group: return {'n':0}
        first=min(r['client_first_ns'] for r in group if r.get('client_first_ns') is not None)
        last=max(r['client_last_ns'] for r in group if r.get('client_last_ns') is not None)
        return {'n':len(group),'mean_e2e_ms':mean([r['client_e2e_ms'] for r in group]),
            'mean_ttft_ms':mean([r['ttft_ms'] for r in group]),'mean_tpot_ms':mean([r['tpot_ms'] for r in group]),
            'output_tokens_per_second':sum(r.get('actual_output_tokens',0) for r in group)/((last-first)/1e9)}
    hot_batch_gpu=[];hot_batch_executor=[]
    hot_iid=mapping_ids[hot_index]
    for ev in events:
        if ev.get('kind')!='diagnostic': continue
        d=ev.get('data') or {}
        if d.get('instance_id')!=hot_iid: continue
        if d.get('event')=='gpu_execution_completed' and d.get('num_decode_requests')==43:
            hot_batch_gpu.append(d.get('gpu_elapsed_ms'))
        if d.get('event')=='engine_iteration_completed' and d.get('num_decode_requests')==43:
            hot_batch_executor.append(d.get('executor_wall_ms'))
    gpu_dist=sorted(x for x in hot_batch_gpu if x is not None)
    exec_dist=sorted(x for x in hot_batch_executor if x is not None)
    records[env]={
      'root':str(root.relative_to(ROOT)), 'comparison_file':str((root/'comparisons'/f'{suffix}.json').relative_to(ROOT)),
      'episode_dir':str(epdir.relative_to(ROOT)), 'base_trace_hash':comp.get('base_trace_hash'),
      'full_trace_hash':comp.get('full_trace_hash'),'candidate_metadata':comp['candidate_metadata'],
      'metrics':{'mean_e2e_ms':e2e['mean'],'p50_e2e_ms':e2e['median'],'p95_e2e_ms':e2e['p95'],
          'p99_e2e_ms':e2e['p99'],'max_e2e_ms':e2e['max'],'cvar95_e2e_ms':e2e['cvar95'],
          'C_ms':e2e['mean']+e2e['cvar95'],'mean_ttft_ms':ttft['mean'],'p95_ttft_ms':ttft['p95'],
          'mean_tpot_ms':tpot['mean'],'p95_tpot_ms':tpot['p95'],
          'send_lag_mean_ms':e2e.get('mean_send_lag_ms')},
      'cohort':{'trace_n':len(trace),'measured_n':episode.get('measured_request_count'),
          'successes':sum(r.get('outcome')=='success' for r in reqs),'non_successes':sum(r.get('outcome')!='success' for r in reqs),
          'input_tokens_total':sum(r.get('expected_input_tokens',0) for r in reqs),
          'output_tokens_total':sum(r.get('actual_output_tokens',0) for r in reqs),
          'output_token_distribution':{str(k):v for k,v in output_counts.items()},
          'span_from_first_send_to_last_completion_s':(max(terminals)-min(sends))/1e9,
          'cluster_output_tokens_per_second':sum(r.get('actual_output_tokens',0) for r in reqs)/((max(terminals)-min(sends))/1e9),
          'planned_arrival_window_s':comp['candidate_metadata']['arrival_time_domain_sec'],
          'hot_requests':route_group_stats(hot_reqs),'nonhot_requests':route_group_stats(nonhot_reqs),
          'actual_prompt_token_distribution':{str(k):v for k,v in Counter(x.get('data',{}).get('input_tokens') for x in events if x.get('kind')=='server_terminal').items()},
          'actual_send_lag_mean_ms':mean([r['send_lag_ms'] for r in reqs]),
          'actual_send_lag_p95_ms':pct([r['send_lag_ms'] for r in reqs],.95),
          'actual_send_lag_max_ms':max(r['send_lag_ms'] for r in reqs)},
      'route':{'status':episode.get('route_audit',{}).get('status'),
          'routing_mode':episode.get('routing_mode'), 'hotspot_index':comp['candidate_metadata']['hotspot_instance_index'],
          'actual_route_counts':{str(k):v for k,v in sorted(route_counts.items())},
          'logical_to_instance_id':mapping_ids,'logical_to_gpu_id':[gpu_by_instance.get(i) for i in mapping_ids],
          'manager_first_route_count':len(episode.get('route_audit',{}).get('manager_first_routes',{})),
          'engine_admission_count':len(episode.get('route_audit',{}).get('engine_first_admissions',{})),
          'manager_engine_mismatch_count':sum(episode.get('route_audit',{}).get('manager_first_routes',{}).get(rid)!=episode.get('route_audit',{}).get('engine_first_admissions',{}).get(rid) for rid in episode.get('route_audit',{}).get('manager_first_routes',{})),
          'admission_join_count':sum(r['request_id'] in admissions for r in reqs),
          'manager_route_join_count':sum(r['request_id'] in manager_routes for r in reqs),
          'admission_instance_mismatch_count':sum(admissions.get(r['request_id'],{}).get('instance_id') != episode.get('route_audit',{}).get('engine_first_admissions',{}).get(r['request_id']) for r in reqs if r['request_id'] in admissions),
          'manager_route_instance_mismatch_count':sum(manager_routes.get(r['request_id'],{}).get('instance_id') != episode.get('route_audit',{}).get('manager_first_routes',{}).get(r['request_id']) for r in reqs if r['request_id'] in manager_routes)},
      'execution':{'execution_audit_status':execution.get('status'), 'stay_migration_attempts':opportunity.get('started',0),
          'stay_migration_successes':opportunity.get('succeeded',0),'stay_migration_failures':opportunity.get('failed',0),
          'coordinator_attempt_records':(execution.get('coordinator_attempts',0) if isinstance(execution.get('coordinator_attempts',0),int) else len(execution.get('coordinator_attempts',[]))),
          'reconciled_migrations':len(execution.get('reconciled_migrations',[])),
          'decision_count':len(decisions),'stay_decision_classes':dict(Counter(d.get('action_class') for d in decisions)),
          'scale_enabled':False},
      'hot_batch_execution':{'gpu_elapsed_ms_n_decode_43':len(gpu_dist),'gpu_elapsed_ms_mean':mean(gpu_dist),'gpu_elapsed_ms_values':gpu_dist,'executor_wall_ms_n_decode_43':len(exec_dist),'executor_wall_ms_mean':mean(exec_dist),'executor_wall_ms_values':exec_dist},
      'instance_state':{}, 'diagnostic_kind_counts':dict(diag_kinds),
      'time_series':{'pressure':pressure,'instances':state_rows},'top_tail_requests':top,
      'request_rows':reqs,
    }
    # per logical instance and hotspot aggregate
    for i in range(4):
        vals=[d for d in diag if d['logical_index']==i]
        records[env]['instance_state'][str(i)]={
            'gpu_id':gpu_by_instance.get(mapping_ids[i]),'samples':len(vals),
            'running_mean':mean([d['running'] for d in vals if d['running'] is not None]),
            'running_p95':pct([d['running'] for d in vals if d['running'] is not None],.95),
            'running_peak':max((d['running'] for d in vals if d['running'] is not None),default=None),
            'waiting_mean':mean([d['waiting'] for d in vals if d['waiting'] is not None]),
            'waiting_p95':pct([d['waiting'] for d in vals if d['waiting'] is not None],.95),
            'waiting_peak':max((d['waiting'] for d in vals if d['waiting'] is not None),default=None),
            'kv_mean_pct':100*mean([d['kv_utilization'] for d in vals if d['kv_utilization'] is not None]),
            'kv_p95_pct':100*pct([d['kv_utilization'] for d in vals if d['kv_utilization'] is not None],.95),
            'kv_peak_pct':100*max((d['kv_utilization'] for d in vals if d['kv_utilization'] is not None),default=0),
        }

# Parse actual runtime config and compare after excluding per-run path strings.
roots=[ROOT/'benchmark/results/mig_ppo_v2_followup_20261106T080500Z',ROOT/'benchmark/results/mig_ppo_v2_multienv_20261105T130307Z']
configs=[]
for root in roots:
    cfg=yaml.safe_load((root/'runtime_config.yaml').read_text())
    cfg['service']['ray_temp_dir']='<run-specific>'
    cfg['service']['server_argv'][cfg['service']['server_argv'].index('--config-file')+1]='<run-specific-real_llumnix.yaml>'
    configs.append(cfg)
ll=[yaml.safe_load((root/'real_llumnix.yaml').read_text()) for root in roots]
model=[read_json(root/'environment_audit.json')['model_identity']['files_sha256'] for root in roots]
gpu=[read_json(root/'environment_audit.json')['gpu_inventory']['stdout'] for root in roots]
config_comparison={'normalized_runtime_configs_equal':configs[0]==configs[1],
    'real_llumnix_configs_equal':ll[0]==ll[1], 'model_file_hashes_equal':model[0]==model[1],
    'gpu_inventory_equal':gpu[0]==gpu[1], 'gpu_inventory':gpu[0]}

# Auxiliary HOT1-H repeated Stay evidence and L/M comparisons.
aux=[]
for rootname,pattern in [
 ('mig_ppo_v2_followup_20261106T080500Z','HOT1-H_backup_n48_q1p0_s26110523_followup.json'),
 ('mig_ppo_v2_hot1h_repeat_20261106T083209Z','HOT1-H_repeat_n48_q1p0_s26110671.json'),
 ('mig_ppo_v2_hot1h_repeat2_20261106T084608Z','HOT1-H_repeat_n48_q1_s26110681.json'),
 ('mig_ppo_v2_hot1h_q1p25_20261106T085546Z','HOT1-H_repeat_n48_q1p25_s26110691.json'),
]:
 d=read_json(ROOT/'benchmark/results'/rootname/'comparisons'/pattern)
 st=d['episodes']['Stay']['metrics']
 aux.append({'run':rootname,'candidate_id':d['candidate_id'],'qps':d['candidate_metadata']['target_qps'],
  'arrival_seed':d['candidate_metadata']['arrival_seed'],'prompt_seed':d['candidate_metadata']['prompt_seed'],
  'base_trace_hash':d['base_trace_hash'],'stay_C_ms':d['stay_C_ms'],
  'mean_e2e_ms':st['mean_e2e_ms'],'p95_e2e_ms':st['p95_e2e_ms'],'cvar95_e2e_ms':st['cvar95_e2e_ms'],
  'actual_output_tokens':d['episodes']['Stay']['output_token_total'],
  'hotspot_fraction':d['candidate_metadata']['actual_hotspot_fraction'],
  'route_audit':d['episodes']['Stay']['route_audit']['status'],
  'execution_audit':d['episodes']['Stay']['execution_audit_status']})

# H1-L/M and table context.
main=ROOT/'benchmark/results/mig_ppo_v2_multienv_20261105T130307Z'
other_envs=[]
for path in sorted((main/'comparisons').glob('HOT*-*.json')):
    d=read_json(path)
    env=d.get('environment_id','')
    if env in ('HOT1-L','HOT1-M'):
        st=d.get('episodes',{}).get('Stay',{}).get('metrics') or {}
        other_envs.append({'environment_id':env,'candidate_id':d['candidate_id'],
            'qps':d.get('candidate_metadata',{}).get('target_qps'),'n':d.get('candidate_metadata',{}).get('n'),
            'stay_C_ms':d.get('stay_C_ms'),'mean_e2e_ms':st.get('mean_e2e_ms'),
            'cvar95_e2e_ms':st.get('cvar95_e2e_ms'),'status':d.get('status')})

# save audit datasets, summary tables, tail rows, and chart input curves
summary={'primary_H':{k:{key:val for key,val in v.items() if key not in ('request_rows','time_series','top_tail_requests')} for k,v in records.items()},
    'config_comparison':config_comparison,'hot1h_repeats':aux,'hot1_lm':other_envs}
(ASSET/'audit_data.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
for name,rows in [('pressure_curves.csv',[r for d in records.values() for r in d['time_series']['pressure']]),
                  ('instance_state_curves.csv',[r for d in records.values() for r in d['time_series']['instances']]),
                  ('tail_request_details.csv',[r for d in records.values() for r in d['top_tail_requests']])]:
    if rows:
        with (ASSET/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

import csv, math, html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'benchmark/results/hot1h_stay_anomaly_audit_assets'
def load(name):
    with (ASSET/name).open(encoding='utf-8') as f: return list(csv.DictReader(f))
pressure=load('pressure_curves.csv')
instances=load('instance_state_curves.csv')
ENVS=['HOT1-H','HOT2-H','HOT3-H','HOT4-H']
COLORS={'Outstanding':'#7b3294','Running':'#008837','Waiting':'#c51b7d','Completed':'#333333'}
ICOLORS={0:'#1b9e77',1:'#d95f02',2:'#7570b3',3:'#e7298a'}

def polyline(points,color,width=1.7,dash=None):
    if not points: return ''
    extra=f' stroke-dasharray="{dash}"' if dash else ''
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{extra} points="'+ ' '.join(f'{x:.1f},{y:.1f}' for x,y in points) +'"/>'
def text(x,y,s,size=11,anchor='start',weight='normal',fill='#222'):
    return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{html.escape(str(s))}</text>'
def panel_base(x,y,w,h,title,xmax,ymax,ylabel):
    L,R,T,B=47,16,32,37
    px=x+L;py=y+T;pw=w-L-R;ph=h-T-B
    a=[text(x+8,y+19,title,14,weight='bold')]
    a.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="white" stroke="#bbb"/>')
    for tick in range(0,int(ymax)+1,max(1,int(ymax/5))):
        yy=py+ph-tick/ymax*ph
        a.append(f'<line x1="{px}" y1="{yy}" x2="{px+pw}" y2="{yy}" stroke="#ddd"/>')
        a.append(text(px-7,yy+4,tick,10,'end'))
    for tick in range(0,int(xmax)+1,40):
        xx=px+tick/xmax*pw
        a.append(f'<line x1="{xx}" y1="{py}" x2="{xx}" y2="{py+ph}" stroke="#eee"/>')
        a.append(text(xx,py+ph+16,tick,10,'middle'))
    a.append(text(x+8,py+ph/2,ylabel,10,'middle'))
    return a,(px,py,pw,ph)

def chart_pressure():
    W,H=1280,770; pw,ph=620,365; xmax=190; ymax=50
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>',text(W/2,28,'Stay: request pressure and completed requests (1-second bins)',19,'middle','bold')]
    legends=[('Outstanding',COLORS['Outstanding']),('Running',COLORS['Running']),('Waiting',COLORS['Waiting']),('Arrivals cumulative', '#666666'),('Completed cumulative',COLORS['Completed'])]
    lx=250
    for label,color in legends:
        out.append(f'<line x1="{lx}" y1="50" x2="{lx+22}" y2="50" stroke="{color}" stroke-width="2"/>');out.append(text(lx+28,54,label,11));lx+=180
    for j,env in enumerate(ENVS):
        x=15+(j%2)*635;y=65+(j//2)*375
        p=[r for r in pressure if r['environment']==env]
        parts,(px,py,plotw,ploth)=panel_base(x,y,pw,ph,env,xmax,ymax,'Requests')
        out+=parts
        for key,color in [('outstanding',COLORS['Outstanding']),('running_total',COLORS['Running']),('waiting_total',COLORS['Waiting'])]:
            pts=[]
            for r in p:
                t=float(r['second'])+.5;v=float(r[key]);pts.append((px+t/xmax*plotw,py+ploth-v/ymax*ploth))
            out.append(polyline(pts,color,1.8))
        for key,color,dash in [('sent_cumulative','#666666','5 4'),('completed_cumulative',COLORS['Completed'],None)]:
            pts=[]
            for r in p:
                t=float(r['second'])+.5;v=float(r[key]);pts.append((px+t/xmax*plotw,py+ploth-v/48*ploth))
            out.append(polyline(pts,color,1.8,dash))
        for tick in [0,12,24,36,48]:
            yy=py+ploth-tick/48*ploth;out.append(text(px+plotw+5,yy+4,tick,9))
        out.append(text(px+plotw/2,py+ploth+32,'Seconds from first planned arrival',10,'middle'))
    out.append(text(15,755,'Left axis: outstanding, running, waiting. Right axis: cumulative actual sends and client completions.',10,fill='#555'))
    out.append('</svg>')
    (ASSET/'hot1h_pressure_curves.svg').write_text('\n'.join(out),encoding='utf-8')

def chart_instances():
    W,H=1500,1040; left=180; top=70; colw=430; rowh=230; xmax=190
    metrics=[('running_mean','Running requests',50,1),('waiting_mean','Waiting requests',10,1),('kv_mean','KV Cache used (%)',50,100)]
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>',text(W/2,30,'Stay: per-instance state samples (1-second means)',19,'middle','bold')]
    for c,(metric,title,ymax,mult) in enumerate(metrics):
        out.append(text(left+c*colw+colw/2,55,title,14,'middle','bold'))
        for tick in range(0,int(ymax)+1,max(1,int(ymax/5))):
            xx=left+c*colw+tick/ymax*colw
            out.append(text(xx,72,str(tick),9,'middle',fill='#555'))
    for ri,env in enumerate(ENVS):
        y=top+ri*rowh
        out.append(text(18,y+115,env,14,weight='bold'))
        rows=[r for r in instances if r['environment']==env]
        for ci,(metric,title,ymax,mult) in enumerate(metrics):
            x=left+ci*colw; plotx=x+35; ploty=y+10; plotw=colw-55; ploth=165
            out.append(f'<rect x="{plotx}" y="{ploty}" width="{plotw}" height="{ploth}" fill="white" stroke="#bbb"/>')
            for tick in range(0,int(ymax)+1,max(1,int(ymax/5))):
                yy=ploty+ploth-tick/ymax*ploth
                out.append(f'<line x1="{plotx}" y1="{yy}" x2="{plotx+plotw}" y2="{yy}" stroke="#e3e3e3"/>')
                out.append(text(plotx-5,yy+3,tick,9,'end',fill='#555'))
            for tick in range(0,191,40):
                xx=plotx+tick/xmax*plotw
                out.append(f'<line x1="{xx}" y1="{ploty}" x2="{xx}" y2="{ploty+ploth}" stroke="#eee"/>')
                if ri==len(ENVS)-1: out.append(text(xx,ploty+ploth+15,tick,9,'middle',fill='#555'))
            for i in range(4):
                rr=sorted([r for r in rows if int(r['logical_instance'])==i and r.get(metric) not in ('',None)],key=lambda z:int(z['second']))
                points=[]
                for r in rr:
                    t=int(r['second']);v=float(r[metric])*mult
                    points.append((plotx+t/xmax*plotw,ploty+ploth-min(ymax,v)/ymax*ploth))
                out.append(polyline(points,ICOLORS[i],1.6))
            if ri==0:
                # compact instance-index legend in each column
                for i in range(4):
                    lx=plotx+10+i*86
                    out.append(f'<line x1="{lx}" y1="{ploty+ploth+25}" x2="{lx+14}" y2="{ploty+ploth+25}" stroke="{ICOLORS[i]}" stroke-width="2"/>')
                    out.append(text(lx+18,ploty+ploth+29,f'idx {i}',9))
            if ci==1 and ri==0: out.append(text(plotx+plotw-4,ploty+ploth-5,'all four instances overlap at zero',9,'end',fill='#555'))
    out.append(text(W/2,H-12,'Seconds from first planned arrival; four logical instances are colored by index.',10,'middle',fill='#555'))
    out.append('</svg>')
    (ASSET/'hot1h_instance_state_curves.svg').write_text('\n'.join(out),encoding='utf-8')
chart_pressure();chart_instances()


print(json.dumps({'assets':str(ASSET.relative_to(ROOT)),'scenes':{e:{'C_ms':round(d['metrics']['C_ms'],1),'mean_E2E_ms':round(d['metrics']['mean_e2e_ms'],1),'mean_TPOT_ms':round(d['metrics']['mean_tpot_ms'],2),'hot_GPU43_samples':d['hot_batch_execution']} for e,d in records.items()}},indent=2,ensure_ascii=False))
