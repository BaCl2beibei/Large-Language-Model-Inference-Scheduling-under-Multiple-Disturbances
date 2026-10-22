"""P3 protocol, frozen traces, constrained Balanced, and paired gate (CPU pure)."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .real_bridge import PAIRS, digest
from .real_budget import BudgetLedger


SEEDS=(2101,2102,2103)
PROBE_SEEDS=(7101,7102,7103)
FINAL_SEEDS=(5101,5102,5103)


def protocol(run_dir, runtime_config):
    """Freeze choices before observing new traffic; all scales remain unmeasured."""
    return {
        'schema':'real-opportunity-v1','run_dir':str(Path(run_dir).resolve()),
        'runtime_config':str(Path(runtime_config).resolve()),
        'scope':'06 CPU preparation; 07 runs only on separate user instruction and sufficient budget',
        'prerequisite_exception':{'user_instruction':'跳过smoke，直接执行06_sol_p3_prepare.md',
            '05_smoke':'skipped_by_user','P1_P2_live_status':'not_verified',
            'reason':'Explicit user override of the 06 live prerequisite; no synthetic passed live gate',
            'inherited_for_opportunity_protocol':True,'GPU_started_in_06':False},
        'horizons':{'arrival_sec':120,'drain_sec':120,'cleanup_sec':60,'decision_interval_sec':1},
        'probe':{'qps':[3,6,9],'seeds':list(PROBE_SEEDS),'scenes':['A','B'],
            'arrival_sec':30,'drain_sec':60,'policy':'Stay','runs':6,
            'order':'qps ascending, A then B; no automatic retry or escalation beyond 9'},
        'health':{'min_completion_rate':.99,'max_nonmanual_error_rate':.01,
            'max_first_output_censored_fraction':.01,'max_send_lateness_p95_ms':1000.,
            'require_all_offered_sent':True,'require_passed_score_cleanup_route_action_audits':True},
        'selection':{'A':'lower median of healthy normal probe QPS (sorted ascending)',
            'B':'largest healthy skew probe QPS with waiting>0 in at least half of last-third arrival samples',
            'missing_healthy_or_queued_scene':'stop incomplete; do not tune workload or thresholds'},
        'scenes':{'A':{'routing':'native load dispatcher'},
                  'B':{'routing':'fixed trace route seed into stable instance indices','route_shares':[.7,.1,.1,.1]}},
        'opportunity':{'seeds':list(SEEDS),'policies':['Stay','Balanced'],'runs':12,
            'pair_order':'scene A then B, seeds ascending; policy order alternates by scene_index+seed_index',
            'paired_trace_identical':True,'same_sampling_and_finite_window':True,'reuse_service':True},
        'workload':{'arrival':'piecewise Poisson with one initial request at t=0',
            'input_budgets':[256,512,1048],'output_budgets':[256,512,1048],
            'profile_mix':'all 9 Cartesian combinations equally likely',
            'max_requests':4096,'max_tokens_are_budgets_not_realized_lengths':True,
            'prompt_pool_seed':8300,'route_seed':'seed*1000000+request_index',
            'separate_profile_rng_offset':100000},
        'balanced':{'version':'constrained-Balanced-remaining-capacity-SR-v1',
            'load':'(free_blocks-waiting_blocks)/(running+waiting), +inf only for measured empty queue',
            'threshold':9000.,'source_load_at_most_threshold':True,'destination_load_at_least_threshold':True,
            'pairing':'busiest sources (ascending numeric capacity/request) with least busy destinations, stable ID ties',
            'move':'exact observed SR candidate; after-load uses its actual block count and one running request',
            'predicate':'source_after < destination_after and 0 < abs(after_difference) < abs(before_difference)',
            'relation_to_native':'native Balanced remaining_steps ordering/filter/predicate; actual SR blocks replace stale last-queue estimate',
            'control':'same lease, one global migration, terminal cooldown 5s, reserve max(16,ceil(5% total)), freshness <=2s'},
        'slos':{'ttft_ms':2000.,'tpot_ms':150.,'single_token':'TTFT only; TPOT N/A'},
        'metrics':{'primary':'SLO-goodput over fixed 240s','completion_denominator':'all offered',
            'errors':'timeout/service_error client terminals by deadline; unknown invalid, manual cancel excluded',
            'censoring':'first-output, terminal-outstanding and offered-not-sent fractions reported separately',
            'zero_denominators':'null + incomplete; no infinite relative gain'},
        'gate':{'min_paired_relative_gain':.10,'min_winning_seeds':2,'max_error_rate_increase_pp':1.,
            'healthy_probe_selection_required':True,'zero_Stay_goodput':'report absolute delta; never relative-pass',
            'majority_zero_scenes':'incomplete if >=2 of 2 scenes both policies have zero mean goodput'},
        'reward':{'weights':{'good':1.,'hold':.2,'first':.5,'gap':.2,'attempt':.02,'bad':.2,'tail':2.},
            'C_ref':None,'N_ref':None,'scale_status':'unmeasured',
            'scale_rule':'pooled completed/seconds and outstanding-request-seconds/seconds from healthy Stay opportunity only; each floor 1',
            'gap_limit':'max(1s,5*tpot_slo)','window_partition':'left-closed right-open; final deadline included once',
            'weight_revisions_before_training_max':1,'automatic_weight_revision':False},
        'final_burst':{'seeds':list(FINAL_SEEDS),'not_for_training_or_selection':True,
            'base_qps':'selected scene A QPS, determined only by opportunity probes',
            'arrival_sec':120,'drain_sec':120,'routing':'native dispatcher',
            'rate_segments':[[0,30,1],[30,45,3],[45,90,1],[90,105,3],[105,120,1]],
            'profiles':'same frozen nine-profile distribution and prompt pool',
            'generation_rule':'same Poisson generator; freeze/hash traces after load selection, before any final policy evaluation'},
        'stop_rules':['first request/action/route/cleanup audit failure stops all later runs',
            'unknown migration terminal is invalid, not failed/cancelled',
            'no performance-based reruns, discarded seeds, workload changes or automatic retry',
            'reserve complete probe+12-run maximum before service activation'],
    }


def episode_schedule(spec, selected):
    rows=[]
    for scene_index,scene in enumerate(('A','B')):
        for seed_index,seed in enumerate(spec['opportunity']['seeds']):
            policies=('Stay','Balanced') if (scene_index+seed_index)%2==0 else ('Balanced','Stay')
            for policy in policies:
                rows.append({'phase':'opportunity','scene':scene,'seed':seed,'policy':policy,'qps':selected[scene]})
    return rows


def balanced_action(observation, snapshot, *, threshold=9000.):
    if observation is None:return 0
    raw=observation['raw']
    if raw['snapshot_version']!=snapshot['version']:raise ValueError('policy must use the saved decision snapshot')
    states={s['instance_id']:s for s in snapshot['instances']};mapping=snapshot['mapping']
    def load(s, delta=0, blocks=0):
        fields=[s.get(k) for k in ('free_blocks','waiting_blocks','running','waiting')]
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in fields):return None
        free,waiting_blocks,running,waiting=fields;n=running+waiting+delta
        return (free-waiting_blocks-blocks)/n if n>0 else math.inf if n==0 else None
    loads={key:load(state) for key,state in states.items()}
    sources=sorted((key for key in states if loads[key] is not None and loads[key]<=threshold),key=lambda key:(loads[key],key))
    dests=sorted((key for key in states if loads[key] is not None and loads[key]>=threshold),key=lambda key:(-loads[key],key))
    indices={entry[0]:i for i,entry in enumerate(mapping)}
    for src,dst in zip(sources,dests):
        if src==dst:continue
        action=PAIRS.index((indices[src],indices[dst]))+1
        if not raw['action_mask'][action]:continue
        blocks=states[src].get('candidate_blocks')
        if type(blocks) is not int or blocks<1:continue
        a=load(states[src],-1,-blocks);b=load(states[dst],1,blocks)
        if a is not None and b is not None and a<b and 0<abs(a-b)<abs(loads[src]-loads[dst]):return action
    return 0


def metric_summary(ledger):
    score=ledger.freeze_score()
    if score['status']!='passed':raise ValueError('score audit did not pass')
    metrics=deepcopy(score['fixed']);rows,_=ledger._rows(occurred_by_ns=ledger.deadline_ns)
    offered=metrics['offered']
    errors=sum(r['outcome'] in ('timeout','service_error') and r['client_terminal_ns'] is not None
               and r['client_terminal_ns']<=ledger.deadline_ns for r in rows)
    metrics.update(nonmanual_errors=errors,nonmanual_error_rate=errors/offered if offered else None,
        first_output_censored_fraction=metrics['first_output_censored']/offered if offered else None,
        terminal_censored_fraction=len(metrics['outstanding_ids'])/offered if offered else None,
        offered_not_sent_fraction=(offered-metrics['sent'])/offered if offered else None)
    return metrics


def healthy(row,spec):
    m=row['metrics'];h=spec['health']
    return (row.get('status')=='passed' and m['offered']>0
            and m['sent']==m['offered'] and m['completion_rate'] is not None and m['completion_rate']>=h['min_completion_rate']
            and m['nonmanual_error_rate'] is not None and m['nonmanual_error_rate']<=h['max_nonmanual_error_rate']
            and m['first_output_censored_fraction'] is not None
            and m['first_output_censored_fraction']<=h['max_first_output_censored_fraction']
            and row.get('send_lateness_p95_ms') is not None and row['send_lateness_p95_ms']<=h['max_send_lateness_p95_ms'])


def select_loads(probes,spec):
    required={(scene,qps,seed) for scene in ('A','B') for qps,seed in zip(spec['probe']['qps'],spec['probe']['seeds'])}
    keys=[(row['scene'],row['qps'],row['seed']) for row in probes]
    if len(keys)!=len(set(keys)) or set(keys)!=required:raise ValueError('full frozen probe inventory required')
    if any(row.get('phase')!='probe' or row.get('policy')!='Stay' or row.get('status')!='passed' for row in probes):
        raise ValueError('invalid probe evidence; stop before selecting loads')
    a=sorted(r['qps'] for r in probes if r['scene']=='A' and healthy(r,spec))
    b=sorted(r['qps'] for r in probes if r['scene']=='B' and healthy(r,spec)
             and r.get('queue_last_third_samples',0)>0 and r.get('queue_last_third_positive',0)/r['queue_last_third_samples']>=.5)
    if not a or not b:return {'status':'incomplete','reason':'no healthy normal or healthy persistently queued skew probe','selected':None}
    return {'status':'passed','selected':{'A':a[(len(a)-1)//2],'B':b[-1]},
            'healthy_normal_qps':a,'healthy_queued_skew_qps':b,'rule':spec['selection']}


def paired_gate(episodes,spec):
    required={(scene,seed,policy) for scene in ('A','B') for seed in spec['opportunity']['seeds'] for policy in ('Stay','Balanced')}
    keys=[(e['scene'],e['seed'],e['policy']) for e in episodes]
    if set(keys)!=required or len(keys)!=len(set(keys)):
        return {'status':'incomplete','reason':'full 12-run paired matrix missing or duplicated','pairs':[]}
    if any(e.get('status')!='passed' for e in episodes):
        return {'status':'incomplete','reason':'invalid episode retained; request/action/cleanup audit failed','pairs':[]}
    if any(e['metrics'].get('offered',0)<=0 or e['metrics'].get('nonmanual_error_rate') is None for e in episodes):
        return {'status':'incomplete','reason':'zero offered denominator; report null rates','pairs':[]}
    by_key={key:row for key,row in zip(keys,episodes)};pairs=[];scenes={}
    for scene in ('A','B'):
        rows=[]
        for seed in spec['opportunity']['seeds']:
            a,b=(by_key[scene,seed,p] for p in ('Stay','Balanced'))
            if (a['trace_hash']!=b['trace_hash'] or a['external_case_hash']!=b['external_case_hash']
                    or a['metrics']['duration_sec']!=240 or b['metrics']['duration_sec']!=240
                    or a['metrics']['offered']!=b['metrics']['offered'] or a['mapping']!=b['mapping']):
                return {'status':'incomplete','reason':'pair trace/routing/profile/window/mapping mismatch','pairs':pairs}
            ga,gb=a['metrics']['slo_goodput_rps'],b['metrics']['slo_goodput_rps']
            ea,eb=a['metrics']['nonmanual_error_rate'],b['metrics']['nonmanual_error_rate']
            relative=(gb-ga)/ga if ga>0 else None
            error_pp=(eb-ea)*100 if ea is not None and eb is not None else None
            error_ok=error_pp is not None and error_pp<=spec['gate']['max_error_rate_increase_pp']+1e-12
            win=relative is not None and relative+1e-12>=spec['gate']['min_paired_relative_gain'] and error_ok
            row={'scene':scene,'seed':seed,'Stay_goodput_rps':ga,'Balanced_goodput_rps':gb,
                 'absolute_goodput_delta':gb-ga,'relative_gain':relative,'error_rate_delta_pp':error_pp,
                 'error_guard_passed':error_ok,'winning_pair':win,
                 'Stay_completion_rate':a['metrics']['completion_rate'],'Balanced_completion_rate':b['metrics']['completion_rate'],
                 'Stay_first_censored_fraction':a['metrics']['first_output_censored_fraction'],
                 'Balanced_first_censored_fraction':b['metrics']['first_output_censored_fraction'],
                 'Stay_terminal_censored_fraction':a['metrics']['terminal_censored_fraction'],
                 'Balanced_terminal_censored_fraction':b['metrics']['terminal_censored_fraction']}
            rows.append(row);pairs.append(row)
        scenes[scene]={'winning_seeds':sum(r['winning_pair'] for r in rows),
            'error_guard_all_seeds':all(r['error_guard_passed'] for r in rows),
            'both_mean_goodput_zero':all(r['Stay_goodput_rps']==r['Balanced_goodput_rps']==0 for r in rows)}
        scenes[scene]['passed']=(scenes[scene]['winning_seeds']>=spec['gate']['min_winning_seeds'] and scenes[scene]['error_guard_all_seeds'])
    all_zero=sum(s['both_mean_goodput_zero'] for s in scenes.values())>=2
    passed=any(s['passed'] for s in scenes.values()) and not all_zero
    return {'status':'passed' if passed else 'incomplete' if all_zero else 'failed',
            'reason':'paired exploratory gate only; not PPO effect proof' if passed else 'no qualifying paired opportunity evidence',
            'pairs':pairs,'scenes':scenes,'zero_denominator_auto_pass':False,'PPO_training_authorized':False}


def maximum_duration(spec,runtime):
    p=spec['probe'];h=spec['horizons']
    return (runtime['service']['startup_timeout_sec']+60
        +p['runs']*(p['arrival_sec']+p['drain_sec']+h['cleanup_sec'])
        +spec['opportunity']['runs']*(h['arrival_sec']+h['drain_sec']+h['cleanup_sec']))


def readonly_budget(run_dir):
    ledger=BudgetLedger(run_dir);state=json.loads(ledger.path.read_text())
    if state['first_started_wall'] is not None:
        delta=max(0.,time.time()-state['last_wall'])
        if state['boot_id']==ledger.boot_id:delta=max(delta,time.monotonic()-state['last_mono'])
        state['elapsed_sec']+=delta
        if state['phase']:state['phase_usage'][state['phase']]+=delta
    return {'remaining':ledger._remaining(state),'exhausted':state['exhausted'],'read_only':True,'as_of_wall':time.time()}


def make_trace(spec, *, seed,qps,duration,bursty=False):
    """Generate small prompt-reference manifests; materialisation uses frozen pool."""
    arrivals=random.Random(seed);profiles=random.Random(seed+spec['workload']['separate_profile_rng_offset'])
    segments=spec['final_burst']['rate_segments'] if bursty else [[0,duration,1]]
    times=[0.]
    for lo,hi,multiplier in segments:
        at=lo+arrivals.expovariate(qps*multiplier)
        while at<hi:
            times.append(at);at+=arrivals.expovariate(qps*multiplier)
    if len(times)>=spec['workload']['max_requests']:raise ValueError('trace cap reached; do not silently truncate offered arrivals')
    rows=[];mix=[(i,o) for i in spec['workload']['input_budgets'] for o in spec['workload']['output_budgets']]
    for index,at in enumerate(times):
        inp,out=profiles.choice(mix)
        rows.append({'request_id':f'r{index+1:05d}','scheduled_arrival_sec':round(at,9),
            'request_type':f'in{inp}_out{out}','prompt_key':str(inp),'input_budget':inp,'output_tokens':out,
            'route_seed':seed*1000000+index})
    return rows


def load_frozen_trace(run_dir,spec,*,seed,qps,phase):
    directory=Path(run_dir)/'traces/opportunity07'
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['protocol_hash']!=digest(spec):raise ValueError('trace protocol changed')
    pool=json.loads((directory/'prompt_pool.json').read_text())
    if digest(pool)!=manifest['prompt_pool_hash']:raise ValueError('frozen prompt pool changed')
    key=f'{phase}_q{qps}_seed{seed}';item=manifest['traces'][key]
    path=directory/item['path'];raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('frozen trace changed')
    rows=json.loads(raw)
    expected=make_trace(spec,seed=seed,qps=qps,duration=spec['probe']['arrival_sec'] if phase=='probe' else spec['horizons']['arrival_sec'])
    if rows!=expected:raise ValueError('trace disagrees with preregistered generator')
    return [{**r,'prompt':pool[r['prompt_key']]['prompt'],'input_tokens':pool[r['prompt_key']]['input_tokens']} for r in rows]
