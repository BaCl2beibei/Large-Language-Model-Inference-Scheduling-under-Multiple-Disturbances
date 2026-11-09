"""Portable native Windows/Linux CPU bundle and on-policy multi-environment PPO.

No live service imports, Linux-only locks, model downloads or real RPCs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from rl_scheduler.cohort_ppo import (CohortActorCritic, FrozenController, build_rollout,
    normalizer_definition, policy_hash, ppo_update, tensors)
from rl_scheduler.cohort_reward import MigPPOV2Reward
from rl_scheduler.cohort_statistics import summarize_values
from rl_scheduler.portable_io import atomic_json
from rl_scheduler.q15b_sim import Q15BCohortSim
from rl_scheduler.real_bridge import PAIRS, digest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = ROOT/'benchmark/results/mig_ppo_v2_refit_20261106T100025Z'
BUNDLE_SCHEMA = 'mig-ppo-v2-native-cpu-bundle-v1'
CHECKPOINT_SCHEMA = 'mig-ppo-v2-local-checkpoint-v1'
SETTINGS = dict(actor_sample_mode='choice_only', update_epochs=2, minibatch_size=64,
    clip_epsilon=.2, value_coefficient=.5, entropy_coefficient=.01, max_grad_norm=.5,
    target_kl=.03, learning_rate=.0003)
CPU_FILES = ['__init__.py','action_mask.py','cost_estimator.py','observation.py','reward.py',
    'service_model.py','env.py','real_bridge.py','real_observation.py','real_opportunity.py',
    'cohort_statistics.py','portable_io.py','cohort_reward.py','cohort_ppo.py','q15b_sim.py','ppo_runtime.py']
ENV_SPECS = [
    ('HOT1-L',48,1.,.5,'HOT1-L_backup_n48_q1p0_s26110523'),
    ('HOT1-M',48,.6,.7,'HOT1-M_primary_s26110521'),
    ('HOT1-H',48,1.,.9,'HOT1-H_backup_n48_q1p0_s26110523_followup'),
    ('HOT2-L',48,.6,.5,'HOT2-L_primary_s26110521'),
    ('HOT2-M',32,.6,.7,'HOT2-M_backup_n32_q0p6_s26110522'),
    ('HOT2-H',48,1.,.9,'HOT2-H_backup_n48_q1p0_s26110523'),
    ('HOT3-L',48,.6,.5,'HOT3-L_primary_s26110521'),
    ('HOT3-M',48,.6,.7,'HOT3-M_primary_s26110521'),
    ('HOT3-H',48,1.,.9,'HOT3-H_backup_n48_q1p0_s26110523'),
    ('HOT4-L',48,.6,.5,'HOT4-L_primary_s26110521'),
    ('HOT4-M',48,.6,.7,'HOT4-M_primary_s26110521'),
    ('HOT4-H',48,1.,.9,'HOT4-H_backup_n48_q1p0_s26110523'),
    ('NATIVE',48,.6,None,'NATIVE_primary_s26110521')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def inside(root, relative):
    if not isinstance(relative,str) or '\\' in relative or ':' in relative:
        raise ValueError('bundle paths must be relative POSIX paths')
    p=(Path(root)/relative).resolve()
    if not p.is_relative_to(Path(root).resolve()):raise ValueError('path escapes bundle root')
    return p


def load_bundle(root, verify=True):
    root=Path(root).resolve();m=read(root/'manifest.json')
    if m.get('schema')!=BUNDLE_SCHEMA:raise ValueError('native CPU bundle schema required')
    if verify:
        for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
            checksum,name=line.split('  ',1)
            if sha(inside(root,name))!=checksum:raise ValueError('bundle integrity failed: '+name)
    envs=read(root/'configs/environments.json')['environments']
    if [e['env_id'] for e in envs]!=[e[0] for e in ENV_SPECS] or len(envs)!=13:
        raise ValueError('exact frozen 13-environment pool required')
    if any(not math.isclose(e['probability'],1/13) for e in envs):raise ValueError('uniform environment probabilities required')
    reward=MigPPOV2Reward.from_state_dict(read(root/'calibration/reward_definition.json'))
    for e in envs:
        model=read(inside(root,e['model_path']))
        if model['decision_interval_sec']!=1.5 or model['migration_cooldown_sec']!=0:
            raise ValueError('V2 requires 1.5-second sampling and zero cooldown')
    return root,m,envs,reward


def numeric_trace(env,seed):
    """Exactly the measured fixed-N arrival and fixed-count hotspot generator."""
    rng=random.Random(int(seed));n=int(env['request_count']);qps=float(env['qps'])
    times=sorted(rng.random()*n/qps for _ in range(n))
    routes=[None]*n
    if env['env_id']!='NATIVE':
        route_rng=random.Random(int(seed)^0x51A1);order=list(range(n));route_rng.shuffle(order)
        count=math.floor(n*env['hot_fraction']+.5);hot=env['hot_index']
        for idx in order[:count]:routes[idx]=hot
        others=[i for i in range(4) if i!=hot]
        for k,idx in enumerate(order[count:]):routes[idx]=others[k%3]
    return [dict(request_id=f'trace_{seed}_{i:05d}',scheduled_arrival_sec=at,
        input_tokens=128,output_tokens=1536,initial_instance_index=routes[i],target_qps=qps,
        routing_mode='native' if env['env_id']=='NATIVE' else 'fixed') for i,at in enumerate(times)]


def trace_group(trace):
    # Request labels and routes do not make identical base workload independent.
    return digest([[r['scheduled_arrival_sec'],r['input_tokens'],r['output_tokens']] for r in trace])


def write_rollout(path,value):
    path=Path(path);temp=Path(str(path)+'.tmp')
    with gzip.open(temp,'wt',encoding='utf-8') as f:json.dump(value,f,allow_nan=False)
    os.replace(temp,path)


def read_rollout(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def episode_metrics(summary,decisions,migrations,reward_audit):
    all_stats=summarize_values([r['client_e2e_ms'] for r in summary['requests']])
    successful=[r for r in summary['requests'] if r['outcome']=='success']
    success_stats=summarize_values([r['client_e2e_ms'] for r in successful])
    moves=[m for m in migrations if m['status']=='success'];seen={};repeated=reverse=0
    for m in moves:
        old=seen.get(m['request_id'])
        if old:
            repeated+=1
            reverse+=old==(m['target_instance'],m['source_instance'])
        seen[m['request_id']]=(m['source_instance'],m['target_instance'])
    choices=sum(sum(bool(x) for x in d['action_mask'][1:])>0 for d in decisions)
    voluntary=sum(sum(d['action_mask'][1:])>0 and d['sampled_action']==0 for d in decisions)
    windows=reward_audit['windows'];elapsed=sum(w['dt_sec'] for w in windows)
    return dict(return_=reward_audit['return'],C_ms=all_stats['mean']+all_stats['cvar95'],
        mean_e2e_ms=all_stats['mean'],p95_e2e_ms=all_stats['p95'],latency_scope='all terminal requests; timeout lifetimes censored at episode deadline',
        success_C_ms=success_stats['mean']+success_stats['cvar95'] if successful else None,
        request_count=len(summary['requests']),failure_count=len(summary['requests'])-len(successful),
        failure_rate=1-len(successful)/len(summary['requests']),migration_count=len(moves),
        distinct_migrated_requests=len(seen),repeated_migrations=repeated,reverse_migrations=reverse,
        eligible_decisions=choices,forced_stay_decisions=len(decisions)-choices,voluntary_stay_decisions=voluntary,
        conditional_migration_rate=(choices-voluntary)/choices if choices else None,
        D_mean=sum(w['D_integral'] for w in windows)/elapsed,
        **{k+'_mean':sum(w['pressures'][k]*w['dt_sec'] for w in windows)/elapsed for k in ('Q','H','B_star')},
        terminal_sec=max(r['client_terminal_ns'] for r in summary['requests'])/1e9,
        source='simulation',formal_ppo_ready=False)


def run_episode(root,env,seed,out,policy=None,version=0,baseline=None,wall_limit=120.):
    model=read(inside(root,env['model_path']));trace=numeric_trace(env,seed)
    sim=Q15BCohortSim(model,trace,int(seed),out,minimal_logging=True,finalize_deadline=True,wall_time_limit_sec=wall_limit)
    try:
        atomic_json(Path(out)/'trace.json',trace)
        definition=MigPPOV2Reward.from_state_dict(read(Path(root)/'calibration/reward_definition.json'))
        controller=FrozenController(policy,normalizer_definition(),version) if policy is not None else None
        summary,decisions,migrations=sim.run(baseline or 'PPO',1.,controller=controller,reward_definition=definition,episode_horizon_sec=200.)
        reward_audit=read(Path(out)/'reward.json')
        metrics=episode_metrics(summary,decisions,migrations,reward_audit)
        rollout=build_rollout(summary,decisions,definition) if controller is not None else None
        if rollout is not None:write_rollout(Path(out)/'rollout.json.gz',rollout)
        for name in ('decisions','requests','migrations','curve'):
            path=Path(out)/(name+'.jsonl')
            with path.open('rb') as src,gzip.open(str(path)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
            path.unlink()
        atomic_json(Path(out)/'metrics.json',metrics)
        return metrics,rollout
    finally:sim.event_file.close()


def new_policy(seed):
    torch.manual_seed(int(seed));return CohortActorCritic(64).cpu()


def checkpoint(path,policy,optimizer,manifest,state,env_rng,config):
    payload=dict(schema=CHECKPOINT_SCHEMA,policy=policy.state_dict(),optimizer=optimizer.state_dict(),
        normalizer=normalizer_definition(),settings=config['settings'],bundle_contract_sha256=manifest['contract_sha256'],
        config=config,state=deepcopy(state),environment_rng=env_rng.getstate(),torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),policy_sha256=policy_hash(policy))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=Path(str(path)+'.tmp')
    with temp.open('wb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def load_checkpoint(path,manifest):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('schema')!=CHECKPOINT_SCHEMA or ck.get('normalizer')!=normalizer_definition():
        raise ValueError('incompatible checkpoint; native Mig-PPO V2 checkpoint required')
    if ck.get('bundle_contract_sha256')!=manifest['contract_sha256']:
        raise ValueError('checkpoint belongs to another simulator/source bundle')
    policy=CohortActorCritic(64).cpu();policy.load_state_dict(ck['policy'],strict=True)
    if policy_hash(policy)!=ck['policy_sha256']:raise ValueError('checkpoint tensor hash mismatch')
    opt=torch.optim.Adam(policy.parameters(),lr=ck['settings']['learning_rate']);opt.load_state_dict(ck['optimizer'])
    return policy,opt,ck


def restore_rng(ck,env_rng):
    env_rng.setstate(ck['environment_rng']);torch.set_rng_state(ck['torch_rng'])
    np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])


def doctor(bundle_root):
    root,m,envs,reward=load_bundle(bundle_root)
    cfg=read(root/'configs/local_train.json');torch.set_num_threads(cfg['torch_threads'])
    policy,_,ck=load_checkpoint(root/'checkpoints/initial_v2.pt',m)
    ref=read(root/'validation/policy_reference.json')
    with torch.no_grad():dist,val=policy.distribution(*tensors([ref['observation']]))
    if not np.allclose(dist.probs[0].numpy(),ref['probabilities'],atol=1e-6,rtol=1e-5) or not math.isclose(val.item(),ref['value'],abs_tol=1e-6,rel_tol=1e-5):
        raise ValueError('cross-platform reference output mismatch')
    return dict(status='passed',platform=platform.platform(),python=sys.version,device='cpu',
        torch=torch.__version__,numpy=np.__version__,cpu_count=os.cpu_count(),environment_count=len(envs),
        policy_parameters=sum(p.numel() for p in policy.parameters()),independent_validation_pending=True,
        native_windows_executed=platform.system()=='Windows',bundle_contract_sha256=m['contract_sha256'],
        cpu_module_paths={name:str(Path(sys.modules[name].__file__).resolve()) for name in ('rl_scheduler.q15b_sim','rl_scheduler.cohort_ppo')})


def smoke(bundle_root,run_dir):
    root,m,envs,_=load_bundle(bundle_root);out=Path(run_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=read(root/'configs/local_train.json');torch.set_num_threads(cfg['torch_threads'])
    policy,opt,ck=load_checkpoint(root/'checkpoints/initial_v2.pt',m);rng=random.Random();restore_rng(ck,rng)
    started=time.monotonic();rollouts=[];reports=[]
    for i,e in enumerate(envs):
        metrics,r=run_episode(root,e,700_000_000+i,out/e['env_id'],policy,0)
        rollouts.append(r);reports.append(dict(env_id=e['env_id'],**metrics))
    # This is one disposable contract update, never a training checkpoint.
    before=policy_hash(policy);update=ppo_update(policy,opt,rollouts,cfg['settings'],seed=cfg['update_seed'])
    state=dict(next_update=1,next_episode=13,pending=[],elapsed_sec=0.,history=[])
    checkpoint(out/'smoke_only.pt',policy,opt,m,state,rng,cfg)
    reloaded,_,_=load_checkpoint(out/'smoke_only.pt',m)
    if policy_hash(reloaded)!=policy_hash(policy):raise ValueError('save/reload tensor mismatch')
    # Explicit deadline test with waiting requests and no output.
    e=envs[2];model=read(inside(root,e['model_path']));trace=numeric_trace(e,709_999_999)
    for i,r in enumerate(trace):r['scheduled_arrival_sec']=i*.0001
    slow=deepcopy(model);slow['prefill_ms']=2_000_000.
    failure=Q15BCohortSim(slow,trace,99,out/'deadline_failure',minimal_logging=True,finalize_deadline=True)
    try:
        summary,decs,events=failure.run('Stay',1.,reward_definition=MigPPOV2Reward.from_state_dict(read(root/'calibration/reward_definition.json')),episode_horizon_sec=200.)
        rr=read(out/'deadline_failure/reward.json')
        if any(r['outcome']!='timeout' for r in summary['requests']) or sum(w['failure_count'] for w in rr['windows'])!=e['request_count']:
            raise ValueError('deadline failures lost from shared reward')
        if rr['windows'][-1]['end_ns']!=200_000_000_000 or rr['terminal_potential']!=0:raise ValueError('deadline GAE terminal invalid')
    finally:failure.event_file.close()
    result=dict(status='passed',scope='disposable CPU contract smoke; not policy performance',
        native_windows_executed=platform.system()=='Windows',environment_count=13,episodes=reports,
        update=update,policy_changed=before!=policy_hash(policy),save_reload_identical=True,
        deadline_failure_count=e['request_count'],elapsed_sec=time.monotonic()-started,
        gpu_runs=0,formal_ppo_ready=False)
    atomic_json(out/'smoke_report.json',result);return result


@contextmanager
def run_lock(out,recover=False):
    path=Path(out)/'.training.lock'
    if recover and path.exists():path.unlink()
    try:fd=os.open(str(path),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:
        raise RuntimeError('Run is locked. If its old process exited, resume with --recover-lock.') from exc
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),host=platform.node()),f)
        yield
    finally:path.unlink(missing_ok=True)


def evaluation_spec(envs):
    specs={}
    for split,offset in [('development',0),('heldout',10000)]:
        specs[split]=[dict(env_id=e['env_id'],trace_seed=900_000_000+offset+k,
            action_seeds=[910_000_000+offset+k*10+j for j in range(2)]) for e in envs for k in range(2)]
    return dict(schema='numeric-trace-split-v1',generator='fixed-n-uniform-count-hotspot-v1',splits=specs,
                independent_real_validation_pending=True)


def _cached_eval_episode(root,env,case,out,policy,version,action_seed,baseline=None):
    key=dict(env_id=env['env_id'],trace_seed=case['trace_seed'],action_seed=action_seed if policy is not None else None,
        policy_sha256=policy_hash(policy) if policy is not None else baseline,
        model_sha256=sha(inside(root,env['model_path'])),reward_sha256=sha(root/'calibration/reward_definition.json'))
    if (out/'metrics.json').exists():
        if read(out/'evaluation_input.json')!=key:raise ValueError('evaluation cache belongs to other inputs')
        return read(out/'metrics.json')
    if out.exists():
        # Preserve an interrupted attempt; same inputs and seeds are retried.
        attempt=1
        while Path(str(out)+f'.interrupted_{attempt}').exists():attempt+=1
        out.rename(Path(str(out)+f'.interrupted_{attempt}'))
    if policy is not None:torch.manual_seed(action_seed)
    result,_=run_episode(root,env,case['trace_seed'],out,policy,version,baseline)
    atomic_json(out/'evaluation_input.json',key)
    return result


def evaluate(bundle_root,run_dir,split='development',checkpoint_path=None,deadline=None):
    root,m,envs,_=load_bundle(bundle_root);out=Path(run_dir).resolve()
    spec=read(out/'evaluation_spec.json')
    cp=Path(checkpoint_path) if checkpoint_path is not None else out/'checkpoints/selected.pt'
    if not cp.exists():raise ValueError('no selected checkpoint; complete development evaluation first')
    by_env={e['env_id']:e for e in envs};rows=[]
    with torch.random.fork_rng(devices=[]):
        policy,_,ck=load_checkpoint(cp,m);version=ck['state']['next_update'];ph=policy_hash(policy)
        destination=out/'evaluation'/split/ph
        if split=='heldout':
            frozen=dict(policy_sha256=ph,evaluation_spec_sha256=sha(out/'evaluation_spec.json'))
            path=out/'heldout_selection.json'
            if path.exists() and read(path)!=frozen:raise ValueError('heldout selection already frozen; cannot select another policy using test results')
            atomic_json(path,frozen)
        for case in spec['splits'][split]:
            env=by_env[case['env_id']];case_result=dict(case=case,policies={})
            for baseline in ('Stay','Balanced'):
                if deadline is not None and time.monotonic()>=deadline:return dict(status='budget_exhausted',completed_cases=len(rows))
                directory=out/'evaluation/baselines'/split/env['env_id']/str(case['trace_seed'])/baseline
                case_result['policies'][baseline]=_cached_eval_episode(root,env,case,directory,None,0,None,baseline)
            sampled=[]
            for action_seed in case['action_seeds']:
                if deadline is not None and time.monotonic()>=deadline:return dict(status='budget_exhausted',completed_cases=len(rows))
                directory=destination/env['env_id']/str(case['trace_seed'])/str(action_seed)
                sampled.append(_cached_eval_episode(root,env,case,directory,policy,version,action_seed))
            case_result['policies']['PPO']={k:float(np.mean([r[k] for r in sampled])) for k in ('return_','C_ms','mean_e2e_ms','p95_e2e_ms','failure_rate','migration_count')}
            case_result['action_seed_results']=sampled;rows.append(case_result)
        per_env={}
        for eid in by_env:
            rr=[r for r in rows if r['case']['env_id']==eid]
            per_env[eid]={p:{key:float(np.mean([r['policies'][p][key] for r in rr])) for key in ('return_','C_ms','mean_e2e_ms','p95_e2e_ms','failure_rate','migration_count')} for p in ('PPO','Stay','Balanced')}
        macro={p:{key:float(np.mean([v[p][key] for v in per_env.values()])) for key in per_env[envs[0]['env_id']][p]} for p in ('PPO','Stay','Balanced')}
        result=dict(status='passed',scope='simulation development only; no independent real validation',split=split,
            policy_sha256=ph,checkpoint_version=version,per_environment=per_env,macro=macro,cases=rows,
            real_validation_pending=True,formal_ppo_ready=False)
        atomic_json(destination/'report.json',result);return result


def train(bundle_root,run_dir,config_path=None,resume=None,max_updates=None,wall_budget_sec=None,
          skip_evaluation=False,recover_lock=False):
    root,m,envs,_=load_bundle(bundle_root);out=Path(run_dir).resolve()
    cfg=read(config_path or root/'configs/local_train.json')
    if cfg.get('scope')!='local_simulation_development' or cfg['settings']!=SETTINGS:
        raise ValueError('frozen local development PPO settings required')
    if max_updates is not None:cfg['max_updates']=int(max_updates)
    if wall_budget_sec is not None:cfg['wall_budget_sec']=float(wall_budget_sec)
    if cfg['max_updates']<1 or cfg['wall_budget_sec']<=0:raise ValueError('positive training budget required')
    torch.set_num_threads(cfg['torch_threads'])
    out.mkdir(parents=True,exist_ok=bool(resume))
    by_env={e['env_id']:e for e in envs};rng=random.Random()
    with run_lock(out,recover_lock):
        if resume:
            cp=out/'checkpoints/latest.pt' if resume=='latest' else Path(resume)
            policy,opt,ck=load_checkpoint(cp,m)
            frozen_cfg={k:v for k,v in ck['config'].items() if k not in ('max_updates','wall_budget_sec')}
            if frozen_cfg!={k:v for k,v in cfg.items() if k not in ('max_updates','wall_budget_sec')}:
                raise ValueError('only total update/wall budgets may change on resume')
            state=ck['state'];restore_rng(ck,rng)
            if max_updates is None:cfg['max_updates']=ck['config']['max_updates']
            if wall_budget_sec is None:cfg['wall_budget_sec']=ck['config']['wall_budget_sec']
            clock_file=out/'active_clock.json'
            if clock_file.exists():
                state['elapsed_sec']=max(state['elapsed_sec'],read(clock_file)['elapsed_sec'])
        else:
            policy,opt,ck=load_checkpoint(root/'checkpoints/initial_v2.pt',m);restore_rng(ck,rng)
            state=dict(next_update=0,next_episode=0,pending=[],inflight=None,elapsed_sec=0.,history=[],evaluated=[],best=None)
            atomic_json(out/'run_manifest.json',dict(bundle_contract_sha256=m['contract_sha256'],config=cfg,scope='simulation',formal_ppo_ready=False))
            shutil.copyfile(root/'traces/evaluation_spec.json',out/'evaluation_spec.json')
            checkpoint(out/'checkpoints/initial_v2.pt',policy,opt,m,state,rng,cfg)
        atomic_json(out/'config_effective.json',cfg)
        spec=read(out/'evaluation_spec.json')
        split_groups={split:{trace_group(numeric_trace(by_env[c['env_id']],c['trace_seed']))
                            for c in cases} for split,cases in spec['splits'].items()}
        if split_groups['development'] & split_groups['heldout']:
            raise ValueError('development and heldout base workloads overlap')
        evaluation_groups=split_groups['development'] | split_groups['heldout']
        start=time.monotonic();previous_elapsed=state['elapsed_sec'];deadline=start+max(0.,cfg['wall_budget_sec']-previous_elapsed)
        def persist():
            state['elapsed_sec']=previous_elapsed+time.monotonic()-start
            checkpoint(out/'checkpoints/latest.pt',policy,opt,m,state,rng,cfg)
        def assess(cp):
            result=evaluate(root,out,'development',cp,deadline)
            if result['status']!='passed':return False
            metrics=result['macro']['PPO'];score=(metrics['return_'],-metrics['C_ms'])
            if state['best'] is None:state['initial_failure_rate']=metrics['failure_rate']
            if metrics['failure_rate']<=state.get('initial_failure_rate',0.)+1e-12 and (state['best'] is None or score>tuple(state['best']['score'])):
                state['best']=dict(score=list(score),policy_sha256=result['policy_sha256'],update=state['next_update'])
                temp=out/'checkpoints/selected.pt.tmp';shutil.copyfile(cp,temp);os.replace(temp,out/'checkpoints/selected.pt')
            state['evaluated'].append(state['next_update']);persist();return True
        persist()
        status='completed';error=None
        try:
            version=state['next_update']
            if not skip_evaluation and version not in state['evaluated'] and (
                version==0 or version%cfg['evaluate_every_updates']==0 or version>=cfg['max_updates']):
                if not assess(out/'checkpoints/latest.pt'):status='budget_exhausted'
            while status=='completed' and state['next_update']<cfg['max_updates']:
                if time.monotonic()>=deadline:status='budget_exhausted';break
                while len(state['pending'])<cfg['episodes_per_update']:
                    if time.monotonic()>=deadline:status='budget_exhausted';break
                    if state['inflight'] is None:
                        env=rng.choice(envs)
                        state['inflight']=dict(env_id=env['env_id'],trace_seed=cfg['training_trace_seed_base']+state['next_episode'])
                        persist()  # RNG/behavior state before sampling survives Ctrl-C.
                    current=state['inflight'];env=by_env[current['env_id']]
                    if trace_group(numeric_trace(env,current['trace_seed'])) in evaluation_groups:
                        raise ValueError('training workload overlaps an evaluation split')
                    prefix=f"u{state['next_update']:05d}_e{state['next_episode']:06d}"
                    attempts=len(list((out/'episodes').glob(prefix+'_attempt_*'))) if (out/'episodes').exists() else 0
                    episode=out/'episodes'/(prefix+f'_attempt_{attempts:02d}')
                    metrics,_=run_episode(root,env,current['trace_seed'],episode,policy,state['next_update'],wall_limit=cfg['episode_wall_limit_sec'])
                    item=dict(**current,rollout_path=(episode/'rollout.json.gz').relative_to(out).as_posix(),metrics=metrics)
                    state['pending'].append(item);state['history'].append(dict(**current,update=state['next_update']))
                    state['next_episode']+=1;state['inflight']=None;persist()
                    print(f"episode {state['next_episode']}: {env['env_id']} C={metrics['C_ms']:.1f} failures={metrics['failure_count']}",flush=True)
                if status!='completed':break
                # Latest checkpoint is pre-update; partial optimizer work never replaces it.
                rollouts=[read_rollout(inside(out,e['rollout_path'])) for e in state['pending']]
                update=ppo_update(policy,opt,rollouts,cfg['settings'],seed=cfg['update_seed']+state['next_update'])
                state['next_update']+=1;state['pending']=[]
                atomic_json(out/'updates'/f"{state['next_update']:05d}.json",update)
                persist();cp=out/'checkpoints'/f"update_{state['next_update']:05d}.pt"
                shutil.copyfile(out/'checkpoints/latest.pt',cp)
                if not skip_evaluation and (state['next_update']%cfg['evaluate_every_updates']==0 or state['next_update']==cfg['max_updates']):
                    if not assess(cp):status='budget_exhausted';break
        except KeyboardInterrupt:
            status='interrupted';error='Resume latest.pt; in-flight episode/partial optimizer work is replayed from its pre-sampling/pre-update checkpoint.'
        except Exception as exc:
            status='failed';error=type(exc).__name__+': '+str(exc)
            atomic_json(out/'training_error.json',dict(error=error))
        # Never save possibly half-updated memory on exception.
        if status not in ('failed','interrupted'):persist()
        atomic_json(out/'active_clock.json',dict(elapsed_sec=previous_elapsed+time.monotonic()-start))
        latest=torch.load(out/'checkpoints/latest.pt',map_location='cpu',weights_only=False)['state']
        result=dict(status=status,error=error,completed_updates=latest['next_update'],completed_episodes=latest['next_episode'],
            pending_batch_episodes=len(latest['pending']),inflight_episode=latest['inflight'],
            active_wall_sec=previous_elapsed+time.monotonic()-start,environment_coverage=dict(Counter(x['env_id'] for x in latest['history'])),
            candidate_selection='development_selected' if latest['best'] else 'not_evaluated',
            native_windows_executed=platform.system()=='Windows',scope='local_simulation_development',
            formal_ppo_ready=False,independent_real_validation_pending=True)
        atomic_json(out/'local_report.json',result)
        if status=='failed':raise RuntimeError(error)
        return result


def export_policy(bundle_root,run_dir,output_dir):
    root,m,envs,_=load_bundle(bundle_root);run=Path(run_dir).resolve();out=Path(output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    cp=run/'checkpoints/selected.pt';selected=cp.exists()
    if not selected:cp=run/'checkpoints/latest.pt'
    policy,_,ck=load_checkpoint(cp,m)
    torch.save(dict(schema='mig-ppo-v2-portable-policy-v1',policy=policy.state_dict(),normalizer=normalizer_definition(),
        hidden_size=64,policy_sha256=policy_hash(policy),source='simulation',formal_ppo_ready=False),out/'policy.pt')
    for src,dst in [('calibration/reward_definition.json','reward_definition.json'),('configs/environments.json','environments.json'),('manifest.json','simulator_manifest.json')]:shutil.copyfile(root/src,out/dst)
    atomic_json(out/'normalizer.json',normalizer_definition())
    shutil.copyfile(run/'evaluation_spec.json',out/'evaluation_spec.json')
    ref=read(root/'validation/policy_reference.json')
    with torch.no_grad():dist,val=policy.distribution(*tensors([ref['observation']]))
    atomic_json(out/'diagnostic_outputs.json',dict(observation=ref['observation'],probabilities=dist.probs[0].tolist(),value=val.item()))
    atomic_json(out/'policy_config.json',dict(schema='mig-ppo-v2',hidden_size=64,action_pairs=list(PAIRS),sampling='stochastic',scope='simulation_development_only'))
    reports=list((run/'evaluation').glob('*/'+policy_hash(policy)+'/report.json'))
    for report_path in reports:shutil.copyfile(report_path,out/(report_path.parent.parent.name+'_evaluation.json'))
    atomic_json(out/'export_report.json',dict(status='exported',development_selected=selected,policy_sha256=policy_hash(policy),
        independent_real_validation_pending=True,formal_ppo_ready=False,real_validation='not_run'))
    write_checksums(out)
    return dict(status='exported',output_dir=str(out),development_selected=selected,formal_ppo_ready=False)


def write_checksums(root):
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts)
    (root/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(root).as_posix()+'\n' for p in files),encoding='utf-8')


def portable_export(output_dir,calibration_dir=None):
    """Create a real ZIP with portable sources, weights and 13 frozen models."""
    source=Path(calibration_dir or DEFAULT_CALIBRATION).resolve()
    upstream=read(source/'model_registry.json');by_pair={r['pair_id']:r for r in upstream['environment_models']}
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    bundle=out/'mig_ppo_v2_cpu';bundle.mkdir()
    for name in CPU_FILES:
        target=bundle/'rl_scheduler'/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'rl_scheduler'/name,target)
    (bundle/'benchmark').mkdir();(bundle/'benchmark/__init__.py').write_text('"""Portable CPU entrypoints."""\n',encoding='utf-8')
    for name in ('run_ppo_sim.py','ppo_local.py'):shutil.copyfile(ROOT/'benchmark'/name,bundle/'benchmark'/name)
    for p in (ROOT/'benchmark/windows_bundle').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc':
            target=bundle/p.relative_to(ROOT/'benchmark/windows_bundle');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    environments=[]
    for eid,n,qps,f,pair in ENV_SPECS:
        entry=by_pair[pair];path=source/entry['model_path']
        if sha(path)!=entry['sha256']:raise ValueError('frozen upstream model changed: '+pair)
        relative='calibration/'+eid+'.json';target=bundle/relative;target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
        environments.append(dict(env_id=eid,request_count=n,qps=qps,hot_fraction=f,
            hot_index=int(eid[3])-1 if eid!='NATIVE' else None,probability=1/13,model_path=relative,
            source_pair_id=pair,source_model_sha256=sha(path),source_all_modes_passed=entry['all_modes_passed'],
            local_role='training_and_development_evaluation',upstream_native_test_only=entry['native_test_only']))
    envcfg=dict(schema='mig-ppo-v2-thirteen-environments-v1',sampling='uniform_with_replacement',environments=environments,
        native_role_change='User requested all thirteen for local development; historical test-only registry is not modified.')
    atomic_json(bundle/'configs/environments.json',envcfg)
    cfg=dict(scope='local_simulation_development',allow_unvalidated_physics=True,formal_ppo_ready=False,
        max_updates=16,episodes_per_update=4,wall_budget_sec=1800.,episode_wall_limit_sec=120.,
        torch_threads=2,evaluate_every_updates=8,training_trace_seed_base=80_000_000,
        policy_seed=133702,environment_seed=133701,update_seed=133704,settings=SETTINGS)
    atomic_json(bundle/'configs/local_train.json',cfg)
    shutil.copyfile(source/'reward_definition.json',bundle/'calibration/reward_definition.json')
    shutil.copyfile(source/'physical_model.json',bundle/'calibration/shared_physical_model.json')
    (bundle/'validation').mkdir(exist_ok=True)
    for name in ('report.md','accuracy_summary.json','model_registry.json','integrity_check.json','method_comparison.json'):
        shutil.copyfile(source/name,bundle/'validation'/('upstream_'+name))
    atomic_json(bundle/'traces/evaluation_spec.json',evaluation_spec(environments))
    source_hashes={p.relative_to(bundle).as_posix():sha(p) for dirname in ('rl_scheduler','benchmark','calibration')
                   for p in sorted((bundle/dirname).rglob('*')) if p.is_file()}
    contract=digest(dict(sources=source_hashes,environments=envcfg,reward=read(bundle/'calibration/reward_definition.json')))
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit='unavailable'
    manifest=dict(schema=BUNDLE_SCHEMA,contract_sha256=contract,source_commit=commit,source_file_hashes=source_hashes,
        source_calibration_registry_sha256=sha(source/'model_registry.json'),
        target='Native Windows 10/11 x64 and Linux x86_64, CPython 3.10-3.12, CPU',
        windows_native_validation='not_run_on_export_host',export_host=platform.platform(),
        hidden_size=64,normalizer=normalizer_definition(),decision_interval_sec=1.5,migration_cooldown_sec=0.,
        episode_horizon_sec=200.,environment_count=13,formal_ppo_ready=False,independent_real_validation_pending=True,
        exported_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(bundle/'manifest.json',manifest)
    policy=new_policy(cfg['policy_seed']);opt=torch.optim.Adam(policy.parameters(),lr=SETTINGS['learning_rate']);rng=random.Random(cfg['environment_seed'])
    state=dict(next_update=0,next_episode=0,pending=[],inflight=None,elapsed_sec=0.,history=[],evaluated=[],best=None)
    checkpoint(bundle/'checkpoints/initial_v2.pt',policy,opt,manifest,state,rng,cfg)
    # A fixed, nontrivial normalized state for cross-platform numerical checks.
    from rl_scheduler.reward import MigPPORewardMetrics
    env=environments[2];sim=Q15BCohortSim(read(bundle/env['model_path']),numeric_trace(env,699_999_991),123,
        out/'reference_temporary',minimal_logging=True)
    try:
        sim.episode_horizon_sec=200.
        sim._advance(MigPPORewardMetrics(window_sec=12.,instance_count=4),0.,12.)
        raw,_=sim.raw();record=FrozenController(policy,normalizer_definition(),0,True)(raw,0)
        with torch.no_grad():dist,val=policy.distribution(*tensors([record['observation']]))
        atomic_json(bundle/'validation/policy_reference.json',dict(observation=record['observation'],
            probabilities=dist.probs[0].tolist(),value=val.item(),atol=1e-6,rtol=1e-5))
    finally:sim.event_file.close()
    shutil.rmtree(out/'reference_temporary')
    write_checksums(bundle)
    archive=out/'mig_ppo_v2_cpu_windows_linux.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(out).as_posix())
    result=dict(status='exported',bundle_root=str(bundle),zip_path=str(archive),bytes=archive.stat().st_size,
        zip_sha256=sha(archive),environment_count=13,source_schema='mig-ppo-v2',
        windows_native_validation='pending user machine doctor and smoke',gpu_runs=0,formal_ppo_ready=False)
    atomic_json(out/'export_report.json',result)
    (out/(archive.name+'.sha256')).write_text(result['zip_sha256']+'  '+archive.name+'\n',encoding='utf-8')
    return result


def dispatch(args):
    if args.command=='portable-export':return portable_export(args.output_dir,args.calibration_dir)
    if args.command=='local-doctor':
        result=doctor(args.bundle_root)
        if args.output_file:atomic_json(args.output_file,result)
        return result
    if args.command=='local-smoke':return smoke(args.bundle_root,args.run_dir)
    if args.command=='local-train':
        return train(args.bundle_root,args.run_dir,args.config,args.resume,args.max_updates,args.wall_budget_sec,
                     args.skip_evaluation,args.recover_lock)
    if args.command=='local-evaluate':return evaluate(args.bundle_root,args.run_dir,args.split,args.checkpoint)
    if args.command=='local-export-policy':return export_policy(args.bundle_root,args.run_dir,args.output_dir)
    raise ValueError('unknown portable command')
