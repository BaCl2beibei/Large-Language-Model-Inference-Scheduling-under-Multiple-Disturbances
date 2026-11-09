#!/usr/bin/env python3
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, subprocess, sys, time, traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
RUN_DIR = Path(__file__).resolve().parent
BUNDLE = ROOT / 'benchmark/results/migppo_local_001_best_v304'
SPEC_PATH = RUN_DIR / 'external_policy_test_spec.json'
RESULTS_PATH = RUN_DIR / 'external_policy_full_pilot_results.jsonl'
SUMMARY_PATH = RUN_DIR / 'external_policy_full_pilot_summary.json'
POLICY_VERSION = 'migppo-local-v304-dc558f13'
MAX_TOTAL_SEC = 14400.0
MAX_PILOT_SEC = 3600.0
EPISODE_HORIZON_SEC = 200.0
DECISION_INTERVAL_SEC = 1.5
MIGRATION_COOLDOWN_SEC = 0.0
PILOT_ENV_IDS = ('NATIVE', 'HOT2-H')
POLICY_ORDERS = (('Stay', 'Balanced', 'PPO'),
                 ('Balanced', 'PPO', 'Stay'),
                 ('PPO', 'Stay', 'Balanced'))

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def prepare():
    import torch
    from tokenizers import Tokenizer
    from benchmark import mig_ppo_v2_multienv as multienv
    from rl_scheduler.cohort_ppo import CohortActorCritic, normalizer_definition, policy_hash
    from rl_scheduler.real_bridge import PAIRS

    if SPEC_PATH.exists() or RESULTS_PATH.exists():
        raise FileExistsError('evaluation artifacts already exist; preserve them')
    plan = load_json(RUN_DIR / 'plan.json')
    package_env = load_json(BUNDLE / 'environments.json')
    envs = package_env['environments']
    env_ids = [x['env_id'] for x in envs]
    if len(envs) != 13 or env_ids != multienv.ENV_IDS:
        raise RuntimeError('uploaded environment list differs from registered 13-environment order')
    saved = torch.load(BUNDLE / 'policy.pt', map_location='cpu', weights_only=False)
    model = CohortActorCritic(hidden_size=saved['hidden_size']).cpu().eval()
    model.load_state_dict(saved['policy'], strict=True)
    if policy_hash(model) != saved['policy_sha256']:
        raise RuntimeError('policy hash mismatch')
    if saved['normalizer'] != normalizer_definition():
        raise RuntimeError('exported normalizer differs from real observation normalizer')
    policy_cfg = load_json(BUNDLE / 'policy_config.json')
    if policy_cfg['action_pairs'] != [list(x) for x in PAIRS]:
        raise RuntimeError('action pair mapping mismatch')
    if policy_cfg.get('sampling') != 'stochastic':
        raise RuntimeError('unexpected policy sampling mode')
    model_path = ROOT / 'model/Qwen/Qwen2-1.5B'
    tokenizer = Tokenizer.from_file(str(model_path / 'tokenizer.json'))
    arrival_seed = int(plan['evaluation_seed'])
    prompt_seed = int(plan['validation_seed'])
    scenarios = []
    for profile in envs:
        env_id = profile['env_id']
        candidate_id = f'v304_{env_id}_a{arrival_seed}_p{prompt_seed}'
        out = RUN_DIR / 'traces' / 'v304_real_eval' / env_id / (candidate_id + '.json')
        meta = multienv._make_candidate_trace(
            tokenizer=tokenizer, candidate_id=candidate_id,
            n=int(profile['request_count']), qps=float(profile['qps']),
            arrival_seed=arrival_seed, prompt_seed=prompt_seed,
            environment_id=env_id, output_dir=out,
            hot_fraction=profile.get('hot_fraction'), hot_index=profile.get('hot_index'))
        meta.update({
            'env_id': env_id, 'request_count': int(profile['request_count']),
            'qps': float(profile['qps']), 'hot_fraction': profile.get('hot_fraction'),
            'hot_index': profile.get('hot_index'),
            'profile_source_pair_id': profile.get('source_pair_id'),
            'trace_sha256': sha_file(out),
            'trace_path': str(out.relative_to(ROOT)),
        })
        scenarios.append(meta)
    spec = {
        'schema': 'migppo-v304-real-13-env-evaluation-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'run_id': plan['run_id'],
        'policy': {
            'bundle_dir': str(BUNDLE.relative_to(ROOT)),
            'bundle_zip_sha256': sha_file(ROOT / 'benchmark/results/migppo_local_001_best_v304.zip'),
            'policy_sha256': saved['policy_sha256'],
            'policy_file_sha256': sha_file(BUNDLE / 'policy.pt'),
            'normalizer_sha256': sha_file(BUNDLE / 'normalizer.json'),
            'config': policy_cfg,
            'selection_scope': 'simulation development only; independent real validation pending',
        },
        'source_code': {
            'test_runner_sha256': sha_file(__file__),
            'multienv_module_sha256': sha_file(ROOT / 'benchmark/mig_ppo_v2_multienv.py'),
            'full_completion_sha256': sha_file(ROOT / 'rl_scheduler/full_completion.py'),
            'real_observation_sha256': sha_file(ROOT / 'rl_scheduler/real_observation.py'),
            'cohort_ppo_sha256': sha_file(ROOT / 'rl_scheduler/cohort_ppo.py'),
        },
        'seeds': {'arrival': arrival_seed, 'prompt': prompt_seed, 'ppo_torch_seed_base': 26110905},
        'comparison': {
            'strategies': ['Stay', 'Balanced', 'PPO'],
            'same_trace_for_all_strategies': True,
            'profile_source': str((BUNDLE / 'environments.json').relative_to(ROOT)),
            'decision_interval_sec': DECISION_INTERVAL_SEC,
            'migration_cooldown_sec': MIGRATION_COOLDOWN_SEC,
            'episode_horizon_sec': EPISODE_HORIZON_SEC,
            'ppo_execution': 'active real migrations using the frozen uploaded policy',
            'policy_sampling': 'stochastic; one fixed independent torch seed per environment',
            'method_order': 'rotating counterbalance across the registered 13-environment order',
            'audit_adapter': 'read durable ppo_action_intents.jsonl into the existing PPO execution audit',
        },
        'resource_limits': {
            'total_gpu_service_wall_sec': MAX_TOTAL_SEC,
            'pilot_gpu_service_wall_sec': MAX_PILOT_SEC,
            'visible_gpu_count': 4,
            'pilot_env_ids': list(PILOT_ENV_IDS),
        },
        'pilot_advance_gate': {
            'both_pilot_envs_all_3_strategies_complete_and_audited': True,
            'PPO_macro_C_ms_lower_than_Balanced': True,
            'PPO_C_ms_no_more_than_5_percent_worse_per_pilot_env': True,
            'PPO_request_failures_not_greater_than_Balanced_per_env': True,
            'unknown_or_unreconciled_migrations': 0,
            'otherwise': 'stop after pilot; do not run other 11 environments',
        },
        'scenarios': scenarios,
    }
    spec['spec_sha256'] = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    atomic_json(SPEC_PATH, spec)
    RESULTS_PATH.touch()
    print(json.dumps({
        'status': 'prepared', 'run_dir': str(RUN_DIR), 'run_id': plan['run_id'],
        'policy_sha256': saved['policy_sha256'], 'environment_count': len(scenarios),
        'pilot_envs': list(PILOT_ENV_IDS), 'arrival_seed': arrival_seed,
        'prompt_seed': prompt_seed,
        'trace_hashes': {x['env_id']: x['trace_sha256'] for x in scenarios},
        'spec_sha256': spec['spec_sha256'],
    }, ensure_ascii=False, indent=2))

def pilot_gate(rows):
    reasons = []
    required = {(env_id, strategy) for env_id in PILOT_ENV_IDS for strategy in ('Stay', 'Balanced', 'PPO')}
    found = {(r['environment_id'], r['strategy']) for r in rows}
    if not required.issubset(found):
        reasons.append('pilot lacks required environment/strategy episodes')
        return {'passed': False, 'reasons': reasons}
    chosen = [r for r in rows if (r['environment_id'], r['strategy']) in required]
    for row in chosen:
        if row.get('status') != 'passed' or row.get('cohort_complete') is not True:
            reasons.append(f"{row['environment_id']}/{row['strategy']} episode incomplete")
        if row.get('execution_audit_status') != 'passed':
            reasons.append(f"{row['environment_id']}/{row['strategy']} execution audit not passed")
        if row.get('unknown_migrations', 0):
            reasons.append(f"{row['environment_id']}/{row['strategy']} has unknown migrations")
        if row.get('C_ms') is None:
            reasons.append(f"{row['environment_id']}/{row['strategy']} lacks primary C metric")
    if reasons: return {'passed': False, 'reasons': reasons}
    by = {(r['environment_id'], r['strategy']): r for r in chosen}
    ppo_costs = [by[(env, 'PPO')]['C_ms'] for env in PILOT_ENV_IDS]
    balanced_costs = [by[(env, 'Balanced')]['C_ms'] for env in PILOT_ENV_IDS]
    macro_gain = (sum(balanced_costs)-sum(ppo_costs))/sum(balanced_costs) if sum(balanced_costs) else None
    per_env_gain = {}
    for env in PILOT_ENV_IDS:
        ppo = by[(env, 'PPO')]
        balanced = by[(env, 'Balanced')]
        per_env_gain[env] = (balanced['C_ms']-ppo['C_ms'])/balanced['C_ms'] if balanced['C_ms'] else None
        if ppo['request_failures'] > balanced['request_failures']:
            reasons.append(f'{env}: PPO request failures exceed Balanced')
        if per_env_gain[env] is None or per_env_gain[env] < -0.05:
            reasons.append(f'{env}: PPO C more than 5% worse than Balanced')
    if macro_gain is None or macro_gain <= 0:
        reasons.append('PPO macro C is not lower than Balanced')
    return {'passed': not reasons, 'reasons': reasons, 'macro_gain_vs_balanced': macro_gain,
            'per_environment_gain_vs_balanced': per_env_gain,
            'C_ms': {'PPO': ppo_costs, 'Balanced': balanced_costs}}

def write_summary(rows, status, extra=None):
    by_env = {}
    for row in rows:
        by_env.setdefault(row['environment_id'], {})[row['strategy']] = row
    macro = {}
    for strategy in ('Stay', 'Balanced', 'PPO'):
        valid = [case[strategy] for case in by_env.values()
                 if strategy in case and case[strategy].get('C_ms') is not None]
        if valid:
            macro[strategy] = {
                'environment_count': len(valid),
                'mean_C_ms': sum(x['C_ms'] for x in valid)/len(valid),
                'mean_migration_attempts': sum(x['actual_migration_attempts'] for x in valid)/len(valid),
                'mean_successful_migrations': sum(x['migration_successes'] for x in valid)/len(valid),
                'mean_request_failures': sum(x['request_failures'] for x in valid)/len(valid),
            }
    summary = {
        'schema': 'migppo-v304-real-13-results-v1', 'status': status,
        'policy_sha256': load_json(BUNDLE/'export_report.json')['policy_sha256'],
        'environment_count_requested': 13, 'environment_count_completed': len(by_env),
        'pilot_gate': extra.get('pilot_gate') if extra else None,
        'macro': macro, 'environments': by_env,
        'updated_utc': datetime.now(timezone.utc).isoformat(),
    }
    if extra: summary.update(extra)
    atomic_json(SUMMARY_PATH, summary)
    return summary

def append_result(row):
    with RESULTS_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)+'\n')
        f.flush(); os.fsync(f.fileno())

async def execute():
    import aiohttp, torch, yaml
    import rl_scheduler.full_completion as full_completion
    import rl_scheduler.cohort_ppo as cohort_ppo
    from tokenizers import Tokenizer
    from benchmark import mig_ppo_v2_multienv as multienv
    from benchmark.run_full_completion_balance import ManagedFullCompletionService, _warmup_trace
    from rl_scheduler.cohort_ppo import CohortActorCritic, FrozenController, policy_hash
    from rl_scheduler.real_bridge import RealBridge

    if not SPEC_PATH.is_file(): raise FileNotFoundError('run --prepare first')
    existing_rows=[json.loads(line) for line in RESULTS_PATH.read_text(encoding='utf-8').splitlines() if line.strip()] if RESULTS_PATH.exists() else []
    prior={(r.get('environment_id'),r.get('strategy')) for r in existing_rows}
    required_native={('NATIVE','Stay'),('NATIVE','Balanced'),('NATIVE','PPO')}
    if prior != required_native or len(existing_rows)!=3: raise RuntimeError('resume file must contain exactly the three prior NATIVE results')
    if not RESULTS_PATH.exists(): RESULTS_PATH.touch()
    spec=load_json(SPEC_PATH)
    plan=load_json(RUN_DIR/'plan.json')
    package=torch.load(BUNDLE/'policy.pt',map_location='cpu',weights_only=False)
    model=CohortActorCritic(hidden_size=package['hidden_size']).cpu().eval()
    model.load_state_dict(package['policy'],strict=True)
    if policy_hash(model)!=spec['policy']['policy_sha256']: raise RuntimeError('policy changed after preparation')
    if package['normalizer']!=load_json(BUNDLE/'normalizer.json'): raise RuntimeError('normalizer file mismatch')
    current={'controller':None}
    original_run_episode=full_completion.run_episode
    original_audit=cohort_ppo.audit_execution
    async def injected_run_episode(*args,**kwargs):
        if kwargs.get('policy')=='PPO':
            if current['controller'] is None: raise RuntimeError('PPO controller not prepared')
            kwargs['decision_controller']=current['controller']
            kwargs['controller_version']=POLICY_VERSION
        return await original_run_episode(*args,**kwargs)
    def audit_with_ppo_intents(summary, decisions, event_rows, intent_rows):
        if summary.get('policy')=='PPO':
            intent_path=RUN_DIR/'runs'/summary['episode_id']/'ppo_action_intents.jsonl'
            if not intent_path.is_file(): raise RuntimeError('PPO action-intent evidence missing')
            intent_rows=[json.loads(line) for line in intent_path.read_text().splitlines() if line.strip()]
        return original_audit(summary,decisions,event_rows,intent_rows)
    full_completion.run_episode=injected_run_episode
    cohort_ppo.audit_execution=audit_with_ppo_intents

    cfg=yaml.safe_load((RUN_DIR/'runtime_config.yaml').read_text())
    tokenizer=Tokenizer.from_file(str(Path(cfg['service']['model_path'])/'tokenizer.json'))
    profiles={x['env_id']:x for x in load_json(BUNDLE/'environments.json')['environments']}
    scenarios={x['env_id']:x for x in spec['scenarios']}
    service=ManagedFullCompletionService(RUN_DIR)
    wall_start=time.time(); mono_start=time.monotonic()
    budget_before=load_json(RUN_DIR/'budget.json')
    used_gpu_sec=float(budget_before.get('gpu_active_sec',0.))
    total_deadline=mono_start+max(0.,MAX_TOTAL_SEC-used_gpu_sec); pilot_deadline=mono_start+max(0.,MAX_PILOT_SEC-used_gpu_sec)
    rows=existing_rows; gate=None; global_error=None; mapping=None; threshold=0.; active_start=None
    service_started=False
    try:
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],
                           capture_output=True,text=True,timeout=30)
        if gpu.returncode!=0 or gpu.stdout.strip(): raise RuntimeError('GPU is not idle before test; preserving active work')
        supervisor=subprocess.run(['supervisorctl','status'],capture_output=True,text=True,timeout=30)
        active=[line for line in supervisor.stdout.splitlines()
                if any(x in line for x in ('full_completion_','p23_')) and ' RUNNING ' in line]
        if active: raise RuntimeError('another Llumnix experiment is running: '+str(active))
        service_started=True
        service.start(); active_start=time.monotonic()
        await asyncio.wait_for(full_completion.wait_service_ready(cfg,service.is_running),
                               timeout=float(cfg['service'].get('startup_timeout_sec',480.)))
        timeout=aiohttp.ClientTimeout(total=None,connect=5.,sock_connect=5.)
        async with aiohttp.ClientSession(timeout=timeout,
                connector=aiohttp.TCPConnector(limit=4096,keepalive_timeout=3.)) as session:
            warm=_warmup_trace(tokenizer)
            warm_summary=await original_run_episode(
                cfg=cfg,run_id=plan['run_id'],episode_id='v304_real_test_warmup_hot2h_resume',
                policy='Stay',trace=warm,run_dir=RUN_DIR,control_session=session,
                frozen_mapping=None,service_probe=service.is_running,stop_owned_service=service.stop,
                episode_horizon_sec=EPISODE_HORIZON_SEC,decision_interval_sec=DECISION_INTERVAL_SEC,
                migration_cooldown_sec=MIGRATION_COOLDOWN_SEC,no_progress_timeout_sec=120.,
                hard_stop_monotonic=pilot_deadline)
            if warm_summary.get('status')!='passed' or not warm_summary.get('cohort_complete'):
                raise RuntimeError('warmup cohort failed: '+str(warm_summary.get('issues')))
            mapping=warm_summary['frozen_mapping']
            capacity=await RealBridge(cfg['service']['base_url'],session,instance_order=mapping).snapshot()
            capacities=[int(x.get('total_blocks') or 0) for x in capacity['instances']]
            if len(capacities)!=4 or min(capacities)<=0: raise RuntimeError('four measured KV capacities required')
            threshold=min(capacities)/8.
            atomic_json(RUN_DIR/'external_policy_runtime.json',{
                'service_started_utc':datetime.now(timezone.utc).isoformat(),
                'frozen_mapping':mapping,'kv_capacities':capacities,
                'balanced_threshold_min_capacity_over_8':threshold,
                'warmup_status':warm_summary.get('status'),'service_name':service.name})
            env_order=load_json(BUNDLE/'environments.json')['environments']
            order_index={x['env_id']:i for i,x in enumerate(env_order)}
            method_order={env:POLICY_ORDERS[order_index[env]%3] for env in profiles}
            for env_id in PILOT_ENV_IDS:
                done={(r['environment_id'],r['strategy']) for r in rows}
                if all((env_id,strategy) in done for strategy in ('Stay','Balanced','PPO')): continue
                ok=await run_environment(env_id,method_order[env_id],profiles,scenarios,
                    cfg=cfg,run_id=plan['run_id'],session=session,service=service,mapping=mapping,
                    threshold=threshold,controller_state=current,policy=model,normalizer=package['normalizer'],
                    policy_version=POLICY_VERSION,seed_base=int(spec['seeds']['ppo_torch_seed_base']),
                    deadline=pilot_deadline,rows=rows)
                if not ok: break
            gate=pilot_gate(rows)
            atomic_json(RUN_DIR/'pilot_gate.json',gate)
            write_summary(rows,'pilot_passed' if gate['passed'] else 'pilot_failed',
                          {'pilot_gate':gate,'continuation':'pending' if gate['passed'] else 'stopped'})
            if gate['passed']:
                for profile in env_order:
                    env_id=profile['env_id']
                    if env_id in PILOT_ENV_IDS: continue
                    ok=await run_environment(env_id,method_order[env_id],profiles,scenarios,
                        cfg=cfg,run_id=plan['run_id'],session=session,service=service,mapping=mapping,
                        threshold=threshold,controller_state=current,policy=model,normalizer=package['normalizer'],
                        policy_version=POLICY_VERSION,seed_base=int(spec['seeds']['ppo_torch_seed_base']),
                        deadline=total_deadline,rows=rows)
                    if not ok: break
            completed={(r['environment_id'],r['strategy']) for r in rows
                       if r.get('status')=='passed' and r.get('execution_audit_status')=='passed'}
            complete=all((env,strategy) in completed for env in profiles
                         for strategy in ('Stay','Balanced','PPO'))
            if not gate['passed']: status='pilot_failed'
            elif complete: status='complete'
            else: status='incomplete_after_pilot'
            write_summary(rows,status,{'pilot_gate':gate,
                'continuation':'completed_13_envs' if complete else ('not_run_pilot_gate_failed' if not gate['passed'] else 'partial'),
                'global_error':global_error})
    except BaseException as exc:
        global_error=type(exc).__name__+': '+str(exc)
        atomic_json(RUN_DIR/'external_policy_full_pilot_failure.json',{
            'status':'incomplete','reason':global_error,'traceback':traceback.format_exc(),
            'utc':datetime.now(timezone.utc).isoformat()})
        write_summary(rows,'incomplete',{'pilot_gate':gate,'global_error':global_error})
    finally:
        try: await asyncio.to_thread(service.stop)
        except BaseException as exc:
            global_error=(global_error+'; ' if global_error else '')+'cleanup:'+type(exc).__name__+':'+str(exc)
        wall_elapsed=max(0.,time.time()-wall_start)
        gpu_active=max(0.,time.monotonic()-active_start) if active_start else 0.
        budget=load_json(RUN_DIR/'budget.json')
        budget['gpu_reserved_origin_wall']=wall_start
        budget['gpu_reserved_elapsed_sec']=wall_elapsed
        budget['gpu_active_sec']=float(budget.get('gpu_active_sec',0.))+gpu_active
        budget['gpu_reserved_total_sec']=float(budget.get('prior_gpu_reservation_sec',0.))+wall_elapsed
        budget['gpu_hours']=(float(budget.get('prior_gpu_reservation_sec',0.))+gpu_active)*4/3600.
        budget.setdefault('phases',[]).append({
            'stage':'v304_external_policy_pilot_and_conditional_evaluation',
            'service_elapsed_sec':gpu_active,'wall_elapsed_sec':wall_elapsed,
            'ended_utc':datetime.now(timezone.utc).isoformat(),'episodes':len(rows),
            'gpu_hours_four_cards':budget['gpu_hours']})
        atomic_json(RUN_DIR/'budget.json',budget)
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],
                           capture_output=True,text=True,timeout=30)
        status=subprocess.run(['supervisorctl','status',service.name],capture_output=True,text=True,timeout=30)
        atomic_json(RUN_DIR/'cleanup_verification.json',{
            'service_stopped':not service.is_running(),'service_status':status.stdout.strip(),
            'gpu_processes_after':gpu.stdout.strip(),'wall_elapsed_sec':wall_elapsed,
            'gpu_active_sec':gpu_active,'gpu_hours_four_cards':budget['gpu_hours'],
            'verified_utc':datetime.now(timezone.utc).isoformat()})
    final=load_json(SUMMARY_PATH) if SUMMARY_PATH.exists() else {}
    print(json.dumps({'status':final.get('status'),'run_dir':str(RUN_DIR),
        'pilot_gate':final.get('pilot_gate'),'gpu_active_sec':final.get('gpu_active_sec')},
        ensure_ascii=False,indent=2),flush=True)
    return 0 if final.get('status') in ('complete','pilot_failed') else 1

async def run_environment(env_id,method_order,profiles,scenarios,*,cfg,run_id,session,
                          service,mapping,threshold,controller_state,policy,normalizer,
                          policy_version,seed_base,deadline,rows):
    import torch
    from benchmark import mig_ppo_v2_multienv as multienv
    from rl_scheduler.cohort_ppo import FrozenController
    spec=scenarios[env_id]
    trace_path=ROOT/spec['trace_path']
    if sha_file(trace_path)!=spec['trace_sha256']: raise RuntimeError('trace hash changed: '+env_id)
    trace=load_json(trace_path)
    for strategy in method_order:
        if time.monotonic()+EPISODE_HORIZON_SEC+120.>=deadline:
            atomic_json(RUN_DIR/'budget_stop.json',{
                'environment_id':env_id,'next_strategy':strategy,
                'reason':'insufficient time including cleanup reserve',
                'utc':datetime.now(timezone.utc).isoformat()})
            return False
        controller_state['controller']=None
        controller_seed=None
        if strategy=='PPO':
            controller_seed=seed_base+list(profiles).index(env_id)
            torch.manual_seed(controller_seed)
            controller_state['controller']=FrozenController(policy,normalizer,policy_version,deterministic=False)
        episode_id=f"v304_{env_id}_{strategy.lower()}"
        result,elapsed=await multienv._episode(
            cfg=cfg,run_dir=RUN_DIR,run_id=run_id,name=episode_id,
            policy=strategy,trace=trace,frozen_mapping=mapping,threshold=threshold,
            service=service,hard_deadline=deadline,traffic_session=session)
        metrics=multienv._metrics(result) or {}
        receipts=list((result.get('migration_receipts') or {}).values())
        terminals=Counter(x.get('terminal_status') for x in receipts if x.get('migration_id'))
        expected={'succeeded','failed','rejected'}
        unknown=sum(x.get('migration_id') is not None and x.get('terminal_status') not in expected for x in receipts)
        failures=sum(x.get('outcome')!='success' for x in (result.get('requests') or []))
        row={
            'environment_id':env_id,'strategy':strategy,'episode_id':episode_id,
            'controller_seed':controller_seed,'trace_sha256':spec['trace_sha256'],
            'profile':{k:profiles[env_id].get(k) for k in ('request_count','qps','hot_fraction','hot_index','source_pair_id')},
            'status':result.get('status'),'cohort_complete':result.get('cohort_complete'),
            'execution_audit_status':result.get('execution_audit',{}).get('status'),
            'planned_request_count':result.get('planned_request_count'),
            'measured_request_count':result.get('measured_request_count'),
            'request_failures':int(failures),'C_ms':metrics.get('C_ms'),
            'mean_e2e_ms':metrics.get('mean_e2e_ms'),'p95_e2e_ms':metrics.get('p95_e2e_ms'),
            'cvar95_e2e_ms':metrics.get('cvar95_e2e_ms'),
            'ttft_p95_ms':metrics.get('ttft_p95_ms'),'tpot_p95_ms':metrics.get('tpot_p95_ms'),
            'actual_migration_attempts':int(result.get('actual_migration_attempts') or 0),
            'migration_successes':int(terminals.get('succeeded',0)),
            'migration_failed':int(terminals.get('failed',0)),
            'migration_rejected':int(terminals.get('rejected',0)),
            'unknown_migrations':int(unknown),'migration_receipts':len(receipts),
            'wall_sec':float(elapsed),'route_audit_passed':result.get('route_audit_passed'),
            'control_issues':result.get('control_issues',[]),
        }
        rows.append(row);append_result(row)
        print(json.dumps({'environment':env_id,'strategy':strategy,'status':row['status'],
            'C_ms':row['C_ms'],'migrations':row['actual_migration_attempts'],
            'migration_successes':row['migration_successes'],'request_failures':row['request_failures'],
            'audit':row['execution_audit_status'],'wall_sec':round(elapsed,1)},
            ensure_ascii=False),flush=True)
        if row['status']!='passed' or not row['cohort_complete'] or row['execution_audit_status']!='passed':
            atomic_json(RUN_DIR/'episode_stop.json',{
                'environment_id':env_id,'strategy':strategy,
                'reason':'episode or execution audit incomplete; remaining non-pilot runs stopped',
                'row':row,'utc':datetime.now(timezone.utc).isoformat()})
            if env_id not in PILOT_ENV_IDS: return False
    return True

def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(); return 0
    return asyncio.run(execute())

if __name__=='__main__': raise SystemExit(main())
