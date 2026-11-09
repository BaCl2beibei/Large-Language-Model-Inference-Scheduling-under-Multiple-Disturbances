#!/usr/bin/env python3
"""Frozen, bounded Qwen2-1.5B P0/P1 screen. Never creates training rollouts."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_scheduler.full_completion import make_trace, run_episode, summarize_values, wait_service_ready
from rl_scheduler.real_bridge import RealBridge, PAIRS, digest, eligible_pair
from rl_scheduler.real_budget import atomic_json
from benchmark.run_full_completion_balance import ManagedFullCompletionService, _warmup_trace


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def command(*args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def environment():
    import importlib.metadata
    return {'wall': time.time(), 'gpu': command('nvidia-smi'),
            'gpu_uuids': command('nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv'),
            'gpu_processes': command('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv'),
            'supervisor': command('supervisorctl', 'status'),
            'versions': {k: importlib.metadata.version(k) for k in ('torch', 'ray', 'vllm', 'tokenizers')},
            'cpu_affinity_count': len(os.sched_getaffinity(0))}


def validate_config(cfg, server):
    svc = cfg['service']; argv = svc['server_argv']
    model = str(ROOT / 'model/Qwen/Qwen2-1.5B')
    expected = {'--model': model, '--tensor-parallel-size': '1', '--max-model-len': '2112',
                '--max-num-seqs': '128', '--gpu-memory-utilization': '0.65', '--dtype': 'half',
                '--max-num-batched-tokens': '2112', '--migration-backend': 'rayrpc'}
    for key, value in expected.items():
        if argv[argv.index(key)+1] != value:
            raise ValueError('config mismatch: '+key)
    assert svc['model_path'] == model and (Path(model)/'tokenizer.json').is_file()
    assert '--enforce-eager' in argv and svc['ray_num_cpus'] == 16
    assert server['SERVER']['REQUEST_OUTPUT_QUEUE_TYPE'] == 'rayqueue'
    assert server['INSTANCE']['REQUEST_MIGRATION_POLICY'] == 'SR'
    assert cfg['sampling'] == dict(n=1, best_of=1, temperature=0., top_k=1, ignore_eos=True)
    return expected


class MeasurementStay:
    """Same CPU architecture with seeded random weights; no behavior log-prob."""
    def __init__(self):
        import torch
        from rl_scheduler.cohort_ppo import CohortActorCritic, normalizer_definition
        torch.set_num_threads(1)
        torch.manual_seed(141100)
        self.net = CohortActorCritic().cpu().eval()
        self.normalizer = normalizer_definition(128.)
        self.weights_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in self.net.parameters())).hexdigest()

    def __call__(self, raw, start):
        import torch
        from rl_scheduler.cohort_ppo import causal_observation, tensors
        t = time.perf_counter()
        obs = causal_observation(raw, start, self.normalizer)
        inputs = tensors([obs])
        feature = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        with torch.no_grad():
            dist, value = self.net.distribution(*inputs)
        return {'action': 0, 'executed_action': 0, 'suggested_action': int(dist.probs.argmax(-1).item()),
                'measurement_only': True, 'network_path_only': True, 'value': float(value.item()),
                'probabilities': dist.probs[0].tolist(), 'cpu_conversion_ms': feature,
                'cpu_forward_ms': (time.perf_counter()-t)*1000, 'decision_ns': raw['decision_ns'],
                'observation': obs, 'weights_sha256': self.weights_hash}


def can_start(elapsed, previous_durations=()):
    reserve = max(480., max(previous_durations, default=0.) + 30.)
    return elapsed < 2700. and 2700.-elapsed >= reserve


def opportunity_stats(decisions, receipts, requests, n=128, threshold=None):
    end = max((r['client_terminal_ns'] for r in requests if r.get('client_terminal_ns')), default=None)
    ds = [d for d in decisions if end is not None and d['monotonic_ns'] < end]
    opp = sum(any(d['action_mask'][1:]) for d in ds)
    move = sum(d.get('executed_action', d['sampled_action']) != 0 for d in ds)
    voluntary = sum(any(d['action_mask'][1:]) and d.get('executed_action', d['sampled_action']) == 0 for d in ds)
    # Final coordinator receipts, deduplicated by migration identity.
    unique = {r['migration_id']: dict(r) for r in receipts.values() if r.get('migration_id')}
    for d in ds:
        mid = d.get('receipt', {}).get('migration_id')
        if mid in unique:
            unique[mid] = dict(unique[mid], src_id=d.get('src_id'), dst_id=d.get('dst_id'))
    succeeded = [r for r in unique.values() if r.get('terminal_status') == 'succeeded']
    started = [r for r in unique.values() if r.get('migration_start_ns') is not None]
    coverage = {r.get('actual_request_id') for r in succeeded} - {None}
    first = min((r['sent_ns'] for r in requests if r.get('sent_ns')), default=None)
    minutes = (end-first)/60e9 if end is not None and first is not None and end > first else None
    reasons = Counter(); pair_reasons = Counter(); filtered = 0
    for d in ds:
        if not any(d['action_mask'][1:]):
            s = d['snapshot']; now = d['monotonic_ns']
            found = set()
            if s.get('busy'): found.add('busy')
            if s.get('last_terminal_ns') is not None and now-s['last_terminal_ns'] < 5e9: found.add('cooldown')
            states = {x['instance_id']: x for x in s['instances']}
            for a,b in PAIRS:
                _, reason = eligible_pair(states[s['mapping'][a][0]], states[s['mapping'][b][0]], now)
                if reason: found.add(reason); pair_reasons[reason] += 1
            reasons.update(found)
        if any(d['action_mask'][1:]) and threshold is not None:
            from rl_scheduler.real_opportunity import balanced_action
            raw = {'action_mask': d['action_mask'], 'snapshot_version': d['snapshot_version']}
            if balanced_action({'raw': raw}, d['snapshot'], threshold=threshold) == 0: filtered += 1
    counts = Counter(r.get('actual_request_id') for r in succeeded)
    directions = {}; reverse = 0
    for r in sorted(succeeded, key=lambda x: x.get('migration_start_ns', 0)):
        req = r.get('actual_request_id'); direction = (r.get('src_id'), r.get('dst_id'))
        if direction[::-1] in directions.get(req, set()): reverse += 1
        directions.setdefault(req, set()).add(direction)
    timing = lambda k: summarize_values([d['ppo'][k] for d in ds if k in d.get('ppo', {})])
    return dict(N_all=len(ds), N_opp=opp, N_forced=len(ds)-opp, N_voluntary=voluntary, N_move=move,
                opportunity_ratio=opp/len(ds) if ds else None, move_per_opportunity=move/opp if opp else None,
                succeeded=len(succeeded), started=len(started), rejected=sum(r.get('terminal_status')=='rejected' for r in unique.values()),
                failed=sum(r.get('terminal_status')=='failed' for r in unique.values()),
                distinct_requests=len(coverage), coverage=len(coverage)/n,
                success_per_minute=len(succeeded)/minutes if minutes else None,
                success_per_started=len(succeeded)/len(started) if started else None,
                repeated_migrations=sum(max(0,v-1) for v in counts.values()), reverse_migrations=reverse,
                no_opportunity_reasons=dict(reasons), no_opportunity_pair_reasons=dict(pair_reasons),
                balanced_filter_stay_despite_opportunity=filtered,
                # Includes threshold, source/destination ordering and post-move balance checks.
                balanced_filter_is_not_threshold_only=True,
                cpu_conversion_ms=timing('cpu_conversion_ms'), cpu_forward_ms=timing('cpu_forward_ms'),
                feature_ms=summarize_values([d['feature_ms'] for d in ds if 'feature_ms' in d]),
                control_work_ms=summarize_values([d['control_work_ms'] for d in ds if 'control_work_ms' in d]),
                actual_control_interval_ms=summarize_values([(b['monotonic_ns']-a['monotonic_ns'])/1e6 for a,b in zip(ds,ds[1:])]))


def valid(e):
    return bool(e and e.get('status') == 'passed' and e.get('execution_audit', {}).get('status') == 'passed'
                and e.get('measured_request_count') == 128 and e.get('cohort_complete') and e.get('primary_metrics'))


def metrics(e):
    p = e['primary_metrics']; x = p['client_e2e_ms']
    return dict(cost=x['mean']+x['cvar95'], mean=x['mean'], p95=x['p95'], cvar95=x['cvar95'],
                ttft=p['ttft_ms']['p95'], tpot=p['tpot_ms']['p95'])


def p0_gate(es):
    if len(es) != 3 or not all(valid(e) for e in es):
        return {'passed': False, 'reason': 'missing_or_invalid_P0'}
    a,p,b = map(metrics, es); checks = {}
    for k in ('cost','mean'):
        ref = (a[k]+b[k])/2
        checks[k+'_drift'] = abs(a[k]-b[k])/ref if ref > 0 else None
        checks[k+'_network_deviation'] = abs(p[k]-ref)/ref if ref > 0 else None
    ref = (a['p95']+b['p95'])/2
    checks['p95_network_worsening'] = (p['p95']-ref)/ref if ref > 0 else None
    zero = all(e['opportunity']['N_move']==0 and e['opportunity']['succeeded']==0 for e in es)
    return {'passed': zero and all(v is not None and v <= (.10 if k=='p95_network_worsening' else .05) for k,v in checks.items()),
            'zero_migrations': zero, 'deviations': checks, 'scope': 'network path overhead only; engineering screen, not equivalence'}


def pair_gate(stays, balanced):
    if not stays or not all(valid(e) for e in [*stays, balanced]):
        return {'passed': False, 'reason': 'missing_or_invalid_pair'}
    if len({e['trace_sha256'] for e in [*stays,balanced]}) != 1 or any(e['frozen_mapping'] != balanced['frozen_mapping'] for e in stays):
        return {'passed': False, 'reason': 'trace_or_mapping_mismatch'}
    ms = [metrics(e) for e in stays]; bm = metrics(balanced)
    ref = {k: sum(m[k] for m in ms)/len(ms) for k in bm}
    gains = {k:(ref[k]-bm[k])/ref[k] if ref[k]>0 else None for k in ref}
    o = balanced['opportunity']
    checks = {'positive_cost': gains['cost'] is not None and gains['cost']>0,
              'positive_mean': gains['mean'] is not None and gains['mean']>0,
              'cost_lower_than_each_stay': all(bm['cost'] < m['cost'] for m in ms),
              'p95': gains['p95'] is not None and gains['p95'] >= -.05,
              'ttft': gains['ttft'] is not None and gains['ttft'] >= -.10,
              'tpot': gains['tpot'] is not None and gains['tpot'] >= -.10,
              'succeeded': o['succeeded'] >= 20, 'coverage': o['coverage'] >= .15,
              'frequency': o['success_per_minute'] is not None and o['success_per_minute']>=3,
              'success_rate': o['success_per_started'] is not None and o['success_per_started']>=.9,
              'stay_opportunities': all(e['opportunity']['N_opp']>=30 and e['opportunity']['opportunity_ratio'] is not None and e['opportunity']['opportunity_ratio']>=.2 for e in stays)}
    return {'passed': all(checks.values()), 'checks': checks, 'relative_improvement': gains,
            'versus_each_stay': [{k:(m[k]-bm[k])/m[k] if m[k]>0 else None for k in m} for m in ms]}


def candidate_gate(p0, pairs):
    return bool(p0.get('passed') and len(pairs)==2 and all(p.get('passed') for p in pairs)
                and all(sum(p['relative_improvement'][k] for p in pairs)/2 >= .05 for k in ('cost','mean')))


def plan(directory, retry_from=None):
    import yaml
    from tokenizers import Tokenizer
    directory.mkdir(parents=True, exist_ok=False)
    for sub in ('traces', 'runtime', 'services'): (directory/sub).mkdir()
    cfg = yaml.safe_load((ROOT/'rl_scheduler/configs/real_pilot.yaml').read_text())
    srv = yaml.safe_load((ROOT/'rl_scheduler/configs/real_llumnix.yaml').read_text())
    srv['SERVER']['REQUEST_OUTPUT_QUEUE_TYPE'] = 'rayqueue'
    cfg['service'].update(ray_num_cpus=16, startup_timeout_sec=480,
                          ray_temp_dir='/tmp/p0p1_'+hashlib.sha256(str(directory).encode()).hexdigest()[:10],
                          request_output_queue_type='rayqueue')
    argv = cfg['service']['server_argv']
    argv[argv.index('--config-file')+1] = str(directory/'real_llumnix.yaml')
    argv += ['--max-num-batched-tokens','2112']
    validate_config(cfg,srv)
    (directory/'runtime_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (directory/'real_llumnix.yaml').write_text(yaml.safe_dump(srv,sort_keys=False))
    tokenizer = Tokenizer.from_file(str(Path(cfg['service']['model_path'])/'tokenizer.json'))
    # Check existing experiment ledgers before freezing new seeds.
    prior = read(retry_from/'plan.json') if retry_from else None
    seeds=prior['seeds'] if prior else [141101,141102]; used=[]
    for path in (ROOT/'benchmark/results').rglob('*'):
        if directory in path.parents or not path.is_file(): continue
        if ('ledger' in path.name or path.name in ('plan.json','run_manifest.json')) and path.suffix in ('.json','.jsonl'):
            content=path.read_text(errors='replace')
            if any(str(s) in content for s in seeds): used.append(str(path.relative_to(ROOT)))
    if used and not prior: raise RuntimeError('proposed seeds used: '+str(used))
    if prior and prior.get('retry_from'): raise RuntimeError('only one repair/retry permitted')
    hashes={}
    for label, rate in [('Q15-A',1.),('Q15-B',.6)]:
        for seed in seeds:
            trace=make_trace(candidate=dict(family='S1',profile='decode1536',rho=1.,hot_fraction=.9),
                             n=128,seed=seed,lambda_ref=rate,tokenizer=tokenizer,episode_id_prefix=f'{label}_{seed}_')
            assert all(len(tokenizer.encode(r['prompt']).ids)==128 and r['output_tokens']==1536 for r in trace)
            assert sum(r['initial_instance_index']==seed%4 for r in trace)==115
            name=f'{label}_{seed}'; atomic_json(directory/'traces'/f'{name}.json',trace); hashes[name]=digest(trace)
    sources=[ROOT/'benchmark/run_p0p1_fast.py',ROOT/'benchmark/run_full_completion_balance.py', *sorted((ROOT/'rl_scheduler').glob('*.py')),
             ROOT/'llumnix/real_control.py',ROOT/'llumnix/manager.py',ROOT/'rl_scheduler/docs/p0p1_fast_track_prompt.md']
    result=dict(schema='p0p1-fast-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                objective='real service improvement plus migration opportunity and distinct request coverage',
                model=cfg['service']['model_path'],tokenizer=cfg['service']['model_path'], seeds=seeds,
                seed_search='existing ledgers/manifests/plans checked; no matches',
                candidates={'Q15-A':1.,'Q15-B':.6}, input_tokens=128,output_tokens=1536,n=128,hot_fraction=.9,
                lambda_ref_meaning='generator rate only, not measured service capacity',
                formal_sequence=['A seed1 Stay-A','A seed1 PPO-STAY','A seed1 Stay-B','A seed1 Balanced','A seed2 Balanced','A seed2 Stay'],
                fallback_sequence=['B seed1 Stay','B seed1 Balanced','B seed2 Balanced','B seed2 Stay'],
                gates_source='rl_scheduler/docs/p0p1_fast_track_prompt.md sections 5-6',
                gates_source_sha256=hashlib.sha256((ROOT/'rl_scheduler/docs/p0p1_fast_track_prompt.md').read_bytes()).hexdigest(),
                gates_implementation=inspect.getsource(p0_gate)+inspect.getsource(pair_gate)+inspect.getsource(candidate_gate),
                budgets=dict(no_new_after_sec=2700,min_remaining_sec=480,hard_abort_sec=3600,startup_sec=480,no_progress_sec=120,max_formal=10),
                threshold_rule='min(actual total_blocks_i)/8 after idle warmup; then immutable',balanced_threshold=None,
                trace_hashes=hashes,git_commit=command('git','rev-parse','HEAD')['stdout'].strip(),
                code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                config_sha256=hashlib.sha256((directory/'runtime_config.yaml').read_bytes()).hexdigest(),
                no_training=True,measurement='seeded random current CPU actor/critic, torch threads=1; network path overhead check')
    if prior:
        previous=read(retry_from/'run_state.json')
        result.update(retry_from=str(retry_from), repair_reason='rolling online token increments incorrectly treated as cumulative progress; use durable token_high',
                      budget_origin_wall=previous.get('budget_origin_wall',previous['started_wall']),
                      prior_formal_rounds=previous['formal_rounds'],
                      seed_search='explicit one-time repair retry; same frozen seeds/traces, not additional independent evidence')
        for name,sha in result['trace_hashes'].items():
            if sha!=prior['trace_hashes'][name]: raise ValueError('retry must retain exact trace')
        result['original_trace_hashes']=prior['trace_hashes']
    atomic_json(directory/'plan.json',result)
    atomic_json(directory/'training_environment_handoff.json',{'recommended':False,'status':'planned','model':result['model']})
    return result


def analyze(directory):
    plan_=read(directory/'plan.json'); inventory=rows(directory/'episode_ledger.jsonl')
    all_es=[]
    for item in inventory:
        e=read(directory/'runs'/item['episode_id']/'episode.json')
        e.update({k:item[k] for k in ('candidate','seed','mode','wall_sec')})
        e['execution_audit']=read(directory/'runs'/item['episode_id']/'execution_audit.json')
        requests=rows(directory/'runs'/item['episode_id']/'requests.jsonl')
        decisions=rows(directory/'runs'/item['episode_id']/'decisions.jsonl')
        e['opportunity']=opportunity_stats(decisions,e['migration_receipts'],requests,n=e['planned_request_count'],threshold=plan_['balanced_threshold'])
        atomic_json(directory/'runs'/item['episode_id']/'opportunity.json',e['opportunity'])
        if item['candidate']!='warmup': all_es.append(e)
    p0=p0_gate(all_es[:3]); comparisons={}; winner=None
    for label in ('Q15-A','Q15-B'):
        pairs=[]
        for seed in plan_['seeds']:
            group=[e for e in all_es if e['candidate']==label and e['seed']==seed]
            stays=[e for e in group if e['mode'].startswith('Stay')]
            balanced=next((e for e in group if e['mode']=='Balanced'),None)
            pairs.append(pair_gate(stays,balanced))
        passed=candidate_gate(p0,pairs)
        comparisons[label]={'pairs':pairs,'passed':passed,'P0_extrapolation':label=='Q15-B'}
        if passed: winner=label
    state=read(directory/'run_state.json') if (directory/'run_state.json').exists() else {'status':'planned'}
    status='fast_candidate_passed' if winner else state['status']
    if status=='completed': status='not_found'
    result={'status':status,'p0':p0,'candidates':comparisons,'recommended':bool(winner),'winner':winner}
    atomic_json(directory/'comparison.json',result)
    output=[]
    for e in all_es:
        m=metrics(e) if valid(e) else {}; o=e['opportunity']; p=e.get('primary_metrics') or {}
        output.append(dict(episode_id=e['episode_id'],candidate=e['candidate'],seed=e['seed'],mode=e['mode'],status=e['status'],
                           wall_sec=e['wall_sec'],completed=e['measured_request_count'],**m,
                           send_lag_p95_ms=p.get('send_lag_ms',{}).get('p95'),send_lag_max_ms=p.get('send_lag_ms',{}).get('max'),
                           **{k:v for k,v in o.items() if not isinstance(v,(dict,list))}))
    if output:
        with (directory/'results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    else: (directory/'results.csv').write_text('episode_id,status\n')
    handoff=dict(recommended=bool(winner),status=status,model=plan_['model'],evidence=str(directory),
                 applicability='128 short-input/long-output requests; artificial 90% single-instance hotspot; two new seeds only',
                 no_PPO_training=True,comparison='comparison.json')
    if winner:
        handoff.update(candidate=winner,rate=plan_['candidates'][winner],plan=plan_,results=output)
    atomic_json(directory/'training_environment_handoff.json',handoff)
    text_=['# P0/P1 快速筛选', '',f'状态：{status}；推荐 PPO 环境：{bool(winner)}。',
           '模型 Qwen2-1.5B，输入 128 / 实际输出 1536 token，128 请求，90% 热点。',
           'PPO-STAY 为固定随机权重的 CPU 网络路径开销检查；没有训练。',
           f'P0：{json.dumps(p0,ensure_ascii=False)}','',
           '| 场景 | 种子 | 模式 | 状态 | mean ms | P95 ms | cost ms | 成功搬迁 | 不同请求 | 机会/决策 | 只能 Stay |',
           '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in all_es:
        m=metrics(e) if valid(e) else {}; o=e['opportunity']
        text_.append(f"| {e['candidate']} | {e['seed']} | {e['mode']} | {e['status']} | {m.get('mean')} | {m.get('p95')} | {m.get('cost')} | {o['succeeded']} | {o['distinct_requests']} | {o['N_opp']}/{o['N_all']} | {o['N_forced']} |")
    text_ += ['',f'运行状态及真实时间：`{json.dumps(state,ensure_ascii=False)}`',
              '逐种子收益及逐项门槛见 comparison.json。缺失回合均不判通过。完整原始证据在 runs/、causal_events/ 和 runtime/。',
              '未达标时不进入 PPO；不能由大量迁移或 GPU 均衡推出服务改善。',
              '原因统计可重叠。Balanced 过滤计数同时含阈值、配对排序和迁后检查，不能全部归因于阈值。',
              'Q15-B 复用 P0 仅属不同请求率的有限外推；控制及发包开销另存每轮 opportunity.json。',
              '下一项建议由失败证据决定；未完成两种子比较前不调整当前冻结配置。']
    (directory/'report.md').write_text('\n'.join(text_)+'\n')
    return result


async def run(directory):
    import aiohttp
    import yaml
    from tokenizers import Tokenizer
    from rl_scheduler.cohort_ppo import audit_execution
    plan_=read(directory/'plan.json')
    if (directory/'run_state.json').exists(): raise RuntimeError('never overwrite a prior attempt; create a new plan directory')
    for rel,sha in plan_['code_hashes'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=sha: raise RuntimeError('code changed since plan: '+rel)
    cfg=yaml.safe_load((directory/'runtime_config.yaml').read_text()); srv=yaml.safe_load((directory/'real_llumnix.yaml').read_text())
    validate_config(cfg,srv)
    if hashlib.sha256((directory/'runtime_config.yaml').read_bytes()).hexdigest()!=plan_['config_sha256']: raise RuntimeError('config changed')
    env={'before':environment()};atomic_json(directory/'environment_audit.json',env)
    service=ManagedFullCompletionService(directory);attempt_start=time.monotonic();wall=time.time()
    origin=plan_.get('budget_origin_wall',wall);start=attempt_start-max(0.,wall-origin)
    durations=[];formal=plan_.get('prior_formal_rounds',0);mapping=None
    state={'status':'running','started_wall':wall,'budget_origin_wall':origin};atomic_json(directory/'run_state.json',state)
    loop=asyncio.get_running_loop();stop=asyncio.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): loop.add_signal_handler(sig,stop.set)
    async def stop_on_signal():
        await stop.wait(); await asyncio.to_thread(service.stop)
    signal_task=asyncio.create_task(stop_on_signal())
    async def hard_budget_watchdog():
        await asyncio.sleep(max(0.,3600-(time.monotonic()-start)))
        state.update(status='incomplete_budget',reason='60 minute hard abort')
        await asyncio.to_thread(service.stop)
    budget_task=asyncio.create_task(hard_budget_watchdog())
    controller=MeasurementStay()
    try:
        await asyncio.to_thread(service.start)
        await asyncio.wait_for(wait_service_ready(cfg,service.is_running), timeout=max(.1,480-(time.monotonic()-attempt_start)))
        timeout=aiohttp.ClientTimeout(total=None,connect=5,sock_connect=5)
        async with aiohttp.ClientSession(timeout=timeout,connector=aiohttp.TCPConnector(keepalive_timeout=3.)) as session:
            async def episode(candidate,seed,mode,trace):
                nonlocal mapping,formal
                if candidate!='warmup':
                    if not can_start(time.monotonic()-start,durations):
                        raise TimeoutError('insufficient remaining 45-minute budget for new complete episode')
                    if formal>=10: raise TimeoutError('maximum formal rounds reached')
                    formal+=1
                if stop.is_set(): raise RuntimeError('user_stop')
                name=f'{formal:02d}_{candidate}_{seed}_{mode}'
                before=time.monotonic()
                print(json.dumps({'event':'episode_start','id':name,'elapsed_sec':before-start}),flush=True)
                e=await run_episode(cfg=cfg,run_id=directory.name,episode_id=name,
                     policy='PPO-STAY' if mode=='PPO-STAY' else 'Balanced' if mode=='Balanced' else 'Stay',trace=trace,run_dir=directory,
                     control_session=session,frozen_mapping=mapping,service_probe=service.is_running,stop_owned_service=service.stop,
                     decision_controller=controller if mode=='PPO-STAY' else None,controller_version='p0p1-fast-measurement-v1',
                     balanced_threshold=plan_['balanced_threshold'] or 9000.,no_progress_timeout_sec=120.,hard_stop_monotonic=start+3600)
                elapsed=time.monotonic()-before
                if candidate!='warmup': durations.append(elapsed)
                mapping=e['frozen_mapping']
                local=directory/'runs'/name
                full=read(local/'episode.json')
                try:
                    audit=audit_execution(full,rows(local/'decisions.jsonl'),rows(local/'events.jsonl'),rows(local/'ppo_action_intents.jsonl'))
                    if mode.startswith('Stay') or mode=='PPO-STAY':
                        if audit['coordinator_attempts'] or e['actual_migration_attempts']: raise ValueError('Stay had actual migrations')
                except Exception as exc:
                    audit={'status':'failed','error':str(exc)}
                atomic_json(local/'execution_audit.json',audit)
                (local/'migrations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audit.get('reconciled_migrations', [])))
                with (directory/'episode_ledger.jsonl').open('a') as f:
                    f.write(json.dumps(dict(episode_id=name,candidate=candidate,seed=seed,mode=mode,wall_sec=elapsed))+'\n');f.flush();os.fsync(f.fileno())
                print(json.dumps({'event':'episode_end','id':name,'status':e['status'],'audit':audit['status'],'wall_sec':elapsed,'issues':e.get('issues')}),flush=True)
                if e['status']!='passed' or audit['status']!='passed': raise RuntimeError('first episode/lifecycle/routing/execution audit failure; stop formal comparisons')
                return e
            tokenizer=Tokenizer.from_file(str(Path(cfg['service']['model_path'])/'tokenizer.json'))
            warm=await episode('warmup',141100,'Stay',_warmup_trace(tokenizer))
            snap=await RealBridge(cfg['service']['base_url'],session).snapshot()
            capacities=[s.get('total_blocks',0) for s in snap['instances']]
            if len(capacities)!=4 or min(capacities)<=0: raise RuntimeError('unknown or zero block capacity')
            plan_['balanced_threshold']=min(capacities)/8
            plan_['capacity_snapshot']=snap
            plan_['threshold_frozen_wall']=time.time();atomic_json(directory/'plan.json',plan_)
            env['after_warmup']=environment();env['capacity_snapshot']=snap
            # Preserve actual worker/instance/GPU identities from Ray state and worker logs.
            from rl_scheduler.migration_scan_runner import gpu_mapping
            acks=await RealBridge(cfg['service']['base_url'],session).flush()
            env['worker_flush']=acks
            env['instance_gpu_uuid']=gpu_mapping(acks)
            atomic_json(directory/'environment_audit.json',env)
            s1,s2=plan_['seeds']
            def trace(c,s):
                value=read(directory/'traces'/f'{c}_{s}.json')
                if digest(value)!=plan_['trace_hashes'][f'{c}_{s}']: raise ValueError('trace hash changed')
                return value
            for mode in ('Stay-A','PPO-STAY','Stay-B'): await episode('Q15-A',s1,mode,trace('Q15-A',s1))
            analysis=analyze(directory)
            if not analysis['p0']['passed']:
                state['status']='p0_failed';return
            await episode('Q15-A',s1,'Balanced',trace('Q15-A',s1));analysis=analyze(directory)
            if analysis['candidates']['Q15-A']['pairs'][0]['passed']:
                await episode('Q15-A',s2,'Balanced',trace('Q15-A',s2))
                await episode('Q15-A',s2,'Stay',trace('Q15-A',s2));analysis=analyze(directory)
                if analysis['recommended']: state['status']='fast_candidate_passed';return
            # Only attempt the one preregistered fallback when a complete pair fits.
            if 2700-(time.monotonic()-start) < max(960,2*max(durations,default=480)+60):
                state['status']='incomplete_budget';return
            await episode('Q15-B',s1,'Stay',trace('Q15-B',s1))
            await episode('Q15-B',s1,'Balanced',trace('Q15-B',s1));analysis=analyze(directory)
            if analysis['candidates']['Q15-B']['pairs'][0]['passed']:
                await episode('Q15-B',s2,'Balanced',trace('Q15-B',s2))
                await episode('Q15-B',s2,'Stay',trace('Q15-B',s2))
            state['status']='completed'
    except TimeoutError as exc:
        state.update(status='incomplete_budget',reason=str(exc))
    except BaseException as exc:
        state.update(status='execution_failed',reason=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal_task.cancel();budget_task.cancel()
        await asyncio.gather(signal_task,budget_task,return_exceptions=True)
        await asyncio.to_thread(service.stop)
        state.update(elapsed_sec=time.monotonic()-start,attempt_elapsed_sec=time.monotonic()-attempt_start,formal_rounds=formal,ended_wall=time.time(),episode_durations_sec=durations)
        env['after_cleanup']=environment()
        env['service_stopped']=read(directory/'service_stopped.json') if (directory/'service_stopped.json').exists() else None
        atomic_json(directory/'environment_audit.json',env);atomic_json(directory/'run_state.json',state)
        analyze(directory)
        print(json.dumps(state),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','run','analyze'))
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--retry-from',type=Path,help='one justified repair, preserving traces and original overall budget')
    args=parser.parse_args(); directory=(args.run_dir or ROOT/'benchmark/results'/('p0p1_fast_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    if args.command=='plan': plan(directory,args.retry_from.resolve() if args.retry_from else None);print(directory)
    elif args.command=='run': asyncio.run(run(directory))
    else: print(json.dumps(analyze(directory),indent=2))

if __name__=='__main__': main()
