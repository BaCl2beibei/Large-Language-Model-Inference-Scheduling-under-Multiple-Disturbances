import json
from pathlib import Path
import subprocess
import sys

import pytest

from rl_scheduler.migration_scan import (weighted_summary, improvement, select_thresholds,
    coarse_matrix, refinement, paired_schedule, migration_statistics, resource_summary, remaining_steps,
    trace_for, opportunity)
from rl_scheduler.migration_scan_analysis import analyze, recommendation


def test_time_weighting_and_missing_gaps():
    result=weighted_summary([(0,10),(1,30),(4,None),(5,100)],0,10,max_gap=2)
    assert result['mean']==pytest.approx((10+30*2+100*2)/5)
    assert result['coverage']==.5
    assert result['p95']==100
    assert weighted_summary([(0,None)],0,10)['mean'] is None
    assert weighted_summary([],0,10)['coverage']==0
    assert weighted_summary([(-1,5),(1,9)],0,2,max_gap=2)['mean']==7
    with pytest.raises(ValueError):weighted_summary([(0,1),(0,2)],0,1)


def test_ratios_zero_missing_and_directions():
    assert improvement(0,2)==dict(percent=None,absolute=2)
    assert improvement(None,2)==dict(percent=None,absolute=None)
    assert improvement(10,12,lower=True)['percent']==-20
    assert improvement(10,8,lower=True)['percent']==20
    assert improvement(float('nan'),2)['percent'] is None


def test_threshold_empirical_nonidle_no_invention():
    assert select_thresholds([None,float('inf'),1,2,3,4,5,6,7,8])=={
        'Balanced-L':2,'Balanced-M':4,'Balanced-H':6}
    assert remaining_steps(dict(free_blocks=100,waiting_blocks=20,running=2,waiting=2))==20
    assert remaining_steps(dict(free_blocks=100,waiting_blocks=0,running=0,waiting=0)) is None
    with pytest.raises(ValueError):select_thresholds([1,1,1,2,3])
    with pytest.raises(ValueError):select_thresholds([None,float('inf')])


def test_matrix_and_counterbalanced_pairing():
    cases=coarse_matrix()
    assert len(cases)==9
    assert all(c['input_tokens']+c['output_tokens']<=c['max_model_len'] for c in cases)
    assert {r['max_num_seqs'] for r in refinement(cases[0])}=={128,192,256}
    thresholds={c['scenario']:{'Balanced-L':1,'Balanced-M':2,'Balanced-H':3} for c in cases}
    rows=paired_schedule(cases[:2],[1,2,3],thresholds)
    assert len(rows)==24
    assert {r['policy'] for r in rows[:4]}=={'Stay','Balanced-L','Balanced-M','Balanced-H'}
    assert [r['policy'] for r in rows[:4]]!=[r['policy'] for r in rows[4:8]]
    with pytest.raises(ValueError):paired_schedule(cases,[1,1,2],thresholds)


def test_poisson_trace_reproducible_and_context(monkeypatch):
    import rl_scheduler.full_completion as module
    monkeypatch.setattr(module,'exact_prompt',lambda tokenizer,n,seed:('prompt',n))
    case=coarse_matrix()[0]
    a=trace_for(case,10,None)
    assert a==trace_for(case,10,None)
    assert a!=trace_for(case,11,None)
    assert all(0<r['scheduled_arrival_sec']<120 for r in a)
    assert len({r['request_id'] for r in a})==len(a)
    with pytest.raises(ValueError):trace_for({**case,'max_model_len':100},10,None)


def test_migration_denominators_distinct():
    decisions=[dict(sampled_action=a,opportunity={'legal_pairs':l}) for a,l in [(0,0),(1,2),(0,3),(1,1)]]
    migrations=[dict(terminal_status='succeeded',actual_request_id='r1',attempt_ns=i*10**9,
        terminal_ns=(i+1)*10**9,src_id='a',dst_id='b') for i in (1,3)]
    result=migration_statistics(decisions,migrations,10,120)
    assert result['request_coverage']==.1
    assert result['opportunity_ratio']==.75
    assert result['action_given_opportunity']==pytest.approx(2/3)
    assert result['migrations_per_accepted_request']==.2
    assert result['successes_per_minute']==1
    assert result['duration_p95_sec']==1
    assert result['success_intervals_sec']==[2]
    empty=migration_statistics([],[],0,120)
    assert empty['request_coverage'] is empty['action_given_opportunity'] is empty['success_rate'] is None


def test_four_gpu_dispersion_missing_not_zero():
    samples=[dict(at=0,instances=[dict(instance_id=str(i),kv_pct=i*10) for i in range(4)]),
        dict(at=1,instances=[dict(instance_id=str(i),kv_pct=None if i==0 else i*10) for i in range(4)])]
    result=resource_summary(samples,0,2)
    assert result['kv_pct']['cluster_mean']==15
    assert result['kv_pct']['dispersion']['coverage']==.5
    assert result['kv_pct']['dispersion']['mean']==pytest.approx(125**.5)
    assert result['gpu_pct']['cluster_mean'] is None
    assert result['kv_pct']['instances']['0']['coverage']==.5


def test_no_real_data_no_fabricated_recommendations(tmp_path):
    (tmp_path/'plots').mkdir()
    analyze(tmp_path)
    assert json.loads((tmp_path/'recommendations.json').read_text())=={}
    assert json.loads((tmp_path/'candidate_shortlist.json').read_text())==[]
    assert not list((tmp_path/'plots').glob('*.png'))
    assert '0' in (tmp_path/'stay_scan_report.md').read_text()


def test_recommendation_requires_evidence_and_three_seeds():
    assert recommendation([])['recommendation']=='conditional'
    rows=[dict(policy=p,seed=s,goodput_improvement_pct=12,error_rate_absolute_change=0,
        kv_pressure_supported=True,resource_balance_supported=True,succeeded=3,success_rate=1,
        opportunity_ratio=.2) for p in ('Balanced-L','Balanced-M','Balanced-H') for s in (1,2,3)]
    assert recommendation(rows)['recommendation']=='recommended'
    for row in rows:row['resource_balance_supported']=False
    assert recommendation(rows)['recommendation']=='not_recommended'


def test_cli_plan_never_starts_gpu_and_freezes_sources(tmp_path):
    from benchmark.run_migration_opportunity_scan import plan, verify_sources
    target=tmp_path/'run'
    plan(target)
    manifest=json.loads((target/'experiment_manifest.json').read_text())
    assert manifest['status']=='planned_no_gpu_experiments'
    assert not list((target/'raw_runs').iterdir())
    verify_sources(manifest)
    manifest['code_hashes']['rl_scheduler/migration_scan.py']='invalid'
    with pytest.raises(RuntimeError):verify_sources(manifest)
    with pytest.raises(FileExistsError):plan(target)


def test_scheduler_observers_preserve_decisions_and_restore_on_error(monkeypatch):
    from collections import Counter
    from types import SimpleNamespace
    from llumnix.backends.vllm.scheduler import SchedulerLlumnix, Scheduler, AllocStatus
    from vllm.core.scheduler import SchedulingBudget
    scheduler=object.__new__(SchedulerLlumnix)
    scheduler._scan_enabled=True
    scheduler.scan_counters=Counter()
    allocate=lambda *a,**kw:AllocStatus.LATER
    scheduler.block_manager=SimpleNamespace(can_allocate=allocate)
    budget=SchedulingBudget(token_budget=10,max_num_seqs=2)
    original=budget.can_schedule
    def parent(self,budget,curr_loras,enable_chunking=False):
        assert self.block_manager.can_allocate(None)==AllocStatus.LATER
        assert budget.can_schedule(num_new_tokens=11,num_new_seqs=3) is False
        raise RuntimeError('parent scheduling error')
    monkeypatch.setattr(Scheduler,'_schedule_prefills',parent)
    with pytest.raises(RuntimeError,match='parent scheduling error'):
        scheduler._schedule_prefills(budget,None)
    assert scheduler.scan_counters=={'kv_allocation_later':1,'token_budget_blocked':1,'sequence_limit_blocked':1}
    assert scheduler.block_manager.can_allocate is allocate
    assert budget.can_schedule==original
    scheduler._scan_enabled=False
    with pytest.raises(RuntimeError):scheduler._schedule_prefills(budget,None)
    assert scheduler.scan_counters['kv_allocation_later']==1


@pytest.mark.asyncio
async def test_new_episode_end_to_end_cpu_http(tmp_path,monkeypatch):
    import asyncio
    import time
    from aiohttp import web,ClientSession,ClientTimeout
    from aiohttp.test_utils import TestServer
    from rl_scheduler.real_bridge import ControlArbiter
    from rl_scheduler.real_env import RealEnvConfig
    from rl_scheduler.real_budget import atomic_json
    from rl_scheduler.real_runner import server_request_id
    from rl_scheduler.tests.test_real_runner import Producer
    from rl_scheduler.tests.test_real_bridge import states,Clock
    from rl_scheduler.migration_scan_runner import episode,WallBudget
    import rl_scheduler.migration_scan_runner as runner
    causal=tmp_path/'causal';causal.mkdir()
    atomic_json(tmp_path/'service_session.json',{'causal_dir':str(causal)})
    producers={name:Producer(causal,name) for name in ('api','manager',*[f'l{i}' for i in range(4)],*[f'w{i}' for i in range(4)])}
    current=states(Clock())
    for state in current:state.update(running=0,used_blocks=0,free_blocks=1000,scan_counters={})
    async def provider():return [dict(s,measured_ns=time.monotonic_ns()) for s in current]
    async def dispatch(receipt):raise AssertionError('Stay must not migrate')
    core=ControlArbiter(provider,dispatch)
    async def control(request):
        method=request.match_info['method'];body=await request.json()
        if method=='flush':return web.json_response({'api':producers['api'].checkpoint(),'cluster':{
            'manager':producers['manager'].checkpoint(),'instances':[{'local':producers[f'l{i}'].checkpoint(),
                'workers':[producers[f'w{i}'].checkpoint()]} for i in range(4)]}})
        if method=='release_episode':return web.json_response({'status':'released'})
        return web.json_response(await getattr(core,method)(**body))
    async def generate(request):
        body=await request.json();rid=body['request_id'];ep=body['episode_id'];sid=server_request_id('cpu',ep,rid)
        instance=body['initial_instance_id']
        producers['manager'].emit('real_initial_route',request_id=sid,instance_id=instance)
        producers['l'+instance].emit('request_engine_admitted',request_id=sid,instance_id=instance)
        def event(kind,**data):producers['api'].emit('real_metrics_event',metric_kind=kind,run_id='cpu',episode_id=ep,
            request_id=rid,server_request_id=sid,data=data)
        def frame(kind,**data):return (json.dumps(dict(run_id='cpu',episode_id=ep,request_id=rid,
            server_request_id=sid,kind=kind,**data))+'\0').encode()
        response=web.StreamResponse();await response.prepare(request)
        event('accepted');await response.write(frame('accepted'))
        event('api_first');await response.write(frame('output',cumulative_tokens=1,text='a'))
        await asyncio.sleep(.001)
        await response.write(frame('output',cumulative_tokens=4,text='abcd'))
        event('server_terminal',outcome='success');await response.write(frame('terminal',outcome='success'))
        return response
    app=web.Application();app.router.add_post('/real_control/{method}',control);app.router.add_post('/generate_real_metrics',generate)
    monkeypatch.setattr(runner,'gpu_mapping',lambda ack:{str(i):f'g{i}' for i in range(4)})
    monkeypatch.setattr(runner,'gpu_sample',lambda:{f'g{i}':dict(gpu_pct=10,memory_mib=1024,memory_pct=10) for i in range(4)})
    trace=[dict(request_id=f'r{i}',scheduled_arrival_sec=.1*i+.02,prompt='CPU',output_tokens=4,route_seed=i) for i in range(3)]
    try:
        async with TestServer(app) as server,ClientSession(timeout=ClientTimeout(total=2)) as session:
            summary=await episode(cfg={'service':{'base_url':str(server.make_url('')).rstrip('/')},'sampling':{'n':1}},
                service_dir=tmp_path,run_id='cpu',episode_id='e',policy='Stay',threshold=None,
                trace=trace,session=session,budget=WallBudget(5),
                horizons=RealEnvConfig(arrival_sec=.4,drain_sec=.2,cleanup_sec=1,decision_interval_sec=.05))
        assert summary['status']=='passed'
        assert summary['metrics']['completed']==3
        assert summary['metrics']['client_output_tokens']==12
        assert summary['metrics']['completed_throughput_rps']==pytest.approx(5)
        assert summary['migration_statistics']['succeeded']==0
        assert summary['resources']['gpu_pct']['cluster_mean']==10
        assert summary['routes']['status']=='passed'
    finally:
        for producer in producers.values():producer.stream.close()


def test_offline_nonempty_reports_plots_and_frozen_thresholds(tmp_path):
    from rl_scheduler.real_budget import atomic_json
    from rl_scheduler.real_bridge import digest
    from benchmark.run_migration_opportunity_scan import freeze
    from rl_scheduler.migration_scan_runner import append
    (tmp_path/'plots').mkdir()
    cases=coarse_matrix()[:2]
    manifest={'code_hashes':{},'seeds':[1,2,3]}
    atomic_json(tmp_path/'experiment_manifest.json',manifest)
    candidates=tmp_path/'candidates.json'
    atomic_json(candidates,cases)
    for case in cases:
        for phase,policies in [('scan',['Stay']),('paired',['Stay','Balanced-L','Balanced-M','Balanced-H'])]:
            for policy in policies:
                path=tmp_path/phase/case['scenario']/policy
                path.mkdir(parents=True)
                rows=[dict(at=0.,instances=[dict(instance_id=str(i),kv_pct=60 if i==0 else 10,
                    gpu_pct=60 if i==0 else 10,memory_mib=1000,memory_pct=10,running=i,waiting=1 if i==0 else 0,
                    free_blocks=100,used_blocks=100,head_wait_sec=1,prefill_tps=1,decode_tps=1,output_tps=1) for i in range(4)])]
                summary=dict(status='passed',mapping=[[str(i),'e'] for i in range(4)],
                    metrics=dict(accepted=10,completed=10,slo_goodput_rps=0 if policy=='Stay' else .1,
                        ttft_p95_ms=10,tpot_p95_ms=10,completed_throughput_rps=.1,client_token_throughput_tps=1,
                        e2e_mean_ms=100,e2e_p95_ms=100,nonmanual_error_rate=0),
                    resources=resource_summary(rows,0,1),migrations=[],f_values=list(range(1,9)),
                    migration_statistics=migration_statistics([],[],10,1),trace_hash='same',
                    window=dict(start=0,arrival_end=1,end=1),waiting_fraction=1,
                    counter_deltas={'0':{'kv_allocation_later':100}})
                atomic_json(path/'scan_summary.json',summary)
                append(path/'resources.jsonl',rows[0])
                append(tmp_path/'completed_runs.jsonl',dict(case=case,phase=phase,policy=policy,seed=1,
                    threshold=None if policy=='Stay' else 2,summary=str((path/'scan_summary.json').relative_to(tmp_path))))
    analyze(tmp_path)
    assert len(list((tmp_path/'plots').glob('*.png')))==8
    assert 'N/A' in (tmp_path/'performance_improvement.csv').read_text()
    assert 'conditional' in (tmp_path/'ppo_scenario_recommendations.md').read_text()
    freeze(tmp_path,candidates)
    thresholds=json.loads((tmp_path/'threshold_selection.json').read_text())
    assert thresholds['selection'][cases[0]['scenario']]['thresholds']['Balanced-M']==4
    with pytest.raises(RuntimeError):freeze(tmp_path,candidates)


def test_live_gpu_mapping_and_missing_samples(monkeypatch):
    from types import SimpleNamespace
    import rl_scheduler.migration_scan_runner as module
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='g0, [N/A], 512, 1024\n'))
    assert module.gpu_sample()=={'g0':dict(gpu_pct=None,memory_mib=512,memory_pct=50)}
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='\n'.join(f'{i}, g{i}' for i in range(4))))
    ack={'instances':[dict(workers=[dict(worker_pid=i,instance_id=f'i{i}')]) for i in range(4)]}
    assert module.gpu_mapping(ack)=={f'i{i}':f'g{i}' for i in range(4)}
    with pytest.raises(RuntimeError):module.gpu_mapping({'workers':[{'worker_pid':99,'instance_id':'unknown'}]})


def test_legal_opportunity_remains_distinct_from_threshold_block():
    from rl_scheduler.tests.test_real_bridge import states,Clock
    clock=Clock();measured=states(clock)
    for state in measured:state.update(running=2,free_blocks=100,waiting_blocks=0)
    snapshot=dict(instances=measured,mapping=[[str(i),f'e{i}'] for i in range(4)],version='v',busy=False)
    raw=dict(snapshot_version='v',action_mask=[True]*13,global_features=[0.]*13)
    diag=opportunity(snapshot,raw,1000,clock())
    assert diag['legal_pairs']==12
    assert diag['potential_sources']==4
    assert diag['potential_destinations']==0
    assert diag['threshold_blocked'] is True
    assert diag['action']==0


def test_preemption_records_actual_mode_not_unspecified_argument(monkeypatch):
    from collections import Counter
    from types import SimpleNamespace
    from llumnix.backends.vllm.scheduler import SchedulerLlumnix,Scheduler,PreemptionMode
    scheduler=object.__new__(SchedulerLlumnix)
    scheduler._scan_enabled=True;scheduler.scan_counters=Counter()
    monkeypatch.setattr(Scheduler,'_preempt',lambda *a,**kw:PreemptionMode.RECOMPUTE)
    request=SimpleNamespace(last_preemption_time=None)
    assert scheduler._preempt(request,[])==PreemptionMode.RECOMPUTE
    assert scheduler.scan_counters['preemptions']==1
    assert scheduler.scan_counters['preemption_'+str(PreemptionMode.RECOMPUTE)]==1
    assert 'preemption_None' not in scheduler.scan_counters
