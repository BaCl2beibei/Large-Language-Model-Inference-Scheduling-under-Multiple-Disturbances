"""CPU invariants for the frozen P3 protocol and sequential orchestration."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import patch

from rl_scheduler.real_bridge import digest
from rl_scheduler.real_metrics import Event,NS
from rl_scheduler.real_opportunity import (protocol,episode_schedule,balanced_action,healthy,select_loads,paired_gate,
                                         make_trace,maximum_duration,readonly_budget,metric_summary)
from rl_scheduler.real_opportunity_runner import reconcile_migrations,route_audit,run_opportunity
from rl_scheduler.tests.test_real_metrics import Fixture
from rl_scheduler.tests.test_real_bridge import Clock,states,extract_method,load_file
from rl_scheduler.tests.test_real_env import budget_file,ROOT


def row(scene='A',seed=2101,policy='Stay',qps=3,phase='opportunity'):
    return {'episode_id':f'{phase}-{scene}-{seed}-{policy}','phase':phase,'scene':scene,'seed':seed,'policy':policy,'qps':qps,
        'status':'passed','trace_hash':f'{qps}-{seed}','external_case_hash':f'{qps}-{seed}-{scene}',
        'mapping':[[str(i),'e'+str(i)] for i in range(4)],'send_lateness_p95_ms':5.,
        'queue_last_third_samples':10,'queue_last_third_positive':5,
        'metrics':{'offered':100,'sent':100,'completed':100,'duration_sec':240,'completion_rate':1.,
            'nonmanual_error_rate':0.,'first_output_censored_fraction':0.,'terminal_censored_fraction':0.,
            'slo_goodput_rps':1.,'outstanding_request_seconds':480.}}


class ProtocolTests(unittest.TestCase):
    def setUp(self):self.spec=protocol('/tmp/p3-test','/tmp/runtime.yaml')

    def probes(self):
        return [row(scene,s,'Stay',q,'probe') for q,s in zip((3,6,9),(7101,7102,7103)) for scene in ('A','B')]

    def matrix(self):return [row(**case) for case in episode_schedule(self.spec,{'A':6,'B':9})]

    def test_fixed_12_matrix_counterbalances_same_trace_pair(self):
        matrix=self.matrix();self.assertEqual(len(matrix),12)
        self.assertEqual([(r['scene'],r['seed'],r['policy']) for r in matrix[:4]],
                         [('A',2101,'Stay'),('A',2101,'Balanced'),('A',2102,'Balanced'),('A',2102,'Stay')])
        for a,b in zip(matrix[::2],matrix[1::2]):self.assertEqual(a['trace_hash'],b['trace_hash'])
        self.assertFalse(set(self.spec['final_burst']['seeds'])&set(self.spec['opportunity']['seeds']))

    def test_selection_frozen_median_and_persistent_queue(self):
        probes=self.probes();selected=select_loads(probes,self.spec)
        self.assertEqual(selected['selected'],{'A':6,'B':9})
        for r in probes:
            if r['scene']=='B' and r['qps']==9:r['queue_last_third_positive']=4
        self.assertEqual(select_loads(probes,self.spec)['selected']['B'],6)
        for r in probes:r['metrics']['nonmanual_error_rate']=.5
        self.assertEqual(select_loads(probes,self.spec)['status'],'incomplete')

    def test_selection_never_omits_failed_or_missing_probe(self):
        probes=self.probes();probes[0]['status']='incomplete'
        with self.assertRaises(ValueError):select_loads(probes,self.spec)
        with self.assertRaises(ValueError):select_loads(self.probes()[:-1],self.spec)
        with self.assertRaises(ValueError):select_loads(self.probes()+self.probes()[:1],self.spec)

    def test_no_healthy_queued_skew_scene_cannot_proceed(self):
        probes=self.probes()
        for r in probes:r['queue_last_third_positive']=0
        self.assertEqual(select_loads(probes,self.spec)['status'],'incomplete')
        r=probes[0];r['metrics']['sent']=99;self.assertFalse(healthy(r,self.spec))
        r['metrics']['sent']=100;r['send_lateness_p95_ms']=1001;self.assertFalse(healthy(r,self.spec))

    def test_gate_2_of_3_ten_percent_plus_all_seed_error_guard(self):
        matrix=self.matrix()
        for r in matrix:
            if r['scene']=='B' and r['policy']=='Balanced' and r['seed'] in (2101,2102):r['metrics']['slo_goodput_rps']=1.1
        result=paired_gate(matrix,self.spec);self.assertEqual(result['status'],'passed')
        self.assertEqual(result['scenes']['B']['winning_seeds'],2)
        for r in matrix:
            if r['scene']=='B' and r['policy']=='Balanced' and r['seed']==2103:r['metrics']['nonmanual_error_rate']=.010001
        self.assertEqual(paired_gate(matrix,self.spec)['status'],'failed')

    def test_integer_count_thresholds_and_no_epsilon_relaxation(self):
        matrix=self.matrix()
        for r in matrix:
            r['metrics'].update(offered=300,completed=300,slo_completed=100,nonmanual_errors=10)
            if r['policy']=='Balanced':r['metrics'].update(slo_completed=110,slo_goodput_rps=1.1,nonmanual_errors=13)
        self.assertEqual(paired_gate(matrix,self.spec)['status'],'passed')
        for r in matrix:
            if r['policy']=='Balanced':r['metrics']['nonmanual_errors']=14
        self.assertEqual(paired_gate(matrix,self.spec)['status'],'failed')
        matrix=self.matrix()
        for r in matrix:
            if r['policy']=='Balanced':r['metrics']['slo_goodput_rps']=1.0999999999999
        self.assertEqual(paired_gate(matrix,self.spec)['status'],'failed')

    def test_zero_goodput_and_zero_offered_do_not_auto_pass(self):
        matrix=self.matrix()
        for r in matrix:r['metrics']['slo_goodput_rps']=0.
        self.assertEqual(paired_gate(matrix,self.spec)['status'],'incomplete')
        for r in matrix:
            if r['policy']=='Balanced':r['metrics']['slo_goodput_rps']=2.
        result=paired_gate(matrix,self.spec);self.assertEqual(result['status'],'failed')
        self.assertTrue(all(r['relative_gain'] is None for r in result['pairs']))
        matrix[0]['metrics']['offered']=0;matrix[0]['metrics']['nonmanual_error_rate']=None
        self.assertEqual(paired_gate(matrix,self.spec)['status'],'incomplete')

    def test_trace_route_window_mapping_mismatch_and_invalid_episode_fail_closed(self):
        for change in ('trace','routing','window','mapping','audit','missing'):
            matrix=self.matrix()
            if change=='trace':matrix[0]['trace_hash']='other'
            elif change=='routing':matrix[0]['external_case_hash']='other'
            elif change=='window':matrix[0]['metrics']['duration_sec']=239
            elif change=='mapping':matrix[0]['mapping'][0][1]='rebuilt'
            elif change=='audit':matrix[0]['status']='incomplete'
            else:matrix.pop()
            self.assertEqual(paired_gate(matrix,self.spec)['status'],'incomplete',change)

    def test_deterministic_trace_profiles_and_future_burst_registration(self):
        a=make_trace(self.spec,seed=2101,qps=3,duration=120)
        self.assertEqual(a,make_trace(self.spec,seed=2101,qps=3,duration=120))
        self.assertTrue(all(0<=r['scheduled_arrival_sec']<120 for r in a))
        self.assertEqual({(r['input_budget'],r['output_tokens']) for r in a},
                         {(i,o) for i in (256,512,1048) for o in (256,512,1048)})
        burst=make_trace(self.spec,seed=5101,qps=3,duration=120,bursty=True)
        self.assertNotEqual(a,burst);self.assertEqual(self.spec['reward']['C_ref'],None)
        self.assertGreater(len(burst),len(a))
        self.assertEqual(maximum_duration(self.spec,{'service':{'startup_timeout_sec':1200}}),5760)

    def test_balanced_matches_native_order_and_predicate_using_actual_SR_blocks(self):
        clock=Clock();ss=states(clock)
        for s in ss:s.update(total_blocks=100000,free_blocks=90000,used_blocks=10000,running=0,waiting=0)
        ss[0]['running']=20
        snap={'instances':ss,'mapping':[[s['instance_id'],s['epoch']] for s in ss],'version':'v'}
        obs={'raw':{'snapshot_version':'v','action_mask':[True,True]+[False]*11}}
        self.assertEqual(balanced_action(obs,snap),1)
        class Load:
            def __init__(self,x):self.remaining_steps=x
            def __lt__(self,other):return self.remaining_steps>=other.remaining_steps
        source=Obj(instance_id='0',migration_load_metric=Load(90000/20),migration_load_metric_after_migrate_out=Load(90020/19))
        destination=Obj(instance_id='1',migration_load_metric=Load(float('inf')),migration_load_metric_after_migrate_in=Load(89980))
        method=extract_method('llumnix/global_scheduler/migration_policy.py','Balanced','pair_migration',{})
        owner=Obj(sort_instance_infos=lambda values,descending:values,_metric_value=lambda metric:metric.remaining_steps,migrate_out_load_threshold=9000.)
        self.assertEqual(method(owner,[source],[destination]),[('0','1')])
        obs['raw']['action_mask'][1]=False;self.assertEqual(balanced_action(obs,snap),0)
        obs['raw']['action_mask'][1]=True;ss[0]['free_blocks']=None;self.assertEqual(balanced_action(obs,snap),0)

    def test_budget_peek_does_not_change_clock_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);budget_file(run)
            from rl_scheduler.real_budget import BudgetLedger
            BudgetLedger(run).status();before=(run/'experiment_clock.json').read_bytes()
            status=readonly_budget(run);self.assertTrue(status['read_only'])
            self.assertEqual((run/'experiment_clock.json').read_bytes(),before)

    def test_cli_dry_run_reports_budget_and_live_plan_refuses_without_clock_write(self):
        cli=load_file('_test_opportunity_cli','benchmark/run_real_opportunity.py')
        from rl_scheduler.real_budget import BudgetLedger
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);budget_file(run);BudgetLedger(run).status()
            value=json.loads((run/'budget.json').read_text());value['phase_caps_sec']['P0_P3_including_GPU_checks_and_opportunity']=5000
            (run/'budget.json').write_text(json.dumps(value));before=(run/'experiment_clock.json').read_bytes()
            args=Obj(output_dir=run,config=run/'runtime.yaml',manage_services=True,dry_run=True,invocation_id='test')
            args.config.write_text('{}')
            validator=Obj(validate_gate=lambda *a:{'stage':'P3_CPU_preparation','dependency_hashes':{}})
            with patch.object(cli,'validate_inputs',return_value={}),patch.object(cli,'smoke_cli',return_value=validator):
                result=cli.plan(args,self.spec,{'service':{'startup_timeout_sec':1200}})
                self.assertFalse(result['GPU_will_start']);self.assertFalse(result['ready_for_live'])
                self.assertTrue(any('budget insufficient' in reason for reason in result['blocked_reasons']))
                args.dry_run=False
                with self.assertRaises(RuntimeError):cli.plan(args,self.spec,{'service':{'startup_timeout_sec':1200}})
            self.assertEqual((run/'experiment_clock.json').read_bytes(),before)

    def test_frozen_trace_and_prompt_pool_tampering_is_detected(self):
        from rl_scheduler.real_opportunity import load_frozen_trace
        from rl_scheduler.real_budget import atomic_json
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);directory=run/'traces/opportunity07';directory.mkdir(parents=True)
            spec=protocol(run,'/tmp/runtime');pool={str(i):{'prompt':'CPU','input_tokens':1} for i in (256,512,1048)}
            trace=make_trace(spec,seed=7101,qps=3,duration=30)
            atomic_json(directory/'trace.json',trace);atomic_json(directory/'prompt_pool.json',pool)
            manifest={'protocol_hash':digest(spec),'prompt_pool_hash':digest(pool),'traces':{
                'probe_q3_seed7101':{'path':'trace.json','sha256':hashlib.sha256((directory/'trace.json').read_bytes()).hexdigest()}}}
            atomic_json(directory/'manifest.json',manifest)
            self.assertEqual(len(load_frozen_trace(run,spec,seed=7101,qps=3,phase='probe')),len(trace))
            pool['256']['prompt']='changed';atomic_json(directory/'prompt_pool.json',pool)
            with self.assertRaises(ValueError):load_frozen_trace(run,spec,seed=7101,qps=3,phase='probe')

    def test_client_metrics_errors_manual_cancel_and_censoring(self):
        f=Fixture();f.plan_send('a');f.output(1,1,'a');f.terminal(2,'a')
        f.plan_send('b');f.terminal(3,'b',outcome='timeout',http_status=None)
        f.plan_send('c');f.add('cancel_requested',4,'c');f.terminal(5,'c',outcome='cancelled',http_status=499,reason='manual_drain');f.add('cancel_ack',6,'c',confirmed=True)
        f.plan_send('d');f.ack()
        metrics=metric_summary(f.ledger)
        self.assertEqual(metrics['nonmanual_errors'],1);self.assertEqual(metrics['nonmanual_error_rate'],.25)
        self.assertEqual(metrics['terminal_censored_fraction'],.25)
        self.assertEqual(metrics['first_output_censored_fraction'],.75)

    def test_original_attempt_timestamp_and_explicit_rejection_reconcile(self):
        decision=dict(decision_id='d',sampled_action=1,monotonic_time=NS)
        receipt=dict(receipt_status='accepted',terminal_status='succeeded',terminal_ns=4*NS,migration_id='m')
        event=Event('s',1,'diagnostic',2*NS,2*NS,'r','e',data={'event':'migration_attempt_started','migration_id':'m'})
        records=reconcile_migrations([decision],{'d':receipt},[event]);self.assertEqual(records[0]['attempt_ns'],2*NS)
        with self.assertRaises(ValueError):reconcile_migrations([decision],{'d':dict(receipt,terminal_status='unknown')},[event])
        with self.assertRaises(ValueError):reconcile_migrations([],{},[event])
        local={'receipt_status':'rejected','terminal_status':'rejected'}
        self.assertEqual(reconcile_migrations([decision],{'d':local},[])[0]['terminal_ns'],NS)

    def test_actual_initial_route_uses_engine_admission_not_latest_owner(self):
        f=Fixture();f.plan_send('a');f.output(1,1,'a');f.terminal(2,'a');f.ack()
        mapping=[[str(i),'e'+str(i)] for i in range(4)]
        event=Event('s',1,'diagnostic',NS,NS,'r','e','a',data={'event':'request_engine_admitted','instance_id':'0'})
        io=Obj(route_shares=None,expected_routes={})
        self.assertEqual(route_audit(io,f.ledger,[event],mapping)['status'],'passed')
        io.route_shares=[.7,.1,.1,.1];io.expected_routes={'a':'1'}
        self.assertEqual(route_audit(io,f.ledger,[event],mapping)['status'],'incomplete')
        self.assertEqual(route_audit(io,f.ledger,[],mapping)['status'],'incomplete')


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_episode_over_CPU_HTTP_reconciles_normal_and_skew_routes(self):
        from aiohttp import web,ClientSession,ClientTimeout
        from aiohttp.test_utils import TestServer
        from rl_scheduler.real_bridge import ControlArbiter,RealBridge
        from rl_scheduler.real_budget import atomic_json
        from rl_scheduler.real_runner import server_request_id
        from rl_scheduler.real_opportunity_runner import run_episode
        from rl_scheduler.tests.test_real_runner import Producer
        import time
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);budget_file(run);causal=run/'causal';causal.mkdir()
            atomic_json(run/'service_session.json',{'causal_dir':str(causal)})
            producers={name:Producer(causal,name) for name in ('api','manager',*[f'local{i}' for i in range(4)],*[f'worker{i}' for i in range(4)])}
            measured=states(Clock())
            for s in measured:s.update(running=0,used_blocks=0,free_blocks=1000)
            async def provider():return [dict(s,measured_ns=time.monotonic_ns()) for s in measured]
            async def dispatch(receipt):raise AssertionError('Stay dispatched migration')
            core=ControlArbiter(provider,dispatch)
            async def control(request):
                method=request.match_info['method'];body=await request.json()
                if method=='flush':return web.json_response({'api':producers['api'].checkpoint(),'cluster':{
                    'manager':producers['manager'].checkpoint(),'instances':[{'local':producers[f'local{i}'].checkpoint(),
                    'workers':[producers[f'worker{i}'].checkpoint()]} for i in range(4)]}})
                if method=='release_episode':return web.json_response({'status':'released'})
                return web.json_response(await getattr(core,method)(**body))
            async def generate(request):
                body=await request.json();rid=body['request_id'];ep=body['episode_id'];sid=server_request_id('run',ep,rid)
                instance=body.get('initial_instance_id','0')
                if 'initial_instance_id' in body:producers['manager'].emit('real_initial_route',request_id=sid,instance_id=instance)
                producers['local'+instance].emit('request_engine_admitted',request_id=sid,instance_id=instance)
                def event(kind,**data):producers['api'].emit('real_metrics_event',metric_kind=kind,run_id='run',episode_id=ep,
                    request_id=rid,server_request_id=sid,data=data)
                def frame(kind,**data):return (json.dumps({'run_id':'run','episode_id':ep,'request_id':rid,'server_request_id':sid,
                    'kind':kind,**data})+'\0').encode()
                response=web.StreamResponse();await response.prepare(request)
                event('accepted');await response.write(frame('accepted'))
                event('api_first');await response.write(frame('output',cumulative_tokens=1,text='a'))
                await asyncio.sleep(.001)
                await response.write(frame('output',cumulative_tokens=4,text='abcd'))
                event('server_terminal',outcome='success');await response.write(frame('terminal',outcome='success'))
                return response
            app=web.Application();app.router.add_post('/real_control/{method}',control);app.router.add_post('/generate_real_metrics',generate)
            spec=protocol(run,'/tmp/runtime');spec['horizons'].update(arrival_sec=.1,drain_sec=.1,decision_interval_sec=.02,cleanup_sec=1.)
            trace=[dict(request_id='a',scheduled_arrival_sec=0.,prompt='CPU',output_tokens=4,route_seed=1),
                   dict(request_id='b',scheduled_arrival_sec=.03,prompt='CPU',output_tokens=4,route_seed=2)]
            try:
                async with TestServer(app) as server,ClientSession(timeout=ClientTimeout(total=2.)) as session:
                    bridge=RealBridge(str(server.make_url('')).rstrip('/'),session)
                    runtime={'sampling':{'n':1}};mapping=None
                    with patch('rl_scheduler.real_opportunity_runner.load_frozen_trace',return_value=trace):
                        for scene in ('A','B'):
                            result=await run_episode(spec,runtime,dict(phase='opportunity',scene=scene,seed=2101,qps=3,policy='Stay'),
                                bridge=bridge,session=session,run_dir=run,episode_id='cpu_'+scene,
                                budget=Obj(require=lambda *args:None),session_mapping=mapping)
                            r,ledger,edges,migrations,through=result;mapping=r['mapping']
                            self.assertEqual(r['status'],'passed');self.assertEqual(r['metrics']['completed'],2)
                            self.assertEqual(r['route_audit']['status'],'passed');self.assertEqual(r['metrics']['client_output_tokens'],8)
                            self.assertFalse(migrations);self.assertEqual(ledger.audit(through_ns=through,require_cleanup=True)['status'],'passed')
            finally:
                for producer in producers.values():producer.stream.close()

    async def test_sequential_six_probes_then_twelve_pairs_and_deferred_scales(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);budget_file(run);spec=protocol(run,'/tmp/runtime');calls=[]
            async def episode(spec,runtime,case,**kw):
                calls.append(case);e=row(**case);e['episode_id']=kw['episode_id']
                if e['policy']=='Balanced':e['metrics']['slo_goodput_rps']=1.2
                (run/'runs'/e['episode_id']).mkdir(parents=True)
                return e,Obj(),[0,240*NS],[],240*NS
            with patch('rl_scheduler.real_opportunity_runner.run_episode',side_effect=episode), \
                 patch('rl_scheduler.real_opportunity_runner.reconstruct_rewards',return_value=[]):
                result=await run_opportunity(spec,{'service':{'base_url':'http://127.0.0.1:1'}},run_dir=run,invocation_id='cpu')
            self.assertEqual(result['status'],'passed');self.assertEqual(len(calls),18)
            self.assertTrue(all(c['phase']=='probe' and c['policy']=='Stay' for c in calls[:6]))
            scales=json.loads((run/'opportunity_runs/cpu/reward_scales.json').read_text())
            self.assertEqual(len(scales['provenance']['episode_ids']),6)
            self.assertTrue((run/'opportunity_runs/cpu/final_burst_traces.json').exists())

    async def test_first_invalid_episode_stops_without_retry_or_later_cases(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);budget_file(run);spec=protocol(run,'/tmp/runtime')
            with patch('rl_scheduler.real_opportunity_runner.run_episode',side_effect=RuntimeError('unknown receipt')) as episode:
                with self.assertRaises(RuntimeError):await run_opportunity(spec,{'service':{'base_url':'http://127.0.0.1:1'}},run_dir=run,invocation_id='failed')
            self.assertEqual(episode.await_count,1)
            gate=json.loads((run/'opportunity_runs/failed/opportunity_gate.json').read_text())
            self.assertEqual(gate['status'],'incomplete');self.assertFalse(gate['automatic_retry'])


if __name__=='__main__':unittest.main(verbosity=2)
