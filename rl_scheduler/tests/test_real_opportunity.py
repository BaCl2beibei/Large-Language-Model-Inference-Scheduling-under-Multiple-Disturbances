"""CPU invariants for the frozen P3 protocol and sequential orchestration."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import patch

from rl_scheduler.real_bridge import digest
from rl_scheduler.real_metrics import Checkpoint,Event,EventCursor,NS,RequestLedger
from rl_scheduler.real_opportunity import (protocol,episode_schedule,balanced_action,healthy,select_loads,paired_gate,
                                         make_trace,maximum_duration,readonly_budget,metric_summary)
from rl_scheduler.real_opportunity_runner import (prime_causal_history,reconcile_migrations,route_audit,
                                                  OpportunityRequestLedger,request_submission_audit,run_opportunity)
from rl_scheduler.tests.test_real_metrics import Fixture
from rl_scheduler.tests.test_real_bridge import Clock,states,extract_method,load_file
from rl_scheduler.tests.test_real_env import budget_file,ROOT


def row(scene='A',seed=2101,policy='Stay',qps=3,phase='opportunity'):
    return {'episode_id':f'{phase}-{scene}-{seed}-{policy}','phase':phase,'scene':scene,'seed':seed,'policy':policy,'qps':qps,
        'status':'passed','trace_hash':f'{qps}-{seed}','external_case_hash':f'{qps}-{seed}-{scene}',
        'mapping':[[str(i),'e'+str(i)] for i in range(4)],'send_lateness_p95_ms':5.,
        'queue_last_third_samples':10,'queue_last_third_positive':5,
        'metrics':{'offered':100,'sent':100,'accepted':100,'completed':100,'duration_sec':240,'completion_rate':1.,
            'nonmanual_error_rate':0.,'first_output_censored_fraction':0.,'terminal_censored_fraction':0.,
            'slo_goodput_rps':1.,'outstanding_request_seconds':480.}}


def invocation_lease(run,invocation_id):
    from rl_scheduler.real_budget import atomic_json
    path=Path(run)/'test_lease.json'
    atomic_json(path,{'schema':'real-opportunity-budget-lease-v1','run_id':'run','invocation_id':invocation_id,
        'created_wall':time.time(),'created_monotonic_ns':time.monotonic_ns(),'max_runtime_sec':3600.,
        'deadline_monotonic_ns':time.monotonic_ns()+3600_000_000_000,'closed_wall':None,'close_reason':None})
    return path


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
        r=probes[0]
        r['metrics'].update(offered=99,sent=97,accepted=97,completed=97,completion_rate=1.)
        self.assertTrue(healthy(r,self.spec))
        r['send_lateness_p95_ms']=1001
        self.assertTrue(healthy(r,self.spec))

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
            r['metrics'].update(offered=300,accepted=300,completed=300,slo_completed=100,nonmanual_errors=10)
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
            args=Obj(output_dir=run,config=run/'runtime.yaml',manage_services=True,dry_run=True,invocation_id='test',
                     user_authorized_budget_override=False,continuation_id=None)
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
        f=Fixture();f.plan_send('a');f.add('accepted',.1,'a',server_request_id='server-a');f.output(1,1,'a');f.terminal(2,'a')
        f.plan_send('b');f.add('accepted',.1,'b',server_request_id='server-b');f.terminal(3,'b',outcome='timeout',http_status=None)
        f.plan_send('c');f.add('accepted',.1,'c',server_request_id='server-c');f.add('cancel_requested',4,'c');f.terminal(5,'c',outcome='cancelled',http_status=499,reason='manual_drain');f.add('cancel_ack',6,'c',confirmed=True)
        f.plan_send('d');f.add('accepted',.1,'d',server_request_id='server-d');f.ack()
        metrics=metric_summary(f.ledger)
        self.assertEqual(metrics['nonmanual_errors'],1);self.assertEqual(metrics['nonmanual_error_rate'],.25)
        self.assertEqual(metrics['terminal_censored_fraction'],.25)
        self.assertEqual(metrics['first_output_censored_fraction'],.75)

    def test_submission_audit_separates_unissued_requests_from_missing_acceptance(self):
        rows=[{'request_id':'a','sent_ns':10,'accepted_ns':20},
              {'request_id':'b','sent_ns':30,'accepted_ns':None},
              {'request_id':'c','sent_ns':None,'accepted_ns':None}]
        result=request_submission_audit(rows,['a','b','c'])
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['not_sent_ids'],['c'])
        self.assertEqual(result['sent_without_acceptance_ids'],['b'])
        self.assertEqual(result['accepted'],1)

    def test_p3_sequence_audit_uses_durable_episode_baseline_and_keeps_new_gaps(self):
        def make_audit(sequences):
            ledger=OpportunityRequestLedger('run','episode',start_ns=1,arrival_end_ns=2,
                score_deadline_ns=3,expected_sources=['worker'],expected_request_ids=[],
                sequence_baselines={'worker':4})
            for sequence in sequences:
                ledger.ingest([Event('worker',sequence,'diagnostic',2,2,'run','episode')])
            ledger.acknowledge(Checkpoint('worker',6,3,3,pending_delivery=0,durable=True))
            return ledger.audit()
        self.assertEqual(make_audit([5,6])['status'],'passed')
        missing=make_audit([6])
        self.assertEqual(missing['status'],'incomplete')
        self.assertEqual(missing['sequence_gaps'],{'worker':[5]})

    def test_p3_online_observation_matches_full_ledger_without_rebuilding_output_history(self):
        fast=OpportunityRequestLedger('run','episode',start_ns=NS,arrival_end_ns=31*NS,
            score_deadline_ns=91*NS,expected_sources=['client'],expected_request_ids=['a','b'],
            sequence_baselines={'client':0})
        full=RequestLedger('run','episode',start_ns=NS,arrival_end_ns=31*NS,
            score_deadline_ns=91*NS,expected_sources=['client'],expected_request_ids=['a','b'])
        records=[
            ('planned','a',2*NS,{'scheduled_ns':2*NS}),('sent','a',2*NS+1,{}),
            ('client_first','a',2*NS+2,{}),('client_output','a',2*NS+2,{'cumulative_tokens':1}),
            ('client_output','a',6*NS,{'cumulative_tokens':4}),
            ('planned','b',9*NS,{'scheduled_ns':9*NS}),('sent','b',9*NS+1,{}),
            ('client_terminal','a',10*NS,{'outcome':'success'}),
        ]
        events=[Event('client',i,kind,stamp,stamp,'run','episode',rid,data=data)
                for i,(kind,rid,stamp,data) in enumerate(records,1)]
        fast.ingest(events);full.ingest(events)
        now=11*NS
        observed=fast.observe(now)['requests'];reference=full.observe(now)['requests']
        by_id={row['request_id']:row for row in observed}
        for expected in reference:
            actual=by_id[expected['request_id']]
            for field in ('sent_ns','client_first_ns','client_last_ns','client_terminal_ns'):
                self.assertEqual(actual[field],expected[field],(expected['request_id'],field))
            window=lambda row:sum(count for stamp,count in row['token_increments'] if now-5*NS<stamp<=now)
            self.assertEqual(window(actual),window(expected))
        self.assertEqual(fast.observe(now)['outstanding'],1)

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
        f=Fixture();f.plan_send('a');f.add('accepted',.1,'a',server_request_id='server-a')
        f.output(1,1,'a');f.terminal(2,'a');f.ack()
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
        from rl_scheduler.real_env import RealEnv
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
            spec=protocol(run,'/tmp/runtime');spec['horizons'].update(arrival_sec=.5,drain_sec=.5,decision_interval_sec=.02,cleanup_sec=1.)
            trace=[dict(request_id='a',scheduled_arrival_sec=0.,prompt='CPU',output_tokens=4,route_seed=1),
                   dict(request_id='b',scheduled_arrival_sec=.03,prompt='CPU',output_tokens=4,route_seed=2),
                   dict(request_id='c',scheduled_arrival_sec=.45,prompt='CPU',output_tokens=4,route_seed=3)]
            try:
                async with TestServer(app) as server,ClientSession(timeout=ClientTimeout(total=2.)) as session:
                    bridge=RealBridge(str(server.make_url('')).rstrip('/'),session)
                    runtime={'sampling':{'n':1}};mapping=None
                    original_observation=RealEnv._observation;blocked_once=[False]
                    async def stall_control_loop(env):
                        if not blocked_once[0]:
                            blocked_once[0]=True
                            time.sleep(.15)
                        return await original_observation(env)
                    with patch('rl_scheduler.real_opportunity_runner.load_frozen_trace',return_value=trace), \
                         patch.object(RealEnv,'_observation',stall_control_loop):
                        for scene in ('A','B'):
                            result=await run_episode(spec,runtime,dict(phase='opportunity',scene=scene,seed=2101,qps=3,policy='Stay'),
                                bridge=bridge,session=session,run_dir=run,episode_id='cpu_'+scene,
                                budget=Obj(require=lambda *args:None),session_mapping=mapping)
                            r,ledger,edges,migrations,through=result;mapping=r['mapping']
                            self.assertEqual(r['status'],'passed');self.assertEqual(r['metrics']['sent'],3)
                            self.assertEqual(r['metrics']['completed'],3)
                            self.assertEqual(r['route_audit']['status'],'passed');self.assertEqual(r['metrics']['client_output_tokens'],12)
                            lateness=[row['send_lateness_ms'] for row in ledger._rows()[0]]
                            self.assertLess(max(lateness),50.)
                            self.assertFalse(migrations);self.assertEqual(ledger.audit(through_ns=through,require_cleanup=True)['status'],'passed')
                    async def fail_control(*args,**kwargs):raise RuntimeError('synthetic control failure')
                    with patch('rl_scheduler.real_opportunity_runner.load_frozen_trace',return_value=trace), \
                         patch.object(RealEnv,'step',fail_control):
                        with self.assertRaisesRegex(RuntimeError,'synthetic control failure'):
                            await run_episode(spec,runtime,dict(phase='opportunity',scene='A',seed=2101,qps=3,policy='Stay'),
                                bridge=bridge,session=session,run_dir=run,episode_id='cpu_control_failure',
                                budget=Obj(require=lambda *args:None),session_mapping=mapping)
                    interrupted=json.loads((run/'runs/cpu_control_failure/interrupted.json').read_text())
                    scoring=interrupted['workload_scoring']
                    self.assertFalse(scoring['valid_for_analysis'])
                    self.assertEqual(scoring['offered'],3)
                    self.assertEqual(scoring['sent_by_deadline'],3)
                    self.assertEqual(scoring['completed_by_deadline'],3)
                    self.assertEqual(scoring['accepted_in_arrival_window'],3)
                    self.assertEqual(scoring['completion_rate_accepted'],1.)
                    self.assertEqual(scoring['unfinished_ids'],[])
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
                result=await run_opportunity(spec,{'service':{'base_url':'http://127.0.0.1:1'}},run_dir=run,invocation_id='cpu',
                                             budget_override_lease=invocation_lease(run,'cpu'))
            self.assertEqual(result['status'],'passed');self.assertEqual(len(calls),18)
            self.assertTrue(all(c['phase']=='probe' and c['policy']=='Stay' for c in calls[:6]))
            scales=json.loads((run/'opportunity_runs/cpu/reward_scales.json').read_text())
            self.assertEqual(len(scales['provenance']['episode_ids']),6)
            self.assertTrue((run/'opportunity_runs/cpu/final_burst_traces.json').exists())

    async def test_continuation_reuses_audited_probes_without_repeating_their_keys(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);budget_file(run);spec=protocol(run,'/tmp/runtime');calls=[]
            prior=[row(scene,7101,'Stay',3,'probe') for scene in ('A','B')]
            for item in prior:item['request_submission_audit']={'status':'passed'}
            async def episode(spec,runtime,case,**kw):
                calls.append(case);e=row(**case);e['episode_id']=kw['episode_id']
                if e['policy']=='Balanced':e['metrics']['slo_goodput_rps']=1.2
                (run/'runs'/e['episode_id']).mkdir(parents=True)
                return e,Obj(),[0,240*NS],[],240*NS
            with patch('rl_scheduler.real_opportunity_runner.run_episode',side_effect=episode), \
                 patch('rl_scheduler.real_opportunity_runner.reconstruct_rewards',return_value=[]):
                result=await run_opportunity(spec,{'service':{'base_url':'http://127.0.0.1:1'}},run_dir=run,
                    invocation_id='cpu',budget_override_lease=invocation_lease(run,'cpu'),
                    artifact_dir=run/'attempts/recovery2',episode_prefix='cpu_recovery2',prior_probes=prior)
            self.assertEqual(result['status'],'passed');self.assertEqual(len(calls),16)
            self.assertEqual([(r['scene'],r['qps'],r['seed']) for r in json.loads((run/'attempts/recovery2/probes.json').read_text())],
                             [('A',3,7101),('B',3,7101),('A',6,7102),('B',6,7102),('A',9,7103),('B',9,7103)])
            self.assertEqual([(r['scene'],r['qps']) for r in calls[:2]],[('A',6),('B',6)])

    async def test_continuation_rejects_prior_probes_without_complete_submission_audit(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);budget_file(run);spec=protocol(run,'/tmp/runtime')
            prior=[row(scene,7101,'Stay',3,'probe') for scene in ('A','B')]
            with self.assertRaisesRegex(ValueError,'cannot reuse probes without at least one service-accepted request'):
                await run_opportunity(spec,{'service':{'base_url':'http://127.0.0.1:1'}},run_dir=run,
                    invocation_id='cpu',budget_override_lease=invocation_lease(run,'cpu'),prior_probes=prior)

    def test_causal_cursor_starts_at_durable_baseline_then_reads_new_records(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.jsonl'
            first=Event('worker',1,'diagnostic',1,1,'run','old',data={'value':'old'})
            second=Event('worker',2,'diagnostic',2,2,'run','new',data={'value':'new'})
            path.write_text(json.dumps(asdict(first))+'\n')
            cursor=EventCursor();prime_causal_history(cursor,[path])
            with path.open('a') as stream:stream.write(json.dumps(asdict(second))+'\n')
            found=cursor.poll(path)
            self.assertEqual([(event.sequence,event.episode_id) for event in found],[(2,'new')])
            self.assertEqual(cursor.issues,[])

    async def test_first_invalid_episode_stops_without_retry_or_later_cases(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);budget_file(run);spec=protocol(run,'/tmp/runtime')
            with patch('rl_scheduler.real_opportunity_runner.run_episode',side_effect=RuntimeError('unknown receipt')) as episode:
                with self.assertRaises(RuntimeError):await run_opportunity(spec,{'service':{'base_url':'http://127.0.0.1:1'}},run_dir=run,
                    invocation_id='failed',budget_override_lease=invocation_lease(run,'failed'))
            self.assertEqual(episode.await_count,1)
            gate=json.loads((run/'opportunity_runs/failed/opportunity_gate.json').read_text())
            self.assertEqual(gate['status'],'incomplete');self.assertFalse(gate['automatic_retry'])


if __name__=='__main__':unittest.main(verbosity=2)
