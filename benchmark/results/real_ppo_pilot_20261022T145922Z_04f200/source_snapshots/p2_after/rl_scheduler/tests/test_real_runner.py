"""CPU end-to-end traffic/ledger/control fixture. No Ray or model imports."""
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rl_scheduler.real_bridge import RealBridge,ControlArbiter
from rl_scheduler.real_budget import atomic_json
from rl_scheduler.real_env import RealEnv,RealEnvConfig
from rl_scheduler.real_runner import LiveEpisodeIO,server_request_id
from rl_scheduler.tests.test_real_bridge import Clock,states,load_file,extract_method
from rl_scheduler.tests.test_real_env import budget_file,ROOT


class Producer:
    def __init__(self,directory,source):
        self.source,self.sequence=source,0
        self.stream=(Path(directory)/('events_'+source+'.jsonl')).open('wb')
    def emit(self,event,**data):
        self.sequence+=1
        payload={'stream_id':self.source,'event_sequence':self.sequence,'event':event,
                 'monotonic_ns':time.monotonic_ns(),**data}
        self.stream.write((json.dumps(payload)+'\n').encode());self.stream.flush()
    def checkpoint(self):
        self.stream.flush();os.fsync(self.stream.fileno());now=time.monotonic_ns()
        return {'status':'passed','source':self.source,'last_sequence':self.sequence,
                'through_ns':now,'received_ns':now,'pending_delivery':0,'durable':True}


class LiveFixtureTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_trace_generator_receives_required_arrival_contract(self):
        from rl_scheduler.real_runner import make_trace
        class Tokenizer:
            def get_vocab(self):return {'a':1}
            def get_vocab_size(self):return 2
            def decode(self,ids,**kwargs):return ' '.join('a' for x in ids)
            def encode(self,text,**kwargs):return [1]*len(text.split())
        cfg={'service':{'model_path':'unused','max_model_len':2112}}
        case={'arrival_sec':20,'qps':.2,'input_tokens':16,'output_tokens':4,'max_requests':20}
        with patch('disturbance_exp.workload.load_local_tokenizer',return_value=Tokenizer()):
            rows=make_trace(cfg,case,6101)
        self.assertGreater(len(rows),1)
        self.assertTrue(all(row['input_tokens']+row['output_tokens']<=2112 for row in rows))

    async def test_complete_CPU_HTTP_episode_through_real_env_IO_and_ledger(self):
        from aiohttp import web,ClientSession,ClientTimeout
        from aiohttp.test_utils import TestServer
        with tempfile.TemporaryDirectory() as directory:
            run_dir=Path(directory);causal=run_dir/'causal';causal.mkdir();budget_file(run_dir)
            atomic_json(run_dir/'service_session.json',{'causal_dir':str(causal)})
            producers={name:Producer(causal,name) for name in ('api','manager',*[f'local{i}' for i in range(4)],*[f'worker{i}' for i in range(4)])}
            current=states(Clock())
            for s in current:s.update(running=0,used_blocks=0,free_blocks=1000)
            async def provider():
                now=time.monotonic_ns()
                return [dict(s,measured_ns=now) for s in current]
            async def dispatch(receipt):raise AssertionError('Stay test must not dispatch migration')
            core=ControlArbiter(provider,dispatch)
            async def control(request):
                method=request.match_info['method'];body=await request.json()
                if method=='flush':
                    return web.json_response({'api':producers['api'].checkpoint(),'cluster':{
                        'manager':producers['manager'].checkpoint(),'instances':[{
                            'local':producers[f'local{i}'].checkpoint(),'workers':[producers[f'worker{i}'].checkpoint()]} for i in range(4)]}})
                return web.json_response(await getattr(core,method)(**body))
            async def generate(request):
                body=await request.json();rid=body['request_id'];sid=server_request_id('run','ep',rid)
                def emit(kind,**data):
                    producers['api'].emit('real_metrics_event',metric_kind=kind,run_id='run',episode_id='ep',
                                           request_id=rid,server_request_id=sid,data=data)
                def frame(kind,**data):return (json.dumps({'run_id':'run','episode_id':'ep','request_id':rid,
                    'server_request_id':sid,'kind':kind,**data})+'\0').encode()
                response=web.StreamResponse();await response.prepare(request)
                emit('accepted');await response.write(frame('accepted'))
                producers['local0'].emit('engine_request_output',request_id=sid,instance_id='0',output_tokens=1)
                emit('api_first');await response.write(frame('output',cumulative_tokens=1,text='a'))
                await asyncio.sleep(.001)
                await response.write(frame('output',cumulative_tokens=4,text='abcd'))
                emit('server_terminal',outcome='success');await response.write(frame('terminal',outcome='success'))
                return response
            app=web.Application();app.router.add_post('/real_control/{method}',control);app.router.add_post('/generate_real_metrics',generate)
            io=None;env=None
            try:
                async with TestServer(app) as server,ClientSession(timeout=ClientTimeout(total=2)) as session:
                    bridge=RealBridge(str(server.make_url('')).rstrip('/'),session)
                    trace=[{'request_id':'r0','scheduled_arrival_sec':0.,'prompt':'test','output_tokens':4},
                           {'request_id':'r1','scheduled_arrival_sec':.03,'prompt':'test','output_tokens':4}]
                    io=LiveEpisodeIO(bridge,session,run_id='run',episode_id='ep',run_dir=run_dir,trace=trace,
                                     sampling={'n':1},arrival_sec=.1)
                    budget=SimpleNamespace(require=lambda required:None)
                    env=RealEnv(bridge,io,budget,run_id='run',episode_id='ep',ledger_factory=io.ledger,
                        config=RealEnvConfig(arrival_sec=.1,drain_sec=.1,decision_interval_sec=.02,cleanup_sec=1.))
                    await env.reset(mode='stay')
                    done=False
                    while not done:_,_,done,_,_=await env.step(0)
                    result=await env.finish()
                    self.assertEqual(result['status'],'passed',result['invalid_reasons'])
                    self.assertEqual(result['score']['fixed']['completed'],2)
                    self.assertEqual(result['score']['fixed']['client_output_tokens'],8)
                    self.assertEqual(result['score']['fixed']['slo_goodput_rps'],10.)
                    self.assertEqual(len(env.ledger.sources),11)
                    self.assertTrue((io.directory/'flush_history.json').exists())
            finally:
                if io:await io.close()
                for producer in producers.values():producer.stream.close()

    async def test_backend_prepare_fence_is_checked_after_destination_rebuild(self):
        result_type=lambda **kw:SimpleNamespace(**kw)
        method=extract_method('llumnix/llumlet/migration_coordinator.py','MigrationCoordinator','pre_alloc_cache',
                              {'MigrationResponse':result_type})
        called=[];obj=SimpleNamespace(_real_capacity_guards={},has_migration_slot=lambda:True,
                                     _pre_alloc_cache=lambda *args:called.append(args) or result_type(success=True))
        result=method(obj,'r',None,0,20,[],True,real_expected_epoch='old_epoch')
        self.assertFalse(result.success);self.assertFalse(called)
        obj._real_capacity_guards['r']='old_epoch'
        self.assertTrue(method(obj,'r',None,0,20,[],True,real_expected_epoch='old_epoch').success)
        # No extra arguments retains the legacy contract.
        self.assertTrue(method(obj,'legacy',None,0,20,[],True).success)

    async def test_guard_release_failure_keeps_unknown_source_fence(self):
        module=load_file('_guard_release_failure','llumnix/real_control.py');clock=Clock();src,dst=states(clock)[:2]
        request=SimpleNamespace(request_id='request-0',is_migrating=False,stage_num_blocks_list=[])
        candidates=[request]
        coordinator=SimpleNamespace(enable_engine_migration_interface=False,
            migration_scheduler=SimpleNamespace(request_migration_policy='SR',_get_shortest_running_request=lambda *args:candidates))
        owner=SimpleNamespace(migration_coordinator=coordinator)
        async def target_state():return dict(dst)
        async def prepare(*args):candidates.clear();return {'status':'prepared'}
        async def release(*args):raise ConnectionError('lost destination')
        actor=SimpleNamespace(real_control_state=SimpleNamespace(remote=target_state),real_control_reserve=SimpleNamespace(remote=prepare),
                              real_control_unreserve=SimpleNamespace(remote=release))
        with patch.object(module,'instance_state',lambda owner:dict(src)),patch.object(module.time,'monotonic_ns',clock):
            result=await module.controlled_migrate(owner,actor,SimpleNamespace(instance_id='1'),None,
                expected_candidate_id='request-0',decision_id='d',migration_id='m',source_epoch='e0',destination_epoch='e1')
        self.assertEqual(result['terminal_status'],'unknown');self.assertEqual(owner._real_control_inflight,'d')

    async def test_budget_watchdog_expires_and_stops_only_run_service(self):
        import rl_scheduler.real_service as module
        calls=[]
        budget=SimpleNamespace(status=lambda:{'started':True,'exhausted':False,
            'remaining':{'total_experiment_wall_time_sec':59.,'P0_P3_sec':59.}},expire=lambda:calls.append('expired'))
        service=SimpleNamespace(server_name='owned_task',_supervisor=lambda *a,**k:calls.append(a),
                                owner=SimpleNamespace(stop=lambda **kw:calls.append('owned_cleanup')))
        with patch.object(module,'BudgetLedger',return_value=budget),patch.object(module,'ManagedService',return_value=service), \
             patch.object(module.signal,'signal'):
            module.run_watchdog(SimpleNamespace(run_dir='x',root='x',config='x'))
        self.assertEqual(calls,['expired',('stop','owned_task'),'owned_cleanup'])


if __name__=='__main__':unittest.main()
