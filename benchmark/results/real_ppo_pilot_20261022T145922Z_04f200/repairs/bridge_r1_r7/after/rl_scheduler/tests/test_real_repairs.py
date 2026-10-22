"""CPU protocol regressions for independent bridge review R1--R7.

Run against real method bodies; only actors, KV storage and engine steps are mocked.
No model, Ray initialization, service installation or shared budget writes.
"""
import ast
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import patch

from rl_scheduler.real_bridge import ControlArbiter, RealBridge, digest
from rl_scheduler.real_metrics import RequestLedger, Checkpoint
from rl_scheduler.real_runner import flatten_acks, role_inventory, checkpoint
from rl_scheduler.tests.test_real_bridge import Clock, states, extract_method, load_file
from rl_scheduler.tests.test_real_env import environment, ROOT


def ack(source='p', seq=0, through=100, **extra):
    return dict(source=source, status='passed', durable=True, pending_delivery=0,
                last_sequence=seq, through_ns=through, **extra)


def tree():
    return {'api':ack('api'), 'cluster':{'manager':ack('manager'), 'instances':[
        {'local':ack(f'l{i}'), 'workers':[ack(f'w{i}')]} for i in range(4)]}}


class FlushRepairs(unittest.TestCase):
    def test_shared_and_independent_producers_and_cross_round_regression(self):
        raw=tree();raw['cluster']['instances'][0]={'local':ack('shared',through=200),'workers':[ack('shared',through=100)]}
        self.assertEqual(len(role_inventory(raw)),10)
        merged=list(flatten_acks(raw));self.assertEqual(len(merged),9)
        self.assertEqual(next(a['through_ns'] for a in merged if a['source']=='shared'),200)
        ledger=RequestLedger('r','e',start_ns=0,arrival_end_ns=1,score_deadline_ns=2,
                             expected_sources=['shared'],expected_request_ids=[])
        ledger.acknowledge(checkpoint(ack('shared',through=200)))
        ledger.acknowledge(checkpoint(ack('shared',through=100)))
        self.assertIn('checkpoint_regression:shared',ledger.issues)
        self.assertEqual(len(list(flatten_acks(tree()))),10)

    def test_failed_worker_and_missing_role_never_disappear(self):
        for reason in ('pending_cuda_telemetry','writer_closed','tracing_disabled'):
            for shared in (True,False):
                raw=tree();instance=raw['cluster']['instances'][0]
                if shared:instance['local']=ack('same-process')
                instance['workers']=[{'status':'incomplete','reason':reason}]
                with self.subTest(reason=reason,shared=shared),self.assertRaises(RuntimeError):role_inventory(raw)
                with self.assertRaises(RuntimeError):list(flatten_acks(raw))
        for change in ('missing','empty','extra'):
            raw=tree()
            if change=='missing':raw['cluster']['instances'].pop()
            elif change=='empty':raw['cluster']['instances'][0]['workers']=[]
            else:raw['cluster']['instances'][0]['workers'].append(ack('extra'))
            with self.assertRaises(RuntimeError):role_inventory(raw)

    def test_every_role_requires_explicit_zero_pending_and_durability(self):
        for role in ('api','manager','local','worker'):
            for pending in (None,1):
                raw=tree()
                target={'api':raw['api'],'manager':raw['cluster']['manager'],
                        'local':raw['cluster']['instances'][0]['local'],
                        'worker':raw['cluster']['instances'][0]['workers'][0]}[role]
                if pending is None:target.pop('pending_delivery')
                else:target['pending_delivery']=pending
                with self.subTest(role=role,pending=pending),self.assertRaises(RuntimeError):role_inventory(raw)


class DecisionRepairs(unittest.IsolatedAsyncioTestCase):
    async def test_env_unknown_stops_new_action_preserves_sample_and_cleans(self):
        env,clock,bridge,io=environment();calls=[]
        async def submit(*args):
            calls.append(('submit',args[0]));return {'receipt_status':'unknown','terminal_status':None}
        async def query(rid):calls.append(('query',rid));return {'receipt_status':'unknown','terminal_status':None}
        bridge.submit_migration=submit;bridge.query_receipt=query
        await env.reset();await env.step(1,old_logp=-2.)
        original=deepcopy(env.decisions[0])
        with self.assertRaises(RuntimeError):await env.step(1,old_logp=-3.)
        self.assertEqual(env.decisions,[original]);self.assertEqual(original['sampled_action'],1)
        self.assertEqual(original['old_logp'],-2.);self.assertEqual(original['receipt_status'],'unknown')
        self.assertEqual([c[0] for c in calls],['submit','query'])
        result=await env.close();self.assertFalse(result['valid_episode'])

    async def test_transport_fence_queries_old_id_and_resolves_only_known_terminal(self):
        bridge=RealBridge('http://127.0.0.1:1',None);bridge.lease={'lease_id':'lease'};calls=[];known=[False]
        async def rpc(method,**payload):
            calls.append((method,payload['decision_id']))
            if method=='submit_migration':raise ConnectionError()
            return {'terminal_status':'failed' if known[0] else 'unknown'}
        bridge.call=rpc
        await bridge.submit_migration('old','0','1','c','v')
        with self.assertRaises(RuntimeError):await bridge.submit_migration('new','0','1','c','v')
        self.assertEqual(calls,[('submit_migration','old'),('query_receipt','old')])
        known[0]=True
        await bridge.submit_migration('new','0','1','c','v')
        self.assertEqual(calls[-2:],[('query_receipt','old'),('submit_migration','new')])

    async def test_acquisition_fence_cleanup_and_failover_intervention(self):
        clock=Clock();core=None
        async def provider():
            self.assertTrue(core.ordinary_blocked)
            core.external_intervention('pre_stop')
            return states(clock)
        core=ControlArbiter(provider,None,clock=clock)
        with self.assertRaises(RuntimeError):await core.acquire_controller('r','v','e')
        self.assertFalse(core.acquiring);self.assertIsNone(core.lease)
        self.assertIn('pre_stop',core.invalid_reasons)
        async def unavailable():raise ConnectionError()
        core.state_provider=unavailable
        with self.assertRaises(ConnectionError):await core.acquire_controller('r','v','e')
        self.assertFalse(core.ordinary_blocked)


def cancellation_owner(module, index='0'):
    scheduler=Obj(running=[],waiting=[],swapped=[],pre_alloc_cache_dict={},migrating_out_request_last_stage={})
    engine=Obj(scheduler=[scheduler],request_ids=set(),output_forwarder=Obj(thread_forwarder=Obj(real_pending_delivery=lambda:0)))
    engine.clear_request=engine.request_ids.discard
    coordinator=Obj(migrating_out_request_id_set=set(),migrating_in_request_id_set=set(),pending_migrate_in_request_time={})
    def free(rid):
        scheduler.pre_alloc_cache_dict.pop(rid,None);coordinator.migrating_in_request_id_set.discard(rid)
        coordinator.pending_migrate_in_request_time.pop(rid,None)
    coordinator.free_pre_alloc_cache=free
    async def abort(ids):
        for queue in (scheduler.running,scheduler.waiting,scheduler.swapped):queue[:]=[r for r in queue if r.request_id not in ids]
    async def boundary(rid):
        for r in list(scheduler.running):
            if r.request_id==rid:scheduler.running.remove(r);scheduler.migrating_out_request_last_stage[rid]=r
        await abort({rid})
    backend=Obj(engine=engine,abort_request=abort,remove_running_request=boundary,
                free_src_request=lambda req:engine.clear_request(req.request_id))
    return Obj(backend_engine=backend,migration_coordinator=coordinator,instance_id=index)


class CancellationRepairs(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.module=load_file('_repair_real_control','llumnix/real_control.py')
        self.state_patch=patch.object(self.module,'instance_state',lambda owner:{'epoch':'epoch-'+owner.instance_id})
        self.state_patch.start();self.addCleanup(self.state_patch.stop)

    async def test_migrated_owner_and_paused_preallocated_are_fenced(self):
        owners=[cancellation_owner(self.module,str(i)) for i in range(4)]
        dst=owners[1];dst.backend_engine.engine.scheduler[0].running.append(Obj(request_id='r'))
        dst.backend_engine.engine.request_ids.add('r')
        src=owners[0];s=src.backend_engine.engine.scheduler[0]
        s.migrating_out_request_last_stage['r']=Obj(request_id='r');s.pre_alloc_cache_dict['r']=object()
        src.migration_coordinator.migrating_in_request_id_set.add('r');src.migration_coordinator.pending_migrate_in_request_time['r']=1
        async def actor_call(owner,rid):return await self.module.cancel_instance_request(owner,rid)
        actors={o.instance_id:Obj(real_cancel_request=Obj(remote=lambda rid,o=o:actor_call(o,rid)),
                    real_control_state=Obj(remote=lambda o=o:asyncio.sleep(0,result={
                        'instance_id':o.instance_id,'epoch':'epoch-'+o.instance_id,
                        'healthy':True,'type':'neutral','is_migrating':False}))) for o in owners}
        manager=Obj(instances=actors,enable_pd_disagg=False)
        result=await self.module.ManagerControlMixin.real_cancel_request(manager,'r')
        self.assertEqual(result['status'],'absent')
        for owner in owners:
            self.assertIn('r',owner._real_cancelled_requests)
            self.assertIn('r',owner.migration_coordinator._real_cancelled_requests)
            self.assertNotIn('r',owner.backend_engine.engine.request_ids)
        self.assertFalse(s.migrating_out_request_last_stage);self.assertFalse(s.pre_alloc_cache_dict)

    async def test_unfinished_dispatch_transfer_and_delivery_stay_unknown(self):
        for reason in ('generate','incoming','outgoing','controlled','delivery'):
            owner=cancellation_owner(self.module)
            if reason=='generate':owner._real_generating={'r'}
            elif reason=='incoming':owner._real_inbound_tasks={object()}
            elif reason=='outgoing':owner.migration_coordinator.migrating_out_request_id_set.add('r')
            elif reason=='controlled':owner._real_control_inflight='d'
            else:owner.backend_engine.engine.output_forwarder.thread_forwarder.real_pending_delivery=lambda:1
            result=await self.module.cancel_instance_request(owner,'r')
            self.assertEqual(result['status'],'unknown',reason)
            self.assertIn('r',owner._real_cancelled_requests)
        epochs={str(i):'epoch-'+str(i) for i in range(4)}
        actors={str(i):Obj(real_cancel_request=Obj(remote=lambda rid,i=i:asyncio.sleep(0,result={
            'status':'absent','fenced':True,'pending_delivery':0,'instance_id':str(i),'epoch':'epoch-'+str(i)})),
            real_control_state=Obj(remote=lambda i=i:asyncio.sleep(0,result={
                'instance_id':str(i),'epoch':epochs[str(i)],'healthy':True,'type':'neutral','is_migrating':False})))
            for i in range(4)}
        manager=Obj(instances=actors,enable_pd_disagg=False,_real_dispatch_tasks={'r':{object()}})
        self.assertEqual((await self.module.ManagerControlMixin.real_cancel_request(manager,'r'))['status'],'unknown')
        manager._real_dispatch_tasks.clear();epochs['1']='rebuilt'
        self.assertEqual((await self.module.ManagerControlMixin.real_cancel_request(manager,'r'))['status'],'unknown')
        epochs['1']='epoch-1'
        self.assertEqual((await self.module.ManagerControlMixin.real_cancel_request(manager,'r'))['status'],'absent')

    async def test_uncertain_RPC_cannot_produce_absence_proof(self):
        async def lost(rid):raise ConnectionError('reply lost')
        actors={str(i):Obj(real_cancel_request=Obj(remote=lost)) for i in range(4)}
        manager=Obj(instances=actors,enable_pd_disagg=False)
        with self.assertRaises(ConnectionError):
            await self.module.ManagerControlMixin.real_cancel_request(manager,'r')
        self.assertIn('r',manager._real_cancelled_requests)

    async def test_late_generate_preallocate_receive_and_commit_do_not_create_request(self):
        owner=cancellation_owner(self.module);await self.module.cancel_instance_request(owner,'r')
        generate=extract_method('llumnix/llumlet/llumlet.py','Llumlet','generate',{})
        await generate(owner,'r',None,1)  # Must return before calling any backend add.
        response=lambda **kw:Obj(**kw)
        for name,args in [('pre_alloc_cache',('r',None,0,5,[],True)),('recv_cache',('r',)),('commit_dst_request',('r',None))]:
            method=extract_method('llumnix/llumlet/migration_coordinator.py','MigrationCoordinator',name,{'MigrationResponse':response})
            result=method(owner.migration_coordinator,*args)
            if asyncio.iscoroutine(result):result=await result
            self.assertFalse(result.success,name)
        manager_generate=extract_method('llumnix/manager.py','Manager','generate',{})
        with self.assertRaises(RuntimeError):await manager_generate(Obj(_real_cancelled_requests={'r'}),'r',None)

    async def test_engine_boundary_and_output_barrier_are_required(self):
        owner=cancellation_owner(self.module);entered=asyncio.Event();release=asyncio.Event()
        owner.backend_engine.engine.request_ids.add('r')
        async def boundary(rid):entered.set();await release.wait()
        owner.backend_engine.remove_running_request=boundary
        task=asyncio.create_task(self.module.cancel_instance_request(owner,'r'))
        await entered.wait();self.assertFalse(task.done());release.set()
        self.assertEqual((await task)['status'],'absent')
        boundary_method=extract_method('llumnix/backends/vllm/llm_engine.py','BackendVLLM','_remove_running_request',{})
        seen=[]
        backend=Obj(engine=Obj(_real_cancelled_requests={'r'},scheduler=[Obj(remove_running_request=lambda rid:seen.append(('pause',rid)))],
                               abort_request=lambda ids:seen.append(('abort',ids))))
        boundary_method(backend,'r')
        self.assertEqual(seen,[('pause','r'),('abort',{'r'})])

    async def test_API_uses_Manager_proof_and_drains_queue_before_clearing(self):
        tree_ast=ast.parse((ROOT/'llumnix/entrypoints/vllm/api_server.py').read_text())
        method=next(n for n in tree_ast.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_real_metrics_abort_confirmed')
        events=[];pending=[1]
        async def call(method,rid,**kw):
            events.append(('manager',rid));return {'status':'absent'}
        client=Obj(manager=Obj(real_cancel_request=object()),request_output_queue=Obj(qsize=lambda:pending[0]),
                   _clear_client_request_states=lambda rid:events.append(('clear',rid)))
        ns={'llumnix_client':client,'asyncio_wait_for_ray_remote_call_with_timeout':call}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'api_server.py','exec'),ns)
        task=asyncio.create_task(ns['_real_metrics_abort_confirmed']('r'))
        await asyncio.sleep(.001);self.assertEqual(events,[('manager','r')]);self.assertFalse(task.done())
        pending[0]=0;self.assertTrue(await task);self.assertEqual(events[-1],('clear','r'))

    async def test_unproven_abort_never_labels_server_cancelled(self):
        import httpx
        from fastapi import FastAPI
        from rl_scheduler.real_client import NulJSONDecoder
        module=load_file('_repair_stream','llumnix/entrypoints/vllm/real_metrics_stream.py')
        ready=asyncio.Event();stopped=asyncio.Event()
        class Client:
            async def generate(self,*args,**kwargs):
                ready.set()
                async def stream():
                    await stopped.wait()
                    if False:yield None
                return Obj(generator=stream)
        async def unproven(rid):stopped.set();return False
        app=FastAPI();events=[]
        module.install_real_metrics_routes(app,Client,lambda **kw:kw,lambda *a,**kw:events.append(kw),None,unproven)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            task=asyncio.create_task(client.post('/generate_real_metrics',json=dict(run_id='r',episode_id='e',request_id='a',prompt='x')))
            await ready.wait()
            result=await client.post('/cancel_real_metrics',json=dict(run_id='r',episode_id='e',request_ids=['a']))
            self.assertEqual(result.json()['results'][0]['status'],'unknown')
            output=await task;frames=NulJSONDecoder().feed(output.content)
            self.assertEqual(frames[-1]['outcome'],'unknown')


class GateRepairs(unittest.TestCase):
    def setUp(self):self.cli=load_file('_repair_cli','benchmark/run_real_ppo.py')

    def fixture(self,directory):
        root=Path(directory);run=root/'run';(run/'reviews').mkdir(parents=True);(run/'gates').mkdir()
        (root/'source.py').write_text('cpu fixture');(run/'cfg.json').write_text('{}');(run/'proof.txt').write_text('cpu proof')
        source={'source.py':self.cli.sha(root/'source.py')}
        gate={'status':'passed','checks':[{'name':'cpu','status':'passed'}], 'source_hashes':source,'source_hash':digest(source),
              'config_hashes':{'cfg.json':self.cli.sha(run/'cfg.json')},'evidence_hashes':{'proof.txt':self.cli.sha(run/'proof.txt')}}
        review=dict(gate,stage='04_bridge_review')
        (run/'reviews/bridge_review.json').write_text(json.dumps(review))
        original=self.cli.validate_gate
        def validate(path,run_dir):return original(path,run_dir,source_root=root)
        return run,gate,patch.object(self.cli,'validate_gate',side_effect=validate)

    def test_review_absent_failed_incomplete_and_stale_are_blocked(self):
        for state in ('missing','failed','incomplete','stale','unbound'):
            with tempfile.TemporaryDirectory() as d:
                run,gate,patched=self.fixture(d);p=run/'reviews/bridge_review.json';review=json.loads(p.read_text())
                if state=='missing':p.unlink()
                elif state=='stale':(Path(d)/'source.py').write_text('changed')
                elif state=='unbound':review['source_hashes']={'source.py':gate['source_hashes']['source.py']};gate['source_hashes']['extra.py']='x'
                else:review['status']=state
                if state!='missing':p.write_text(json.dumps(review))
                with patched,self.assertRaises((OSError,RuntimeError)):
                    self.cli.live_dependencies(run,'telemetry',[gate])

    def test_live_frozen_order_and_result_hash_chain(self):
        with tempfile.TemporaryDirectory() as d:
            run,gate,patched=self.fixture(d)
            with patched:
                dependencies=self.cli.live_dependencies(run,'telemetry',[gate])
                for index,key in enumerate(self.cli.LIVE_ORDER):
                    if index:
                        with self.assertRaises(OSError):self.cli.live_dependencies(run,key,[gate])
                        previous=self.cli.LIVE_ORDER[index-1]
                        folder=run/'smoke_runs'/previous;folder.mkdir(parents=True)
                        checks={name:True for name in self.cli.LIVE_CHECKS[previous]}
                        (folder/'result.json').write_text(json.dumps({'status':'passed','checks':checks,'results':[]}))
                        self.cli.publish_live_gate(run,previous,previous,{'dependency_gate_hashes':dependencies})
                        dependencies=self.cli.live_dependencies(run,key,[gate])
                self.assertEqual(len(dependencies),4)
                (run/'smoke_runs/telemetry/result.json').write_text('{}')
                with self.assertRaises(RuntimeError):self.cli.live_dependencies(run,'overhead',[gate])


if __name__=='__main__':unittest.main(verbosity=2)
