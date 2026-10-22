"""CPU concurrency, ownership, runtime-adapter and control transport checks."""
import ast
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl_scheduler.real_bridge import ControlArbiter, RealBridge, PAIRS, capacity_ok, CumulativeTokenCounter

ROOT=Path(__file__).resolve().parents[2]


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def extract_method(path,class_name,method_name,namespace):
    tree=ast.parse((ROOT/path).read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==class_name)
    method=next(node for node in cls.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==method_name)
    method.decorator_list=[]
    unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    ast.fix_missing_locations(unit);exec(compile(unit,str(ROOT/path),'exec'),namespace)
    return namespace[method_name]


class Clock:
    def __init__(self):self.now=100_000_000_000
    def __call__(self):return self.now
    def advance(self,seconds):self.now+=int(seconds*1e9)


def states(clock):
    return [{'instance_id':str(i),'epoch':'e'+str(i),'measured_ns':clock(),
             'healthy':True,'type':'neutral','is_migrating':False,'has_slot':True,
             'total_blocks':1000,'free_blocks':900,'used_blocks':100,'waiting_blocks':0,
             'candidate_id':'request-'+str(i),'candidate_blocks':20,'candidate_length':128,
             'candidate_stage':'decode','candidate_policy':'SR','running':1,'waiting':0,
             'head_wait_sec':0.,'pending_delivery':0,'preallocated_requests':0,'paused_requests':0,
             'output_tokens_total':0,'last_step_ms':10.} for i in range(4)]


class Harness:
    def __init__(self,mode='active'):
        self.clock=Clock();self.states=states(self.clock);self.calls=[]
        self.started=asyncio.Event();self.finish=asyncio.Event();self.mode=mode
        async def provider():return deepcopy(self.states)
        async def dispatch(receipt):
            self.calls.append(receipt);self.started.set();await self.finish.wait()
            return {'terminal_status':'succeeded','migration_id':receipt['migration_id'],
                    'actual_request_id':receipt['expected_candidate_id'],'copied_blocks':20}
        self.arbiter=ControlArbiter(provider,dispatch,clock=self.clock)

    async def ready(self):
        self.lease=await self.arbiter.acquire_controller('run','policy','ep',self.mode)
        self.snapshot=await self.arbiter.snapshot();return self

    async def submit(self,decision_id='d',**overrides):
        args=dict(lease_id=self.lease['lease_id'],decision_id=decision_id,src_id='0',dst_id='1',
                  expected_candidate_id='request-0',snapshot_version=self.snapshot['version'])
        args.update(overrides);return await self.arbiter.submit_migration(**args)

    async def complete(self):
        self.finish.set();await asyncio.sleep(0);await asyncio.sleep(0)


class ArbiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_instance_production_counter_deduplicates_and_excludes_migrated_prefix(self):
        counter=CumulativeTokenCounter()
        self.assertEqual(counter.observe('r',5),5)
        self.assertEqual(counter.observe('r',3),0)
        self.assertEqual(counter.observe('r',5),0)
        self.assertEqual(counter.observe('r',6),1)
        counter.baseline('migrated',100)
        self.assertEqual(counter.observe('migrated',101),1)
        self.assertEqual(counter.total,7)
    async def test_atomic_global_slot_under_race(self):
        h=await Harness().ready()
        receipts=await asyncio.gather(*(h.submit(str(i)) for i in range(32)))
        self.assertEqual(sum(r['receipt_status']=='accepted' for r in receipts),1)
        self.assertEqual(len(h.calls),1)
        self.assertTrue(h.arbiter.ordinary_blocked)
        await h.complete()

    async def test_idempotent_duplicate_and_changed_payload(self):
        h=await Harness().ready()
        receipts=await asyncio.gather(*(h.submit() for _ in range(16)))
        self.assertEqual(len({r['migration_id'] for r in receipts}),1)
        self.assertEqual(len(h.calls),1)
        changed=await h.submit(dst_id='2')
        self.assertEqual(changed['receipt_status'],'conflict')
        await h.complete()
        repeated=await h.submit();self.assertEqual(repeated['terminal_status'],'succeeded')
        self.assertEqual(len(h.calls),1)

    async def test_candidate_capacity_stale_version_and_index_rejections(self):
        for case in ('candidate','capacity','stale','version','index'):
            with self.subTest(case=case):
                h=await Harness().ready()
                if case=='candidate':h.states[0]['candidate_id']='different'
                if case=='capacity':h.states[1]['free_blocks']=69 # 20 + reserve50 leaves one block short
                if case=='stale':h.clock.advance(3)
                if case=='version':await h.arbiter.snapshot()
                if case=='index':h.states[0]['epoch']='rebuilt'
                result=await h.submit()
                self.assertEqual(result['receipt_status'],'rejected')
                self.assertFalse(h.calls)

    async def test_missing_state_does_not_produce_execution(self):
        h=await Harness().ready();h.states[1]['healthy']=None
        result=await h.submit();self.assertEqual(result['receipt_status'],'rejected')
        self.assertFalse(h.calls)

    async def test_shadow_and_stay_never_dispatch(self):
        for mode in ('shadow','stay'):
            h=await Harness(mode).ready();receipt=await h.submit()
            self.assertEqual(receipt['receipt_status'],mode)
            self.assertIsNone(receipt['migration_id']);self.assertFalse(h.calls)

    async def test_cooldown_begins_after_terminal_not_start(self):
        h=await Harness().ready();await h.submit();h.clock.advance(10)
        self.assertIsNotNone(h.arbiter.slot)
        await h.complete()
        h.states=states(h.clock);h.snapshot=await h.arbiter.snapshot()
        receipt=await h.submit('too_early');self.assertEqual(receipt['rejection_reason'],'cooldown')
        h.clock.advance(5);h.states=states(h.clock);h.snapshot=await h.arbiter.snapshot()
        receipt=await h.submit('after_cooldown');self.assertEqual(receipt['receipt_status'],'accepted')
        await h.complete()

    async def test_timeout_retains_slot_reconciles_by_same_decision(self):
        h=await Harness().ready()
        async def uncertain(receipt):h.calls.append(receipt);raise asyncio.TimeoutError()
        h.arbiter.dispatch=uncertain
        await h.submit();await asyncio.sleep(0);await asyncio.sleep(0)
        self.assertEqual((await h.arbiter.query_receipt(h.lease['lease_id'],'d'))['terminal_status'],'unknown')
        self.assertEqual(h.arbiter.slot,'d')
        self.assertEqual((await h.submit('new'))['rejection_reason'],'global_slot_busy')
        self.assertEqual((await h.submit())['terminal_status'],'unknown')
        self.assertEqual(len(h.calls),1)
        async def reconcile(receipt):return {'terminal_status':'succeeded','migration_id':receipt['migration_id'],'actual_request_id':'request-0'}
        h.arbiter.reconcile=reconcile
        self.assertEqual((await h.arbiter.query_receipt(h.lease['lease_id'],'d'))['terminal_status'],'succeeded')
        self.assertIsNone(h.arbiter.slot)

    async def test_exit_expiry_and_native_intervention_do_not_auto_unlock(self):
        h=await Harness().ready();await h.submit()
        self.assertEqual((await h.arbiter.release_controller(h.lease['lease_id']))['status'],'draining')
        h.clock.advance(100)
        with self.assertRaises(RuntimeError):await h.arbiter.acquire_controller('other','v','ep')
        self.assertTrue(h.arbiter.ordinary_blocked)
        await h.complete()
        self.assertEqual((await h.arbiter.release_controller(h.lease['lease_id']))['status'],'released')
        self.assertFalse(h.arbiter.ordinary_blocked)
        h=await Harness().ready();h.arbiter.external_intervention('PRE_STOP_MIGRATION')
        self.assertEqual((await h.submit())['rejection_reason'],'episode_invalid')

    async def test_wrong_executed_candidate_is_unknown(self):
        h=await Harness().ready()
        async def wrong(receipt):return {'terminal_status':'succeeded','actual_request_id':'another'}
        h.arbiter.dispatch=wrong;await h.submit();await asyncio.sleep(0);await asyncio.sleep(0)
        result=await h.arbiter.query_receipt(h.lease['lease_id'],'d')
        self.assertEqual(result['terminal_status'],'unknown');self.assertEqual(h.arbiter.slot,'d')

    async def test_transport_requeries_uncertain_id_instead_of_resubmit(self):
        bridge=RealBridge('http://127.0.0.1:1',None);bridge.lease={'lease_id':'lease'};calls=[]
        async def rpc(method,**payload):
            calls.append(method)
            if method=='submit_migration':raise asyncio.TimeoutError()
            return {'receipt_status':'accepted','terminal_status':None}
        bridge.call=rpc
        first=await bridge.submit_migration('d','a','b','request','v')
        self.assertEqual(first['receipt_status'],'unknown')
        await bridge.submit_migration('d','a','b','request','v')
        self.assertEqual(calls,['submit_migration','query_receipt'])
        with self.assertRaises(ValueError):RealBridge('http://example.com',None)


class RuntimeAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_revalidates_after_destination_RPC(self):
        module=load_file('_real_control_adapter','llumnix/real_control.py');clock=Clock()
        src,dst=states(clock)[:2];request=SimpleNamespace(request_id='request-0',is_migrating=False,stage_num_blocks_list=[20])
        candidates=[request];calls=[]
        scheduler=SimpleNamespace(request_migration_policy='SR',_get_shortest_running_request=lambda *args:candidates)
        async def migrate(*args,**kwargs):calls.append(args);return ['request-0']
        coordinator=SimpleNamespace(enable_engine_migration_interface=False,migration_scheduler=scheduler,_migrate_out_one_request=migrate)
        owner=SimpleNamespace(migration_coordinator=coordinator)
        async def getstate():return deepcopy(dst)
        async def reserve(*args):candidates[:]=[SimpleNamespace(request_id='new_SR')];src['candidate_id']='new_SR';return {'status':'prepared'}
        async def unreserve(*args):return {'status':'released'}
        actor=SimpleNamespace(real_control_state=SimpleNamespace(remote=getstate),real_control_reserve=SimpleNamespace(remote=reserve),real_control_unreserve=SimpleNamespace(remote=unreserve))
        with patch.object(module,'instance_state',lambda owner:deepcopy(src)),patch.object(module.time,'monotonic_ns',clock):
            result=await module.controlled_migrate(owner,actor,SimpleNamespace(instance_id='1'),None,
                expected_candidate_id='request-0',decision_id='d',migration_id='m',source_epoch='e0',destination_epoch='e1')
        self.assertEqual(result['terminal_status'],'rejected');self.assertFalse(calls)
        self.assertIsNone(owner._real_control_inflight)

    async def test_actual_controlled_migration_path_passes_exact_candidate_and_id(self):
        module=load_file('_real_control_success','llumnix/real_control.py');clock=Clock();src,dst=states(clock)[:2]
        request=SimpleNamespace(request_id='request-0',is_migrating=False,stage_num_blocks_list=[7,13]);calls=[]
        async def migrate(*args,**kwargs):calls.append((args,kwargs));return ['request-0']
        scheduler=SimpleNamespace(request_migration_policy='SR',_get_shortest_running_request=lambda *args:[request])
        coordinator=SimpleNamespace(enable_engine_migration_interface=False,migration_scheduler=scheduler,_migrate_out_one_request=migrate)
        owner=SimpleNamespace(migration_coordinator=coordinator)
        async def getstate():return deepcopy(dst)
        async def reserve(*args):return {'status':'prepared'}
        async def unreserve(*args):return {'status':'released'}
        actor=SimpleNamespace(real_control_state=SimpleNamespace(remote=getstate),real_control_reserve=SimpleNamespace(remote=reserve),real_control_unreserve=SimpleNamespace(remote=unreserve))
        with patch.object(module,'instance_state',lambda owner:deepcopy(src)),patch.object(module.time,'monotonic_ns',clock):
            kwargs=dict(expected_candidate_id='request-0',decision_id='d',migration_id='m',source_epoch='e0',destination_epoch='e1')
            result=await module.controlled_migrate(owner,actor,SimpleNamespace(instance_id='1'),None,**kwargs)
            repeated=await module.controlled_migrate(owner,actor,SimpleNamespace(instance_id='1'),None,**kwargs)
        self.assertEqual(result['terminal_status'],'succeeded');self.assertEqual(result['actual_request_id'],'request-0')
        self.assertEqual(result['copied_blocks'],20);self.assertEqual(repeated,result)
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][1]['migration_id'],'m')

    async def test_migration_scheduler_suppresses_only_ordinary_paths(self):
        native=SimpleNamespace(PRE_STOP_MIGRATION='prestop',NEUTRAL_LOAD_BALANCE='ordinary')
        method=extract_method('llumnix/global_scheduler/migration_scheduler.py','MigrationScheduler','push_migrations',{'MigrationType':native})
        obj=SimpleNamespace(enable_pre_stop_migration=True,enable_pd=False,
                            unit_failover_pipeline='failover_filter',unit_failover_policy='failover_policy')
        obj._pair_migration=lambda *args:[('broken','healthy')]
        self.assertEqual(method(obj,{},allow_routine=False),[('prestop',[('broken','healthy')])])
        obj._pair_migration=lambda *args:[]
        self.assertEqual(method(obj,{},allow_routine=False),[])

    async def test_destination_capacity_guard_rechecks_actual_allocator(self):
        response=lambda **kw:SimpleNamespace(**kw)
        method=extract_method('llumnix/llumlet/migration_coordinator.py','MigrationCoordinator','_pre_alloc_cache',
                              {'MigrationResponse':response})
        info=SimpleNamespace(num_total_gpu_blocks=1000,num_free_gpu_blocks=60,num_blocks_all_waiting_requests=0)
        allocation=[];freed=[]
        backend=SimpleNamespace(engine=SimpleNamespace(scheduler=[SimpleNamespace(_get_instance_info=lambda items:info)]),
                                pre_alloc_cache=lambda *args:allocation.append(args) or response(success=True))
        obj=SimpleNamespace(_real_capacity_guards={'r'},backend_engine=backend,free_pre_alloc_cache=lambda rid:freed.append(rid))
        result=method(obj,'r',None,0.,20,[],True)
        self.assertFalse(result.success);self.assertEqual(allocation,[]);self.assertEqual(freed,['r'])
        info.num_free_gpu_blocks=70
        self.assertTrue(method(obj,'r',None,0.,20,[],True).success)
        self.assertEqual(len(allocation),1)

    async def test_pending_delivery_counts_queued_and_inflight_batches(self):
        method=extract_method('llumnix/backends/output_forwarder.py','ThreadOutputForwarder','put_nowait_to_servers',
                              {'asyncio':asyncio,'_record_output_forwarder_stage':lambda *args:None})
        from concurrent.futures import Future
        future=Future();obj=SimpleNamespace(_real_pending_lock=threading.Lock(),_real_pending=0,_real_delivery_error=False,
            add_trace_timeline=lambda *args,**kwargs:None,forwarder_loop=None,_put_nowait_to_servers=lambda *args:'placeholder')
        with patch.object(asyncio,'run_coroutine_threadsafe',return_value=future):method(obj,{}, {})
        self.assertEqual(obj._real_pending,1)
        future.set_result(None);self.assertEqual(obj._real_pending,0)
        future=Future()
        with patch.object(asyncio,'run_coroutine_threadsafe',return_value=future):method(obj,{}, {})
        future.set_exception(RuntimeError('delivery failed'))
        self.assertTrue(obj._real_delivery_error)

    async def test_loopback_only_control_endpoint_and_forwarding(self):
        from fastapi import FastAPI
        import httpx
        module=load_file('_real_control_http','llumnix/entrypoints/vllm/real_control_api.py')
        calls=[]
        async def rpc(method,payload):calls.append((method,payload));return {'ok':True}
        client=SimpleNamespace(manager=SimpleNamespace(real_control=SimpleNamespace(remote=rpc)))
        app=FastAPI();module.install_real_control_routes(app,lambda:client,{},lambda timeout:{})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,client=('8.8.8.8',123)),base_url='http://test') as http:
            self.assertEqual((await http.post('/real_control/snapshot',json={})).status_code,403)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,client=('127.0.0.1',123)),base_url='http://test') as http:
            self.assertEqual((await http.post('/real_control/snapshot',json={})).json(),{'ok':True})
        self.assertEqual(calls,[('snapshot',{})])


if __name__=='__main__':unittest.main()
