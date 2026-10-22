"""Finite episode, measured observation, persistent budget and owned cleanup tests."""
import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rl_scheduler.real_bridge import digest
from rl_scheduler.real_budget import BudgetLedger,BudgetExhausted,atomic_json
from rl_scheduler.real_env import RealEnv,RealEnvConfig,assert_idle
from rl_scheduler.real_metrics import RequestLedger,Checkpoint
from rl_scheduler.real_observation import RealObservationBuilder,RealNormalizer,SCHEMA_VERSION
from rl_scheduler.real_service import OwnedProcesses,ManagedService,process_identity,identity_alive
from rl_scheduler.tests.test_real_bridge import Clock,states,ROOT


def budget_file(directory):
    atomic_json(Path(directory)/'budget.json',{'run_id':'run','usage':{'task_experiment_wall_time_sec':0,'ppo_training_wall_time_sec':0},
        'total_experiment_wall_time_sec':28800,'phase_caps_sec':{'P0_P3_including_GPU_checks_and_opportunity':7200,'P4_PPO_training':7200,'P5_final_evaluation_reserved':14400}})


def snapshot(clock):
    values=states(clock)
    for s in values:s.update(running=0,waiting=0)
    return {'version':'v1','measured_ns':clock(),'received_ns':clock(),'mapping':[[s['instance_id'],s['epoch']] for s in values],
            'instances':values,'busy':False,'last_terminal_ns':None,'lease_expired':False,'invalid_reasons':[]}


class FakeBridge:
    def __init__(self,clock):self.clock=clock;self.value=snapshot(clock);self.released=False;self.actions=[]
    async def snapshot(self):
        value=deepcopy(self.value);value['received_ns']=self.clock();value['measured_ns']=self.clock()
        for s in value['instances']:s['measured_ns']=self.clock()
        return value
    async def acquire_controller(self,*args):return {'lease_id':'lease','mapping':self.value['mapping']}
    async def renew_controller(self):return {}
    async def submit_migration(self,*args):
        self.actions.append(args);self.value['instances'][0]['running']=3
        return {'receipt_status':'rejected','terminal_status':'rejected','rejection_reason':'candidate_changed','migration_id':None}
    async def query_receipt(self,*args):return {'receipt_status':'accepted','terminal_status':'unknown'}
    async def release_controller(self):self.released=True;return {'status':'released'}


class FakeIO:
    def __init__(self,clock,bridge):self.clock,self.bridge=clock,bridge;self.stopped=0;self.cleaned=0;self.hang=False
    async def prepare(self):pass
    async def begin(self,*args):pass
    async def poll(self,ledger):pass
    async def stop_arrivals(self):self.stopped+=1
    async def sleep(self,seconds):self.clock.advance(seconds)
    async def cleanup(self,deadline,ledger):
        self.cleaned+=1
        if self.hang:await asyncio.Future()
        for s in self.bridge.value['instances']:s['running']=0
    async def flush(self,ledger):ledger.acknowledge(Checkpoint('s',0,self.clock(),self.clock(),0,True))


def environment(*,cleanup_sec=60.):
    clock=Clock();bridge=FakeBridge(clock);io=FakeIO(clock,bridge)
    factory=lambda start,arrival,deadline:RequestLedger('run','ep',start_ns=start,arrival_end_ns=arrival,
        score_deadline_ns=deadline,expected_sources=['s'],expected_request_ids=[])
    budget=type('Budget',(),{'require':lambda self,required:None})()
    env=RealEnv(bridge,io,budget,run_id='run',episode_id='ep',ledger_factory=factory,
                config=RealEnvConfig(arrival_sec=2.,drain_sec=2.,cleanup_sec=cleanup_sec),clock=clock)
    return env,clock,bridge,io


class EnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_KV_baseline_can_be_nonzero_but_cannot_drift(self):
        clock=Clock();s=snapshot(clock)
        baseline=assert_idle(s);self.assertEqual(set(baseline.values()),{100})
        s['instances'][0]['used_blocks']=101
        with self.assertRaises(RuntimeError):assert_idle(s,baseline)
        s['instances'][0]['used_blocks']=100;s['instances'][0]['pending_delivery']=1
        with self.assertRaises(RuntimeError):assert_idle(s,baseline)

    async def test_finite_phases_dt_and_completed_cleanup(self):
        env,clock,bridge,io=environment();obs,_=await env.reset()
        for _ in range(4):_,_,done,_,_=await env.step(0,old_logp=0.,old_value=1.)
        self.assertTrue(done);self.assertEqual(env.state,'deadline')
        self.assertEqual([d['dt_sec'] for d in env.decisions],[0.,1.,1.,1.])
        result=await env.finish()
        self.assertEqual(result['status'],'passed');self.assertTrue(result['terminal_value_zero_allowed'])
        self.assertTrue(bridge.released);self.assertEqual(io.cleaned,1)
        self.assertEqual(result['boundary']['score_deadline_ns'],104_000_000_000)

    async def test_saved_observation_and_rejected_sample_are_not_rewritten(self):
        env,clock,bridge,io=environment();observation,_=await env.reset();original=deepcopy(observation)
        await env.step(1,policy_version='p',old_logp=-.7,old_value=.2)
        row=env.decisions[0]
        self.assertEqual(row['raw_obs'],original['raw'])
        self.assertEqual(row['normalized_obs'],original['normalized'])
        self.assertEqual(row['sampled_action'],1);self.assertEqual(row['old_logp'],-.7)
        self.assertEqual(row['rejection_reason'],'candidate_changed')
        self.assertEqual(row['raw_obs']['instance_features'][0][0],0.)
        self.assertEqual(env.current_observation['raw']['instance_features'][0][0],3.)
        await env.close()

    async def test_delayed_callback_does_not_create_catchup_transitions(self):
        env,clock,bridge,io=environment();await env.reset()
        await env.step(0);clock.advance(1.5);await env.step(0)
        self.assertEqual(len(env.decisions),2);self.assertEqual(env.decisions[1]['dt_sec'],2.5)
        await env.close()

    async def test_cleanup_timeout_and_controller_exit_are_invalid(self):
        env,clock,bridge,io=environment(cleanup_sec=.01);await env.reset();clock.advance(4);io.hang=True
        result=await env.finish()
        self.assertEqual(result['status'],'incomplete');self.assertFalse(result['finite_terminal'])
        self.assertTrue(any('TimeoutError' in issue for issue in result['invalid_reasons']))
        self.assertTrue(bridge.released)
        env,clock,bridge,io=environment();await env.reset();result=await env.close()
        self.assertFalse(result['valid_episode']);self.assertIn('controller_exit',result['invalid_reasons'])

    async def test_index_invalidation_and_disconnection_abort_episode(self):
        env,clock,bridge,io=environment();await env.reset();bridge.value['mapping'][0][1]='rebuilt'
        with self.assertRaises(RuntimeError):await env.step(0)
        result=await env.close();self.assertEqual(result['status'],'incomplete')
        env,clock,bridge,io=environment();await env.reset()
        async def disconnected():raise ConnectionError('lost')
        bridge.renew_controller=disconnected
        with self.assertRaises(ConnectionError):await env.step(0)
        result=await env.close();self.assertFalse(result['valid_episode'])


class ObservationTests(unittest.TestCase):
    def test_versioned_named_normalizer_frozen_and_missing_mask(self):
        clock=Clock();s=snapshot(clock);s['instances'][0]['candidate_blocks']=None;s['instances'][0]['last_step_ms']=None
        ledger=RequestLedger('run','ep',start_ns=clock(),arrival_end_ns=clock()+2_000_000_000,
                             score_deadline_ns=clock()+4_000_000_000,expected_sources=['s'],expected_request_ids=[])
        builder=RealObservationBuilder(s['mapping']);raw=builder.build(s,now_ns=clock(),ledger=ledger,
            episode_start_ns=clock(),arrival_end_ns=clock()+2_000_000_000,deadline_ns=clock()+4_000_000_000)
        normalizer=RealNormalizer();value=normalizer.transform(raw)
        self.assertTrue(raw['instance_missing'][0][6]);self.assertTrue(raw['instance_missing'][0][7])
        self.assertFalse(any(raw['action_mask'][1:4]))
        self.assertEqual(RealNormalizer.from_state_dict(normalizer.state_dict()).transform(raw),value)
        with self.assertRaises(TypeError):normalizer.scales['running']=4
        with self.assertRaises(AttributeError):normalizer.scales={'running':4}
        raw['instance_feature_names'].reverse()
        with self.assertRaises(ValueError):normalizer.transform(raw)
        self.assertFalse(any('predicted' in name or 'future' in name or 'migration_ms' in name for name in normalizer.scales))

    def test_history_rates_and_state_age_use_received_information(self):
        clock=Clock();s=snapshot(clock);builder=RealObservationBuilder(s['mapping'])
        ledger=RequestLedger('run','ep',start_ns=clock(),arrival_end_ns=clock()+10_000_000_000,
                             score_deadline_ns=clock()+20_000_000_000,expected_sources=['s'],expected_request_ids=[])
        def build(value):return builder.build(value,now_ns=clock(),ledger=ledger,episode_start_ns=100_000_000_000,
                                               arrival_end_ns=110_000_000_000,deadline_ns=120_000_000_000)
        first=build(s);self.assertTrue(first['instance_missing'][0][9])
        clock.advance(5);s=snapshot(clock);s['instances'][0]['output_tokens_total']=50;s['instances'][0]['waiting']=10
        raw=build(s);self.assertEqual(raw['instance_features'][0][9],10.);self.assertEqual(raw['instance_features'][0][10],2.)
        s['received_ns']=clock()+1
        with self.assertRaises(ValueError):build(s)


class BudgetAndServiceTests(unittest.TestCase):
    def test_persistent_budget_phase_caps_restart_and_wall_jump(self):
        with tempfile.TemporaryDirectory() as directory:
            budget_file(directory);clock=Clock();wall=[1000.]
            ledger=BudgetLedger(directory,wall=lambda:wall[0],mono=lambda:clock()/1e9,boot_id='boot')
            self.assertFalse(ledger.status()['started']);ledger.activate()
            clock.advance(100);wall[0]+=100
            second=BudgetLedger(directory,wall=lambda:wall[0],mono=lambda:clock()/1e9,boot_id='boot')
            self.assertEqual(second.status()['elapsed_sec'],100.)
            wall[0]-=50;clock.advance(20)
            self.assertEqual(second.status()['elapsed_sec'],120.)
            second.activate('P4');clock.advance(7200);wall[0]+=7200
            with self.assertRaises(BudgetExhausted):second.require(60,'P4')
            with self.assertRaises(BudgetExhausted):second.activate('P4')

    def test_budget_cross_process_origin_is_not_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            budget_file(directory);ledger=BudgetLedger(directory);ledger.activate();before=json.loads(ledger.path.read_text())
            code='from rl_scheduler.real_budget import BudgetLedger; import sys; b=BudgetLedger(sys.argv[1]); b.activate(); print(b.status()["started"])'
            result=subprocess.run([sys.executable,'-c',code,directory],cwd=ROOT,capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.strip(),'True')
            after=json.loads(ledger.path.read_text())
            self.assertEqual(after['first_started_wall'],before['first_started_wall']);self.assertGreaterEqual(after['elapsed_sec'],before['elapsed_sec'])

    def test_owned_cleanup_preserves_unregistered_process(self):
        with tempfile.TemporaryDirectory() as directory:
            owned=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
            foreign=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
            try:
                ledger=OwnedProcesses(Path(directory)/'owned.json');ledger.register(owned.pid,'test_owned')
                ledger.stop(timeout=.5)
                owned.wait(timeout=2)
                self.assertIsNone(foreign.poll())
            finally:
                for child in (owned,foreign):
                    if child.poll() is None:child.terminate()
                    child.wait(timeout=2)

    def test_supervisor_preparation_never_starts_service_and_uses_private_managed_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            budget_file(directory);calls=[]
            service=ManagedService(ROOT,directory,ROOT/'rl_scheduler/configs/real_pilot.yaml',command_runner=lambda *a,**k:calls.append(a))
            paths=service.prepare();self.assertEqual(len(paths),4);self.assertFalse(calls)
            self.assertIn('autostart=false',Path(paths[1]).read_text())
            self.assertIn('stdout_logfile=/dev/stdout',Path(paths[1]).read_text())
            with self.assertRaises(RuntimeError):service._supervisor('stop','caddy')

    def test_unimplemented_CLI_stage_is_rejected_even_dry_run(self):
        result=subprocess.run([sys.executable,'benchmark/run_real_ppo.py','--config','rl_scheduler/configs/real_pilot.yaml',
                               '--stage','train','--output-dir','/tmp/unused-real-ppo','--dry-run'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('stage not implemented',result.stderr)


if __name__=='__main__':unittest.main()
