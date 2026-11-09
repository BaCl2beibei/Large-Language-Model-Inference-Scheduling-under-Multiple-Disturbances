"""Portable CPU contracts; run with python -m pytest -q tests in the ZIP.

Tiny physical fixtures exercise actual simulation/updates, not service mocks.
"""
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from benchmark import ppo_local as local
from rl_scheduler.cohort_ppo import FrozenController, build_rollout, normalizer_definition
from rl_scheduler.cohort_reward import MigPPOV2Reward
from rl_scheduler.q15b_sim import Q15BCohortSim
from rl_scheduler.reward import MigPPORewardMetrics


def model():
    return dict(schema='mig-ppo-v2-shared-physical-v2',capacity=1000,
        migration_intercept_ms=100.,migration_ms_per_block=0.,
        decision_interval_sec=1.5,migration_cooldown_sec=0.,
        hardware_by_logical_instance=['cpu']*4,client_frame_tokens=64,prefill_ms=10.,
        service=dict(method='linear',coefficients=[10.,0.,0.],hardware_factors={},min_ms=1.))


def trace():
    return [dict(request_id='r',scheduled_arrival_sec=0.,input_tokens=128,
        output_tokens=128,initial_instance_index=0,target_qps=.6)]


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    root=tmp_path/'CPU bundle 中文';root.mkdir()
    envs=[]
    for eid,n,qps,f,_ in local.ENV_SPECS:
        relative='calibration/'+eid+'.json'
        local.atomic_json(root/relative,model())
        envs.append(dict(env_id=eid,request_count=n,qps=qps,hot_fraction=f,
            hot_index=int(eid[3])-1 if eid!='NATIVE' else None,probability=1/13,model_path=relative))
    local.atomic_json(root/'configs/environments.json',dict(environments=envs))
    local.atomic_json(root/'calibration/reward_definition.json',MigPPOV2Reward(hotspot_threshold=.75).state_dict())
    cfg=dict(scope='local_simulation_development',settings=local.SETTINGS,max_updates=2,
        episodes_per_update=2,wall_budget_sec=120.,episode_wall_limit_sec=30.,torch_threads=1,
        evaluate_every_updates=1,training_trace_seed_base=80_000_000,environment_seed=1234,
        policy_seed=1235,update_seed=1236)
    local.atomic_json(root/'configs/local_train.json',cfg)
    manifest=dict(schema=local.BUNDLE_SCHEMA,contract_sha256='tiny-fixture')
    local.atomic_json(root/'manifest.json',manifest)
    spec=local.evaluation_spec(envs)
    # One workload/action seed per environment keeps this real integration test short.
    for split,cases in spec['splits'].items():spec['splits'][split]=[dict(c,action_seeds=c['action_seeds'][:1]) for c in cases[::2]]
    local.atomic_json(root/'traces/evaluation_spec.json',spec)
    torch.set_num_threads(1)
    policy=local.new_policy(cfg['policy_seed']);opt=torch.optim.Adam(policy.parameters(),lr=.0003)
    state=dict(next_update=0,next_episode=0,pending=[],inflight=None,elapsed_sec=0.,history=[],evaluated=[],best=None)
    local.checkpoint(root/'checkpoints/initial_v2.pt',policy,opt,manifest,state,random.Random(cfg['environment_seed']),cfg)
    local.write_checksums(root)
    original=local.numeric_trace
    def tiny_trace(env,seed):
        rows=original(env,seed)[:4]
        for r in rows:r['output_tokens']=128
        return rows
    monkeypatch.setattr(local,'numeric_trace',tiny_trace)
    return root


def test_logging_mode_preserves_physical_outputs(tmp_path):
    outputs=[]
    for minimal in (False,True):
        sim=Q15BCohortSim(model(),trace(),1,tmp_path/str(minimal),minimal_logging=minimal)
        try:
            s,d,m=sim.run('Stay',1.,reward_definition=MigPPOV2Reward(hotspot_threshold=.75))
            outputs.append((s['requests'],d,m,local.read(sim.output_dir/'reward.json')))
        finally:sim.event_file.close()
    assert outputs[0]==outputs[1]


def test_deadline_retains_partial_tokens_and_terminal_gae(tmp_path):
    p=local.new_policy(1)
    sim=Q15BCohortSim(model(),trace(),1,tmp_path/'partial',minimal_logging=True,finalize_deadline=True)
    definition=MigPPOV2Reward(hotspot_threshold=.75)
    try:
        s,d,_=sim.run('PPO',1.,FrozenController(p,normalizer_definition(),0),
            reward_definition=definition,episode_horizon_sec=.2)
        r=s['requests'][0]
        assert r['outcome']=='timeout' and 1<r['actual_output_tokens']<128
        assert sum(n for _,n in r['token_increments'])==r['actual_output_tokens']
        assert r['client_terminal_ns']==200_000_000
        audit=local.read(sim.output_dir/'reward.json')
        assert sum(w['failure_count'] for w in audit['windows'])==1
        assert audit['terminal_potential']==0
        rollout=build_rollout(s,d,definition)
        assert rollout['records'][-1]['terminated'] and not rollout['records'][-1]['truncated']
        assert len(rollout['records'])==len(d)
        assert not sim.requests and all(i.used_blocks==i.reserved_blocks==0 for i in sim.instances)
    finally:sim.event_file.close()


@pytest.mark.parametrize('replay',[False,True])
def test_deadline_aborts_migration_and_releases_reservation(tmp_path,replay):
    sim=Q15BCohortSim(model(),trace(),1,tmp_path/'moving',minimal_logging=True)
    try:
        sim._advance(MigPPORewardMetrics(window_sec=.2,instance_count=4),0.,.2)
        if replay:
            assert sim.start_replay_migration(dict(request_id='r',source_instance='instance_0',
                target_instance='instance_1',cost_ms=10000.,pause_stage_ms=10000.,
                blocks=sim.requests['r'].allocated_blocks))=='started'
        else:
            sim.episode_horizon_sec=200.
            raw,_=sim.raw()
            ob=SimpleNamespace(action_mask=SimpleNamespace(pairs=[(sim.mapping[a][0],sim.mapping[b][0])
                for a,b in local.PAIRS]),pair_mask=raw['action_mask'][1:])
            assert sim._start_migration(1,ob,MigPPORewardMetrics(window_sec=1.,instance_count=4)) is None
        assert sim.instances[1].reserved_blocks>0
        sim._finalize_at_deadline()
        assert sim.pending_migration is None
        assert sim.migration_events[-1]['status']=='aborted'
        assert sim.migration_events[-1]['fixed_trace_replay']==replay
        assert sim.request_results[-1]['status']=='timeout'
        assert all(i.used_blocks==i.reserved_blocks==0 for i in sim.instances)
    finally:sim.event_file.close()


def test_watchdog_is_an_error_not_a_fake_terminal(tmp_path):
    sim=Q15BCohortSim(model(),trace(),1,tmp_path/'watchdog',minimal_logging=True,
        finalize_deadline=True,wall_time_limit_sec=0.)
    try:
        with pytest.raises(TimeoutError):sim.run('Stay',1.,reward_definition=MigPPOV2Reward(hotspot_threshold=.75))
        assert not (sim.output_dir/'episode.json').exists()
    finally:sim.event_file.close()


def test_resume_matches_continuous_training_and_preserves_budget(bundle,tmp_path):
    whole=tmp_path/'whole';resumed=tmp_path/'resumed'
    local.train(bundle,whole,max_updates=2,wall_budget_sec=100.,skip_evaluation=True)
    local.train(bundle,resumed,max_updates=1,wall_budget_sec=100.,skip_evaluation=True)
    local.train(bundle,resumed,resume='latest',max_updates=2,skip_evaluation=True)
    manifest=local.read(bundle/'manifest.json')
    a,_,ca=local.load_checkpoint(whole/'checkpoints/latest.pt',manifest)
    b,_,cb=local.load_checkpoint(resumed/'checkpoints/latest.pt',manifest)
    assert local.policy_hash(a)==local.policy_hash(b)
    assert ca['state']['history']==cb['state']['history']
    assert cb['config']['wall_budget_sec']==100.
    assert cb['state']['next_episode']==4


def test_interrupted_episode_replays_same_rng_and_environment(bundle,tmp_path,monkeypatch):
    whole=tmp_path/'whole';resume=tmp_path/'retry'
    local.train(bundle,whole,max_updates=1,skip_evaluation=True)
    original=local.run_episode
    calls=0
    def interrupt(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1:return original(*args,**kwargs)
        torch.rand(30)  # Model sampling would have advanced this generator.
        raise KeyboardInterrupt()
    monkeypatch.setattr(local,'run_episode',interrupt)
    result=local.train(bundle,resume,max_updates=1,skip_evaluation=True)
    assert result['status']=='interrupted' and result['inflight_episode']
    assert result['pending_batch_episodes']==1
    monkeypatch.setattr(local,'run_episode',original)
    local.train(bundle,resume,resume='latest',skip_evaluation=True)
    manifest=local.read(bundle/'manifest.json')
    a,_,ca=local.load_checkpoint(whole/'checkpoints/latest.pt',manifest)
    b,_,cb=local.load_checkpoint(resume/'checkpoints/latest.pt',manifest)
    assert local.policy_hash(a)==local.policy_hash(b)
    assert ca['state']['history']==cb['state']['history']


def test_half_update_is_not_persisted(bundle,tmp_path,monkeypatch):
    whole=tmp_path/'whole';resume=tmp_path/'partial_update'
    local.train(bundle,whole,max_updates=1,skip_evaluation=True)
    original=local.ppo_update
    def interrupt(policy,*args,**kwargs):
        with torch.no_grad():next(policy.parameters()).add_(100.)
        raise KeyboardInterrupt()
    monkeypatch.setattr(local,'ppo_update',interrupt)
    result=local.train(bundle,resume,max_updates=1,skip_evaluation=True)
    assert result['status']=='interrupted' and result['pending_batch_episodes']==2
    monkeypatch.setattr(local,'ppo_update',original)
    local.train(bundle,resume,resume='latest',skip_evaluation=True)
    manifest=local.read(bundle/'manifest.json')
    a,_,_=local.load_checkpoint(whole/'checkpoints/latest.pt',manifest)
    b,_,_=local.load_checkpoint(resume/'checkpoints/latest.pt',manifest)
    assert local.policy_hash(a)==local.policy_hash(b)


def test_heldout_is_frozen_and_development_cache_reusable(bundle,tmp_path):
    run=tmp_path/'evaluated'
    local.train(bundle,run,max_updates=1)
    first=local.evaluate(bundle,run,'development')
    # A kill after metrics but before the cache receipt must be recoverable.
    receipt=next((run/'evaluation/baselines/development').rglob('evaluation_input.json'))
    receipt.unlink()
    second=local.evaluate(bundle,run,'development')
    assert first==second
    assert list((run/'evaluation/baselines/development').rglob('*.interrupted_1'))
    heldout=local.evaluate(bundle,run,'heldout')
    assert len(heldout['per_environment'])==13
    assert heldout['formal_ppo_ready'] is False
    manifest=local.read(bundle/'manifest.json')
    policy,opt,ck=local.load_checkpoint(run/'checkpoints/latest.pt',manifest)
    with torch.no_grad():next(policy.parameters()).add_(.5)
    changed=run/'different.pt'
    rng=random.Random();local.restore_rng(ck,rng)
    local.checkpoint(changed,policy,opt,manifest,ck['state'],rng,ck['config'])
    with pytest.raises(ValueError,match='already frozen'):local.evaluate(bundle,run,'heldout',changed)


def test_integrity_and_path_escape_are_rejected(bundle):
    with pytest.raises(ValueError):local.inside(bundle,'../outside.json')
    with pytest.raises(ValueError):local.inside(bundle,'C:\\outside.json')
    file=bundle/'calibration/HOT1-L.json'
    file.write_text(file.read_text(encoding='utf-8')+' ',encoding='utf-8')
    with pytest.raises(ValueError,match='integrity'):local.load_bundle(bundle)


def test_split_grouping_ignores_labels_and_routes():
    original=trace();changed=deepcopy(original)
    changed[0].update(request_id='renamed',initial_instance_index=3)
    assert local.trace_group(original)==local.trace_group(changed)
    changed[0]['scheduled_arrival_sec']=.1
    assert local.trace_group(original)!=local.trace_group(changed)
