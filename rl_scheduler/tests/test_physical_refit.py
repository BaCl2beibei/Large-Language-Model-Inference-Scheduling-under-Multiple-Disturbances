"""Physical regression checks: client batching, progress, replay and grouping."""
from pathlib import Path

import pytest

from rl_scheduler.q15b_sim import Q15BCohortSim
from rl_scheduler.reward import MigPPORewardMetrics
from rl_scheduler.service_model import predict_physical_cadence


def physical_model():
    return dict(schema='mig-ppo-v2-shared-physical-v2',capacity=1000,
        migration_intercept_ms=100.,migration_ms_per_block=0.,
        decision_interval_sec=1.5,migration_cooldown_sec=0.,
        hardware_by_logical_instance=['gpu']*4,client_frame_tokens=64,prefill_ms=10.,
        service=dict(method='linear',coefficients=[10.,0.,0.],hardware_factors={},min_ms=1.))


def trace():
    return [dict(request_id='r',scheduled_arrival_sec=0.,input_tokens=128,output_tokens=128,
                 initial_instance_index=0,target_qps=.6)]


def test_client_frame_flush_does_not_freeze_engine_or_kv(tmp_path):
    sim=Q15BCohortSim(physical_model(),trace(),seed=1,output_dir=tmp_path/'sim')
    try:
        metrics=MigPPORewardMetrics(window_sec=.2,instance_count=4)
        sim._advance(metrics,0.,.2)
        request=sim.requests['r']
        assert 10<request.output_tokens<64
        assert sim.instances[0].used_blocks>8
        client=sim.observe(round(sim.now*1e9))['requests'][0]
        assert sum(n for _,n in client['token_increments'])==1
        assert client['client_last_ns']==client['client_first_ns']
        assert client['client_last_ns']<round(request.last_token_time*1e9)
        sim._advance(metrics,.2,2.)
        assert [n for _,n in sim.token_history['r']]==[1,63,64]
        assert sum(n for _,n in sim.token_history['r'])==128
        assert sim.instances[0].used_blocks==0
        assert len(sim.request_results)==1
    finally:
        sim.event_file.close()


def test_service_features_change_with_current_load_only():
    model=dict(method='linear',coefficients=[10.,2.,.5],hardware_factors={'gpu':1.2},min_ms=1.)
    base=dict(n=1.,context=1024.,hardware='gpu')
    assert predict_physical_cadence(model,base)==pytest.approx(15.)
    assert predict_physical_cadence(model,dict(base,n=8.))>predict_physical_cadence(model,base)
    assert predict_physical_cadence(model,dict(base,context=8192.))>predict_physical_cadence(model,base)
    # Arbitrary scenario/policy metadata cannot affect the physical predictor.
    assert predict_physical_cadence(model,dict(base,environment='HOT1-H',policy='Balanced',gain=-.4))==15.


def test_engine_time_and_effective_progress_share_one_clock(tmp_path):
    model=physical_model()
    model['engine']=dict(method='constant',coefficients=[4.],hardware_factors={},min_ms=1.)
    sim=Q15BCohortSim(model,trace(),seed=1,output_dir=tmp_path/'two_clocks')
    try:
        sim._advance(MigPPORewardMetrics(window_sec=.2,instance_count=4),0.,.2)
        assert sim.instances[0].last_step_ms==4.
        steps=[s for s in sim.service_iterations if s['phase']=='batch' and s['batch_type']=='decode']
        assert steps
        assert all(s['engine_execution_ms']==4. and s['output_cadence_ms']==10. and s['step_duration_ms']==10. for s in steps)
        assert 18<=sim.requests['r'].output_tokens<=20
    finally:sim.event_file.close()


def test_explicit_zero_pause_replay_is_preserved(tmp_path):
    sim=Q15BCohortSim(physical_model(),trace(),seed=1,output_dir=tmp_path/'replay')
    try:
        sim._advance(MigPPORewardMetrics(window_sec=.2,instance_count=4),0.,.2)
        result=sim.start_replay_migration(dict(request_id='r',source_instance='instance_0',
            target_instance='instance_1',cost_ms=100.,pause_stage_ms=0.,blocks=sim.requests['r'].allocated_blocks))
        assert result=='started'
        assert sim.pending_migration.pause_stage_ms==0.
        assert sim.pending_migration.pause_start_time==sim.pending_migration.end_time
    finally:sim.event_file.close()


def test_trace_groups_ignore_routes_but_keep_workload_changes():
    from benchmark.mig_ppo_v2_refit import base_hash
    a=dict(trace()[0],prompt='hello')
    assert base_hash([a])==base_hash([dict(a,initial_instance_index=3,candidate_id='other')])
    assert base_hash([a])!=base_hash([dict(a,scheduled_arrival_sec=.1)])
