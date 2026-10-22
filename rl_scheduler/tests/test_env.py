import math
from itertools import product

import numpy as np
import pytest

from rl_scheduler.env import LlumnixMigrationEnv, MigPPOEnvConfig, _Request
from rl_scheduler.reward import MigPPORewardMetrics


def test_env_reset_is_seeded_and_stay_advances_real_request_state():
    config = MigPPOEnvConfig(
        workload_scenario="low", arrival_rates_rps={"low": 2.0, "medium": 4.0, "high": 8.0, "bursty": 4.0},
        max_decisions=20, total_kv_blocks=128, safety_margin_blocks=4,
        prompt_profiles=(32,), output_profiles=(16,), request_timeout_sec=5.0,
    )
    env = LlumnixMigrationEnv(config)
    obs_a, _ = env.reset(seed=123, options={"scenario": "low"})
    obs_b, _ = env.reset(seed=123, options={"scenario": "low"})
    assert obs_a.action_mask.action_mask.tolist() == obs_b.action_mask.action_mask.tolist()
    assert obs_a.instance_features.tolist() == obs_b.instance_features.tolist()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    assert info["algorithm"] == "Mig-PPO"
    assert env.request_count >= 0
    assert len(obs.instance_features) == 4


def test_migration_changes_future_queue_placement_and_logs_terminal_result():
    config = MigPPOEnvConfig(
        workload_scenario="low", arrival_rates_rps={"low": 0.0, "medium": 0.0, "high": 0.0, "bursty": 0.0},
        max_decisions=5, total_kv_blocks=128, safety_margin_blocks=4,
        migration_failure_probability=0.0, migration_alpha_ms=10.0,
        migration_beta_ms_per_block=0.0,
    )
    env = LlumnixMigrationEnv(config)
    env.reset(seed=7, options={"scenario": "low"})
    request = _Request("req-manual", 0.0, 64, 512, output_tokens=4,
                       stage="decode", allocated_blocks=5, last_token_time=0.0)
    env.requests[request.request_id] = request
    src = env.instances[0]
    src.running.append(request)
    src.used_blocks = request.allocated_blocks
    src.next_step_time = math.inf
    obs = env._observation()
    env._last_obs = obs
    action = 1 + obs.action_mask.pairs.index(("instance_0", "instance_1"))
    assert obs.action_mask.action_mask[action]
    _next, _reward, _terminated, _truncated, info = env.step(action)
    assert env.migration_events
    event = env.migration_events[0]
    assert event["status"] == "success"
    assert event["request_id"] == request.request_id
    assert any(request in instance.running for instance in env.instances[1:])
    assert info["reward"]["migration_succeeded"] == 1


def test_no_legal_candidate_forces_stay():
    env = LlumnixMigrationEnv(MigPPOEnvConfig(max_decisions=2))
    obs, _ = env.reset(seed=1, options={"scenario": "low"})
    assert obs.action_mask.action_mask.tolist() == [True] + [False] * 12


def test_training_workload_uses_five_qps_levels_nine_profiles_and_128_sequences():
    config = MigPPOEnvConfig.from_yaml("rl_scheduler/configs/train.yaml")
    assert config.active_sequence_capacity == 128
    assert config.max_model_len == 2112
    assert tuple(config.target_qps_levels) == (9, 12, 15, 18, 21)
    assert tuple(config.prompt_profiles) == (256, 512, 1048)
    assert tuple(config.output_profiles) == (256, 512, 1048)

    env = LlumnixMigrationEnv(config)
    env.rng = np.random.default_rng(42)
    seen_scenarios = {env._choose_scenario() for _ in range(100)}
    assert seen_scenarios == {"qps_9", "qps_12", "qps_15", "qps_18", "qps_21"}

    env.reset(seed=42, options={"target_qps": 3})
    profiles = [env._sample_prompt_response() for _ in range(9)]
    expected = set(product(config.prompt_profiles, config.output_profiles))
    assert len(profiles) == 9
    assert set(profiles) == expected


def test_training_profile_product_must_fit_configured_model_length():
    with pytest.raises(ValueError, match="cannot fit the longest"):
        MigPPOEnvConfig(prompt_profiles=(1048,), output_profiles=(1048,), max_model_len=2048)


def test_arrival_trace_replays_only_arrival_fields_and_request_ids():
    config = MigPPOEnvConfig(decision_interval_sec=0.1, max_decisions=2, total_kv_blocks=128,
                             max_num_batched_tokens=128, request_timeout_sec=None,
                             prefill_ms_per_token=0.1, prefill_intercept_ms=1.0,
                             decode_ms_per_request=1.0, decode_intercept_ms=1.0)
    env = LlumnixMigrationEnv(config)
    trace = [{"request_id": "r00001", "arrival_time": 0.0, "prompt_tokens": 32, "output_tokens": 2}]
    env.reset(seed=3, options={"target_qps": 9, "arrival_trace": trace})
    env.step(0)
    assert env.request_count == 1
    assert "r00001" in env.requests or any(row["request_id"] == "r00001" for row in env.request_results)
    assert all(instance.capacity_scale == 1.0 for instance in env.instances)


def test_traced_output_budget_uses_seeded_calibration_length_distribution():
    model = {"version": 1, "profiles": {
        "32:100": {"lengths": [20, 40], "counts": [1, 3], "n": 4},
    }}
    config = MigPPOEnvConfig(total_kv_blocks=128, output_length_model=model)
    trace = [{"request_id": "r00042", "arrival_time": 0.0, "prompt_tokens": 32,
              "output_tokens": 100, "output_budget_tokens": 100}]
    targets = []
    for _ in range(2):
        env = LlumnixMigrationEnv(config)
        env.reset(seed=19, options={"target_qps": 9, "arrival_trace": trace})
        env._process_arrival()
        request = env.requests["r00042"]
        targets.append(request.target_output_tokens)
        assert request.requested_output_tokens == 100
        assert env._expected_output_tokens(request) == 35
    assert targets[0] == targets[1]
    assert targets[0] in {20, 40}


def test_trace_realized_output_length_overrides_calibration_sample_and_validates_budget():
    model = {"version": 1, "profiles": {
        "32:100": {"lengths": [20, 40], "counts": [1, 3], "n": 4},
    }}
    config = MigPPOEnvConfig(total_kv_blocks=128, output_length_model=model)
    trace = [{"request_id": "r00043", "arrival_time": 0.0, "prompt_tokens": 32,
              "output_tokens": 100, "output_budget_tokens": 100,
              "realized_output_tokens": 37, "realized_output_source": "fixed_training_sample"}]
    env = LlumnixMigrationEnv(config)
    env.reset(seed=19, options={"target_qps": 9, "arrival_trace": trace})
    env._process_arrival()
    request = env.requests["r00043"]
    assert request.target_output_tokens == 37
    assert request.requested_output_tokens == 100
    assert env.request_events[-1]["output_length_source"] == "fixed_training_sample"

    invalid_trace = [{**trace[0], "realized_output_tokens": 101}]
    env.reset(seed=19, options={"target_qps": 9, "arrival_trace": invalid_trace})
    with pytest.raises(ValueError, match="outside request budget"):
        env._process_arrival()


def test_decode_token_interval_scale_preserves_iteration_prediction_and_slows_emission():
    config = MigPPOEnvConfig(
        workload_scenario="low", arrival_rates_rps={"low": 0.0, "medium": 0.0,
            "high": 0.0, "bursty": 0.0},
        decode_intercept_ms=10.0, decode_ms_per_request=0.0,
        decode_ms_per_context_token=0.0, decode_token_interval_scale=3.0,
        total_kv_blocks=128,
    )
    env = LlumnixMigrationEnv(config)
    env.reset(seed=23, options={"scenario": "low"})
    request = _Request("decode", 0.0, 32, 16, stage="decode", allocated_blocks=1)
    env.instances[0].running.append(request)

    env._run_engine_step(env.instances[0], MigPPORewardMetrics(window_sec=0.1, instance_count=4), 0.0, 1.0)

    assert env.service_iterations[0]["predicted_ms"] == 10.0
    assert env.service_iterations[0]["effective_token_interval_ms"] == 30.0
    assert env.instances[0].next_step_time == pytest.approx(0.03)
    assert env.instances[0].pending_decode_time == pytest.approx(0.03)


def test_preempted_request_repeats_context_prefill_without_emitting_a_token():
    config = MigPPOEnvConfig(decision_interval_sec=0.1, max_decisions=2,
                             workload_scenario="low", arrival_rates_rps={"low": 0.0, "medium": 0.0,
                                 "high": 0.0, "bursty": 0.0}, total_kv_blocks=128,
                             max_num_batched_tokens=128, min_engine_step_ms=1.0)
    env = LlumnixMigrationEnv(config)
    env.reset(seed=8, options={"scenario": "low"})
    request = _Request("resume", 0.0, 64, 32, output_tokens=5, stage="waiting")
    env.requests[request.request_id] = request
    instance = env.instances[0]
    instance.waiting.append(request)
    assert env._schedule_waiting(instance) == [request]
    assert request.stage == "resume_prefill"
    assert request.allocated_blocks == env._blocks_for_tokens(69)
    instance.running.remove(request)
    instance.used_blocks = 0
    request.allocated_blocks = 0
    request.stage = "waiting"
    instance.waiting.append(request)
    metrics = MigPPORewardMetrics(window_sec=0.1, instance_count=4)
    env._run_engine_step(instance, metrics, 0.0, 1.0)
    env.now = instance.next_step_time
    env._run_engine_step(instance, metrics, 0.0, 1.0)
    assert request.stage == "decode"
    assert request.output_tokens == 5
    assert env.service_iterations[0]["batch_tokens"] == 69


def test_reset_from_trace_to_poisson_preserves_seeded_arrival_sequence():
    config = MigPPOEnvConfig(
        workload_scenario="low",
        arrival_rates_rps={"low": 2.0, "medium": 4.0, "high": 8.0, "bursty": 4.0},
        prompt_profiles=(32,), output_profiles=(8,),
    )
    trace = [{"request_id": "trace", "arrival_time": 0.0, "prompt_tokens": 32, "output_tokens": 8}]
    reused = LlumnixMigrationEnv(config)
    reused.reset(seed=71, options={"scenario": "low", "arrival_trace": trace})
    reused.reset(seed=71, options={"scenario": "low"})
    fresh = LlumnixMigrationEnv(config)
    fresh.reset(seed=71, options={"scenario": "low"})
    assert reused.next_arrival == fresh.next_arrival


def test_mixed_gpu_batch_emits_prefill_and_decode_at_one_batch_completion():
    config = MigPPOEnvConfig(
        decision_interval_sec=0.1, max_decisions=2, kv_dynamics_enabled=False,
        workload_scenario="low", arrival_rates_rps={"low": 0.0, "medium": 0.0,
            "high": 0.0, "bursty": 0.0},
        prefill_intercept_ms=1.0, prefill_ms_per_token=0.5,
        decode_intercept_ms=10.0, decode_ms_per_request=0.0,
        decode_ms_per_context_token=0.0, max_num_batched_tokens=128,
        trace_events_enabled=True, trace_token_events=True,
    )
    env = LlumnixMigrationEnv(config)
    env.reset(seed=72, options={"scenario": "low"})
    prefill = _Request("prefill", 0.0, 32, 4)
    decode = _Request("decode", 0.0, 32, 4, output_tokens=1,
                      stage="decode", first_token_time=0.0, last_token_time=0.0)
    instance = env.instances[0]
    instance.running.append(decode)
    instance.waiting.append(prefill)
    env.requests = {prefill.request_id: prefill, decode.request_id: decode}
    metrics = MigPPORewardMetrics(window_sec=0.1, instance_count=4)

    env._run_engine_step(instance, metrics, 0.0, 0.1)
    assert instance.next_step_time == pytest.approx(0.017)
    first_step_id = env.service_iterations[0]["step_id"]
    assert {row["phase"] for row in env.service_iterations} == {"prefill", "decode"}
    assert {row["step_id"] for row in env.service_iterations} == {first_step_id}
    assert {row["step_duration_ms"] for row in env.service_iterations} == {17.0}

    env.now = instance.next_step_time
    env._run_engine_step(instance, metrics, 0.0, 0.1)
    output_events = [row for row in env.lifecycle_events
                     if row["event"] in {"first_token", "decode_token"}]
    assert len(output_events) == 2
    assert all(row["timestamp_sec"] == pytest.approx(0.017) for row in output_events)
    assert prefill.output_tokens == 1
    assert decode.output_tokens == 2


def test_incoming_migration_reserves_destination_sequence_slot_and_kv_once():
    config = MigPPOEnvConfig(
        num_instances=2, max_num_seqs_per_instance=1, total_kv_blocks=128,
        workload_scenario="low", arrival_rates_rps={"low": 0.0, "medium": 0.0,
            "high": 0.0, "bursty": 0.0}, migration_failure_probability=0.0,
    )
    env = LlumnixMigrationEnv(config)
    env.reset(seed=73, options={"scenario": "low"})
    request = _Request("moving", 0.0, 32, 8, output_tokens=1, stage="decode", allocated_blocks=3)
    waiting = _Request("queued", 0.0, 32, 8)
    source, target = env.instances
    source.running.append(request)
    source.used_blocks = request.allocated_blocks
    target.waiting.append(waiting)
    env.requests = {request.request_id: request, waiting.request_id: waiting}
    assert env.start_replay_migration({
        "request_id": request.request_id, "source_instance": source.instance_id,
        "target_instance": target.instance_id, "blocks": 3, "cost_ms": 10.0,
        "status": "FINISHED",
    }) == "started"
    assert target.reserved_blocks == 3
    assert env._schedule_waiting(target) == []
    env._check_state_invariants()

    env.now = 0.01
    env._finish_migration(MigPPORewardMetrics(window_sec=0.1, instance_count=2), 0.0, 0.1)
    env._check_state_invariants()
    assert request in target.running
    assert waiting in target.waiting
    assert len(target.running) == 1
    assert source.used_blocks == 0
    assert target.used_blocks == request.allocated_blocks == 3
    assert target.reserved_blocks == 0


def test_timeout_during_migration_releases_source_then_target_reservation_once():
    config = MigPPOEnvConfig(
        num_instances=2, max_num_seqs_per_instance=1, total_kv_blocks=128, request_timeout_sec=0.05,
        workload_scenario="low", arrival_rates_rps={"low": 0.0, "medium": 0.0,
            "high": 0.0, "bursty": 0.0},
    )
    env = LlumnixMigrationEnv(config)
    env.reset(seed=74, options={"scenario": "low"})
    request = _Request("timing-out", 0.0, 32, 8, output_tokens=1,
                       stage="decode", allocated_blocks=3)
    waiting = _Request("still-waiting", 0.01, 32, 8)
    source, target = env.instances
    source.running.append(request)
    source.used_blocks = request.allocated_blocks
    target.waiting.append(waiting)
    env.requests = {request.request_id: request, waiting.request_id: waiting}
    assert env.start_replay_migration({
        "request_id": request.request_id, "source_instance": source.instance_id,
        "target_instance": target.instance_id, "blocks": 3, "cost_ms": 100.0,
        "status": "FINISHED",
    }) == "started"

    metrics = MigPPORewardMetrics(window_sec=0.1, instance_count=2)
    env.now = 0.05
    target.next_step_time = env.now
    env._run_engine_step(target, metrics, 0.0, 0.1)
    assert math.isinf(target.next_step_time)  # slot is held while the transfer is pending
    env._expire_requests(metrics, 0.0, 0.1)
    assert request.request_id not in env.requests
    assert source.used_blocks == 0
    assert target.reserved_blocks == 3
    env._run_engine_step(target, metrics, 0.0, 0.1)
    assert math.isinf(target.next_step_time)
    env._check_state_invariants()

    env.now = 0.1
    env._finish_migration(MigPPORewardMetrics(window_sec=0.1, instance_count=2), 0.0, 0.1)
    env._check_state_invariants()
    assert target.reserved_blocks == 0
    assert env.migration_events[0]["status"] == "aborted"
    assert target.next_step_time == pytest.approx(env.now)

