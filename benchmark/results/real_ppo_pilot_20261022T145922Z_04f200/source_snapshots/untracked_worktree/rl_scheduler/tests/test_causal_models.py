import json
import gzip
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_scheduler.causal_calibration import _event_rows, _km_profile, _normalize_training_seeds
from rl_scheduler.causal_event_audit import audit, write_report
from rl_scheduler.causal_event_io import compress_event_stream, event_stream_paths, open_event_stream
from rl_scheduler.causal_system_validation import (
    _fixed_batch_replay, _observed, _routine_balanced_action, _simulated,
)
from rl_scheduler.env import LlumnixMigrationEnv, MigPPOEnvConfig, _Request
from rl_scheduler.reward import MigPPORewardMetrics
from rl_scheduler.service_model import sample_qps_quantile_model

_causal_events_spec = importlib.util.spec_from_file_location(
    "_llumnix_causal_events_test", Path(__file__).parents[2] / "llumnix/metrics/causal_events.py")
_causal_events_module = importlib.util.module_from_spec(_causal_events_spec)
assert _causal_events_spec and _causal_events_spec.loader
_causal_events_spec.loader.exec_module(_causal_events_module)
should_sample_output_update = _causal_events_module.should_sample_output_update


def test_kaplan_meier_keeps_censored_tail_explicit():
    profile = _km_profile([(2, True), (4, False)], budget=5)
    assert profile["lengths"] == [2, 5]
    assert profile["weights"] == pytest.approx([0.5, 0.5])
    assert profile["completed_n"] == 1
    assert profile["censored_n"] == 1
    assert profile["tail_mass_at_budget"] == pytest.approx(0.5)


def test_calibration_seed_normalization_accepts_multiple_whole_run_groups():
    assert _normalize_training_seeds([909, 606, 909]) == ([606, 909], {606, 909})
    assert _normalize_training_seeds(606) == ([606], {606})


def test_output_length_profile_is_conditioned_on_load():
    model = {"version": 2, "profiles": {
        "32:100:9": {"lengths": [20], "counts": [1]},
        "32:100:12": {"lengths": [40], "counts": [1]},
    }}
    env = LlumnixMigrationEnv(MigPPOEnvConfig(total_kv_blocks=128,
                                                output_length_model=model))
    env.reset(seed=31, options={"target_qps": 9})
    assert env._output_distribution(32, 100)[0] == [20]
    env.reset(seed=31, options={"target_qps": 12})
    assert env._output_distribution(32, 100)[0] == [40]


def test_calibrated_batch_model_drives_one_engine_step():
    config = MigPPOEnvConfig(
        decision_interval_sec=0.1, max_decisions=2, kv_dynamics_enabled=False,
        workload_scenario="low", arrival_rates_rps={"low": 0.0, "medium": 0.0,
            "high": 0.0, "bursty": 0.0}, service_model_mode="calibrated",
        service_time_model={"batch_execution": {
            "terms": [{"op": "linear", "feature": "batch_size"}],
            "centers": [0.0], "scales": [1.0], "coefficients": [2.0],
            "intercept_ms": 5.0, "min_prediction_ms": 0.1,
        }},
    )
    env = LlumnixMigrationEnv(config)
    env.reset(seed=32, options={"scenario": "low"})
    request = _Request("decode", 0.0, 32, 8, output_tokens=1,
                       stage="decode", allocated_blocks=1)
    env.instances[0].running.append(request)
    env._run_engine_step(env.instances[0],
                         MigPPORewardMetrics(window_sec=0.1, instance_count=4), 0.0, 1.0)
    batch = next(row for row in env.service_iterations if row.get("phase") == "batch")
    assert batch["predicted_ms"] == pytest.approx(7.0)
    assert env.instances[0].pending_step_end_time == pytest.approx(0.007)


def test_qps_quantile_sampler_interpolates_profiles():
    class FixedRng:
        @staticmethod
        def random():
            return 0.5

    model = {"quantile_levels": [0.0, 1.0], "profiles": {
        "10": {"interval_quantiles_ms": [10.0, 20.0]},
        "20": {"interval_quantiles_ms": [30.0, 60.0]},
    }}
    assert sample_qps_quantile_model(model, 15.0, FixedRng()) == pytest.approx(30.0)


def test_calibrated_api_cadence_is_separate_from_engine_execution():
    config = MigPPOEnvConfig(
        decision_interval_sec=0.2, max_decisions=1, kv_dynamics_enabled=False,
        workload_scenario="low", arrival_rates_rps={"low": 1.0, "medium": 1.0,
            "high": 1.0, "bursty": 1.0}, service_model_mode="calibrated",
        service_time_model={"batch_execution": {
            "terms": [], "centers": [], "scales": [], "coefficients": [],
            "intercept_ms": 5.0, "min_prediction_ms": 0.1,
        }},
        decode_token_interval_model={"quantile_levels": [0.0, 1.0], "profiles": {
            "1": {"interval_quantiles_ms": [100.0, 100.0]},
        }},
        trace_events_enabled=True,
    )
    env = LlumnixMigrationEnv(config)
    env.reset(seed=320, options={"target_qps": 1.0})
    request = _Request("decode", 0.0, 32, 8, output_tokens=1,
                       stage="decode", allocated_blocks=1)
    env.instances[0].running.append(request)
    env._run_engine_step(env.instances[0],
                         MigPPORewardMetrics(window_sec=0.2, instance_count=4), 0.0, 1.0)
    batch = next(row for row in env.service_iterations if row.get("phase") == "batch")
    scheduled = next(row for row in env.lifecycle_events if row.get("event") == "engine_step_scheduled")
    assert batch["engine_execution_ms"] == pytest.approx(5.0)
    assert batch["output_cadence_ms"] == pytest.approx(100.0)
    assert scheduled["batch_duration_ms"] == pytest.approx(100.0)
    assert env.instances[0].pending_step_end_time == pytest.approx(0.1)


def test_migration_total_pause_and_recovery_are_separate_stages():
    config = MigPPOEnvConfig(
        num_instances=2, total_kv_blocks=128, request_timeout_sec=None,
        workload_scenario="low", arrival_rates_rps={"low": 0.0, "medium": 0.0,
            "high": 0.0, "bursty": 0.0}, migration_failure_probability=0.0,
        trace_events_enabled=True, trace_token_events=True,
        validate_state_invariants=True,
    )
    env = LlumnixMigrationEnv(config)
    request = _Request("moving", 0.0, 32, 8, output_tokens=1,
                       stage="decode", allocated_blocks=1,
                       first_token_time=0.0, last_token_time=0.0)
    env.reset(seed=33, options={"scenario": "low", "migration_event_trace": [{
        "migration_id": "m1", "migration_time_sec": 0.0,
        "request_id": request.request_id, "source_instance": "instance_0",
        "target_instance": "instance_1", "blocks": 1, "cost_ms": 100.0,
        "pause_stage_ms": 40.0, "recovery_after_resume_ms": 20.0,
        "status": "FINISHED",
    }]})
    source = env.instances[0]
    source.running.append(request)
    source.used_blocks = 1
    env.requests[request.request_id] = request
    env._advance(MigPPORewardMetrics(window_sec=0.2, instance_count=2), 0.0, 0.2)
    assert env.migration_replay_results[0]["result"] == "started"
    assert env.migration_events[0]["cost_ms"] == pytest.approx(100.0)
    assert env.migration_events[0]["pause_stage_ms"] == pytest.approx(40.0)
    assert any(row["request_id"] == request.request_id and row["status"] == "completed"
               for row in env.request_results)
    pause_events = [row for row in env.lifecycle_events
                    if row["event"] == "migration_request_pause_started"]
    token_events = [row for row in env.lifecycle_events if row["event"] == "decode_token"]
    assert len(pause_events) == 1
    assert token_events
    assert min(row["timestamp_sec"] for row in token_events) >= 0.11


def test_event_audit_reports_gaps_and_joins(tmp_path):
    data_dir = tmp_path / "run"
    event_dir = data_dir / "causal_events"
    event_dir.mkdir(parents=True)
    events = [
        {"event": "request_server_received", "request_id": "r1", "pid": 5,
         "host": "h", "event_sequence": 1, "monotonic_ns": 10},
        {"event": "batch_scheduled", "batch_id": "b1", "pid": 5,
         "host": "h", "event_sequence": 3, "monotonic_ns": 12},
        {"event": "engine_iteration_completed", "batch_id": "b1", "pid": 5,
         "host": "h", "event_sequence": 4, "monotonic_ns": 13},
    ]
    (event_dir / "events_h_5.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8")
    result = audit(data_dir)
    assert result["processes"]["h:5"]["sequence_missing"] == 1
    assert result["batch_joins"]["scheduled_batches"] == 1
    assert result["batch_joins"]["completed_batches"] == 1
    assert result["request_lifecycle"]["received_without_terminal"] == ["r1"]
    write_report(data_dir, result)
    assert (data_dir / "event_integrity_audit.json").exists()
    assert (data_dir / "event_integrity_audit.md").exists()


def test_event_audit_reads_losslessly_compressed_source_streams(tmp_path):
    event_dir = tmp_path / "causal_events"
    event_dir.mkdir(parents=True)
    path = event_dir / "events_host_5.jsonl.gz"
    event = {"event": "request_server_received", "request_id": "r1", "pid": 5,
             "host": "h", "event_sequence": 1, "monotonic_ns": 10}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
    result = audit(tmp_path)
    assert result["source_stream_count"] == 1
    assert result["total_events"] == 1
    assert result["request_lifecycle"]["server_received_requests"] == 1


def test_shard_event_archiver_verifies_and_reopens_gzip(tmp_path):
    run_dir = tmp_path / "run"
    event_dir = run_dir / "causal_events"
    event_dir.mkdir(parents=True)
    path = event_dir / "events.jsonl"
    line = json.dumps({"event": "request_server_received", "request_id": "r1"}) + "\n"
    path.write_text(line, encoding="utf-8")

    archive = compress_event_stream(path)
    archived_path = event_dir / "events.jsonl.gz"
    assert not path.exists()
    assert archive["encoding"] == "gzip"
    assert archive["uncompressed_bytes"] == len(line.encode("utf-8"))
    assert event_stream_paths(event_dir) == [archived_path]
    with open_event_stream(archived_path) as stream:
        assert stream.read() == line
    assert list(_event_rows(run_dir, {}, measurement_only=False)) == [
        {"event": "request_server_received", "request_id": "r1"}
    ]


def test_output_queue_sampling_keeps_boundaries_and_sparse_updates(monkeypatch):
    monkeypatch.setenv("LLUMNIX_CAUSAL_TRACE_DIR", "/tmp/causal-trace-test")
    assert should_sample_output_update("r1", 1, False) is True
    assert should_sample_output_update("r1", 64, False) is True
    assert should_sample_output_update("r1", 65, False) is False
    assert should_sample_output_update("r1", 17, True) is True
    assert should_sample_output_update("r1", 0, False) is False


def test_validation_uses_nonempty_batches_for_queue_and_kv_state(tmp_path):
    run_dir = tmp_path / "run"
    event_dir = run_dir / "causal_events"
    event_dir.mkdir(parents=True)
    events = [
        {"event": "batch_scheduled", "monotonic_ns": 1_050_000_000,
         "instance_id": "i0", "num_waiting_requests": 2, "kv_utilization": 0.25},
        {"event": "batch_scheduled", "monotonic_ns": 1_100_000_000,
         "instance_id": "i0", "num_waiting_requests": 4, "kv_utilization": 0.75},
    ]
    (event_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    observed = _observed(
        run_dir, [], {"measurement_window": {"start_monotonic_ns_estimate": 1_000_000_000}},
        measurement_sec=0.1, drain_sec=0.1)
    assert observed["queue_mean_requests"] == pytest.approx(3.0)
    assert observed["kv_mean_fraction"] == pytest.approx(0.5)
    assert observed["queue_samples"] == 2


def test_validation_reports_output_delivery_lag_and_ignores_empty_token_updates(tmp_path):
    run_dir = tmp_path / "run"
    event_dir = run_dir / "causal_events"
    event_dir.mkdir(parents=True)
    events = [
        {"event": "request_server_received", "request_id": "r1", "monotonic_ns": 1_000_000_000},
        {"event": "request_output_at_api", "request_id": "r1", "output_tokens": 0,
         "new_output_tokens": 0, "monotonic_ns": 1_010_000_000},
        {"event": "engine_request_output", "request_id": "r1", "output_tokens": 1,
         "instance_id": "i0", "monotonic_ns": 1_020_000_000},
        {"event": "output_forwarder_submit", "request_id": "r1", "instance_id": "i0",
         "output_tokens": 1, "monotonic_ns": 1_020_500_000},
        {"event": "output_forwarder_loop_start", "request_id": "r1", "instance_id": "i0",
         "output_tokens": 1, "monotonic_ns": 1_020_600_000},
        {"event": "output_forwarder_queue_client_send", "request_id": "r1", "instance_id": "i0",
         "output_tokens": 1, "monotonic_ns": 1_020_800_000},
        {"event": "output_queue_enqueue", "request_id": "r1", "instance_id": "i0",
         "output_tokens": 1, "queue_depth_messages": 2, "monotonic_ns": 1_021_000_000},
        {"event": "output_queue_dequeue", "request_id": "r1", "instance_id": "i0",
         "output_tokens": 1, "monotonic_ns": 1_025_000_000},
        {"event": "request_output_at_api", "request_id": "r1", "output_tokens": 1,
         "instance_id": "i0", "new_output_tokens": 1, "monotonic_ns": 1_030_000_000},
        {"event": "engine_request_output", "request_id": "r1", "output_tokens": 2,
         "instance_id": "i0", "monotonic_ns": 1_040_000_000},
        {"event": "request_output_at_api", "request_id": "r1", "output_tokens": 2,
         "instance_id": "i0", "new_output_tokens": 1, "monotonic_ns": 1_050_000_000},
        {"event": "request_server_terminal", "request_id": "r1", "terminal_status": "server_completed",
         "output_tokens": 2, "monotonic_ns": 1_060_000_000},
    ]
    (event_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    rows = [{"phase": "measurement", "request_id": "r1", "server_request_id": "r1",
             "actual_send_sec": "0.0", "status_code": "200", "success": "true"}]
    observed = _observed(
        run_dir, rows, {"measurement_window": {"start_monotonic_ns_estimate": 1_000_000_000}},
        measurement_sec=0.1, drain_sec=0.1)
    assert observed["ttft_samples"] == 1
    assert observed["ttft_mean_ms"] == pytest.approx(30.0)
    assert observed["tpot_samples"] == 1
    assert observed["tpot_mean_ms"] == pytest.approx(20.0)
    delivery = observed["output_delivery_diagnostics"]
    assert delivery["engine_output_updates"] == 2
    assert delivery["matched_api_output_updates"] == 2
    assert delivery["first_output_engine_to_api_delay"]["p50_ms"] == pytest.approx(10.0)
    assert delivery["measurement_first_api_token_in_window"] == 1
    assert delivery["first_api_token_window_coverage_of_sent"] == pytest.approx(1.0)
    stages = observed["output_queue_stage_diagnostics"]
    assert stages["matched_enqueue_dequeue_samples"] == 1
    assert stages["matched_engine_enqueue_samples"] == 1
    assert stages["engine_to_enqueue"]["p50_ms"] == pytest.approx(1.0)
    forwarding = stages["forwarder_stages"]
    assert forwarding["engine_to_submit"]["p50_ms"] == pytest.approx(0.5)
    assert forwarding["submit_to_loop_start"]["p50_ms"] == pytest.approx(0.1)
    assert forwarding["loop_start_to_queue_send"]["p50_ms"] == pytest.approx(0.2)
    assert forwarding["queue_send_to_enqueue"]["p50_ms"] == pytest.approx(0.2)
    assert stages["queue_residence"]["p50_ms"] == pytest.approx(4.0)
    assert stages["dequeue_to_api_callback"]["p50_ms"] == pytest.approx(5.0)
    assert stages["queue_depth_messages"]["p95"] == pytest.approx(2.0)


def test_warmup_boundary_is_measured_from_server_lifecycle_events(tmp_path):
    run_dir = tmp_path / "run"
    event_dir = run_dir / "causal_events"
    event_dir.mkdir(parents=True)
    events = [
        {"event": "request_server_received", "request_id": "warm1", "monotonic_ns": 900},
        {"event": "request_server_terminal", "request_id": "warm1", "terminal_status": "server_completed",
         "monotonic_ns": 990},
    ]
    (event_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    rows = [{"phase": "warmup", "request_id": "warm1", "server_request_id": "warm1"}]
    config = {"measurement_window": {"start_monotonic_ns_estimate": 1000}}
    observed = _observed(run_dir, rows, config, measurement_sec=0.1, drain_sec=0.1)
    assert observed["warmup_server_requests"] == 1
    assert observed["warmup_terminal_before_measurement"] == 1
    assert observed["warmup_active_at_measurement_start"] == 0
    assert observed["warmup_phase_cleared_before_measurement"] is True

    events[-1]["monotonic_ns"] = 1001
    (event_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    observed = _observed(run_dir, rows, config, measurement_sec=0.1, drain_sec=0.1)
    assert observed["warmup_active_at_measurement_start"] == 1
    assert observed["warmup_phase_cleared_before_measurement"] is False


def test_fixed_batch_replay_scores_same_observed_batch_for_both_models(tmp_path):
    run_dir = tmp_path / "run"
    event_dir = run_dir / "causal_events"
    event_dir.mkdir(parents=True)
    batch_model = {
        "terms": [{"op": "linear", "feature": "prefill_tokens"}],
        "centers": [0.0], "scales": [1.0], "coefficients": [3.0 / 32.0],
        "intercept_ms": 2.0, "min_prediction_ms": 0.1,
    }
    event = {
        "event": "engine_iteration_completed", "monotonic_ns": 150,
        "executor_wall_ms": 5.0, "succeeded": True, "batch_type": "prefill",
        "batch_size": 1, "num_prefill_requests": 1, "num_decode_requests": 0,
        "num_running_sequences": 1, "num_waiting_requests": 0,
        "num_batched_tokens": 32, "prefill_tokens": 32, "decode_tokens": 0,
        "total_context_tokens": 32, "kv_utilization": 0.1,
    }
    (event_dir / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    historical_batch = {"terms": [{"op": "linear", "feature": "batch_tokens"}],
                        "centers": [0.0], "scales": [1.0], "coefficients": [0.1],
                        "intercept_ms": 2.0, "min_prediction_ms": 0.1}
    historical = MigPPOEnvConfig(service_model_mode="calibrated",
        service_time_model={"prefill": historical_batch, "decode": None})
    result = _fixed_batch_replay(
        run_dir, {"measurement_window": {"start_monotonic_ns_estimate": 100,
                                         "end_monotonic_ns_estimate": 200}},
        batch_model, historical)
    assert result["overall"]["causal_rebuild"]["n"] == 1
    assert result["overall"]["round2_v3"]["n"] == 1
    assert result["overall"]["causal_rebuild"]["mae_ms"] == pytest.approx(0.0)


def test_fixed_batch_replay_uses_joined_full_engine_step_wall(tmp_path):
    run_dir = tmp_path / "run"
    event_dir = run_dir / "causal_events"
    event_dir.mkdir(parents=True)
    batch_event = {
        "event": "engine_iteration_completed", "batch_id": "b1", "monotonic_ns": 150,
        "executor_wall_ms": 5.0, "succeeded": True, "batch_type": "decode",
        "batch_size": 1, "num_prefill_requests": 0, "num_decode_requests": 1,
        "num_running_sequences": 1, "num_waiting_requests": 0,
        "num_batched_tokens": 1, "prefill_tokens": 0, "decode_tokens": 1,
        "total_context_tokens": 32, "kv_utilization": 0.1,
    }
    engine_event = {"event": "llm_engine_step_completed", "batch_id": "b1",
                    "monotonic_ns": 151, "engine_step_wall_ms": 12.0}
    with gzip.open(event_dir / "events.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write("\n".join(json.dumps(event) for event in (batch_event, engine_event)) + "\n")
    model = {"terms": [], "centers": [], "scales": [], "coefficients": [],
             "intercept_ms": 12.0, "min_prediction_ms": 0.1}
    result = _fixed_batch_replay(
        run_dir, {"measurement_window": {"start_monotonic_ns_estimate": 100,
                                         "end_monotonic_ns_estimate": 200}},
        model, MigPPOEnvConfig())
    assert result["overall"]["causal_rebuild"]["mae_ms"] == pytest.approx(0.0)


def test_simulation_metrics_use_relative_lifecycle_timestamp():
    config = MigPPOEnvConfig(
        decision_interval_sec=0.05, request_timeout_sec=None,
        prompt_profiles=(32,), output_profiles=(1, 2), max_model_len=128,
        arrival_rates_rps={"low": 0.0, "medium": 0.0, "high": 0.0, "bursty": 0.0},
        trace_events_enabled=True, trace_token_events=True,
        validate_state_invariants=True,
    )
    trace = [{"phase": "measurement", "request_id": "r1", "arrival_time": 0.0,
              "prompt_tokens": 32, "output_tokens": 1, "output_budget_tokens": 1}]

    result = _simulated(trace, {"r1"}, config, qps=3.0, seed=1, warmup_sec=0.0,
                        measurement_sec=1.0, drain_sec=1.0)

    assert result["completion_count_window"] == 1
    assert result["request_throughput_rps"] == pytest.approx(1.0)
    assert result["output_tokens_window"] == 1
    assert result["ttft_samples"] == 1


def test_routine_balanced_controller_uses_runtime_threshold_and_legal_pair():
    source = SimpleNamespace(
        instance_id="source", preempted_requests=0, free_blocks=80,
        running=[SimpleNamespace(request_len=10), SimpleNamespace(request_len=20)],
        waiting=[],
    )
    destination = SimpleNamespace(
        instance_id="destination", preempted_requests=0, free_blocks=510,
        running=[SimpleNamespace(request_len=10), SimpleNamespace(request_len=10)],
        waiting=[],
    )
    env = SimpleNamespace(
        _last_obs=SimpleNamespace(action_mask=SimpleNamespace(
            pairs=[("source", "destination")]), pair_mask=[True]),
        pending_migration=None,
        instances=[source, destination],
        _instance_views=lambda: {
            "source": SimpleNamespace(remaining_steps=SimpleNamespace(remaining_steps=100.0)),
            "destination": SimpleNamespace(remaining_steps=SimpleNamespace(remaining_steps=200.0)),
        },
        _sr_candidate=lambda instance: object() if instance is source else None,
        _blocks_for_tokens=lambda tokens: float(tokens),
    )
    policy = {"enabled": True, "migration_load_metric": "remaining_steps",
              "pair_migration_policy": "balanced", "migrate_out_threshold": 150.0}

    assert _routine_balanced_action(env, policy) == 1
    assert _routine_balanced_action(env, {**policy, "enabled": False}) == 0
