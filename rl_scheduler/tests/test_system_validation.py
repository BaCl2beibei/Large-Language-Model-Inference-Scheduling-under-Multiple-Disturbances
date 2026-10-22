import csv

import pytest

from rl_scheduler.system_validation import _observed_state, _request_trace


def test_request_trace_preserves_explicit_training_sample_and_budget(tmp_path):
    trace_path = tmp_path / "trace.csv"
    request_path = tmp_path / "requests.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "request_id", "scheduled_arrival_sec", "input_tokens", "output_tokens",
            "realized_output_tokens", "realized_output_source",
        ])
        writer.writeheader()
        writer.writerow({"request_id": "r00001", "scheduled_arrival_sec": 0.25,
                         "input_tokens": 32, "output_tokens": 100,
                         "realized_output_tokens": 37,
                         "realized_output_source": "fixed_training_sample"})
    with request_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["request_id", "phase", "actual_send_sec"])
        writer.writeheader()
        writer.writerow({"request_id": "run_measurement_r00001", "phase": "measurement",
                         "actual_send_sec": 0.3})

    trace = _request_trace(trace_path, request_path)

    assert trace == [{
        "request_id": "r00001", "arrival_time": 0.3,
        "scheduled_arrival_sec": 0.25, "actual_send_sec": 0.3,
        "prompt_tokens": 32, "output_tokens": 100, "output_budget_tokens": 100,
        "realized_output_tokens": 37, "realized_output_source": "fixed_training_sample",
    }]


def test_observed_queue_and_kv_are_sampled_on_simulator_fixed_grid(tmp_path):
    run_dir = tmp_path / "run"
    state_dir = run_dir / "server_logs"
    state_dir.mkdir(parents=True)
    path = state_dir / "on_instance_instance.csv"
    rows = [
        {"instance_id": "i0", "step_id": "1", "timestamp": "1000.01",
         "num_waiting_requests": "2", "gpu_cache_usage": "0.2"},
        {"instance_id": "i1", "step_id": "1", "timestamp": "1000.04",
         "num_waiting_requests": "4", "gpu_cache_usage": "0.4"},
        {"instance_id": "i0", "step_id": "2", "timestamp": "1000.08",
         "num_waiting_requests": "0", "gpu_cache_usage": "0.1"},
        {"instance_id": "i1", "step_id": "2", "timestamp": "1000.12",
         "num_waiting_requests": "2", "gpu_cache_usage": "0.2"},
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = _observed_state(run_dir, start_epoch=1000.0, measurement_sec=0.15,
                             sample_interval_sec=0.05)

    assert result["queue_mean_requests"] == pytest.approx(2.0)
    assert result["queue_p95_requests"] == pytest.approx(2.9)
    assert result["kv_mean_fraction"] == pytest.approx(0.7 / 3.0)
    assert result["kv_p95_fraction"] == pytest.approx(0.295)
    assert result["state_grid_sample_n"] == 3
    assert result["unique_engine_state_rows_used"] == 4
