from benchmark.mig_ppo_v2_multienv import (
    _frozen_matrix_projection,
    _join_instance_gpu_mapping,
    _trace_base,
    _trace_with_routing,
)
from rl_scheduler.real_bridge import _apply_instance_order


def _base_trace(n=48):
    return [
        {
            "request_id": f"r{i}",
            "scheduled_arrival_sec": i / 10,
            "input_tokens": 128,
            "output_tokens": 1536,
            "initial_instance_index": i % 4,
            "hot_instance_index": 0,
            "routing_mode": "fixed",
            "candidate_id": "fixture",
        }
        for i in range(n)
    ]


def test_registered_hotspot_levels_have_exact_stable_counts_and_shared_base():
    trace = _base_trace()
    low = _trace_with_routing(trace, "HOT1-L", "fixed", .50, 0, 8123)
    medium = _trace_with_routing(trace, "HOT1-M", "fixed", .70, 0, 8123)
    high = _trace_with_routing(trace, "HOT1-H", "fixed", .90, 0, 8123)
    assert [sum(r["initial_instance_index"] == 0 for r in rows)
            for rows in (low, medium, high)] == [24, 34, 43]
    assert [sum(r["initial_instance_index"] == i for r in medium) for i in range(4)] == [34, 5, 5, 4]
    assert _trace_base(low) == _trace_base(medium) == _trace_base(high)


def test_native_trace_never_injects_initial_route():
    trace = _trace_with_routing(_base_trace(), "NATIVE", "native")
    assert all(row["routing_mode"] == "native" for row in trace)
    assert all(row["initial_instance_index"] is None for row in trace)


def test_runtime_status_does_not_change_frozen_matrix_projection():
    planned = [{"environment_id": "HOT1-L", "status": "not_run",
                "candidates": [{"candidate_id": "p", "status": "preregistered",
                                "full_trace_hash": "abc"}]}]
    executed = [{"environment_id": "HOT1-L", "status": "selected",
                 "selected_candidate_id": "p", "selection_passed": True,
                 "candidates": [{"candidate_id": "p", "status": "used",
                                 "full_trace_hash": "abc",
                                 "validation_base_trace_hash": "independent"}]}]
    assert _frozen_matrix_projection(planned) == _frozen_matrix_projection(executed)


def test_worker_mapping_reads_flat_causal_engine_event_schema(tmp_path):
    events = tmp_path / "events_fixture.jsonl"
    ids = [f"instance-{i}" for i in range(4)]
    events.write_text("".join(
        '{"event":"request_engine_admitted","instance_id":"%s","pid":%d}\n' % (iid, 100 + i)
        for i, iid in enumerate(ids)
    ))
    pid_gpu = {100 + i: {"gpu_uuid": f"GPU-{i}", "used_memory": "1 GiB"}
               for i in range(4)}
    mapping = _join_instance_gpu_mapping([(iid, f"epoch-{i}") for i, iid in enumerate(ids)],
                                          tmp_path, pid_gpu)
    assert [row["gpu_uuid"] for row in mapping] == [f"GPU-{i}" for i in range(4)]


def test_bridge_restores_frozen_logical_order_after_service_restart():
    ids = [f"instance-{i}" for i in range(4)]
    snapshot = {
        "mapping": [[ids[i], f"epoch-{i}"] for i in range(4)],
        "instances": [{"instance_id": ids[i], "value": i} for i in range(4)],
    }
    expected = [[ids[i], f"epoch-{i}"] for i in (2, 0, 3, 1)]
    aligned = _apply_instance_order(snapshot, expected)
    assert aligned["mapping"] == expected
    assert [row["instance_id"] for row in aligned["instances"]] == [ids[i] for i in (2, 0, 3, 1)]
