"""Validate causal event models against wholly held-out Llumnix run groups.

The simulator receives only the held-out arrival times, prompt sizes, and
output budgets. Server outcomes are read after simulation for scoring.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from rl_scheduler.causal_calibration import _batch_features, _read_csv
from rl_scheduler.causal_event_io import event_stream_paths, open_event_stream
from rl_scheduler.cost_estimator import MigPPOMigrationCostEstimator
from rl_scheduler.env import LlumnixMigrationEnv, MigPPOEnvConfig
from rl_scheduler.service_model import predict_service_model


def _num(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _output_delivery_diagnostics(
    engine_output_times: Mapping[tuple[str, str, int], int],
    api_outputs: Mapping[str, list[tuple[int, int, int, str]]],
    engine_output_updates: int,
    sent_measurement_ids: set[str],
    request_received: Mapping[str, int],
    start_ns: int,
    end_ns: int,
    completion_horizon_ns: int,
) -> dict[str, Any]:
    """Summarize engine-output to API-callback delay and first-token coverage.

    These are observational diagnostics only. They are not fed into the
    simulator or used by the independent accuracy gate.
    """
    delays_ms: list[float] = []
    first_delays_ms: list[float] = []
    first_api_time: dict[str, int] = {}
    unmatched_api_updates = 0
    for request_id, rows in api_outputs.items():
        if request_id not in sent_measurement_ids:
            continue
        token_rows = sorted((row for row in rows
                             if row[1] > 0 and row[0] <= completion_horizon_ns),
                            key=lambda row: row[0])
        if token_rows:
            first_api_time[request_id] = token_rows[0][0]
        for timestamp, output_tokens, _new_tokens, instance_id in token_rows:
            engine_timestamp = engine_output_times.get((request_id, instance_id, output_tokens))
            if engine_timestamp is None or engine_timestamp > timestamp:
                unmatched_api_updates += 1
                continue
            delay_ms = (timestamp - engine_timestamp) / 1e6
            delays_ms.append(delay_ms)
            if token_rows[0][0] == timestamp and token_rows[0][1] == output_tokens:
                first_delays_ms.append(delay_ms)

    def summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"samples": 0, "mean_ms": None, "p50_ms": None,
                    "p95_ms": None, "p99_ms": None, "max_ms": None}
        return {
            "samples": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.quantile(values, .50)),
            "p95_ms": float(np.quantile(values, .95)),
            "p99_ms": float(np.quantile(values, .99)),
            "max_ms": float(max(values)),
        }

    first_in_window = sum(start_ns <= first_api_time.get(request_id, -1) <= end_ns
                          for request_id in sent_measurement_ids)
    first_after_window = sum(end_ns < first_api_time.get(request_id, -1) <= completion_horizon_ns
                             for request_id in sent_measurement_ids)
    received = sum(request_id in request_received for request_id in sent_measurement_ids)
    api_by_horizon = sum(first_api_time.get(request_id, completion_horizon_ns + 1)
                         <= completion_horizon_ns for request_id in sent_measurement_ids)
    engine_request_ids = {request_id for request_id, _instance_id, _tokens in engine_output_times}
    engine_without_api = sum(
        request_id in engine_request_ids
        and first_api_time.get(request_id, completion_horizon_ns + 1) > completion_horizon_ns
        for request_id in sent_measurement_ids
    )
    return {
        "engine_output_updates": int(engine_output_updates),
        "api_output_updates": int(sum(
            request_id in sent_measurement_ids
            for request_id, rows in api_outputs.items()
            for timestamp, _tokens, _new_tokens, _instance_id in rows
            if timestamp <= completion_horizon_ns
        )),
        "matched_api_output_updates": len(delays_ms),
        "unmatched_api_output_updates": unmatched_api_updates,
        "engine_to_api_callback_delay": summary(delays_ms),
        "first_output_engine_to_api_delay": summary(first_delays_ms),
        "measurement_sent_requests": len(sent_measurement_ids),
        "measurement_server_received_requests": received,
        "measurement_first_api_token_in_window": first_in_window,
        "measurement_first_api_token_after_window_by_drain": first_after_window,
        "measurement_first_api_token_by_drain": api_by_horizon,
        "measurement_engine_output_without_api_token_by_drain": engine_without_api,
        "first_api_token_window_coverage_of_sent": (
            first_in_window / len(sent_measurement_ids) if sent_measurement_ids else None
        ),
        "measurement_first_token_right_censored_at_window": max(
            0, len(sent_measurement_ids) - first_in_window
        ),
    }


def _output_queue_stage_diagnostics(
    engine_output_times: Mapping[tuple[str, str, int], int],
    forwarder_stage_times: Mapping[str, Mapping[tuple[str, str, int], int]],
    enqueue_events: Mapping[tuple[str, str, int], tuple[int, int]],
    dequeue_events: Mapping[tuple[str, str, int], int],
    api_callback_times: Mapping[tuple[str, str, int], int],
    measurement_request_ids: set[str],
    completion_horizon_ns: int,
) -> dict[str, Any]:
    """Summarize the sampled ZMQ residence and post-dequeue callback stages."""
    enqueue_events = {
        key: value for key, value in enqueue_events.items()
        if key[0] in measurement_request_ids and value[0] <= completion_horizon_ns
    }
    dequeue_events = {
        key: value for key, value in dequeue_events.items()
        if key[0] in measurement_request_ids and value <= completion_horizon_ns
    }
    api_callback_times = {
        key: value for key, value in api_callback_times.items()
        if key[0] in measurement_request_ids and value <= completion_horizon_ns
    }
    forwarder_stage_times = {
        event_name: {
            key: timestamp for key, timestamp in values.items()
            if key[0] in measurement_request_ids and timestamp <= completion_horizon_ns
        }
        for event_name, values in forwarder_stage_times.items()
    }
    engine_output_times = {
        key: timestamp for key, timestamp in engine_output_times.items()
        if key[0] in measurement_request_ids and timestamp <= completion_horizon_ns
    }
    engine_to_enqueue_ms = [
        (value[0] - engine_output_times[key]) / 1e6
        for key, value in enqueue_events.items()
        if key in engine_output_times and value[0] >= engine_output_times[key]
    ]
    queue_residence_ms = [
        (dequeue_events[key] - value[0]) / 1e6
        for key, value in enqueue_events.items()
        if key in dequeue_events and dequeue_events[key] >= value[0]
    ]
    callback_after_dequeue_ms = [
        (api_callback_times[key] - dequeue_events[key]) / 1e6
        for key in dequeue_events
        if key in api_callback_times and api_callback_times[key] >= dequeue_events[key]
    ]
    queue_depths = [float(depth) for _timestamp, depth in enqueue_events.values()]

    def stage_delays(before: Mapping[tuple[str, str, int], int],
                     after: Mapping[tuple[str, str, int], int]) -> list[float]:
        return [
            (after[key] - timestamp) / 1e6
            for key, timestamp in before.items()
            if key in after and after[key] >= timestamp
        ]

    submit_times = forwarder_stage_times.get("output_forwarder_submit", {})
    loop_start_times = forwarder_stage_times.get("output_forwarder_loop_start", {})
    send_times = forwarder_stage_times.get("output_forwarder_queue_client_send", {})
    enqueue_times = {key: value[0] for key, value in enqueue_events.items()}
    forwarder_delays = {
        "engine_to_submit": stage_delays(engine_output_times, submit_times),
        "submit_to_loop_start": stage_delays(submit_times, loop_start_times),
        "loop_start_to_queue_send": stage_delays(loop_start_times, send_times),
        "queue_send_to_enqueue": stage_delays(send_times, enqueue_times),
    }

    def summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"samples": 0, "mean_ms": None, "p50_ms": None,
                    "p95_ms": None, "p99_ms": None, "max_ms": None}
        return {
            "samples": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.quantile(values, .50)),
            "p95_ms": float(np.quantile(values, .95)),
            "p99_ms": float(np.quantile(values, .99)),
            "max_ms": float(max(values)),
        }

    return {
        "enqueue_samples": len(enqueue_events),
        "dequeue_samples": len(dequeue_events),
        "matched_enqueue_dequeue_samples": len(queue_residence_ms),
        "matched_engine_enqueue_samples": len(engine_to_enqueue_ms),
        "matched_dequeue_api_callback_samples": len(callback_after_dequeue_ms),
        "unmatched_enqueue_samples": len(set(enqueue_events) - set(dequeue_events)),
        "unmatched_dequeue_samples": len(set(dequeue_events) - set(enqueue_events)),
        "queue_residence": summary(queue_residence_ms),
        "engine_to_enqueue": summary(engine_to_enqueue_ms),
        "forwarder_stages": {
            name: {**summary(values), "matched_samples": len(values)}
            for name, values in forwarder_delays.items()
        },
        "dequeue_to_api_callback": summary(callback_after_dequeue_ms),
        "queue_depth_messages": {
            "samples": len(queue_depths),
            "p50": float(np.quantile(queue_depths, .50)) if queue_depths else None,
            "p95": float(np.quantile(queue_depths, .95)) if queue_depths else None,
            "max": max(queue_depths) if queue_depths else None,
        },
    }


def _short_id(value: object) -> str | None:
    match = re.search(r"(r\d{5,})$", str(value or ""))
    return match.group(1) if match else None


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths_for_run(data_dir: Path, qps: float, seed: int) -> Path:
    label = f"qps_{qps:g}"
    path = data_dir / "validation" / f"{label}_seed{seed}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_run_inputs(run_dir: Path, warmup_sec: float) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    request_rows = _read_csv(run_dir / "requests.csv")
    raw_by_phase_short = {
        (str(row.get("phase", "")), _short_id(row.get("request_id"))): row
        for row in request_rows if _short_id(row.get("request_id"))
    }
    trace_paths = list((run_dir / "traces").glob("*.csv"))
    trace_rows: list[dict[str, Any]] = []
    for path in trace_paths:
        is_warmup = path.name.startswith("warmup_")
        if not is_warmup and not path.name.startswith("trace_"):
            continue
        for row in _read_csv(path):
            short = _short_id(row.get("request_id")) or str(row.get("request_id", ""))
            phase = "warmup" if is_warmup else "measurement"
            raw = raw_by_phase_short.get((phase, short))
            if raw is not None:
                phase = str(raw.get("phase", phase))
                actual = _num(raw.get("actual_send_sec"))
            else:
                actual = None
            scheduled = _num(row.get("scheduled_arrival_sec"), 0.0) or 0.0
            phase_send = actual if actual is not None else scheduled
            arrival = phase_send - warmup_sec if phase == "warmup" else phase_send
            if raw is None:
                request_id = str(row.get("request_id", ""))
            else:
                request_id = str(raw.get("request_id", row.get("request_id", "")))
            try:
                prompt = int(row["input_tokens"])
                budget = int(row["output_tokens"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            trace_rows.append({"request_id": request_id, "short_id": short,
                               "arrival_time": float(arrival),
                               "scheduled_arrival_sec": float(scheduled),
                               "actual_send_sec": actual,
                               "prompt_tokens": prompt,
                               "output_tokens": budget,
                               "output_budget_tokens": budget,
                               "phase": phase})
    trace_rows.sort(key=lambda row: float(row["arrival_time"]))
    return trace_rows, request_rows, config


def _grid_state(samples: Mapping[str, list[tuple[int, float, float]]],
                start_ns: int, measurement_sec: float) -> dict[str, Any]:
    interval = 0.05
    grid = np.arange(interval, measurement_sec + 1e-9, interval)
    ordered: dict[str, tuple[list[int], list[tuple[float, float]]]] = {}
    for instance_id, values in samples.items():
        values.sort(key=lambda item: item[0])
        ordered[instance_id] = ([item[0] for item in values], [(item[1], item[2]) for item in values])
    queue_grid: list[float] = []
    kv_grid: list[float] = []
    for relative in grid:
        mono = start_ns + int(relative * 1e9)
        per_queue, per_kv = [], []
        for times, values in ordered.values():
            index = bisect.bisect_right(times, mono) - 1
            if index < 0 and times and times[0] - mono <= int(interval * 1e9):
                index = 0
            if index < 0:
                continue
            queue, kv = values[index]
            per_queue.append(queue)
            per_kv.append(kv)
        if per_queue:
            queue_grid.append(float(np.mean(per_queue)))
        if per_kv:
            kv_grid.append(float(np.mean(per_kv)))
    return {
        "queue_mean_requests": float(np.mean(queue_grid)) if queue_grid else None,
        "queue_p95_requests": float(np.quantile(queue_grid, .95)) if queue_grid else None,
        "kv_mean_fraction": float(np.mean(kv_grid)) if kv_grid else None,
        "kv_p95_fraction": float(np.quantile(kv_grid, .95)) if kv_grid else None,
        "state_sample_interval_sec": interval,
        "queue_samples": len(queue_grid), "kv_samples": len(kv_grid),
        "state_instances": len(ordered),
    }


def _observed(run_dir: Path, request_rows: list[dict[str, str]], config: Mapping[str, Any],
              measurement_sec: float, drain_sec: float) -> dict[str, Any]:
    window = config.get("measurement_window", {})
    start_ns = int(window["start_monotonic_ns_estimate"])
    end_ns = start_ns + int(measurement_sec * 1e9)
    completion_horizon_ns = end_ns + int(drain_sec * 1e9)
    raw_measure = {str(row.get("request_id", "")): row for row in request_rows
                   if row.get("phase") == "measurement"}
    sent_measurement_ids = {
        request_id for request_id, row in raw_measure.items()
        if _num(row.get("actual_send_sec")) is not None
    }
    event_types: Counter[str] = Counter()
    request_received: dict[str, int] = {}
    outputs: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)
    api_output_times: dict[tuple[str, str, int], int] = {}
    engine_output_times: dict[tuple[str, str, int], int] = {}
    engine_output_updates = 0
    output_queue_enqueues: dict[tuple[str, str, int], tuple[int, int]] = {}
    output_queue_dequeues: dict[tuple[str, str, int], int] = {}
    forwarder_stage_times: dict[str, dict[tuple[str, str, int], int]] = defaultdict(dict)
    terminal: dict[str, tuple[str, int, int]] = {}
    idle_samples: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    migration: dict[str, dict[str, Any]] = defaultdict(dict)
    measurement_request_ids = set(raw_measure)
    request_routes: dict[str, str] = {}
    instance_ids: set[str] = set()
    parse_errors = 0
    for path in event_stream_paths(run_dir / "causal_events"):
        with open_event_stream(path) as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                name = str(event.get("event", "unknown"))
                event_types[name] += 1
                request_id = str(event.get("request_id", ""))
                timestamp = event.get("monotonic_ns")
                if name == "request_server_received" and request_id and isinstance(timestamp, int):
                    request_received[request_id] = timestamp
                elif name == "request_output_at_api" and request_id and isinstance(timestamp, int):
                    output_tokens = int(_num(event.get("output_tokens"), 0) or 0)
                    instance_id = str(event.get("instance_id", ""))
                    outputs[request_id].append((timestamp, output_tokens,
                                                int(_num(event.get("new_output_tokens"), 0) or 0),
                                                instance_id))
                    api_output_times[(request_id, instance_id, output_tokens)] = timestamp
                elif name == "engine_request_output" and request_id and isinstance(timestamp, int):
                    if request_id in sent_measurement_ids and timestamp <= completion_horizon_ns:
                        engine_output_updates += 1
                    output_tokens = int(_num(event.get("output_tokens"), 0) or 0)
                    key = (request_id, str(event.get("instance_id", "")), output_tokens)
                    # A request may appear on two instances during migration;
                    # retain the latest matching engine output before joining
                    # it to the API callback below.
                    engine_output_times[key] = max(timestamp, engine_output_times.get(key, -1))
                elif name == "output_queue_enqueue" and request_id and isinstance(timestamp, int):
                    key = (request_id, str(event.get("instance_id", "")),
                           int(_num(event.get("output_tokens"), 0) or 0))
                    depth = int(_num(event.get("queue_depth_messages"), 0) or 0)
                    output_queue_enqueues[key] = (timestamp, depth)
                elif name == "output_queue_dequeue" and request_id and isinstance(timestamp, int):
                    key = (request_id, str(event.get("instance_id", "")),
                           int(_num(event.get("output_tokens"), 0) or 0))
                    output_queue_dequeues[key] = timestamp
                elif name in {"output_forwarder_submit", "output_forwarder_loop_start",
                              "output_forwarder_queue_client_send"} \
                        and request_id and isinstance(timestamp, int):
                    key = (request_id, str(event.get("instance_id", "")),
                           int(_num(event.get("output_tokens"), 0) or 0))
                    forwarder_stage_times[name][key] = timestamp
                elif name == "request_server_terminal" and request_id and isinstance(timestamp, int):
                    terminal[request_id] = (str(event.get("terminal_status", "unknown")),
                                            int(_num(event.get("output_tokens"), 0) or 0), timestamp)
                elif name in {"batch_scheduled", "scheduler_idle"} and isinstance(timestamp, int) \
                        and start_ns <= timestamp <= end_ns:
                    instance_id = str(event.get("instance_id", ""))
                    if instance_id:
                        instance_ids.add(instance_id)
                    if name == "batch_scheduled" and instance_id:
                        for batch_request_id in event.get("request_ids", []) or []:
                            batch_request_id = str(batch_request_id)
                            if batch_request_id in measurement_request_ids:
                                request_routes.setdefault(batch_request_id, instance_id)
                    used = _num(event.get("kv_used_blocks"))
                    free = _num(event.get("kv_free_blocks"))
                    utilization = _num(event.get("kv_utilization"))
                    if used is not None and free is not None and used + free > 0:
                        utilization = used / (used + free)
                    if utilization is not None:
                        idle_samples[str(event.get("instance_id", ""))].append(
                            (timestamp, _num(event.get("num_waiting_requests"), 0.0) or 0.0,
                             utilization))
                elif name.startswith("migration_"):
                    migration_id = str(event.get("migration_id", ""))
                    if migration_id:
                        migration[migration_id][name] = event

    ttft: list[float] = []
    tpot: list[float] = []
    output_tokens_window = 0
    completions_window = 0
    for request_id, rows in outputs.items():
        # Empty RequestOutput updates are not token arrivals and must not be
        # counted as TTFT observations or as the first TPOT interval anchor.
        rows = sorted((row for row in rows if row[1] > 0), key=lambda row: row[0])
        if not rows:
            continue
        first = rows[0]
        receive = request_received.get(request_id)
        if start_ns <= first[0] <= end_ns and receive is not None:
            ttft.append(max(0.0, (first[0] - receive) / 1e6))
        previous = None
        for item in rows:
            timestamp, output_count, new_tokens, _instance_id = item
            if start_ns <= timestamp <= end_ns:
                output_tokens_window += max(0, new_tokens)
            if previous is not None and start_ns < timestamp <= end_ns:
                delta = max(1, output_count - previous[1])
                tpot.extend([(timestamp - previous[0]) / 1e6 / delta] * delta)
            previous = item
    for status, _tokens, timestamp in terminal.values():
        if status == "server_completed" and start_ns <= timestamp <= end_ns:
            completions_window += 1

    qps = _grid_state(idle_samples, start_ns, measurement_sec)
    output_delivery = _output_delivery_diagnostics(
        engine_output_times, outputs, engine_output_updates, sent_measurement_ids,
        request_received, start_ns, end_ns, completion_horizon_ns,
    )
    output_queue_stages = _output_queue_stage_diagnostics(
        engine_output_times, forwarder_stage_times,
        output_queue_enqueues, output_queue_dequeues, api_output_times,
        sent_measurement_ids, completion_horizon_ns,
    )
    cohort = list(raw_measure.values())
    warmup_rows = [row for row in request_rows if row.get("phase") == "warmup"]
    warmup_request_ids = {
        str(row.get("server_request_id") or row.get("request_id") or "")
        for row in warmup_rows
    }
    warmup_request_ids.discard("")
    warmup_received_before_start = {
        request_id for request_id in warmup_request_ids
        if request_received.get(request_id, completion_horizon_ns) < start_ns
    }
    warmup_terminal_before_start = {
        request_id for request_id in warmup_request_ids
        if request_id in terminal and terminal[request_id][2] < start_ns
    }
    warmup_active_at_start = warmup_received_before_start - warmup_terminal_before_start
    warmup_unresolved_at_start = warmup_request_ids - warmup_terminal_before_start
    warmup_completed = sum(str(row.get("status_code", "")) == "200" for row in warmup_rows)
    warmup_censored = sum(str(row.get("status_code", "")) == "499" for row in warmup_rows)
    warmup_timeouts = sum(str(row.get("timeout", "")).lower() == "true" for row in warmup_rows)
    finished = censored = timeouts = failed = unfinished = 0
    completed_by_horizon = 0
    terminal_joined = 0
    for row in cohort:
        rid = str(row.get("server_request_id") or row.get("request_id") or "")
        record = terminal.get(rid)
        terminal_joined += int(record is not None)
        status_code = str(row.get("status_code", ""))
        if record and record[2] <= completion_horizon_ns and record[0] == "server_completed":
            finished += 1
            completed_by_horizon += 1
        elif status_code == "499":
            censored += 1
        elif str(row.get("timeout", "")).lower() == "true":
            timeouts += 1
        elif status_code and status_code not in {"200", "499"}:
            failed += 1
        else:
            unfinished += 1
    client_e2e = [_num(row.get("client_e2e_ms")) for row in cohort
                  if str(row.get("success", "")).lower() == "true"]
    client_e2e = [value for value in client_e2e if value is not None]

    migration_rows = []
    warmup_migrations_started = 0
    warmup_migrations_inflight = 0
    for migration_id, events in migration.items():
        started = events.get("migration_attempt_started")
        completed = events.get("migration_attempt_completed")
        pause = events.get("migration_request_pause_started")
        resumed = events.get("migration_request_resumed")
        if not started:
            continue
        start = started.get("monotonic_ns")
        end = completed.get("monotonic_ns") if completed else None
        if isinstance(start, int) and start < start_ns:
            warmup_migrations_started += 1
            if not isinstance(end, int) or end > start_ns:
                warmup_migrations_inflight += 1
        if not completed:
            continue
        if not isinstance(start, int) or not isinstance(end, int) or not start_ns <= start <= end_ns:
            continue
        row = {"migration_id": migration_id, "status": str(completed.get("status", "unknown")),
               "blocks": _num(completed.get("total_blocks")),
               "total_ms": _num(completed.get("migration_total_ms"), (end - start) / 1e6),
               "migration_time_sec": (start - start_ns) / 1e9,
               "request_id": str(started.get("request_id", "")),
               "source_instance": str(started.get("source_instance_id", "")),
               "target_instance": str(started.get("target_instance_id", ""))}
        if pause and resumed and isinstance(pause.get("monotonic_ns"), int) and isinstance(resumed.get("monotonic_ns"), int):
            pause_ns, resume_ns = pause["monotonic_ns"], resumed["monotonic_ns"]
            request_id = str(started.get("request_id", ""))
            token_times = [time for time, _, _, _instance_id in outputs.get(request_id, [])]
            before = [time for time in token_times if time < pause_ns]
            after = [time for time in token_times if time > resume_ns]
            row["pause_stage_ms"] = (resume_ns - pause_ns) / 1e6
            row["recovery_after_resume_ms"] = (after[0] - resume_ns) / 1e6 if after else None
            row["request_token_gap_ms"] = (after[0] - before[-1]) / 1e6 if before and after else None
        migration_rows.append(row)
    migration_total = [row["total_ms"] for row in migration_rows if row.get("total_ms") is not None]
    migration_pause = [row["pause_stage_ms"] for row in migration_rows if row.get("pause_stage_ms") is not None]
    migration_recovery = [row["recovery_after_resume_ms"] for row in migration_rows
                          if row.get("recovery_after_resume_ms") is not None]
    gpu_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(run_dir / "gpu_metrics.csv"):
        if row.get("phase") != "measurement" or row.get("gpu_index") in {None, "", "None"}:
            continue
        gpu_samples[str(row["gpu_index"])].append(row)
    gpu_summary = {}
    for gpu_index, samples in sorted(gpu_samples.items()):
        utilization = [_num(row.get("gpu_utilization_pct")) for row in samples]
        memory = [_num(row.get("memory_used_mib")) for row in samples]
        temperature = [_num(row.get("temperature_c")) for row in samples]
        gpu_summary[gpu_index] = {
            "samples": len(samples),
            "gpu_name": samples[0].get("gpu_name"),
            "mean_utilization_pct": float(np.mean([value for value in utilization if value is not None]))
            if any(value is not None for value in utilization) else None,
            "mean_memory_used_mib": float(np.mean([value for value in memory if value is not None]))
            if any(value is not None for value in memory) else None,
            "mean_temperature_c": float(np.mean([value for value in temperature if value is not None]))
            if any(value is not None for value in temperature) else None,
        }
    return {
        "measurement_requests": len(cohort),
        "warmup_requests": len(warmup_rows),
        "warmup_http_200": warmup_completed,
        "warmup_http_499": warmup_censored,
        "warmup_timeouts": warmup_timeouts,
        "warmup_server_requests": len(warmup_request_ids),
        "warmup_terminal_before_measurement": len(warmup_terminal_before_start),
        "warmup_active_at_measurement_start": len(warmup_active_at_start),
        "warmup_unresolved_at_measurement_start": len(warmup_unresolved_at_start),
        "warmup_phase_cleared_before_measurement": len(warmup_active_at_start) == 0,
        "warmup_migrations_started": warmup_migrations_started,
        "warmup_migrations_inflight_at_measurement_start": warmup_migrations_inflight,
        "terminal_events_joined_to_measurement_requests": terminal_joined,
        "terminal_event_missing_requests": max(0, len(cohort) - terminal_joined),
        "all_measurement_requests_sent": bool(config.get("all_measurement_requests_sent", False)),
        "finished_by_drain_horizon": finished,
        "failed": failed, "censored_http_499": censored,
        "timeouts": timeouts, "unfinished": unfinished,
        "completion_count_window": completions_window,
        "request_throughput_rps": completions_window / measurement_sec,
        "output_tokens_window": output_tokens_window,
        "output_tokens_per_sec": output_tokens_window / measurement_sec,
        "ttft_mean_ms": float(np.mean(ttft)) if ttft else None,
        "ttft_p95_ms": float(np.quantile(ttft, .95)) if ttft else None,
        "ttft_samples": len(ttft),
        "tpot_mean_ms": float(np.mean(tpot)) if tpot else None,
        "tpot_p95_ms": float(np.quantile(tpot, .95)) if tpot else None,
        "tpot_samples": len(tpot),
        "output_delivery_diagnostics": output_delivery,
        "output_queue_stage_diagnostics": output_queue_stages,
        "client_e2e_mean_ms": float(np.mean(client_e2e)) if client_e2e else None,
        **qps,
        "migration_attempts": len(migration_rows),
        "migration_successes": sum("FINISHED" in str(row["status"]) for row in migration_rows),
        "migration_failures": sum("FAIL" in str(row["status"]).upper() for row in migration_rows),
        "migration_aborted": sum("ABORT" in str(row["status"]).upper() for row in migration_rows),
        "migration_total_mean_ms": float(np.mean(migration_total)) if migration_total else None,
        "migration_pause_stage_mean_ms": float(np.mean(migration_pause)) if migration_pause else None,
        "migration_recovery_mean_ms": float(np.mean(migration_recovery)) if migration_recovery else None,
        "migration_request_token_gap_mean_ms": float(np.mean([
            row["request_token_gap_ms"] for row in migration_rows
            if row.get("request_token_gap_ms") is not None])) if any(
                row.get("request_token_gap_ms") is not None for row in migration_rows) else None,
        "migration_recovery_mean_ms": float(np.mean(migration_recovery)) if migration_recovery else None,
        "migration_rows": migration_rows,
        "fixed_replay_inputs": {
            "request_routes": request_routes if len(instance_ids) == 4 else {},
            "instance_ids": sorted(instance_ids) if len(instance_ids) == 4 else [],
            "migration_events": [
                {"migration_id": row["migration_id"],
                 "migration_time_sec": row["migration_time_sec"],
                 "request_id": row["request_id"],
                 "source_instance": row["source_instance"],
                 "target_instance": row["target_instance"],
                 "blocks": row["blocks"], "cost_ms": row["total_ms"],
                 "pause_stage_ms": row.get("pause_stage_ms", row["total_ms"]),
                 "recovery_after_resume_ms": row.get("recovery_after_resume_ms", 0.0),
                 "status": row["status"]}
                for row in migration_rows
                if row["request_id"] and row["source_instance"] and row["target_instance"]
            ],
        },
        "event_integrity": {
            "event_type_counts": dict(event_types), "json_parse_errors": parse_errors,
            "process_sequence_drop_audit": "requires unsplit source streams; filtered shard gaps are not interpreted as drops",
            "expected_measurement_window_monotonic_ns": [start_ns, end_ns],
            "actual_server_terminal_count": len(terminal),
        },
        "sampling_mode": str(config.get("mode", "unknown")),
        "gpu_metrics": gpu_summary,
    }


def _routine_balanced_action(env: LlumnixMigrationEnv,
                            policy: Mapping[str, Any] | None) -> int:
    """Mirror Llumnix's neutral-instance RemainingStepsLoad/Balanced routine."""
    if not policy or not policy.get("enabled") or str(policy.get("migration_load_metric")) != "remaining_steps":
        return 0
    if str(policy.get("pair_migration_policy", "")).lower() != "balanced":
        return 0
    obs = env._last_obs
    if obs is None or env.pending_migration is not None:
        return 0
    threshold = float(policy.get("migrate_out_threshold", 9000.0))
    views = env._instance_views()

    def numeric_load(instance_id: str) -> float:
        return float(views[instance_id].remaining_steps.remaining_steps)

    source_ids = [instance.instance_id for instance in env.instances
                  if instance.preempted_requests > 0 or numeric_load(instance.instance_id) <= threshold]
    destination_ids = [instance.instance_id for instance in env.instances
                       if instance.preempted_requests == 0 and numeric_load(instance.instance_id) >= threshold]
    # RemainingStepsLoad deliberately reverses comparison order: Llumnix's
    # reverse=True source sort is numeric ascending; destination sort is
    # numeric descending.
    source_ids.sort(key=numeric_load)
    destination_ids.sort(key=numeric_load, reverse=True)
    legal_pairs = {pair for pair, legal in zip(obs.action_mask.pairs, obs.pair_mask) if legal}

    def remaining_after(instance, blocks_delta: int, request_delta: int) -> float:
        waiting_blocks = sum(env._blocks_for_tokens(request.request_len) for request in instance.waiting)
        request_count = len(instance.running) + len(instance.waiting) + request_delta
        if request_count <= 0:
            return math.inf
        available = instance.free_blocks + blocks_delta - waiting_blocks
        return float(available / request_count)

    for source_id, destination_id in zip(source_ids, destination_ids):
        if source_id == destination_id:
            continue
        source = next(item for item in env.instances if item.instance_id == source_id)
        destination = next(item for item in env.instances if item.instance_id == destination_id)
        candidate = env._sr_candidate(source)
        if candidate is None:
            continue
        pair = (source_id, destination_id)
        if pair not in legal_pairs:
            continue

        # llumnix's Balanced policy computes the hypothetical load using each
        # instance's last running request block count, then migrates the SR
        # candidate. Keep those two server semantics separate here as well.
        source_last_blocks = (env._blocks_for_tokens(source.running[-1].request_len)
                              if source.running else 0)
        destination_last_blocks = (env._blocks_for_tokens(destination.running[-1].request_len)
                                   if destination.running else 0)
        source_after = remaining_after(source, source_last_blocks, -1)
        destination_after = remaining_after(destination, -destination_last_blocks, 1)
        # RemainingStepsLoad's comparison operators are reversed, so the
        # server's `destination_after > threshold` means numeric > threshold,
        # while `source_after > destination_after` means numeric <=.
        if destination_after <= threshold or source_after > destination_after:
            continue
        before_difference = abs(numeric_load(source_id) - numeric_load(destination_id))
        after_difference = abs(source_after - destination_after)
        if 0.0 < after_difference < before_difference:
            return obs.action_mask.pairs.index(pair) + 1
    return 0


def _runtime_migration_policy(on_dir: Path) -> dict[str, Any]:
    """Read the actual routine migration policy used by the ON benchmark run."""
    path = on_dir / "configs" / "llumnix_on.yml"
    if not path.exists():
        raise FileNotFoundError(f"ON run config required to reproduce runtime migration policy: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    manager, instance = config.get("MANAGER", {}), config.get("INSTANCE", {})
    return {
        "enabled": bool(manager.get("ENABLE_ROUTINE_MIGRATION", False)),
        "pair_migration_policy": str(manager.get("PAIR_MIGRATION_POLICY", "")),
        "pair_migration_frequency": max(1, int(manager.get("PAIR_MIGRATION_FREQUENCY", 1))),
        "polling_interval_sec": max(0.001, float(manager.get("POLLING_INTERVAL", 0.05))),
        "migrate_out_threshold": float(manager.get("MIGRATE_OUT_THRESHOLD", 9000.0)),
        "migration_load_metric": str(instance.get("MIGRATION_LOAD_METRIC", "remaining_steps")),
        "max_migration_concurrency": max(1, int(instance.get("MAX_MIGRATION_CONCURRENCY", 1))),
        "source_config_sha256": _hash(path),
    }


def _fixed_batch_replay(run_dir: Path, run_config: Mapping[str, Any],
                        causal_batch_model: Mapping[str, Any],
                        historical: MigPPOEnvConfig) -> dict[str, Any]:
    """Score service models on the held-out, observed batch schedule."""
    window = run_config.get("measurement_window", {})
    start = _num(window.get("start_monotonic_ns_estimate"), math.nan)
    end = _num(window.get("end_monotonic_ns_estimate"), math.nan)
    observed_by_type: dict[str, list[float]] = defaultdict(list)
    round2_by_type: dict[str, list[float]] = defaultdict(list)
    causal_by_type: dict[str, list[float]] = defaultdict(list)
    iteration_events: list[dict[str, Any]] = []
    engine_steps: dict[str, float] = {}
    for path in event_stream_paths(run_dir / "causal_events"):
        with open_event_stream(path) as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                timestamp = event.get("monotonic_ns")
                if not isinstance(timestamp, int) or not math.isfinite(start) \
                        or not math.isfinite(end) or not start <= timestamp <= end:
                    continue
                if event.get("event") == "llm_engine_step_completed":
                    batch_id = str(event.get("batch_id", ""))
                    value = _num(event.get("engine_step_wall_ms"))
                    if batch_id and value is not None and value > 0.0:
                        engine_steps[batch_id] = value
                elif event.get("event") == "engine_iteration_completed" \
                        and event.get("succeeded", True):
                    iteration_events.append(event)

    for event in iteration_events:
        target = engine_steps.get(str(event.get("batch_id", "")))
        if target is None:
            # Older audit fixtures have only the executor record.
            target = _num(event.get("engine_step_wall_ms"), _num(event.get("executor_wall_ms")))
        if target is None or target <= 0:
            continue
        features = _batch_features(event)
        batch_type = str(event.get("batch_type") or (
            "mixed" if features["is_mixed"] else
            "prefill" if features["is_prefill"] else "decode"))
        causal_prediction = max(historical.min_engine_step_ms,
                                predict_service_model(causal_batch_model, features))

        prefill_prediction = 0.0
        decode_prediction = 0.0
        model = historical.service_time_model or {}
        if features["num_prefill_requests"] > 0 and model.get("prefill"):
            prefill_count = max(1.0, features["num_prefill_requests"])
            prefill_prediction = predict_service_model(model["prefill"], {
                "batch_tokens": features["prefill_tokens"],
                "total_context_tokens": features["prefill_tokens"],
                "batch_size": prefill_count,
                "running_requests": features["num_running_sequences"],
                "avg_context_tokens": features["prefill_tokens"] / prefill_count,
            })
        if features["num_decode_requests"] > 0 and model.get("decode"):
            decode_context = 0.0
            for group in event.get("scheduled_groups", []) or []:
                if not bool(group.get("is_prompt", False)):
                    decode_context += _num(group.get("context_tokens"), 0.0) or 0.0
            if decode_context <= 0 and batch_type == "decode":
                decode_context = features["total_context_tokens"]
            decode_count = max(1.0, features["num_decode_requests"])
            decode_prediction = predict_service_model(model["decode"], {
                "batch_size": decode_count,
                "total_context_tokens": decode_context,
                "avg_context_tokens": decode_context / decode_count,
                "batch_tokens": features["decode_tokens"] or decode_count,
                "running_requests": features["num_running_sequences"],
            }) * historical.decode_token_interval_scale
        round2_prediction = max(historical.min_engine_step_ms,
                                prefill_prediction, decode_prediction)
        observed_by_type[batch_type].append(target)
        round2_by_type[batch_type].append(round2_prediction)
        causal_by_type[batch_type].append(causal_prediction)
    def summarize(target: list[float], prediction: list[float]) -> dict[str, Any]:
        if not target:
            return {"n": 0, "mae_ms": None, "rmse_ms": None, "mape": None, "bias_ms": None}
        y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
        error = p - y
        return {"n": int(len(y)), "mae_ms": float(np.mean(np.abs(error))),
                "rmse_ms": float(np.sqrt(np.mean(error ** 2))),
                "mape": float(np.mean(np.abs(error) / np.maximum(np.abs(y), 1e-9))),
                "bias_ms": float(np.mean(error))}

    per_type = {}
    for batch_type in sorted(observed_by_type):
        target = observed_by_type[batch_type]
        per_type[batch_type] = {
            "observed_samples": len(target),
            "round2_v3": summarize(target, round2_by_type[batch_type]),
            "causal_rebuild": summarize(target, causal_by_type[batch_type]),
        }
    all_targets = [value for values in observed_by_type.values() for value in values]
    all_round2 = [value for values in round2_by_type.values() for value in values]
    all_causal = [value for values in causal_by_type.values() for value in values]
    return {"protocol": "held-out batch schedule and features are fixed; target is joined full llm_engine_step_completed wall time",
            "overall": {"round2_v3": summarize(all_targets, all_round2),
                        "causal_rebuild": summarize(all_targets, all_causal)},
            "by_batch_type": per_type}


def _simulated(run_trace: list[dict[str, Any]], measurement_ids: set[str], config: MigPPOEnvConfig,
               qps: float, seed: int, warmup_sec: float, measurement_sec: float,
               drain_sec: float, migration_policy: Mapping[str, Any] | None = None,
               fixed_replay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    # execute_trace drains warmup requests and aborts remaining requests before
    # it starts the measurement phase. Start with an empty queue at measurement
    # time zero; warmup arrivals must not leak into the measured request cohort.
    measurement_trace = [row for row in run_trace if row.get("phase") == "measurement"]
    total = measurement_sec + drain_sec
    interval = config.decision_interval_sec
    steps = int(math.ceil(total / interval))
    sim_config = replace(config, max_decisions=steps, decision_interval_sec=interval,
                         workload_scenario=f"qps_{qps:g}", target_qps_levels=(qps,),
                         request_timeout_sec=None, trace_events_enabled=True,
                         trace_token_events=True, state_sample_interval_sec=0.05,
                         validate_state_invariants=True)
    env = LlumnixMigrationEnv(sim_config)
    reset_options: dict[str, Any] = {"target_qps": qps, "arrival_trace": measurement_trace,
                                     "start_time_sec": 0.0}
    if fixed_replay:
        instance_ids = fixed_replay.get("instance_ids") or []
        routes = fixed_replay.get("request_routes") or {}
        if len(instance_ids) == config.num_instances and routes:
            reset_options["instance_ids"] = instance_ids
            reset_options["request_routes"] = routes
        reset_options["migration_event_trace"] = fixed_replay.get("migration_events", [])
    env.reset(seed=seed, options=reset_options)
    polling_sec = float((migration_policy or {}).get("polling_interval_sec", interval))
    controller_steps = max(1, int(round(polling_sec / interval)))
    migration_frequency = max(1, int((migration_policy or {}).get("pair_migration_frequency", 1)))
    for step in range(steps):
        polling_index = step // controller_steps
        routine_tick = step % controller_steps == 0 and polling_index % migration_frequency == 0
        action = (_routine_balanced_action(env, migration_policy)
                  if migration_policy and routine_tick else 0)
        env.step(action)
    events = env.lifecycle_events
    first_tokens = [event for event in events if event.get("event") == "first_token"
                    and 0.0 < float(event.get("timestamp_sec", -math.inf)) <= measurement_sec]
    later_tokens = [event for event in events if event.get("event") == "decode_token"
                    and 0.0 < float(event.get("timestamp_sec", -math.inf)) <= measurement_sec]
    completions = [event for event in events if event.get("event") == "completed"
                   and 0.0 <= float(event.get("timestamp_sec", math.inf)) <= measurement_sec]
    token_values = [float(event.get("tpot_ms", 0.0)) for event in later_tokens]
    state = [row for row in env.state_samples if 0.0 < row["time_sec"] <= measurement_sec]
    queue = [row["queue_mean_requests"] for row in state]
    kv = [row["kv_mean_fraction"] for row in state]
    results = {str(row["request_id"]): row for row in env.request_results}
    finished_cohort = sum(1 for request_id in measurement_ids
                          if results.get(request_id, {}).get("status") == "completed")
    failed_cohort = sum(1 for request_id in measurement_ids
                        if results.get(request_id, {}).get("status") == "timeout")
    unfinished_cohort = max(0, len(measurement_ids) - finished_cohort - failed_cohort)
    return {
        "measurement_requests": len(measurement_ids),
        "finished_by_drain_horizon": finished_cohort,
        "failed": failed_cohort, "censored_or_unfinished": unfinished_cohort,
        "completion_count_window": len(completions),
        "request_throughput_rps": len(completions) / measurement_sec,
        "output_tokens_window": len(first_tokens) + len(later_tokens),
        "output_tokens_per_sec": (len(first_tokens) + len(later_tokens)) / measurement_sec,
        "ttft_mean_ms": float(np.mean([float(event["ttft_ms"]) for event in first_tokens])) if first_tokens else None,
        "ttft_p95_ms": float(np.quantile([float(event["ttft_ms"]) for event in first_tokens], .95)) if first_tokens else None,
        "ttft_samples": len(first_tokens),
        "tpot_mean_ms": float(np.mean(token_values)) if token_values else None,
        "tpot_p95_ms": float(np.quantile(token_values, .95)) if token_values else None,
        "tpot_samples": len(token_values),
        "queue_mean_requests": float(np.mean(queue)) if queue else None,
        "queue_p95_requests": float(np.quantile(queue, .95)) if queue else None,
        "kv_mean_fraction": float(np.mean(kv)) if kv else None,
        "kv_p95_fraction": float(np.quantile(kv, .95)) if kv else None,
        "state_samples": len(state),
        "migration_attempts": len(env.migration_events),
        "migration_successes": sum(event.get("status") == "success" for event in env.migration_events),
        "migration_failures": sum(event.get("status") == "failed" for event in env.migration_events),
        "migration_aborted": sum(event.get("status") == "aborted" for event in env.migration_events),
        "migration_total_mean_ms": float(np.mean([float(event["cost_ms"]) for event in env.migration_events]))
        if env.migration_events else None,
        "migration_pause_stage_mean_ms": float(np.mean([float(event["pause_stage_ms"])
                                                          for event in env.migration_events]))
        if env.migration_events else None,
        "migration_recovery_mean_ms": float(np.mean([
            float(event.get("predicted_recovery_after_resume_ms", 0.0))
            for event in env.migration_events])) if env.migration_events else None,
        "migration_events": env.migration_events,
        "fixed_replay_results": env.migration_replay_results,
        "fixed_route_coverage": len(reset_options.get("request_routes", {})) / max(1, len(measurement_ids)),
        "service_batch_count": sum(row.get("phase") == "batch" for row in env.service_iterations),
        "unfinished_request_count_at_horizon": len(env.requests),
    }


def _error(predicted: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("request_throughput_rps", "output_tokens_per_sec", "ttft_mean_ms", "ttft_p95_ms",
              "tpot_mean_ms", "tpot_p95_ms", "queue_mean_requests", "queue_p95_requests",
              "kv_mean_fraction", "kv_p95_fraction", "migration_total_mean_ms",
              "migration_pause_stage_mean_ms", "migration_recovery_mean_ms")
    out: dict[str, Any] = {}
    for field in fields:
        pred, actual = _num(predicted.get(field)), _num(observed.get(field))
        if pred is None or actual is None:
            continue
        delta = pred - actual
        out[field] = {"predicted": pred, "observed": actual, "error": delta,
                      "absolute_error": abs(delta),
                      "absolute_percentage_error": abs(delta) / abs(actual) if abs(actual) > 1e-9 else None,
                      "kv_error_percentage_points": delta * 100.0 if field.startswith("kv_") else None}
    return out


def _summary(rows: list[dict[str, Any]], versions: list[str]) -> dict[str, Any]:
    fields = ("request_throughput_rps", "output_tokens_per_sec", "ttft_mean_ms", "ttft_p95_ms",
              "tpot_mean_ms", "tpot_p95_ms", "queue_mean_requests", "queue_p95_requests",
              "kv_mean_fraction", "kv_p95_fraction", "migration_total_mean_ms",
              "migration_pause_stage_mean_ms", "migration_recovery_mean_ms")
    result: dict[str, Any] = {}
    for version in versions:
        result[version] = {}
        for field in fields:
            errors = [row["versions"][version]["error"][field]
                      for row in rows if field in row["versions"][version]["error"]]
            if not errors:
                continue
            ape = [value["absolute_percentage_error"] for value in errors
                   if value.get("absolute_percentage_error") is not None]
            result[version][field] = {
                "n": len(errors), "macro_mape": float(np.mean(ape)) if ape else None,
                "macro_mae": float(np.mean([value["absolute_error"] for value in errors])),
                "signed_bias": float(np.mean([value["error"] for value in errors])),
                "macro_mae_pp": float(np.mean([abs(value["kv_error_percentage_points"]) for value in errors
                                               if value.get("kv_error_percentage_points") is not None]))
                if field.startswith("kv_") else None,
            }
    return result


def _acceptance_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    definitions = {
        "request_throughput_rps": ("mape", 0.20),
        "ttft_mean_ms": ("mape", 0.25),
        "ttft_p95_ms": ("mape", 0.30),
        "tpot_mean_ms": ("mape", 0.20),
        "tpot_p95_ms": ("mape", 0.25),
        "queue_mean_requests": ("relative_mae", 0.20),
        "queue_p95_requests": ("relative_mae", 0.20),
        "kv_mean_fraction": ("absolute_pp", 5.0),
        "kv_p95_fraction": ("absolute_pp", 10.0),
    }
    checks: dict[str, Any] = {}
    for metric, (kind, threshold) in definitions.items():
        values = []
        passed = True
        for row in rows:
            sim = row["versions"]["causal_rebuild"]["simulation"]
            observed = row["observed"]
            pred, actual = _num(sim.get(metric)), _num(observed.get(metric))
            if pred is None or actual is None:
                passed = False
                continue
            error = abs(pred - actual)
            if kind == "mape":
                score = error / abs(actual) if abs(actual) > 1e-9 else (0.0 if error <= 1e-9 else math.inf)
                limit = threshold
            elif kind == "relative_mae":
                score = error
                limit = max(0.5 if metric == "queue_mean_requests" else 1.0,
                            threshold * abs(actual))
            else:
                score = error * 100.0
                limit = threshold
            values.append({"mode": row["mode"], "qps": row["qps"], "observed": actual,
                           "predicted": pred, "error_metric": score, "limit": limit,
                           "passed": score <= limit})
            passed = passed and score <= limit
        checks[metric] = {"kind": kind, "threshold": threshold,
                          "passed": bool(passed and values), "run_checks": values,
                          "worst_error": max(values, key=lambda item: item["error_metric"])
                          if values else None}
    lifecycle_ok = all(
        row["observed"]["all_measurement_requests_sent"]
        and row["observed"]["terminal_event_missing_requests"] == 0
        and row["observed"]["warmup_phase_cleared_before_measurement"]
        and row["observed"]["warmup_migrations_inflight_at_measurement_start"] == 0
        and row["observed"]["event_integrity"]["json_parse_errors"] == 0
        for row in rows
    )
    checks["request_lifecycle_and_collection"] = {
        "passed": lifecycle_ok,
        "all_sent": all(row["observed"]["all_measurement_requests_sent"] for row in rows),
        "zero_missing_terminal_events": all(
            row["observed"]["terminal_event_missing_requests"] == 0 for row in rows),
        "zero_warmup_requests_active_at_measurement_start": all(
            row["observed"]["warmup_phase_cleared_before_measurement"] for row in rows),
        "zero_warmup_migrations_in_flight": all(
            row["observed"]["warmup_migrations_inflight_at_measurement_start"] == 0 for row in rows),
        "zero_json_parse_errors": all(
            row["observed"]["event_integrity"]["json_parse_errors"] == 0 for row in rows),
    }
    return {"passed": bool(checks) and all(check["passed"] for check in checks.values()),
            "checks": checks}


def validate(off_dir: Path, on_dir: Path, model_dir: Path, output_dir: Path,
             round2_config_path: Path, validation_seed: int,
             warmup_sec: float, measurement_sec: float, drain_sec: float) -> dict[str, Any]:
    service_model = json.loads((model_dir / "causal_service_model.json").read_text(encoding="utf-8"))
    output_model = json.loads((model_dir / "output_length_model.json").read_text(encoding="utf-8"))
    token_cadence_model = json.loads((model_dir / "token_cadence_model.json").read_text(encoding="utf-8"))
    migration_model = json.loads((model_dir / "migration_model.json").read_text(encoding="utf-8"))
    training_seeds = [int(seed) for seed in service_model.get("training_seeds", [606])]
    historical = MigPPOEnvConfig.from_yaml(round2_config_path)
    total_component = migration_model.get("components", {}).get("total_ms", {})
    migration_cost_state = {"global": {
        "alpha_ms": float(total_component.get("intercept_ms", historical.migration_alpha_ms)),
        "beta_ms_per_block": float(total_component.get("ms_per_block", historical.migration_beta_ms_per_block)),
        "residual_sigma_ms": float(total_component.get("uncertainty_ms", 0.0) or 0.0),
        "samples": int(total_component.get("samples", 0)),
    }}
    runtime_migration_policy = _runtime_migration_policy(on_dir)
    causal_config = replace(
        historical,
        decision_interval_sec=0.05,
        max_num_batched_tokens=2112,
        service_time_model={"batch_execution": service_model["batch_execution"]},
        service_model_mode="calibrated",
        output_length_model=output_model,
        decode_token_interval_model=token_cadence_model,
        decode_token_interval_scale=1.0,
        migration_cost_estimator=migration_cost_state,
        migration_pause_model=migration_model.get("components", {}),
        migration_uncertainty_scale=0.0,
        request_timeout_sec=None,
        state_sample_interval_sec=0.05,
        validate_state_invariants=True,
        calibration_status=f"causal_model_seeds_{'_'.join(map(str, training_seeds))}_holdout_{validation_seed}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run: list[dict[str, Any]] = []
    for mode, data_dir in (("off", off_dir), ("on", on_dir)):
        for config_path in sorted((data_dir / "validation").glob(f"qps_*_seed{validation_seed}/config.json")):
            run_dir = config_path.parent
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
            qps = float(run_config["qps"])
            trace, request_rows, run_config = _load_run_inputs(run_dir, warmup_sec)
            if not trace:
                raise ValueError(f"No held-out arrival trace recovered from {run_dir}")
            observed = _observed(run_dir, request_rows, run_config, measurement_sec, drain_sec)
            fixed_batch_replay = _fixed_batch_replay(
                run_dir, run_config, service_model["batch_execution"], historical)
            measurement_ids = {str(row.get("request_id", "")) for row in request_rows
                               if row.get("phase") == "measurement"}
            run_versions = {}
            for version, base_config in (("round2_v3", historical),
                                         ("causal_rebuild", causal_config),
                                         ("causal_fixed_replay", causal_config)):
                config = replace(base_config,
                                 state_sample_interval_sec=0.05,
                                 validate_state_invariants=True)
                simulation = _simulated(
                    trace, measurement_ids, config, qps, validation_seed,
                    warmup_sec, measurement_sec, drain_sec,
                    migration_policy=(runtime_migration_policy
                                      if mode == "on" and version == "causal_rebuild" else None),
                    fixed_replay=(observed["fixed_replay_inputs"]
                                  if version == "causal_fixed_replay" else None),
                )
                run_versions[version] = {"simulation": simulation,
                                         "error": _error(simulation, observed)}
            event_inventory = [{"path": str(path.relative_to(run_dir)),
                                "bytes": path.stat().st_size, "sha256": _hash(path)}
                               for path in event_stream_paths(run_dir / "causal_events")]
            per_run.append({
                "mode": mode, "qps": qps, "seed": validation_seed,
                "trace_sha256": _hash(run_dir / "traces" / f"trace_{run_dir.name}.csv"),
                "trace_request_count": len(trace), "measurement_request_count": len(measurement_ids),
                "source_config_sha256": _hash(config_path),
                "measurement_window": run_config.get("measurement_window"),
                "observed": observed, "fixed_batch_replay": fixed_batch_replay,
                "versions": run_versions,
                "event_inventory": event_inventory,
                "prediction_protocol": "measurement arrivals, prompt lengths, and budgets only; warmup requests are drained or aborted before measurement and are not carried into the measurement state; no held-out output length, server state, event schedule, or response time is supplied to simulation",
            })
    versions = ["round2_v3", "causal_rebuild", "causal_fixed_replay"]
    summary = _summary(per_run, versions)
    acceptance = _acceptance_gates(per_run)
    result = {
        "data_dirs": {"off": str(off_dir), "on": str(on_dir)},
        "training_seed": training_seeds[0] if len(training_seeds) == 1 else training_seeds,
        "training_seeds": training_seeds, "validation_seed": validation_seed,
        "split_unit": f"complete run seed; calibration seeds {training_seeds} train, seed {validation_seed} held out",
        "measurement_sec": measurement_sec, "drain_sec": drain_sec, "warmup_sec": warmup_sec,
        "metric_definitions": {
            "request_throughput_rps": "server-completed requests in the measurement window divided by measurement seconds",
            "output_tokens_per_sec": "API-visible generated output tokens in the measurement window divided by measurement seconds",
            "ttft": "first API-visible token minus server receive time, filtered by first token time in the measurement window",
            "tpot": "API-visible per-token intervals whose ending token is in the measurement window",
            "queue_and_kv": "per-instance 50 ms last-observation carry-forward; instance means at each grid point",
            "censored": "HTTP 499 is a drain-censored request, not a server failure",
            "migration_total": "attempt start to terminal result",
            "migration_pause_stage": "source pause start to request resumed at target",
            "migration_recovery": "request resumed to first API-visible token after resume",
        },
        "versions": {
            "round2_v3": "published Round 2 calibration replayed on these held-out measurement traces with the same empty phase-boundary state",
            "causal_rebuild": f"causal batch, Kaplan-Meier output, and migration models trained on seeds {training_seeds}; closed-loop Llumnix routine Balanced/RemainingStepsLoad policy when mode is ON",
            "causal_fixed_replay": "causal rebuild using held-out observed first-batch routes and migration start events; diagnostic replay only, not a prediction eligible for acceptance",
        },
        "routine_migration_policy": runtime_migration_policy,
        "aggregate_errors": summary,
        "independent_accuracy_gate": acceptance,
        "per_run": per_run,
        "acceptance_targets": {
            "throughput_mape": 0.20, "ttft_mean_mape": 0.25, "ttft_p95_mape": 0.30,
            "tpot_mean_mape": 0.20, "tpot_p95_mape": 0.25,
            "queue_mean_mae": "max(0.5, 20% of observed mean)",
            "queue_p95_mae": "max(1, 20% of observed P95)",
            "kv_mean_error_pp": 5.0, "kv_p95_error_pp": 10.0,
        },
        "passed_independent_accuracy_gate": acceptance["passed"],
    }
    (output_dir / "system_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_plots(output_dir / "plots", result)
    _write_report(output_dir / "system_validation.md", result)
    return result


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = ["# Causal system validation", "",
             f"Training seed(s): {result.get('training_seeds', result['training_seed'])}; held-out seed: {result['validation_seed']}. ",
             f"Measurement/drain: {result['measurement_sec']:.0f}s / {result['drain_sec']:.0f}s. ",
             "The predictive simulator receives held-out measurement arrivals, prompt lengths, and budgets only.", "",
             "## Held-out error summary", "",
             "| Version | Metric | Macro MAPE | Macro MAE | Signed bias |", "|:---|:---|---:|---:|---:|"]
    for version, metrics in result["aggregate_errors"].items():
        for field, item in metrics.items():
            mape = item.get("macro_mape")
            lines.append(f"| {version} | {field} | {'n/a' if mape is None else f'{mape * 100:.2f}%'} | "
                         f"{item.get('macro_mae', float('nan')):.3f} | {item.get('signed_bias', float('nan')):.3f} |")
    lines.extend(["", "## Warmup phase boundary", "",
                  "| Mode | QPS | HTTP 200 | HTTP 499 | Migrations started | In flight at measurement start |",
                  "|:---:|---:|---:|---:|---:|---:|"])
    for run in result["per_run"]:
        observed = run["observed"]
        lines.append(f"| {run['mode']} | {run['qps']:g} | {observed['warmup_http_200']} | "
                     f"{observed['warmup_http_499']} | {observed['warmup_migrations_started']} | "
                     f"{observed['warmup_migrations_inflight_at_measurement_start']} |")
    lines.extend(["", "## Output delivery and first-token coverage", "",
                  "Engine-to-API callback lag is measured by joining request ID and cumulative output-token count. "
                  "It is an observational diagnostic and is not supplied to the simulator.", "",
                  "| Mode | QPS | Engine updates | API updates | Matched | First-output lag p50/p95 ms | First-token events in window / sent | Right-censored at window |",
                  "|:---:|---:|---:|---:|---:|---:|---:|---:|"])
    for run in result["per_run"]:
        delivery = run["observed"].get("output_delivery_diagnostics", {})
        first = delivery.get("first_output_engine_to_api_delay", {})
        lines.append(
            f"| {run['mode']} | {run['qps']:g} | {delivery.get('engine_output_updates', 0)} | "
            f"{delivery.get('api_output_updates', 0)} | {delivery.get('matched_api_output_updates', 0)} | "
            f"{first.get('p50_ms') or 0:.1f}/{first.get('p95_ms') or 0:.1f} | "
            f"{delivery.get('measurement_first_api_token_in_window', 0)}/"
            f"{delivery.get('measurement_sent_requests', 0)} | "
            f"{delivery.get('measurement_first_token_right_censored_at_window', 0)} |"
        )
    lines.extend(["", "## Output queue stage diagnostics", "",
                  "Sampled first, every 64th, and final output updates are joined by request, instance, and token count. "
                  "Older runs collected before this instrumentation have no stage samples.", "",
                  "| Mode | QPS | Enqueues | Dequeues | Engine/enqueue and enqueue/dequeue matches | Engine→enqueue p50/p95 ms | Forwarder stage p50/p95 ms (engine→submit / submit→loop / loop→send / send→ZMQ) | ZMQ residence p50/p95 ms | Dequeue to API p50/p95 ms | Queue depth P95 messages |",
                  "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for run in result["per_run"]:
        stages = run["observed"].get("output_queue_stage_diagnostics", {})
        engine_enqueue = stages.get("engine_to_enqueue", {})
        residence = stages.get("queue_residence", {})
        callback = stages.get("dequeue_to_api_callback", {})
        depths = stages.get("queue_depth_messages", {})
        forwarding = stages.get("forwarder_stages", {})
        forwarding_order = ("engine_to_submit", "submit_to_loop_start",
                            "loop_start_to_queue_send", "queue_send_to_enqueue")
        forwarding_summary = " / ".join(
            f"{(forwarding.get(name, {}).get('p50_ms') or 0):.1f}/"
            f"{(forwarding.get(name, {}).get('p95_ms') or 0):.1f}"
            for name in forwarding_order
        )
        lines.append(
            f"| {run['mode']} | {run['qps']:g} | {stages.get('enqueue_samples', 0)} | "
            f"{stages.get('dequeue_samples', 0)} | "
            f"{stages.get('matched_engine_enqueue_samples', 0)}/"
            f"{stages.get('matched_enqueue_dequeue_samples', 0)} | "
            f"{engine_enqueue.get('p50_ms') or 0:.1f}/{engine_enqueue.get('p95_ms') or 0:.1f} | "
            f"{forwarding_summary} | "
            f"{residence.get('p50_ms') or 0:.1f}/{residence.get('p95_ms') or 0:.1f} | "
            f"{callback.get('p50_ms') or 0:.1f}/{callback.get('p95_ms') or 0:.1f} | "
            f"{depths.get('p95') or 0:.1f} |"
        )
    lines.extend(["", "## Per-run values", "",
                  "| Mode | QPS | Version | Request throughput | Token throughput | Mean/P95 TTFT ms | Mean/P95 TPOT ms | Queue mean/P95 | KV mean/P95 | Migrations (attempt/success/fail/abort) | Total/pause/recovery ms |",
                  "|:---:|---:|:---|---:|---:|---:|---:|---:|---:|:---:|---:|"])
    for run in result["per_run"]:
        displayed = [("observed", run["observed"])] + [
            (version, item["simulation"]) for version, item in run["versions"].items()]
        for version, values in displayed:
            lines.append(f"| {run['mode']} | {run['qps']:g} | {version} | "
                         f"{values.get('request_throughput_rps', 0):.3f} | "
                         f"{values.get('output_tokens_per_sec', 0):.1f} | "
                         f"{values.get('ttft_mean_ms') or 0:.1f}/{values.get('ttft_p95_ms') or 0:.1f} | "
                         f"{values.get('tpot_mean_ms') or 0:.1f}/{values.get('tpot_p95_ms') or 0:.1f} | "
                         f"{values.get('queue_mean_requests') or 0:.2f}/{values.get('queue_p95_requests') or 0:.2f} | "
                         f"{(values.get('kv_mean_fraction') or 0) * 100:.2f}%/"
                         f"{(values.get('kv_p95_fraction') or 0) * 100:.2f}% | "
                         f"{values.get('migration_attempts', 0)}/{values.get('migration_successes', 0)}/"
                         f"{values.get('migration_failures', 0)}/{values.get('migration_aborted', 0)} | "
                         f"{values.get('migration_total_mean_ms') or 0:.1f}/"
                         f"{values.get('migration_pause_stage_mean_ms') or 0:.1f}/"
                         f"{values.get('migration_recovery_mean_ms') or 0:.1f} |")
    lines.extend(["", "## Fixed batch schedule replay", "",
                  "Observed held-out batches are fixed inputs for this local service model check.", "",
                  "| Mode | QPS | Batches | Round 2 MAE ms / MAPE | Causal MAE ms / MAPE |",
                  "|:---:|---:|---:|---:|---:|"])
    for run in result["per_run"]:
        fixed = run["fixed_batch_replay"]["overall"]
        old, new = fixed["round2_v3"], fixed["causal_rebuild"]
        old_text = ("n/a" if old["mae_ms"] is None else
                    f"{old['mae_ms']:.2f} / {old['mape'] * 100:.2f}%")
        new_text = ("n/a" if new["mae_ms"] is None else
                    f"{new['mae_ms']:.2f} / {new['mape'] * 100:.2f}%")
        lines.append(f"| {run['mode']} | {run['qps']:g} | {new['n']} | "
                     f"{old_text} | {new_text} |")
    lines.extend(["", "## Fixed route and migration replay", "",
                  "| Mode | QPS | Observed migration starts | Replayed starts | Rejected starts | First-batch route coverage |",
                  "|:---:|---:|---:|---:|---:|---:|"])
    for run in result["per_run"]:
        fixed = run["versions"]["causal_fixed_replay"]["simulation"]
        replay = fixed.get("fixed_replay_results", [])
        started = sum(item.get("result") == "started" for item in replay)
        lines.append(f"| {run['mode']} | {run['qps']:g} | "
                     f"{run['observed'].get('migration_attempts', 0)} | {len(replay)} | "
                     f"{len(replay) - started} | {fixed.get('fixed_route_coverage', 0.0):.1%} |")
    lines.extend(["", "## Accuracy gate", "",
                  f"Independent accuracy gate passed: **{result['passed_independent_accuracy_gate']}**.", "",
                  "| Metric | Passed | Worst held-out run | Error | Limit |",
                  "|:---|:---:|:---|---:|---:|"])
    for metric, check in result["independent_accuracy_gate"]["checks"].items():
        worst = check.get("worst_error")
        if worst:
            lines.append(f"| {metric} | {check['passed']} | {worst['mode']} QPS {worst['qps']:g} | "
                         f"{worst['error_metric']:.3f} | {worst['limit']:.3f} |")
        else:
            lines.append(f"| {metric} | {check['passed']} | n/a | n/a | {check.get('threshold')} |")
    lines.extend(["", "## Figures", "",
                  "![OFF held-out metrics](plots/system_validation_off.png)", "",
                  "![ON held-out metrics](plots/system_validation_on.png)", "",
                  "![Round 2 versus causal MAPE](plots/round2_vs_causal_mape.png)",
                  "", "## Definitions and limitations", "",
                  "Queue and KV are resampled at 50 ms per-instance grids. Request output and server completion use the API/server event chain; client E2E is retained as a separate metric. HTTP 499 tails are counted as censored.",
                  "TTFT and TPOT ignore empty RequestOutput updates. TTFT remains an event-window statistic; the sent-request first-token coverage and right-censored count are reported alongside it.",
                  "Engine-to-API callback lag joins cumulative output counts for diagnosis. It includes the output-forwarding path and is not used to calibrate the simulator during holdout scoring.",
                  "Warmup requests are drained or aborted before the measurement phase. The simulator starts with an empty request/KV state at that boundary; it does not model GPU cache or thermal effects from warmup.",
                  "Migration ON uses the runtime Balanced/RemainingStepsLoad routine policy parsed from the real run configuration; the routine policy is mirrored in the simulator and the service strategy is unchanged.",
                  "The fixed replay diagnostic uses held-out first-batch instance routes and observed ON migration starts, durations, pause, and recovery labels. It is a mechanism diagnostic and is excluded from the independent prediction gate.",
                  "A gate passes only when every OFF and ON QPS run meets the per-run threshold and every measurement request is joined to a terminal event.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plots(output_dir: Path, result: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("request_throughput_rps", "Completed requests / s"),
        ("output_tokens_per_sec", "Output tokens / s"),
        ("ttft_mean_ms", "Mean TTFT (ms)"),
        ("ttft_p95_ms", "P95 TTFT (ms)"),
        ("tpot_mean_ms", "Mean TPOT (ms)"),
        ("tpot_p95_ms", "P95 TPOT (ms)"),
        ("queue_mean_requests", "Mean waiting requests / instance"),
        ("queue_p95_requests", "P95 waiting requests / instance"),
        ("kv_mean_fraction", "Mean KV utilization"),
        ("kv_p95_fraction", "P95 KV utilization"),
    ]
    versions = ("round2_v3", "causal_rebuild", "causal_fixed_replay")
    colors = {"observed": "black", "round2_v3": "#d95f02",
              "causal_rebuild": "#1b9e77", "causal_fixed_replay": "#7570b3"}
    for mode in ("off", "on"):
        fig, axes = plt.subplots(5, 2, figsize=(13, 17), sharex=True)
        for axis, (metric, title) in zip(axes.flat, metrics):
            mode_rows = sorted((row for row in result["per_run"] if row["mode"] == mode),
                               key=lambda row: row["qps"])
            if not mode_rows:
                axis.set_visible(False)
                continue
            observed_y = [_num(row["observed"].get(metric)) for row in mode_rows]
            axis.plot([row["qps"] for row in mode_rows], observed_y,
                      marker="o", color=colors["observed"], label="observed")
            for version in versions:
                predicted_y = [_num(row["versions"][version]["simulation"].get(metric))
                               for row in mode_rows]
                axis.plot([row["qps"] for row in mode_rows], predicted_y,
                          marker=".", color=colors[version], label=version)
            axis.set_title(title)
            axis.grid(True, alpha=0.25)
        for axis in axes[-1, :]:
            axis.set_xlabel("Target QPS")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        fig.suptitle(f"Held-out {mode.upper()} system validation", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(output_dir / f"system_validation_{mode}.png", dpi=160)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    metric_names = ("request_throughput_rps", "ttft_mean_ms", "ttft_p95_ms",
                    "tpot_mean_ms", "tpot_p95_ms")
    x = np.arange(len(metric_names))
    width = 0.36
    for offset, version in ((-width / 2, "round2_v3"), (width / 2, "causal_rebuild")):
        values = [result["aggregate_errors"].get(version, {}).get(metric, {}).get("macro_mape")
                  for metric in metric_names]
        axis.bar(x + offset, [float("nan") if value is None else value * 100 for value in values],
                 width, label=version, color=colors[version])
    axis.set_xticks(x, ["Throughput", "Mean TTFT", "P95 TTFT", "Mean TPOT", "P95 TPOT"])
    axis.set_ylabel("Macro MAPE across held-out QPS/mode runs (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "round2_vs_causal_mape.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-dir", type=Path, required=True)
    parser.add_argument("--on-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round2-config", type=Path, required=True)
    parser.add_argument("--validation-seed", type=int, default=707)
    parser.add_argument("--warmup-sec", type=float, required=True)
    parser.add_argument("--measure-sec", type=float, required=True)
    parser.add_argument("--drain-sec", type=float, required=True)
    args = parser.parse_args()
    result = validate(args.off_dir, args.on_dir, args.model_dir, args.output_dir,
                      args.round2_config, args.validation_seed,
                      args.warmup_sec, args.measure_sec, args.drain_sec)
    print(json.dumps({"per_run_count": len(result["per_run"]),
                      "aggregate_errors": result["aggregate_errors"],
                      "passed_independent_accuracy_gate": result["passed_independent_accuracy_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
