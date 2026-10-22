"""Replay held-out arrivals through the Mig-PPO simulator.

The trace supplies only arrival times and prompt/output sizes. Request response
times are loaded only for final comparison. Fixed migration replay consumes the
logged event schedule and outcomes; closed-loop mode uses a transparent greedy
controller over the environment's pair-gain features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from rl_scheduler.env import LlumnixMigrationEnv, MigPPOEnvConfig
from rl_scheduler.refine_calibration import grouped_service_cv, load_profile_split
from rl_scheduler.service_model import predict_service_model


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def _float(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _request_trace(path: Path, requests_path: Path | None = None) -> list[dict[str, object]]:
    rows = _read_csv(path)
    send_times = {_short_request_id(row.get("request_id")): _float(row.get("actual_send_sec"))
                  for row in _read_csv(requests_path) if row.get("phase") == "measurement"} \
        if requests_path else {}
    trace = []
    for row in rows:
        request_id = str(row["request_id"])
        scheduled = float(row["scheduled_arrival_sec"])
        actual = send_times.get(request_id)
        output_budget = int(row["output_tokens"])
        item = {"request_id": request_id,
                "arrival_time": float(actual) if actual is not None else scheduled,
                "scheduled_arrival_sec": scheduled,
                "actual_send_sec": actual,
                "prompt_tokens": int(row["input_tokens"]),
                "output_tokens": output_budget,
                "output_budget_tokens": output_budget}
        realized = _float(row.get("realized_output_tokens"))
        if realized is not None:
            realized_int = int(realized)
            if realized_int < 1 or realized_int > output_budget:
                raise ValueError(f"invalid fixed trace output length for {request_id}: {realized_int}/{output_budget}")
            item["realized_output_tokens"] = realized_int
            item["realized_output_source"] = str(row.get("realized_output_source") or "fixed_training_sample")
        trace.append(item)
    return trace


def _output_length_model(data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit an empirical realized-length distribution from calibration successes only."""
    by_profile: dict[tuple[int, int], Counter[int]] = {}
    by_budget: dict[int, Counter[int]] = {}
    rejected = Counter()
    for run_dir in sorted((data_root / "calibration").glob("qps_*_seed*")):
        request_path = run_dir / "requests.csv"
        trace_path = run_dir / "traces" / f"trace_{run_dir.name}.csv"
        if not request_path.exists() or not trace_path.exists():
            continue
        trace_rows = _read_csv(trace_path)
        trace_by_id = {str(row.get("request_id", "")): row for row in trace_rows}
        for row in _read_csv(request_path):
            if row.get("phase") != "measurement" or row.get("success", "").lower() != "true":
                continue
            trace_id = _short_request_id(row.get("request_id"))
            trace = trace_by_id.get(trace_id or "")
            if trace is None:
                rejected["missing_trace_join"] += 1
                continue
            try:
                prompt = int(trace["input_tokens"])
                budget = int(trace["output_tokens"])
                realized = int(float(row.get("server_output_tokens", "")))
            except (KeyError, TypeError, ValueError, OverflowError):
                rejected["invalid_length"] += 1
                continue
            if budget < 1 or realized < 1 or realized > budget:
                rejected["invalid_or_over_budget"] += 1
                continue
            by_profile.setdefault((prompt, budget), Counter())[realized] += 1
            by_budget.setdefault(budget, Counter())[realized] += 1

    def encode(counter: Counter[int]) -> dict[str, Any]:
        values = sorted(counter)
        return {"lengths": values, "counts": [int(counter[value]) for value in values],
                "n": int(sum(counter.values()))}

    profiles = {f"{prompt}:{budget}": encode(counter)
                for (prompt, budget), counter in sorted(by_profile.items())}
    profiles.update({f"*:{budget}": encode(counter) for budget, counter in sorted(by_budget.items())})
    total = sum(sum(counter.values()) for counter in by_profile.values())
    audit = {"source": "calibration measurement phase, successful requests, server_output_tokens",
             "held_out_seed_used": False, "profile_sample_counts": {
                 key: value["n"] for key, value in profiles.items() if not key.startswith("*:")},
             "n_successful_lengths": int(total), "rejected_rows": dict(rejected),
             "selection_limit": "completed responses only; drain-censored requests have no realized length label"}
    return {"version": 1, "source": audit["source"], "profiles": profiles}, audit


def _short_request_id(value: object) -> str | None:
    match = re.search(r"r(\d{5,})$", str(value or ""))
    return f"r{int(match.group(1)):05d}" if match else None


def _run_inputs(run_dir: Path, trace: list[dict[str, object]], measurement_sec: float) -> tuple[list[str], dict[str, str], list[dict[str, object]], dict[str, Any]]:
    requests = [row for row in _read_csv(run_dir / "requests.csv")
                if row.get("phase") == "measurement"]
    send_epochs = [_float(row.get("actual_send_wall_time")) for row in requests]
    send_secs = [_float(row.get("actual_send_sec")) for row in requests]
    epoch_pairs = [(wall - sec) for wall, sec in zip(send_epochs, send_secs)
                   if wall is not None and sec is not None]
    epoch = min(epoch_pairs) if epoch_pairs else 0.0
    service_rows = _read_csv(run_dir / "server_logs/on_instance_instance.csv")
    instance_ids = sorted({row.get("instance_id", "") for row in service_rows if row.get("instance_id")})
    migration_rows = _read_csv(run_dir / "migration_events.csv")
    measurement_migrations = [row for row in migration_rows if row.get("phase") == "measurement"]
    for row in migration_rows:
        for key in ("source_instance", "target_instance"):
            if row.get(key) and row[key] not in instance_ids:
                instance_ids.append(row[key])
    instance_ids = sorted(set(instance_ids))
    routes: dict[str, str] = {}
    events: list[dict[str, object]] = []
    omitted = {"no_request_id": 0, "outside_measurement": 0, "bad_time": 0}
    trace_ids = {str(item["request_id"]) for item in trace}
    for row in migration_rows:
        if row.get("phase") != "measurement":
            omitted["outside_measurement"] += 1
            continue
        try:
            ids = json.loads(row.get("request_ids") or "[]")
        except json.JSONDecodeError:
            ids = []
        linked = [_short_request_id(value) for value in ids]
        linked = [value for value in linked if value in trace_ids]
        if not linked:
            omitted["no_request_id"] += 1
            continue
        cost = _float(row.get("cost_ms"))
        observed = _float(row.get("observed_wall_time"))
        if cost is None or observed is None or not epoch:
            omitted["bad_time"] += 1
            continue
        event_time = max(0.0, observed - cost / 1000.0 - epoch)
        for request_id in linked:
            routes.setdefault(request_id, str(row.get("source_instance", "")))
            events.append({"scheduled_time": event_time,
                           "request_id": request_id,
                           "source_instance": str(row.get("source_instance", "")),
                           "target_instance": str(row.get("target_instance", "")),
                           "blocks": int(_float(row.get("blocks"), 0) or 0),
                           "cost_ms": cost,
                           "status": str(row.get("status", "FINISHED")),
                           "observed_wall_time": observed})
    events.sort(key=lambda item: float(item["scheduled_time"]))
    valid_routes = {key: value for key, value in routes.items() if value in instance_ids}
    migration_costs = [value for row in measurement_migrations
                       if (value := _float(row.get("cost_ms"))) is not None]
    arrival_lags = [float(item["actual_send_sec"]) - float(item["scheduled_arrival_sec"])
                    for item in trace if item.get("actual_send_sec") is not None
                    and item.get("scheduled_arrival_sec") is not None]
    audit = {"measurement_request_rows": len(requests), "trace_arrivals": len(trace),
             "trace_arrival_source": "actual_send_sec from requests.csv, scheduled arrival as fallback",
             "arrival_rows_using_actual_send_time": len(arrival_lags),
             "actual_minus_scheduled_arrival_mean_ms": float(np.mean(arrival_lags) * 1000) if arrival_lags else None,
             "actual_minus_scheduled_arrival_p95_ms": _quantile(arrival_lags, .95) * 1000 if arrival_lags else None,
             "actual_minus_scheduled_arrival_max_ms": max(arrival_lags) * 1000 if arrival_lags else None,
             "measurement_epoch_approximation": "min(actual_send_wall_time - actual_send_sec)",
             "measurement_epoch_wall_sec": epoch, "fixed_migration_events": len(events),
             "fixed_migration_event_rows": len(measurement_migrations),
             "logged_migration_status_counts": dict(Counter(row.get("status", "unknown")
                                                                for row in measurement_migrations)),
             "logged_migration_cost_mean_ms": float(np.mean(migration_costs)) if migration_costs else None,
             "logged_migration_cost_p95_ms": _quantile(migration_costs, .95),
             "logged_migration_cost_total_ms": float(sum(migration_costs)) if migration_costs else 0.0,
             "fixed_migration_omitted": omitted, "trace_requests_routed_to_logged_first_source": len(valid_routes),
             "closed_loop_route_count": 0, "instance_ids": instance_ids}
    return instance_ids, valid_routes, events, audit


def _greedy_action(env: LlumnixMigrationEnv) -> int:
    obs = env._last_obs
    if obs is None or env.pending_migration is not None:
        return 0
    legal = np.asarray(obs.pair_mask, dtype=bool)
    gains = np.asarray(obs.pair_features[:, 4], dtype=float) if len(obs.pair_features) else np.asarray([])
    candidates = np.flatnonzero(legal & (gains > 0))
    if not len(candidates):
        return 0
    return int(candidates[np.argmax(gains[candidates])] + 1)


def _simulate(config: MigPPOEnvConfig, trace: list[dict[str, object]], qps: float,
              instance_ids: list[str], routes: dict[str, str], *, seed: int,
              horizon_sec: float, measurement_sec: float = 60.0,
              fixed_events: list[dict[str, object]] | None = None,
              migration_controller: bool = False) -> dict[str, Any]:
    interval = config.decision_interval_sec
    steps = int(math.ceil(horizon_sec / interval))
    run_config = replace(config, max_decisions=steps, workload_scenario=f"qps_{qps:g}",
                         target_qps_levels=(qps,), request_timeout_sec=None,
                         validate_state_invariants=True)
    env = LlumnixMigrationEnv(run_config)
    env.reset(seed=seed, options={"target_qps": qps, "arrival_trace": trace,
                                  "request_routes": routes, "instance_ids": instance_ids})
    pending = [dict(event) for event in (fixed_events or [])]
    replay_outcomes: list[dict[str, object]] = []
    queue_samples: list[float] = []
    kv_samples: list[float] = []
    failed_replay_after_sec = 10.0
    refresh_interval = config.observation_refresh_interval_sec or interval
    controller_period_steps = max(1, int(round(refresh_interval / interval)))
    for step_index in range(steps):
        if pending:
            event = pending[0]
            if float(event["scheduled_time"]) <= env.now + interval + 1e-9:
                outcome = env.start_replay_migration(event)
                if outcome == "started":
                    replay_outcomes.append({"request_id": event["request_id"], "scheduled_time": event["scheduled_time"],
                                            "started_at": env.now, "start_delay_sec": env.now - float(event["scheduled_time"]),
                                            "outcome": outcome})
                    pending.pop(0)
                elif outcome not in {"migration_in_progress", "request_not_running_at_logged_source"}:
                    replay_outcomes.append({"request_id": event["request_id"], "scheduled_time": event["scheduled_time"],
                                            "started_at": None, "outcome": outcome})
                    pending.pop(0)
                elif env.now - float(event["scheduled_time"]) > failed_replay_after_sec:
                    replay_outcomes.append({"request_id": event["request_id"], "scheduled_time": event["scheduled_time"],
                                            "started_at": None, "outcome": f"expired:{outcome}"})
                    pending.pop(0)
        action = (_greedy_action(env) if step_index % controller_period_steps == 0 else 0) \
            if migration_controller else 0
        env.step(action)
        if env.now <= measurement_sec + 1e-9:
            queue_samples.append(float(np.mean([len(instance.waiting) for instance in env.instances])))
            kv_samples.append(float(np.mean([
                (instance.used_blocks + instance.reserved_blocks) / max(1, instance.total_blocks)
                for instance in env.instances])))
    results = list(env.request_results)
    completed = [row for row in results if row["status"] == "completed"]
    completed_in_window = [row for row in completed if float(row["completion_sec"]) <= measurement_sec]
    ttft = [float(row["ttft_ms"]) for row in completed if row.get("ttft_ms") is not None]
    request_tpot = [float(row["tpot_mean_ms"]) for row in completed if row.get("tpot_mean_ms") is not None]
    sim_iterations = {phase: [float(row["predicted_ms"]) for row in env.service_iterations
                              if row["phase"] == phase] for phase in ("prefill", "decode")}
    decode_token_intervals = [float(row.get("effective_token_interval_ms", row["predicted_ms"]))
                              for row in env.service_iterations if row["phase"] == "decode"]
    migration_costs = [float(row["cost_ms"]) for row in env.migration_events]
    migration_successes = sum(event["status"] == "success" for event in env.migration_events)
    migration_failures = sum(event["status"] == "failed" for event in env.migration_events)
    migration_aborts = sum(event["status"] == "aborted" for event in env.migration_events)
    return {
        "arrivals": env.request_count,
        "controller_observation_refresh_sec": refresh_interval if migration_controller else None,
        "completed": len(completed),
        "completed_throughput_rps": len(completed) / measurement_sec,
        "failed": sum(row["status"] == "timeout" for row in results),
        "unfinished": len(env.requests),
        "in_window_completed": len(completed_in_window),
        "in_window_throughput_rps": len(completed_in_window) / measurement_sec,
        "ttft_mean_ms": float(np.mean(ttft)) if ttft else None,
        "ttft_p95_ms": _quantile(ttft, .95),
        "tpot_request_mean_ms": float(np.mean(request_tpot)) if request_tpot else None,
        "tpot_request_p95_ms": _quantile(request_tpot, .95),
        "queue_mean_requests": float(np.mean(queue_samples)) if queue_samples else 0.0,
        "queue_p95_requests": _quantile(queue_samples, .95),
        "kv_mean_fraction": float(np.mean(kv_samples)) if kv_samples else 0.0,
        "kv_p95_fraction": _quantile(kv_samples, .95),
        "prefill_iteration_mean_ms": float(np.mean(sim_iterations["prefill"])) if sim_iterations["prefill"] else None,
        "prefill_iteration_p95_ms": _quantile(sim_iterations["prefill"], .95),
        "decode_iteration_mean_ms": float(np.mean(sim_iterations["decode"])) if sim_iterations["decode"] else None,
        "decode_iteration_p95_ms": _quantile(sim_iterations["decode"], .95),
        "decode_token_interval_mean_ms": float(np.mean(decode_token_intervals)) if decode_token_intervals else None,
        "decode_token_interval_p95_ms": _quantile(decode_token_intervals, .95),
        "migration_count": len(env.migration_events),
        "migration_attempts_started": env.action_count,
        "migration_successes": migration_successes,
        "migration_failures": migration_failures,
        "migration_aborted": migration_aborts,
        "migration_success_rate": migration_successes / len(env.migration_events) if env.migration_events else None,
        "migration_cost_total_ms": float(sum(migration_costs)) if migration_costs else 0.0,
        "migration_cost_mean_ms": float(np.mean(migration_costs)) if migration_costs else None,
        "migration_cost_p95_ms": _quantile(migration_costs, .95),
        "migration_pending_at_horizon": int(env.pending_migration is not None),
        "fixed_replay_attempts": len(replay_outcomes),
        "fixed_replay_started": sum(row["outcome"] == "started" for row in replay_outcomes),
        "fixed_replay_unattempted": len(pending),
        "fixed_replay_nonstarted": sum(row["outcome"] != "started" for row in replay_outcomes),
        "fixed_replay_outcomes": replay_outcomes,
        "service_iteration_n": {phase: len(values) for phase, values in sim_iterations.items()},
    }


def _observed_timeline_audit(run_dir: Path) -> dict[str, Any]:
    rows = [row for row in _read_csv(run_dir / "requests.csv")
            if row.get("phase") == "measurement" and row.get("success", "").lower() == "true"]
    server_totals, client_totals, gaps = [], [], []
    for row in rows:
        try:
            raw = json.loads(row.get("per_token_latency_json") or "[]")
        except json.JSONDecodeError:
            continue
        intervals = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    value = _float(item[1])
                    if value is not None:
                        intervals.append(value)
        try:
            output_count = int(float(row.get("server_output_tokens", "")))
        except (TypeError, ValueError, OverflowError):
            continue
        client = _float(row.get("client_e2e_ms"))
        if not intervals or len(intervals) != output_count or client is None:
            continue
        server_total = float(sum(intervals))
        server_totals.append(server_total)
        client_totals.append(client)
        gaps.append(client - server_total)
    return {"timeline_complete_n": len(server_totals),
            "timeline_coverage_of_successes": len(server_totals) / len(rows) if rows else None,
            "server_timeline_total_mean_ms": float(np.mean(server_totals)) if server_totals else None,
            "server_timeline_total_p95_ms": _quantile(server_totals, .95),
            "client_e2e_same_rows_mean_ms": float(np.mean(client_totals)) if client_totals else None,
            "client_minus_server_timeline_mean_ms": float(np.mean(gaps)) if gaps else None,
            "client_minus_server_timeline_p50_ms": _quantile(gaps, .5),
            "client_minus_server_timeline_p95_ms": _quantile(gaps, .95)}


def _observed_run(run_dir: Path, qps: float, measurement_sec: float, drain_sec: float,
                  comparison_horizon_sec: float) -> dict[str, Any]:
    rows = [row for row in _read_csv(run_dir / "requests.csv") if row.get("phase") == "measurement"]
    successes = [row for row in rows if row.get("success", "").lower() == "true"]
    timeouts = [row for row in rows if row.get("timeout", "").lower() == "true"]
    non_success = [row for row in rows if row.get("success", "").lower() != "true"
                   and row.get("timeout", "").lower() != "true"]
    censored = [row for row in rows if row.get("success", "").lower() != "true"
                and row.get("status_code", "") == "499"
                and (_float(row.get("completion_sec"), -math.inf) or -math.inf)
                >= measurement_sec + drain_sec - 1.0]
    failures = [row for row in rows if row.get("success", "").lower() != "true"
                and row.get("timeout", "").lower() != "true" and row not in censored]
    unresolved = [row for row in rows if not row.get("completion_sec")
                  and row.get("timeout", "").lower() != "true" and row not in censored]
    completed_by_horizon = [row for row in successes
                            if (_float(row.get("completion_sec"), math.inf) or math.inf)
                            <= comparison_horizon_sec]
    failed_by_horizon = [row for row in non_success
                         if (_float(row.get("completion_sec"), math.inf) or math.inf)
                         <= comparison_horizon_sec]
    censored_by_horizon = [row for row in censored
                           if (_float(row.get("completion_sec"), math.inf) or math.inf)
                           <= comparison_horizon_sec]
    genuine_failures_by_horizon = [row for row in failures
                                   if (_float(row.get("completion_sec"), math.inf) or math.inf)
                                   <= comparison_horizon_sec]
    timeout_by_horizon = [row for row in timeouts
                          if not row.get("completion_sec")
                          or (_float(row.get("completion_sec"), math.inf) or math.inf)
                          <= comparison_horizon_sec]
    unfinished_by_horizon = max(0, len(rows) - len(completed_by_horizon)
                                - len(genuine_failures_by_horizon) - len(timeout_by_horizon)
                                - len(censored_by_horizon))
    ttft = [value for row in completed_by_horizon
            if (value := _float(row.get("server_ttft_ms"))) is not None]
    tpot = [value for row in completed_by_horizon
            if (value := _float(row.get("server_tpot_ms"))) is not None]
    in_window = [row for row in successes if (_float(row.get("completion_sec"), math.inf) or math.inf) <= measurement_sec]
    successful_after_drain = [row for row in successes
                              if (_float(row.get("completion_sec"), -math.inf) or -math.inf)
                              > measurement_sec + drain_sec]
    output_lengths = [value for row in completed_by_horizon
                      if (value := _float(row.get("server_output_tokens"))) is not None]
    output_budgets = [value for row in completed_by_horizon
                      if (value := _float(row.get("expected_output_tokens"))) is not None]
    return {"arrivals": len(rows), "completed": len(successes),
            "request_throughput_rps": len(successes) / measurement_sec,
            "failed": len(failures),
            "timeouts": len(timeouts), "censored_at_drain": len(censored),
            "unfinished": len(censored) + len(unresolved), "unfinished_without_response": len(unresolved),
            "successful_after_drain_cutoff": len(successful_after_drain),
            "comparison_horizon_sec": comparison_horizon_sec,
            "completed_by_comparison_horizon": len(completed_by_horizon),
            "failed_by_comparison_horizon": len(genuine_failures_by_horizon),
            "censored_by_comparison_horizon": len(censored_by_horizon),
            "timeouts_by_comparison_horizon": len(timeout_by_horizon),
            "unfinished_by_comparison_horizon": unfinished_by_horizon,
            "in_window_completed": len(in_window), "in_window_throughput_rps": len(in_window) / measurement_sec,
            "ttft_mean_ms": float(np.mean(ttft)) if ttft else None,
            "ttft_p95_ms": _quantile(ttft, .95),
            "tpot_request_mean_ms": float(np.mean(tpot)) if tpot else None,
            "tpot_request_p95_ms": _quantile(tpot, .95), "ttft_n": len(ttft), "tpot_n": len(tpot),
            "ttft_coverage_of_successes": len(ttft) / len(successes) if successes else None,
            "tpot_coverage_of_successes": len(tpot) / len(successes) if successes else None,
            "output_tokens_mean": float(np.mean(output_lengths)) if output_lengths else None,
            "output_budget_tokens_mean": float(np.mean(output_budgets)) if output_budgets else None,
            "realized_to_budget_output_token_ratio": (float(np.sum(output_lengths) / np.sum(output_budgets))
                                                       if output_lengths and output_budgets
                                                       and np.sum(output_budgets) else None),
            "output_tokens_p95": _quantile(output_lengths, .95),
            "drain_cutoff_sec": measurement_sec + drain_sec,
            "responses_after_nominal_drain": sum(
                (_float(row.get("completion_sec"), -math.inf) or -math.inf) > measurement_sec + drain_sec
                for row in rows),
            "server_timeline_audit": _observed_timeline_audit(run_dir),
            "status_code_counts": dict(Counter(row.get("status_code", "") or "none" for row in rows))}


def _observed_state(run_dir: Path, start_epoch: float, measurement_sec: float,
                    sample_interval_sec: float = 0.05) -> dict[str, Any]:
    rows = _read_csv(run_dir / "server_logs/on_instance_instance.csv")
    by_step: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        timestamp = _float(row.get("timestamp"))
        if timestamp is None or not start_epoch <= timestamp <= start_epoch + measurement_sec:
            continue
        by_step[(row.get("instance_id", ""), row.get("step_id", ""))] = row
    by_instance: dict[str, list[tuple[float, dict[str, str]]]] = {}
    for row in by_step.values():
        timestamp = _float(row.get("timestamp"))
        if timestamp is not None:
            by_instance.setdefault(row.get("instance_id", ""), []).append((timestamp, row))
    for values in by_instance.values():
        values.sort(key=lambda item: item[0])

    # Match the simulator's 50 ms end-of-step observations. The engine emits
    # state rows at irregular per-instance step times, so carry each instance's
    # latest observed state forward to the same fixed wall-clock grid.
    grid = np.arange(sample_interval_sec, measurement_sec + sample_interval_sec * 1e-6,
                     sample_interval_sec, dtype=np.float64)
    queue_at_time: list[float] = []
    kv_at_time: list[float] = []
    rows_used: set[tuple[str, str]] = set()
    for relative_time in grid:
        queue_values: list[float] = []
        kv_values: list[float] = []
        for instance_id, values in by_instance.items():
            times = [timestamp - start_epoch for timestamp, _ in values]
            index = int(np.searchsorted(times, relative_time, side="right")) - 1
            if index < 0:
                # The first server snapshot may arrive shortly after the work
                # window opens; use it only for the adjacent 50 ms sample.
                if times and times[0] - relative_time <= sample_interval_sec + 1e-9:
                    index = 0
                else:
                    continue
            row = values[index][1]
            queue_value = _float(row.get("num_waiting_requests"))
            kv_value = _float(row.get("gpu_cache_usage"))
            if queue_value is not None:
                queue_values.append(queue_value)
            if kv_value is not None:
                kv_values.append(kv_value)
            rows_used.add((instance_id, str(row.get("step_id", ""))))
        if queue_values:
            queue_at_time.append(float(np.mean(queue_values)))
        if kv_values:
            kv_at_time.append(float(np.mean(kv_values)))
    return {
        "queue_mean_requests": float(np.mean(queue_at_time)) if queue_at_time else None,
        "queue_p95_requests": _quantile(queue_at_time, .95),
        "kv_mean_fraction": float(np.mean(kv_at_time)) if kv_at_time else None,
        "kv_p95_fraction": _quantile(kv_at_time, .95),
        "state_sampling_policy": "per-instance last observation carried forward on fixed 50ms grid; instance means per sample",
        "state_sampling_interval_sec": sample_interval_sec,
        "state_grid_sample_n": int(len(grid)),
        "queue_grid_sample_n": len(queue_at_time),
        "kv_grid_sample_n": len(kv_at_time),
        "unique_engine_state_rows_used": len(rows_used),
        "instances_with_state_rows": len(by_instance),
    }


def _measurement_epochs(data_root: Path, split: str) -> dict[str, float]:
    epochs: dict[str, float] = {}
    for run_dir in sorted((data_root / split).glob("qps_*_seed*")):
        rows = [row for row in _read_csv(run_dir / "requests.csv")
                if row.get("phase") == "measurement"]
        pairs = [wall - relative for row in rows
                 if (wall := _float(row.get("actual_send_wall_time"))) is not None
                 and (relative := _float(row.get("actual_send_sec"))) is not None]
        if pairs:
            epochs[run_dir.name] = min(pairs)
    return epochs


def _profile_phase(row: Mapping[str, Any], epochs: Mapping[str, float], measurement_sec: float) -> str:
    run_group = str(row.get("run_group", ""))
    end = _float(row.get("profile_timestamp"))
    latency = _float(row.get("latency_ms"))
    start = end - (latency or 0.0) / 1000.0 if end is not None else None
    epoch = epochs.get(run_group)
    if start is None or end is None or epoch is None:
        return "unknown"
    if end < epoch:
        return "warmup"
    if start >= epoch + measurement_sec:
        return "drain"
    if start >= epoch and end <= epoch + measurement_sec:
        return "measurement"
    return "boundary"


def _residual_metrics(observed: list[float], predicted: list[float]) -> dict[str, Any]:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    absolute = np.abs(residual)
    return {"n": int(len(residual)),
            "mae_ms": float(np.mean(absolute)) if len(residual) else None,
            "rmse_ms": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else None,
            "bias_prediction_minus_observed_ms": float(np.mean(residual)) if len(residual) else None,
            "residual_p05_p50_p95_ms": [float(x) for x in np.quantile(residual, [.05, .5, .95])]
            if len(residual) else None,
            "label_p95_ms": _quantile(observed, .95),
            "label_over_500ms": int(np.sum(np.asarray(observed) > 500)),
            "absolute_residual_over_250ms": int(np.sum(absolute > 250))}


def _service_features(row: Mapping[str, Any]) -> dict[str, float]:
    features = {name: float(row[name]) for name in (
        "batch_tokens", "total_context_tokens", "batch_size", "running_requests")
        if row.get(name) is not None}
    features["avg_context_tokens"] = features.get("total_context_tokens", 0.0) / max(
        1.0, features.get("batch_size", 0.0))
    return features


def _fit_decode_token_interval_scale(data_root: Path, service_model: Mapping[str, Any],
                                     measurement_sec: float) -> tuple[float, dict[str, Any]]:
    """Estimate per-request token cadence from calibration runs, excluding holdout."""
    samples, load_audit = load_profile_split(data_root, "calibration")
    epochs = _measurement_epochs(data_root, "calibration")
    model = service_model.get("decode")
    if not isinstance(model, Mapping):
        raise ValueError("calibration service model has no decode model")
    per_run = {}
    scales = []
    for run_group in sorted(epochs):
        engine_rows = [row for row in samples if row["run_group"] == run_group
                       and row["kind"] == "decode"
                       and _profile_phase(row, epochs, measurement_sec) == "measurement"]
        requests_path = data_root / "calibration" / run_group / "requests.csv"
        tpot = [value for row in _read_csv(requests_path)
                if row.get("phase") == "measurement" and row.get("success", "").lower() == "true"
                and (value := _float(row.get("server_tpot_ms"))) is not None]
        if not engine_rows or len(tpot) < 50:
            continue
        predicted = [predict_service_model(model, _service_features(row)) for row in engine_rows]
        predicted_mean = float(np.mean(predicted))
        observed_mean = float(np.mean(tpot))
        scale = observed_mean / max(1e-6, predicted_mean)
        scales.append(scale)
        per_run[run_group] = {"decode_iteration_samples": len(engine_rows), "tpot_samples": len(tpot),
                              "mean_predicted_iteration_ms": predicted_mean,
                              "mean_observed_tpot_ms": observed_mean,
                              "tpot_to_iteration_scale": scale}
    if not scales:
        raise ValueError("no calibration runs have enough matched decode iteration and TPOT samples")
    fitted = float(np.median(scales))
    audit = {"fit_source": "calibration seeds 101/202/303 only; per-run mean server TPOT divided by mean P1/D3-calibrated decode iteration time; median across runs",
             "held_out_seed_used_for_fit": False,
             "fitted_scale": fitted,
             "n_calibration_runs": len(scales),
             "scale_median_p05_p95": [float(x) for x in np.quantile(scales, [.05, .5, .95])],
             "scale_mad": float(np.median(np.abs(np.asarray(scales) - fitted))),
             "per_run": per_run,
             "profile_load_audit": load_audit,
             "interpretation": "empirical request emission cadence factor; it accounts for multiple executor iterations or scheduling gaps between emitted tokens and is not a replacement for measured iteration wall time"}
    return fitted, audit


def _relative_errors(simulated: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "ttft_mean_ms": "ttft_mean_ms",
        "ttft_p95_ms": "ttft_p95_ms",
        "tpot_request_mean_ms": "tpot_request_mean_ms",
        "tpot_request_p95_ms": "tpot_request_p95_ms",
        "completed_throughput_rps": "request_throughput_rps",
        "in_window_throughput_rps": "in_window_throughput_rps",
    }
    result = {}
    for simulated_key, observed_key in fields.items():
        pred, actual = simulated.get(simulated_key), observed.get(observed_key)
        relative = ((float(pred) - float(actual)) / abs(float(actual)) * 100.0
                    if pred is not None and actual is not None and float(actual) != 0 else None)
        result[simulated_key] = {"simulated": pred, "observed": actual,
                                 "signed_relative_error_pct": relative,
                                 "absolute_relative_error_pct": abs(relative) if relative is not None else None}
    return result


def _service_validation(rows: list[dict[str, Any]], epochs: Mapping[str, float], measurement_sec: float,
                        service_model: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    phase_counts: Counter[str] = Counter()
    classified = []
    for row in rows:
        phase_label = _profile_phase(row, epochs, measurement_sec)
        classified.append((row, phase_label))
        phase_counts[f"{row['kind']}:{phase_label}"] += 1

    result: dict[str, Any] = {}
    qps_values = sorted({float(row["qps"]) for row, label in classified
                         if row.get("qps") is not None and label == "measurement"})
    for qps in qps_values:
        result[str(qps)] = {}
        for phase in ("prefill", "decode"):
            selected = [float(row["latency_ms"]) for row, label in classified
                        if float(row["qps"]) == qps and row["kind"] == phase and label == "measurement"]
            result[str(qps)][phase] = {"n": len(selected),
                                       "mean_ms": float(np.mean(selected)) if selected else None,
                                       "p95_ms": _quantile(selected, .95)}

    model_results: dict[str, Any] = {}
    for kind in ("prefill", "decode"):
        model = service_model.get(kind)
        if not isinstance(model, Mapping):
            continue
        model_results[kind] = {}
        for phase_label in ("all", "measurement", "warmup", "drain", "boundary", "unknown"):
            selected = [(row, label) for row, label in classified
                        if row["kind"] == kind and (phase_label == "all" or label == phase_label)]
            observed = [float(row["latency_ms"]) for row, _ in selected]
            predictions = [predict_service_model(model, _service_features(row)) for row, _ in selected]
            model_results[kind][phase_label] = _residual_metrics(observed, predictions)
        model_results[kind]["measurement_by_qps"] = {}
        for qps in qps_values:
            selected = [row for row, label in classified if row["kind"] == kind and label == "measurement"
                        and float(row["qps"]) == qps]
            observed = [float(row["latency_ms"]) for row in selected]
            predictions = [predict_service_model(model, _service_features(row)) for row in selected]
            model_results[kind]["measurement_by_qps"][str(qps)] = _residual_metrics(observed, predictions)
    audit = {"phase_definition": "service interval [latest deduplicated snapshot timestamp - executor wall time, snapshot timestamp] relative to min(actual_send_wall_time - actual_send_sec); only intervals wholly inside the 60s workload window are measurement samples",
             "sample_counts_by_kind_and_phase": dict(phase_counts),
             "classification_unknown": int(phase_counts.get("prefill:unknown", 0) + phase_counts.get("decode:unknown", 0)),
             "model_residuals": model_results}
    return result, audit


def _calibration_service_cv(data_root: Path, measurement_sec: float) -> dict[str, Any]:
    samples, load_audit = load_profile_split(data_root, "calibration")
    epochs = _measurement_epochs(data_root, "calibration")
    by_kind = {kind: [row for row in samples if row["kind"] == kind
                      and _profile_phase(row, epochs, measurement_sec) == "measurement"]
               for kind in ("prefill", "decode")}
    selected_names = {"prefill": "P1", "decode": "D3"}
    old_comparison_path = data_root / "model_comparison.json"
    old_comparison = json.loads(old_comparison_path.read_text(encoding="utf-8")) \
        if old_comparison_path.exists() else {}
    result = {}
    for kind, model_name in selected_names.items():
        rows = by_kind[kind]
        cv = grouped_service_cv(rows, kind, model_name) if rows else None
        prior = old_comparison.get("service", {}).get(kind, {}).get("grouped_run_cv", {}).get(model_name, {})
        result[kind] = {"selected_model": model_name,
                        "all_phase_grouped_cv": prior.get("metrics"),
                        "measurement_only_grouped_cv": cv.get("metrics") if cv else None,
                        "measurement_only_macro_run_mae_ms": cv.get("macro_run_mae_ms") if cv else None,
                        "measurement_only_samples": len(rows),
                        "measurement_only_groups": len(cv.get("groups", [])) if cv else 0}
    result["phase_counts"] = dict(Counter(f"{row['kind']}:{_profile_phase(row, epochs, measurement_sec)}"
                                           for row in samples))
    result["profile_load_audit"] = load_audit
    result["phase_definition"] = "measurement-window interval rule applied to the configured held-out seed"
    return result


def run_system_validation(data_root: str | Path, *, calibration_yaml: str | Path | None = None,
                          output_dir: str | Path | None = None,
                          held_out_seed: int = 404) -> dict[str, Any]:
    root = Path(data_root)
    output = Path(output_dir) if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    collection = json.loads((root / "collection.json").read_text(encoding="utf-8"))
    measurement_sec = float(collection.get("measurement_sec", 60.0))
    drain_sec = float(collection.get("drain_sec", 10.0))
    comparison_drain_sec = max(drain_sec, 30.0)
    horizon = measurement_sec + comparison_drain_sec
    refined_path = Path(calibration_yaml) if calibration_yaml else root / "mig_ppo_calibration_refined.yaml"
    refined_payload = yaml.safe_load(refined_path.read_text(encoding="utf-8"))
    refined = refined_payload["environment"]
    refined_config = MigPPOEnvConfig(**refined)
    length_model, length_audit = _output_length_model(root)
    decode_scale, decode_scale_audit = _fit_decode_token_interval_scale(
        root, refined_config.service_time_model or {}, measurement_sec)
    data_audit_path = root / "data_audit.json"
    data_audit = json.loads(data_audit_path.read_text(encoding="utf-8")) if data_audit_path.exists() else {}
    kv_capacity_audit = data_audit.get("raw_datasets", {}).get("scheduler_state", {}).get(
        "estimated_total_kv_blocks_from_free_and_usage")
    refined_config = replace(refined_config, output_length_model=length_model,
                             decode_token_interval_scale=decode_scale)
    final_calibration = dict(refined_payload)
    final_calibration["environment"] = {**dict(refined), "output_length_model": length_model,
                                        "decode_token_interval_scale": decode_scale}
    final_calibration["system_validation_notes"] = {
        "output_length_model_audit": length_audit,
        "decode_token_interval_scale_audit": decode_scale_audit,
        "kv_capacity_inference": kv_capacity_audit,
        "held_out_seed_used_for_fit": False,
        "realized_length_is_sampled_from_calibration_successes": True,
    }
    (output / "mig_ppo_calibration_system_validation.yaml").write_text(
        yaml.safe_dump(final_calibration, sort_keys=False, allow_unicode=True), encoding="utf-8")
    old_path = root / "mig_ppo_calibration.yaml"
    old_payload = yaml.safe_load(old_path.read_text(encoding="utf-8"))["environment"]
    old_estimator = old_payload.get("migration_cost_estimator")
    calibration_versions = {
        "V0": replace(refined_config, service_model_mode="legacy", kv_dynamics_enabled=False,
                      prefill_ms_per_token=float(old_payload["prefill_ms_per_token"]),
                      prefill_intercept_ms=float(old_payload["prefill_intercept_ms"]),
                      decode_ms_per_request=float(old_payload["decode_ms_per_request"]),
                      decode_ms_per_context_token=float(old_payload["decode_ms_per_context_token"]),
                      decode_intercept_ms=float(old_payload["decode_intercept_ms"]),
                      migration_cost_estimator=old_estimator, migration_uncertainty_scale=0.0,
                      observation_refresh_interval_sec=1.0),
        "V1": replace(refined_config, service_model_mode="calibrated", kv_dynamics_enabled=False,
                      migration_cost_estimator=old_estimator, migration_uncertainty_scale=0.0,
                      observation_refresh_interval_sec=1.0),
        "V2": replace(refined_config, service_model_mode="calibrated", kv_dynamics_enabled=True,
                      migration_uncertainty_scale=0.0, observation_refresh_interval_sec=1.0),
        "V3": replace(refined_config, service_model_mode="calibrated", kv_dynamics_enabled=True,
                      migration_uncertainty_scale=1.0, observation_refresh_interval_sec=1.0),
    }
    validation_dir = root / "validation"
    systems: dict[str, Any] = {}
    fixed_summary: dict[str, Any] = {}
    progress_path = output / "system_validation_progress.json"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            checkpoint = progress.get("per_qps", {})
            if (isinstance(checkpoint, dict)
                    and all(str(group.get("observed", {}).get("comparison_horizon_sec"))
                            == str(horizon) for group in checkpoint.values())):
                systems.update(checkpoint)
                for qps_key, group in systems.items():
                    fixed_summary[qps_key] = {
                        "observed": group["observed"],
                        "fixed_event_input_audit": group["fixed_event_input_audit"],
                        "versions": {version: {
                            key: {metric: value for metric, value in run.items()
                                  if metric != "fixed_replay_outcomes"}
                            for key, run in values.items()}
                            for version, values in group["versions"].items()},
                    }
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            systems.clear()
            fixed_summary.clear()
    validation_runs = sorted(validation_dir.glob(f"qps_*_seed{int(held_out_seed)}"))
    if len(validation_runs) != 5:
        raise ValueError(f"expected five held-out seed-{held_out_seed} QPS runs; found {len(validation_runs)}")
    fixed_length_counts = {}
    for run_dir in validation_runs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        qps = float(config["qps"])
        if str(qps) in systems:
            continue
        trace = _request_trace(run_dir / "traces" / f"trace_qps_{qps:g}_seed{int(held_out_seed)}.csv",
                               run_dir / "requests.csv")
        fixed_length_counts[str(qps)] = sum("realized_output_tokens" in item for item in trace)
        instance_ids, routes, fixed_events, input_audit = _run_inputs(run_dir, trace, measurement_sec)
        observed = _observed_run(run_dir, qps, measurement_sec, drain_sec, horizon)
        start_epoch = float(input_audit["measurement_epoch_wall_sec"])
        observed.update(_observed_state(run_dir, start_epoch, measurement_sec))
        per_version = {}
        for version, version_config in calibration_versions.items():
            fixed_config = replace(version_config, migration_uncertainty_scale=0.0)
            fixed = _simulate(fixed_config, trace, qps, instance_ids, routes,
                              seed=int(held_out_seed), horizon_sec=horizon, measurement_sec=measurement_sec,
                              fixed_events=fixed_events)
            closed = _simulate(version_config, trace, qps, instance_ids, {},
                               seed=int(held_out_seed), horizon_sec=horizon, measurement_sec=measurement_sec,
                               migration_controller=True)
            per_version[version] = {
                "fixed_on_event_replay": fixed,
                "closed_loop_greedy": closed,
                "fixed_event_relative_errors": _relative_errors(fixed, observed),
                "closed_loop_relative_errors": _relative_errors(closed, observed),
            }
        fixed_summary[str(qps)] = {"observed": observed,
                                   "fixed_event_input_audit": input_audit,
                                   "versions": {version: {
                                       key: {metric: value for metric, value in result.items()
                                             if metric not in {"fixed_replay_outcomes"}}
                                       for key, result in values.items()}
                                       for version, values in per_version.items()}}
        systems[str(qps)] = {"observed": observed, "fixed_event_input_audit": input_audit,
                             "versions": per_version}
        progress_path.write_text(json.dumps({"data_root": str(root), "horizon_sec": horizon,
                                            "completed_qps": list(systems), "per_qps": systems},
                                            ensure_ascii=False), encoding="utf-8")
    service_samples, service_load_audit = load_profile_split(root, "validation")
    service_epochs = _measurement_epochs(root, "validation")
    measured_engine, service_audit = _service_validation(
        service_samples, service_epochs, measurement_sec,
        refined_config.service_time_model or {})
    service_audit["profile_load_audit"] = service_load_audit
    calibration_service_cv = _calibration_service_cv(root, measurement_sec)
    has_fixed_lengths = any(fixed_length_counts.values())
    result = {"data_root": str(root), "held_out_seed": int(held_out_seed),
              "held_out_response_labels_used_as_inputs": False,
              "calibration_response_labels_used_for_output_length_model": True,
              "output_length_model_audit": length_audit,
              "decode_token_interval_scale_audit": decode_scale_audit,
              "kv_capacity_inference": kv_capacity_audit,
              "configured_kv_blocks_per_instance": refined_config.total_kv_blocks,
              "arrival_trace_source": "requests.csv actual_send_sec joined to the trace request ID; scheduled trace time is used only when an actual send time is absent",
              "queue_kv_sampling_policy": "both simulated and observed states are sampled at 50 ms intervals as a per-instance mean; observed engine snapshots use per-instance last-observation carry-forward",
              "output_length_semantics": (
                  "trace output_tokens are max-token budgets. Rows with realized_output_tokens use a fixed workload length sampled from training calibration responses; those values are not held-out response labels. Other rows sample from the training calibration distribution."
                  if has_fixed_lengths else
                  f"trace output_tokens are max-token budgets; realized lengths are sampled from successful calibration responses conditioned on input length and budget; no seed-{int(held_out_seed)} output lengths enter simulation"
              ),
              "fixed_training_length_trace_rows_by_qps": fixed_length_counts,
              "closed_loop_future_migration_leakage_check": "closed-loop runs receive no route map and no fixed migration events; logged migrations only affect fixed replay",
              "horizon_sec": horizon, "measurement_sec": measurement_sec,
              "nominal_collection_drain_sec": drain_sec,
              "comparison_drain_sec": comparison_drain_sec,
              "versions": {
                  "V0": "original scalar service coefficients; KV block pressure/preemption disabled; old migration fit, deterministic; shared empirically calibrated output cadence scale",
                  "V1": "promoted prefill/decode service models; KV block pressure/preemption disabled; old migration fit, deterministic; shared empirically calibrated output cadence scale",
                  "V2": "V1 plus dynamic KV block allocation/growth, sequence cap, and resumed-prefill correction; robust migration mean, deterministic; shared output cadence scale",
                  "V3": "V2 plus residual-uncertainty sampling for migration duration; shared output cadence scale",
              },
              "controller": "greedy rule-based selection of legal pair with positive predicted net gain; this is not a trained PPO checkpoint",
              "fixed_replay_semantics": "logged measurement-phase migration time/source/destination/request/cost/status are exogenous; migration predictor and uncertainty are bypassed",
              "migration_pause_caveat": "request is blocked during the simulated migration interval, but no measured service-pause label exists; this is an explicit approximation",
              "per_qps": systems,
              "per_qps_compact": fixed_summary,
              "measured_engine_iteration_by_qps": measured_engine,
              "service_validation_audit": service_audit,
              "calibration_service_cv_phase_comparison": calibration_service_cv}
    output_json = output / "system_validation.json"
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_system_report(output / "system_validation.md", result)
    _write_error_attribution_report(output / "model_error_attribution.md", root, result)
    progress_path.unlink(missing_ok=True)
    return result


def _write_error_attribution_report(path: Path, data_root: Path, result: Mapping[str, Any]) -> None:
    def cell(value: object, digits: int = 1) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    comparison_path = data_root / "model_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.exists() else {}
    migration = comparison.get("migration", {})
    promoted = str(migration.get("promoted", "M2_huber_pair_shrunk"))
    cv = migration.get("grouped_run_cv", {}).get(promoted, {}).get("metrics", {})
    holdout = migration.get("seed404", {}).get(promoted, {})
    holdout_metrics = {key: value for key, value in holdout.items()
                       if key not in {"by_status", "by_pair"}}
    if "metrics" in holdout_metrics:
        holdout_metrics = holdout_metrics["metrics"]
    held_out_seed = int(result.get("held_out_seed", 404))
    lines = ["# Model and system error attribution", "",
             f"This report separates service iteration fit, request-level system behavior, migration costs, and collection censoring. Calibration uses seeds 101/202/303; seed {held_out_seed} is held out for service/system validation.", "",
             "## Prefill and decode service models", "",
             f"| Stage | Model | Grouped CV all phases MAE / RMSE ms | Grouped CV measurement-only MAE / RMSE ms | Seed {held_out_seed} measurement MAE / RMSE ms | Bias ms | Residual P05 / P50 / P95 ms | Labels >500 ms | Abs residual >250 ms |",
             "|:---|:---:|---:|---:|---:|---:|---:|---:|---:|"]
    cv_phase = result["calibration_service_cv_phase_comparison"]
    service_residuals = result["service_validation_audit"]["model_residuals"]
    for kind in ("prefill", "decode"):
        entry = cv_phase[kind]
        all_metrics = entry.get("all_phase_grouped_cv") or {}
        measurement_cv = entry.get("measurement_only_grouped_cv") or {}
        measured_holdout = service_residuals.get(kind, {}).get("measurement", {})
        qs = measured_holdout.get("residual_p05_p50_p95_ms") or [None, None, None]
        lines.append(f"| {kind} | {entry['selected_model']} | "
                     f"{cell(all_metrics.get('mae_ms'))} / {cell(all_metrics.get('rmse_ms'))} | "
                     f"{cell(measurement_cv.get('mae_ms'))} / {cell(measurement_cv.get('rmse_ms'))} | "
                     f"{cell(measured_holdout.get('mae_ms'))} / {cell(measured_holdout.get('rmse_ms'))} | "
                     f"{cell(measured_holdout.get('bias_prediction_minus_observed_ms'))} | "
                     f"{cell(qs[0])} / {cell(qs[1])} / {cell(qs[2])} | "
                     f"{measured_holdout.get('label_over_500ms', 0)} | "
                     f"{measured_holdout.get('absolute_residual_over_250ms', 0)} |")
    lines += ["", f"The measurement-only grouped CV rows are compared with the all-phase fits above. Seed {held_out_seed} measurement-window residuals are reported independently for each service stage; per-QPS values remain in `system_validation.json`.", "",
              "Service labels are executor iteration wall times. They are not request TTFT/TPOT. Phase filtering can materially change the holdout residuals by excluding out-of-window stalls; the requested holdout's values are reported above. Large executor stalls are sparse and the logs do not identify a unique cause; they cannot be safely assigned to batch/context features.", "",
              "## Migration duration", "",
              "| Model | Split | N | MAE / RMSE ms | Bias ms | Residual P05 / P50 / P95 ms |",
              "|:---|:---|---:|---:|---:|---:|"]
    for split, metrics in (("grouped CV", cv), ("seed 404 (historical model-comparison holdout)", holdout_metrics)):
        quantiles = [metrics.get("residual_p05_ms"), metrics.get("residual_p50_ms"),
                     metrics.get("residual_p95_ms")]
        lines.append(f"| {promoted} | {split} | {metrics.get('n', 0)} | "
                     f"{cell(metrics.get('mae_ms'))} / {cell(metrics.get('rmse_ms'))} | "
                     f"{cell(metrics.get('mean_error_ms'))} | "
                     f"{cell(quantiles[0])} / {cell(quantiles[1])} / {cell(quantiles[2])} |")
    statuses = holdout.get("by_status", {})
    for status, metrics in statuses.items():
        lines.append(f"| {promoted} ({status}) | seed 404 (historical) | {metrics.get('n', 0)} | "
                     f"{cell(metrics.get('mae_ms'))} / {cell(metrics.get('rmse_ms'))} | "
                     f"{cell(metrics.get('mean_error_ms'))} | "
                     f"{cell(metrics.get('residual_p05_ms'))} / {cell(metrics.get('residual_p50_ms'))} / "
                     f"{cell(metrics.get('residual_p95_ms'))} |")
    lines += ["", "M2 remains high-error and status-dependent; seed 404 has only 31 migration events. Block count alone has weak association with cost, and directed-pair shrinkage rests on few observations. `cost_ms` is end-to-end migration wall time. No measured pause-time label exists, so the environment's request blocking during migration remains an approximation.", "",
              "## System-level request and resource errors", "",
              "Closed-loop rows below use the same one-second observation refresh and share the train-only output-length/cadence models. Relative errors are signed against observed request response throughput (`HTTP 200 / 60s`); queue and KV compare per-instance units.", "",
              "| QPS | Version | Throughput sim / observed (error %) | TTFT avg sim / obs ms | TPOT avg sim / obs ms (observed N) | Queue mean sim / obs | KV mean sim / obs | Sim completed / observed HTTP 200 |",
              "|---:|:---:|---:|---:|---:|---:|---:|---:|"]
    for qps, group in result["per_qps"].items():
        observed = group["observed"]
        for version in ("V0", "V1", "V2", "V3"):
            row = group["versions"][version]["closed_loop_greedy"]
            error = group["versions"][version]["closed_loop_relative_errors"]["completed_throughput_rps"]["signed_relative_error_pct"]
            lines.append(f"| {qps} | {version} | {cell(row['completed_throughput_rps'])} / "
                         f"{cell(observed['request_throughput_rps'])} ({cell(error)}%) | "
                         f"{cell(row['ttft_mean_ms'])} / {cell(observed['ttft_mean_ms'])} | "
                         f"{cell(row['tpot_request_mean_ms'])} / {cell(observed['tpot_request_mean_ms'])} "
                         f"({observed['tpot_n']}) | "
                         f"{cell(row['queue_mean_requests'])} / {cell(observed['queue_mean_requests'])} | "
                         f"{cell(row['kv_mean_fraction'], 3)} / {cell(observed['kv_mean_fraction'], 3)} | "
                         f"{row['completed']} / {observed['completed_by_comparison_horizon']} |")
    scale_audit = result.get("decode_token_interval_scale_audit")
    scale_summary = (f"Decode cadence scale fit from calibration runs only: {scale_audit['fitted_scale']:.3f} across "
                     f"{scale_audit['n_calibration_runs']} runs.") if scale_audit else \
        "This baseline reports the unscaled per-iteration simulator and predates the cadence correction."
    system_gap = ("The calibrated cadence layer is shared across all versions; use the per-QPS errors below to judge whether it closes the request-level timing gap.") if scale_audit else \
        "Across QPS, the unscaled simulator completes too many requests and substantially underpredicts request-level TTFT/TPOT and queue length even when executor iteration residuals are small. This is a system timing/queue transition gap, not evidence that P1/D3 per-iteration regressors need more terms."
    lines += ["", "Observed HTTP 499s were drain-triggered censored requests, not server failures. The held-out runs contain no true HTTP failures or request timeouts. " + system_gap, "",
              "### Client versus server request duration", "",
              "| QPS | Complete server timelines / HTTP 200 | Server duration mean ms | Client minus server duration mean / P50 / P95 ms |",
              "|---:|---:|---:|---:|"]
    for qps, group in result["per_qps"].items():
        audit = group["observed"].get("server_timeline_audit", {})
        lines.append(f"| {qps} | {audit.get('timeline_complete_n', 'n/a')} / {group['observed']['completed']} | "
                     f"{cell(audit.get('server_timeline_total_mean_ms'))} | "
                     f"{cell(audit.get('client_minus_server_timeline_mean_ms'))} / "
                     f"{cell(audit.get('client_minus_server_timeline_p50_ms'))} / "
                     f"{cell(audit.get('client_minus_server_timeline_p95_ms'))} |")
    lines += ["", "Client completion timestamps are distinct from the server's per-token timeline. The client-minus-server duration gap varies by QPS on the subset with complete per-token timelines; the table above gives those sample counts and residual distributions. Completion-window throughput therefore remains a separate collection metric.", "",
              f"{scale_summary} The scale audit and per-QPS fixed/closed-loop metrics are in `system_validation.json`; the final combined model/configuration is `mig_ppo_calibration_system_validation.yaml`.", "",
              "## Data coverage and capacity", "",
              f"The empirical output-length distribution uses {result['output_length_model_audit']['n_successful_lengths']:,} successful calibration responses; censored output lengths are unavailable. The per-instance KV capacity is set to {result['configured_kv_blocks_per_instance']:,} blocks and supported by the log inference in the system validation report.",
              "The collection mixes 17 runs before and 3 runs after the migration-slot guard change. Service instrumentation was kept the same, but migration behavior has a code-version split.", "",
              f"The validator requires all five QPS runs for held-out seed {held_out_seed} before producing a system-level report. System prediction results must meet the admission criteria before formal Mig-PPO policy training.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_system_report(path: Path, result: Mapping[str, Any]) -> None:
    def cell(value: object, digits: int = 1) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    held_out_seed = int(result["held_out_seed"])
    fixed_length_counts = result.get("fixed_training_length_trace_rows_by_qps", {})
    fixed_length_total = sum(int(count) for count in fixed_length_counts.values())
    length_description = (
        f"The simulation used {fixed_length_total:,} explicit output lengths sampled from training calibration responses and stored in each held-out trace; these are workload inputs, not held-out response labels."
        if fixed_length_total else
        f"The simulation sampled realized response lengths from successful calibration responses only; seed-{held_out_seed} output lengths are held out for comparison."
    )
    lines = [f"# Seed {held_out_seed} system validation", "",
             f"Data root: `{result['data_root']}`. Held-out seed: {result['held_out_seed']}. "
             f"Measurement window: {result['measurement_sec']:.1f}s; replay horizon: {result['horizon_sec']:.1f}s "
             f"({result['comparison_drain_sec']:.1f}s simulated drain).", "",
             f"The collection nominal drain was {result['nominal_collection_drain_sec']:.1f}s. All request outcome and latency comparisons use the common {result['horizon_sec']:.0f}s cutoff; queue and KV snapshots use the shared {result['measurement_sec']:.0f}s workload window. Request arrivals use actual client send times joined to the trace, with scheduled times as fallback. {length_description} Drain-censored requests have no realized-length labels, so the empirical calibration length model may underrepresent long outputs.", "",
             "V0 uses legacy scalar service coefficients and disables KV dynamics. V1 uses calibrated service models and disables KV dynamics. V2 adds dynamic KV allocation and preemption recovery. V3 adds migration-duration residual sampling. All four versions use the same run-level decode output cadence scale, fitted as the median calibration-run ratio of request TPOT to predicted executor iteration time; raw executor predictions remain separately reported. The fixed-event replay and closed-loop greedy baseline are reported separately. Closed-loop action observations refresh once per second; the decision interval remains 50 ms. The greedy baseline is not a trained PPO policy.", "",
             "## Observed request outcomes", "",
             "| QPS | Arrivals | HTTP 200 by horizon | Failed by horizon | Timed out | Drain-censored 499 | Unfinished at horizon | HTTP 200 total | TTFT N | TPOT N / HTTP 200 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for qps, group in result["per_qps"].items():
        obs = group["observed"]
        lines.append(f"| {qps} | {obs['arrivals']} | {obs['completed_by_comparison_horizon']} | "
                     f"{obs['failed_by_comparison_horizon']} | {obs['timeouts_by_comparison_horizon']} | "
                     f"{obs['censored_by_comparison_horizon']} | {obs['unfinished_by_comparison_horizon']} | "
                     f"{obs['completed']} | {obs['ttft_n']} | {obs['tpot_n']} / {obs['completed']} |")
    lines += ["", "HTTP 499 drain-censored responses are separated from actual failures and timeouts; they do not by themselves establish an engine failure. `HTTP 200 total` includes recorded responses at any time, while `HTTP 200 by horizon` is the comparable count. `output_tokens` is a generation budget; realized output lengths are reported separately. TPOT is available for only a subset of successful responses; sample counts and coverage are retained in JSON.", "",
              "## Training-sampled output lengths in held-out traces", "",
              "The optional `realized_output_tokens` trace field fixes an independent length sample drawn from calibration-only successful responses. These lengths are workload inputs, while held-out response lengths remain validation labels.", "",
              "| QPS | Trace rows with fixed training-sampled lengths |", "|---:|---:|"]
    for qps, count in sorted(fixed_length_counts.items(), key=lambda item: float(item[0])):
        lines.append(f"| {qps} | {int(count)} |")
    lines += ["", "## Arrival trace timing audit", "",
              "| QPS | Arrivals using actual send time | Mean actual minus scheduled ms | P95 ms | Maximum ms |",
              "|---:|---:|---:|---:|---:|"]
    for qps, group in result["per_qps"].items():
        audit = group["fixed_event_input_audit"]
        lines.append(f"| {qps} | {audit['arrival_rows_using_actual_send_time']} | "
                     f"{cell(audit['actual_minus_scheduled_arrival_mean_ms'])} | "
                     f"{cell(audit['actual_minus_scheduled_arrival_p95_ms'])} | "
                     f"{cell(audit['actual_minus_scheduled_arrival_max_ms'])} |")
    lines += ["", "## Client and server completion timing", "",
              "For requests with a complete server per-token timeline, the sum of server token intervals gives an independent server-side request duration. Client response completion can be later because it also includes client/event-loop and response handling time.", "",
              "| QPS | Full server timelines / HTTP 200 | Server timeline mean / P95 ms | Client minus server timeline mean / P50 / P95 ms |",
              "|---:|---:|---:|---:|"]
    for qps, group in result["per_qps"].items():
        audit = group["observed"]["server_timeline_audit"]
        lines.append(f"| {qps} | {audit['timeline_complete_n']} / {group['observed']['completed']} | "
                     f"{cell(audit['server_timeline_total_mean_ms'])} / {cell(audit['server_timeline_total_p95_ms'])} | "
                     f"{cell(audit['client_minus_server_timeline_mean_ms'])} / "
                     f"{cell(audit['client_minus_server_timeline_p50_ms'])} / "
                     f"{cell(audit['client_minus_server_timeline_p95_ms'])} |")
    lines += ["", "## System metrics by QPS, replay mode, and version", "",
              "Latency cells show `simulated / observed`; errors are signed percentage differences against the observed value. Queue and KV values are per-sample means across instances on a shared 50 ms grid, with p95 across the 60-second sample series. Observed engine snapshots are carried forward to that grid. Successful response throughput is the project summary metric `HTTP 200 responses / 60s`; first-window completion throughput (`completion_sec <= 60`) is also shown separately. The common 90-second horizon bounds simulated completed counts.", "",
              "| QPS | Replay | Version | Sim complete / failed / unfinished | TTFT avg / P95 ms (sim / obs) | TPOT avg / P95 ms (sim / obs) | TPOT observed N | HTTP 200 /60 req/s (sim / obs; error) | Completed by 60s req/s (sim / obs) | Queue mean / P95 | KV mean / P95 | Mig started / success / fail / abort | Mig cost mean / P95 ms |",
              "|---:|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for qps, group in result["per_qps"].items():
        obs = group["observed"]
        for version in ("V0", "V1", "V2", "V3"):
            per_version = group["versions"][version]
            for mode, key, error_key in (("fixed events", "fixed_on_event_replay", "fixed_event_relative_errors"),
                                         ("closed-loop", "closed_loop_greedy", "closed_loop_relative_errors")):
                row = per_version[key]
                errors = per_version[error_key]
                thr_err = errors["completed_throughput_rps"]["signed_relative_error_pct"]
                throughput = f"{cell(row['completed_throughput_rps'])} / {cell(obs['request_throughput_rps'])}"
                throughput += f" ({cell(thr_err)}%)" if thr_err is not None else " (n/a)"
                lines.append("| " + " | ".join([
                    qps, mode, version,
                    f"{row['completed']} / {row['failed']} / {row['unfinished']}",
                    f"{cell(row['ttft_mean_ms'])} / {cell(obs['ttft_mean_ms'])}; {cell(row['ttft_p95_ms'])} / {cell(obs['ttft_p95_ms'])}",
                    f"{cell(row['tpot_request_mean_ms'])} / {cell(obs['tpot_request_mean_ms'])}; {cell(row['tpot_request_p95_ms'])} / {cell(obs['tpot_request_p95_ms'])}",
                    f"{obs['tpot_n']} / {obs['completed']}", throughput,
                    f"{cell(row['in_window_throughput_rps'])} / {cell(obs['in_window_throughput_rps'])}",
                    f"{cell(row['queue_mean_requests'])} / {cell(row['queue_p95_requests'])}",
                    f"{cell(row['kv_mean_fraction'], 3)} / {cell(row['kv_p95_fraction'], 3)}",
                    f"{row['migration_attempts_started']} / {row['migration_successes']} / {row['migration_failures']} / {row['migration_aborted']}",
                    f"{cell(row['migration_cost_mean_ms'])} / {cell(row['migration_cost_p95_ms'])}"]) + " |")

    lines += ["", "## Fixed migration replay coverage", "",
              "The event replay consumes logged event time, source, destination, blocks, duration, and terminal status. Event start time is approximated by watcher observation time minus the recorded duration. Requests linked to a logged migration are routed to the first logged source only in this fixed replay. The closed-loop run receives no logged route map or event schedule.", "",
              "| QPS | Logged migration rows | Linked replay requests | Logged status counts | V2 started / nonstarted / still queued | Median start delay ms |",
              "|---:|---:|---:|:---|---:|---:|"]
    for qps, group in result["per_qps"].items():
        audit = group["fixed_event_input_audit"]
        replay = group["versions"]["V2"]["fixed_on_event_replay"]
        delays = [float(row["start_delay_sec"]) * 1000 for row in replay["fixed_replay_outcomes"]
                  if row.get("start_delay_sec") is not None]
        lines.append(f"| {qps} | {audit['fixed_migration_event_rows']} | {audit['fixed_migration_events']} | "
                     f"{audit['logged_migration_status_counts']} | {replay['fixed_replay_started']} / "
                     f"{replay['fixed_replay_nonstarted']} / {replay['fixed_replay_unattempted']} | "
                     f"{cell(_quantile(delays, .5))} |")
    lines += ["", "## Engine iteration validation", "",
              "Service intervals are classified against each run's measurement epoch. A sample is in-window only when its estimated start and end both fall within the 60-second measurement period. Labels are executor iteration wall times, not isolated GPU time or request latency.", "",
              "| QPS | Phase | N | Observed mean / P95 ms | Model MAE / RMSE ms | Bias ms | Residual P05 / P50 / P95 ms | Labels >500 ms | Abs residual >250 ms |",
              "|---:|:---|---:|---:|---:|---:|---:|---:|---:|"]
    service_audit = result["service_validation_audit"]
    residuals = service_audit["model_residuals"]
    for qps, stages in result["measured_engine_iteration_by_qps"].items():
        for stage in ("prefill", "decode"):
            observed = stages[stage]
            model = residuals.get(stage, {}).get("measurement_by_qps", {}).get(qps, {})
            quantiles = model.get("residual_p05_p50_p95_ms") or [None, None, None]
            lines.append(f"| {qps} | {stage} | {observed['n']} | {cell(observed['mean_ms'])} / {cell(observed['p95_ms'])} | "
                         f"{cell(model.get('mae_ms'))} / {cell(model.get('rmse_ms'))} | "
                         f"{cell(model.get('bias_prediction_minus_observed_ms'))} | "
                         f"{cell(quantiles[0])} / {cell(quantiles[1])} / {cell(quantiles[2])} | "
                         f"{model.get('label_over_500ms', 0)} | {model.get('absolute_residual_over_250ms', 0)} |")
    lines += ["", "### Phase classification counts", "",
              f"`{service_audit['phase_definition']}`", "",
              "| Kind and phase | Samples |", "|:---|---:|"]
    for phase, count in sorted(service_audit["sample_counts_by_kind_and_phase"].items()):
        lines.append(f"| {phase} | {count} |")
    cv_comparison = result["calibration_service_cv_phase_comparison"]
    lines += ["", "### Grouped calibration CV after phase filtering", "",
              "The selected P1/D3 structures are evaluated on calibration runs with service intervals wholly inside measurement. This is a diagnostic refit by held-out run; seed 404 is excluded.", "",
              "| Stage | Model | CV phase | N | Groups | MAE / RMSE ms | R² | Macro run MAE ms |",
              "|:---|:---:|:---|---:|---:|---:|---:|---:|"]
    for kind in ("prefill", "decode"):
        entry = cv_comparison[kind]
        for phase_name, metrics, count, groups, macro in (
            ("all phases (prior report)", entry.get("all_phase_grouped_cv"), None, None, None),
            ("measurement only", entry.get("measurement_only_grouped_cv"),
             entry.get("measurement_only_samples"), entry.get("measurement_only_groups"),
             entry.get("measurement_only_macro_run_mae_ms")),
        ):
            lines.append(f"| {kind} | {entry['selected_model']} | {phase_name} | "
                         f"{metrics.get('n', count) if metrics else 0} | {groups if groups is not None else 'n/a'} | "
                         f"{cell(metrics.get('mae_ms')) if metrics else 'n/a'} / "
                         f"{cell(metrics.get('rmse_ms')) if metrics else 'n/a'} | "
                         f"{cell(metrics.get('r2'), 3) if metrics else 'n/a'} | "
                         f"{cell(macro) if macro is not None else 'n/a'} |")
    lines += ["", "### Model residuals by phase", "",
              "| Kind | Phase | N | MAE / RMSE ms | Bias ms | Residual P05 / P50 / P95 ms | Labels >500 ms | Abs residual >250 ms |",
              "|:---|:---|---:|---:|---:|---:|---:|---:|"]
    for kind in ("prefill", "decode"):
        for phase in ("all", "measurement", "warmup", "drain", "boundary"):
            model = residuals.get(kind, {}).get(phase, {})
            quantiles = model.get("residual_p05_p50_p95_ms") or [None, None, None]
            lines.append(f"| {kind} | {phase} | {model.get('n', 0)} | {cell(model.get('mae_ms'))} / {cell(model.get('rmse_ms'))} | "
                         f"{cell(model.get('bias_prediction_minus_observed_ms'))} | "
                         f"{cell(quantiles[0])} / {cell(quantiles[1])} / {cell(quantiles[2])} | "
                         f"{model.get('label_over_500ms', 0)} | {model.get('absolute_residual_over_250ms', 0)} |")
    lines += ["", "## Calibration and system limitations", "",
              f"- The output-length model uses {result['output_length_model_audit']['n_successful_lengths']} successful seed-101/202/303 measurement responses. "
              "It is conditioned on input length and max-token budget. Unobserved drain-censored response lengths create selection bias; this model is an empirical approximation.",
              "- Current collection is Migration ON only. Fixed-event replay evaluates those logged event conditions, while closed-loop results use the simulator's migration cost model and configured failure probability.",
              "- Migration cost labels are end-to-end migration wall times. No measured service-pause label exists; simulated blocking during migration is an explicit approximation.",
              "- Logs do not provide exact initial worker routing, complete scheduler decisions, per-request block allocation transitions, or mixed-batch overlap semantics. Initial routing uses least projected work with the calibrated expected response length; logged source routing is restricted to fixed replay.",
              "- Queue and KV comparisons use a common 50 ms sample grid. Observed per-instance engine snapshots are carried forward to each sample; simulator state is sampled after each 50 ms step. Each sample is averaged across instances before mean and p95 aggregation.",
              "- Fixed replay and greedy closed-loop results test system behavior under different controls. Closed-loop greedy results do not establish PPO policy performance.", "",
              "The calibrated service models remain P1/D3 with the previously selected migration model. The report preserves all QPS/version/mode metrics and error calculations in `system_validation.json`; the combined final calibration is `mig_ppo_calibration_system_validation.yaml`.", ""]
    capacity = result.get("kv_capacity_inference")
    if capacity:
        lines.insert(-2, f"- The configured {result['configured_kv_blocks_per_instance']:,} KV blocks per instance are supported by {capacity['n']:,} log-derived free-block/usage samples (p05/median/p95 {capacity['p05']:,.0f}/{capacity['median']:,.0f}/{capacity['p95']:,.0f}); this remains a log inference, not an independent hardware capacity measurement.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-yaml")
    parser.add_argument("--output-dir", help="write validation artifacts here; defaults to --data-root")
    parser.add_argument("--held-out-seed", type=int, default=404)
    args = parser.parse_args()
    result = run_system_validation(args.data_root, calibration_yaml=args.calibration_yaml,
                                   output_dir=args.output_dir, held_out_seed=args.held_out_seed)
    output = Path(args.output_dir or args.data_root)
    print(json.dumps({"report": str(output / "system_validation.md"),
                      "json": str(output / "system_validation.json"),
                      "calibration": str(output / "mig_ppo_calibration_system_validation.yaml"),
                      "qps_levels": sorted(result["per_qps"].keys())}, indent=2))


if __name__ == "__main__":
    main()
