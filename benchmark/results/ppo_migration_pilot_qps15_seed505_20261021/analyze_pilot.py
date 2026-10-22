#!/usr/bin/env python3
"""Create a paired OFF/ON pilot summary from preserved raw Llumnix artifacts."""

from __future__ import annotations

import ast
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PILOT_ROOT = Path(__file__).resolve().parent
CAPACITY_BLOCKS = 22084


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def quantile(values: list[float], probability: float):
    values = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def stats(values: list[float]) -> dict:
    clean = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "p50": quantile(clean, 0.50),
        "p95": quantile(clean, 0.95),
        "max": max(clean) if clean else None,
    }


def bool_value(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_timeline(row: dict[str, str]) -> list[tuple[float, float]]:
    try:
        value = json.loads(row.get("per_token_latency_json") or "[]")
    except (ValueError, TypeError):
        return []
    result = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                timestamp, latency = number(item[0]), number(item[1])
                if timestamp is not None and latency is not None:
                    result.append((timestamp, latency))
    return result


def analyze_mode(mode: str, mono_wall_offset: float, clock_metadata: dict) -> dict:
    path = PILOT_ROOT / mode
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    rows_all = read_csv(path / "requests.csv")
    rows = [row for row in rows_all if row.get("phase") == "measurement"]
    if not rows:
        raise RuntimeError(f"No measurement request rows for {mode}")
    send_bases = [
        float(row["actual_send_wall_time"]) - float(row["actual_send_sec"])
        for row in rows if row.get("actual_send_wall_time") and row.get("actual_send_sec")
    ]
    phase_start_wall = statistics.median(send_bases)
    phase_start_monotonic = phase_start_wall - mono_wall_offset
    trace_hash = config["trace_sha256"]

    client_completion = []
    server_end = []
    server_ttft_90 = []
    server_tpot_90 = []
    server_elapsed_90 = []
    response_lag_90 = []
    pre_handler_delay_90 = []
    output_tokens_90 = []
    timeline_complete_by_horizon = []
    client_response_lag_all = []
    outcomes_by_status = Counter(row.get("status_code") or "none" for row in rows)
    completion_times = []
    arrival_times = []
    server_request_records = []

    for row in rows:
        sent = number(row.get("actual_send_sec"))
        received = number(row.get("completion_sec"))
        if sent is not None:
            arrival_times.append(sent)
        if received is not None:
            completion_times.append(received)
        if row.get("status_code") != "200":
            continue
        timeline = parse_timeline(row)
        if not timeline:
            continue
        first_perf, first_latency_ms = timeline[0]
        last_perf = timeline[-1][0]
        handler_start_perf = first_perf - first_latency_ms / 1000.0
        handler_start_sec = handler_start_perf - phase_start_monotonic
        first_token_sec = first_perf - phase_start_monotonic
        end_sec = last_perf - phase_start_monotonic
        elapsed_ms = (last_perf - handler_start_perf) * 1000.0
        server_end.append(end_sec)
        within_horizon = end_sec <= 90.0
        within_work = end_sec <= 60.0
        if received is not None:
            client_completion.append(received)
            if received - end_sec >= -0.1:
                client_response_lag_all.append((received - end_sec) * 1000.0)
        if within_horizon:
            output_tokens = number(row.get("server_output_tokens"))
            ttft = number(row.get("server_ttft_ms"))
            tpot = number(row.get("server_tpot_ms"))
            if ttft is not None:
                server_ttft_90.append(ttft)
            if tpot is not None:
                server_tpot_90.append(tpot)
            if elapsed_ms >= 0:
                server_elapsed_90.append(elapsed_ms)
            if sent is not None:
                pre_handler_delay_90.append((handler_start_sec - sent) * 1000.0)
            if received is not None and received >= end_sec:
                response_lag_90.append((received - end_sec) * 1000.0)
            if output_tokens is not None:
                output_tokens_90.append(output_tokens)
            if bool_value(row.get("timeline_matches_output")):
                timeline_complete_by_horizon.append(row)
        server_request_records.append({
            "request_id": row.get("request_id"),
            "client_send_sec": sent,
            "server_handler_start_sec": handler_start_sec,
            "server_first_token_sec": first_token_sec,
            "server_final_token_sec": end_sec,
            "client_response_sec": received,
            "server_elapsed_ms": elapsed_ms,
            "server_final_token_to_client_response_ms": (
                (received - end_sec) * 1000.0 if received is not None else None
            ),
            "server_completed_by_60_sec": within_work,
            "server_completed_by_90_sec": within_horizon,
            "actual_output_tokens": number(row.get("server_output_tokens")),
            "timeline_count": len(timeline),
            "timeline_matches_output": bool_value(row.get("timeline_matches_output")),
        })

    horizon_end = 90.0
    client_response_le_horizon = sum(value <= horizon_end for value in completion_times)
    server_done_le_work = sum(value <= 60.0 for value in server_end)
    server_done_le_horizon = sum(value <= horizon_end for value in server_end)
    server_done_after_horizon = sum(value > horizon_end for value in server_end)
    abort_censored = outcomes_by_status.get("499", 0)
    other_non200 = sum(count for status, count in outcomes_by_status.items() if status not in {"200", "499"})

    req_count = len(rows)
    client_success = outcomes_by_status.get("200", 0)
    sends_complete = sum(number(row.get("actual_send_sec")) is not None for row in rows)
    client_e2e = [
        number(row.get("client_e2e_ms")) for row in rows
        if row.get("status_code") == "200" and number(row.get("client_e2e_ms")) is not None
    ]
    ttft_all_success = [
        number(row.get("server_ttft_ms")) for row in rows
        if row.get("status_code") == "200" and number(row.get("server_ttft_ms")) is not None
    ]
    tpot_all_valid = [
        number(row.get("server_tpot_ms")) for row in rows
        if row.get("status_code") == "200" and number(row.get("server_tpot_ms")) is not None
    ]

    instance_rows = read_csv(path / "server_logs" / f"{mode}_instance_instance.csv")
    window_end_wall = phase_start_wall + 60.0
    instance_rows = [
        row for row in instance_rows
        if (number(row.get("timestamp")) is not None
            and phase_start_wall <= number(row.get("timestamp")) <= window_end_wall)
    ]
    queue_field = "num_waiting_requests"
    running_field = "num_running_requests"
    queue_values = [number(row.get(queue_field)) for row in instance_rows]
    running_values = [number(row.get(running_field)) for row in instance_rows]
    cache_values = [number(row.get("gpu_cache_usage")) for row in instance_rows]
    block_values = []
    engine_samples: dict[str, list[float]] = defaultdict(list)
    per_time: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_instance: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in instance_rows:
        timestamp = number(row.get("timestamp"))
        relative = timestamp - phase_start_wall if timestamp is not None else None
        bin_id = math.floor(relative / 5.0) if relative is not None else None
        instance_id = row.get("instance_id", "unknown")
        for key, field in (("waiting", queue_field), ("running", running_field),
                           ("kv_usage_pct", "gpu_cache_usage")):
            value = number(row.get(field))
            if value is not None:
                if key == "kv_usage_pct" and value <= 1.0:
                    value *= 100.0
                per_instance[instance_id][key].append(value)
                if bin_id is not None:
                    per_time[bin_id][key].append(value)
        available = number(row.get("num_available_gpu_blocks"))
        if available is not None:
            block_values.append(max(0.0, CAPACITY_BLOCKS - available) / CAPACITY_BLOCKS * 100.0)
        try:
            profiling = ast.literal_eval(row.get("profiling_data", ""))
            if isinstance(profiling, (tuple, list)) and len(profiling) >= 4:
                kind = str(profiling[0]).strip().lower()
                latency = number(profiling[3])
                if kind in {"prefill", "decode"} and latency is not None and latency > 0:
                    engine_samples[kind].append(latency)
        except (ValueError, SyntaxError, TypeError):
            pass

    time_means = {}
    for metric in ("waiting", "running", "kv_usage_pct"):
        values = [statistics.fmean(group[metric]) for group in per_time.values() if group.get(metric)]
        time_means[metric] = stats(values)

    migration_rows = read_csv(path / "migration_events.csv")
    migration_rows = [row for row in migration_rows if row.get("run_id") == config["run_id"]]
    migration_by_phase = {}
    for phase in ("warmup", "measurement"):
        subset = [row for row in migration_rows if row.get("phase") == phase]
        costs = [number(row.get("cost_ms")) for row in subset if number(row.get("cost_ms")) is not None]
        blocks = [number(row.get("blocks")) for row in subset if number(row.get("blocks")) is not None]
        migration_by_phase[phase] = {
            "events": len(subset),
            "statuses": dict(Counter(row.get("status", "") for row in subset)),
            "successes": sum(bool_value(row.get("success")) for row in subset),
            "failures_or_aborts": sum(not bool_value(row.get("success")) for row in subset),
            "cost_ms": stats(costs),
            "blocks": stats(blocks),
            "pause_time_ground_truth_present": False,
        }

    # All metrics below retain the seed 505 request-level paired cohort and 90s server horizon.
    return {
        "mode": mode,
        "run_id": config["run_id"],
        "trace_sha256": trace_hash,
        "measurement": {
            "trace_requests": config["trace_requests"],
            "requests_sent": sends_complete,
            "requested_qps": config["qps"],
            "actual_send_qps": ((len(arrival_times) - 1) / (max(arrival_times) - min(arrival_times)))
                if len(arrival_times) > 1 and max(arrival_times) > min(arrival_times) else None,
            "arrival_window_sec": 60.0,
            "shared_result_horizon_sec": 90.0,
            "client_response_status_counts": dict(outcomes_by_status),
            "client_responses_received_by_90_sec": client_response_le_horizon,
            "client_responses_received_after_90_sec": sum(value > 90.0 for value in completion_times),
            "client_timeout_count": sum(bool_value(row.get("timeout")) for row in rows),
            "other_http_or_transport_failures": other_non200,
            "abort_censored_http_499": abort_censored,
            "client_http_200_count": client_success,
            "client_e2e_ms_for_http_200": stats(client_e2e),
            "server_final_token_completions_by_60_sec": server_done_le_work,
            "server_final_token_completions_60_to_90_sec": server_done_le_horizon - server_done_le_work,
            "server_final_token_completions_by_90_sec": server_done_le_horizon,
            "server_final_token_completions_after_90_sec": server_done_after_horizon,
            "server_unfinished_at_90_sec": max(0, req_count - server_done_le_horizon),
            "server_completed_throughput_by_60_per_arrival_window_sec": server_done_le_work / 60.0,
            "server_completed_throughput_by_90_per_arrival_window_sec": server_done_le_horizon / 60.0,
            "server_ttft_ms_all_http_200": stats(ttft_all_success),
            "server_ttft_ms_server_completed_by_90": stats(server_ttft_90),
            "server_tpot_ms_all_valid_http_200_timelines": stats(tpot_all_valid),
            "server_tpot_ms_server_completed_by_90": stats(server_tpot_90),
            "server_elapsed_ms_server_completed_by_90": stats(server_elapsed_90),
            "server_to_client_response_delay_ms_by_90_server_completions": stats(response_lag_90),
            "server_to_client_response_delay_ms_all_timelined_http_200": stats(client_response_lag_all),
            "client_send_to_api_handler_start_ms_by_90_server_completions": stats(pre_handler_delay_90),
            "actual_output_tokens_server_completed_by_90": stats(output_tokens_90),
            "full_token_timeline_matches_server_output_count_by_90": len(timeline_complete_by_horizon),
            "server_completion_time_sec": stats(server_end),
            "queue_and_kv": {
                "instance_snapshot_rows_in_arrival_window": len(instance_rows),
                "five_second_time_bin_means": time_means,
                "per_instance_snapshot_waiting_requests": stats([x for x in queue_values if x is not None]),
                "per_instance_snapshot_running_requests": stats([x for x in running_values if x is not None]),
                "per_instance_snapshot_gpu_cache_usage_pct": stats([
                    x * 100.0 if x is not None and x <= 1.0 else x for x in cache_values
                ]),
                "per_instance_snapshot_kv_used_blocks_pct_from_22084_capacity": stats(block_values),
                "per_instance": {
                    instance: {metric: stats(values) for metric, values in metrics.items()}
                    for instance, metrics in per_instance.items()
                },
                "kv_capacity_blocks_per_instance_runtime_reported": CAPACITY_BLOCKS,
            },
            "executor_iteration_wall_ms_descriptive_only": {
                kind: stats(values) for kind, values in engine_samples.items()
            },
            "migration_events": migration_by_phase,
        },
        "clock_alignment": {
            **clock_metadata,
            "phase_start_wall_time": phase_start_wall,
            "phase_start_monotonic": phase_start_monotonic,
            "server_token_clock_offset_estimated_after_run": mono_wall_offset,
            "basis": (
                "API per_token_latency timestamps use time.perf_counter; Python reported the same "
                "CLOCK_MONOTONIC implementation for perf_counter and monotonic. The shared "
                "wall/monotonic offset maps the final server token into the client phase clock."
            ),
        },
        "request_timelines": server_request_records,
    }


def main() -> None:
    monotonic_info = time.get_clock_info("monotonic")
    perf_info = time.get_clock_info("perf_counter")
    if monotonic_info.implementation != perf_info.implementation:
        raise RuntimeError("perf_counter and monotonic use different clock implementations")
    metadata = {
        "monotonic_implementation": monotonic_info.implementation,
        "perf_counter_implementation": perf_info.implementation,
        "same_clock_implementation": monotonic_info.implementation == perf_info.implementation,
        "clock_resolution_sec": perf_info.resolution,
    }
    mono_wall_offset = time.time() - time.monotonic()
    off = analyze_mode("off", mono_wall_offset, metadata)
    on = analyze_mode("on", mono_wall_offset, metadata)
    if off["trace_sha256"] != on["trace_sha256"]:
        raise RuntimeError("OFF and ON did not consume the same request trace")
    paired = {
        "experiment_id": "mig_ppo_migration_off_on_pilot_qps15_seed505",
        "created_at_wall_time": time.time(),
        "trace_sha256": off["trace_sha256"],
        "paired_trace_confirmed": True,
        "comparison": {
            "server_final_token_completions_by_60_sec_off": off["measurement"]["server_final_token_completions_by_60_sec"],
            "server_final_token_completions_by_60_sec_on": on["measurement"]["server_final_token_completions_by_60_sec"],
            "server_final_token_completions_by_90_sec_off": off["measurement"]["server_final_token_completions_by_90_sec"],
            "server_final_token_completions_by_90_sec_on": on["measurement"]["server_final_token_completions_by_90_sec"],
            "server_completion_delta_on_minus_off_by_90": (
                on["measurement"]["server_final_token_completions_by_90_sec"]
                - off["measurement"]["server_final_token_completions_by_90_sec"]
            ),
            "server_ttft_mean_ms_off_by_90": off["measurement"]["server_ttft_ms_server_completed_by_90"]["mean"],
            "server_ttft_mean_ms_on_by_90": on["measurement"]["server_ttft_ms_server_completed_by_90"]["mean"],
            "server_tpot_mean_ms_off_by_90": off["measurement"]["server_tpot_ms_server_completed_by_90"]["mean"],
            "server_tpot_mean_ms_on_by_90": on["measurement"]["server_tpot_ms_server_completed_by_90"]["mean"],
            "migration_measurement_events_off": off["measurement"]["migration_events"]["measurement"]["events"],
            "migration_measurement_events_on": on["measurement"]["migration_events"]["measurement"]["events"],
        },
        "mode_results": {"off": off, "on": on},
        "interpretation_limits": [
            "One paired trace and one process startup per condition do not estimate run-to-run variance.",
            "Client response completion timestamps include delayed non-streaming body delivery; server final-token timestamps are reported separately.",
            "HTTP 499 responses are abort-barrier censoring; they have no returned token timeline.",
            "TTFT and TPOT summaries exclude censored HTTP 499 requests and can therefore have survivor bias.",
            "The migration log cost field is not a measured request pause-time label.",
            "Engine executor iteration summaries are descriptive labels; this pilot was not used to refit P1/D3 or cadence.",
        ],
    }
    json_path = PILOT_ROOT / "paired_pilot_analysis.json"
    json_path.write_text(json.dumps(paired, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")

    def fmt(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Matched Migration OFF/ON pilot: QPS 15, seed 505",
        "",
        "This is a targeted pilot on a new seed. Both modes used the same saved 942-request trace, four eager vLLM instances, max model length 2112, 128 sequences per instance, and the calibrated nine-profile workload. The two YAML configs differ only in `ENABLE_ROUTINE_MIGRATION`.",
        "",
        f"Trace SHA-256: `{paired['trace_sha256']}`. Arrival window: 60 seconds. Shared result horizon: 90 seconds.",
        "",
        "## Paired results",
        "",
        "| Measure | Migration OFF | Migration ON |",
        "|---|---:|---:|",
        f"| Requests sent | {off['measurement']['requests_sent']} | {on['measurement']['requests_sent']} |",
        f"| Server final tokens by 60s | {off['measurement']['server_final_token_completions_by_60_sec']} | {on['measurement']['server_final_token_completions_by_60_sec']} |",
        f"| Server final tokens by 90s | {off['measurement']['server_final_token_completions_by_90_sec']} | {on['measurement']['server_final_token_completions_by_90_sec']} |",
        f"| Requests unfinished at server horizon 90s | {off['measurement']['server_unfinished_at_90_sec']} | {on['measurement']['server_unfinished_at_90_sec']} |",
        f"| HTTP 200 bodies received | {off['measurement']['client_http_200_count']} | {on['measurement']['client_http_200_count']} |",
        f"| HTTP 499 abort-censored | {off['measurement']['abort_censored_http_499']} | {on['measurement']['abort_censored_http_499']} |",
        f"| Client bodies received by 90s | {off['measurement']['client_responses_received_by_90_sec']} | {on['measurement']['client_responses_received_by_90_sec']} |",
        f"| Server TTFT mean / P95 by 90s (ms) | {fmt(off['measurement']['server_ttft_ms_server_completed_by_90']['mean'])} / {fmt(off['measurement']['server_ttft_ms_server_completed_by_90']['p95'])} | {fmt(on['measurement']['server_ttft_ms_server_completed_by_90']['mean'])} / {fmt(on['measurement']['server_ttft_ms_server_completed_by_90']['p95'])} |",
        f"| Server TPOT mean / P95 by 90s (ms; timeline matched) | {fmt(off['measurement']['server_tpot_ms_server_completed_by_90']['mean'])} / {fmt(off['measurement']['server_tpot_ms_server_completed_by_90']['p95'])} | {fmt(on['measurement']['server_tpot_ms_server_completed_by_90']['mean'])} / {fmt(on['measurement']['server_tpot_ms_server_completed_by_90']['p95'])} |",
        f"| Queue mean / P95 per instance snapshot | {fmt(off['measurement']['queue_and_kv']['per_instance_snapshot_waiting_requests']['mean'])} / {fmt(off['measurement']['queue_and_kv']['per_instance_snapshot_waiting_requests']['p95'])} | {fmt(on['measurement']['queue_and_kv']['per_instance_snapshot_waiting_requests']['mean'])} / {fmt(on['measurement']['queue_and_kv']['per_instance_snapshot_waiting_requests']['p95'])} |",
        f"| Queue mean / P95 of five-second instance means | {fmt(off['measurement']['queue_and_kv']['five_second_time_bin_means']['waiting']['mean'])} / {fmt(off['measurement']['queue_and_kv']['five_second_time_bin_means']['waiting']['p95'])} | {fmt(on['measurement']['queue_and_kv']['five_second_time_bin_means']['waiting']['mean'])} / {fmt(on['measurement']['queue_and_kv']['five_second_time_bin_means']['waiting']['p95'])} |",
        f"| KV usage mean / P95 from GPU cache metric (%) | {fmt(off['measurement']['queue_and_kv']['per_instance_snapshot_gpu_cache_usage_pct']['mean'])} / {fmt(off['measurement']['queue_and_kv']['per_instance_snapshot_gpu_cache_usage_pct']['p95'])} | {fmt(on['measurement']['queue_and_kv']['per_instance_snapshot_gpu_cache_usage_pct']['mean'])} / {fmt(on['measurement']['queue_and_kv']['per_instance_snapshot_gpu_cache_usage_pct']['p95'])} |",
        f"| KV usage mean / P95 of five-second instance means (%) | {fmt(off['measurement']['queue_and_kv']['five_second_time_bin_means']['kv_usage_pct']['mean'])} / {fmt(off['measurement']['queue_and_kv']['five_second_time_bin_means']['kv_usage_pct']['p95'])} | {fmt(on['measurement']['queue_and_kv']['five_second_time_bin_means']['kv_usage_pct']['mean'])} / {fmt(on['measurement']['queue_and_kv']['five_second_time_bin_means']['kv_usage_pct']['p95'])} |",
        f"| Measurement migrations: attempts, successes, failures/aborts | {off['measurement']['migration_events']['measurement']['events']}, {off['measurement']['migration_events']['measurement']['successes']}, {off['measurement']['migration_events']['measurement']['failures_or_aborts']} | {on['measurement']['migration_events']['measurement']['events']}, {on['measurement']['migration_events']['measurement']['successes']}, {on['measurement']['migration_events']['measurement']['failures_or_aborts']} |",
        "",
        "## Timing and censoring",
        "",
        f"For OFF, {off['measurement']['client_http_200_count']} complete HTTP 200 responses arrived after 90 seconds. Their server final-token timestamps show {off['measurement']['server_final_token_completions_by_90_sec']} had completed by 90 seconds, with {off['measurement']['server_final_token_completions_after_90_sec']} finishing afterward. The remaining {off['measurement']['abort_censored_http_499']} requests were aborted at the drain barrier. ON returned {on['measurement']['client_http_200_count']} HTTP 200 responses; server token timestamps place {on['measurement']['server_final_token_completions_by_90_sec']} by 90 seconds and {on['measurement']['server_final_token_completions_after_90_sec']} after 90 seconds, with {on['measurement']['abort_censored_http_499']} abort-censored 499s.",
        "",
        f"The mean final-token-to-client-body delay for time-lined HTTP 200 requests was {fmt(off['measurement']['server_to_client_response_delay_ms_all_timelined_http_200']['mean'])} ms OFF and {fmt(on['measurement']['server_to_client_response_delay_ms_all_timelined_http_200']['mean'])} ms ON. These client completion timestamps cannot stand in for server token completion timestamps.",
        "",
        "TTFT and TPOT summaries by 90 seconds use only successful HTTP 200 responses with server final-token timestamps within the horizon. HTTP 499 requests have no returned token timeline. TPOT additionally requires the timeline count to equal the response output-token count. The actual generated output count is recorded separately from each request’s output-token budget.",
        "",
        "## Queue, KV, engine, and migration observations",
        "",
        f"The server startup logs report {off['measurement']['queue_and_kv']['kv_capacity_blocks_per_instance_runtime_reported']:,} GPU KV blocks per instance in both modes; each instance CSV also records available blocks. This directly confirms the runtime-reported capacity for this deployment, though it remains a vLLM-reported rather than independent hardware measurement.",
        "",
        f"OFF measurement engine iteration samples: prefill n={off['measurement']['executor_iteration_wall_ms_descriptive_only'].get('prefill', {}).get('n', 0)}, decode n={off['measurement']['executor_iteration_wall_ms_descriptive_only'].get('decode', {}).get('n', 0)}. ON: prefill n={on['measurement']['executor_iteration_wall_ms_descriptive_only'].get('prefill', {}).get('n', 0)}, decode n={on['measurement']['executor_iteration_wall_ms_descriptive_only'].get('decode', {}).get('n', 0)}. These are descriptive executor iteration wall times, not full Engine Step or request latency labels, and were not used to refit the simulator.",
        "",
        f"ON had {on['measurement']['migration_events']['measurement']['events']} terminal measurement migration records: {on['measurement']['migration_events']['measurement']['successes']} successful and {on['measurement']['migration_events']['measurement']['failures_or_aborts']} unsuccessful. Logged cost mean/P95 was {fmt(on['measurement']['migration_events']['measurement']['cost_ms']['mean'])}/{fmt(on['measurement']['migration_events']['measurement']['cost_ms']['p95'])} ms. No migration Pause Time ground truth was collected.",
        "",
        "## Limits",
        "",
        "This is one paired seed and one server startup per mode, so it identifies a useful causal contrast but does not quantify run-to-run variance. It adds a Migration OFF service baseline and a matched ON comparison; it does not satisfy the five-QPS system admission gate or validate migration pause time. The simulator remains ineligible for formal PPO training based on the separate five-QPS system validation.",
        "",
        "The JSON retains all per-request server timeline conversions, queue/KV samples, engine iteration summaries, and migration events. Clock alignment uses the shared Linux `CLOCK_MONOTONIC` implementation: API token timestamps are `time.perf_counter()`, while the benchmark records `time.monotonic()` offsets and wall-clock send/response times.",
        "",
        "Queue and KV fields are shown both per recorded instance snapshot and as means within five-second bins. The latter gives each time bin equal weight; the raw snapshot distribution is included for tails and per-instance inspection. These pilot summaries use five-second bins and should not be substituted for the 50 ms system-validation queue gate.",
        "",
    ]
    (PILOT_ROOT / "paired_pilot_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(paired["comparison"], indent=2))


if __name__ == "__main__":
    main()
