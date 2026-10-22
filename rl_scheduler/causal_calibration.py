"""Fit causal Llumnix batch and output-length models from held-out run groups.

Calibration rows come only from the declared calibration seed. The module
streams event shards, keeps executor wall time separate from CUDA time, and
uses Kaplan-Meier estimates for output lengths with drain-censored requests.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from rl_scheduler.cost_estimator import MigPPOMigrationCostEstimator
from rl_scheduler.causal_event_audit import audit as audit_event_streams, write_report as write_event_audit
from rl_scheduler.causal_event_io import event_stream_paths, open_event_stream
from rl_scheduler.service_model import evaluate_term, predict_service_model


BATCH_NUMERIC = (
    "batch_size", "num_prefill_requests", "num_decode_requests",
    "num_running_sequences", "num_waiting_requests", "num_batched_tokens",
    "prefill_tokens", "decode_tokens", "total_context_tokens",
    "kv_utilization", "num_preempted", "num_blocks_to_swap_in",
    "num_blocks_to_swap_out", "num_blocks_to_copy",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _normalize_training_seeds(training_seed: int | Iterable[int]) -> tuple[list[int], set[int]]:
    """Accept a legacy single seed or an explicit list of calibration seeds."""
    if isinstance(training_seed, (int, np.integer)):
        seeds = [int(training_seed)]
    else:
        seeds = sorted({int(seed) for seed in training_seed})
    if not seeds:
        raise ValueError("at least one calibration seed is required")
    return seeds, set(seeds)


def _batch_features(event: Mapping[str, Any]) -> dict[str, float]:
    prefill = _number(event.get("num_prefill_requests"))
    decode = _number(event.get("num_decode_requests"))
    batch_type = str(event.get("batch_type", ""))
    if batch_type not in {"prefill", "decode", "mixed"}:
        batch_type = "mixed" if prefill and decode else "prefill" if prefill else "decode" if decode else ""
    features = {name: _number(event.get(name)) for name in BATCH_NUMERIC}
    if not features["batch_size"]:
        features["batch_size"] = prefill + decode
    if not features["total_context_tokens"]:
        features["total_context_tokens"] = _number(event.get("context_tokens"))
    if not features["prefill_tokens"] and batch_type == "prefill":
        features["prefill_tokens"] = _number(event.get("context_tokens"))
    features.update({
        "is_prefill": float(batch_type == "prefill"),
        "is_decode": float(batch_type == "decode"),
        "is_mixed": float(batch_type == "mixed"),
    })
    # The scheduler event is authoritative for decode token count. Older
    # instrumented shards may only contain a per-request count.
    if not features["decode_tokens"]:
        features["decode_tokens"] = float(features["num_decode_requests"])
    if not features["num_batched_tokens"]:
        features["num_batched_tokens"] = features["prefill_tokens"] + features["decode_tokens"]
    return features


def _terms(nonlinear: bool) -> list[dict[str, object]]:
    terms: list[dict[str, object]] = [
        {"op": "linear", "feature": name}
        for name in ("is_prefill", "is_decode", "is_mixed", "kv_utilization",
                     "num_preempted", "num_blocks_to_swap_in",
                     "num_blocks_to_swap_out", "num_blocks_to_copy")
    ]
    logged = ("batch_size", "num_running_sequences", "num_waiting_requests",
              "num_batched_tokens", "prefill_tokens", "decode_tokens",
              "total_context_tokens", "num_prefill_requests", "num_decode_requests")
    terms.extend({"op": "log1p", "feature": name} for name in logged)
    if nonlinear:
        for name in ("num_batched_tokens", "prefill_tokens", "decode_tokens",
                     "total_context_tokens", "num_running_sequences"):
            terms.append({"op": "square_log1p", "feature": name})
        terms.extend([
            {"op": "interaction", "left": "num_batched_tokens", "right": "num_running_sequences"},
            {"op": "interaction", "left": "total_context_tokens", "right": "num_decode_requests"},
            {"op": "interaction", "left": "prefill_tokens", "right": "num_prefill_requests"},
            {"op": "interaction", "left": "kv_utilization", "right": "num_running_sequences"},
        ])
    return terms


def _fit_ridge(rows: list[dict[str, Any]], target_name: str,
               model_name: str, nonlinear: bool, alpha: float = 8.0) -> dict[str, Any]:
    terms = _terms(nonlinear)
    features = [_batch_features(row) for row in rows]
    matrix = np.asarray([[evaluate_term(term, row) for term in terms]
                         for row in features], dtype=np.float64)
    target = np.asarray([_number(row.get(target_name)) for row in rows], dtype=np.float64)
    centers = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-9)] = 1.0
    standardized = (matrix - centers) / scales
    intercept = float(np.median(target))
    centered = target - intercept
    coefficients = np.linalg.solve(
        standardized.T @ standardized + float(alpha) * np.eye(standardized.shape[1]),
        standardized.T @ centered,
    )
    return {
        "name": model_name, "terms": terms,
        "centers": centers.astype(float).tolist(), "scales": scales.astype(float).tolist(),
        "coefficients": coefficients.astype(float).tolist(),
        "intercept_ms": intercept, "min_prediction_ms": 0.1,
        "ridge_alpha": float(alpha), "train_samples": len(rows),
    }


def _fit_type_median(rows: list[dict[str, Any]], target_name: str) -> dict[str, Any]:
    medians: dict[str, list[float]] = {}
    for batch_type, flag in (("prefill", "is_prefill"), ("decode", "is_decode"), ("mixed", "is_mixed")):
        values = [_number(row.get(target_name)) for row in rows
                  if _batch_features(row)[flag] == 1.0]
        medians[flag] = [float(np.median(values)) if values else 0.1]
    return {
        "name": "batch_type_median", "terms": [
            {"op": "linear", "feature": "is_prefill"},
            {"op": "linear", "feature": "is_decode"},
            {"op": "linear", "feature": "is_mixed"},
        ], "centers": [0.0, 0.0, 0.0], "scales": [1.0, 1.0, 1.0],
        "coefficients": [medians["is_prefill"][0], medians["is_decode"][0], medians["is_mixed"][0]],
        "intercept_ms": 0.0, "min_prediction_ms": 0.1,
        "train_samples": len(rows),
    }


def _metric(target: list[float], prediction: list[float]) -> dict[str, float | int]:
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if not len(y):
        return {"n": 0}
    residual = p - y
    return {
        "n": int(len(y)), "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(np.sqrt(np.mean(residual ** 2))),
        "p95_absolute_error_ms": float(np.quantile(np.abs(residual), 0.95)),
        "bias_prediction_minus_observed_ms": float(np.mean(residual)),
    }


TOKEN_CADENCE_QUANTILES = np.linspace(0.0, 1.0, 101)


def _interval_quantiles(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0.0)]
    if not len(array):
        return []
    return np.quantile(array, TOKEN_CADENCE_QUANTILES).astype(float).tolist()


def _interpolate_interval_quantiles(profiles: Mapping[str, Mapping[str, Any]], qps: float) -> list[float]:
    usable: list[tuple[float, list[float]]] = []
    for key, profile in profiles.items():
        values = profile.get("interval_quantiles_ms", [])
        try:
            x = float(key)
            row = [float(value) for value in values]
        except (TypeError, ValueError, OverflowError):
            continue
        if row and all(math.isfinite(value) and value > 0.0 for value in row):
            usable.append((x, row))
    usable.sort(key=lambda item: item[0])
    if not usable:
        return []
    if qps <= usable[0][0]:
        return usable[0][1]
    if qps >= usable[-1][0]:
        return usable[-1][1]
    for (left_qps, left), (right_qps, right) in zip(usable, usable[1:]):
        if left_qps <= qps <= right_qps:
            fraction = (qps - left_qps) / (right_qps - left_qps)
            return [a + (b - a) * fraction for a, b in zip(left, right)]
    return usable[-1][1]


def _token_cadence_selection(samples_by_qps: Mapping[float, list[float]]) -> dict[str, Any]:
    """Compare load-conditioned cadence quantiles with a global CDF by held-out QPS."""
    folds: list[dict[str, Any]] = []
    for held_out in sorted(samples_by_qps):
        test = samples_by_qps[held_out]
        train = {qps: values for qps, values in samples_by_qps.items() if qps != held_out}
        pooled = [value for values in train.values() for value in values]
        if not test or not pooled:
            continue
        global_profile = _interval_quantiles(pooled)
        local_profiles = {f"{qps:g}": {"interval_quantiles_ms": _interval_quantiles(values)}
                          for qps, values in train.items() if values}
        qps_profile = _interpolate_interval_quantiles(local_profiles, held_out)
        observed = np.asarray(_interval_quantiles(test), dtype=float)
        global_values = np.asarray(global_profile, dtype=float)
        qps_values = np.asarray(qps_profile, dtype=float)
        folds.append({
            "held_out_qps": held_out,
            "n_intervals": len(test),
            "global_empirical_mean_abs_quantile_error_ms": float(np.mean(np.abs(global_values - observed))),
            "qps_interpolated_mean_abs_quantile_error_ms": float(np.mean(np.abs(qps_values - observed))),
        })
    global_error = float(np.mean([
        fold["global_empirical_mean_abs_quantile_error_ms"] for fold in folds
    ])) if folds else None
    qps_error = float(np.mean([
        fold["qps_interpolated_mean_abs_quantile_error_ms"] for fold in folds
    ])) if folds else None
    winner = "qps_interpolated_empirical_quantiles"
    if global_error is not None and qps_error is not None and global_error < qps_error:
        winner = "global_empirical_quantiles"
    return {"split": "whole QPS groups held out across calibration seeds",
            "selection_metric": "mean absolute error between empirical quantile profiles (ms)",
            "selected_candidate": winner, "mean_cv_error_ms": {
                "global_empirical_quantiles": global_error,
                "qps_interpolated_empirical_quantiles": qps_error,
            }, "folds": folds}


def _load_token_cadence_model(result_dirs: list[Path], training_seed: int | Iterable[int],
                              training_mode: str = "off") -> tuple[dict[str, Any], dict[str, Any]]:
    training_seeds, training_seed_set = _normalize_training_seeds(training_seed)
    samples_by_qps: dict[float, list[float]] = collections.defaultdict(list)
    audit: dict[str, Any] = {"training_seeds": training_seeds, "training_mode": training_mode,
                             "held_out_seed_used": False, "runs": [],
                             "excluded_non_calibration_seed": 0, "excluded_modes": 0,
                             "matched_requests": 0, "api_output_events": 0,
                             "positive_token_intervals": 0, "unchanged_output_events": 0,
                             "invalid_intervals": 0}
    for result_dir in result_dirs:
        for run_dir in sorted((result_dir / "calibration").glob("qps_*_seed*")):
            try:
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            seed = int(config.get("seed", -1))
            if seed not in training_seed_set:
                audit["excluded_non_calibration_seed"] += 1
                continue
            mode = str(config.get("mode", "")) or ("on" if result_dir.name.endswith("_on") else "off")
            if mode != training_mode:
                audit["excluded_modes"] += 1
                continue
            qps = float(config["qps"])
            window = config.get("measurement_window", {})
            start = _number(window.get("start_monotonic_ns_estimate"), math.nan)
            end = _number(window.get("end_monotonic_ns_estimate"), math.nan)
            with (run_dir / "requests.csv").open(newline="", encoding="utf-8", errors="replace") as stream:
                request_ids = {str(row.get("server_request_id") or row.get("request_id") or "")
                               for row in csv.DictReader(stream) if row.get("phase") == "measurement"}
            request_events: dict[str, list[tuple[int, int, int]]] = collections.defaultdict(list)
            event_count = 0
            for event in _event_rows(run_dir, config, measurement_only=False):
                if event.get("event") != "request_output_at_api":
                    continue
                request_id = str(event.get("request_id", ""))
                timestamp = event.get("monotonic_ns")
                if request_id not in request_ids or not isinstance(timestamp, int):
                    continue
                if not (math.isfinite(start) and math.isfinite(end) and start <= timestamp <= end):
                    continue
                try:
                    output_tokens = int(event.get("output_tokens", 0))
                    update_index = int(event.get("output_update_index", 0))
                except (TypeError, ValueError, OverflowError):
                    audit["invalid_intervals"] += 1
                    continue
                request_events[request_id].append((timestamp, update_index, output_tokens))
                event_count += 1
            interval_count = 0
            for request_id, events in request_events.items():
                events.sort(key=lambda row: (row[0], row[1]))
                previous: tuple[int, int, int] | None = None
                for current in events:
                    if previous is not None:
                        delta_tokens = current[2] - previous[2]
                        gap_ms = (current[0] - previous[0]) / 1_000_000.0
                        if delta_tokens == 0:
                            audit["unchanged_output_events"] += 1
                        elif delta_tokens < 0 or gap_ms <= 0.0:
                            audit["invalid_intervals"] += 1
                        else:
                            # API updates may coalesce several generated tokens. Divide the
                            # observed interval across that token delta for the TPOT model.
                            per_token_ms = gap_ms / delta_tokens
                            if math.isfinite(per_token_ms) and per_token_ms > 0.0:
                                samples_by_qps[qps].extend([per_token_ms] * delta_tokens)
                                interval_count += delta_tokens
                                audit["positive_token_intervals"] += delta_tokens
                            else:
                                audit["invalid_intervals"] += 1
                    previous = current
            audit["matched_requests"] += len(request_events)
            audit["api_output_events"] += event_count
            audit["runs"].append({"path": str(run_dir), "qps": qps, "mode": mode,
                                  "seed": seed, "matched_requests": len(request_events),
                                  "api_output_events": event_count,
                                  "positive_token_intervals": interval_count})
    profiles: dict[str, Any] = {}
    for qps, values in sorted(samples_by_qps.items()):
        array = np.asarray(values, dtype=float)
        profiles[f"{qps:g}"] = {
            "n_token_intervals": int(len(array)),
            "mean_ms": float(np.mean(array)), "median_ms": float(np.median(array)),
            "p95_ms": float(np.quantile(array, 0.95)), "p99_ms": float(np.quantile(array, 0.99)),
            "interval_quantiles_ms": _interval_quantiles(array),
        }
    if not profiles:
        raise ValueError("no request_output_at_api token intervals found for cadence calibration")
    selection = _token_cadence_selection(samples_by_qps)
    audit["profiles"] = {qps: {key: value for key, value in profile.items()
                                if key != "interval_quantiles_ms"}
                          for qps, profile in profiles.items()}
    audit["selection"] = selection
    model = {"version": 1, "source_event": "request_output_at_api",
             "interval_semantics": "successive server API-visible output token timestamps; coalesced token deltas divide the observed gap",
             "training_seeds": training_seeds, "training_modes": [training_mode],
             "quantile_levels": TOKEN_CADENCE_QUANTILES.astype(float).tolist(),
             "profiles": profiles, "selected_candidate": selection["selected_candidate"],
             "selection": selection}
    return model, audit


def _qps_fold(rows: list[dict[str, Any]], target_name: str,
              candidate: str) -> dict[str, Any]:
    groups = sorted({float(row["qps"]) for row in rows})
    targets: list[float] = []
    predictions: list[float] = []
    folds: list[dict[str, Any]] = []
    for qps in groups:
        train = [row for row in rows if float(row["qps"]) != qps]
        test = [row for row in rows if float(row["qps"]) == qps]
        if not train or not test:
            continue
        if candidate == "type_median":
            model = _fit_type_median(train, target_name)
        else:
            model = _fit_ridge(train, target_name, candidate,
                               nonlinear=(candidate == "ridge_nonlinear"))
        expected = [_number(row.get(target_name)) for row in test]
        predicted = [predict_service_model(model, _batch_features(row)) for row in test]
        targets.extend(expected)
        predictions.extend(predicted)
        folds.append({"held_out_qps": qps, **_metric(expected, predicted)})
    macro_mae = float(np.mean([fold["mae_ms"] for fold in folds if "mae_ms" in fold])) if folds else None
    result = {"candidate": candidate, "groups_qps": groups,
              "pooled_metrics": _metric(targets, predictions),
              "macro_qps_mae_ms": macro_mae, "folds": folds}
    return result


def _event_rows(run_dir: Path, config: Mapping[str, Any], measurement_only: bool = True) -> Iterable[dict[str, Any]]:
    window = config.get("measurement_window", {})
    start = _number(window.get("start_monotonic_ns_estimate"), math.nan)
    end = _number(window.get("end_monotonic_ns_estimate"), math.nan)
    for path in event_stream_paths(run_dir / "causal_events"):
        with open_event_stream(path) as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_name = event.get("event")
                if measurement_only and event_name in {
                        "engine_iteration_completed", "gpu_execution_completed",
                        "llm_engine_step_completed"}:
                    timestamp = event.get("monotonic_ns")
                    if not isinstance(timestamp, int) or not math.isfinite(start) or not math.isfinite(end):
                        continue
                    if not start <= timestamp <= end:
                        continue
                yield event


def _load_batch_rows(result_dirs: list[Path], training_seed: int | Iterable[int],
                     include_modes: tuple[str, ...] = ("off", "on")) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_seeds, training_seed_set = _normalize_training_seeds(training_seed)
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"training_seeds": training_seeds, "runs": [], "excluded_non_calibration_seed": 0,
                             "excluded_modes": 0, "batch_counts": collections.Counter()}
    for result_dir in result_dirs:
        for run_dir in sorted((result_dir / "calibration").glob("qps_*_seed*")):
            try:
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            seed = int(config.get("seed", -1))
            if seed not in training_seed_set:
                audit["excluded_non_calibration_seed"] += 1
                continue
            mode = str(config.get("mode", ""))
            mode_from_path = "on" if "_on_" in str(run_dir) or result_dir.name.endswith("_on") else "off"
            mode = mode or mode_from_path
            if mode not in include_modes:
                audit["excluded_modes"] += 1
                continue
            qps = float(config["qps"])
            before = len(rows)
            batch_events: list[dict[str, Any]] = []
            engine_steps: dict[str, dict[str, Any]] = {}
            for event in _event_rows(run_dir, config):
                if event.get("event") == "llm_engine_step_completed":
                    batch_id = str(event.get("batch_id", ""))
                    if batch_id and _number(event.get("engine_step_wall_ms"), math.nan) > 0:
                        engine_steps[batch_id] = event
                elif event.get("event") == "engine_iteration_completed" and event.get("succeeded", True):
                    if _number(event.get("executor_wall_ms"), math.nan) > 0:
                        batch_events.append(event)
            joined = 0
            for event in batch_events:
                step = engine_steps.get(str(event.get("batch_id", "")))
                engine_step_wall_ms = _number(step.get("engine_step_wall_ms"), math.nan) if step else math.nan
                if math.isfinite(engine_step_wall_ms) and engine_step_wall_ms > 0:
                    joined += 1
                row = {**_batch_features(event), "batch_type": event.get("batch_type", ""),
                       "executor_wall_ms": _number(event.get("executor_wall_ms")),
                       "engine_step_wall_ms": engine_step_wall_ms,
                       "gpu_elapsed_ms": None, "qps": qps, "mode": mode,
                       "run_group": f"qps_{qps:g}", "run_dir": str(run_dir)}
                rows.append(row)
            events = len(batch_events)
            audit["batch_counts"][f"{mode}_qps_{qps:g}"] = events
            audit["runs"].append({"path": str(run_dir), "mode": mode, "qps": qps,
                                  "seed": seed, "batch_events": len(rows) - before,
                                  "engine_step_joined": joined,
                                  "engine_step_unmatched": events - joined,
                                  "measurement_window": config.get("measurement_window")})
    audit["batch_counts"] = dict(audit["batch_counts"])
    return rows, audit


def _load_gpu_durations(result_dirs: list[Path], training_seed: int | Iterable[int]) -> list[dict[str, Any]]:
    _, training_seed_set = _normalize_training_seeds(training_seed)
    rows: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        for run_dir in sorted((result_dir / "calibration").glob("qps_*_seed*")):
            try:
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if int(config.get("seed", -1)) not in training_seed_set:
                continue
            qps = float(config["qps"])
            for event in _event_rows(run_dir, config):
                if event.get("event") != "gpu_execution_completed" or not event.get("succeeded", True):
                    continue
                if _number(event.get("gpu_elapsed_ms"), math.nan) <= 0:
                    continue
                features = _batch_features(event)
                batch_type = ("mixed" if features["is_mixed"] else
                              "prefill" if features["is_prefill"] else
                              "decode" if features["is_decode"] else "")
                rows.append({**features, "batch_type": batch_type,
                             "gpu_elapsed_ms": _number(event.get("gpu_elapsed_ms")),
                             "qps": qps, "mode": config.get("mode", "unknown"),
                             "run_group": f"qps_{qps:g}"})
    return rows


def _trace_rows(run_dir: Path) -> dict[str, tuple[int, int, float]]:
    trace_dir = run_dir / "traces"
    matches = sorted(trace_dir.glob("trace_*.csv"))
    if not matches:
        return {}
    result: dict[str, tuple[int, int, float]] = {}
    for row in _read_csv(matches[0]):
        try:
            result[str(row["request_id"])] = (
                int(row["input_tokens"]), int(row["output_tokens"]),
                float(row.get("scheduled_arrival_sec", "0")),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return result


def _km_profile(samples: list[tuple[int, bool]], budget: int) -> dict[str, Any]:
    """Discrete Kaplan-Meier PMF; censored token prefixes remain at risk."""
    if not samples:
        return {"lengths": [], "counts": [], "n": 0, "completed_n": 0,
                "censored_n": 0, "tail_mass": 1.0}
    completed_n = sum(1 for _, event in samples if event)
    censored_n = len(samples) - completed_n
    mass: dict[int, float] = collections.defaultdict(float)
    survival = 1.0
    last_observed = max(length for length, _ in samples)
    for length in range(1, int(budget) + 1):
        # A censored prefix of c tokens establishes survival through c, so it
        # leaves the risk set at c+0.5, after a possible event at c.
        risk = sum(1 for observed, _ in samples if observed + 0.5 >= length)
        events = sum(1 for observed, event in samples if event and observed == length)
        if risk and events:
            mass[length] += survival * events / risk
            survival *= 1.0 - events / risk
    # Survival after the last observed event is not identified beyond the
    # observed support. Keep it explicit and use budget as a conservative
    # upper-bound continuation when the simulator needs a finite draw.
    mass[int(budget)] += survival
    total = sum(mass.values())
    if total <= 0:
        mass[int(budget)] = 1.0
        total = 1.0
    lengths = sorted(mass)
    weights = [mass[length] / total for length in lengths]
    counts = [max(1, int(round(weight * 1_000_000))) for weight in weights]
    return {
        "lengths": lengths, "counts": counts, "weights": weights,
        "n": len(samples), "completed_n": completed_n, "censored_n": censored_n,
        "tail_mass_at_budget": float(survival), "last_observed_tokens": int(last_observed),
        "estimator": "discrete Kaplan-Meier; unidentified survival tail assigned to budget",
    }


def _load_output_profiles(result_dirs: list[Path], training_seed: int | Iterable[int],
                          training_mode: str = "off") -> tuple[dict[str, Any], dict[str, Any]]:
    training_seeds, training_seed_set = _normalize_training_seeds(training_seed)
    samples: dict[tuple[int, int, float], list[tuple[int, bool]]] = collections.defaultdict(list)
    audit: dict[str, Any] = {"training_seeds": training_seeds, "runs": [], "matched_requests": 0, "missing_terminal_events": 0,
                             "excluded_non_calibration_seed": 0, "excluded_modes": 0,
                             "censored_status_counts": collections.Counter(),
                             "excluded_failures": 0}
    for result_dir in result_dirs:
        for run_dir in sorted((result_dir / "calibration").glob("qps_*_seed*")):
            try:
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            seed = int(config.get("seed", -1))
            if seed not in training_seed_set:
                audit["excluded_non_calibration_seed"] += 1
                continue
            mode = str(config.get("mode", "")) or ("on" if result_dir.name.endswith("_on") else "off")
            if mode != training_mode:
                audit["excluded_modes"] += 1
                continue
            qps = float(config["qps"])
            trace = _trace_rows(run_dir)
            req_rows = [row for row in _read_csv(run_dir / "requests.csv")
                        if row.get("phase") == "measurement"]
            request_meta = {}
            for row in req_rows:
                rid = str(row.get("server_request_id") or row.get("request_id") or "")
                try:
                    prompt = int(float(row.get("expected_input_tokens", "")))
                    budget = int(float(row.get("expected_output_tokens", "")))
                except (TypeError, ValueError, OverflowError):
                    item = trace.get(str(row.get("request_id", "")))
                    if item is None:
                        continue
                    prompt, budget = item[0], item[1]
                request_meta[rid] = {"prompt": prompt, "budget": budget,
                                    "status_code": str(row.get("status_code", "")),
                                    "client_success": str(row.get("success", "")).lower() == "true"}
            found: set[str] = set()
            run_censored = 0
            for event in _event_rows(run_dir, config, measurement_only=False):
                if event.get("event") != "request_server_terminal":
                    continue
                rid = str(event.get("request_id", ""))
                meta = request_meta.get(rid)
                if meta is None:
                    continue
                found.add(rid)
                try:
                    observed = max(0, int(event.get("output_tokens", 0)))
                    budget = int(event.get("output_budget", meta["budget"]))
                except (TypeError, ValueError, OverflowError):
                    continue
                if budget < 1:
                    continue
                status = str(event.get("terminal_status", "unknown"))
                if status == "server_completed" or observed >= budget:
                    completed = True
                    length = min(budget, max(1, observed))
                elif meta["status_code"] == "499" or "abort" in status.lower() or "cancel" in status.lower():
                    completed = False
                    length = min(budget, observed)
                    audit["censored_status_counts"][status] += 1
                    run_censored += 1
                elif observed > 0:
                    completed = False
                    length = min(budget, observed)
                    audit["censored_status_counts"][status] += 1
                    run_censored += 1
                else:
                    audit["excluded_failures"] += 1
                    continue
                prompt = int(meta["prompt"])
                samples[(prompt, budget, qps)].append((length, completed))
                audit["matched_requests"] += 1
            audit["missing_terminal_events"] += len(set(request_meta) - found)
            audit["runs"].append({"path": str(run_dir), "qps": qps, "mode": mode, "seed": seed,
                                  "measurement_requests": len(request_meta),
                                  "terminal_events_joined": len(found),
                                  "censored": run_censored})
    profiles: dict[str, Any] = {}
    for (prompt, budget, qps), values in sorted(samples.items()):
        profiles[f"{prompt}:{budget}:{qps:g}"] = _km_profile(values, budget)
    # Fallback profiles pool prompt lengths at the same budget and load. They
    # are still training-only and preserve censoring indicators.
    pooled: dict[tuple[int, float], list[tuple[int, bool]]] = collections.defaultdict(list)
    for (prompt, budget, qps), values in samples.items():
        pooled[(budget, qps)].extend(values)
    for (budget, qps), values in pooled.items():
        profiles[f"*:{budget}:{qps:g}"] = _km_profile(values, budget)
    all_budget: dict[int, list[tuple[int, bool]]] = collections.defaultdict(list)
    for (prompt, budget, qps), values in samples.items():
        all_budget[budget].extend(values)
    for budget, values in all_budget.items():
        profiles[f"*:{budget}"] = _km_profile(values, budget)
    audit["censored_status_counts"] = dict(audit["censored_status_counts"])
    audit["profiles"] = {key: {"n": value["n"], "completed_n": value["completed_n"],
                               "censored_n": value["censored_n"],
                               "tail_mass_at_budget": value.get("tail_mass_at_budget")}
                         for key, value in profiles.items()}
    audit["held_out_seed_used"] = False
    audit["training_seeds"] = training_seeds
    audit["training_mode"] = training_mode
    return {"version": 2, "estimator": "discrete Kaplan-Meier with explicit unidentified tail",
            "training_seeds": training_seeds,
            "profiles": profiles}, audit


def _fit_duration_component(values: list[tuple[float, float]]) -> dict[str, Any]:
    fitted = MigPPOMigrationCostEstimator._fit(values)
    if fitted is None:
        durations = [duration for _, duration in values]
        return {"intercept_ms": float(np.median(durations)) if durations else 0.0,
                "ms_per_block": 0.0,
                "uncertainty_ms": float(np.std(durations)) if durations else None,
                "samples": len(durations), "fallback": "median"}
    alpha, beta, sigma, count = fitted
    return {"intercept_ms": alpha, "ms_per_block": beta,
            "uncertainty_ms": sigma, "samples": count}


def _load_migrations(result_dirs: list[Path], training_seed: int | Iterable[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    training_seeds, training_seed_set = _normalize_training_seeds(training_seed)
    samples: list[dict[str, Any]] = []
    audit = {"runs": [], "held_out_seed_used": False, "training_seeds": training_seeds}
    for result_dir in result_dirs:
        if not result_dir.name.endswith("_on"):
            continue
        for run_dir in sorted((result_dir / "calibration").glob("qps_*_seed*")):
            try:
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            seed = int(config.get("seed", -1))
            if seed not in training_seed_set:
                continue
            start = _number(config.get("measurement_window", {}).get("start_monotonic_ns_estimate"), math.nan)
            end = _number(config.get("measurement_window", {}).get("end_monotonic_ns_estimate"), math.nan)
            migrations: dict[str, dict[str, Any]] = collections.defaultdict(dict)
            for event in _event_rows(run_dir, config, measurement_only=False):
                name = str(event.get("event", ""))
                migration_id = str(event.get("migration_id", ""))
                if name.startswith("migration_") and migration_id:
                    migrations[migration_id][name] = event
            request_ids = {
                str(events.get("migration_attempt_started", {}).get("request_id", ""))
                for events in migrations.values()
            }
            request_ids.discard("")
            token_times: dict[str, list[int]] = collections.defaultdict(list)
            if request_ids:
                for event in _event_rows(run_dir, config, measurement_only=False):
                    if event.get("event") != "request_output_at_api":
                        continue
                    request_id = str(event.get("request_id", ""))
                    timestamp = event.get("monotonic_ns")
                    if request_id in request_ids and isinstance(timestamp, int):
                        token_times[request_id].append(timestamp)
            count = 0
            for migration_id, events in migrations.items():
                started = events.get("migration_attempt_started")
                finished = events.get("migration_attempt_completed")
                pause = events.get("migration_request_pause_started")
                resumed = events.get("migration_request_resumed")
                if not started or not finished or not pause or not resumed:
                    continue
                start_ns = started.get("monotonic_ns")
                end_ns = finished.get("monotonic_ns")
                if not isinstance(start_ns, int) or not isinstance(end_ns, int):
                    continue
                if math.isfinite(start) and math.isfinite(end) and not start <= start_ns <= end:
                    continue
                pause_ns, resume_ns = pause.get("monotonic_ns"), resumed.get("monotonic_ns")
                if not isinstance(pause_ns, int) or not isinstance(resume_ns, int):
                    continue
                request_id = str(started.get("request_id", ""))
                outputs = sorted(token_times.get(request_id, []))
                before = [value for value in outputs if value < pause_ns]
                after = [value for value in outputs if value > resume_ns]
                if not before or not after:
                    continue
                blocks = _number(finished.get("total_blocks"),
                                 _number(events.get("migration_transfer_completed", {}).get("blocks")))
                total_ms = _number(finished.get("migration_total_ms"), (end_ns - start_ns) / 1e6)
                pause_ms = (resume_ns - pause_ns) / 1e6
                recovery_ms = (after[0] - resume_ns) / 1e6
                token_gap_ms = (after[0] - before[-1]) / 1e6
                row = {"migration_id": migration_id, "request_id": request_id,
                       "qps": float(config["qps"]), "mode": "on", "blocks": blocks,
                       "total_ms": total_ms, "pause_stage_ms": pause_ms,
                       "token_gap_ms": token_gap_ms, "recovery_after_resume_ms": recovery_ms,
                       "status": str(finished.get("status", "unknown")),
                       "source_instance": started.get("source_instance_id"),
                       "target_instance": started.get("target_instance_id")}
                samples.append(row)
                count += 1
            audit["runs"].append({"path": str(run_dir), "qps": float(config["qps"]), "seed": seed,
                                  "matched_migration_samples": count})
    components = {}
    for name in ("total_ms", "pause_stage_ms", "token_gap_ms", "recovery_after_resume_ms"):
        components[name] = _fit_duration_component([
            (float(row["blocks"]), float(row[name])) for row in samples
        ])
    audit.update({"sample_count": len(samples), "status_counts": dict(collections.Counter(
        row["status"] for row in samples)), "samples": samples,
        "components": components,
        "definitions": {
            "total_ms": "attempt start to terminal result",
            "pause_stage_ms": "source pause start to request resumed at target",
            "token_gap_ms": "last API token before pause to first API token after resume",
            "recovery_after_resume_ms": "request resumed to first API token after resume",
        }})
    return {"version": 1, "components": components,
            "source": "calibration ON migration events only"}, audit


def fit(data_dirs: list[Path], output_dir: Path, training_seed: int | Iterable[int]) -> dict[str, Any]:
    training_seeds, _ = _normalize_training_seeds(training_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_integrity = []
    for data_dir in data_dirs:
        report = audit_event_streams(data_dir)
        write_event_audit(data_dir, report)
        process_rows = list(report["processes"].values())
        sequence_missing = sum(row["sequence_missing"] for row in process_rows)
        sequence_duplicates = sum(row["sequence_duplicates"] for row in process_rows)
        unmatched_batches = len(report["batch_joins"]["scheduled_without_executor_completion"])
        missing_terminals = len(report["request_lifecycle"]["received_without_terminal"])
        checks = {
            "zero_json_parse_errors": len(report["json_parse_errors"]) == 0,
            "zero_missing_process_sequence_values": sequence_missing == 0,
            "zero_duplicate_process_sequence_values": sequence_duplicates == 0,
            "zero_unmatched_scheduled_batches": unmatched_batches == 0,
            "zero_missing_request_terminals": missing_terminals == 0,
        }
        event_integrity.append({
            "data_dir": str(data_dir), "source_stream_count": report["source_stream_count"],
            "total_events": report["total_events"], "event_type_counts": report["event_type_counts"],
            "sequence_missing": sequence_missing, "sequence_duplicates": sequence_duplicates,
            "unmatched_scheduled_batches": unmatched_batches,
            "missing_request_terminals": missing_terminals,
            "checks": checks, "passed": all(checks.values()),
        })
    if not all(item["passed"] for item in event_integrity):
        raise ValueError(f"causal source event integrity gate failed: {event_integrity}")
    batch_rows, batch_audit = _load_batch_rows(data_dirs, training_seeds)
    if not batch_rows:
        raise ValueError("no calibration engine_iteration_completed events found")
    targets = ("engine_step_wall_ms", "executor_wall_ms")
    selection: dict[str, Any] = {"split": "whole QPS groups held out across calibration seeds",
                                 "training_seeds": training_seeds, "held_out_seed_used": False,
                                 "service_targets": {}, "batch_collection": batch_audit,
                                 "event_integrity": event_integrity}
    selected: dict[str, Any] = {}
    for target_name in targets:
        target_rows = [row for row in batch_rows
                       if _number(row.get(target_name), math.nan) > 0.0]
        if not target_rows:
            selection["service_targets"][target_name] = {
                "selected_candidate": None, "candidate_comparison": [],
                "excluded_missing_target_rows": len(batch_rows),
            }
            continue
        candidates = [_qps_fold(target_rows, target_name, candidate)
                      for candidate in ("type_median", "ridge_batch", "ridge_nonlinear")]
        # Macro per-QPS MAE is the selection criterion; pooled MAE is retained.
        winner = min(candidates, key=lambda item: (item["macro_qps_mae_ms"] is None,
                                                   item["macro_qps_mae_ms"] or math.inf))
        nonlinear = winner["candidate"] == "ridge_nonlinear"
        model = (_fit_type_median(target_rows, target_name) if winner["candidate"] == "type_median"
                 else _fit_ridge(target_rows, target_name, winner["candidate"], nonlinear))
        selected[target_name] = model
        selection["service_targets"][target_name] = {
            "selected_candidate": winner["candidate"], "criterion": "lowest macro held-out-QPS MAE",
            "candidate_comparison": candidates,
            "excluded_missing_target_rows": len(batch_rows) - len(target_rows),
            "calibration_fit_metrics": _metric(
                [_number(row[target_name]) for row in target_rows],
                [predict_service_model(model, _batch_features(row)) for row in target_rows]),
        }
    gpu_rows = _load_gpu_durations(data_dirs, training_seeds)
    if gpu_rows:
        gpu_cv = [_qps_fold(gpu_rows, "gpu_elapsed_ms", candidate)
                  for candidate in ("type_median", "ridge_batch", "ridge_nonlinear")]
        gpu_winner = min(gpu_cv, key=lambda item: (item["macro_qps_mae_ms"] is None,
                                                   item["macro_qps_mae_ms"] or math.inf))
        gpu_model = (_fit_type_median(gpu_rows, "gpu_elapsed_ms") if gpu_winner["candidate"] == "type_median"
                     else _fit_ridge(gpu_rows, "gpu_elapsed_ms", gpu_winner["candidate"],
                                     gpu_winner["candidate"] == "ridge_nonlinear"))
        selected["gpu_elapsed_ms"] = gpu_model
        selection["gpu_target"] = {"selected_candidate": gpu_winner["candidate"],
                                    "candidate_comparison": gpu_cv,
                                    "fit_metrics": _metric([_number(row["gpu_elapsed_ms"]) for row in gpu_rows],
                                                           [predict_service_model(gpu_model, _batch_features(row))
                                                            for row in gpu_rows])}
    output_model, output_audit = _load_output_profiles(data_dirs, training_seeds)
    token_cadence_model, token_cadence_audit = _load_token_cadence_model(
        data_dirs, training_seeds, training_mode="off")
    migration_model, migration_audit = _load_migrations(data_dirs, training_seeds)
    selection["output_length_audit"] = output_audit
    selection["token_cadence_audit"] = token_cadence_audit
    selection["token_cadence_model_selection"] = token_cadence_model["selection"]
    selection["migration_audit"] = migration_audit
    selection["service_event_count"] = len(batch_rows)
    selection["engine_step_joined_count"] = sum(
        int(run.get("engine_step_joined", 0)) for run in batch_audit["runs"])
    selection["gpu_event_count"] = len(gpu_rows)
    selection["training_run_count"] = len(batch_audit["runs"])
    batch_execution_model = selected.get("engine_step_wall_ms", selected.get("executor_wall_ms"))
    if batch_execution_model is None:
        raise ValueError("no engine step or executor timing model could be fitted")
    (output_dir / "causal_service_model.json").write_text(
        json.dumps({"version": 2, "training_seeds": training_seeds,
                    "batch_execution": batch_execution_model,
                    "engine_step_wall": selected.get("engine_step_wall_ms"),
                    "executor_wall": selected.get("executor_wall_ms"),
                    "gpu_execution": selected.get("gpu_elapsed_ms"),
                    "feature_names": list(BATCH_NUMERIC) + ["is_prefill", "is_decode", "is_mixed"],
                    "batch_execution_target": "engine_step_wall_ms"
                    if "engine_step_wall_ms" in selected else "executor_wall_ms",
                    "selection": selection["service_targets"].get(
                        "engine_step_wall_ms", selection["service_targets"].get("executor_wall_ms"))},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "token_cadence_model.json").write_text(
        json.dumps(token_cadence_model, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "output_length_model.json").write_text(
        json.dumps(output_model, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "migration_model.json").write_text(
        json.dumps(migration_model, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "calibration_audit.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    return selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True,
                        help="OFF/ON result directories; only calibration/<run> shards are read")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, action="append", required=True,
                        help="calibration seed; repeat for multiple whole calibration run groups")
    args = parser.parse_args()
    result = fit(args.data_dir, args.output_dir, args.training_seed)
    print(json.dumps({"service_event_count": result["service_event_count"],
                      "gpu_event_count": result["gpu_event_count"],
                      "training_run_count": result["training_run_count"],
                      "engine_step_joined_count": result["engine_step_joined_count"],
                      "token_cadence_intervals": result["token_cadence_audit"]["positive_token_intervals"],
                      "output_profile_count": len(result["output_length_audit"]["profiles"]),
                      "output_censored": sum(v["censored_n"] for k, v in
                                              result["output_length_audit"]["profiles"].items()
                                              if k.count(":") == 2)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
