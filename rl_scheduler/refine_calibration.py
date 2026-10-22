"""Run-aware calibration, data audit, and validation for the Mig-PPO simulator.

All fitting uses calibration seeds 101/202/303. Seed 404 is read only after
model selection and is never used to estimate coefficients or regularization.
"""

from __future__ import annotations

import ast
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from rl_scheduler.calibrate import parse_profiling_data
from rl_scheduler.cost_estimator import MigPPOMigrationCostEstimator, nonnegative_lstsq
from rl_scheduler.service_model import evaluate_term, predict_service_model


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def _parse_profile(value: object) -> tuple[str, float, float, float] | None:
    parsed = parse_profiling_data(str(value or ""))
    if parsed is None:
        return None
    kind, batch, total_tokens, latency = parsed
    if kind not in {"prefill", "decode"}:
        return None
    if not all(math.isfinite(item) for item in (batch, total_tokens, latency)) or latency <= 0:
        return None
    if batch <= 0 or total_tokens <= 0:
        return None
    return kind, batch, total_tokens, latency


def _snapshot_match(profile: tuple[str, float, float, float], row: Mapping[str, str]) -> tuple[int, dict[str, float | None]]:
    kind, batch, total_tokens, _latency = profile
    score = 0
    fields: dict[str, float | None] = {}
    if str(row.get("inference_type", "")).strip().lower() == kind:
        score += 1
    row_batch = _safe_float(row.get("num_seqs"))
    if row_batch is not None and int(row_batch) == int(batch):
        score += 1
    row_tokens = None
    try:
        seq_lens = ast.literal_eval(row.get("seq_lens", "[]"))
        if isinstance(seq_lens, (tuple, list)):
            row_tokens = float(sum(float(value) for value in seq_lens))
    except (ValueError, SyntaxError, TypeError, OverflowError):
        pass
    if row_tokens is not None and int(row_tokens) == int(total_tokens):
        score += 1
    batched_tokens = _safe_float(row.get("bs"))
    running = _safe_float(row.get("num_running_requests"))
    waiting = _safe_float(row.get("num_waiting_requests"))
    kv_usage = _safe_float(row.get("gpu_cache_usage"))
    free_blocks = _safe_float(row.get("num_available_gpu_blocks"))
    fields.update({
        "batch_tokens_logged": batched_tokens,
        "running_requests": running,
        "waiting_requests": waiting,
        "kv_usage": kv_usage,
        "free_blocks": free_blocks,
    })
    return score, fields


def load_profile_split(data_root: str | Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read sharded logs, deduplicate polling snapshots, and align state fields.

    `profiling_data` is self-contained for batch size, aggregate context length,
    label, and executor-iteration latency. Other scheduler fields are joined
    from the preceding `step_id`, as the collector snapshots scheduler state
    before the profiling tuple is copied into the next manager poll.
    """
    root = Path(data_root)
    shard_root = root / split
    files = sorted(shard_root.glob("qps_*_seed*/server_logs/on_instance_instance.csv"))
    samples: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "split": split, "run_count": len(files), "runs": {},
        "raw_rows": 0, "valid_profile_rows_before_dedup": 0,
        "valid_unique_profile_samples": 0, "duplicate_snapshot_rows": 0,
        "conflicting_profile_duplicates": 0, "previous_step_joins": 0,
        "previous_step_full_feature_matches": 0,
        "previous_step_batch_token_matches": 0,
        "profile_label_matches_predecessor": 0,
        "profile_batch_matches_predecessor": 0,
        "profile_length_matches_predecessor": 0,
        "row_same_step_label_mismatches": 0,
        "row_same_step_feature_mismatches": 0,
        "batch_tokens_available": 0,
    }
    for path in files:
        config_path = path.parents[1] / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
        run_name = path.parents[1].name
        match = re.fullmatch(r"qps_([0-9.]+)_seed(\d+)", run_name)
        qps = _safe_float(config.get("qps"))
        seed = int(config.get("seed", -1))
        if match:
            qps = qps if qps is not None else float(match.group(1))
            seed = seed if seed >= 0 else int(match.group(2))
        role = str(config.get("dataset_role", "calibration" if split == "calibration" else "validation"))
        run_group = f"qps_{qps:g}_seed{seed}" if qps is not None else run_name
        raw_rows = _read_rows(path)
        audit["raw_rows"] += len(raw_rows)
        # The same step is polled more than once. Profiling tuples are stable
        # within a step; retain one copy and report any conflict.
        by_step: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in raw_rows:
            try:
                step_id = int(row.get("step_id", ""))
            except (TypeError, ValueError):
                continue
            by_step[(str(row.get("instance_id", "")), step_id)].append(row)
        audit["duplicate_snapshot_rows"] += sum(max(0, len(rows) - 1) for rows in by_step.values())
        dedup_rows: dict[tuple[str, int], dict[str, str]] = {}
        for key, rows in by_step.items():
            profiles = {_parse_profile(row.get("profiling_data")) for row in rows}
            profiles.discard(None)
            if len(profiles) > 1:
                audit["conflicting_profile_duplicates"] += 1
            # Select a snapshot with the latest timestamp for scheduler-state
            # features; the profiling tuple itself is identical across polls.
            dedup_rows[key] = max(rows, key=lambda row: _safe_float(row.get("timestamp")) or -math.inf)
        run_counts = Counter()
        for (instance_id, step_id), row in dedup_rows.items():
            profile = _parse_profile(row.get("profiling_data"))
            if profile is None:
                continue
            audit["valid_profile_rows_before_dedup"] += sum(
                1 for candidate in by_step[(instance_id, step_id)]
                if _parse_profile(candidate.get("profiling_data")) is not None
            )
            audit["valid_unique_profile_samples"] += 1
            run_counts[profile[0]] += 1
            kind, batch, total_tokens, latency = profile
            same_score, _same_fields = _snapshot_match(profile, row)
            if str(row.get("inference_type", "")).strip().lower() != kind:
                audit["row_same_step_label_mismatches"] += 1
            if same_score < 3:
                audit["row_same_step_feature_mismatches"] += 1

            predecessor = dedup_rows.get((instance_id, step_id - 1))
            aligned: dict[str, float | None] = {}
            if predecessor is not None:
                score, aligned = _snapshot_match(profile, predecessor)
                audit["previous_step_joins"] += 1
                if score == 3:
                    audit["previous_step_full_feature_matches"] += 1
                if str(predecessor.get("inference_type", "")).strip().lower() == kind:
                    audit["profile_label_matches_predecessor"] += 1
                if _safe_float(predecessor.get("num_seqs")) is not None and int(float(predecessor["num_seqs"])) == int(batch):
                    audit["profile_batch_matches_predecessor"] += 1
                try:
                    prev_lens = ast.literal_eval(predecessor.get("seq_lens", "[]"))
                    if int(sum(float(x) for x in prev_lens)) == int(total_tokens):
                        audit["profile_length_matches_predecessor"] += 1
                except (ValueError, SyntaxError, TypeError, OverflowError):
                    pass
                expected_bs = int(total_tokens) if kind == "prefill" else int(batch)
                if aligned.get("batch_tokens_logged") is not None and int(aligned["batch_tokens_logged"]) == expected_bs:
                    audit["previous_step_batch_token_matches"] += 1
            batch_tokens = aligned.get("batch_tokens_logged")
            if batch_tokens is not None:
                audit["batch_tokens_available"] += 1
            # Profile tuple total length is the observed aggregate sequence
            # length. For prefill it matches scheduler bs for nearly all rows;
            # keep that field separate from the actual-bs audit join.
            sample = {
                "kind": kind, "batch_size": float(batch),
                "total_context_tokens": float(total_tokens),
                "batch_tokens": float(total_tokens) if kind == "prefill" else float(batch),
                "scheduler_batch_tokens": batch_tokens,
                "running_requests": aligned.get("running_requests"),
                "waiting_requests": aligned.get("waiting_requests"),
                "kv_usage": aligned.get("kv_usage"), "free_blocks": aligned.get("free_blocks"),
                "latency_ms": float(latency), "instance_id": instance_id,
                "step_id": int(step_id), "run_group": run_group,
                "profile_timestamp": _safe_float(row.get("timestamp")),
                "qps": qps, "seed": seed, "role": role,
                "code_provenance": (
                    "supplemental_after_migration_slot_guard"
                    if qps == 21 and seed in {202, 303, 404}
                    else "reused_before_migration_slot_guard"
                ),
                "profile_source": str(path),
            }
            samples.append(sample)
        audit["runs"][run_group] = {
            "qps": qps, "seed": seed, "role": role,
            "raw_rows": len(raw_rows), "deduplicated_rows": len(dedup_rows),
            "valid_prefill_samples": run_counts["prefill"],
            "valid_decode_samples": run_counts["decode"],
        }
    return samples, audit


def _prefill_terms(model_name: str, train_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    if model_name == "P0":
        return [{"op": "linear", "feature": "total_context_tokens"}]
    if model_name == "P1":
        return [{"op": "linear", "feature": "batch_tokens"}, {"op": "linear", "feature": "batch_size"}]
    if model_name == "P2":
        return [
            {"op": "linear", "feature": "batch_tokens"},
            {"op": "linear", "feature": "batch_size"},
            {"op": "linear", "feature": "running_requests"},
        ]
    if model_name == "P3":
        return [
            {"op": "linear", "feature": "batch_tokens"},
            {"op": "linear", "feature": "batch_size"},
            {"op": "linear", "feature": "running_requests"},
            {"op": "interaction", "left": "batch_tokens", "right": "batch_size"},
            {"op": "interaction", "left": "batch_tokens", "right": "running_requests"},
        ]
    if model_name == "P4":
        return [
            {"op": "log1p", "feature": "batch_tokens"},
            {"op": "square_log1p", "feature": "batch_tokens"},
            {"op": "log1p", "feature": "batch_size"},
            {"op": "square_log1p", "feature": "batch_size"},
            {"op": "linear", "feature": "running_requests"},
        ]
    raise ValueError(f"unknown prefill candidate: {model_name}")


def _decode_terms(model_name: str, _train_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    if model_name == "D0":
        return [
            {"op": "linear", "feature": "batch_size"},
            {"op": "linear", "feature": "total_context_tokens"},
        ]
    if model_name == "D1":
        return [{"op": "linear", "feature": "batch_size"}]
    if model_name == "D2":
        return [
            {"op": "linear", "feature": "batch_size"},
            {"op": "linear", "feature": "avg_context_tokens"},
        ]
    if model_name == "D3":
        return [
            {"op": "linear", "feature": "batch_size"},
            {"op": "linear", "feature": "avg_context_tokens"},
            {"op": "interaction", "left": "batch_size", "right": "avg_context_tokens"},
        ]
    if model_name == "D4":
        return [
            {"op": "log1p", "feature": "batch_size"},
            {"op": "square_log1p", "feature": "batch_size"},
            {"op": "log1p", "feature": "avg_context_tokens"},
            {"op": "square_log1p", "feature": "avg_context_tokens"},
        ]
    raise ValueError(f"unknown decode candidate: {model_name}")


def _features(row: Mapping[str, Any]) -> dict[str, float]:
    result = {key: float(value) for key, value in row.items()
              if isinstance(value, (int, float)) and value is not None and math.isfinite(float(value))}
    batch = result.get("batch_size", 0.0)
    result["avg_context_tokens"] = result.get("total_context_tokens", 0.0) / max(1.0, batch)
    return result


def _design(rows: Sequence[Mapping[str, Any]], terms: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([[evaluate_term(term, _features(row)) for term in terms] for row in rows], dtype=np.float64)


def _robust_sigma(residuals: np.ndarray) -> float:
    if len(residuals) == 0:
        return 0.0
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    return max(1e-6, 1.4826 * mad)


def fit_service_candidate(
    rows: Sequence[Mapping[str, Any]], kind: str, model_name: str,
    *, ridge_alpha: float = 10.0,
) -> dict[str, Any]:
    terms = _prefill_terms(model_name, rows) if kind == "prefill" else _decode_terms(model_name, rows)
    matrix = _design(rows, terms)
    target = np.asarray([float(row["latency_ms"]) for row in rows], dtype=np.float64)
    centers = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-9)] = 1.0
    standardized = (matrix - centers) / scales
    intercept = float(target.mean())
    centered_target = target - intercept
    regularized = model_name in {"P3", "P4", "D3", "D4"}
    if regularized:
        gram = standardized.T @ standardized
        coefficients = np.linalg.solve(gram + float(ridge_alpha) * np.eye(gram.shape[0]),
                                       standardized.T @ centered_target)
    else:
        coefficients = nonnegative_lstsq(standardized, centered_target)
    model = {
        "name": model_name, "kind": kind, "terms": list(terms),
        "centers": centers.astype(float).tolist(), "scales": scales.astype(float).tolist(),
        "coefficients": np.asarray(coefficients, dtype=float).tolist(),
        "intercept_ms": intercept, "min_prediction_ms": 1.0,
        "ridge_alpha": float(ridge_alpha) if regularized else 0.0,
        "regularized": regularized,
    }
    predictions = np.asarray([predict_service_model(model, _features(row)) for row in rows])
    model["residual_sigma_ms"] = _robust_sigma(target - predictions)
    model["train_samples"] = len(rows)
    return model


def service_metrics(target: Sequence[float], prediction: Sequence[float]) -> dict[str, float | int]:
    y = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    residual = pred - y
    if len(y) == 0:
        return {"n": 0}
    total = float(np.sum(np.square(y - np.mean(y))))
    return {
        "n": int(len(y)), "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": float(1.0 - np.sum(np.square(residual)) / total) if total > 0 else 0.0,
        "mean_error_ms": float(np.mean(residual)),
        "residual_p05_ms": float(np.quantile(residual, 0.05)),
        "residual_p50_ms": float(np.quantile(residual, 0.50)),
        "residual_p95_ms": float(np.quantile(residual, 0.95)),
        "residual_sigma_mad_ms": _robust_sigma(residual),
    }


def grouped_service_cv(rows: Sequence[Mapping[str, Any]], kind: str, model_name: str) -> dict[str, Any]:
    groups = sorted({str(row["run_group"]) for row in rows})
    prediction = np.full(len(rows), np.nan, dtype=np.float64)
    target = np.asarray([float(row["latency_ms"]) for row in rows])
    fold_rows = []
    for group in groups:
        test_idx = [index for index, row in enumerate(rows) if str(row["run_group"]) == group]
        train_rows = [row for row in rows if str(row["run_group"]) != group]
        test_rows = [rows[index] for index in test_idx]
        model = fit_service_candidate(train_rows, kind, model_name)
        pred = np.asarray([predict_service_model(model, _features(row)) for row in test_rows])
        prediction[test_idx] = pred
        metrics = service_metrics([row["latency_ms"] for row in test_rows], pred)
        fold_rows.append({"held_out_run": group, **metrics})
    result = {"model": model_name, "groups": groups,
              "metrics": service_metrics(target, prediction), "run_metrics": fold_rows,
              "macro_run_mae_ms": float(np.mean([row["mae_ms"] for row in fold_rows]))}
    result["predictions"] = prediction
    return result


def _diagnostics(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"features": {}, "correlation": {}, "vif": {}, "condition_number": None}
    matrix = []
    for name in features:
        vals = np.asarray([float(row[name]) for row in rows if row.get(name) is not None], dtype=np.float64)
        if len(vals):
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            iqr = q3 - q1
            out["features"][name] = {
                "n": int(len(vals)), "missing_fraction": float(1.0 - len(vals) / max(1, len(rows))),
                "min": float(vals.min()), "p01": float(np.quantile(vals, 0.01)),
                "p25": float(q1), "median": float(np.median(vals)), "mean": float(vals.mean()),
                "p75": float(q3), "p99": float(np.quantile(vals, 0.99)), "max": float(vals.max()),
                "iqr_outlier_fraction": float(np.mean((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr))),
            }
    complete = [row for row in rows if all(row.get(name) is not None for name in features)]
    if len(complete) >= 3 and len(features) > 1:
        matrix = np.asarray([[float(row[name]) for name in features] for row in complete])
        corr = np.corrcoef(matrix, rowvar=False)
        out["correlation"] = {features[i]: {features[j]: float(corr[i, j]) for j in range(len(features))}
                               for i in range(len(features))}
        standardized = (matrix - matrix.mean(axis=0)) / np.where(matrix.std(axis=0) > 1e-12, matrix.std(axis=0), 1.0)
        out["condition_number"] = float(np.linalg.cond(standardized))
        for index, name in enumerate(features):
            other = np.delete(standardized, index, axis=1)
            if other.shape[1] == 0:
                vif = 1.0
            else:
                fitted = other @ np.linalg.lstsq(other, standardized[:, index], rcond=None)[0]
                ss_total = np.sum(np.square(standardized[:, index] - standardized[:, index].mean()))
                r2 = 1.0 - np.sum(np.square(standardized[:, index] - fitted)) / max(ss_total, 1e-12)
                vif = float(1.0 / max(1e-8, 1.0 - r2))
            out["vif"][name] = vif
    return out


def _raw_csv_summary(path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    headers = list(rows[0]) if rows else []
    columns: dict[str, Any] = {}
    for name in headers:
        values = [_safe_float(row.get(name)) for row in rows]
        usable = np.asarray([value for value in values if value is not None], dtype=float)
        info: dict[str, Any] = {"missing": len(rows) - len(usable),
                                "missing_fraction": float(1 - len(usable) / max(1, len(rows)))}
        if len(usable):
            q1, q3 = np.quantile(usable, [.25, .75])
            iqr = q3 - q1
            info.update({"min": float(usable.min()), "p01": float(np.quantile(usable, .01)),
                         "median": float(np.median(usable)), "mean": float(usable.mean()),
                         "p95": float(np.quantile(usable, .95)), "max": float(usable.max()),
                         "iqr_outlier_fraction": float(np.mean((usable < q1 - 1.5 * iqr)
                                                                  | (usable > q3 + 1.5 * iqr)))})
        else:
            categories = Counter(str(row.get(name, "")) for row in rows if str(row.get(name, "")).strip())
            info["unique_nonempty"] = len(categories)
            if len(categories) <= 30:
                info["values"] = dict(categories)
            elif categories:
                info["top_values"] = dict(categories.most_common(10))
        columns[name] = info
    timestamp_ranges = {}
    for name in ("timestamp", "observed_wall_time", "wall_time", "actual_send_wall_time",
                 "completion_wall_time", "scheduled_arrival_sec", "actual_send_sec",
                 "completion_sec", "relative_sec"):
        vals = [_safe_float(row.get(name)) for row in rows]
        vals = [value for value in vals if value is not None]
        if vals:
            timestamp_ranges[name] = {"min": min(vals), "max": max(vals)}
    return {"path": str(path), "rows": len(rows), "columns": headers,
            "column_quality": columns, "timestamp_ranges": timestamp_ranges}


def _audit_raw_datasets(root: Path, cal_samples: Sequence[Mapping[str, Any]],
                        val_samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    locations = {
        "calibration_service_instance": root / "calibration_inputs/service_instance.csv",
        "validation_service_instance": root / "validation_inputs/service_instance.csv",
        "calibration_migration_events": root / "calibration_inputs/migration_events.csv",
        "validation_migration_events": root / "validation_inputs/migration_events.csv",
        "requests": root / "requests.csv", "summary": root / "summary.csv",
        "gpu_metrics": root / "gpu_metrics.csv", "migration_events": root / "migration_events.csv",
    }
    summaries = {name: _raw_csv_summary(path) for name, path in locations.items()}
    trace_paths = sorted((root / "traces").glob("trace_qps_*_seed*.csv"))
    trace_rows = [row for path in trace_paths for row in _read_rows(path)]
    trace_qps_counts = dict(Counter(path.stem.replace("trace_qps_", "") for path in trace_paths))
    requests = _read_rows(root / "requests.csv")
    measurement = [row for row in requests if row.get("phase") == "measurement"]
    timeline_consistent = 0
    for row in measurement:
        try:
            timeline = json.loads(row.get("per_token_latency_json") or "[]")
            if len(timeline) == int(float(row.get("timeline_count") or -1)):
                timeline_consistent += 1
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    services = [*cal_samples, *val_samples]
    block_estimates = []
    for row in services:
        usage = _safe_float(row.get("kv_usage"))
        free = _safe_float(row.get("free_blocks"))
        if usage is not None and free is not None and 0 <= usage < 1 and free >= 0:
            block_estimates.append(free / max(1e-9, 1 - usage))
    gpu_rows = _read_rows(root / "gpu_metrics.csv")
    gpu_by_id = defaultdict(list)
    for row in gpu_rows:
        util = _safe_float(row.get("gpu_utilization_pct"))
        if util is not None:
            gpu_by_id[str(row.get("gpu_index", "unknown"))].append(util)
    collection_path = root / "collection.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8")) if collection_path.exists() else {}
    return {
        "datasets": summaries,
        "trace_files": {"count": len(trace_paths), "rows": len(trace_rows),
                        "file_counts_by_qps_seed": trace_qps_counts,
                        "join_key": "request_id suffix rNNNN links trace requests to requests.csv and migration_events.request_ids",
                        "columns": list(trace_rows[0]) if trace_rows else []},
        "request_consistency": {
            "measurement_rows": len(measurement),
            "success_true": sum(row.get("success", "").lower() == "true" for row in measurement),
            "timeout_true": sum(row.get("timeout", "").lower() == "true" for row in measurement),
            "timeline_count_matches_json_rows": timeline_consistent,
            "timeline_match_flag_true": sum(row.get("timeline_matches_output", "").lower() == "true" for row in measurement),
            "successful_output_equals_expected": sum(
                row.get("success", "").lower() == "true"
                and _safe_float(row.get("server_output_tokens")) == _safe_float(row.get("expected_output_tokens"))
                for row in measurement),
            "successful_server_input_matches_expected": sum(
                row.get("success", "").lower() == "true"
                and _safe_float(row.get("server_input_tokens")) == _safe_float(row.get("expected_input_tokens"))
                for row in measurement),
            "measurement_request_id_unique": len({row.get("request_id") for row in measurement}),
            "linkage_keys": ["run_id", "phase", "request_id", "server_request_id", "qps_label", "seed"],
        },
        "gpu_metrics": {"rows": len(gpu_rows), "mean_utilization_pct_by_gpu": {
            key: float(np.mean(values)) for key, values in gpu_by_id.items()},
            "linkage_keys": ["run_id", "phase", "gpu_index", "wall_time"]},
        "scheduler_state": {
            "unique_engine_samples_calibration": sum(row["kind"] in {"prefill", "decode"} for row in cal_samples),
            "unique_engine_samples_validation": sum(row["kind"] in {"prefill", "decode"} for row in val_samples),
            "batch_semantics": "profiling_data tuple (stage, num_seqs, sum(seq_lens), executor_iteration_wall_ms); outer bs is scheduler num_batched_tokens and must be predecessor-step aligned",
            "state_join_key": "(instance_id, step_id - 1); sample key (instance_id, step_id)",
            "estimated_total_kv_blocks_from_free_and_usage": {
                "n": len(block_estimates), "median": float(np.median(block_estimates)) if block_estimates else None,
                "p05": float(np.quantile(block_estimates, .05)) if block_estimates else None,
                "p95": float(np.quantile(block_estimates, .95)) if block_estimates else None,
            },
        },
        "units_and_time_semantics": {
            "service timestamp": "Unix wall seconds; profiling_data latency is milliseconds",
            "migration observed_wall_time": "Unix wall seconds; blocks is KV blocks; cost_ms is milliseconds",
            "request scheduled_arrival_sec/actual_send_sec/completion_sec": "seconds relative to run phase; latency columns are milliseconds",
            "trace scheduled_arrival_sec": "seconds relative to measurement workload start; input_tokens/output_tokens are tokens",
            "GPU wall_time/relative_sec": "Unix wall seconds / run-relative seconds; memory MiB, utilization percent, temperature Celsius, power watts",
        },
        "collection_metadata": {key: collection.get(key) for key in
                                ("qps_levels", "calibration_seeds", "validation_seed", "warmup_sec",
                                 "measurement_sec", "drain_sec", "input_profiles", "output_profiles",
                                 "max_model_len", "max_num_seqs_per_instance", "instance_count",
                                 "migration_modes", "engine_latency_measurement")},
    }


def _load_migrations(data_root: Path, split: str) -> list[dict[str, Any]]:
    path = data_root / f"{split}_inputs" / "migration_events.csv"
    rows = _read_rows(path)
    output = []
    for row in rows:
        if str(row.get("mode", "")).lower() != "on" or not str(row.get("status", "")).strip():
            continue
        blocks, duration = _safe_float(row.get("blocks")), _safe_float(row.get("cost_ms"))
        if blocks is None or duration is None or blocks < 0 or duration < 0:
            continue
        output.append({**row, "blocks_value": blocks, "duration_ms": duration,
                       "run_group": str(row.get("run_id", "")),
                       "status_category": str(row.get("status", "")).split(".")[-1].upper()})
    return output


def _migration_linear_fit(rows: Sequence[Mapping[str, Any]], robust: bool) -> dict[str, float]:
    x = np.asarray([float(row["blocks_value"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["duration_ms"]) for row in rows], dtype=np.float64)
    design = np.column_stack((np.ones_like(x), x))
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    if robust:
        for _ in range(50):
            residual = y - design @ coef
            scale = _robust_sigma(residual)
            cutoff = 1.345 * scale
            weights = np.ones_like(residual)
            nonzero = np.abs(residual) > cutoff
            weights[nonzero] = cutoff / np.maximum(np.abs(residual[nonzero]), 1e-12)
            weighted = design * np.sqrt(weights[:, None])
            updated = np.linalg.lstsq(weighted, y * np.sqrt(weights), rcond=None)[0]
            if np.max(np.abs(updated - coef)) < 1e-8:
                coef = updated
                break
            coef = updated
    alpha, beta = float(coef[0]), float(coef[1])
    if beta < 0:
        beta = 0.0
        alpha = max(0.0, float(np.median(y)))
    alpha = max(0.0, alpha)
    residual = y - (alpha + beta * x)
    return {"alpha_ms": alpha, "beta_ms_per_block": beta,
            "residual_sigma_ms": _robust_sigma(residual), "samples": int(len(rows))}


def _fit_migration_model(rows: Sequence[Mapping[str, Any]], *, robust: bool,
                         pair_correction: bool) -> dict[str, Any]:
    global_fit = _migration_linear_fit(rows, robust=robust)
    result: dict[str, Any] = {"global": global_fit, "pair_corrections": {}, "pair_min_samples": 8,
                              "pair_shrinkage": 8.0, "robust": robust}
    if not pair_correction:
        return result
    by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        src, dst = row.get("source_instance"), row.get("target_instance")
        if src and dst:
            by_pair[(str(src), str(dst))].append(row)
    alpha, beta = global_fit["alpha_ms"], global_fit["beta_ms_per_block"]
    for pair, pair_rows in by_pair.items():
        if len(pair_rows) < result["pair_min_samples"]:
            continue
        residual = np.asarray([
            float(row["duration_ms"]) - (alpha + beta * float(row["blocks_value"]))
            for row in pair_rows
        ])
        shrink = len(pair_rows) / (len(pair_rows) + result["pair_shrinkage"])
        correction = float(np.median(residual)) * shrink
        result["pair_corrections"][f"{pair[0]}|{pair[1]}"] = {
            "intercept_correction_ms": correction,
            "samples": len(pair_rows),
            "uncertainty_ms": _robust_sigma(residual - np.median(residual)),
        }
    return result


def predict_migration_model(model: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[float, float, str]:
    global_fit = model["global"]
    alpha = float(global_fit["alpha_ms"])
    beta = float(global_fit["beta_ms_per_block"])
    uncertainty = float(global_fit["residual_sigma_ms"])
    pair_key = f"{row.get('source_instance', '')}|{row.get('target_instance', '')}"
    pair = model.get("pair_corrections", {}).get(pair_key)
    source = "global"
    if pair:
        alpha += float(pair["intercept_correction_ms"])
        uncertainty = math.hypot(uncertainty, float(pair.get("uncertainty_ms", 0.0)))
        source = "pair_shrunk"
    duration = max(0.0, alpha + beta * float(row["blocks_value"]))
    return duration, uncertainty, source


def migration_metrics(rows: Sequence[Mapping[str, Any]], model: Mapping[str, Any]) -> dict[str, Any]:
    target = [float(row["duration_ms"]) for row in rows]
    pred = [predict_migration_model(model, row)[0] for row in rows]
    result: dict[str, Any] = service_metrics(target, pred)
    status_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        status_groups[str(row["status_category"])].append(row)
    result["by_status"] = {
        status: service_metrics([row["duration_ms"] for row in subset],
                                [predict_migration_model(model, row)[0] for row in subset])
        for status, subset in sorted(status_groups.items())
    }
    return result


def grouped_migration_cv(rows: Sequence[Mapping[str, Any]], robust: bool,
                         pair_correction: bool) -> dict[str, Any]:
    groups = sorted({str(row["run_group"]) for row in rows})
    preds: list[float] = []
    target: list[float] = []
    fold_metrics = []
    for group in groups:
        train = [row for row in rows if str(row["run_group"]) != group]
        test = [row for row in rows if str(row["run_group"]) == group]
        model = _fit_migration_model(train, robust=robust, pair_correction=pair_correction)
        fold_pred = [predict_migration_model(model, row)[0] for row in test]
        preds.extend(fold_pred)
        target.extend(float(row["duration_ms"]) for row in test)
        fold_metrics.append({"held_out_run": group, **service_metrics(
            [row["duration_ms"] for row in test], fold_pred)})
    return {"groups": groups, "metrics": service_metrics(target, preds),
            "run_metrics": fold_metrics,
            "macro_run_mae_ms": float(np.mean([row["mae_ms"] for row in fold_metrics]))}


def _status_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["status_category"]) for row in rows)
    by_pair = Counter((str(row.get("source_instance")), str(row.get("target_instance"))) for row in rows)
    block_counts: dict[str, Any] = {}
    for bucket, predicate in {
        "0-16": lambda x: 0 <= x <= 16,
        "17-32": lambda x: 16 < x <= 32,
        "33-64": lambda x: 32 < x <= 64,
        "65+": lambda x: x > 64,
    }.items():
        selected = [row for row in rows if predicate(float(row["blocks_value"]))]
        block_counts[bucket] = {"n": len(selected), "statuses": dict(Counter(row["status_category"] for row in selected))}
    durations = np.asarray([row["duration_ms"] for row in rows], dtype=np.float64)
    blocks = np.asarray([row["blocks_value"] for row in rows], dtype=np.float64)
    return {
        "samples": len(rows), "status_counts": dict(counts),
        "pair_counts": {f"{src}|{dst}": count for (src, dst), count in sorted(by_pair.items())},
        "block_range": [float(blocks.min()), float(blocks.max())] if len(blocks) else [],
        "block_buckets": block_counts,
        "cost_ms": ({"min": float(durations.min()), "median": float(np.median(durations)),
                     "p95": float(np.quantile(durations, .95)), "max": float(durations.max())}
                    if len(durations) else {}),
        "contains_phase_breakdown": False,
        "duration_semantics": "coordinator _migrate_out_one_request wall time from entry to terminal status; no independent service-pause label",
    }


def _metrics_by_qps(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, np.ndarray],
                    model_names: Sequence[str]) -> dict[str, Any]:
    result = {}
    qps_levels = sorted({float(row["qps"]) for row in rows if row.get("qps") is not None})
    for qps in qps_levels:
        indices = [i for i, row in enumerate(rows) if float(row["qps"]) == qps]
        target = [float(rows[i]["latency_ms"]) for i in indices]
        result[str(qps)] = {name: service_metrics(target, predictions[name][indices]) for name in model_names}
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_refined_calibration(data_root: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(data_root)
    output = Path(output_dir) if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    cal_samples, cal_audit = load_profile_split(root, "calibration")
    val_samples, val_audit = load_profile_split(root, "validation")
    cal_prefill = [row for row in cal_samples if row["kind"] == "prefill" and row.get("running_requests") is not None]
    val_prefill = [row for row in val_samples if row["kind"] == "prefill" and row.get("running_requests") is not None]
    cal_decode = [row for row in cal_samples if row["kind"] == "decode"]
    val_decode = [row for row in val_samples if row["kind"] == "decode"]
    if not cal_prefill or not val_prefill or not cal_decode or not val_decode:
        raise ValueError("calibration/validation service data is incomplete")

    p_names, d_names = ["P0", "P1", "P2", "P3", "P4"], ["D0", "D1", "D2", "D3", "D4"]
    prefill_cv = {name: grouped_service_cv(cal_prefill, "prefill", name) for name in p_names}
    decode_cv = {name: grouped_service_cv(cal_decode, "decode", name) for name in d_names}
    best_prefill = min(p_names, key=lambda name: (prefill_cv[name]["metrics"]["mae_ms"], p_names.index(name)))
    best_decode = min(d_names, key=lambda name: (decode_cv[name]["metrics"]["mae_ms"], d_names.index(name)))
    prefill_models = {name: fit_service_candidate(cal_prefill, "prefill", name) for name in p_names}
    decode_models = {name: fit_service_candidate(cal_decode, "decode", name) for name in d_names}
    prefill_val_predictions = {
        name: np.asarray([predict_service_model(prefill_models[name], _features(row)) for row in val_prefill])
        for name in p_names
    }
    decode_val_predictions = {
        name: np.asarray([predict_service_model(decode_models[name], _features(row)) for row in val_decode])
        for name in d_names
    }
    prefill_val_metrics = {
        name: service_metrics([row["latency_ms"] for row in val_prefill], prediction)
        for name, prediction in prefill_val_predictions.items()
    }
    decode_val_metrics = {
        name: service_metrics([row["latency_ms"] for row in val_decode], prediction)
        for name, prediction in decode_val_predictions.items()
    }
    # Candidate promotion uses calibration-only leave-one-run-out CV. Seed 404
    # is strictly a report-only independent validation set.
    promote_prefill = best_prefill
    promote_decode = best_decode

    cal_migrations = _load_migrations(root, "calibration")
    val_migrations = _load_migrations(root, "validation")
    migration_models = {
        "M0_global_ols": _fit_migration_model(cal_migrations, robust=False, pair_correction=False),
        "M1_global_huber": _fit_migration_model(cal_migrations, robust=True, pair_correction=False),
        "M2_huber_pair_shrunk": _fit_migration_model(cal_migrations, robust=True, pair_correction=True),
    }
    migration_cv = {
        "M0_global_ols": grouped_migration_cv(cal_migrations, robust=False, pair_correction=False),
        "M1_global_huber": grouped_migration_cv(cal_migrations, robust=True, pair_correction=False),
        "M2_huber_pair_shrunk": grouped_migration_cv(cal_migrations, robust=True, pair_correction=True),
    }
    migration_val = {name: migration_metrics(val_migrations, model)
                     for name, model in migration_models.items()}
    best_migration = min(migration_models, key=lambda name: migration_cv[name]["metrics"]["mae_ms"])
    promote_migration = min(migration_models,
                            key=lambda name: (migration_cv[name]["metrics"]["mae_ms"],
                                              migration_cv[name]["metrics"]["rmse_ms"], name))

    selected_pre = prefill_models[promote_prefill]
    selected_dec = decode_models[promote_decode]
    selected_mig = migration_models[promote_migration]
    environment = {
        "num_instances": 4,
        "decision_interval_sec": 0.05,
        "max_decisions": 4000,
        "max_running_requests": None,
        "max_num_seqs_per_instance": 128,
        "total_kv_blocks": 22084,
        "block_size_tokens": 16,
        "max_model_len": 2112,
        "max_num_batched_tokens": 2112,
        "ttft_slo_ms": 2000.0,
        "tpot_slo_ms": 150.0,
        "request_timeout_sec": None,
        "migration_cooldown_sec": 0.0,
        "max_state_age_sec": 2.0,
        "safety_margin_blocks": 16,
        "safety_margin_fraction": 0.05,
        "workload_scenario": "mix",
        "arrival_model": "poisson",
        "target_qps_levels": [9.0, 12.0, 15.0, 18.0, 21.0],
        "prompt_profiles": [256, 512, 1048],
        "output_profiles": [256, 512, 1048],
        "service_model_mode": "calibrated",
        "kv_dynamics_enabled": True,
        "instance_capacity_scale": 1.0,
        # Legacy scalar coefficients are always present for old environment
        # configs and for explicit model rollback.
        "prefill_ms_per_token": float(prefill_models["P0"]["coefficients"][0] / prefill_models["P0"]["scales"][0]),
        "prefill_intercept_ms": float(prefill_models["P0"]["intercept_ms"]
            - prefill_models["P0"]["coefficients"][0] * prefill_models["P0"]["centers"][0] / prefill_models["P0"]["scales"][0]),
        "decode_ms_per_request": 0.0,
        "decode_ms_per_context_token": 0.0,
        "decode_intercept_ms": float(decode_models["D0"]["intercept_ms"]),
        "service_time_model": {
            "version": 1, "prefill_name": promote_prefill, "decode_name": promote_decode,
            "prefill": selected_pre, "decode": selected_dec,
            "fallback_prefill": prefill_models["P0"], "fallback_decode": decode_models["D0"],
            "selection_rule": "lowest grouped run-CV MAE; seed404 is report-only and does not affect model choice",
        },
        "migration_cost_estimator": selected_mig,
        "migration_alpha_ms": float(selected_mig["global"]["alpha_ms"]),
        "migration_beta_ms_per_block": float(selected_mig["global"]["beta_ms_per_block"]),
        "migration_uncertainty_scale": 1.0,
        "calibration_status": "service_models_run_cv_and_seed404_validated",
    }
    # Keep coefficients from the selected service model in YAML while preserving
    # P0/D0 as embedded fallback models. Older scalar parameters remain usable.
    payload = {
        "environment": environment,
        "calibration_metadata": {
            "data_root": str(root), "fit_seeds": [101, 202, 303],
            "held_out_seed": 404, "held_out_excluded_from_fit": True,
            "held_out_used_for_model_selection": False,
            "service_measurement": "executor_iteration_wall_ms",
            "service_fit_groups": sorted({str(row["run_group"]) for row in cal_samples}),
            "feature_join_note": "profiling tuple is self-contained; running/queue/KV fields use predecessor step only after feature match",
        },
    }
    audit = {
        "data_root": str(root), "calibration_split": cal_audit, "validation_split": val_audit,
        "service_features": {
            "prefill_calibration": _diagnostics(cal_prefill, ["batch_tokens", "batch_size", "running_requests", "total_context_tokens", "latency_ms"]),
            "decode_calibration": _diagnostics(cal_decode, ["batch_size", "total_context_tokens", "latency_ms"]),
            "prefill_candidate_n": len(cal_prefill), "decode_candidate_n": len(cal_decode),
            "prefill_holdout_candidate_n": len(val_prefill), "decode_holdout_candidate_n": len(val_decode),
        },
        "migration_calibration": _status_audit(cal_migrations),
        "migration_validation": _status_audit(val_migrations),
        "raw_datasets": _audit_raw_datasets(root, cal_samples, val_samples),
        "migration_cost_ms_unit": "milliseconds", "request_and_gpu_units": {
            "trace_arrival_sec": "seconds relative to workload start",
            "requests_client_e2e_ms_server_ttft_ms_server_tpot_ms": "milliseconds",
            "gpu_utilization_pct": "percent", "memory_mib": "MiB", "temperature_c": "Celsius", "power_w": "watts",
        },
        "provenance": {
            "preexisting_scheduler_version_runs": 17,
            "post_migration_slot_guard_runs": ["qps_21_seed202", "qps_21_seed303", "qps_21_seed404"],
            "raw_csv_has_per_row_source_revision": False,
            "metric_instrumentation_same_per_collection_note": True,
        },
    }
    model_results = {
        "service": {
            "prefill": {"grouped_run_cv": {name: {k: v for k, v in result.items() if k != "predictions"}
                                                   for name, result in prefill_cv.items()},
                        "seed404": prefill_val_metrics, "cv_winner": best_prefill,
                        "promoted": promote_prefill,
                        "per_qps_seed404": _metrics_by_qps(val_prefill, prefill_val_predictions, p_names),
                        "coefficients": prefill_models},
            "decode": {"grouped_run_cv": {name: {k: v for k, v in result.items() if k != "predictions"}
                                                  for name, result in decode_cv.items()},
                       "seed404": decode_val_metrics, "cv_winner": best_decode,
                       "promoted": promote_decode,
                       "per_qps_seed404": _metrics_by_qps(val_decode, decode_val_predictions, d_names),
                       "coefficients": decode_models},
        },
        "migration": {
            "grouped_run_cv": migration_cv, "seed404": migration_val,
            "cv_winner": best_migration, "promoted": promote_migration,
            "models": migration_models,
        },
    }
    result = {"audit": audit, "model_results": model_results, "calibration": payload}
    _write_artifacts(output, result, cal_prefill, val_prefill, cal_decode, val_decode,
                     prefill_cv, decode_cv, val_migrations, migration_models)
    return result


def _write_artifacts(output: Path, result: dict[str, Any], cal_prefill, val_prefill, cal_decode, val_decode,
                     prefill_cv, decode_cv, val_migrations, migration_models) -> None:
    audit_path = output / "data_audit.json"
    models_path = output / "model_comparison.json"
    calibration_path = output / "mig_ppo_calibration_refined.yaml"
    audit_path.write_text(json.dumps(_json_safe(result["audit"]), indent=2, ensure_ascii=False), encoding="utf-8")
    models_path.write_text(json.dumps(_json_safe(result["model_results"]), indent=2, ensure_ascii=False), encoding="utf-8")
    calibration_path.write_text(yaml.safe_dump(_json_safe(result["calibration"]), sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_markdown_reports(output, result)
    _write_residual_plots(output, cal_prefill, val_prefill, cal_decode, val_decode,
                          prefill_cv, decode_cv, val_migrations, migration_models)


def _write_markdown_reports(output: Path, result: dict[str, Any]) -> None:
    audit = result["audit"]
    models = result["model_results"]
    cal = audit["calibration_split"]
    val = audit["validation_split"]
    lines = [
        "# Mig-PPO data audit and refined calibration", "",
        "## Data coverage", "",
        f"Calibration: {cal['run_count']} runs, {cal['raw_rows']} service rows, "
        f"{cal['valid_unique_profile_samples']} unique valid engine samples after deduplicating "
        f"{cal['duplicate_snapshot_rows']} repeated snapshots.",
        f"Validation seed 404: {val['run_count']} runs, {val['raw_rows']} service rows, "
        f"{val['valid_unique_profile_samples']} unique valid engine samples after deduplicating "
        f"{val['duplicate_snapshot_rows']} repeated snapshots.", "",
        "Service latency is `executor_iteration_wall_ms`, a host wall-clock interval from Ray executor dispatch through the awaited worker result. It is neither isolated GPU time nor request TTFT. `profiling_data` supplies batch size, aggregate sequence length, stage label, and latency. `bs` is actual scheduler `num_batched_tokens`; it lives in the scheduler snapshot and is aligned through the preceding step when available.", "",
        f"Same-step outer label mismatches: calibration {cal['row_same_step_label_mismatches']}, validation {val['row_same_step_label_mismatches']}. "
        f"Predecessor-step joins: calibration {cal['previous_step_joins']}, validation {val['previous_step_joins']}; "
        f"full feature matches {cal['previous_step_full_feature_matches']}/{cal['previous_step_joins']} and "
        f"{val['previous_step_full_feature_matches']}/{val['previous_step_joins']}. "
        "Unmatched running/queue/KV features are not filled with fabricated values; P2+ Prefill candidates use only rows with a validated predecessor join.", "",
        "## Migration label semantics", "",
        f"Calibration terminal statuses: `{audit['migration_calibration']['status_counts']}`; "
        f"validation statuses: `{audit['migration_validation']['status_counts']}`.",
        "`cost_ms` is the wall-clock duration of `_migrate_out_one_request` from method entry through terminal status handling. The CSV has no phase-level durations and no direct request service-pause label. The simulator may model an in-flight request pause as an explicit approximation; the data do not identify that pause duration.", "",
        "## Service model selection", "",
        "Coefficients, regularization and candidate selection use only calibration seeds 101/202/303. Selection uses leave-one-run-out CV over QPS-seed run groups. Seed 404 is report-only independent validation and does not affect the promoted candidate.", "",
        "| Stage | Candidate | grouped CV MAE | grouped CV RMSE | CV R² | seed404 MAE | seed404 RMSE | seed404 R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, names in (("prefill", ["P0", "P1", "P2", "P3", "P4"]),
                         ("decode", ["D0", "D1", "D2", "D3", "D4"])):
        cv = models["service"][stage]["grouped_run_cv"]
        hold = models["service"][stage]["seed404"]
        for name in names:
            cvm, hm = cv[name]["metrics"], hold[name]
            lines.append(f"| {stage} | {name} | {cvm['mae_ms']:.3f} | {cvm['rmse_ms']:.3f} | {cvm['r2']:.4f} | "
                         f"{hm['mae_ms']:.3f} | {hm['rmse_ms']:.3f} | {hm['r2']:.4f} |")
    for stage in ("prefill", "decode"):
        row = models["service"][stage]
        lines += ["", f"Promoted {stage} candidate: **{row['promoted']}** (CV winner {row['cv_winner']})."]
    lines += ["", "## Migration model selection", "",
              "| Candidate | grouped CV MAE | grouped CV RMSE | seed404 MAE | seed404 RMSE |",
              "|---|---:|---:|---:|---:|"]
    for name in ("M0_global_ols", "M1_global_huber", "M2_huber_pair_shrunk"):
        cvm, hm = models["migration"]["grouped_run_cv"][name]["metrics"], models["migration"]["seed404"][name]
        lines.append(f"| {name} | {cvm['mae_ms']:.3f} | {cvm['rmse_ms']:.3f} | {hm['mae_ms']:.3f} | {hm['rmse_ms']:.3f} |")
    lines += ["", f"Promoted migration model: **{models['migration']['promoted']}**.",
              "", "## Raw file schemas and linkage", "",
              "Full field-by-field column names, numeric distributions, missingness, IQR outlier fractions, and categorical counts are in `data_audit.json`. The request CSV keys are `run_id/phase/request_id/server_request_id`; trace IDs join by the terminal `rNNNNN`; per-instance scheduler samples join on `(instance_id, step_id)` and service state is taken from `step_id - 1`; migration events join through `run_id/phase/request_ids`; GPU metrics join through `run_id/phase/gpu_index/wall_time`.", "",
              "Timestamp fields are Unix seconds for `timestamp`, `observed_wall_time`, `wall_time`, and `actual_send_wall_time`; `scheduled_arrival_sec`, `actual_send_sec`, `completion_sec`, `relative_sec`, and trace arrivals are phase-relative seconds. `cost_ms`, request latency fields, and profiling tuple latency are milliseconds; input/output/sequence lengths are tokens; `blocks` is KV blocks; GPU memory is MiB, utilization is percent, temperature is Celsius, and power is watts.", "",
              "The raw `summary.csv` is a 163-column aggregate with run rows and aggregate rows (`row_type` distinguishes them); only per-run rows should be compared with individual traces. `service_instance.csv` is the scheduler polling schema (`profiling_data`, `bs`, `seq_lens`, queue/KV/admission fields) and includes repeated polling snapshots.", "",
              "## Limitations", "",
              "Request-level responses and GPU metrics are kept for system validation, not used as labels in the service-time fits. The fixed ON replay and autonomous closed-loop simulation are separate outputs. Initial instance routing, omitted request-to-instance assignments, aborted events without request IDs, model output distributions, and the absence of an explicit migration pause label remain assumptions and are listed in the system validation report.", ""]
    (output / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _write_residual_plots(output: Path, cal_prefill, val_prefill, cal_decode, val_decode,
                          prefill_cv, decode_cv, val_migrations, migration_models) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots = output / "refined_plots"
    plots.mkdir(exist_ok=True)
    for kind, train_rows, val_rows, cv, names in (
        ("prefill", cal_prefill, val_prefill, prefill_cv, ["P0", "P1", "P2", "P3", "P4"]),
        ("decode", cal_decode, val_decode, decode_cv, ["D0", "D1", "D2", "D3", "D4"]),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        target = np.asarray([row["latency_ms"] for row in train_rows])
        for name in names:
            pred = np.asarray(cv[name]["predictions"])
            residual = pred - target
            axes[0].scatter(pred, residual, s=5, alpha=.18, label=name)
            qps = np.asarray([row["qps"] for row in train_rows], dtype=float)
            axes[1].scatter(qps, residual, s=5, alpha=.18, label=name)
        axes[0].axhline(0, color="black", linewidth=.8)
        axes[0].set(xlabel="Grouped-CV prediction (ms)", ylabel="prediction − observed (ms)", title=f"{kind} grouped-CV residuals")
        axes[1].axhline(0, color="black", linewidth=.8)
        axes[1].set(xlabel="QPS", ylabel="prediction − observed (ms)", title=f"{kind} residuals by QPS")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / f"{kind}_residuals.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in migration_models.items():
        pred = np.asarray([predict_migration_model(model, row)[0] for row in val_migrations])
        obs = np.asarray([row["duration_ms"] for row in val_migrations])
        ax.scatter(pred, pred - obs, s=20, alpha=.7, label=name)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel="Seed 404 predicted attempt duration (ms)", ylabel="prediction − observed (ms)",
           title="Migration holdout residuals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "migration_residuals.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="existing 20-run calibration collection directory")
    parser.add_argument("--output-dir", help="defaults to --data-root")
    args = parser.parse_args()
    result = run_refined_calibration(args.data_root, output_dir=args.output_dir)
    print(json.dumps({
        "audit": str(Path(args.output_dir or args.data_root) / "data_audit.md"),
        "models": str(Path(args.output_dir or args.data_root) / "model_comparison.json"),
        "calibration": str(Path(args.output_dir or args.data_root) / "mig_ppo_calibration_refined.yaml"),
        "prefill_promoted": result["model_results"]["service"]["prefill"]["promoted"],
        "decode_promoted": result["model_results"]["service"]["decode"]["promoted"],
        "migration_promoted": result["model_results"]["migration"]["promoted"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

