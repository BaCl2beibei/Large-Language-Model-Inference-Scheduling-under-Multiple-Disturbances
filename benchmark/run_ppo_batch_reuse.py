#!/usr/bin/env python3
"""Collect the full PPO dataset while reusing one eager four-instance server."""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import shutil
import statistics
import time
import traceback
from pathlib import Path

import pandas as pd

import run_baseline_4gpu as BASE
from rl_scheduler.causal_event_io import compress_event_stream


ROOT = Path(__file__).resolve().parents[1]
PROFILE_LENGTHS = (256, 512, 1048)
REQUEST_PROFILES = tuple((prompt, output) for prompt in PROFILE_LENGTHS for output in PROFILE_LENGTHS)
PROFILE_MIX = [
    {"name": f"input{prompt}_output{output}", "input_tokens": prompt,
     "output_tokens": output, "weight": 1.0}
    for prompt, output in REQUEST_PROFILES
]
MAX_MODEL_LEN = 2112
THRESHOLD = 9000.0
FREQUENCY = 1
DEFRAG = False
CAUSAL_SOURCE_FILES = (
    "benchmark/run_baseline_4gpu.py",
    "benchmark/run_ppo_batch_reuse.py",
    "llumnix/metrics/causal_events.py",
    "llumnix/backends/vllm/scheduler.py",
    "llumnix/backends/vllm/llm_engine.py",
    "llumnix/backends/vllm/executor.py",
    "llumnix/backends/vllm/worker.py",
    "llumnix/llumlet/migration_coordinator.py",
    "llumnix/entrypoints/vllm/api_server.py",
    "llumnix/entrypoints/setup.py",
    "llumnix/scaler.py",
    "llumnix/ray_utils.py",
    "llumnix/envs.py",
    "rl_scheduler/service_model.py",
    "rl_scheduler/env.py",
    "rl_scheduler/causal_calibration.py",
    "rl_scheduler/causal_event_io.py",
    "rl_scheduler/causal_event_audit.py",
    "rl_scheduler/causal_system_validation.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> dict:
    hashes = {name: sha256(ROOT / name) for name in CAUSAL_SOURCE_FILES}
    commit_result = BASE.run_cmd(["git", "rev-parse", "HEAD"], check=False)
    diff_result = BASE.run_cmd(["git", "diff", "--binary", "HEAD"], check=False)
    diff_digest = hashlib.sha256(diff_result.stdout.encode("utf-8", errors="replace")).hexdigest()
    aggregate = hashlib.sha256()
    for name, digest in sorted(hashes.items()):
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "git_commit": commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        "git_dirty_diff_sha256": diff_digest,
        "instrumented_source_sha256": aggregate.hexdigest(),
        "source_file_sha256": hashes,
    }


def event_inventory(out: Path) -> list[dict]:
    entries = []
    for path in sorted((out / "causal_events").glob("events_*.jsonl*")):
        if path.name.endswith(".jsonl.gz") and path.with_suffix("").exists():
            # A crash can leave a verified archive beside its source. Prefer the
            # source until compression finishes and the source is removed.
            continue
        entries.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "encoding": "gzip" if path.name.endswith(".jsonl.gz") else "plain"})
    return entries


def compress_root_event_streams(out: Path) -> list[dict]:
    """Losslessly archive duplicate unsplit streams after run shards are written."""
    event_dir = out / "causal_events"
    uncompressed_metadata = {}
    entries = []
    for path in sorted(event_dir.glob("events_*.jsonl")):
        original_bytes = path.stat().st_size
        original_hash = sha256(path)
        archived = path.with_suffix(path.suffix + ".gz")
        temporary = archived.with_name(archived.name + ".tmp")
        try:
            with path.open("rb") as source, temporary.open("wb") as target:
                with gzip.GzipFile(fileobj=target, mode="wb", compresslevel=1, mtime=0) as compressed:
                    shutil.copyfileobj(source, compressed, length=1024 * 1024)
            digest = hashlib.sha256()
            decoded_bytes = 0
            with gzip.open(temporary, "rb") as restored:
                for block in iter(lambda: restored.read(1024 * 1024), b""):
                    digest.update(block)
                    decoded_bytes += len(block)
            if decoded_bytes != original_bytes or digest.hexdigest() != original_hash:
                raise IOError(f"gzip verification failed for {path}")
            temporary.replace(archived)
            path.unlink()
            uncompressed_metadata[archived.name] = (original_bytes, original_hash)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    for archived in sorted(event_dir.glob("events_*.jsonl.gz")):
        if archived.with_suffix("").exists():
            continue
        decoded_bytes, decoded_hash = uncompressed_metadata.get(archived.name, (None, None))
        if decoded_bytes is None or decoded_hash is None:
            digest = hashlib.sha256()
            decoded_bytes = 0
            with gzip.open(archived, "rb") as restored:
                for block in iter(lambda: restored.read(1024 * 1024), b""):
                    digest.update(block)
                    decoded_bytes += len(block)
            decoded_hash = digest.hexdigest()
        entries.append({
            "path": str(archived.relative_to(out)), "bytes": archived.stat().st_size,
            "sha256": sha256(archived), "encoding": "gzip",
            "uncompressed_bytes": decoded_bytes, "uncompressed_sha256": decoded_hash,
        })
    return entries


def atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or not path.stat().st_size:
        return [], []
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def drop_incomplete_run_rows(path: Path, incomplete_run_ids: set[str]) -> None:
    fields, rows = read_csv(path)
    if fields and rows:
        atomic_csv(path, fields, [row for row in rows if row.get("run_id") not in incomplete_run_ids])


def prepare_resume_artifacts(out: Path, state: dict, mode: str = "on") -> Path | None:
    """Preserve the previous server logs and discard telemetry from a failed window."""
    server_dir = out / "server_logs"
    incomplete_run_ids = {
        run_id for run_id, value in state.get("runs", {}).items()
        if value.get("status") != "complete"
    }
    if not incomplete_run_ids:
        return None

    for metrics_path in (out / "gpu_metrics.csv", out / "migration_events.csv"):
        drop_incomplete_run_rows(metrics_path, incomplete_run_ids)

    attempt = 1
    while (server_dir / f"{mode}_attempt_{attempt}.log").exists():
        attempt += 1
    old_server_log = server_dir / f"{mode}.log"
    if old_server_log.exists():
        shutil.copy2(old_server_log, server_dir / f"{mode}_attempt_{attempt}.log")
    old_instance_log = server_dir / f"{mode}_instance_instance.csv"
    archive_instance_log = server_dir / f"{mode}_instance_instance_attempt_{attempt}.csv"
    if old_instance_log.exists():
        shutil.copy2(old_instance_log, archive_instance_log)
        return archive_instance_log
    return None


def merge_preserved_instance_log(out: Path, archive_path: Path | None, mode: str = "on") -> None:
    current_path = out / "server_logs" / f"{mode}_instance_instance.csv"
    if archive_path is None or not archive_path.exists() or not current_path.exists():
        return
    old_fields, old_rows = read_csv(archive_path)
    new_fields, new_rows = read_csv(current_path)
    fields = list(dict.fromkeys([*old_fields, *new_fields]))
    if fields:
        atomic_csv(current_path, fields, [*old_rows, *new_rows])


def _event_request_ids(value, request_ids: set[str], found: set[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _event_request_ids(item, request_ids, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _event_request_ids(item, request_ids, found)
    if isinstance(value, str):
        if value in request_ids:
            found.add(value)
        elif ":" in value:
            found.update(part for part in value.split(":") if part in request_ids)


def split_causal_events(out: Path, shard_request_ids: dict[Path, set[str]],
                        shard_time_ranges: dict[Path, tuple[int, int] | None]) -> dict[Path, int]:
    event_dir = out / "causal_events"
    if not event_dir.exists():
        return {shard: 0 for shard in shard_request_ids}
    request_to_shard = {
        request_id: shard for shard, request_ids in shard_request_ids.items()
        for request_id in request_ids
    }
    counts = {shard: 0 for shard in shard_request_ids}
    streams = {}
    for shard in shard_request_ids:
        shard_event_dir = shard / "causal_events"
        shard_event_dir.mkdir(parents=True, exist_ok=True)
        streams[shard] = (shard_event_dir / "events.jsonl").open("w", encoding="utf-8")
    all_request_ids = set(request_to_shard)
    all_shards = set(shard_request_ids)
    try:
        sources = [path for path in sorted(event_dir.glob("events_*.jsonl"))]
        sources.extend(path for path in sorted(event_dir.glob("events_*.jsonl.gz"))
                       if not path.with_suffix("").exists())
        for source in sources:
            opener = gzip.open if source.name.endswith(".jsonl.gz") else open
            with opener(source, "rt", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") == "clock_sync":
                        destinations = all_shards
                    else:
                        matched_ids: set[str] = set()
                        _event_request_ids(event, all_request_ids, matched_ids)
                        destinations = {request_to_shard[request_id] for request_id in matched_ids}
                        if not destinations and event.get("event") == "scheduler_idle":
                            event_time = event.get("monotonic_ns")
                            if isinstance(event_time, int):
                                destinations = {
                                    shard for shard, bounds in shard_time_ranges.items()
                                    if bounds is not None and bounds[0] <= event_time <= bounds[1]
                                }
                    if destinations:
                        record = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
                        for destination in destinations:
                            streams[destination].write(record)
                            counts[destination] += 1
    finally:
        for destination in streams.values():
            destination.close()
    return counts


def write_run_shards(out: Path, levels: list[dict], seeds: list[int], calibration_seeds: list[int],
                     validation_seed: int, traces: dict, state: dict, gate: dict,
                     environment: dict, config: dict, mode: str) -> None:
    request_fields, requests = read_csv(out / "requests.csv")
    instance_fields, instances = read_csv(out / "server_logs" / f"{mode}_instance_instance.csv")
    migration_fields, migrations = read_csv(out / "migration_events.csv")
    gpu_fields, gpu_rows = read_csv(out / "gpu_metrics.csv")
    shard_request_ids: dict[Path, set[str]] = {}
    shard_time_ranges: dict[Path, tuple[int, int] | None] = {}
    for level in levels:
        for seed in seeds:
            label = level["label"]
            run_id = f"formal_{mode}_{label}_seed{seed}"
            role = "calibration" if seed in calibration_seeds else "validation"
            qps_tag = BASE.qps_label(level["qps"])
            shard = out / role / f"qps_{qps_tag}_seed{seed}"
            (shard / "server_logs").mkdir(parents=True, exist_ok=True)
            (shard / "traces").mkdir(exist_ok=True)
            run_rows = [row for row in requests if row.get("run_id") == run_id]
            shard_request_ids[shard] = {
                str(row["request_id"]) for row in run_rows if row.get("request_id")
            }
            candidate_starts = []
            for row in run_rows:
                try:
                    sent_mono = int(row.get("client_send_monotonic_ns", ""))
                    relative_sec = float(row.get("actual_send_sec", ""))
                    candidate_starts.append(sent_mono - int(relative_sec * 1_000_000_000))
                except (TypeError, ValueError):
                    continue
            candidate_ends = []
            for row in run_rows:
                try:
                    candidate_ends.append(int(row["client_complete_monotonic_ns"]))
                except (KeyError, TypeError, ValueError):
                    continue
            if not candidate_ends and candidate_starts:
                candidate_ends = [max(candidate_starts) + int(
                    (float(config["warmup_sec"]) + float(config["measure_sec"])
                     + float(config["drain_sec"])) * 1_000_000_000
                )]
            shard_time_ranges[shard] = (
                (min(candidate_starts), max(candidate_ends))
                if candidate_starts and candidate_ends else None
            )
            measure_rows = [row for row in run_rows if row.get("phase") == "measurement"]
            formal_sent = sum(1 for row in measure_rows if row.get("actual_send_sec", "") not in ("", "None"))
            trace = traces[(label, seed)]
            start_candidates = []
            for row in run_rows:
                if row.get("actual_send_wall_time") and row.get("actual_send_sec"):
                    try:
                        start_candidates.append(float(row["actual_send_wall_time"]) - float(row["actual_send_sec"]))
                    except (TypeError, ValueError):
                        pass
            if start_candidates:
                warmup_start = min(start_candidates)
            else:
                warmup_start = float("inf")
            measurement_start_candidates = []
            for row in measure_rows:
                if row.get("actual_send_wall_time") and row.get("actual_send_sec"):
                    try:
                        measurement_start_candidates.append(
                            float(row["actual_send_wall_time"]) - float(row["actual_send_sec"])
                        )
                    except (TypeError, ValueError):
                        pass
            measurement_start = min(measurement_start_candidates) if measurement_start_candidates else float("inf")
            measurement_end = measurement_start + float(config["measure_sec"])
            run_instances = []
            for row in instances:
                try:
                    timestamp = float(row.get("timestamp", "nan"))
                except (TypeError, ValueError):
                    continue
                if warmup_start <= timestamp <= measurement_end:
                    run_instances.append(row)
            run_migrations = [row for row in migrations if row.get("run_id") == run_id]
            run_gpu = [row for row in gpu_rows if row.get("run_id") == run_id]
            atomic_csv(shard / "requests.csv", request_fields, run_rows)
            atomic_csv(shard / "server_logs" / f"{mode}_instance_instance.csv", instance_fields, run_instances)
            atomic_csv(shard / "migration_events.csv", migration_fields, run_migrations)
            atomic_csv(shard / "gpu_metrics.csv", gpu_fields, run_gpu)
            trace_source = out / "traces" / f"trace_{label}_seed{seed}.csv"
            shutil.copy2(trace_source, shard / "traces" / trace_source.name)
            shutil.copy2(out / "environment_audit.md", shard / "environment_audit.md")
            shutil.copy2(out / "configs" / f"llumnix_{mode}.yml", shard / f"configs_llumnix_{mode}.yml")
            gate_copy = {**gate, "mode": mode, "reused_for_full_batch": True,
                         "total_qps_seed_runs": len(levels) * len(seeds)}
            BASE.atomic_json(shard / f"worker_startup_gate_{mode}.json", gate_copy)
            trace_hash = sha256(shard / "traces" / trace_source.name)
            def int_field(row: dict, name: str) -> int | None:
                try:
                    return int(row[name]) if row.get(name) not in (None, "", "None") else None
                except (KeyError, TypeError, ValueError):
                    return None

            measurement_mono = [int_field(row, "client_send_monotonic_ns") for row in measure_rows]
            measurement_wall = [int_field(row, "client_send_wall_time_ns") for row in measure_rows]
            measurement_offsets = []
            for row in measure_rows:
                mono, relative = int_field(row, "client_send_monotonic_ns"), row.get("actual_send_sec")
                if mono is not None:
                    try:
                        measurement_offsets.append(mono - int(float(relative) * 1_000_000_000))
                    except (TypeError, ValueError):
                        pass
            window_start_mono = min(measurement_offsets) if measurement_offsets else None
            window_start_wall = None
            if window_start_mono is not None and measurement_wall and any(x is not None for x in measurement_wall):
                wall_mono_pairs = [
                    (int_field(row, "client_send_wall_time_ns"), int_field(row, "client_send_monotonic_ns"))
                    for row in measure_rows
                ]
                offsets = [wall - mono for wall, mono in wall_mono_pairs if wall is not None and mono is not None]
                if offsets:
                    window_start_wall = window_start_mono + int(statistics.median(offsets))
            completion_mono = [int_field(row, "client_complete_monotonic_ns") for row in run_rows]
            completion_wall = [int_field(row, "client_complete_wall_time_ns") for row in run_rows]
            completed_mono = [value for value in completion_mono if value is not None]
            completed_wall = [value for value in completion_wall if value is not None]
            sent = state.get("runs", {}).get(run_id, {})
            run_config = {
                **config,
                "experiment_id": f"ppo_calibration_mixed_qps{qps_tag}_seed{seed}_{role}_seq{BASE.MAX_NUM_SEQS}",
                "qps": float(level["qps"]), "seed": int(seed), "dataset_role": role,
                "measurement_request_count": len(trace), "trace_request_count": len(trace),
                "trace_sha256": trace_hash, "output_dir": str(shard), "status": "complete",
                "effective_config_verified": True,
                "all_measurement_requests_sent": formal_sent == len(trace),
                "reused_server_instance_group": True,
                "startup_gate_verified_once_for_entire_batch": True,
                "configs": {mode: f"configs_llumnix_{mode}.yml"},
                "causal_event_count": 0,
                "causal_event_time_range_monotonic_ns": list(shard_time_ranges[shard])
                if shard_time_ranges[shard] is not None else None,
                "measurement_window": {
                    "start_monotonic_ns_estimate": window_start_mono,
                    "start_wall_time_ns_estimate": window_start_wall,
                    "end_monotonic_ns_estimate": (window_start_mono + int(float(config["measure_sec"]) * 1e9))
                    if window_start_mono is not None else None,
                    "end_wall_time_ns_estimate": (window_start_wall + int(float(config["measure_sec"]) * 1e9))
                    if window_start_wall is not None else None,
                    "first_request_send_monotonic_ns": min(x for x in measurement_mono if x is not None)
                    if any(x is not None for x in measurement_mono) else None,
                    "last_request_send_monotonic_ns": max(x for x in measurement_mono if x is not None)
                    if any(x is not None for x in measurement_mono) else None,
                    "observed_drain_end_monotonic_ns": max(completed_mono) if completed_mono else None,
                    "observed_drain_end_wall_time_ns": max(completed_wall) if completed_wall else None,
                },
            }
            BASE.atomic_json(shard / "config.json", run_config)
            run_state = {
                "experiment_id": run_config["experiment_id"], "status": "complete",
                "started_at_utc": state.get("started_at_utc"), "completed_at_utc": BASE.now_iso(),
                "completed_formal_runs": 1,
                "runs": {run_id: {**sent, "trace_requests": len(trace), "sent": formal_sent,
                                  "status": "complete" if formal_sent == len(trace) else "partial"}},
            }
            BASE.atomic_json(shard / "state.json", run_state)

    event_counts = split_causal_events(out, shard_request_ids, shard_time_ranges)
    for shard, count in event_counts.items():
        run_config_path = shard / "config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config["causal_event_count"] = count
        event_path = shard / "causal_events" / "events.jsonl"
        archived_event = compress_event_stream(event_path)
        archived_event["path"] = str(Path(archived_event["path"]).relative_to(shard))
        run_config["causal_events_inventory"] = [archived_event]
        BASE.atomic_json(run_config_path, run_config)
        BASE.atomic_json(shard / "causal_shard_archive_manifest.json", {
            "source": "per-run causal telemetry shard",
            "verified_lossless": True,
            "archives": [archived_event],
            "uncompressed_bytes": archived_event["uncompressed_bytes"],
            "compressed_bytes": archived_event["bytes"],
        })

    # The per-run copies are now independently archived and verified, so the
    # original root streams can also be compressed without losing provenance.
    root_event_dir = out / "causal_events"
    root_archives = [compress_event_stream(path) for path in sorted(root_event_dir.glob("events_*.jsonl"))]
    BASE.atomic_json(out / "causal_shard_archive_manifest.json", {
        "source": "root causal telemetry streams after verified per-run splitting",
        "verified_lossless": True,
        "archives": root_archives,
        "uncompressed_bytes": sum(item["uncompressed_bytes"] for item in root_archives),
        "compressed_bytes": sum(item["bytes"] for item in root_archives),
    })


async def run_batch(qps_values: list[float], calibration_seeds: list[int], validation_seed: int,
                    max_num_seqs: int, warmup_sec: float, measure_sec: float, drain_sec: float,
                    output_dir: str, resume: bool = False, mode: str = "on",
                    causal_trace: bool = True, validation_only: bool = False,
                    calibration_only: bool = False) -> int:
    if mode not in {"off", "on"}:
        raise ValueError(f"mode must be 'off' or 'on', got {mode!r}")
    if validation_only and calibration_only:
        raise ValueError("--validation-only and --calibration-only are mutually exclusive")
    out = Path(output_dir).resolve()
    if not out.exists():
        out.mkdir(parents=True)
    if any(any(out.iterdir()) for _ in (0,)):
        # collection.json is created by the parent collector before this runner starts.
        extras = [path.name for path in out.iterdir()
                  if path.name not in {"collection.json", "postprocess_status.json"}]
        if extras and not resume:
            raise FileExistsError(f"Output directory contains unexpected files: {extras}")
    resume_state_path = out / "state.json"
    if resume and not resume_state_path.exists():
        raise FileNotFoundError(f"Cannot resume without state.json: {resume_state_path}")
    for directory in ("configs", "traces", "server_logs", "raw_requests", "plots",
                      "calibration", "validation", "causal_events"):
        (out / directory).mkdir(exist_ok=True)
    if causal_trace:
        os.environ["LLUMNIX_CAUSAL_TRACE_DIR"] = str(out / "causal_events")
    else:
        os.environ.pop("LLUMNIX_CAUSAL_TRACE_DIR", None)

    seeds = ([validation_seed] if validation_only else
             list(calibration_seeds) if calibration_only else
             [*calibration_seeds, validation_seed])
    BASE.OUT = out
    BASE.SEEDS = [int(seed) for seed in seeds]
    BASE.MAX_NUM_SEQS = max_num_seqs
    BASE.GPU_MEMORY_UTIL = 0.5
    BASE.MODEL_MAX_LEN = MAX_MODEL_LEN
    BASE.ENFORCE_EAGER = True
    BASE.WARMUP_SEC = float(warmup_sec)
    BASE.MEASURE_SEC = float(measure_sec)
    BASE.DRAIN_SEC = float(drain_sec)
    BASE.REQUEST_TIMEOUT_SEC = None
    BASE.MONITOR_INTERVAL_SEC = 1.0
    BASE.INPUT_PROFILE = list(PROFILE_LENGTHS)
    BASE.OUTPUT_PROFILE = list(PROFILE_LENGTHS)
    BASE.PROFILE_MIX = PROFILE_MIX
    BASE.PROFILE_CYCLE = True
    BASE.ENABLE_CAUSAL_TRACE = causal_trace
    # Keep the budget as an upper bound; the model's normal EOS behavior is part
    # of the output-length distribution being measured.
    BASE.IGNORE_EOS = False

    if not BASE.MODEL.exists():
        raise RuntimeError(f"Model directory not found: {BASE.MODEL}")
    BASE.owned_process_preflight()
    environment = BASE.system_snapshot()
    if environment["gpu_count"] != 4 or not environment["torch"].get("cuda_available") \
            or environment["torch"].get("device_count") != 4:
        raise RuntimeError(f"Expected four usable GPUs; environment={environment}")
    topology = BASE.run_cmd(["nvidia-smi", "topo", "-m"], check=False).stdout
    environment["gpu_topology"] = topology
    (out / "environment_audit.md").write_text(
        BASE.environment_audit(environment) + "\n## GPU topology\n\n```text\n" + topology + "```\n",
        encoding="utf-8",
    )

    levels = [{"label": f"qps_{BASE.qps_label(qps)}", "qps": float(qps)} for qps in qps_values]
    config_path = out / "configs" / f"llumnix_{mode}.yml"
    mode_config = BASE.yaml_config(mode, "balanced", migrate_out_threshold=THRESHOLD,
                                   pair_migration_frequency=FREQUENCY)
    mode_config["INSTANCE"]["ENABLE_DEFRAG"] = DEFRAG
    config_path.write_text(BASE.yaml.safe_dump(mode_config, sort_keys=False), encoding="utf-8")
    if bool(mode_config["MANAGER"].get("ENABLE_ROUTINE_MIGRATION")) != (mode == "on"):
        raise RuntimeError(f"Migration {mode.upper()} configuration has wrong routine migration setting")

    traces = BASE.create_traces(levels)
    for level in levels:
        for seed in seeds:
            trace = traces[(level["label"], seed)]
            observed = {(req.input_tokens, req.output_tokens) for req in trace}
            if not trace or observed != set(REQUEST_PROFILES):
                raise RuntimeError(f"Trace {level['label']} seed {seed} did not include all nine profiles")

    config = {
        "experiment_id": "ppo_calibration_mixed_2112_eager_reused_server",
        "created_at_utc": BASE.now_iso(), "environment": environment,
        "model": {"path": str(BASE.MODEL), "max_model_len": BASE.MODEL_MAX_LEN,
                  "tensor_parallel_size": 1},
        "model_path": str(BASE.MODEL), "qps_levels": qps_values,
        "model_id": {
            "path": str(BASE.MODEL),
            "weights_sha256": sha256(BASE.MODEL / "model.safetensors"),
            "config_sha256": sha256(BASE.MODEL / "config.json"),
        },
        "calibration_seeds": calibration_seeds, "validation_seed": validation_seed,
        "validation_only_collection": bool(validation_only),
        "calibration_only_collection": bool(calibration_only),
        "dataset_role": "batch", "mode": mode, "policy": "balanced", "threshold": THRESHOLD,
        "frequency": FREQUENCY, "input_profiles": list(PROFILE_LENGTHS),
        "output_profiles": list(PROFILE_LENGTHS), "request_mode_count": 9,
        "measure_sec": measure_sec,
        "measurement_window_contract": {
            "warmup_sec_per_run": warmup_sec, "measurement_sec_per_run": measure_sec,
            "drain_sec_per_run": drain_sec, "arrival_model": "poisson",
            "trace_ids_are_shared_across_modes": True,
        },
        "runtime": {"initial_instances": 4, "tensor_parallel_size": 1,
                    "gpu_memory_utilization": BASE.GPU_MEMORY_UTIL,
                    "max_model_len": BASE.MODEL_MAX_LEN,
                    "max_num_seqs_per_instance": BASE.MAX_NUM_SEQS,
                    "aggregate_max_num_seqs": BASE.MAX_NUM_SEQS * 4, "dtype": "half",
                    "enforce_eager": True, "dispatch_policy": "load",
                    "load_metric": "remaining_steps", "pair_migration_policy": "balanced",
                    "migration_backend": "rayrpc", "enable_defrag": DEFRAG,
                    "migrate_out_threshold": THRESHOLD,
                    "pair_migration_frequency": FREQUENCY,
                    "routine_migration_enabled": mode == "on",
                    "placement_group_ready_timeout_sec": float(
                        os.environ.get("LLUMNIX_WAIT_PLACEMENT_GROUP_TIMEOUT", "5.0")
                    ),
                    "init_instances_rpc_timeout_sec": float(os.environ.get(
                        "LLUMNIX_INIT_INSTANCES_TIMEOUT",
                        str(max(
                            60.0,
                            4 * float(os.environ.get("LLUMNIX_WAIT_PLACEMENT_GROUP_TIMEOUT", "5.0")) + 30.0,
                        )),
                    )),
                    "instance_ready_timeout_sec": float(
                        os.environ.get("LLUMNIX_INSTANCE_READY_TIMEOUT", "900.0")
                    ),
                    "instance_readiness_checks_concurrent": True,
                    "server_startup_timeout_sec": float(
                        os.environ.get("LLUMNIX_SERVER_STARTUP_TIMEOUT", "900.0")
                    )},
        "workload": {"arrival": "poisson", "input_profiles": list(PROFILE_LENGTHS),
                     "output_profiles": list(PROFILE_LENGTHS), "profile_mix": PROFILE_MIX,
                     "profile_cycle": True},
        "profile_mix": PROFILE_MIX, "warmup_sec": warmup_sec,
        "drain_sec": drain_sec, "request_timeout_sec": None,
        "arrival_model": "poisson", "reused_four_instance_server": True,
        "server_startups_for_entire_dataset": 1,
        "causal_telemetry": causal_trace,
        "causal_event_schema_version": 1 if causal_trace else None,
        "ignore_eos": BASE.IGNORE_EOS,
        "engine_latency_measurement": {
            "metric": "executor_iteration_wall_ms",
            "clock": "time.perf_counter",
            "scope": "from Ray executor dispatch through awaited worker result",
            "ray_spmd_and_non_spmd": True,
            "cuda_synchronize_added": False,
        },
        "configs": {mode: str(config_path.relative_to(out))},
        **source_manifest(),
        "output_dir": str(out), "status": "running",
    }
    archive_instance_log = None
    if resume:
        state = json.loads(resume_state_path.read_text(encoding="utf-8"))
        if state.get("experiment_id") != config["experiment_id"]:
            raise RuntimeError(f"Resume experiment id differs: {state.get('experiment_id')!r}")
        archive_instance_log = prepare_resume_artifacts(out, state, mode)
        if state.get("failure"):
            state.setdefault("failure_history", []).append({
                "failure": state["failure"], "failed_at_utc": state.get("failed_at_utc")
            })
        state.pop("failure", None)
        state.pop("failed_at_utc", None)
        state["status"] = "running"
        state["resumed_at_utc"] = BASE.now_iso()
        print(f"[resume] preserving completed windows and retrying incomplete ones; "
              f"completed={state.get('completed_formal_runs', 0)}", flush=True)
    else:
        state = {"experiment_id": config["experiment_id"], "status": "running",
                 "started_at_utc": BASE.now_iso(), "runs": {}, "completed_formal_runs": 0}
    BASE.atomic_json(out / "batch_config.json", config)
    BASE.atomic_json(out / "state.json", state)

    original_wait_ready = BASE.wait_ready
    original_verify_effective_config = BASE.verify_effective_config

    def startup_gate_snapshot(log_path: Path) -> dict | None:
        startup_log = log_path.read_text(errors="replace")
        ready_counts = [int(value) for value in re.findall(
            r"Init Llumnix components done, (\d+) instances are ready", startup_log
        )]
        manager_states = re.findall(r"num_instances:\s*(\d+),\s*instances:\s*\{([^}]*)\}", startup_log)
        worker_allocations = {
            worker_id: gpu_id for worker_id, gpu_id in re.findall(
                r"Llumlet\(actor_name=instance_([0-9a-f]{32}).*?gpu_ids=\['(\d+)'\]", startup_log
            )
        }
        manager_matches = []
        for count_text, instance_text in manager_states:
            instance_ids = set(re.findall(r"[0-9a-f]{32}", instance_text))
            if int(count_text) == 4 and len(instance_ids) == 4:
                manager_matches.append(instance_ids)
        if not ready_counts or max(ready_counts) != 4 or not manager_matches:
            return None
        registered_ids = manager_matches[-1]
        if (set(worker_allocations) != registered_ids
                or set(worker_allocations.values()) != {"0", "1", "2", "3"}):
            return None
        return {
            "passed": True, "mode": mode, "enforce_eager": True,
            "ready_instances_reported": max(ready_counts), "manager_instance_count": 4,
            "manager_instance_ids": sorted(registered_ids),
            "worker_gpu_assignments": worker_allocations,
            "readiness_monitor_stopped_before_workload": True,
            "verified_at_utc": BASE.now_iso(),
        }

    server_startup_timeout = float(os.environ.get("LLUMNIX_SERVER_STARTUP_TIMEOUT", "900.0"))

    async def wait_ready_for_four_workers(process, log_path: Path, timeout_sec: float | None = None) -> None:
        if timeout_sec is None:
            timeout_sec = server_startup_timeout
        await original_wait_ready(process, log_path, timeout_sec=timeout_sec)
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = log_path.read_text(errors="replace")[-10000:] if log_path.exists() else ""
                raise RuntimeError(f"Llumnix exited before four-worker gate; code={process.returncode}; log tail:\n{tail}")
            gate = startup_gate_snapshot(log_path)
            if gate is not None:
                ray_status = await BASE.verify_ray_four_gpu()
                gate["ray_gpu_status"] = ray_status
                gate["ray_gpu_resources"] = 4
                BASE.atomic_json(out / f"worker_startup_gate_{mode}.json", gate)
                print(f"[gate] all four workers ready on GPUs 0-3; this group will serve "
                      f"{len(levels) * len(seeds)} windows", flush=True)
                return
            await asyncio.sleep(2)
        tail = log_path.read_text(errors="replace")[-10000:] if log_path.exists() else ""
        raise TimeoutError(f"Four-worker startup gate deadline; log tail:\n{tail}")

    def verify_effective_config(log_path, *args, **kwargs):
        gate = startup_gate_snapshot(log_path)
        if gate is None:
            raise RuntimeError("The four-worker startup gate was not passed before testing")
        if "enforce_eager=True" not in log_path.read_text(errors="replace"):
            raise RuntimeError("Effective vLLM configuration did not confirm enforce_eager=True")
        original_verify_effective_config(log_path, *args, **kwargs)
        current = json.loads((out / f"worker_startup_gate_{mode}.json").read_text(encoding="utf-8"))
        current["passed"] = True
        BASE.atomic_json(out / f"worker_startup_gate_{mode}.json", current)

    BASE.wait_ready = wait_ready_for_four_workers
    BASE.verify_effective_config = verify_effective_config
    gpu_rows, migration_events, capacity_rows = [], [], None
    try:
        try:
            await BASE.run_mode(
                mode, config_path, {"experiment_id": config["experiment_id"]}, state,
                out / "raw_requests", gpu_rows, migration_events, traces, levels, "balanced",
                expected_migration_enabled=(mode == "on"),
                expected_migrate_out_threshold=THRESHOLD,
                expected_pair_migration_frequency=FREQUENCY,
            )
        finally:
            merge_preserved_instance_log(out, archive_instance_log)
        BASE.write_final_requests(out / "raw_requests", out / "requests.csv")
        migration_path = out / "migration_events.csv"
        if not migration_path.exists():
            migration_path.write_text(",".join(BASE.MIGRATION_FIELDS) + "\n", encoding="utf-8")
        summary = BASE.aggregate_summary(out / "requests.csv", out / "gpu_metrics.csv",
                                         migration_path, out / "summary.csv")
        BASE.plot_results(summary, out / "plots")
        gate = json.loads((out / f"worker_startup_gate_{mode}.json").read_text(encoding="utf-8"))
        write_run_shards(out, levels, seeds, calibration_seeds, validation_seed,
                         traces, state, gate, environment,
                         {**config, "measure_sec": measure_sec}, mode)
        all_sent = all(
            state.get("runs", {}).get(f"formal_{mode}_{level['label']}_seed{seed}", {}).get("sent")
            == len(traces[(level["label"], seed)])
            for level in levels for seed in seeds
        )
        state["status"] = "complete" if all_sent else "partial"
        state["completed_at_utc"] = BASE.now_iso()
        config["status"] = state["status"]
        config["effective_config_verified"] = gate.get("passed") is True
        config["all_measurement_requests_sent"] = all_sent
        config["completed_formal_runs"] = state["completed_formal_runs"]
        config["causal_event_inventory"] = compress_root_event_streams(out)
        BASE.atomic_json(out / "state.json", state)
        BASE.atomic_json(out / "batch_config.json", config)
        print(f"[done] reused one eager server for {len(levels) * len(seeds)} QPS/seed windows; "
              f"all_measurements_sent={all_sent}; output={out}", flush=True)
        return 0 if all_sent else 2
    except BaseException as exc:
        state["status"] = "failed_or_interrupted"
        state["failure"] = f"{type(exc).__name__}: {exc}"
        state["failed_at_utc"] = BASE.now_iso()
        BASE.atomic_json(out / "state.json", state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", type=float, nargs="+", required=True)
    parser.add_argument("--calibration-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--validation-seed", type=int, required=True)
    parser.add_argument("--warmup-sec", type=float, default=30.0)
    parser.add_argument("--measure-sec", type=float, default=200.0)
    parser.add_argument("--drain-sec", type=float, default=15.0)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("off", "on"), default="on")
    parser.add_argument("--no-causal-trace", action="store_true",
                        help="disable causal telemetry for a paired overhead control")
    parser.add_argument("--validation-only", action="store_true",
                        help="collect only validation-seed runs; calibration seeds remain recorded in metadata")
    parser.add_argument("--calibration-only", action="store_true",
                        help="collect only the declared calibration seeds; validation seed remains recorded in metadata")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_num_seqs < 1 or len(args.qps) != len(set(args.qps)):
        parser.error("--max-num-seqs must be positive and --qps values unique")
    if args.validation_seed in args.calibration_seeds or len(args.calibration_seeds) != len(set(args.calibration_seeds)):
        parser.error("validation seed must be distinct from unique calibration seeds")
    try:
        return asyncio.run(run_batch(args.qps, args.calibration_seeds, args.validation_seed,
                                     args.max_num_seqs, args.warmup_sec, args.measure_sec,
                                     args.drain_sec, args.output_dir, args.resume, args.mode,
                                     not args.no_causal_trace, args.validation_only,
                                     args.calibration_only))
    except KeyboardInterrupt:
        process = BASE.CURRENT_SERVER_PROCESS
        if process is not None and process.poll() is None:
            try:
                process.send_signal(BASE.signal.SIGTERM)
                process.wait(timeout=90)
            except Exception:
                process.kill()
                process.wait(timeout=20)
        if BASE.CURRENT_CLUSTER_OWNED:
            BASE.run_cmd([str(ROOT / ".venv" / "bin" / "ray"), "stop", "--force"], timeout=90, check=False)
        return 130
    except Exception as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=BASE.sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
