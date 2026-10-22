#!/usr/bin/env python3
"""Run Migration OFF/ON with matched prompts, arrivals, and train-sampled output lengths."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SWEEP_ROOT = ROOT / "benchmark" / "results" / "ppo_system_validation_round3_fixed_lengths_20261022"
PILOT_ROOT = SWEEP_ROOT
SOURCE = ROOT / "benchmark" / "results" / "ppo_calibration_mixed_2112_eager_seq128_latencyfix_20261021_162505"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))

import run_baseline_4gpu as BASE  # noqa: E402
from disturbance_exp.workload import TraceRequest  # noqa: E402
from rl_scheduler.system_validation import _output_length_model  # noqa: E402


QPS = 15.0
SEED = 505
MEASURE_SEC = 60.0
DRAIN_SEC = 30.0
WARMUP_SEC = 15.0
THRESHOLD = 9000.0
FREQUENCY = 1
POLICY = "balanced"
FORCED_REQUEST_IDS: set[str] = set()
CURRENT_OUTPUT_LENGTHS: dict[str, int] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path: Path) -> list[TraceRequest]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [TraceRequest(
            request_id=row["request_id"],
            scheduled_arrival_sec=float(row["scheduled_arrival_sec"]),
            request_type=row["request_type"],
            prompt=row["prompt"],
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
        ) for row in csv.DictReader(stream)]


def _fixed_length_manifest(trace: list[TraceRequest], qps: float, manifest_path: Path) -> dict[str, int]:
    model, audit = _output_length_model(SOURCE)
    profiles = model["profiles"]
    seed = SEED * 1_000_000 + int(round(qps * 1000))
    rng = np.random.default_rng(seed)
    sampled: dict[str, int] = {}
    rows = []
    for request in trace:
        key = f"{request.input_tokens}:{request.output_tokens}"
        profile_key = key if key in profiles else f"*:{request.output_tokens}"
        profile = profiles.get(profile_key)
        if not profile or not profile.get("lengths"):
            raise RuntimeError(f"No training calibration length distribution for profile {key}")
        lengths = [int(value) for value in profile["lengths"]]
        counts = [int(value) for value in profile["counts"]]
        value = int(rng.choice(lengths, p=np.asarray(counts, dtype=float) / sum(counts)))
        if not 1 <= value <= request.output_tokens:
            raise RuntimeError(f"Training sample {value} exceeds request budget {request.output_tokens}")
        sampled[request.request_id] = value
        rows.append({"request_id": request.request_id, "input_tokens": request.input_tokens,
                     "output_budget_tokens": request.output_tokens,
                     "realized_output_tokens": value, "training_profile_key": profile_key,
                     "training_profile_n": int(profile["n"]),
                     "sample_seed": seed, "sample_source": "calibration_measurement_successes"})
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            existing = {row["request_id"]: int(row["realized_output_tokens"])
                        for row in csv.DictReader(stream)}
        if existing != sampled:
            raise RuntimeError("Existing fixed output length manifest does not match the deterministic training sample")
    else:
        with manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if len(sampled) != len(trace) or len(Counter(sampled.values())) == 0:
        raise RuntimeError("Fixed output length manifest is incomplete")
    BASE.atomic_json(manifest_path.with_suffix(".audit.json"), {
        "source": audit["source"], "n_training_successful_lengths": audit["n_successful_lengths"],
        "training_profile_sample_counts": audit["profile_sample_counts"],
        "held_out_response_labels_used": False, "qps": qps, "seed": SEED,
        "sample_seed": seed, "manifest_rows": len(rows),
        "length_counts": {str(value): count for value, count in sorted(Counter(sampled.values()).items())},
    })
    return sampled


def write_fixed_trace(source: Path, target: Path, sampled: dict[str, int]) -> None:
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0]) + ["realized_output_tokens", "realized_output_source"]
    for row in rows:
        row["realized_output_tokens"] = sampled[row["request_id"]]
        row["realized_output_source"] = "fixed_training_sample"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_globals() -> tuple[dict, Path, Path, list[TraceRequest], dict[str, int]]:
    source_config = json.loads((SOURCE / "batch_config.json").read_text(encoding="utf-8"))
    runtime = source_config["runtime"]
    workload = source_config["workload"]
    BASE.MODEL_MAX_LEN = int(source_config["model"]["max_model_len"])
    BASE.MAX_NUM_SEQS = int(runtime["max_num_seqs_per_instance"])
    BASE.GPU_MEMORY_UTIL = float(runtime["gpu_memory_utilization"])
    BASE.WARMUP_SEC = WARMUP_SEC
    BASE.MEASURE_SEC = MEASURE_SEC
    BASE.DRAIN_SEC = DRAIN_SEC
    BASE.REQUEST_TIMEOUT_SEC = None
    BASE.MONITOR_INTERVAL_SEC = 1.0
    BASE.INPUT_PROFILE = list(source_config["input_profiles"])
    BASE.OUTPUT_PROFILE = list(source_config["output_profiles"])
    BASE.PROFILE_MIX = list(workload["profile_mix"])
    BASE.PROFILE_CYCLE = bool(workload["profile_cycle"])
    BASE.SEEDS = [SEED]

    if not BASE.MODEL.exists():
        raise RuntimeError(f"Model directory is missing: {BASE.MODEL}")
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    trace_path = PILOT_ROOT / f"trace_qps_{BASE.qps_label(QPS)}_seed{SEED}.csv"
    spec_path = PILOT_ROOT / "pilot_spec.json"
    spec = {
        "experiment_id": "mig_ppo_seed505_fixed_output_lengths_round3",
        "source_calibration_config": str((SOURCE / "batch_config.json").relative_to(ROOT)),
        "qps": QPS,
        "seed": SEED,
        "measurement_sec": MEASURE_SEC,
        "drain_sec_after_arrivals": DRAIN_SEC,
        "shared_result_horizon_sec": MEASURE_SEC + DRAIN_SEC,
        "warmup_sec": WARMUP_SEC,
        "max_model_len": BASE.MODEL_MAX_LEN,
        "max_num_seqs_per_instance": BASE.MAX_NUM_SEQS,
        "gpu_memory_utilization": BASE.GPU_MEMORY_UTIL,
        "enforce_eager": True,
        "profile_mix": BASE.PROFILE_MIX,
        "profile_cycle": BASE.PROFILE_CYCLE,
        "migration_policy": POLICY,
        "migrate_out_threshold": THRESHOLD,
        "pair_migration_frequency": FREQUENCY,
        "mode_config_difference": "MANAGER.ENABLE_ROUTINE_MIGRATION only",
        "output_length_policy": "per-request realized output lengths sampled from calibration seeds 101/202/303 successes; forced with min_tokens=max_tokens; same manifest in OFF and ON",
        "generation_seed_policy": "request seed = experiment seed * 1,000,000 + numeric trace request suffix; same seed in OFF and ON",
    }
    if spec_path.exists():
        recorded = json.loads(spec_path.read_text(encoding="utf-8"))
        for key, value in spec.items():
            if recorded.get(key) != value:
                raise RuntimeError(f"Pilot spec differs at {key}: {recorded.get(key)!r} != {value!r}")
    else:
        BASE.atomic_json(spec_path, spec)

    if not trace_path.exists():
        trace = BASE.generated_trace(SEED, MEASURE_SEC, QPS)
        if not trace:
            raise RuntimeError("Trace generator returned no requests")
        BASE.write_trace(trace_path, trace)
    trace = load_trace(trace_path)
    profiles = {(int(item["input_tokens"]), int(item["output_tokens"])) for item in BASE.PROFILE_MIX}
    observed = {(row.input_tokens, row.output_tokens) for row in trace}
    if not trace or not profiles.issubset(observed):
        raise RuntimeError(f"Trace is missing calibrated profiles: {profiles - observed}")
    if any(row.scheduled_arrival_sec >= MEASURE_SEC for row in trace):
        raise RuntimeError("Trace includes an arrival outside the 60-second work window")

    configs_dir = PILOT_ROOT / "configs"
    configs_dir.mkdir(exist_ok=True)
    off_path, on_path = configs_dir / "llumnix_off.yml", configs_dir / "llumnix_on.yml"
    if not off_path.exists() and not on_path.exists():
        for mode, path in (("off", off_path), ("on", on_path)):
            mode_config = BASE.yaml_config(mode, POLICY, migrate_out_threshold=THRESHOLD,
                                           pair_migration_frequency=FREQUENCY)
            mode_config["INSTANCE"]["ENABLE_DEFRAG"] = False
            path.write_text(BASE.yaml.safe_dump(mode_config, sort_keys=False), encoding="utf-8")
    if not off_path.exists() or not on_path.exists():
        raise RuntimeError("Both migration mode configs must exist together")
    BASE.check_mode_configs({"off": off_path, "on": on_path})
    sampled = _fixed_length_manifest(trace, QPS, PILOT_ROOT / "fixed_output_lengths.csv")
    return spec, trace_path, off_path, trace, sampled


def startup_gate_snapshot(log_path: Path, mode: str) -> dict | None:
    log = log_path.read_text(errors="replace")
    ready_counts = [int(value) for value in re.findall(
        r"Init Llumnix components done, (\d+) instances are ready", log
    )]
    registrations = re.findall(r"num_instances:\s*(\d+),\s*instances:\s*\{([^}]*)\}", log)
    registered_ids = set()
    for count_text, instance_text in registrations:
        ids = set(re.findall(r"[0-9a-f]{32}", instance_text))
        if int(count_text) == 4 and len(ids) == 4:
            registered_ids = ids
    allocations = {
        worker_id: gpu_id for worker_id, gpu_id in re.findall(
            r"Llumlet\(actor_name=instance_([0-9a-f]{32}).*?gpu_ids=\['(\d+)'\]", log
        )
    }
    if (not ready_counts or max(ready_counts) != 4 or len(registered_ids) != 4
            or set(allocations) != registered_ids
            or set(allocations.values()) != {"0", "1", "2", "3"}
            or "enforce_eager=True" not in log):
        return None
    return {
        "passed": True,
        "mode": mode,
        "enforce_eager": True,
        "ready_instances_reported": max(ready_counts),
        "manager_instance_count": len(registered_ids),
        "manager_instance_ids": sorted(registered_ids),
        "worker_gpu_assignments": allocations,
        "ray_status_checked_by_runner": True,
        "verified_at_utc": BASE.now_iso(),
    }


async def run(mode: str) -> int:
    global CURRENT_OUTPUT_LENGTHS
    spec, trace_path, off_path, trace, sampled_output_lengths = prepare_globals()
    CURRENT_OUTPUT_LENGTHS = sampled_output_lengths
    out = PILOT_ROOT / mode
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite pilot output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    BASE.OUT = out
    for name in ("traces", "server_logs", "raw_requests", "plots"):
        (out / name).mkdir(exist_ok=True)
    BASE.owned_process_preflight()
    config_path = out / "configs" / f"llumnix_{mode}.yml"
    config_path.parent.mkdir(exist_ok=True)
    shutil.copy2(PILOT_ROOT / "configs" / f"llumnix_{mode}.yml", config_path)
    environment = BASE.system_snapshot()
    if (environment["gpu_count"] != 4 or not environment["torch"].get("cuda_available")
            or environment["torch"].get("device_count") != 4):
        raise RuntimeError(f"Expected four usable GPUs, observed: {environment}")
    topology = BASE.run_cmd(["nvidia-smi", "topo", "-m"], check=False).stdout
    environment["gpu_topology"] = topology
    (out / "environment_audit.md").write_text(
        BASE.environment_audit(environment) + "\n## GPU topology\n\n```text\n" + topology + "```\n",
        encoding="utf-8",
    )
    trace_hash = sha256(trace_path)
    run_id = f"formal_{mode}_qps_{BASE.qps_label(QPS)}_seed{SEED}"
    level = {"label": f"qps_{BASE.qps_label(QPS)}", "qps": QPS}
    trace_copy = out / "traces" / f"trace_{level['label']}_seed{SEED}.csv"
    write_fixed_trace(trace_path, trace_copy, sampled_output_lengths)
    mode_config = BASE.yaml.safe_load(config_path.read_text(encoding="utf-8"))
    report_config = {
        **spec,
        "mode": mode,
        "run_id": run_id,
        "trace_path": str(trace_path.relative_to(ROOT)),
        "trace_sha256": trace_hash,
        "fixed_output_length_manifest": str((PILOT_ROOT / "fixed_output_lengths.csv").relative_to(ROOT)),
        "fixed_output_length_manifest_sha256": sha256(PILOT_ROOT / "fixed_output_lengths.csv"),
        "trace_requests": len(trace),
        "mode_config_path": str(config_path.relative_to(ROOT)),
        "mode_config": mode_config,
        "environment": environment,
        "command": f"{sys.executable} {Path(__file__).relative_to(ROOT)} --mode {mode} --qps {QPS:g} --seed {SEED}",
        "status": "running",
        "started_at_utc": BASE.now_iso(),
    }
    BASE.atomic_json(out / "config.json", report_config)
    state = {"experiment_id": spec["experiment_id"], "status": "running",
             "started_at_utc": BASE.now_iso(), "runs": {}, "completed_formal_runs": 0}
    BASE.atomic_json(out / "state.json", state)

    original_wait_ready = BASE.wait_ready
    original_verify = BASE.verify_effective_config

    async def wait_ready_with_gate(process, log_path, timeout_sec=900.0):
        await original_wait_ready(process, log_path, timeout_sec=timeout_sec)
        deadline = asyncio.get_running_loop().time() + timeout_sec
        while asyncio.get_running_loop().time() < deadline:
            if process.poll() is not None:
                tail = log_path.read_text(errors="replace")[-10000:]
                raise RuntimeError(f"Server exited before four-worker gate; tail:\n{tail}")
            gate = startup_gate_snapshot(log_path, mode)
            if gate is not None:
                ray_status = await BASE.verify_ray_four_gpu()
                gate["ray_gpu_status"] = ray_status
                gate["ray_gpu_resources"] = 4
                BASE.atomic_json(out / f"worker_startup_gate_{mode}.json", gate)
                return
            await asyncio.sleep(2)
        raise TimeoutError("Four-worker, one-GPU-per-worker startup gate not reached")

    def verify_with_gate(log_path, *args, **kwargs):
        gate_path = out / f"worker_startup_gate_{mode}.json"
        if not gate_path.exists() or not json.loads(gate_path.read_text())["passed"]:
            raise RuntimeError("Four-worker startup gate was not passed")
        gate = json.loads(gate_path.read_text())
        if gate["mode"] != mode or gate["enforce_eager"] is not True:
            raise RuntimeError("Startup gate mode/eager settings do not match this run")
        original_verify(log_path, *args, **kwargs)

    BASE.wait_ready = wait_ready_with_gate
    BASE.verify_effective_config = verify_with_gate
    gpu_rows: list[dict] = []
    migration_events: list[dict] = []
    try:
        await BASE.run_mode(
            mode, config_path, {"experiment_id": spec["experiment_id"]}, state,
            out / "raw_requests", gpu_rows, migration_events,
            {(level["label"], SEED): trace}, [level], POLICY,
            expected_migration_enabled=(mode == "on"),
            expected_migrate_out_threshold=THRESHOLD,
            expected_pair_migration_frequency=FREQUENCY,
        )
        BASE.write_final_requests(out / "raw_requests", out / "requests.csv")
        with (out / "requests.csv").open(newline="", encoding="utf-8") as stream:
            measured_rows = [row for row in csv.DictReader(stream) if row.get("phase") == "measurement"]
        mismatches = []
        for row in measured_rows:
            if str(row.get("success", "")).lower() != "true":
                continue
            match = re.search(r"(r\d{5,})$", str(row.get("request_id", "")))
            request_suffix = match.group(1) if match else ""
            target = sampled_output_lengths.get(request_suffix)
            actual = int(float(row["server_output_tokens"])) if row.get("server_output_tokens") else None
            if target is None or actual != target:
                mismatches.append({"request_id": row.get("request_id"), "target": target, "actual": actual})
        successful_count = sum(str(row.get("success", "")).lower() == "true" for row in measured_rows)
        output_length_audit = {
            "passed": bool(successful_count and not mismatches),
            "mode": mode, "qps": QPS, "manifest_rows": len(sampled_output_lengths),
            "measurement_requests_forced": len(FORCED_REQUEST_IDS),
            "measurement_rows": len(measured_rows), "successful_rows": successful_count,
            "successful_output_length_mismatches": mismatches,
            "forced_min_tokens_equals_max_tokens": True,
            "verified_at_utc": BASE.now_iso(),
        }
        BASE.atomic_json(out / "output_length_enforcement_audit.json", output_length_audit)
        if not output_length_audit["passed"]:
            raise RuntimeError(f"Fixed output length enforcement failed: {output_length_audit}")
        if not (out / "migration_events.csv").exists():
            (out / "migration_events.csv").write_text(",".join(BASE.MIGRATION_FIELDS) + "\n", encoding="utf-8")
        summary = BASE.aggregate_summary(out / "requests.csv", out / "gpu_metrics.csv",
                                         out / "migration_events.csv", out / "summary.csv")
        result = BASE.calculate_metrics(
            pd_read(out / "requests.csv"), pd_read(out / "gpu_metrics.csv"),
            pd_read(out / "migration_events.csv"), QPS, duration=MEASURE_SEC,
        )
        result.update({"mode": mode, "trace_sha256": trace_hash, "trace_requests": len(trace),
                       "shared_result_horizon_sec": MEASURE_SEC + DRAIN_SEC,
                       "scheduled_arrival_window_sec": MEASURE_SEC,
                       "startup_gate": json.loads((out / f"worker_startup_gate_{mode}.json").read_text()),
                       "summary_rows": int(len(summary))})
        BASE.atomic_json(out / "pilot_metrics.json", result)
        state["status"] = "complete"
        state["completed_at_utc"] = BASE.now_iso()
        BASE.atomic_json(out / "state.json", state)
        report_config["status"] = "complete"
        report_config["completed_at_utc"] = BASE.now_iso()
        BASE.atomic_json(out / "config.json", report_config)
        return 0
    except BaseException as exc:
        state["status"] = "failed_or_interrupted"
        state["failure"] = f"{type(exc).__name__}: {exc}"
        state["failed_at_utc"] = BASE.now_iso()
        BASE.atomic_json(out / "state.json", state)
        report_config["status"] = state["status"]
        report_config["failure"] = state["failure"]
        report_config["failed_at_utc"] = state["failed_at_utc"]
        BASE.atomic_json(out / "config.json", report_config)
        raise
    finally:
        BASE.wait_ready = original_wait_ready
        BASE.verify_effective_config = original_verify


def pd_read(path: Path):
    import pandas as pd
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def main() -> int:
    global PILOT_ROOT, QPS, SEED, FORCED_REQUEST_IDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--qps", type=float, choices=(9.0, 12.0, 15.0, 18.0, 21.0), required=True)
    parser.add_argument("--seed", type=int, default=505)
    args = parser.parse_args()

    QPS = args.qps
    SEED = args.seed
    FORCED_REQUEST_IDS = set()
    PILOT_ROOT = SWEEP_ROOT / f"qps_{BASE.qps_label(QPS)}"
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)

    # Force the same training-sampled realized output length in OFF and ON.
    original_post = BASE.aiohttp.ClientSession.post

    def seeded_post(session, url, *post_args, **post_kwargs):
        payload = post_kwargs.get("json")
        if str(url).rstrip("/").endswith("/generate_benchmark") and isinstance(payload, dict):
            request_id = str(payload.get("request_id", ""))
            match = re.search(r"(r\d{5,})$", request_id)
            request_suffix = match.group(1) if match else ""
            payload = dict(payload)
            if "_measurement_" in request_id:
                if request_suffix not in CURRENT_OUTPUT_LENGTHS:
                    raise RuntimeError(f"No fixed output length for measurement request {request_id}")
                target = int(CURRENT_OUTPUT_LENGTHS[request_suffix])
                payload["min_tokens"] = target
                payload["max_tokens"] = target
                payload["ignore_eos"] = True
                FORCED_REQUEST_IDS.add(request_suffix)
            payload["seed"] = (SEED * 1_000_000 + int(request_suffix[1:])) % (2**32 - 1) if request_suffix else SEED
            post_kwargs["json"] = payload
        return original_post(session, url, *post_args, **post_kwargs)

    BASE.aiohttp.ClientSession.post = seeded_post
    return asyncio.run(run(args.mode))


if __name__ == "__main__":
    raise SystemExit(main())
