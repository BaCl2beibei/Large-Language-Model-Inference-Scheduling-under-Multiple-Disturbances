#!/usr/bin/env python3
"""Collect the full PPO dataset while reusing one eager four-instance server."""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import hashlib
import json
import re
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd

import run_baseline_4gpu as BASE


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def prepare_resume_artifacts(out: Path, state: dict) -> Path | None:
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
    while (server_dir / f"on_attempt_{attempt}.log").exists():
        attempt += 1
    old_server_log = server_dir / "on.log"
    if old_server_log.exists():
        shutil.copy2(old_server_log, server_dir / f"on_attempt_{attempt}.log")
    old_instance_log = server_dir / "on_instance_instance.csv"
    archive_instance_log = server_dir / f"on_instance_instance_attempt_{attempt}.csv"
    if old_instance_log.exists():
        shutil.copy2(old_instance_log, archive_instance_log)
        return archive_instance_log
    return None


def merge_preserved_instance_log(out: Path, archive_path: Path | None) -> None:
    current_path = out / "server_logs" / "on_instance_instance.csv"
    if archive_path is None or not archive_path.exists() or not current_path.exists():
        return
    old_fields, old_rows = read_csv(archive_path)
    new_fields, new_rows = read_csv(current_path)
    fields = list(dict.fromkeys([*old_fields, *new_fields]))
    if fields:
        atomic_csv(current_path, fields, [*old_rows, *new_rows])


def write_run_shards(out: Path, levels: list[dict], seeds: list[int], calibration_seeds: list[int],
                     validation_seed: int, traces: dict, state: dict, gate: dict,
                     environment: dict, config: dict) -> None:
    request_fields, requests = read_csv(out / "requests.csv")
    instance_fields, instances = read_csv(out / "server_logs" / "on_instance_instance.csv")
    migration_fields, migrations = read_csv(out / "migration_events.csv")
    gpu_fields, gpu_rows = read_csv(out / "gpu_metrics.csv")
    for level in levels:
        for seed in seeds:
            label = level["label"]
            run_id = f"formal_on_{label}_seed{seed}"
            role = "calibration" if seed in calibration_seeds else "validation"
            qps_tag = BASE.qps_label(level["qps"])
            shard = out / role / f"qps_{qps_tag}_seed{seed}"
            (shard / "server_logs").mkdir(parents=True, exist_ok=True)
            (shard / "traces").mkdir(exist_ok=True)
            run_rows = [row for row in requests if row.get("run_id") == run_id]
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
            atomic_csv(shard / "server_logs" / "on_instance_instance.csv", instance_fields, run_instances)
            atomic_csv(shard / "migration_events.csv", migration_fields, run_migrations)
            atomic_csv(shard / "gpu_metrics.csv", gpu_fields, run_gpu)
            trace_source = out / "traces" / f"trace_{label}_seed{seed}.csv"
            shutil.copy2(trace_source, shard / "traces" / trace_source.name)
            shutil.copy2(out / "environment_audit.md", shard / "environment_audit.md")
            shutil.copy2(out / "configs" / "llumnix_on.yml", shard / "configs_llumnix_on.yml")
            gate_copy = {**gate, "reused_for_full_batch": True, "total_qps_seed_runs": len(levels) * len(seeds)}
            BASE.atomic_json(shard / "worker_startup_gate_on.json", gate_copy)
            trace_hash = sha256(shard / "traces" / trace_source.name)
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
                "configs": {"on": "configs_llumnix_on.yml"},
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


async def run_batch(qps_values: list[float], calibration_seeds: list[int], validation_seed: int,
                    max_num_seqs: int, warmup_sec: float, measure_sec: float, drain_sec: float,
                    output_dir: str, resume: bool = False) -> int:
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
    for directory in ("configs", "traces", "server_logs", "raw_requests", "plots", "calibration", "validation"):
        (out / directory).mkdir(exist_ok=True)

    seeds = [*calibration_seeds, validation_seed]
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
    config_path = out / "configs" / "llumnix_on.yml"
    mode_config = BASE.yaml_config("on", "balanced", migrate_out_threshold=THRESHOLD,
                                   pair_migration_frequency=FREQUENCY)
    mode_config["INSTANCE"]["ENABLE_DEFRAG"] = DEFRAG
    config_path.write_text(BASE.yaml.safe_dump(mode_config, sort_keys=False), encoding="utf-8")
    if not mode_config["MANAGER"].get("ENABLE_ROUTINE_MIGRATION"):
        raise RuntimeError("Migration ON configuration did not enable routine migration")

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
        "calibration_seeds": calibration_seeds, "validation_seed": validation_seed,
        "dataset_role": "batch", "policy": "balanced", "threshold": THRESHOLD,
        "frequency": FREQUENCY, "input_profiles": list(PROFILE_LENGTHS),
        "output_profiles": list(PROFILE_LENGTHS), "request_mode_count": 9,
        "measure_sec": measure_sec,
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
                    "routine_migration_enabled": True},
        "workload": {"arrival": "poisson", "input_profiles": list(PROFILE_LENGTHS),
                     "output_profiles": list(PROFILE_LENGTHS), "profile_mix": PROFILE_MIX,
                     "profile_cycle": True},
        "profile_mix": PROFILE_MIX, "warmup_sec": warmup_sec,
        "drain_sec": drain_sec, "request_timeout_sec": None,
        "arrival_model": "poisson", "reused_four_instance_server": True,
        "server_startups_for_entire_dataset": 1,
        "engine_latency_measurement": {
            "metric": "executor_iteration_wall_ms",
            "clock": "time.perf_counter",
            "scope": "from Ray executor dispatch through awaited worker result",
            "ray_spmd_and_non_spmd": True,
            "cuda_synchronize_added": False,
        },
        "configs": {"on": str(config_path.relative_to(out))},
        "output_dir": str(out), "status": "running",
    }
    archive_instance_log = None
    if resume:
        state = json.loads(resume_state_path.read_text(encoding="utf-8"))
        if state.get("experiment_id") != config["experiment_id"]:
            raise RuntimeError(f"Resume experiment id differs: {state.get('experiment_id')!r}")
        archive_instance_log = prepare_resume_artifacts(out, state)
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
            "passed": True, "mode": "on", "enforce_eager": True,
            "ready_instances_reported": max(ready_counts), "manager_instance_count": 4,
            "manager_instance_ids": sorted(registered_ids),
            "worker_gpu_assignments": worker_allocations,
            "readiness_monitor_stopped_before_workload": True,
            "verified_at_utc": BASE.now_iso(),
        }

    async def wait_ready_for_four_workers(process, log_path: Path, timeout_sec: float = 900.0) -> None:
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
                BASE.atomic_json(out / "worker_startup_gate_on.json", gate)
                print("[gate] all four workers ready on GPUs 0-3; the same group will serve all 20 windows", flush=True)
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
        current = json.loads((out / "worker_startup_gate_on.json").read_text(encoding="utf-8"))
        current["passed"] = True
        BASE.atomic_json(out / "worker_startup_gate_on.json", current)

    BASE.wait_ready = wait_ready_for_four_workers
    BASE.verify_effective_config = verify_effective_config
    gpu_rows, migration_events, capacity_rows = [], [], None
    try:
        try:
            await BASE.run_mode(
                "on", config_path, {"experiment_id": config["experiment_id"]}, state,
                out / "raw_requests", gpu_rows, migration_events, traces, levels, "balanced",
                expected_migration_enabled=True,
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
        gate = json.loads((out / "worker_startup_gate_on.json").read_text(encoding="utf-8"))
        write_run_shards(out, levels, seeds, calibration_seeds, validation_seed,
                         traces, state, gate, environment,
                         {**config, "measure_sec": measure_sec})
        all_sent = all(
            state.get("runs", {}).get(f"formal_on_{level['label']}_seed{seed}", {}).get("sent")
            == len(traces[(level["label"], seed)])
            for level in levels for seed in seeds
        )
        state["status"] = "complete" if all_sent else "partial"
        state["completed_at_utc"] = BASE.now_iso()
        config["status"] = state["status"]
        config["effective_config_verified"] = gate.get("passed") is True
        config["all_measurement_requests_sent"] = all_sent
        config["completed_formal_runs"] = state["completed_formal_runs"]
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_num_seqs < 1 or len(args.qps) != len(set(args.qps)):
        parser.error("--max-num-seqs must be positive and --qps values unique")
    if args.validation_seed in args.calibration_seeds or len(args.calibration_seeds) != len(set(args.calibration_seeds)):
        parser.error("validation seed must be distinct from unique calibration seeds")
    try:
        return asyncio.run(run_batch(args.qps, args.calibration_seeds, args.validation_seed,
                                     args.max_num_seqs, args.warmup_sec, args.measure_sec,
                                     args.drain_sec, args.output_dir, args.resume))
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
