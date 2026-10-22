#!/usr/bin/env python3
"""Collect eager-only Migration ON traces with nine prompt/output length modes."""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH_RUNNER = ROOT / "benchmark" / "run_ppo_batch_reuse.py"
DEFAULT_QPS = [9.0, 12.0, 15.0, 18.0, 21.0]
DEFAULT_CALIBRATION_SEEDS = [101, 202, 303]
DEFAULT_VALIDATION_SEED = 404
PROFILE_LENGTHS = [256, 512, 1048]
REQUEST_PROFILE_MIX = [
    {"name": f"input{prompt}_output{output}", "input_tokens": prompt,
     "output_tokens": output, "weight": 1.0}
    for prompt in PROFILE_LENGTHS for output in PROFILE_LENGTHS
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_name(qps: float, seed: int) -> str:
    return f"qps_{qps:g}_seed{seed}"


def positive_latency_samples(path: Path) -> dict[str, int]:
    counts = {"prefill": 0, "decode": 0}
    if not path.exists():
        return counts
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for row in csv.DictReader(stream):
            try:
                item = ast.literal_eval(row.get("profiling_data", ""))
                if not isinstance(item, (tuple, list)) or len(item) < 4:
                    continue
                kind = str(item[0]).strip(" ' \"").lower()
                latency_ms = float(item[3])
            except (ValueError, SyntaxError, TypeError):
                continue
            if kind in counts and latency_ms > 0 and latency_ms < float("inf"):
                counts[kind] += 1
    return counts


def validate_run(path: Path, qps: float, seed: int, role: str) -> dict | None:
    config_path = path / "config.json"
    state_path = path / "state.json"
    if not config_path.exists() or not state_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (float(config.get("qps", -1)) != float(qps)
            or int(config.get("seed", -1)) != int(seed)
            or config.get("dataset_role") != role
            or config.get("request_mode_count") != 9
            or config.get("runtime", {}).get("enforce_eager") is not True
            or config.get("workload", {}).get("profile_cycle") is not True
            or config.get("profile_mix") != REQUEST_PROFILE_MIX
            or not config.get("effective_config_verified")
            or not config.get("all_measurement_requests_sent")):
        return None
    gates = {}
    mode = "on"
    gate_path = path / f"worker_startup_gate_{mode}.json"
    if not gate_path.exists():
        return None
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (not gate.get("passed") or gate.get("enforce_eager") is not True
            or gate.get("ready_instances_reported") != 4
            or gate.get("manager_instance_count") != 4
            or len(gate.get("worker_gpu_assignments", {})) != 4
            or set(gate.get("worker_gpu_assignments", {}).values()) != {"0", "1", "2", "3"}):
        return None
    gates[mode] = gate
    mode_runs = state.get("runs", {})
    row = mode_runs.get(f"formal_on_qps_{qps:g}_seed{seed}", {})
    if row.get("sent") != row.get("trace_requests") or row.get("trace_requests", 0) <= 0:
        return None
    latency_counts = positive_latency_samples(
        path / "server_logs" / "on_instance_instance.csv"
    )
    if latency_counts["prefill"] < 2 or latency_counts["decode"] < 3:
        return None
    return {
        "status": "complete",
        "qps": float(qps),
        "seed": int(seed),
        "dataset_role": role,
        "experiment_dir": str(path.relative_to(ROOT)),
        "state_status": state.get("status"),
        "all_measurement_requests_sent": True,
        "startup_gate_passed": True,
        "startup_gates": gates,
        "request_completions_may_be_censored": True,
        "positive_engine_latency_samples": latency_counts,
        "recorded_at_utc": now_iso(),
    }


def combine_csv(files: list[Path], output: Path, *, predicate=None) -> int:
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for path in files:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open(newline="", encoding="utf-8", errors="replace") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in fields:
                        fields.append(name)
            rows.extend(row for row in reader if predicate is None or predicate(row))
    if not fields:
        raise RuntimeError(f"No CSV headers found for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_combined_inputs(root: Path, runs: list[dict], calibration_seeds: list[int],
                          validation_seed: int) -> dict:
    def selected(role: str):
        return [ROOT / row["experiment_dir"] for row in runs if row["dataset_role"] == role]

    calibration_dirs = selected("calibration")
    validation_dirs = selected("validation")
    cal_service = combine_csv(
        [path / "server_logs" / "on_instance_instance.csv" for path in calibration_dirs],
        root / "calibration_inputs" / "service_instance.csv",
    )
    cal_migration = combine_csv(
        [path / "migration_events.csv" for path in calibration_dirs],
        root / "calibration_inputs" / "migration_events.csv",
        predicate=lambda row: row.get("mode") == "on" and row.get("status", "").strip() != "",
    )
    val_service = combine_csv(
        [path / "server_logs" / "on_instance_instance.csv" for path in validation_dirs],
        root / "validation_inputs" / "service_instance.csv",
    )
    val_migration = combine_csv(
        [path / "migration_events.csv" for path in validation_dirs],
        root / "validation_inputs" / "migration_events.csv",
        predicate=lambda row: row.get("mode") == "on" and row.get("status", "").strip() != "",
    )
    return {
        "calibration": {
            "seeds": calibration_seeds,
            "instance_csv": str((root / "calibration_inputs" / "service_instance.csv").relative_to(ROOT)),
            "migration_csv": str((root / "calibration_inputs" / "migration_events.csv").relative_to(ROOT)),
            "instance_rows": cal_service,
            "migration_terminal_events": cal_migration,
        },
        "validation": {
            "seed": validation_seed,
            "instance_csv": str((root / "validation_inputs" / "service_instance.csv").relative_to(ROOT)),
            "migration_csv": str((root / "validation_inputs" / "migration_events.csv").relative_to(ROOT)),
            "instance_rows": val_service,
            "migration_terminal_events": val_migration,
            "excluded_from_calibration": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", type=float, nargs="+", default=DEFAULT_QPS)
    parser.add_argument("--calibration-seeds", type=int, nargs="+", default=DEFAULT_CALIBRATION_SEEDS)
    parser.add_argument("--validation-seed", type=int, default=DEFAULT_VALIDATION_SEED)
    parser.add_argument("--warmup-sec", type=float, default=30.0)
    parser.add_argument("--measure-sec", type=float, default=200.0)
    parser.add_argument("--drain-sec", type=float, default=15.0)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.qps or any(qps <= 0 for qps in args.qps) or len(args.qps) != len(set(args.qps)):
        parser.error("--qps must contain unique positive values")
    if (not args.calibration_seeds or len(args.calibration_seeds) != len(set(args.calibration_seeds))
            or args.validation_seed in args.calibration_seeds):
        parser.error("Use unique calibration seeds and a distinct validation seed")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir).resolve() if args.output_dir else (
        ROOT / "benchmark" / "results" / f"ppo_calibration_mixed_2112_eager_{stamp}"
    )
    manifest_path = root / "collection.json"
    specification = {
        "experiment_id": "mig_ppo_real_calibration_9mode_eager",
        "qps_levels": [float(qps) for qps in args.qps],
        "calibration_seeds": [int(seed) for seed in args.calibration_seeds],
        "validation_seed": int(args.validation_seed),
        "warmup_sec": float(args.warmup_sec),
        "measurement_sec": float(args.measure_sec),
        "drain_sec": float(args.drain_sec),
        "input_profiles": PROFILE_LENGTHS,
        "output_profiles": PROFILE_LENGTHS,
        "request_mode_count": 9,
        "profile_mix": REQUEST_PROFILE_MIX,
        "profile_cycle": True,
        "max_model_len": 2112,
        "enforce_eager": True,
        "max_num_seqs_per_instance": int(args.max_num_seqs),
        "model": "Qwen/Qwen2-0.5B",
        "instance_count": 4,
        "tensor_parallel_size": 1,
        "migration_modes": ["on"],
        "reuse_one_four_instance_server_for_full_batch": True,
        "server_side_abort_barrier_between_workload_phases": True,
        "engine_latency_measurement": {
            "metric": "executor_iteration_wall_ms",
            "clock": "time.perf_counter",
            "scope": "from Ray executor dispatch through awaited worker result",
            "ray_spmd_and_non_spmd": True,
            "cuda_synchronize_added": False,
        },
        "readiness_gate": "before measurement: 4 ready Llumnix workers, 4 Manager registrations, Ray GPU resources 4/4, one worker per GPU",
    }
    if root.exists():
        if not args.resume or not manifest_path.exists():
            raise FileExistsError(f"Output directory exists: {root}; pass --resume to continue it")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in specification.items()):
            raise RuntimeError("Resume specification differs from collection.json")
    else:
        root.mkdir(parents=True)
        manifest = {**specification, "status": "running", "started_at_utc": now_iso(), "runs": []}
        atomic_json(manifest_path, manifest)

    expected_runs = [
        (qps, seed, "calibration")
        for qps in args.qps for seed in args.calibration_seeds
    ] + [
        (qps, args.validation_seed, "validation") for qps in args.qps
    ]
    completed_by_key = {
        (float(row["qps"]), int(row["seed"]), row["dataset_role"]): row
        for row in manifest.get("runs", [])
    }
    pending_runs = []
    for qps, seed, role in expected_runs:
        rel_dir = Path("calibration" if role == "calibration" else "validation") / run_name(qps, seed)
        run_dir = root / rel_dir
        previous = validate_run(run_dir, qps, seed, role)
        if previous:
            completed_by_key[(float(qps), int(seed), role)] = previous
            continue
        pending_runs.append((qps, seed, role))

    if pending_runs:
        if any((root / ("calibration" if role == "calibration" else "validation")
                / run_name(qps, seed)).exists() for qps, seed, role in pending_runs):
            raise RuntimeError("Incomplete per-run shards exist; use a new output directory for a clean batch")
        command = [
            sys.executable, str(BATCH_RUNNER),
            "--qps", *[str(qps) for qps in args.qps],
            "--calibration-seeds", *[str(seed) for seed in args.calibration_seeds],
            "--validation-seed", str(args.validation_seed),
            "--warmup-sec", str(args.warmup_sec), "--measure-sec", str(args.measure_sec),
            "--drain-sec", str(args.drain_sec), "--max-num-seqs", str(args.max_num_seqs),
            "--output-dir", str(root),
        ]
        if args.resume:
            command.append("--resume")
            if manifest.get("last_failure"):
                manifest.setdefault("failure_history", []).append(manifest["last_failure"])
            manifest.pop("last_failure", None)
            manifest["status"] = "running"
            manifest["resumed_at_utc"] = now_iso()
            atomic_json(manifest_path, manifest)
        print(f"[run] starting one eager four-instance server for {len(pending_runs)} pending windows "
              f"({len(expected_runs)} total)", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        missing_or_invalid = []
        for qps, seed, role in expected_runs:
            rel_dir = Path("calibration" if role == "calibration" else "validation") / run_name(qps, seed)
            run_dir = root / rel_dir
            valid = validate_run(run_dir, qps, seed, role)
            if valid is None:
                missing_or_invalid.append((qps, seed, role, run_dir))
            else:
                completed_by_key[(float(qps), int(seed), role)] = valid
        if result.returncode != 0 or missing_or_invalid:
            record = {
                "status": "failed", "qps": None, "seed": None, "dataset_role": "batch",
                "experiment_dir": str(root.relative_to(ROOT)), "return_code": result.returncode,
                "missing_or_invalid_runs": [
                    {"qps": float(qps), "seed": int(seed), "dataset_role": role,
                     "experiment_dir": str(path.relative_to(ROOT))}
                    for qps, seed, role, path in missing_or_invalid
                ], "recorded_at_utc": now_iso(),
            }
            manifest["runs"] = [completed_by_key[key] for key in completed_by_key]
            manifest["status"] = "blocked_or_failed"
            manifest["last_failure"] = record
            atomic_json(manifest_path, manifest)
            raise RuntimeError(
                f"Persistent batch collection failed; runner exit={result.returncode}, "
                f"invalid windows={len(missing_or_invalid)}. Inspect {root} before retrying."
            )
        manifest["runs"] = [completed_by_key[(float(q), int(s), r)]
                             for q, s, r in expected_runs]
        manifest["last_completed_run"] = manifest["runs"][-1]
        atomic_json(manifest_path, manifest)
    else:
        manifest["runs"] = [completed_by_key[(float(q), int(s), r)]
                             for q, s, r in expected_runs]
        print(f"[skip] all {len(expected_runs)} windows already collected", flush=True)

    manifest["runs"] = [completed_by_key[(float(q), int(s), r)] for q, s, r in expected_runs]
    manifest["combined_inputs"] = build_combined_inputs(root, manifest["runs"],
                                                           args.calibration_seeds,
                                                           args.validation_seed)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = now_iso()
    atomic_json(manifest_path, manifest)
    print(f"[done] collected {len(expected_runs)} QPS/seed pairs at {root}", flush=True)
    print(f"[calibration] {manifest['combined_inputs']['calibration']}", flush=True)
    print(f"[validation] {manifest['combined_inputs']['validation']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
