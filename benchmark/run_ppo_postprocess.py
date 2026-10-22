#!/usr/bin/env python3
"""After real trace collection, fit calibration and run the requested PPO training."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", required=True)
    parser.add_argument("--collection-pid", type=int)
    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args()
    collection = Path(args.collection_dir).resolve()
    manifest_path = collection / "collection.json"
    status_path = collection / "postprocess_status.json"
    training_dir = collection / "mig_ppo_training"
    calibration_path = collection / "mig_ppo_calibration.yaml"
    write_json(status_path, {"status": "waiting_for_collection", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})

    while True:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Collection manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        collection_state = manifest.get("status")
        if collection_state == "complete":
            break
        if collection_state != "running":
            raise RuntimeError(f"Collection stopped with status {collection_state!r}; refusing to train on incomplete inputs")
        if args.collection_pid is not None:
            try:
                os.kill(args.collection_pid, 0)
            except ProcessLookupError as exc:
                raise RuntimeError("Collection process exited before writing a complete manifest") from exc
        time.sleep(30)

    runs = manifest.get("runs", [])
    expected = len(manifest.get("qps_levels", [])) * (
        len(manifest.get("calibration_seeds", [])) + 1
    )
    if len(runs) != expected or any(row.get("status") != "complete" for row in runs):
        raise RuntimeError(f"Collection manifest has {len(runs)} runs; expected {expected} complete runs")

    inputs = manifest["combined_inputs"]
    instance_csv = ROOT / inputs["calibration"]["instance_csv"]
    migration_csv = ROOT / inputs["calibration"]["migration_csv"]
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(status_path, {"status": "calibrating", "collection_runs": len(runs), "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    run([
        str(PYTHON), "-m", "rl_scheduler.calibrate",
        "--instance-csv", str(instance_csv), "--migration-csv", str(migration_csv),
        "--output", str(calibration_path),
    ])

    from rl_scheduler.calibrate import fit_migration_profile, fit_service_profiles

    validation = inputs["validation"]
    val_instance = ROOT / validation["instance_csv"]
    val_migration = ROOT / validation["migration_csv"]
    calibration = __import__("yaml").safe_load(calibration_path.read_text(encoding="utf-8"))
    calibrated = calibration["environment"]
    validation_report: dict = {
        "held_out_seed": validation["seed"],
        "excluded_from_calibration": True,
        "calibration_migration_mode": "on",
        "validation_service_fit": fit_service_profiles(val_instance),
    }
    val_service = validation_report["validation_service_fit"]
    validation_report["service_parameter_difference"] = {
        key: {"calibration": calibrated[key], "validation": val_service[key],
              "absolute_difference": abs(float(calibrated[key]) - float(val_service[key]))}
        for key in ("prefill_ms_per_token", "prefill_intercept_ms", "decode_ms_per_request",
                    "decode_ms_per_context_token", "decode_intercept_ms")
    }
    try:
        validation_report["validation_migration_fit"] = fit_migration_profile(val_migration)
        validation_report["migration_validation_status"] = "fit"
    except (ValueError, KeyError) as exc:
        validation_report["migration_validation_status"] = "insufficient_terminal_events"
        validation_report["migration_validation_error"] = str(exc)
    validation_report_path = collection / "validation_report.json"
    write_json(validation_report_path, validation_report)

    training_dir.mkdir(parents=True, exist_ok=True)
    write_json(status_path, {"status": "training", "episodes": args.episodes, "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    run([
        str(PYTHON), "-m", "rl_scheduler.train_ppo",
        "--config", str(ROOT / "rl_scheduler" / "configs" / "train.yaml"),
        "--calibration", str(calibration_path), "--episodes", str(args.episodes),
        "--output-dir", str(training_dir),
    ])
    best_checkpoint = training_dir / "mig_ppo_best.pt"
    run([
        str(PYTHON), "-m", "rl_scheduler.evaluate",
        "--checkpoint", str(best_checkpoint),
        "--scenarios", "qps_9", "qps_12", "qps_15", "qps_18", "qps_21",
        "--seeds", str(validation["seed"]), "--device", "cuda:0",
        "--output", str(collection / "simulation_evaluation_seed404.json"),
    ])
    write_json(status_path, {
        "status": "complete", "episodes": args.episodes,
        "calibration_yaml": str(calibration_path.relative_to(ROOT)),
        "validation_report": str(validation_report_path.relative_to(ROOT)),
        "training_dir": str(training_dir.relative_to(ROOT)),
        "simulation_evaluation": str((collection / "simulation_evaluation_seed404.json").relative_to(ROOT)),
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    print(f"[done] postprocessing and {args.episodes} PPO episodes completed", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
