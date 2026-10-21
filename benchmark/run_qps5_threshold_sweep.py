#!/usr/bin/env python3
"""Compare four Balanced migration thresholds and OFF at QPS 5 on four GPUs."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import run_baseline_4gpu as BASE


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [2000.0, 5000.0, 8000.0, 20000.0]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def as_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def fmt(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def create_plots(summary: pd.DataFrame, out_dir: Path, labels: dict[str, str]) -> list[str]:
    agg = summary[summary["row_type"].eq("aggregate")].set_index("mode")
    modes = [(mode, label) for mode, label in labels.items() if mode in agg.index]
    colors = ["#2474B5", "#E87500", "#52A36B", "#8C64B0", "#C44E52"]
    metrics = [("e2e_p95_ms", "P95 E2E (ms)"),
               ("ttft_mean_ms", "Mean TTFT (ms)"),
               ("tpot_mean_ms", "Mean TPOT (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (column, title) in zip(axes, metrics):
        values = [as_float(agg.loc[mode].get(f"{column}_mean")) for mode, _ in modes]
        errors = [as_float(agg.loc[mode].get(f"{column}_std")) or 0.0 for mode, _ in modes]
        ax.bar([label for _, label in modes], values, yerr=errors,
               color=colors[:len(modes)], capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("QPS 5 — migration threshold sweep (mean ± sample SD, 3 seeds)")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar([label for _, label in modes],
                [as_float(agg.loc[mode].get("migration_successes_mean")) or 0.0 for mode, _ in modes],
                color=colors[:len(modes)])
    axes[0].set_title("Successful migrations per run")
    axes[0].set_ylabel("Mean count across seeds")
    axes[1].bar([label for _, label in modes],
                [as_float(agg.loc[mode].get("migration_mean_cost_ms_mean"))
                 if as_float(agg.loc[mode].get("migration_mean_cost_ms_mean")) is not None else math.nan
                 for mode, _ in modes],
                color=colors[:len(modes)])
    axes[1].set_title("Mean logged migration cost")
    axes[1].set_ylabel("Milliseconds")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "migration_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ["latency_comparison.png", "migration_comparison.png"]


def write_report(out: Path, config: dict, summary: pd.DataFrame, state: dict) -> None:
    labels = config["mode_labels"]
    aggregate = summary[summary["row_type"].eq("aggregate")].set_index("mode")
    replicates = summary[summary["row_type"].eq("replicate")]
    off = replicates[replicates["mode"].eq("off")].set_index("seed")
    rows, delta_rows = [], []
    for mode, label in labels.items():
        if mode not in aggregate.index:
            continue
        row = aggregate.loc[mode]
        cell = lambda field, digits=2: (
            f"{fmt(as_float(row.get(field + '_mean')), digits)} ± "
            f"{fmt(as_float(row.get(field + '_std')), digits)}"
        )
        rows.append(
            f"| {label} | {cell('e2e_p95_ms')} | {cell('ttft_mean_ms')} | {cell('tpot_mean_ms')} "
            f"| {cell('request_throughput_rps')} | {cell('migration_successes', 1)} | "
            f"{cell('migration_mean_cost_ms', 1)} |"
        )
        if mode == "off":
            continue
        treatment = replicates[replicates["mode"].eq(mode)].set_index("seed")
        deltas = []
        for seed in config["seeds"]:
            if seed in off.index and seed in treatment.index:
                base_p95 = as_float(off.loc[seed].get("e2e_p95_ms"))
                candidate_p95 = as_float(treatment.loc[seed].get("e2e_p95_ms"))
                if base_p95 and candidate_p95 is not None:
                    deltas.append((candidate_p95 / base_p95 - 1.0) * 100.0)
        delta_mean = float(pd.Series(deltas).mean()) if deltas else None
        delta_std = float(pd.Series(deltas).std(ddof=1)) if len(deltas) > 1 else None
        delta_rows.append(f"| {label} vs OFF | {fmt(delta_mean, 1)}% | {fmt(delta_std, 1)}% | {len(deltas)} |")

    events_path = out / "migration_events.csv"
    events = pd.read_csv(events_path) if events_path.exists() and events_path.stat().st_size else pd.DataFrame()
    event_rows = []
    if not events.empty:
        for mode, label in labels.items():
            subset = events[events["mode"].astype(str).eq(mode)]
            measurement = subset[subset["phase"].astype(str).eq("measurement")]
            successes = int(measurement["success"].astype(str).str.lower().eq("true").sum())
            failures = len(measurement) - successes
            event_rows.append(f"| {label} | {len(measurement)} | {successes} | {failures} |")
    event_table = "\n".join(event_rows) if event_rows else "| — | 0 | 0 | 0 |"

    plot_names = create_plots(summary, out / "plots", labels)
    plots_md = "\n".join(f"- `plots/{name}`" for name in plot_names)
    report = f"""# QPS 5 Migration Threshold Sweep

## Comparison

Values are mean ± sample standard deviation across seeds 101, 202, and 303. TTFT and TPOT use the service's per-token timing data.

| Mode | P95 E2E (ms) | Mean TTFT (ms) | Mean TPOT (ms) | Completed throughput (req/s) | Successful migrations/run (all phases) | Mean migration cost (ms, all phases) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

| Paired P95 difference vs OFF | Mean change | Across-seed SD | Seed pairs |
|---|---:|---:|---:|
{chr(10).join(delta_rows)}

## Migration events during measurement windows

Warmup and drain events are recorded in `migration_events.csv`; this table counts only events attributed to the 20-second measurement windows.

| Mode | Events | Successful | Failed |
|---|---:|---:|---:|
{event_table}

## Method and validation

- Status: `{state.get('status')}`; target QPS {config['qps']}; seeds {', '.join(map(str, config['seeds']))}; 4 independent Llumnix instances, TP=1.
- All migration runs use Balanced, `rayrpc`, `remaining_steps`, the same dispatch policy and model. The only ON-mode setting varied is `MIGRATE_OUT_THRESHOLD`; OFF disables routine migration.
- Each seed has one Poisson request trace reused unchanged by all five modes. Trace SHA-256 by seed: `{json.dumps(config['trace_sha256'], sort_keys=True)}`.
- Startup logs verify the migration enable flag, threshold value, Balanced policy, and Ray's four-GPU resources for every mode. No CPU/GPU load, network, or fault injection was used.
- Warmup {config['warmup_sec']}s, measurement {config['measure_sec']}s, drain deadline {config['drain_sec']}s. Request success, timing sample counts, and GPU metrics are in the CSV files.
- Migration count and average migration cost in the comparison table include all observed phases (warmup, measurement, drain). The separate event table isolates the measurement windows.
- This is a short, three-seed comparison; it is exploratory and does not establish statistical significance.

## Plots

{plots_md}

## Files

- `requests.csv`, `gpu_metrics.csv`, `migration_events.csv`, `summary.csv`
- `config.json`, `state.json`, `environment_audit.md`, `configs/`, `traces/`, `server_logs/`

## Reproduce

```bash
cd {ROOT}
.venv/bin/python benchmark/run_qps5_threshold_sweep.py --seeds {' '.join(map(str, config['seeds']))} --qps {config['qps']} --warmup-sec {config['warmup_sec']} --measure-sec {config['measure_sec']} --drain-sec {config['drain_sec']} --policy {config['policy']}
```
"""
    (out / "report.md").write_text(report, encoding="utf-8")


async def run(args) -> int:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT / "benchmark" / "results" / f"qps5_threshold_sweep_{timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("configs", "traces", "server_logs", "raw_requests", "plots"):
        (out / directory).mkdir()

    BASE.OUT = out
    BASE.SEEDS = args.seeds
    BASE.WARMUP_SEC = args.warmup_sec
    BASE.MEASURE_SEC = args.measure_sec
    BASE.DRAIN_SEC = args.drain_sec
    BASE.MONITOR_INTERVAL_SEC = 1.0
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

    levels = [{"label": f"qps_{args.qps:g}".replace(".", "p"), "qps": args.qps}]
    modes = ["off"] + [f"threshold_{int(value)}" for value in THRESHOLDS]
    labels = {"off": "OFF"}
    for threshold in THRESHOLDS:
        labels[f"threshold_{int(threshold)}"] = f"Threshold {int(threshold)}"

    configs = {}
    off_path = out / "configs" / "llumnix_off.yml"
    off_path.write_text(BASE.yaml.safe_dump(BASE.yaml_config("off", args.policy), sort_keys=False),
                        encoding="utf-8")
    configs["off"] = off_path
    for threshold in THRESHOLDS:
        mode = f"threshold_{int(threshold)}"
        path = out / "configs" / f"llumnix_{mode}.yml"
        path.write_text(BASE.yaml.safe_dump(
            BASE.yaml_config("on", args.policy, migrate_out_threshold=threshold), sort_keys=False),
            encoding="utf-8",
        )
        configs[mode] = path
    BASE.check_mode_configs({"off": configs["off"], "on": configs["threshold_20000"]})
    reference_on = BASE.yaml.safe_load(configs["threshold_20000"].read_text())
    for threshold in THRESHOLDS:
        mode = f"threshold_{int(threshold)}"
        cfg = BASE.yaml.safe_load(configs[mode].read_text())
        if cfg["MANAGER"]["MIGRATE_OUT_THRESHOLD"] != threshold:
            raise RuntimeError(f"Threshold config mismatch for {mode}")
        for key in cfg["MANAGER"]:
            if key not in {"MIGRATE_OUT_THRESHOLD"} and cfg["MANAGER"][key] != reference_on["MANAGER"][key]:
                raise RuntimeError(f"Non-threshold manager setting differs for {mode}: {key}")

    formal_traces = BASE.create_traces(levels)
    trace_hashes = {
        str(seed): digest(out / "traces" / f"trace_{levels[0]['label']}_seed{seed}.csv")
        for seed in args.seeds
    }
    config = {
        "experiment_id": "qps5_migration_threshold_sweep", "created_at_utc": BASE.now_iso(),
        "environment": environment, "model_path": str(BASE.MODEL), "qps": args.qps,
        "seeds": args.seeds, "policy": args.policy, "thresholds": THRESHOLDS,
        "mode_labels": labels, "warmup_sec": args.warmup_sec, "measure_sec": args.measure_sec,
        "drain_sec": args.drain_sec, "arrival_model": "poisson", "profile_mix": BASE.PROFILE_MIX,
        "runtime": {"initial_instances": 4, "tensor_parallel_size": 1,
                    "gpu_memory_utilization": BASE.GPU_MEMORY_UTIL,
                    "max_model_len": BASE.MODEL_MAX_LEN, "max_num_seqs_per_instance": BASE.MAX_NUM_SEQS,
                    "dtype": "half", "dispatch_policy": "load",
                    "dispatch_load_metric": "remaining_steps", "migration_policy": args.policy,
                    "migration_backend": "rayrpc", "routine_migration_frequency": 1,
                    "off_threshold": 20000.0, "migration_enabled_threshold_modes": True},
        "trace_sha256": trace_hashes,
        "configs": {mode: str(path.relative_to(out)) for mode, path in configs.items()},
        "output_dir": str(out), "status": "running",
    }
    state = {"experiment_id": config["experiment_id"], "status": "running",
             "started_at_utc": BASE.now_iso(), "runs": {}, "completed_formal_runs": 0}
    BASE.atomic_json(out / "config.json", config)
    BASE.atomic_json(out / "state.json", state)

    request_parts = out / "raw_requests"
    gpu_rows, migration_events = [], []
    context = {"experiment_id": config["experiment_id"]}
    try:
        for mode in modes:
            print(f"[mode] {labels[mode]}", flush=True)
            await BASE.run_mode(
                mode, configs[mode], context, state, request_parts, gpu_rows, migration_events,
                formal_traces, levels, args.policy,
                expected_migration_enabled=(mode != "off"),
                expected_migrate_out_threshold=(20000.0 if mode == "off" else float(mode.split("_")[-1])),
            )
        BASE.write_final_requests(request_parts, out / "requests.csv")
        if not (out / "migration_events.csv").exists():
            (out / "migration_events.csv").write_text(",".join(BASE.MIGRATION_FIELDS) + "\n", encoding="utf-8")
        summary = BASE.aggregate_summary(out / "requests.csv", out / "gpu_metrics.csv",
                                         out / "migration_events.csv", out / "summary.csv")
        expected_rows = len(args.seeds) * len(modes)
        actual_rows = summary[summary["row_type"].eq("replicate")].shape[0]
        if summary.empty or actual_rows != expected_rows:
            raise RuntimeError(f"Expected {expected_rows} summary replicates, got {actual_rows}")

        state["status"] = "complete"
        state["completed_at_utc"] = BASE.now_iso()
        state["completed_formal_runs"] = sum(1 for row in state["runs"].values()
                                               if row.get("status") == "complete")
        config["status"] = "complete"
        config["completed_formal_runs"] = state["completed_formal_runs"]
        config["effective_config_verified"] = True
        BASE.atomic_json(out / "state.json", state)
        BASE.atomic_json(out / "config.json", config)
        write_report(out, config, summary, state)
        print(f"[done] status=complete runs={state['completed_formal_runs']} output={out}", flush=True)
        return 0
    except BaseException as exc:
        state["status"] = "failed_or_interrupted"
        state["failure"] = f"{type(exc).__name__}: {exc}"
        state["failed_at_utc"] = BASE.now_iso()
        BASE.atomic_json(out / "state.json", state)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--qps", type=float, default=5.0)
    parser.add_argument("--warmup-sec", type=float, default=5.0)
    parser.add_argument("--measure-sec", type=float, default=20.0)
    parser.add_argument("--drain-sec", type=float, default=180.0)
    parser.add_argument("--policy", choices=("balanced", "defrag"), default="balanced")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.qps <= 0 or args.warmup_sec < 0 or args.measure_sec <= 0 or args.drain_sec <= 0:
        raise SystemExit("QPS must be positive; warmup must be nonnegative; measurement and drain must be positive")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise SystemExit("Seeds must be a nonempty list of unique integers")
    try:
        return asyncio.run(run(args))
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
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
