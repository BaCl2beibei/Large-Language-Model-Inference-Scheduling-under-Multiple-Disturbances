#!/usr/bin/env python3
"""Run a paired three-mode QPS=5 Llumnix comparison on the local four-GPU host."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(row, name):
    value = row.get(name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def create_plot(summary: pd.DataFrame, out_dir: Path) -> str:
    aggregate = summary[summary["row_type"].eq("aggregate")].set_index("mode")
    modes = [("off", "OFF", "#2474B5"), ("on", "ON", "#E87500"),
             ("dryrun", "Dry-run", "#35936E")]
    plots = [("e2e_p95_ms", "P95 E2E (ms)"),
             ("ttft_mean_ms", "Mean TTFT (ms)"),
             ("tpot_mean_ms", "Mean TPOT (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (column, title) in zip(axes, plots):
        values, errors, labels, colors = [], [], [], []
        for mode, label, color in modes:
            if mode not in aggregate.index:
                continue
            row = aggregate.loc[mode]
            values.append(metric(row, f"{column}_mean"))
            errors.append(metric(row, f"{column}_std") or 0.0)
            labels.append(label)
            colors.append(color)
        ax.bar(labels, values, yerr=errors, color=colors, capsize=4)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Llumnix at 5 QPS — three seeds (mean ± sample SD)")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return "latency_comparison.png"


def write_report(out: Path, config: dict, summary: pd.DataFrame, state: dict) -> None:
    agg = summary[summary["row_type"].eq("aggregate")].set_index("mode")
    reps = summary[summary["row_type"].eq("replicate")]
    modes = [("off", "OFF"), ("on", "ON"), ("dryrun", "Dry-run")]
    rows = []
    for mode, label in modes:
        if mode not in agg.index:
            continue
        row = agg.loc[mode]
        cell = lambda field, digits=2: f"{fmt(metric(row, field + '_mean'), digits)} ± {fmt(metric(row, field + '_std'), digits)}"
        rows.append(
            f"| {label} | {cell('e2e_p95_ms')} | {cell('ttft_mean_ms')} | {cell('tpot_mean_ms')} "
            f"| {cell('request_throughput_rps')} | {fmt(metric(row, 'successful_requests_mean'), 1)}/"
            f"{fmt(metric(row, 'sent_requests_mean'), 1)} | {fmt(metric(row, 'migration_successes_mean'), 1)} |"
        )

    paired = []
    off = reps[reps["mode"].eq("off")].set_index("seed")
    for mode, label in (("on", "ON"), ("dryrun", "Dry-run")):
        treatment = reps[reps["mode"].eq(mode)].set_index("seed")
        deltas = []
        for seed in config["seeds"]:
            if seed in off.index and seed in treatment.index:
                off_p95 = metric(off.loc[seed], "e2e_p95_ms")
                mode_p95 = metric(treatment.loc[seed], "e2e_p95_ms")
                if off_p95 and mode_p95 is not None:
                    deltas.append((mode_p95 / off_p95 - 1) * 100)
        mean_delta = float(pd.Series(deltas).mean()) if deltas else None
        std_delta = float(pd.Series(deltas).std(ddof=1)) if len(deltas) > 1 else None
        paired.append(f"| {label} vs OFF | {fmt(mean_delta, 1)}% | {fmt(std_delta, 1)}% | {len(deltas)} |")

    suppression_path = out / "pairing_ablation_events.csv"
    if suppression_path.exists() and suppression_path.stat().st_size:
        suppression = pd.read_csv(suppression_path)
        discarded = int(pd.to_numeric(suppression.get("suppressed_pair_count"), errors="coerce").fillna(0).sum())
    else:
        discarded = 0
    event_path = out / "migration_events.csv"
    events = pd.read_csv(event_path) if event_path.exists() and event_path.stat().st_size else pd.DataFrame()
    on_events = events[events["mode"].astype(str).eq("on")] if not events.empty else events
    dry_events = events[events["mode"].astype(str).eq("dryrun")] if not events.empty else events
    on_success = int(on_events.get("success", pd.Series(dtype=str)).astype(str).str.lower().eq("true").sum())
    on_failures = len(on_events) - on_success
    dry_count = len(dry_events)
    trace_hashes = config["trace_sha256"]
    plot = create_plot(summary, out / "plots")
    report = f"""# QPS 5：Migration OFF / ON / Dry-run 三种模式对比

## 实验结果

每个数值为 3 个种子的均值 ± 样本标准差。TTFT/TPOT 使用服务端逐 Token 时延；样本数及 N/A 字段见 `summary.csv`。

| 模式 | P95 E2E (ms) | Mean TTFT (ms) | Mean TPOT (ms) | 完成请求吞吐 (req/s) | 成功/发送请求 | 成功迁移数 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

| 配对差值 | P95 E2E 相对 OFF 平均变化 | 跨种子标准差 | 配对种子数 |
|---|---:|---:|---:|
{chr(10).join(paired)}

## 实验设置与验证

- 状态：`{state.get('status')}`；QPS：{config['qps']}；种子：{', '.join(map(str, config['seeds']))}。
- 四 GPU、4 个独立 Llumnix 实例、TP=1、Balanced 策略、`rayrpc` 后端、`remaining_steps` 负载指标；没有负载/网络/故障注入。
- OFF 设置 `ENABLE_ROUTINE_MIGRATION=false`。ON 设置为 `true` 并执行正常配对和迁移。Dry-run 同样设置为 `true`，保留迁移过滤与 Balanced 配对计算，但在调度迁移任务前丢弃配对。
- ON 与 Dry-run 的启动日志分别验证了开关；Ray 启动状态确认 4 张 GPU。迁移事件：ON 共记录 {len(on_events)} 次（成功 {on_success}、失败 {on_failures}）；Dry-run 迁移事件 {dry_count} 次；dry-run 被丢弃候选配对合计 {discarded} 个。
- 每种模式使用同一 seed 的请求轨迹；运行器为每个 seed 只生成一条轨迹，并将同一轨迹对象依次交给 OFF、ON、Dry-run。各 seed 的轨迹 SHA-256：`{json.dumps(trace_hashes, sort_keys=True)}`。
- 每组预热 {config['warmup_sec']} 秒、测量 {config['measure_sec']} 秒，发送停止后最多排空 {config['drain_sec']} 秒。完成率和 SLO 指标见 CSV。
- 样本量只有 3 个种子，且测量窗口为快速实验设置；结果用于模式对比，不代表正式显著性检验。

## 图表

![QPS 5 三种模式时延对比](plots/{plot})

## 文件

- `requests.csv`：逐请求数据。
- `gpu_metrics.csv`：逐 GPU 资源采样。
- `migration_events.csv`、`pairing_ablation_events.csv`：迁移完成事件和 dry-run 丢弃配对日志。
- `summary.csv`：每种模式每个种子的指标及均值、样本标准差。
- `config.json`、`state.json`、`environment_audit.md`、`configs/`、`traces/`、`server_logs/`：配置、状态、环境、轨迹与启动日志。

## 复现

```bash
cd {ROOT}
.venv/bin/python benchmark/run_qps5_three_mode.py --seeds {' '.join(map(str, config['seeds']))} --qps {config['qps']} --warmup-sec {config['warmup_sec']} --measure-sec {config['measure_sec']} --drain-sec {config['drain_sec']} --policy {config['policy']}
```
"""
    (out / "report.md").write_text(report, encoding="utf-8")


async def run(args) -> int:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT / "benchmark" / "results" / f"qps5_three_mode_{timestamp}"
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
    (out / "environment_audit.md").write_text(BASE.environment_audit(environment) +
                                               "\n## 本次 GPU 拓扑\n\n```text\n" + topology + "```\n",
                                               encoding="utf-8")

    levels = [{"label": f"qps_{args.qps:g}".replace(".", "p"), "qps": args.qps}]
    configs = {}
    for mode in ("off", "on"):
        path = out / "configs" / f"llumnix_{mode}.yml"
        path.write_text(BASE.yaml.safe_dump(BASE.yaml_config(mode, args.policy), sort_keys=False), encoding="utf-8")
        configs[mode] = path
    BASE.check_mode_configs(configs)
    # Dry-run has the same Llumnix config as ON. Its single difference is the
    # experiment-only environment variable that discards selected pairs.
    (out / "configs" / "llumnix_dryrun.yml").write_text(configs["on"].read_text(), encoding="utf-8")

    formal_traces = BASE.create_traces(levels)
    trace_hashes = {}
    for seed in args.seeds:
        trace_path = out / "traces" / f"trace_{levels[0]['label']}_seed{seed}.csv"
        trace_hashes[str(seed)] = sha256(trace_path)

    config = {
        "experiment_id": "qps5_three_mode_ablation", "created_at_utc": BASE.now_iso(),
        "environment": environment, "model_path": str(BASE.MODEL), "qps": args.qps,
        "seeds": args.seeds, "policy": args.policy, "warmup_sec": args.warmup_sec,
        "measure_sec": args.measure_sec, "drain_sec": args.drain_sec,
        "arrival_model": "poisson", "profile_mix": BASE.PROFILE_MIX,
        "runtime": {"initial_instances": 4, "tensor_parallel_size": 1,
                    "gpu_memory_utilization": BASE.GPU_MEMORY_UTIL, "max_model_len": BASE.MODEL_MAX_LEN,
                    "max_num_seqs_per_instance": BASE.MAX_NUM_SEQS, "dtype": "half",
                    "dispatch_policy": "load", "dispatch_load_metric": "remaining_steps",
                    "migration_policy": args.policy, "migration_backend": "rayrpc",
                    "routine_migration_frequency": 1, "migrate_out_threshold": 20000.0,
                    "routine_migration_enabled": {"off": False, "on": True, "dryrun": True},
                    "dryrun": "run the configured filters and migration pairing policy, then discard all pairs before migration dispatch"},
        "trace_sha256": trace_hashes, "same_seed_trace_reused_across_modes": True,
        "configs": {"off": "configs/llumnix_off.yml",
                                                      "on": "configs/llumnix_on.yml",
                                                      "dryrun": "configs/llumnix_dryrun.yml"},
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
        # Each mode gets a fresh experiment-owned Ray cluster, but all three
        # reuse these exact per-seed Poisson traces.
        for mode in ("off", "on", "dryrun"):
            config_path = configs["off"] if mode == "off" else configs["on"]
            await BASE.run_mode(
                mode, config_path, context, state, request_parts, gpu_rows, migration_events,
                formal_traces, levels, args.policy, discard_migration_pairs=(mode == "dryrun"),
            )
        BASE.write_final_requests(request_parts, out / "requests.csv")
        if not (out / "migration_events.csv").exists():
            (out / "migration_events.csv").write_text(",".join(BASE.MIGRATION_FIELDS) + "\n", encoding="utf-8")
        if not (out / "pairing_ablation_events.csv").exists():
            (out / "pairing_ablation_events.csv").write_text(",".join(BASE.PAIR_SUPPRESSION_FIELDS) + "\n", encoding="utf-8")

        summary = BASE.aggregate_summary(out / "requests.csv", out / "gpu_metrics.csv",
                                         out / "migration_events.csv", out / "summary.csv")
        if summary.empty or summary[summary["row_type"].eq("replicate")].shape[0] != len(args.seeds) * 3:
            raise RuntimeError("Expected one completed summary row per mode and seed")
        if not summary[summary["row_type"].eq("replicate")]["mode"].isin(["off", "on", "dryrun"]).all():
            raise RuntimeError("Unexpected mode label in results")

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
