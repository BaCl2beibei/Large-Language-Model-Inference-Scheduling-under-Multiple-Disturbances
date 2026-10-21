#!/usr/bin/env python3
"""Paired fixed-count 512-input/512-output OFF/ON Llumnix run."""

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

import pandas as pd

import run_baseline_4gpu as BASE


ROOT = Path(__file__).resolve().parents[1]
INPUT_TOKENS = 512
OUTPUT_TOKENS = 512
THRESHOLD = 9000.0
FREQUENCY = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value, digits=2):
    try:
        number = float(value)
        if not math.isfinite(number):
            return "N/A"
        return f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def write_report(out: Path, config: dict, summary: pd.DataFrame, state: dict,
                 plot_names: list[str]) -> None:
    aggregate = summary[summary["row_type"].eq("aggregate")]
    rows = []
    for _, row in aggregate.iterrows():
        rows.append(
            f"| {str(row['mode']).upper()} | {fmt(row.get('sent_requests_mean'), 1)} "
            f"| {fmt(row.get('successful_requests_mean'), 1)} | {fmt(row.get('failed_requests_mean'), 1)} "
            f"| {fmt(row.get('actual_send_qps_mean'), 2)} | {fmt(row.get('in_window_completion_qps_mean'), 2)} "
            f"| {fmt(row.get('measurement_output_tokens_per_sec_mean'), 2)} "
            f"| {fmt(row.get('ttft_p50_ms_mean'))} | {fmt(row.get('ttft_p95_ms_mean'))} "
            f"| {fmt(row.get('tpot_p50_ms_mean'))} | {fmt(row.get('tpot_p95_ms_mean'))} "
            f"| {fmt(row.get('e2e_p50_ms_mean'))} | {fmt(row.get('e2e_p95_ms_mean'))} "
            f"| {fmt(row.get('ttft_samples_mean'), 0)}/{fmt(row.get('tpot_samples_mean'), 0)} |"
        )
    plots = "\n".join(f"- `plots/{name}`" for name in plot_names)
    qps_tag = BASE.qps_label(float(config["qps"]))
    trace_label = f"qps_{qps_tag}"
    traces = pd.read_csv(out / "traces" / f"trace_{trace_label}_seed{config['seed']}.csv")
    input_lengths = sorted(pd.to_numeric(traces["input_tokens"], errors="coerce").dropna().unique().tolist())
    output_lengths = sorted(pd.to_numeric(traces["output_tokens"], errors="coerce").dropna().unique().tolist())
    requests = pd.read_csv(out / "requests.csv")
    measurement = requests[requests["phase"].eq("measurement")]
    request_lengths = {
        mode: {
            "expected_input": sorted(pd.to_numeric(group["expected_input_tokens"], errors="coerce").dropna().unique().tolist()),
            "expected_output": sorted(pd.to_numeric(group["expected_output_tokens"], errors="coerce").dropna().unique().tolist()),
            "server_input": sorted(pd.to_numeric(group["server_input_tokens"], errors="coerce").dropna().unique().tolist()),
            "server_output": sorted(pd.to_numeric(group["server_output_tokens"], errors="coerce").dropna().unique().tolist()),
        }
        for mode, group in measurement.groupby("mode")
    }
    events = pd.read_csv(out / "migration_events.csv") if (out / "migration_events.csv").stat().st_size else pd.DataFrame()
    if not events.empty and "observed_wall_time" in events:
        events["observed_wall_time"] = pd.to_numeric(events["observed_wall_time"], errors="coerce")
        request_rows = requests[requests["phase"].eq("measurement")].dropna(
            subset=["actual_send_wall_time", "actual_send_sec"]
        )
        bounds = {}
        for mode, group in request_rows.groupby("mode"):
            start = (group["actual_send_wall_time"] - group["actual_send_sec"]).median()
            bounds[mode] = (start, start + float(config["measure_sec"]))
        events = events[
            events.apply(lambda row: row["mode"] in bounds and
                         bounds[row["mode"]][0] <= row["observed_wall_time"] <= bounds[row["mode"]][1], axis=1)
        ]
    elif not events.empty:
        events = events.iloc[0:0]
    successes = int(events["success"].astype(str).str.lower().eq("true").sum()) if not events.empty else 0
    failures = len(events) - successes
    migration_costs = pd.to_numeric(events.loc[events["success"].astype(str).str.lower().eq("true"), "cost_ms"], errors="coerce").dropna() if not events.empty else pd.Series(dtype=float)
    plot_text = plots or "- 无图表"
    report = f"""# QPS {config['qps']:g}、512 输入 / 512 输出：Migration OFF/ON

## 结果

单种子 `{config['seed']}`；延迟使用服务端逐 Token 时间线计算，无法取得的值保留为 N/A。`summary.csv` 保留各重复层级的汇总列，逐请求数据在 `requests.csv`。

| 模式 | 发送请求 | 成功请求 | 失败/超时 | 实际发送 QPS | 测量窗完成请求/s | 测量窗输出 token/s | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) | TTFT/TPOT 样本数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 配置与校验

- 状态：`{state.get('status')}`；目标 QPS {config['qps']}；正式请求 {config['measurement_request_count']} 条/模式；测量 {config['measure_sec']} 秒；随机种子 {config['seed']}。
- 每条请求的正式轨迹输入长度 `{input_lengths}`、输出长度 `{output_lengths}`；两种模式复用同一条轨迹，SHA-256 `{config['trace_sha256']}`。
- 服务器返回的输入/输出 token 长度按模式记录在报告内嵌数据中：`{json.dumps(request_lengths, ensure_ascii=False, sort_keys=True)}`。请求使用 `ignore_eos`，目标输出长度为 512；实际返回长度以 CSV 记录为准。
- 服务端返回的 `server_input_tokens` 字段缺失，记为 N/A；输入文本在本地 trace 生成时按 tokenizer 截到 512 tokens。实际输出 token 数由响应记录。
- 迁移 ON 配置：阈值 `{config['threshold']}`、`PAIR_MIGRATION_FREQUENCY={config['frequency']}`、Balanced、`rayrpc`、负载指标 `remaining_steps`。OFF/ON 的启动日志确认迁移开关和参数；Ray 状态确认四卡。
- 测量阶段记录迁移事件 {len(events)} 次（成功 {successes}、非成功 {failures}）；成功迁移平均日志耗时 `{fmt(migration_costs.mean()) if len(migration_costs) else 'N/A'}` ms，耗时样本 {len(migration_costs)}。这些事件来自迁移完成日志，不代表未产生日志的所有潜在尝试。
- 请求长度使 Offered output load 达到约 QPS × 512 输出 token/s；是否达到稳态取决于实测完成速率，见 CSV 和 GPU 采样。
- 每个正式请求的客户端 E2E 从实际发送计至完整响应或传输错误返回；TTFT/TPOT 使用服务端逐 Token 时间线。延迟统计覆盖正式轨迹中的全部请求，包括测量窗结束后完成的请求。
- TTFT、TPOT、E2E 百分位以正式测量阶段发送且最终成功的请求计算；HTTP/传输错误单独记录。客户端没有总请求、连接或读取超时，停止发送后无期限等待所有请求任务返回再汇总。
- 预热 {config['warmup_sec']} 秒，预热请求单独记录、不计入每种模式的 {config['measurement_request_count']} 条正式请求；停止正式发送后持续排空，直到所有正式请求任务结束。

## 图表

{plot_text}

## 文件

- `requests.csv`：逐请求数据，包括期望与服务端 token 长度、TTFT/TPOT、E2E、成功/超时状态。
- `gpu_metrics.csv`：每 GPU 利用率和显存采样。
- `migration_events.csv`：可解析的迁移事件。
- `summary.csv`：OFF/ON 运行指标及统计汇总。
- `config.json`、`state.json`、`configs/`、`traces/`、`server_logs/`：配置、状态、共用请求轨迹及日志。

## 复现

```bash
cd {ROOT}
.venv/bin/python benchmark/run_qps5_512x512_off_on.py --seed {config['seed']} --qps {config['qps']} --request-count {config['measurement_request_count']} --measure-sec {config['measure_sec']} --warmup-sec {config['warmup_sec']}
```
"""
    (out / "report.md").write_text(report, encoding="utf-8")


async def run(args) -> int:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    qps_tag = BASE.qps_label(args.qps)
    run_label = f"qps{qps_tag}_512x512"
    out = ROOT / "benchmark" / "results" / f"{run_label}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("configs", "traces", "server_logs", "raw_requests", "plots"):
        (out / directory).mkdir()

    BASE.OUT = out
    BASE.SEEDS = [args.seed]
    BASE.MAX_NUM_SEQS = 64
    BASE.WARMUP_SEC = args.warmup_sec
    BASE.MEASURE_SEC = args.measure_sec
    BASE.DRAIN_SEC = None
    BASE.REQUEST_TIMEOUT_SEC = None
    BASE.MONITOR_INTERVAL_SEC = 1.0
    BASE.PROFILE_MIX = [{"name": "512x512", "input_tokens": INPUT_TOKENS,
                         "output_tokens": OUTPUT_TOKENS, "weight": 1.0}]
    BASE.INPUT_PROFILE = [INPUT_TOKENS]
    BASE.OUTPUT_PROFILE = [OUTPUT_TOKENS]

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

    levels = [{"label": f"qps_{qps_tag}", "qps": args.qps}]
    config_paths = {}
    for mode in ("off", "on"):
        path = out / "configs" / f"llumnix_{mode}.yml"
        path.write_text(BASE.yaml.safe_dump(
            BASE.yaml_config(mode, "balanced", migrate_out_threshold=THRESHOLD,
                             pair_migration_frequency=FREQUENCY), sort_keys=False), encoding="utf-8")
        config_paths[mode] = path
    BASE.check_mode_configs(config_paths)

    formal_traces = BASE.create_traces(levels, max_requests=args.request_count)
    trace = formal_traces[(levels[0]["label"], args.seed)]
    if len(trace) != args.request_count or any(
            req.input_tokens != INPUT_TOKENS or req.output_tokens != OUTPUT_TOKENS for req in trace):
        raise RuntimeError(
            f"Generated formal trace must contain exactly {args.request_count} 512-input/512-output requests"
        )
    trace_path = out / "traces" / f"trace_{levels[0]['label']}_seed{args.seed}.csv"
    trace_hash = sha256(trace_path)
    config = {
        "experiment_id": f"{run_label}_off_on", "created_at_utc": BASE.now_iso(),
        "environment": environment, "model_path": str(BASE.MODEL), "qps": args.qps,
        "seed": args.seed, "policy": "balanced", "threshold": THRESHOLD, "frequency": FREQUENCY,
        "input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
        "measurement_request_count": args.request_count,
        "profile_mix": BASE.PROFILE_MIX,
        "runtime": {"initial_instances": 4, "tensor_parallel_size": 1,
                    "gpu_memory_utilization": BASE.GPU_MEMORY_UTIL,
                    "max_model_len": BASE.MODEL_MAX_LEN, "max_num_seqs_per_instance": BASE.MAX_NUM_SEQS,
                    "dtype": "half", "dispatch_policy": "load", "load_metric": "remaining_steps",
                    "migration_backend": "rayrpc", "migrate_out_threshold": THRESHOLD,
                    "pair_migration_frequency": FREQUENCY,
                    "routine_migration_enabled": {"off": False, "on": True}},
        "warmup_sec": args.warmup_sec, "measure_sec": args.measure_sec,
        "drain_sec": None, "request_timeout_sec": None,
        "unbounded_request_completion": True,
        "arrival_model": "poisson", "trace_request_count": len(trace), "trace_sha256": trace_hash,
        "configs": {mode: str(path.relative_to(out)) for mode, path in config_paths.items()},
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
        for mode in ("off", "on"):
            await BASE.run_mode(
                mode, config_paths[mode], context, state, request_parts, gpu_rows, migration_events,
                formal_traces, levels, "balanced",
                expected_migration_enabled=(mode == "on"),
                expected_migrate_out_threshold=THRESHOLD,
                expected_pair_migration_frequency=FREQUENCY,
            )
            raw_path = request_parts / f"formal_{mode}_qps_{qps_tag}_seed{args.seed}_measurement.csv"
            completed_rows = pd.read_csv(raw_path)
            if len(completed_rows) != args.request_count:
                raise RuntimeError(f"{mode} recorded {len(completed_rows)} formal requests, expected {args.request_count}")
            if completed_rows["completion_wall_time"].isna().any():
                raise RuntimeError(f"{mode} has formal requests without a recorded completion/error time")
            if completed_rows["timeout"].astype(str).str.lower().eq("true").any():
                raise RuntimeError(f"{mode} unexpectedly timed out a formal request")
        BASE.write_final_requests(request_parts, out / "requests.csv")
        if not (out / "migration_events.csv").exists():
            (out / "migration_events.csv").write_text(",".join(BASE.MIGRATION_FIELDS) + "\n", encoding="utf-8")
        summary = BASE.aggregate_summary(out / "requests.csv", out / "gpu_metrics.csv",
                                         out / "migration_events.csv", out / "summary.csv")
        expected_rows = 2
        actual_rows = summary[summary["row_type"].eq("replicate")].shape[0] if not summary.empty else 0
        if actual_rows != expected_rows:
            raise RuntimeError(f"Expected {expected_rows} summary rows, got {actual_rows}")
        request_rows = pd.read_csv(out / "requests.csv")
        measurement = request_rows[request_rows["phase"].eq("measurement")]
        for mode, rows in measurement.groupby("mode"):
            input_lengths = set(pd.to_numeric(rows["expected_input_tokens"], errors="coerce").dropna())
            output_lengths = set(pd.to_numeric(rows["expected_output_tokens"], errors="coerce").dropna())
            if input_lengths != {INPUT_TOKENS} or output_lengths != {OUTPUT_TOKENS}:
                raise RuntimeError(f"Request lengths changed in {mode}: input={input_lengths}, output={output_lengths}")

        plot_names = [name for name in BASE.plot_results(summary, out / "plots") if name != "migration.png"]
        percentile_plot = out / "plots" / "latency_percentiles.png"
        aggregate = summary[summary["row_type"].eq("aggregate")]
        if not aggregate.empty:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
            for ax, metric, title in zip(
                axes,
                ("ttft", "tpot", "e2e"),
                ("TTFT", "TPOT", "End-to-end latency"),
            ):
                for mode, color in (("off", "#2474B5"), ("on", "#E87500")):
                    row = aggregate[aggregate["mode"].eq(mode)]
                    if row.empty:
                        continue
                    values = [
                        pd.to_numeric(row[f"{metric}_{percentile}_ms_mean"], errors="coerce").iloc[0]
                        for percentile in ("p50", "p95")
                    ]
                    ax.plot(("P50", "P95"), values, marker="o", color=color, label=mode.upper())
                ax.set_title(title)
                ax.set_ylabel("Milliseconds")
                ax.grid(True, alpha=0.25)
                ax.legend()
            fig.tight_layout()
            fig.savefig(percentile_plot, dpi=160, bbox_inches="tight")
            plt.close(fig)
            plot_names.append(percentile_plot.name)
        state["status"] = "complete"
        state["completed_at_utc"] = BASE.now_iso()
        state["completed_formal_runs"] = sum(1 for row in state["runs"].values()
                                               if row.get("status") == "complete")
        config["status"] = "complete"
        config["completed_formal_runs"] = state["completed_formal_runs"]
        config["effective_config_verified"] = True
        BASE.atomic_json(out / "state.json", state)
        BASE.atomic_json(out / "config.json", config)
        write_report(out, config, summary, state, plot_names)
        analysis_script = ROOT / "benchmark" / "analyze_instance_metrics.py"
        BASE.run_cmd([sys.executable, str(analysis_script), str(out)], timeout=120.0, check=True)
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
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--qps", type=float, default=30.0)
    parser.add_argument("--request-count", type=int, default=500)
    parser.add_argument("--warmup-sec", type=float, default=30.0)
    parser.add_argument("--measure-sec", type=float, default=120.0)
    parser.add_argument("--drain-sec", type=float, default=None,
                        help="Optional drain limit; omitted means wait for every request")
    parser.add_argument("--request-timeout-sec", type=float, default=None,
                        help="Optional total HTTP timeout; omitted means no request timeout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.qps <= 0 or args.warmup_sec < 0 or args.measure_sec <= 0 or args.request_count <= 0 \
            or (args.drain_sec is not None and args.drain_sec <= 0) \
            or (args.request_timeout_sec is not None and args.request_timeout_sec <= 0):
        raise SystemExit("QPS, measurement, and request count must be positive; warmup must be nonnegative")
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
