#!/usr/bin/env python3
"""Paired QPS 15 1024-input/1024-output Llumnix OFF/ON experiment."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import math
import re
import traceback
from pathlib import Path

import pandas as pd

import run_baseline_4gpu as BASE


ROOT = Path(__file__).resolve().parents[1]
INPUT_TOKENS = 1024
OUTPUT_TOKENS = 1024
THRESHOLD = 9000.0
FREQUENCY = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value, digits: int = 2) -> str:
    try:
        number = float(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def write_report(out: Path, config: dict, summary: pd.DataFrame, state: dict,
                 plot_names: list[str]) -> None:
    aggregate = summary[summary["row_type"].eq("aggregate")]
    rows = []
    for _, row in aggregate.iterrows():
        rows.append(
            f"| {row['mode'].upper()} | {fmt(row.get('sent_requests_mean'), 0)} "
            f"| {fmt(row.get('successful_requests_mean'), 0)} "
            f"| {fmt(row.get('failed_requests_mean'), 0)} "
            f"| {fmt(row.get('actual_send_qps_mean'))} "
            f"| {fmt(row.get('in_window_completion_qps_mean'))} "
            f"| {fmt(row.get('ttft_p50_ms_mean'))} | {fmt(row.get('ttft_p95_ms_mean'))} "
            f"| {fmt(row.get('tpot_p50_ms_mean'))} | {fmt(row.get('tpot_p95_ms_mean'))} "
            f"| {fmt(row.get('e2e_p50_ms_mean'))} | {fmt(row.get('e2e_p95_ms_mean'))} |"
        )
    request_rows = pd.read_csv(out / "requests.csv")
    formal = request_rows[request_rows["phase"].eq("measurement")]
    per_mode = []
    for mode, group in formal.groupby("mode"):
        sent = int(group["actual_send_sec"].notna().sum())
        successful = int(group["success"].astype(str).str.lower().eq("true").sum())
        timed_out = int(group["timeout"].astype(str).str.lower().eq("true").sum())
        incomplete = int(group["completion_wall_time"].isna().sum())
        per_mode.append(
            f"| {mode.upper()} | {len(group)} | {sent} | {successful} | "
            f"{len(group) - successful} | {timed_out} | {incomplete} |"
        )
    report = f"""# QPS {config['qps']:g}、1024 输入 / 1024 输出：Migration OFF/ON，ENABLE_DEFRAG={config['runtime']['enable_defrag']}

## 结果

使用同一条 Poisson 到达轨迹配对比较 Migration OFF/ON。服务端逐 Token 时间线用于 TTFT/TPOT；客户端 E2E 从实际发送计至完整响应或传输错误。百分位只使用成功请求；完整请求状态见 `requests.csv`。

| 模式 | 发送请求 | 成功 | 失败 | 实际发送 QPS | 测量窗完成请求/s | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 请求完整性

| 模式 | 轨迹请求数 | 已发送 | 成功响应 | 失败/超时 | 排空上限触发超时 | 无完成时间 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(per_mode)}

- 本次状态：`{state.get('status')}`。目标 QPS {config['qps']}；实际轨迹在 {config['measure_sec']} 秒测量窗生成，seed `{config['seed']}`。两种模式的轨迹 SHA-256：`{config['trace_sha256']}`。
- 输入/输出目标长度均为 {INPUT_TOKENS} token。模型上下文长度为 {config['runtime']['max_model_len']}；请求使用 `ignore_eos`，返回的实际输出长度按请求记录。
- 启动配置：Qwen2-0.5B、4 个 Llumnix 实例（每卡一个，TP=1）、`MAX_NUM_SEQS={config['runtime']['max_num_seqs_per_instance']}`、`GPU_MEMORY_UTIL={config['runtime']['gpu_memory_utilization']}`，Balanced 配对策略，`ENABLE_DEFRAG={config['runtime']['enable_defrag']}`；ready 等待上限 {config['runtime']['instance_ready_timeout_sec']} 秒。
- 客户端每阶段 HTTP 连接池上限为 {config['client_connector_limit_per_phase']}，覆盖该阶段整条请求轨迹，避免连接池排队改变目标到达率。
- 预热 {config['warmup_sec']} 秒；正式测量 {config['measure_sec']} 秒；每个阶段请求无 HTTP 总时限。正式测量结束后最多排空 {config['drain_sec']} 秒；达到上限仍未结束的任务在 CSV 中标记，不纳入成功延迟分位数。
- OFF/ON 配置文件除迁移开关外相同；运行时同时核对四卡 Ray 资源、4 个实例 ready 日志及 Manager 实际登记的实例数。整组核验结果：`{config.get('effective_config_verified', False)}`；未通过的模式不会作为有效四实例测量纳入汇总。

## 资源、排队、KV cache 与迁移遥测

沿用 qps30 的采集流程：`gpu_metrics.csv` 逐秒记录四张 GPU 的利用率、显存、温度和功耗；Llumnix 实例日志汇总运行/等待请求数、KV cache 使用率及 GPU blocks；迁移日志记录可解析的迁移完成事件。`analyze_instance_metrics.py` 会生成相同口径的汇总 CSV 和图表。

## 图表

{chr(10).join(f'- `plots/{name}`' for name in plot_names) if plot_names else '- 无'}

## 文件

- `requests.csv`：预热与正式测量的逐请求时间戳、状态、E2E、TTFT/TPOT 和 token 数。
- `gpu_metrics.csv`：四卡 GPU 遥测；`instance_metrics.csv`：实例级队列和 KV cache 原始采样。
- `summary.csv`、`instance_metrics_summary.csv`、`gpu_metrics_summary.csv`、`migration_metrics_summary.csv`：汇总数据。
- `instance_gpu_metrics.csv`、`instance_gpu_summary.csv`：实例到 GPU 的可确认映射和统计。
- `migration_events.csv`、`migration_measurement_events.csv`：原始迁移日志与测量窗事件。
- `config.json`、`state.json`、`configs/`、`traces/`、`server_logs/`：实验配置、状态、共用轨迹和服务日志。

## 复现

```bash
cd {ROOT}
.venv/bin/python benchmark/run_qps15_1024x1024_off_on.py
```
"""
    (out / "report.md").write_text(report, encoding="utf-8")


async def run() -> int:
    qps, seed = 15.0, 101
    warmup_sec, measure_sec, drain_sec = 60.0, 180.0, 300.0
    BASE.OUT = ROOT / "benchmark" / "results" / (
        f"qps15_1024x1024_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    out = BASE.OUT
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("configs", "traces", "server_logs", "raw_requests", "plots"):
        (out / directory).mkdir()

    BASE.SEEDS = [seed]
    BASE.MAX_NUM_SEQS = 128
    BASE.GPU_MEMORY_UTIL = 0.5
    BASE.MODEL_MAX_LEN = 2048
    BASE.WARMUP_SEC = warmup_sec
    BASE.MEASURE_SEC = measure_sec
    BASE.DRAIN_SEC = drain_sec
    BASE.REQUEST_TIMEOUT_SEC = None
    BASE.MONITOR_INTERVAL_SEC = 1.0
    BASE.INPUT_PROFILE = [INPUT_TOKENS]
    BASE.OUTPUT_PROFILE = [OUTPUT_TOKENS]
    BASE.PROFILE_MIX = [{"name": "1024x1024", "input_tokens": INPUT_TOKENS,
                         "output_tokens": OUTPUT_TOKENS, "weight": 1.0}]

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

    qps_tag = BASE.qps_label(qps)
    levels = [{"label": f"qps_{qps_tag}", "qps": qps}]
    config_paths = {}
    for mode in ("off", "on"):
        path = out / "configs" / f"llumnix_{mode}.yml"
        path.write_text(BASE.yaml.safe_dump(
            BASE.yaml_config(mode, "balanced", migrate_out_threshold=THRESHOLD,
                             pair_migration_frequency=FREQUENCY), sort_keys=False), encoding="utf-8")
        config_paths[mode] = path
    BASE.check_mode_configs(config_paths)

    original_verify_effective_config = BASE.verify_effective_config

    def verify_exactly_four_instances(log_path, *verify_args, **verify_kwargs):
        startup_log = log_path.read_text(errors="replace")
        ready_counts = [int(value) for value in re.findall(
            r"Init Llumnix components done, (\d+) instances are ready", startup_log
        )]
        if not ready_counts or max(ready_counts) != 4:
            raise RuntimeError(
                "Expected four ready Llumnix instances before benchmarking; "
                f"startup log reported {ready_counts or 'no instance count'}"
            )
        manager_states = re.findall(
            r"num_instances:\s*(\d+),\s*instances:\s*\{([^}]*)\}", startup_log
        )
        if not manager_states:
            raise RuntimeError("Could not verify the Manager's registered instance count")
        registered_count = int(manager_states[-1][0])
        registered_ids = set(re.findall(r"[0-9a-f]{32}", manager_states[-1][1]))
        if registered_count != 4 or len(registered_ids) != 4:
            raise RuntimeError(
                "Expected four instances registered in the Llumnix Manager; "
                f"latest Manager state reports count={registered_count}, ids={len(registered_ids)}"
            )
        return original_verify_effective_config(log_path, *verify_args, **verify_kwargs)

    BASE.verify_effective_config = verify_exactly_four_instances

    formal_traces = BASE.create_traces(levels)
    trace = formal_traces[(levels[0]["label"], seed)]
    if not trace or any(req.input_tokens != INPUT_TOKENS or req.output_tokens != OUTPUT_TOKENS for req in trace):
        raise RuntimeError("Generated formal trace is empty or does not match 1024/1024 token lengths")
    trace_path = out / "traces" / f"trace_{levels[0]['label']}_seed{seed}.csv"
    trace_hash = sha256(trace_path)
    config = {
        "experiment_id": "qps15_1024x1024_off_on", "created_at_utc": BASE.now_iso(),
        "environment": environment, "model": {"path": str(BASE.MODEL), "max_model_len": BASE.MODEL_MAX_LEN,
                                                 "tensor_parallel_size": 1},
        "model_path": str(BASE.MODEL), "qps": qps, "seed": seed, "policy": "balanced",
        "threshold": THRESHOLD, "frequency": FREQUENCY, "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS, "measure_sec": measure_sec,
        "runtime": {"initial_instances": 4, "tensor_parallel_size": 1,
                    "gpu_memory_utilization": BASE.GPU_MEMORY_UTIL,
                    "max_model_len": BASE.MODEL_MAX_LEN,
                    "max_num_seqs_per_instance": BASE.MAX_NUM_SEQS, "dtype": "half",
                    "dispatch_policy": "load", "load_metric": "remaining_steps",
                    "pair_migration_policy": "balanced", "migration_backend": "rayrpc",
                    "enable_defrag": True,
                    "instance_ready_timeout_sec": 900.0,
                    "migrate_out_threshold": THRESHOLD, "pair_migration_frequency": FREQUENCY,
                    "routine_migration_enabled": {"off": False, "on": True}},
        "workload": {"arrival": "poisson", "input_tokens": INPUT_TOKENS,
                     "output_tokens": OUTPUT_TOKENS, "profile_mix": BASE.PROFILE_MIX},
        "profile_mix": BASE.PROFILE_MIX, "warmup_sec": warmup_sec,
        "measurement_request_count": len(trace), "drain_sec": drain_sec,
        "request_timeout_sec": None, "unbounded_request_completion": False,
        "arrival_model": "poisson", "trace_request_count": len(trace),
        "trace_sha256": trace_hash,
        "client_connector_limit_per_phase": min(max(512, len(trace)), 4096),
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
            raw_path = request_parts / f"formal_{mode}_{levels[0]['label']}_seed{seed}_measurement.csv"
            rows = pd.read_csv(raw_path)
            sent = int(rows["actual_send_sec"].notna().sum())
            successful = int(rows["success"].astype(str).str.lower().eq("true").sum())
            incomplete = int(rows["completion_wall_time"].isna().sum())
            entry = state["runs"][f"formal_{mode}_{levels[0]['label']}_seed{seed}"]
            entry.update({"trace_requests": len(trace), "sent": sent, "successful": successful,
                          "incomplete_at_drain_deadline": incomplete,
                          "status": "complete" if successful == len(trace) and incomplete == 0 else "partial"})
            state["completed_formal_runs"] = sum(
                1 for run in state["runs"].values() if run.get("status") == "complete"
            )
            BASE.atomic_json(out / "state.json", state)

        BASE.write_final_requests(request_parts, out / "requests.csv")
        migration_path = out / "migration_events.csv"
        if not migration_path.exists():
            migration_path.write_text(",".join(BASE.MIGRATION_FIELDS) + "\n", encoding="utf-8")
        summary = BASE.aggregate_summary(out / "requests.csv", out / "gpu_metrics.csv",
                                         migration_path, out / "summary.csv")
        plot_names = BASE.plot_results(summary, out / "plots")
        (out / "plots" / "migration.png").unlink(missing_ok=True)
        plot_names = [name for name in plot_names if name != "migration.png"]
        aggregate = summary[summary["row_type"].eq("aggregate")]
        if not aggregate.empty:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
            for ax, metric, title in zip(axes, ("ttft", "tpot", "e2e"), ("TTFT", "TPOT", "End-to-end latency")):
                for mode, color in (("off", "#2474B5"), ("on", "#E87500")):
                    row = aggregate[aggregate["mode"].eq(mode)]
                    if row.empty:
                        continue
                    values = [pd.to_numeric(row[f"{metric}_{p}_ms_mean"], errors="coerce").iloc[0]
                              for p in ("p50", "p95")]
                    ax.plot(("P50", "P95"), values, marker="o", color=color, label=mode.upper())
                ax.set_title(title)
                ax.set_ylabel("Milliseconds")
                ax.grid(True, alpha=0.25)
                ax.legend()
            fig.tight_layout()
            fig.savefig(out / "plots" / "latency_percentiles.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            plot_names.append("latency_percentiles.png")

        all_complete = state["completed_formal_runs"] == 2
        state["status"] = "complete" if all_complete else "partial"
        state["completed_at_utc"] = BASE.now_iso()
        config["status"] = state["status"]
        config["completed_formal_runs"] = state["completed_formal_runs"]
        config["effective_config_verified"] = True
        BASE.atomic_json(out / "state.json", state)
        BASE.atomic_json(out / "config.json", config)
        write_report(out, config, summary, state, plot_names)
        analysis_script = ROOT / "benchmark" / "analyze_instance_metrics.py"
        BASE.run_cmd([str(ROOT / ".venv" / "bin" / "python"), str(analysis_script), str(out)],
                     timeout=180.0, check=True)
        print(f"[done] status={state['status']} requests={len(trace)} per mode output={out}", flush=True)
        return 0 if all_complete else 2
    except BaseException as exc:
        state["status"] = "failed_or_interrupted"
        state["failure"] = f"{type(exc).__name__}: {exc}"
        state["failed_at_utc"] = BASE.now_iso()
        BASE.atomic_json(out / "state.json", state)
        raise


def main() -> int:
    try:
        return asyncio.run(run())
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
