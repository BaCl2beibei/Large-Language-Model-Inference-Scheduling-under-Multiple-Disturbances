#!/usr/bin/env python3
"""Run one Migration ON Llumnix sample with nine mixed prompt/output profiles."""

from __future__ import annotations

import asyncio
import argparse
import datetime as dt
import hashlib
import json
import math
import re
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
    profile_table = "\n".join(
        f"| {entry['name']} | {entry['input_tokens']} | {entry['output_tokens']} | 1/9 |"
        for entry in config["profile_mix"]
    )
    report = f"""# QPS {config['qps']:g}、九种请求长度混合：Migration ON，ENABLE_DEFRAG={config['runtime']['enable_defrag']}

## 结果

本实验仅运行 Migration ON。请求以等比例循环混合九种 prompt/output 长度。服务端逐 Token 时间线用于 TTFT/TPOT；客户端 E2E 从实际发送计至完整响应或传输错误。百分位只使用成功请求；完整请求状态见 `requests.csv`。

| 请求模式 | 输入 token | 输出 token | 权重 |
|---|---:|---:|---:|
{profile_table}

| 模式 | 发送请求 | 成功 | 失败 | 实际发送 QPS | 测量窗完成请求/s | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 请求完整性

| 模式 | 轨迹请求数 | 已发送 | 成功响应 | 失败/超时 | 排空上限触发超时 | 无完成时间 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(per_mode)}

- 本次状态：`{state.get('status')}`。目标 QPS {config['qps']}；实际轨迹在 {config['measure_sec']} 秒测量窗生成，seed `{config['seed']}`。轨迹 SHA-256：`{config['trace_sha256']}`。
- 九种模式的输入/输出长度档位均为 {PROFILE_LENGTHS}，最长组合为 1048+1048 token。模型上下文长度为 {config['runtime']['max_model_len']}；请求使用 `ignore_eos`，返回的实际输出长度按请求记录。
- 启动配置：Qwen2-0.5B、4 个 Llumnix 实例（每卡一个，TP=1）、每实例 `MAX_NUM_SEQS={config['runtime']['max_num_seqs_per_instance']}`（合计上限 {config['runtime']['aggregate_max_num_seqs']}）、`GPU_MEMORY_UTIL={config['runtime']['gpu_memory_utilization']}`、`enforce_eager=True`（不捕获 CUDA graph），Balanced 配对策略，`ENABLE_DEFRAG={config['runtime']['enable_defrag']}`。
- 客户端每阶段 HTTP 连接池上限为 {config['client_connector_limit_per_phase']}，覆盖该阶段整条请求轨迹，避免连接池排队改变目标到达率。
- 预热 {config['warmup_sec']} 秒；正式测量 {config['measure_sec']} 秒；每个阶段请求无 HTTP 总时限。正式测量结束后最多排空 {config['drain_sec']} 秒；达到上限仍未结束的任务在 CSV 中标记，不纳入成功延迟分位数。
- 正式请求启动前必须确认四个实例 ready、Manager 登记四个实例，且四个 Llumlet worker 分别分配到 GPU 0–3。整组核验结果：`{config.get('effective_config_verified', False)}`；未通过门禁不会开始测量。

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
.venv/bin/python benchmark/run_qps15_1024x1024_no_defrag.py --max-num-seqs {config['runtime']['max_num_seqs_per_instance']}
```
"""
    (out / "report.md").write_text(report, encoding="utf-8")


async def run(max_num_seqs: int = 64, qps: float = 15.0, seed: int = 101,
              warmup_sec: float = 30.0, measure_sec: float = 200.0,
              drain_sec: float = 15.0, dataset_role: str = "calibration",
              output_dir: str | None = None) -> int:
    if qps <= 0 or seed < 0:
        raise ValueError("qps must be positive and seed must be non-negative")
    if min(warmup_sec, measure_sec, drain_sec) < 0 or measure_sec <= 0:
        raise ValueError("measurement duration must be positive; other durations cannot be negative")
    if dataset_role not in {"calibration", "validation"}:
        raise ValueError("dataset_role must be calibration or validation")
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')
    BASE.OUT = Path(output_dir).resolve() if output_dir else ROOT / "benchmark" / "results" / (
        f"ppo_calibration_qps{BASE.qps_label(qps)}_seed{seed}_{dataset_role}_{stamp}"
    )
    out = BASE.OUT
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("configs", "traces", "server_logs", "raw_requests", "plots"):
        (out / directory).mkdir()

    BASE.SEEDS = [int(seed)]
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

    qps_tag = BASE.qps_label(qps)
    levels = [{"label": f"qps_{qps_tag}", "qps": qps}]
    config_paths = {}
    for mode in ("on",):
        path = out / "configs" / f"llumnix_{mode}.yml"
        mode_config = BASE.yaml_config(mode, "balanced", migrate_out_threshold=THRESHOLD,
                                       pair_migration_frequency=FREQUENCY)
        mode_config["INSTANCE"]["ENABLE_DEFRAG"] = DEFRAG
        path.write_text(BASE.yaml.safe_dump(mode_config, sort_keys=False), encoding="utf-8")
        config_paths[mode] = path
        if not mode_config["MANAGER"].get("ENABLE_ROUTINE_MIGRATION"):
            raise RuntimeError("Migration ON configuration did not enable routine migration")

    original_verify_effective_config = BASE.verify_effective_config
    original_wait_ready = BASE.wait_ready

    def startup_gate_snapshot(log_path: Path) -> dict | None:
        startup_log = log_path.read_text(errors="replace")
        ready_counts = [int(value) for value in re.findall(
            r"Init Llumnix components done, (\d+) instances are ready", startup_log
        )]
        manager_states = re.findall(
            r"num_instances:\s*(\d+),\s*instances:\s*\{([^}]*)\}", startup_log
        )
        worker_allocations = {
            worker_id: gpu_id for worker_id, gpu_id in re.findall(
                r"Llumlet\(actor_name=instance_([0-9a-f]{32}).*?gpu_ids=\['(\d+)'\]",
                startup_log,
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
            "passed": True,
            "mode": "on",
            "enforce_eager": True,
            "ready_instances_reported": max(ready_counts),
            "manager_instance_count": 4,
            "manager_instance_ids": sorted(registered_ids),
            "worker_gpu_assignments": worker_allocations,
            "ray_gpu_resources": 4,
            "readiness_monitor_stopped_before_workload": True,
            "verified_at_utc": BASE.now_iso(),
        }

    async def wait_ready_for_four_workers(process, log_path: Path,
                                          timeout_sec: float = 900.0) -> None:
        await original_wait_ready(process, log_path, timeout_sec=timeout_sec)
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = log_path.read_text(errors="replace")[-10000:] if log_path.exists() else ""
                raise RuntimeError(
                    f"Llumnix exited before all four workers passed startup gate; "
                    f"code={process.returncode}; log tail:\n{tail}"
                )
            gate = startup_gate_snapshot(log_path)
            if gate is not None:
                ray_status = await BASE.verify_ray_four_gpu()
                gate["ray_gpu_status"] = ray_status
                gate["ray_gpu_resources"] = 4
                BASE.atomic_json(out / "worker_startup_gate_on.json", gate)
                print("[gate] four ready workers registered, one on each GPU; workload may start", flush=True)
                return
            await asyncio.sleep(2)
        tail = log_path.read_text(errors="replace")[-10000:] if log_path.exists() else ""
        raise TimeoutError(f"Four-worker startup gate deadline; log tail:\n{tail}")

    BASE.wait_ready = wait_ready_for_four_workers

    def verify_exactly_four_instances(log_path, *verify_args, **verify_kwargs):
        gate = startup_gate_snapshot(log_path)
        if gate is None:
            raise RuntimeError("The four-worker startup gate was not passed before testing")
        if "enforce_eager=True" not in log_path.read_text(errors="replace"):
            raise RuntimeError("Effective vLLM configuration did not confirm enforce_eager=True")
        original_verify_effective_config(log_path, *verify_args, **verify_kwargs)
        BASE.atomic_json(out / "worker_startup_gate_on.json", gate)

    BASE.verify_effective_config = verify_exactly_four_instances

    formal_traces = BASE.create_traces(levels)
    trace = formal_traces[(levels[0]["label"], seed)]
    observed_profiles = {(req.input_tokens, req.output_tokens) for req in trace}
    if not trace or observed_profiles != set(REQUEST_PROFILES):
        raise RuntimeError(
            "Generated formal trace must include all nine configured prompt/output length pairs; "
            f"observed={sorted(observed_profiles)}"
        )
    trace_path = out / "traces" / f"trace_{levels[0]['label']}_seed{seed}.csv"
    trace_hash = sha256(trace_path)
    config = {
        "experiment_id": f"ppo_calibration_mixed_qps{qps_tag}_seed{seed}_{dataset_role}_seq{max_num_seqs}",
        "created_at_utc": BASE.now_iso(),
        "environment": environment, "model": {"path": str(BASE.MODEL), "max_model_len": BASE.MODEL_MAX_LEN,
                                                 "tensor_parallel_size": 1},
        "model_path": str(BASE.MODEL), "qps": qps, "seed": seed,
        "dataset_role": dataset_role, "policy": "balanced",
        "threshold": THRESHOLD, "frequency": FREQUENCY,
        "input_profiles": list(PROFILE_LENGTHS), "output_profiles": list(PROFILE_LENGTHS),
        "request_mode_count": len(PROFILE_MIX), "measure_sec": measure_sec,
        "runtime": {"initial_instances": 4, "tensor_parallel_size": 1,
                    "gpu_memory_utilization": BASE.GPU_MEMORY_UTIL,
                    "max_model_len": BASE.MODEL_MAX_LEN,
                    "max_num_seqs_per_instance": BASE.MAX_NUM_SEQS,
                    "aggregate_max_num_seqs": BASE.MAX_NUM_SEQS * 4, "dtype": "half",
                    "enforce_eager": True,
                    "dispatch_policy": "load", "load_metric": "remaining_steps",
                    "pair_migration_policy": "balanced", "migration_backend": "rayrpc",
                    "enable_defrag": DEFRAG,
                    "migrate_out_threshold": THRESHOLD, "pair_migration_frequency": FREQUENCY,
                    "routine_migration_enabled": True},
        "workload": {"arrival": "poisson", "input_profiles": list(PROFILE_LENGTHS),
                     "output_profiles": list(PROFILE_LENGTHS), "profile_mix": BASE.PROFILE_MIX,
                     "profile_cycle": True},
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
        for mode in ("on",):
            run_id = f"formal_{mode}_{levels[0]['label']}_seed{seed}"
            try:
                await BASE.run_mode(
                    mode, config_paths[mode], context, state, request_parts, gpu_rows, migration_events,
                    formal_traces, levels, "balanced",
                    expected_migration_enabled=True,
                    expected_migrate_out_threshold=THRESHOLD,
                    expected_pair_migration_frequency=FREQUENCY,
                )
            except Exception as exc:
                state.setdefault("runs", {})[run_id] = {
                    "status": "aborted_invalid_topology",
                    "mode": mode,
                    "qps": qps,
                    "seed": seed,
                    "formal_measurement_started": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                state["startup_failure"] = state["runs"][run_id]["reason"]
                state["status"] = "partial"
                BASE.atomic_json(out / "state.json", state)
                print(f"[mode-aborted] {mode.upper()}: {state['startup_failure']}", flush=True)
                break
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

        all_measurements_recorded = (
            state.get("runs", {}).get(f"formal_on_{levels[0]['label']}_seed{seed}", {}).get("sent", -1)
            == len(trace)
        )
        state["status"] = "complete" if all_measurements_recorded else "partial"
        state["completed_at_utc"] = BASE.now_iso()
        config["status"] = state["status"]
        config["completed_formal_runs"] = state["completed_formal_runs"]
        config["all_measurement_requests_sent"] = all_measurements_recorded
        config["effective_config_verified"] = (
            (out / "worker_startup_gate_on.json").exists()
            and state.get("runs", {}).get(f"formal_on_{levels[0]['label']}_seed{seed}", {}).get("status")
            in {"complete", "partial"}
        )
        BASE.atomic_json(out / "state.json", state)
        BASE.atomic_json(out / "config.json", config)
        write_report(out, config, summary, state, plot_names)
        analysis_script = ROOT / "benchmark" / "analyze_instance_metrics.py"
        if pd.read_csv(out / "requests.csv")["phase"].eq("measurement").any():
            BASE.run_cmd([str(ROOT / ".venv" / "bin" / "python"), str(analysis_script), str(out)],
                         timeout=180.0, check=True)
        print(f"[done] status={state['status']} requests={len(trace)} per mode output={out}", flush=True)
        return 0 if all_measurements_recorded else 2
    except BaseException as exc:
        state["status"] = "failed_or_interrupted"
        state["failure"] = f"{type(exc).__name__}: {exc}"
        state["failed_at_utc"] = BASE.now_iso()
        BASE.atomic_json(out / "state.json", state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-num-seqs", type=int, default=64,
                        help="maximum concurrent sequences per Llumnix instance")
    parser.add_argument("--qps", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--dataset-role", choices=("calibration", "validation"), default="calibration")
    parser.add_argument("--warmup-sec", type=float, default=30.0)
    parser.add_argument("--measure-sec", type=float, default=200.0)
    parser.add_argument("--drain-sec", type=float, default=15.0)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be positive")
    try:
        return asyncio.run(run(args.max_num_seqs, args.qps, args.seed, args.warmup_sec,
                               args.measure_sec, args.drain_sec, args.dataset_role,
                               args.output_dir))
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
