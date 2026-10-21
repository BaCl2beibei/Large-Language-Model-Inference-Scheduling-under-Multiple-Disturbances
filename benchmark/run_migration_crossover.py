#!/usr/bin/env python3
"""Run a short paired OFF/ON QPS sweep to locate a latency crossover.

Uses the existing real Llumnix serving/measurement path. One seeded Poisson
trace is created per QPS and reused byte-for-byte in both migration modes.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "benchmark" / "run_baseline_4gpu.py"
SPEC = importlib.util.spec_from_file_location("baseline_4gpu_helper", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import benchmark helper at {HELPER_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qps_levels(values: list[float]) -> list[dict]:
    result = []
    for qps in values:
        label = f"qps_{qps:g}".replace(".", "p")
        result.append({"label": label, "qps": qps})
    return result


def add_idle_control_rows(summary, gpu_path: Path, migration_path: Path, seed: int):
    """Add QPS=0 rows with N/A request latency and measured idle GPU metrics."""
    pd = BASE.pd
    gpu = pd.read_csv(gpu_path) if gpu_path.exists() and gpu_path.stat().st_size else pd.DataFrame()
    migrations = pd.read_csv(migration_path) if migration_path.exists() and migration_path.stat().st_size else pd.DataFrame()
    empty_requests = pd.DataFrame(columns=BASE.REQUEST_FIELDS)
    empty_migrations = pd.DataFrame(columns=BASE.MIGRATION_FIELDS)
    rows = summary[summary["row_type"].eq("replicate")].to_dict("records") if not summary.empty else []
    for mode in ("off", "on"):
        run_id = f"formal_{mode}_qps_0_seed{seed}"
        gpu_group = gpu[gpu["run_id"].astype(str).eq(run_id)] if not gpu.empty else pd.DataFrame()
        migration_group = migrations[migrations["run_id"].astype(str).eq(run_id)] if not migrations.empty else empty_migrations
        metrics = BASE.calculate_metrics(empty_requests, gpu_group, migration_group, 0.0)
        rows.append({"row_type": "replicate", "mode": mode, "qps_label": "qps_0",
                     "target_qps": 0.0, "seed": seed, "run_id": run_id, **metrics})

    replicates = pd.DataFrame(rows)
    numeric = [col for col in replicates.columns
               if col not in {"row_type", "mode", "qps_label", "run_id"}
               and pd.api.types.is_numeric_dtype(replicates[col])]
    aggregates = []
    for (mode, label, target), group in replicates.groupby(["mode", "qps_label", "target_qps"], dropna=False):
        aggregate = {"row_type": "aggregate", "mode": mode, "qps_label": label,
                     "target_qps": target, "seed": "all", "run_id": f"aggregate_{mode}_{label}"}
        for col in numeric:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            aggregate[f"{col}_mean"] = float(values.mean()) if len(values) else None
            aggregate[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else None)
        aggregates.append(aggregate)
    return pd.concat([replicates, pd.DataFrame(aggregates)], ignore_index=True, sort=False)


def write_report(out: Path, config: dict, summary, plots: list[str], state: dict) -> None:
    replicates = summary[summary["row_type"].eq("replicate")].copy()
    lookup = {(str(r.mode), str(r.qps_label)): r for r in replicates.itertuples(index=False)}
    rows = []
    crosses = []
    for level in config["qps_levels"]:
        label, target = level["label"], level["qps"]
        off = lookup.get(("off", label))
        on = lookup.get(("on", label))
        if off is None or on is None:
            continue
        def value(row, name):
            v = getattr(row, name, None)
            if v is None:
                return None
            try:
                number = float(v)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None
        e2e_off, e2e_on = value(off, "e2e_p95_ms"), value(on, "e2e_p95_ms")
        ttft_mean_off, ttft_mean_on = value(off, "ttft_mean_ms"), value(on, "ttft_mean_ms")
        ttft_off, ttft_on = value(off, "ttft_p95_ms"), value(on, "ttft_p95_ms")
        tpot_mean_off, tpot_mean_on = value(off, "tpot_mean_ms"), value(on, "tpot_mean_ms")
        tpot_off, tpot_on = value(off, "tpot_p95_ms"), value(on, "tpot_p95_ms")
        sent_off, sent_on = int(value(off, "sent_requests") or 0), int(value(on, "sent_requests") or 0)
        send_qps_off, send_qps_on = value(off, "actual_send_qps"), value(on, "actual_send_qps")
        window_qps_off, window_qps_on = value(off, "in_window_completion_qps"), value(on, "in_window_completion_qps")
        success_off = value(off, "successful_requests")
        success_on = value(on, "successful_requests")
        mig_on = int(value(on, "migration_successes") or 0)
        pct = ((e2e_on / e2e_off) - 1.0) * 100.0 if e2e_off and e2e_on is not None else None
        rows.append(
            f"| {target:g} | {send_qps_off:.2f}/{send_qps_on:.2f} "
            f"| {window_qps_off:.2f}/{window_qps_on:.2f} | {sent_off}/{sent_on} | {success_off:g}/{success_on:g} "
            f"| {e2e_off:.1f}/{e2e_on:.1f} | {ttft_mean_off:.1f}/{ttft_mean_on:.1f} "
            f"| {ttft_off:.1f}/{ttft_on:.1f} | {tpot_mean_off:.1f}/{tpot_mean_on:.1f} "
            f"| {tpot_off:.1f}/{tpot_on:.1f} | {pct:+.1f}% | {mig_on} |"
            if None not in (send_qps_off, send_qps_on, window_qps_off, window_qps_on,
                            e2e_off, e2e_on, ttft_mean_off, ttft_mean_on, ttft_off, ttft_on,
                            tpot_mean_off, tpot_mean_on, tpot_off, tpot_on, pct,
                            success_off, success_on)
            else f"| {target:g} | {send_qps_off if send_qps_off is not None else 'N/A'}/{send_qps_on if send_qps_on is not None else 'N/A'} "
                 f"| {window_qps_off if window_qps_off is not None else 'N/A'}/{window_qps_on if window_qps_on is not None else 'N/A'} "
                 f"| {sent_off}/{sent_on} | {success_off}/{success_on} | N/A | N/A | N/A | N/A | N/A | N/A | {mig_on} |"
        )
        if e2e_off is not None and e2e_on is not None and e2e_on < e2e_off:
            crosses.append({"qps": target, "label": label, "p95_e2e_improvement_pct": -pct,
                            "successful_migrations_on": mig_on})

    first_cross = min(crosses, key=lambda x: x["qps"]) if crosses else None
    if config.get("pair_suppression_ablation"):
        crossing_text = (
            "本次是迁移配对抑制消融：ON 保留迁移初始化、过滤、Balanced 配对计算和后台轮询，"
            "但不执行任何迁移。该组与 OFF 的性能差异对应这套启用路径在无实际迁移时的观测差异。"
        )
    elif first_cross:
        crossing_text = (
            f"首次测得 ON 的 P95 端到端延迟低于 OFF：**{first_cross['qps']:g} QPS**，"
            f"差异约 {first_cross['p95_e2e_improvement_pct']:.1f}%。ON 记录到 "
            f"{first_cross['successful_migrations_on']} 次成功迁移。"
        )
        if first_cross["successful_migrations_on"] == 0:
            crossing_text += " 该点没有观测到成功迁移，因此不能把延迟差异归因于 KV Cache 迁移。"
        else:
            crossing_text += " 这只是单种子、稀疏 QPS 网格得到的候选交叉点，不是统计确认的临界值。"
    else:
        crossing_text = (
            f"在本次测试范围（{min(x['qps'] for x in config['qps_levels']):g}–"
            f"{max(x['qps'] for x in config['qps_levels']):g} QPS）内，没有测到 ON 的 P95 端到端延迟低于 OFF。"
        )

    plots_md = "\n".join(f"- `plots/{name}`" for name in plots) or "- 无图表"
    command = (
        "python benchmark/run_migration_crossover.py --seed " + str(config["seed"]) +
        " --qps " + " ".join(str(x["qps"]) for x in config["qps_levels"]) +
        f" --warmup-sec {config['warmup_sec']:g} --measure-sec {config['measure_sec']:g} "
        f"--drain-sec {config['drain_sec']:g} --policy {config['policy']}"
    )
    if config.get("pair_suppression_ablation"):
        command += " --suppress-pairs"
    suppressed_pairs = 0
    suppression_log = out.parent / "pairing_ablation_events.csv"
    if suppression_log.exists():
        suppressions = BASE.pd.read_csv(suppression_log)
        if not suppressions.empty:
            suppressed_pairs = int(BASE.pd.to_numeric(
                suppressions["suppressed_pair_count"], errors="coerce").fillna(0).sum())
    ablation_note = ""
    if config.get("pair_suppression_ablation"):
        ablation_note = (
            "- ON 保持 `ENABLE_ROUTINE_MIGRATION=true` 和 `balanced` 策略；过滤与 Balanced 配对计算照常运行，"
            "但配对结果在提交迁移任务前被实验开关丢弃。启动日志确认了该开关。\n"
            f"- 启动、预热、测量和排空日志中记录的被丢弃候选配对数：{suppressed_pairs}；ON 实际迁移事件数：0。"
        )
    idle_note = (
        "- QPS 0 是空闲控制：两个模式下均不发送请求，延迟及成功率记为 N/A；GPU 指标仍在测量窗口采集。\n"
        if any(level["qps"] == 0 for level in config["qps_levels"]) else ""
    )
    title = "Llumnix Migration Pair-Suppression 消融实验" if config.get("pair_suppression_ablation") \
        else "Llumnix Migration ON/OFF 快速交叉点实验"
    on_mode_label = "ON（配对丢弃）" if config.get("pair_suppression_ablation") else "ON"
    out.write_text(f"""# {title}

## 结论

{crossing_text}

## 实验设置

- 状态：`{state.get('status')}`；种子：`{config['seed']}`；迁移策略：`{config['policy']}`。
- 四个独立实例，TP=1，模型 `{config['model_path']}`；GPU 显存利用率配置 `{config['gpu_memory_utilization']}`。
- OFF/ON 配置仅 `ENABLE_ROUTINE_MIGRATION` 不同；调度策略、rayrpc、`remaining_steps`、模型及其他参数相同。启动日志确认了开关和 Balanced 策略，Ray 状态确认四张 GPU。
- 比较组为 OFF / {on_mode_label}。
{ablation_note}
- 每个 QPS 的 Poisson 请求轨迹只生成一次，OFF/ON 共用相同轨迹文件；实际发送 QPS、请求完成情况和逐请求延迟保存在 `requests.csv`。
{idle_note}
- 预热 {config['warmup_sec']:g}s、测量 {config['measure_sec']:g}s，停止发送后最多排空 {config['drain_sec']:g}s。单种子结果用于快速筛查，不能代表正式重复实验或统计显著性。
- GPU 每秒采样；迁移次数按 Llumnix server log 中可解析的迁移完成日志计数。未解析到事件不代表没有尝试迁移。

## 配对结果

延迟列按 `OFF / ON` 排列，单位 ms；相对变化按 ON 相对 OFF 计算，负数代表 ON 更快。

| 目标 QPS | 实际发送 QPS OFF/ON | 测量窗完成 QPS OFF/ON | 发送数 OFF/ON | 排空后成功数 OFF/ON | P95 E2E OFF/ON | Mean TTFT OFF/ON | P95 TTFT OFF/ON | Mean TPOT OFF/ON | P95 TPOT OFF/ON | P95 E2E 变化 | ON 成功迁移数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows) if rows else '| — | — | — | — | — | — | — | — | — | — | — | — |'}

TTFT/TPOT 优先来自服务端逐 Token 延迟时间线；缺失值不会估算。SLO 阈值沿用基准脚本默认值（TTFT {BASE.SLO_TTFT_MS:g}ms，TPOT {BASE.SLO_TPOT_MS:g}ms），有效样本和违约率见 `summary.csv`。

“测量窗完成 QPS”只统计 20 秒发送窗口内完成的请求；“排空后成功数”还包含窗口结束后完成的请求，不能单独作为稳态吞吐量。

## 图表

{plots_md}

## 文件

- `requests.csv`：逐请求测量和预热记录。
- `gpu_metrics.csv`：逐卡 GPU 利用率、显存等采样。
- `migration_events.csv`：可解析的迁移完成事件。
- `pairing_ablation_events.csv`：Balanced 计算出配对、随后被消融开关丢弃的记录（如有）。
- `summary.csv`：每个 QPS × 模式的单种子指标。
- `config.json`、`state.json`、`configs/`、`traces/`、`server_logs/`：真实配置、配对轨迹、日志和运行状态。

## 复现命令

```bash
cd {ROOT}
source .venv/bin/activate
{command}
```
""", encoding="utf-8")


async def run(args) -> int:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_prefix = "migration_pair_suppression" if args.suppress_pairs else "migration_crossover"
    out = ROOT / "benchmark" / "results" / f"{experiment_prefix}_{timestamp}"
    BASE.OUT = out
    BASE.SEEDS = [args.seed]
    BASE.WARMUP_SEC = args.warmup_sec
    BASE.MEASURE_SEC = args.measure_sec
    BASE.DRAIN_SEC = args.drain_sec
    BASE.MONITOR_INTERVAL_SEC = 1.0

    if not BASE.MODEL.exists():
        raise RuntimeError(f"Model directory not found: {BASE.MODEL}")
    BASE.owned_process_preflight()
    environment = BASE.system_snapshot()
    if environment["gpu_count"] != 4 or not environment["torch"].get("cuda_available") or environment["torch"].get("device_count") != 4:
        raise RuntimeError(f"Expected four usable GPUs; environment={environment}")

    levels = qps_levels(args.qps)
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("configs", "traces", "server_logs", "raw_requests", "plots"):
        (out / directory).mkdir()
    config_paths = {}
    for mode in ("off", "on"):
        path = out / "configs" / f"llumnix_{mode}.yml"
        path.write_text(BASE.yaml.safe_dump(BASE.yaml_config(mode, args.policy), sort_keys=False), encoding="utf-8")
        config_paths[mode] = path
    BASE.check_mode_configs(config_paths)

    formal_traces = BASE.create_traces(levels)
    trace_hashes = {}
    for level in levels:
        path = out / "traces" / f"trace_{level['label']}_seed{args.seed}.csv"
        trace_hashes[f"{level['label']}_seed{args.seed}"] = sha256(path)

    config = {
        "experiment_id": "migration_pair_suppression_ablation" if args.suppress_pairs
        else "migration_crossover_quick",
        "created_at_utc": BASE.now_iso(),
        "environment": environment,
        "model_path": str(BASE.MODEL),
        "policy": args.policy,
        "gpu_memory_utilization": BASE.GPU_MEMORY_UTIL,
        "runtime": {
            "initial_instances": 4, "tensor_parallel_size": 1,
            "max_model_len": BASE.MODEL_MAX_LEN, "max_num_seqs_per_instance": BASE.MAX_NUM_SEQS,
            "gpu_memory_utilization": BASE.GPU_MEMORY_UTIL, "dtype": "half",
            "dispatch_policy": "load", "dispatch_load_metric": "remaining_steps",
            "routine_migration_policy": args.policy, "routine_migration_frequency": 1,
            "migration_backend": "rayrpc", "pre_stop_migration": False,
            "routine_migration_enabled": {"off": False, "on": True},
            "pair_suppression_ablation": {
                "enabled": args.suppress_pairs,
                "applied_to": "on" if args.suppress_pairs else None,
                "method": "run configured policy and filters, then discard all selected pairs before migration dispatch"
                if args.suppress_pairs else None,
            },
        },
        "pair_suppression_ablation": args.suppress_pairs,
        "seed": args.seed,
        "qps_levels": levels,
        "arrival_model": "poisson",
        "profile_mix": BASE.PROFILE_MIX,
        "warmup_sec": args.warmup_sec,
        "measure_sec": args.measure_sec,
        "drain_sec": args.drain_sec,
        "request_timeout_sec": BASE.REQUEST_TIMEOUT_SEC,
        "slo": {"ttft_ms": BASE.SLO_TTFT_MS, "tpot_ms": BASE.SLO_TPOT_MS},
        "trace_sha256": trace_hashes,
        "configs": {mode: str(path.relative_to(out)) for mode, path in config_paths.items()},
        "output_dir": str(out),
    }
    state = {"experiment_id": config["experiment_id"], "status": "running",
             "started_at_utc": BASE.now_iso(), "runs": {}, "completed_formal_runs": 0}
    BASE.atomic_json(out / "config.json", config)
    BASE.atomic_json(out / "state.json", state)
    (out / "environment_audit.md").write_text(BASE.environment_audit(environment), encoding="utf-8")

    request_parts = out / "raw_requests"
    gpu_rows: list[dict] = []
    migration_events: list[dict] = []
    context = {"experiment_id": config["experiment_id"]}
    try:
        for mode in ("off", "on"):
            await BASE.run_mode(mode, config_paths[mode], context, state, request_parts, gpu_rows,
                                migration_events, formal_traces, levels, args.policy,
                                discard_migration_pairs=(mode == "on" and args.suppress_pairs))
        BASE.write_final_requests(request_parts, out / "requests.csv")
        migration_path = out / "migration_events.csv"
        if not migration_path.exists():
            migration_path.write_text(",".join(BASE.MIGRATION_FIELDS) + "\n", encoding="utf-8")
        suppression_path = out / "pairing_ablation_events.csv"
        if not suppression_path.exists():
            suppression_path.write_text(",".join(BASE.PAIR_SUPPRESSION_FIELDS) + "\n", encoding="utf-8")
        if args.suppress_pairs:
            migration_df = BASE.pd.read_csv(migration_path)
            on_migrations = migration_df[migration_df["mode"].astype(str).eq("on")] if not migration_df.empty else migration_df
            if not on_migrations.empty:
                raise RuntimeError(f"Pair-suppression ablation recorded actual ON migration events: {len(on_migrations)}")
            suppressed_df = BASE.pd.read_csv(suppression_path)
            config["suppressed_candidate_pair_count"] = int(
                BASE.pd.to_numeric(suppressed_df["suppressed_pair_count"], errors="coerce").fillna(0).sum()
            ) if not suppressed_df.empty else 0
        summary = BASE.aggregate_summary(out / "requests.csv", out / "gpu_metrics.csv",
                                         migration_path, out / "summary.csv")
        if any(level["qps"] == 0 for level in levels):
            summary = add_idle_control_rows(summary, out / "gpu_metrics.csv", migration_path, args.seed)
            summary.to_csv(out / "summary.csv", index=False)
        mode_labels = {"off": "Migration OFF", "on": "Migration ON (pairs suppressed)"} \
            if args.suppress_pairs else None
        plots = BASE.plot_results(summary, out / "plots", mode_labels=mode_labels)
        state["status"] = "complete"
        state["completed_at_utc"] = BASE.now_iso()
        state["completed_formal_runs"] = sum(
            1 for run in state.get("runs", {}).values() if run.get("status") == "complete"
        )
        BASE.atomic_json(out / "state.json", state)
        config["completed_formal_runs"] = state["completed_formal_runs"]
        config["effective_config_verified"] = True
        BASE.atomic_json(out / "config.json", config)
        write_report(out / "report.md", config, summary, plots, state)
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
    parser.add_argument("--seed", type=int, default=101, help="single paired trace seed")
    parser.add_argument("--qps", type=float, nargs="+", default=[5.5675, 8, 10, 12, 16, 20],
                        help="offered QPS values; each level is paired across migration modes")
    parser.add_argument("--warmup-sec", type=float, default=5.0)
    parser.add_argument("--measure-sec", type=float, default=20.0)
    parser.add_argument("--drain-sec", type=float, default=180.0)
    parser.add_argument("--policy", choices=("balanced", "defrag"), default="balanced")
    parser.add_argument("--suppress-pairs", action="store_true",
                        help="ablation: keep routine migration active and run filtering/policy, then discard selected pairs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (not args.qps or len(set(args.qps)) != len(args.qps) or any(qps < 0 for qps in args.qps)
            or not any(qps > 0 for qps in args.qps)
            or args.measure_sec <= 0 or args.warmup_sec < 0 or args.drain_sec <= 0):
        raise SystemExit("QPS values must be nonnegative with at least one positive load; measurement and drain must be positive.")
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        process = BASE.CURRENT_SERVER_PROCESS
        if process is not None and process.poll() is None:
            try:
                process.send_signal(BASE.signal.SIGTERM)
                process.wait(timeout=90)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=20)
                except Exception:
                    pass
        if BASE.CURRENT_CLUSTER_OWNED:
            try:
                BASE.run_cmd([str(ROOT / ".venv" / "bin" / "ray"), "stop", "--force"],
                             timeout=90, check=False)
            except Exception:
                pass
        print("Interrupted. Flushed request/GPU/event records remain in the timestamped output directory.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
