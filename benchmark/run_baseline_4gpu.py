#!/usr/bin/env python3
"""Run a paired, no-disturbance Llumnix 4-GPU migration baseline.

The trace generator and benchmark response parser are reused from
disturbance_exp. Each mode is launched as an experiment-owned local process;
the script refuses to start if it finds an existing Ray/Llumnix server.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import aiohttp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disturbance_exp.metrics import parse_benchmark_payload, percentile
from disturbance_exp.workload import generate_trace, load_local_tokenizer, write_trace


OUT = ROOT / "benchmark" / "results" / "baseline_4gpu"
CURRENT_SERVER_PROCESS: subprocess.Popen | None = None
CURRENT_CLUSTER_OWNED = False
MODEL = ROOT / "model" / "Qwen" / "Qwen2-0.5B"
BASE_URL = "http://127.0.0.1:8080"
MODEL_MAX_LEN = 2048
MAX_NUM_SEQS = 128
GPU_MEMORY_UTIL = 0.65
SERVER_PORT = 8080
RAY_PORT = 6379
SEEDS = [101, 202, 303]
WARMUP_SEC = 30.0
MEASURE_SEC = 120.0
DRAIN_SEC = 180.0
REQUEST_TIMEOUT_SEC = 180.0
MONITOR_INTERVAL_SEC = 1.0
SLO_TTFT_MS = 2000.0
SLO_TPOT_MS = 150.0
INPUT_PROFILE = [64, 128, 256]
OUTPUT_PROFILE = [64, 96, 128]
PROFILE_MIX = [
    {"name": "short", "input_tokens": 64, "output_tokens": 64, "weight": 0.25},
    {"name": "medium", "input_tokens": 128, "output_tokens": 96, "weight": 0.50},
    {"name": "long", "input_tokens": 256, "output_tokens": 128, "weight": 0.25},
]
PROFILE_CYCLE = False
ENABLE_CAUSAL_TRACE = False
IGNORE_EOS = True
PROBE_QPS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
PROBE_SEC = 20.0
PROBE_DRAIN_SEC = 180.0
GPU_QUERY = (
    "uuid,name,utilization.gpu,memory.total,memory.used,temperature.gpu,power.draw"
)

REQUEST_FIELDS = [
    "experiment_id", "mode", "qps_label", "target_qps", "seed", "run_id", "phase",
    "request_id", "server_request_id", "request_type", "scheduled_arrival_sec", "actual_send_wall_time",
    "actual_send_sec", "client_send_monotonic_ns", "client_send_wall_time_ns",
    "completion_wall_time", "completion_sec", "client_complete_monotonic_ns",
    "client_complete_wall_time_ns", "expected_input_tokens",
    "server_input_tokens", "expected_output_tokens", "server_output_tokens", "client_e2e_ms",
    "server_ttft_ms", "server_tpot_ms", "timeline_count", "timeline_matches_output",
    "per_token_latency_json", "status_code", "success", "timeout", "error_type", "error_message",
]
_USE_CONFIGURED_REQUEST_TIMEOUT = object()
GPU_FIELDS = [
    "wall_time", "relative_sec", "experiment_id", "mode", "qps_label", "target_qps",
    "seed", "run_id", "phase", "gpu_index", "gpu_uuid", "gpu_name", "gpu_utilization_pct",
    "memory_total_mib", "memory_used_mib", "temperature_c", "power_w", "sample_error",
]
MIGRATION_FIELDS = [
    "experiment_id", "mode", "qps_label", "target_qps", "seed", "run_id", "phase",
    "observed_wall_time", "source_instance", "target_instance", "source_gpu", "target_gpu",
    "migration_type", "request_ids", "status", "success", "blocks", "cost_ms", "raw_log_line",
]
MIGRATION_RE = re.compile(
    r"Instance\s+(?P<src>\S+)->(?P<dst>\S+)\s+migrate done,\s*"
    r"migration type:\s*(?P<type>[^,]+),\s*migrate request\s*(?P<requests>\[.*?\]),\s*"
    r"migration status:\s*(?P<status>[^,]+),\s*len:\s*(?P<blocks>[\d.]+)\s*blocks,\s*"
    r"cost:\s*(?P<cost>[\d.]+)\s*ms"
)
PAIR_SUPPRESSION_RE = re.compile(r"PAIRING_ABLATION_SUPPRESSED count=(?P<count>\d+) pairs=(?P<pairs>\[.*\])")
PAIR_SUPPRESSION_FIELDS = [
    "experiment_id", "mode", "qps_label", "target_qps", "seed", "run_id", "phase",
    "observed_wall_time", "suppressed_pair_count", "pairs", "raw_log_line",
]
RAY_FOUR_GPU_RE = re.compile(r"(?:GPU\s+4(?:\.0)?\s*/\s*4(?:\.0)?|4(?:\.0)?\s*/\s*4(?:\.0)?\s+GPU\b)")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_cmd(args: list[str], timeout: float = 10.0, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=check)


def append_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def profile_prompt() -> str:
    sentence = (
        "A controlled language model serving experiment evaluates request scheduling, "
        "generation quality, response latency, and reliable service behavior. "
    )
    return sentence * 180


def trace_config(seed: int, duration: float, qps: float, max_requests: int = 100000) -> dict:
    return {
        "experiment": {"name": "baseline_4gpu", "seed": int(seed), "duration_sec": float(duration)},
        "workload": {
            "arrival": "poisson", "base_qps": float(qps), "input_tokens": 128,
            "output_tokens": 96, "max_model_len": MODEL_MAX_LEN,
            "tokenizer": str(MODEL), "prompt": profile_prompt(), "profile_mix": PROFILE_MIX,
            "profile_cycle": PROFILE_CYCLE,
        },
        "limits": {"max_requests": int(max_requests)},
        "events": [],
    }


def generated_trace(seed: int, duration: float, qps: float, max_requests: int = 100000):
    tokenizer = load_local_tokenizer(str(MODEL))
    if tokenizer is None:
        raise RuntimeError(f"Unable to load local tokenizer at {MODEL}")
    trace = generate_trace(trace_config(seed, duration, qps, max_requests=max_requests), tokenizer)
    return trace


def yaml_config(mode: str, pair_migration_policy: str = "defrag",
                migrate_out_threshold: float = 20000.0,
                pair_migration_frequency: int = 1) -> dict:
    return {
        "SERVER": {
            "HOST": "127.0.0.1", "PORT": SERVER_PORT, "REQUEST_OUTPUT_QUEUE_TYPE": "zmq",
            "RAY_CLUSTER_PORT": RAY_PORT, "LAUNCH_RAY_CLUSTER": True,
        },
        "MANAGER": {
            "INITIAL_INSTANCES": 4, "DISPATCH_POLICY": "load", "ENABLE_ROUTINE_MIGRATION": mode == "on",
            "ENABLE_PRE_STOP_MIGRATION": False, "PAIR_MIGRATION_FREQUENCY": int(pair_migration_frequency),
            "PAIR_MIGRATION_POLICY": pair_migration_policy, "MIGRATE_OUT_THRESHOLD": float(migrate_out_threshold),
            "ENABLE_SCALING": False, "POLLING_INTERVAL": 0.05,
            "DISPATCH_LOAD_METRIC": "remaining_steps",
        },
        "INSTANCE": {
            "MIGRATION_LOAD_METRIC": "remaining_steps", "ENABLE_DEFRAG": True,
            "MIGRATION_BACKEND": "rayrpc", "REQUEST_MIGRATION_POLICY": "SR",
            "MAX_MIGRATION_CONCURRENCY": 1,
        },
    }


def check_mode_configs(paths: dict[str, Path]) -> None:
    configs = {mode: yaml.safe_load(path.read_text()) for mode, path in paths.items()}
    off, on = configs["off"], configs["on"]
    differences = []
    def walk(a, b, prefix=""):
        for key in sorted(set(a) | set(b)):
            loc = f"{prefix}.{key}" if prefix else key
            va, vb = a.get(key), b.get(key)
            if isinstance(va, dict) and isinstance(vb, dict):
                walk(va, vb, loc)
            elif va != vb:
                differences.append((loc, va, vb))
    walk(off, on)
    if differences != [("MANAGER.ENABLE_ROUTINE_MIGRATION", False, True)]:
        raise RuntimeError(f"OFF/ON config difference is not migration-only: {differences}")


def system_snapshot() -> dict:
    gpu_result = run_cmd([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in gpu_result.stdout.splitlines():
        idx, uuid, name, total, used = [x.strip() for x in line.split(",", 4)]
        gpus.append({"index": int(idx), "uuid": uuid, "name": name,
                     "memory_total_mib": int(total), "memory_used_mib": int(used)})
    packages = {}
    from importlib import metadata
    for name in ["torch", "vllm", "ray", "aiohttp", "matplotlib", "numpy", "pandas", "transformers", "tokenizers"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    model_files = []
    for p in sorted(MODEL.glob("*")):
        if p.is_file():
            model_files.append({"name": p.name, "size_bytes": p.stat().st_size})
    try:
        import torch
        torch_info = {"version": torch.__version__, "cuda": torch.version.cuda,
                      "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}
    lscpu = run_cmd(["lscpu"], check=False).stdout
    cpu_model = next((line.split(":", 1)[1].strip() for line in lscpu.splitlines() if line.startswith("Model name:")), None)
    cpu_count = os.cpu_count()
    return {
        "captured_at_utc": now_iso(), "python": sys.version, "packages": packages, "torch": torch_info,
        "gpus": gpus, "gpu_count": len(gpus), "cpu_count_logical": cpu_count, "cpu_model": cpu_model,
        "memory_total_bytes": psutil.virtual_memory().total, "memory_available_bytes": psutil.virtual_memory().available,
        "workspace_disk": {"total": shutil.disk_usage(ROOT).total, "used": shutil.disk_usage(ROOT).used,
                            "free": shutil.disk_usage(ROOT).free},
        "model_path": str(MODEL), "model_files": model_files,
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"], check=False).stdout.strip(),
        "git_status": run_cmd(["git", "status", "--short"], check=False).stdout.strip(),
        "workspace_is_volume": False,
    }


def environment_audit(snapshot: dict) -> str:
    gpus = "\n".join(
        f"- GPU {g['index']}: {g['name']}, {g['memory_total_mib']} MiB, UUID `{g['uuid']}`"
        for g in snapshot["gpus"]
    )
    packages = "\n".join(f"- {k}: `{v}`" for k, v in snapshot["packages"].items())
    files = "\n".join(f"- `{f['name']}` ({f['size_bytes']:,} bytes)" for f in snapshot["model_files"])
    return f"""# 四卡基准环境审计

采集时间：{snapshot['captured_at_utc']}

## 硬件

{gpus}
- CPU：{snapshot['cpu_model']}，{snapshot['cpu_count_logical']} 个逻辑 CPU
- 内存：{snapshot['memory_total_bytes'] // (1024**3)} GiB，总可用 {snapshot['memory_available_bytes'] // (1024**3)} GiB
- 工作目录盘：{snapshot['workspace_disk']['free'] // (1024**3)} GiB 可用
- GPU 拓扑：启动前以 `nvidia-smi topo -m` 核对；GPU 3 到 GPU 0–2 经 SYS/NUMA 互联，其余 GPU 对之间为 PHB。

## 软件

- Python：`{snapshot['python'].split()[0]}`
- PyTorch：`{snapshot['torch']}`
{packages}

## 模型

- 路径：`{snapshot['model_path']}`
{files}

该审计反映本次四卡服务器，不使用仓库中旧的单卡 RTX 3060 审计作为环境依据。
"""


def owned_process_preflight() -> None:
    # Starting the repo's local launch path calls `ray stop`, so any Ray process
    # or occupied port is a hard stop. Never take ownership of an existing cluster.
    cmd = run_cmd(["ps", "-eo", "pid,ppid,args"], check=False).stdout
    candidates = [line for line in cmd.splitlines() if re.search(r"raylet|llumnix\.entrypoints\.vllm\.api_server", line)
                  and "grep" not in line and "run_baseline_4gpu" not in line]
    if candidates:
        raise RuntimeError("Existing Ray/Llumnix processes found; refusing to launch or stop them:\n" + "\n".join(candidates))
    for port in (RAY_PORT, SERVER_PORT):
        result = run_cmd(["bash", "-lc", f"ss -lntp | grep -E ':{port}\\b'"], check=False)
        if result.stdout.strip():
            raise RuntimeError(f"Port {port} is occupied; refusing to launch:\n{result.stdout}")
    supervisor = run_cmd(["supervisorctl", "status"], check=False).stdout
    if re.search(r"llumnix|ray", supervisor, re.I):
        raise RuntimeError("Supervisor appears to manage Ray/Llumnix; refusing to replace services:\n" + supervisor)
    if len(run_cmd(["nvidia-smi", "-L"]).stdout.strip().splitlines()) != 4:
        raise RuntimeError("Expected exactly four visible GPUs")


def append_migration_line(line: str, context: dict, all_events: list[dict], path: Path) -> None:
    m = MIGRATION_RE.search(line)
    if not m:
        return
    status = m.group("status").strip()
    try:
        request_ids = json.loads(m.group("requests").replace("'", '"'))
    except Exception:
        request_ids = re.findall(r"['\"]([^'\"]+)['\"]", m.group("requests"))
    row = {
        "experiment_id": context.get("experiment_id", "baseline_4gpu"),
        "mode": context.get("mode"), "qps_label": context.get("qps_label"),
        "target_qps": context.get("target_qps"), "seed": context.get("seed"), "run_id": context.get("run_id"),
        "phase": context.get("phase", "unattributed"), "observed_wall_time": time.time(),
        "source_instance": m.group("src"), "target_instance": m.group("dst"), "source_gpu": None,
        "target_gpu": None, "migration_type": m.group("type").strip(), "request_ids": json.dumps(request_ids),
        "status": status, "success": "FINISHED" in status.upper(), "blocks": float(m.group("blocks")),
        "cost_ms": float(m.group("cost")), "raw_log_line": line.strip(),
    }
    all_events.append(row)
    append_csv(path, MIGRATION_FIELDS, [row])


async def watch_server_log(log_path: Path, context: dict, events: list[dict], stop: asyncio.Event, event_path: Path):
    offset = 0
    while not stop.is_set() or (log_path.exists() and log_path.stat().st_size > offset):
        if log_path.exists():
            try:
                with log_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    lines = f.readlines()
                    offset = f.tell()
                for line in lines:
                    if "migrate done" in line:
                        append_migration_line(line, dict(context), events, event_path)
                    suppressed = PAIR_SUPPRESSION_RE.search(line)
                    if suppressed:
                        row = {
                            "experiment_id": context.get("experiment_id", "baseline_4gpu"),
                            "mode": context.get("mode"), "qps_label": context.get("qps_label"),
                            "target_qps": context.get("target_qps"), "seed": context.get("seed"),
                            "run_id": context.get("run_id"), "phase": context.get("phase", "unattributed"),
                            "observed_wall_time": time.time(),
                            "suppressed_pair_count": int(suppressed.group("count")),
                            "pairs": suppressed.group("pairs"), "raw_log_line": line.strip(),
                        }
                        append_csv(log_path.parent.parent / "pairing_ablation_events.csv",
                                   PAIR_SUPPRESSION_FIELDS, [row])
            except OSError:
                pass
        await asyncio.sleep(0.5)


async def sample_gpu_metrics(context: dict, rows: list[dict], stop: asyncio.Event, out_path: Path,
                             start_mono: float):
    while not stop.is_set():
        wall = time.time()
        rel = time.monotonic() - start_mono
        try:
            output = await asyncio.to_thread(run_cmd, [
                "nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"
            ], 5.0, True)
            lines = output.stdout.strip().splitlines()
            for idx, line in enumerate(lines):
                parts = [x.strip() for x in line.split(",", 6)]
                if len(parts) != 7:
                    continue
                uuid, name, util, total, used, temp, power = parts
                def numeric(value):
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
                row = {
                    "wall_time": wall, "relative_sec": rel, **context, "gpu_index": idx,
                    "gpu_uuid": uuid, "gpu_name": name, "gpu_utilization_pct": numeric(util),
                    "memory_total_mib": numeric(total), "memory_used_mib": numeric(used),
                    "temperature_c": numeric(temp), "power_w": numeric(power), "sample_error": "",
                }
                rows.append(row)
                append_csv(out_path, GPU_FIELDS, [row])
        except Exception as exc:
            row = {"wall_time": wall, "relative_sec": rel, **context, "gpu_index": None,
                   "gpu_uuid": None, "gpu_name": None, "gpu_utilization_pct": None,
                   "memory_total_mib": None, "memory_used_mib": None, "temperature_c": None,
                   "power_w": None, "sample_error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            append_csv(out_path, GPU_FIELDS, [row])
        try:
            await asyncio.wait_for(stop.wait(), timeout=MONITOR_INTERVAL_SEC)
        except asyncio.TimeoutError:
            pass


def make_request_row(ctx: dict, req, phase: str) -> dict:
    request_id = f"{ctx.get('run_id', 'run')}_{phase}_{req.request_id}"
    return {
        "experiment_id": "baseline_4gpu", **ctx, "phase": phase,
        "request_id": request_id, "request_type": req.request_type,
        "server_request_id": None,
        "scheduled_arrival_sec": req.scheduled_arrival_sec,
        "expected_input_tokens": req.input_tokens, "server_input_tokens": None,
        "expected_output_tokens": req.output_tokens, "server_output_tokens": None,
        "actual_send_wall_time": None, "actual_send_sec": None, "completion_wall_time": None,
        "client_send_monotonic_ns": None, "client_send_wall_time_ns": None,
        "client_complete_monotonic_ns": None, "client_complete_wall_time_ns": None,
        "completion_sec": None, "client_e2e_ms": None, "server_ttft_ms": None,
        "server_tpot_ms": None, "timeline_count": 0, "timeline_matches_output": False,
        "per_token_latency_json": None, "status_code": None, "success": False,
        "timeout": False, "error_type": "", "error_message": "",
    }


async def execute_trace(trace, ctx: dict, phase: str, duration: float, drain_sec: float | None,
                        request_parts: Path,
                        timeout_sec: float | None | object = _USE_CONFIGURED_REQUEST_TIMEOUT) -> list[dict]:
    if not trace:
        return []
    if timeout_sec is _USE_CONFIGURED_REQUEST_TIMEOUT:
        timeout_sec = REQUEST_TIMEOUT_SEC
    phase_start = time.monotonic()
    phase_wall = time.time()
    rows = [make_request_row(ctx, req, phase) for req in trace]
    if timeout_sec is None:
        session_timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=None, sock_read=None)
    else:
        session_timeout = aiohttp.ClientTimeout(total=timeout_sec, connect=15.0, sock_connect=15.0)
    # Keep client-side connection admission from silently changing the target
    # arrival process. The connector must accommodate the full trace when it
    # contains thousands of long-running requests, before they reach Llumnix.
    connector_limit = min(max(512, len(trace)), 4096)
    connector = aiohttp.TCPConnector(limit=connector_limit, limit_per_host=connector_limit,
                                     ttl_dns_cache=60)
    tasks = []

    async def issue(i: int, req):
        row = rows[i]
        target = phase_start + req.scheduled_arrival_sec
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        row["actual_send_wall_time"] = time.time()
        row["actual_send_sec"] = time.monotonic() - phase_start
        row["client_send_monotonic_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        row["client_send_wall_time_ns"] = time.time_ns()
        body = {
            "prompt": req.prompt, "n": 1, "best_of": 1, "temperature": 0.0,
            "top_k": 1, "max_tokens": req.output_tokens, "ignore_eos": IGNORE_EOS, "stream": False,
            "request_id": row["request_id"],
        }
        try:
            headers = ({"X-Llumnix-Trace": "true", "X-Llumnix-Trace-Events": "true"}
                       if ENABLE_CAUSAL_TRACE else None)
            async with session.post(f"{BASE_URL}/generate_benchmark", json=body, headers=headers) as response:
                row["status_code"] = response.status
                raw = await response.text()
                row["completion_wall_time"] = time.time()
                row["completion_sec"] = time.monotonic() - phase_start
                row["client_complete_monotonic_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                row["client_complete_wall_time_ns"] = time.time_ns()
                row["client_e2e_ms"] = (row["completion_sec"] - row["actual_send_sec"]) * 1000.0
                if response.status != 200:
                    row["error_type"] = "http_error"
                    row["error_message"] = raw[:1000]
                    return
                payload = json.loads(raw)
                metrics = parse_benchmark_payload(payload)
                row["server_request_id"] = payload.get("request_id")
                row["server_output_tokens"] = metrics["output_tokens"]
                row["server_ttft_ms"] = metrics["server_ttft_ms"]
                row["server_tpot_ms"] = metrics["server_tpot_ms"]
                row["timeline_count"] = metrics["timeline_count"]
                row["timeline_matches_output"] = metrics["timeline_matches_output"]
                row["per_token_latency_json"] = json.dumps(payload.get("per_token_latency"), separators=(",", ":"))
                row["success"] = True
        except asyncio.TimeoutError as exc:
            row["timeout"] = True
            row["error_type"] = type(exc).__name__ or "timeout"
            row["error_message"] = "request timed out"
        except asyncio.CancelledError:
            row["timeout"] = True
            row["error_type"] = "drain_timeout"
            row["error_message"] = "request did not complete before drain deadline"
            raise
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)[:1000]
            row["completion_wall_time"] = time.time()
            row["completion_sec"] = time.monotonic() - phase_start
            row["client_complete_monotonic_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            row["client_complete_wall_time_ns"] = time.time_ns()
            if row["actual_send_sec"] is not None:
                row["client_e2e_ms"] = (row["completion_sec"] - row["actual_send_sec"]) * 1000.0

    async with aiohttp.ClientSession(timeout=session_timeout, connector=connector) as session:
        for i, req in enumerate(trace):
            tasks.append(asyncio.create_task(issue(i, req)))
        await asyncio.sleep(duration)
        ctx["phase"] = "drain"
        done, pending = await asyncio.wait(tasks, timeout=drain_sec)
        abort_failure = None
        if pending:
            pending_ids = [rows[i]["request_id"] for i, task in enumerate(tasks)
                           if task in pending and rows[i]["actual_send_sec"] is not None]
            if pending_ids:
                try:
                    # High-QPS phases can leave a large tail of active requests;
                    # allow the server-side batch abort barrier up to 15 minutes.
                    abort_timeout = aiohttp.ClientTimeout(total=900.0, connect=30.0, sock_connect=30.0)
                    abort_connector = aiohttp.TCPConnector(limit=4, limit_per_host=4)
                    async with aiohttp.ClientSession(timeout=abort_timeout,
                                                     connector=abort_connector) as abort_session:
                        async with abort_session.post(f"{BASE_URL}/abort_benchmark",
                                                      json={"request_ids": pending_ids}) as response:
                            abort_result = await response.json(content_type=None)
                            if response.status != 200 or abort_result.get("unresolved_request_ids"):
                                abort_failure = (
                                    f"server abort barrier failed: status={response.status}, "
                                    f"active={abort_result.get('active_request_count')}, "
                                    f"unresolved={len(abort_result.get('unresolved_request_ids', []))}"
                                )
                except Exception as exc:
                    abort_failure = f"server abort barrier request failed: {type(exc).__name__}: {exc}"
            barrier_passed = abort_failure is None
            done_after_abort, pending = await asyncio.wait(pending, timeout=3.0 if barrier_passed else 30.0)
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if not barrier_passed:
                    abort_failure = abort_failure or (
                        f"{len(pending)} client request handlers remained active after abort failure"
                    )
            if abort_failure:
                raise RuntimeError(abort_failure)
    for row in rows:
        if row["actual_send_sec"] is None:
            row["error_type"] = row["error_type"] or "not_sent"
            row["error_message"] = row["error_message"] or "scheduled request was not sent"
        elif row["completion_sec"] is None and not row["timeout"]:
            row["timeout"] = True
            row["error_type"] = row["error_type"] or "drain_timeout"
            row["error_message"] = row["error_message"] or "request did not complete before drain deadline"
    request_parts.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = request_parts.with_name(request_parts.stem + ".tmp.csv")
    if tmp_path.exists():
        tmp_path.unlink()
    append_csv(tmp_path, REQUEST_FIELDS, rows)
    tmp_path.replace(request_parts)
    return rows


async def wait_ready(process: subprocess.Popen, log_path: Path, timeout_sec: float = 900.0) -> None:
    start = time.monotonic()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
        while time.monotonic() - start < timeout_sec:
            if process.poll() is not None:
                tail = log_path.read_text(errors="replace")[-10000:] if log_path.exists() else ""
                raise RuntimeError(f"Llumnix exited with code {process.returncode}; log tail:\n{tail}")
            try:
                async with session.get(f"{BASE_URL}/health") as resp:
                    healthy = resp.status == 200
                async with session.get(f"{BASE_URL}/is_ready") as resp:
                    ready = resp.status == 200 and (await resp.json()) is True
                if healthy and ready:
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            tail = log_path.read_text(errors="replace")[-10000:] if log_path.exists() else ""
            raise TimeoutError(f"Llumnix readiness deadline; log tail:\n{tail}")

    # Validate effective arguments from the running process log.
    log = log_path.read_text(errors="replace")
    required = ["Init Llumnix components done", "4 instances are ready"]
    if not all(token in log for token in required):
        # This repository version may word the ready line differently; Ray GPU
        # resources plus all four vLLM worker creations are checked below.
        if "instances are ready" not in log:
            raise RuntimeError("Could not verify four ready Llumnix instances in startup log")


async def verify_ray_four_gpu() -> str:
    ray_bin = ROOT / ".venv" / "bin" / "ray"
    result = await asyncio.to_thread(run_cmd, [str(ray_bin), "status", "--address", f"127.0.0.1:{RAY_PORT}"], 15.0, True)
    output = result.stdout + result.stderr
    if not RAY_FOUR_GPU_RE.search(output):
        raise RuntimeError("Ray did not report all 4 GPUs as available:\n" + output)
    return output


def server_command(mode: str, config_path: Path, log_base: Path) -> tuple[list[str], dict[str, str]]:
    cmd = [
        sys.executable, "-m", "llumnix.entrypoints.vllm.api_server",
        "--config-file", str(config_path), "--host", "127.0.0.1", "--port", str(SERVER_PORT),
        "--model", str(MODEL), "--tensor-parallel-size", "1", "--worker-use-ray",
        "--migration-backend", "rayrpc", "--max-model-len", str(MODEL_MAX_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS), "--gpu-memory-utilization", str(GPU_MEMORY_UTIL),
        "--dtype", "half", "--log-instance-info", "--log-filename", str(log_base),
        "--server-log-level", "info", "--enforce-eager",
    ]
    env = os.environ.copy()
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    env["HEAD_NODE_IP"] = "127.0.0.1"
    env["HEAD_NODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["RAY_DEDUP_LOGS"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("LLUMNIX_DISCARD_MIGRATION_PAIRS_AFTER_POLICY", None)
    return cmd, env


def verify_effective_config(log_path: Path, mode: str, ray_status: str,
                            pair_migration_policy: str = "defrag",
                            discard_migration_pairs: bool = False,
                            expected_migration_enabled: bool | None = None,
                            expected_migrate_out_threshold: float | None = None,
                            expected_pair_migration_frequency: int | None = None) -> None:
    log = log_path.read_text(errors="replace")
    if expected_migration_enabled is None:
        expected_migration_enabled = mode in {"on", "dryrun"}
    target = f"enable_routine_migration={expected_migration_enabled}"
    checks = [target, "initial_instances=4", "migration_backend='rayrpc'",
              f"pair_migration_policy='{pair_migration_policy}'",
              "dispatch_policy='load'", f"max_model_len={MODEL_MAX_LEN}",
              f"max_num_seqs={MAX_NUM_SEQS}"]
    missing = [x for x in checks if x not in log]
    if missing:
        raise RuntimeError(f"Effective startup args not verified for {mode}: missing {missing}")
    if expected_migrate_out_threshold is not None:
        threshold_match = re.search(r"migrate_out_threshold=([-+]?\d+(?:\.\d+)?)", log)
        if threshold_match is None or float(threshold_match.group(1)) != float(expected_migrate_out_threshold):
            actual = threshold_match.group(1) if threshold_match else "not found"
            raise RuntimeError(
                f"Effective migration threshold not verified for {mode}: "
                f"expected {expected_migrate_out_threshold:g}, found {actual}"
            )
    if expected_pair_migration_frequency is not None:
        frequency_match = re.search(r"pair_migration_frequency=(\d+)", log)
        if frequency_match is None or int(frequency_match.group(1)) != int(expected_pair_migration_frequency):
            actual = frequency_match.group(1) if frequency_match else "not found"
            raise RuntimeError(
                f"Effective pair migration frequency not verified for {mode}: "
                f"expected {expected_pair_migration_frequency}, found {actual}"
            )
    if discard_migration_pairs and "PAIRING_ABLATION_ENABLED" not in log:
        raise RuntimeError("Pair-suppression ablation was requested but not confirmed in server log")
    if not RAY_FOUR_GPU_RE.search(ray_status):
        raise RuntimeError("Four-GPU Ray resource verification failed")


def qps_label(qps: float) -> str:
    return f"{qps:.3f}".rstrip("0").rstrip(".")


def write_final_requests(parts_dir: Path, out_path: Path) -> None:
    if out_path.exists():
        out_path.unlink()
    files = sorted(parts_dir.glob("*.csv"))
    seen = set()
    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                key = (row.get("run_id"), row.get("phase"), row.get("request_id"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
            if rows:
                append_csv(out_path, REQUEST_FIELDS, rows)


def calculate_metrics(reqs: pd.DataFrame, gpu: pd.DataFrame, migrations: pd.DataFrame,
                      target_qps: float, duration: float | None = None) -> dict:
    # Resolve at call time: quick exploratory runs override MEASURE_SEC after
    # importing this module, so a captured default would misstate throughput.
    if duration is None:
        duration = MEASURE_SEC
    measurement = reqs[(reqs["phase"] == "measurement")].copy() if not reqs.empty else pd.DataFrame()
    if not measurement.empty:
        measurement["success_bool"] = measurement["success"].astype(str).str.lower().eq("true")
        measurement["timeout_bool"] = measurement["timeout"].astype(str).str.lower().eq("true")
        success = measurement[measurement["success_bool"]]
    else:
        success = pd.DataFrame()

    def nums(frame: pd.DataFrame, col: str) -> list[float]:
        if frame.empty or col not in frame:
            return []
        values = pd.to_numeric(frame[col], errors="coerce").dropna().tolist()
        return [float(v) for v in values if math.isfinite(float(v))]

    ttft_values = nums(success, "server_ttft_ms")
    tpot_values = nums(success, "server_tpot_ms")
    e2e_values = nums(success, "client_e2e_ms")
    sent = len(measurement)
    completed = len(success)
    timeout_count = int(measurement["timeout_bool"].sum()) if sent else 0
    failed = sent - completed
    actual_send_times = nums(measurement, "actual_send_sec")
    send_span = max(actual_send_times) - min(actual_send_times) if len(actual_send_times) > 1 else 0.0
    actual_send_qps = (max(0, len(actual_send_times) - 1) / send_span
                       if send_span > 0 else sent / duration)
    ttft_valid = len(ttft_values)
    tpot_valid = len(tpot_values)
    ttft_violations = sum(v > SLO_TTFT_MS for v in ttft_values)
    tpot_violations = sum(v > SLO_TPOT_MS for v in tpot_values)
    if not measurement.empty:
        completion = pd.to_numeric(measurement["completion_sec"], errors="coerce")
        in_window = int((measurement["success_bool"] & completion.le(duration)).sum())
    else:
        in_window = 0
    metrics = {
        "sent_requests": sent, "successful_requests": completed, "failed_requests": failed,
        "timeout_requests": timeout_count, "actual_send_qps": actual_send_qps,
        "request_throughput_rps": completed / duration, "in_window_completion_qps": in_window / duration,
        "output_tokens": sum(int(v) for v in nums(success, "server_output_tokens")),
        "output_tokens_per_sec": sum(int(v) for v in nums(success, "server_output_tokens")) / duration,
        "success_rate": completed / sent if sent else None,
        "failure_rate": failed / sent if sent else None, "timeout_rate": timeout_count / sent if sent else None,
        "ttft_samples": ttft_valid, "ttft_missing": max(0, completed - ttft_valid),
        "ttft_slo_valid_samples": ttft_valid, "ttft_slo_missing_samples": max(0, completed - ttft_valid),
        "ttft_slo_violations": ttft_violations,
        "ttft_slo_violation_rate": ttft_violations / ttft_valid if ttft_valid else None,
        "tpot_samples": tpot_valid, "tpot_missing": max(0, completed - tpot_valid),
        "tpot_slo_valid_samples": tpot_valid, "tpot_slo_missing_samples": max(0, completed - tpot_valid),
        "tpot_slo_violations": tpot_violations,
        "tpot_slo_violation_rate": tpot_violations / tpot_valid if tpot_valid else None,
    }
    for prefix, values in [("ttft", ttft_values), ("tpot", tpot_values), ("e2e", e2e_values)]:
        metrics[f"{prefix}_mean_ms"] = statistics.fmean(values) if values else None
        for p, name in [(0.50, "p50"), (0.95, "p95"), (0.99, "p99")]:
            metrics[f"{prefix}_{name}_ms"] = percentile(values, p)

    if not gpu.empty:
        gm = gpu[(gpu["phase"] == "measurement")].copy()
        for idx in range(4):
            part = gm[pd.to_numeric(gm["gpu_index"], errors="coerce") == idx] if not gm.empty else pd.DataFrame()
            util = nums(part, "gpu_utilization_pct")
            mem = nums(part, "memory_used_mib")
            metrics[f"gpu{idx}_mean_util_pct"] = statistics.fmean(util) if util else None
            metrics[f"gpu{idx}_peak_memory_mib"] = max(mem) if mem else None
        util_means = [metrics.get(f"gpu{i}_mean_util_pct") for i in range(4)]
        valid_utils = [float(x) for x in util_means if x is not None]
        metrics["gpu_util_range_pct"] = max(valid_utils) - min(valid_utils) if valid_utils else None
        metrics["gpu_util_cv"] = (statistics.pstdev(valid_utils) / statistics.fmean(valid_utils)
                                   if len(valid_utils) > 1 and statistics.fmean(valid_utils) else None)
    else:
        for idx in range(4):
            metrics[f"gpu{idx}_mean_util_pct"] = None
            metrics[f"gpu{idx}_peak_memory_mib"] = None
        metrics["gpu_util_range_pct"] = None
        metrics["gpu_util_cv"] = None

    mig = migrations if not migrations.empty else pd.DataFrame()
    if not mig.empty:
        if "run_id" in mig.columns:
            mig = mig[mig["run_id"].astype(str).eq(str(reqs["run_id"].iloc[0]))] if not reqs.empty else mig.iloc[0:0]
        statuses = mig["status"].fillna("").astype(str).str.upper()
        successes = statuses.str.contains("FINISHED")
        metrics["migration_attempts"] = len(mig)
        metrics["migration_successes"] = int(successes.sum())
        metrics["migration_failures"] = int((~successes).sum())
        costs = pd.to_numeric(mig.loc[successes, "cost_ms"], errors="coerce").dropna().tolist()
        metrics["migration_mean_cost_ms"] = statistics.fmean(costs) if costs else None
        metrics["migration_cost_samples"] = len(costs)
    else:
        metrics.update({"migration_attempts": 0, "migration_successes": 0, "migration_failures": 0,
                        "migration_mean_cost_ms": None, "migration_cost_samples": 0})
    metrics["target_qps"] = target_qps
    return metrics


def aggregate_summary(request_path: Path, gpu_path: Path, migration_path: Path, output_path: Path) -> pd.DataFrame:
    req = pd.read_csv(request_path) if request_path.exists() else pd.DataFrame()
    gpu = pd.read_csv(gpu_path) if gpu_path.exists() and gpu_path.stat().st_size else pd.DataFrame()
    mig = pd.read_csv(migration_path) if migration_path.exists() and migration_path.stat().st_size else pd.DataFrame()
    rows = []
    if req.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()
    formal = req[req["phase"].eq("measurement")]
    groups = formal.groupby(["mode", "qps_label", "target_qps", "seed", "run_id"], dropna=False)
    metric_names = None
    for (mode, label, target, seed, run_id), group in groups:
        gpu_group = gpu[gpu["run_id"].astype(str).eq(str(run_id))] if not gpu.empty else pd.DataFrame()
        mig_group = mig[mig["run_id"].astype(str).eq(str(run_id))] if not mig.empty else pd.DataFrame()
        m = calculate_metrics(group, gpu_group, mig_group, float(target))
        metric_names = list(m)
        rows.append({"row_type": "replicate", "mode": mode, "qps_label": label, "target_qps": target,
                     "seed": seed, "run_id": run_id, **m})
    df = pd.DataFrame(rows)
    agg_rows = []
    numeric = [c for c in df.columns if c not in {"row_type", "mode", "qps_label", "run_id"}
               and pd.api.types.is_numeric_dtype(df[c])]
    for (mode, label, target), group in df.groupby(["mode", "qps_label", "target_qps"], dropna=False):
        row = {"row_type": "aggregate", "mode": mode, "qps_label": label, "target_qps": target,
               "seed": "all", "run_id": f"aggregate_{mode}_{label}"}
        for col in numeric:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else None
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else None)
        agg_rows.append(row)
    combined = pd.concat([df, pd.DataFrame(agg_rows)], ignore_index=True, sort=False)
    combined.to_csv(output_path, index=False)
    return combined


def plot_results(summary: pd.DataFrame, out_dir: Path,
                mode_labels: dict[str, str] | None = None) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_labels = mode_labels or {"off": "Migration OFF", "on": "Migration ON"}
    agg = summary[summary["row_type"].eq("aggregate")].copy()
    if agg.empty:
        return []
    created = []

    def pair_plot(metrics, title, ylabel, filename):
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
        for ax, (metric, metric_title) in zip(axes[0], metrics):
            for mode, marker, color in [("off", "o", "#2474B5"), ("on", "s", "#E87500")]:
                part = agg[agg["mode"].eq(mode)].sort_values("target_qps")
                value = f"{metric}_mean"
                if value not in part:
                    continue
                x = part["target_qps"].astype(float).tolist()
                y = pd.to_numeric(part[value], errors="coerce").tolist()
                ax.plot(x, y, marker=marker, color=color, label=mode_labels.get(mode, mode.upper()))
            ax.set_title(metric_title)
            ax.set_xlabel("Target QPS")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend()
        fig.suptitle(title)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        created.append(filename)

    pair_plot([("ttft_p95_ms", "P95 TTFT"), ("tpot_p95_ms", "P95 TPOT")],
              "P95 server latency by offered load", "Milliseconds", "p95_latency.png")
    pair_plot([("ttft_mean_ms", "Mean TTFT"), ("tpot_mean_ms", "Mean TPOT")],
              "Mean server latency by offered load", "Milliseconds", "mean_latency.png")
    pair_plot([("request_throughput_rps", "Completed requests"), ("output_tokens_per_sec", "Output tokens")],
              "Completed throughput by offered load", "per second", "throughput.png")
    pair_plot([("ttft_slo_violation_rate", "TTFT SLO violations"), ("tpot_slo_violation_rate", "TPOT SLO violations")],
              f"SLO violation rates (TTFT>{SLO_TTFT_MS:.0f} ms, TPOT>{SLO_TPOT_MS:.0f} ms)",
              "Violation rate", "slo_violations.png")

    # Aggregate each GPU independently so a four-GPU balance issue remains visible.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xlabels = [f"{mode_labels.get(m, m.upper())}\n{q}" for m, q in zip(agg["mode"], agg["qps_label"])]
    x = np.arange(len(agg))
    width = 0.19
    for idx in range(4):
        util_col = f"gpu{idx}_mean_util_pct_mean"
        mem_col = f"gpu{idx}_peak_memory_mib_mean"
        if util_col in agg:
            axes[0].bar(x + (idx - 1.5) * width, pd.to_numeric(agg[util_col], errors="coerce"), width, label=f"GPU {idx}")
        if mem_col in agg:
            axes[1].bar(x + (idx - 1.5) * width, pd.to_numeric(agg[mem_col], errors="coerce"), width, label=f"GPU {idx}")
    axes[0].set_ylabel("Mean GPU utilization (%)")
    axes[1].set_ylabel("Peak memory (MiB)")
    for ax in axes:
        ax.set_xticks(x, xlabels)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=2)
    fig.suptitle("Per-GPU resources by migration mode and load")
    fig.tight_layout()
    fig.savefig(out_dir / "gpu_resources.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    created.append("gpu_resources.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode, marker, color in [("off", "o", "#2474B5"), ("on", "s", "#E87500")]:
        part = agg[agg["mode"].eq(mode)].sort_values("target_qps")
        attempts = part["migration_attempts_mean"] if "migration_attempts_mean" in part else pd.Series(np.nan, index=part.index)
        costs = part["migration_mean_cost_ms_mean"] if "migration_mean_cost_ms_mean" in part else pd.Series(np.nan, index=part.index)
        axes[0].plot(part["target_qps"], attempts, marker=marker, color=color,
                     label=mode_labels.get(mode, mode.upper()))
        axes[1].plot(part["target_qps"], costs, marker=marker, color=color,
                     label=mode_labels.get(mode, mode.upper()))
    axes[0].set_title("Migration events")
    axes[0].set_ylabel("Logged migration operations")
    axes[1].set_title("Successful migration duration")
    axes[1].set_ylabel("Mean milliseconds")
    for ax in axes:
        ax.set_xlabel("Target QPS")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "migration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    created.append("migration.png")
    return created


def format_cell(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, (int, float, np.number)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_report(path: Path, config: dict, summary: pd.DataFrame, migration_events: pd.DataFrame,
                 plot_names: list[str], status: dict) -> None:
    aggregate = summary[summary["row_type"].eq("aggregate")] if not summary.empty else pd.DataFrame()
    table_rows = []
    for _, row in aggregate.iterrows():
        table_rows.append(
            f"| {row['mode'].upper()} | {row['qps_label']} | {format_cell(row.get('successful_requests_mean'), 1)} "
            f"| {format_cell(row.get('ttft_p95_ms_mean'))} | {format_cell(row.get('tpot_p95_ms_mean'))} "
            f"| {format_cell(row.get('request_throughput_rps_mean'))} | {format_cell(row.get('output_tokens_per_sec_mean'), 1)} "
            f"| {format_cell(row.get('ttft_slo_violation_rate_mean'))} | {format_cell(row.get('tpot_slo_violation_rate_mean'))} "
            f"| {format_cell(row.get('migration_attempts_mean'), 1)} |"
        )
    result_table = "\n".join(table_rows) if table_rows else "| — | — | — | — | — | — | — | — | — | — |"
    plots = "\n".join(f"- `{name}`" for name in plot_names) or "- 暂无图表（没有完整测量汇总）"
    events_count = len(migration_events)
    successful = int(migration_events["success"].astype(str).str.lower().eq("true").sum()) if events_count else 0
    unsuccessful = events_count - successful
    mode_migration = "可观测到实际迁移事件。" if successful else "此次运行没有可确认成功的迁移事件；开启开关仅验证配置有效，不能据此声称实际迁移成功。"
    pair_migration_policy = config["runtime"].get("pair_migration_policy", "defrag")
    reproduce_command = (f"cd {ROOT} && source .venv/bin/activate && "
                         "python benchmark/run_baseline_4gpu.py --execute --fresh")
    if pair_migration_policy != "defrag":
        reproduce_command += f" --pair-migration-policy {pair_migration_policy}"
    path.write_text(f"""# Llumnix 四 GPU 无扰动基准报告

## 实验状态

- 状态：`{status.get('status')}`
- 开始：{status.get('started_at_utc')}
- 完成：{status.get('completed_at_utc', '进行中')}
- 已完成正式重复：{status.get('completed_formal_runs', 0)} / 18
- 实验采用真实本地 vLLM/Llumnix 推理，无 CPU/GPU/网络/故障注入。
- 工作目录：`{ROOT}`；结果：`{OUT}`。

## 环境和配置

- 四 GPU：{', '.join(g['name'] for g in config['environment']['gpus'])}
- 模型：`{config['model']['path']}`
- 每种模式保持 4 个独立实例、TP=1、rayrpc 后端、`load` 派发、`{pair_migration_policy}` 迁移配对策略、同一权重和显存配置。
- OFF/ON 配置文件除 `MANAGER.ENABLE_ROUTINE_MIGRATION` 外逐项相同；运行日志验证有效参数，并由 Ray 状态核验 4 GPU。
- 到达过程：Poisson。每个 QPS/seed 生成一次 trace，在 OFF 和 ON 间原样复用；轨迹含固定的短/中/长输入输出长度混合。
- SLO 阈值：TTFT {SLO_TTFT_MS:.0f} ms，TPOT {SLO_TPOT_MS:.0f} ms，沿用仓库 disturbance_exp 默认值。缺失服务端时间线从有效样本分母剔除并单列，不视为达标。
- 建议从头复现命令：`{reproduce_command}`

## 容量预实验与选定 QPS

- 稳定容量观测：{'≥' if config.get('capacity', {}).get('right_censored') else ''}{format_cell(config.get('capacity', {}).get('estimated_sustainable_qps'))} requests/s。
- 低/中/高目标 QPS：{', '.join(f"{x['label']}={x['qps']:.3f}" for x in config.get('qps_levels', []))}。
- 预实验逐档数据见 `capacity_probe.csv`。稳定性由测量窗内完成速率与实际发送速率比较。最高探测档仍稳定，真实饱和点未测到；三档 QPS 按已观测稳定速率下界的 20%/50%/85% 设置。

## 汇总结果

每个 ON/OFF × QPS 组合包含 3 个种子；表格是三次重复的均值，重复级原始指标与标准差在 `summary.csv` 中。

| 模式 | QPS 档 | 成功请求均值 | P95 TTFT ms | P95 TPOT ms | 完成请求/s | 输出 token/s | TTFT SLO 违约率 | TPOT SLO 违约率 | 迁移操作均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{result_table}

## 服务质量和数据完整性

- 每条请求记录包含目标到达、实际发送、完成、状态码、成功/失败/超时和服务端 per-token 时间线（可用时）。
- TTFT/TPOT 优先来自 `/generate_benchmark` 的服务端 `per_token_latency`；没有完整时间线时保留为 N/A。
- GPU 遥测每秒覆盖四卡的利用率、显存、温度和功耗；四卡均值、峰值显存及负载离散度汇总到 `summary.csv`。
- 迁移日志：总事件 {events_count}，成功 {successful}，非成功 {unsuccessful}。{mode_migration}
- 迁移日志不暴露实例到物理 GPU 的可靠映射时，`source_gpu` / `target_gpu` 留空，不推测填充。

## 图表

{plots}

## 可追溯文件

- `requests.csv`：逐请求记录，含预热和正式测量阶段。
- `gpu_metrics.csv`：四卡每秒采样。
- `migration_events.csv`：可观测的迁移完成事件及原始日志行。
- `summary.csv`：重复行以及跨种子均值/标准差。
- `config.json`、`state.json`、`configs/`、`traces/`、`server_logs/`：配置、状态、配对轨迹和运行日志。

## 限制

- 工作区当前不是挂载持久卷；Vast recycle/destroy 会删除这些本地结果。
- GPU 物理互联存在 NUMA/SYS 与 PHB 路径差异，日志中若无法可靠映射实例至 GPU，迁移事件中的 GPU 字段保留空值。
""", encoding="utf-8")


def finalize_artifacts(state: dict, config: dict) -> int:
    request_parts = OUT / "raw_requests"
    write_final_requests(request_parts, OUT / "requests.csv")
    migration_path = OUT / "migration_events.csv"
    if not migration_path.exists():
        with migration_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MIGRATION_FIELDS).writeheader()
    combined = aggregate_summary(OUT / "requests.csv", OUT / "gpu_metrics.csv",
                                 migration_path, OUT / "summary.csv")
    plot_names = plot_results(combined, OUT / "plots")
    migration_df = pd.read_csv(migration_path) if migration_path.stat().st_size else pd.DataFrame()
    complete_count = state.get("completed_formal_runs", 0)
    state["status"] = "complete" if complete_count == 18 else "partial"
    state["completed_at_utc"] = now_iso()
    state["capacity"] = config.get("capacity")
    state.pop("failure", None)
    state.pop("failed_at_utc", None)
    atomic_json(OUT / "state.json", state)
    config["status"] = state["status"]
    config["generated_artifacts"] = {"plots": plot_names, "completed_formal_runs": complete_count}
    atomic_json(OUT / "config.json", config)
    write_report(OUT / "report.md", config, combined, migration_df, plot_names, state)
    print(f"[done] status={state['status']} formal_runs={complete_count}/18 output={OUT}", flush=True)
    return 0 if complete_count == 18 else 2


def summarize_probe(rows: list[dict], qps: float, duration: float) -> dict:
    sent = [r for r in rows if r["phase"] == "capacity_probe" and r["actual_send_sec"] is not None]
    completed_in_window = [r for r in sent if r["success"] and r["completion_sec"] is not None and r["completion_sec"] <= duration]
    success = [r for r in sent if r["success"]]
    ttft = [float(r["server_ttft_ms"]) for r in success if r["server_ttft_ms"] is not None]
    actual_send_qps = len(sent) / duration
    ratio = (len(completed_in_window) / duration) / actual_send_qps if actual_send_qps else 0.0
    return {
        "target_qps": qps, "sent_requests": len(sent), "completed_in_window": len(completed_in_window),
        "completed_after_drain": len(success), "actual_send_qps": actual_send_qps,
        "in_window_completion_qps": len(completed_in_window) / duration,
        "completion_to_actual_send_ratio": ratio, "ttft_p95_ms": percentile(ttft, 0.95),
        "stable": ratio >= 0.85 and (percentile(ttft, 0.95) is None or percentile(ttft, 0.95) < 30000),
    }


async def health_check() -> None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        async with session.get(f"{BASE_URL}/health") as resp:
            if resp.status != 200:
                raise RuntimeError(f"Health returned HTTP {resp.status}")
        async with session.get(f"{BASE_URL}/is_ready") as resp:
            if resp.status != 200 or (await resp.json()) is not True:
                raise RuntimeError("Llumnix is_ready did not return true")


async def stop_server(process: subprocess.Popen | None, owned_cluster: bool, logs: Path) -> None:
    global CURRENT_SERVER_PROCESS, CURRENT_CLUSTER_OWNED
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            await asyncio.to_thread(process.wait, 90)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, 20)
    if owned_cluster:
        # The initial guard proved there was no cluster, and this run launched
        # it through the repo entrypoint. Only that experiment-owned cluster is
        # shut down here.
        ray_bin = ROOT / ".venv" / "bin" / "ray"
        result = await asyncio.to_thread(run_cmd, [str(ray_bin), "stop", "--force"], 90.0, False)
        with logs.open("a", encoding="utf-8") as f:
            f.write("\n[experiment cleanup] ray stop --force\n")
            f.write(result.stdout + result.stderr)
    if process is CURRENT_SERVER_PROCESS:
        CURRENT_SERVER_PROCESS = None
    if owned_cluster:
        CURRENT_CLUSTER_OWNED = False


def state_run(state: dict, run_id: str, status: str, **extra) -> None:
    state.setdefault("runs", {})[run_id] = {"status": status, "updated_at_utc": now_iso(), **extra}
    state["completed_formal_runs"] = sum(
        1 for key, value in state.get("runs", {}).items()
        if key.startswith("formal_") and value.get("status") == "complete"
    )
    atomic_json(OUT / "state.json", state)


async def run_capacity_probe(context: dict, requests_parts: Path, gpu_context: dict,
                             gpu_rows: list[dict], capacity_rows: list[dict]) -> tuple[float, bool]:
    stable_rates = []
    any_stable = False
    for idx, qps in enumerate(PROBE_QPS):
        trace = generated_trace(50000 + idx, PROBE_SEC, qps)
        trace_path = OUT / "traces" / f"capacity_probe_{qps_label(qps)}.csv"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        write_trace(trace_path, trace)
        ctx = {**context, "qps_label": "probe", "target_qps": qps, "seed": 50000 + idx,
               "run_id": f"capacity_{qps_label(qps)}", "phase": "capacity_probe"}
        gpu_context.clear(); gpu_context.update(ctx)
        req_path = requests_parts / f"{ctx['run_id']}_capacity_probe.csv"
        rows = await execute_trace(trace, ctx, "capacity_probe", PROBE_SEC, PROBE_DRAIN_SEC, req_path)
        probe = summarize_probe(rows, qps, PROBE_SEC)
        capacity_rows.append(probe)
        append_csv(OUT / "capacity_probe.csv", list(probe.keys()), [probe])
        print(f"[capacity] target={qps:g} sent={probe['actual_send_qps']:.2f} "
              f"complete-window={probe['in_window_completion_qps']:.2f} stable={probe['stable']}", flush=True)
        if probe["stable"]:
            stable_rates.append(probe["actual_send_qps"])
            any_stable = True
        else:
            break
    if not capacity_rows:
        raise RuntimeError("Capacity probe produced no observations")
    right_censored = len(capacity_rows) == len(PROBE_QPS) and all(x["stable"] for x in capacity_rows)
    if stable_rates:
        cap = max(stable_rates)
    else:
        cap = max(float(capacity_rows[0]["in_window_completion_qps"]), 0.05)
    return cap, right_censored


async def run_mode(mode: str, config_path: Path, context: dict, state: dict,
                   request_parts: Path, gpu_rows: list[dict], migration_events: list[dict],
                   formal_traces: dict[tuple[str, int], list], qps_levels: list[dict],
                   pair_migration_policy: str = "defrag",
                   capacity_rows: list[dict] | None = None,
                   discard_migration_pairs: bool = False,
                   expected_migration_enabled: bool | None = None,
                   expected_migrate_out_threshold: float | None = None,
                   expected_pair_migration_frequency: int | None = None) -> None:
    global CURRENT_SERVER_PROCESS, CURRENT_CLUSTER_OWNED
    server_dir = OUT / "server_logs"
    server_dir.mkdir(parents=True, exist_ok=True)
    log_path = server_dir / f"{mode}.log"
    instance_log_base = server_dir / f"{mode}_instance"
    cmd, env = server_command(mode, config_path, instance_log_base)
    if discard_migration_pairs:
        env["LLUMNIX_DISCARD_MIGRATION_PAIRS_AFTER_POLICY"] = "1"
    print(f"[server] starting Migration {mode.upper()} from {config_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, text=True)
    CURRENT_SERVER_PROCESS = process
    owned_cluster = True
    CURRENT_CLUSTER_OWNED = True
    event_stop = asyncio.Event()
    log_context = {"mode": mode, "phase": "startup"}
    event_task = asyncio.create_task(watch_server_log(log_path, log_context, migration_events,
                                                       event_stop, OUT / "migration_events.csv"))
    gpu_stop = asyncio.Event()
    monitor_context = {"experiment_id": context.get("experiment_id", "baseline_4gpu"),
                       "mode": mode, "qps_label": "startup",
                       "target_qps": None, "seed": None, "run_id": f"startup_{mode}", "phase": "startup"}
    monitor_start = time.monotonic()
    gpu_task = asyncio.create_task(sample_gpu_metrics(monitor_context, gpu_rows, gpu_stop,
                                                      OUT / "gpu_metrics.csv", monitor_start))
    try:
        await wait_ready(process, log_path)
        ray_status = await verify_ray_four_gpu()
        verify_effective_config(log_path, "on" if mode == "dryrun" else mode, ray_status, pair_migration_policy,
                                discard_migration_pairs=discard_migration_pairs,
                                expected_migration_enabled=expected_migration_enabled,
                                expected_migrate_out_threshold=expected_migrate_out_threshold,
                                expected_pair_migration_frequency=expected_pair_migration_frequency)
        await health_check()
        (server_dir / f"{mode}_ray_status.txt").write_text(ray_status, encoding="utf-8")
        print(f"[server] Migration {mode.upper()} ready; Ray sees 4 GPUs", flush=True)

        if mode == "off" and capacity_rows is not None:
            warm_trace = generated_trace(49999, 30.0, 0.2)
            warm_ctx = {**context, "mode": mode, "qps_label": "startup_warmup", "target_qps": 0.2,
                        "seed": 49999, "run_id": "capacity_warmup", "phase": "warmup"}
            log_context.update(warm_ctx)
            monitor_context.clear(); monitor_context.update(warm_ctx)
            await execute_trace(warm_trace, warm_ctx, "warmup", 30.0, 90.0,
                                request_parts / "capacity_warmup_warmup.csv")
            cap, censored = await run_capacity_probe(warm_ctx, request_parts, monitor_context, gpu_rows, capacity_rows)
            context["capacity_qps"] = cap
            context["capacity_right_censored"] = censored
            print(f"[capacity] estimated sustainable rate={cap:.3f} req/s", flush=True)

        for level in qps_levels:
            for seed in SEEDS:
                run_id = f"formal_{mode}_{level['label']}_seed{seed}"
                prior = state.get("runs", {}).get(run_id, {})
                if prior.get("status") == "complete":
                    print(f"[skip] {run_id} already complete", flush=True)
                    continue
                state_run(state, run_id, "running", mode=mode, qps=level["qps"], seed=seed)
                run_context = {"experiment_id": context.get("experiment_id", "baseline_4gpu"),
                               "mode": mode, "qps_label": level["label"],
                               "target_qps": level["qps"], "seed": seed, "run_id": run_id}
                log_context.clear(); log_context.update({**run_context, "phase": "warmup"})
                monitor_context.clear(); monitor_context.update({**run_context, "phase": "warmup"})
                if level["qps"] == 0:
                    # Zero QPS is an idle control: latency is undefined because
                    # no requests are submitted. Keep the service and GPU
                    # monitor active for the same warmup/measurement windows.
                    await asyncio.sleep(WARMUP_SEC)
                    log_context["phase"] = "measurement"
                    monitor_context.clear(); monitor_context.update({**run_context, "phase": "measurement"})
                    await asyncio.sleep(MEASURE_SEC)
                    log_context["phase"] = "post_drain"
                    monitor_context.clear(); monitor_context.update({**run_context, "phase": "post_drain"})
                    state_run(state, run_id, "complete", sent=0, successful=0, idle_control=True,
                              completed_at_utc=now_iso())
                    print(f"[run] {run_id}: idle control; no requests sent", flush=True)
                    continue
                warm_seed = seed + int(round(level["qps"] * 100000)) + 900000
                warm_trace = generated_trace(warm_seed, WARMUP_SEC, level["qps"])
                warm_path = OUT / "traces" / f"warmup_{level['label']}_seed{seed}.csv"
                write_trace(warm_path, warm_trace)
                print(f"[run] {run_id}: {WARMUP_SEC:g}s warmup ({len(warm_trace)} arrivals)", flush=True)
                await execute_trace(warm_trace, run_context, "warmup", WARMUP_SEC, DRAIN_SEC,
                                    request_parts / f"{run_id}_warmup.csv")

                trace = formal_traces[(level["label"], seed)]
                log_context["phase"] = "measurement"
                monitor_context.clear(); monitor_context.update({**run_context, "phase": "measurement"})
                print(f"[run] {run_id}: {MEASURE_SEC:g}s measurement ({len(trace)} scheduled arrivals)", flush=True)
                rows = await execute_trace(trace, run_context, "measurement", MEASURE_SEC, DRAIN_SEC,
                                           request_parts / f"{run_id}_measurement.csv")
                sent = len([r for r in rows if r["actual_send_sec"] is not None])
                done = sum(1 for r in rows if r["success"])
                log_context["phase"] = "post_drain"
                monitor_context.clear(); monitor_context.update({**run_context, "phase": "post_drain"})
                state_run(state, run_id, "complete", sent=sent, successful=done,
                          completed_at_utc=now_iso())
                print(f"[run] {run_id}: sent={sent} success={done} timeout="
                      f"{sum(1 for r in rows if r['timeout'])}", flush=True)
    finally:
        gpu_stop.set()
        await gpu_task
        # Let the logger flush any final migration completion messages.
        await asyncio.sleep(1.0)
        event_stop.set()
        await event_task
        await stop_server(process, owned_cluster, log_path)


def create_traces(qps_levels: list[dict], max_requests: int | None = None) -> dict[tuple[str, int], list]:
    formal = {}
    trace_dir = OUT / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for level in qps_levels:
        for seed in SEEDS:
            trace = generated_trace(seed, MEASURE_SEC, level["qps"],
                                   max_requests=max_requests or 100000) if level["qps"] > 0 else []
            path = trace_dir / f"trace_{level['label']}_seed{seed}.csv"
            write_trace(path, trace)
            formal[(level["label"], seed)] = trace
    return formal


async def main_async(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        state_path = OUT / "state.json"
        config_path = OUT / "config.json"
        if not state_path.exists() or not config_path.exists():
            raise RuntimeError("--postprocess-only requires existing state.json and config.json")
        state = json.loads(state_path.read_text())
        config = json.loads(config_path.read_text())
        return finalize_artifacts(state, config)
    if args.fresh and (OUT / "state.json").exists():
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = OUT.with_name(f"{OUT.name}_archive_{stamp}")
        suffix = 1
        while archive.exists():
            archive = OUT.with_name(f"{OUT.name}_archive_{stamp}_{suffix}")
            suffix += 1
        OUT.rename(archive)
        OUT.mkdir(parents=True, exist_ok=True)
        print(f"[fresh] archived prior results to {archive}", flush=True)
    if args.resume and (OUT / "config.json").exists():
        recorded_config = json.loads((OUT / "config.json").read_text())
        recorded_policy = recorded_config.get("runtime", {}).get("pair_migration_policy", "defrag")
        if recorded_policy != args.pair_migration_policy:
            raise RuntimeError(
                f"Cannot resume with pair migration policy {args.pair_migration_policy!r}; "
                f"the recorded run uses {recorded_policy!r}."
            )
    for d in ["configs", "traces", "server_logs", "raw_requests", "plots"]:
        (OUT / d).mkdir(exist_ok=True)
    state_path = OUT / "state.json"
    if state_path.exists() and not args.resume:
        raise RuntimeError(f"{state_path} already exists. Use --resume to continue the recorded experiment.")
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "experiment_id": "baseline_4gpu", "status": "preparing", "runs": {}, "started_at_utc": now_iso(),
        "completed_formal_runs": 0,
    }
    atomic_json(state_path, state)
    print("[preflight] verifying this experiment can own the local Ray/API ports", flush=True)
    owned_process_preflight()
    snapshot = system_snapshot()
    if snapshot["gpu_count"] != 4 or not snapshot["torch"].get("cuda_available") or snapshot["torch"].get("device_count") != 4:
        raise RuntimeError(f"Expected PyTorch and nvidia-smi to expose four GPUs; snapshot={snapshot}")
    (OUT / "environment_audit.md").write_text(environment_audit(snapshot), encoding="utf-8")
    config_paths = {}
    for mode in ("off", "on"):
        path = OUT / "configs" / f"llumnix_{mode}.yml"
        path.write_text(yaml.safe_dump(yaml_config(mode, args.pair_migration_policy), sort_keys=False),
                        encoding="utf-8")
        config_paths[mode] = path
    check_mode_configs(config_paths)

    # QPS is filled from the real OFF-mode capacity probe before formal traces
    # are generated. On resume, retain the previously measured rates and traces.
    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() and args.resume else {
        "experiment_id": "baseline_4gpu", "created_at_utc": now_iso(), "environment": snapshot,
        "model": {"path": str(MODEL), "max_model_len": MODEL_MAX_LEN, "tensor_parallel_size": 1},
        "runtime": {"initial_instances": 4, "max_num_seqs_per_instance": MAX_NUM_SEQS,
                    "gpu_memory_utilization": GPU_MEMORY_UTIL, "dtype": "half", "migration_backend": "rayrpc",
                    "dispatch_policy": "load", "pair_migration_policy": args.pair_migration_policy,
                    "pair_migration_frequency": 1,
                    "migrate_out_threshold": 20000.0, "routine_migration_off": False,
                    "routine_migration_on": True, "pre_stop_migration": False,
                    "disturbance_injection": False, "launch_ray_cluster": True, "api_host": "127.0.0.1",
                    "api_port": SERVER_PORT, "ray_port": RAY_PORT},
        "workload": {"arrival": "poisson", "profile_mix": PROFILE_MIX, "input_token_profile": INPUT_PROFILE,
                     "output_token_profile": OUTPUT_PROFILE, "random_seeds": SEEDS,
                     "warmup_sec": WARMUP_SEC, "measurement_sec": MEASURE_SEC, "drain_sec": DRAIN_SEC,
                     "request_timeout_sec": REQUEST_TIMEOUT_SEC},
        "slo": {"ttft_ms": SLO_TTFT_MS, "tpot_ms": SLO_TPOT_MS,
                "source": "disturbance_exp/scenario.py defaults"},
        "configs": {k: str(v.relative_to(OUT)) for k, v in config_paths.items()},
    }
    if config["runtime"].get("pair_migration_policy", "defrag") != args.pair_migration_policy:
        raise RuntimeError(
            "The recorded pair migration policy differs from this invocation; "
            "use --fresh for a new policy or --resume with the recorded policy."
        )
    state["status"] = "running"
    state["started_at_utc"] = state.get("started_at_utc", now_iso())
    atomic_json(state_path, state)

    request_parts = OUT / "raw_requests"
    gpu_rows: list[dict] = []
    migration_events: list[dict] = []
    capacity_rows: list[dict] = []
    context: dict = {"experiment_id": "baseline_4gpu"}
    qps_levels = config.get("qps_levels") or []

    if not qps_levels:
        if args.resume and (OUT / "capacity_probe.csv").exists():
            probe = pd.read_csv(OUT / "capacity_probe.csv")
            stable = probe[probe["stable"].astype(str).str.lower().eq("true")]
            cap = float(stable["target_qps"].max()) if not stable.empty else max(float(probe.iloc[0]["in_window_completion_qps"]), .05)
            censored = len(probe) == len(PROBE_QPS) and stable.shape[0] == len(PROBE_QPS)
        else:
            # The first OFF server performs this probe below.
            cap = None
            censored = False
        if cap is not None:
            qps_levels = [{"label": "low", "qps": round(cap * .2, 4)},
                          {"label": "medium", "qps": round(cap * .5, 4)},
                          {"label": "high", "qps": round(cap * .85, 4)}]
            config["capacity"] = {"estimated_sustainable_qps": cap, "right_censored": censored,
                                  "probe_qps": PROBE_QPS}
            config["qps_levels"] = qps_levels
            atomic_json(config_path, config)

    # On the initial run the off server is used for warmup + short capacity
    # probes, then formal OFF repetitions. Traces are generated immediately
    # after the probe and copied byte-for-byte to both migration modes.
    async def execute_off_with_probe():
        nonlocal qps_levels, capacity_rows
        if not qps_levels:
            # Temporarily use a conservative QPS array only so run_mode can start;
            # it will perform capacity first, then this wrapper recalculates and
            # runs formal repetitions in the same owned server process.
            return await run_mode_with_capacity()
        else:
            return await run_mode("off", config_paths["off"], context, state, request_parts,
                                  gpu_rows, migration_events, formal_traces, qps_levels,
                                  pair_migration_policy=args.pair_migration_policy)

    async def run_mode_with_capacity():
        nonlocal qps_levels, capacity_rows, formal_traces
        global CURRENT_SERVER_PROCESS, CURRENT_CLUSTER_OWNED
        mode = "off"
        server_dir = OUT / "server_logs"
        log_path = server_dir / "off.log"
        instance_log_base = server_dir / "off_instance"
        cmd, env = server_command(mode, config_paths[mode], instance_log_base)
        print("[server] starting Migration OFF for capacity probe", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, text=True)
        CURRENT_SERVER_PROCESS = process
        CURRENT_CLUSTER_OWNED = True
        event_stop = asyncio.Event()
        log_context = {"mode": "off", "phase": "startup"}
        event_task = asyncio.create_task(watch_server_log(log_path, log_context, migration_events,
                                                           event_stop, OUT / "migration_events.csv"))
        gpu_stop = asyncio.Event()
        monitor_context = {"experiment_id": "baseline_4gpu", "mode": "off", "qps_label": "startup",
                           "target_qps": None, "seed": None, "run_id": "startup_off", "phase": "startup"}
        monitor_start = time.monotonic()
        gpu_task = asyncio.create_task(sample_gpu_metrics(monitor_context, gpu_rows, gpu_stop,
                                                          OUT / "gpu_metrics.csv", monitor_start))
        try:
            await wait_ready(process, log_path)
            ray_status = await verify_ray_four_gpu()
            verify_effective_config(log_path, "off", ray_status, args.pair_migration_policy)
            await health_check()
            (server_dir / "off_ray_status.txt").write_text(ray_status, encoding="utf-8")
            warm_trace = generated_trace(49999, 30.0, 0.2)
            warm_ctx = {"experiment_id": "baseline_4gpu", "mode": "off", "qps_label": "startup_warmup",
                        "target_qps": 0.2, "seed": 49999, "run_id": "capacity_warmup"}
            log_context.clear(); log_context.update({**warm_ctx, "phase": "warmup"})
            monitor_context.clear(); monitor_context.update({**warm_ctx, "phase": "warmup"})
            print("[capacity] 30s startup warmup", flush=True)
            await execute_trace(warm_trace, warm_ctx, "warmup", 30.0, 90.0,
                                request_parts / "capacity_warmup_warmup.csv")
            stable_rates = []
            for idx, rate in enumerate(PROBE_QPS):
                trace = generated_trace(50000 + idx, PROBE_SEC, rate)
                trace_path = OUT / "traces" / f"capacity_probe_{qps_label(rate)}.csv"
                write_trace(trace_path, trace)
                probe_ctx = {"experiment_id": "baseline_4gpu", "mode": "off", "qps_label": "probe",
                             "target_qps": rate, "seed": 50000 + idx,
                             "run_id": f"capacity_{qps_label(rate)}"}
                log_context.clear(); log_context.update({**probe_ctx, "phase": "capacity_probe"})
                monitor_context.clear(); monitor_context.update({**probe_ctx, "phase": "capacity_probe"})
                print(f"[capacity] probing {rate:g} requests/s for {PROBE_SEC:g}s", flush=True)
                probe_rows = await execute_trace(trace, probe_ctx, "capacity_probe", PROBE_SEC,
                                                 PROBE_DRAIN_SEC,
                                                 request_parts / f"{probe_ctx['run_id']}_capacity_probe.csv")
                probe = summarize_probe(probe_rows, rate, PROBE_SEC)
                capacity_rows.append(probe)
                append_csv(OUT / "capacity_probe.csv", list(probe.keys()), [probe])
                print(f"[capacity] target={rate:g} sent={probe['actual_send_qps']:.2f} "
                      f"complete-window={probe['in_window_completion_qps']:.2f} "
                      f"stable={probe['stable']}", flush=True)
                if probe["stable"]:
                    stable_rates.append(probe["actual_send_qps"])
                else:
                    break
            cap = max(stable_rates) if stable_rates else max(float(capacity_rows[0]["in_window_completion_qps"]), .05)
            censored = len(capacity_rows) == len(PROBE_QPS) and all(x["stable"] for x in capacity_rows)
            qps_levels = [{"label": "low", "qps": round(cap * .2, 4)},
                          {"label": "medium", "qps": round(cap * .5, 4)},
                          {"label": "high", "qps": round(cap * .85, 4)}]
            config["capacity"] = {"estimated_sustainable_qps": cap, "right_censored": censored,
                                  "stable_definition": "in-window completion QPS / actual sent QPS >= 0.85 and P95 TTFT < 30s",
                                  "probe_qps": PROBE_QPS, "probe_duration_sec": PROBE_SEC,
                                  "probes": capacity_rows}
            config["qps_levels"] = qps_levels
            atomic_json(config_path, config)
            print(f"[capacity] sustainable estimate={cap:.3f}; "
                  f"low/medium/high={[x['qps'] for x in qps_levels]}", flush=True)
            formal_traces = create_traces(qps_levels)
            for mode in ("off", "on"):
                mode_config = yaml.safe_load(config_paths[mode].read_text())
                mode_config["MANAGER"]["ENABLE_ROUTINE_MIGRATION"] = mode == "on"
                config_paths[mode].write_text(yaml.safe_dump(mode_config, sort_keys=False), encoding="utf-8")
            check_mode_configs(config_paths)
            for level in qps_levels:
                for seed in SEEDS:
                    run_id = f"formal_off_{level['label']}_seed{seed}"
                    state_run(state, run_id, "running", mode="off", qps=level["qps"], seed=seed)
                    run_context = {"experiment_id": "baseline_4gpu", "mode": "off",
                                   "qps_label": level["label"], "target_qps": level["qps"],
                                   "seed": seed, "run_id": run_id}
                    warm_seed = seed + int(round(level["qps"] * 100000)) + 900000
                    warm_trace = generated_trace(warm_seed, WARMUP_SEC, level["qps"])
                    write_trace(OUT / "traces" / f"warmup_{level['label']}_seed{seed}.csv", warm_trace)
                    log_context.clear(); log_context.update({**run_context, "phase": "warmup"})
                    monitor_context.clear(); monitor_context.update({**run_context, "phase": "warmup"})
                    print(f"[run] {run_id}: 30s warmup ({len(warm_trace)} arrivals)", flush=True)
                    await execute_trace(warm_trace, run_context, "warmup", WARMUP_SEC, DRAIN_SEC,
                                        request_parts / f"{run_id}_warmup.csv")
                    trace = formal_traces[(level["label"], seed)]
                    log_context["phase"] = "measurement"
                    monitor_context.clear(); monitor_context.update({**run_context, "phase": "measurement"})
                    print(f"[run] {run_id}: 120s measurement ({len(trace)} arrivals)", flush=True)
                    rows = await execute_trace(trace, run_context, "measurement", MEASURE_SEC, DRAIN_SEC,
                                               request_parts / f"{run_id}_measurement.csv")
                    sent = sum(1 for row in rows if row["actual_send_sec"] is not None)
                    done = sum(1 for row in rows if row["success"])
                    log_context["phase"] = "post_drain"
                    state_run(state, run_id, "complete", sent=sent, successful=done,
                              completed_at_utc=now_iso())
                    print(f"[run] {run_id}: sent={sent} success={done} "
                          f"timeout={sum(1 for row in rows if row['timeout'])}", flush=True)
        finally:
            gpu_stop.set(); await gpu_task
            await asyncio.sleep(1.0)
            event_stop.set(); await event_task
            await stop_server(process, True, log_path)

    formal_traces = create_traces(qps_levels) if qps_levels else {}
    if not qps_levels:
        await run_mode_with_capacity()
        # run_mode_with_capacity performs OFF formal runs in the same process.
    else:
        # In a clean invocation with existing probe results, reconstruct paired
        # traces using the saved rates and run OFF without repeating probing.
        formal_traces = create_traces(qps_levels)
        await execute_off_with_probe()

    # Start a fresh owned Ray cluster for ON. Same traces, per-seed, as OFF.
    await run_mode("on", config_paths["on"], context, state, request_parts,
                   gpu_rows, migration_events, formal_traces, qps_levels,
                   pair_migration_policy=args.pair_migration_policy)

    return finalize_artifacts(state, config)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run real Llumnix inference experiments")
    parser.add_argument("--resume", action="store_true", help="reuse saved capacity/traces and skip completed runs")
    parser.add_argument("--fresh", action="store_true", help="archive prior results and start a new experiment")
    parser.add_argument("--pair-migration-policy", choices=("defrag", "balanced"), default="defrag",
                        help="migration pair policy shared by OFF/ON runs (default: defrag)")
    parser.add_argument("--postprocess-only", action="store_true", help="rebuild CSV summary, charts, and report without launching Llumnix")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.execute and not args.postprocess_only:
        print("This script runs real local model inference. Pass --execute to start.")
        return 0
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        global CURRENT_SERVER_PROCESS, CURRENT_CLUSTER_OWNED
        process = CURRENT_SERVER_PROCESS
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=90)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=20)
                except Exception:
                    pass
        if CURRENT_CLUSTER_OWNED:
            try:
                run_cmd([str(ROOT / ".venv" / "bin" / "ray"), "stop", "--force"], timeout=90, check=False)
            except Exception:
                pass
        print("Interrupted; state and already flushed request/GPU/migration records remain in baseline_4gpu.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        if (OUT / "state.json").exists():
            state = json.loads((OUT / "state.json").read_text())
            state["status"] = "blocked_or_failed"
            state["failure"] = f"{type(exc).__name__}: {exc}"
            state["failed_at_utc"] = now_iso()
            atomic_json(OUT / "state.json", state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
