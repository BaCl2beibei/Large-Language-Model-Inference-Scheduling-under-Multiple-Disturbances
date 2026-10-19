"""Request metrics, host telemetry, and summary calculations."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


REQUEST_FIELDS = [
    "run_id", "request_id", "request_type", "scheduled_arrival_sec", "actual_send_sec",
    "completion_sec", "input_tokens", "output_tokens", "total_latency_ms", "server_ttft_ms",
    "server_tpot_ms", "client_ttft_ms", "client_tpot_ms", "status_code", "success", "timeout",
    "error_type", "error_message",
]
CPU_FIELDS = [
    "wall_time", "relative_sec", "cpu_percent", "memory_percent", "memory_available_mib",
    "gpu_uuid", "gpu_utilization_percent", "gpu_memory_total_mib", "gpu_memory_used_mib",
    "gpu_memory_free_mib", "gpu_temperature_c", "gpu_power_w", "gpu_error",
    "cpu_load_active", "cpu_load_workers", "gpu_load_active", "gpu_load_pid",
    "network_proxy_active", "api_mock_fault_active",
]


@dataclass
class RequestResult:
    run_id: str
    request_id: str
    request_type: str
    scheduled_arrival_sec: float
    actual_send_sec: float | None = None
    completion_sec: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_latency_ms: float | None = None
    server_ttft_ms: float | None = None
    server_tpot_ms: float | None = None
    client_ttft_ms: float | None = None
    client_tpot_ms: float | None = None
    status_code: int | None = None
    success: bool = False
    timeout: bool = False
    error_type: str = ""
    error_message: str = ""


def parse_benchmark_payload(payload: dict) -> dict:
    """Extract server-side timing only when the token timeline matches output count."""
    raw = payload.get("per_token_latency")
    token_count = payload.get("num_output_tokens_cf")
    try:
        token_count = int(token_count)
    except (TypeError, ValueError):
        token_count = None
    values = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    values.append(float(item[1]))
                except (TypeError, ValueError):
                    pass
    ttft = values[0] if values else None
    tpot = None
    if token_count and token_count > 1 and len(values) == token_count:
        intervals = values[1:]
        tpot = sum(intervals) / len(intervals) if intervals else None
    return {"output_tokens": token_count, "server_ttft_ms": ttft, "server_tpot_ms": tpot,
            "timeline_count": len(values), "timeline_matches_output": token_count == len(values) if token_count is not None else False}


def percentile(values: list[float], p: float) -> float | None:
    clean = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * p
    low, high = int(math.floor(rank)), int(math.ceil(rank))
    return clean[low] + (clean[high] - clean[low]) * (rank - low)


def _nvml_sample() -> dict:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        values = {
            "gpu_uuid": pynvml.nvmlDeviceGetUUID(handle),
            "gpu_utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "gpu_memory_total_mib": memory.total // (1024 * 1024),
            "gpu_memory_used_mib": memory.used // (1024 * 1024),
            "gpu_memory_free_mib": memory.free // (1024 * 1024),
        }
        try:
            values["gpu_temperature_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            values["gpu_temperature_c"] = None
        try:
            values["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            values["gpu_power_w"] = None
        return values
    except Exception as exc:
        if shutil.which("nvidia-smi"):
            try:
                output = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.total,memory.used,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ], text=True, timeout=2).strip().splitlines()[0].split(",")
                uuid, util, total, used, temp, power = [x.strip() for x in output]
                return {"gpu_uuid": uuid, "gpu_utilization_percent": float(util),
                        "gpu_memory_total_mib": float(total), "gpu_memory_used_mib": float(used),
                        "gpu_memory_free_mib": float(total) - float(used),
                        "gpu_temperature_c": float(temp), "gpu_power_w": float(power),
                        "gpu_error": "NVML fallback via nvidia-smi"}
            except Exception:
                pass
        return {"gpu_uuid": None, "gpu_utilization_percent": None, "gpu_memory_total_mib": None,
                "gpu_memory_used_mib": None, "gpu_memory_free_mib": None,
                "gpu_temperature_c": None, "gpu_power_w": None,
                "gpu_error": f"GPU metrics unavailable: {type(exc).__name__}"}


def sample_host(relative_sec: float, wall_time: float | None = None, state: dict | None = None) -> dict:
    memory = psutil.virtual_memory()
    return {
        "wall_time": wall_time or time.time(),
        "relative_sec": relative_sec,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": memory.percent,
        "memory_available_mib": memory.available // (1024 * 1024),
        **_nvml_sample(),
        **(state or {}),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


async def collect_host_metrics(path: Path, start_monotonic: float, interval_sec: float,
                               stop_event: asyncio.Event, rows: list[dict], state_provider=None) -> None:
    while not stop_event.is_set():
        state = state_provider() if state_provider else None
        row = sample_host(asyncio.get_running_loop().time() - start_monotonic, state=state)
        rows.append(row)
        await asyncio.to_thread(_append_csv_row, path, CPU_FIELDS, row)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            pass


def _append_csv_row(path: Path, fields: list[str], row: dict):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
