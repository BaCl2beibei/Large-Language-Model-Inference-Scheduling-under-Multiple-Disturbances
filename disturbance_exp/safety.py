"""Read-only preflight checks and safety gates."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp


async def _get(base_url: str, endpoint: str) -> dict:
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(base_url.rstrip("/") + endpoint) as response:
                body = await response.text()
                return {"ok": response.status == 200, "status_code": response.status,
                        "body": body[:200]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def preflight(cfg: dict) -> dict:
    started = time.time()
    service = cfg["service"]
    health, ready = await asyncio.gather(
        _get(service["base_url"], service["health_endpoint"]),
        _get(service["base_url"], service["ready_endpoint"]),
    )
    gpu = {"available": False, "free_mib": None}
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu = {"available": True, "name": pynvml.nvmlDeviceGetName(handle),
               "uuid": pynvml.nvmlDeviceGetUUID(handle),
               "total_mib": memory.total // (1024 * 1024),
               "used_mib": memory.used // (1024 * 1024),
               "free_mib": memory.free // (1024 * 1024)}
    except Exception as exc:
        gpu["error"] = type(exc).__name__
    cap_eff = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("CapEff:"):
                cap_eff = int(line.split()[1], 16)
                break
    except OSError:
        pass
    return {
        "timestamp_unix": started,
        "service_health": health,
        "service_ready": ready,
        "service_ok": health.get("ok") and ready.get("ok") and ready.get("body", "").strip().lower() == "true",
        "gpu": gpu,
        "cpu_count": __import__("os").cpu_count(),
        "network_admin_capability": bool(cap_eff is not None and (cap_eff & (1 << 12))),
        "tools": {name: shutil.which(name) for name in ("tc", "ip", "stress-ng", "nvidia-smi")},
        "modules": {name: importlib.util.find_spec(name) is not None
                    for name in ("aiohttp", "yaml", "psutil", "pynvml", "pytest", "torch", "transformers")},
    }


def doctor_sync(cfg: dict) -> dict:
    return asyncio.run(preflight(cfg))


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
    except Exception:
        return None


def model_name(cfg: dict) -> str | None:
    tokenizer = cfg["workload"].get("tokenizer")
    return str(tokenizer) if tokenizer else None
