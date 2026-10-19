"""Bounded experiment lifecycle, event scheduling, and automatic recovery."""

from __future__ import annotations

import asyncio
import csv
import importlib.metadata
import json
import re
import signal
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import yaml

from . import __version__
from .analyzer import plot_results, summarize, write_report
from .injectors import CpuLoadInjector, GpuLoadInjector, LocalHttpProxy, MockFaultInjector
from .metrics import CPU_FIELDS, REQUEST_FIELDS, RequestResult, collect_host_metrics, parse_benchmark_payload, write_csv, write_summary
from .safety import git_commit, model_name, preflight
from .scenario import ScenarioError, load_scenario
from .workload import TraceRequest, generate_trace, load_local_tokenizer, write_trace


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "experiment"


def _make_result_dir(root: Path, name: str, seed: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = root / f"{_safe_name(name)}_seed{seed}_{stamp}"
    candidate, index = base, 1
    while candidate.exists():
        candidate = root / f"{base.name}_{index}"
        index += 1
    candidate.mkdir()
    return candidate


def _package_versions() -> dict:
    result = {}
    for name in ("aiohttp", "PyYAML", "psutil", "pynvml", "torch", "ray", "vllm", "transformers", "matplotlib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class _EventLog:
    def __init__(self, path: Path, loop, start: float):
        self.path, self.loop, self.start = path, loop, start
        self.rows = []
        self.stream = path.open("w", encoding="utf-8")

    def add(self, event: dict, status: str, message: str = "") -> dict:
        row = {"at_sec": event.get("at_sec"), "actual_sec": round(self.loop.time() - self.start, 6),
               "type": event.get("type"), "action": event.get("action", "start"),
               "status": status, "message": message}
        self.rows.append(row)
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stream.flush()
        return row

    def close(self):
        self.stream.close()


class _Router:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.proxy: LocalHttpProxy | None = None

    def endpoint(self, path: str) -> str:
        return self.proxy.url_for(path) if self.proxy else self.base_url + path


async def run_experiment(scenario_path: str | Path, execute: bool = False,
                         allow_gpu_load: bool = False, result_root: str | Path | None = None) -> Path:
    cfg = load_scenario(scenario_path)
    root = Path(result_root) if result_root else Path(__file__).resolve().parent / "results"
    result_dir = _make_result_dir(root, cfg["experiment"]["name"], int(cfg["experiment"]["seed"]))
    run_id = result_dir.name
    cfg_path = result_dir / "scenario.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    pre = await preflight(cfg)
    _json(result_dir / "preflight.json", pre)
    tokenizer = load_local_tokenizer(cfg["workload"].get("tokenizer"))
    trace = generate_trace(cfg, tokenizer=tokenizer)
    write_trace(result_dir / "request_trace.csv", trace)

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "disturbance_exp_version": __version__,
        "git_commit": git_commit(Path.cwd()),
        "python_version": sys.version,
        "dependency_versions": _package_versions(),
        "gpu": pre.get("gpu"),
        "model_or_tokenizer_path": model_name(cfg),
        "seed": cfg["experiment"]["seed"],
        "planned_duration_sec": cfg["experiment"]["duration_sec"],
        "execution_mode": "execute" if execute else "dry_run",
        "tokenizer_loaded": tokenizer is not None,
    }
    _json(result_dir / "manifest.json", manifest)

    if not execute:
        event_rows = []
        for event in sorted(cfg.get("events", []), key=lambda item: float(item["at_sec"])):
            event_rows.append({"at_sec": event["at_sec"], "actual_sec": None,
                               "type": event["type"], "action": event.get("action", "start"),
                               "status": "planned", "message": "dry-run; no resources or requests changed"})
        (result_dir / "events.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in event_rows), encoding="utf-8")
        write_csv(result_dir / "requests.csv", REQUEST_FIELDS, [])
        write_csv(result_dir / "cpu_metrics.csv", CPU_FIELDS, [])
        summary = {"status": "dry_run", "submitted_requests": 0,
                   "planned_requests": len(trace), "service_preflight_ok": pre["service_ok"],
                   "note": "No request was sent and no injector was started."}
        write_summary(result_dir / "summary.json", summary)
        write_report(result_dir / "report.md", cfg, summary, pre)
        return result_dir

    if not pre.get("service_ok"):
        summary = {"status": "failed_preflight", "submitted_requests": 0,
                   "message": "health/readiness preflight did not pass", "service_preflight_ok": False}
        write_csv(result_dir / "requests.csv", REQUEST_FIELDS, [])
        write_csv(result_dir / "cpu_metrics.csv", CPU_FIELDS, [])
        write_summary(result_dir / "summary.json", summary)
        write_report(result_dir / "report.md", cfg, summary, pre)
        return result_dir

    loop = asyncio.get_running_loop()
    start = loop.time()
    current_task = asyncio.current_task()
    handled_signals = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, current_task.cancel)
            handled_signals.append(signum)
        except (NotImplementedError, RuntimeError, ValueError):
            pass
    duration = float(cfg["experiment"]["duration_sec"])
    request_rows: list[dict] = []
    host_rows: list[dict] = []
    route = _Router(cfg["service"]["base_url"])
    event_log = _EventLog(result_dir / "events.jsonl", loop, start)
    cpu = CpuLoadInjector()
    gpu = GpuLoadInjector()
    fault = MockFaultInjector(seed=int(cfg["experiment"]["seed"]))
    auto_stop_tasks: list[asyncio.Task] = []
    metric_stop = asyncio.Event()
    event_task: asyncio.Task | None = None
    request_tasks: list[asyncio.Task] = []
    interrupted = False
    def injector_state():
        cpu_state, gpu_state, fault_state = cpu.status(), gpu.status(), fault.status()
        return {
            "cpu_load_active": cpu_state["active"],
            "cpu_load_workers": cpu_state["workers"],
            "gpu_load_active": gpu_state["active"],
            "gpu_load_pid": gpu_state["pid"],
            "network_proxy_active": bool(route.proxy and route.proxy.status()["active"]),
            "api_mock_fault_active": fault_state["active"],
        }

    metric_task = asyncio.create_task(collect_host_metrics(
        result_dir / "cpu_metrics.csv", start, float(cfg["monitoring"]["interval_sec"]),
        metric_stop, host_rows, state_provider=injector_state,
    ))

    async def stop_injector(kind: str, event: dict, message: str = "scheduled stop"):
        if kind == "cpu_compute":
            cpu.stop()
        elif kind == "gpu_compute":
            gpu.stop()
        elif kind == "network_proxy" and route.proxy:
            proxy, route.proxy = route.proxy, None
            proxy_state = proxy.status()
            await proxy.stop()
            message = json.dumps({"reason": message, **proxy_state})
        elif kind == "fault":
            fault.stop()
        event_log.add({**event, "action": "stop"}, "stopped", message)

    async def apply_event(event: dict):
        kind, action = event["type"], event.get("action", "start")
        if kind == "workload":
            event_log.add(event, "scheduled", "QPS change is reflected in the pre-generated deterministic trace")
            return
        if kind == "restore":
            cpu.stop()
            gpu.stop()
            fault.stop()
            if route.proxy:
                proxy, route.proxy = route.proxy, None
                await proxy.stop()
            event_log.add(event, "restored", "experiment-owned injectors stopped")
            return
        if action in {"stop", "restore"}:
            await stop_injector(kind, event)
            return
        if kind == "cpu_compute":
            status = cpu.start(workers=event.get("workers", 1), duty_cycle=event.get("duty_cycle", 0.25))
            row = event_log.add(event, "active" if status["active"] else "skipped", json.dumps(status))
        elif kind == "gpu_compute":
            if not allow_gpu_load:
                status = {"supported": False, "reason": "requires --allow-gpu-load"}
            else:
                current_health = await preflight(cfg)
                if not current_health.get("service_ok"):
                    status = {"supported": False, "reason": "health/readiness check failed immediately before GPU injection"}
                else:
                    status = gpu.start(device_id=int(str(event.get("target", "gpu_0")).split("_")[-1]),
                                       duty_cycle=event.get("duty_cycle", 0.1),
                                       memory_limit_mib=event.get("memory_limit_mib", 32))
            row = event_log.add(event, "active" if status.get("active") else "skipped", json.dumps(status))
        elif kind == "network_proxy":
            if route.proxy:
                await route.proxy.stop()
            proxy = LocalHttpProxy(
                cfg["service"]["base_url"], host="127.0.0.1", port=0,
                request_delay_ms=event.get("request_delay_ms", 0),
                response_delay_ms=event.get("response_delay_ms", 0),
                jitter_ms=event.get("jitter_ms", 0),
                failure_probability=event.get("failure_probability", 0),
                seed=int(cfg["experiment"]["seed"]),
            )
            try:
                status = await proxy.start()
                route.proxy = proxy
                row = event_log.add(event, "active", json.dumps(status))
            except OSError as exc:
                row = event_log.add(event, "failed", f"proxy failed to bind: {exc}")
        elif kind == "fault":
            status = fault.start(failure_probability=event.get("failure_probability", 0.01))
            row = event_log.add(event, "active", json.dumps(status))
        else:
            row = event_log.add(event, "unsupported", "no implementation")
        event_duration = event.get("duration_sec")
        if row["status"] == "active" and event_duration:
            async def delayed_stop():
                await asyncio.sleep(float(event_duration))
                await stop_injector(kind, event, "duration elapsed")
            auto_stop_tasks.append(asyncio.create_task(delayed_stop()))

    async def run_event_schedule():
        for event in sorted(cfg.get("events", []), key=lambda item: float(item["at_sec"])):
            if float(event["at_sec"]) == 0:
                continue
            target = start + float(event["at_sec"])
            await asyncio.sleep(max(0.0, target - loop.time()))
            await apply_event(event)
        await asyncio.sleep(max(0.0, start + duration - loop.time()))

    def make_payload(item: TraceRequest) -> dict:
        is_generate = cfg["service"]["request_endpoint"] == "/generate"
        return {"prompt": item.prompt, "n": 1, "best_of": 1, "temperature": 0.0,
                "top_k": 1, "max_tokens": item.output_tokens, "ignore_eos": True,
                "stream": bool(cfg["service"].get("stream")) if is_generate else False}

    def output_token_count(text: str) -> int | None:
        if tokenizer is None:
            return None
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            return len(encoded.ids if hasattr(encoded, "ids") else encoded)
        except Exception:
            return None

    async def read_nul_stream(response: aiohttp.ClientResponse, prompt: str):
        buffer = b""
        arrival_times: list[float] = []
        token_counts: list[int | None] = []
        latest_text = ""

        def consume(raw: bytes):
            nonlocal latest_text
            if not raw:
                return
            payload = json.loads(raw.decode("utf-8"))
            text_list = payload.get("text") if isinstance(payload, dict) else None
            if isinstance(text_list, list) and text_list and isinstance(text_list[0], str):
                latest_text = text_list[0]
                completion = latest_text[len(prompt):] if latest_text.startswith(prompt) else latest_text
                arrival_times.append(loop.time())
                token_counts.append(output_token_count(completion))

        async for chunk in response.content.iter_any():
            buffer += chunk
            while b"\0" in buffer:
                raw, buffer = buffer.split(b"\0", 1)
                consume(raw)
        if buffer.strip():
            consume(buffer)
        completion = latest_text[len(prompt):] if latest_text.startswith(prompt) else latest_text
        return completion, arrival_times, token_counts

    async def send_one(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, item: TraceRequest):
        target = start + item.scheduled_arrival_sec
        await asyncio.sleep(max(0.0, target - loop.time()))
        async with semaphore:
            sent = loop.time()
            row = RequestResult(run_id=run_id, request_id=item.request_id, request_type=item.request_type,
                                scheduled_arrival_sec=item.scheduled_arrival_sec,
                                actual_send_sec=sent - start, input_tokens=item.input_tokens,
                                output_tokens=None)
            if fault.should_fail():
                row.completion_sec = loop.time() - start
                row.total_latency_ms = (loop.time() - sent) * 1000
                row.status_code = 503
                row.error_type = "api_level_mock_failure"
                row.error_message = "simulated request failure; no TCP loss or backend process failure"
                request_rows.append(asdict(row))
                return
            try:
                timeout = aiohttp.ClientTimeout(total=float(cfg["limits"]["request_timeout_sec"]))
                async with session.post(route.endpoint(cfg["service"]["request_endpoint"]),
                                        json=make_payload(item), timeout=timeout,
                                        headers={"Connection": "close"}) as response:
                    row.status_code = response.status
                    if cfg["service"]["request_endpoint"] == "/generate" and cfg["service"].get("stream"):
                        completion, arrival_times, token_counts = await read_nul_stream(response, item.prompt)
                        row.output_tokens = token_counts[-1] if token_counts else None
                        row.client_ttft_ms = (arrival_times[0] - sent) * 1000 if arrival_times else None
                        if row.output_tokens and row.output_tokens > 1 and len(token_counts) == row.output_tokens and len(arrival_times) == row.output_tokens:
                            row.client_tpot_ms = (arrival_times[-1] - arrival_times[0]) * 1000 / (row.output_tokens - 1)
                        row.completion_sec = loop.time() - start
                        row.total_latency_ms = (loop.time() - sent) * 1000
                        row.success = response.status == 200 and bool(arrival_times)
                        if not row.success:
                            row.error_type = "service_http_error"
                            row.error_message = "stream contained no NUL-delimited JSON token output"
                        request_rows.append(asdict(row))
                        return
                    payload = await response.json(content_type=None)
                    row.completion_sec = loop.time() - start
                    row.total_latency_ms = (loop.time() - sent) * 1000
                    if cfg["service"]["request_endpoint"] == "/generate":
                        text_list = payload.get("text") if isinstance(payload, dict) else None
                        generated = text_list[0] if isinstance(text_list, list) and text_list and isinstance(text_list[0], str) else ""
                        completion = generated[len(item.prompt):] if generated.startswith(item.prompt) else generated
                        row.output_tokens = output_token_count(completion)
                        row.success = response.status == 200 and bool(generated)
                    else:
                        parsed = parse_benchmark_payload(payload if isinstance(payload, dict) else {})
                        if parsed["output_tokens"] is not None:
                            row.output_tokens = parsed["output_tokens"]
                        row.server_ttft_ms = parsed["server_ttft_ms"]
                        row.server_tpot_ms = parsed["server_tpot_ms"]
                        row.success = response.status == 200 and isinstance(payload, dict) and "generated_text" in payload
                    if not row.success:
                        row.error_type = "service_http_error"
                        row.error_message = str(payload)[:200]
            except asyncio.TimeoutError:
                row.completion_sec = loop.time() - start
                row.total_latency_ms = (loop.time() - sent) * 1000
                row.timeout = True
                row.error_type = "timeout"
                row.error_message = "request timeout"
            except (aiohttp.ClientError, ValueError) as exc:
                row.completion_sec = loop.time() - start
                row.total_latency_ms = (loop.time() - sent) * 1000
                row.error_type = "network_proxy_connection_failure" if route.proxy else type(exc).__name__
                row.error_message = str(exc)[:200]
            request_rows.append(asdict(row))

    try:
        # Apply t=0 events before launching request tasks so proxy routing is deterministic.
        for event in sorted(cfg.get("events", []), key=lambda item: float(item["at_sec"])):
            if float(event["at_sec"]) == 0:
                await apply_event(event)
        event_task = asyncio.create_task(run_event_schedule())
        connector = aiohttp.TCPConnector(limit=int(cfg["limits"]["max_inflight"]))
        timeout = aiohttp.ClientTimeout(total=float(cfg["limits"]["request_timeout_sec"]))
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            semaphore = asyncio.Semaphore(int(cfg["limits"]["max_inflight"]))
            request_tasks = [asyncio.create_task(send_one(session, semaphore, item)) for item in trace]
            await asyncio.gather(*request_tasks)
        await event_task
    except asyncio.CancelledError:
        event_log.add({"type": "controller", "action": "interrupt"}, "partial", "experiment interrupted")
        interrupted = True
    except Exception as exc:
        event_log.add({"type": "controller", "action": "error"}, "failed", f"{type(exc).__name__}: {exc}")
    finally:
        pending = [task for task in ([event_task] if event_task else []) + request_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in auto_stop_tasks:
            if not task.done():
                task.cancel()
        if auto_stop_tasks:
            await asyncio.gather(*auto_stop_tasks, return_exceptions=True)
        cpu.stop()
        gpu.stop()
        fault.stop()
        if route.proxy:
            proxy, route.proxy = route.proxy, None
            await proxy.stop()
        metric_stop.set()
        try:
            await asyncio.wait_for(metric_task, timeout=3)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            metric_task.cancel()
        event_log.close()
        for signum in handled_signals:
            loop.remove_signal_handler(signum)

    actual_duration = max(0.001, min(duration, loop.time() - start))
    post = await preflight(cfg)
    _json(result_dir / "postflight.json", post)
    write_csv(result_dir / "requests.csv", REQUEST_FIELDS, request_rows)
    summary = summarize(request_rows, host_rows, event_log.rows, actual_duration, cfg["slo"])
    summary["planned_duration_sec"] = duration
    summary["actual_duration_sec"] = actual_duration
    summary["service_postflight_ok"] = post.get("service_ok")
    if interrupted:
        summary["status"] = "partial"
        summary["note"] = "experiment interrupted; completed request and metric data retained"
    if not post.get("service_ok"):
        summary["status"] = "partial"
        summary["note"] = "postflight health/readiness failed; inspect service manually"
    if len(request_rows) < len(trace):
        summary["status"] = "partial"
        summary["note"] = "not all pre-generated requests completed"
    write_summary(result_dir / "summary.json", summary)
    plots = plot_results(result_dir)
    write_report(result_dir / "report.md", cfg, summary, pre, plots)
    return result_dir


async def report_result(result_dir: str | Path) -> Path:
    result_dir = Path(result_dir)
    cfg = load_scenario(result_dir / "scenario.yaml")
    pre = json.loads((result_dir / "preflight.json").read_text(encoding="utf-8"))
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    plots = plot_results(result_dir)
    report_path = result_dir / "report.md"
    write_report(report_path, cfg, summary, pre, plots)
    return report_path
