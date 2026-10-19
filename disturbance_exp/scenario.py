"""Scenario loading and conservative validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


class ScenarioError(ValueError):
    pass


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ScenarioError(f"{name} must be a mapping")
    return value


def validate_scenario(raw: dict) -> dict:
    cfg = deepcopy(_mapping(raw, "scenario"))
    cfg.setdefault("experiment", {})
    cfg.setdefault("service", {})
    cfg.setdefault("limits", {})
    cfg.setdefault("workload", {})
    cfg.setdefault("monitoring", {})
    cfg.setdefault("slo", {})
    for key in ("experiment", "service", "limits", "workload", "monitoring", "slo"):
        _mapping(cfg[key], key)

    exp = cfg["experiment"]
    exp.setdefault("name", "single_host")
    exp.setdefault("seed", 42)
    exp.setdefault("duration_sec", 20)
    if not isinstance(exp["name"], str) or not exp["name"].strip():
        raise ScenarioError("experiment.name must be a non-empty string")
    if not isinstance(exp["seed"], int):
        raise ScenarioError("experiment.seed must be an integer")
    if not isinstance(exp["duration_sec"], (int, float)) or not 1 <= exp["duration_sec"] <= 3600:
        raise ScenarioError("experiment.duration_sec must be between 1 and 3600")

    service = cfg["service"]
    service.setdefault("base_url", "http://127.0.0.1:8080")
    service.setdefault("request_endpoint", "/generate_benchmark")
    service.setdefault("health_endpoint", "/health")
    service.setdefault("ready_endpoint", "/is_ready")
    service.setdefault("stream", False)
    parsed = urlparse(service["base_url"])
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ScenarioError("service.base_url must use HTTP and a loopback host")
    if not service["request_endpoint"].startswith("/"):
        raise ScenarioError("service.request_endpoint must start with /")
    if service["request_endpoint"] not in {"/generate", "/generate_benchmark"}:
        raise ScenarioError("service.request_endpoint must be /generate or /generate_benchmark")
    if service["stream"] and service["request_endpoint"] != "/generate":
        raise ScenarioError("streaming mode is supported only for /generate")

    limits = cfg["limits"]
    limits.setdefault("max_inflight", 2)
    limits.setdefault("max_requests", 20)
    limits.setdefault("request_timeout_sec", 60)
    if not 1 <= int(limits["max_inflight"]) <= 16:
        raise ScenarioError("limits.max_inflight must be between 1 and 16")
    if not 1 <= int(limits["max_requests"]) <= 10000:
        raise ScenarioError("limits.max_requests must be between 1 and 10000")
    if not 1 <= float(limits["request_timeout_sec"]) <= 180:
        raise ScenarioError("limits.request_timeout_sec must be between 1 and 180")

    workload = cfg["workload"]
    workload.setdefault("arrival", "poisson")
    workload.setdefault("base_qps", 0.25)
    workload.setdefault("input_tokens", 64)
    workload.setdefault("output_tokens", 8)
    workload.setdefault("max_model_len", 8192)
    workload.setdefault("tokenizer", "/workspace/llumnix-ray/model")
    workload.setdefault("dataset_type", "sharegpt")
    if workload["arrival"] not in {"fixed", "poisson", "burst"}:
        raise ScenarioError("workload.arrival must be fixed, poisson, or burst")
    if not 0.01 <= float(workload["base_qps"]) <= 50:
        raise ScenarioError("workload.base_qps must be between 0.01 and 50")
    if int(workload["input_tokens"]) < 1 or int(workload["output_tokens"]) < 1:
        raise ScenarioError("input_tokens and output_tokens must be positive")
    if int(workload["input_tokens"]) + int(workload["output_tokens"]) > int(workload["max_model_len"]):
        raise ScenarioError("input_tokens + output_tokens exceeds max_model_len")
    workload.setdefault("profile_mix", None)
    if workload["profile_mix"] is not None:
        profiles = workload["profile_mix"]
        if not isinstance(profiles, list) or not profiles:
            raise ScenarioError("workload.profile_mix must be a non-empty list")
        total_weight = 0.0
        for i, profile in enumerate(profiles):
            _mapping(profile, f"workload.profile_mix[{i}]")
            if int(profile.get("input_tokens", 0)) < 1 or int(profile.get("output_tokens", 0)) < 1:
                raise ScenarioError("profile token counts must be positive")
            if int(profile["input_tokens"]) + int(profile["output_tokens"]) > int(workload["max_model_len"]):
                raise ScenarioError("profile token lengths exceed max_model_len")
            total_weight += float(profile.get("weight", 1.0))
        if total_weight <= 0:
            raise ScenarioError("profile weights must sum to a positive value")

    cfg["monitoring"].setdefault("interval_sec", 1)
    if not 0.2 <= float(cfg["monitoring"]["interval_sec"]) <= 10:
        raise ScenarioError("monitoring.interval_sec must be between 0.2 and 10")
    cfg["slo"].setdefault("ttft_ms", 2000)
    cfg["slo"].setdefault("tpot_ms", 150)

    events = cfg.setdefault("events", [])
    if not isinstance(events, list):
        raise ScenarioError("events must be a list")
    known = {"workload", "cpu_compute", "gpu_compute", "network_proxy", "fault", "restore"}
    for i, event in enumerate(events):
        _mapping(event, f"events[{i}]")
        kind = event.get("type")
        at = float(event.get("at_sec", -1))
        if kind not in known:
            raise ScenarioError(f"events[{i}].type is unsupported")
        if not 0 <= at < float(exp["duration_sec"]):
            raise ScenarioError(f"events[{i}].at_sec must fall inside experiment duration")
        if "duration_sec" in event and not 0 < float(event["duration_sec"]) <= 300:
            raise ScenarioError("event duration_sec must be between 0 and 300")
        if kind == "workload" and event.get("action") == "set_qps":
            if not 0.01 <= float(event.get("value", 0)) <= 50:
                raise ScenarioError("workload set_qps must be between 0.01 and 50")
        if kind == "cpu_compute":
            if int(event.get("workers", 1)) not in (1, 2):
                raise ScenarioError("CPU workers are limited to 1 or 2")
            if not 0 < float(event.get("duty_cycle", 0.25)) <= 0.5:
                raise ScenarioError("CPU duty_cycle must be in (0, 0.5]")
        if kind == "gpu_compute":
            if not 0 < float(event.get("duty_cycle", 0.1)) <= 0.5:
                raise ScenarioError("GPU duty_cycle must be in (0, 0.5]")
            if not 1 <= int(event.get("memory_limit_mib", 32)) <= 256:
                raise ScenarioError("GPU memory_limit_mib must be between 1 and 256")
        if kind == "network_proxy":
            for key, ceiling in (("request_delay_ms", 1000), ("response_delay_ms", 1000), ("jitter_ms", 250)):
                if not 0 <= float(event.get(key, 0)) <= ceiling:
                    raise ScenarioError(f"{key} must be between 0 and {ceiling}")
            if not 0 <= float(event.get("failure_probability", 0)) <= 0.05:
                raise ScenarioError("proxy failure_probability is capped at 0.05")
        if kind == "fault":
            if event.get("mode", "mock") != "mock":
                raise ScenarioError("only API-level mock faults are supported")
            if not 0 <= float(event.get("failure_probability", 0.01)) <= 0.05:
                raise ScenarioError("mock failure_probability is capped at 0.05")
    return cfg


def load_scenario(path: str | Path) -> dict:
    file_path = Path(path)
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScenarioError(f"cannot read scenario {file_path}: {exc}") from exc
    return validate_scenario(raw)
