"""Result aggregation, Markdown report, and optional plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .metrics import percentile


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _numbers(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (ValueError, TypeError, KeyError):
            pass
    return values


def summarize(requests: list[dict], host_metrics: list[dict], events: list[dict],
              duration_sec: float, slo: dict) -> dict:
    submitted = len(requests)
    successful = [row for row in requests if str(row.get("success", "")).lower() in {"true", "1"}]
    failed = submitted - len(successful)
    total_tokens = sum(int(float(row.get("output_tokens") or 0)) for row in successful)
    e2e = _numbers(successful, "total_latency_ms")
    ttft = _numbers(successful, "server_ttft_ms")
    tpot = _numbers(successful, "server_tpot_ms")
    client_ttft = _numbers(successful, "client_ttft_ms")
    client_tpot = _numbers(successful, "client_tpot_ms")
    timeout_count = sum(str(row.get("timeout", "")).lower() in {"true", "1"} for row in requests)
    ttft_limit, tpot_limit = float(slo.get("ttft_ms", 0)), float(slo.get("tpot_ms", 0))
    violations = sum(
        (ttft_limit > 0 and _float(row.get("server_ttft_ms")) is not None and _float(row.get("server_ttft_ms")) > ttft_limit)
        or (tpot_limit > 0 and _float(row.get("server_tpot_ms")) is not None and _float(row.get("server_tpot_ms")) > tpot_limit)
        for row in successful
    )
    cpu_values, gpu_values = _numbers(host_metrics, "cpu_percent"), _numbers(host_metrics, "gpu_utilization_percent")
    return {
        "status": "complete",
        "submitted_requests": submitted,
        "successful_requests": len(successful),
        "failed_requests": failed,
        "timed_out_requests": timeout_count,
        "completion_rate": len(successful) / submitted if submitted else None,
        "failure_rate": failed / submitted if submitted else None,
        "timeout_rate": timeout_count / submitted if submitted else None,
        "throughput_requests_per_sec": len(successful) / duration_sec if duration_sec else None,
        "output_tokens": total_tokens,
        "output_tokens_per_sec": total_tokens / duration_sec if duration_sec else None,
        "e2e_latency_ms": _distribution(e2e),
        "server_ttft_ms": _distribution(ttft),
        "server_tpot_ms": _distribution(tpot),
        "client_ttft_ms": _distribution(client_ttft),
        "client_tpot_ms": _distribution(client_tpot),
        "slo_violations": violations,
        "slo_violation_rate": violations / len(successful) if successful else None,
        "cpu_percent_mean": sum(cpu_values) / len(cpu_values) if cpu_values else None,
        "gpu_utilization_percent_mean": sum(gpu_values) / len(gpu_values) if gpu_values else None,
        "host_metric_samples": len(host_metrics),
        "events": events,
    }


def _float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _distribution(values: list[float]) -> dict:
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
            "p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99)}


def write_report(path: Path, cfg: dict, summary: dict, preflight: dict,
                 plot_paths: list[Path] | None = None) -> None:
    plots = "\n".join(f"- `{p.name}`" for p in (plot_paths or [])) or "- 当前没有足够数据绘图。"
    gpu = preflight.get("gpu", {})
    text = f"""# {cfg['experiment']['name']} 实验报告

- 状态：`{summary.get('status')}`
- 单机运行时长：{cfg['experiment']['duration_sec']} 秒
- 请求成功率：{_fmt(summary.get('completion_rate'))}
- 成功吞吐：{_fmt(summary.get('throughput_requests_per_sec'))} 请求/秒
- 输出 Token 吞吐：{_fmt(summary.get('output_tokens_per_sec'))} Token/秒
- 服务端 TTFT：{_fmt_dist(summary.get('server_ttft_ms'))} 毫秒
- 服务端 TPOT：{_fmt_dist(summary.get('server_tpot_ms'))} 毫秒
- 客户端流式 TTFT：{_fmt_dist(summary.get('client_ttft_ms'))} 毫秒
- 客户端流式 TPOT：{_fmt_dist(summary.get('client_tpot_ms'))} 毫秒
- 客户端端到端时延：{_fmt_dist(summary.get('e2e_latency_ms'))} 毫秒
- SLO 违约率：{_fmt(summary.get('slo_violation_rate'))}
- GPU：{gpu.get('name', '不可用')}；空闲显存 {gpu.get('free_mib', '未知')} MiB

## 扰动事件

""" + ("\n".join(f"- {e.get('type')} @ {e.get('actual_sec', e.get('at_sec'))}s：{e.get('status')} — {e.get('message', '')}" for e in summary.get("events", [])) or "- 无") + f"""

## 图表

{plots}

## 解释范围

网络代理造成的等待只表示 API 代理层额外时延，不代表 TCP packet loss 或跨节点网络故障。没有服务端可观测数据时，相关指标保持为空。
"""
    path.write_text(text, encoding="utf-8")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _fmt_dist(values) -> str:
    if not values or not values.get("count"):
        return "n/a"
    return f"mean {_fmt(values['mean'])}, P50 {_fmt(values['p50'])}, P95 {_fmt(values['p95'])}, P99 {_fmt(values['p99'])}"


def plot_results(result_dir: Path) -> list[Path]:
    requests = read_csv(result_dir / "requests.csv")
    metrics = read_csv(result_dir / "cpu_metrics.csv")
    events = []
    event_path = result_dir / "events.jsonl"
    if event_path.exists():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not requests and not metrics:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    created = []
    plots = [
        ("ttft_timeline.png", "server_ttft_ms", "Server TTFT (ms)"),
        ("tpot_timeline.png", "server_tpot_ms", "Server TPOT (ms)"),
    ]
    for filename, field, title in plots:
        points = [(float(r.get("completion_sec") or 0), _float(r.get(field))) for r in requests]
        points = [(x, y) for x, y in points if y is not None]
        if not points:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", linewidth=1)
        ax.set(xlabel="Experiment time (s)", ylabel=title, title=title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        target = plot_dir / filename
        fig.savefig(target, dpi=120)
        plt.close(fig)
        created.append(target)
    throughput = {}
    for row in requests:
        if str(row.get("success", "")).lower() not in {"true", "1"}:
            continue
        try:
            second = int(float(row.get("completion_sec") or 0))
        except (TypeError, ValueError):
            continue
        throughput[second] = throughput.get(second, 0) + 1
    if throughput:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        xs = sorted(throughput)
        ax.step(xs, [throughput[x] for x in xs], where="post", linewidth=1.5)
        ax.set(xlabel="Experiment time (s)", ylabel="Completed requests/s", title="Throughput timeline")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        target = plot_dir / "throughput_timeline.png"
        fig.savefig(target, dpi=120)
        plt.close(fig)
        created.append(target)
    gpu = [(float(r.get("relative_sec") or 0), _float(r.get("gpu_utilization_percent"))) for r in metrics]
    gpu = [(x, y) for x, y in gpu if y is not None]
    if gpu:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot([x for x, _ in gpu], [y for _, y in gpu], linewidth=1)
        ax.set(xlabel="Experiment time (s)", ylabel="GPU utilization (%)", title="GPU utilization")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        target = plot_dir / "gpu_utilization.png"
        fig.savefig(target, dpi=120)
        plt.close(fig)
        created.append(target)
    timed_events = [e for e in events if _float(e.get("at_sec")) is not None]
    if timed_events:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        for index, event in enumerate(timed_events):
            x = float(event.get("actual_sec", event.get("at_sec", 0)))
            ax.axvline(x, alpha=0.7)
            ax.text(x, index % 3, str(event.get("type", "event")), rotation=90, va="bottom", fontsize=8)
        ax.set(xlabel="Experiment time (s)", yticks=[], title="Disturbance timeline")
        fig.tight_layout()
        target = plot_dir / "disturbance_timeline.png"
        fig.savefig(target, dpi=120)
        plt.close(fig)
        created.append(target)
    return created
