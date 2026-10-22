"""Audit unsplit causal event streams after an instrumented serving run."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any


def audit(data_dir: Path) -> dict[str, Any]:
    """Summarize parse, process sequence, lifecycle, and batch join integrity."""
    event_dir = data_dir / "causal_events"
    files = sorted(event_dir.glob("events_*.jsonl"))
    files.extend(path for path in sorted(event_dir.glob("events_*.jsonl.gz"))
                 if not path.with_suffix("").exists())
    process: dict[str, dict[str, Any]] = {}
    event_counts: collections.Counter[str] = collections.Counter()
    request_received: set[str] = set()
    request_terminal: set[str] = set()
    scheduled_batches: set[str] = set()
    completed_batches: set[str] = set()
    failed_batches: set[str] = set()
    json_errors: list[dict[str, Any]] = []
    total_events = 0
    for path in files:
        stream_events = 0
        file_processes: set[str] = set()
        opener = gzip.open if path.name.endswith(".jsonl.gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    json_errors.append({"path": str(path.relative_to(data_dir)),
                                        "line": line_number, "error": str(exc)})
                    continue
                stream_events += 1
                total_events += 1
                event_name = str(event.get("event", "unknown"))
                event_counts[event_name] += 1
                host, pid = str(event.get("host", "unknown")), str(event.get("pid", "unknown"))
                key = f"{host}:{pid}"
                file_processes.add(key)
                stats = process.setdefault(key, {
                    "host": host, "pid": pid, "source_files": [], "events": 0,
                    "sequence_values": set(), "sequence_missing": 0,
                    "sequence_duplicates": 0, "sequence_invalid": 0,
                    "monotonic_inversions": 0, "worst_monotonic_inversion_ms": 0.0,
                    "last_monotonic_ns": None,
                })
                sequence = event.get("event_sequence")
                if isinstance(sequence, int) and sequence > 0:
                    if sequence in stats["sequence_values"]:
                        stats["sequence_duplicates"] += 1
                    else:
                        stats["sequence_values"].add(sequence)
                else:
                    stats["sequence_invalid"] += 1
                mono = event.get("monotonic_ns")
                if isinstance(mono, int):
                    previous = stats["last_monotonic_ns"]
                    if isinstance(previous, int) and mono < previous:
                        stats["monotonic_inversions"] += 1
                        stats["worst_monotonic_inversion_ms"] = max(
                            stats["worst_monotonic_inversion_ms"], (previous - mono) / 1e6)
                    stats["last_monotonic_ns"] = mono
                stats["events"] += 1
                request_id = str(event.get("request_id", ""))
                if event_name == "request_server_received" and request_id:
                    request_received.add(request_id)
                elif event_name == "request_server_terminal" and request_id:
                    request_terminal.add(request_id)
                batch_id = str(event.get("batch_id", ""))
                if event_name == "batch_scheduled" and batch_id:
                    scheduled_batches.add(batch_id)
                elif event_name == "engine_iteration_completed" and batch_id:
                    completed_batches.add(batch_id)
                    if not event.get("succeeded", True):
                        failed_batches.add(batch_id)
        for key in file_processes:
            process[key]["source_files"].append({
                "path": str(path.relative_to(data_dir)), "events": stream_events,
                "bytes": path.stat().st_size,
            })

    process_report = {}
    for key, stats in process.items():
        sequences = stats.pop("sequence_values")
        if sequences:
            low, high = min(sequences), max(sequences)
            stats["sequence_min"] = low
            stats["sequence_max"] = high
            stats["sequence_missing"] = max(0, high - low + 1 - len(sequences)) + max(0, low - 1)
        else:
            stats["sequence_min"] = None
            stats["sequence_max"] = None
        process_report[key] = stats

    unmatched_scheduled = sorted(scheduled_batches - completed_batches)
    completed_without_schedule = sorted(completed_batches - scheduled_batches)
    report = {
        "data_dir": str(data_dir),
        "source_stream_count": len(files),
        "source_streams": [str(path.relative_to(data_dir)) for path in files],
        "total_events": total_events,
        "event_type_counts": dict(sorted(event_counts.items())),
        "json_parse_errors": json_errors,
        "processes": process_report,
        "request_lifecycle": {
            "server_received_requests": len(request_received),
            "server_terminal_requests": len(request_terminal),
            "received_without_terminal": sorted(request_received - request_terminal),
            "terminal_without_received": sorted(request_terminal - request_received),
        },
        "batch_joins": {
            "scheduled_batches": len(scheduled_batches),
            "completed_batches": len(completed_batches),
            "failed_executor_batches": len(failed_batches),
            "scheduled_without_executor_completion": unmatched_scheduled,
            "executor_completion_without_schedule": completed_without_schedule,
        },
        "interpretation": {
            "process_sequence": "event_sequence values are checked as a set per PID, independent of thread write order",
            "monotonic_order": "inversions are reported because concurrent emit calls can enqueue out of timestamp order",
            "batch_join": "batch_scheduled is joined to engine_iteration_completed by batch_id",
            "source_streams": "only unsplit root streams are audited; filtered run shards cannot prove process sequence completeness",
        },
    }
    return report


def write_report(data_dir: Path, report: dict[str, Any]) -> None:
    (data_dir / "event_integrity_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Causal event integrity audit", "",
             f"Source streams: {report['source_stream_count']}; events: {report['total_events']}; "
             f"JSON parse errors: {len(report['json_parse_errors'])}.", "",
             "## Process streams", "",
             "| Host:PID | Events | Missing sequence values | Duplicate sequence values | Monotonic inversions | Worst inversion ms |",
             "|:---|---:|---:|---:|---:|---:|"]
    for key, stats in sorted(report["processes"].items()):
        lines.append(f"| {key} | {stats['events']} | {stats['sequence_missing']} | "
                     f"{stats['sequence_duplicates']} | {stats['monotonic_inversions']} | "
                     f"{stats['worst_monotonic_inversion_ms']:.6f} |")
    lifecycle = report["request_lifecycle"]
    batches = report["batch_joins"]
    lines.extend(["", "## Joins", "",
                  f"Server requests received/terminal: {lifecycle['server_received_requests']}/"
                  f"{lifecycle['server_terminal_requests']}; missing terminals: "
                  f"{len(lifecycle['received_without_terminal'])}.",
                  f"Scheduled/executor batches: {batches['scheduled_batches']}/"
                  f"{batches['completed_batches']}; unmatched schedules: "
                  f"{len(batches['scheduled_without_executor_completion'])}; failed batches: "
                  f"{batches['failed_executor_batches']}.", "",
                  "## Event counts", "",
                  "| Event | Count |", "|:---|---:|"])
    lines.extend(f"| {name} | {count} |" for name, count in report["event_type_counts"].items())
    lines.extend(["", "Sequence completeness is computed from unsplit root streams. Monotonic inversions are reported but are not treated as drops because emits can race across threads.", ""])
    (data_dir / "event_integrity_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="experiment result root containing causal_events/")
    args = parser.parse_args()
    result = audit(args.data_dir)
    write_report(args.data_dir, result)
    print(json.dumps({
        "source_stream_count": result["source_stream_count"],
        "total_events": result["total_events"],
        "json_parse_errors": len(result["json_parse_errors"]),
        "missing_sequence_values": sum(item["sequence_missing"]
                                        for item in result["processes"].values()),
        "missing_request_terminals": len(result["request_lifecycle"]["received_without_terminal"]),
        "unmatched_batches": len(result["batch_joins"]["scheduled_without_executor_completion"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
