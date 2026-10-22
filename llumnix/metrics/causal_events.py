"""Low-overhead, append-only causal telemetry for Llumnix profiling runs.

Enable by setting ``LLUMNIX_CAUSAL_TRACE_DIR``. Events use the host's
``CLOCK_MONOTONIC`` clock for ordering and also carry a wall-clock pair for
external correlation. Request text and token contents are never recorded.
"""

from __future__ import annotations

import atexit
import enum
import json
import math
import os
import queue
import socket
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
_WRITER = None
_WRITER_PID = None
_WRITER_LOCK = threading.Lock()
_SEQUENCE_LOCK = threading.Lock()
_EVENT_SEQUENCE = 0
_BATCH_SEQUENCE = 0
_SCHEDULED_BATCHES: dict[tuple, deque[str]] = defaultdict(deque)
_BATCH_FEATURES: dict[str, dict[str, Any]] = {}


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return _safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except Exception:  # pragma: no cover - defensive for optional array types
            pass
    return str(value)


class _JsonlWriter:
    def __init__(self, directory: str, pid: int) -> None:
        self.pid = pid
        self.stream_id = f"{socket.gethostname()}-{pid}-{uuid.uuid4().hex}"
        self.error: str | None = None
        self.path = Path(directory) / f"events_{socket.gethostname()}_{pid}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._closed = False
        self._close_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="llumnix-causal-writer", daemon=True)
        self.thread.start()

    def put(self, event: dict[str, Any]) -> None:
        self.queue.put_nowait(event)

    def _run(self) -> None:
        try:
            self._write_loop()
        except Exception as exc:  # surfaced by flush(), never claim a dead writer is complete
            self.error = type(exc).__name__

    def _write_loop(self) -> None:
        buffered_events = 0
        with self.path.open("a", encoding="utf-8", buffering=64 * 1024) as stream:
            while True:
                try:
                    item = self.queue.get(timeout=0.25)
                except queue.Empty:
                    if buffered_events:
                        stream.flush()
                        buffered_events = 0
                    continue
                try:
                    if item is None:
                        stream.flush()
                        return
                    if isinstance(item, _FlushBarrier):
                        stream.flush()
                        os.fsync(stream.fileno())
                        item.done.set()
                        buffered_events = 0
                        continue
                    stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"),
                                             allow_nan=False) + "\n")
                    buffered_events += 1
                    if buffered_events >= 128:
                        stream.flush()
                        buffered_events = 0
                finally:
                    self.queue.task_done()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.queue.put_nowait(None)
        if threading.current_thread() is not self.thread:
            # The sentinel follows all queued events, so joining drains and
            # flushes the complete process stream before its interpreter exits.
            self.thread.join()


class _FlushBarrier:
    def __init__(self) -> None:
        self.done = threading.Event()


def flush(timeout_sec: float = 5.0) -> dict[str, Any]:
    """Acknowledge durable process-local sequence progress, not cluster quiescence.

    The caller must query every registered producer; pending output delivery and
    asynchronous work need separate confirmation. Disabled/dead writers return
    incomplete. Sequence assignment, enqueue and the barrier share one lock.
    """
    if timeout_sec < 0:
        raise ValueError("negative flush timeout")
    writer = _writer_for_process()
    if writer is None:
        return {"status": "incomplete", "reason": "tracing_disabled"}
    with _SEQUENCE_LOCK:
        barrier = _FlushBarrier()
        sequence = _EVENT_SEQUENCE
        through_ns = time.monotonic_ns()
        if writer._closed or writer.error:
            return {"status": "incomplete", "reason": writer.error or "writer_closed"}
        writer.queue.put_nowait(barrier)
    done = barrier.done.wait(timeout_sec)
    return {"status": "passed" if done and writer.error is None else "incomplete",
            "source": writer.stream_id, "last_sequence": sequence,
            "through_ns": through_ns, "received_ns": time.monotonic_ns(),
            "durable": bool(done and writer.error is None),
            "reason": writer.error if writer.error else (None if done else "flush_timeout")}


def _after_fork_child() -> None:
    global _WRITER, _WRITER_PID, _WRITER_LOCK, _SEQUENCE_LOCK, _EVENT_SEQUENCE
    global _SCHEDULED_BATCHES, _BATCH_FEATURES, _BATCH_SEQUENCE
    _WRITER = None
    _WRITER_PID = None
    _WRITER_LOCK = threading.Lock()
    _SEQUENCE_LOCK = threading.Lock()
    _EVENT_SEQUENCE = 0
    _BATCH_SEQUENCE = 0
    _SCHEDULED_BATCHES = defaultdict(deque)
    _BATCH_FEATURES = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _writer_for_process() -> _JsonlWriter | None:
    global _WRITER, _WRITER_PID
    directory = os.environ.get("LLUMNIX_CAUSAL_TRACE_DIR")
    if not directory:
        return None
    pid = os.getpid()
    if _WRITER is not None and _WRITER_PID == pid:
        return _WRITER
    with _WRITER_LOCK:
        if _WRITER is not None and _WRITER_PID == pid:
            return _WRITER
        _WRITER = _JsonlWriter(directory, pid)
        _WRITER_PID = pid
        atexit.register(_WRITER.close)
        return _WRITER


def enabled() -> bool:
    return bool(os.environ.get("LLUMNIX_CAUSAL_TRACE_DIR"))


def should_sample_output_update(request_id: str, output_tokens: int, finished: bool) -> bool:
    """Sample first, every 64th, and final token updates for queue diagnostics."""
    if not enabled():
        return False
    count = int(output_tokens)
    return bool(finished or count == 1 or (count > 0 and count % 64 == 0))


def emit(event: str, **fields: Any) -> int | None:
    """Queue one JSONL event without blocking the inference or scheduler loop."""
    global _EVENT_SEQUENCE
    writer = _writer_for_process()
    if writer is None:
        return None
    with _SEQUENCE_LOCK:
        mono_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        wall_ns = time.time_ns()
        _EVENT_SEQUENCE += 1
        event_sequence = _EVENT_SEQUENCE
        payload = {
            **_safe(fields),
            "schema_version": SCHEMA_VERSION,
            "event_sequence": event_sequence,
            "event": str(event),
            "clock_id": "CLOCK_MONOTONIC",
            "monotonic_ns": mono_ns,
            "wall_time_ns": wall_ns,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "host": socket.gethostname(),
            "process_name": os.environ.get("PROC_NAME", ""),
            "stream_id": writer.stream_id,
        }
        try:
            writer.put(payload)
        except Exception:
            # A failed event remains visible through the sequence gap.
            return None
    return event_sequence


def metadata_signature(metadata_list: Iterable[Any]) -> tuple:
    """Stable signature shared by scheduler and executor for a scheduled batch."""
    records = []
    for metadata in metadata_list or ():
        request_id = str(getattr(metadata, "request_id", ""))
        seq_data = getattr(metadata, "seq_data", {}) or {}
        sequences = []
        values = seq_data.values() if isinstance(seq_data, Mapping) else ()
        for sequence in values:
            try:
                length = int(sequence.get_len())
            except (AttributeError, TypeError, ValueError):
                length = -1
            try:
                computed = int(sequence.get_num_computed_tokens())
            except (AttributeError, TypeError, ValueError):
                computed = -1
            sequences.append((length, computed))
        records.append((request_id, bool(getattr(metadata, "is_prompt", False)), tuple(sorted(sequences))))
    return tuple(sorted(records))


def register_scheduled_batch(metadata_list: Iterable[Any], fields: Mapping[str, Any]) -> str | None:
    """Register a scheduler batch so its Ray-executor interval can be joined."""
    global _BATCH_SEQUENCE
    if not enabled():
        return None
    signature = metadata_signature(metadata_list)
    _BATCH_SEQUENCE += 1
    batch_id = f"{os.getpid()}-{_BATCH_SEQUENCE}"
    _SCHEDULED_BATCHES[signature].append(batch_id)
    _BATCH_FEATURES[batch_id] = dict(fields)
    emit("batch_scheduled", batch_id=batch_id, metadata_signature=signature, **dict(fields))
    return batch_id


def start_scheduled_batch(metadata_list: Iterable[Any]) -> tuple[str | None, dict[str, Any]]:
    if not enabled():
        return None, {}
    signature = metadata_signature(metadata_list)
    pending = _SCHEDULED_BATCHES.get(signature)
    if not pending:
        return None, {}
    batch_id = pending.popleft()
    return batch_id, dict(_BATCH_FEATURES.get(batch_id, {}))


def finish_scheduled_batch(batch_id: str | None, executor_wall_ms: float,
                           *, succeeded: bool = True, error_type: str | None = None) -> None:
    if batch_id is None:
        return
    features = _BATCH_FEATURES.pop(batch_id, {})
    emit("engine_iteration_completed", batch_id=batch_id,
         executor_wall_ms=float(executor_wall_ms), succeeded=bool(succeeded),
         error_type=error_type, **features)


def emit_clock_sync() -> None:
    """Record a local wall/monotonic pair and its sampling uncertainty."""
    before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    wall = time.time_ns()
    after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    emit("clock_sync", sampled_monotonic_ns=(before + after) // 2,
         sampled_wall_time_ns=wall, uncertainty_bound_ns=(after - before + 1) // 2)
