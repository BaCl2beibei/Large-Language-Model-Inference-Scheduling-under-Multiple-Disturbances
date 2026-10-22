"""CPU-only request accounting. All times are host CLOCK_MONOTONIC nanoseconds.

Event time reconstructs final scores; received time gates online observations.
A producer manifest + fsynced sequence/watermark acknowledgements are required
for completeness. An empty engine queue is never a delivery acknowledgement.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Iterable

NS = 1_000_000_000
KINDS = frozenset({'planned', 'sent', 'accepted', 'engine_first', 'api_first',
                   'client_first', 'client_output', 'client_chunk', 'server_terminal',
                   'client_terminal', 'cancel_requested', 'cancel_ack', 'score_deadline',
                   'diagnostic'})


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Event:
    source: str
    sequence: int
    kind: str
    occurred_ns: int
    received_ns: int
    run_id: str
    episode_id: str
    request_id: str | None = None
    server_request_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.source or self.sequence < 1 or self.kind not in KINDS:
            raise ValueError('invalid source, sequence or event kind')
        if self.occurred_ns < 0 or self.received_ns < self.occurred_ns:
            raise ValueError('invalid monotonic occurrence/receipt pair')
        if not self.run_id or not self.episode_id:
            raise ValueError('run and episode IDs are required')
        if self.kind not in ('score_deadline', 'diagnostic') and not self.request_id:
            raise ValueError('request ID is required')


@dataclass(frozen=True)
class Checkpoint:
    source: str
    last_sequence: int
    through_ns: int
    received_ns: int
    pending_delivery: int | None = None
    durable: bool = False


class EventJournal:
    """One producer, ordered append + explicit durable watermark, no background sleep.

    The owner must stop/drain asynchronous work before acknowledging pending_delivery=0.
    A checkpoint asserts all occurrences <= through_ns have been emitted. This is
    automatic only for local events stamped by record(); reconstructed events must
    not be appended behind an earlier watermark.
    """
    def __init__(self, path: Path, run_id: str, episode_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source = f'{os.getpid()}-{uuid.uuid4().hex}'
        self.run_id, self.episode_id = run_id, episode_id
        self._lock = threading.Lock()
        self._stream = self.path.open('xb')  # never append a new session to old evidence
        self._sequence = 0
        self._watermark = -1
        self._closed = False

    def record(self, kind: str, request_id: str | None = None, *,
               occurred_ns: int | None = None, server_request_id: str | None = None,
               **data: Any) -> Event:
        with self._lock:
            now = time.monotonic_ns()
            occurred = now if occurred_ns is None else occurred_ns
            if occurred <= self._watermark:
                raise ValueError('cannot append behind acknowledged watermark')
            event = Event(self.source, self._sequence + 1, kind, occurred, now,
                          self.run_id, self.episode_id, request_id,
                          server_request_id, data)
            self._stream.write((json.dumps(asdict(event), allow_nan=False) + '\n').encode())
            self._sequence += 1
            return event

    def checkpoint(self, *, pending_delivery: int | None = None) -> Checkpoint:
        with self._lock:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            now = time.monotonic_ns()
            self._watermark = now
            cp = Checkpoint(self.source, self._sequence, now, now, pending_delivery, True)
            target = self.path.with_suffix(self.path.suffix + '.checkpoint.json')
            temporary = target.with_suffix(target.suffix + '.tmp')
            with temporary.open('w') as stream:
                json.dump(asdict(cp), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            return cp

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._stream.close()
                self._closed = True


class EventCursor:
    """Tail whole JSONL records; preserve partial bytes and stamp actual receipt.

    Never use a persisted producer received_ns as the collector arrival time.
    Rotation/truncation and malformed complete lines are sticky integrity errors.
    """
    def __init__(self):
        self.offsets: dict[str, int] = {}
        self.identities: dict[str, tuple[int, int]] = {}
        self.issues: list[str] = []

    def poll(self, path: Path, *, received_ns: int | None = None,
             adapter: CausalEventAdapter | None = None) -> list[Event]:
        key = str(Path(path).resolve())
        stat = Path(path).stat()
        identity = (stat.st_dev, stat.st_ino)
        offset = self.offsets.get(key, 0)
        if (key in self.identities and identity != self.identities[key]) or stat.st_size < offset:
            self.issues.append(f'stream_replaced_or_truncated:{key}')
            return []
        self.identities[key] = identity
        events = []
        with Path(path).open('rb') as stream:
            stream.seek(offset)
            while True:
                line = stream.readline()
                if not line.endswith(b'\n'):
                    break
                offset += len(line)
                try:
                    payload = json.loads(line)
                    arrival = time.monotonic_ns() if received_ns is None else received_ns
                    if adapter is None:
                        payload['received_ns'] = arrival
                        events.append(Event(**payload))
                    else:
                        events.append(adapter.adapt(payload, received_ns=arrival))
                except (ValueError, TypeError, KeyError) as exc:
                    self.issues.append(f'invalid_record:{key}:{offset}:{type(exc).__name__}')
        self.offsets[key] = offset
        return events

    def checkpoint(self, path: Path, *, received_ns: int | None = None) -> Checkpoint:
        payload = json.loads(Path(path).read_text())
        payload['received_ns'] = time.monotonic_ns() if received_ns is None else received_ns
        return Checkpoint(**payload)


def _overlap(start: int, end: int, lo: int, hi: int) -> float:
    return max(0, min(end, hi) - max(start, lo)) / NS


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    a, b = math.floor(index), math.ceil(index)
    return values[a] + (values[b] - values[a]) * (index - a)


class RequestLedger:
    def __init__(self, run_id: str, episode_id: str, *, start_ns: int,
                 arrival_end_ns: int, score_deadline_ns: int,
                 expected_sources: Iterable[str], ttft_slo_ms: float = 2000,
                 tpot_slo_ms: float = 150, expected_request_ids: Iterable[str] | None = None):
        if not start_ns < arrival_end_ns <= score_deadline_ns:
            raise ValueError('invalid scoring boundaries')
        if ttft_slo_ms < 0 or tpot_slo_ms < 0:
            raise ValueError('negative SLO')
        self.run_id, self.episode_id = run_id, episode_id
        self.start_ns, self.arrival_end_ns, self.deadline_ns = start_ns, arrival_end_ns, score_deadline_ns
        self.sources = frozenset(expected_sources)
        self.expected_request_ids = (frozenset(expected_request_ids)
                                     if expected_request_ids is not None else None)
        if not self.sources:
            raise ValueError('explicit producer inventory required')
        self.ttft_slo_ms, self.tpot_slo_ms = ttft_slo_ms, tpot_slo_ms
        self.events: dict[tuple[str, int], Event] = {}
        self.checkpoints: dict[str, Checkpoint] = {}
        self.issues: set[str] = set()
        self._frozen: dict | None = None

    def ingest(self, events: Iterable[Event]) -> None:
        for event in events:
            if (event.run_id, event.episode_id) != (self.run_id, self.episode_id):
                self.issues.add('foreign_run_or_episode')
                continue
            if event.source not in self.sources:
                self.issues.add(f'unregistered_source:{event.source}')
            key = (event.source, event.sequence)
            old = self.events.get(key)
            if old:
                # Re-polling can have a different collector receipt. Preserve the first.
                a, b = asdict(old), asdict(event)
                a.pop('received_ns'); b.pop('received_ns')
                if a != b:
                    self.issues.add(f'conflicting_sequence:{key}')
                continue
            cp = self.checkpoints.get(event.source)
            if cp and event.sequence > cp.last_sequence and event.occurred_ns <= cp.through_ns:
                self.issues.add(f'event_behind_watermark:{key}')
            if self._frozen and event.occurred_ns <= self.deadline_ns:
                self.issues.add(f'late_scoring_evidence_after_freeze:{key}')
            self.events[key] = deepcopy(event)

    def acknowledge(self, checkpoint: Checkpoint) -> None:
        if (checkpoint.last_sequence < 0 or checkpoint.through_ns < 0
                or checkpoint.received_ns < checkpoint.through_ns
                or (checkpoint.pending_delivery is not None and checkpoint.pending_delivery < 0)):
            self.issues.add(f'invalid_checkpoint:{checkpoint.source}')
            return
        old = self.checkpoints.get(checkpoint.source)
        if old and (checkpoint.last_sequence < old.last_sequence or checkpoint.through_ns < old.through_ns):
            self.issues.add(f'checkpoint_regression:{checkpoint.source}')
            return
        self.checkpoints[checkpoint.source] = checkpoint

    def poll_events(self, cursor: EventCursor, path: Path, *,
                    adapter: CausalEventAdapter | None = None) -> list[Event]:
        events = cursor.poll(path, adapter=adapter)
        self.ingest(events)
        self.issues.update(cursor.issues)
        return events

    def _rows(self, *, received_by_ns: int | None = None, occurred_by_ns: int | None = None):
        grouped = defaultdict(list)
        for event in self.events.values():
            if received_by_ns is not None and event.received_ns > received_by_ns:
                continue
            if occurred_by_ns is not None and event.occurred_ns > occurred_by_ns:
                continue
            if event.request_id:
                grouped[event.request_id].append(event)
        rows, problems = [], set()
        for request_id, events in sorted(grouped.items()):
            events.sort(key=lambda e: (e.occurred_ns, e.source, e.sequence))
            by_kind = defaultdict(list)
            for event in events:
                by_kind[event.kind].append(event)
            row: dict[str, Any] = {'run_id': self.run_id, 'episode_id': self.episode_id,
                                   'request_id': request_id}
            for kind in ('planned', 'sent', 'accepted', 'engine_first', 'api_first',
                         'client_first', 'server_terminal', 'client_terminal',
                         'cancel_requested', 'cancel_ack'):
                items = by_kind[kind]
                row[kind + '_ns'] = items[0].occurred_ns if items else None
                if len({(e.occurred_ns, canonical_hash(e.data)) for e in items}) > 1:
                    problems.add(f'conflicting_{kind}:{request_id}')
            if not by_kind['planned']:
                problems.add(f'unplanned_request:{request_id}')
            row['scheduled_ns'] = by_kind['planned'][0].data.get('scheduled_ns') if by_kind['planned'] else None
            server_ids = {e.server_request_id for e in events if e.server_request_id}
            row['server_request_id'] = next(iter(server_ids)) if len(server_ids) == 1 else None
            if len(server_ids) > 1:
                problems.add(f'ambiguous_server_mapping:{request_id}')
            if by_kind['accepted'] and not server_ids:
                problems.add(f'missing_server_mapping:{request_id}')
            high = 0
            outputs = []
            for event in by_kind['client_output']:
                if event.data.get('sampled', False):
                    problems.add(f'sampled_client_output:{request_id}')
                    continue
                count = event.data.get('cumulative_tokens')
                if type(count) is not int or count < 0:
                    problems.add(f'invalid_token_count:{request_id}')
                    continue
                if count < high:
                    problems.add(f'token_count_regression:{request_id}')
                if count > high:
                    outputs.append((event.occurred_ns, count - high))
                    high = count
            row['actual_output_tokens'] = high
            row['token_increments'] = outputs
            engine_times = [e.occurred_ns for e in events
                            if e.kind == 'engine_first' or
                            (e.data.get('event') == 'engine_request_output'
                             and e.data.get('output_tokens', 0) > 0
                             and not e.data.get('sampled', False))]
            if engine_times:
                row['engine_first_ns'] = min(engine_times)
            row['client_first_ns'] = outputs[0][0] if outputs else row['client_first_ns']
            row['client_last_ns'] = outputs[-1][0] if outputs else None
            row['chunk_times_ns'] = sorted({e.occurred_ns for e in by_kind['client_chunk']})
            sent, first, last, terminal = (row[k] for k in ('sent_ns', 'client_first_ns',
                                                          'client_last_ns', 'client_terminal_ns'))
            if sent is None and any(by_kind[k] for k in ('accepted', 'client_output', 'client_terminal')):
                problems.add(f'missing_sent:{request_id}')
            if sent is not None and (sent < self.start_ns or sent > self.deadline_ns):
                problems.add(f'send_outside_episode:{request_id}')
            if sent is not None and any(t is not None and t < sent for t in (first, terminal)):
                problems.add(f'client_time_order:{request_id}')
            if terminal is not None and last is not None and last > terminal:
                problems.add(f'output_after_terminal:{request_id}')
            row['ttft_ms'] = (first - sent) / 1e6 if first is not None and sent is not None else None
            row['tpot_ms'] = (last - first) / 1e6 / (high - 1) if high >= 2 and last is not None and first is not None else None
            terminal_event = by_kind['client_terminal'][0] if by_kind['client_terminal'] else None
            confirmed_cancel = any(e.data.get('confirmed') is True
                                   and row['cancel_requested_ns'] is not None
                                   and e.occurred_ns >= row['cancel_requested_ns']
                                   for e in by_kind['cancel_ack'])
            row['cancel_confirmed'] = confirmed_cancel
            row['manual_cancel_terminal'] = bool(terminal_event and
                terminal_event.data.get('reason') == 'manual_drain' and
                (terminal_event.data.get('http_status') == 499 or
                 terminal_event.data.get('outcome') == 'cancelled'))
            row['outcome'] = self._outcome(terminal_event, row, confirmed_cancel)
            row['score_outcome'] = ('not_sent' if sent is None else
                'censored' if terminal is None or terminal > self.deadline_ns else row['outcome'])
            row['cleanup_outcome'] = ('not_sent' if sent is None else
                                      row['outcome'] if terminal is not None else 'unresolved')
            row['server_outcome'] = by_kind['server_terminal'][0].data.get('outcome', 'unknown') if by_kind['server_terminal'] else None
            if row['outcome'] == 'success' and (first is None or last is None or high == 0):
                problems.add(f'success_without_complete_client_output:{request_id}')
            row['send_lateness_ms'] = (sent - row['scheduled_ns']) / 1e6 if sent is not None and row['scheduled_ns'] is not None else None
            row['event_times'] = [{'kind': e.kind, 'source': e.source, 'sequence': e.sequence,
                                   'occurred_ns': e.occurred_ns, 'received_ns': e.received_ns}
                                  for e in events]
            rows.append(row)
        # A server request must not count as two logical successes.
        owners = defaultdict(set)
        for row in rows:
            if row['server_request_id']:
                owners[row['server_request_id']].add(row['request_id'])
        for server_id, ids in owners.items():
            if len(ids) > 1:
                problems.add(f'server_id_reused:{server_id}')
        return rows, problems

    @staticmethod
    def _outcome(event: Event | None, row: dict, confirmed_cancel: bool) -> str | None:
        if event is None:
            return None
        outcome, http = event.data.get('outcome', 'unknown'), event.data.get('http_status')
        if http == 499 or outcome == 'cancelled':
            if (row['cancel_requested_ns'] is not None
                    and row['cancel_requested_ns'] <= event.occurred_ns
                    and event.data.get('reason') == 'manual_drain' and confirmed_cancel):
                return 'manual_cancel'
            return 'unknown'
        if outcome == 'success' and http in (None, 200):
            return 'success'
        if outcome in ('timeout', 'service_error'):
            return outcome
        return 'unknown'

    def observe(self, decision_ns: int) -> dict:
        rows, problems = self._rows(received_by_ns=decision_ns, occurred_by_ns=decision_ns)
        return {'as_of_ns': decision_ns, 'requests': rows, 'issues': sorted(problems),
                'outstanding': sum(r['sent_ns'] is not None and r['client_terminal_ns'] is None for r in rows)}

    def audit(self, *, through_ns: int | None = None, require_cleanup: bool = False) -> dict:
        through = self.deadline_ns if through_ns is None else through_ns
        rows, problems = self._rows(occurred_by_ns=through)
        problems.update(self.issues)
        planned_ids = {r['request_id'] for r in rows if r['planned_ns'] is not None}
        if self.expected_request_ids is None:
            problems.add('offered_request_manifest_missing')
        elif planned_ids != self.expected_request_ids:
            problems.add('offered_request_manifest_mismatch')
        gaps = {}
        for source in self.sources:
            seqs = {seq for (src, seq) in self.events if src == source}
            cp = self.checkpoints.get(source)
            last = max(max(seqs, default=0), cp.last_sequence if cp else 0)
            missing = sorted(set(range(1, last + 1)) - seqs)
            if missing:
                gaps[source] = missing
                problems.add(f'sequence_gap:{source}')
            if cp is None or not cp.durable or cp.through_ns < through or cp.last_sequence < max(seqs, default=0):
                problems.add(f'flush_unconfirmed:{source}')
            if require_cleanup and (cp is None or cp.pending_delivery != 0):
                problems.add(f'delivery_not_drained:{source}')
        for row in rows:
            if row['outcome'] == 'unknown':
                problems.add(f'unknown_client_terminal:{row["request_id"]}')
            if require_cleanup and row['sent_ns'] is not None:
                if row['client_terminal_ns'] is None:
                    problems.add(f'missing_client_terminal:{row["request_id"]}')
                if row['accepted_ns'] is not None and row['server_terminal_ns'] is None:
                    problems.add(f'missing_server_terminal:{row["request_id"]}')
                if row['server_outcome'] is not None and row['server_outcome'] not in ('success', 'cancelled', 'service_error', 'timeout', 'not_accepted'):
                    problems.add(f'unknown_server_terminal:{row["request_id"]}')
                if row['manual_cancel_terminal'] and not any(
                        e.kind == 'cancel_ack' and e.request_id == row['request_id']
                        and e.data.get('confirmed') is True
                        and e.occurred_ns >= row['cancel_requested_ns']
                        and e.occurred_ns <= through for e in self.events.values()):
                    problems.add(f'cancel_unconfirmed:{row["request_id"]}')
        return {'status': 'incomplete' if problems else 'passed', 'issues': sorted(problems),
                'sequence_gaps': gaps, 'request_count': len(rows), 'through_ns': through,
                'require_cleanup': require_cleanup,
                'offered_request_ids': sorted(self.expected_request_ids) if self.expected_request_ids is not None else None}

    def _window(self, rows: list[dict], lo: int, hi: int) -> dict:
        duration = (hi - lo) / NS
        completions, good, tokens, completed_tokens, hold, first_hold, gap_hold = 0, 0, 0, 0, 0., 0., 0.
        ttfts, tpots, chunk_gaps, longest_silences = [], [], [], []
        one_token = 0
        for row in rows:
            sent, terminal, first = row['sent_ns'], row['client_terminal_ns'], row['client_first_ns']
            if sent is None:
                continue
            end = min(terminal if terminal is not None else hi, hi)
            hold += _overlap(sent, end, lo, hi)
            first_hold += _overlap(sent + int(self.ttft_slo_ms * 1e6),
                                   min(first if first is not None else end, end), lo, hi)
            output_times = [t for t, _ in row['token_increments'] if t <= end]
            gap_limit = int(max(1., 5 * self.tpot_slo_ms / 1000) * NS)
            for index, t in enumerate(output_times):
                next_t = output_times[index + 1] if index + 1 < len(output_times) else end
                gap_hold += _overlap(t + gap_limit, next_t, lo, hi)
            tokens += sum(n for t, n in row['token_increments'] if lo <= t <= hi)
            if first is not None and lo <= first <= hi and row['ttft_ms'] is not None:
                ttfts.append(row['ttft_ms'])
            chunks = [t for t in row['chunk_times_ns'] if lo <= t <= hi]
            chunk_gaps.extend((b-a)/1e6 for a,b in zip(chunks, chunks[1:]))
            if output_times:
                longest_silences.append(max(_overlap(a,b,lo,hi) for a,b in
                    zip(output_times, output_times[1:] + [end])) * 1000)
            if terminal is not None and lo <= terminal <= hi and row['outcome'] == 'success':
                completions += 1
                completed_tokens += row['actual_output_tokens']
                if row['actual_output_tokens'] == 1:
                    one_token += 1
                if row['tpot_ms'] is not None:
                    tpots.append(row['tpot_ms'])
                if (row['ttft_ms'] is not None and row['ttft_ms'] <= self.ttft_slo_ms
                        and (row['actual_output_tokens'] == 1 or
                             row['tpot_ms'] is not None and row['tpot_ms'] <= self.tpot_slo_ms)):
                    good += 1
        offered = len(rows)
        sent_rows = [r for r in rows if r['sent_ns'] is not None and r['sent_ns'] <= hi]
        outstanding = [r['request_id'] for r in sent_rows if r['client_terminal_ns'] is None or r['client_terminal_ns'] > hi]
        return {'start_ns': lo, 'end_ns': hi, 'duration_sec': duration,
                'offered': offered, 'sent': len(sent_rows), 'completed': completions,
                'completion_rate': completions / offered if offered else None,
                'slo_completed': good, 'slo_goodput_rps': good / duration,
                'client_output_tokens': tokens, 'client_token_throughput_tps': tokens / duration,
                'completed_output_tokens': completed_tokens, 'single_token_completed': one_token,
                'ttft_conditional_ms': ttfts, 'ttft_p95_ms': _percentile(ttfts, .95),
                'tpot_conditional_ms': tpots, 'tpot_p95_ms': _percentile(tpots, .95),
                'chunk_intervals_ms': chunk_gaps, 'longest_no_output_ms': max(longest_silences, default=None),
                'exact_token_itl_available': False,
                'outstanding_request_seconds': hold, 'overdue_first_request_seconds': first_hold,
                'output_gap_request_seconds': gap_hold, 'outstanding_ids': outstanding,
                'first_output_censored': sum(r['client_first_ns'] is None or r['client_first_ns'] > hi for r in sent_rows),
                'terminal_outcomes': dict(Counter(r['outcome'] for r in sent_rows
                    if r['client_terminal_ns'] is not None and lo <= r['client_terminal_ns'] <= hi))}

    def freeze_score(self) -> dict:
        if self._frozen is not None:
            return deepcopy(self._frozen)
        audit = self.audit()
        # Ignore cleanup occurrences, even when already collected before finalisation.
        rows, _ = self._rows(occurred_by_ns=self.deadline_ns)
        score = {'status': audit['status'], 'score_deadline_ns': self.deadline_ns,
                 'arrival': self._window(rows, self.start_ns, self.arrival_end_ns),
                 'fixed': self._window(rows, self.start_ns, self.deadline_ns), 'audit': audit}
        if audit['status'] == 'passed':
            self._frozen = deepcopy(score)
        return score

    def export(self, directory: Path, *, cleanup_through_ns: int) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        rows, _ = self._rows()
        score = self.freeze_score()
        audit = self.audit(through_ns=cleanup_through_ns, require_cleanup=True)
        late = [r for r in rows if r['client_terminal_ns'] is not None and r['client_terminal_ns'] > self.deadline_ns]
        for name, payload in [('window_metrics.json', score), ('audit.json', audit),
                              ('checkpoints.json', [asdict(cp) for cp in self.checkpoints.values()]),
                              ('producer_manifest.json', {'run_id': self.run_id, 'episode_id': self.episode_id,
                                  'expected_sources': sorted(self.sources),
                                  'expected_request_ids': sorted(self.expected_request_ids) if self.expected_request_ids is not None else None,
                                  'start_ns': self.start_ns, 'arrival_end_ns': self.arrival_end_ns,
                                  'score_deadline_ns': self.deadline_ns,
                                  'ttft_slo_ms': self.ttft_slo_ms, 'tpot_slo_ms': self.tpot_slo_ms}),
                              ('cleanup.json', {'late_completed': sum(r['outcome']=='success' for r in late),
                                                'late_requests': late})]:
            (directory / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        for name, payload in [('requests.jsonl', rows), ('events.jsonl', [asdict(e) for e in self.events.values()])]:
            (directory / name).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in payload))
        return {'score_status': score['status'], 'cleanup_status': audit['status']}


class CausalEventAdapter:
    """Adapt whole causal process streams without discarding sequence positions.

    Supply the explicit server->logical mapping before replay. Sampled forwarder/
    queue records stay diagnostic. Legacy logs lack durable watermarks and hence
    cannot pass completeness simply because this adapter can read them.
    """
    def __init__(self, run_id: str, episode_id: str, request_map: dict[str, str]):
        self.run_id, self.episode_id = run_id, episode_id
        self.request_map = dict(request_map)
        self._first: set[tuple[str, str]] = set()

    def adapt(self, raw: dict, *, received_ns: int) -> Event:
        source = raw.get('stream_id') or f'legacy:{raw.get("host")}:{raw.get("pid")}'
        server_id = raw.get('server_request_id') or raw.get('request_id')
        logical_id = self.request_map.get(server_id)
        kind, data = 'diagnostic', dict(raw)
        if (raw.get('event') == 'real_metrics_event'
                and raw.get('run_id') == self.run_id and raw.get('episode_id') == self.episode_id):
            logical_id = raw['request_id']
            kind = raw['metric_kind']
            data = raw.get('data', {})
        elif logical_id is not None and raw.get('event') == 'engine_request_output':
            key = ('engine_first', logical_id)
            if raw.get('output_tokens', 0) > 0 and not raw.get('sampled', False) and key not in self._first:
                kind = 'engine_first'
                self._first.add(key)
        # Other sampled updates are diagnostics, never a client output stream.
        return Event(source, raw['event_sequence'], kind, raw['monotonic_ns'], received_ns,
                     self.run_id, self.episode_id, logical_id, server_id if logical_id else None, data)


def migration_diagnostics(raw_events: Iterable[dict]) -> list[dict]:
    """Pair directly measured migration stages; missing pairs remain None.

    Recovery here means commit-ack -> first engine output at the destination,
    not client-visible recovery. No pause is inferred from total migration time.
    """
    events = list(raw_events)
    grouped = defaultdict(list)
    for event in events:
        if event.get('migration_id'):
            grouped[event['migration_id']].append(event)
    result = []
    for migration_id, records in sorted(grouped.items()):
        records.sort(key=lambda e: e['monotonic_ns'])
        by_kind = defaultdict(list)
        for event in records:
            by_kind[event['event']].append(event)
        def first(kind):
            return by_kind[kind][0]['monotonic_ns'] if by_kind[kind] else None
        def interval(a, b):
            return (b-a)/1e6 if a is not None and b is not None and b >= a else None
        start, end = first('migration_attempt_started'), first('migration_attempt_completed')
        pause = first('migration_request_pause_started')
        commit = first('migration_target_commit_started')
        resumes = [e for e in by_kind['migration_request_resumed']
                   if e.get('resume_reason') == 'destination_commit_acknowledged']
        resumed = resumes[0]['monotonic_ns'] if resumes else None
        request_id = records[0].get('request_id')
        target = records[0].get('target_instance_id')
        outputs = [e['monotonic_ns'] for e in events if e.get('event') == 'engine_request_output'
                   and e.get('request_id') == request_id and e.get('instance_id') == target
                   and e.get('output_tokens', 0) > 0 and not e.get('sampled', False)
                   and resumed is not None and e['monotonic_ns'] >= resumed]
        result.append({'migration_id': migration_id, 'total_ms': interval(start, end),
                       'source_pause_ms': interval(pause, resumed),
                       'target_commit_ms': interval(commit, resumed),
                       'post_commit_first_engine_output_ms': interval(resumed, min(outputs) if outputs else None),
                       'client_recovery_ms': None})
    return result
