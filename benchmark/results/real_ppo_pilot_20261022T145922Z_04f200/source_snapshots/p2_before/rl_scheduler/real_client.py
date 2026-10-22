"""Opt-in streaming metrics client; no retry, model loading, or cluster lifecycle.

Wire records are NUL-delimited JSON, independent of HTTP packet boundaries.
Only cumulative token counts from decoded records measure tokens. The first/last
positive update is client-visible; server frame timestamps stay diagnostic.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .real_metrics import EventJournal


class NulJSONDecoder:
    def __init__(self, max_record_bytes: int = 8 * 1024 * 1024):
        self.buffer = b''
        self.max_record_bytes = max_record_bytes

    def feed(self, data: bytes) -> list[dict]:
        self.buffer += data
        records = []
        while b'\0' in self.buffer:
            raw, self.buffer = self.buffer.split(b'\0', 1)
            if len(raw) > self.max_record_bytes:
                raise ValueError('stream record too large')
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError('stream record is not an object')
            records.append(record)
        if len(self.buffer) > self.max_record_bytes:
            raise ValueError('stream record too large')
        return records

    def finish(self) -> None:
        if self.buffer:
            raise ValueError('incomplete stream record')


class StreamingRequestCollector:
    def __init__(self, journal: EventJournal):
        self.journal = journal
        self.cancel_requested: set[str] = set()
        self.sent: set[str] = set()

    def request_cancel(self, request_id: str) -> None:
        self.cancel_requested.add(request_id)
        self.journal.record('cancel_requested', request_id, reason='manual_drain')

    async def issue(self, session: Any, url: str, request_id: str, body: dict) -> None:
        """Caller registers planned events first and owns a finite HTTP timeout.

        Local task cancellation ends the client handler, but cannot confirm backend
        cancellation. A later cancel_ack must come from an explicit server barrier.
        """
        if request_id in self.sent:
            raise ValueError('automatic retry is forbidden; request already sent')
        self.sent.add(request_id)
        self.journal.record('sent', request_id)
        decoder, high, terminal, first, server_id = NulJSONDecoder(), 0, None, False, None
        try:
            payload = {**body, 'request_id': request_id, 'run_id': self.journal.run_id,
                       'episode_id': self.journal.episode_id}
            async with session.post(url, json=payload) as response:
                status = response.status
                if status != 200:
                    await response.read()
                    terminal = {'outcome': 'cancelled' if status == 499 else 'service_error',
                                'http_status': status,
                                'reason': 'manual_drain' if status == 499 and request_id in self.cancel_requested else 'http_error'}
                else:
                    async for packet in response.content.iter_any():
                        self.journal.record('client_chunk', request_id, bytes=len(packet))
                        for record in decoder.feed(packet):
                            if (record.get('run_id'), record.get('episode_id'), record.get('request_id')) != (
                                    self.journal.run_id, self.journal.episode_id, request_id):
                                raise ValueError('stream identity mismatch')
                            mapped_id = record.get('server_request_id')
                            if not mapped_id or server_id is not None and server_id != mapped_id:
                                raise ValueError('missing or changed server request mapping')
                            server_id = mapped_id
                            kind = record.get('kind')
                            if terminal is not None:
                                raise ValueError('record after terminal')
                            if kind == 'accepted':
                                self.journal.record('diagnostic', request_id, server_request_id=server_id,
                                                    stage='accepted_frame_received')
                            elif kind == 'output':
                                count = record.get('cumulative_tokens')
                                if type(count) is not int or count < 0:
                                    raise ValueError('invalid cumulative token count')
                                # Record duplicate/regression evidence too; ledger audits it.
                                first_event = None
                                if count > high and not first:
                                    first_event = self.journal.record('client_first', request_id, server_request_id=server_id)
                                    first = True
                                self.journal.record('client_output', request_id,
                                                    occurred_ns=first_event.occurred_ns if first_event else None,
                                                    server_request_id=server_id, cumulative_tokens=count)
                                high = max(high, count)
                            elif kind == 'terminal':
                                terminal = {'outcome': record.get('outcome', 'unknown'), 'http_status': 200,
                                            'reason': record.get('reason')}
                            else:
                                raise ValueError('unknown frame kind')
                    decoder.finish()
                    if terminal is None:
                        terminal = {'outcome': 'unknown', 'reason': 'EOF_without_terminal', 'http_status': 200}
        except asyncio.TimeoutError:
            terminal = {'outcome': 'timeout', 'reason': 'client_timeout'}
        except asyncio.CancelledError:
            terminal = {'outcome': 'cancelled', 'reason': 'manual_drain' if request_id in self.cancel_requested else 'unconfirmed_local_cancel'}
            raise
        except Exception as exc:
            terminal = {'outcome': 'unknown', 'reason': type(exc).__name__}
        finally:
            self.journal.record('client_terminal', request_id, server_request_id=server_id,
                                **(terminal or {'outcome': 'unknown'}))

    async def cancel(self, session: Any, url: str, request_ids: list[str]) -> dict:
        import aiohttp
        for request_id in request_ids:
            if request_id not in self.cancel_requested:
                self.request_cancel(request_id)
        # This is a separate cleanup timeout, never the old 900-second barrier.
        async with session.post(url, json={'run_id': self.journal.run_id,
                                          'episode_id': self.journal.episode_id,
                                          'request_ids': request_ids},
                                timeout=aiohttp.ClientTimeout(total=60)) as response:
            result = await response.json()
            for row in result.get('results', []):
                if row.get('request_id') not in request_ids:
                    raise ValueError('foreign cancellation acknowledgement')
                self.journal.record('cancel_ack', row['request_id'],
                                    confirmed=row.get('status') == 'cancelled_confirmed',
                                    server_request_id=row.get('server_request_id'),
                                    status=row.get('status'))
            return result
