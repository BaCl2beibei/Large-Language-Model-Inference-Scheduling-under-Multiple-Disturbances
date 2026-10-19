"""Loopback-only TCP proxy for measured HTTP delay/failure experiments."""

from __future__ import annotations

import asyncio
import random
from urllib.parse import urlparse

from .base import Injector


class LocalHttpProxy(Injector):
    name = "network_proxy"

    def __init__(self, upstream_url: str, host: str = "127.0.0.1", port: int = 0,
                 request_delay_ms: float = 0, response_delay_ms: float = 0,
                 jitter_ms: float = 0, failure_probability: float = 0, seed: int = 42):
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("proxy may only bind to loopback")
        parsed = urlparse(upstream_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("proxy upstream must be a loopback HTTP endpoint")
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port or 80
        self.host, self.port = host, int(port)
        self.request_delay_ms = max(0.0, float(request_delay_ms))
        self.response_delay_ms = max(0.0, float(response_delay_ms))
        self.jitter_ms = max(0.0, float(jitter_ms))
        self.failure_probability = max(0.0, min(float(failure_probability), 0.05))
        self.rng = random.Random(seed)
        self.server = None
        self.connections = 0
        self.injected_failures = 0

    async def start(self, **kwargs) -> dict:
        if self.server:
            return self.status()
        self.server = await asyncio.start_server(self._handle, self.host, self.port, limit=65536)
        sockets = self.server.sockets or []
        if sockets:
            self.port = int(sockets[0].getsockname()[1])
        return self.status()

    def url_for(self, endpoint: str) -> str:
        return f"http://127.0.0.1:{self.port}{endpoint}"

    def _delay(self, base_ms: float) -> float:
        jitter = self.rng.uniform(-self.jitter_ms, self.jitter_ms) if self.jitter_ms else 0.0
        return max(0.0, base_ms + jitter) / 1000.0

    async def _copy(self, source, dest, delay_ms: float):
        while True:
            data = await source.read(65536)
            if not data:
                break
            delay = self._delay(delay_ms)
            if delay:
                await asyncio.sleep(delay)
            dest.write(data)
            await dest.drain()
        try:
            dest.write_eof()
        except (AttributeError, OSError, RuntimeError):
            pass

    async def _handle(self, client_reader, client_writer):
        self.connections += 1
        if self.rng.random() < self.failure_probability:
            self.injected_failures += 1
            client_writer.close()
            await client_writer.wait_closed()
            return
        try:
            delay = self._delay(self.request_delay_ms)
            if delay:
                await asyncio.sleep(delay)
            upstream_reader, upstream_writer = await asyncio.open_connection(
                self.upstream_host, self.upstream_port, limit=65536
            )
            tasks = [
                asyncio.create_task(self._copy(client_reader, upstream_writer, 0)),
                asyncio.create_task(self._copy(upstream_reader, client_writer, self.response_delay_ms)),
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            upstream_writer.close()
            await upstream_writer.wait_closed()
        except (OSError, asyncio.CancelledError):
            pass
        finally:
            client_writer.close()
            try:
                await client_writer.wait_closed()
            except OSError:
                pass

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    def status(self) -> dict:
        return {"active": self.server is not None, "bind_host": self.host, "port": self.port,
                "connections": self.connections, "injected_failures": self.injected_failures,
                "classification": "proxy_api_failure_not_packet_loss"}
