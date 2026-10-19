import asyncio
import time
import unittest

from disturbance_exp.injectors.cpu import CpuLoadInjector
from disturbance_exp.injectors.fault import MockFaultInjector
from disturbance_exp.injectors.network import LocalHttpProxy


class InjectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_adds_request_delay_and_stops(self):
        async def upstream(reader, writer):
            data = await reader.read(1024)
            writer.write(data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(upstream, "127.0.0.1", 0)
        upstream_port = server.sockets[0].getsockname()[1]
        proxy = LocalHttpProxy(f"http://127.0.0.1:{upstream_port}", request_delay_ms=50)
        await proxy.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
            started = time.monotonic()
            writer.write(b"probe")
            await writer.drain()
            self.assertEqual(await asyncio.wait_for(reader.read(5), timeout=2), b"probe")
            self.assertGreaterEqual(time.monotonic() - started, 0.045)
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.stop()
            server.close()
            await server.wait_closed()
        self.assertFalse(proxy.status()["active"])

    async def test_cpu_worker_cleanup(self):
        injector = CpuLoadInjector()
        result = injector.start(workers=1, duty_cycle=0.1)
        self.assertTrue(result["active"])
        injector.stop()
        self.assertFalse(injector.status()["active"])

    async def test_fault_is_labeled_as_mock(self):
        injector = MockFaultInjector(seed=1)
        state = injector.start(failure_probability=0.05)
        self.assertEqual(state["classification"], "api_level_mock")
        self.assertTrue(injector.status()["active"])
        injector.stop()
        self.assertFalse(injector.status()["active"])


if __name__ == "__main__":
    unittest.main()
