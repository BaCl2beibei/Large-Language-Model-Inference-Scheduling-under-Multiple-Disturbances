import asyncio
import unittest

from disturbance_exp.injectors.cpu import CpuLoadInjector


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_owned_cpu_process_stops_on_explicit_cleanup(self):
        injector = CpuLoadInjector()
        injector.start(workers=1, duty_cycle=0.1)
        await asyncio.sleep(0.1)
        injector.stop()
        self.assertFalse(injector.status()["active"])


if __name__ == "__main__":
    unittest.main()
