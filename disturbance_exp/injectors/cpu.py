"""Low-priority, duty-cycled CPU load in experiment-owned child processes."""

from __future__ import annotations

import multiprocessing as mp
import os
import time

from .base import Injector


def _cpu_worker(stop_event, duty_cycle: float) -> None:
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    period = 0.5
    while not stop_event.is_set():
        work_until = time.monotonic() + period * duty_cycle
        value = 17
        while time.monotonic() < work_until and not stop_event.is_set():
            value = (value * 1103515245 + 12345) & 0x7FFFFFFF
        stop_event.wait(max(0.0, period * (1.0 - duty_cycle)))


class CpuLoadInjector(Injector):
    name = "cpu_compute"

    def __init__(self):
        self._processes = []
        self._events = []
        self._ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()

    def start(self, workers: int = 1, duty_cycle: float = 0.25, **kwargs) -> dict:
        self.stop()
        workers = max(1, min(int(workers), 2))
        duty_cycle = max(0.01, min(float(duty_cycle), 0.5))
        for index in range(workers):
            stop_event = self._ctx.Event()
            proc = self._ctx.Process(target=_cpu_worker, args=(stop_event, duty_cycle), name=f"disturbance-cpu-{index}")
            proc.daemon = True
            proc.start()
            self._events.append(stop_event)
            self._processes.append(proc)
        return self.status()

    def stop(self):
        for event in self._events:
            event.set()
        for proc in self._processes:
            proc.join(timeout=1.5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        self._processes.clear()
        self._events.clear()

    def status(self) -> dict:
        alive = [proc for proc in self._processes if proc.is_alive()]
        return {"active": bool(alive), "workers": len(alive), "pids": [p.pid for p in alive]}
