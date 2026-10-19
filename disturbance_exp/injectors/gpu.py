"""Small, memory-bounded CUDA matmul load with a conservative free-memory gate."""

from __future__ import annotations

import multiprocessing as mp
import time

from .base import Injector


def _gpu_worker(stop_event, device_id: int, duty_cycle: float, memory_limit_mib: int):
    import torch

    torch.set_num_threads(1)
    torch.cuda.set_device(device_id)
    side = max(128, min(512, int((memory_limit_mib * 1024 * 1024 / 16) ** 0.5)))
    a = torch.randn((side, side), device=f"cuda:{device_id}")
    b = torch.randn_like(a)
    period = 0.5
    while not stop_event.is_set():
        end = time.monotonic() + period * duty_cycle
        while time.monotonic() < end and not stop_event.is_set():
            torch.mm(a, b)
        torch.cuda.synchronize(device_id)
        stop_event.wait(max(0, period * (1 - duty_cycle)))
    del a, b


class GpuLoadInjector(Injector):
    name = "gpu_compute"

    def __init__(self, minimum_free_mib: int = 3072):
        self.minimum_free_mib = int(minimum_free_mib)
        self._ctx = mp.get_context("spawn") if "spawn" in mp.get_all_start_methods() else mp.get_context()
        self._event = None
        self._process = None
        self.last_result = {"supported": False, "reason": "not started"}

    @staticmethod
    def _free_memory(device_id: int) -> int | None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free // (1024 * 1024))
        except Exception:
            return None

    def start(self, device_id: int = 0, duty_cycle: float = 0.1,
              memory_limit_mib: int = 32, **kwargs) -> dict:
        self.stop()
        free_mib = self._free_memory(int(device_id))
        if free_mib is None:
            self.last_result = {"supported": False, "reason": "NVML unavailable; memory gate cannot be checked"}
            return self.last_result
        if free_mib < self.minimum_free_mib:
            self.last_result = {
                "supported": False,
                "reason": f"only {free_mib} MiB free; require {self.minimum_free_mib} MiB safety reserve",
                "free_mib": free_mib,
            }
            return self.last_result
        try:
            import torch

            if not torch.cuda.is_available() or int(device_id) >= torch.cuda.device_count():
                self.last_result = {"supported": False, "reason": "requested CUDA device is unavailable"}
                return self.last_result
        except Exception as exc:
            self.last_result = {"supported": False, "reason": f"CUDA check failed: {type(exc).__name__}"}
            return self.last_result
        memory_limit_mib = max(1, min(int(memory_limit_mib), 256))
        self._event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_gpu_worker,
            args=(self._event, int(device_id), max(0.01, min(float(duty_cycle), 0.5)), memory_limit_mib),
            name="disturbance-gpu",
            daemon=True,
        )
        self._process.start()
        time.sleep(0.25)
        if not self._process.is_alive():
            self.stop()
            self.last_result = {"supported": False, "reason": "CUDA worker exited during startup", "free_mib": free_mib}
            return self.last_result
        self.last_result = {"supported": True, "free_mib": free_mib, **self.status()}
        return self.last_result

    def stop(self):
        if self._event is not None:
            self._event.set()
        if self._process is not None:
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
        self._event = None
        self._process = None

    def status(self) -> dict:
        return {"active": bool(self._process and self._process.is_alive()),
                "pid": self._process.pid if self._process else None}
