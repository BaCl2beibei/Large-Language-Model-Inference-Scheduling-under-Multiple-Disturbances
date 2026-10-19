"""API-level fault simulation; never stops an Llumnix or Ray process."""

from __future__ import annotations

import random

from .base import Injector


class MockFaultInjector(Injector):
    name = "fault_mock"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.failure_probability = 0.0
        self.active = False
        self.injected = 0

    def start(self, failure_probability: float = 0.01, **kwargs) -> dict:
        self.failure_probability = max(0.0, min(float(failure_probability), 0.05))
        self.injected = 0
        self.active = True
        return self.status()

    def stop(self):
        self.active = False
        self.failure_probability = 0.0

    def should_fail(self) -> bool:
        fail = self.active and self.rng.random() < self.failure_probability
        self.injected += int(fail)
        return fail

    def status(self) -> dict:
        return {"active": self.active, "failure_probability": self.failure_probability,
                "injected_failures": self.injected, "classification": "api_level_mock"}
