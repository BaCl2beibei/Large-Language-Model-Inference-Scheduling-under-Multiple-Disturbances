from __future__ import annotations

from abc import ABC, abstractmethod


class Injector(ABC):
    name = "injector"

    @abstractmethod
    def start(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def status(self) -> dict:
        raise NotImplementedError
