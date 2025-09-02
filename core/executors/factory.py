from __future__ import annotations

import logging
from typing import Dict, Type

from .base import BaseExecutor
from .builtins import GreetingExecutor, SlotFillingExecutor, DefaultExecutor

logger = logging.getLogger(__name__)


class ExecutorFactory:
    """Simple factory for creating executors (built-ins only)."""

    def __init__(self) -> None:
        self.executors: Dict[str, Type[BaseExecutor]] = {
            'initial': GreetingExecutor,
            'greeting': GreetingExecutor,
            'start': GreetingExecutor,
            'slot_filling': SlotFillingExecutor,
            'info_collection': SlotFillingExecutor,
            'confirmation': DefaultExecutor,
            'final_confirmation': DefaultExecutor,
            'final': DefaultExecutor,
            'completion': DefaultExecutor,
            'general_chat': DefaultExecutor,
            'default': DefaultExecutor,
        }
        self._cache: Dict[str, BaseExecutor] = {}

    def get(self, stage: str) -> BaseExecutor:
        if stage not in self._cache:
            executor_cls = self.executors.get(stage, DefaultExecutor)
            self._cache[stage] = executor_cls()
        return self._cache[stage]

    def register(self, stage: str, executor_class: Type[BaseExecutor]):
        self.executors[stage] = executor_class
        self._cache.pop(stage, None)


executor_factory = ExecutorFactory() 