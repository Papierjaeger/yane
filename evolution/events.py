"""Lightweight publish/subscribe event bus for YANE training events.

Events fired by NeuroEvolution.train():
    "generation_end"  — payload: diagnostics dict + "iteration" key
    "new_best"        — payload: {"genome": Genome, "fitness": float, "iteration": int}
    "stagnation"      — payload: {"stagnation_count": int, "iteration": int}
    "run_end"         — payload: {"stop_reason": str, "iterations": int}
    "anomaly"         — payload: AnomalyReport dict

User code may emit arbitrary events via emit().
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class EventBus:
    """Thread-unsafe, synchronous event bus.

    Handlers are called in registration order. Exceptions inside handlers are
    silently swallowed so one bad callback cannot interrupt training.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, fn: Callable) -> None:
        """Register *fn* as a handler for *event* (idempotent)."""
        handlers = self._handlers[event]
        if fn not in handlers:
            handlers.append(fn)

    def off(self, event: str, fn: Callable) -> None:
        """Unregister *fn* from *event* (no-op if not registered)."""
        try:
            self._handlers[event].remove(fn)
        except ValueError:
            pass

    def emit(self, event: str, payload: Any = None) -> None:
        """Fire *event*, calling all registered handlers with *payload*."""
        for fn in list(self._handlers.get(event, [])):
            try:
                fn(payload)
            except Exception:
                pass

    def clear(self, event: str | None = None) -> None:
        """Remove all handlers for *event*, or every handler when *event* is None."""
        if event is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event, None)

    def handler_count(self, event: str) -> int:
        return len(self._handlers.get(event, []))
