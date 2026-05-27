from __future__ import annotations
import math


def sanitize_fitness(
    value: float,
    fallback: float = 0.0,
    clip_low: float | None = None,
    clip_high: float | None = None,
) -> tuple[float, bool, bool]:
    """Sanitize a raw fitness value.

    Returns (sanitized_value, was_invalid, was_clipped).

    was_invalid: True when value was nan or inf (replaced by fallback).
    was_clipped: True when value was finite but outside [clip_low, clip_high].
    """
    if not math.isfinite(value):
        return fallback, True, False
    clipped = False
    if clip_low is not None and value < clip_low:
        value = clip_low
        clipped = True
    if clip_high is not None and value > clip_high:
        value = clip_high
        clipped = True
    return value, False, clipped


class FitnessSanitizer:
    """Owns fitness sanitizing state and applies it to raw fitness values."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.fallback: float = 0.0
        self.clip_low: float | None = None
        self.clip_high: float | None = None
        self.n_invalid: int = 0
        self.n_clipped: int = 0

    def configure(
        self,
        fallback: float = 0.0,
        clip_low: float | None = None,
        clip_high: float | None = None,
    ) -> None:
        self.enabled = True
        self.fallback = fallback
        self.clip_low = clip_low
        self.clip_high = clip_high

    def apply(self, fitness: float) -> float:
        """Apply sanitizing; update counters and log anomalies. No-op when disabled."""
        if not self.enabled:
            return fitness
        clean, invalid, clipped = sanitize_fitness(
            fitness, self.fallback, self.clip_low, self.clip_high
        )
        if invalid:
            self.n_invalid += 1
            from yane.util.logger import get_logger
            get_logger().warning(
                "sanitize_fitness: invalid value %r replaced with fallback %r "
                "(total invalid: %d)",
                fitness, self.fallback, self.n_invalid,
            )
        elif clipped:
            self.n_clipped += 1
        return clean
