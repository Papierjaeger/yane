"""Normalization helpers for dataset-style examples.

All normalizers share the same interface:

    normalize_input(values)   → list[float]
    normalize_output(values)  → list[float]
    denormalize_input(values) → list[float]
    denormalize_output(values)→ list[float]
    normalize_samples(samples)→ list[dict]

They are plain Python objects and fully picklable, so they can be stored
alongside checkpoints via NeuroEvolution.set_normalizer() /
NeuroEvolution.get_normalizer().
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _broadcast(scale_or_pair: tuple, n: int) -> list:
    """Return a list of length n, broadcasting the last element if needed."""
    s = list(scale_or_pair)
    if len(s) >= n:
        return s[:n]
    return s + [s[-1]] * (n - len(s))


def _normalize_samples(normalizer, samples: Iterable[dict]) -> list[dict[str, list[float]]]:
    return [
        {
            "input": normalizer.normalize_input(sample["input"]),
            "output": normalizer.normalize_output(sample["output"]),
        }
        for sample in samples
    ]


# ---------------------------------------------------------------------------
# ScaleNormalizer  (original)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleNormalizer:
    """Per-channel division/multiplication normalizer.

    Values are mapped by division on input and multiplication on output.
    Mirrors the GUI's input_scale / output_scale convention so examples
    and Inspect can share the same metadata.
    """

    input_scale: tuple[float, ...] | None = None
    output_scale: tuple[float, ...] | None = None

    @staticmethod
    def _apply(values: Sequence[float], scale: tuple[float, ...] | None, op) -> list[float]:
        if not scale:
            return [float(v) for v in values]
        return [
            op(float(v), float(scale[min(i, len(scale) - 1)]))
            for i, v in enumerate(values)
        ]

    def normalize_input(self, values: Sequence[float]) -> list[float]:
        return self._apply(values, self.input_scale, lambda v, s: v / s)

    def normalize_output(self, values: Sequence[float]) -> list[float]:
        return self._apply(values, self.output_scale, lambda v, s: v / s)

    def denormalize_input(self, values: Sequence[float]) -> list[float]:
        return self._apply(values, self.input_scale, lambda v, s: v * s)

    def denormalize_output(self, values: Sequence[float]) -> list[float]:
        return self._apply(values, self.output_scale, lambda v, s: v * s)

    def normalize_samples(self, samples: Iterable[dict]) -> list[dict[str, list[float]]]:
        return _normalize_samples(self, samples)

    @property
    def input_scale_list(self) -> list[float] | None:
        return list(self.input_scale) if self.input_scale else None

    @property
    def output_scale_list(self) -> list[float] | None:
        return list(self.output_scale) if self.output_scale else None


# ---------------------------------------------------------------------------
# MinMaxNormalizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinMaxNormalizer:
    """Maps each channel from [data_min, data_max] → [0, 1].

    Args:
        input_range:  list/tuple of (min, max) per input channel.
                      A single (min, max) pair is broadcast to all channels.
        output_range: list/tuple of (min, max) per output channel.

    Example::

        n = MinMaxNormalizer(input_range=[(0, 255)] * 3, output_range=[(0, 1)])
    """

    input_range: tuple[tuple[float, float], ...] = ()
    output_range: tuple[tuple[float, float], ...] = ()

    @staticmethod
    def _norm(values: Sequence[float],
              ranges: tuple[tuple[float, float], ...]) -> list[float]:
        if not ranges:
            return [float(v) for v in values]
        out = []
        for i, v in enumerate(values):
            lo, hi = ranges[min(i, len(ranges) - 1)]
            span = hi - lo
            out.append((float(v) - lo) / span if span > 0 else 0.0)
        return out

    @staticmethod
    def _denorm(values: Sequence[float],
                ranges: tuple[tuple[float, float], ...]) -> list[float]:
        if not ranges:
            return [float(v) for v in values]
        out = []
        for i, v in enumerate(values):
            lo, hi = ranges[min(i, len(ranges) - 1)]
            out.append(float(v) * (hi - lo) + lo)
        return out

    def normalize_input(self, values: Sequence[float]) -> list[float]:
        return self._norm(values, self.input_range)

    def normalize_output(self, values: Sequence[float]) -> list[float]:
        return self._norm(values, self.output_range)

    def denormalize_input(self, values: Sequence[float]) -> list[float]:
        return self._denorm(values, self.input_range)

    def denormalize_output(self, values: Sequence[float]) -> list[float]:
        return self._denorm(values, self.output_range)

    def normalize_samples(self, samples: Iterable[dict]) -> list[dict[str, list[float]]]:
        return _normalize_samples(self, samples)


# ---------------------------------------------------------------------------
# ZScoreNormalizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZScoreNormalizer:
    """Maps each channel to zero mean, unit variance: (x - mean) / std.

    Args:
        input_stats:  list/tuple of (mean, std) per input channel.
        output_stats: list/tuple of (mean, std) per output channel.
        eps:          minimum std to avoid division by zero (default 1e-8).

    Example::

        n = ZScoreNormalizer(
            input_stats=[(0.5, 0.25), (0.0, 1.0)],
            output_stats=[(10.0, 5.0)],
        )
    """

    input_stats: tuple[tuple[float, float], ...] = ()
    output_stats: tuple[tuple[float, float], ...] = ()
    eps: float = 1e-8

    @staticmethod
    def _norm(values: Sequence[float],
              stats: tuple[tuple[float, float], ...],
              eps: float) -> list[float]:
        if not stats:
            return [float(v) for v in values]
        out = []
        for i, v in enumerate(values):
            mean, std = stats[min(i, len(stats) - 1)]
            out.append((float(v) - mean) / max(std, eps))
        return out

    @staticmethod
    def _denorm(values: Sequence[float],
                stats: tuple[tuple[float, float], ...],
                eps: float) -> list[float]:
        if not stats:
            return [float(v) for v in values]
        out = []
        for i, v in enumerate(values):
            mean, std = stats[min(i, len(stats) - 1)]
            out.append(float(v) * max(std, eps) + mean)
        return out

    def normalize_input(self, values: Sequence[float]) -> list[float]:
        return self._norm(values, self.input_stats, self.eps)

    def normalize_output(self, values: Sequence[float]) -> list[float]:
        return self._norm(values, self.output_stats, self.eps)

    def denormalize_input(self, values: Sequence[float]) -> list[float]:
        return self._denorm(values, self.input_stats, self.eps)

    def denormalize_output(self, values: Sequence[float]) -> list[float]:
        return self._denorm(values, self.output_stats, self.eps)

    def normalize_samples(self, samples: Iterable[dict]) -> list[dict[str, list[float]]]:
        return _normalize_samples(self, samples)


# ---------------------------------------------------------------------------
# ClipNormalizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClipNormalizer:
    """Clips each channel to [low, high] and optionally rescales to [0, 1].

    When rescale=True (default), the clipped value is linearly mapped to
    [0, 1].  When rescale=False, values are only clipped — no rescaling.

    Args:
        input_clip:  list/tuple of (low, high) per input channel.
        output_clip: list/tuple of (low, high) per output channel.
        rescale:     whether to map the clipped range to [0, 1].

    Example::

        n = ClipNormalizer(input_clip=[(-5.0, 5.0)], rescale=True)
    """

    input_clip: tuple[tuple[float, float], ...] = ()
    output_clip: tuple[tuple[float, float], ...] = ()
    rescale: bool = True

    @staticmethod
    def _norm(values: Sequence[float],
              clips: tuple[tuple[float, float], ...],
              rescale: bool) -> list[float]:
        if not clips:
            return [float(v) for v in values]
        out = []
        for i, v in enumerate(values):
            lo, hi = clips[min(i, len(clips) - 1)]
            clipped = max(lo, min(hi, float(v)))
            if rescale:
                span = hi - lo
                out.append((clipped - lo) / span if span > 0 else 0.0)
            else:
                out.append(clipped)
        return out

    @staticmethod
    def _denorm(values: Sequence[float],
                clips: tuple[tuple[float, float], ...],
                rescale: bool) -> list[float]:
        if not clips:
            return [float(v) for v in values]
        out = []
        for i, v in enumerate(values):
            lo, hi = clips[min(i, len(clips) - 1)]
            if rescale:
                raw = float(v) * (hi - lo) + lo
            else:
                raw = float(v)
            out.append(max(lo, min(hi, raw)))
        return out

    def normalize_input(self, values: Sequence[float]) -> list[float]:
        return self._norm(values, self.input_clip, self.rescale)

    def normalize_output(self, values: Sequence[float]) -> list[float]:
        return self._norm(values, self.output_clip, self.rescale)

    def denormalize_input(self, values: Sequence[float]) -> list[float]:
        return self._denorm(values, self.input_clip, self.rescale)

    def denormalize_output(self, values: Sequence[float]) -> list[float]:
        return self._denorm(values, self.output_clip, self.rescale)

    def normalize_samples(self, samples: Iterable[dict]) -> list[dict[str, list[float]]]:
        return _normalize_samples(self, samples)


# ---------------------------------------------------------------------------
# RunningStatsNormalizer
# ---------------------------------------------------------------------------

class RunningStatsNormalizer:
    """Online z-score normalizer that estimates mean and std from streaming data.

    Useful for gym-style environments where the input distribution is unknown
    upfront.  Call ``update(inputs, outputs)`` inside the fitness function to
    accumulate statistics, then ``normalize_input`` / ``normalize_output`` use
    the current estimates.

    Thread-safety: not thread-safe — use one instance per process.

    Args:
        n_inputs:  number of input channels.
        n_outputs: number of output channels.
        eps:       minimum std (default 1e-8) to avoid division by zero.

    Example::

        norm = RunningStatsNormalizer(n_inputs=4, n_outputs=1)

        def evaluate(genome):
            obs, _ = env.reset()
            norm.update_inputs(obs)
            ...
            out = genome.forward(norm.normalize_input(obs))
    """

    def __init__(self, n_inputs: int, n_outputs: int = 0, eps: float = 1e-8) -> None:
        self.eps = eps
        self._in_n  = [0] * n_inputs
        self._in_m  = [0.0] * n_inputs   # running mean
        self._in_m2 = [0.0] * n_inputs   # running sum of squared deviations (Welford)
        self._out_n  = [0] * n_outputs
        self._out_m  = [0.0] * n_outputs
        self._out_m2 = [0.0] * n_outputs

    # -- Welford online update --------------------------------------------------

    @staticmethod
    def _welford_update(n: list[int], m: list[float], m2: list[float],
                        values: Sequence[float]) -> None:
        for i, v in enumerate(values):
            if i >= len(n):
                break
            n[i] += 1
            delta = float(v) - m[i]
            m[i] += delta / n[i]
            m2[i] += delta * (float(v) - m[i])

    def update_inputs(self, values: Sequence[float]) -> None:
        """Feed one observation vector to update running input statistics."""
        self._welford_update(self._in_n, self._in_m, self._in_m2, values)

    def update_outputs(self, values: Sequence[float]) -> None:
        """Feed one target vector to update running output statistics."""
        self._welford_update(self._out_n, self._out_m, self._out_m2, values)

    def update(self, inputs: Sequence[float],
               outputs: Sequence[float] | None = None) -> None:
        """Update both input and output statistics in one call."""
        self.update_inputs(inputs)
        if outputs is not None:
            self.update_outputs(outputs)

    # -- Normalize / denormalize -----------------------------------------------

    def _std(self, n: list[int], m2: list[float], i: int) -> float:
        ni = n[min(i, len(n) - 1)]
        m2i = m2[min(i, len(m2) - 1)]
        return math.sqrt(m2i / ni) if ni > 1 else self.eps

    def normalize_input(self, values: Sequence[float]) -> list[float]:
        return [
            (float(v) - self._in_m[min(i, len(self._in_m) - 1)]) /
            max(self._std(self._in_n, self._in_m2, i), self.eps)
            for i, v in enumerate(values)
        ]

    def normalize_output(self, values: Sequence[float]) -> list[float]:
        if not self._out_n:
            return [float(v) for v in values]
        return [
            (float(v) - self._out_m[min(i, len(self._out_m) - 1)]) /
            max(self._std(self._out_n, self._out_m2, i), self.eps)
            for i, v in enumerate(values)
        ]

    def denormalize_input(self, values: Sequence[float]) -> list[float]:
        return [
            float(v) * max(self._std(self._in_n, self._in_m2, i), self.eps)
            + self._in_m[min(i, len(self._in_m) - 1)]
            for i, v in enumerate(values)
        ]

    def denormalize_output(self, values: Sequence[float]) -> list[float]:
        if not self._out_n:
            return [float(v) for v in values]
        return [
            float(v) * max(self._std(self._out_n, self._out_m2, i), self.eps)
            + self._out_m[min(i, len(self._out_m) - 1)]
            for i, v in enumerate(values)
        ]

    def normalize_samples(self, samples: Iterable[dict]) -> list[dict[str, list[float]]]:
        return _normalize_samples(self, samples)

    @property
    def input_means(self) -> list[float]:
        return list(self._in_m)

    @property
    def input_stds(self) -> list[float]:
        return [max(self._std(self._in_n, self._in_m2, i), self.eps)
                for i in range(len(self._in_n))]

    @property
    def output_means(self) -> list[float]:
        return list(self._out_m)

    @property
    def output_stds(self) -> list[float]:
        return [max(self._std(self._out_n, self._out_m2, i), self.eps)
                for i in range(len(self._out_n))]

    def sample_count(self) -> int:
        return self._in_n[0] if self._in_n else 0
