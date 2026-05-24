"""Small normalization helpers for dataset-style examples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ScaleNormalizer:
    """Per-channel scale normalizer.

    Values are mapped by division on input and multiplication on output.  This
    intentionally mirrors the GUI's existing `input_scale` / `output_scale`
    convention, so examples and Inspect can share the same metadata.
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
        return [
            {
                "input": self.normalize_input(sample["input"]),
                "output": self.normalize_output(sample["output"]),
            }
            for sample in samples
        ]

    @property
    def input_scale_list(self) -> list[float] | None:
        return list(self.input_scale) if self.input_scale else None

    @property
    def output_scale_list(self) -> list[float] | None:
        return list(self.output_scale) if self.output_scale else None
