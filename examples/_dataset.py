"""Shared helpers for bundled dataset examples."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


Sample = dict[str, list[float]]


def load_json_dataset(package_file: str, filename: str) -> list[Sample]:
    with Path(package_file).with_name(filename).open() as fh:
        return json.load(fh)


def make_absolute_error_eval(samples: Iterable[Sample], reset_each_sample: bool = True):
    data = tuple(samples)

    def evaluate(genome):
        fitness = 0.0
        if reset_each_sample:
            for sample in data:
                genome.reset()
                outputs = genome.forward(sample["input"])
                for i, target in enumerate(sample["output"]):
                    fitness -= abs(outputs[i] - target)
        else:
            genome.reset()
            for sample in data:
                outputs = genome.forward(sample["input"])
                for i, target in enumerate(sample["output"]):
                    fitness -= abs(outputs[i] - target)
        return fitness

    return evaluate


def sample_pairs(samples: Iterable[Sample]) -> list[tuple[list[float], list[float]]]:
    return [
        (list(map(float, sample["input"])), list(map(float, sample["output"])))
        for sample in samples
    ]
