"""Sequence recall PI — predicts the next digit of pi as a classification task.

Input: current pi digit (0–9), normalised to [0, 1].
Output: one-hot over digits 0–9 (argmax = predicted next digit).

Note: with a single scalar input there is no positional information, so the
network can only learn the most likely successor for each digit — not the full
sequence. Increasing DECIMAL_PLACES makes the task harder (more ambiguous
digit transitions) while keeping it tractable as a memorisation benchmark.
"""
import json
import os

import numpy as np

from yane import NeuroEvolution

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "dataset_PI.json")) as f:
    _raw = json.load(f)

DECIMAL_PLACES = 10  # number of pi digits to learn

# Normalise digit input 0–9 → 0–1 so activation functions operate in a
# useful range from the start (avoids the network fighting input scale).
dataset = [
    {"input": [s["input"][0] / 9.0], "output": s["output"]}
    for s in _raw
]


def _sample_fitness(outputs, target_index):
    fitness = 0.0
    for i, v in enumerate(outputs):
        expected = 1.0 if i == target_index else 0.0
        fitness -= abs(v - expected)
    return fitness


def main():
    yane = NeuroEvolution()
    yane.configure(n_inputs=1, n_outputs=10, max_nodes=30, max_connections=100)
    yane.set_resource_limits(max_process_gb=2.0)
    yane.set_min_fitness(-1.0 * DECIMAL_PLACES)  # perfect = 0; -DECIMAL_PLACES = all wrong

    def evaluate(genome):
        fitness = 0.0
        for sample in dataset[:DECIMAL_PLACES]:
            outputs = genome.forward(sample["input"])
            fitness += _sample_fitness(outputs, sample["output"][0])
        return fitness

    n = yane.train(evaluate)
    best = yane.get_best()
    print(f"Done in {n} iterations. Fitness: {best.fitness:.4f}")

    correct = 0
    for sample, raw in zip(dataset[:DECIMAL_PLACES], _raw[:DECIMAL_PLACES]):
        outputs = best.forward(sample["input"])
        predicted = int(np.argmax(outputs))
        expected  = raw["output"][0]
        mark = "✓" if predicted == expected else "✗"
        print(f"  {mark}  Input digit: {raw['input'][0]}  Predicted: {predicted}  Expected: {expected}")
        if predicted == expected:
            correct += 1
    print(f"Accuracy: {correct}/{DECIMAL_PLACES}")


if __name__ == "__main__":
    main()
