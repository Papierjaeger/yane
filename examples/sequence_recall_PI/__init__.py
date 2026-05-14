"""Sequence recall PI — predicts digits of pi as a classification task."""
import json
import os

import numpy as np

from yane import NeuroEvolution

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "dataset_PI.json")) as f:
    dataset = json.load(f)

DECIMAL_PLACES = 5


def _sample_fitness(outputs, target_index):
    fitness = 0.0
    for i, v in enumerate(outputs):
        expected = 1.0 if i == target_index else 0.0
        fitness -= abs(v - expected)
    return fitness


def main():
    n_inputs = len(dataset[0]["input"]) if dataset else 1

    yane = NeuroEvolution()
    yane.configure(n_inputs=n_inputs, n_outputs=10, max_nodes=30, max_connections=100)
    yane.set_min_fitness(-0.5)

    def evaluate(genome):
        fitness = 0.0
        for sample in dataset[:DECIMAL_PLACES]:
            outputs = genome.forward(sample["input"])
            fitness += _sample_fitness(outputs, sample["output"][0])
        return fitness

    n = yane.train(evaluate)
    best = yane.get_best()
    print(f"Done in {n} iterations. Fitness: {best.fitness:.4f}")

    for sample in dataset[:DECIMAL_PLACES]:
        outputs = best.forward(sample["input"])
        predicted = int(np.argmax(outputs))
        print(f"  Input: {sample['input']}  Predicted: {predicted}  Expected: {sample['output'][0]}")


if __name__ == "__main__":
    main()
