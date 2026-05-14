"""Basic multiplication example — learns to multiply two numbers."""
import json
import os

import numpy as np

from yane import NeuroEvolution

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "multiplication_table.json")) as f:
    dataset = json.load(f)


def main():
    yane = NeuroEvolution()
    yane.configure(n_inputs=2, n_outputs=1, max_nodes=30, max_connections=100)
    yane.set_resource_limits(max_process_gb=2.0)
    yane.set_min_fitness(-0.1)

    def evaluate(genome):
        fitness = 0.0
        for sample in dataset:
            outputs = genome.forward(sample["input"])
            for i, target in enumerate(sample["output"]):
                fitness -= abs(outputs[i] - target)
        return fitness

    n = yane.train(evaluate)
    best = yane.get_best()
    print(f"Done in {n} iterations. Fitness: {best.fitness:.4f}")

    for sample in dataset:
        outputs = best.forward(sample["input"])
        print(f"  {sample['input']} -> {outputs[0]:.3f}  (expected {sample['output'][0]})")


if __name__ == "__main__":
    main()
