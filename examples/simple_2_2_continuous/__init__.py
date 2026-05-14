"""Continuous 2-input 2-output regression example."""
import json
import os

from yane import NeuroEvolution

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "dataset_2_2.json")) as f:
    dataset = json.load(f)


def main():
    yane = NeuroEvolution()
    yane.configure(n_inputs=2, n_outputs=2, max_nodes=20, max_connections=60)
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
        print(f"  {sample['input']} -> {[f'{v:.3f}' for v in outputs]}  (expected {sample['output']})")


if __name__ == "__main__":
    main()
