"""XOR example — teaches the network to compute XOR of two binary inputs."""
import json
import os

from yane import NeuroEvolution

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "dataset_XOR.json")) as f:
    dataset = json.load(f)


def main():
    yane = NeuroEvolution()
    yane.configure(n_inputs=2, n_outputs=1, max_nodes=20, max_connections=50)
    yane.set_resource_limits(max_process_gb=2.0)
    yane.set_min_fitness(-0.1)

    def evaluate(genome):
        fitness = 0.0
        for sample in dataset:
            outputs = genome.forward(sample["input"])
            fitness -= abs(outputs[0] - sample["output"][0])
        return fitness

    n = yane.train(evaluate)
    best = yane.get_best()
    print(f"Done in {n} iterations. Fitness: {best.fitness:.4f}")

    for sample in dataset:
        outputs = best.forward(sample["input"])
        print(f"  {sample['input']} -> {outputs[0]:.3f}  (expected {sample['output'][0]})")


if __name__ == "__main__":
    main()
