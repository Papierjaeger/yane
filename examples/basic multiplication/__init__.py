"""Basic multiplication example — learns to multiply two numbers.

Inputs and outputs are normalised to [-1, 1] before being fed into the network.
The network therefore learns the *shape* of multiplication, not its raw scale.
Without normalisation the fitness landscape is dominated by the large output
magnitudes (0–81) and sigmoid-like activations can never reach the target range.
"""
import json
import os

from yane import NeuroEvolution

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "multiplication_table.json")) as f:
    _raw = json.load(f)

# Normalise: inputs 0–9 → 0–1, outputs 0–81 → 0–1
_IN_MAX  = 9.0
_OUT_MAX = 81.0

dataset = [
    {
        "input":  [x / _IN_MAX for x in s["input"]],
        "output": [y / _OUT_MAX for y in s["output"]],
    }
    for s in _raw
]


def main():
    yane = NeuroEvolution()
    yane.configure(n_inputs=2, n_outputs=1, max_nodes=30, max_connections=100)
    yane.set_resource_limits(max_process_gb=2.0)
    yane.set_min_fitness(-0.5)  # fitness per sample is in [-1, 0]; -0.5 = good avg

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

    for s_raw, s_norm in zip(_raw, dataset):
        outputs = best.forward(s_norm["input"])
        predicted = outputs[0] * _OUT_MAX
        expected  = s_raw["output"][0]
        print(f"  {s_raw['input']} -> {predicted:.2f}  (expected {expected})")


if __name__ == "__main__":
    main()
