"""Basic multiplication example — learns to multiply two numbers.

Inputs 0–9 and outputs 0–81 are normalised to [0, 1] by default so activation
functions operate in a useful range and the fitness landscape is scale-independent.
Pass normalize=False to make_eval() to train on raw values instead.
"""
import json
import os

from yane import NeuroEvolution

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "multiplication_table.json")) as f:
    _raw = json.load(f)

_IN_MAX  = 9.0
_OUT_MAX = 81.0

dataset = [
    {
        "input":  [x / _IN_MAX  for x in s["input"]],
        "output": [y / _OUT_MAX for y in s["output"]],
    }
    for s in _raw
]

N_INPUTS       = 2
N_OUTPUTS      = 1
TARGET_FITNESS = -5.0   # total |error| ≤ 5 across 100 normalised samples (avg 0.05)

# Normalised test cases: inputs a/9, b/9 → output a*b/81
TEST_CASES = [
    ([a / _IN_MAX, b / _IN_MAX], [a * b / _OUT_MAX])
    for a, b in [(0, 0), (2, 3), (5, 5), (7, 4), (9, 9), (1, 9)]
]


def make_eval(render_callback=None, step_callback=None, demo=False, normalize=True):
    data = dataset if normalize else _raw

    def evaluate(genome):
        fitness = 0.0
        for sample in data:
            # Reset between samples so any persistent hidden nodes (developed
            # through mutation) can't accumulate state and produce NaN/Inf.
            genome.reset()
            outputs = genome.forward(sample["input"])
            for i, target in enumerate(sample["output"]):
                fitness -= abs(outputs[i] - target)
        return fitness
    return evaluate


def main():
    yane = NeuroEvolution()
    yane.configure(n_inputs=N_INPUTS, n_outputs=N_OUTPUTS,
                   max_nodes=30, max_connections=100)
    yane.set_resource_limits(max_process_gb=2.0)
    yane.set_min_fitness(TARGET_FITNESS)

    n = yane.train(make_eval())
    best = yane.get_best()
    print(f"Done in {n} iterations. Fitness: {best.fitness:.4f}")

    for s_raw, s_norm in zip(_raw, dataset):
        best.reset()
        outputs   = best.forward(s_norm["input"])
        predicted = outputs[0] * _OUT_MAX
        expected  = s_raw["output"][0]
        print(f"  {s_raw['input']} -> {predicted:.2f}  (expected {expected})")


if __name__ == "__main__":
    main()
