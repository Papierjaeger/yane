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
TARGET_FITNESS = -0.5   # total |error| ≤ 0.5 across 100 normalised samples


def make_eval(render_callback=None, step_callback=None, demo=False, normalize=True):
    data = dataset if normalize else _raw

    def evaluate(genome):
        fitness = 0.0
        for sample in data:
            genome.reset()  # stateless: reset before each sample
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
