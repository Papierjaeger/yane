"""Sequence recall PI — predicts the next digit of pi.

Input:  current digit (0–9), normalised to [0, 1] by default.
Output: next digit (0–9), normalised to [0, 1] by default.
Pass normalize=False to train on raw digit values 0–9 instead.
"""
import json
import os

from yane import NeuroEvolution

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "dataset_PI.json")) as f:
    _raw = json.load(f)

DECIMAL_PLACES = 10
_SCALE         = 9.0

dataset = [
    {"input": [s["input"][0] / _SCALE], "output": [s["output"][0] / _SCALE]}
    for s in _raw
]

N_INPUTS       = 1
N_OUTPUTS      = 1
TARGET_FITNESS = -float(DECIMAL_PLACES)


def make_eval(render_callback=None, step_callback=None, demo=False, normalize=True):
    data = dataset if normalize else [
        {"input": [s["input"][0]], "output": [s["output"][0]]}
        for s in _raw
    ]
    samples = data[:DECIMAL_PLACES]

    def evaluate(genome):
        genome.reset()
        fitness = 0.0
        for sample in samples:
            outputs = genome.forward(sample["input"])
            fitness -= abs(outputs[0] - sample["output"][0])
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

    best.reset()
    correct = 0
    for sample, raw in zip(dataset[:DECIMAL_PLACES], _raw[:DECIMAL_PLACES]):
        outputs   = best.forward(sample["input"])
        predicted = round(outputs[0] * _SCALE)
        expected  = raw["output"][0]
        mark = "✓" if predicted == expected else "✗"
        print(f"  {mark}  Input: {raw['input'][0]}  Predicted: {predicted}  Expected: {expected}")
        if predicted == expected:
            correct += 1
    print(f"Accuracy: {correct}/{DECIMAL_PLACES}")


if __name__ == "__main__":
    main()
