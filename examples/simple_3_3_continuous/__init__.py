"""Continuous 3-input 3-output regression example."""
import json
import os

from yane import NeuroEvolution

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "dataset_3_3.json")) as f:
    dataset = json.load(f)

N_INPUTS       = 3
N_OUTPUTS      = 3
TARGET_FITNESS = -2.0   # 8 samples × 3 outputs = 24 errors; avg ≤ 0.08 per output

TEST_CASES = [
    (list(map(float, s["input"])), list(map(float, s["output"])))
    for s in dataset
]


def make_eval(render_callback=None, step_callback=None, demo=False):
    def evaluate(genome):
        fitness = 0.0
        for sample in dataset:
            genome.reset()   # stateless: clear persistent memory between samples
            outputs = genome.forward(sample["input"])
            for i, target in enumerate(sample["output"]):
                fitness -= abs(outputs[i] - target)
        return fitness
    return evaluate


def main():
    yane = NeuroEvolution()
    yane.configure(n_inputs=N_INPUTS, n_outputs=N_OUTPUTS,
                   max_nodes=20, max_connections=60)
    yane.set_resource_limits(max_process_gb=2.0)
    yane.set_min_fitness(TARGET_FITNESS)

    n = yane.train(make_eval())
    best = yane.get_best()
    print(f"Done in {n} iterations. Fitness: {best.fitness:.4f}")

    for sample in dataset:
        best.reset()
        outputs = best.forward(sample["input"])
        print(f"  {sample['input']} -> {[f'{v:.3f}' for v in outputs]}  (expected {sample['output']})")


if __name__ == "__main__":
    main()
