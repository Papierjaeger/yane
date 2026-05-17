"""XOR example — teaches the network to compute XOR of two binary inputs."""
import json
import os

from yane import NeuroEvolution

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "dataset_XOR.json")) as f:
    dataset = json.load(f)

N_INPUTS       = 2
N_OUTPUTS      = 1
TARGET_FITNESS = -0.1

TEST_CASES = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]


def make_eval(render_callback=None, step_callback=None, demo=False):
    def evaluate(genome):
        fitness = 0.0
        for sample in dataset:
            # stateless: _forward already zeros output nodes; no reset needed
            outputs = genome.forward(sample["input"])
            fitness -= abs(outputs[0] - sample["output"][0])
        return fitness
    return evaluate


def main():
    yane = NeuroEvolution()
    yane.configure(n_inputs=N_INPUTS, n_outputs=N_OUTPUTS,
                   max_nodes=20, max_connections=50)
    yane.set_resource_limits(max_process_gb=2.0)
    yane.set_min_fitness(TARGET_FITNESS)

    n = yane.train(make_eval())
    best = yane.get_best()
    print(f"Done in {n} iterations. Fitness: {best.fitness:.4f}")

    for sample in dataset:
        best.reset()
        outputs = best.forward(sample["input"])
        print(f"  {sample['input']} -> {outputs[0]:.3f}  (expected {sample['output'][0]})")


if __name__ == "__main__":
    main()
