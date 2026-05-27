"""XOR example — teaches the network to compute XOR of two binary inputs."""

from yane import NeuroEvolution
from yane.examples._dataset import load_json_dataset, make_absolute_error_eval

dataset = load_json_dataset(__file__, "dataset_XOR.json")

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
    return make_absolute_error_eval(dataset)


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
