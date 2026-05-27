"""Continuous 3-input 3-output regression example."""

from yane import NeuroEvolution
from yane.examples._dataset import (
    load_json_dataset,
    make_absolute_error_eval,
    sample_pairs,
)

dataset = load_json_dataset(__file__, "dataset_3_3.json")

N_INPUTS       = 3
N_OUTPUTS      = 3
TARGET_FITNESS = -5.0   # 8 samples × 3 outputs = 24 errors; avg ≤ 0.21 per output (robust to seed variance)

TEST_CASES = sample_pairs(dataset)


def make_eval(render_callback=None, step_callback=None, demo=False):
    return make_absolute_error_eval(dataset)


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
