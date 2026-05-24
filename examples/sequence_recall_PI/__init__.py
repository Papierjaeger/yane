"""Sequence recall PI — predicts the next digit of pi.

Input:  current digit (0–9), normalised to [0, 1] by default.
Output: next digit (0–9), normalised to [0, 1] by default.
Pass normalize=False to train on raw digit values 0–9 instead.
"""
import json
import os

from yane import NeuroEvolution
from yane.util.normalization import ScaleNormalizer

_here = os.path.dirname(__file__)

with open(os.path.join(_here, "dataset_PI.json")) as f:
    _raw = json.load(f)

DECIMAL_PLACES = 10
_SCALE         = 9.0
NORMALIZER = ScaleNormalizer(input_scale=(_SCALE,), output_scale=(_SCALE,))

dataset = NORMALIZER.normalize_samples(_raw)

N_INPUTS       = 1
N_OUTPUTS      = 1
TARGET_FITNESS = 0.0


def _digit_error(output: float, target: float, normalize: bool) -> float:
    """Return zero once the predicted digit rounds to the expected digit."""
    if normalize:
        predicted = max(0, min(9, round(output * _SCALE)))
        expected = max(0, min(9, round(target * _SCALE)))
    else:
        predicted = max(0, min(9, round(output)))
        expected = max(0, min(9, round(target)))
    if predicted == expected:
        return 0.0
    return abs(output - target)


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
            fitness -= _digit_error(outputs[0], sample["output"][0], normalize)
        return fitness
    return evaluate


def make_curriculum_eval(
    n_digits: int,
    normalize: bool = True,
    protected_digits: int = 0,
    protect_tolerance: float = 0.0,
):
    """Return an evaluation function for the first *n_digits* digits of pi.

    A perfect score is 0.0.  Curriculum stages after the first protect the
    already learned prefix lexicographically: regressions on earlier digits are
    punished harder than the still-new digit can compensate, while the base
    fitness keeps the same scale as the GUI target.
    """
    data = dataset if normalize else [
        {"input": [s["input"][0]], "output": [s["output"][0]]}
        for s in _raw
    ]
    samples = data[:n_digits]
    protected_digits = max(0, min(protected_digits, len(samples)))
    protect_tolerance = max(0.0, float(protect_tolerance))
    # Later digits must not be able to buy fitness by worsening the prefix.
    prefix_weight = float(max(10, n_digits * 10))

    def evaluate(genome):
        genome.reset()
        errors = []
        for sample in samples:
            outputs = genome.forward(sample["input"])
            errors.append(_digit_error(outputs[0], sample["output"][0], normalize))
        base_error = sum(errors)
        regressions = [
            max(0.0, err - protect_tolerance)
            for err in errors[:protected_digits]
        ]
        regression_penalty = prefix_weight * sum(regressions)
        return -(base_error + regression_penalty)

    evaluate.__name__ = f"pi_{n_digits}digits"
    return evaluate


# Curriculum specification: (n_digits, label)
#
# The curriculum expands exactly one output at a time.  Earlier outputs are
# heavily weighted in later stages so the search cannot trade away already
# learned digits for progress on the new one.  Stage targets are derived from
# the GUI's final target fitness by make_curriculum_targets().
CURRICULUM_SPEC = [
    (1, "1st digit"),
    (2, "2 digits"),
    (3, "3 digits"),
    (4, "4 digits"),
    (5, "5 digits"),
    (6, "6 digits"),
    (7, "7 digits"),
    (8, "8 digits"),
    (9, "9 digits"),
    (10, "10 digits"),
]


def make_curriculum_targets(
    final_target_fitness: float = TARGET_FITNESS,
    total_digits: int = DECIMAL_PLACES,
) -> list[float]:
    """Scale the final GUI target to per-prefix curriculum targets.

    Fitness is negative error with optimum 0.0.  If the GUI target is -0.5 for
    ten digits, the 1-digit stage targets -0.05, the 2-digit stage -0.10, etc.
    A GUI target of 0.0 demands perfect output at every stage.
    """
    final_target = min(0.0, float(final_target_fitness))
    total_digits = max(1, int(total_digits))
    return [
        final_target * (n_digits / total_digits)
        for n_digits, _label in CURRICULUM_SPEC
    ]


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
