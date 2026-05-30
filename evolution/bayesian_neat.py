"""Probabilistic / Bayesian NEAT.

Adds Monte-Carlo dropout-style Gaussian noise to genome outputs, enabling
uncertainty estimation via repeated stochastic sampling.

Usage::

    from yane.evolution.bayesian_neat import set_probabilistic, bayesian_forward

    set_probabilistic(genome, enabled=True, noise_std=0.05)
    mean, std = bayesian_forward(genome, [0.0, 1.0], n=100)

    # Deterministic inference (mean only, no noise):
    set_probabilistic(genome, enabled=True, inference_mode=True)
    result = genome.forward([0.0, 1.0])
"""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


def set_probabilistic(
    genome: "Genome",
    enabled: bool = True,
    noise_std: float = 0.05,
    inference_mode: bool = False,
) -> None:
    """Enable or disable probabilistic (stochastic) output on *genome*.

    When enabled and ``inference_mode=False``, each call to
    ``genome.forward()`` adds per-output Gaussian noise with standard
    deviation ``noise_std``.  Use :func:`bayesian_forward` to collect
    multiple samples and estimate output mean and standard deviation.

    When ``inference_mode=True``, no noise is added — forward passes are
    fully deterministic.

    Parameters
    ----------
    genome:
        The genome to configure.
    enabled:
        Whether probabilistic noise is active.
    noise_std:
        Standard deviation of the per-output Gaussian noise.
    inference_mode:
        When True, forward() is deterministic regardless of ``enabled``.
    """
    genome._prob_enabled = enabled
    genome._prob_noise_std = float(noise_std)
    genome._prob_inference_mode = inference_mode


def bayesian_forward(
    genome: "Genome",
    inputs: "list[float]",
    n: int = 100,
) -> "tuple[list[float], list[float]]":
    """Monte-Carlo forward pass: run *n* stochastic samples and aggregate.

    Temporarily forces probabilistic (non-deterministic) mode, collects *n*
    output samples, then returns per-output mean and population standard
    deviation.

    Parameters
    ----------
    genome:
        Genome.  Probabilistic mode must have been enabled via
        :func:`set_probabilistic`.
    inputs:
        Input vector passed to each forward call.
    n:
        Number of stochastic samples (higher = lower estimation variance).

    Returns
    -------
    (mean_outputs, std_outputs):
        Both lists have length ``n_outputs``.
        ``std_outputs[i]`` converges as O(1/sqrt(n)).
    """
    if n < 1:
        raise ValueError("bayesian_forward: n must be >= 1")

    # Force stochastic mode for sampling
    prev_enabled = getattr(genome, "_prob_enabled", False)
    prev_inference = getattr(genome, "_prob_inference_mode", False)

    genome._prob_enabled = True
    genome._prob_inference_mode = False

    n_outputs = len(genome.output_nodes)
    # Collect samples
    samples: list[list[float]] = []
    for _ in range(n):
        genome.reset()
        out = genome.forward(inputs)
        samples.append(list(out))

    # Restore original state
    genome._prob_enabled = prev_enabled
    genome._prob_inference_mode = prev_inference

    mean_outputs: list[float] = []
    std_outputs: list[float] = []
    for i in range(n_outputs):
        col = [s[i] for s in samples]
        mu = sum(col) / n
        mean_outputs.append(mu)
        if n > 1:
            variance = sum((x - mu) ** 2 for x in col) / n
            std_outputs.append(math.sqrt(variance))
        else:
            std_outputs.append(0.0)

    return mean_outputs, std_outputs
