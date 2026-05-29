"""Evolutionary Data Augmentation — co-evolving input transformation pipelines.

A small population of ``AugmentationPipeline`` objects evolves alongside the
main NEAT population.  Each pipeline is a sequence of stochastic input
transformations (noise injection, dropout, scaling, etc.).  During training,
the current pipeline wraps ``genome.forward()`` so that every sample the NEAT
genome sees is drawn from the augmented distribution — this acts as a
regulariser that can prevent overfitting on small datasets.

Integration
-----------
::

    yane.set_evolutionary_augmentation(
        augmentation_space=["gaussian_noise", "dropout_noise", "scaling"],
        population_augmentations=8,
        pipeline_length=3,
        evolution_interval=20,
    )
    yane.train(eval_fn)   # pipeline wraps genome.forward() automatically

How the co-evolution works
--------------------------
1. A pool of ``population_augmentations`` random pipelines is created.
2. At each generation boundary one pipeline is selected from the pool
   (UCB1 balancing exploration and exploitation).
3. All genome evaluations in that generation use the selected pipeline.
4. The pipeline's reward is the fitness improvement of the best genome in
   that generation vs. the best of the previous generation.
5. Every ``evolution_interval`` generations the pool is evolved: the worst
   half is replaced with offspring (tournament crossover + mutation) of the
   best half.
"""
from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Supported augmentation types
# ---------------------------------------------------------------------------

AUGMENTATION_TYPES: list[str] = [
    "gaussian_noise",  # add Gaussian noise to every input element
    "dropout_noise",   # zero out each input element independently
    "scaling",         # multiply each element by a random factor near 1
    "translation",     # add a uniform offset to all elements
    "cutout",          # zero a contiguous block of elements
]


def _apply_gaussian_noise(inputs: list[float], prob: float, mag: float,
                          rng: _random.Random) -> list[float]:
    return [v + rng.gauss(0.0, mag) if rng.random() < prob else v
            for v in inputs]


def _apply_dropout_noise(inputs: list[float], prob: float, mag: float,
                         rng: _random.Random) -> list[float]:
    # mag used as extra drop probability (on top of prob)
    drop_p = max(0.0, min(1.0, prob * mag))
    return [0.0 if rng.random() < drop_p else v for v in inputs]


def _apply_scaling(inputs: list[float], prob: float, mag: float,
                   rng: _random.Random) -> list[float]:
    if rng.random() >= prob:
        return inputs
    return [v * (1.0 + rng.uniform(-mag, mag)) for v in inputs]


def _apply_translation(inputs: list[float], prob: float, mag: float,
                       rng: _random.Random) -> list[float]:
    if rng.random() >= prob:
        return inputs
    offset = rng.uniform(-mag, mag)
    return [v + offset for v in inputs]


def _apply_cutout(inputs: list[float], prob: float, mag: float,
                  rng: _random.Random) -> list[float]:
    if rng.random() >= prob or not inputs:
        return inputs
    n = len(inputs)
    cut_len = max(1, int(n * mag))
    start = rng.randint(0, max(0, n - cut_len))
    result = list(inputs)
    for i in range(start, min(start + cut_len, n)):
        result[i] = 0.0
    return result


_APPLY_FN: dict[str, Callable] = {
    "gaussian_noise": _apply_gaussian_noise,
    "dropout_noise":  _apply_dropout_noise,
    "scaling":        _apply_scaling,
    "translation":    _apply_translation,
    "cutout":         _apply_cutout,
}


# ---------------------------------------------------------------------------
# AugmentationGene
# ---------------------------------------------------------------------------

@dataclass
class AugmentationGene:
    """Single augmentation operation with evolvable probability and magnitude."""

    aug_type:    str    # one of AUGMENTATION_TYPES
    probability: float  # [0, 1] — chance this transform is applied per call
    magnitude:   float  # [0, 1] — strength of the transformation

    def apply(self, inputs: list[float], rng: _random.Random) -> list[float]:
        fn = _APPLY_FN.get(self.aug_type)
        if fn is None or not inputs:
            return inputs
        return fn(inputs, self.probability, self.magnitude, rng)

    def mutate(self, rng: _random.Random, sigma: float = 0.1) -> "AugmentationGene":
        """Return a mutated copy of this gene."""
        prob = float(max(0.0, min(1.0, self.probability + rng.gauss(0.0, sigma))))
        mag  = float(max(0.0, min(1.0, self.magnitude  + rng.gauss(0.0, sigma))))
        return AugmentationGene(self.aug_type, prob, mag)

    def to_dict(self) -> dict:
        return {"type": self.aug_type, "probability": self.probability,
                "magnitude": self.magnitude}

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentationGene":
        return cls(d["type"], float(d["probability"]), float(d["magnitude"]))

    @classmethod
    def random(cls, aug_types: list[str], rng: _random.Random) -> "AugmentationGene":
        return cls(
            aug_type=rng.choice(aug_types),
            probability=rng.uniform(0.3, 0.9),
            magnitude=rng.uniform(0.05, 0.4),
        )


# ---------------------------------------------------------------------------
# AugmentationPipeline
# ---------------------------------------------------------------------------

@dataclass
class AugmentationPipeline:
    """Ordered sequence of augmentation genes applied to every input sample."""

    genes:          list[AugmentationGene]
    _reward_sum:    float = field(default=0.0, repr=False)
    _n_selections:  int   = field(default=0,   repr=False)

    def apply(self, inputs: list[float], rng: _random.Random) -> list[float]:
        """Apply all genes in sequence."""
        result = list(inputs)
        for gene in self.genes:
            result = gene.apply(result, rng)
        return result

    def mutate(self, rng: _random.Random, sigma: float = 0.1) -> "AugmentationPipeline":
        """Return a copy with all genes slightly mutated."""
        return AugmentationPipeline([g.mutate(rng, sigma) for g in self.genes])

    def crossover(
        self, other: "AugmentationPipeline", rng: _random.Random
    ) -> "AugmentationPipeline":
        """Uniform gene-level crossover; length = max(len(self), len(other))."""
        n = max(len(self.genes), len(other.genes))
        a_genes = self.genes  + [None] * (n - len(self.genes))
        b_genes = other.genes + [None] * (n - len(other.genes))
        child_genes = []
        for a, b in zip(a_genes, b_genes):
            if a is None:
                child_genes.append(b)
            elif b is None:
                child_genes.append(a)
            else:
                child_genes.append(a if rng.random() < 0.5 else b)
        return AugmentationPipeline([g for g in child_genes if g is not None])

    @property
    def mean_reward(self) -> float:
        return self._reward_sum / max(1, self._n_selections)

    def ucb1_score(self, total_selections: int, c: float = 1.414) -> float:
        if self._n_selections == 0:
            return float("inf")
        return self.mean_reward + c * math.sqrt(
            math.log(max(1, total_selections)) / self._n_selections
        )

    def to_dict(self) -> dict:
        return {"genes": [g.to_dict() for g in self.genes]}

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentationPipeline":
        genes = [AugmentationGene.from_dict(g) for g in d.get("genes", [])]
        return cls(genes)

    @classmethod
    def random(
        cls,
        aug_types: list[str],
        length: int,
        rng: _random.Random,
    ) -> "AugmentationPipeline":
        genes = [AugmentationGene.random(aug_types, rng) for _ in range(length)]
        return cls(genes)


# ---------------------------------------------------------------------------
# AugmentationPool
# ---------------------------------------------------------------------------

class AugmentationPool:
    """Small evolving population of augmentation pipelines.

    Selection uses UCB1 to balance between trying new pipelines and
    exploiting the best-performing ones.  Every ``evolution_interval``
    calls to ``evolve()`` the worst half is replaced with offspring of
    the best half.
    """

    def __init__(
        self,
        augmentation_space: list[str],
        population_size: int = 8,
        pipeline_length: int = 3,
        evolution_interval: int = 20,
        mutation_sigma: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self._rng = _random.Random(seed)
        self._space = [t for t in augmentation_space if t in AUGMENTATION_TYPES]
        if not self._space:
            raise ValueError(
                f"No valid augmentation types given.  "
                f"Available: {AUGMENTATION_TYPES}"
            )
        self._pop_size = max(1, population_size)
        self._pipe_len = max(1, pipeline_length)
        self._evo_interval = max(1, evolution_interval)
        self._sigma = mutation_sigma
        self._generation = 0
        self._total_selections = 0
        self._pipelines: list[AugmentationPipeline] = [
            AugmentationPipeline.random(self._space, self._pipe_len, self._rng)
            for _ in range(self._pop_size)
        ]
        self._active_idx: int = 0
        self._active: AugmentationPipeline = self._pipelines[0]

    def select(self) -> AugmentationPipeline:
        """Return the next pipeline to use (UCB1)."""
        best_score = -float("inf")
        best_idx   = 0
        for i, p in enumerate(self._pipelines):
            score = p.ucb1_score(self._total_selections)
            if score > best_score:
                best_score = score
                best_idx   = i
        self._active_idx = best_idx
        self._active = self._pipelines[best_idx]
        self._active._n_selections += 1
        self._total_selections += 1
        return self._active

    def update_reward(self, reward: float) -> None:
        """Record a reward for the current active pipeline."""
        self._active._reward_sum += reward

    def evolve(self) -> None:
        """One round of evolution: replace worst half with offspring of best half."""
        self._generation += 1
        sorted_pipes = sorted(self._pipelines, key=lambda p: p.mean_reward, reverse=True)
        n_keep = max(1, len(sorted_pipes) // 2)
        elite  = sorted_pipes[:n_keep]
        offspring: list[AugmentationPipeline] = []
        while len(offspring) < len(sorted_pipes) - n_keep:
            p1 = self._rng.choice(elite)
            p2 = self._rng.choice(elite)
            child = p1.crossover(p2, self._rng).mutate(self._rng, self._sigma)
            offspring.append(child)
        self._pipelines = elite + offspring

    def should_evolve(self, generation: int) -> bool:
        return generation > 0 and generation % self._evo_interval == 0

    @property
    def rng(self) -> _random.Random:
        return self._rng

    def best_pipeline(self) -> AugmentationPipeline:
        return max(self._pipelines, key=lambda p: p.mean_reward)

    def get_diagnostics(self) -> dict:
        best = self.best_pipeline()
        return {
            "population_size":   len(self._pipelines),
            "total_selections":  self._total_selections,
            "best_mean_reward":  round(best.mean_reward, 4),
            "best_genes":        [g.to_dict() for g in best.genes],
        }
