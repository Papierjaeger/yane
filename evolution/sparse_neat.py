"""Sparse NEAT -- Lottery Ticket Hypothesis via Iterative Magnitude Pruning.

Finds the "lottery ticket" -- the minimal sparse sub-network that retains
near-original performance after iterative weight pruning.

Algorithm (IMP):
  1. Evaluate original genome fitness.
  2. Prune the weakest prune_frac fraction of connections.
  3. Optional: fine-tune via LamarckRefiner.
  4. Re-evaluate.  If fitness >= original - max_fitness_drop, continue.
  5. Repeat until target_sparsity is reached.

Usage::

    from yane.evolution.sparse_neat import find_lottery_ticket, apply_ticket, LotteryTicket

    ticket = find_lottery_ticket(
        genome, fitness_fn,
        target_sparsity=0.5, max_fitness_drop=0.05, iterations=5,
    )
    apply_ticket(genome, ticket)   # disables pruned connections in-place
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from yane.core.genome import Genome


@dataclass
class LotteryTicket:
    """Result of find_lottery_ticket."""
    mask: frozenset
    sparsity: float
    fitness: float
    original_fitness: float

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


def _get_active_connections(genome):
    return [c for nd in genome.nodes for c in nd.connections if c.enabled and c.innovation != -1]


def _get_all_connections(genome):
    return [c for nd in genome.nodes for c in nd.connections]


def find_lottery_ticket(genome, fitness_fn, target_sparsity=0.5, max_fitness_drop=0.05, iterations=5, lamarck_steps=0, lamarck_sigma=0.1):
    """Find the sparse lottery ticket via Iterative Magnitude Pruning (IMP)."""
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError(f"target_sparsity must be in [0, 1), got {target_sparsity}")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    original_states = {}
    all_conns = _get_all_connections(genome)
    for c in all_conns:
        original_states[id(c)] = (c._weight, c.enabled)

    genome.reset()
    original_fitness = fitness_fn(genome)

    n_original_active = sum(1 for nd in genome.nodes for c in nd.connections if c.enabled and c.innovation != -1)
    n_total_to_prune = int(n_original_active * target_sparsity)
    n_pruned_so_far = [0]

    best_ticket = LotteryTicket(
        mask=frozenset(c.innovation for c in _get_active_connections(genome)),
        sparsity=0.0,
        fitness=original_fitness,
        original_fitness=original_fitness,
    )

    for iteration in range(iterations):
        active = _get_active_connections(genome)
        if not active:
            break
        remaining = n_total_to_prune - n_pruned_so_far[0]
        if remaining <= 0:
            break
        iters_left = iterations - iteration
        n_prune = max(1, round(remaining / iters_left))
        n_prune = min(n_prune, remaining, len(active))
        if n_prune <= 0:
            break
        by_mag = sorted(active, key=lambda c: abs(c._weight))
        to_prune = by_mag[:n_prune]
        for c in to_prune:
            c.enabled = False
        n_pruned_so_far[0] += len(to_prune)
        genome._invalidate_topology()

        if lamarck_steps > 0:
            _lamarck_finetune(genome, fitness_fn, lamarck_steps, lamarck_sigma)

        genome.reset()
        current_fitness = fitness_fn(genome)
        remaining_active = _get_active_connections(genome)
        n_orig = sum(1 for c in all_conns if original_states[id(c)][1])
        current_sparsity = 1.0 - (len(remaining_active) / max(1, n_orig))

        if current_fitness >= original_fitness - max_fitness_drop:
            ticket = LotteryTicket(
                mask=frozenset(c.innovation for c in remaining_active),
                sparsity=current_sparsity,
                fitness=current_fitness,
                original_fitness=original_fitness,
            )
            if ticket.sparsity > best_ticket.sparsity:
                best_ticket = ticket
        else:
            for c in to_prune:
                c.enabled = True
            genome._invalidate_topology()
            break

    for c in all_conns:
        orig_w, orig_enabled = original_states[id(c)]
        c._weight = orig_w
        if c._weight_arr is not None:
            c._weight_arr[c._weight_idx] = orig_w
        c.enabled = orig_enabled
    genome._invalidate_topology()

    return best_ticket


def _lamarck_finetune(genome, fitness_fn, steps, sigma):
    """Simple hill-climbing fine-tune."""
    import random
    conns = _get_active_connections(genome)
    nodes = genome.nodes
    genome.reset()
    best_fitness = fitness_fn(genome)
    for _ in range(steps):
        saved_w = [c._weight for c in conns]
        saved_b = [n.bias for n in nodes]
        for c in conns:
            c._weight += random.gauss(0.0, sigma)
            if c._weight_arr is not None:
                c._weight_arr[c._weight_idx] = c._weight
        for n in nodes:
            n.bias += random.gauss(0.0, sigma)
        genome.reset()
        new_fitness = fitness_fn(genome)
        if new_fitness > best_fitness:
            best_fitness = new_fitness
        else:
            for c, w in zip(conns, saved_w):
                c._weight = w
                if c._weight_arr is not None:
                    c._weight_arr[c._weight_idx] = w
            for n, b in zip(nodes, saved_b):
                n.bias = b


def apply_ticket(genome, ticket):
    """Disable connections not in the ticket mask."""
    mask = ticket.mask
    for node in genome.nodes:
        for c in node.connections:
            if c.innovation == -1:
                continue
            if c.innovation not in mask:
                c.enabled = False
    genome._invalidate_topology()
