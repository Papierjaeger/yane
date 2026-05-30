"""Tests for Sparse NEAT / Lottery Ticket Hypothesis (evolution/sparse_neat.py).

Acceptance criteria:
  1. find_lottery_ticket returns a LotteryTicket with correct structure.
  2. Ticket fitness >= original - max_fitness_drop.
  3. apply_ticket disables connections not in the mask.
  4. IMP finds sparser tickets than target_sparsity=0.
  5. LotteryTicket is serializable (pickle roundtrip).
  6. Genome.find_lottery_ticket() method delegates correctly.
  7. Genome.apply_ticket() method delegates correctly.
  8. NeuroEvolution.find_lottery_ticket() wrapper works.
  9. Genome is restored to original state after find_lottery_ticket.
 10. Connections with innovation=-1 are left as-is by apply_ticket.
"""
from __future__ import annotations

import pickle
import unittest

import pytest

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType
from yane.evolution.sparse_neat import (
    find_lottery_ticket,
    apply_ticket,
    LotteryTicket,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_connected_genome(n_inputs: int = 2, n_outputs: int = 1) -> Genome:
    """Genome with multiple weighted connections (easier to prune)."""
    g = Genome()
    inps = []
    for i in range(n_inputs):
        nd = Node(NodeType.INPUT, i)
        nd.activation = ActivationType.LINEAR
        nd.input_index = i
        g.nodes.append(nd)
        g.input_nodes.append(nd)
        inps.append(nd)

    out = Node(NodeType.OUTPUT, n_inputs)
    out.activation = ActivationType.LINEAR
    out.bias = 0.0
    g.nodes.append(out)
    g.output_nodes.append(out)

    weights = [0.001, 0.5, 0.8, 0.003, 0.9]
    for i, w in enumerate(weights[:n_inputs]):
        conn = Connection(out, innovation=10 + i)
        conn.weight = w
        conn.enabled = True
        inps[i % n_inputs].connections.append(conn)

    # Add extra connections with different weights
    for i in range(2, min(4, n_inputs + 1)):
        src = inps[0]
        conn = Connection(out, innovation=20 + i)
        conn.weight = weights[i] if i < len(weights) else 0.001
        conn.enabled = True
        src.connections.append(conn)

    g._invalidate_topology()
    return g


def _make_yane() -> NeuroEvolution:
    yane = NeuroEvolution(seed=42)
    yane.set_population_size(5)
    yane.configure(2, 1)
    return yane


def _count_active(genome: Genome) -> int:
    """Count enabled connections."""
    return sum(
        1 for nd in genome.nodes for c in nd.connections
        if c.enabled and c.innovation != -1
    )


# ---------------------------------------------------------------------------
# 1. Basic ticket structure
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestLotteryTicketStructure(unittest.TestCase):

    def test_returns_lottery_ticket(self):
        """find_lottery_ticket returns a LotteryTicket."""
        g = _make_connected_genome()
        ticket = find_lottery_ticket(g, lambda g: 0.0, target_sparsity=0.3)
        self.assertIsInstance(ticket, LotteryTicket)

    def test_ticket_has_correct_attributes(self):
        """LotteryTicket has mask, sparsity, fitness, original_fitness."""
        g = _make_connected_genome()
        ticket = find_lottery_ticket(g, lambda g: 5.0, target_sparsity=0.0)
        self.assertIsInstance(ticket.mask, frozenset)
        self.assertIsInstance(ticket.sparsity, float)
        self.assertIsInstance(ticket.fitness, float)
        self.assertIsInstance(ticket.original_fitness, float)
        self.assertAlmostEqual(ticket.original_fitness, 5.0)

    def test_zero_sparsity_keeps_all_connections(self):
        """target_sparsity=0 → ticket keeps all connections (sparsity=0)."""
        g = _make_connected_genome()
        ticket = find_lottery_ticket(g, lambda g: 1.0, target_sparsity=0.0, iterations=3)
        self.assertAlmostEqual(ticket.sparsity, 0.0)

    def test_invalid_sparsity_raises(self):
        """target_sparsity outside [0, 1) raises ValueError."""
        g = _make_connected_genome()
        with self.assertRaises(ValueError):
            find_lottery_ticket(g, lambda g: 0.0, target_sparsity=1.0)
        with self.assertRaises(ValueError):
            find_lottery_ticket(g, lambda g: 0.0, target_sparsity=-0.1)

    def test_invalid_iterations_raises(self):
        """iterations < 1 raises ValueError."""
        g = _make_connected_genome()
        with self.assertRaises(ValueError):
            find_lottery_ticket(g, lambda g: 0.0, iterations=0)


# ---------------------------------------------------------------------------
# 2. Fitness constraint
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTicketFitnessConstraint(unittest.TestCase):

    def test_ticket_fitness_within_max_drop(self):
        """Ticket fitness >= original_fitness - max_fitness_drop."""
        g = _make_connected_genome()
        original = 10.0
        drop = 2.0
        ticket = find_lottery_ticket(
            g, lambda g: original, target_sparsity=0.4, max_fitness_drop=drop
        )
        self.assertGreaterEqual(ticket.fitness, original - drop - 1e-9)

    def test_genome_restored_after_search(self):
        """Genome connections are fully restored after find_lottery_ticket."""
        g = _make_connected_genome()
        before_count = _count_active(g)
        before_weights = [c._weight for nd in g.nodes for c in nd.connections if c.enabled]

        find_lottery_ticket(g, lambda g: 1.0, target_sparsity=0.5)

        after_count = _count_active(g)
        after_weights = [c._weight for nd in g.nodes for c in nd.connections if c.enabled]

        self.assertEqual(before_count, after_count)
        for w1, w2 in zip(before_weights, after_weights):
            self.assertAlmostEqual(w1, w2, places=10)


# ---------------------------------------------------------------------------
# 3. apply_ticket
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestApplyTicket(unittest.TestCase):

    def test_apply_disables_pruned_connections(self):
        """apply_ticket disables connections not in the ticket mask."""
        g = _make_connected_genome()
        original_count = _count_active(g)
        ticket = find_lottery_ticket(g, lambda g: 1.0, target_sparsity=0.4, iterations=2)
        apply_ticket(g, ticket)
        after_count = _count_active(g)
        # At least as sparse as the ticket
        self.assertLessEqual(after_count, original_count)

    def test_apply_preserves_mask_connections(self):
        """Connections in the ticket mask remain enabled."""
        g = _make_connected_genome()
        ticket = find_lottery_ticket(g, lambda g: 1.0, target_sparsity=0.3)
        apply_ticket(g, ticket)
        for nd in g.nodes:
            for c in nd.connections:
                if c.innovation != -1 and c.enabled:
                    self.assertIn(c.innovation, ticket.mask)

    def test_apply_leaves_untracked_connections(self):
        """apply_ticket leaves connections with innovation=-1 unchanged."""
        g = _make_connected_genome()
        # Add an untracked connection
        out = g.output_nodes[0]
        untracked = Connection(out, innovation=-1)
        untracked.weight = 0.0001
        untracked.enabled = True
        g.input_nodes[0].connections.append(untracked)
        g._invalidate_topology()

        # Find a ticket that prunes a lot
        ticket = find_lottery_ticket(g, lambda g: 0.0, target_sparsity=0.5)
        apply_ticket(g, ticket)

        # Untracked connection must still be enabled
        self.assertTrue(untracked.enabled)

    def test_genome_apply_ticket_method(self):
        """Genome.apply_ticket() works as expected."""
        g = _make_connected_genome()
        ticket = g.find_lottery_ticket(lambda g: 1.0, target_sparsity=0.0)
        g.apply_ticket(ticket)  # Should not raise


# ---------------------------------------------------------------------------
# 4. Ticket serialization
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestTicketSerialization(unittest.TestCase):

    def test_pickle_roundtrip(self):
        """LotteryTicket survives pickle serialization."""
        g = _make_connected_genome()
        ticket = find_lottery_ticket(g, lambda g: 3.0, target_sparsity=0.2)
        data = pickle.dumps(ticket)
        restored = pickle.loads(data)
        self.assertEqual(restored.mask, ticket.mask)
        self.assertAlmostEqual(restored.sparsity, ticket.sparsity)
        self.assertAlmostEqual(restored.fitness, ticket.fitness)
        self.assertAlmostEqual(restored.original_fitness, ticket.original_fitness)


# ---------------------------------------------------------------------------
# 5. Genome.find_lottery_ticket() method
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestGenomeMethod(unittest.TestCase):

    def test_genome_find_lottery_ticket(self):
        """Genome.find_lottery_ticket() delegates to sparse_neat module."""
        g = _make_connected_genome()
        ticket = g.find_lottery_ticket(lambda g: 1.0, target_sparsity=0.0)
        self.assertIsInstance(ticket, LotteryTicket)

    def test_genome_find_with_sparsity(self):
        """Genome.find_lottery_ticket() with actual sparsity target."""
        g = _make_connected_genome()
        ticket = g.find_lottery_ticket(
            lambda g: 5.0, target_sparsity=0.4, max_fitness_drop=5.0, iterations=3
        )
        self.assertIsInstance(ticket, LotteryTicket)
        self.assertAlmostEqual(ticket.original_fitness, 5.0)


# ---------------------------------------------------------------------------
# 6. NeuroEvolution.find_lottery_ticket() wrapper
# ---------------------------------------------------------------------------

@pytest.mark.ci
class TestNeuroEvolutionLotteryTicket(unittest.TestCase):

    def test_ne_find_lottery_ticket(self):
        """NeuroEvolution.find_lottery_ticket() calls the IMP on the best genome."""
        yane = _make_yane()
        # Give all genomes a fitness so get_best() works
        for g in yane._population._unevaluated:
            g.fitness = 1.0
        yane._population._evaluated.extend(yane._population._unevaluated)
        yane._population._unevaluated.clear()

        ticket = yane.find_lottery_ticket(lambda g: 1.0, target_sparsity=0.0)
        self.assertIsInstance(ticket, LotteryTicket)

    def test_ne_find_lottery_ticket_not_configured_raises(self):
        """find_lottery_ticket raises when not configured."""
        yane = NeuroEvolution(seed=0)
        with self.assertRaises(Exception):
            yane.find_lottery_ticket(lambda g: 0.0)
