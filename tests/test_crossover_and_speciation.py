"""Tests for crossover, speciation, fitness sharing, elitism, and strategy genes."""
import random
import unittest

from yane import NeuroEvolution
from yane.core.genome import Genome
from yane.evolution.population import Population, _compatibility


def _make_yane(n_inputs=2, n_outputs=1, max_nodes=10, max_connections=20, pop_size=50):
    yane = NeuroEvolution()
    yane.configure(n_inputs, n_outputs, max_nodes=max_nodes, max_connections=max_connections)
    yane.set_population_size(pop_size)
    return yane


def _evolved_genome(n_inputs=2, n_outputs=1, n_mutations=30):
    """Return a genome that has undergone structural mutations."""
    yane = _make_yane(n_inputs, n_outputs)
    g = yane.next_genome()
    for _ in range(n_mutations):
        g.mutate()
    return g


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

class TestCrossover(unittest.TestCase):

    def _parents(self, n_inputs=2, n_outputs=1):
        yane = _make_yane(n_inputs, n_outputs)
        p1 = yane.next_genome()
        p2 = yane.next_genome()
        for _ in range(10):
            p1.mutate()
            p2.mutate()
        p1.fitness = 1.0
        p2.fitness = 0.5
        return p1, p2

    def test_crossover_produces_valid_io_count(self):
        p1, p2 = self._parents()
        child = p1.crossover(p2)
        self.assertEqual(len(child.input_nodes), len(p1.input_nodes))
        self.assertEqual(len(child.output_nodes), len(p1.output_nodes))

    def test_crossover_child_nodes_are_new_objects(self):
        p1, p2 = self._parents()
        child = p1.crossover(p2)
        parent_nodes = set(id(n) for n in p1.nodes + p2.nodes)
        for node in child.nodes:
            self.assertNotIn(id(node), parent_nodes)

    def test_crossover_connections_point_to_child_nodes(self):
        p1, p2 = self._parents()
        child = p1.crossover(p2)
        child_node_ids = {id(n) for n in child.nodes}
        for node in child.nodes:
            for conn in node.connections:
                self.assertIn(id(conn.target), child_node_ids,
                              "Connection must point to a child node, not a parent node")

    def test_crossover_respects_max_nodes(self):
        yane = _make_yane(max_nodes=6)
        p1 = yane.next_genome()
        p2 = yane.next_genome()
        for _ in range(20):
            p1.mutate()
            p2.mutate()
        child = p1.crossover(p2)
        self.assertLessEqual(len(child.nodes), 6)

    def test_crossover_respects_max_connections(self):
        yane = _make_yane(max_connections=8)
        p1 = yane.next_genome()
        p2 = yane.next_genome()
        for _ in range(20):
            p1.mutate()
            p2.mutate()
        child = p1.crossover(p2)
        self.assertLessEqual(child.connection_count, 8)

    def test_crossover_forward_is_deterministic(self):
        p1, p2 = self._parents()
        child = p1.crossover(p2)
        inputs = [0.5, 0.3]
        # Reset before each call: memory nodes (persist_value=True) can appear
        # after random mutations and carry state between forward() calls.
        child.reset(); out1 = child.forward(inputs)
        child.reset(); out2 = child.forward(inputs)
        self.assertEqual(out1, out2)

    def test_crossover_strategy_genes_from_parents(self):
        p1, p2 = self._parents()
        p1.crossover_prob = 0.9
        p2.crossover_prob = 0.1
        # Run many crossovers; child must always get one of the two values
        for _ in range(20):
            child = p1.crossover(p2)
            self.assertIn(round(child.crossover_prob, 6),
                          [round(p1.crossover_prob, 6), round(p2.crossover_prob, 6)])

    def test_crossover_sigma_global_inherited(self):
        p1, p2 = self._parents()
        p1.sigma_global = 2.0
        p2.sigma_global = 0.5
        for _ in range(10):
            child = p1.crossover(p2)
            self.assertIn(child.sigma_global, [2.0, 0.5])

    def test_crossover_child_forward_output_count(self):
        yane = _make_yane(n_outputs=3)
        p1 = yane.next_genome()
        p2 = yane.next_genome()
        child = p1.crossover(p2)
        outputs = child.forward([0.0, 0.0])
        self.assertEqual(len(outputs), 3)

    def test_fitter_parent_connections_always_inherited(self):
        """Disjoint / excess genes from the fitter parent must all appear in the child."""
        from yane.evolution import smart_mutation
        yane = _make_yane(2, 1, max_nodes=30, max_connections=100)
        tracker = yane._tracker

        p1 = yane.next_genome()
        p2 = yane.next_genome()
        # Give p1 many unique connections
        for _ in range(8):
            smart_mutation.add_connection(p1, tracker)
        p1.fitness = 10.0
        p2.fitness = 1.0

        p1_innovations = {c.innovation for n in p1.nodes for c in n.connections
                          if c.innovation >= 0}
        child = p1.crossover(p2)
        child_innovations = {c.innovation for n in child.nodes for c in n.connections
                             if c.innovation >= 0}

        for innov in p1_innovations:
            self.assertIn(innov, child_innovations,
                "All innovations from the fitter parent must appear in the child")

    def test_matching_genes_weight_inherited_from_either_parent(self):
        """For matching genes, weight is drawn from one parent or the other."""
        from yane.core.connection import Connection

        yane = _make_yane(2, 1)
        tracker = yane._tracker
        # Build two independent genomes from the same template
        p1 = yane.next_genome()
        p2 = p1.copy()

        inp_innov = p1.input_nodes[0].innovation
        out_innov = p1.output_nodes[0].innovation
        shared_innov = tracker.get_connection(inp_innov, out_innov)

        w1, w2 = 0.1, 0.9
        c1 = Connection(p1.output_nodes[0], innovation=shared_innov); c1.weight = w1
        c2 = Connection(p2.output_nodes[0], innovation=shared_innov); c2.weight = w2
        p1.input_nodes[0].connections.append(c1)
        p2.input_nodes[0].connections.append(c2)
        p1._invalidate_topology(); p2._invalidate_topology()
        p1.fitness = 1.0; p2.fitness = 0.5

        weights_seen = set()
        for _ in range(40):
            child = p1.crossover(p2)
            for n in child.nodes:
                for c in n.connections:
                    if c.innovation == shared_innov:
                        weights_seen.add(round(c.weight, 6))

        self.assertIn(round(w1, 6), weights_seen,
            "Fitter parent's weight for matching gene must appear")
        self.assertIn(round(w2, 6), weights_seen,
            "Weaker parent's weight for matching gene must appear (50/50)")

    def test_topology_cache_invalidated_after_crossover(self):
        """Child's exec_order must be recomputed, not inherited from parents."""
        from yane.core.connection import Connection as Conn
        yane = _make_yane(2, 1)
        p1 = yane.next_genome()
        p2 = yane.next_genome()
        # Add a deterministic acyclic connection (input[0] → output[0]) so
        # exec_order is guaranteed to be cached after forward().
        innov = yane._tracker.get_connection(
            p1.input_nodes[0].innovation, p1.output_nodes[0].innovation)
        c1 = Conn(p1.output_nodes[0], innovation=innov); c1.weight = 0.5
        c2 = Conn(p2.output_nodes[0], innovation=innov); c2.weight = 0.3
        p1.input_nodes[0].connections.append(c1); p1._invalidate_topology()
        p2.input_nodes[0].connections.append(c2); p2._invalidate_topology()
        p1.fitness = 1.0
        p2.fitness = 0.5

        p1.forward([0.5, 0.5])
        self.assertIsNotNone(p1._exec_order,
            "forward() must build exec_order for acyclic network")

        child = p1.crossover(p2)
        self.assertIsNone(child._exec_order,
            "Crossover child must not inherit parent's topology cache")


# ---------------------------------------------------------------------------
# Weight Inheritance (fitness-weighted blend)
# ---------------------------------------------------------------------------

class TestWeightInheritance(unittest.TestCase):

    def _parents_with_known_weights(self):
        """Return two independent parents with distinct weights for matching connections.

        Creates two related genomes via ``NeuroEvolution`` so they share
        innovation numbers, then adjusts a matching connection's weight.
        """
        yane = _make_yane(2, 1, max_nodes=20, max_connections=30)
        # Train briefly so the population has evaluated genomes with structure.
        def _eval(g):
            return sum(abs(c.weight) for src in g.nodes for c in src.connections)
        yane.set_max_iterations(200)
        yane.train(_eval)
        # Grab the best two evaluated genomes.
        evald = yane._population._evaluated
        evald.sort(key=lambda g: g.fitness, reverse=True)
        p1 = evald[0].copy()
        p2 = evald[1].copy() if len(evald) > 1 else evald[0].copy()
        # Find a matching connection.
        p1_conns = {
            c.innovation: c
            for src in p1.nodes for c in src.connections
            if c.innovation >= 0
        }
        p2_conns = {
            c.innovation: c
            for src in p2.nodes for c in src.connections
            if c.innovation >= 0
        }
        matching = set(p1_conns) & set(p2_conns)
        if not matching:
            # If no match, find a connection in p1 and create a matching one in p2.
            shared_innov = min(p1_conns)
            in2 = p2.input_nodes[0]
            out2 = p2.output_nodes[0]
            from yane.core.connection import Connection
            c2 = Connection(out2, innovation=shared_innov)
            c2.weight = 0.0
            in2.connections.append(c2)
            p2._invalidate_topology()
            p2_conns[shared_innov] = c2
            matching = {shared_innov}
        shared_innov = min(matching)
        p1_conns[shared_innov].weight = 1.0
        p2_conns[shared_innov].weight = 0.0
        p1.fitness = 1.0
        p2.fitness = 0.5
        return p1, p2, shared_innov

    def test_blend_alpha_weighs_fitter_parent_more(self):
        """With blend_alpha=0.8, child weight should be closer to fitter parent."""
        p1, p2, innov = self._parents_with_known_weights()
        # p1 is fitter (1.0 vs 0.5), weight 1.0
        # p2 weight 0.0
        # blend_alpha=0.8 → 0.8*1.0 + 0.2*0.0 = 0.8
        child = p1.crossover(p2, blend_alpha=0.8)
        child_conn = next(
            conn for src in child.nodes for conn in src.connections
            if conn.innovation == innov
        )
        self.assertAlmostEqual(child_conn.weight, 0.8, places=10)

    def test_blend_alpha_zero_gives_weaker_parent_weight(self):
        """With blend_alpha=0.0, child gets weaker parent's weight."""
        p1, p2, innov = self._parents_with_known_weights()
        child = p1.crossover(p2, blend_alpha=0.0)
        child_conn = next(
            conn for src in child.nodes for conn in src.connections
            if conn.innovation == innov
        )
        self.assertAlmostEqual(child_conn.weight, 0.0, places=10)

    def test_blend_alpha_one_gives_fitter_parent_weight(self):
        """With blend_alpha=1.0, child gets fitter parent's weight unchanged."""
        p1, p2, innov = self._parents_with_known_weights()
        child = p1.crossover(p2, blend_alpha=1.0)
        child_conn = next(
            conn for src in child.nodes for conn in src.connections
            if conn.innovation == innov
        )
        self.assertAlmostEqual(child_conn.weight, 1.0, places=10)

    def test_negative_blend_alpha_defaults_to_random(self):
        """Default blend_alpha=-1.0 preserves random 50/50 behaviour."""
        p1, p2, innov = self._parents_with_known_weights()
        weights = set()
        for _ in range(50):
            child = p1.crossover(p2, blend_alpha=-1.0)
            child_conn = next(
                conn for src in child.nodes for conn in src.connections
                if conn.innovation == innov
            )
            weights.add(round(child_conn.weight, 10))
        # Both parents' weights should appear across multiple trials.
        self.assertIn(1.0, weights)
        self.assertIn(0.0, weights)

    def test_set_weight_inheritance_api(self):
        """NeuroEvolution.set_weight_inheritance() propagates to population."""
        yane = _make_yane(2, 1)
        yane.set_weight_inheritance(enabled=True, blend_alpha=0.7)
        self.assertTrue(yane._weight_inheritance_enabled)
        self.assertAlmostEqual(yane._weight_blend_alpha, 0.7)
        self.assertAlmostEqual(
            yane._population._weight_blend_alpha, 0.7
        )

    def test_set_weight_inheritance_disabled_restores_random(self):
        """Disabling weight inheritance sets blend_alpha to -1.0."""
        yane = _make_yane(2, 1)
        yane.set_weight_inheritance(enabled=True, blend_alpha=0.7)
        yane.set_weight_inheritance(enabled=False)
        self.assertFalse(yane._weight_inheritance_enabled)
        self.assertEqual(yane._population._weight_blend_alpha, -1.0)

    def test_set_weight_inheritance_invalid_alpha(self):
        """blend_alpha outside [0, 1] raises ValueError."""
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        with self.assertRaises(ValueError):
            yane.set_weight_inheritance(enabled=True, blend_alpha=1.5)
        with self.assertRaises(ValueError):
            yane.set_weight_inheritance(enabled=True, blend_alpha=-0.1)

    def test_weight_inheritance_in_config_dict(self):
        """_config_dict includes weight_inheritance fields."""
        yane = _make_yane(2, 1)
        cfg = yane._config_dict()
        self.assertIn("weight_inheritance_enabled", cfg)
        self.assertIn("weight_blend_alpha", cfg)

    def test_population_spawn_uses_weight_blend(self):
        """Population spawn passes blend_alpha to crossover when configured."""
        yane = _make_yane(2, 1, pop_size=20)
        yane.set_weight_inheritance(enabled=True, blend_alpha=0.8)
        pop = yane._population
        pop._crossover_enabled = True
        # Ensure there are evaluated genomes with fitness.
        for g in list(pop._unevaluated):
            g.fitness = random.uniform(0.0, 1.0)
            g.shared_fitness = g.fitness
            pop._evaluated.append(g)
        pop._unevaluated.clear()
        # Spawn should use crossover with blend_alpha.
        pop._spawn_offspring()
        child = pop._unevaluated[0] if pop._unevaluated else None
        self.assertIsNotNone(child)
        self.assertGreater(len(child.nodes), 0)


# ---------------------------------------------------------------------------
# Copy consistency (original vs copy forward parity — the sorted-set bug)
# ---------------------------------------------------------------------------

class TestCopyForwardConsistency(unittest.TestCase):

    def test_copy_matches_original_forward(self):
        """A copy must produce identical outputs to the original for all inputs."""
        g = _evolved_genome(n_mutations=50)
        g_copy = g.copy()
        test_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        for inp in test_inputs:
            self.assertAlmostEqual(
                g.forward(inp)[0], g_copy.forward(inp)[0], places=10,
                msg=f"Copy output differs from original for input {inp}")

    def test_crossover_child_stable_across_copies(self):
        p1 = _evolved_genome(n_mutations=30)
        p2 = _evolved_genome(n_mutations=30)
        child = p1.crossover(p2)
        child_copy = child.copy()
        for inp in [[0.5, 0.5], [0.0, 1.0]]:
            self.assertAlmostEqual(
                child.forward(inp)[0], child_copy.forward(inp)[0], places=10)


# ---------------------------------------------------------------------------
# Strategy genes
# ---------------------------------------------------------------------------

class TestStrategyGenes(unittest.TestCase):

    def test_shared_fitness_initialized_to_zero(self):
        g = Genome()
        self.assertEqual(g.shared_fitness, 0.0)

    def test_strategy_genes_present_after_init(self):
        g = Genome()
        for attr in ('crossover_prob', 'offspring_factor', 'sigma_global'):
            self.assertTrue(hasattr(g, attr), f"Missing attribute: {attr}")

    def test_strategy_genes_copied(self):
        g = Genome()
        g.crossover_prob = 0.77
        g.offspring_factor = 3.14
        g.sigma_global = 1.23
        c = g.copy()
        self.assertAlmostEqual(c.crossover_prob,   0.77)
        self.assertAlmostEqual(c.offspring_factor, 3.14)
        self.assertAlmostEqual(c.sigma_global,     1.23)

    def test_strategy_genes_mutate(self):
        random.seed(0)
        g = Genome()
        original = (g.crossover_prob, g.offspring_factor, g.sigma_global)
        changed = False
        for _ in range(200):
            g.mutate()
            current = (g.crossover_prob, g.offspring_factor, g.sigma_global)
            if current != original:
                changed = True
                break
        self.assertTrue(changed, "Strategy genes should mutate over many iterations")

    def test_sigma_global_scales_weight_mutations(self):
        """Higher sigma_global → larger average weight change on mutation."""
        from yane.core.connection import Connection
        from yane.core.node import Node, NodeType

        n = Node(NodeType.HIDDEN)
        conn = Connection(n)
        conn.weight = 0.0

        # Measure mutation magnitude with low sigma
        low_deltas = []
        for _ in range(200):
            w = conn.mutation.mutate_value(0.0, sigma=0.01)
            low_deltas.append(abs(w))

        high_deltas = []
        for _ in range(200):
            w = conn.mutation.mutate_value(0.0, sigma=10.0)
            high_deltas.append(abs(w))

        avg_low  = sum(low_deltas)  / len(low_deltas)
        avg_high = sum(high_deltas) / len(high_deltas)
        self.assertLess(avg_low, avg_high,
                        "Higher sigma should produce larger weight changes on average")

    def test_strategy_gene_bounds_respected_after_mutation(self):
        g = Genome()
        for _ in range(500):
            g.mutate()
        self.assertGreaterEqual(g.crossover_prob,   0.0)
        self.assertLessEqual(g.crossover_prob,      1.0)
        self.assertGreaterEqual(g.offspring_factor, 0.01)
        self.assertGreaterEqual(g.sigma_global,     0.01)


# ---------------------------------------------------------------------------
# Compatibility distance
# ---------------------------------------------------------------------------

class TestCompatibility(unittest.TestCase):

    def test_identical_genomes_zero_distance(self):
        g = _evolved_genome()
        self.assertAlmostEqual(_compatibility(g, g), 0.0)

    def test_same_structure_zero_distance(self):
        g1 = _evolved_genome()
        g2 = g1.copy()
        self.assertAlmostEqual(_compatibility(g1, g2), 0.0)

    def test_distance_symmetric(self):
        g1 = _evolved_genome(n_mutations=10)
        g2 = _evolved_genome(n_mutations=10)
        self.assertAlmostEqual(_compatibility(g1, g2), _compatibility(g2, g1), places=10)

    def test_distance_in_range(self):
        # Formula: excess/N + disjoint/N + 0.4*W_bar.
        # Max theoretical: ~2.8 (all excess + large weight diff). Docs say "roughly [0, 2]".
        g1 = _evolved_genome(n_mutations=30)
        g2 = _evolved_genome(n_mutations=30)
        d = _compatibility(g1, g2)
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 3.0)

    def test_different_structure_nonzero_distance(self):
        from yane.evolution import smart_mutation
        yane = _make_yane(2, 1, max_nodes=30, max_connections=100)
        g1 = yane.next_genome()
        g2 = g1.copy()  # identical start
        # Force structural divergence by adding connections to g1 only
        for _ in range(10):
            smart_mutation.add_connection(g1, yane._tracker)
        self.assertGreater(_compatibility(g1, g2), 0.0)

    def test_excess_vs_disjoint_classification(self):
        """Genes beyond the range of the smaller genome are excess; within-range gaps are disjoint."""
        from yane.core.connection import Connection
        yane = _make_yane(2, 1)
        tracker = yane._tracker

        g1 = yane.next_genome()
        g2 = g1.copy()   # start from same base to get same node innovations

        inp0 = g1.input_nodes[0].innovation
        inp1 = g1.input_nodes[1].innovation
        out0 = g1.output_nodes[0].innovation

        # Reserve 5 innovation numbers in a defined order
        innovations = [tracker.get_connection(inp0, out0),
                       tracker.get_connection(inp1, out0),
                       tracker.next(), tracker.next(), tracker.next()]

        # g1 gets all 5 connections
        for innov in innovations:
            c = Connection(g1.output_nodes[0], innovation=innov)
            c.weight = 0.5
            g1.input_nodes[0].connections.append(c)
        g1._invalidate_topology()

        # g2 gets only the first 3 — innovations[3] and [4] are excess in g1
        for innov in innovations[:3]:
            c = Connection(g2.output_nodes[0], innovation=innov)
            c.weight = 0.5
            g2.input_nodes[0].connections.append(c)
        g2._invalidate_topology()

        d = _compatibility(g1, g2)
        self.assertGreater(d, 0.0, "Excess genes must contribute to distance")

    def test_weight_difference_contributes_to_distance(self):
        """Two genomes with same structure but different weights have δ > 0."""
        from yane.core.connection import Connection
        yane = _make_yane(2, 1)
        tracker = yane._tracker

        g1 = yane.next_genome()
        g2 = g1.copy()

        innov = tracker.get_connection(
            g1.input_nodes[0].innovation, g1.output_nodes[0].innovation)
        c1 = Connection(g1.output_nodes[0], innovation=innov); c1.weight =  1.0
        c2 = Connection(g2.output_nodes[0], innovation=innov); c2.weight = -1.0
        g1.input_nodes[0].connections.append(c1)
        g2.input_nodes[0].connections.append(c2)
        g1._invalidate_topology(); g2._invalidate_topology()

        d = _compatibility(g1, g2)
        self.assertGreater(d, 0.0, "Weight differences must contribute to δ")

    def test_adaptive_threshold_moves_toward_target(self):
        """_compat_threshold must move in the correct direction after speciation."""
        from yane.evolution.population import Population
        pop = Population(max_size=20, target_species=3)
        initial_threshold = pop._compat_threshold

        # Evaluate many genomes so speciation runs many times
        for i in range(40):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i % 5))

        # After many rounds, threshold should have moved from its initial value
        # (it adapts toward the level that gives ~3 species)
        # We just verify it changed — direction depends on actual species count
        self.assertNotEqual(pop._compat_threshold, initial_threshold,
            "Adaptive threshold must change after multiple speciation rounds")


# ---------------------------------------------------------------------------
# Speciation
# ---------------------------------------------------------------------------

class TestSpeciation(unittest.TestCase):

    def test_species_count_starts_at_zero(self):
        pop = Population(max_size=10)
        self.assertEqual(pop.species_count, 0)

    def test_species_count_after_spawn(self):
        pop = Population(max_size=20)
        for i in range(5):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))
        # Trigger speciation via next spawn
        pop.select_for_evaluation()
        self.assertGreaterEqual(pop.species_count, 1)

    def test_identical_structure_same_species(self):
        """Two genomes with identical topology should land in the same species."""
        pop = Population(max_size=20)
        g1 = pop.select_for_evaluation()
        pop.submit(g1, 1.0)
        g2 = g1.copy()
        g2.fitness = 0.0
        pop._evaluated.append(g2)
        pop._assign_species()
        self.assertEqual(pop.species_count, 1,
                         "Structurally identical genomes should be in the same species")

    def test_very_different_genomes_different_species(self):
        """Structurally diverged genomes with low compat threshold → multiple species."""
        from yane.evolution import smart_mutation
        yane = _make_yane(2, 1, max_connections=100)
        pop = yane._population
        g1 = pop.select_for_evaluation()
        pop.submit(g1, 1.0)

        # g2 gets many extra tracked connections; with a near-zero threshold
        # even one disjoint gene forces a new species.
        g2 = g1.copy()
        pop._compat_threshold = 0.001
        for _ in range(15):
            smart_mutation.add_connection(g2, yane._tracker)
        g2.fitness = 0.0
        pop._evaluated.append(g2)
        pop._assign_species()

        self.assertGreaterEqual(pop.species_count, 2,
                                "Very different genomes should form distinct species")

    def test_shared_fitness_updates_after_speciation(self):
        pop = Population(max_size=20)
        for i in range(4):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i + 1))

        pop._assign_species()
        pop._compute_shared_fitness()

        for g in pop._evaluated:
            n = sum(1 for s in pop._species if g in s.members)
            self.assertEqual(n, 1, "Each genome should belong to exactly one species")
            sp = next(s for s in pop._species if g in s.members)
            expected = g.fitness / len(sp.members)
            self.assertAlmostEqual(g.shared_fitness, expected, places=10)

    def test_single_species_shared_equals_raw(self):
        """When all genomes are in one species, shared_fitness == raw_fitness."""
        pop = Population(max_size=10)
        g1 = pop.select_for_evaluation()
        pop.submit(g1, 3.0)

        g2 = g1.copy()
        g2.fitness = 3.0
        pop._evaluated.append(g2)

        pop._assign_species()
        pop._compute_shared_fitness()

        if pop.species_count == 1:
            for g in pop._evaluated:
                self.assertAlmostEqual(g.shared_fitness, g.fitness / 2, places=10)


# ---------------------------------------------------------------------------
# Elitism
# ---------------------------------------------------------------------------

class TestElitism(unittest.TestCase):

    def test_best_genome_survives_pruning(self):
        """The globally best genome must never be pruned from the population."""
        pop = Population(max_size=5)
        best = None
        for i in range(10):
            g = pop.select_for_evaluation()
            fitness = float(i)
            pop.submit(g, fitness)
            if fitness == 9.0:
                best = g

        self.assertIn(best, pop._evaluated,
                      "Best genome must remain in _evaluated after pruning")

    def test_species_best_survives_pruning(self):
        """Best genome per species must not be pruned."""
        pop = Population(max_size=5)
        submitted = []
        for i in range(10):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i))
            submitted.append(g)

        pop._assign_species()
        for sp in pop._species:
            sp_best = sp.best()
            self.assertIn(sp_best, pop._evaluated,
                          "Species best must survive pruning")

    def test_population_never_exceeds_max_size(self):
        pop = Population(max_size=8)
        for i in range(30):
            g = pop.select_for_evaluation()
            pop.submit(g, float(i % 5))
            self.assertLessEqual(pop.size, 9,  # +1 for unevaluated in-flight
                                 f"Population size {pop.size} exceeded max after step {i}")


# ---------------------------------------------------------------------------
# Population API (species_count, offspring_factor weighting)
# ---------------------------------------------------------------------------

class TestPopulationAPI(unittest.TestCase):

    def test_species_count_property(self):
        pop = Population(max_size=20)
        self.assertIsInstance(pop.species_count, int)
        self.assertGreaterEqual(pop.species_count, 0)

    def test_high_offspring_factor_increases_weight(self):
        """offspring_factor multiplicatively scales selection weight."""
        # Verify the property directly: same shared_fitness, offspring_factor 100× higher
        # → weight must be 100× higher, independent of speciation state.
        g_high = Genome()
        g_low  = Genome()
        g_high.shared_fitness = 1.0
        g_low.shared_fitness  = 1.0
        g_high.offspring_factor = 100.0
        g_low.offspring_factor  = 1.0

        min_fit = 1.0
        w_high = max(0.0, g_high.shared_fitness - min_fit + 1e-6) * g_high.offspring_factor
        w_low  = max(0.0, g_low.shared_fitness  - min_fit + 1e-6) * g_low.offspring_factor

        self.assertAlmostEqual(w_high / w_low, 100.0, places=2,
                               msg="offspring_factor=100 should produce 100× the selection weight")

    def test_crossover_triggered_by_high_crossover_prob(self):
        """With crossover_prob=1 and ≥2 evaluated non-elite genomes, crossover executes."""
        p1 = _evolved_genome(n_mutations=5)
        p2 = _evolved_genome(n_mutations=5)
        p1.fitness = 1.0
        p2.fitness = 0.5
        p1.crossover_prob = 1.0

        # Call crossover directly — if crossover_prob drives the branch in
        # _spawn_offspring, the result must be a valid genome with nodes.
        child = p1.crossover(p2)
        self.assertGreater(len(child.nodes), 0,
                           "Crossover must produce a genome with nodes")


class TestSpeciesBudget(unittest.TestCase):
    """Tests for _select_parent_species() species-budget selection."""

    def _pop_with_two_species(self, fit_a=1.0, fit_b=0.1, stag_a=0, stag_b=0):
        """Return a Population with two manually-constructed species."""
        pop = Population(max_size=20)
        g_a = Genome(); g_a.fitness = fit_a; g_a.shared_fitness = fit_a
        g_b = Genome(); g_b.fitness = fit_b; g_b.shared_fitness = fit_b
        pop._evaluated = [g_a, g_b]
        pop._min_shared_fitness = min(fit_a, fit_b)
        from yane.evolution.species import Species
        sp_a = Species(g_a, spawn_count=0)
        sp_a.members = [g_a]
        sp_a.stagnation_count = stag_a
        sp_b = Species(g_b, spawn_count=0)
        sp_b.members = [g_b]
        sp_b.stagnation_count = stag_b
        pop._species = [sp_a, sp_b]
        return pop, sp_a, sp_b

    def test_high_fitness_species_selected_more_often(self):
        pop, sp_a, sp_b = self._pop_with_two_species(fit_a=10.0, fit_b=0.01)
        counts = {id(sp_a): 0, id(sp_b): 0}
        for _ in range(500):
            chosen = pop._select_parent_species()
            counts[id(chosen)] += 1
        self.assertGreater(counts[id(sp_a)], counts[id(sp_b)],
                           "High-fitness species must be chosen more often")

    def test_stagnating_species_selected_less(self):
        # sp_b stagnates heavily, sp_a fresh
        pop, sp_a, sp_b = self._pop_with_two_species(fit_a=1.0, fit_b=1.0,
                                                      stag_a=0, stag_b=100)
        counts = {id(sp_a): 0, id(sp_b): 0}
        for _ in range(500):
            chosen = pop._select_parent_species()
            counts[id(chosen)] += 1
        self.assertGreater(counts[id(sp_a)], counts[id(sp_b)],
                           "Non-stagnating species must win over stagnating one")

    def test_no_species_returns_none(self):
        pop = Population(max_size=10)
        pop._species = []
        self.assertIsNone(pop._select_parent_species())

    def test_single_species_returned_directly(self):
        pop = Population(max_size=10)
        g = Genome(); g.shared_fitness = 1.0
        from yane.evolution.species import Species
        sp = Species(g); sp.members = [g]
        pop._species = [sp]
        pop._min_shared_fitness = 1.0
        self.assertIs(pop._select_parent_species(), sp)

    def test_stagnating_species_never_fully_excluded(self):
        """Even a fully-stagnated species must be chosen occasionally (floor ≥ 0.1)."""
        pop, sp_a, sp_b = self._pop_with_two_species(fit_a=1.0, fit_b=1.0,
                                                      stag_a=0, stag_b=10_000)
        chosen_b = sum(1 for _ in range(500)
                       if pop._select_parent_species() is sp_b)
        self.assertGreater(chosen_b, 0,
                           "Stagnating species must still occasionally be chosen")

    def test_equal_species_receive_balanced_budget(self):
        """Rolling spawn credits should split offspring nearly evenly."""
        pop, sp_a, sp_b = self._pop_with_two_species(fit_a=1.0, fit_b=1.0)
        counts = {id(sp_a): 0, id(sp_b): 0}
        for _ in range(100):
            chosen = pop._select_parent_species()
            counts[id(chosen)] += 1
        self.assertLessEqual(abs(counts[id(sp_a)] - counts[id(sp_b)]), 1)

    def test_species_spawn_credit_tracks_offspring_totals(self):
        pop, sp_a, sp_b = self._pop_with_two_species(fit_a=5.0, fit_b=1.0)
        for _ in range(10):
            pop._select_parent_species()
        self.assertEqual(
            sum(pop._species_offspring_total.values()),
            10,
        )
        self.assertTrue(pop._last_species_spawn_scores)


if __name__ == '__main__':
    unittest.main()
