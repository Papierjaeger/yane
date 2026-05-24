import unittest

from yane import NeuroEvolution
from yane.benchmarks.profile_serialization import profile_one
from yane.evolution.async_evaluation import evaluate_batch_async
from yane.evolution.descriptors import (
    DEFAULT_DESCRIPTORS,
    FitnessComponent,
    scalarize_components,
    topology_descriptor,
)
from yane.evolution.matrix_export import MatrixForwardCache, forward_compatible_batch


class TestAsyncDescriptorsProfile(unittest.TestCase):
    def test_async_evaluate_batch(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        genomes = [yane.next_genome()]

        results = evaluate_batch_async(genomes, lambda g: float(len(g.nodes)), max_workers=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 2.0)

    def test_descriptor_registry_and_scalarization(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        g = yane.next_genome()

        self.assertIn("topology", DEFAULT_DESCRIPTORS.names())
        self.assertEqual(topology_descriptor(g), (0.0, 0.0))
        score, values = scalarize_components(
            g,
            [FitnessComponent("nodes", lambda genome: len(genome.nodes), weight=0.5)],
        )
        self.assertEqual(values, (2,))
        self.assertEqual(score, 1.0)

    def test_fitness_components_shape_submitted_fitness_and_diagnostics(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        yane.set_fitness_components([
            FitnessComponent("nodes", lambda genome: len(genome.nodes), weight=0.5),
        ])

        yane.next_genome()
        yane.submit_fitness(1.0)

        best = yane.population.get_best()
        self.assertEqual(best.fitness_component_values, (2,))
        self.assertEqual(best.fitness, 2.0)
        # raw_fitness must hold the pure task score, not inflated by component bonus
        self.assertEqual(best.raw_fitness, 1.0)
        info = yane.population_memory_info()
        self.assertIn("fitness_component_weights", info)
        self.assertEqual(info["fitness_component_weights"]["weights"]["nodes"], 0.5)
        # diagnostics max_fitness shows task score (not component-inflated genome.fitness)
        self.assertEqual(info["max_fitness"], 1.0)

    def test_adaptive_component_weights_record_history_and_avoid_collapse(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        components = [
            FitnessComponent("constant", lambda genome: 1.0, weight=1.9),
            FitnessComponent("metric", lambda genome: getattr(genome, "test_metric", 0.0), weight=0.1),
        ]
        yane.set_fitness_components(components, mode="adaptive", adaptation_rate=1.0, collapse_floor=0.1)

        g1 = yane.next_genome()
        yane.submit_fitness(1.0)
        g2 = g1.copy()
        g2.test_metric = 10.0
        yane.population._evaluated.append(g2)
        yane.get_fitness_component_weights().scalarize(g2)
        yane.population._stagnation_count = yane.population.stagnation_threshold

        yane.get_fitness_component_weights().tick(yane.population)
        diagnostics = yane.population_memory_info()["fitness_component_weights"]

        self.assertEqual(diagnostics["last_reason"], "adaptive:stagnation")
        self.assertGreaterEqual(len(diagnostics["history"]), 2)
        total = sum(diagnostics["weights"].values())
        for weight in diagnostics["weights"].values():
            self.assertGreaterEqual(weight / total, 0.1 - 1e-9)

    def test_serialization_profile_smoke(self):
        result = profile_one(0, 1, repeats=1)
        self.assertGreater(result.pickle_bytes, 0)
        self.assertGreaterEqual(result.pickle_ms, 0.0)

    def test_matrix_forward_cache_batch(self):
        yane = NeuroEvolution()
        yane.configure(1, 1)
        g = yane.next_genome()
        cache = MatrixForwardCache()
        outputs = forward_compatible_batch([g], [[0.0], [1.0]], cache=cache)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0]), 2)


if __name__ == "__main__":
    unittest.main()
