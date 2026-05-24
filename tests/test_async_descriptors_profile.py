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
