import math
import tempfile
import unittest
from pathlib import Path

from yane.util.normalization import (
    ClipNormalizer,
    MinMaxNormalizer,
    RunningStatsNormalizer,
    ScaleNormalizer,
    ZScoreNormalizer,
)


class TestScaleNormalizer(unittest.TestCase):

    def test_normalize_and_denormalize_roundtrip(self):
        n = ScaleNormalizer(input_scale=(9.0, 3.0), output_scale=(81.0,))
        self.assertEqual(n.normalize_input([9.0, 1.5]), [1.0, 0.5])
        self.assertEqual(n.denormalize_input([1.0, 0.5]), [9.0, 1.5])
        self.assertEqual(n.normalize_output([40.5]), [0.5])
        self.assertEqual(n.denormalize_output([0.5]), [40.5])

    def test_last_scale_reused_for_extra_channels(self):
        n = ScaleNormalizer(input_scale=(10.0,), output_scale=(2.0,))
        self.assertEqual(n.normalize_input([5.0, 10.0]), [0.5, 1.0])

    def test_normalize_samples(self):
        n = ScaleNormalizer(input_scale=(10.0,), output_scale=(5.0,))
        samples = [{"input": [5], "output": [2.5]}]
        self.assertEqual(
            n.normalize_samples(samples),
            [{"input": [0.5], "output": [0.5]}],
        )


class TestMinMaxNormalizer(unittest.TestCase):

    def test_normalize_maps_to_unit_interval(self):
        n = MinMaxNormalizer(input_range=((0.0, 10.0),))
        self.assertAlmostEqual(n.normalize_input([0.0])[0], 0.0)
        self.assertAlmostEqual(n.normalize_input([5.0])[0], 0.5)
        self.assertAlmostEqual(n.normalize_input([10.0])[0], 1.0)

    def test_denormalize_roundtrip(self):
        n = MinMaxNormalizer(input_range=((0.0, 100.0),), output_range=((-1.0, 1.0),))
        for v in [0.0, 25.0, 75.0, 100.0]:
            normed = n.normalize_input([v])
            self.assertAlmostEqual(n.denormalize_input(normed)[0], v, places=10)
        for v in [-1.0, 0.0, 1.0]:
            normed = n.normalize_output([v])
            self.assertAlmostEqual(n.denormalize_output(normed)[0], v, places=10)

    def test_last_range_broadcast(self):
        n = MinMaxNormalizer(input_range=((0.0, 10.0),))
        result = n.normalize_input([0.0, 5.0, 10.0])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 1.0)

    def test_zero_span_returns_zero(self):
        n = MinMaxNormalizer(input_range=((5.0, 5.0),))
        self.assertEqual(n.normalize_input([5.0]), [0.0])

    def test_no_range_passthrough(self):
        n = MinMaxNormalizer()
        self.assertEqual(n.normalize_input([3.0, 7.0]), [3.0, 7.0])
        self.assertEqual(n.denormalize_input([0.5]), [0.5])

    def test_normalize_samples(self):
        n = MinMaxNormalizer(input_range=((0.0, 10.0),), output_range=((0.0, 4.0),))
        samples = [{"input": [5.0], "output": [2.0]}]
        result = n.normalize_samples(samples)
        self.assertAlmostEqual(result[0]["input"][0], 0.5)
        self.assertAlmostEqual(result[0]["output"][0], 0.5)


class TestZScoreNormalizer(unittest.TestCase):

    def test_normalize_standardizes(self):
        n = ZScoreNormalizer(input_stats=((10.0, 2.0),))
        result = n.normalize_input([10.0])
        self.assertAlmostEqual(result[0], 0.0)
        result2 = n.normalize_input([12.0])
        self.assertAlmostEqual(result2[0], 1.0)

    def test_denormalize_roundtrip(self):
        n = ZScoreNormalizer(
            input_stats=((0.5, 0.25),),
            output_stats=((10.0, 5.0),),
        )
        for v in [0.0, 0.5, 1.0]:
            normed = n.normalize_input([v])
            self.assertAlmostEqual(n.denormalize_input(normed)[0], v, places=10)
        for v in [5.0, 10.0, 15.0]:
            normed = n.normalize_output([v])
            self.assertAlmostEqual(n.denormalize_output(normed)[0], v, places=10)

    def test_no_stats_passthrough(self):
        n = ZScoreNormalizer()
        self.assertEqual(n.normalize_input([1.0, 2.0]), [1.0, 2.0])
        self.assertEqual(n.denormalize_output([3.0]), [3.0])

    def test_eps_prevents_division_by_zero(self):
        n = ZScoreNormalizer(input_stats=((0.0, 0.0),), eps=1e-8)
        result = n.normalize_input([1.0])
        self.assertTrue(math.isfinite(result[0]))

    def test_normalize_samples(self):
        n = ZScoreNormalizer(input_stats=((0.0, 1.0),), output_stats=((0.0, 2.0),))
        samples = [{"input": [1.0], "output": [2.0]}]
        result = n.normalize_samples(samples)
        self.assertAlmostEqual(result[0]["input"][0], 1.0)
        self.assertAlmostEqual(result[0]["output"][0], 1.0)


class TestClipNormalizer(unittest.TestCase):

    def test_clip_and_rescale(self):
        n = ClipNormalizer(input_clip=((-5.0, 5.0),), rescale=True)
        self.assertAlmostEqual(n.normalize_input([-5.0])[0], 0.0)
        self.assertAlmostEqual(n.normalize_input([0.0])[0], 0.5)
        self.assertAlmostEqual(n.normalize_input([5.0])[0], 1.0)
        self.assertAlmostEqual(n.normalize_input([100.0])[0], 1.0)

    def test_clip_no_rescale(self):
        n = ClipNormalizer(input_clip=((0.0, 10.0),), rescale=False)
        self.assertAlmostEqual(n.normalize_input([-5.0])[0], 0.0)
        self.assertAlmostEqual(n.normalize_input([5.0])[0], 5.0)
        self.assertAlmostEqual(n.normalize_input([15.0])[0], 10.0)

    def test_denormalize_roundtrip_rescale(self):
        n = ClipNormalizer(input_clip=((0.0, 100.0),), rescale=True)
        for v in [0.0, 25.0, 50.0, 100.0]:
            normed = n.normalize_input([v])
            self.assertAlmostEqual(n.denormalize_input(normed)[0], v, places=10)

    def test_denormalize_clamps_output_values(self):
        n = ClipNormalizer(input_clip=((0.0, 1.0),), rescale=True)
        self.assertAlmostEqual(n.denormalize_input([2.0])[0], 1.0)
        self.assertAlmostEqual(n.denormalize_input([-1.0])[0], 0.0)

    def test_no_clip_passthrough(self):
        n = ClipNormalizer()
        self.assertEqual(n.normalize_input([99.0]), [99.0])
        self.assertEqual(n.denormalize_output([-3.0]), [-3.0])

    def test_normalize_samples(self):
        n = ClipNormalizer(input_clip=((0.0, 10.0),), output_clip=((0.0, 2.0),))
        samples = [{"input": [5.0], "output": [1.0]}]
        result = n.normalize_samples(samples)
        self.assertAlmostEqual(result[0]["input"][0], 0.5)
        self.assertAlmostEqual(result[0]["output"][0], 0.5)


class TestRunningStatsNormalizer(unittest.TestCase):

    def test_single_update_uses_eps_as_std(self):
        n = RunningStatsNormalizer(n_inputs=1, eps=1e-8)
        n.update_inputs([5.0])
        # Only one sample: std is undefined → eps used
        result = n.normalize_input([5.0])
        self.assertAlmostEqual(result[0], 0.0, places=6)

    def test_converges_to_zscore_after_many_updates(self):
        import random as _random
        _random.seed(0)
        n = RunningStatsNormalizer(n_inputs=1)
        values = [_random.gauss(10.0, 2.0) for _ in range(1000)]
        for v in values:
            n.update_inputs([v])
        self.assertAlmostEqual(n.input_means[0], 10.0, delta=0.3)
        self.assertAlmostEqual(n.input_stds[0], 2.0, delta=0.3)

    def test_normalize_denormalize_roundtrip(self):
        n = RunningStatsNormalizer(n_inputs=2, n_outputs=1)
        for _ in range(200):
            n.update([1.0, 2.0])
            n.update_outputs([3.0])
        v_in = [1.0, 2.0]
        normed = n.normalize_input(v_in)
        recovered = n.denormalize_input(normed)
        for orig, rec in zip(v_in, recovered):
            self.assertAlmostEqual(orig, rec, places=8)

    def test_no_outputs_passthrough(self):
        n = RunningStatsNormalizer(n_inputs=1, n_outputs=0)
        self.assertEqual(n.normalize_output([42.0]), [42.0])
        self.assertEqual(n.denormalize_output([42.0]), [42.0])

    def test_sample_count(self):
        n = RunningStatsNormalizer(n_inputs=2)
        for i in range(7):
            n.update_inputs([float(i), float(i)])
        self.assertEqual(n.sample_count(), 7)

    def test_normalize_samples(self):
        n = RunningStatsNormalizer(n_inputs=1, n_outputs=1)
        for _ in range(50):
            n.update([0.0], [1.0])
        samples = [{"input": [0.0], "output": [1.0]}]
        result = n.normalize_samples(samples)
        self.assertEqual(len(result), 1)
        self.assertIn("input", result[0])
        self.assertIn("output", result[0])

    def test_properties(self):
        n = RunningStatsNormalizer(n_inputs=2, n_outputs=2)
        self.assertEqual(len(n.input_means), 2)
        self.assertEqual(len(n.input_stds), 2)
        self.assertEqual(len(n.output_means), 2)
        self.assertEqual(len(n.output_stds), 2)


class TestNormalizerCheckpointPersistence(unittest.TestCase):

    def _make_yane(self):
        from yane import NeuroEvolution

        def _xor(genome) -> float:
            total = 0.0
            for a, b, expected in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
                out = genome.forward([float(a), float(b)])[0]
                total -= (out - expected) ** 2
            return total

        yane = NeuroEvolution(seed=0)
        yane.configure(n_inputs=2, n_outputs=1)
        yane.set_max_iterations(5)
        yane.train(_xor)
        return yane

    def test_scale_normalizer_round_trips_checkpoint(self):
        from yane import NeuroEvolution

        yane = self._make_yane()
        norm = ScaleNormalizer(input_scale=(2.0,), output_scale=(4.0,))
        yane.set_normalizer(norm)
        self.assertIs(yane.get_normalizer(), norm)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            yane.save_checkpoint(path)
            yane2 = NeuroEvolution()
            yane2.load_checkpoint(path)
            restored = yane2.get_normalizer()
            self.assertIsInstance(restored, ScaleNormalizer)
            self.assertEqual(restored.input_scale, (2.0,))
            self.assertEqual(restored.output_scale, (4.0,))
        finally:
            path.unlink(missing_ok=True)

    def test_minmax_normalizer_round_trips_checkpoint(self):
        from yane import NeuroEvolution

        yane = self._make_yane()
        norm = MinMaxNormalizer(input_range=((0.0, 1.0),), output_range=((0.0, 2.0),))
        yane.set_normalizer(norm)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            yane.save_checkpoint(path)
            yane2 = NeuroEvolution()
            yane2.load_checkpoint(path)
            restored = yane2.get_normalizer()
            self.assertIsInstance(restored, MinMaxNormalizer)
            self.assertEqual(restored.input_range, ((0.0, 1.0),))
        finally:
            path.unlink(missing_ok=True)

    def test_running_stats_normalizer_round_trips_checkpoint(self):
        import pickle
        from yane import NeuroEvolution

        yane = self._make_yane()
        norm = RunningStatsNormalizer(n_inputs=2, n_outputs=1)
        for i in range(10):
            norm.update([float(i), float(i * 2)], [float(i)])
        yane.set_normalizer(norm)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            yane.save_checkpoint(path)
            yane2 = NeuroEvolution()
            yane2.load_checkpoint(path)
            restored = yane2.get_normalizer()
            self.assertIsInstance(restored, RunningStatsNormalizer)
            self.assertEqual(restored.sample_count(), 10)
            self.assertAlmostEqual(restored.input_means[0], norm.input_means[0], places=10)
        finally:
            path.unlink(missing_ok=True)

    def test_none_normalizer_after_checkpoint_without_normalizer(self):
        from yane import NeuroEvolution

        yane = self._make_yane()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            yane.save_checkpoint(path)
            yane2 = NeuroEvolution()
            yane2.load_checkpoint(path)
            self.assertIsNone(yane2.get_normalizer())
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
