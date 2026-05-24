import unittest

from yane.util.normalization import ScaleNormalizer


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


if __name__ == "__main__":
    unittest.main()
