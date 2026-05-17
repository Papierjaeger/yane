import unittest

from yane.util.activation import ActivationFunction, ActivationType


class TestActivationFunction(unittest.TestCase):

    def test_linear(self):
        self.assertEqual(ActivationFunction.activate(ActivationType.LINEAR, 1.0), 1.0)
        self.assertEqual(ActivationFunction.activate(ActivationType.LINEAR, -5.0), -5.0)

    def test_sigmoid(self):
        result = ActivationFunction.activate(ActivationType.SIGMOID, 0.0)
        self.assertAlmostEqual(result, 0.5)

    def test_sigmoid_large_positive(self):
        result = ActivationFunction.activate(ActivationType.SIGMOID, 1000.0)
        self.assertAlmostEqual(result, 1.0)

    def test_tanh(self):
        self.assertAlmostEqual(ActivationFunction.activate(ActivationType.TANH, 0.0), 0.0)

    def test_relu_negative(self):
        self.assertEqual(ActivationFunction.activate(ActivationType.RELU, -1.0), 0.0)

    def test_relu_positive(self):
        self.assertEqual(ActivationFunction.activate(ActivationType.RELU, 2.0), 2.0)

    def test_binary_above_threshold(self):
        self.assertEqual(ActivationFunction.activate(ActivationType.BINARY, 1.0), 1.0)

    def test_binary_below_threshold(self):
        self.assertEqual(ActivationFunction.activate(ActivationType.BINARY, 0.0), 0.0)

    def test_unknown_raises(self):
        with self.assertRaises((ValueError, Exception)):
            ActivationFunction.activate("nonexistent", 0.0)

    # ----- Clipping / extreme values -----

    def test_sigmoid_clips_large_positive(self):
        """Values beyond ±CLIP must not raise OverflowError and must stay in [0,1]."""
        from yane.util.activation import _sigmoid
        result = _sigmoid(1e9)
        self.assertAlmostEqual(result, 1.0, places=3)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_sigmoid_clips_large_negative(self):
        from yane.util.activation import _sigmoid
        result = _sigmoid(-1e9)
        self.assertAlmostEqual(result, 0.0, places=3)

    def test_swish_clips_extreme_values(self):
        from yane.util.activation import _swish
        pos = _swish(1e9)
        neg = _swish(-1e9)
        self.assertFalse(pos != pos, "swish(+inf) must not be NaN")
        self.assertFalse(neg != neg, "swish(-inf) must not be NaN")

    def test_softplus_at_boundary_v20(self):
        """softplus fast-path: for v>20 returns v directly; check continuity near 20."""
        from yane.util.activation import _softplus
        import math
        # At v=19.999 the full formula is used; at v=20.001 the fast path kicks in.
        # Both must be very close (softplus is smooth).
        slow = _softplus(19.999)
        fast = _softplus(20.001)
        self.assertAlmostEqual(slow, fast, delta=0.005,
            msg="softplus must be continuous near v=20 fast-path boundary")

    def test_elu_exact_negative_plateau(self):
        """elu(-CLIP) ≈ -1.0 (exp(-500)-1 ≈ -1)."""
        from yane.util.activation import _elu, _CLIP
        result = _elu(-_CLIP)
        self.assertAlmostEqual(result, -1.0, places=3)

    def test_elu_extreme_negative(self):
        """elu beyond -CLIP must not overflow."""
        from yane.util.activation import _elu
        result = _elu(-1e9)
        self.assertFalse(result != result, "elu(-1e9) must not be NaN")
        self.assertAlmostEqual(result, -1.0, places=3)

    def test_all_activations_produce_finite_output(self):
        """Every activation function must handle ±large values without NaN/Inf."""
        from yane.util.activation import ACTIVATION_FNS
        for act_type, fn in ACTIVATION_FNS.items():
            for v in (-1e6, -1.0, 0.0, 1.0, 1e6):
                result = fn(v)
                self.assertFalse(result != result,
                    f"{act_type.value}({v}) produced NaN")
                self.assertFalse(result == float('inf') or result == float('-inf'),
                    f"{act_type.value}({v}) produced Inf")


if __name__ == "__main__":
    unittest.main()
