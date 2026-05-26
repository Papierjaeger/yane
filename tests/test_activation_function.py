import unittest
import pytest

from yane.util.activation import ActivationFunction, ActivationType


@pytest.mark.ci
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


# ---------------------------------------------------------------------------
# Custom / registrierbare Aktivierungsfunktionen
# ---------------------------------------------------------------------------

class TestCustomActivationRegistry(unittest.TestCase):

    def test_register_and_resolve_custom(self):
        from yane.util.activation import (
            register_activation, resolve_activation_fn, list_activations,
            CUSTOM_ACTIVATION_FNS,
        )
        # Use a named function so pickle works.
        def _square_plus_one(v: float) -> float:
            return v * v + 1.0
        register_activation("sq1", _square_plus_one)
        try:
            fn = resolve_activation_fn("sq1")
            self.assertAlmostEqual(fn(3.0), 10.0)
            self.assertIn("sq1", list_activations())
        finally:
            CUSTOM_ACTIVATION_FNS.pop("sq1", None)

    def test_register_duplicate_raises(self):
        from yane.util.activation import register_activation, CUSTOM_ACTIVATION_FNS
        def _dummy(v): return v
        register_activation("dup_test", _dummy)
        try:
            with self.assertRaises(ValueError):
                register_activation("dup_test", _dummy)
        finally:
            CUSTOM_ACTIVATION_FNS.pop("dup_test", None)

    def test_register_shadows_builtin_raises(self):
        from yane.util.activation import register_activation
        def _dummy(v): return v
        with self.assertRaises(ValueError):
            register_activation("sigmoid", _dummy)

    def test_register_empty_name_raises(self):
        from yane.util.activation import register_activation
        def _dummy(v): return v
        with self.assertRaises(ValueError):
            register_activation("", _dummy)

    def test_node_activation_setter_accepts_string(self):
        from yane.core.node import Node, NodeType
        n = Node(NodeType.HIDDEN, 0)
        n.activation = "gelu"
        self.assertEqual(n.activation, "gelu")
        # Forward pass should produce correct GELU output
        result = n._activate_fn(0.0)
        self.assertAlmostEqual(result, 0.0)
        result = n._activate_fn(1.0)
        self.assertGreater(result, 0.8)  # GELU(1) ≈ 0.841

    def test_node_activation_setter_still_accepts_enum(self):
        from yane.core.node import Node, NodeType
        n = Node(NodeType.HIDDEN, 0)
        n.activation = ActivationType.RELU
        self.assertIs(n.activation, ActivationType.RELU)
        self.assertAlmostEqual(n._activate_fn(-1.0), 0.0)
        self.assertAlmostEqual(n._activate_fn(2.0), 2.0)

    def test_custom_activation_via_neuroevolution_api(self):
        from yane import NeuroEvolution
        def _my_act(v: float) -> float:
            return v * 2.0
        NeuroEvolution.register_activation("my_double", _my_act)
        from yane.util.activation import resolve_activation_fn, CUSTOM_ACTIVATION_FNS
        try:
            fn = resolve_activation_fn("my_double")
            self.assertAlmostEqual(fn(3.0), 6.0)
        finally:
            CUSTOM_ACTIVATION_FNS.pop("my_double", None)

    def test_custom_activation_forward_via_genome(self):
        """Genome forward pass works with custom activation on a node."""
        from yane.util.activation import register_activation, CUSTOM_ACTIVATION_FNS
        from yane import NeuroEvolution
        def _double_it(v: float) -> float:
            return v * 2.0
        register_activation("double_it", _double_it)
        try:
            yane = NeuroEvolution()
            yane.configure(1, 1)
            g = yane.next_genome()
            # Set output node to custom activation
            g.output_nodes[0].activation = "double_it"
            # Add a connection from input to output
            from yane.core.connection import Connection
            innov = yane._tracker.get_connection(
                g.input_nodes[0].innovation, g.output_nodes[0].innovation
            )
            conn = Connection(g.output_nodes[0], innovation=innov)
            conn.weight = 3.0
            g.input_nodes[0].connections.append(conn)
            g._invalidate_topology()
            result = g.forward([2.0])
            # input=2.0 * weight=3.0 = 6.0, then double_it → 12.0
            self.assertAlmostEqual(result[0], 12.0, places=10)
        finally:
            CUSTOM_ACTIVATION_FNS.pop("double_it", None)

    def test_custom_activation_forward_batch_via_genome(self):
        """Genome batch forward falls back to scalar custom activations."""
        from yane.util.activation import register_activation, CUSTOM_ACTIVATION_FNS
        from yane import NeuroEvolution
        from yane.core.connection import Connection

        def _triple_it(v: float) -> float:
            return v * 3.0

        register_activation("triple_it", _triple_it)
        try:
            yane = NeuroEvolution()
            yane.configure(1, 1)
            g = yane.next_genome()
            g.output_nodes[0].activation = "triple_it"
            innov = yane._tracker.get_connection(
                g.input_nodes[0].innovation, g.output_nodes[0].innovation
            )
            conn = Connection(g.output_nodes[0], innovation=innov)
            conn.weight = 2.0
            g.input_nodes[0].connections.append(conn)
            g._invalidate_topology()
            self.assertEqual(g.forward_batch([[1.0], [2.0]]), [[6.0], [12.0]])
        finally:
            CUSTOM_ACTIVATION_FNS.pop("triple_it", None)

    def test_custom_activation_mutation_does_not_change(self):
        """Custom activation should not be mutated to a random builtin."""
        from yane.util.activation import register_activation, CUSTOM_ACTIVATION_FNS
        from yane.core.node import Node, NodeType
        def _custom(v): return v
        register_activation("frozen_act", _custom)
        try:
            n = Node(NodeType.HIDDEN, 0)
            n.activation = "frozen_act"
            for _ in range(100):
                n.mutate()
                self.assertEqual(n.activation, "frozen_act",
                    "Custom activation changed after mutation")
        finally:
            CUSTOM_ACTIVATION_FNS.pop("frozen_act", None)

    def test_gelu_values(self):
        from yane.util.activation import _gelu
        self.assertAlmostEqual(_gelu(0.0), 0.0)
        self.assertAlmostEqual(_gelu(1.0), 0.841192, places=4)
        self.assertAlmostEqual(_gelu(-1.0), -0.158808, places=4)
        # Extreme values
        self.assertAlmostEqual(_gelu(10.0), 10.0, places=3)
        self.assertAlmostEqual(_gelu(-10.0), 0.0)

    def test_mish_values(self):
        from yane.util.activation import _mish
        self.assertAlmostEqual(_mish(0.0), 0.0)
        self.assertAlmostEqual(_mish(1.0), 0.865098, places=4)
        self.assertAlmostEqual(_mish(-1.0), -0.303401, places=4)
        # Large positive: mish(x) → x
        self.assertAlmostEqual(_mish(20.0), 20.0, places=3)

    def test_silu_values(self):
        from yane.util.activation import _silu
        self.assertAlmostEqual(_silu(0.0), 0.0)
        self.assertAlmostEqual(_silu(1.0), 0.731058, places=4)
        self.assertAlmostEqual(_silu(-1.0), -0.268941, places=4)

    def test_gelu_mish_silu_in_list_activations(self):
        from yane.util.activation import list_activations
        acts = list_activations()
        self.assertIn("gelu", acts)
        self.assertIn("mish", acts)
        self.assertIn("silu", acts)

    def test_custom_activation_pickle_roundtrip(self):
        """Genome with custom activation survives pickle round-trip."""
        import pickle
        from yane.util.activation import (
            register_activation, CUSTOM_ACTIVATION_FNS,
        )
        from yane import NeuroEvolution

        # Use GELU (already registered at module level, so pickle-safe)
        try:
            yane = NeuroEvolution()
            yane.configure(1, 1)
            g = yane.next_genome()
            g.output_nodes[0].activation = "gelu"
            # Pickle round-trip
            data = pickle.dumps(g)
            g2 = pickle.loads(data)
            self.assertEqual(g2.output_nodes[0].activation, "gelu")
            # Forward should work after unpickling
            out = g2.output_nodes[0]._activate_fn(1.0)
            self.assertAlmostEqual(out, 0.841192, places=4)
        finally:
            pass  # gelu is always-registered, no cleanup needed


if __name__ == "__main__":
    unittest.main()
