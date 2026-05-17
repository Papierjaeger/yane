import unittest
from yane.core.node import Node, NodeType
from yane.core.connection import Connection


def _node():
    return Node(NodeType.HIDDEN)


class TestConnectionBasics(unittest.TestCase):

    def test_initial_weight_in_range(self):
        for _ in range(50):
            n = _node()
            c = Connection(n)
            self.assertGreaterEqual(c.weight, -1.0)
            self.assertLessEqual(c.weight, 1.0)

    def test_target_is_correct_node(self):
        n = _node()
        c = Connection(n)
        self.assertIs(c.target, n)

    def test_target_setter(self):
        n1, n2 = _node(), _node()
        c = Connection(n1)
        c.target = n2
        self.assertIs(c.target, n2)

    def test_mutate_changes_weight(self):
        n = _node()
        c = Connection(n)
        c.weight = 0.0
        # With many mutations, weight should eventually move
        changed = False
        for _ in range(200):
            c.mutate()
            if abs(c.weight) > 1e-6:
                changed = True
                break
        self.assertTrue(changed, "mutate() never changed the weight")

    def test_copy_has_same_weight(self):
        n1, n2 = _node(), _node()
        c = Connection(n1)
        c.weight = 0.42
        node_map = {n1: n2}
        copy = c.copy(node_map)
        self.assertAlmostEqual(copy.weight, 0.42)

    def test_copy_points_to_mapped_node(self):
        n1, n2 = _node(), _node()
        c = Connection(n1)
        node_map = {n1: n2}
        copy = c.copy(node_map)
        self.assertIs(copy.target, n2)

    def test_copy_has_independent_mutation(self):
        n1, n2 = _node(), _node()
        c = Connection(n1)
        node_map = {n1: n2}
        copy = c.copy(node_map)
        original_rate = c.mutation.shift_rate
        copy.mutation.shift_rate = 0.999
        self.assertEqual(c.mutation.shift_rate, original_rate,
            "copy() must produce an independent Mutation object")

    def test_connection_freed_when_node_cleared(self):
        import gc
        import weakref
        n1, n2 = _node(), _node()
        c = Connection(n2)
        n1.connections.append(c)
        ref = weakref.ref(c)

        n1.connections.clear()  # releases c
        del c
        gc.collect()
        self.assertIsNone(ref(), "Connection must be freed when removed from node.connections")


class TestConnectionNumerics(unittest.TestCase):

    def test_weight_stays_finite_after_many_mutations(self):
        """Weights must remain finite even after thousands of mutations with large sigma."""
        n = _node()
        c = Connection(n)
        for _ in range(2000):
            c.weight = c.mutation.mutate_value(c.weight, sigma=10.0)
            c.mutation.mutate_rates()
        self.assertFalse(c.weight != c.weight, "weight became NaN after many mutations")
        self.assertFalse(abs(c.weight) == float('inf'), "weight became Inf after many mutations")

    def test_mutation_rate_stays_above_minimum(self):
        """mutate_rates must never push any rate below MIN_RATE."""
        from yane.evolution.mutation import Mutation
        m = Mutation()
        for _ in range(5000):
            m.mutate_rates()
        self.assertGreaterEqual(m.shift_rate,        Mutation.MIN_RATE)
        self.assertGreaterEqual(m.custom_rate,       Mutation.MIN_RATE)
        self.assertGreaterEqual(m.bool_rate,         Mutation.MIN_RATE)
        self.assertGreaterEqual(m.int_rate,          Mutation.MIN_RATE)
        self.assertGreaterEqual(m.rate_mutation_rate, Mutation.MIN_RATE)
        self.assertGreater(m.value_delta, 0.0)

    def test_mutation_rate_stays_below_maximum(self):
        """mutate_rates must never push any rate above 0.999."""
        from yane.evolution.mutation import Mutation
        m = Mutation()
        m.shift_rate = m.custom_rate = m.bool_rate = m.int_rate = 0.99
        for _ in range(1000):
            m.mutate_rates()
        self.assertLessEqual(m.shift_rate,  0.999)
        self.assertLessEqual(m.custom_rate, 0.999)


if __name__ == "__main__":
    unittest.main()
