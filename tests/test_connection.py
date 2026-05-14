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


if __name__ == "__main__":
    unittest.main()
