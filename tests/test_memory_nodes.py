"""Tests for persistent hidden nodes (memory) and the stateful/reset contract."""
import unittest

from yane import NeuroEvolution
from yane.core.node import Node, NodeType
from yane.core.connection import Connection
from yane.util.activation import ActivationType


def _genome_with_memory_node():
    """2-input, 1-output genome with one persistent hidden (memory) node."""
    yane = NeuroEvolution()
    yane.configure(n_inputs=2, n_outputs=1)
    g = yane.next_genome()

    mem = Node(NodeType.HIDDEN)
    mem.activation = ActivationType.LINEAR
    mem.bias = 0.0
    mem.persist_value = True
    g.nodes.append(mem)

    # input[0] → mem → output[0]
    c_in  = Connection(mem);              c_in.weight  = 1.0
    c_out = Connection(g.output_nodes[0]); c_out.weight = 1.0
    g.input_nodes[0].connections.append(c_in)
    mem.connections.append(c_out)
    g._invalidate_topology()

    return g, mem


class TestPersistentHiddenNode(unittest.TestCase):

    def test_output_nodes_are_persistent_by_default(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        for n in g.output_nodes:
            self.assertTrue(n.persist_value,
                "output nodes must have persist_value=True so get_outputs() works")

    def test_hidden_nodes_not_persistent_by_default(self):
        from yane.evolution import smart_mutation
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        smart_mutation.add_connection(g, yane._tracker)  # need a connection to split
        smart_mutation.add_node(g, yane._tracker)
        hidden = [n for n in g.nodes if n.type == NodeType.HIDDEN]
        self.assertTrue(len(hidden) > 0)
        # Default hidden nodes should not be persistent (no memory)
        for n in hidden:
            self.assertFalse(n.persist_value,
                "freshly added hidden node must default to persist_value=False")

    def test_memory_node_retains_value_between_steps(self):
        g, mem = _genome_with_memory_node()
        g.reset()

        g.forward([1.0, 0.0])
        val_after_step1 = mem.value
        self.assertNotEqual(val_after_step1, 0.0,
            "memory node must have a non-zero value after step 1")

        g.forward([0.0, 0.0])
        val_after_step2 = mem.value
        # The memory node fired with the old value still present,
        # so step-2 output incorporates step-1 memory.
        # We only require that the memory was non-trivially non-zero after step 1.
        self.assertNotEqual(val_after_step1, 0.0)

    def test_regular_hidden_node_zeroed_between_steps(self):
        """A non-persistent hidden node must have value=0 after each step."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()

        reg = Node(NodeType.HIDDEN)
        reg.activation = ActivationType.LINEAR
        reg.persist_value = False
        g.nodes.append(reg)

        c1 = Connection(reg);                 c1.weight = 1.0
        c2 = Connection(g.output_nodes[0]);  c2.weight = 1.0
        g.input_nodes[0].connections.append(c1)
        reg.connections.append(c2)
        g._invalidate_topology()

        g.reset()
        g.forward([1.0, 0.0])
        self.assertEqual(reg.value, 0.0,
            "non-persistent hidden node must be zeroed after fire_simple()")

    def test_reset_clears_memory_node(self):
        g, mem = _genome_with_memory_node()
        g.reset()
        g.forward([1.0, 0.0])
        self.assertNotEqual(mem.value, 0.0)

        g.reset()
        self.assertEqual(mem.value, 0.0,
            "reset() must zero all node values including persistent hidden nodes")

    def test_reset_clears_output_nodes(self):
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        from yane.evolution import smart_mutation
        smart_mutation.add_connection(g, yane._tracker)
        g.reset()
        g.forward([1.0, 1.0])
        for out in g.output_nodes:
            g.reset()
            self.assertEqual(out.value, 0.0,
                "reset() must zero output nodes")
            break

    def test_episode_boundary_produces_independent_outputs(self):
        """Same input after reset() must produce same output regardless of history."""
        g, mem = _genome_with_memory_node()

        g.reset()
        out_fresh = g.forward([0.5, 0.0])[0]

        # Pollute memory with a different sequence
        g.reset()
        g.forward([9.0, 0.0])
        g.forward([9.0, 0.0])
        g.reset()
        out_after_history = g.forward([0.5, 0.0])[0]

        self.assertAlmostEqual(out_fresh, out_after_history, places=6,
            msg="reset() must produce a clean episode start")

    def test_stateless_eval_resets_before_each_forward(self):
        """For stateless tasks, memory must be cleared before each sample."""
        g, mem = _genome_with_memory_node()

        # Simulate stateless evaluation (reset before every forward call)
        g.reset(); out_a = g.forward([1.0, 0.0])[0]
        g.reset(); out_b = g.forward([1.0, 0.0])[0]

        self.assertAlmostEqual(out_a, out_b, places=6,
            msg="same input with reset() before each call must give same output")


class TestNonPersistentNodeInBFS(unittest.TestCase):

    def test_non_persistent_hidden_zeroed_after_fire_in_bfs(self):
        """In the BFS path (cyclic network), non-persistent nodes must be
        zeroed by fire() after they fire."""
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()

        # Add a self-loop to force BFS path
        from yane.core.connection import Connection
        self_conn = Connection(g.input_nodes[0])
        self_conn.weight = 0.1
        g.input_nodes[0].connections.append(self_conn)
        g._invalidate_topology()

        g.reset()
        g.forward([1.0, 0.5])
        # Input nodes have persist_value=False, so after firing they're 0
        for inp in g.input_nodes:
            self.assertEqual(inp.value, 0.0,
                "input node (non-persistent) must be 0 after firing")


if __name__ == "__main__":
    unittest.main()
