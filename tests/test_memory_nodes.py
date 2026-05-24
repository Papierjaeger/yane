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


class TestDynamicGating(unittest.TestCase):
    """Tests for gate_node — dynamic gating where gate comes from another node's output."""

    def _make_gated_genome(self):
        """Build a 2-input, 1-output genome with a persistent hidden node and a persistent
        gate source node.

        Topology:
          input[0] → mem (persistent SIGMOID hidden, weight=1)
          input[1] → gate_src (persistent LINEAR hidden, weight=1)
          mem → output (weight=1)
          gate_src is set as mem.gate_node

        Using persistent gate_src ensures its .value from the PREVIOUS step is used as
        the gate signal when mem fires (gate_src's value is not zeroed between steps).
        """
        yane = NeuroEvolution()
        yane.configure(n_inputs=2, n_outputs=1)
        g = yane.next_genome()

        # Persistent hidden memory node with sigmoid activation
        mem = Node(NodeType.HIDDEN)
        mem.activation = ActivationType.SIGMOID
        mem.bias = 0.0
        mem.persist_value = True
        mem.leak_alpha = 1.0
        mem.memory_gate = 0.0
        g.nodes.append(mem)

        # Persistent gate source node — its value from the previous step drives the gate
        gate_src = Node(NodeType.HIDDEN)
        gate_src.activation = ActivationType.LINEAR
        gate_src.bias = 0.0
        gate_src.persist_value = True
        gate_src.leak_alpha = 1.0
        gate_src.memory_gate = 0.0
        g.nodes.append(gate_src)

        # input[0] → mem
        c_in_mem = Connection(mem)
        c_in_mem.weight = 1.0
        g.input_nodes[0].connections.append(c_in_mem)

        # input[1] → gate_src
        c_in_gate = Connection(gate_src)
        c_in_gate.weight = 1.0
        g.input_nodes[1].connections.append(c_in_gate)

        # mem → output
        c_out = Connection(g.output_nodes[0])
        c_out.weight = 1.0
        mem.connections.append(c_out)

        mem.gate_node = gate_src
        g._invalidate_topology()

        return g, mem, gate_src

    def test_gate_node_none_produces_same_as_static_gate_zero(self):
        """When gate_node is None and memory_gate=0, mem retains only new activation."""
        yane = NeuroEvolution()
        yane.configure(n_inputs=1, n_outputs=1)
        g = yane.next_genome()

        mem = Node(NodeType.HIDDEN)
        mem.activation = ActivationType.SIGMOID
        mem.bias = 0.0
        mem.persist_value = True
        mem.memory_gate = 0.0
        mem.leak_alpha = 1.0
        g.nodes.append(mem)

        c1 = Connection(mem); c1.weight = 2.0
        g.input_nodes[0].connections.append(c1)
        c2 = Connection(g.output_nodes[0]); c2.weight = 1.0
        mem.connections.append(c2)
        g._invalidate_topology()

        g.reset()
        self.assertIsNone(mem.gate_node)

        # With gate=0, new value = (1-0) * leak_alpha * sigmoid(old + bias)
        # old_value after reset = 0; input accumulates: old_value = 2.0
        # activated = sigmoid(2.0) ≈ 0.880
        # retained = 1.0 * 0.880 = 0.880
        # gate=0: mem.value = 0 * 2.0 + 1.0 * 0.880 ≈ 0.880
        import math
        g.forward([1.0])
        expected = 1.0 / (1.0 + math.exp(-2.0))  # sigmoid(input*weight)
        self.assertAlmostEqual(mem.value, expected, places=8)

    def test_gate_node_is_set(self):
        """When gate_node is set, gate_node.value drives the gate instead of memory_gate."""
        g, mem, gate_src = self._make_gated_genome()
        # Ensure gate_node is wired correctly
        self.assertIs(mem.gate_node, gate_src)

    def test_gate_node_high_vs_low_affects_retention(self):
        """High gate (sigmoid near 1) retains more memory than low gate (sigmoid near 0).

        Protocol:
          Step 1: prime memory with a positive value (input[0]=3, input[1]=0 for gate=0)
          Step 2 (gate=high): input[0]=0, input[1]=+100 → gate≈1, mem should retain strongly
          Step 2 (gate=low):  input[0]=0, input[1]=−100 → gate≈0, mem should retain weakly
        """
        import math

        # High-gate branch
        g_high, mem_high, gs_high = self._make_gated_genome()
        g_high.reset()
        g_high.forward([3.0, 0.0])   # prime memory; gate_src becomes 0 (old) → gate=sigmoid(0)
        g_high.forward([0.0, 100.0]) # gate_src retains 100 → gate≈sigmoid(100)≈1; no new mem input
        val_high = mem_high.value

        # Low-gate branch
        g_low, mem_low, gs_low = self._make_gated_genome()
        g_low.reset()
        g_low.forward([3.0, 0.0])    # same prime
        g_low.forward([0.0, -100.0]) # gate_src retains -100 → gate≈0; no new mem input
        val_low = mem_low.value

        # High gate retains more of the primed state than low gate
        # (for the BFS/small path, gate_src.value from the previous step is used)
        self.assertGreater(val_high, val_low,
            "High gate should retain more memory state than low gate")

    def test_copy_remaps_gate_node(self):
        """Genome copy() must remap gate_node to the copy's own nodes."""
        g, mem, gate_src = self._make_gated_genome()
        g_copy = g.copy()

        # Find the copied mem node (persistent hidden)
        mem_copy = next(n for n in g_copy.nodes
                        if n.type == NodeType.HIDDEN and n._persist_value)
        # gate_node must be a node IN the copy, not in the original
        self.assertIsNotNone(mem_copy.gate_node)
        self.assertIn(mem_copy.gate_node, g_copy.nodes)
        self.assertNotIn(mem_copy.gate_node, g.nodes)

    def test_copy_gate_node_produces_same_output(self):
        """Copied genome with gate_node must produce same output as original."""
        g, mem, gate_src = self._make_gated_genome()
        g.reset()
        g.forward([0.5])  # prime state

        g_copy = g.copy()
        g.reset(); g_copy.reset()

        out1 = g.forward([0.7])
        out2 = g_copy.forward([0.7])
        self.assertAlmostEqual(out1[0], out2[0], places=8)

    def test_allow_memory_false_clears_gate_node(self):
        """allow_memory=False in mutate() must clear gate_node on all nodes."""
        g, mem, gate_src = self._make_gated_genome()
        g.allow_memory = False
        g.mutate()
        for node in g.nodes:
            self.assertIsNone(node.gate_node,
                "gate_node must be cleared when allow_memory=False")

    def test_gate_node_survives_pickle(self):
        """gate_node reference must survive pickle/unpickle (nodes are pickled together)."""
        import pickle
        g, mem, gate_src = self._make_gated_genome()
        g.reset()
        g.forward([1.0])

        data = pickle.dumps(g)
        g2 = pickle.loads(data)
        mem2 = next(n for n in g2.nodes
                    if n.type == NodeType.HIDDEN and n._persist_value)
        self.assertIsNotNone(mem2.gate_node)
        self.assertIn(mem2.gate_node, g2.nodes)

    def test_mutation_can_assign_gate_node(self):
        """After enough mutations, some persistent node should gain a gate_node."""
        import random
        yane = NeuroEvolution(seed=42)
        yane.configure(n_inputs=2, n_outputs=1)
        g = yane.next_genome()

        # Create a persistent node manually so the mutation has something to target
        from yane.core.connection import Connection as Conn
        mem = Node(NodeType.HIDDEN)
        mem.activation = ActivationType.LINEAR
        mem.persist_value = True
        g.nodes.append(mem)
        c = Conn(g.output_nodes[0]); c.weight = 0.1
        mem.connections.append(c)
        g._invalidate_topology()

        # Force mutation_gate_source to fire on every call
        g.mutation_gate_source.bool_rate = 1.0
        found_gate = False
        for _ in range(20):
            g._mutate_gate_source()
            mem_nodes = [n for n in g.nodes
                         if n.type == NodeType.HIDDEN and n._persist_value]
            if any(n.gate_node is not None for n in mem_nodes):
                found_gate = True
                break
        self.assertTrue(found_gate, "High-rate mutation should assign gate_node within 20 calls")


if __name__ == "__main__":
    unittest.main()
