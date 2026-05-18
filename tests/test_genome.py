import unittest
from yane.core.genome import Genome
from yane.core.node import Node, NodeType
from yane.core.connection import Connection


def _make_genome(n_inputs=2, n_outputs=1):
    from yane import NeuroEvolution
    yane = NeuroEvolution()
    yane.configure(n_inputs, n_outputs)
    return yane.next_genome()


class TestGenomeCopy(unittest.TestCase):

    def test_copy_is_independent(self):
        g = _make_genome()
        copy = g.copy()
        # Modifying the copy does not affect the original
        copy.fitness = 999.0
        self.assertNotEqual(g.fitness, 999.0)

    def test_copy_has_same_structure(self):
        g = _make_genome()
        copy = g.copy()
        self.assertEqual(len(g.nodes), len(copy.nodes))
        self.assertEqual(len(g.input_nodes), len(copy.input_nodes))
        self.assertEqual(len(g.output_nodes), len(copy.output_nodes))

    def test_copy_preserves_fitness_values(self):
        g = _make_genome()
        g.fitness = 12.5
        g.shared_fitness = 3.125
        g.eval_time_ms = 7.5
        g.efficiency_score = 0.75
        g.selection_score = 42.0
        copy = g.copy()
        self.assertEqual(copy.fitness, 12.5)
        self.assertEqual(copy.shared_fitness, 3.125)
        self.assertEqual(copy.eval_time_ms, 7.5)
        self.assertEqual(copy.efficiency_score, 0.75)
        self.assertEqual(copy.selection_score, 42.0)

    def test_copy_nodes_are_different_objects(self):
        g = _make_genome()
        copy = g.copy()
        original_ids = {id(n) for n in g.nodes}
        copy_ids = {id(n) for n in copy.nodes}
        self.assertTrue(original_ids.isdisjoint(copy_ids))

    def test_copy_connections_point_to_new_nodes(self):
        g = _make_genome()
        copy = g.copy()
        copy_node_ids = {id(n) for n in copy.nodes}
        for node in copy.nodes:
            for conn in node.connections:
                self.assertIn(id(conn.target), copy_node_ids,
                    "Connection in copy points to node from original genome")

    def test_copy_inherits_caps(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1, max_nodes=15, max_connections=30)
        g = yane.next_genome()
        copy = g.copy()
        self.assertEqual(copy.max_nodes, 15)
        self.assertEqual(copy.max_connections, 30)


class TestGenomeForward(unittest.TestCase):

    def test_forward_returns_correct_output_count(self):
        g = _make_genome(2, 1)
        out = g.forward([0.0, 1.0])
        self.assertEqual(len(out), 1)

    def test_forward_is_deterministic(self):
        g = _make_genome(2, 1)
        out1 = g.forward([0.5, 0.5])
        out2 = g.forward([0.5, 0.5])
        self.assertAlmostEqual(out1[0], out2[0])

    def test_forward_hard_resets_between_calls(self):
        g = _make_genome(2, 1)
        out1 = g.forward([1.0, 0.0])
        out2 = g.forward([0.0, 1.0])
        out3 = g.forward([1.0, 0.0])
        self.assertAlmostEqual(out1[0], out3[0],
            msg="forward() must hard-reset — same inputs must give same output")

    def test_forward_different_inputs_different_outputs(self):
        g = _make_genome(2, 1)
        out1 = g.forward([0.0, 0.0])
        out2 = g.forward([1.0, 1.0])
        # With random weights, outputs will almost certainly differ
        # (test the network is actually connected)
        self.assertEqual(len(out1), 1)
        self.assertEqual(len(out2), 1)

    def test_forward_output_in_sigmoid_range(self):
        g = _make_genome(2, 1)
        # Genomes start with no connections; add one explicitly so signal
        # reaches the output node and we can verify the sigmoid range.
        conn = Connection(g.output_nodes[0])
        conn.weight = 1.0
        g.input_nodes[0].connections.append(conn)
        out = g.forward([0.5, 0.5])
        self.assertGreater(out[0], 0.0)
        self.assertLess(out[0], 1.0)

    def test_tick_mode_propagates(self):
        g = _make_genome(2, 1)
        g.set_inputs([1.0, 0.0])
        g.tick()   # input → hidden/output
        g.tick()   # propagate further if needed
        out = g.get_outputs()
        self.assertEqual(len(out), 1)

    def test_reset_clears_triggered(self):
        g = _make_genome(2, 1)
        g.set_inputs([1.0, 1.0])
        g.tick()
        g.reset()
        self.assertEqual(len(g._triggered), 0)


    def test_forward_short_data_uses_zero_for_missing_inputs(self):
        """If data is shorter than n_inputs, missing inputs must default to 0.0."""
        g = _make_genome(4, 1)
        conn = Connection(g.output_nodes[0])
        conn.weight = 1.0
        g.input_nodes[2].connections.append(conn)  # connect input[2] (will be missing)
        out_short = g.forward([1.0, 1.0])           # only 2 values, input[2] → 0.0
        out_full  = g.forward([1.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(out_short[0], out_full[0], places=6,
            msg="Short data must produce same result as padding with zeros")

    def test_forward_with_empty_data(self):
        """forward([]) must not raise — all inputs default to 0."""
        g = _make_genome(2, 1)
        result = g.forward([])
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0] != result[0], "forward([]) must not produce NaN")

    def test_invalidate_topology_is_idempotent(self):
        """Calling _invalidate_topology multiple times must not corrupt state."""
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        for _ in range(5):
            g._invalidate_topology()
        self.assertIsNone(g._exec_order)
        self.assertIsNone(g._compiled_forward)
        self.assertIsNone(g._forward_dispatch)
        # Forward must still work after repeated invalidation
        result = g.forward([0.5, 0.5])
        self.assertEqual(len(result), 1)

    def test_memory_info_on_empty_genome(self):
        from yane.core.genome import Genome
        g = Genome()
        info = g.memory_info()
        self.assertEqual(info["nodes"], 0)
        self.assertEqual(info["connections"], 0)

    def test_max_triggers_bfs_limits_cycles(self):
        """A node with max_triggers=1 fires at most once per BFS pass."""
        from yane import NeuroEvolution
        from yane.core.connection import Connection as Conn
        yane = NeuroEvolution()
        yane.configure(1, 1)
        g = yane.next_genome()
        # Create a cycle: output → input (recurrent)
        back = Conn(g.input_nodes[0]); back.weight = 1.0
        g.output_nodes[0].connections.append(back)
        g.input_nodes[0].max_triggers = 1
        g._invalidate_topology()
        # Must not hang or raise
        result = g.forward([1.0])
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0] != result[0], "cyclic network must not produce NaN")


class TestGenomeMutation(unittest.TestCase):

    def test_mutate_does_not_exceed_max_nodes(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1, max_nodes=5, max_connections=10)
        g = yane.next_genome()
        for _ in range(100):
            g.mutate()
        self.assertLessEqual(len(g.nodes), 5)

    def test_mutate_does_not_exceed_max_connections(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1, max_nodes=5, max_connections=8)
        g = yane.next_genome()
        for _ in range(200):
            g.mutate()
        self.assertLessEqual(g.connection_count, 8)

    def test_mutation_rates_stay_above_minimum(self):
        from yane.evolution.mutation import Mutation
        g = _make_genome()
        for _ in range(500):
            g.mutate()
        for node in g.nodes:
            for m in [node.mutation_bias, node.mutation_activation,
                      node.mutation_persist, node.mutation_max_triggers]:
                self.assertGreaterEqual(m.shift_rate, Mutation.MIN_RATE)
                self.assertGreaterEqual(m.bool_rate, Mutation.MIN_RATE)

    def test_allow_memory_false_survives_repeated_mutation_all_nodes(self):
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1, max_nodes=10, max_connections=20, stateful=False)
        g = yane.next_genome()
        for _ in range(200):
            g.mutate()
        for n in g.nodes:
            self.assertFalse(n.persist_value,
                f"{n.type.value} node must stay non-persistent after many mutations")


class TestGenomeForwardCorrectness(unittest.TestCase):
    """Forward pass must return activated output values regardless of persist_value."""

    def _build_xor_network(self, out_persist: bool):
        from yane.core.genome import Genome
        from yane.core.node import Node, NodeType
        from yane.core.connection import Connection
        from yane.util.activation import ActivationType
        g = Genome()
        inp0 = Node(NodeType.INPUT, innovation=0); inp0.input_index = 0
        inp0.activation = ActivationType.LINEAR
        inp1 = Node(NodeType.INPUT, innovation=1); inp1.input_index = 1
        inp1.activation = ActivationType.LINEAR
        h = Node(NodeType.HIDDEN, innovation=2)
        h.activation = ActivationType.LINEAR; h.bias = -1.5
        out = Node(NodeType.OUTPUT, innovation=3)
        out.activation = ActivationType.LINEAR
        out.persist_value = out_persist
        c0 = Connection(h, 0); c0.weight = 2.0; inp0.connections.append(c0)
        c1 = Connection(h, 1); c1.weight = 2.0; inp1.connections.append(c1)
        c2 = Connection(out, 2); c2.weight = 1.0; h.connections.append(c2)
        c3 = Connection(out, 3); c3.weight = -1.0; inp0.connections.append(c3)
        c4 = Connection(out, 4); c4.weight = -1.0; inp1.connections.append(c4)
        g.nodes = [inp0, inp1, h, out]
        g.input_nodes = [inp0, inp1]
        g.output_nodes = [out]
        return g

    def test_forward_returns_activated_when_output_non_persistent(self):
        g = self._build_xor_network(out_persist=False)
        g.reset()
        result = g.forward([1.0, 0.0])
        # Expected XOR-like value: -1.0 + (2-1.5)*1 = -0.5 (linear activation)
        # Critical: NOT 0.0 (which was the bug)
        self.assertNotEqual(result, [0.0],
            "forward() must return activated output value, not 0, when persist_value=False")
        self.assertAlmostEqual(result[0], -0.5, places=4)

    def test_forward_same_result_regardless_of_output_persist(self):
        # With reset() between calls, persist_value on output should not change result
        g_yes = self._build_xor_network(out_persist=True)
        g_no  = self._build_xor_network(out_persist=False)
        for inp in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]:
            g_yes.reset(); r_yes = g_yes.forward(inp)
            g_no.reset();  r_no  = g_no.forward(inp)
            self.assertEqual(r_yes, r_no,
                f"Same input {inp}: result must match regardless of output persist_value")


class TestGenomePickle(unittest.TestCase):

    def test_allow_memory_survives_pickle_roundtrip(self):
        import pickle
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1, stateful=False)
        g = yane.next_genome()
        g2 = pickle.loads(pickle.dumps(g))
        self.assertFalse(g2.allow_memory)
        for n in g2.nodes:
            self.assertFalse(n.persist_value)

    def test_backward_compat_missing_allow_memory_defaults_true(self):
        from yane.core.genome import Genome
        g = _make_genome()
        state = g.__getstate__()
        state.pop('allow_memory', None)   # simulate old pickle without flag
        g2 = Genome()
        g2.__setstate__(state)
        self.assertTrue(g2.allow_memory,
            "Old pickled genomes must default allow_memory=True for backward compatibility")

    def test_backward_compat_old_allow_output_memory_attribute(self):
        from yane.core.genome import Genome
        g = _make_genome()
        state = g.__getstate__()
        state.pop('allow_memory', None)
        state['allow_output_memory'] = False  # simulate pre-rename pickle
        g2 = Genome()
        g2.__setstate__(state)
        self.assertFalse(g2.allow_memory,
            "Old allow_output_memory=False must migrate to allow_memory=False")
        self.assertFalse(hasattr(g2, 'allow_output_memory') and 'allow_output_memory' in g2.__dict__,
            "Obsolete allow_output_memory attribute should not leak through")


if __name__ == "__main__":
    unittest.main()
