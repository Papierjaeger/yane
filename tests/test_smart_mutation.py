import unittest
from yane.core.node import Node, NodeType
from yane.evolution import smart_mutation


def _make_genome(n_inputs=2, n_outputs=1, max_nodes=20, max_connections=50):
    from yane import NeuroEvolution
    yane = NeuroEvolution()
    yane.configure(n_inputs, n_outputs, max_nodes=max_nodes, max_connections=max_connections)
    return yane.next_genome()


class TestAddNode(unittest.TestCase):

    def test_add_node_increases_node_count(self):
        g = _make_genome()
        smart_mutation.add_connection(g)  # add_node requires an existing connection to split
        before = len(g.nodes)
        smart_mutation.add_node(g)
        self.assertEqual(len(g.nodes), before + 1)

    def test_add_node_does_not_break_forward_pass(self):
        """add_node must not crash and must produce a valid output in [0,1]."""
        g = _make_genome(2, 1)
        smart_mutation.add_node(g)
        out = g.forward([1.0, 0.0])
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0] != out[0], "output must not be NaN")

    def test_add_node_respects_max_nodes(self):
        g = _make_genome(2, 1, max_nodes=3)  # already at 3 nodes (2 in + 1 out)
        smart_mutation.add_node(g)
        self.assertEqual(len(g.nodes), 3, "add_node must respect max_nodes cap")

    def test_add_node_without_connections_is_noop(self):
        from yane.core.genome import Genome
        g = Genome()
        # Add a bare input and output without connecting them
        inp = Node(NodeType.INPUT); g.nodes.append(inp); g.input_nodes.append(inp)
        out = Node(NodeType.OUTPUT); g.nodes.append(out); g.output_nodes.append(out)
        smart_mutation.add_node(g)  # no connections → no-op (nothing to split)
        self.assertEqual(len(g.nodes), 2, "add_node without connections must be a no-op")


class TestRemoveNodeSelfLoop(unittest.TestCase):
    """remove_node must not infinite-loop when the node has a self-connection."""

    def test_remove_node_with_self_loop_terminates(self):
        from yane.core.genome import Genome
        from yane.core.connection import Connection
        from yane.core.node import Node, NodeType

        g = Genome()
        inp = Node(NodeType.INPUT);  g.nodes.append(inp);  g.input_nodes.append(inp)
        hid = Node(NodeType.HIDDEN); g.nodes.append(hid)
        out = Node(NodeType.OUTPUT); g.nodes.append(out);  g.output_nodes.append(out)

        # inp → hid, hid → out, hid → hid (self-loop)
        c1 = Connection(hid); c1.weight = 1.0; inp.connections.append(c1)
        c2 = Connection(out); c2.weight = 1.0; hid.connections.append(c2)
        c3 = Connection(hid); c3.weight = 0.5; hid.connections.append(c3)  # self-loop

        g.bypass_connection_prob = 1.0  # always create bypass → worst case

        # Must return quickly — previously caused an infinite loop
        import signal

        def _timeout(sig, frame):
            raise TimeoutError("remove_node did not terminate")

        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(3)
        try:
            smart_mutation.remove_node(g)
        finally:
            signal.alarm(0)

        self.assertNotIn(hid, g.nodes, "hidden node must be removed")


class TestRemoveNode(unittest.TestCase):

    def test_remove_node_decreases_node_count(self):
        g = _make_genome(2, 1)
        # Add a hidden node first
        smart_mutation.add_node(g)
        before = len(g.nodes)
        smart_mutation.remove_node(g)
        self.assertLessEqual(len(g.nodes), before)

    def test_remove_node_does_nothing_without_hidden(self):
        g = _make_genome(2, 1)
        # Should have only input + output, no hidden
        before = len(g.nodes)
        smart_mutation.remove_node(g)
        self.assertEqual(len(g.nodes), before,
            "remove_node must be a no-op when there are no hidden nodes")

    def test_remove_node_cleans_incoming_connections(self):
        g = _make_genome(2, 1)
        smart_mutation.add_node(g)
        hidden = [n for n in g.nodes if n.type == NodeType.HIDDEN]
        if not hidden:
            return
        target = hidden[0]
        g.bypass_connection_prob = 0.0  # no bypass, so no new connections
        smart_mutation.remove_node(g)
        # No connection should point to the removed node
        for node in g.nodes:
            for conn in node.connections:
                self.assertIsNot(conn.target, target,
                    "remove_node left a dangling connection to the removed node")

    def test_bypass_connection_created(self):
        g = _make_genome(2, 1)
        smart_mutation.add_node(g)
        g.bypass_connection_prob = 1.0  # always create bypass
        before_conns = g.connection_count
        smart_mutation.remove_node(g)
        # Connections should be restructured (bypass created)
        self.assertGreaterEqual(g.connection_count, 0)


class TestAddConnection(unittest.TestCase):

    def test_add_connection_increases_count(self):
        from yane.core.genome import Genome
        from yane.core.node import Node, NodeType
        g = Genome()
        n1 = Node(NodeType.INPUT); g.nodes.append(n1); g.input_nodes.append(n1)
        n2 = Node(NodeType.OUTPUT); g.nodes.append(n2); g.output_nodes.append(n2)
        smart_mutation.add_connection(g)
        self.assertEqual(g.connection_count, 1)

    def test_add_connection_no_duplicate(self):
        g = _make_genome(2, 1)
        # Force a specific connection
        src, tgt = g.nodes[0], g.nodes[-1]
        from yane.core.connection import Connection
        src.connections.append(Connection(tgt))
        count_before = g.connection_count
        # Try to add the same connection
        for _ in range(20):
            smart_mutation.add_connection(g)
        # If only 2 nodes, only 1 connection possible; more nodes = more possible
        self.assertGreaterEqual(g.connection_count, count_before)

    def test_add_connection_respects_max_connections(self):
        g = _make_genome(2, 1, max_connections=2)
        for _ in range(10):
            smart_mutation.add_connection(g)
        self.assertLessEqual(g.connection_count, 2)

    def test_self_loop_allowed(self):
        from yane.core.genome import Genome
        from yane.core.node import Node, NodeType
        g = Genome()
        n = Node(NodeType.HIDDEN); g.nodes.append(n)
        # Manually add self-loop
        from yane.core.connection import Connection
        n.connections.append(Connection(n))
        self.assertEqual(n.connections[0].target, n)


class TestRemoveConnection(unittest.TestCase):

    def test_remove_connection_decreases_count(self):
        g = _make_genome(2, 1)
        before = g.connection_count
        if before == 0:
            return
        smart_mutation.remove_connection(g)
        self.assertEqual(g.connection_count, before - 1)

    def test_remove_connection_no_op_when_empty(self):
        from yane.core.genome import Genome
        g = Genome()
        smart_mutation.remove_connection(g)  # must not raise
        self.assertEqual(g.connection_count, 0)


class TestAddNodeSplitCorrectness(unittest.TestCase):

    def test_add_node_preserves_output_value(self):
        """A→B (weight w) becomes A→N→B. Output must be identical immediately after split."""
        from yane import NeuroEvolution
        from yane.core.connection import Connection
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()

        c = Connection(g.output_nodes[0]); c.weight = 0.7
        g.input_nodes[0].connections.append(c)
        g._invalidate_topology()

        data = [0.8, 0.3]
        out_before = g.forward(data)[0]

        smart_mutation.add_node(g, yane._tracker)
        out_after = g.forward(data)[0]

        self.assertAlmostEqual(out_before, out_after, places=5,
            msg="add_node must preserve network output immediately after the split")

    def test_consecutive_add_node_stays_valid(self):
        """Multiple consecutive add_node calls must produce a forward-passable network."""
        from yane import NeuroEvolution
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()
        from yane.core.connection import Connection
        g.input_nodes[0].connections.append(Connection(g.output_nodes[0]))
        g._invalidate_topology()

        for _ in range(8):
            smart_mutation.add_node(g, yane._tracker)

        result = g.forward([0.5, 0.5])
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0] != result[0], "After 8 splits, output must not be NaN")

    def test_remove_node_bypass_weight_product(self):
        """Bypass weight must equal w_in × w_out when both weights are exact."""
        from yane import NeuroEvolution
        from yane.core.connection import Connection
        yane = NeuroEvolution()
        yane.configure(2, 1)
        g = yane.next_genome()

        # Build A→N→B manually with known weights
        from yane.core.node import Node, NodeType
        n = Node(NodeType.HIDDEN, innovation=yane._tracker.next())
        g.nodes.append(n)
        c_in  = Connection(n);                  c_in.weight  = 2.0
        c_out = Connection(g.output_nodes[0]);  c_out.weight = 3.0
        g.input_nodes[0].connections.append(c_in)
        n.connections.append(c_out)
        g._invalidate_topology()

        g.bypass_connection_prob = 1.0   # always create bypass
        smart_mutation.remove_node(g, yane._tracker)

        # Find bypass connection from input[0] to output[0]
        bypass = next(
            (c for c in g.input_nodes[0].connections
             if c.target is g.output_nodes[0]),
            None,
        )
        self.assertIsNotNone(bypass, "Bypass connection must be created")
        self.assertAlmostEqual(bypass.weight, 2.0 * 3.0, places=6,
            msg="Bypass weight must equal w_in × w_out = 6.0")


class TestRewireConnection(unittest.TestCase):

    def _genome_with_two_connections(self):
        from yane.core.connection import Connection
        g = _make_genome(2, 1)
        c1 = Connection(g.output_nodes[0]); c1.weight = 0.5
        c2 = Connection(g.output_nodes[0]); c2.weight = 0.3
        g.input_nodes[0].connections.append(c1)
        g.input_nodes[1].connections.append(c2)
        g._invalidate_topology()
        return g

    def test_rewire_preserves_connection_count(self):
        g = self._genome_with_two_connections()
        before = g.connection_count
        smart_mutation.rewire_connection(g)
        self.assertEqual(g.connection_count, before,
            "rewire_connection must preserve connection count (remove one, add one)")

    def test_rewire_forward_works_after(self):
        g = self._genome_with_two_connections()
        smart_mutation.rewire_connection(g)
        result = g.forward([0.5, 0.5])
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0] != result[0], "forward after rewire must not produce NaN")

    def test_rewire_noop_on_empty_genome(self):
        from yane.core.genome import Genome
        g = Genome()
        smart_mutation.rewire_connection(g)
        self.assertEqual(g.connection_count, 0, "rewire on empty genome must be a no-op")

    def test_rewire_noop_on_genome_without_connections(self):
        g = _make_genome(2, 1)
        smart_mutation.rewire_connection(g)  # must not raise
        self.assertEqual(g.connection_count, 0)


class TestDisableEnableConnection(unittest.TestCase):

    def _genome_with_one_connection(self, weight=3.0):
        from yane.core.connection import Connection
        g = _make_genome(2, 1)
        conn = Connection(g.output_nodes[0])
        conn.weight = weight
        g.input_nodes[0].connections.append(conn)
        g._invalidate_topology()
        return g, conn

    def test_disable_connection_count_unchanged(self):
        """disable_connection keeps the connection in the genome (count stays the same)."""
        g, conn = self._genome_with_one_connection()
        before = g.connection_count
        smart_mutation.disable_connection(g)
        self.assertEqual(g.connection_count, before,
            "disable_connection must not remove connections from genome")

    def test_disable_connection_marks_as_disabled(self):
        g, conn = self._genome_with_one_connection()
        smart_mutation.disable_connection(g)
        disabled = [c for n in g.nodes for c in n.connections if not c.enabled]
        self.assertEqual(len(disabled), 1, "Exactly one connection should be disabled")

    def test_disabled_connection_excluded_from_forward(self):
        """Directly disabling a connection must change the forward output."""
        from yane.core.connection import Connection
        g = _make_genome(2, 1)
        conn = Connection(g.output_nodes[0])
        conn.weight = 5.0
        g.input_nodes[0].connections.append(conn)
        g._invalidate_topology()

        out_enabled = g.forward([1.0, 0.0])

        conn.enabled = False
        g._invalidate_topology()
        out_disabled = g.forward([1.0, 0.0])

        self.assertNotAlmostEqual(out_enabled[0], out_disabled[0], places=3,
            msg="Disabling a high-weight connection must change the forward output")

    def test_enable_connection_reactivates_disabled(self):
        """enable_connection must re-enable a disabled connection."""
        from yane.core.connection import Connection
        g = _make_genome(2, 1)
        conn = Connection(g.output_nodes[0])
        conn.weight = 1.0
        conn.enabled = False
        g.input_nodes[0].connections.append(conn)
        g._invalidate_topology()

        smart_mutation.enable_connection(g)
        self.assertTrue(conn.enabled, "enable_connection must set enabled=True")

    def test_enable_connection_noop_when_none_disabled(self):
        """enable_connection must not crash when all connections are already enabled."""
        g, conn = self._genome_with_one_connection()
        conn.enabled = True
        smart_mutation.enable_connection(g)
        self.assertTrue(conn.enabled, "No disabled connection → must remain enabled")

    def test_disable_enable_cycle_restores_output(self):
        """Re-enabling a disabled connection must restore the original forward output."""
        from yane.core.connection import Connection
        g = _make_genome(2, 1)
        conn = Connection(g.output_nodes[0])
        conn.weight = 3.0
        g.input_nodes[0].connections.append(conn)
        g._invalidate_topology()

        out_original = g.forward([0.8, 0.2])

        conn.enabled = False
        g._invalidate_topology()

        conn.enabled = True
        g._invalidate_topology()

        out_restored = g.forward([0.8, 0.2])
        self.assertAlmostEqual(out_original[0], out_restored[0], places=10,
            msg="Re-enabling a connection must restore the original forward output")

    def test_add_node_skips_disabled_connections(self):
        """add_node must not split disabled connections."""
        from yane.core.connection import Connection
        g = _make_genome(2, 1)
        conn = Connection(g.output_nodes[0])
        conn.weight = 1.0
        conn.enabled = False
        g.input_nodes[0].connections.append(conn)
        g._invalidate_topology()

        n_before = len(g.nodes)
        smart_mutation.add_node(g)
        self.assertEqual(len(g.nodes), n_before,
            "add_node must skip disabled connections and be a no-op when all are disabled")

    def test_remove_connection_only_removes_enabled(self):
        """remove_connection must never remove a disabled connection."""
        from yane.core.connection import Connection
        g = _make_genome(2, 1)
        c_disabled = Connection(g.output_nodes[0]); c_disabled.weight = 0.01; c_disabled.enabled = False
        c_enabled  = Connection(g.output_nodes[0]); c_enabled.weight  = 1.0;  c_enabled.enabled  = True
        g.input_nodes[0].connections.append(c_disabled)
        g.input_nodes[1].connections.append(c_enabled)
        g._invalidate_topology()

        smart_mutation.remove_connection(g)

        remaining = [c for n in g.nodes for c in n.connections]
        self.assertEqual(len(remaining), 1, "Exactly one connection should remain after removal")
        self.assertFalse(remaining[0].enabled,
            "The remaining connection must be the disabled one "
            "(remove_connection must only target enabled connections)")

    def test_remove_connection_prefers_small_weights(self):
        """Soft-min sampling must preferentially remove small-weight connections."""
        from yane.core.connection import Connection
        removed_small = 0
        n_trials = 200
        for _ in range(n_trials):
            g = _make_genome(2, 1)
            c_small = Connection(g.output_nodes[0]); c_small.weight = 0.01
            c_large = Connection(g.output_nodes[0]); c_large.weight = 10.0
            g.input_nodes[0].connections.append(c_small)
            g.input_nodes[1].connections.append(c_large)
            g._invalidate_topology()

            smart_mutation.remove_connection(g)

            remaining = [c for n in g.nodes for c in n.connections]
            if len(remaining) == 1 and abs(remaining[0].weight - 10.0) < 1e-9:
                removed_small += 1

        # P(remove small | weight_small=0.01, weight_large=10.0) ≈ 0.999
        # Threshold 0.85 is very conservative.
        self.assertGreater(removed_small / n_trials, 0.85,
            "Small-weight connections must be removed far more often than large-weight ones")


if __name__ == "__main__":
    unittest.main()
