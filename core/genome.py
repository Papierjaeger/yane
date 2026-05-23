from __future__ import annotations
import random
from collections import deque

from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.evolution.mutation import Mutation

# Attributes shared between copy() and crossover() — scalars and Mutation objects.
_SCALAR_GENES = (
    'bypass_connection_prob', 'crossover_prob', 'offspring_factor',
    'sigma_global', 'allow_memory',
)
_MUTATION_GENES = (
    'mutation_bypass', 'mutation_add_node', 'mutation_remove_node',
    'mutation_add_connection', 'mutation_remove_connection',
    'mutation_rewire', 'mutation_disable_connection', 'mutation_enable_connection',
    'mutation_crossover', 'mutation_offspring',
    'mutation_sigma',
)

# Strategy-gene mutation specs: (attr, mutation_attr, min_val, max_val | None)
_STRATEGY_MUTATION_SPECS = (
    ('bypass_connection_prob', 'mutation_bypass',    0.0,  1.0),
    ('crossover_prob',         'mutation_crossover', 0.0,  1.0),
    ('offspring_factor',       'mutation_offspring',  0.01, None),
    ('sigma_global',           'mutation_sigma',      0.01, 20.0),
)


def _pick(a, b):
    """Return a or b with equal probability (used in crossover for gene selection)."""
    return a if random.random() < 0.5 else b


class Genome:
    def __init__(self) -> None:
        self.fitness: float = 0.0
        self.eval_time_ms: float | None = None
        self.efficiency_score: float = 1.0
        self.selection_score: float = 0.0
        self.nodes: list[Node] = []
        self.input_nodes: list[Node] = []
        self.output_nodes: list[Node] = []
        self._triggered: set[Node] = set()
        self._connection_count: int = 0   # cached; updated by _invalidate_topology()
        self._exec_order: list | None = None  # topological order; None = not yet computed
        self._has_cycles: bool = False        # True = skip topo sort, use BFS
        self._reset_nodes: list | None = None # nodes that need explicit reset before forward
        self._compiled_forward = None         # cached closure; avoids attribute lookup in hot loop
        self._forward_dispatch = None         # set after first forward(); direct call from then on
        self._innov_cache: tuple | None = None  # (innov_dict, max_innov); cleared by _invalidate_topology
        self._values_arr = None               # np.float64 array — reused node-value buffer for forward()

        # Optional size caps — set by NeuroEvolution.configure()
        self.max_nodes: int | None = None
        self.max_connections: int | None = None
        # When False, NO neurons (hidden or output) may hold persist_value=True.
        # Set by NeuroEvolution.configure(stateful=...).
        self.allow_memory: bool = True

        # Structural mutation rates
        self.mutation_add_node = Mutation()
        self.mutation_remove_node = Mutation()
        self.mutation_add_connection = Mutation()
        self.mutation_remove_connection = Mutation()
        self.mutation_rewire = Mutation()
        self.mutation_disable_connection = Mutation()
        self.mutation_disable_connection.bool_rate = 0.03  # low initial rate — reversible op
        self.mutation_enable_connection = Mutation()
        self.mutation_enable_connection.bool_rate = 0.03
        self.bypass_connection_prob: float = 0.5
        self.mutation_bypass = Mutation()

        # ── Self-adaptive strategy genes ─────────────────────────────────────
        # These are inherited and mutated like any other gene, so the
        # population discovers good values without any external tuning.

        # Probability this genome reproduces via crossover (vs. pure mutation)
        # Original NEAT uses 0.75; higher crossover combines good features faster.
        self.crossover_prob: float = 0.6
        self.mutation_crossover = Mutation()

        # Relative reproduction drive; higher → more likely to be selected
        # as a parent. Balanced by selection pressure over time.
        self.offspring_factor: float = 1.0
        self.mutation_offspring = Mutation()

        # Global step-size scale for weight/bias mutations (CMA-ES style)
        self.sigma_global: float = 1.0
        self.mutation_sigma = Mutation()

        # Set by Population.submit(); initialised here so the attribute always exists.
        self.shared_fitness: float = 0.0

        # Species placement cache for Population._assign_species() fast path.
        # Stores id(Species) of the last species this genome was placed in so
        # the fast path can skip the full O(species) search.
        self._last_species_id: int | None = None

        # Dirty flag for lazy species re-assignment.
        # Set True whenever the topology changes (_invalidate_topology).
        # Weight-only mutations leave this False so _assign_species() can skip
        # the full compatibility check and restore the genome to its last species.
        self._species_stale: bool = True

    # -------------------------------------------------------------------------
    # Tick mode
    # -------------------------------------------------------------------------

    def set_inputs(self, data: list[float]) -> None:
        n = len(data)
        for node in self.input_nodes:
            if node.input_index < n:
                node.value = data[node.input_index] * node.input_scale
                self._triggered.add(node)

    def tick(self) -> None:
        if not self._triggered:
            return
        next_triggered: set[Node] = set()
        for node in self._triggered:
            node.fire(next_triggered)
        self._triggered = next_triggered

    def get_outputs(self) -> list[float]:
        return [node.value * node.output_scale for node in self.output_nodes]

    def reset(self) -> None:
        """Reset all node values for a new episode — clears persistent memory too.

        Call this after env.reset() so hidden memory nodes start fresh.
        Between steps within an episode, use forward() which preserves
        persistent hidden node values across calls.
        """
        self._triggered.clear()
        for node in self.nodes:
            node.value = 0.0
        if self._values_arr is not None:
            self._values_arr.fill(0.0)

    # -------------------------------------------------------------------------
    # Forward mode (full pass with cycle protection)
    # -------------------------------------------------------------------------

    # Connections-per-node threshold for switching from Python loop to NumPy.
    # Below: loop over pre-compiled (target_idx, conn) pairs — fast for
    #        small/medium networks (conn._weight is a direct slot read).
    # Above: np.multiply into pre-allocated scratch + fancy scatter-add via
    #        wt_arr (kept live by the Connection.weight property setter).
    # Crossover measured at ~12 connections per node.
    _FIRE_NUMPY_THRESHOLD: int = 30  # see docstring below

    def _compile_forward(self):
        """Build an optimised closure for the acyclic forward pass.

        Selects between two implementations at compile time:

        Small-network path (all nodes have < _FIRE_NUMPY_THRESHOLD connections):
          Uses fire_simple() via the original compiled approach.  Avoids
          values_arr overhead; conn._weight is a direct slot read (not property).

        Large-network path (any node has >= _FIRE_NUMPY_THRESHOLD connections):
          Pre-allocates a float64 _values_arr buffer shared across calls.
          Per-node dispatch:
            < threshold : Python loop, conn._weight slot read.
            >= threshold: np.multiply into pre-alloc scratch + scatter-add.
          wt_arr kept live by the Connection.weight property setter.
        """
        import numpy as _np
        from yane.util.activation import ActivationType
        from yane.core.node import NodeType as _NT
        _thr = self._FIRE_NUMPY_THRESHOLD

        exec_order   = self._exec_order
        output_nodes = self.output_nodes

        # ── Decide which path to use ─────────────────────────────────────────
        max_conns = 0
        for n in exec_order:
            c = len(n.connections)
            if c > max_conns:
                max_conns = c
        for n in self.input_nodes:
            c = len(n.connections)
            if c > max_conns:
                max_conns = c

        if max_conns < _thr:
            return self._compile_forward_small()

        # ── Large-network path ─────────────────────────────────────────────
        all_nodes = self.nodes
        N = len(all_nodes)
        node_to_idx = {id(n): i for i, n in enumerate(all_nodes)}

        if self._values_arr is None or len(self._values_arr) != N:
            self._values_arr = _np.zeros(N, dtype=_np.float64)
        values = self._values_arr

        def _build(node):
            ni    = node_to_idx[id(node)]
            conns = [c for c in node.connections if c.enabled]
            n_c   = len(conns)
            if n_c >= _thr:
                tgt_arr = _np.array([node_to_idx[id(c.target)] for c in conns],
                                     dtype=_np.int32)
                wt_arr  = _np.empty(n_c, dtype=_np.float64)
                scratch = _np.empty(n_c, dtype=_np.float64)
                for j, conn in enumerate(conns):
                    wt_arr[j]        = conn._weight
                    conn._weight_arr = wt_arr
                    conn._weight_idx = j
                return (ni, None, tgt_arr, wt_arr, scratch, True)
            elif n_c > 0:
                return (ni, [(node_to_idx[id(c.target)], c) for c in conns],
                        None, None, None, False)
            else:
                return (ni, None, None, None, None, False)

        trivial_data  = []
        general_data  = []
        for node in self.input_nodes:
            entry = _build(node)
            if (node._activation is ActivationType.LINEAR
                    and node.bias == 0.0
                    and not node._persist_value):
                trivial_data.append((node,) + entry)
            else:
                general_data.append((node,) + entry)

        exec_compiled = [(n,) + _build(n) for n in exec_order]
        output_idx    = [node_to_idx[id(n)] for n in output_nodes]
        persistent_hidden = [
            (n, node_to_idx[id(n)])
            for n in exec_order
            if n.type is _NT.HIDDEN and n._persist_value
        ]
        self._reset_nodes = list(output_nodes)

        def _forward(data: list[float]) -> list[float]:
            n_data = len(data)
            for idx in output_idx:
                values[idx] = 0.0

            for node, ni, pairs, tgt_arr, wt_arr, scratch, use_np in trivial_data:
                val = (data[node.input_index] * node.input_scale
                       if node.input_index < n_data else 0.0)
                if use_np:
                    _np.multiply(wt_arr, val, out=scratch)
                    values[tgt_arr] += scratch
                elif pairs:
                    for tgt, conn in pairs:
                        values[tgt] += conn._weight * val
                values[ni] = 0.0

            for node, ni, pairs, tgt_arr, wt_arr, scratch, use_np in general_data:
                val = (data[node.input_index] * node.input_scale
                       if node.input_index < n_data else 0.0)
                values[ni] = val
                v = val + node.bias
                try:
                    activated = node._activate_fn(v)
                except (ValueError, OverflowError):
                    activated = 0.0
                if use_np:
                    _np.multiply(wt_arr, activated, out=scratch)
                    values[tgt_arr] += scratch
                elif pairs:
                    for tgt, conn in pairs:
                        values[tgt] += conn._weight * activated
                if node._retain_value:
                    values[ni] = activated
                else:
                    values[ni] = 0.0

            for node, ni, pairs, tgt_arr, wt_arr, scratch, use_np in exec_compiled:
                v = values[ni] + node.bias
                try:
                    activated = node._activate_fn(v)
                except (ValueError, OverflowError):
                    activated = 0.0
                if use_np:
                    _np.multiply(wt_arr, activated, out=scratch)
                    values[tgt_arr] += scratch
                elif pairs:
                    for tgt, conn in pairs:
                        values[tgt] += conn._weight * activated
                if node._retain_value:
                    values[ni] = activated
                else:
                    values[ni] = 0.0

            for node, idx in persistent_hidden:
                node.value = float(values[idx])
            result = []
            for node, idx in zip(output_nodes, output_idx):
                v = float(values[idx])
                node.value = v
                result.append(v * node.output_scale)
            return result

        return _forward

    def _compile_forward_small(self):
        """Original fast-path for small networks (all nodes < _FIRE_NUMPY_THRESHOLD).

        Uses fire_simple() which reads conn._weight directly (slot, no property).
        No values_arr overhead — node.value is the primary state.
        """
        from yane.util.activation import ActivationType
        reset_nodes  = list(self.output_nodes)
        self._reset_nodes = reset_nodes
        exec_order   = self._exec_order
        output_nodes = self.output_nodes

        trivial_inputs = []
        general_inputs = []
        for node in self.input_nodes:
            if (node._activation is ActivationType.LINEAR
                    and node.bias == 0.0
                    and not node._persist_value):
                # Snapshot only enabled connections at compile time — zero runtime
                # overhead for disabled connections in the acyclic fast path.
                trivial_inputs.append((node, [c for c in node.connections if c.enabled]))
            else:
                general_inputs.append(node)

        if general_inputs:
            def _forward(data: list[float]) -> list[float]:
                for node in reset_nodes:
                    node.value = 0.0
                n = len(data)
                for node, conns in trivial_inputs:
                    val = (data[node.input_index] * node.input_scale
                           if node.input_index < n else 0.0)
                    for conn in conns:
                        conn.target.value += conn._weight * val
                    node.value = 0.0
                for node in general_inputs:
                    node.value = (data[node.input_index] * node.input_scale
                                  if node.input_index < n else 0.0)
                    node.fire_simple()
                for node in exec_order:
                    node.fire_simple()
                return [node.value * node.output_scale for node in output_nodes]
        else:
            def _forward(data: list[float]) -> list[float]:
                for node in reset_nodes:
                    node.value = 0.0
                n = len(data)
                for node, conns in trivial_inputs:
                    val = (data[node.input_index] * node.input_scale
                           if node.input_index < n else 0.0)
                    for conn in conns:
                        conn.target.value += conn._weight * val
                    node.value = 0.0
                for node in exec_order:
                    node.fire_simple()
                return [node.value * node.output_scale for node in output_nodes]

        return _forward


    def _build_exec_order(self) -> list[Node] | None:
        """Topological sort (Kahn's algorithm) for acyclic networks.

        Input nodes are treated as already-processed sources: their outgoing
        edges are counted but their in_degree contribution is immediately
        removed so their targets can enter the queue first.

        Returns non-input nodes in execution order, or None if cycles exist.
        """
        inputs = set(self.input_nodes)
        in_degree: dict[Node, int] = {n: 0 for n in self.nodes if n not in inputs}
        for node in self.nodes:
            for conn in node.connections:
                if conn.enabled and conn.target in in_degree:
                    in_degree[conn.target] += 1

        # Input nodes are "pre-processed" — decrement their targets immediately.
        queue: deque[Node] = deque()
        for inp in self.input_nodes:
            for conn in inp.connections:
                if conn.enabled and conn.target in in_degree:
                    in_degree[conn.target] -= 1
                    if in_degree[conn.target] == 0:
                        queue.append(conn.target)
        # Nodes with no incoming connections at all (isolated hidden nodes).
        queued = set(queue)
        for node, deg in in_degree.items():
            if deg == 0 and node not in queued:
                queue.append(node)

        order: list[Node] = []
        while queue:
            node = queue.popleft()  # O(1) vs list.pop(0) O(n)
            order.append(node)
            for conn in node.connections:
                if not conn.enabled:
                    continue
                t = conn.target
                if t in in_degree:
                    in_degree[t] -= 1
                    if in_degree[t] == 0:
                        queue.append(t)

        if len(order) != len(in_degree):
            return None  # cycle detected → fall back to BFS
        return order

    def _bfs_forward(self, data: list[float]) -> list[float]:
        """Slow path: BFS with cycle protection for recurrent networks.

        Only output nodes are zeroed per step; persistent hidden nodes
        keep their value from the previous call (memory).
        """
        from yane.core.node import NodeType as _NT
        self._triggered.clear()
        for node in self.nodes:
            # Persistent hidden nodes keep value (memory between steps).
            # Output nodes reset (fresh accumulation). Non-persistent nodes
            # are zeroed by fire() after firing, but we zero them here too
            # in case they receive no signal this step.
            if not (node.persist_value and node.type == _NT.HIDDEN):
                node.value = 0.0
        self.set_inputs(data)

        trigger_counts: dict[Node, int] = {}
        # Use the _triggered set directly — avoids O(all_nodes) scan per iteration.
        # list() creates a snapshot so fire() can safely modify _triggered via next_pending.
        pending: list[Node] = list(self._triggered)
        next_pending: set[Node] = set()

        while pending:
            next_pending.clear()
            for node in pending:
                cnt = trigger_counts.get(node, 0)
                if cnt >= node.max_triggers:
                    continue
                trigger_counts[node] = cnt + 1
                node.fire(next_pending)
            # Convert set directly to list — O(pending) instead of O(all_nodes).
            pending = list(next_pending)

        return self.get_outputs()

    def forward(self, data: list[float]) -> list[float]:
        # Guard: gym environments (CartPole, etc.) return numpy arrays as observations.
        # numpy.float64 values propagate through node.value slots, causing spurious
        # numpy overflow/cast warnings and potential instability.  Convert to a plain
        # Python list of floats once, at the boundary, so the rest of the network
        # always sees Python floats.
        if type(data) is not list:
            data = [float(x) for x in data]

        # _forward_dispatch is set once after topology is resolved:
        # either to the compiled fast-path closure or to _bfs_forward.
        # After the first call, this is a direct 1-attribute-read + 1-call.
        fn = self._forward_dispatch
        if fn is not None:
            return fn(data)

        # First call: resolve topology and install the dispatch function.
        if not self._has_cycles:
            exec_order = self._build_exec_order()
            if exec_order is not None:
                self._exec_order = exec_order
                compiled = self._compile_forward()
                self._compiled_forward = compiled
                self._forward_dispatch = compiled
                return compiled(data)
            self._has_cycles = True

        self._forward_dispatch = self._bfs_forward
        return self._bfs_forward(data)

    # -------------------------------------------------------------------------
    # Mutation
    # -------------------------------------------------------------------------

    # Minimum probability for structural mutations — prevents self-adaptive rates
    # from drifting to ~0 when Lamarck makes weight-only improvements dominate.
    # The floor is intentionally small (1%) so it only kicks in when normal
    # self-adaptation has already suppressed the rates below this level.
    _STRUCT_FLOOR = 0.01

    def mutate(self, tracker=None) -> None:
        from yane.evolution import smart_mutation

        floor = self._STRUCT_FLOOR
        if self.mutation_add_node.mutate_bool(False) or random.random() < floor:
            smart_mutation.add_node(self, tracker)
        if self.mutation_remove_node.mutate_bool(False) or random.random() < floor:
            smart_mutation.remove_node(self, tracker)
        if self.mutation_add_connection.mutate_bool(False) or random.random() < floor:
            smart_mutation.add_connection(self, tracker)
        if self.mutation_remove_connection.mutate_bool(False) or random.random() < floor:
            smart_mutation.remove_connection(self, tracker)
        # Rewire and disable/enable have no floor — purely self-adaptive.
        # Rewire is net-neutral (remove + add) so it doesn't need a minimum
        # exploration guarantee.  Disable/enable are reversible and should
        # not skew the structural add/remove balance.
        if self.mutation_rewire.mutate_bool(False):
            smart_mutation.rewire_connection(self, tracker)
        if self.mutation_disable_connection.mutate_bool(False):
            smart_mutation.disable_connection(self)
        if self.mutation_enable_connection.mutate_bool(False):
            smart_mutation.enable_connection(self)

        sigma = self.sigma_global
        for node in self.nodes:
            node.mutate(sigma)
        if not self.allow_memory:
            # Tasks with allow_memory=False allow no persistent neurons at all.
            for node in self.nodes:
                node.persist_value = False

        for attr, mut_attr, lo, hi in _STRATEGY_MUTATION_SPECS:
            val = getattr(self, mut_attr).mutate_value(getattr(self, attr))
            if lo is not None:
                val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
            setattr(self, attr, val)

        for mut_attr in _MUTATION_GENES:
            getattr(self, mut_attr).mutate_rates()

    # -------------------------------------------------------------------------
    # Crossover
    # -------------------------------------------------------------------------

    def crossover(self, other: Genome) -> Genome:
        """NEAT-style crossover aligned by innovation numbers.

        self is assumed to be the fitter parent.

        Gene alignment rules (original NEAT paper):
        - Matching genes (same innovation number): inherit weight randomly
          from either parent with 50/50 probability.
        - Disjoint / excess genes from the FITTER parent (self): always
          inherited — these represent structure that was already selected for.
        - Disjoint / excess genes from the WEAKER parent (other): discarded,
          because they haven't proven their fitness in self's lineage.

        Node inheritance follows the same logic: nodes that appear in self are
        always included; nodes only in other are discarded (their connections
        are already filtered above anyway).
        """
        child = Genome()
        child.max_nodes = self.max_nodes
        child.max_connections = self.max_connections

        # --- Node alignment by innovation number ---
        # Build innovation → node for both parents.
        self_nodes: dict[int, Node] = {n.innovation: n for n in self.nodes if n.innovation >= 0}
        other_nodes: dict[int, Node] = {n.innovation: n for n in other.nodes if n.innovation >= 0}

        node_map: dict[Node, Node] = {}  # parent node → child node

        # Copy all nodes from fitter parent (self); for matching nodes, bias
        # remains from self but activation is picked randomly.
        for node in self.nodes:
            n = node.copy()
            node_map[node] = n
            child.nodes.append(n)
            if node.type == NodeType.INPUT:
                child.input_nodes.append(n)
            elif node.type == NodeType.OUTPUT:
                child.output_nodes.append(n)

        # Map other's fixed nodes to child's equivalents (same structural role)
        for s, o in zip(self.input_nodes, other.input_nodes):
            node_map[o] = node_map[s]
        for s, o in zip(self.output_nodes, other.output_nodes):
            node_map[o] = node_map[s]

        # For matching hidden nodes (same innovation), map other's node to
        # child's copy of self's node and blend bias randomly.
        for innov, other_node in other_nodes.items():
            if other_node.type != NodeType.HIDDEN:
                continue
            if innov in self_nodes:
                child_node = node_map[self_nodes[innov]]
                node_map[other_node] = child_node
                # 50/50: take bias and activation from either parent
                if random.random() < 0.5:
                    child_node.bias = other_node.bias
                if random.random() < 0.5:
                    child_node.activation = other_node.activation
            # Disjoint / excess from weaker parent: skip.

        # --- Connection alignment by innovation number ---
        # Build innovation → (source_node, connection) for both parents.
        self_conns: dict[int, tuple[Node, Connection]] = {
            conn.innovation: (src, conn)
            for src in self.nodes
            for conn in src.connections
            if conn.innovation >= 0
        }
        other_conns: dict[int, tuple[Node, Connection]] = {
            conn.innovation: (src, conn)
            for src in other.nodes
            for conn in src.connections
            if conn.innovation >= 0
        }

        conn_count = 0

        # Inherit all connections from self (fitter parent, disjoint/excess kept)
        for innov, (old_src, conn) in self_conns.items():
            if conn.target not in node_map:
                continue
            child_src = node_map[old_src]
            child_tgt = node_map[conn.target]
            weight = conn.weight
            # Matching gene: randomly pick weight from either parent
            if innov in other_conns and random.random() < 0.5:
                weight = other_conns[innov][1].weight
            if child.max_connections is not None and conn_count >= child.max_connections:
                break
            new_conn = Connection(child_tgt, innovation=innov)
            new_conn.weight = weight
            new_conn.mutation = conn.mutation.copy()
            child_src.connections.append(new_conn)
            conn_count += 1

        # Fallback for untracked connections (innovation == -1): use old topology-based logic
        untracked_self = [
            (src, conn) for src in self.nodes for conn in src.connections
            if conn.innovation < 0
        ]
        if untracked_self:
            existing_keys: set[tuple[int, int]] = {
                (id(node_map[src]), id(node_map[conn.target]))
                for src, conn in self_conns.values()
                if conn.target in node_map
            }
            for old_src, conn in untracked_self:
                if old_src not in node_map or conn.target not in node_map:
                    continue
                child_src = node_map[old_src]
                child_tgt = node_map[conn.target]
                key = (id(child_src), id(child_tgt))
                if key in existing_keys:
                    continue
                if child.max_connections is not None and conn_count >= child.max_connections:
                    break
                new_conn = Connection(child_tgt, innovation=-1)
                new_conn.weight = conn.weight
                new_conn.mutation = conn.mutation.copy()
                child_src.connections.append(new_conn)
                existing_keys.add(key)
                conn_count += 1

        # Strategy genes: random from either parent
        for attr in _SCALAR_GENES:
            setattr(child, attr, _pick(getattr(self, attr), getattr(other, attr)))
        for attr in _MUTATION_GENES:
            setattr(child, attr, _pick(getattr(self, attr), getattr(other, attr)).copy())

        child._invalidate_topology()
        return child

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy(self) -> Genome:
        genome = Genome()

        node_map: dict[Node, Node] = {}
        for node in self.nodes:
            new_node = node.copy()
            node_map[node] = new_node
            genome.nodes.append(new_node)
            if node.type == NodeType.INPUT:
                genome.input_nodes.append(new_node)
            elif node.type == NodeType.OUTPUT:
                genome.output_nodes.append(new_node)

        for old_node, new_node in node_map.items():
            for conn in old_node.connections:
                if conn.target in node_map:
                    new_node.connections.append(conn.copy(node_map))

        genome.fitness = self.fitness
        genome.shared_fitness = self.shared_fitness
        genome.eval_time_ms = self.eval_time_ms
        genome.efficiency_score = self.efficiency_score
        genome.selection_score = self.selection_score
        genome.max_nodes = self.max_nodes
        genome.max_connections = self.max_connections
        for attr in _SCALAR_GENES:
            setattr(genome, attr, getattr(self, attr))
        for attr in _MUTATION_GENES:
            setattr(genome, attr, getattr(self, attr).copy())

        genome._connection_count = self._connection_count
        # Topology caches reference old Node objects — must recompute for the copy.
        genome._exec_order = None
        genome._reset_nodes = None
        genome._compiled_forward = None
        genome._forward_dispatch = None
        genome._has_cycles = self._has_cycles
        genome._values_arr = None   # fresh buffer; allocated on first forward()
        return genome

    # -------------------------------------------------------------------------
    # Pickle support (multiprocessing)
    # -------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Return picklable state — strip compiled closures that can't be pickled.

        _compiled_forward and _forward_dispatch are nested closures created by
        _compile_forward().  They capture local variables and are not importable,
        so pickle rejects them.  Clearing them here is safe: the first forward()
        call in the subprocess will rebuild them from the node/connection data,
        which IS pickled correctly.
        """
        state = self.__dict__.copy()
        state['_compiled_forward'] = None
        state['_forward_dispatch'] = None
        state['_values_arr'] = None   # numpy buffer; rebuilt on first forward() in subprocess
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Backward compat: old pickles may carry either old or no flag.
        if 'allow_memory' not in state:
            self.allow_memory = state.get('allow_output_memory', True)
        # Drop the obsolete attribute if it leaked in.
        self.__dict__.pop('allow_output_memory', None)
        # Backward compat: old pickles won't have these attributes.
        self.__dict__.setdefault('_last_species_id', None)
        self.__dict__.setdefault('_species_stale', True)
        self.__dict__.setdefault('_values_arr', None)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def _clear(self) -> None:
        """Break all internal reference cycles so Python can immediately GC this genome.

        Node A → Connection → Node B → Connection → Node A creates a cycle that
        Python's reference counter cannot resolve alone; it waits for the cyclic GC.
        _forward_dispatch may hold a bound method (self._bfs_forward) which also
        creates a genome → method → genome cycle; clear it here too.
        """
        for node in self.nodes:
            node.connections.clear()
        self.nodes.clear()
        self.input_nodes.clear()
        self.output_nodes.clear()
        self._triggered.clear()
        self._compiled_forward = None
        self._forward_dispatch = None
        self._reset_nodes = None
        self._innov_cache = None

    def all_connections(self) -> list[tuple[Node, Connection]]:
        """All (source, connection) pairs in the network."""
        return [(node, conn) for node in self.nodes for conn in node.connections]

    def _get_innov_cache(self) -> tuple:
        """Return (innov_dict, max_innov, n_innov, key_frozenset, sorted_arr) for this genome.

        Built once on first call after any structural change, then reused.
        sorted_arr is a sorted np.int32 array of innovation keys — used by the
        NumPy path in _compatibility() for large networks.  None for small networks
        (below _COMPAT_NUMPY_THRESHOLD) to avoid allocation overhead.
        """
        if self._innov_cache is None:
            from yane.evolution.population import _COMPAT_NUMPY_THRESHOLD
            import numpy as _np
            d = {conn.innovation: conn
                 for src in self.nodes
                 for conn in src.connections
                 if conn.innovation >= 0}
            n = len(d)
            sorted_arr = _np.array(sorted(d), dtype=_np.int32) if n >= _COMPAT_NUMPY_THRESHOLD else None
            self._innov_cache = (d, max(d, default=-1), n, frozenset(d), sorted_arr)
        return self._innov_cache

    def _invalidate_topology(self) -> None:
        self._connection_count = sum(len(n.connections) for n in self.nodes)
        self._exec_order = None
        self._has_cycles = False
        self._reset_nodes = None
        self._compiled_forward = None
        self._forward_dispatch = None
        self._innov_cache = None
        self._species_stale = True  # topology changed → needs species re-assignment
        self._values_arr = None     # will be reallocated in next _compile_forward()
        # Disconnect all connections from their weight arrays so stale array
        # references don't survive recompilation.
        for node in self.nodes:
            for conn in node.connections:
                conn._weight_arr = None

    @property
    def connection_count(self) -> int:
        return self._connection_count

    def memory_info(self) -> dict:
        """Returns a breakdown of node/connection counts for memory profiling."""
        return {
            "nodes": len(self.nodes),
            "connections": self.connection_count,
            "input_nodes": len(self.input_nodes),
            "output_nodes": len(self.output_nodes),
            "max_nodes": self.max_nodes,
            "max_connections": self.max_connections,
        }
