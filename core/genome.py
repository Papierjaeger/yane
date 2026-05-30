from __future__ import annotations
import random
from collections import deque

from yane.core.connection import Connection
from yane.core.node import Node, NodeType
from yane.evolution.mutation import Mutation

# Attributes shared between copy() and crossover() — scalars and Mutation objects.
_SCALAR_GENES = (
    'bypass_connection_prob', 'crossover_prob', 'offspring_factor',
    'sigma_global', 'lamarck_sigma', 'allow_memory',
)
# Global genome ID counter for lineage tracking
_GENOME_ID_COUNTER: int = 0

_MUTATION_GENES = (
    'mutation_bypass', 'mutation_add_node', 'mutation_remove_node',
    'mutation_add_connection', 'mutation_remove_connection',
    'mutation_rewire', 'mutation_disable_connection', 'mutation_enable_connection',
    'mutation_crossover', 'mutation_offspring',
    'mutation_sigma', 'mutation_lamarck_sigma',
)

# Strategy-gene mutation specs: (attr, mutation_attr, min_val, max_val | None)
_STRATEGY_MUTATION_SPECS = (
    ('bypass_connection_prob', 'mutation_bypass',         0.0,   1.0),
    ('crossover_prob',         'mutation_crossover',      0.0,   1.0),
    ('offspring_factor',       'mutation_offspring',      0.01,  None),
    ('sigma_global',           'mutation_sigma',          0.01,  20.0),
    ('lamarck_sigma',          'mutation_lamarck_sigma',  0.001, 10.0),
)


def _pick(a, b):
    """Return a or b with equal probability (used in crossover for gene selection)."""
    return a if random.random() < 0.5 else b


class Genome:
    def __init__(self) -> None:
        global _GENOME_ID_COUNTER
        self._genome_id: int = _GENOME_ID_COUNTER
        _GENOME_ID_COUNTER += 1
        self._parent_ids: list[int] = []
        self.fitness: float = 0.0
        self.raw_fitness: float = 0.0  # task score before fitness-component bonus
        self.objectives: tuple[float, ...] | None = None
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
        self._active_structure_cache: dict | None = None  # cleared by _invalidate_topology
        self._values_arr = None               # np.float64 array — reused node-value buffer for forward()

        # Output sanitizing (NaN/Inf → fallback after every forward pass)
        self._output_sanitize: bool = False
        self._output_fallback: float = 0.0
        self.n_output_sanitized: int = 0      # cumulative, session-only (not copied to children)

        # Probabilistic / Bayesian NEAT — add Gaussian noise after each forward().
        # Set via NeuroEvolution.set_probabilistic() or bayesian_neat.set_probabilistic().
        self._prob_enabled: bool = False
        self._prob_noise_std: float = 0.05
        self._prob_inference_mode: bool = False

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

        # ── Mutation diagnostics ───────────────────────────────────────────
        # Populated by mutate() — list of mutation type names that fired.
        # Reset on each mutate() call. Used by Population to track success rates.
        self._mutation_types_fired: list[str] = []
        # Parent fitness at spawn time — set by Population, used to measure
        # whether the mutations were beneficial.
        self._parent_fitness: float | None = None

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

        # Dedicated step-size for Lamarckian hill-climbing — evolves independently
        # of sigma_global so the population can find the right refinement scale
        # without interfering with the structural mutation pressure.
        self.lamarck_sigma: float = 1.0
        self.mutation_lamarck_sigma = Mutation()

        # Gate-source mutation: rewire the gate_node reference on a random persistent
        # hidden node.  Lower initial rate than structural mutations (rarer operation).
        self.mutation_gate_source = Mutation()
        self.mutation_gate_source.bool_rate = 0.05

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

        # DARTS-Lite: gate values per connection innovation (lazy; None = disabled).
        # Updated each generation from |weight| via sigmoid by NeuroEvolution.
        self._darts_gates: dict[int, float] | None = None

        # Shared Weights: group_id → shared weight value.
        # Connections assigned to a group via set_weight_group() share this value.
        self.weight_groups: dict[str, float] = {}
        # group_id → list[Connection] — rebuilt lazily by sync_shared_weights().
        self._weight_group_members: dict[str, list] = {}

        # Evolvable Input Aggregation — optional preprocessing layer.
        # None = disabled (no cost).  Set via NeuroEvolution.set_input_grouping().
        self.grouper = None  # InputGrouper | None

        # Evolvable Output Synergy — optional postprocessing layer.
        # None = disabled (no cost).  Set via NeuroEvolution.set_output_grouping().
        self.out_grouper = None  # OutputGrouper | None

        # Convolutional NEAT — optional image-processing front-end.
        # None = disabled (no cost).  Set via NeuroEvolution.set_conv_neat().
        self.conv_stack = None  # ConvStack | None

        # Developmental NEAT — ontogenesis during evaluation.
        # dev_rules: list of DevelopmentalRule that may fire after forward().
        # _dev_added: (src_node, conn) pairs added this episode (cleared on reset()).
        # _dev_frozen: when True, no rules are evaluated.
        self.dev_rules: list = []
        self._dev_added: list = []
        self._dev_frozen: bool = False

        # Evolvable Attention Heads — optional attention preprocessing layer.
        # None = disabled (no cost).  Set via NeuroEvolution.set_attention().
        self.attention_block = None  # AttentionBlock | None

    # -------------------------------------------------------------------------
    # Developmental NEAT
    # -------------------------------------------------------------------------

    def developmental_forward(self, inputs: "list[float]") -> "list[float]":
        """Run forward pass then evaluate developmental rules.

        After the standard ``forward()`` computes node activations, each rule
        in ``self.dev_rules`` is checked.  Triggered rules add connections
        (which take effect on the *next* call within the same episode).

        ``reset()`` removes all episode-local connections and resets rule
        fire counters.

        Returns
        -------
        list[float]
            Same as ``forward(inputs)``.
        """
        from yane.evolution.developmental import developmental_forward as _dev_fwd
        return _dev_fwd(self, inputs)

    def freeze_development(self) -> None:
        """Disable all developmental rules for the remainder of this episode.

        Call ``reset()`` (or set ``genome._dev_frozen = False``) to re-enable.
        """
        self._dev_frozen = True

    # -------------------------------------------------------------------------
    # Convolutional NEAT
    # -------------------------------------------------------------------------

    def forward_image(
        self,
        pixels: "list[float]",
        height: int,
        width: int,
        channels: int = 1,
    ) -> "list[float]":
        """Process a flat image through the conv stack and then the NEAT network.

        Requires ``self.conv_stack`` to be set (via
        ``NeuroEvolution.set_conv_neat()``).

        Parameters
        ----------
        pixels :
            Flat image data, channel-last ordering: pixel at (r, c, ch) is at
            index ``r * width * channels + c * channels + ch``.
        height, width :
            Spatial dimensions.
        channels :
            Number of image channels.

        Returns
        -------
        list[float]
            Same as ``self.forward()``.

        Raises
        ------
        RuntimeError
            When no :class:`~yane.evolution.conv_neat.ConvStack` is attached.
        """
        if self.conv_stack is None:
            raise RuntimeError(
                "forward_image() requires a ConvStack. "
                "Call NeuroEvolution.set_conv_neat() first."
            )
        flat_features = self.conv_stack.forward_image(pixels, height, width, channels)
        return self.forward(flat_features)

    # -------------------------------------------------------------------------
    # Tick mode
    # -------------------------------------------------------------------------

    def set_inputs(self, data: list[float]) -> None:
        n = len(data)
        n_in = len(self.input_nodes)
        if n != n_in:
            import warnings
            warnings.warn(
                f"set_inputs: expected {n_in} inputs, got {n}. "
                f"{'Padding with zeros.' if n < n_in else 'Ignoring extra inputs.'}",
                stacklevel=2,
            )
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

    def _sanitize_outputs(self, result: list[float]) -> list[float]:
        """Replace NaN/Inf in output vector with fallback. Mutates result in-place."""
        import math
        fallback = self._output_fallback
        for i, v in enumerate(result):
            if not math.isfinite(v):
                result[i] = fallback
                self.n_output_sanitized += 1
        return result

    def _apply_prob_noise(self, result: list[float]) -> list[float]:
        """Add Gaussian noise to forward output if probabilistic mode is active."""
        if not self._prob_enabled:
            return result
        if self._prob_inference_mode:
            return result
        import random as _rnd
        std = self._prob_noise_std
        if std <= 0.0:
            return result
        return [v + _rnd.gauss(0.0, std) for v in result]

    def set_probabilistic(
        self,
        enabled: bool = True,
        noise_std: float = 0.05,
        inference_mode: bool = False,
    ) -> None:
        """Enable or disable probabilistic (stochastic) output noise.

        Parameters
        ----------
        enabled:
            Whether probabilistic noise is active.
        noise_std:
            Standard deviation of the per-output Gaussian noise.
        inference_mode:
            When True, forward() is deterministic (no noise).
        """
        self._prob_enabled = enabled
        self._prob_noise_std = float(noise_std)
        self._prob_inference_mode = inference_mode

    def bayesian_forward(
        self,
        inputs: list,
        n: int = 100,
    ) -> tuple:
        """Monte-Carlo forward pass: collect *n* stochastic samples.

        Returns (mean_outputs, std_outputs), both lists of length n_outputs.
        std_outputs[i] converges as O(1/sqrt(n)).

        Temporarily forces probabilistic mode regardless of current settings.
        """
        from yane.evolution.bayesian_neat import bayesian_forward as _bf
        return _bf(self, inputs, n=n)

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
        # Developmental NEAT: remove episode-local connections and reset rule counters.
        dev_added = getattr(self, "_dev_added", [])
        if dev_added:
            from yane.evolution.developmental import reset_developmental
            reset_developmental(self)

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

        # Pre-compute gate source indices for persistent nodes that have a gate_node.
        # Keyed by ni (node index); value is gate_idx (index of the gate source node).
        # Read at compile time so the hot loop avoids attribute lookups each step.
        import math as _math
        _exp = _math.exp
        _gate_source: dict[int, int] = {}
        for node in exec_order:
            if node._persist_value and node.gate_node is not None:
                gid = id(node.gate_node)
                if gid in node_to_idx:
                    _gate_source[node_to_idx[id(node)]] = node_to_idx[gid]

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
                old_value = values[ni]
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
                    if node._persist_value:
                        retained = node.leak_alpha * activated
                        _gi = _gate_source.get(ni, -1)
                        if _gi >= 0:
                            _gv = values[_gi]
                            try:
                                gate = 1.0 / (1.0 + _exp(-_gv))
                            except OverflowError:
                                gate = 0.0 if _gv < 0 else 1.0
                        else:
                            gate = node.memory_gate
                        values[ni] = gate * old_value + (1.0 - gate) * retained
                    else:
                        values[ni] = activated
                else:
                    values[ni] = 0.0

            for node, ni, pairs, tgt_arr, wt_arr, scratch, use_np in exec_compiled:
                old_value = values[ni]
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
                    if node._persist_value:
                        retained = node.leak_alpha * activated
                        _gi = _gate_source.get(ni, -1)
                        if _gi >= 0:
                            _gv = values[_gi]
                            try:
                                gate = 1.0 / (1.0 + _exp(-_gv))
                            except OverflowError:
                                gate = 0.0 if _gv < 0 else 1.0
                        else:
                            gate = node.memory_gate
                        values[ni] = gate * old_value + (1.0 - gate) * retained
                    else:
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
            result = fn(data)
            if self._output_sanitize:
                result = self._sanitize_outputs(result)
            return self._apply_prob_noise(result) if self._prob_enabled else result

        # First call: resolve topology and install the dispatch function.
        if not self._has_cycles:
            exec_order = self._build_exec_order()
            if exec_order is not None:
                self._exec_order = exec_order
                compiled = self._compile_forward()
                self._compiled_forward = compiled
                self._forward_dispatch = compiled
                result = compiled(data)
                if self._output_sanitize:
                    result = self._sanitize_outputs(result)
                return self._apply_prob_noise(result) if self._prob_enabled else result
            self._has_cycles = True

        self._forward_dispatch = self._bfs_forward
        result = self._bfs_forward(data)
        if self._output_sanitize:
            result = self._sanitize_outputs(result)
        return self._apply_prob_noise(result) if self._prob_enabled else result

    def forward_batch(self, batch) -> list[list[float]]:
        """Vectorized forward pass for a batch of input vectors.

        ~10–100× faster than sequential forward() for feed-forward (acyclic,
        non-stateful) networks because it replaces per-sample Python loops with
        NumPy matrix operations.

        Falls back to sequential forward() for:
          - Cyclic (recurrent) networks
          - Networks with persistent (memory) hidden nodes
          - Empty batches

        Args:
            batch: sequence of N input vectors (list-of-lists or 2-D ndarray).
        Returns:
            list of N output vectors, each of length n_outputs.
        """
        import numpy as _np
        from yane.util.activation import get_batch_activation_fns
        from yane.core.node import NodeType as _NT

        N = len(batch)
        if N == 0:
            return []

        # --- Fallback conditions -------------------------------------------
        if self._has_cycles:
            return [self.forward(row) for row in batch]

        has_memory = any(
            n._persist_value for n in self.nodes if n.type is _NT.HIDDEN
        )
        if has_memory:
            return [self.forward(row) for row in batch]

        # --- Ensure topology is resolved ------------------------------------
        if self._exec_order is None:
            exec_order = self._build_exec_order()
            if exec_order is None:
                self._has_cycles = True
                return [self.forward(row) for row in batch]
            self._exec_order = exec_order

        exec_order = self._exec_order
        batch_fns = get_batch_activation_fns()

        def _batch_activate(node, values):
            fn = batch_fns.get(node._activation)
            if fn is not None:
                return fn(values)
            return _np.vectorize(node._activate_fn, otypes=[_np.float64])(values)

        # --- Build node-index map (per-call; cheap for typical net sizes) --
        all_nodes = self.nodes
        n_nodes = len(all_nodes)
        node_to_idx = {id(n): i for i, n in enumerate(all_nodes)}

        # --- Allocate value matrix -----------------------------------------
        vals = _np.zeros((N, n_nodes), dtype=_np.float64)
        data = _np.asarray(batch, dtype=_np.float64)  # (N, n_inputs)
        n_inp = data.shape[1] if data.ndim > 1 else 0

        # --- Input nodes: activate + push to targets -----------------------
        for node in self.input_nodes:
            ni = node_to_idx[id(node)]
            col = (data[:, node.input_index] * node.input_scale
                   if node.input_index < n_inp
                   else _np.zeros(N, dtype=_np.float64))
            col = col + node.bias
            activated = _batch_activate(node, col)  # (N,)
            for conn in node.connections:
                if conn.enabled:
                    vals[:, node_to_idx[id(conn.target)]] += conn._weight * activated
            # Input node values reset to zero after pushing
            # (output_scale on inputs is unused; node.value stays 0)

        # --- Hidden / output nodes in topological order --------------------
        for node in exec_order:
            ni = node_to_idx[id(node)]
            pre = vals[:, ni] + node.bias
            activated = _batch_activate(node, pre)  # (N,)
            for conn in node.connections:
                if conn.enabled:
                    vals[:, node_to_idx[id(conn.target)]] += conn._weight * activated
            if node._retain_value:
                vals[:, ni] = activated
            else:
                vals[:, ni] = 0.0

        # --- Collect outputs -----------------------------------------------
        result = []
        sanitize = self._output_sanitize
        for i in range(N):
            row = [float(vals[i, node_to_idx[id(n)]]) * n.output_scale
                   for n in self.output_nodes]
            if sanitize:
                self._sanitize_outputs(row)
            result.append(row)
        return result

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

        self._mutation_types_fired = []  # reset diagnostics

        floor = self._STRUCT_FLOOR
        if self.mutation_add_node.mutate_bool(False) or random.random() < floor:
            smart_mutation.add_node(self, tracker)
            self._mutation_types_fired.append("add_node")
        if self.mutation_remove_node.mutate_bool(False) or random.random() < floor:
            smart_mutation.remove_node(self, tracker)
            self._mutation_types_fired.append("remove_node")
        if self.mutation_add_connection.mutate_bool(False) or random.random() < floor:
            smart_mutation.add_connection(self, tracker)
            self._mutation_types_fired.append("add_connection")
        if self.mutation_remove_connection.mutate_bool(False) or random.random() < floor:
            smart_mutation.remove_connection(self, tracker)
            self._mutation_types_fired.append("remove_connection")
        if self.mutation_rewire.mutate_bool(False):
            smart_mutation.rewire_connection(self, tracker)
            self._mutation_types_fired.append("rewire")
        if self.mutation_disable_connection.mutate_bool(False):
            smart_mutation.disable_connection(self)
            self._mutation_types_fired.append("disable")
        if self.mutation_enable_connection.mutate_bool(False):
            smart_mutation.enable_connection(self)
            self._mutation_types_fired.append("enable")
        if self.mutation_gate_source.mutate_bool(False):
            self._mutate_gate_source()
            self._mutation_types_fired.append("gate_source")

        sigma = self.sigma_global
        for node in self.nodes:
            node.mutate(sigma)
        if not self.allow_memory:
            # Tasks with allow_memory=False allow no persistent neurons at all.
            for node in self.nodes:
                node.persist_value = False
                node.gate_node = None  # dynamic gating requires memory

        for attr, mut_attr, lo, hi in _STRATEGY_MUTATION_SPECS:
            val = getattr(self, mut_attr).mutate_value(getattr(self, attr))
            if lo is not None:
                val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
            setattr(self, attr, val)

        for mut_attr in _MUTATION_GENES:
            getattr(self, mut_attr).mutate_rates()

        # Shared weights: resync group values from first-encountered rep per group.
        if self.weight_groups:
            _seen_grps: set[str] = set()
            for _node in self.nodes:
                for _conn in _node.connections:
                    _gid = getattr(_conn, 'weight_group', None)
                    if _gid is not None and _gid not in _seen_grps:
                        _seen_grps.add(_gid)
                        self.weight_groups[_gid] = _conn.weight
            self.sync_shared_weights()

        # Lamarck-Momentum nudge: bias mutation direction toward last Lamarck delta.
        _mom_prob = getattr(self, '_lamarck_momentum_prob', 0.0)
        if _mom_prob > 0.0:
            _mw: dict = getattr(self, '_lamarck_momentum', {})
            _mb: list = getattr(self, '_lamarck_bias_momentum', [])
            _decay: float = getattr(self, '_lamarck_momentum_decay', 0.9)
            bi = 0
            for node in self.nodes:
                for conn in node.connections:
                    if conn.enabled and random.random() < _mom_prob:
                        m = _mw.get(conn.innovation, 0.0)
                        if m:
                            conn.weight += m
                if bi < len(_mb) and random.random() < _mom_prob:
                    node.bias += _mb[bi]
                bi += 1
            for inn in _mw:
                _mw[inn] *= _decay
            for i in range(len(_mb)):
                _mb[i] *= _decay

    def _mutate_gate_source(self) -> None:
        """Randomly assign or clear gate_node on a persistent hidden node."""
        persistent = [n for n in self.nodes
                      if n.type == NodeType.HIDDEN and n._persist_value]
        if not persistent:
            return
        target = random.choice(persistent)
        # 40% clear, 60% assign; bias toward clearing when gate is already set
        if target.gate_node is not None and random.random() < 0.4:
            target.gate_node = None
        else:
            candidates = [n for n in self.nodes if n is not target]
            if candidates:
                target.gate_node = random.choice(candidates)
        self.mutation_gate_source.mutate_rates()
        # Force recompilation so compiled forward paths pick up the new gate source.
        self._forward_dispatch = None
        self._compiled_forward = None

    # -------------------------------------------------------------------------
    # Crossover
    # -------------------------------------------------------------------------

    def crossover(self, other: Genome, blend_alpha: float = -1.0) -> Genome:
        """NEAT-style crossover aligned by innovation numbers.

        self is assumed to be the fitter parent.

        Gene alignment rules (original NEAT paper):
        - Matching genes (same innovation number): inherit weight randomly
          from either parent with 50/50 probability, or fitness-blended when
          ``blend_alpha >= 0``.
        - Disjoint / excess genes from the FITTER parent (self): always
          inherited — these represent structure that was already selected for.
        - Disjoint / excess genes from the WEAKER parent (other): discarded,
          because they haven't proven their fitness in self's lineage.

        Node inheritance follows the same logic: nodes that appear in self are
        always included; nodes only in other are discarded (their connections
        are already filtered above anyway).

        Parameters
        ----------
        other : Genome
            The weaker parent.
        blend_alpha : float, optional
            When >= 0, matching gene weights are blended as
            ``blend_alpha * w_self + (1 - blend_alpha) * w_other`` instead
            of 50/50 random pick.  Negative (default) preserves the original
            random-inheritance behaviour.
        """
        child = Genome()
        child._parent_ids = [self._genome_id, other._genome_id]
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
                if blend_alpha >= 0.0:
                    # Fitness-weighted blend for bias
                    child_node.bias = blend_alpha * child_node.bias + (1.0 - blend_alpha) * other_node.bias
                else:
                    # 50/50: take bias randomly from either parent
                    if random.random() < 0.5:
                        child_node.bias = other_node.bias
                # Activation is always 50/50 (enum, not blendable)
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
            # Matching gene: blend or randomly pick weight from either parent
            if innov in other_conns:
                if blend_alpha >= 0.0:
                    # Fitness-weighted blend
                    weight = blend_alpha * weight + (1.0 - blend_alpha) * other_conns[innov][1].weight
                elif random.random() < 0.5:
                    weight = other_conns[innov][1].weight
            if child.max_connections is not None and conn_count >= child.max_connections:
                break
            new_conn = Connection(child_tgt, innovation=innov)
            new_conn.weight = weight
            new_conn.mutation = conn.mutation.copy()
            # Inherit Hebb plasticity coefficients from the fitter parent's connection.
            new_conn.hebb_a = conn.hebb_a
            new_conn.hebb_b = conn.hebb_b
            new_conn.hebb_c = conn.hebb_c
            new_conn.hebb_d = conn.hebb_d
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

        # Remap gate_node references from fitter parent (self) to child nodes.
        for old_node, new_node in node_map.items():
            if old_node in self.nodes and old_node.gate_node is not None:
                new_node.gate_node = node_map.get(old_node.gate_node)

        # Strategy genes: random from either parent
        for attr in _SCALAR_GENES:
            setattr(child, attr, _pick(getattr(self, attr), getattr(other, attr)))
        for attr in _MUTATION_GENES:
            setattr(child, attr, _pick(getattr(self, attr), getattr(other, attr)).copy())
        child.mutation_gate_source = _pick(
            self.mutation_gate_source, other.mutation_gate_source
        ).copy()

        child._output_sanitize = self._output_sanitize
        child._output_fallback = self._output_fallback
        child._prob_enabled = self._prob_enabled
        child._prob_noise_std = self._prob_noise_std
        child._prob_inference_mode = self._prob_inference_mode
        # DARTS gates from fitter parent (self); child gets fresh copy to evolve.
        if self._darts_gates is not None:
            child._darts_gates = dict(self._darts_gates)
        # Shared weight groups from fitter parent.
        child.weight_groups = dict(self.weight_groups)
        child._weight_group_members = {}
        # Evolvable input grouper: crossover if both parents have one; inherit self's otherwise.
        if self.grouper is not None and other.grouper is not None:
            try:
                child.grouper = self.grouper.crossover(other.grouper)
            except Exception:
                child.grouper = self.grouper.copy()
        elif self.grouper is not None:
            child.grouper = self.grouper.copy()
        else:
            child.grouper = None
        # Output grouper crossover.
        if self.out_grouper is not None and other.out_grouper is not None:
            try:
                child.out_grouper = self.out_grouper.crossover(other.out_grouper)
            except Exception:
                child.out_grouper = self.out_grouper.copy()
        elif self.out_grouper is not None:
            child.out_grouper = self.out_grouper.copy()
        else:
            child.out_grouper = None
        # Conv stack crossover.
        if self.conv_stack is not None and other.conv_stack is not None:
            try:
                child.conv_stack = self.conv_stack.crossover(other.conv_stack)
            except Exception:
                child.conv_stack = self.conv_stack.copy()
        elif self.conv_stack is not None:
            child.conv_stack = self.conv_stack.copy()
        else:
            child.conv_stack = None
        # Attention block crossover.
        if self.attention_block is not None and other.attention_block is not None:
            try:
                child.attention_block = self.attention_block.crossover(other.attention_block)
            except Exception:
                child.attention_block = self.attention_block.copy()
        elif self.attention_block is not None:
            child.attention_block = self.attention_block.copy()
        else:
            child.attention_block = None
        # Developmental rules: inherit from fitter parent (self); per-gene crossover.
        if self.dev_rules and other.dev_rules:
            shared = min(len(self.dev_rules), len(other.dev_rules))
            import random as _rnd
            child.dev_rules = [
                (self.dev_rules[i] if _rnd.random() < 0.5 else other.dev_rules[i]).copy()
                for i in range(shared)
            ] + [r.copy() for r in self.dev_rules[shared:]]
        else:
            child.dev_rules = [r.copy() for r in self.dev_rules]
        child._dev_added = []
        child._dev_frozen = False
        child._invalidate_topology()
        return child

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy(self) -> Genome:
        genome = Genome()
        genome._parent_ids = [self._genome_id]

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
            # Remap gate_node reference to the corresponding node in the new genome.
            if old_node.gate_node is not None:
                new_node.gate_node = node_map.get(old_node.gate_node)

        genome.fitness = self.fitness
        genome.raw_fitness = self.raw_fitness
        genome.objectives = self.objectives
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
        genome.mutation_gate_source = self.mutation_gate_source.copy()

        genome._output_sanitize = self._output_sanitize
        genome._output_fallback = self._output_fallback
        # Probabilistic mode: inherit from parent.
        genome._prob_enabled = self._prob_enabled
        genome._prob_noise_std = self._prob_noise_std
        genome._prob_inference_mode = self._prob_inference_mode
        # n_output_sanitized starts fresh (per-instance counter, not inherited)
        genome._connection_count = self._connection_count
        # Topology caches reference old Node objects — must recompute for the copy.
        genome._exec_order = None
        genome._reset_nodes = None
        genome._compiled_forward = None
        genome._forward_dispatch = None
        genome._has_cycles = self._has_cycles
        genome._values_arr = None   # fresh buffer; allocated on first forward()
        # active_structure_cache holds only integer counts — safe to share until
        # the copy is mutated (which calls _invalidate_topology → clears the cache).
        genome._active_structure_cache = self._active_structure_cache
        # DARTS gates: copy dict so child can diverge independently.
        if self._darts_gates is not None:
            genome._darts_gates = dict(self._darts_gates)
        # Shared weights: copy group values; members rebuilt lazily on sync.
        genome.weight_groups = dict(self.weight_groups)
        genome._weight_group_members = {}
        # Evolvable input grouper: deep-copy so child can mutate independently.
        genome.grouper = self.grouper.copy() if self.grouper is not None else None
        # Evolvable output grouper.
        genome.out_grouper = self.out_grouper.copy() if self.out_grouper is not None else None
        # Convolutional NEAT stack.
        genome.conv_stack = self.conv_stack.copy() if self.conv_stack is not None else None
        genome.attention_block = self.attention_block.copy() if self.attention_block is not None else None
        # Developmental rules: copy each rule; episode state (_dev_added) starts fresh.
        genome.dev_rules = [r.copy() for r in self.dev_rules]
        genome._dev_added = []
        genome._dev_frozen = self._dev_frozen
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
        state['_triggered'] = set()
        state['_exec_order'] = None
        state['_reset_nodes'] = None
        state['_compiled_forward'] = None
        state['_forward_dispatch'] = None
        state['_innov_cache'] = None
        state['_active_structure_cache'] = None
        state['_values_arr'] = None   # numpy buffer; rebuilt on first forward() in subprocess
        # _dev_added contains Node/Connection references — episode-local, not persisted.
        state['_dev_added'] = []
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Backward compat: old pickles may carry either old or no flag.
        if 'allow_memory' not in state:
            self.allow_memory = state.get('allow_output_memory', True)
        # Drop the obsolete attribute if it leaked in.
        self.__dict__.pop('allow_output_memory', None)
        # Backward compat: old pickles won't have these attributes.
        self.__dict__.setdefault('objectives', None)
        self.__dict__.setdefault('_last_species_id', None)
        self.__dict__.setdefault('_species_stale', True)
        self.__dict__.setdefault('_values_arr', None)
        self.__dict__.setdefault('_active_structure_cache', None)
        self.__dict__.setdefault('_darts_gates', None)
        self.__dict__.setdefault('weight_groups', {})
        self.__dict__.setdefault('_weight_group_members', {})
        self.__dict__.setdefault('grouper', None)
        self.__dict__.setdefault('out_grouper', None)
        self.__dict__.setdefault('conv_stack', None)
        self.__dict__.setdefault('attention_block', None)
        self.__dict__.setdefault('dev_rules', [])
        self.__dict__.setdefault('_dev_added', [])
        self.__dict__.setdefault('_dev_frozen', False)
        if 'mutation_gate_source' not in state:
            m = Mutation()
            m.bool_rate = 0.05
            self.mutation_gate_source = m

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

    # ── Shared Weights ────────────────────────────────────────────────────────

    def set_weight_group(self, conn: "Connection", group_id: str) -> None:
        """Assign *conn* to a shared weight group.

        All connections in the same group share one weight value stored in
        ``genome.weight_groups[group_id]``.  The group is initialised to the
        connection's current weight the first time it is created.
        """
        if group_id not in self.weight_groups:
            self.weight_groups[group_id] = conn.weight
        conn.weight_group = group_id
        # Invalidate member cache so it is rebuilt on next sync.
        self._weight_group_members.pop(group_id, None)

    def sync_shared_weights(self) -> None:
        """Propagate each group's weight value to all its member connections."""
        if not self.weight_groups:
            return
        # Rebuild member cache lazily.
        for gid in list(self.weight_groups):
            if gid not in self._weight_group_members:
                members = [
                    conn
                    for node in self.nodes
                    for conn in node.connections
                    if getattr(conn, 'weight_group', None) == gid
                ]
                self._weight_group_members[gid] = members
            w = self.weight_groups[gid]
            for conn in self._weight_group_members[gid]:
                conn.weight = w

    def get_lamarck_connections(self) -> list:
        """Return enabled connections deduplicated by shared weight group.

        For grouped connections, only the first representative of each group is
        included so Lamarck refinement updates the group value once rather than
        having the last connection's delta silently overwrite earlier deltas.
        """
        conns = []
        seen_groups: set[str] = set()
        for node in self.nodes:
            for conn in node.connections:
                if not conn.enabled:
                    continue
                gid = getattr(conn, 'weight_group', None)
                if gid is not None:
                    if gid in seen_groups:
                        continue
                    seen_groups.add(gid)
                conns.append(conn)
        return conns

    def _sync_groups_from_reps(self, rep_conns: list) -> None:
        """Update group weights from representative connections, then sync all members.

        Call this after perturbing/reverting representative connection weights in
        Lamarck refinement so all group members stay in sync.
        """
        if not self.weight_groups:
            return
        for conn in rep_conns:
            gid = getattr(conn, 'weight_group', None)
            if gid is not None:
                self.weight_groups[gid] = conn.weight
        self.sync_shared_weights()

    # ── DARTS-Lite ────────────────────────────────────────────────────────────

    def update_darts_gates(self) -> None:
        """Recompute gate values from |weight| via sigmoid for all connections.

        gate = sigmoid(|weight| * 2).  Initialises _darts_gates on first call.
        """
        import math
        if self._darts_gates is None:
            self._darts_gates = {}
        gates = self._darts_gates
        for node in self.nodes:
            for conn in node.connections:
                if conn.innovation >= 0:
                    gates[conn.innovation] = 1.0 / (1.0 + math.exp(-abs(conn.weight) * 2.0))

    def prune_darts_connections(self, threshold: float = 0.1) -> int:
        """Remove connections whose DARTS gate is below *threshold*.

        Returns the number of connections removed.
        """
        if self._darts_gates is None:
            return 0
        removed = 0
        gates = self._darts_gates
        for node in list(self.nodes):
            to_remove = [
                conn for conn in node.connections
                if conn.innovation >= 0 and gates.get(conn.innovation, 1.0) < threshold
            ]
            for conn in to_remove:
                node.connections.remove(conn)
                removed += 1
        if removed:
            self._invalidate_topology()
        return removed

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
        self._active_structure_cache = None
        self._species_stale = True  # topology changed → needs species re-assignment
        self._last_species_id = None  # old species ID is now stale too
        self._values_arr = None     # will be reallocated in next _compile_forward()
        # Disconnect all connections from their weight arrays so stale array
        # references don't survive recompilation.
        for node in self.nodes:
            for conn in node.connections:
                conn._weight_arr = None

    @property
    def connection_count(self) -> int:
        return self._connection_count

    def active_structure_info(self) -> dict:
        """Return structural activity counts for enabled input->output paths.

        A hidden node is active when it is reachable from an input and can reach
        an output through enabled connections.  A connection is active when it is
        enabled and both endpoints lie on such a path.
        """
        if self._active_structure_cache is not None:
            return self._active_structure_cache

        enabled_edges = [
            (src, conn.target, conn)
            for src in self.nodes
            for conn in src.connections
            if conn.enabled
        ]
        forward: dict[Node, list[Node]] = {}
        reverse: dict[Node, list[Node]] = {}
        for src, tgt, _conn in enabled_edges:
            forward.setdefault(src, []).append(tgt)
            reverse.setdefault(tgt, []).append(src)

        reachable: set[Node] = set()
        stack = list(self.input_nodes)
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(forward.get(node, ()))

        can_reach_output: set[Node] = set()
        stack = list(self.output_nodes)
        while stack:
            node = stack.pop()
            if node in can_reach_output:
                continue
            can_reach_output.add(node)
            stack.extend(reverse.get(node, ()))

        active_nodes = reachable & can_reach_output
        active_connections = sum(
            1
            for src, tgt, _conn in enabled_edges
            if src in active_nodes and tgt in active_nodes
        )
        hidden_nodes = [n for n in self.nodes if n.type is NodeType.HIDDEN]
        active_hidden = sum(1 for n in hidden_nodes if n in active_nodes)
        enabled_connections = len(enabled_edges)
        result = {
            "active_nodes": len(active_nodes),
            "active_hidden_nodes": active_hidden,
            "inactive_hidden_nodes": len(hidden_nodes) - active_hidden,
            "enabled_connections": enabled_connections,
            "active_connections": active_connections,
            "inactive_connections": self.connection_count - active_connections,
            "inactive_enabled_connections": enabled_connections - active_connections,
        }
        self._active_structure_cache = result
        return result

    def memory_info(self) -> dict:
        """Returns a breakdown of node/connection counts for memory profiling."""
        active = self.active_structure_info()
        return {
            "nodes": len(self.nodes),
            "connections": self.connection_count,
            "input_nodes": len(self.input_nodes),
            "output_nodes": len(self.output_nodes),
            "max_nodes": self.max_nodes,
            "max_connections": self.max_connections,
            **active,
        }

    def sensitivity_analysis(
        self,
        test_cases: "list[tuple[list[float], list[float]]]",
        delta: float = 0.1,
    ) -> "list[float]":
        """Per-input influence score via symmetric ±delta perturbation.

        For each input *i* and each test case, the output vector is evaluated
        with input *i* raised by *delta* and lowered by *delta*.  The mean
        absolute output change (averaged over outputs and test cases) is the
        influence score for input *i*.

        Args:
            test_cases: List of ``(inputs, expected)`` pairs.  Only the inputs
                are used; expected values are ignored.
            delta: Perturbation magnitude (default 0.1).

        Returns:
            List of ``n_inputs`` floats ≥ 0.  Higher = more influential.
        """
        n_inputs = len(self.input_nodes)
        if not test_cases or n_inputs == 0:
            return [0.0] * n_inputs

        n_outputs = len(self.output_nodes)
        denom = max(1, n_outputs) * max(1, len(test_cases)) * 2.0 * delta
        scores = [0.0] * n_inputs

        for inputs, _ in test_cases:
            base = list(inputs)
            for i in range(n_inputs):
                perturbed = list(base)

                perturbed[i] = base[i] + delta
                self.reset()
                out_plus = self._bfs_forward(perturbed)

                perturbed[i] = base[i] - delta
                self.reset()
                out_minus = self._bfs_forward(perturbed)

                scores[i] += sum(
                    abs(op - om) for op, om in zip(out_plus, out_minus)
                )

        return [s / denom for s in scores]

    def dead_nodes(
        self,
        test_cases: "list[tuple[list[float], list[float]]]",
        threshold: float = 1e-6,
    ) -> "set[int]":
        """Return innovation IDs of hidden nodes that never activate.

        A hidden node is considered dead if |activation| < *threshold* for
        every test case.

        Args:
            test_cases: List of ``(inputs, expected)`` pairs.
            threshold: Activation magnitude below which a node is "dead"
                (default 1e-6).

        Returns:
            Set of innovation IDs (integers) of dead hidden nodes.
        """
        hidden = [n for n in self.nodes if n.type is NodeType.HIDDEN]
        if not hidden or not test_cases:
            return set()

        dead = {n.innovation for n in hidden}
        for inputs, _ in test_cases:
            self.reset()
            self._bfs_forward(list(inputs))
            alive = {n.innovation for n in hidden if abs(n.value) >= threshold}
            dead -= alive
            if not dead:
                break

        return dead

    # -------------------------------------------------------------------------
    # Pruning & compression
    # -------------------------------------------------------------------------

    def prune(self, threshold: float = 0.01, method: str = "weight") -> int:
        """Remove connections whose absolute weight is below *threshold*.

        Args:
            threshold: Minimum absolute weight to keep a connection.
            method: ``"weight"`` (remove by weight magnitude) or
                    ``"activation_frequency"`` (not yet implemented).

        Returns:
            Number of connections removed.
        """
        if method == "activation_frequency":
            raise NotImplementedError("prune method 'activation_frequency' is not yet implemented")
        if method != "weight":
            raise ValueError(f"Unknown prune method: {method!r}")
        total_before = sum(
            1 for src in self.nodes for c in src.connections if c.enabled
        )
        removed = 0
        for src in list(self.nodes):
            to_remove = []
            for conn in src.connections:
                if conn.enabled and abs(conn.weight) < threshold:
                    to_remove.append(conn)
            for conn in to_remove:
                src.connections.remove(conn)
                removed += 1
        if removed:
            self._invalidate_topology()
        total_after = total_before - removed
        comp_rate = removed / total_before if total_before > 0 else 0.0
        self._prune_stats = {
            "connections_removed": removed,
            "nodes_removed": 0,
            "fitness_delta": 0.0,
            "compression_rate": comp_rate,
            "rolled_back": False,
        }
        return removed

    def compress(self, target_size: int) -> int:
        """Remove the smallest-weight connections until *target_size* remains.

        Args:
            target_size: Desired number of connections.

        Returns:
            Total connections removed.
        """
        all_conns = [
            (abs(conn.weight), src, conn)
            for src in self.nodes
            for conn in src.connections
            if conn.enabled
        ]
        total_before = len(all_conns)
        n_remove = total_before - target_size
        if n_remove <= 0:
            return 0
        all_conns.sort(key=lambda t: t[0])
        for _, src, conn in all_conns[:n_remove]:
            src.connections.remove(conn)
        self._invalidate_topology()
        comp_rate = n_remove / total_before if total_before > 0 else 0.0
        self._prune_stats = {
            "connections_removed": n_remove,
            "nodes_removed": 0,
            "fitness_delta": 0.0,
            "compression_rate": comp_rate,
            "rolled_back": False,
        }
        return n_remove

    def prune_stats(self) -> dict:
        """Return pruning statistics from the last ``prune()`` or ``compress()`` call.

        Returns:
            Dict with keys: ``connections_removed``, ``nodes_removed``,
            ``fitness_delta``, ``compression_rate``, ``rolled_back``.
            All values are zero / False before any pruning has occurred.
        """
        return dict(getattr(self, '_prune_stats', {
            "connections_removed": 0,
            "nodes_removed": 0,
            "fitness_delta": 0.0,
            "compression_rate": 0.0,
            "rolled_back": False,
        }))

    def lineage(self, tracker=None) -> list[int]:
        """Return the ancestor chain (oldest first) for this genome.

        Requires an ``InnovationTracker`` with lineage recording enabled.
        If *tracker* is None, returns the direct parent IDs.
        """
        if tracker is not None:
            return tracker.get_ancestors(self._genome_id)
        return list(self._parent_ids)

    # -------------------------------------------------------------------------
    # Sparse NEAT / Lottery Ticket
    # -------------------------------------------------------------------------

    def find_lottery_ticket(
        self,
        fitness_fn,
        target_sparsity: float = 0.5,
        max_fitness_drop: float = 0.05,
        iterations: int = 5,
        lamarck_steps: int = 0,
        lamarck_sigma: float = 0.1,
    ):
        """Find the sparse lottery ticket via Iterative Magnitude Pruning.

        Delegates to :func:`yane.evolution.sparse_neat.find_lottery_ticket`.
        The genome is temporarily modified and then fully restored.

        Parameters
        ----------
        fitness_fn:
            ``(genome) -> float`` fitness function.
        target_sparsity:
            Target fraction of connections to prune (0–1).
        max_fitness_drop:
            Maximum allowed absolute fitness drop from original.
        iterations:
            Number of IMP rounds.
        lamarck_steps:
            Hill-climbing fine-tuning steps per IMP round (0 = disabled).
        lamarck_sigma:
            Step size for Lamarckian refinement.

        Returns
        -------
        LotteryTicket
        """
        from yane.evolution.sparse_neat import find_lottery_ticket as _flt
        return _flt(
            self,
            fitness_fn,
            target_sparsity=target_sparsity,
            max_fitness_drop=max_fitness_drop,
            iterations=iterations,
            lamarck_steps=lamarck_steps,
            lamarck_sigma=lamarck_sigma,
        )

    def apply_ticket(self, ticket) -> None:
        """Apply a lottery ticket, disabling connections not in its mask.

        Delegates to :func:`yane.evolution.sparse_neat.apply_ticket`.

        Parameters
        ----------
        ticket:
            :class:`~yane.evolution.sparse_neat.LotteryTicket` from
            :meth:`find_lottery_ticket`.
        """
        from yane.evolution.sparse_neat import apply_ticket as _at
        _at(self, ticket)

    # -------------------------------------------------------------------------
    # Symbolic Regression Export
    # -------------------------------------------------------------------------

    def to_symbolic(
        self,
        input_names: "list[str] | None" = None,
        format: str = "python",
        fold_constants: bool = True,
    ) -> str:
        """Export genome as a closed-form symbolic expression.

        Only acyclic genomes are supported.  Cyclic genomes (memory nodes)
        cannot be represented as a closed-form mathematical expression.

        Parameters
        ----------
        input_names:
            Names for each input variable.  Defaults to ``["x0", "x1", ...]``.
        format:
            ``"python"``  — Python expression string (evaluable with eval).
            ``"text"``    — human-readable infix notation.
            ``"latex"``   — LaTeX math string (for use with \\( … \\)).
            ``"sympy"``   — sympy-parseable string (same as "python" format).
        fold_constants:
            Whether to simplify: remove zero-weight terms, collapse
            ``1.0 * x`` → ``x``, and ``0.0 * x`` → ``0.0``.

        Returns
        -------
        str
            Symbolic representation of the genome's forward function.

        Raises
        ------
        ValueError
            If the genome contains cycles (cannot be represented in closed form).
        """
        from yane.core._symbolic import genome_to_symbolic
        return genome_to_symbolic(
            self,
            input_names=input_names,
            fmt=format,
            fold_constants=fold_constants,
        )
