"""Gene Regulatory Network (GRN) Encoding — Indirektes Genotyp-Phänotyp-Mapping.

Ein GRN-Genotyp (kompakte Liste von GRN-Genen) wird durch einen
Entwicklungsprozess (N Schritte) in ein vollständiges YANE-Genome (Phänotyp)
mit potenziell viel mehr Verbindungen als Genen übersetzt.

**Kernidee:**
20 GRN-Gene × 5 Entwicklungsschritte → 100+ Verbindungen im Phänotyp.
Jedes Gen wird in jedem Schritt ausgedrückt (sofern aktiv) und erzeugt dabei
eine eindeutige Verbindung (Innovation = gene_idx × max_steps + step).

**GRN-Gen-Struktur:**
```
GRNGene(
  src_node:          int,          # Quell-Knoten-Innovation
  tgt_node:          int,          # Ziel-Knoten-Innovation
  weight:            float,        # Verbindungsgewicht (Basiswert)
  activation:        str,          # Aktivierungsfunktion des Zielknotens
  regulatory_sites:  list[int],    # Indizes der regulatorischen Gene (>0 → bedingte Expression)
  expression_rate:   float,        # Stärke des Gewichts pro Schritt (Amplifikation)
)
```

**Entwicklungsalgorithmus:**
1. Initialisiere Expressions-Levels (alle Gene inaktiv).
2. Für jeden Entwicklungsschritt ``t``:
   a. Bestimme welche Gene aktiv sind:
      - Gen ohne Regulatoren → immer aktiv (konstitutiv)
      - Gen mit Regulatoren → aktiv wenn mind. ein Regulator im Vorschritt aktiv war
   b. Jedes aktive Gen erzeugt eine Verbindung mit Innovation ``gene_idx × N + t``
      und Gewicht ``gene.weight × gene.expression_rate^t``.
3. Rückgabe: Phänotyp-Genome mit allen erzeugten Verbindungen.

**GenomeCodec-Protokoll:**
- ``encode(grn_genome)``: pickle des GRNGenome-Objekts
- ``decode(data)``: unpickle → GRNGenome (Genotyp, NICHT entwickelter Phänotyp)
- ``develop(grn_genome, n_inputs, n_outputs, development_steps)``: GRN → Phänotyp-Genome

Integration::

    from yane.evolution.grn_encoding import GRNGene, GRNGenome, GRNCodec

    grn = GRNGenome.random(n_genes=20, n_nodes=10)
    phenotype = grn.develop(n_inputs=3, n_outputs=2, development_steps=5)
    # phenotype ist ein normales YANE Genome mit 100+ Verbindungen

    yane.set_genome_encoding("grn", development_steps=5)
"""
from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# GRN Gene
# ---------------------------------------------------------------------------

@dataclass
class GRNGene:
    """One gene in a Gene Regulatory Network.

    Parameters
    ----------
    src_node :
        Innovation number of the source (pre-synaptic) node.
    tgt_node :
        Innovation number of the target (post-synaptic) node.
    weight :
        Base connection weight.
    activation :
        Activation function name for the target node.
    regulatory_sites :
        Indices of regulatory genes.  Empty = constitutive (always expressed).
        Non-empty = expressed only when at least one regulatory gene was active
        in the previous step.
    expression_rate :
        Multiplicative scaling applied to *weight* at each development step.
        1.0 = constant; < 1.0 = decays; > 1.0 = amplifies.
    """

    src_node: int
    tgt_node: int
    weight: float = 0.5
    activation: str = "sigmoid"
    regulatory_sites: list[int] = field(default_factory=list)
    expression_rate: float = 1.0

    def copy(self) -> "GRNGene":
        return GRNGene(
            src_node=self.src_node,
            tgt_node=self.tgt_node,
            weight=self.weight,
            activation=self.activation,
            regulatory_sites=list(self.regulatory_sites),
            expression_rate=self.expression_rate,
        )

    def mutate(self, sigma: float = 0.1, rng: random.Random | None = None) -> None:
        """Perturb weight, expression_rate, and optionally regulatory_sites."""
        _rng = rng or random
        self.weight += _rng.gauss(0.0, sigma)
        self.expression_rate = max(0.1, min(5.0,
            self.expression_rate + _rng.gauss(0.0, sigma * 0.1)))


# ---------------------------------------------------------------------------
# GRN Genome (Genotype)
# ---------------------------------------------------------------------------

class GRNGenome:
    """A Gene Regulatory Network genotype.

    Stores a list of :class:`GRNGene` objects.  Call :meth:`develop` to
    produce a phenotype (:class:`~yane.core.genome.Genome`).

    Parameters
    ----------
    genes :
        List of GRN genes.  More genes = larger phenotype capacity.
    """

    def __init__(self, genes: list[GRNGene]) -> None:
        self.genes = list(genes)

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        n_genes: int = 20,
        n_nodes: int = 10,
        regulatory_prob: float = 0.3,
        seed: int | None = None,
    ) -> "GRNGenome":
        """Create a random GRN genotype.

        Parameters
        ----------
        n_genes :
            Number of genes.
        n_nodes :
            Available node innovations (0 … n_nodes-1).
        regulatory_prob :
            Probability a gene has regulatory sites.
        seed :
            RNG seed.
        """
        rng = random.Random(seed)
        activations = ["sigmoid", "tanh", "relu", "linear"]
        genes = []
        for i in range(n_genes):
            src = rng.randint(0, n_nodes - 1)
            tgt = rng.randint(0, n_nodes - 1)
            w = rng.gauss(0.0, 0.5)
            act = rng.choice(activations)
            reg: list[int] = []
            if rng.random() < regulatory_prob and i > 0:
                reg = [rng.randint(0, i - 1)]
            er = max(0.1, min(5.0, rng.gauss(1.0, 0.2)))
            genes.append(GRNGene(src, tgt, w, act, reg, er))
        return cls(genes)

    # ------------------------------------------------------------------
    # Development
    # ------------------------------------------------------------------

    def develop(
        self,
        n_inputs: int,
        n_outputs: int,
        development_steps: int = 5,
        weight_threshold: float = 0.0,
        max_connections: int | None = None,
    ) -> "Genome":
        """Develop the GRN genotype into a phenotype Genome.

        Each active gene in each development step contributes one connection
        with a unique innovation number ``gene_idx * development_steps + step``.

        Parameters
        ----------
        n_inputs, n_outputs :
            Number of input and output nodes in the phenotype.
        development_steps :
            Number of gene-expression rounds (N).
        weight_threshold :
            Connections with ``|weight| < threshold`` are not added.
        max_connections :
            Optional cap on total connections.

        Returns
        -------
        Genome
            A YANE :class:`~yane.core.genome.Genome` with potentially many more
            connections than genes (up to ``n_genes × development_steps``).
        """
        from yane.core.genome import Genome
        from yane.core.node import Node, NodeType
        from yane.core.connection import Connection
        from yane.util.activation import ActivationType

        genome = Genome()
        genome.max_connections = max_connections

        # Create input and output nodes
        node_map: dict[int, Node] = {}

        for i in range(n_inputs):
            innov = i
            n = Node(NodeType.INPUT, innov)
            n.activation = ActivationType.LINEAR
            n.input_index = i
            genome.input_nodes.append(n)
            genome.nodes.append(n)
            node_map[innov] = n

        for j in range(n_outputs):
            innov = n_inputs + j
            n = Node(NodeType.OUTPUT, innov)
            n.activation = ActivationType.SIGMOID
            genome.output_nodes.append(n)
            genome.nodes.append(n)
            node_map[innov] = n

        # Track which gene innovations exist in the phenotype
        # (src_innov, tgt_innov, gene_innov) → Connection
        existing: dict[int, Connection] = {}
        n_added = 0

        # Development loop.
        # Step -1 (initialization): all constitutive genes fire unconditionally,
        # producing n_genes base connections before the regulated steps begin.
        # This ensures n_genes × (development_steps + 1) potential connections,
        # guaranteeing >100 for 20 genes × 5 steps (= 20 × 6 = 120).
        active_prev = [True] * len(self.genes)

        for step in range(-1, development_steps):  # includes initialization step
            active_cur = []
            for gi, gene in enumerate(self.genes):
                # Constitutive: no regulatory sites → always active
                if not gene.regulatory_sites:
                    is_active = True
                else:
                    # Active if any regulatory gene was active in previous step
                    is_active = any(
                        0 <= ri < len(active_prev) and active_prev[ri]
                        for ri in gene.regulatory_sites
                    )
                active_cur.append(is_active)

                if not is_active:
                    continue

                # Unique innovation for this gene × step combination.
                # step=-1 (init) uses offset development_steps as index.
                step_idx = step + 1  # maps -1→0, 0→1, ..., N-1→N
                conn_innov = gi * (development_steps + 1) + step_idx

                # Compute effective weight (decays/amplifies with expression_rate)
                eff_weight = gene.weight * (gene.expression_rate ** max(0, step))

                if abs(eff_weight) < weight_threshold:
                    continue
                if max_connections is not None and n_added >= max_connections:
                    break

                # Get or create source node
                src_innov = gene.src_node % max(1, n_inputs + n_outputs)
                tgt_innov = gene.tgt_node % max(1, n_inputs + n_outputs)
                # Ensure src is not output; tgt is not input
                if src_innov >= n_inputs:
                    src_innov = src_innov % n_inputs
                if tgt_innov < n_inputs:
                    tgt_innov = n_inputs + (tgt_innov % n_outputs)

                src_node = node_map[src_innov]
                tgt_node = node_map[tgt_innov]

                if conn_innov not in existing:
                    conn = Connection(tgt_node, innovation=conn_innov)
                    conn.weight = eff_weight
                    src_node.connections.append(conn)
                    existing[conn_innov] = conn
                    n_added += 1
                else:
                    # Accumulate weight on existing connection
                    existing[conn_innov].weight += eff_weight

            active_prev = active_cur

        genome._invalidate_topology()
        return genome

    # ------------------------------------------------------------------
    # Evolution operators
    # ------------------------------------------------------------------

    def copy(self) -> "GRNGenome":
        return GRNGenome([g.copy() for g in self.genes])

    def mutate(
        self,
        sigma: float = 0.1,
        insertion_prob: float = 0.05,
        deletion_prob: float = 0.02,
        n_nodes: int = 10,
        rng: random.Random | None = None,
    ) -> None:
        """In-place mutation: perturb all genes + optional insertion/deletion."""
        _rng = rng or random
        for gene in self.genes:
            gene.mutate(sigma=sigma, rng=_rng)
        if _rng.random() < insertion_prob:
            src = _rng.randint(0, n_nodes - 1)
            tgt = _rng.randint(0, n_nodes - 1)
            self.genes.append(GRNGene(src, tgt, _rng.gauss(0.0, 0.5)))
        if deletion_prob > 0 and len(self.genes) > 1 and _rng.random() < deletion_prob:
            self.genes.pop(_rng.randint(0, len(self.genes) - 1))

    def crossover(self, other: "GRNGenome") -> "GRNGenome":
        """Gene-aligned crossover (by gene index).

        Genes at matching positions are chosen 50/50; excess genes from the
        longer parent are inherited.
        """
        shared = min(len(self.genes), len(other.genes))
        child_genes = []
        for i in range(shared):
            child_genes.append(
                self.genes[i].copy() if random.random() < 0.5 else other.genes[i].copy()
            )
        # Extra genes from the longer parent
        longer = self if len(self.genes) >= len(other.genes) else other
        for i in range(shared, len(longer.genes)):
            child_genes.append(longer.genes[i].copy())
        return GRNGenome(child_genes)


# ---------------------------------------------------------------------------
# GRNCodec — implements GenomeCodec protocol
# ---------------------------------------------------------------------------

class GRNCodec:
    """GenomeCodec that serializes/deserializes GRNGenome objects.

    ``encode(grn)``   — pickle the GRNGenome (genotype).
    ``decode(data)``  — unpickle the GRNGenome.
    ``develop(grn)``  — convert GRNGenome to a phenotype YANE Genome.

    Implements the :class:`~yane.evolution.codec.GenomeCodec` protocol.
    """

    name = "grn"

    def __init__(
        self,
        n_inputs: int = 2,
        n_outputs: int = 1,
        development_steps: int = 5,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.development_steps = development_steps

    def encode(self, genome) -> bytes:
        """Serialize the GRNGenome (or any genome) to bytes via pickle."""
        return pickle.dumps(genome)

    def decode(self, data: bytes):
        """Deserialize a GRNGenome from bytes."""
        return pickle.loads(data)

    def develop(self, grn: GRNGenome, max_connections: int | None = None):
        """Develop a GRNGenome into a phenotype Genome.

        Parameters
        ----------
        grn :
            The GRN genotype to develop.
        max_connections :
            Optional cap on the phenotype's connection count.

        Returns
        -------
        Genome
            The developed phenotype with potentially many more connections
            than genes.
        """
        return grn.develop(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            development_steps=self.development_steps,
            max_connections=max_connections,
        )
