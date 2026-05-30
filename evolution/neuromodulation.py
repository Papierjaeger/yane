"""Neuromodulation für YANE — kontextabhängige Gewichtung ganzer Verbindungsgruppen.

Ein MODULATOR-Knoten (``node.is_modulator = True``) skaliert die eingehenden
Verbindungen seiner Zielknoten proportional zu seiner eigenen Aktivierung.

**Architektur:**
- MODULATOR-Knoten berechnen ihre Aktivierung wie normale HIDDEN-Knoten.
- Ihr Ausgabewert ``node.value`` wird als multiplikativer Gain für die
  **eingehenden** Verbindungen aller Knoten gesetzt, mit denen sie
  verbunden sind.
- Effekt ist **one-step-delayed**: der Gain aus Pass T wirkt auf Pass T+1.
  (Eine True-same-step-Modulation würde zwei Forward-Pässe erfordern.)

**Evolvierbarkeit:**
- NEAT-Mutation kann Knoten zu MODULATORen machen (``is_modulator = True``).
- Welche Knoten beeinflusst werden: durch Standard-NEAT-Verbindungen von
  MODULATOR → Zielknoten.
- Stärke: ``modulator.value`` (0 ≈ voll moduliert, 1.0 = neutral, >1 = Verstärkung).

**Zero-Cost wenn deaktiviert:**
``node.is_modulator = False`` (Default) → kein Overhead.

Integration::

    yane.set_neuromodulation(enabled=True)
    # Optional: Knoten manuell als Modulator markieren (für Tests)
    from yane.evolution.neuromodulation import make_node_modulator
    make_node_modulator(genome, node_idx=2)
    yane.train(fitness_fn)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome

_MODULATION_GAIN_MIN = 0.0
_MODULATION_GAIN_MAX = 2.0


# ---------------------------------------------------------------------------
# Core modulation
# ---------------------------------------------------------------------------

def apply_modulation_to_weights(genome: "Genome") -> None:
    """Apply stored modulation gains to connection working weights.

    For each non-modulator node that has a ``modulation_gain != 1.0``,
    multiply its incoming connection weights by that gain.  The base weight
    is preserved in ``conn._base_weight``; the working weight is in
    ``conn.weight``.

    Called **before** ``genome.forward()`` so the modulated weights are in
    effect for the current timestep.
    """
    for src in genome.nodes:
        for conn in src.connections:
            if not conn.enabled:
                continue
            tgt = conn.target
            if tgt.modulation_gain == 1.0:
                continue  # no modulation — skip (zero cost)
            if conn._base_weight is None:
                conn._base_weight = conn.weight
            conn.weight = max(
                -_MODULATION_GAIN_MAX * abs(conn._base_weight),
                min(
                    _MODULATION_GAIN_MAX * abs(conn._base_weight),
                    conn._base_weight * tgt.modulation_gain,
                ),
            )


def update_modulation_gains(genome: "Genome") -> None:
    """Read MODULATOR activations and update target nodes' ``modulation_gain``.

    Called **after** ``genome.forward()`` so MODULATOR node values reflect
    the latest computation.  The gains are clamped to
    ``[_MODULATION_GAIN_MIN, _MODULATION_GAIN_MAX]``.
    """
    for src in genome.nodes:
        if not src.is_modulator:
            continue
        gain = max(_MODULATION_GAIN_MIN, min(_MODULATION_GAIN_MAX, float(src.value)))
        for conn in src.connections:
            if conn.enabled and conn.target is not src:
                conn.target.modulation_gain = gain


def restore_modulation_weights(genome: "Genome") -> None:
    """Restore all modulated connection weights to their base values.

    Called on ``genome.reset()`` to ensure evolved weights are not permanently
    distorted by modulation across episodes.
    """
    for src in genome.nodes:
        for conn in src.connections:
            if conn._base_weight is not None:
                conn.weight = conn._base_weight
    # Reset all modulation gains to neutral
    for node in genome.nodes:
        node.modulation_gain = 1.0


def genome_has_modulators(genome: "Genome") -> bool:
    """Return True when *genome* has at least one MODULATOR node."""
    return any(n.is_modulator for n in genome.nodes)


# ---------------------------------------------------------------------------
# Helpers for setup / testing
# ---------------------------------------------------------------------------

def make_node_modulator(genome: "Genome", node_idx: int) -> None:
    """Mark ``genome.nodes[node_idx]`` as a MODULATOR node."""
    genome.nodes[node_idx].is_modulator = True


def mutate_modulator_flags(
    genome: "Genome",
    add_prob: float = 0.02,
    remove_prob: float = 0.01,
) -> None:
    """Randomly promote hidden nodes to MODULATOR or demote existing ones.

    Called by ``NeuroEvolution`` during NEAT mutation when neuromodulation is
    active.

    Parameters
    ----------
    add_prob :
        Probability of promoting a non-modulator HIDDEN node to MODULATOR.
    remove_prob :
        Probability of demoting a MODULATOR to a normal HIDDEN node.
    """
    import random
    for node in genome.nodes:
        if node.type == 'hidden' or (
            hasattr(node, 'type') and
            str(node.type).lower() in ('hidden', 'nodetype.hidden')
        ):
            if node.is_modulator:
                if random.random() < remove_prob:
                    node.is_modulator = False
            else:
                if random.random() < add_prob:
                    node.is_modulator = True
