"""Developmental NEAT — Ontogenese während der Evaluation.

Genome können ihre Topologie *während einer Episode* dynamisch anpassen:
Entwicklungsregeln werden nach jedem ``developmental_forward()``-Aufruf
ausgewertet und fügen bei Bedarf neue Verbindungen hinzu.

**Architektur:**
- ``DevelopmentalRule(trigger_condition, action)``: feuert wenn
  ``trigger_condition(genome)`` True ist → ``action(genome)`` modifiziert
  die Topologie (z.B. neue Verbindung).
- ``genome.dev_rules: list[DevelopmentalRule]`` — evolvierable Regeln.
- ``genome.developmental_forward(inputs)``: standard-forward + Regelauswertung.
- ``genome.reset()``: stellt die Basis-Topologie wieder her (entfernt
  episoden-lokal hinzugefügte Verbindungen).
- ``genome.freeze_development()``: deaktiviert alle Regeln temporär.

**Episodenlokalität:**
Hinzugefügte Verbindungen werden in ``genome._dev_added`` getrackt.
``reset()`` räumt sie auf → Basisgenotyp bleibt unverändert.

**Zero-Cost wenn deaktiviert:**
``genome.dev_rules = []`` (Default) → kein Overhead in ``forward()``.

Integration::

    from yane.evolution.developmental import DevelopmentalRule, make_threshold_rule

    rule = make_threshold_rule(
        trigger_node_idx=2, threshold=0.5,
        src_idx=0, tgt_idx=3, weight=0.3,
    )
    genome.dev_rules = [rule]
    genome.developmental_forward([0.5, 0.5])  # rule may fire
    genome.reset()                             # topology restored
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome
    from yane.core.node import Node


# ---------------------------------------------------------------------------
# DevelopmentalRule
# ---------------------------------------------------------------------------

@dataclass
class DevelopmentalRule:
    """One developmental rule: trigger → action.

    Parameters
    ----------
    trigger_condition :
        Callable ``(genome) -> bool``.  If *True*, the action fires.
    action :
        Callable ``(genome) -> None``.  Modifies the genome (typically
        adds a connection).  Called when *trigger_condition* returns True.
    name :
        Optional human-readable identifier.
    max_fires :
        Maximum number of times this rule can fire per episode (0 = unlimited).
    _fires_this_episode :
        Internal counter — reset by ``genome.reset()``.
    """

    trigger_condition: Callable[["Genome"], bool]
    action: Callable[["Genome"], None]
    name: str = "rule"
    max_fires: int = 0  # 0 = unlimited
    _fires_this_episode: int = field(default=0, repr=False, compare=False)

    def copy(self) -> "DevelopmentalRule":
        """Shallow copy (callables are shared — they're stateless)."""
        return DevelopmentalRule(
            trigger_condition=self.trigger_condition,
            action=self.action,
            name=self.name,
            max_fires=self.max_fires,
        )

    def mutate(
        self,
        weight_sigma: float = 0.05,
        threshold_sigma: float = 0.05,
        rng: random.Random | None = None,
    ) -> None:
        """Mutate the rule's parameters if it is a ``ParametricRule``."""
        if isinstance(self, ParametricRule):
            _rng = rng or random
            self.weight += _rng.gauss(0.0, weight_sigma)
            self.threshold = max(0.0, min(1.0,
                self.threshold + _rng.gauss(0.0, threshold_sigma)))

    def reset_episode(self) -> None:
        """Reset per-episode fire counter."""
        self._fires_this_episode = 0

    def should_fire(self, genome: "Genome") -> bool:
        """Evaluate trigger condition and fire limit."""
        if self.max_fires > 0 and self._fires_this_episode >= self.max_fires:
            return False
        try:
            return bool(self.trigger_condition(genome))
        except Exception:
            return False

    def fire(self, genome: "Genome") -> None:
        """Execute action and increment fire counter."""
        try:
            self.action(genome)
            self._fires_this_episode += 1
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ParametricRule — mutable threshold + weight parameters
# ---------------------------------------------------------------------------

class ParametricRule(DevelopmentalRule):
    """A developmental rule whose trigger and action share evolvable parameters.

    This concrete subclass is what NEAT evolution mutates and inherits.

    Parameters
    ----------
    trigger_node_idx :
        Index into ``genome.nodes`` of the node to monitor.
    threshold :
        Activation threshold for the trigger.
    trigger_mode :
        ``"above"`` — fires when ``node.value >= threshold``.
        ``"below"`` — fires when ``node.value <= threshold``.
    src_idx :
        Index of the connection source node (into ``genome.nodes``).
    tgt_idx :
        Index of the connection target node (into ``genome.nodes``).
    weight :
        Weight of the connection added by the action.
    """

    def __init__(
        self,
        trigger_node_idx: int = 0,
        threshold: float = 0.5,
        trigger_mode: str = "above",
        src_idx: int = 0,
        tgt_idx: int = 1,
        weight: float = 0.1,
        max_fires: int = 1,
    ) -> None:
        self.trigger_node_idx = trigger_node_idx
        self.threshold = threshold
        self.trigger_mode = trigger_mode
        self.src_idx = src_idx
        self.tgt_idx = tgt_idx
        self.weight = weight
        self.max_fires = max_fires
        self._fires_this_episode = 0
        self.name = f"param_rule({trigger_node_idx},{threshold:.2f},{src_idx}→{tgt_idx})"

        super().__init__(
            trigger_condition=self._check_trigger,
            action=self._add_connection,
            name=self.name,
            max_fires=max_fires,
        )

    def _check_trigger(self, genome: "Genome") -> bool:
        idx = self.trigger_node_idx % max(1, len(genome.nodes))
        val = genome.nodes[idx].value
        if self.trigger_mode == "above":
            return val >= self.threshold
        return val <= self.threshold

    def _add_connection(self, genome: "Genome") -> None:
        from yane.core.connection import Connection
        n_nodes = len(genome.nodes)
        if n_nodes < 2:
            return
        src = genome.nodes[self.src_idx % n_nodes]
        tgt = genome.nodes[self.tgt_idx % n_nodes]
        if src is tgt or tgt in genome.input_nodes:
            return
        # Use a negative innovation so it never conflicts with NEAT-tracked ones
        innov = -(len(genome.nodes) * 1000 + self.src_idx * 100 + self.tgt_idx + 1)
        conn = Connection(tgt, innovation=innov)
        conn.weight = self.weight
        src.connections.append(conn)
        genome._invalidate_topology()
        genome._dev_added.append((src, conn))

    def copy(self) -> "ParametricRule":
        return ParametricRule(
            trigger_node_idx=self.trigger_node_idx,
            threshold=self.threshold,
            trigger_mode=self.trigger_mode,
            src_idx=self.src_idx,
            tgt_idx=self.tgt_idx,
            weight=self.weight,
            max_fires=self.max_fires,
        )

    def mutate(  # type: ignore[override]
        self,
        weight_sigma: float = 0.05,
        threshold_sigma: float = 0.05,
        rng: random.Random | None = None,
    ) -> None:
        _rng = rng or random
        self.weight += _rng.gauss(0.0, weight_sigma)
        self.threshold = max(0.0, min(1.0,
            self.threshold + _rng.gauss(0.0, threshold_sigma)))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_threshold_rule(
    trigger_node_idx: int = 0,
    threshold: float = 0.5,
    trigger_mode: str = "above",
    src_idx: int = 0,
    tgt_idx: int = 1,
    weight: float = 0.1,
    max_fires: int = 1,
) -> ParametricRule:
    """Create a :class:`ParametricRule` with sensible defaults."""
    return ParametricRule(
        trigger_node_idx=trigger_node_idx,
        threshold=threshold,
        trigger_mode=trigger_mode,
        src_idx=src_idx,
        tgt_idx=tgt_idx,
        weight=weight,
        max_fires=max_fires,
    )


# ---------------------------------------------------------------------------
# Developmental forward + episode management (module-level functions for Genome)
# ---------------------------------------------------------------------------

def developmental_forward(
    genome: "Genome",
    inputs: list[float],
) -> list[float]:
    """Run standard forward then evaluate developmental rules.

    After ``genome.forward(inputs)``, each enabled rule is checked.
    Triggered rules add connections, which take effect on the *next* call.

    Parameters
    ----------
    genome :
        The genome to evaluate.
    inputs :
        Raw input vector.

    Returns
    -------
    list[float]
        Network outputs (same as ``genome.forward(inputs)``).
    """
    # Run normal forward pass
    genome.reset()
    result = genome.forward(inputs)

    # Skip rule evaluation if frozen or no rules
    if getattr(genome, "_dev_frozen", False):
        return result
    rules = getattr(genome, "dev_rules", [])
    if not rules:
        return result

    # Evaluate rules in order
    for rule in rules:
        if rule.should_fire(genome):
            rule.fire(genome)

    return result


def reset_developmental(genome: "Genome") -> None:
    """Remove all ephemerally added connections and restore base topology.

    Called automatically by ``genome.reset()`` when developmental rules are
    present.  Also resets per-episode fire counters on all rules.
    """
    dev_added = getattr(genome, "_dev_added", [])
    if dev_added:
        for src_node, conn in dev_added:
            try:
                src_node.connections.remove(conn)
            except ValueError:
                pass
        dev_added.clear()
        genome._invalidate_topology()

    for rule in getattr(genome, "dev_rules", []):
        rule.reset_episode()


def mutate_rules(
    genome: "Genome",
    weight_sigma: float = 0.05,
    threshold_sigma: float = 0.05,
    rng: random.Random | None = None,
) -> None:
    """Perturb all evolvable rule parameters in *genome*."""
    for rule in getattr(genome, "dev_rules", []):
        rule.mutate(weight_sigma=weight_sigma, threshold_sigma=threshold_sigma, rng=rng)
