"""Synaptische Plastizität (STDP / Hebbsches Lernen) für YANE.

Implementiert intra-lifetime Gewichtsanpassung innerhalb einer Episode:
Verbindungen tragen evolvierbare Hebb-Koeffizienten A, B, C, D und passen
ihre Gewichte nach jedem ``genome.forward()``-Aufruf an.

**Hebb-Regel:**
``Δw = A * pre + B * post + C * pre * post + D``

Dabei ist:
- ``pre``  = Aktivierung des Quell-Knotens (``src.value``)
- ``post`` = Aktivierung des Ziel-Knotens (``target.value``)
- A, B, C, D = evolvierbare Koeffizienten pro Verbindung (alle 0.0 by default)

Klassische Spezialfälle:
- Reines Hebb: ``A=0, B=0, C=η, D=0`` → ``Δw = η * pre * post``
- Oja-ähnlich: negativer B-Term stabilisiert Gewichte
- Moduliert: D gibt konstante Drift (kann kombiniert werden)

**Episodenlokalität:**
- Beim Start einer Evaluation: ``init_stdp_base_weights()`` speichert
  ``conn._base_weight = conn.weight`` für alle STDP-fähigen Verbindungen.
- Nach jeder Episode (``genome.reset()`` oder end-of-eval):
  ``restore_stdp_weights()`` setzt ``conn.weight = conn._base_weight``
  zurück. Das Basisgewicht wird durch Evolution angepasst, nicht durch STDP.

**Nullkosten wenn deaktiviert:**
Alle Koeffizienten default auf 0.0 → kein Delta → keine Wirkung.
``_base_weight = None`` zeigt an, dass STDP für diese Verbindung nicht
aktiv ist → kein Overhead in ``apply_stdp_update()``.

Integration::

    yane.set_stdp(enabled=True)
    # Optional: Koeffizienten initialisieren (für Tests / Demo)
    from yane.evolution.stdp import set_hebb_coeffs
    set_hebb_coeffs(genome, c=0.01)
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome

_HEBB_COEFF_MAX = 2.0   # clamp range for evolved hebb coefficients


# ---------------------------------------------------------------------------
# Core STDP update
# ---------------------------------------------------------------------------

def apply_stdp_update(
    genome: "Genome",
    weight_min: float = -5.0,
    weight_max: float = 5.0,
) -> None:
    """Apply one Hebb-rule weight update using current node activations.

    Called **after** ``genome.forward()`` so ``src.value`` and
    ``target.value`` reflect the most recent activation values.

    Only connections with ``_base_weight is not None`` are updated (i.e.,
    those that were initialised for STDP in the current evaluation).

    Parameters
    ----------
    genome :
        Genome whose connections are updated.
    weight_min, weight_max :
        Clamp range for the working weight — prevents runaway plasticity.
    """
    for src in genome.nodes:
        pre = src.value
        for conn in src.connections:
            if not conn.enabled or conn._base_weight is None:
                continue
            tgt = conn.target
            post = tgt.value
            delta = (
                conn.hebb_a * pre
                + conn.hebb_b * post
                + conn.hebb_c * pre * post
                + conn.hebb_d
            )
            conn.weight = max(weight_min, min(weight_max, conn.weight + delta))


def init_stdp_base_weights(genome: "Genome") -> None:
    """Save the base weight for every STDP-capable connection.

    A connection is STDP-capable when at least one of its Hebb coefficients
    is non-zero.  Connections that are already initialised (``_base_weight is
    not None``) are skipped so repeated calls are idempotent.
    """
    for src in genome.nodes:
        for conn in src.connections:
            if conn._base_weight is None and _has_hebb(conn):
                conn._base_weight = conn.weight


def restore_stdp_weights(genome: "Genome") -> None:
    """Restore all STDP connection weights to their base (evolved) values.

    Call at the start of each new episode (wraps ``genome.reset()``) and at
    the end of each evaluation to ensure the evolved base weight is preserved
    for crossover, checkpointing, and fitness evaluation.
    """
    for src in genome.nodes:
        for conn in src.connections:
            if conn._base_weight is not None:
                conn.weight = conn._base_weight


def _has_hebb(conn) -> bool:
    """True when any Hebb coefficient is non-zero."""
    return conn.hebb_a != 0.0 or conn.hebb_b != 0.0 or conn.hebb_c != 0.0 or conn.hebb_d != 0.0


def genome_has_stdp(genome: "Genome") -> bool:
    """Return True when *genome* has at least one STDP-capable connection."""
    return any(
        _has_hebb(conn)
        for src in genome.nodes
        for conn in src.connections
    )


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def set_hebb_coeffs(
    genome: "Genome",
    a: float = 0.0,
    b: float = 0.0,
    c: float = 0.01,
    d: float = 0.0,
    sigma: float = 0.0,
    rng: random.Random | None = None,
) -> None:
    """Set Hebb coefficients on all connections in *genome*.

    Use for testing or as a warm-start for evolution.

    Parameters
    ----------
    genome :
        Target genome.
    a, b, c, d :
        Base coefficient values.
    sigma :
        Gaussian noise added to each coefficient (0 = deterministic).
    rng :
        Optional RNG for reproducibility.
    """
    _rng = rng or random
    for src in genome.nodes:
        for conn in src.connections:
            noise = (lambda: _rng.gauss(0.0, sigma)) if sigma > 0 else (lambda: 0.0)
            conn.hebb_a = max(-_HEBB_COEFF_MAX, min(_HEBB_COEFF_MAX, a + noise()))
            conn.hebb_b = max(-_HEBB_COEFF_MAX, min(_HEBB_COEFF_MAX, b + noise()))
            conn.hebb_c = max(-_HEBB_COEFF_MAX, min(_HEBB_COEFF_MAX, c + noise()))
            conn.hebb_d = max(-_HEBB_COEFF_MAX, min(_HEBB_COEFF_MAX, d + noise()))


def mutate_hebb_coeffs(
    genome: "Genome",
    sigma: float = 0.05,
    rng: random.Random | None = None,
) -> None:
    """Perturb Hebb coefficients of all STDP-capable connections.

    Called by ``NeuroEvolution`` after each standard NEAT mutation when STDP
    is enabled.  Only connections that already have non-zero coefficients are
    mutated — this prevents the silent initialisation of coefficients on
    connections that should remain non-plastic.
    """
    _rng = rng or random
    for src in genome.nodes:
        for conn in src.connections:
            if not _has_hebb(conn):
                continue
            conn.hebb_a = max(-_HEBB_COEFF_MAX, min(_HEBB_COEFF_MAX,
                conn.hebb_a + _rng.gauss(0.0, sigma)))
            conn.hebb_b = max(-_HEBB_COEFF_MAX, min(_HEBB_COEFF_MAX,
                conn.hebb_b + _rng.gauss(0.0, sigma)))
            conn.hebb_c = max(-_HEBB_COEFF_MAX, min(_HEBB_COEFF_MAX,
                conn.hebb_c + _rng.gauss(0.0, sigma)))
            conn.hebb_d = max(-_HEBB_COEFF_MAX, min(_HEBB_COEFF_MAX,
                conn.hebb_d + _rng.gauss(0.0, sigma)))
