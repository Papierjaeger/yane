"""Liquid Time-Constant (LTC) Nodes — ODE-basierte Neuronen für YANE.

Implementiert diskrete ODE-Dynamik (Forward Euler) per Knoten als Wrapper-Layer.
Knoten mit ``node.tau < inf`` verwenden die LTC-Update-Regel statt des Standard-
Aktivierungsschritts.

**LTC-ODE (Forward Euler Approximation):**
``x_{t+1} = x_t + dt * (-x_t/τ + activation(sum(inputs) + bias))``

Wobei:
- ``x_t`` = Knotenstate (``node.value``, durch ``persist_value=True`` erhalten)
- ``τ`` (tau) = Zeitkonstante: τ→∞ → Standard-Knoten, τ→0 → instantane Antwort
- ``dt`` = Diskretisierungsschritt (evolvierbar pro Knoten)

**Extremwert-Analyse:**
- τ→∞: ``-x_t/τ → 0`` → ``x_{t+1} ≈ x_t + dt * f(inputs)``
  (State akkumuliert monoton → langsam ändernder Zustand)
- τ→0: ``dt/τ → groß`` → State folgt sofort dem Eingangssignal
  (Verhält sich wie Standardknoten ohne Dynamics)

**Zero-Cost wenn deaktiviert:** ``node.tau = float("inf")`` → kein LTC-Overhead.

Integration::

    yane.set_ltc(enabled=True)
    # Knoten mit finitem tau werden automatisch als LTC behandelt
    from yane.evolution.ltc import make_node_ltc
    make_node_ltc(genome, node_idx=2, tau=1.0, dt=0.05)
    yane.train(fitness_fn)
"""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome

_TAU_MIN = 1e-6   # below this, behave as instant node
_TAU_MAX = 1e6    # effectively inf
_DT_MIN = 1e-4
_DT_MAX = 1.0


# ---------------------------------------------------------------------------
# Core LTC update
# ---------------------------------------------------------------------------

def apply_ltc_update(genome: "Genome") -> None:
    """Apply LTC ODE update to all LTC-enabled nodes.

    Called **after** ``genome.forward()`` so that the standard forward pass
    has already set pre-activation sums on each node.  LTC nodes override
    their value with the ODE-corrected state.

    Nodes with ``tau == inf`` (default) are skipped — zero cost.

    Parameters
    ----------
    genome :
        Genome whose LTC nodes are updated.
    """
    for node in genome.nodes:
        tau = node.tau
        if tau == float("inf") or tau >= _TAU_MAX:
            continue  # standard node — skip
        dt = max(_DT_MIN, min(_DT_MAX, node.dt))

        # f(inputs) = current node.value (set by forward() standard pass)
        f_in = node.value
        # LTC state = node.value (persist_value should be True for LTC nodes)
        x_t = node.value
        # ODE: x_{t+1} = x_t + dt*(-x_t/tau + f_in)
        effective_tau = max(_TAU_MIN, tau)
        x_next = x_t + dt * (-x_t / effective_tau + f_in)
        node.value = x_next


def genome_has_ltc(genome: "Genome") -> bool:
    """Return True when *genome* has at least one LTC node (tau < inf)."""
    return any(n.tau < _TAU_MAX for n in genome.nodes)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def make_node_ltc(
    genome: "Genome",
    node_idx: int,
    tau: float = 1.0,
    dt: float = 0.05,
) -> None:
    """Mark ``genome.nodes[node_idx]`` as an LTC node.

    Also sets ``persist_value = True`` so the ODE state carries over
    between timesteps (required for meaningful temporal dynamics).
    """
    node = genome.nodes[node_idx]
    node.tau = max(_TAU_MIN, float(tau))
    node.dt = max(_DT_MIN, min(_DT_MAX, float(dt)))
    node.persist_value = True


def mutate_ltc_params(
    genome: "Genome",
    tau_sigma: float = 0.1,
    dt_sigma: float = 0.005,
    rng: random.Random | None = None,
) -> None:
    """Perturb tau and dt of all LTC-enabled nodes.

    Called during NEAT mutation when LTC is active.
    """
    _rng = rng or random
    for node in genome.nodes:
        if node.tau >= _TAU_MAX:
            continue
        node.tau = max(_TAU_MIN, min(_TAU_MAX,
            node.tau * math.exp(_rng.gauss(0.0, tau_sigma))))
        node.dt = max(_DT_MIN, min(_DT_MAX,
            node.dt + _rng.gauss(0.0, dt_sigma)))
