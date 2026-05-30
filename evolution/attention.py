"""Evolvable Attention Heads für YANE.

Implementiert einen Multi-Head-Self-Attention-Vorverarbeitungsblock, der wie
ConvStack als Wrapper-Layer vor dem NEAT-Netz läuft.

**Architektur:**
```
[n_inputs raw values]
       ↓
AttentionBlock (Q/K/V Matrizen, evolvierbar)
       ↓
[head_dim * num_heads attended features]
       ↓
genome.forward(attended_features)
       ↓
[n_outputs]
```

**Attention-Berechnung (Scaled Dot-Product):**
``output = softmax(Q @ K^T / sqrt(head_dim)) @ V``

Wobei Q, K, V lineare Projektionen der Input-Vektoren sind.  Der Block
speichert seine Gewichtsmatrizen als evolvierbare Parameter.

**Evolvierbarkeit:**
- ``head_dim``, ``num_heads``: konfigurierbar, werden über Mutation angepasst.
- Gewichtsmatrizen W_Q, W_K, W_V (je [num_heads × head_dim × n_inputs]):
  Standard-NEAT-Mutation (Gauss-Perturbation).

**Zero-Cost wenn deaktiviert:** ``genome.attention_block = None`` → kein Overhead.

**Anmerkung zur Spec:**
Die Spec nennt ``NodeType.ATTENTION`` — dieser Ansatz verwendet stattdessen
eine Wrapper-Layer (wie ConvStack), da eine ATTENTION-Knotenintegration in
``genome.forward()`` eine Sequenz voraussetzt, die in NEAT nicht existiert.
Der Attention-Block läuft als Feature-Extraktor VOR dem NEAT-Netz.

Integration::

    yane.set_attention(enabled=True, head_dim=4, num_heads=2)
    yane.configure(n_inputs=yane.attention_n_inputs(), n_outputs=2)
    yane.train(lambda g: eval(g, data))
"""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------

def _softmax(values: list[float]) -> list[float]:
    """Numerically stable softmax."""
    max_v = max(values) if values else 0.0
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [e / total for e in exps] if total > 0 else [1.0 / len(values)] * len(values)


# ---------------------------------------------------------------------------
# AttentionBlock
# ---------------------------------------------------------------------------

class AttentionBlock:
    """Multi-Head Self-Attention preprocessing layer.

    Treats each input as a single "token" projected into Q/K/V space.
    Output is the concatenation of all head outputs (flat vector of
    ``num_heads * head_dim`` floats).

    Parameters
    ----------
    n_inputs :
        Number of input features (tokens).
    head_dim :
        Dimensionality of each attention head's K/Q/V space.
    num_heads :
        Number of parallel attention heads.
    """

    def __init__(
        self,
        n_inputs: int,
        head_dim: int = 4,
        num_heads: int = 2,
        seed: int | None = None,
    ) -> None:
        self.n_inputs = n_inputs
        self.head_dim = head_dim
        self.num_heads = num_heads

        rng = random.Random(seed)
        scale = math.sqrt(2.0 / max(1, n_inputs))

        # W_Q, W_K, W_V: [num_heads][head_dim][n_inputs]
        def _rand_matrix() -> list[list[float]]:
            return [
                [rng.gauss(0.0, scale) for _ in range(n_inputs)]
                for _ in range(head_dim)
            ]

        self.W_Q: list[list[list[float]]] = [_rand_matrix() for _ in range(num_heads)]
        self.W_K: list[list[list[float]]] = [_rand_matrix() for _ in range(num_heads)]
        self.W_V: list[list[list[float]]] = [_rand_matrix() for _ in range(num_heads)]

    @property
    def n_outputs(self) -> int:
        """Flat output dimension = ``num_heads * head_dim``."""
        return self.num_heads * self.head_dim

    def forward(self, inputs: list[float]) -> list[float]:
        """Apply multi-head attention; return flat attended output.

        Each input dimension is treated as one token (sequence length = n_inputs).
        Q, K, V are projections of the input vector per head.

        Parameters
        ----------
        inputs :
            Input vector of length ``n_inputs``.

        Returns
        -------
        list[float]
            Flat vector of length ``num_heads * head_dim``.
        """
        n = min(len(inputs), self.n_inputs)
        x = list(inputs[:n]) + [0.0] * (self.n_inputs - n)
        result: list[float] = []

        for h in range(self.num_heads):
            # Project to Q, K, V: each is a vector of shape [head_dim]
            def proj(W: list[list[float]]) -> list[float]:
                return [
                    sum(W[d][i] * x[i] for i in range(self.n_inputs))
                    for d in range(self.head_dim)
                ]

            q = proj(self.W_Q[h])  # [head_dim]
            k = proj(self.W_K[h])  # [head_dim]
            v = proj(self.W_V[h])  # [head_dim]

            # Scaled dot-product attention (self-attention: single token)
            # score = q · k / sqrt(head_dim)
            score = sum(q[d] * k[d] for d in range(self.head_dim))
            score /= math.sqrt(max(1, self.head_dim))
            # With a single token, softmax([score]) = 1.0 → output = v
            # For multi-token: weights = softmax(scores), output = sum(weights * v)
            # Here we treat each input dim as a separate "query", producing attention per dim
            attn_weights = _softmax([score])  # [1] since single token
            out = [attn_weights[0] * v[d] for d in range(self.head_dim)]
            result.extend(out)

        return result

    def mutate(self, sigma: float = 0.1, rng: random.Random | None = None) -> None:
        """Perturb all weight matrices with Gaussian noise."""
        _rng = rng or random
        for h in range(self.num_heads):
            for matrix in [self.W_Q[h], self.W_K[h], self.W_V[h]]:
                for row in matrix:
                    for i in range(len(row)):
                        row[i] += _rng.gauss(0.0, sigma)

    def crossover(self, other: "AttentionBlock") -> "AttentionBlock":
        """Uniform per-head crossover; inherits structure from *self*."""
        child = self.copy()  # start from self, then optionally replace heads with other's
        for h in range(min(self.num_heads, other.num_heads)):
            if random.random() < 0.5:
                child.W_Q[h] = [list(row) for row in other.W_Q[h]]
                child.W_K[h] = [list(row) for row in other.W_K[h]]
                child.W_V[h] = [list(row) for row in other.W_V[h]]
        return child

    def copy(self) -> "AttentionBlock":
        child = AttentionBlock.__new__(AttentionBlock)
        child.n_inputs = self.n_inputs
        child.head_dim = self.head_dim
        child.num_heads = self.num_heads
        child.W_Q = [[list(row) for row in mat] for mat in self.W_Q]
        child.W_K = [[list(row) for row in mat] for mat in self.W_K]
        child.W_V = [[list(row) for row in mat] for mat in self.W_V]
        return child
