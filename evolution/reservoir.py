"""Evolutionary Reservoir Computing — Echo State Networks mit evolvierbaren Readout-Verbindungen.

Das Reservoir ist **fixiert** (zufällig initialisiert, nicht evolviert).
Nur die Readout-Gewichte (W_out) werden durch Evolution oder Ridge-Regression optimiert.

**Echo State Network (ESN) Dynamik:**
``x(t+1) = (1-α) * x(t) + α * tanh(W * x(t) + W_in * u(t) + b)``

Wobei:
- x(t) = Reservoir-State (n_reservoir,)
- u(t) = Eingabe (n_inputs,)
- W = Interne Reservoir-Verbindungen (fixiert, spectral_radius < 1)
- W_in = Input-Projektion (fixiert)
- b = Bias-Vektor (fixiert)
- α = Leaking Rate (0 < α ≤ 1)

**Echo-State-Property:**
spectral_radius(W) < 1 → Reservoir "vergisst" alte Eingaben → stabiles Gedächtnis.

**Readout:**
``y(t) = W_out @ x(t)``

Nur W_out wird optimiert — entweder durch:
1. **Ridge Regression** (analytisch): W_out = Y @ X^T @ (X @ X^T + λI)^{-1}
2. **Evolution** (via Fitness-Funktion und NEAT-Kompatibilität).

**Serialisierbarkeit:**
`ReservoirGenome` ist vollständig pickle-fähig (kein Genome-Objekt nötig).

Integration::

    reservoir = yane.configure_reservoir(
        n_reservoir=100, spectral_radius=0.9,
        input_scaling=0.5, leaking_rate=0.3,
    )
    # Analytische Lösung:
    result = yane.train_reservoir(reservoir, train_inputs, train_targets)
    print(result.train_mse)
    # Evolutionäre Lösung:
    yane.set_reservoir(reservoir)
    yane.train(lambda g: -compute_mse(g, test_inputs, test_targets))
"""
from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# ReservoirGenome
# ---------------------------------------------------------------------------

class ReservoirGenome:
    """Echo State Network with fixed reservoir and evolvable readout.

    Parameters
    ----------
    n_inputs :
        Number of input channels.
    n_reservoir :
        Number of reservoir neurons (hidden units).
    n_outputs :
        Number of output channels (readout dimension).
    spectral_radius :
        Spectral radius of the reservoir weight matrix W.
        Must be < 1 to satisfy the Echo State Property.
    input_scaling :
        Scaling factor for W_in (input → reservoir).
    leaking_rate :
        Leaking coefficient α ∈ (0, 1].  α=1 = no leaking (standard ESN);
        α < 1 = leaky integration (slower time-scale).
    seed :
        RNG seed for deterministic reservoir initialization.
    """

    def __init__(
        self,
        n_inputs: int,
        n_reservoir: int,
        n_outputs: int,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leaking_rate: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self._seed = seed

        # Initialize fixed reservoir and evolvable readout
        self._W, self._W_in, self._b = self._init_reservoir(seed)
        self._W_out: list[list[float]] = _zeros(n_outputs, n_reservoir)
        self._state: list[float] = [0.0] * n_reservoir

        # For evolution: expose readout as flat weight list
        self.readout_flat: list[float] = [
            0.0 for _ in range(n_outputs * n_reservoir)
        ]

        # fitness tracking (NEAT-compatible)
        self.fitness: float = 0.0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_reservoir(
        self, seed: int | None
    ) -> tuple[list[list[float]], list[list[float]], list[float]]:
        """Build W (reservoir), W_in (input projection), b (bias)."""
        rng = random.Random(seed)
        n_r = self.n_reservoir
        n_i = self.n_inputs

        # Random sparse reservoir W
        W = [[rng.gauss(0.0, 1.0) if rng.random() < 0.1 else 0.0
              for _ in range(n_r)] for _ in range(n_r)]
        # Scale to desired spectral_radius
        radius = _spectral_radius(W)
        if radius > 1e-9:
            factor = self.spectral_radius / radius
            W = [[w * factor for w in row] for row in W]

        # Random input projection W_in
        W_in = [[rng.uniform(-self.input_scaling, self.input_scaling)
                 for _ in range(n_i)] for _ in range(n_r)]

        # Bias
        b = [rng.uniform(-0.1, 0.1) for _ in range(n_r)]

        return W, W_in, b

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset reservoir state to zeros (start of new episode)."""
        self._state = [0.0] * self.n_reservoir

    def forward(self, inputs: list[float]) -> list[float]:
        """Run one ESN step and return readout output.

        Updates the reservoir state in-place (stateful, call ``reset()``
        between independent episodes).
        """
        n_r = self.n_reservoir
        α = self.leaking_rate
        x = self._state

        # Compute new pre-activation
        pre = [self._b[i] for i in range(n_r)]
        for i in range(n_r):
            for j in range(n_r):
                pre[i] += self._W[i][j] * x[j]
            for k, u in enumerate(inputs[:self.n_inputs]):
                pre[i] += self._W_in[i][k] * u

        # Leaky integration: x_new = (1-α) * x + α * tanh(pre)
        new_state = [(1.0 - α) * x[i] + α * math.tanh(pre[i]) for i in range(n_r)]
        self._state = new_state

        # Readout: y = W_out @ x_new
        W_out = _flat_to_matrix(self.readout_flat, self.n_outputs, n_r)
        return [sum(W_out[o][i] * new_state[i] for i in range(n_r))
                for o in range(self.n_outputs)]

    def collect_states(
        self,
        inputs_sequence: list[list[float]],
        washout: int = 10,
    ) -> list[list[float]]:
        """Collect reservoir states for all timesteps.

        Parameters
        ----------
        inputs_sequence :
            List of input vectors (one per timestep).
        washout :
            Initial steps to discard (let reservoir warm up).

        Returns
        -------
        list[list[float]]
            States after washout — shape ``(T - washout) × n_reservoir``.
        """
        self.reset()
        states = []
        for t, inp in enumerate(inputs_sequence):
            self.forward(inp)
            if t >= washout:
                states.append(list(self._state))
        return states

    # ------------------------------------------------------------------
    # Readout weight management (for evolution)
    # ------------------------------------------------------------------

    def set_readout_from_flat(self, flat: list[float]) -> None:
        """Set readout weights from a flat list."""
        self.readout_flat = list(flat)

    def get_readout_flat(self) -> list[float]:
        return list(self.readout_flat)

    def mutate_readout(
        self,
        sigma: float = 0.1,
        rng: random.Random | None = None,
    ) -> None:
        """Perturb readout weights with Gaussian noise."""
        _rng = rng or random
        self.readout_flat = [
            w + _rng.gauss(0.0, sigma) for w in self.readout_flat
        ]

    def copy(self) -> "ReservoirGenome":
        c = ReservoirGenome.__new__(ReservoirGenome)
        c.n_inputs = self.n_inputs
        c.n_reservoir = self.n_reservoir
        c.n_outputs = self.n_outputs
        c.spectral_radius = self.spectral_radius
        c.input_scaling = self.input_scaling
        c.leaking_rate = self.leaking_rate
        c._seed = self._seed
        c._W = [list(row) for row in self._W]
        c._W_in = [list(row) for row in self._W_in]
        c._b = list(self._b)
        c._W_out = [list(row) for row in self._W_out]
        c._state = list(self._state)
        c.readout_flat = list(self.readout_flat)
        c.fitness = self.fitness
        return c

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: "str | Path") -> None:
        with open(str(path), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: "str | Path") -> "ReservoirGenome":
        with open(str(path), "rb") as f:
            return pickle.load(f)

    @property
    def actual_spectral_radius(self) -> float:
        """True spectral radius of the current reservoir matrix W."""
        return _spectral_radius(self._W)


# ---------------------------------------------------------------------------
# Ridge Regression Readout (analytical solution)
# ---------------------------------------------------------------------------

@dataclass
class ReservoirTrainResult:
    """Result of analytical Ridge readout training."""
    reservoir: ReservoirGenome
    train_mse: float
    n_samples: int


def train_ridge_readout(
    reservoir: ReservoirGenome,
    inputs_sequence: list[list[float]],
    targets_sequence: list[list[float]],
    lambda_ridge: float = 1e-4,
    washout: int = 10,
) -> ReservoirTrainResult:
    """Train readout weights analytically via Ridge Regression.

    Solves:  W_out = Y @ X^T @ (X @ X^T + λI)^{-1}

    Parameters
    ----------
    reservoir :
        The reservoir to train (modified in-place).
    inputs_sequence :
        Training inputs, one vector per timestep.
    targets_sequence :
        Corresponding target outputs.
    lambda_ridge :
        Ridge regularization strength.
    washout :
        Initial steps to skip (reservoir warm-up).

    Returns
    -------
    ReservoirTrainResult
        Contains the reservoir (updated W_out) and train MSE.
    """
    # Collect reservoir states (X) and targets (Y)
    states = reservoir.collect_states(inputs_sequence, washout=washout)
    targets = targets_sequence[washout:]

    if not states or not targets:
        return ReservoirTrainResult(reservoir, float("inf"), 0)

    n_r = reservoir.n_reservoir
    n_o = reservoir.n_outputs
    T = len(states)

    # X: (n_r, T), Y: (n_o, T)
    # W_out = Y @ X^T @ (X @ X^T + λI)^{-1}
    # Using normal equations: W_out * (X @ X^T + λI) = Y @ X^T

    # Compute X @ X^T (n_r × n_r) + λI
    XXT = [[sum(states[t][i] * states[t][j] for t in range(T))
             for j in range(n_r)] for i in range(n_r)]
    for i in range(n_r):
        XXT[i][i] += lambda_ridge

    # Compute Y @ X^T (n_o × n_r)
    YXT = [[sum(targets[t][o] * states[t][r] for t in range(T))
             for r in range(n_r)] for o in range(n_o)]

    # Solve via Gaussian elimination (simple for moderate n_r)
    W_out = _solve_linear(XXT, YXT)

    # Update reservoir readout
    reservoir._W_out = W_out
    reservoir.readout_flat = _matrix_to_flat(W_out)

    # Compute train MSE
    mse = 0.0
    for t in range(T):
        pred = [sum(W_out[o][r] * states[t][r] for r in range(n_r)) for o in range(n_o)]
        mse += sum((p - e) ** 2 for p, e in zip(pred, targets[t]))
    mse /= max(1, T * n_o)

    return ReservoirTrainResult(reservoir=reservoir, train_mse=mse, n_samples=T)


# ---------------------------------------------------------------------------
# Linear algebra helpers (pure Python, no numpy dependency)
# ---------------------------------------------------------------------------

def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _spectral_radius(W: list[list[float]]) -> float:
    """Power iteration approximation of spectral radius (largest eigenvalue magnitude)."""
    n = len(W)
    if n == 0:
        return 0.0
    rng = random.Random(0)
    v = [rng.gauss(0.0, 1.0) for _ in range(n)]
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-9:
        return 0.0
    v = [x / norm for x in v]
    radius = 0.0
    for _ in range(50):  # power iterations
        v_new = [sum(W[i][j] * v[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x * x for x in v_new))
        if norm < 1e-9:
            return 0.0
        radius = norm
        v = [x / norm for x in v_new]
    return radius


def _flat_to_matrix(flat: list[float], rows: int, cols: int) -> list[list[float]]:
    return [[flat[r * cols + c] for c in range(cols)] for r in range(rows)]


def _matrix_to_flat(matrix: list[list[float]]) -> list[float]:
    return [v for row in matrix for v in row]


def _solve_linear(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Solve A @ X = B^T via Gaussian elimination.

    A: (n × n), B: (m × n) — returns X = B @ A^{-1}: (m × n).
    """
    n = len(A)
    m = len(B)
    # Augment: [A | B^T]
    aug = [[A[i][j] for j in range(n)] + [B[k][i] for k in range(m)]
           for i in range(n)]
    # Forward elimination
    for col in range(n):
        # Find pivot
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue
        aug[col] = [v / pivot for v in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][j] - factor * aug[col][j]
                            for j in range(n + m)]
    # Extract solution
    X_T = [[aug[i][n + k] for i in range(n)] for k in range(m)]
    return X_T
