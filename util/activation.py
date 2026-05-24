from __future__ import annotations
from enum import Enum
import math
from typing import Callable


class ActivationType(Enum):
    LINEAR     = "linear"
    SIGMOID    = "sigmoid"
    TANH       = "tanh"
    RELU       = "relu"
    BINARY     = "binary"
    LEAKY_RELU = "leaky_relu"   # solves dying-ReLU; negative slope = 0.01
    ELU        = "elu"          # smooth negative region: α(e^x - 1) for x < 0
    SWISH      = "swish"        # x * sigmoid(x) — smooth, often beats ReLU
    SOFTPLUS   = "softplus"     # log(1 + e^x) — smooth ReLU approximation
    SINE       = "sine"         # sin(x) — periodic/oscillatory tasks
    SQUARE     = "square"       # x² — enables polynomial computation (e.g. x*y via (x+y)²)
    ABS        = "abs"          # |x| — piecewise linear, symmetric around 0
    GAUSSIAN   = "gaussian"     # exp(-x²) — radial basis; peaks at 0, decays symmetrically
    CUBE       = "cube"         # x³ — odd polynomial; can flip sign with input
    COSINE     = "cosine"       # cos(x) — phase-shifted periodic (complements SINE)


_CLIP = 500.0  # prevent overflow in exp-based functions
_GAUSS_CLIP = 26.0  # exp(-26²) underflows to 0; no need to compute


def _linear(v: float) -> float:     return v
def _relu(v: float) -> float:       return v if v > 0.0 else 0.0
def _binary(v: float) -> float:     return 1.0 if v >= 0.5 else 0.0
def _leaky_relu(v: float) -> float: return v if v > 0.0 else 0.01 * v
_SQUARE_CLIP = 1e4   # input clip for SQUARE: max output = 1e8, prevents inf cascade
_CUBE_CLIP   = 1e3   # input clip for CUBE:   max output = 1e9

def _abs(v: float) -> float:
    return v if v >= 0.0 else -v

def _square(v: float) -> float:
    if v > _SQUARE_CLIP:  return _SQUARE_CLIP * _SQUARE_CLIP
    if v < -_SQUARE_CLIP: return _SQUARE_CLIP * _SQUARE_CLIP
    return v * v

def _cube(v: float) -> float:
    if v > _CUBE_CLIP:  return _CUBE_CLIP * _CUBE_CLIP * _CUBE_CLIP
    if v < -_CUBE_CLIP: return -(_CUBE_CLIP * _CUBE_CLIP * _CUBE_CLIP)
    return v * v * v


def _gaussian(v: float) -> float:
    if v > _GAUSS_CLIP or v < -_GAUSS_CLIP:
        return 0.0
    return math.exp(-(v * v))


def _sigmoid(v: float) -> float:
    if v > _CLIP:
        v = _CLIP
    elif v < -_CLIP:
        v = -_CLIP
    return 1.0 / (1.0 + math.exp(-v))


def _swish(v: float) -> float:
    if v > _CLIP:
        v = _CLIP
    elif v < -_CLIP:
        v = -_CLIP
    return v / (1.0 + math.exp(-v))


def _softplus(v: float) -> float:
    if v > 20.0:    return v           # log(1+exp(v)) ≈ v for v > 20
    if v < -_CLIP:  return 0.0         # log(1+exp(-500)) ≈ 0
    return math.log(1.0 + math.exp(v))


def _elu(v: float) -> float:
    if v >= 0.0:    return v
    if v < -_CLIP:  return -1.0        # exp(-500) - 1 ≈ -1
    return math.exp(v) - 1.0


# Lookup table: ActivationType → named module-level function.
# Named functions (not lambdas) are required for pickle/multiprocessing.
# Node caches its entry directly so fire() calls it without dispatch overhead.
ACTIVATION_FNS: dict[ActivationType, Callable[[float], float]] = {
    ActivationType.LINEAR:     _linear,
    ActivationType.SIGMOID:    _sigmoid,
    ActivationType.TANH:       math.tanh,
    ActivationType.RELU:       _relu,
    ActivationType.BINARY:     _binary,
    ActivationType.LEAKY_RELU: _leaky_relu,
    ActivationType.ELU:        _elu,
    ActivationType.SWISH:      _swish,
    ActivationType.SOFTPLUS:   _softplus,
    ActivationType.SINE:       math.sin,
    ActivationType.SQUARE:     _square,
    ActivationType.ABS:        _abs,
    ActivationType.GAUSSIAN:   _gaussian,
    ActivationType.CUBE:       _cube,
    ActivationType.COSINE:     math.cos,
}


class ActivationFunction:
    @classmethod
    def activate(cls, activation: ActivationType, value: float) -> float:
        return ACTIVATION_FNS[activation](value)


# ---------------------------------------------------------------------------
# NumPy-vectorised activation functions for forward_batch().
# Each function accepts a 1-D np.ndarray and returns a new 1-D np.ndarray.
# Loaded lazily (numpy imported at first call) so scalar paths stay light.
# ---------------------------------------------------------------------------

def _get_batch_activation_fns():
    """Return the BATCH_ACTIVATION_FNS dict, building it on first call."""
    import numpy as _np

    _C = 500.0     # clip for exp-based functions
    _GC = 26.0     # gaussian clip: exp(-26²) underflows
    _SQ = 1e4      # square clip
    _CU = 1e3      # cube clip

    def _b_linear(x):     return x
    def _b_sigmoid(x):    return 1.0 / (1.0 + _np.exp(-_np.clip(x, -_C, _C)))
    def _b_tanh(x):       return _np.tanh(x)
    def _b_relu(x):       return _np.maximum(0.0, x)
    def _b_binary(x):     return _np.where(x >= 0.5, 1.0, 0.0)
    def _b_leaky(x):      return _np.where(x > 0.0, x, 0.01 * x)
    def _b_elu(x):
        safe = _np.clip(x, -_C, 0.0)
        return _np.where(x >= 0.0, x, _np.expm1(safe))
    def _b_swish(x):
        cx = _np.clip(x, -_C, _C)
        return cx / (1.0 + _np.exp(-cx))
    def _b_softplus(x):
        return _np.where(x > 20.0, x, _np.log1p(_np.exp(_np.clip(x, -_C, 20.0))))
    def _b_sine(x):       return _np.sin(x)
    def _b_square(x):     xc = _np.clip(x, -_SQ, _SQ); return xc * xc
    def _b_abs(x):        return _np.abs(x)
    def _b_gaussian(x):   xc = _np.clip(x, -_GC, _GC); return _np.exp(-(xc * xc))
    def _b_cube(x):       xc = _np.clip(x, -_CU, _CU); return xc * xc * xc
    def _b_cosine(x):     return _np.cos(x)

    return {
        ActivationType.LINEAR:     _b_linear,
        ActivationType.SIGMOID:    _b_sigmoid,
        ActivationType.TANH:       _b_tanh,
        ActivationType.RELU:       _b_relu,
        ActivationType.BINARY:     _b_binary,
        ActivationType.LEAKY_RELU: _b_leaky,
        ActivationType.ELU:        _b_elu,
        ActivationType.SWISH:      _b_swish,
        ActivationType.SOFTPLUS:   _b_softplus,
        ActivationType.SINE:       _b_sine,
        ActivationType.SQUARE:     _b_square,
        ActivationType.ABS:        _b_abs,
        ActivationType.GAUSSIAN:   _b_gaussian,
        ActivationType.CUBE:       _b_cube,
        ActivationType.COSINE:     _b_cosine,
    }


_BATCH_FNS: dict | None = None


def get_batch_activation_fns() -> dict:
    """Return the BATCH_ACTIVATION_FNS dict (built once, cached)."""
    global _BATCH_FNS
    if _BATCH_FNS is None:
        _BATCH_FNS = _get_batch_activation_fns()
    return _BATCH_FNS
