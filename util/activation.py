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
