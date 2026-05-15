from __future__ import annotations
from enum import Enum
import math
from typing import Callable


class ActivationType(Enum):
    LINEAR   = "linear"
    SIGMOID  = "sigmoid"
    TANH     = "tanh"
    RELU     = "relu"
    BINARY   = "binary"
    LEAKY_RELU = "leaky_relu"   # solves dying-ReLU; negative slope = 0.01
    ELU      = "elu"            # smooth negative region: α(e^x - 1) for x < 0
    SWISH    = "swish"          # x * sigmoid(x) — smooth, often beats ReLU
    SOFTPLUS = "softplus"       # log(1 + e^x) — smooth ReLU approximation
    SINE     = "sine"           # sin(x) — useful for periodic/oscillatory tasks


_CLIP = 500.0  # prevent overflow in exp-based functions


def _sigmoid(v: float) -> float:
    v = max(-_CLIP, min(_CLIP, v))
    return 1.0 / (1.0 + math.exp(-v))


def _swish(v: float) -> float:
    v = max(-_CLIP, min(_CLIP, v))
    return v / (1.0 + math.exp(-v))


def _softplus(v: float) -> float:
    v = max(-_CLIP, min(_CLIP, v))
    return math.log(1.0 + math.exp(v))


def _elu(v: float) -> float:
    return v if v >= 0.0 else math.exp(max(-_CLIP, v)) - 1.0


# Lookup table: ActivationType → bare Python function (no method dispatch overhead).
# Node caches its entry directly so fire() calls it without going through activate().
ACTIVATION_FNS: dict[ActivationType, Callable[[float], float]] = {
    ActivationType.LINEAR:     lambda v: v,
    ActivationType.SIGMOID:    _sigmoid,
    ActivationType.TANH:       math.tanh,
    ActivationType.RELU:       lambda v: v if v > 0.0 else 0.0,
    ActivationType.BINARY:     lambda v: 1.0 if v >= 0.5 else 0.0,
    ActivationType.LEAKY_RELU: lambda v: v if v > 0.0 else 0.01 * v,
    ActivationType.ELU:        _elu,
    ActivationType.SWISH:      _swish,
    ActivationType.SOFTPLUS:   _softplus,
    ActivationType.SINE:       math.sin,
}


class ActivationFunction:
    @classmethod
    def activate(cls, activation: ActivationType, value: float) -> float:
        return ACTIVATION_FNS[activation](value)
