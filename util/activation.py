from enum import Enum
import math


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


class ActivationFunction:
    @classmethod
    def activate(cls, activation: ActivationType, value: float) -> float:
        match activation:
            case ActivationType.LINEAR:
                return value
            case ActivationType.SIGMOID:
                v = max(-_CLIP, min(_CLIP, value))
                return 1.0 / (1.0 + math.exp(-v))
            case ActivationType.TANH:
                return math.tanh(value)
            case ActivationType.RELU:
                return value if value > 0.0 else 0.0
            case ActivationType.BINARY:
                return 1.0 if value >= 0.5 else 0.0
            case ActivationType.LEAKY_RELU:
                return value if value > 0.0 else 0.01 * value
            case ActivationType.ELU:
                return value if value >= 0.0 else math.exp(max(-_CLIP, value)) - 1.0
            case ActivationType.SWISH:
                v = max(-_CLIP, min(_CLIP, value))
                return value / (1.0 + math.exp(-v))
            case ActivationType.SOFTPLUS:
                v = max(-_CLIP, min(_CLIP, value))
                return math.log(1.0 + math.exp(v))
            case ActivationType.SINE:
                return math.sin(value)
            case _:
                raise ValueError(f"Unknown activation: {activation}")
