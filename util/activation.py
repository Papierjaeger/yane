from enum import Enum
import numpy as np


class ActivationType(Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    BINARY = "binary"


class ActivationFunction:
    @classmethod
    def activate(cls, activation: ActivationType, value: float) -> float:
        match activation:
            case ActivationType.LINEAR:
                return value
            case ActivationType.SIGMOID:
                return float(1.0 / (1.0 + np.exp(-np.clip(value, -500, 500))))
            case ActivationType.TANH:
                return float(np.tanh(value))
            case ActivationType.RELU:
                return float(np.maximum(0, value))
            case ActivationType.BINARY:
                return 1.0 if value >= 0.5 else 0.0
            case _:
                raise ValueError(f"Unknown activation: {activation}")
