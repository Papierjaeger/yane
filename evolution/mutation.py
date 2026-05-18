import math
import random
from functools import lru_cache


@lru_cache(maxsize=None)
def _enum_choices(enum_class) -> list:
    return list(enum_class)


class Mutation:
    MIN_RATE = 0.001
    MAX_DELTA = 100.0  # value_delta cap: prevents multiplicative drift to inf over many generations

    def __init__(self) -> None:
        self.shift_rate = 0.5    # higher than original 0.1; self-adaptive rates adjust over time
        self.custom_rate = 0.1
        self.bool_rate = 0.1
        self.int_rate = 0.1
        self.rate_mutation_rate = 0.3   # faster self-adaptation (was 0.1)
        self.value_delta = 0.3   # moderate step size; fine-tunes via self-adaptation

    def mutate_value(self, value: float, sigma: float = 1.0) -> float:
        if random.random() < self.shift_rate:
            step = self.value_delta * sigma
            # Guard against inf/nan sigma (e.g. from sigma_global drift or overflow)
            if not math.isfinite(step) or step <= 0.0:
                return value
            return value + random.gauss(0, step)
        return value

    def mutate_bool(self, value: bool) -> bool:
        if random.random() < self.bool_rate:
            return not value
        return value

    def mutate_enum(self, value, enum_class):
        if random.random() < self.custom_rate:
            return random.choice(_enum_choices(enum_class))
        return value

    def mutate_int(self, value: int, lo: int = 0, hi: int = 10) -> int:
        if random.random() < self.int_rate:
            return random.randint(lo, hi)
        return value

    def mutate_rates(self) -> None:
        _r = random.random   # local alias avoids repeated attribute lookup
        if _r() < self.rate_mutation_rate:
            scale = _r() * 0.2 + 0.9   # uniform(0.9, 1.1) without function call
            lo = self.MIN_RATE
            # if/else avoids max()/min() Python function-call overhead
            r = self.shift_rate * scale
            self.shift_rate = r if lo <= r <= 0.999 else (lo if r < lo else 0.999)
            r = self.custom_rate * scale
            self.custom_rate = r if lo <= r <= 0.999 else (lo if r < lo else 0.999)
            r = self.bool_rate * scale
            self.bool_rate = r if lo <= r <= 0.999 else (lo if r < lo else 0.999)
            r = self.int_rate * scale
            self.int_rate = r if lo <= r <= 0.999 else (lo if r < lo else 0.999)
            r = self.rate_mutation_rate * scale
            self.rate_mutation_rate = r if lo <= r <= 0.999 else (lo if r < lo else 0.999)

        if _r() < self.shift_rate:
            d = self.value_delta * (_r() * 0.2 + 0.9)
            self.value_delta = d if 1e-6 <= d <= self.MAX_DELTA else (1e-6 if d < 1e-6 else self.MAX_DELTA)

    def copy(self) -> 'Mutation':
        m = Mutation()
        m.shift_rate = self.shift_rate
        m.custom_rate = self.custom_rate
        m.bool_rate = self.bool_rate
        m.int_rate = self.int_rate
        m.rate_mutation_rate = self.rate_mutation_rate
        m.value_delta = self.value_delta
        return m

    @classmethod
    def _clamp(cls, rate: float) -> float:
        r = rate
        return r if cls.MIN_RATE <= r <= 0.999 else (cls.MIN_RATE if r < cls.MIN_RATE else 0.999)
