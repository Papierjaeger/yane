import random
from functools import lru_cache


@lru_cache(maxsize=None)
def _enum_choices(enum_class) -> list:
    return list(enum_class)


class Mutation:
    MIN_RATE = 0.001

    def __init__(self) -> None:
        self.shift_rate = 0.1
        self.custom_rate = 0.1
        self.bool_rate = 0.1
        self.int_rate = 0.1
        self.rate_mutation_rate = 0.1
        self.value_delta = 0.1

    def mutate_value(self, value: float, sigma: float = 1.0) -> float:
        if random.random() < self.shift_rate:
            return value + random.gauss(0, self.value_delta * sigma)
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
        if random.random() < self.rate_mutation_rate:
            scale = random.uniform(0.9, 1.1)
            self.shift_rate = self._clamp(self.shift_rate * scale)
            self.custom_rate = self._clamp(self.custom_rate * scale)
            self.bool_rate = self._clamp(self.bool_rate * scale)
            self.int_rate = self._clamp(self.int_rate * scale)
            self.rate_mutation_rate = self._clamp(self.rate_mutation_rate * scale)

        if random.random() < self.shift_rate:
            self.value_delta = max(1e-6, self.value_delta * random.uniform(0.9, 1.1))

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
        return max(cls.MIN_RATE, min(0.999, rate))
