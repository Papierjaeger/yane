"""Built-in example configurations for the GUI."""
from __future__ import annotations
from typing import Callable

from yane.core.genome import Genome

# ---------------------------------------------------------------------------
# XOR — always available
# ---------------------------------------------------------------------------

_XOR_DATA = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]


def _xor_eval(genome: Genome) -> float:
    fitness = 0.0
    for inputs, target in _XOR_DATA:
        out = genome.forward(inputs)
        fitness -= abs(out[0] - target[0])
    return fitness


# ---------------------------------------------------------------------------
# Gym examples — optional
# ---------------------------------------------------------------------------

def _make_discrete_action_eval(env_id: str, early_stop: float | None = None):
    """Returns make(render_callback) → eval_fn for discrete-action gym envs."""
    def make(render_callback=None):
        import numpy as np
        import gymnasium as gym
        env = gym.make(env_id, render_mode="rgb_array" if render_callback else None)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                out = genome.forward(list(state))
                action = int(np.argmax(out))
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += reward
                if early_stop is not None and total < early_stop:
                    break
                if render_callback is not None:
                    frame = env.render()
                    if frame is not None:
                        render_callback(frame)
            return total

        evaluate._env = env
        return evaluate
    return make


def _make_mountaincar_eval():
    """Returns make(render_callback) → eval_fn for MountainCarContinuous."""
    def make(render_callback=None):
        import gymnasium as gym
        env = gym.make("MountainCarContinuous-v0",
                       render_mode="rgb_array" if render_callback else None)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset()
            total = 0.0
            for _ in range(800):
                action = genome.forward(list(state))
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward + state[1]
                if terminated or truncated:
                    break
                if render_callback is not None:
                    frame = env.render()
                    if frame is not None:
                        render_callback(frame)
            return total

        evaluate._env = env
        return evaluate
    return make


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ExampleConfig:
    def __init__(
        self,
        name: str,
        description: str,
        n_inputs: int,
        n_outputs: int,
        max_nodes: int,
        max_connections: int,
        make_eval: Callable,   # signature: (render_callback | None) -> eval_fn
        target_fitness: float,
        supports_render: bool = False,
        test_cases: list[tuple[list[float], list[float]]] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.max_nodes = max_nodes
        self.max_connections = max_connections
        self.make_eval = make_eval
        self.target_fitness = target_fitness
        self.supports_render = supports_render
        self.test_cases = test_cases


def load_examples() -> list[ExampleConfig]:
    examples = [
        ExampleConfig(
            name="XOR",
            description="Learn the XOR function (2 inputs, 1 output).",
            n_inputs=2, n_outputs=1,
            max_nodes=20, max_connections=50,
            make_eval=lambda cb=None: _xor_eval,
            target_fitness=-0.1,
            supports_render=False,
            test_cases=[
                ([0.0, 0.0], [0.0]),
                ([0.0, 1.0], [1.0]),
                ([1.0, 0.0], [1.0]),
                ([1.0, 1.0], [0.0]),
            ],
        ),
    ]

    try:
        import gymnasium  # noqa: F401
        examples += [
            ExampleConfig(
                name="CartPole",
                description="Balance a pole on a cart (4 inputs, 2 outputs).",
                n_inputs=4, n_outputs=2,
                max_nodes=30, max_connections=100,
                make_eval=_make_discrete_action_eval("CartPole-v1"),
                target_fitness=500,
                supports_render=True,
            ),
            ExampleConfig(
                name="Acrobot",
                description="Swing up a two-link robot arm (6 inputs, 3 outputs).",
                n_inputs=6, n_outputs=3,
                max_nodes=30, max_connections=100,
                make_eval=_make_discrete_action_eval("Acrobot-v1", early_stop=-200),
                target_fitness=-64,
                supports_render=True,
            ),
            ExampleConfig(
                name="MountainCar (Continuous)",
                description="Drive a car up a hill with continuous actions (2 inputs, 1 output).",
                n_inputs=2, n_outputs=1,
                max_nodes=20, max_connections=60,
                make_eval=_make_mountaincar_eval(),
                target_fitness=90,
                supports_render=True,
            ),
        ]
    except ImportError:
        pass

    return examples
