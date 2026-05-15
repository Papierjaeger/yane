"""Built-in example configurations for the GUI."""
from __future__ import annotations
import time
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

def _step_hooks(total: float, env, step_callback, render_callback) -> None:
    """Handle per-step callbacks and GIL yielding after each env step."""
    if step_callback is not None:
        delay = step_callback(total)
        if delay > 0:
            time.sleep(delay)
    elif render_callback is None:
        # Yield the GIL so the Qt main thread can process events.
        # Without this, tight gym loops starve the UI thread.
        time.sleep(0)
    if render_callback is not None:
        frame = env.render()
        if frame is not None:
            render_callback(frame)


def _make_discrete_action_eval(
    env_id: str,
    early_stop: float | None = None,
    max_steps: int = 100_000,
):
    """Returns make(render_callback, step_callback, demo) → eval_fn.

    max_steps caps episode length. Keep it low for envs like Acrobot so that
    bad genomes don't run for an unreasonably long time in demo mode.
    """
    def make(render_callback=None, step_callback=None, demo=False):
        import numpy as np
        import gymnasium as gym
        env = gym.make(env_id,
                       render_mode="rgb_array" if render_callback else None,
                       max_episode_steps=max_steps)

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
                if not demo and early_stop is not None and total < early_stop:
                    break
                _step_hooks(total, env, step_callback, render_callback)
            return total

        evaluate._env = env
        return evaluate
    return make


def _make_mountaincar_eval(max_train_steps: int = 1000):
    """Returns make(render_callback, step_callback, demo) → eval_fn."""
    def make(render_callback=None, step_callback=None, demo=False):
        import gymnasium as gym
        env = gym.make("MountainCarContinuous-v0",
                       render_mode="rgb_array" if render_callback else None,
                       max_episode_steps=100_000)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset()
            total = 0.0
            episode_cap = 100_000 if demo else max_train_steps
            for _ in range(episode_cap):
                action = genome.forward(list(state))
                state, reward, terminated, truncated, _ = env.step(action)
                # velocity shaping: encourages momentum so training doesn't
                # converge to "do nothing" (action=0 → reward=0 → local optimum)
                total += reward + state[1]
                _step_hooks(total, env, step_callback, render_callback)
                if terminated or truncated:
                    break
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
        make_eval: Callable,   # (render_callback=None, step_callback=None, demo=False) -> eval_fn
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
            make_eval=lambda cb=None, step_cb=None, demo=False: _xor_eval,
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
                make_eval=_make_discrete_action_eval("CartPole-v1", max_steps=100_000),
                target_fitness=1000,
                supports_render=True,
            ),
            ExampleConfig(
                name="Acrobot",
                description="Swing up a two-link robot arm (6 inputs, 3 outputs).",
                n_inputs=6, n_outputs=3,
                max_nodes=30, max_connections=100,
                make_eval=_make_discrete_action_eval("Acrobot-v1", early_stop=-200, max_steps=1000),
                target_fitness=-100,
                supports_render=True,
            ),
            ExampleConfig(
                name="MountainCar (Continuous)",
                description="Drive a car up a hill with continuous actions (2 inputs, 1 output).",
                n_inputs=2, n_outputs=1,
                max_nodes=20, max_connections=60,
                make_eval=_make_mountaincar_eval(max_train_steps=1000),
                target_fitness=90,
                supports_render=True,
            ),
        ]
    except ImportError:
        pass

    return examples
