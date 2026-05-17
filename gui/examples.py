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
    genome.reset()
    fitness = 0.0
    for inputs, target in _XOR_DATA:
        out = genome.forward(inputs)
        fitness -= abs(out[0] - target[0])
    return fitness


# ---------------------------------------------------------------------------
# Gym examples — optional
# ---------------------------------------------------------------------------

def _step_hooks(total: float, env, step_callback, render_callback) -> None:
    """Handle per-step callbacks. Only called when at least one callback is active."""
    if step_callback is not None:
        delay = step_callback(total)
        if delay > 0:
            time.sleep(delay)
    if render_callback is not None:
        frame = env.render()
        if frame is not None:
            render_callback(frame)


def _make_step_hooks(step_callback, render_callback):
    """Return a per-step hook callable, or None if no callbacks are active.

    Callers check `if _hooks:` once per step instead of calling _step_hooks
    unconditionally — eliminates 600k+ function calls per training run.
    """
    if step_callback is None and render_callback is None:
        return None
    sc, rc = step_callback, render_callback

    def _hooks(total, env):
        if sc is not None:
            delay = sc(total)
            if delay > 0:
                time.sleep(delay)
        if rc is not None:
            frame = env.render()
            if frame is not None:
                rc(frame)

    return _hooks


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
        _hooks = _make_step_hooks(step_callback, render_callback)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset(); genome.reset()
            done = False
            total = 0.0
            while not done:
                out = genome.forward(state)
                action = int(np.argmax(out))
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += reward
                if not demo and early_stop is not None and total < early_stop:
                    break
                if _hooks: _hooks(total, env)
            return total

        evaluate._env = env
        return evaluate
    return make


def _make_continuous_action_eval(
    env_id: str,
    n_outputs: int,
    early_stop: float | None = None,
    max_steps: int = 1000,
):
    """Generic factory for envs with Box action space in [-1, 1].

    Sigmoid genome outputs [0, 1] are scaled to [-1, 1] per output.
    """
    def make(render_callback=None, step_callback=None, demo=False):
        import gymnasium as gym
        env = gym.make(env_id,
                       render_mode="rgb_array" if render_callback else None,
                       max_episode_steps=max_steps if not demo else None)
        _hooks = _make_step_hooks(step_callback, render_callback)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset(); genome.reset()
            total = 0.0
            done = False
            while not done:
                raw = genome.forward(state)
                action = [r * 2.0 - 1.0 for r in raw[:n_outputs]]
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
                if not demo and early_stop is not None and total < early_stop:
                    break
                if _hooks: _hooks(total, env)
            return total

        evaluate._env = env
        return evaluate
    return make


def _make_carracing_eval(grid: int = 12, max_steps: int = 500):
    """Returns make(render_callback, step_callback, demo) → eval_fn.

    CarRacing has 96×96×3 pixel observations — far too many raw inputs for
    NEAT. We downsample to grid×grid grayscale (default 144 inputs) so
    topology stays manageable. Action space is [steering, gas, brake]:
    steering uses the full [-1,1] range; gas and brake are [0,1].
    """
    n_inputs = grid * grid

    def make(render_callback=None, step_callback=None, demo=False):
        import numpy as np
        import gymnasium as gym
        env = gym.make("CarRacing-v3",
                       render_mode="rgb_array" if render_callback else None,
                       continuous=True)
        _hooks = _make_step_hooks(step_callback, render_callback)

        def evaluate(genome: Genome) -> float:
            obs, _ = env.reset(); genome.reset()
            total = 0.0
            done = False
            steps = 0
            cap = 10_000 if demo else max_steps
            while not done and steps < cap:
                # downsample to grid×grid grayscale, flatten to [0, 1]
                gray = obs.mean(axis=2)
                h, w = gray.shape
                small = gray.reshape(grid, h // grid, grid, w // grid).mean(axis=(1, 3))
                inputs = (small.flatten() / 255.0).tolist()

                raw = genome.forward(inputs)
                steering = raw[0] * 2.0 - 1.0          # sigmoid → [-1, 1]
                gas      = max(0.0, min(1.0, raw[1]))   # sigmoid → [0, 1]
                brake    = max(0.0, min(1.0, raw[2]))   # sigmoid → [0, 1]
                obs, reward, terminated, truncated, _ = env.step(np.array([steering, gas, brake], dtype=np.float64))
                total += reward
                done = terminated or truncated
                steps += 1
                if _hooks: _hooks(total, env)
            return total

        evaluate._env = env
        return evaluate
    return make


def _make_acrobot_eval(max_steps: int = 500):
    """Returns make(render_callback, step_callback, demo) → eval_fn.

    Acrobot-v1 reward is -1/step regardless of behavior → with empty-start
    genomes all get the same fitness, killing the selection gradient.
    Shaping adds tip height so evolution has a signal before fully solving.

    Tip height = -cos(θ1) - cos(θ1+θ2) ∈ [-2, 2]; goal is height ≥ 1.
    From observation: obs = [cos θ1, sin θ1, cos θ2, sin θ2, dθ1, dθ2]
      tip_height = -obs[0] - (obs[0]*obs[2] - obs[1]*obs[3])
    """
    def make(render_callback=None, step_callback=None, demo=False):
        import numpy as np
        import gymnasium as gym
        env = gym.make("Acrobot-v1",
                       render_mode="rgb_array" if render_callback else None,
                       max_episode_steps=max_steps)
        _hooks = _make_step_hooks(step_callback, render_callback)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset(); genome.reset()
            max_tip = -2.0
            solved = False
            done = False
            while not done:
                out = genome.forward(state)
                action = int(np.argmax(out))
                state, reward, terminated, truncated, _ = env.step(action)
                # tip height = -cos(θ1) - cos(θ1+θ2) ∈ [-2, 2]
                tip = -state[0] - (state[0] * state[2] - state[1] * state[3])
                if tip > max_tip:
                    max_tip = tip
                done = terminated or truncated
                if terminated:
                    solved = True
                if _hooks: _hooks(max_tip, env)
            # max_tip gives dense gradient; bonus ensures solved > unsolved
            return max_tip + (10.0 if solved else 0.0)

        evaluate._env = env
        return evaluate
    return make


def _make_mountaincar_discrete_eval(max_train_steps: int = 200):
    """Returns make(render_callback, step_callback, demo) → eval_fn.

    MountainCar-v0 gives -1/step for 200 steps regardless → all genomes get
    -200 until one accidentally solves it → no gradient for evolution.
    Reward shaping adds: max position reached + |velocity| * 10 per step,
    giving a meaningful fitness gradient before the task is solved.
    """
    def make(render_callback=None, step_callback=None, demo=False):
        import numpy as np
        import gymnasium as gym
        env = gym.make("MountainCar-v0",
                       render_mode="rgb_array" if render_callback else None,
                       max_episode_steps=200)
        _hooks = _make_step_hooks(step_callback, render_callback)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset(); genome.reset()
            max_pos = state[0]
            solved = False
            for _ in range(200):
                out = genome.forward(state)
                action = int(np.argmax(out))
                state, reward, terminated, truncated, _ = env.step(action)
                pos = state[0]
                if pos > max_pos: max_pos = pos
                if _hooks: _hooks(max_pos, env)
                if terminated or truncated:
                    solved = terminated
                    break
            return max_pos + (10.0 if solved else 0.0)

        evaluate._env = env
        return evaluate
    return make


def _make_pendulum_eval(max_train_steps: int = 500):
    """Returns make(render_callback, step_callback, demo) → eval_fn.

    Pendulum-v1 has a continuous action space in [-2, 2].
    Genome outputs are sigmoid [0, 1], so we scale: action = out * 4 - 2.
    """
    def make(render_callback=None, step_callback=None, demo=False):
        import gymnasium as gym
        env = gym.make("Pendulum-v1",
                       render_mode="rgb_array" if render_callback else None,
                       max_episode_steps=200)
        _hooks = _make_step_hooks(step_callback, render_callback)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset(); genome.reset()
            total = 0.0
            episode_cap = 200 if demo else max_train_steps
            for _ in range(episode_cap):
                raw = genome.forward(state)
                # sigmoid output [0,1] → action in [-2, 2]
                action = [raw[0] * 4.0 - 2.0]
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if _hooks: _hooks(total, env)
                if terminated or truncated:
                    break
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
        _hooks = _make_step_hooks(step_callback, render_callback)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset(); genome.reset()
            max_pos = state[0]
            solved = False
            episode_cap = 100_000 if demo else max_train_steps
            for _ in range(episode_cap):
                raw = genome.forward(state)
                # sigmoid [0,1] → action in [-1, 1] so genome can push both ways
                action = [raw[0] * 2.0 - 1.0]
                state, reward, terminated, truncated, _ = env.step(action)
                pos = state[0]
                if pos > max_pos: max_pos = pos
                if _hooks: _hooks(max_pos, env)
                if terminated or truncated:
                    solved = terminated  # truncated = timeout, terminated = goal reached
                    break
            # max_pos gives a gradient from -1.2 (stuck) to 0.45+ (solved);
            # bonus ensures any solved genome outranks every unsolved one
            return max_pos + (10.0 if solved else 0.0)

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
        n_initial_hidden: int = 0,
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
        self.n_initial_hidden = n_initial_hidden
        self.supports_render = supports_render
        self.test_cases = test_cases


def load_examples() -> list[ExampleConfig]:
    examples = [
        ExampleConfig(
            name="XOR",
            description="Learn the XOR function (2 inputs, 1 output).",
            n_inputs=2, n_outputs=1,
            max_nodes=20, max_connections=50,
            n_initial_hidden=2,
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
                make_eval=_make_acrobot_eval(max_steps=500),
                target_fitness=0,
                supports_render=True,
            ),
            ExampleConfig(
                name="MountainCar (Continuous)",
                description="Drive a car up a hill with continuous actions (2 inputs, 1 output).",
                n_inputs=2, n_outputs=1,
                max_nodes=20, max_connections=60,
                make_eval=_make_mountaincar_eval(max_train_steps=1000),
                target_fitness=10.0,
                supports_render=True,
            ),
            ExampleConfig(
                name="MountainCar (Discrete)",
                description="Drive a car up a hill with 3 discrete actions (2 inputs, 3 outputs).",
                n_inputs=2, n_outputs=3,
                max_nodes=20, max_connections=60,
                make_eval=_make_mountaincar_discrete_eval(max_train_steps=200),
                target_fitness=10.0,
                supports_render=True,
            ),
            ExampleConfig(
                name="Pendulum",
                description="Swing up and balance a pendulum with continuous torque (3 inputs, 1 output).",
                n_inputs=3, n_outputs=1,
                max_nodes=20, max_connections=60,
                make_eval=_make_pendulum_eval(max_train_steps=200),
                target_fitness=-300,
                supports_render=True,
            ),
            ExampleConfig(
                name="LunarLander",
                description="Land a rocket between two flags with 4 discrete thrusters (8 inputs, 4 outputs).",
                n_inputs=8, n_outputs=4,
                max_nodes=40, max_connections=150,
                make_eval=_make_discrete_action_eval("LunarLander-v3", early_stop=-200, max_steps=1000),
                target_fitness=200,
                supports_render=True,
            ),
            ExampleConfig(
                name="BipedalWalker",
                description="Walk with a two-legged robot using 4 continuous joint torques (24 inputs, 4 outputs).",
                n_inputs=24, n_outputs=4,
                max_nodes=60, max_connections=300,
                make_eval=_make_continuous_action_eval("BipedalWalker-v3", n_outputs=4, early_stop=-50, max_steps=1000),
                target_fitness=200,
                supports_render=True,
            ),
            ExampleConfig(
                name="CarRacing",
                description="Race a car around a track — pixel obs downsampled to 12×12 grayscale (144 inputs, 3 outputs).",
                n_inputs=144, n_outputs=3,
                max_nodes=80, max_connections=500,
                make_eval=_make_carracing_eval(grid=12, max_steps=500),
                target_fitness=800,
                supports_render=True,
            ),
        ]
    except ImportError:
        pass

    return examples
