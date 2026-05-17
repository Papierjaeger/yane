"""Built-in example configurations for the GUI."""
from __future__ import annotations
import time
from typing import Callable

from yane.core.genome import Genome

# Dataset examples — import make_eval and metadata directly from each example
# package so there is exactly one implementation (no duplication).
from yane.examples.XOR import (
    make_eval as _xor_make_eval, TEST_CASES as _XOR_TEST_CASES,
    N_INPUTS as _XOR_NI, N_OUTPUTS as _XOR_NO, TARGET_FITNESS as _XOR_FIT,
)
from yane.examples.basic_multiplication import (
    make_eval as _mult_make_eval,
    N_INPUTS as _MULT_NI, N_OUTPUTS as _MULT_NO, TARGET_FITNESS as _MULT_FIT,
    TEST_CASES as _MULT_TEST_CASES,
)
from yane.examples.simple_2_2_continuous import (
    make_eval as _reg22_make_eval,
    N_INPUTS as _REG22_NI, N_OUTPUTS as _REG22_NO, TARGET_FITNESS as _REG22_FIT,
    TEST_CASES as _REG22_TEST_CASES,
)
from yane.examples.simple_3_3_continuous import (
    make_eval as _reg33_make_eval,
    N_INPUTS as _REG33_NI, N_OUTPUTS as _REG33_NO, TARGET_FITNESS as _REG33_FIT,
    TEST_CASES as _REG33_TEST_CASES,
)
from yane.examples.sequence_recall_PI import (
    make_eval as _pi_make_eval,
    N_INPUTS as _PI_NI, N_OUTPUTS as _PI_NO, TARGET_FITNESS as _PI_FIT,
    dataset as _PI_DATASET, DECIMAL_PLACES as _PI_DECIMAL_PLACES,
)


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


_RENDER_INTERVAL = 1.0 / 30  # at most 30 fps for env.render() — the expensive part


def _make_step_hooks(step_callback, render_callback):
    """Return a per-step hook callable, or None if no callbacks are active.

    Callers check `if _hooks:` once per step instead of calling _step_hooks
    unconditionally — eliminates 600k+ function calls per training run.

    Critical: env.render() is only called when a frame will actually be
    displayed (rate-limited to 30 fps).  Calling render() on every step
    regardless costs several ms each time (pygame → RGB numpy), which
    adds up to ~1 s of pure render overhead per 500-step Acrobot episode
    even though only ~10 frames are ever shown.
    """
    if step_callback is None and render_callback is None:
        return None
    sc, rc = step_callback, render_callback

    if rc is not None:
        _last_render = [0.0]

        def _hooks(total, env):
            if sc is not None:
                delay = sc(total)
                if delay > 0:
                    time.sleep(delay)
            now = time.perf_counter()
            if now - _last_render[0] >= _RENDER_INTERVAL:
                _last_render[0] = now
                frame = env.render()
                if frame is not None:
                    rc(frame)
    else:
        def _hooks(total, env):
            delay = sc(total)
            if delay > 0:
                time.sleep(delay)

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
        env = gym.make(env_id, disable_env_checker=True,
                       render_mode="rgb_array" if render_callback else None,
                       max_episode_steps=max_steps)
        _hooks = _make_step_hooks(step_callback, render_callback)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset(); genome.reset()
            done = False
            total = 0.0
            while not done:
                out = genome.forward(state)
                action = out.index(max(out))
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
        env = gym.make(env_id, disable_env_checker=True,
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


def _make_blackjack_eval(n_episodes: int = 500):
    """Blackjack-v1: hit (1) oder stick (0) entscheiden.

    Obs-Tuple (player_sum, dealer_card, usable_ace) → normalisiert auf [0,1].
    Fitness = mittlere Belohnung über n_episodes (+1 gewonnen, -1 verloren, 0 unentschieden).
    Viele Episoden nötig wegen der hohen Varianz pro Runde.
    """
    def make(render_callback=None, step_callback=None, demo=False):
        import numpy as np
        import gymnasium as gym
        env = gym.make("Blackjack-v1", disable_env_checker=True,
                       render_mode="rgb_array" if render_callback else None)
        _hooks = _make_step_hooks(step_callback, render_callback)
        eps = 20 if demo else n_episodes

        def evaluate(genome: Genome) -> float:
            total = 0.0
            for _ in range(eps):
                obs, _ = env.reset()
                genome.reset()
                done = False
                while not done:
                    inputs = [obs[0] / 31.0, obs[1] / 10.0, float(obs[2])]
                    out = genome.forward(inputs)
                    action = out.index(max(out))   # 0=stick, 1=hit
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total += reward
                    if _hooks: _hooks(total, env)
            return total / eps

        evaluate._env = env
        return evaluate
    return make


def _make_cliffwalking_eval(max_steps: int = 200):
    """CliffWalking-v1: navigiere über ein 4×12-Gitter ohne in die Klippe zu fallen.

    Obs = integer 0–47 (Zeile × 12 + Spalte).
    Inputs: [Zeile/3, Spalte/11]. Belohnung: -1/Schritt, -100 Klippe.
    Reward-Shaping: Bonus für Annäherung an das Ziel (Zeile=3, Spalte=11).
    """
    def make(render_callback=None, step_callback=None, demo=False):
        import numpy as np
        import gymnasium as gym
        env = gym.make("CliffWalking-v1", disable_env_checker=True,
                       render_mode="rgb_array" if render_callback else None)
        _hooks = _make_step_hooks(step_callback, render_callback)
        cap = 10_000 if demo else max_steps
        goal_row, goal_col = 3, 11

        def evaluate(genome: Genome) -> float:
            obs, _ = env.reset()
            genome.reset()
            total = 0.0
            done = False
            steps = 0
            prev_dist = abs(obs // 12 - goal_row) + abs(obs % 12 - goal_col)
            while not done and steps < cap:
                row, col = obs // 12, obs % 12
                inputs = [row / 3.0, col / 11.0]
                out = genome.forward(inputs)
                action = out.index(max(out))
                obs, reward, terminated, truncated, _ = env.step(action)
                new_dist = abs(obs // 12 - goal_row) + abs(obs % 12 - goal_col)
                # Shaping: +0.5 when moving toward goal, -0.5 away
                total += reward + 0.5 * (prev_dist - new_dist)
                prev_dist = new_dist
                done = terminated or truncated
                steps += 1
                if _hooks: _hooks(total, env)
            return total

        evaluate._env = env
        return evaluate
    return make


def _make_frozenlake_eval(n_episodes: int = 20):
    """FrozenLake-v1 (nicht-rutschig): Erreiche das Ziel ohne in Löcher zu fallen.

    Obs = integer 0–15 (Zeile × 4 + Spalte). Ziel: (3, 3).
    Inputs: [Zeile/3, Spalte/3].
    Reward-Shaping: Bonus für Annäherung an das Ziel pro Schritt.
    Fitness = mittlere Belohnung über n_episodes.
    """
    def make(render_callback=None, step_callback=None, demo=False):
        import numpy as np
        import gymnasium as gym
        env = gym.make("FrozenLake-v1", disable_env_checker=True, is_slippery=False,
                       render_mode="rgb_array" if render_callback else None)
        _hooks = _make_step_hooks(step_callback, render_callback)
        eps = 5 if demo else n_episodes
        goal_row, goal_col = 3, 3

        def evaluate(genome: Genome) -> float:
            total = 0.0
            for _ in range(eps):
                obs, _ = env.reset()
                genome.reset()
                done = False
                prev_dist = abs(obs // 4 - goal_row) + abs(obs % 4 - goal_col)
                while not done:
                    row, col = obs // 4, obs % 4
                    inputs = [row / 3.0, col / 3.0]
                    out = genome.forward(inputs)
                    action = out.index(max(out))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    new_dist = abs(obs // 4 - goal_row) + abs(obs % 4 - goal_col)
                    # Shaping: 0.1 pro Schritt näher am Ziel
                    total += reward + 0.1 * (prev_dist - new_dist)
                    prev_dist = new_dist
                    done = terminated or truncated
                    if _hooks: _hooks(total, env)
            return total / eps

        evaluate._env = env
        return evaluate
    return make


def _make_taxi_eval(max_steps: int = 500):
    """Taxi-v4: Fahrgast aufnehmen und zum Ziel bringen (5×5-Gitter).

    Obs = integer 0–499, kodiert (row, col, passenger_loc, dest).
    Inputs: [Zeile/4, Spalte/4, Fahrgast/4, Ziel/3].
    Reward-Shaping: Bonus für Annäherung an Fahrgast oder Zielort.
    Belohnung: +20 Abgabe, -10 illegale Aktion, -1/Schritt.
    """
    def make(render_callback=None, step_callback=None, demo=False):
        import numpy as np
        import gymnasium as gym
        env = gym.make("Taxi-v4", disable_env_checker=True,
                       render_mode="rgb_array" if render_callback else None)
        _hooks = _make_step_hooks(step_callback, render_callback)
        cap = 10_000 if demo else max_steps
        # Fixed pickup/dropoff locations (R, G, Y, B) in the 5×5 grid
        _locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

        def _decode(obs):
            dest     = obs % 4
            pass_loc = (obs // 4) % 5
            col      = (obs // 20) % 5
            row      = obs // 100
            return row, col, pass_loc, dest

        def evaluate(genome: Genome) -> float:
            obs, _ = env.reset()
            genome.reset()
            total = 0.0
            done = False
            steps = 0
            while not done and steps < cap:
                row, col, pass_loc, dest = _decode(obs)
                inputs = [row / 4.0, col / 4.0, pass_loc / 4.0, dest / 3.0]
                out = genome.forward(inputs)
                action = out.index(max(out))
                prev_row, prev_col = row, col
                obs, reward, terminated, truncated, _ = env.step(action)
                row, col, pass_loc_new, dest_new = _decode(obs)

                # Reward shaping: encourage moving toward passenger or destination
                if pass_loc < 4:   # passenger not yet in taxi
                    pr, pc = _locs[pass_loc]
                    prev_d = abs(prev_row - pr) + abs(prev_col - pc)
                    new_d  = abs(row - pr) + abs(col - pc)
                else:              # passenger in taxi → head for destination
                    dr, dc = _locs[dest]
                    prev_d = abs(prev_row - dr) + abs(prev_col - dc)
                    new_d  = abs(row - dr) + abs(col - dc)
                total += reward + 0.5 * (prev_d - new_d)
                done = terminated or truncated
                steps += 1
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
        env = gym.make("CarRacing-v3", disable_env_checker=True,
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
        env = gym.make("Acrobot-v1", disable_env_checker=True,
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
                action = out.index(max(out))
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
        env = gym.make("MountainCar-v0", disable_env_checker=True,
                       render_mode="rgb_array" if render_callback else None,
                       max_episode_steps=200)
        _hooks = _make_step_hooks(step_callback, render_callback)

        def evaluate(genome: Genome) -> float:
            state, _ = env.reset(); genome.reset()
            max_pos = state[0]
            solved = False
            for _ in range(200):
                out = genome.forward(state)
                action = out.index(max(out))
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
        env = gym.make("Pendulum-v1", disable_env_checker=True,
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
        env = gym.make("MountainCarContinuous-v0", disable_env_checker=True,
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
        make_eval: Callable,
        target_fitness: float,
        category: str = "Sonstiges",
        n_initial_hidden: int = 0,
        supports_render: bool = False,
        supports_normalization: bool = False,
        test_cases: list[tuple[list[float], list[float]]] | None = None,
        stateful: bool = True,
        sequence_samples: list[tuple[list[float], list[float]]] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.max_nodes = max_nodes
        self.max_connections = max_connections
        self.make_eval = make_eval
        self.target_fitness = target_fitness
        self.category = category
        self.n_initial_hidden = n_initial_hidden
        self.supports_render = supports_render
        self.supports_normalization = supports_normalization
        self.test_cases = test_cases
        self.stateful = stateful
        self.sequence_samples = sequence_samples


def load_examples() -> list[ExampleConfig]:
    examples = [
        ExampleConfig(
            name="XOR",
            description="Learn the XOR function (2 inputs, 1 output).",
            n_inputs=_XOR_NI, n_outputs=_XOR_NO,
            max_nodes=20, max_connections=50,
            n_initial_hidden=2,
            make_eval=_xor_make_eval,
            target_fitness=_XOR_FIT,
            category="Dataset",
            supports_render=False,
            stateful=False,
            test_cases=_XOR_TEST_CASES,
        ),
        ExampleConfig(
            name="Multiplication",
            description=(
                "Lernt die Multiplikationstabelle (2 Inputs, 1 Output, 100 Samples).\n"
                "Inputs 0–9 und Outputs 0–81 werden intern auf [0,1] normalisiert."
            ),
            n_inputs=_MULT_NI, n_outputs=_MULT_NO,
            max_nodes=30, max_connections=100,
            make_eval=_mult_make_eval,
            target_fitness=_MULT_FIT,
            category="Dataset",
            supports_normalization=True,
            stateful=False,
            test_cases=_MULT_TEST_CASES,
        ),
        ExampleConfig(
            name="Regression 2→2",
            description="Lernt eine kontinuierliche 2→2 Abbildung (4 Samples).",
            n_inputs=_REG22_NI, n_outputs=_REG22_NO,
            max_nodes=20, max_connections=60,
            make_eval=_reg22_make_eval,
            target_fitness=_REG22_FIT,
            category="Dataset",
            stateful=False,
            test_cases=_REG22_TEST_CASES,
        ),
        ExampleConfig(
            name="Regression 3→3",
            description="Lernt eine kontinuierliche 3→3 Abbildung (8 Samples).",
            n_inputs=_REG33_NI, n_outputs=_REG33_NO,
            max_nodes=20, max_connections=80,
            make_eval=_reg33_make_eval,
            target_fitness=_REG33_FIT,
            category="Dataset",
            stateful=False,
            test_cases=_REG33_TEST_CASES,
        ),
        ExampleConfig(
            name="Sequence: Pi-Ziffern",
            description="Sagt die nächste Ziffer von Pi voraus — braucht Gedächtnis (1 Input, 1 Output, 10 Samples). Digit /9 → [0,1].",
            n_inputs=_PI_NI, n_outputs=_PI_NO,
            max_nodes=20, max_connections=60,
            make_eval=_pi_make_eval,
            target_fitness=_PI_FIT,
            category="Dataset",
            supports_normalization=True,
            stateful=True,
            sequence_samples=[(s["input"], s["output"]) for s in _PI_DATASET[:_PI_DECIMAL_PLACES]],
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
                category="Classic Control",
                supports_render=True,
            ),
            ExampleConfig(
                name="Acrobot",
                description="Swing up a two-link robot arm (6 inputs, 3 outputs).",
                n_inputs=6, n_outputs=3,
                max_nodes=30, max_connections=100,
                make_eval=_make_acrobot_eval(max_steps=500),
                target_fitness=0,
                category="Classic Control",
                supports_render=True,
            ),
            ExampleConfig(
                name="MountainCar (Continuous)",
                description="Drive a car up a hill with continuous actions (2 inputs, 1 output).",
                n_inputs=2, n_outputs=1,
                max_nodes=20, max_connections=60,
                make_eval=_make_mountaincar_eval(max_train_steps=1000),
                target_fitness=10.0,
                category="Classic Control",
                supports_render=True,
            ),
            ExampleConfig(
                name="MountainCar (Discrete)",
                description="Drive a car up a hill with 3 discrete actions (2 inputs, 3 outputs).",
                n_inputs=2, n_outputs=3,
                max_nodes=20, max_connections=60,
                make_eval=_make_mountaincar_discrete_eval(max_train_steps=200),
                target_fitness=10.0,
                category="Classic Control",
                supports_render=True,
            ),
            ExampleConfig(
                name="Pendulum",
                description="Swing up and balance a pendulum with continuous torque (3 inputs, 1 output).",
                n_inputs=3, n_outputs=1,
                max_nodes=20, max_connections=60,
                make_eval=_make_pendulum_eval(max_train_steps=200),
                target_fitness=-300,
                category="Classic Control",
                supports_render=True,
            ),
            ExampleConfig(
                name="LunarLander",
                description="Land a rocket between two flags with 4 discrete thrusters (8 inputs, 4 outputs).",
                n_inputs=8, n_outputs=4,
                max_nodes=40, max_connections=150,
                make_eval=_make_discrete_action_eval("LunarLander-v3", early_stop=-200, max_steps=1000),
                target_fitness=200,
                category="Box2D",
                supports_render=True,
            ),
            ExampleConfig(
                name="BipedalWalker",
                description="Walk with a two-legged robot using 4 continuous joint torques (24 inputs, 4 outputs).",
                n_inputs=24, n_outputs=4,
                max_nodes=60, max_connections=300,
                make_eval=_make_continuous_action_eval("BipedalWalker-v3", n_outputs=4, early_stop=-50, max_steps=1000),
                target_fitness=200,
                category="Box2D",
                supports_render=True,
            ),
            ExampleConfig(
                name="CarRacing",
                description="Race a car around a track — pixel obs downsampled to 12×12 grayscale (144 inputs, 3 outputs).",
                n_inputs=144, n_outputs=3,
                max_nodes=80, max_connections=500,
                make_eval=_make_carracing_eval(grid=12, max_steps=500),
                target_fitness=800,
                category="Pixel",
                supports_render=True,
            ),
            ExampleConfig(
                name="Blackjack",
                description=(
                    "Lerne Blackjack zu spielen (hit/stick).\n"
                    "Inputs: Kartensumme/31, Dealerkarte/10, Ass nutzbar.\n"
                    "Fitness: Durchschnitt über 500 Runden (max=1.0, min=-1.0)."
                ),
                n_inputs=3, n_outputs=2,
                max_nodes=20, max_connections=60,
                make_eval=_make_blackjack_eval(n_episodes=500),
                target_fitness=-0.05,
                category="Toy Text",
                supports_render=True,
            ),
            ExampleConfig(
                name="Cliff Walking",
                description=(
                    "Navigiere über ein 4×12-Gitter ohne in die Klippe zu fallen.\n"
                    "Inputs: Zeile/3, Spalte/11. Strafe: -1/Schritt, -100 Klippe.\n"
                    "Ziel: Fitness ≥ -50 (optimal: -37)."
                ),
                n_inputs=2, n_outputs=4,
                max_nodes=20, max_connections=60,
                make_eval=_make_cliffwalking_eval(max_steps=200),
                target_fitness=-50.0,
                category="Toy Text",
                supports_render=True,
            ),
            ExampleConfig(
                name="Frozen Lake",
                description=(
                    "Gleite über ein vereistes 4×4-Gitter zum Ziel ohne ins Loch zu fallen.\n"
                    "Inputs: Zeile/3, Spalte/3. Belohnung: +1 Ziel, 0 sonst.\n"
                    "Nicht-rutschig. Fitness = Erfolgsrate über 20 Episoden."
                ),
                n_inputs=2, n_outputs=4,
                max_nodes=20, max_connections=60,
                make_eval=_make_frozenlake_eval(n_episodes=20),
                target_fitness=0.8,
                category="Toy Text",
                supports_render=True,
            ),
            ExampleConfig(
                name="Taxi",
                description=(
                    "Fahre einen Taxi-Fahrgast zum Zielort (5×5-Gitter).\n"
                    "Inputs: Zeile/4, Spalte/4, Fahrgast/4, Ziel/3.\n"
                    "Belohnung: +20 Abgabe, -10 illegale Aktion, -1/Schritt."
                ),
                n_inputs=4, n_outputs=6,
                max_nodes=30, max_connections=100,
                make_eval=_make_taxi_eval(max_steps=500),
                target_fitness=5.0,
                category="Toy Text",
                supports_render=True,
            ),
        ]
    except ImportError:
        pass

    return examples
