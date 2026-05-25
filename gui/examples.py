"""Built-in example configurations for the GUI."""
from __future__ import annotations
import time
from typing import Callable

from yane.core.genome import Genome
from yane.evolution.evaluator_components import (
    EvaluatorSpec, GraphPolicyScore, StateEncoder,
)

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
    dataset as _MULT_DATASET, NORMALIZER as _MULT_NORMALIZER,
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
    NORMALIZER as _PI_NORMALIZER,
    make_curriculum_eval as _pi_make_curriculum_eval,
    make_curriculum_targets as _pi_make_curriculum_targets,
    CURRICULUM_SPEC as _PI_CURRICULUM_SPEC,
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
                # Clamp to [-1, 1] — LINEAR-activated outputs could otherwise blow up.
                action = [max(-1.0, min(1.0, r * 2.0 - 1.0)) for r in raw[:n_outputs]]
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
    Inputs: mixed one-hot + scaled (Zeile, Spalte).
    Components: rollout_score (env reward + distance shaping), oracle_score (oracle-action match).
    """
    encoder = StateEncoder(mode="mixed", sizes=(4, 12), scales=(3.0, 11.0))
    _COMPONENTS = ("rollout_score", "oracle_score")
    _BASE_WEIGHTS = {"rollout_score": 1.0, "oracle_score": 0.4}

    def _inputs(row: int, col: int) -> list[float]:
        return encoder.encode((row, col))

    def make(render_callback=None, step_callback=None, demo=False, enabled_components=None):
        import gymnasium as gym
        env = gym.make("CliffWalking-v1", disable_env_checker=True,
                       render_mode="rgb_array" if render_callback else None)
        _hooks = _make_step_hooks(step_callback, render_callback)
        cap = 10_000 if demo else max_steps
        goal_row, goal_col = 3, 11
        start_states = [36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        enabled = (
            frozenset(enabled_components)
            if enabled_components is not None
            else None
        )
        spec = EvaluatorSpec(
            component_weights=_BASE_WEIGHTS,
            enabled_components=enabled,
        )
        _scores: dict = {}

        def _oracle_action(row: int, col: int) -> int:
            if row == 3 and col == 0:
                return 0  # up, away from the cliff
            if row < 2:
                return 2  # down to the safe corridor
            if col < goal_col:
                return 1  # right
            return 2      # down into the goal

        def evaluate(genome: Genome) -> float:
            rollout_total = 0.0
            oracle_total = 0.0
            for start in start_states:
                env.reset()
                env.unwrapped.s = start
                obs = start
                genome.reset()
                done = False
                steps = 0
                prev_dist = abs(obs // 12 - goal_row) + abs(obs % 12 - goal_col)
                best_dist = prev_dist
                case_rollout = 0.0
                case_oracle = 0.0
                while not done and steps < cap:
                    row, col = obs // 12, obs % 12
                    out = genome.forward(_inputs(row, col))
                    action = out.index(max(out))
                    case_oracle += 4.0 if action == _oracle_action(row, col) else -1.0
                    obs, reward, terminated, truncated, _ = env.step(action)
                    new_row, new_col = obs // 12, obs % 12
                    new_dist = abs(new_row - goal_row) + abs(new_col - goal_col)
                    delta = prev_dist - new_dist
                    shape = 2.0 * delta if delta > 0 else 0.75 * delta
                    case_rollout += reward + shape - 0.05 * new_dist
                    if new_dist < best_dist:
                        case_rollout += 3.0
                        best_dist = new_dist
                    if new_row == goal_row and new_col == goal_col:
                        case_rollout += 80.0
                    prev_dist = new_dist
                    done = terminated or truncated
                    steps += 1
                    if _hooks: _hooks(rollout_total + case_rollout, env)
                rollout_total += case_rollout
                oracle_total += case_oracle
            components = {
                "rollout_score": rollout_total / len(start_states),
                "oracle_score": oracle_total / len(start_states),
            }
            _scores.update(components)
            return spec.combine(components)

        evaluate._env = env
        evaluate._component_scores = _scores
        evaluate._available_components = list(_COMPONENTS)
        return evaluate
    return make


def _make_frozenlake_eval(n_episodes: int = 20):
    """FrozenLake-v1 (nicht-rutschig): Erreiche das Ziel ohne in Löcher zu fallen.

    Obs = integer 0–15 (Zeile × 4 + Spalte). Ziel: (3, 3).
    Inputs: mixed one-hot + scaled (Zeile, Spalte).
    Components: rollout_score (env reward + distance shaping), subgoal_score (distance progress).
    """
    encoder = StateEncoder(mode="mixed", sizes=(4, 4), scales=(3.0, 3.0))
    _COMPONENTS = ("rollout_score", "subgoal_score")
    _BASE_WEIGHTS = {"rollout_score": 1.0, "subgoal_score": 0.5}

    def _inputs(row: int, col: int) -> list[float]:
        return encoder.encode((row, col))

    def make(render_callback=None, step_callback=None, demo=False, enabled_components=None):
        import gymnasium as gym
        env = gym.make("FrozenLake-v1", disable_env_checker=True, is_slippery=False,
                       render_mode="rgb_array" if render_callback else None)
        _hooks = _make_step_hooks(step_callback, render_callback)
        goal_row, goal_col = 3, 3
        holes = {5, 7, 11, 12}
        start_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]
        if demo:
            start_states = [0]
        enabled = (
            frozenset(enabled_components)
            if enabled_components is not None
            else None
        )
        spec = EvaluatorSpec(
            component_weights=_BASE_WEIGHTS,
            enabled_components=enabled,
        )
        _scores: dict = {}

        def evaluate(genome: Genome) -> float:
            rollout_total = 0.0
            subgoal_total = 0.0
            for start in start_states:
                env.reset()
                env.unwrapped.s = start
                obs = start
                genome.reset()
                done = False
                prev_dist = abs(obs // 4 - goal_row) + abs(obs % 4 - goal_col)
                steps = 0
                case_rollout = 0.0
                case_subgoal = 0.0
                while not done and steps < 20:
                    row, col = obs // 4, obs % 4
                    out = genome.forward(_inputs(row, col))
                    action = out.index(max(out))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    new_dist = abs(obs // 4 - goal_row) + abs(obs % 4 - goal_col)
                    delta = prev_dist - new_dist
                    case_rollout += reward - 0.02
                    if obs in holes:
                        case_rollout -= 1.0
                    case_subgoal += 0.25 * delta
                    prev_dist = new_dist
                    done = terminated or truncated
                    steps += 1
                    if _hooks: _hooks(rollout_total + case_rollout, env)
                rollout_total += case_rollout
                subgoal_total += case_subgoal
            components = {
                "rollout_score": rollout_total / len(start_states),
                "subgoal_score": subgoal_total / len(start_states),
            }
            _scores.update(components)
            return spec.combine(components)

        evaluate._env = env
        evaluate._component_scores = _scores
        evaluate._available_components = list(_COMPONENTS)
        return evaluate
    return make


def _make_taxi_eval(max_steps: int = 500):
    """Taxi-v4: Fahrgast aufnehmen und zum Ziel bringen (5×5-Gitter).

    Obs = integer 0–499, kodiert (row, col, passenger_loc, dest).
    Inputs: mixed one-hot + scaled (row, col, pass_loc, dest).
    Components: policy_score (GraphPolicyScore over state space),
                rollout_score (episodic reward + shaping),
                subgoal_score (movement toward current subgoal).
    """
    encoder = StateEncoder(mode="mixed", sizes=(5, 5, 5, 4), scales=(4.0, 4.0, 4.0, 3.0))
    _COMPONENTS = ("policy_score", "rollout_score", "subgoal_score")
    _BASE_WEIGHTS = {"policy_score": 45.0, "rollout_score": 1.0, "subgoal_score": 1.0}

    def _inputs(row: int, col: int, pass_loc: int, dest: int) -> list[float]:
        return encoder.encode((row, col, pass_loc, dest))

    def make(render_callback=None, step_callback=None, demo=False, enabled_components=None):
        import gymnasium as gym
        env = gym.make("Taxi-v4", disable_env_checker=True,
                       render_mode="rgb_array" if render_callback else None)
        _hooks = _make_step_hooks(step_callback, render_callback)
        cap = 10_000 if demo else max_steps
        _locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        enabled = (
            frozenset(enabled_components)
            if enabled_components is not None
            else None
        )
        spec = EvaluatorSpec(
            component_weights=_BASE_WEIGHTS,
            enabled_components=enabled,
        )
        _scores: dict = {}

        def _decode(obs):
            dest     = obs % 4
            pass_loc = (obs // 4) % 5
            col      = (obs // 20) % 5
            row      = obs // 100
            return row, col, pass_loc, dest

        train_cases = [
            (0, 0, 1, 2), (0, 4, 2, 3), (4, 0, 0, 1), (4, 3, 1, 0),
            (2, 2, 4, 0), (1, 3, 4, 2), (3, 1, 3, 1), (2, 4, 0, 3),
        ]

        transitions = env.unwrapped.P
        _policy_scorer = GraphPolicyScore(transitions, env.action_space.n, terminal_reward=0.0)
        dist = _policy_scorer.distances

        policy_probe_states = [
            env.unwrapped.encode(row, col, pass_loc, dest)
            for row in range(5)
            for col in range(5)
            for pass_loc in range(5)
            for dest in range(4)
            if dist[env.unwrapped.encode(row, col, pass_loc, dest)] < float("inf")
        ]

        def _subgoal(pass_loc: int, dest: int) -> tuple[int, int]:
            if pass_loc < 4:
                return _locs[pass_loc]
            return _locs[dest]

        def _movement_action_score(
            action: int,
            row: int,
            col: int,
            pass_loc: int,
            dest: int,
        ) -> float:
            goal_r, goal_c = _subgoal(pass_loc, dest)
            if pass_loc < 4 and (row, col) == (goal_r, goal_c):
                return 8.0 if action == 4 else -2.0
            if pass_loc == 4 and (row, col) == (goal_r, goal_c):
                return 10.0 if action == 5 else -2.0
            prev_d = abs(row - goal_r) + abs(col - goal_c)
            candidates = {
                0: (min(4, row + 1), col),
                1: (max(0, row - 1), col),
                2: (row, min(4, col + 1)),
                3: (row, max(0, col - 1)),
            }
            if action not in candidates:
                return -2.0
            nr, nc = candidates[action]
            new_d = abs(nr - goal_r) + abs(nc - goal_c)
            if new_d < prev_d:
                return 3.0
            if new_d > prev_d:
                return -1.5
            return -0.5

        def evaluate(genome: Genome) -> float:
            cases = train_cases[:1] if demo else train_cases
            # policy_score component
            if enabled is None or "policy_score" in enabled:
                policy_total = 0.0
                for probe_state in policy_probe_states:
                    row, col, pass_loc, dest = _decode(probe_state)
                    out = genome.forward(_inputs(row, col, pass_loc, dest))
                    action = out.index(max(out))
                    policy_total += _policy_scorer.action_score(probe_state, action)
                policy_val = policy_total / len(policy_probe_states)
            else:
                policy_val = 0.0
            # rollout_score and subgoal_score components
            rollout_total = 0.0
            subgoal_total = 0.0
            for row0, col0, pass0, dest0 in cases:
                env.reset()
                obs = env.unwrapped.encode(row0, col0, pass0, dest0)
                env.unwrapped.s = obs
                genome.reset()
                case_rollout = 0.0
                case_subgoal = 0.0
                done = False
                steps = 0
                prev_pass_in_taxi = pass0 == 4
                while not done and steps < cap:
                    row, col, pass_loc, dest = _decode(obs)
                    out = genome.forward(_inputs(row, col, pass_loc, dest))
                    action = out.index(max(out))
                    case_subgoal += _movement_action_score(action, row, col, pass_loc, dest)
                    prev_row, prev_col, prev_pass_loc = row, col, pass_loc
                    obs, reward, terminated, truncated, _ = env.step(action)
                    row, col, pass_loc_new, _dest_new = _decode(obs)

                    if prev_pass_loc < 4:
                        pr, pc = _locs[prev_pass_loc]
                        prev_d = abs(prev_row - pr) + abs(prev_col - pc)
                        new_d = abs(row - pr) + abs(col - pc)
                    else:
                        dr, dc = _locs[dest]
                        prev_d = abs(prev_row - dr) + abs(prev_col - dc)
                        new_d = abs(row - dr) + abs(col - dc)
                    delta = prev_d - new_d
                    shape = 2.0 * delta if delta > 0 else 0.75 * delta
                    case_rollout += reward + shape - 0.05 * new_d
                    if pass_loc_new == 4 and not prev_pass_in_taxi:
                        case_rollout += 40.0
                        prev_pass_in_taxi = True
                    if reward == -10:
                        case_rollout -= 5.0
                    if terminated:
                        case_rollout += 80.0
                    done = terminated or truncated
                    steps += 1
                    if _hooks: _hooks(rollout_total + case_rollout, env)
                rollout_total += case_rollout
                subgoal_total += case_subgoal
            components = {
                "policy_score": policy_val,
                "rollout_score": rollout_total / len(cases),
                "subgoal_score": subgoal_total / len(cases),
            }
            _scores.update(components)
            return spec.combine(components)

        evaluate._env = env
        evaluate._component_scores = _scores
        evaluate._available_components = list(_COMPONENTS)
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
                steering = max(-1.0, min(1.0, raw[0] * 2.0 - 1.0))   # clamp to [-1, 1]
                gas      = max( 0.0, min(1.0, raw[1]))                # clamp to [ 0, 1]
                brake    = max( 0.0, min(1.0, raw[2]))                # clamp to [ 0, 1]
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
                # Clamp to action space [-2, 2] — output activation can evolve to
                # LINEAR (any value), which would otherwise crash gym with NaN/Overflow.
                action = [max(-2.0, min(2.0, raw[0] * 4.0 - 2.0))]
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
                # Clamp to action space [-1, 1] — output activation can evolve to
                # LINEAR (any value), which would otherwise crash gym with NaN/Overflow.
                action = [max(-1.0, min(1.0, raw[0] * 2.0 - 1.0))]
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

def _gui_defaults(**overrides) -> dict:
    defaults = {
        "n_workers": 0,
        "multi_eval": 1,
        "aggregation": "mean",
        "sigma_penalty": 0.0,
        "fitness_shaping": False,
        "multi_objective": False,
        "multi_objective_complexity_weight": 0.01,
        "quality_diversity": False,
        "quality_diversity_descriptor": "Topology",
        "fitness_components": False,
        "fitness_component_mode": "Fix",
        "matrix_forward": False,
        "cppn_substrate": False,
        "cppn_hidden": 2,
        "remote_eval": False,
        "remote_urls": "",
        "remote_timeout_s": 30.0,
        "remote_retries": 2,
        "remote_batch_size": 0,
        "novelty": True,
        "speciation": True,
        "crossover": True,
        "diversity_injection": True,
        "convergence_spread": 0.0,
        "convergence_stagnation": 1.0,
        "early_stop_factor": 0.0,
        "efficiency_max_ms": 0.0,
        "efficiency_penalty": 0.0,
        "elite_global": 1,
        "elite_species": 1,
        "normalize": True,
        "lamarck_schedule": "Adaptiv",
        "lamarck_optimizer": "Hill-Climbing",
        "lamarck_steps": 5,
    }
    defaults.update(overrides)
    return defaults


def _adaptive_defaults(**overrides) -> dict:
    defaults = {
        "adaptive_controller": False,
        "operator_scheduler": False,
        "interspecies_mode": "Fix",
        "interspecies_min_rate": 0.0,
        "interspecies_max_rate": 0.2,
        "lamarck_schedule": "Adaptiv",
        "lamarck_optimizer": "Hill-Climbing",
        "lamarck_budget": 0,
        "meta_adaptive": False,
        "module_library": False,
        "module_insert_rate": 0.02,
    }
    defaults.update(overrides)
    return defaults


_FAST_DATASET_DEFAULTS = _gui_defaults(
    n_workers=1,
    fitness_shaping=True,
    matrix_forward=True,
    novelty=True,
)
_HARD_DATASET_DEFAULTS = _gui_defaults(
    n_workers=1,
    fitness_shaping=True,
    matrix_forward=True,
    quality_diversity=True,
    quality_diversity_descriptor="Topology",
    fitness_components=True,
    fitness_component_mode="Adaptiv",
)
_HARD_DATASET_ADAPTIVE = _adaptive_defaults(
    adaptive_controller=True,
    operator_scheduler=True,
    interspecies_mode="Adaptiv",
    interspecies_min_rate=0.01,
    interspecies_max_rate=0.15,
    meta_adaptive=True,
    module_library=True,
    module_insert_rate=0.02,
)
_SEQUENCE_DEFAULTS = _gui_defaults(
    n_workers=1,
    fitness_shaping=True,
    matrix_forward=False,
    quality_diversity=True,
    quality_diversity_descriptor="Behavior",
    fitness_components=True,
    fitness_component_mode="Adaptiv",
)
_SEQUENCE_ADAPTIVE = _adaptive_defaults(
    adaptive_controller=True,
    operator_scheduler=True,
    interspecies_mode="Adaptiv",
    interspecies_min_rate=0.01,
    interspecies_max_rate=0.2,
    meta_adaptive=True,
    module_library=False,
    lamarck_budget=100,
)
_CLASSIC_CONTROL_DEFAULTS = _gui_defaults(
    n_workers=0,
    multi_eval=1,
    matrix_forward=True,
    fitness_shaping=True,
    quality_diversity=True,
    quality_diversity_descriptor="Behavior",
    fitness_components=False,
    normalize=True,
    early_stop_factor=0.0,
)
_CLASSIC_CONTROL_ADAPTIVE = _adaptive_defaults(
    adaptive_controller=True,
    operator_scheduler=True,
    interspecies_mode="Adaptiv",
    interspecies_min_rate=0.01,
    interspecies_max_rate=0.15,
)
_NOISY_CONTROL_DEFAULTS = _gui_defaults(
    n_workers=0,
    multi_eval=3,
    aggregation="mean",
    sigma_penalty=0.1,
    matrix_forward=True,
    fitness_shaping=True,
    quality_diversity=True,
    quality_diversity_descriptor="Behavior",
)
_LARGE_CONTROL_DEFAULTS = _gui_defaults(
    n_workers=0,
    multi_eval=1,
    matrix_forward=False,
    fitness_shaping=True,
    quality_diversity=True,
    quality_diversity_descriptor="Behavior",
    fitness_components=False,
    early_stop_factor=0.5,
    elite_global=2,
    elite_species=1,
)
_LARGE_CONTROL_ADAPTIVE = _adaptive_defaults(
    adaptive_controller=True,
    operator_scheduler=True,
    interspecies_mode="Adaptiv",
    interspecies_min_rate=0.01,
    interspecies_max_rate=0.2,
    module_library=True,
    module_insert_rate=0.01,
)


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
        input_scale: list[float] | None = None,
        output_scale: list[float] | None = None,
        make_curriculum: Callable | None = None,
        default_population: int = 100,
        default_target_species: int = 5,
        default_curriculum: bool = False,
        default_fitness_shaping: bool = False,
        default_lamarck_steps: int = 0,
        default_config: dict | None = None,
        default_adaptive_policies: dict | None = None,
        action_display_fn: "Callable[[list[float]], str] | None" = None,
        evaluator_components: "list[str] | None" = None,
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
        # Per-channel scale factors mapping normalized [0,1] values to raw units
        # (e.g. Multiplication: input_scale=[9,9], output_scale=[81]). When set,
        # the Inspect tab exposes a toggle to view/enter values in raw units.
        self.input_scale = input_scale
        self.output_scale = output_scale
        # Optional curriculum factory: called with (normalize=bool) →
        # list[CurriculumStage].  When set, the training tab shows a
        # "Curriculum" checkbox.
        self.make_curriculum = make_curriculum
        self.default_population = default_population
        self.default_target_species = default_target_species
        self.default_curriculum = default_curriculum
        self.default_fitness_shaping = default_fitness_shaping
        self.default_lamarck_steps = default_lamarck_steps
        self.default_config = _gui_defaults(**(default_config or {}))
        self.default_adaptive_policies = _adaptive_defaults(**(default_adaptive_policies or {}))
        # Optional callable: (network_outputs: list[float]) -> str
        # When set, the Inspect tab shows the interpreted action next to raw outputs.
        self.action_display_fn: "Callable[[list[float]], str] | None" = action_display_fn
        # Named fitness components available for this evaluator (None = no per-component UI).
        # Each name corresponds to a component tracked in evaluate._component_scores.
        self.evaluator_components: list[str] | None = list(evaluator_components) if evaluator_components else None


def _pi_make_curriculum(
    normalize: bool = True,
    target_fitness: float = _PI_FIT,
):
    """Build CurriculumStage list for the Pi-digits example.

    Stages expand the learned prefix one digit at a time.  The final target is
    the value chosen in the GUI; intermediate targets are scaled to the prefix
    length, and already learned prefix digits are strongly protected.
    """
    from yane.evolution.curriculum import CurriculumStage
    targets = _pi_make_curriculum_targets(
        final_target_fitness=target_fitness,
        total_digits=_PI_DECIMAL_PLACES,
    )
    return [
        CurriculumStage(
            _pi_make_curriculum_eval(
                n,
                normalize=normalize,
                protected_digits=n - 1,
                protect_tolerance=0.0,
            ),
            target_fitness=target,
            name=label,
        )
        for (n, label), target in zip(_PI_CURRICULUM_SPEC, targets)
    ]


def load_examples() -> list[ExampleConfig]:
    examples = [
        ExampleConfig(
            name="XOR",
            description="Learn the XOR function (2 inputs, 1 output).",
            n_inputs=_XOR_NI, n_outputs=_XOR_NO,
            max_nodes=20, max_connections=50,
            n_initial_hidden=4,
            make_eval=_xor_make_eval,
            target_fitness=_XOR_FIT,
            category="Dataset",
            supports_render=False,
            stateful=False,
            test_cases=_XOR_TEST_CASES,
            default_target_species=2,
            default_fitness_shaping=True,
            default_lamarck_steps=2,
            default_config={
                **_FAST_DATASET_DEFAULTS,
                "lamarck_schedule": "Explizit",
                "lamarck_steps": 2,
                "quality_diversity": False,
                "fitness_components": False,
            },
        ),
        ExampleConfig(
            name="Multiplication",
            description=(
                "Lernt die Multiplikationstabelle (2 Inputs, 1 Output, 100 Samples).\n"
                "Inputs 0–9 und Outputs 0–81 werden intern auf [0,1] normalisiert."
            ),
            n_inputs=_MULT_NI, n_outputs=_MULT_NO,
            max_nodes=30, max_connections=100,
            n_initial_hidden=4,    # multiplication is non-linear; needs hidden layer
            make_eval=_mult_make_eval,
            target_fitness=-0.5,  # avg raw error ≤ 0.4/sample → predictions correct to rounding (overrides _MULT_FIT=-5.0 which allows 4-unit error)
            category="Dataset",
            supports_normalization=True,
            stateful=False,
            # Full multiplication table (100 entries) — user can scan all a*b combinations
            test_cases=[(s["input"], s["output"]) for s in _MULT_DATASET],
            input_scale=_MULT_NORMALIZER.input_scale_list,
            output_scale=_MULT_NORMALIZER.output_scale_list,
            default_population=150,
            default_target_species=4,
            default_lamarck_steps=2,
            default_config={
                **_HARD_DATASET_DEFAULTS,
                "lamarck_schedule": "Explizit",
                "lamarck_steps": 2,
                "quality_diversity_descriptor": "Behavior",
            },
            default_adaptive_policies=_HARD_DATASET_ADAPTIVE,
        ),
        ExampleConfig(
            name="Regression 2→2",
            description="Lernt eine kontinuierliche 2→2 Abbildung (4 Samples) — enthält XOR-artige Muster.",
            n_inputs=_REG22_NI, n_outputs=_REG22_NO,
            max_nodes=25, max_connections=80,
            n_initial_hidden=4,    # 2 outputs × 2 hidden each — enough capacity for parallel XOR + linear
            make_eval=_reg22_make_eval,
            target_fitness=_REG22_FIT,
            category="Dataset",
            stateful=False,
            test_cases=_REG22_TEST_CASES,
            default_target_species=2,
            default_fitness_shaping=True,
            default_lamarck_steps=8,
            default_config={
                **_HARD_DATASET_DEFAULTS,
                "lamarck_schedule": "Explizit",
                "lamarck_steps": 8,
                "quality_diversity_descriptor": "Behavior",
            },
            default_adaptive_policies=_HARD_DATASET_ADAPTIVE,
        ),
        ExampleConfig(
            name="Regression 3→3",
            description="Lernt eine kontinuierliche 3→3 Abbildung (8 Samples) — nichtlinear.",
            n_inputs=_REG33_NI, n_outputs=_REG33_NO,
            max_nodes=30, max_connections=120,
            n_initial_hidden=12,   # extra capacity for parallel non-linear outputs
            make_eval=_reg33_make_eval,
            target_fitness=-1.0,  # avg error ≤ 0.042/output (normalized) → well-fitted (overrides _REG33_FIT=-5.0)
            category="Dataset",
            stateful=False,
            test_cases=_REG33_TEST_CASES,
            default_population=150,
            default_target_species=4,
            default_fitness_shaping=True,
            default_lamarck_steps=5,
            default_config={
                **_HARD_DATASET_DEFAULTS,
                "lamarck_schedule": "Explizit",
                "lamarck_steps": 5,
                "quality_diversity_descriptor": "Behavior",
                "cppn_substrate": True,
                "cppn_hidden": 4,
            },
            default_adaptive_policies=_HARD_DATASET_ADAPTIVE,
        ),
        ExampleConfig(
            name="Sequence: Pi-Ziffern",
            description="Sagt die nächste Ziffer von Pi voraus — braucht Gedächtnis (1 Input, 1 Output, 10 Samples). Digit /9 → [0,1].",
            n_inputs=_PI_NI, n_outputs=_PI_NO,
            max_nodes=30, max_connections=100,
            make_eval=_pi_make_eval,
            target_fitness=_PI_FIT,
            category="Dataset",
            supports_normalization=True,
            stateful=True,
            test_cases=[(s["input"], s["output"]) for s in _PI_DATASET[:_PI_DECIMAL_PLACES]],
            sequence_samples=[(s["input"], s["output"]) for s in _PI_DATASET[:_PI_DECIMAL_PLACES]],
            input_scale=_PI_NORMALIZER.input_scale_list,
            output_scale=_PI_NORMALIZER.output_scale_list,
            make_curriculum=_pi_make_curriculum,
            default_population=100,
            default_target_species=4,
            default_curriculum=True,
            default_config={
                **_SEQUENCE_DEFAULTS,
                "lamarck_schedule": "Adaptiv",
                "lamarck_steps": 5,
                "normalize": True,
            },
            default_adaptive_policies=_SEQUENCE_ADAPTIVE,
        ),
    ]

    # ── Action display helpers for Gym examples ────────────────────────────────
    def _action_cartpole(outs): return ["← Links", "→ Rechts"][outs.index(max(outs))]
    def _action_acrobot(outs): return ["+CCW", "⊘ Kein Drehmoment", "-CW"][outs.index(max(outs))]
    def _action_mountaincar_d(outs): return ["← Links", "⊘ Kein Antrieb", "→ Rechts"][outs.index(max(outs))]
    def _action_lunarlander(outs): return [
        "⊘ Keine", "← Links", "▲ Haupt", "→ Rechts"][outs.index(max(outs))]
    def _action_cartpole_2(outs): return f"{'← Links' if outs.index(max(outs)) == 0 else '→ Rechts'} (raw: {outs[0]:.3f} | {outs[1]:.3f})"

    try:
        import gymnasium  # noqa: F401
        examples += [
            ExampleConfig(
                name="CartPole",
                description="Balance a pole on a cart (4 inputs, 2 outputs).",
                n_inputs=4, n_outputs=2,
                max_nodes=30, max_connections=100,
                make_eval=_make_discrete_action_eval("CartPole-v1", max_steps=500),
                target_fitness=500,
                category="Classic Control",
                supports_render=True,
                stateful=False,
                default_population=120,
                default_target_species=4,
                default_config=_CLASSIC_CONTROL_DEFAULTS,
                default_adaptive_policies=_CLASSIC_CONTROL_ADAPTIVE,
                action_display_fn=_action_cartpole,
            ),
            ExampleConfig(
                name="Acrobot",
                description="Swing up a two-link robot arm (6 inputs, 3 outputs).",
                n_inputs=6, n_outputs=3,
                max_nodes=30, max_connections=100,
                make_eval=_make_acrobot_eval(max_steps=500),
                target_fitness=10.5,  # requires solving (bonus +10) + max_tip ≥ 0.5
                category="Classic Control",
                supports_render=True,
                stateful=False,
                default_population=160,
                default_target_species=5,
                default_config=_CLASSIC_CONTROL_DEFAULTS,
                default_adaptive_policies=_CLASSIC_CONTROL_ADAPTIVE,
                action_display_fn=_action_acrobot,
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
                stateful=False,
                default_population=140,
                default_target_species=4,
                default_config=_CLASSIC_CONTROL_DEFAULTS,
                default_adaptive_policies=_CLASSIC_CONTROL_ADAPTIVE,
                action_display_fn=lambda outs: f"Drehmoment: {outs[0]:.4f}",
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
                stateful=False,
                default_population=140,
                default_target_species=4,
                default_config=_CLASSIC_CONTROL_DEFAULTS,
                default_adaptive_policies=_CLASSIC_CONTROL_ADAPTIVE,
                action_display_fn=_action_mountaincar_d,
            ),
            ExampleConfig(
                name="Pendulum",
                description="Swing up and balance a pendulum with continuous torque (3 inputs, 1 output).",
                n_inputs=3, n_outputs=1,
                max_nodes=20, max_connections=60,
                n_initial_hidden=2,    # continuous control benefits from hidden layer
                make_eval=_make_pendulum_eval(max_train_steps=200),
                target_fitness=-200,    # decent balance; baseline ≈ -507, good control ≈ -200
                category="Classic Control",
                supports_render=True,
                stateful=False,
                default_population=160,
                default_target_species=5,
                default_config={
                    **_NOISY_CONTROL_DEFAULTS,
                    "matrix_forward": True,
                    "multi_eval": 2,
                    "sigma_penalty": 0.05,
                },
                default_adaptive_policies=_CLASSIC_CONTROL_ADAPTIVE,
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
                stateful=False,
                default_population=220,
                default_target_species=8,
                default_config={
                    **_LARGE_CONTROL_DEFAULTS,
                    "matrix_forward": True,
                    "multi_eval": 2,
                    "sigma_penalty": 0.05,
                },
                default_adaptive_policies=_LARGE_CONTROL_ADAPTIVE,
                action_display_fn=_action_lunarlander,
            ),
            ExampleConfig(
                name="BipedalWalker",
                description="Walk with a two-legged robot using 4 continuous joint torques (24 inputs, 4 outputs).",
                n_inputs=24, n_outputs=4,
                max_nodes=60, max_connections=300,
                n_initial_hidden=4,
                make_eval=_make_continuous_action_eval("BipedalWalker-v3", n_outputs=4, early_stop=-50, max_steps=1000),
                target_fitness=100,  # substantial walking; baseline ≈ -100, solved ≈ 300
                category="Box2D",
                supports_render=True,
                stateful=False,
                default_population=300,
                default_target_species=10,
                default_config={
                    **_LARGE_CONTROL_DEFAULTS,
                    "multi_eval": 2,
                },
                default_adaptive_policies=_LARGE_CONTROL_ADAPTIVE,
            ),
            ExampleConfig(
                name="CarRacing",
                description="Race a car around a track — pixel obs downsampled to 12×12 grayscale (144 inputs, 3 outputs).",
                n_inputs=144, n_outputs=3,
                max_nodes=80, max_connections=500,
                make_eval=_make_carracing_eval(grid=12, max_steps=500),
                target_fitness=100,    # baseline 0-50; 100 = consistent on-track driving
                category="Pixel",
                supports_render=True,
                stateful=False,
                default_population=300,
                default_target_species=12,
                default_config={
                    **_LARGE_CONTROL_DEFAULTS,
                    "matrix_forward": False,
                    "quality_diversity_descriptor": "Topology",
                    "fitness_components": True,
                    "fitness_component_mode": "Adaptiv",
                    "cppn_substrate": True,
                    "cppn_hidden": 8,
                    "elite_global": 3,
                },
                default_adaptive_policies={
                    **_LARGE_CONTROL_ADAPTIVE,
                    "interspecies_max_rate": 0.25,
                    "meta_adaptive": True,
                },
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
                stateful=False,
                default_population=180,
                default_target_species=4,
                default_config={
                    **_NOISY_CONTROL_DEFAULTS,
                    "multi_eval": 1,
                    "sigma_penalty": 0.0,
                },
                default_adaptive_policies=_CLASSIC_CONTROL_ADAPTIVE,
            ),
            ExampleConfig(
                name="Cliff Walking",
                description=(
                    "Navigiere über ein 4×12-Gitter ohne in die Klippe zu fallen.\n"
                    "Inputs: Zeile/3, Spalte/11. Strafe: -1/Schritt, -100 Klippe.\n"
                    "Reward-Shaping + Bonus für neue Distanz-Bestmarken und Zielerreichung."
                ),
                n_inputs=18, n_outputs=4,
                max_nodes=40, max_connections=140,
                n_initial_hidden=8,
                make_eval=_make_cliffwalking_eval(max_steps=200),
                target_fitness=0.0,
                category="Toy Text",
                supports_render=True,
                stateful=False,
                default_population=180,
                default_target_species=6,
                default_config=_CLASSIC_CONTROL_DEFAULTS,
                default_adaptive_policies=_CLASSIC_CONTROL_ADAPTIVE,
                evaluator_components=["rollout_score", "oracle_score"],
            ),
            ExampleConfig(
                name="Frozen Lake",
                description=(
                    "Gleite über ein vereistes 4×4-Gitter zum Ziel ohne ins Loch zu fallen.\n"
                    "Inputs: Zeile/3, Spalte/3. Belohnung: +1 Ziel, 0 sonst.\n"
                    "Nicht-rutschig. Fitness = Erfolgsrate über 20 Episoden."
                ),
                n_inputs=10, n_outputs=4,
                max_nodes=30, max_connections=100,
                n_initial_hidden=6,
                make_eval=_make_frozenlake_eval(n_episodes=20),
                target_fitness=0.8,
                category="Toy Text",
                supports_render=True,
                default_target_species=4,
                stateful=False,
                default_population=160,
                default_config={
                    **_NOISY_CONTROL_DEFAULTS,
                    "multi_eval": 1,
                    "sigma_penalty": 0.0,
                },
                default_adaptive_policies=_CLASSIC_CONTROL_ADAPTIVE,
                evaluator_components=["rollout_score", "subgoal_score"],
            ),
            ExampleConfig(
                name="Taxi",
                description=(
                    "Fahre einen Taxi-Fahrgast zum Zielort (5×5-Gitter).\n"
                    "Inputs: Zeile/4, Spalte/4, Fahrgast/4, Ziel/3.\n"
                    "Belohnung: +20 Abgabe, -10 illegale Aktion, -1/Schritt."
                ),
                n_inputs=23, n_outputs=6,
                max_nodes=60, max_connections=220,
                n_initial_hidden=12,    # one-hot state channels × 6 actions need capacity
                make_eval=_make_taxi_eval(max_steps=500),
                target_fitness=50.0,  # pickup bonus (30) + delivery bonus (50) + shaping minus steps
                category="Toy Text",
                supports_render=True,
                stateful=False,
                default_population=260,
                default_target_species=8,
                default_config={
                    **_LARGE_CONTROL_DEFAULTS,
                    "early_stop_factor": 0.0,
                },
                default_adaptive_policies=_LARGE_CONTROL_ADAPTIVE,
                evaluator_components=["policy_score", "rollout_score", "subgoal_score"],
            ),
        ]
    except ImportError:
        pass

    return examples
