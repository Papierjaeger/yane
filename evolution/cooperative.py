"""Multi-Agent Cooperation — Kooperative Co-Evolution.

N Genome evolvieren gemeinsam und werden als Team evaluiert.
Das Credit-Assignment bestimmt, wie die Team-Fitness auf die einzelnen
Agenten verteilt wird.

**Credit-Assignment-Modi:**

``"shared"``
    Alle Agenten erhalten die Team-Fitness (gleiche Belohnung).

``"difference"``
    Shapley-Approximation: ``credit_i = f(team) - f(team_without_i)``.
    Misst den marginalen Beitrag jedes Agenten.
    Erfordert N+1 Evaluierungen pro Team.

``"individual"``
    Jeder Agent wird einzeln evaluiert (keine Kooperation).
    Nützlich als Baseline.

``"hierarchical"``
    Agent 0 erhält 50% der Team-Fitness, Agent 1 bekommt 30%, etc.
    Modelliert hierarchische Strukturen (Manager → Arbeiter).

**Rollen-Spezialisierung:**
Wenn ``role_specialization=True``: Agenten mit ähnlichen Outputs auf
denselben Eingaben erhalten einen Diversitäts-Penalty.
``role_similarity`` sinkt über Generationen → Agenten spezialisieren sich.

Integration::

    result = yane.train_cooperative(
        team_fitness_fn=lambda agents: sum(a.forward([0.5])[0] for a in agents),
        n_agents=3,
        credit="difference",
        n_generations=100,
    )
    print(result.role_similarity_history)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from yane.core.genome import Genome

TeamFitnessFn = Callable[[list["Genome"]], float]


# ---------------------------------------------------------------------------
# Credit Assignment
# ---------------------------------------------------------------------------

def assign_shared(
    team_fitness: float,
    agents: list["Genome"],
    team_fitness_fn: TeamFitnessFn | None = None,
) -> list[float]:
    """All agents share the same team fitness."""
    return [team_fitness] * len(agents)


def assign_difference(
    team_fitness: float,
    agents: list["Genome"],
    team_fitness_fn: TeamFitnessFn,
) -> list[float]:
    """Shapley-approximated difference reward.

    ``credit_i = f(team) - f(team_without_i)``

    Higher credit = agent was important.  Requires len(agents) extra evaluations.
    """
    credits = []
    for i, agent in enumerate(agents):
        team_without_i = [a for j, a in enumerate(agents) if j != i]
        if team_without_i:
            try:
                fit_without_i = team_fitness_fn(team_without_i)
            except Exception:
                fit_without_i = 0.0
        else:
            fit_without_i = 0.0
        credits.append(team_fitness - fit_without_i)
    return credits


def assign_individual(
    team_fitness: float,
    agents: list["Genome"],
    individual_fitness_fn: Callable[["Genome"], float] | None = None,
) -> list[float]:
    """Each agent is evaluated independently."""
    if individual_fitness_fn is not None:
        credits = []
        for agent in agents:
            try:
                credits.append(float(individual_fitness_fn(agent)))
            except Exception:
                credits.append(0.0)
        return credits
    # Fallback: return current fitness
    return [getattr(a, "fitness", 0.0) for a in agents]


def assign_hierarchical(
    team_fitness: float,
    agents: list["Genome"],
    team_fitness_fn: TeamFitnessFn | None = None,
) -> list[float]:
    """Hierarchical credit: agent 0 gets most, descending."""
    n = len(agents)
    if n == 0:
        return []
    # Weights: [0.5, 0.3, 0.15, 0.05, ...] (geometric decay, sum ≤ 1)
    weights = [0.5 ** (i + 1) for i in range(n)]
    total = sum(weights)
    weights = [w / total for w in weights]
    return [team_fitness * w for w in weights]


CREDIT_FUNCTIONS = {
    "shared": assign_shared,
    "difference": assign_difference,
    "individual": assign_individual,
    "hierarchical": assign_hierarchical,
}


# ---------------------------------------------------------------------------
# Role Similarity
# ---------------------------------------------------------------------------

def compute_role_similarity(
    agents: list["Genome"],
    probe_inputs: list[list[float]],
) -> float:
    """Compute average pairwise cosine similarity of agent outputs.

    High value (→ 1.0) = agents behave identically (no specialization).
    Low value (→ 0.0) = agents behave differently (good specialization).

    Parameters
    ----------
    agents :
        List of agents to compare.
    probe_inputs :
        Input vectors used to measure behaviour.

    Returns
    -------
    float
        Average cosine similarity ∈ [0, 1].
    """
    if len(agents) < 2:
        return 0.0

    # Collect output vectors for each agent
    outputs: list[list[float]] = []
    for agent in agents:
        agent.reset()
        flat: list[float] = []
        for inp in probe_inputs:
            try:
                out = agent.forward(inp)
                flat.extend(float(v) for v in out)
            except Exception:
                flat.extend([0.0] * max(1, len(agents[0].output_nodes)))
        outputs.append(flat)

    # Average pairwise cosine similarity
    total_sim = 0.0
    n_pairs = 0
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            sim = _cosine_similarity(outputs[i], outputs[j])
            total_sim += sim
            n_pairs += 1
    return total_sim / max(1, n_pairs)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def role_diversity_penalty(similarity: float, diversity_weight: float = 0.1) -> float:
    """Fitness penalty proportional to role similarity."""
    return -diversity_weight * similarity


# ---------------------------------------------------------------------------
# CooperativeSystem
# ---------------------------------------------------------------------------

@dataclass
class CooperativeResult:
    """Result of one ``train_cooperative()`` run."""

    agents: list["Genome"]
    """Final evolved agents (sorted by fitness, best first within each team)."""

    team_fitness_history: list[float]
    """Mean team fitness per generation."""

    role_similarity_history: list[float]
    """Mean role similarity per generation."""

    n_generations: int

    @property
    def best_agent(self) -> "Genome":
        """Agent with the highest individual fitness."""
        return max(self.agents, key=lambda g: getattr(g, "fitness", 0.0))

    @property
    def mean_final_fitness(self) -> float:
        return sum(getattr(g, "fitness", 0.0) for g in self.agents) / max(1, len(self.agents))


class CooperativeSystem:
    """Manages N cooperative agents with configurable credit assignment.

    Parameters
    ----------
    n_agents :
        Number of cooperative agents.
    credit :
        Credit-assignment mode (see module docstring).
    role_specialization :
        When True, measure role similarity and apply diversity penalty.
    diversity_weight :
        Weight of the diversity penalty (only used when role_specialization=True).
    """

    def __init__(
        self,
        n_agents: int = 3,
        credit: str = "shared",
        role_specialization: bool = False,
        diversity_weight: float = 0.1,
    ) -> None:
        if credit not in CREDIT_FUNCTIONS:
            raise ValueError(f"credit must be one of {list(CREDIT_FUNCTIONS)}, got {credit!r}")
        self.n_agents = n_agents
        self.credit = credit
        self.role_specialization = role_specialization
        self.diversity_weight = diversity_weight
        self._agents: list["Genome"] = []
        self._role_similarity_history: list[float] = []
        self._team_fitness_history: list[float] = []

    def set_agents(self, agents: list["Genome"]) -> None:
        self._agents = list(agents)

    def evaluate_team(
        self,
        agents: list["Genome"],
        team_fitness_fn: TeamFitnessFn,
        probe_inputs: list[list[float]] | None = None,
    ) -> None:
        """Evaluate team and assign credit to each agent.

        Updates each agent's ``.fitness`` attribute in-place.

        Parameters
        ----------
        agents :
            Team of agents to evaluate.
        team_fitness_fn :
            ``(agents) -> float`` — team-level fitness.
        probe_inputs :
            Inputs used for role-similarity measurement.
        """
        try:
            team_fitness = float(team_fitness_fn(agents))
        except Exception:
            team_fitness = 0.0

        # Credit assignment
        credit_fn = CREDIT_FUNCTIONS[self.credit]
        if self.credit == "difference":
            credits = credit_fn(team_fitness, agents, team_fitness_fn)
        elif self.credit == "individual":
            credits = credit_fn(team_fitness, agents)
        else:
            credits = credit_fn(team_fitness, agents)

        # Role specialization diversity penalty
        if self.role_specialization and probe_inputs:
            sim = compute_role_similarity(agents, probe_inputs)
            penalty = role_diversity_penalty(sim, self.diversity_weight)
            credits = [c + penalty for c in credits]
        else:
            sim = 0.0

        # Assign to genomes
        for agent, c in zip(agents, credits):
            agent.fitness = float(c)

        self._team_fitness_history.append(team_fitness)
        self._role_similarity_history.append(sim)

    @property
    def role_similarity_history(self) -> list[float]:
        return list(self._role_similarity_history)

    @property
    def team_fitness_history(self) -> list[float]:
        return list(self._team_fitness_history)


# ---------------------------------------------------------------------------
# Standalone training loop
# ---------------------------------------------------------------------------

def train_cooperative(
    agents: list["Genome"],
    team_fitness_fn: TeamFitnessFn,
    mutation_fn: Callable[["Genome"], "Genome"] | None = None,
    n_generations: int = 100,
    n_survivors: int | None = None,
    credit: str = "shared",
    role_specialization: bool = False,
    diversity_weight: float = 0.1,
    probe_inputs: list[list[float]] | None = None,
    seed: int | None = None,
) -> CooperativeResult:
    """Evolve cooperative agents via team fitness and credit assignment.

    Parameters
    ----------
    agents :
        Initial list of agents (one population, all evaluated together).
    team_fitness_fn :
        ``(agents) -> float`` — team fitness signal.
    mutation_fn :
        ``(genome) -> genome`` — produce offspring.
    n_generations :
        Training iterations.
    n_survivors :
        Elites kept per generation.
    credit :
        Credit-assignment mode.
    role_specialization :
        Apply diversity penalty for similar agents.
    probe_inputs :
        Inputs for role-similarity measurement.
    seed :
        RNG seed.

    Returns
    -------
    CooperativeResult
    """
    system = CooperativeSystem(
        n_agents=len(agents),
        credit=credit,
        role_specialization=role_specialization,
        diversity_weight=diversity_weight,
    )
    rng = random.Random(seed)
    n = len(agents)
    survivors = n_survivors or max(1, n // 2)

    # Default probe inputs: random unit vectors
    if probe_inputs is None and role_specialization:
        n_in = len(agents[0].input_nodes) if agents else 1
        probe_inputs = [[rng.uniform(-1, 1) for _ in range(n_in)] for _ in range(5)]

    for gen in range(n_generations):
        # Evaluate team and assign credit
        system.evaluate_team(agents, team_fitness_fn, probe_inputs)

        # Selection + mutation
        sorted_agents = sorted(agents, key=lambda g: g.fitness, reverse=True)
        top = sorted_agents[:survivors]
        while len(top) < n:
            parent = rng.choice(sorted_agents[:max(1, survivors)])
            if mutation_fn is not None:
                top.append(mutation_fn(parent))
            else:
                child = parent.copy()
                top.append(child)
        agents = top

    # Final evaluation
    system.evaluate_team(agents, team_fitness_fn, probe_inputs)

    return CooperativeResult(
        agents=sorted(agents, key=lambda g: g.fitness, reverse=True),
        team_fitness_history=system.team_fitness_history,
        role_similarity_history=system.role_similarity_history,
        n_generations=n_generations,
    )
