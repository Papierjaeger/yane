"""GUI helpers for optional research feature configuration.

The training tab owns widgets and layout. This module owns the translation from
widget values into NeuroEvolution configuration so experimental feature wiring
stays testable and out of the main UI flow.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from yane import CPPNGenome, FitnessComponent, generate_genome_from_cppn, hyperneat_substrate
from yane.evolution.population import Population

if TYPE_CHECKING:
    from yane import NeuroEvolution


@dataclass(frozen=True)
class ResearchFeatureConfig:
    n_inputs: int
    n_outputs: int
    max_nodes: int | None
    max_connections: int | None
    population_size: int
    target_species: int
    allow_memory: bool
    output_sanitize: bool
    output_fallback: float
    cppn_substrate: bool = False
    cppn_hidden: int = 2
    matrix_forward: bool = False
    fitness_components: bool = False
    fitness_component_mode: str = "fixed"
    meta_adaptive: bool = False
    module_library: bool = False
    module_insert_rate: float = 0.02


def configure_cppn_substrate_population(
    yane: "NeuroEvolution",
    cfg: ResearchFeatureConfig,
) -> None:
    """Replace the configured seed genome with a CPPN-generated substrate."""
    hidden_layers = (cfg.cppn_hidden,) if cfg.cppn_hidden > 0 else ()
    substrate = hyperneat_substrate(cfg.n_inputs, cfg.n_outputs, hidden_layers=hidden_layers)
    initial = generate_genome_from_cppn(
        CPPNGenome(),
        substrate,
        threshold=0.0,
        tracker=yane._tracker,
    )
    initial.max_nodes = cfg.max_nodes
    initial.max_connections = cfg.max_connections
    initial.allow_memory = cfg.allow_memory
    initial._output_sanitize = cfg.output_sanitize
    initial._output_fallback = cfg.output_fallback
    yane._population = Population(
        max_size=cfg.population_size,
        initial_genome=initial,
        tracker=yane._tracker,
        target_species=cfg.target_species,
    )
    yane._population.elite_count = yane._elite_count
    yane._population.species_elite_count = yane._species_elite_count


def make_gui_fitness_components(n_inputs: int) -> list[FitnessComponent]:
    """Return the lightweight fitness-shaping components used by the GUI."""
    rng = random.Random(43)
    probes = [[rng.uniform(-1.0, 1.0) for _ in range(n_inputs)] for _ in range(3)]

    def _probe_outputs(genome):
        values: list[float] = []
        for inp in probes:
            genome.reset()
            # Clamp to [0, 1]: linear-activation genomes can produce extreme
            # out-of-distribution outputs for these random probes, which would
            # otherwise make output_span/centering scores explode.
            values.extend(max(0.0, min(1.0, float(v))) for v in genome.forward(inp))
        return values

    def _output_span(genome):
        values = _probe_outputs(genome)
        return max(values) - min(values) if values else 0.0

    def _output_centering(genome):
        values = _probe_outputs(genome)
        return abs((sum(values) / len(values)) - 0.5) if values else 0.0

    return [
        FitnessComponent(
            "fewer_hidden",
            lambda g: max(0, len(g.nodes) - len(g.input_nodes) - len(g.output_nodes)),
            weight=0.001,
            maximize=False,
        ),
        FitnessComponent(
            "fewer_connections",
            lambda g: g.connection_count,
            weight=0.0005,
            maximize=False,
        ),
        FitnessComponent("output_span", _output_span, weight=0.01),
        FitnessComponent("output_centering", _output_centering, weight=0.005, maximize=False),
    ]


def apply_research_features(yane: "NeuroEvolution", cfg: ResearchFeatureConfig) -> None:
    """Apply optional research features after base NeuroEvolution configuration."""
    if cfg.meta_adaptive:
        yane.set_meta_adaptive_policies(enabled=True)
    yane.set_module_library(
        enabled=cfg.module_library,
        insert_rate=cfg.module_insert_rate,
    )
    yane.set_matrix_forward(cfg.matrix_forward)
    if cfg.fitness_components:
        yane.set_fitness_components(
            make_gui_fitness_components(cfg.n_inputs),
            mode=("adaptive" if cfg.fitness_component_mode == "Adaptiv" else "fixed"),
            collapse_floor=0.03,
        )


def format_research_diagnostics(mem: dict) -> dict[str, str]:
    """Format research-feature diagnostics for compact GUI labels."""
    hits = mem.get("matrix_forward_hits")
    misses = mem.get("matrix_forward_misses")
    matrix = f"{hits or 0}/{misses or 0}" if hits is not None or misses is not None else "—"

    fc = mem.get("fitness_component_weights")
    if fc:
        weights = fc.get("weights", {})
        reason = fc.get("last_reason", "—")
        preview = ", ".join(f"{k}:{v:.3g}" for k, v in list(weights.items())[:3])
        fitness_components = f"{fc.get('mode', '?')} {reason} {preview}".strip()
    else:
        fitness_components = "—"

    meta = mem.get("meta_adaptive_policies")
    if meta:
        genes = meta.get("global_genes", {})
        meta_policy = (
            f"{meta.get('last_reason', '—')} "
            f"op={genes.get('operator_exploration', 0):.2f} "
            f"budget={genes.get('lamarck_budget', 0)} "
            f"inter={genes.get('interspecies_rate', 0):.3f}"
        )
    else:
        meta_policy = "—"

    module_info = mem.get("module_library")
    if module_info:
        module_library = (
            f"{module_info.get('module_count', 0)} modules, "
            f"uses={module_info.get('total_uses', 0)}, "
            f"reuse={module_info.get('reuse_rate', 0.0):.2f}"
        )
    else:
        module_library = "—"

    return {
        "matrix_forward": matrix,
        "fitness_components": fitness_components,
        "meta_policy": meta_policy,
        "module_library": module_library,
    }
