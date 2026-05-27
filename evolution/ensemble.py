"""Ensemble genome wrapper for aggregating multiple top genomes."""
from __future__ import annotations

from typing import Callable

from yane.core.genome import Genome


class EnsembleGenome:
    """Wrapper around multiple genomes that aggregates their outputs.

    The ensemble provides a unified ``forward()`` interface like a single
    Genome, delegating to the top-k individuals and combining their results.

    Parameters
    ----------
    members : list[Genome]
        The top-k genomes (higher fitness = higher weight in ``"weighted"`` mode).
    mode : str
        Aggregation strategy: ``"mean"``, ``"vote"``, or ``"weighted"``.
    """

    def __init__(self, members: list[Genome], mode: str = "mean") -> None:
        if not members:
            raise ValueError("Ensemble must have at least one member")
        self.members = list(members)
        self.mode = mode

    def forward(self, inputs: list[float]) -> list[float]:
        """Run inputs through all ensemble members and aggregate outputs."""
        all_outputs = [g.forward(inputs) for g in self.members]
        n_out = len(all_outputs[0])

        if self.mode == "mean":
            return [
                sum(out[i] for out in all_outputs) / len(all_outputs)
                for i in range(n_out)
            ]

        if self.mode == "vote":
            votes = [0] * n_out
            for out in all_outputs:
                winner = max(range(n_out), key=lambda i: out[i])
                votes[winner] += 1
            total = len(all_outputs)
            return [v / total for v in votes]

        if self.mode == "weighted":
            raw_fitnesses = [float(g.fitness) for g in self.members]
            shifted = [max(f, 0.0) for f in raw_fitnesses]
            total_fit = sum(shifted)
            if total_fit <= 0.0:
                min_fit = min(raw_fitnesses)
                shifted = [f - min_fit for f in raw_fitnesses]
                total_fit = sum(shifted)
            weights = (
                [f / total_fit for f in shifted]
                if total_fit > 0.0
                else [1.0 / len(self.members)] * len(self.members)
            )
            result = [0.0] * n_out
            for w, out in zip(weights, all_outputs):
                for i in range(n_out):
                    result[i] += w * out[i]
            return result

        raise ValueError(f"Unknown ensemble mode: {self.mode!r}")

    def to_python(self, class_name: str = "EnsembleModel") -> str:
        """Export the ensemble as a standalone Python source file.

        Each member genome is emitted as a module-level ``memberN_forward()``
        function.  The ensemble is a module-level
        ``<class_name_lower>_forward()`` function that aggregates outputs.

        Returns
        -------
        str
            Complete Python source code for the ensemble model.
        """
        from yane.evolution.genome_export import genome_to_python

        lines: list[str] = []
        member_fns: list[str] = []
        helpers_emitted = False

        for i, g in enumerate(self.members):
            fn_src = genome_to_python(g)

            # Split into lines; the source starts with helpers then def forward.
            src_lines = fn_src.split("\n")
            current_fn_lines: list[str] = []
            in_forward = False
            for line in src_lines:
                stripped = line.strip()
                if stripped.startswith("def forward"):
                    in_forward = True
                    current_fn_lines.append(f"def member{i}_forward(inputs):")
                elif in_forward:
                    if not stripped:
                        continue
                    current_fn_lines.append(line)
                elif not helpers_emitted:
                    # Module-level helpers: import math, _swish, etc.
                    lines.append(line)
            helpers_emitted = True
            member_fns.append("\n".join(current_fn_lines))

        # Append members after helpers
        for i, fn_text in enumerate(member_fns):
            lines.append(f"\n# --- Member {i} ---")
            lines.append(fn_text)

        # Ensemble wrapper
        safe_name = class_name.lower().replace(" ", "_")
        lines.append(f"\n\ndef {safe_name}_forward(inputs):")
        lines.append(f'    """Ensemble of {len(self.members)} genomes ({self.mode} aggregation)."""')
        lines.append("    all_outputs = [")
        for i in range(len(self.members)):
            lines.append(f"        member{i}_forward(inputs),")
        lines.append("    ]")
        lines.append("    n_out = len(all_outputs[0])")

        if self.mode == "mean":
            lines.extend([
                "    return [",
                "        sum(out[i] for out in all_outputs) / len(all_outputs)",
                "        for i in range(n_out)",
                "    ]",
            ])
        elif self.mode == "vote":
            lines.extend([
                "    votes = [0] * n_out",
                "    for out in all_outputs:",
                "        winner = max(range(n_out), key=lambda i: out[i])",
                "        votes[winner] += 1",
                "    total = len(all_outputs)",
                "    return [v / total for v in votes]",
            ])
        elif self.mode == "weighted":
            fitnesses = [max(g.fitness, 0.0) for g in self.members]
            lines.append(f"    fitnesses = {fitnesses!r}")
            lines.extend([
                "    total_fit = sum(fitnesses) or 1.0",
                "    weights = [f / total_fit for f in fitnesses]",
                "    result = [0.0] * n_out",
                "    for w, out in zip(weights, all_outputs):",
                "        for i in range(n_out):",
                "            result[i] += w * out[i]",
                "    return result",
            ])
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"EnsembleGenome(members={len(self.members)}, "
                f"mode={self.mode!r})")
