"""Fitness landscape visualization tools for YANE.

Provides:
- ``GenomeDescriptor`` — fixed-size numeric feature vector for any genome
- ``population_pca()`` — 2-component PCA projection of all evaluated genomes
- ``landscape_snapshot()`` — full snapshot dict with projection + metadata
"""
from __future__ import annotations

from typing import Any

from yane.core.genome import Genome


def genome_descriptor_vector(genome: Genome) -> list[float]:
    """Extract a fixed-size numeric feature vector from a genome.

    The vector comprises::

        [n_nodes, n_connections, n_enabled, n_disabled, mean_weight,
         std_weight, mean_bias, std_bias, n_inputs, n_outputs,
         n_hidden, species_id_hash]

    These features are topology-agnostic and comparable across genomes of
    different sizes.  The vector length is fixed (12 floats).
    """
    weights = []
    biases = []
    n_enabled = 0
    n_disabled = 0

    for src in genome.nodes:
        biases.append(float(src.bias))
        for conn in src.connections:
            weights.append(float(conn.weight))
            if conn.enabled:
                n_enabled += 1
            else:
                n_disabled += 1

    n_nodes = len(genome.nodes)
    n_conns = n_enabled + n_disabled
    n_inputs = len(genome.input_nodes)
    n_outputs = len(genome.output_nodes)
    n_hidden = n_nodes - n_inputs - n_outputs

    mean_w = sum(weights) / len(weights) if weights else 0.0
    std_w = (sum((w - mean_w) ** 2 for w in weights) / len(weights)) ** 0.5 if weights else 0.0
    mean_b = sum(biases) / len(biases) if biases else 0.0
    std_b = (sum((b - mean_b) ** 2 for b in biases) / len(biases)) ** 0.5 if biases else 0.0

    # Species ID as a deterministic hash for coloring
    sid = id(getattr(genome, "_last_species_id", None))

    return [
        float(n_nodes), float(n_conns), float(n_enabled), float(n_disabled),
        round(mean_w, 6), round(std_w, 6),
        round(mean_b, 6), round(std_b, 6),
        float(n_inputs), float(n_outputs), float(n_hidden),
        float(sid % 1000),  # bounded hash for stability
    ]


def population_pca(
    genomes: list[Genome],
) -> dict[str, Any]:
    """Compute a 2-component PCA projection of *genomes* via SVD.

    Returns a dict::

        {"x": list[float], "y": list[float],
         "fitness": list[float], "species_id": list[int],
         "explained_var": list[float]}

    Uses only the Python stdlib + basic math.  The data is mean-centred
    but not scaled (features are already on comparable scales).
    """
    n = len(genomes)
    if n == 0:
        return {"x": [], "y": [], "fitness": [], "species_id": [],
                "explained_var": [0.0, 0.0]}

    # Build feature matrix (n × 12)
    data = [genome_descriptor_vector(g) for g in genomes]
    n_features = len(data[0])
    # Mean-centre
    means = [sum(row[j] for row in data) / n for j in range(n_features)]
    X = [[row[j] - means[j] for j in range(n_features)] for row in data]

    # SVD via covariance matrix (works for n_features << n or n_features > n)
    # Build X^T * X (12×12)
    cov = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(n_features):
            cov[i][j] = sum(X[k][i] * X[k][j] for k in range(n)) / max(n - 1, 1)

    # Power iteration for top 2 eigenvectors
    def _power_iteration(mat, n_iter: int = 50):
        vec = [1.0] * n_features
        for _ in range(n_iter):
            # Multiply mat × vec
            new_vec = [sum(mat[i][j] * vec[j] for j in range(n_features))
                       for i in range(n_features)]
            norm = sum(v * v for v in new_vec) ** 0.5
            if norm > 1e-12:
                vec = [v / norm for v in new_vec]
            else:
                break
        eigenvalue = sum(vec[i] * sum(cov[i][j] * vec[j] for j in range(n_features))
                        for i in range(n_features))
        return vec, eigenvalue

    # First component
    v1, e1 = _power_iteration(cov)

    # Second component: deflate
    cov_deflated = [
        [cov[i][j] - e1 * v1[i] * v1[j] for j in range(n_features)]
        for i in range(n_features)
    ]
    v2, e2 = _power_iteration(cov_deflated)

    total_var = sum(cov[i][i] for i in range(n_features))
    explained = [e1 / total_var if total_var > 0 else 0.0,
                 e2 / total_var if total_var > 0 else 0.0]

    # Project
    proj_x = [sum(X[i][j] * v1[j] for j in range(n_features)) for i in range(n)]
    proj_y = [sum(X[i][j] * v2[j] for j in range(n_features)) for i in range(n)]

    # Normalize projections to [−3, 3] for stable rendering
    def _normalize(vals):
        mx = max(abs(v) for v in vals) if vals else 1.0
        return [v * 3.0 / mx if mx > 0 else v for v in vals]

    return {
        "x": _normalize(proj_x),
        "y": _normalize(proj_y),
        "fitness": [float(g.fitness) for g in genomes],
        "species_id": [hash(getattr(g, "_last_species_id", None)) % 1000
                       for g in genomes],
        "explained_var": [round(v, 4) for v in explained],
    }
