"""Fitness landscape visualization tools for YANE.

Provides:
- ``GenomeDescriptor`` — fixed-size numeric feature vector for any genome
- ``population_pca()`` — 2-component PCA projection of all evaluated genomes
- ``landscape_snapshot()`` — full snapshot dict with projection + metadata
"""
from __future__ import annotations

import csv
import struct
import zlib
from pathlib import Path
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


def export_landscape_csv(snapshot: dict[str, Any], path: str | Path) -> None:
    """Write a PCA landscape snapshot as CSV.

    The CSV contains one row per projected genome and a small metadata comment
    with the explained variance in the first line.
    """
    xs = list(snapshot.get("x", []))
    ys = list(snapshot.get("y", []))
    fitness = list(snapshot.get("fitness", []))
    species = list(snapshot.get("species_id", []))
    explained = list(snapshot.get("explained_var", [0.0, 0.0]))
    if not (len(xs) == len(ys) == len(fitness) == len(species)):
        raise ValueError("Landscape snapshot arrays must have equal length")

    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        fh.write(
            "# explained_var_pc1="
            f"{explained[0] if explained else 0.0},"
            "explained_var_pc2="
            f"{explained[1] if len(explained) > 1 else 0.0}\n"
        )
        writer = csv.writer(fh)
        writer.writerow(["index", "x", "y", "fitness", "species_id"])
        for idx, (x, y, fit, sid) in enumerate(zip(xs, ys, fitness, species)):
            writer.writerow([idx, x, y, fit, sid])


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _write_rgb_png(path: str | Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(pixels[start:start + stride])
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(_png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
    png.extend(_png_chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def export_landscape_png(
    snapshot: dict[str, Any],
    path: str | Path,
    *,
    width: int = 900,
    height: int = 640,
) -> None:
    """Render a PCA landscape snapshot to a simple PNG scatterplot.

    This intentionally uses only the standard library so the export path works
    in headless benchmark runs and without GUI dependencies.
    """
    xs = list(snapshot.get("x", []))
    ys = list(snapshot.get("y", []))
    fitness = list(snapshot.get("fitness", []))
    species = list(snapshot.get("species_id", []))
    if not (len(xs) == len(ys) == len(fitness) == len(species)):
        raise ValueError("Landscape snapshot arrays must have equal length")
    if width < 120 or height < 120:
        raise ValueError("PNG dimensions must be at least 120x120")

    pixels = bytearray([30, 30, 46] * width * height)
    margin = 52
    plot_w = max(width - 2 * margin, 1)
    plot_h = max(height - 2 * margin, 1)

    def set_pixel(px: int, py: int, color: tuple[int, int, int]) -> None:
        if 0 <= px < width and 0 <= py < height:
            idx = (py * width + px) * 3
            pixels[idx:idx + 3] = bytes(color)

    def draw_line(x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            set_pixel(x1, y1, color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    grid = (42, 42, 62)
    axis = (170, 170, 190)
    for i in range(5):
        gx = margin + int(i * plot_w / 4)
        gy = margin + int(i * plot_h / 4)
        draw_line(gx, margin, gx, height - margin, grid)
        draw_line(margin, gy, width - margin, gy, grid)
    draw_line(margin, height - margin, width - margin, height - margin, axis)
    draw_line(margin, margin, margin, height - margin, axis)

    if not xs:
        _write_rgb_png(path, width, height, pixels)
        return

    min_fit = min(fitness) if fitness else 0.0
    max_fit = max(fitness) if fitness else 0.0
    fit_span = max(max_fit - min_fit, 1e-12)
    palette = [
        (137, 180, 250), (166, 227, 161), (249, 226, 175), (245, 194, 231),
        (148, 226, 213), (250, 179, 135), (203, 166, 247), (243, 139, 168),
    ]

    for x, y, fit, sid in zip(xs, ys, fitness, species):
        px = margin + int(((float(x) + 3.0) / 6.0) * plot_w)
        py = height - margin - int(((float(y) + 3.0) / 6.0) * plot_h)
        base = palette[int(sid) % len(palette)]
        strength = (float(fit) - min_fit) / fit_span
        color = tuple(int(c * (0.45 + 0.55 * strength)) for c in base)
        for oy in range(-3, 4):
            for ox in range(-3, 4):
                if ox * ox + oy * oy <= 9:
                    set_pixel(px + ox, py + oy, color)

    _write_rgb_png(path, width, height, pixels)
