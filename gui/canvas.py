"""Network topology canvas and fitness history chart."""
from __future__ import annotations
import math
import random

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF, QRectF, QSize
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath


_C_BG       = QColor("#1e1e2e")
_C_INPUT    = QColor("#4a90d9")
_C_HIDDEN   = QColor("#7c7c9c")
_C_MEMORY   = QColor("#c97bdb")   # persistent hidden (memory) node — warm violet
_C_OUTPUT   = QColor("#5ba85a")
_C_BORDER   = QColor("#2d2d4e")
_C_POS_CONN = QColor("#4a90d9")
_C_NEG_CONN = QColor("#d94a4a")
_C_TEXT     = QColor("#cccccc")
_C_GRID     = QColor("#2a2a3e")
_C_CHART    = QColor("#4CAF50")
_C_MEM_RING = QColor("#f0c0ff")   # outer ring on memory nodes
_NODE_R     = 13


# ---------------------------------------------------------------------------
# Force-directed layout
# ---------------------------------------------------------------------------

def _force_layout(
    nodes_all: list,
    inputs: list,
    outputs: list,
    hidden: list,
    w: int,
    h: int,
    iterations: int = 120,
) -> dict[int, list[float]]:
    """Fruchterman-Reingold spring layout.

    Input and output nodes are fixed (left/right columns).
    Hidden nodes are placed by minimising a spring energy:
    - Attraction along edges proportional to |weight|
    - Repulsion between every pair of nodes
    - Soft boundary keep nodes inside the canvas
    """
    pad = _NODE_R + 18

    def spread_y(idx: int, total: int) -> float:
        if total == 1:
            return h / 2.0
        return pad + (h - 2 * pad) * idx / (total - 1)

    pos: dict[int, list[float]] = {}

    # Fixed anchors
    x_in  = pad + _NODE_R
    x_out = w - pad - _NODE_R
    for i, n in enumerate(inputs):
        pos[id(n)] = [x_in, spread_y(i, len(inputs))]
    for i, n in enumerate(outputs):
        pos[id(n)] = [x_out, spread_y(i, len(outputs))]

    if not hidden:
        return pos

    # Initialise hidden nodes in the centre with small random offsets
    rng = random.Random(42)
    cx, cy = (x_in + x_out) / 2.0, h / 2.0
    spread = min(w, h) * 0.15
    for n in hidden:
        pos[id(n)] = [
            cx + rng.uniform(-spread, spread),
            cy + rng.uniform(-spread, spread),
        ]

    # Build weighted edge list (skip self-loops for layout)
    edges: list[tuple[int, int, float]] = []
    for src in nodes_all:
        for conn in src.connections:
            if conn.target is not src and id(conn.target) in pos:
                edges.append((id(src), id(conn.target), abs(conn.weight)))

    hidden_ids = {id(n) for n in hidden}
    all_ids    = list(pos.keys())

    # Area-based optimal distance (Fruchterman-Reingold)
    area = (w - 2 * pad) * (h - 2 * pad)
    k    = math.sqrt(area / max(len(all_ids), 1)) * 0.7

    for step in range(iterations):
        temp = k * (1.0 - step / iterations) * 0.5 + k * 0.05

        disp: dict[int, list[float]] = {nid: [0.0, 0.0] for nid in hidden_ids}

        # Repulsion between every pair
        for i in range(len(all_ids)):
            aid = all_ids[i]
            ax, ay = pos[aid]
            for j in range(i + 1, len(all_ids)):
                bid = all_ids[j]
                bx, by = pos[bid]
                dx, dy = ax - bx, ay - by
                dist2  = dx * dx + dy * dy
                dist   = math.sqrt(dist2) if dist2 > 0.01 else 0.1
                f_rep  = (k * k) / dist          # F-R repulsion
                ux, uy = dx / dist, dy / dist
                if aid in disp:
                    disp[aid][0] += f_rep * ux
                    disp[aid][1] += f_rep * uy
                if bid in disp:
                    disp[bid][0] -= f_rep * ux
                    disp[bid][1] -= f_rep * uy

        # Attraction along weighted edges
        for (sid, tid, w_edge) in edges:
            if sid not in pos or tid not in pos:
                continue
            dx = pos[tid][0] - pos[sid][0]
            dy = pos[tid][1] - pos[sid][1]
            dist2 = dx * dx + dy * dy
            dist  = math.sqrt(dist2) if dist2 > 0.01 else 0.1
            # Stronger connection = stronger pull; scale so that weight≈1 gives
            # a pull similar to the F-R attractive force
            f_attr = (dist * dist / k) * (0.4 + 0.8 * w_edge)
            ux, uy = dx / dist, dy / dist
            if sid in disp:
                disp[sid][0] += f_attr * ux
                disp[sid][1] += f_attr * uy
            if tid in disp:
                disp[tid][0] -= f_attr * ux
                disp[tid][1] -= f_attr * uy

        # Apply displacement, capped by temperature
        for nid in hidden_ids:
            fx, fy = disp[nid]
            mag = math.sqrt(fx * fx + fy * fy)
            if mag > 0.01:
                scale = min(mag, temp) / mag
                fx *= scale
                fy *= scale
            x = pos[nid][0] + fx
            y = pos[nid][1] + fy
            # Soft clamp: stay inside canvas with padding
            x = max(pad + _NODE_R, min(w - pad - _NODE_R, x))
            y = max(pad + _NODE_R, min(h - pad - _NODE_R, y))
            pos[nid] = [x, y]

    return pos


# ---------------------------------------------------------------------------
# NetworkCanvas
# ---------------------------------------------------------------------------

class NetworkCanvas(QWidget):
    """Draws the topology of a Genome using a force-directed layout.

    Inputs are pinned to the left, outputs to the right.  Hidden nodes
    cluster near strongly-connected neighbours and spread away from nodes
    they share no connections with.
    """

    _REPAINT_INTERVAL_S = 0.5

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._genome = None
        self._pos_cache: dict = {}
        self._cached_size: tuple[int, int] = (0, 0)
        self._last_repaint: float = 0.0
        self.setMinimumSize(260, 220)

    def set_genome(self, genome) -> None:
        import time as _t
        # Do NOT call _clear() on the old genome — it is still owned by the
        # caller (Population); clearing it here would destroy its node data.
        self._genome = genome
        self._pos_cache = {}
        now = _t.perf_counter()
        if now - self._last_repaint >= self._REPAINT_INTERVAL_S:
            self._last_repaint = now
            self.update()

    # ------------------------------------------------------------------

    def _positions(self, w: int, h: int) -> dict:
        from yane.core.node import NodeType
        g = self._genome
        if not g:
            return {}
        inputs  = g.input_nodes
        outputs = g.output_nodes
        hidden  = [n for n in g.nodes if n.type == NodeType.HIDDEN]
        raw = _force_layout(g.nodes, inputs, outputs, hidden, w, h)
        return {k: QPointF(v[0], v[1]) for k, v in raw.items()}

    # ------------------------------------------------------------------

    def _arrow_head(self, painter: QPainter, src: QPointF, dst: QPointF,
                    color: QColor, size: int = 7) -> None:
        dx, dy = dst.x() - src.x(), dst.y() - src.y()
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        tx = dst.x() - ux * (_NODE_R + 2)
        ty = dst.y() - uy * (_NODE_R + 2)
        path = QPainterPath()
        path.moveTo(tx, ty)
        path.lineTo(tx - ux * size - uy * size * 0.45,
                    ty - uy * size + ux * size * 0.45)
        path.lineTo(tx - ux * size + uy * size * 0.45,
                    ty - uy * size - ux * size * 0.45)
        path.closeSubpath()
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), _C_BG)

            if not self._genome or not self._genome.nodes:
                painter.setPen(_C_TEXT)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                                 "No genome loaded")
                return

            w, h = self.width(), self.height()
            if not self._pos_cache or self._cached_size != (w, h):
                self._pos_cache  = self._positions(w, h)
                self._cached_size = (w, h)
            pos = self._pos_cache

            from yane.core.node import NodeType

            # --- Connections ---
            _conn_color = QColor()
            _pr, _pg, _pb = _C_POS_CONN.red(), _C_POS_CONN.green(), _C_POS_CONN.blue()
            _nr, _ng, _nb = _C_NEG_CONN.red(), _C_NEG_CONN.green(), _C_NEG_CONN.blue()
            _pen = QPen(_conn_color, 1.0)

            for node in self._genome.nodes:
                src = pos.get(id(node))
                if src is None:
                    continue
                for conn in node.connections:
                    dst = pos.get(id(conn.target))
                    if dst is None:
                        continue

                    # Alpha and width both scale with |weight|
                    w_abs  = abs(conn.weight)
                    alpha  = min(220, max(35, int(w_abs * 150 + 35)))
                    lwidth = min(3.5, max(0.8, w_abs * 1.8))

                    if conn.weight >= 0:
                        _conn_color.setRgb(_pr, _pg, _pb, alpha)
                    else:
                        _conn_color.setRgb(_nr, _ng, _nb, alpha)

                    _pen.setColor(_conn_color)
                    _pen.setWidthF(lwidth)
                    painter.setPen(_pen)

                    if conn.target is node:
                        # Self-loop: small arc above the node
                        r = _NODE_R * 1.6
                        loop_rect = QRectF(
                            src.x() - r * 0.5,
                            src.y() - _NODE_R - r * 1.8,
                            r, r,
                        )
                        painter.drawArc(loop_rect, 0, 360 * 16)
                    else:
                        painter.drawLine(src, dst)
                        self._arrow_head(painter, src, dst, _conn_color,
                                         size=max(5, int(5 + lwidth)))

            # --- Nodes ---
            font = QFont()
            font.setPointSize(7)
            font.setBold(True)
            painter.setFont(font)

            for node in self._genome.nodes:
                p = pos.get(id(node))
                if p is None:
                    continue

                is_memory = (node.type == NodeType.HIDDEN and node.persist_value)
                color = (_C_INPUT  if node.type == NodeType.INPUT  else
                         _C_OUTPUT if node.type == NodeType.OUTPUT else
                         _C_MEMORY if is_memory else
                         _C_HIDDEN)

                # Memory nodes: dashed outer ring to signal persistence
                if is_memory:
                    ring_pen = QPen(_C_MEM_RING, 1.5, Qt.PenStyle.DashLine)
                    painter.setPen(ring_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    r_ring = _NODE_R + 5
                    painter.drawEllipse(
                        QRectF(p.x() - r_ring, p.y() - r_ring,
                               r_ring * 2, r_ring * 2)
                    )

                # Glow: faint larger circle underneath
                glow = QColor(color)
                glow.setAlpha(45)
                painter.setBrush(QBrush(glow))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(
                    QRectF(p.x() - _NODE_R - 3, p.y() - _NODE_R - 3,
                           (_NODE_R + 3) * 2, (_NODE_R + 3) * 2)
                )

                # Main circle
                border_color = _C_MEM_RING if is_memory else _C_BORDER
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(border_color, 2.0 if is_memory else 1.5))
                painter.drawEllipse(
                    QRectF(p.x() - _NODE_R, p.y() - _NODE_R,
                           _NODE_R * 2, _NODE_R * 2)
                )

                # Activation label; memory nodes show "M" superscript dot
                label = node.activation.value[0].upper()
                painter.setPen(_C_BG)
                painter.drawText(
                    QRectF(p.x() - _NODE_R, p.y() - _NODE_R,
                           _NODE_R * 2, _NODE_R * 2),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )

                # Memory indicator: small filled dot in top-right of node
                if is_memory:
                    dot_r = 3.5
                    painter.setBrush(QBrush(_C_MEM_RING))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(
                        QRectF(p.x() + _NODE_R - dot_r * 1.8,
                               p.y() - _NODE_R - dot_r * 0.5,
                               dot_r * 2, dot_r * 2)
                    )

        finally:
            painter.end()

    def sizeHint(self) -> QSize:
        return QSize(280, 260)


# ---------------------------------------------------------------------------

class FitnessChart(QWidget):
    """Scrolling line chart for fitness history with best-so-far line."""

    _MAX_POINTS = 400

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._history: list[float] = []
        self._best_history: list[float] = []
        self._lo: float = 0.0
        self._hi: float = 0.0
        self.setMinimumHeight(110)

    def add_point(self, fitness: float) -> None:
        self._history.append(fitness)
        best_so_far = (max(self._best_history[-1] if self._best_history else fitness,
                           fitness))
        self._best_history.append(best_so_far)

        if len(self._history) > self._MAX_POINTS:
            self._history     = self._history[-self._MAX_POINTS:]
            self._best_history = self._best_history[-self._MAX_POINTS:]
            self._lo, self._hi = min(self._history), max(self._best_history)
        elif len(self._history) == 1:
            self._lo = self._hi = fitness
        else:
            self._lo = min(self._lo, fitness)
            self._hi = max(self._hi, best_so_far)
        self.update()

    def clear(self) -> None:
        self._history.clear()
        self._best_history.clear()
        self._lo = self._hi = 0.0
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            self._paint(painter)
        finally:
            painter.end()

    @staticmethod
    def _percentile(sorted_vals: list[float], p: float) -> float:
        """p in [0, 1]. Requires sorted input."""
        if not sorted_vals:
            return 0.0
        idx = p * (len(sorted_vals) - 1)
        lo_i, hi_i = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
        frac = idx - lo_i
        return sorted_vals[lo_i] * (1 - frac) + sorted_vals[hi_i] * frac

    def _paint(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _C_BG)

        w, h = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 48, 10, 10, 22

        # Grid lines
        painter.setPen(QPen(_C_GRID, 1))
        for i in range(4):
            y = pad_t + (h - pad_t - pad_b) * i / 3
            painter.drawLine(pad_l, int(y), w - pad_r, int(y))

        if len(self._history) < 2:
            painter.setPen(_C_TEXT)
            font = QFont(); font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Waiting for data…")
            return

        n = len(self._history)

        # ── Outlier-resistant y-axis range (IQR method) ───────────────────
        # Lower fence = Q1 - 1.5×IQR: the standard Tukey outlier boundary.
        # Values below the fence are clipped to the chart edge and annotated.
        # The best-so-far maximum is always fully visible (sets the top).
        sorted_h = sorted(self._history)
        actual_lo = sorted_h[0]
        hi = max(self._best_history) if self._best_history else sorted_h[-1]

        q1 = self._percentile(sorted_h, 0.25)
        q3 = self._percentile(sorted_h, 0.75)
        iqr = q3 - q1
        lo_fence = (q1 - 1.5 * iqr) if iqr > 0 else actual_lo
        lo = max(actual_lo, lo_fence)

        # Add 10 % padding so lines don't touch the edges
        span = hi - lo if hi != lo else 1e-6
        lo -= span * 0.10
        hi += span * 0.05
        if lo == hi:
            hi = lo + 1e-6

        # Clipping helper: keep y inside [pad_t, h - pad_b]
        y_top = float(pad_t)
        y_bot = float(h - pad_b)

        def px(i: int) -> float:
            return pad_l + (w - pad_l - pad_r) * i / (n - 1)

        def py(v: float) -> float:
            raw = h - pad_b - (h - pad_t - pad_b) * (v - lo) / (hi - lo)
            return max(y_top, min(y_bot, raw))

        def is_clipped(v: float) -> bool:
            return v < lo or v > hi

        # ── Current-fitness line (thin, dimmed, clipped) ──────────────────
        pen_cur = QPen(QColor("#4CAF5088"), 1)
        pen_clip = QPen(QColor("#ff6b6b60"), 1, Qt.PenStyle.DotLine)
        for i in range(1, n):
            v0, v1 = self._history[i - 1], self._history[i]
            p = pen_clip if (is_clipped(v0) or is_clipped(v1)) else pen_cur
            painter.setPen(p)
            painter.drawLine(int(px(i - 1)), int(py(v0)),
                             int(px(i)),     int(py(v1)))

        # ── Best-so-far line (bright, always fully visible) ───────────────
        pen_best = QPen(_C_CHART, 2)
        painter.setPen(pen_best)
        for i in range(1, len(self._best_history)):
            painter.drawLine(int(px(i - 1)), int(py(self._best_history[i - 1])),
                             int(px(i)),     int(py(self._best_history[i])))

        # ── Outlier indicator ─────────────────────────────────────────────
        # If the true minimum is below the visible lo, draw a subtle marker
        if actual_lo < lo:
            painter.setPen(QPen(QColor("#ff6b6b99"), 1))
            painter.drawLine(pad_l, int(y_bot), w - pad_r, int(y_bot))
            font_sm = QFont(); font_sm.setPointSize(6)
            painter.setFont(font_sm)
            painter.setPen(QColor("#ff6b6b"))
            painter.drawText(pad_l + 2, int(y_bot) - 2, f"min {actual_lo:.3g}")

        # ── Axis labels ───────────────────────────────────────────────────
        font = QFont(); font.setPointSize(7)
        painter.setFont(font)
        painter.setPen(_C_TEXT)
        painter.drawText(2, pad_t + 10, f"{hi:.3g}")
        painter.drawText(2, pad_t + int((h - pad_t - pad_b) * 0.5) + 5, f"{(lo + hi) / 2:.3g}")
        painter.drawText(2, h - pad_b,  f"{lo:.3g}")
        painter.drawText(pad_l,          h - 5, "0")
        painter.drawText(w - pad_r - 26, h - 5, str(n))

        # ── Legend ────────────────────────────────────────────────────────
        painter.setPen(QPen(_C_CHART, 2))
        painter.drawLine(w - 90, pad_t + 8, w - 70, pad_t + 8)
        painter.setPen(_C_TEXT)
        painter.drawText(w - 66, pad_t + 12, "best")


# ---------------------------------------------------------------------------

class SpeciesChart(QWidget):
    """Compact line chart that tracks species count over training iterations."""

    _MAX_POINTS = 400
    _C_LINE = QColor("#fab387")  # peach

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._history: list[int] = []
        self.setMinimumHeight(65)
        self.setMaximumHeight(75)

    def add_point(self, count: int) -> None:
        self._history.append(count)
        if len(self._history) > self._MAX_POINTS:
            self._history = self._history[-self._MAX_POINTS:]
        self.update()

    def clear(self) -> None:
        self._history.clear()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), _C_BG)
            w, h = self.width(), self.height()
            pad_l, pad_r, pad_t, pad_b = 28, 8, 5, 14

            font = QFont(); font.setPointSize(7)
            painter.setFont(font)

            if len(self._history) < 2:
                painter.setPen(_C_TEXT)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Warte auf Daten…")
                return

            n = len(self._history)
            hi = max(max(self._history), 1)
            chart_h = h - pad_t - pad_b

            painter.setPen(QPen(_C_GRID, 1))
            painter.drawLine(pad_l, pad_t, pad_l, h - pad_b)

            def px(i): return pad_l + (w - pad_l - pad_r) * i / (n - 1)
            def py(v): return h - pad_b - chart_h * v / hi

            painter.setPen(QPen(self._C_LINE, 1.5))
            for i in range(1, n):
                painter.drawLine(int(px(i - 1)), int(py(self._history[i - 1])),
                                 int(px(i)),     int(py(self._history[i])))

            painter.setPen(_C_TEXT)
            painter.drawText(2, pad_t + 8, str(hi))
            painter.drawText(2, h - pad_b + 1, "0")
            painter.setPen(self._C_LINE)
            painter.drawText(pad_l + 4, pad_t + 9, f"Arten: {self._history[-1]}")
        finally:
            painter.end()


# ---------------------------------------------------------------------------

class WeightHistogram(QWidget):
    """Bar chart of connection-weight distribution for the best genome."""

    _BINS = 24
    _C_BAR  = QColor("#89b4fa70")
    _C_ZERO = QColor("#cba6f760")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._weights: list[float] = []
        self.setMinimumHeight(65)
        self.setMaximumHeight(75)

    def set_weights(self, weights: list[float]) -> None:
        self._weights = weights
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), _C_BG)
            w, h = self.width(), self.height()
            pad_l, pad_r, pad_t, pad_b = 6, 6, 5, 14

            font = QFont(); font.setPointSize(6)
            painter.setFont(font)

            if not self._weights:
                painter.setPen(_C_TEXT)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Keine Gewichte")
                return

            lo, hi_val = min(self._weights), max(self._weights)
            if lo == hi_val:
                lo -= 0.5; hi_val += 0.5
            span = hi_val - lo

            bins = [0] * self._BINS
            for wv in self._weights:
                idx = min(int((wv - lo) / span * self._BINS), self._BINS - 1)
                bins[idx] += 1

            max_count = max(bins) if bins else 1
            chart_w = w - pad_l - pad_r
            chart_h = h - pad_t - pad_b
            bar_w = chart_w / self._BINS

            # Zero line
            if lo < 0 < hi_val:
                zero_x = pad_l + (-lo) / span * chart_w
                painter.setPen(QPen(self._C_ZERO, 1))
                painter.drawLine(int(zero_x), pad_t, int(zero_x), h - pad_b)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self._C_BAR))
            for i, count in enumerate(bins):
                bh = chart_h * count / max_count
                painter.drawRect(QRectF(pad_l + i * bar_w + 0.5,
                                        h - pad_b - bh, bar_w - 1.0, bh))

            painter.setPen(_C_TEXT)
            painter.drawText(pad_l, h - 2, f"{lo:.2f}")
            tw = len(f"{hi_val:.2f}") * 5
            painter.drawText(w - pad_r - tw, h - 2, f"{hi_val:.2f}")
            painter.drawText(pad_l + 2, pad_t + 8, f"Gewichte ({len(self._weights)})")
        finally:
            painter.end()
