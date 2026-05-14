"""Network topology canvas and fitness history chart."""
from __future__ import annotations
import math

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF, QRectF, QSize
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath


_C_BG       = QColor("#1e1e2e")
_C_INPUT    = QColor("#4a90d9")
_C_HIDDEN   = QColor("#7c7c9c")
_C_OUTPUT   = QColor("#5ba85a")
_C_BORDER   = QColor("#2d2d4e")
_C_POS_CONN = QColor("#4a90d9")
_C_NEG_CONN = QColor("#d94a4a")
_C_TEXT     = QColor("#cccccc")
_C_GRID     = QColor("#2a2a3e")
_C_CHART    = QColor("#4CAF50")
_NODE_R     = 13


class NetworkCanvas(QWidget):
    """Draws the topology of a Genome: nodes as circles, connections as arrows."""

    _REPAINT_INTERVAL_S = 0.5   # repaint at most every 500 ms (2 fps)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._genome = None
        self._pos_cache: dict = {}
        self._cached_size: tuple[int, int] = (0, 0)
        self._last_repaint: float = 0.0
        self.setMinimumSize(260, 220)

    def set_genome(self, genome) -> None:
        import time as _t
        if self._genome is not None and self._genome is not genome:
            self._genome._clear()
        self._genome = genome
        self._pos_cache = {}
        # Throttle repaints: no more than 2 fps to avoid blocking the event loop
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

        positions: dict[int, QPointF] = {}
        pad = _NODE_R + 14
        inputs  = g.input_nodes
        outputs = g.output_nodes
        hidden  = [n for n in g.nodes if n.type == NodeType.HIDDEN]

        def col_x(frac):
            return pad + frac * (w - 2 * pad)

        def spread_y(idx, total):
            if total == 1:
                return h / 2
            return pad + (h - 2 * pad) * idx / (total - 1)

        for i, n in enumerate(inputs):
            positions[id(n)] = QPointF(col_x(0.08), spread_y(i, len(inputs)))

        for i, n in enumerate(outputs):
            positions[id(n)] = QPointF(col_x(0.92), spread_y(i, len(outputs)))

        if hidden:
            cols = max(1, math.ceil(math.sqrt(len(hidden))))
            rows = math.ceil(len(hidden) / cols)
            x_start = col_x(0.35)
            x_step  = col_x(0.30) / max(cols, 1)
            y_step  = (h - 2 * pad) / max(rows, 1)
            for i, n in enumerate(hidden):
                c, r = i % cols, i // cols
                positions[id(n)] = QPointF(
                    x_start + x_step * (c + 0.5),
                    pad + y_step * (r + 0.5),
                )

        return positions

    def _arrow_head(self, painter: QPainter, src: QPointF, dst: QPointF, color: QColor) -> None:
        dx, dy = dst.x() - src.x(), dst.y() - src.y()
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        # move tip to node border
        tx = dst.x() - ux * (_NODE_R + 2)
        ty = dst.y() - uy * (_NODE_R + 2)
        size = 7
        path = QPainterPath()
        path.moveTo(tx, ty)
        path.lineTo(tx - ux * size - uy * size * 0.4, ty - uy * size + ux * size * 0.4)
        path.lineTo(tx - ux * size + uy * size * 0.4, ty - uy * size - ux * size * 0.4)
        path.closeSubpath()
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)
        painter.setPen(Qt.PenStyle.NoPen)  # reset handled by caller

    def paintEvent(self, event) -> None:
        import time as _t
        _t0 = _t.perf_counter()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), _C_BG)

            if not self._genome or not self._genome.nodes:
                painter.setPen(_C_TEXT)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No genome loaded")
                return

            w, h = self.width(), self.height()
            if not self._pos_cache or self._cached_size != (w, h):
                self._pos_cache = self._positions(w, h)
                self._cached_size = (w, h)
            pos = self._pos_cache

            from yane.core.node import NodeType

            # Connections — reuse a single QColor/QPen per connection (avoid per-conn alloc)
            _conn_color = QColor()
            _pr, _pg, _pb = _C_POS_CONN.red(), _C_POS_CONN.green(), _C_POS_CONN.blue()
            _nr, _ng, _nb = _C_NEG_CONN.red(), _C_NEG_CONN.green(), _C_NEG_CONN.blue()
            _pen = QPen(_conn_color, 1.5)

            for node in self._genome.nodes:
                src = pos.get(id(node))
                if src is None:
                    continue
                for conn in node.connections:
                    dst = pos.get(id(conn.target))
                    if dst is None:
                        continue

                    alpha = min(230, max(40, int(abs(conn.weight) * 160 + 40)))
                    if conn.weight >= 0:
                        _conn_color.setRgb(_pr, _pg, _pb, alpha)
                    else:
                        _conn_color.setRgb(_nr, _ng, _nb, alpha)
                    _pen.setColor(_conn_color)
                    painter.setPen(_pen)

                    if conn.target is node:
                        loop_rect = QRectF(src.x(), src.y() - _NODE_R * 2.8, _NODE_R * 2, _NODE_R * 2)
                        painter.drawArc(loop_rect, 0, 360 * 16)
                    else:
                        painter.drawLine(src, dst)
                        self._arrow_head(painter, src, dst, _conn_color)

            # Nodes
            font = QFont(); font.setPointSize(7)
            painter.setFont(font)

            for node in self._genome.nodes:
                p = pos.get(id(node))
                if p is None:
                    continue
                color = (_C_INPUT if node.type == NodeType.INPUT
                         else _C_OUTPUT if node.type == NodeType.OUTPUT
                         else _C_HIDDEN)
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(_C_BORDER, 1.5))
                painter.drawEllipse(QRectF(p.x() - _NODE_R, p.y() - _NODE_R, _NODE_R * 2, _NODE_R * 2))

                label = node.activation.value[0].upper()
                painter.setPen(_C_BG)
                painter.drawText(
                    QRectF(p.x() - _NODE_R, p.y() - _NODE_R, _NODE_R * 2, _NODE_R * 2),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )
        finally:
            painter.end()
            elapsed = (_t.perf_counter() - _t0) * 1000
            if elapsed > 30:
                from yane.util.logger import get_logger
                n = len(self._genome.nodes) if self._genome else 0
                c = self._genome.connection_count if self._genome else 0
                get_logger().warning(
                    "SLOW NetworkCanvas.paintEvent: %.1fms  nodes=%d  conns=%d",
                    elapsed, n, c,
                )

    def sizeHint(self) -> QSize:
        return QSize(280, 260)


# ---------------------------------------------------------------------------

class FitnessChart(QWidget):
    """Simple scrolling line chart for fitness history."""

    _MAX_POINTS = 300

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._history: list[float] = []
        self._lo: float = 0.0
        self._hi: float = 0.0
        self.setMinimumHeight(110)

    def add_point(self, fitness: float) -> None:
        self._history.append(fitness)
        trimmed = len(self._history) > self._MAX_POINTS
        if trimmed:
            self._history = self._history[-self._MAX_POINTS:]
            self._lo, self._hi = min(self._history), max(self._history)
        elif len(self._history) == 1:
            self._lo = self._hi = fitness
        else:
            self._lo = min(self._lo, fitness)
            self._hi = max(self._hi, fitness)
        self.update()

    def clear(self) -> None:
        self._history.clear()
        self._lo = self._hi = 0.0
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _C_BG)

        w, h = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 42, 8, 8, 20

        # Grid
        painter.setPen(QPen(_C_GRID, 1))
        for i in range(4):
            y = pad_t + (h - pad_t - pad_b) * i / 3
            painter.drawLine(pad_l, int(y), w - pad_r, int(y))

        if len(self._history) < 2:
            painter.setPen(_C_TEXT)
            font = QFont(); font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for data…")
            return

        vals = self._history
        lo, hi = self._lo, self._hi
        if lo == hi:
            hi = lo + 1e-6

        def px(i: int) -> float:
            return pad_l + (w - pad_l - pad_r) * i / (len(vals) - 1)

        def py(v: float) -> float:
            return h - pad_b - (h - pad_t - pad_b) * (v - lo) / (hi - lo)

        # Line
        pen = QPen(_C_CHART, 2)
        painter.setPen(pen)
        for i in range(1, len(vals)):
            painter.drawLine(int(px(i - 1)), int(py(vals[i - 1])), int(px(i)), int(py(vals[i])))

        # Axis labels
        font = QFont(); font.setPointSize(7)
        painter.setFont(font)
        painter.setPen(_C_TEXT)
        painter.drawText(2, pad_t + 10, f"{hi:.3f}")
        painter.drawText(2, h - pad_b, f"{lo:.3f}")
        painter.drawText(pad_l, h - 4, "0")
        painter.drawText(w - pad_r - 20, h - 4, f"{len(vals)}")
