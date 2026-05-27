"""Quality-diversity and Pareto diagnostic widgets."""
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget, QToolTip

_C_BG = QColor("#1e1e2e")
_C_TEXT = QColor("#cccccc")
_C_GRID = QColor("#2a2a3e")


class ParetoScatter(QWidget):
    """Scatter plot for the first two Pareto objectives with hover tooltips."""

    _PAD_L = 38
    _PAD_B = 22
    _PAD_T = 14
    _PAD_R = 8

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._points: list[dict] = []
        self._px: list[float] = []
        self._py: list[float] = []
        self.setMinimumHeight(120)
        self.setMouseTracking(True)

    def set_points(self, points: list[dict] | None) -> None:
        self._points = list(points or [])
        self._px = []
        self._py = []
        self.update()

    def _to_pixel(
        self,
        w: int,
        h: int,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        ox: float,
        oy: float,
    ) -> tuple[float, float]:
        px = self._PAD_L + (ox - x0) / (x1 - x0) * (w - self._PAD_L - self._PAD_R)
        py = self._PAD_T + (1 - (oy - y0) / (y1 - y0)) * (h - self._PAD_T - self._PAD_B)
        return px, py

    def mouseMoveEvent(self, event) -> None:
        mx, my = event.position().x(), event.position().y()
        best_d, best_i = 64.0, -1
        for i, (px, py) in enumerate(zip(self._px, self._py)):
            d = (mx - px) ** 2 + (my - py) ** 2
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            p = self._points[best_i]
            objs = p.get("objectives", [])
            tip = (
                f"Obj1: {objs[0]:.4f}\n"
                f"Obj2: {objs[1]:.4f}\n"
                f"Fitness: {p.get('fitness', '—'):.4f}\n"
                f"Nodes: {p.get('nodes', '—')}  "
                f"Conns: {p.get('connections', '—')}"
            )
            QToolTip.showText(event.globalPosition().toPoint(), tip, self)
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), _C_BG)
            w, h = self.width(), self.height()
            pad_l, pad_b, pad_t, pad_r = self._PAD_L, self._PAD_B, self._PAD_T, self._PAD_R
            if not self._points:
                painter.setPen(_C_TEXT)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Pareto data")
                return
            xs = [p["objectives"][0] for p in self._points]
            ys = [p["objectives"][1] for p in self._points]
            fits = [p.get("fitness", 0.0) for p in self._points]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            f0, f1 = min(fits), max(fits)
            if x0 == x1:
                x0 -= 0.5
                x1 += 0.5
            if y0 == y1:
                y0 -= 0.5
                y1 += 0.5
            fspan = max(f1 - f0, 1e-9)
            painter.setPen(QPen(_C_GRID, 1))
            plot_rect = QRectF(pad_l, pad_t, w - pad_l - pad_r, h - pad_t - pad_b)
            painter.drawRect(plot_rect)
            small_font = QFont()
            small_font.setPointSize(7)
            painter.setFont(small_font)
            painter.setPen(_C_TEXT)
            painter.drawText(
                QRectF(pad_l, h - pad_b, (w - pad_l - pad_r) / 2, pad_b),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                f"{x0:.3g}",
            )
            painter.drawText(
                QRectF(pad_l + (w - pad_l - pad_r) / 2, h - pad_b,
                       (w - pad_l - pad_r) / 2, pad_b),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                f"{x1:.3g}",
            )
            painter.drawText(
                QRectF(0, pad_t, pad_l - 2, 12),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                f"{y1:.3g}",
            )
            painter.drawText(
                QRectF(0, h - pad_b - 12, pad_l - 2, 12),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                f"{y0:.3g}",
            )
            painter.drawText(
                pad_l + 2,
                pad_t - 2,
                f"Pareto ({len(self._points)})  [Obj1 vs Obj2, color=fitness]",
            )
            self._px = []
            self._py = []
            painter.setPen(Qt.PenStyle.NoPen)
            for p in self._points:
                px, py = self._to_pixel(
                    w, h, x0, x1, y0, y1, p["objectives"][0], p["objectives"][1]
                )
                self._px.append(px)
                self._py.append(py)
                rel = (p.get("fitness", f0) - f0) / fspan
                painter.setBrush(QBrush(QColor.fromHsvF(0.33 * rel, 0.7, 0.9)))
                painter.drawEllipse(QRectF(px - 3.0, py - 3.0, 6.0, 6.0))
        finally:
            painter.end()


class MapElitesHeatmap(QWidget):
    """Two-dimensional MAP-Elites cell heatmap with hover tooltips."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cells: list[dict] = []
        self._cell_rects: list[tuple[float, float, float, float]] = []
        self.setMinimumHeight(110)
        self.setMouseTracking(True)

    def set_cells(self, cells: list[dict] | None) -> None:
        self._cells = [c for c in (cells or []) if len(c.get("cell", [])) >= 2]
        self._cell_rects = []
        self.update()

    def mouseMoveEvent(self, event) -> None:
        mx, my = event.position().x(), event.position().y()
        for i, (rx, ry, cw, ch) in enumerate(self._cell_rects):
            if rx <= mx <= rx + cw and ry <= my <= ry + ch:
                c = self._cells[i]
                QToolTip.showText(
                    event.globalPosition().toPoint(),
                    f"Cell: {c['cell']}\nFitness: {c['fitness']:.4f}",
                    self,
                )
                return
        QToolTip.hideText()
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), _C_BG)
            if not self._cells:
                painter.setPen(_C_TEXT)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No QD cells")
                return
            w, h = self.width(), self.height()
            pad = 10
            title_h = 12
            max_x = max(c["cell"][0] for c in self._cells)
            max_y = max(c["cell"][1] for c in self._cells)
            min_fit = min(c["fitness"] for c in self._cells)
            max_fit = max(c["fitness"] for c in self._cells)
            span = max(max_fit - min_fit, 1e-9)
            plot_w = w - 2 * pad
            plot_h = h - 2 * pad - title_h
            cw = plot_w / (max_x + 1)
            ch = plot_h / (max_y + 1)
            self._cell_rects = []
            for c in self._cells:
                rel = (c["fitness"] - min_fit) / span
                color = QColor.fromHsvF(0.33 * rel, 0.6, 0.9, 0.9)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                rx = pad + c["cell"][0] * cw
                ry = pad + title_h + (max_y - c["cell"][1]) * ch
                painter.drawRect(QRectF(rx + 1, ry + 1, max(1.0, cw - 2), max(1.0, ch - 2)))
                self._cell_rects.append((rx, ry, cw, ch))
            painter.setPen(_C_TEXT)
            small_font = QFont()
            small_font.setPointSize(7)
            painter.setFont(small_font)
            painter.drawText(
                pad + 2,
                pad + title_h - 2,
                f"MAP-Elites ({len(self._cells)})  [fitness: {min_fit:.3g} → {max_fit:.3g}]",
            )
        finally:
            painter.end()
