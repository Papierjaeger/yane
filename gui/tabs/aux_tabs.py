"""Auxiliary tabs: API server control and debug/diagnostics view."""
from __future__ import annotations
import time as _time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QFormLayout,
    QPlainTextEdit, QSpinBox, QApplication, QFileDialog,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from yane.gui.worker import ServerThread
from yane.gui.canvas import LandscapeScatter
from yane.gui._helpers import _label, _divider

class ServerTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._server_thread: ServerThread | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(14, 14, 14, 14)

        # Controls
        ctrl = QGroupBox("API Server")
        ctrl_layout = QFormLayout(ctrl)
        self.spin_port = QSpinBox()
        self.spin_port.setRange(1024, 65535)
        self.spin_port.setValue(8000)
        ctrl_layout.addRow("Port:", self.spin_port)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_server = QPushButton("Start Server")
        self.btn_server.setObjectName("serverBtn")
        self.btn_server.clicked.connect(self._toggle_server)
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #585b70; font-size: 18px;")
        self.status_lbl = QLabel("Offline")
        row_layout.addWidget(self.btn_server)
        row_layout.addWidget(self.status_dot)
        row_layout.addWidget(self.status_lbl)
        row_layout.addStretch()
        ctrl_layout.addRow(row)
        layout.addWidget(ctrl)

        # Endpoints
        ep = QGroupBox("Endpoints")
        ep_layout = QVBoxLayout(ep)
        endpoints = [
            ("POST", "/configure",           "Initialise with n_inputs / n_outputs"),
            ("POST", "/population/next",      "Select next genome for evaluation"),
            ("POST", "/population/fitness",   "Submit fitness for current genome"),
            ("GET",  "/population/status",    "Population size, best fitness"),
            ("GET",  "/population/best",      "Best genome info"),
            ("POST", "/network/inputs",       "Set input values"),
            ("POST", "/network/tick",         "Execute one tick"),
            ("GET",  "/network/outputs",      "Read current outputs"),
            ("POST", "/network/forward",      "Full forward pass"),
            ("POST", "/network/reset",        "Reset network state"),
        ]
        for method, path, desc in endpoints:
            color = "#a6e3a1" if method == "GET" else "#89b4fa"
            lbl = QLabel(f'<span style="color:{color};font-weight:bold;">{method}</span>'
                         f'&nbsp;&nbsp;<tt>{path}</tt>&nbsp;&nbsp;'
                         f'<span style="color:#6c7086;">{desc}</span>')
            lbl.setTextFormat(Qt.TextFormat.RichText)
            ep_layout.addWidget(lbl)
        layout.addWidget(ep)

        layout.addStretch()

    def _toggle_server(self) -> None:
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.stop()
            self._server_thread = None
            self.btn_server.setText("Start Server")
            self.status_dot.setStyleSheet("color: #585b70; font-size: 18px;")
            self.status_lbl.setText("Offline")
        else:
            port = self.spin_port.value()
            self._server_thread = ServerThread(port=port)
            self._server_thread.start()
            self.btn_server.setText("Stop Server")
            self.status_dot.setStyleSheet("color: #a6e3a1; font-size: 18px;")
            self.status_lbl.setText(f"Running on http://127.0.0.1:{port}")


# ---------------------------------------------------------------------------
# Debug tab
# ---------------------------------------------------------------------------

class DebugTab(QWidget):
    """Live debug log — compact tabular snapshot every 0.5 s when enabled."""

    _HEADER = (
        "  iter     t(s)    best     avg     min  sp tgt   thr   pop/max"
        "   stag  sinj    xov    mut   inj   ms  spawn adj_n  adj#"
    )
    _HEADER_EVERY = 25   # repeat column header every N data lines

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._enabled = False
        self._data_lines = 0
        self._t0: float | None = None
        self._ne = None  # set by MainWindow after training starts
        self._last_landscape: dict = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # ── Controls ────────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self.btn_toggle = QPushButton("Debug: Off")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setFixedWidth(110)
        self.btn_toggle.toggled.connect(self._on_toggle)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(70)
        btn_clear.clicked.connect(self._clear)

        self.btn_copy = QPushButton("Copy to Clipboard")
        self.btn_copy.setFixedWidth(150)
        self.btn_copy.clicked.connect(self._copy)

        hint = _label(
            "Enable before training to capture a live log you can paste into chat.",
            "mutRate",
        )
        hint.setWordWrap(True)

        ctrl.addWidget(self.btn_toggle)
        ctrl.addWidget(btn_clear)

        # Report export button
        self.btn_report = QPushButton("Export Report")
        self.btn_report.setFixedWidth(120)
        self.btn_report.clicked.connect(self._export_report)
        self.btn_report.setEnabled(False)
        ctrl.addWidget(self.btn_report)

        # Landscape PCA button
        self.btn_pca = QPushButton("Landscape PCA")
        self.btn_pca.setFixedWidth(120)
        self.btn_pca.clicked.connect(self._run_landscape_pca)
        self.btn_pca.setEnabled(False)
        ctrl.addWidget(self.btn_pca)

        self.btn_pca_png = QPushButton("PCA PNG")
        self.btn_pca_png.setFixedWidth(90)
        self.btn_pca_png.clicked.connect(self._export_landscape_png)
        self.btn_pca_png.setEnabled(False)
        ctrl.addWidget(self.btn_pca_png)

        self.btn_pca_csv = QPushButton("PCA CSV")
        self.btn_pca_csv.setFixedWidth(90)
        self.btn_pca_csv.clicked.connect(self._export_landscape_csv)
        self.btn_pca_csv.setEnabled(False)
        ctrl.addWidget(self.btn_pca_csv)

        ctrl.addStretch()
        ctrl.addWidget(hint)
        ctrl.addStretch()
        ctrl.addWidget(self.btn_copy)
        outer.addLayout(ctrl)

        self._landscape = LandscapeScatter()
        outer.addWidget(self._landscape)

        # ── Log area ────────────────────────────────────────────────────────
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Monospace", 10))
        self._log.setStyleSheet(
            "QPlainTextEdit {"
            "  background: #11111b;"
            "  color: #cdd6f4;"
            "  border: 1px solid #313244;"
            "  border-radius: 4px;"
            "}"
        )
        outer.addWidget(self._log)

    # ── Slots called by MainWindow ──────────────────────────────────────────

    def on_training_started(self, mem: dict | None = None) -> None:
        self._t0 = _time.perf_counter()
        self._data_lines = 0
        if self._enabled:
            self._log.appendPlainText("\n=== Training started ===")

    def on_update(self, genome, mem: dict, _do_heavy: bool) -> None:
        if not self._enabled:
            return

        t = _time.perf_counter() - (self._t0 or _time.perf_counter())

        # Repeat header every N data lines
        if self._data_lines % self._HEADER_EVERY == 0:
            self._log.appendPlainText(self._HEADER)
        self._data_lines += 1

        total_iter = (mem.get("n_crossover", 0)
                      + mem.get("n_mutation_only", 0)
                      + mem.get("n_diversity_injection", 0))
        pop_eval = mem.get("pop_evaluated", mem.get("total_genomes", 0))
        pop_max  = mem.get("pop_max", 0)
        ev_ms    = mem.get("eval_time_mean_ms", 0.0)

        line = (
            f"{total_iter:6d}"
            f"  {t:7.1f}"
            f"  {mem.get('max_fitness', 0.0):7.4f}"
            f"  {mem.get('avg_fitness', 0.0):7.4f}"
            f"  {mem.get('min_fitness', 0.0):7.4f}"
            f"  {mem.get('species_count', 0):2d}"
            f"  {mem.get('target_species', '?'):>3}"
            f"  {mem.get('compat_threshold', 0.0):.3f}"
            f"  {pop_eval:3d}/{pop_max:<3d}"
            f"  {mem.get('stagnation_count', 0):5d}"
            f"  {mem.get('since_last_injection', 0):4d}"
            f"  {mem.get('n_crossover', 0):5d}"
            f"  {mem.get('n_mutation_only', 0):5d}"
            f"  {mem.get('n_diversity_injection', 0):4d}"
            f"  {ev_ms:5.1f}"
            f"  {mem.get('spawn_count', 0):6d}"
            f"  {mem.get('dbg_last_adj_n', -1):5d}"
            f"  {mem.get('dbg_adj_count', 0):5d}"
        )
        self._log.appendPlainText(line)

        # Auto-scroll to bottom
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Internal helpers ────────────────────────────────────────────────────

    def _on_toggle(self, checked: bool) -> None:
        self._enabled = checked
        self.btn_toggle.setText("Debug: On" if checked else "Debug: Off")
        self.btn_toggle.setStyleSheet(
            "color: #a6e3a1; font-weight: bold;" if checked else ""
        )

    def _clear(self) -> None:
        self._log.clear()
        self._data_lines = 0

    def set_ne(self, ne) -> None:
        """Set NeuroEvolution instance reference (called by MainWindow)."""
        self._ne = ne
        self.btn_report.setEnabled(ne is not None)
        self.btn_pca.setEnabled(ne is not None)
        has_landscape = bool(self._last_landscape.get("x"))
        self.btn_pca_png.setEnabled(ne is not None and has_landscape)
        self.btn_pca_csv.setEnabled(ne is not None and has_landscape)

    def _run_landscape_pca(self) -> None:
        """Run landscape PCA analysis and show results in the log."""
        if self._ne is None:
            return
        self._log.appendPlainText("\n=== Landscape PCA ===")
        try:
            result = self._ne.landscape_pca()
            if not result or not result.get("x"):
                self._log.appendPlainText("❌ No result (population empty?).")
                self._last_landscape = {}
                self._landscape.clear()
                self.btn_pca_png.setEnabled(False)
                self.btn_pca_csv.setEnabled(False)
                return
            self._last_landscape = result
            self._landscape.set_snapshot(result)
            self.btn_pca_png.setEnabled(True)
            self.btn_pca_csv.setEnabled(True)
            ev = result.get("explained_var", [0.0, 0.0])
            total = sum(ev) if ev else 0.0
            self._log.appendPlainText(
                f"  Explained variance (first 2 components): "
                f"{ev}"
            )
            self._log.appendPlainText(
                f"  Total variance captured: {total:.1%}"
            )
            n_genomes = len(result.get("x", []))
            self._log.appendPlainText(f"  Genomes projected: {n_genomes}")
            self._log.appendPlainText("✅ Landscape PCA complete.")
        except Exception as exc:
            self._log.appendPlainText(f"❌ Landscape PCA failed: {exc}")

    def _export_landscape_png(self) -> None:
        """Export the latest landscape PCA snapshot as PNG."""
        if self._ne is None or not self._last_landscape.get("x"):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Landscape PNG exportieren", "landscape_pca.png", "PNG (*.png)",
        )
        if not path:
            return
        try:
            self._ne.export_landscape_png(path)
            self._log.appendPlainText(f"\n✅ Landscape PNG exported: {path}")
        except Exception as exc:
            self._log.appendPlainText(f"\n❌ Landscape PNG export failed: {exc}")

    def _export_landscape_csv(self) -> None:
        """Export the latest landscape PCA snapshot as CSV."""
        if self._ne is None or not self._last_landscape.get("x"):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Landscape CSV exportieren", "landscape_pca.csv", "CSV (*.csv)",
        )
        if not path:
            return
        try:
            self._ne.export_landscape_csv(path)
            self._log.appendPlainText(f"\n✅ Landscape CSV exported: {path}")
        except Exception as exc:
            self._log.appendPlainText(f"\n❌ Landscape CSV export failed: {exc}")

    def _export_report(self) -> None:
        """Export a run report for the current training session."""
        if self._ne is None:
            return
        from pathlib import Path
        path, _ = QFileDialog.getSaveFileName(
            self, "Report exportieren", "report.html",
            "HTML (*.html);;Markdown (*.md);;JSON (*.json)",
        )
        if not path:
            return
        try:
            fmt = Path(path).suffix.lstrip(".") or "html"
            self._ne.export_run_report(path, fmt=fmt)
            self._log.appendPlainText(f"\n✅ Report exported: {path}")
        except Exception as exc:
            self._log.appendPlainText(f"\n❌ Report export failed: {exc}")

    def _copy(self) -> None:
        text = self._log.toPlainText()
        if text.strip():
            QApplication.clipboard().setText(text)
            original = self.btn_copy.text()
            self.btn_copy.setText("Copied!")
            QTimer.singleShot(1500, lambda: self.btn_copy.setText(original))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
