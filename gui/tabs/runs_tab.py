"""Runs tab — browse and inspect runs stored in a RunDatabase."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QAbstractItemView,
    QHeaderView,
)

from yane.gui._helpers import _label, _divider
from yane.gui.canvas import MultiRunChart


class RunsTab(QWidget):
    """Browse all runs stored in a RunDatabase (.db file).

    Left panel:
        - DB path selector
        - Experiment filter
        - Sortable run list

    Right panel (tabbed):
        - Overview: fitness chart + summary
        - Diagnostics: full key-value table
        - Configuration: saved config key-value table
        - Artifacts: clickable file paths
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._db = None          # RunDatabase | None
        self._db_path: str = ""
        self._runs: list = []    # list[Run]

        main = QHBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # ── Left panel ──────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(280)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)

        ll.addWidget(_label("Datenbank", "sectionTitle"))

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("benchmarks/benchmark_runs.db")
        self._path_edit.setReadOnly(True)
        path_row.addWidget(self._path_edit, stretch=1)
        open_btn = QPushButton("Öffnen…")
        open_btn.setFixedWidth(70)
        open_btn.clicked.connect(self._browse_db)
        path_row.addWidget(open_btn)
        ll.addLayout(path_row)

        ll.addWidget(_label("Experiment", "sectionTitle"))
        self._exp_combo = QComboBox()
        self._exp_combo.addItem("Alle")
        self._exp_combo.currentTextChanged.connect(self._rebuild_list)
        ll.addWidget(self._exp_combo)

        self._refresh_btn = QPushButton("↻  Aktualisieren")
        self._refresh_btn.clicked.connect(self._refresh)
        ll.addWidget(self._refresh_btn)

        ll.addWidget(_divider())
        ll.addWidget(_label("Runs", "sectionTitle"))

        self._run_table = QTableWidget(0, 5)
        self._run_table.setHorizontalHeaderLabels(
            ["Name", "Experiment", "Datum", "Best", "Stop"]
        )
        self._run_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._run_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._run_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._run_table.setSortingEnabled(True)
        self._run_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._run_table.verticalHeader().setVisible(False)
        self._run_table.itemSelectionChanged.connect(self._on_run_selected)
        ll.addWidget(self._run_table, stretch=1)

        self._count_lbl = _label("", "sectionTitle")
        ll.addWidget(self._count_lbl)

        main.addWidget(left)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("background: #313244;")
        main.addWidget(sep)

        # ── Right panel ─────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 10, 10, 10)
        rl.setSpacing(8)

        self._detail_tabs = QTabWidget()
        self._detail_tabs.setDocumentMode(True)

        # Tab 1 — Übersicht
        overview = QWidget()
        ov_layout = QVBoxLayout(overview)
        ov_layout.setContentsMargins(4, 4, 4, 4)
        chart_group = QGroupBox("Lernkurve")
        chart_inner = QVBoxLayout(chart_group)
        self._chart = MultiRunChart()
        self._chart.setMinimumHeight(200)
        chart_inner.addWidget(self._chart)
        ov_layout.addWidget(chart_group, stretch=2)

        summary_group = QGroupBox("Zusammenfassung")
        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._summary_inner = QWidget()
        self._summary_layout = QVBoxLayout(self._summary_inner)
        self._summary_layout.setContentsMargins(4, 4, 4, 4)
        self._summary_layout.setSpacing(2)
        self._summary_layout.addWidget(
            _label("Kein Run ausgewählt.", "sectionTitle")
        )
        summary_scroll.setWidget(self._summary_inner)
        summary_inner_l = QVBoxLayout(summary_group)
        summary_inner_l.addWidget(summary_scroll)
        ov_layout.addWidget(summary_group, stretch=1)
        self._detail_tabs.addTab(overview, "Übersicht")

        # Tab 2 — Diagnose
        diag_widget = QWidget()
        diag_layout = QVBoxLayout(diag_widget)
        diag_layout.setContentsMargins(4, 4, 4, 4)
        diag_scroll = QScrollArea()
        diag_scroll.setWidgetResizable(True)
        diag_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._diag_inner = QWidget()
        self._diag_layout = QVBoxLayout(self._diag_inner)
        self._diag_layout.setContentsMargins(4, 4, 4, 4)
        self._diag_layout.setSpacing(2)
        self._diag_layout.addWidget(_label("Kein Run ausgewählt.", "sectionTitle"))
        diag_scroll.setWidget(self._diag_inner)
        diag_layout.addWidget(diag_scroll)
        self._detail_tabs.addTab(diag_widget, "Diagnose")

        # Tab 3 — Konfiguration
        cfg_widget = QWidget()
        cfg_layout = QVBoxLayout(cfg_widget)
        cfg_layout.setContentsMargins(4, 4, 4, 4)
        cfg_scroll = QScrollArea()
        cfg_scroll.setWidgetResizable(True)
        cfg_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._cfg_inner = QWidget()
        self._cfg_layout = QVBoxLayout(self._cfg_inner)
        self._cfg_layout.setContentsMargins(4, 4, 4, 4)
        self._cfg_layout.setSpacing(2)
        self._cfg_layout.addWidget(_label("Kein Run ausgewählt.", "sectionTitle"))
        cfg_scroll.setWidget(self._cfg_inner)
        cfg_layout.addWidget(cfg_scroll)
        self._detail_tabs.addTab(cfg_widget, "Konfiguration")

        # Tab 4 — Artefakte
        art_widget = QWidget()
        art_layout = QVBoxLayout(art_widget)
        art_layout.setContentsMargins(4, 4, 4, 4)
        self._art_list = QListWidget()
        self._art_list.itemDoubleClicked.connect(self._on_artifact_double_clicked)
        art_layout.addWidget(QLabel("Doppelklick öffnet den Pfad im Datei-Manager."))
        art_layout.addWidget(self._art_list)
        self._detail_tabs.addTab(art_widget, "Artefakte")

        rl.addWidget(self._detail_tabs)
        main.addWidget(right, stretch=1)

        # Try to open default benchmark DB on startup
        from yane.benchmarks import BENCHMARK_DB_PATH
        if BENCHMARK_DB_PATH.exists():
            self._open_db(str(BENCHMARK_DB_PATH))

    # ── DB management ──────────────────────────────────────────────────

    def _browse_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "RunDatabase öffnen",
            str(Path.home()),
            "SQLite Datenbanken (*.db);;Alle Dateien (*)",
        )
        if path:
            self._open_db(path)

    def _open_db(self, path: str) -> None:
        try:
            from yane.util.run_database import RunDatabase
            self._db = RunDatabase(path)
            self._db_path = path
            self._path_edit.setText(path)
            self._refresh()
        except Exception as exc:
            self._path_edit.setText(f"Fehler: {exc}")
            self._db = None

    def _refresh(self) -> None:
        if self._db is None:
            return
        # Reload experiment list
        current_exp = self._exp_combo.currentText()
        self._exp_combo.blockSignals(True)
        self._exp_combo.clear()
        self._exp_combo.addItem("Alle")
        try:
            exps = self._db.list_experiments()
            for e in exps:
                self._exp_combo.addItem(e.name)
        except Exception:
            pass
        idx = self._exp_combo.findText(current_exp)
        self._exp_combo.setCurrentIndex(max(0, idx))
        self._exp_combo.blockSignals(False)
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        if self._db is None:
            return
        exp_name = self._exp_combo.currentText()
        try:
            if exp_name == "Alle":
                self._runs = self._db.list_runs()
            else:
                exps = [
                    e for e in self._db.list_experiments() if e.name == exp_name
                ]
                if exps:
                    self._runs = self._db.list_runs(experiment_id=exps[0].experiment_id)
                else:
                    self._runs = []
        except Exception:
            self._runs = []

        self._run_table.setSortingEnabled(False)
        self._run_table.setRowCount(len(self._runs))
        for row, run in enumerate(self._runs):
            self._run_table.setItem(row, 0, QTableWidgetItem(run.name or ""))
            self._run_table.setItem(
                row, 1, QTableWidgetItem(run.experiment_id or "")
            )
            self._run_table.setItem(
                row, 2, QTableWidgetItem((run.start_time or "")[:16])
            )
            best = run.final_best
            self._run_table.setItem(
                row, 3,
                QTableWidgetItem(f"{best:.4f}" if best is not None else "—"),
            )
            self._run_table.setItem(
                row, 4, QTableWidgetItem(run.stop_reason or "—")
            )
        self._run_table.setSortingEnabled(True)
        n = len(self._runs)
        self._count_lbl.setText(f"{n} Run{'s' if n != 1 else ''}")

    # ── Run selection ──────────────────────────────────────────────────

    def _on_run_selected(self) -> None:
        rows = self._run_table.selectionModel().selectedRows()
        if not rows:
            self._clear_detail()
            return
        visual_row = rows[0].row()
        # Map visual row back to self._runs — sort may have reordered rows
        name_item = self._run_table.item(visual_row, 0)
        date_item = self._run_table.item(visual_row, 2)
        if name_item is None:
            self._clear_detail()
            return
        name = name_item.text()
        date = date_item.text() if date_item else ""
        run = next(
            (r for r in self._runs
             if r.name == name and (r.start_time or "")[:16] == date),
            None,
        )
        if run is None:
            run = next((r for r in self._runs if r.name == name), None)
        if run is None:
            self._clear_detail()
            return
        self._show_run(run)

    def _clear_detail(self) -> None:
        self._chart.clear()
        _replace_layout_content(
            self._summary_layout,
            [_label("Kein Run ausgewählt.", "sectionTitle")],
        )
        _replace_layout_content(
            self._diag_layout,
            [_label("Kein Run ausgewählt.", "sectionTitle")],
        )
        _replace_layout_content(
            self._cfg_layout,
            [_label("Kein Run ausgewählt.", "sectionTitle")],
        )
        self._art_list.clear()

    def _show_run(self, run) -> None:
        # ── Fitness chart ────────────────────────────────────────────
        if run.fitness_history:
            iters = [r.get("iteration", i) for i, r in enumerate(run.fitness_history)]
            bests = [r.get("best_fitness") for r in run.fitness_history]
            bests_clean = [
                b if isinstance(b, (int, float)) else None for b in bests
            ]
            self._chart.set_runs([{
                "name": run.name,
                "iterations": iters,
                "best_fitness": [b for b in bests_clean if b is not None],
            }])
        else:
            self._chart.clear()

        # ── Summary ──────────────────────────────────────────────────
        best = run.final_best
        rows = [
            ("Run-ID", run.run_id),
            ("Name", run.name or "—"),
            ("Gestartet", run.start_time or "—"),
            ("Beendet", run.end_time or "—"),
            ("Stop-Grund", run.stop_reason or "—"),
            ("Best Fitness", f"{best:.6f}" if best is not None else "—"),
            ("Iterationen", str(run.total_iterations)),
            ("Seed", str(run.seed) if run.seed is not None else "—"),
        ]
        _replace_layout_content(
            self._summary_layout,
            [_kv_row(k, v) for k, v in rows],
        )

        # ── Diagnostics ──────────────────────────────────────────────
        _replace_layout_content(
            self._diag_layout,
            _flatten_dict_widgets(run.diagnostics) or [
                _label("Keine Diagnosedaten.", "sectionTitle")
            ],
        )

        # ── Config ───────────────────────────────────────────────────
        _replace_layout_content(
            self._cfg_layout,
            _flatten_dict_widgets(run.config) or [
                _label("Keine Konfigurationsdaten.", "sectionTitle")
            ],
        )

        # ── Artifacts ────────────────────────────────────────────────
        self._art_list.clear()
        for key, path_str in run.artifacts.items():
            item = QListWidgetItem(f"{key}: {path_str}")
            item.setData(Qt.ItemDataRole.UserRole, path_str)
            item.setToolTip(path_str)
            self._art_list.addItem(item)

    def _on_artifact_double_clicked(self, item: QListWidgetItem) -> None:
        path_str = item.data(Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        import subprocess, sys
        p = Path(path_str)
        target = str(p.parent) if p.is_file() else path_str
        try:
            if sys.platform == "linux":
                subprocess.Popen(["xdg-open", target])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["explorer", target])
        except Exception:
            pass


# ── Helpers ────────────────────────────────────────────────────────────────

def _replace_layout_content(layout: QVBoxLayout, widgets: list[QWidget]) -> None:
    """Remove all existing widgets from *layout* and add *widgets*."""
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
    for w in widgets:
        layout.addWidget(w)
    layout.addStretch()


def _kv_row(key: str, value: str) -> QWidget:
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    key_lbl = QLabel(f"<b>{key}</b>")
    key_lbl.setFixedWidth(160)
    key_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    val_lbl = QLabel(str(value))
    val_lbl.setWordWrap(True)
    val_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    row.addWidget(key_lbl)
    row.addWidget(val_lbl, stretch=1)
    return w


def _flatten_dict_widgets(d: dict, prefix: str = "") -> list[QWidget]:
    """Recursively flatten a dict into a list of key-value row widgets."""
    widgets: list[QWidget] = []
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            widgets.append(_label(full_key, "sectionTitle"))
            widgets.extend(_flatten_dict_widgets(v, full_key))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            # List of dicts — show length only to avoid flooding
            widgets.append(_kv_row(full_key, f"[{len(v)} Einträge]"))
        elif isinstance(v, list):
            widgets.append(_kv_row(full_key, str(v)[:200]))
        else:
            widgets.append(_kv_row(full_key, str(v) if v is not None else "—"))
    return widgets
