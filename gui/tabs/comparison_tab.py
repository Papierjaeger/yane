"""Comparison tab — overlaid learning curves for multiple training runs."""
from __future__ import annotations

import csv
import statistics

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QComboBox,
    QFrame, QFileDialog, QScrollArea,
)

from yane.gui._helpers import _label, _divider
from yane.gui.canvas import MultiRunChart
from yane.util.run_history import RunRecord


_MAX_SELECTED = 4


class ComparisonTab(QWidget):
    """Shows overlaid fitness curves for up to 4 training runs.

    Runs are loaded from the ``logs/`` directory on demand.  The user can
    filter by category, select individual runs, and export the chart as PNG
    or the raw data as CSV.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records: list[RunRecord] = []
        self._selected: list[RunRecord] = []

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Left panel: run selector ────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(230)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        left_layout.addWidget(_label("Kategorie", "sectionTitle"))
        self._cat_combo = QComboBox()
        self._cat_combo.addItem("Alle")
        self._cat_combo.currentTextChanged.connect(self._on_category_changed)
        left_layout.addWidget(self._cat_combo)

        self._refresh_btn = QPushButton("↻  Aktualisieren")
        self._refresh_btn.clicked.connect(self.refresh)
        left_layout.addWidget(self._refresh_btn)

        left_layout.addWidget(_divider())
        left_layout.addWidget(_label(f"Runs (max. {_MAX_SELECTED} auswählen)", "sectionTitle"))

        self._run_list = QListWidget()
        self._run_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._run_list.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._run_list, stretch=1)

        self._sel_lbl = _label("0 ausgewählt", "sectionTitle")
        left_layout.addWidget(self._sel_lbl)

        left_layout.addWidget(_divider())

        self._export_png_btn = QPushButton("⬇  Chart als PNG speichern")
        self._export_png_btn.setEnabled(False)
        self._export_png_btn.clicked.connect(self._export_png)
        left_layout.addWidget(self._export_png_btn)

        self._export_csv_btn = QPushButton("⬇  Daten als CSV speichern")
        self._export_csv_btn.setEnabled(False)
        self._export_csv_btn.clicked.connect(self._export_csv)
        left_layout.addWidget(self._export_csv_btn)

        main_layout.addWidget(left)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("background: #313244;")
        main_layout.addWidget(sep)

        # ── Right panel: chart + stats ──────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        chart_group = QGroupBox("Lernkurven-Vergleich")
        chart_layout = QVBoxLayout(chart_group)
        self._chart = MultiRunChart()
        chart_layout.addWidget(self._chart)
        right_layout.addWidget(chart_group, stretch=3)

        # Stats section
        stats_group = QGroupBox("Statistische Zusammenfassung (gleiche Konfiguration)")
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setFrameShape(QFrame.Shape.NoFrame)
        stats_inner = QWidget()
        self._stats_layout = QVBoxLayout(stats_inner)
        self._stats_layout.setContentsMargins(4, 4, 4, 4)
        self._stats_layout.setSpacing(4)
        self._stats_placeholder = _label(
            "Wähle mehrere Runs mit gleicher Konfiguration aus, um Statistiken zu sehen.",
            "sectionTitle"
        )
        self._stats_placeholder.setWordWrap(True)
        self._stats_layout.addWidget(self._stats_placeholder)
        stats_scroll.setWidget(stats_inner)
        stats_group_layout = QVBoxLayout(stats_group)
        stats_group_layout.addWidget(stats_scroll)
        right_layout.addWidget(stats_group, stretch=1)

        main_layout.addWidget(right, stretch=1)

        self.refresh()

    # ------------------------------------------------------------------
    # Public

    def refresh(self) -> None:
        """Reload the run list from the logs/ directory."""
        # Preserve current category selection
        current_cat = self._cat_combo.currentText()

        cats = RunRecord.list_categories()
        self._cat_combo.blockSignals(True)
        self._cat_combo.clear()
        self._cat_combo.addItem("Alle")
        for c in cats:
            self._cat_combo.addItem(c)
        idx = self._cat_combo.findText(current_cat)
        self._cat_combo.setCurrentIndex(max(0, idx))
        self._cat_combo.blockSignals(False)

        self._reload_records()

    # ------------------------------------------------------------------

    def _on_category_changed(self, _cat: str) -> None:
        self._reload_records()

    def _reload_records(self) -> None:
        cat = self._cat_combo.currentText()
        category = None if cat == "Alle" else cat
        self._records = RunRecord.list_runs(category=category)
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        self._run_list.blockSignals(True)
        self._run_list.clear()
        for rec in self._records:
            item = QListWidgetItem(rec.label)
            item.setToolTip(
                f"Iterationen: {rec.total_iterations}\n"
                f"Bestes Fitness: {rec.final_best:.4f}" if rec.final_best is not None else ""
            )
            self._run_list.addItem(item)
        self._run_list.blockSignals(False)
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        indices = [self._run_list.row(item)
                   for item in self._run_list.selectedItems()]

        # Enforce max selection
        if len(indices) > _MAX_SELECTED:
            # Deselect the oldest extra items
            extra = indices[:-_MAX_SELECTED]
            self._run_list.blockSignals(True)
            for i in extra:
                self._run_list.item(i).setSelected(False)
            self._run_list.blockSignals(False)
            indices = indices[-_MAX_SELECTED:]

        self._selected = [self._records[i] for i in indices if i < len(self._records)]
        n = len(self._selected)
        self._sel_lbl.setText(f"{n} ausgewählt")
        self._export_png_btn.setEnabled(n > 0)
        self._export_csv_btn.setEnabled(n > 0)

        self._chart.set_runs([r.to_chart_dict() for r in self._selected])
        self._update_stats()

    def _update_stats(self) -> None:
        # Clear existing stats widgets
        while self._stats_layout.count():
            item = self._stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        selected = self._selected
        if len(selected) < 2:
            lbl = _label(
                "Wähle mindestens 2 Runs aus, um Statistiken zu sehen.",
                "sectionTitle"
            )
            lbl.setWordWrap(True)
            self._stats_layout.addWidget(lbl)
            return

        # Group selected runs by config hash
        groups: dict[str, list[RunRecord]] = {}
        for rec in selected:
            groups.setdefault(rec.config_hash, []).append(rec)

        for hash_key, group in groups.items():
            if len(group) < 2:
                continue
            finals = [r.final_best for r in group if r.final_best is not None]
            if not finals:
                continue
            med   = statistics.median(finals)
            mean  = statistics.mean(finals)
            best  = max(finals)
            worst = min(finals)
            stdev = statistics.stdev(finals) if len(finals) >= 2 else 0.0

            n_cfg = len(group)
            grp_label = group[0].name
            text = (
                f"{grp_label}  ({n_cfg} Runs, gleiche Konfig)\n"
                f"  Median:  {med:.4f}   Mittel:  {mean:.4f}\n"
                f"  Bestes:  {best:.4f}   Schlechtestes: {worst:.4f}\n"
                f"  Stdev:   {stdev:.4f}"
            )
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "font-family: monospace; font-size: 12px; color: #cdd6f4;"
                " background: #313244; border-radius: 4px; padding: 6px;"
            )
            lbl.setWordWrap(True)
            self._stats_layout.addWidget(lbl)

        if not self._stats_layout.count():
            lbl = _label(
                "Keine zwei Runs mit gleicher Konfiguration ausgewählt.",
                "sectionTitle"
            )
            lbl.setWordWrap(True)
            self._stats_layout.addWidget(lbl)

        self._stats_layout.addStretch()

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Chart als PNG speichern", "lernkurven.png",
            "PNG Images (*.png)"
        )
        if path:
            self._chart.export_png(path)

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Daten als CSV speichern", "lernkurven.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        fieldnames = ["run", "iteration", "best_fitness", "mean_fitness",
                      "median_fitness", "iqr_fitness", "species_count"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self._selected:
                for row in rec.to_csv_rows():
                    writer.writerow({"run": rec.label, **row})
