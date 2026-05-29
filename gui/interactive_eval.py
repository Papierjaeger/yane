"""GUI panel for Interactive / Human-in-the-Loop evaluation.

Provides :class:`InteractiveEvalPanel` — a PySide6 widget that shows genome
outputs side-by-side and collects human ratings.  Attach it to an
:class:`~yane.evolution.interactive_eval.InteractiveEvaluator` via
:meth:`set_evaluator`.

The panel polls for pending genomes (genomes waiting for feedback) and
renders them in the active mode (rating slider, pairwise left/right buttons,
ranking drag-list, or implicit dwell-time timer).

Note: The GUI layer requires PySide6.  All core logic lives in
``evolution/interactive_eval.py`` and works without a GUI.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QListWidget, QListWidgetItem, QAbstractItemView,
    QSplitter, QScrollArea, QFrame,
)

if TYPE_CHECKING:
    from yane.evolution.interactive_eval import InteractiveEvaluator


class _GenomeCard(QFrame):
    """Compact display for one genome (id + basic topology info)."""

    def __init__(self, genome, parent=None) -> None:
        super().__init__(parent)
        self.genome = genome
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        gid = genome._genome_id
        n_nodes = len(genome.nodes)
        n_conn = sum(len(n.connections) for n in genome.nodes)
        layout.addWidget(QLabel(f"<b>Genome #{gid}</b>"))
        layout.addWidget(QLabel(f"Nodes: {n_nodes}   Connections: {n_conn}"))
        if hasattr(genome, "fitness") and genome.fitness is not None:
            layout.addWidget(QLabel(f"Fitness: {genome.fitness:.4f}"))


class InteractiveEvalPanel(QWidget):
    """PySide6 widget for interactive genome evaluation.

    Parameters
    ----------
    parent :
        Optional parent widget.
    poll_interval_ms :
        How often (ms) to check for new pending genomes (default: 500).
    """

    def __init__(self, parent=None, poll_interval_ms: int = 500) -> None:
        super().__init__(parent)
        self._evaluator: InteractiveEvaluator | None = None
        self._pending_snapshot: list = []   # genomes currently displayed
        self._implicit_start: float = 0.0

        self._build_ui()
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(poll_interval_ms)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_evaluator(self, evaluator: "InteractiveEvaluator") -> None:
        """Attach the evaluator whose pending genomes this panel services."""
        self._evaluator = evaluator
        mode = evaluator.mode
        # Show only the relevant input widget for this mode
        self._rating_widget.setVisible(mode in ("rating", "implicit"))
        self._pairwise_widget.setVisible(mode == "pairwise")
        self._ranking_widget.setVisible(mode == "ranking")
        self._mode_label.setText(f"Mode: <b>{mode}</b>")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QHBoxLayout()
        self._mode_label = QLabel("Mode: —")
        header.addWidget(self._mode_label)
        header.addStretch()
        self._stats_label = QLabel("Queries: 0  |  Surrogate skips: 0")
        header.addWidget(self._stats_label)
        root.addLayout(header)

        # Genome display area (two cards side-by-side)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._card_a = QLabel("No pending genome")
        self._card_a.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._card_a.setWordWrap(True)
        self._card_b = QLabel("")
        self._card_b.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._card_b.setWordWrap(True)
        scroll_a = QScrollArea()
        scroll_a.setWidgetResizable(True)
        scroll_a.setWidget(self._card_a)
        scroll_b = QScrollArea()
        scroll_b.setWidgetResizable(True)
        scroll_b.setWidget(self._card_b)
        self._splitter.addWidget(scroll_a)
        self._splitter.addWidget(scroll_b)
        root.addWidget(self._splitter, stretch=1)

        # --- Rating / implicit mode ---
        self._rating_widget = QGroupBox("Rate this genome (0 = bad, 100 = perfect)")
        rw_layout = QVBoxLayout(self._rating_widget)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(50)
        rw_layout.addWidget(self._slider)
        self._submit_rating_btn = QPushButton("Submit Rating")
        self._submit_rating_btn.clicked.connect(self._on_submit_rating)
        rw_layout.addWidget(self._submit_rating_btn)
        root.addWidget(self._rating_widget)

        # --- Pairwise mode ---
        self._pairwise_widget = QGroupBox("Pairwise Comparison")
        pw_layout = QHBoxLayout(self._pairwise_widget)
        self._prefer_a_btn = QPushButton("Prefer LEFT")
        self._prefer_a_btn.clicked.connect(lambda: self._on_pairwise(0))
        self._prefer_b_btn = QPushButton("Prefer RIGHT")
        self._prefer_b_btn.clicked.connect(lambda: self._on_pairwise(1))
        pw_layout.addWidget(self._prefer_a_btn)
        pw_layout.addWidget(self._prefer_b_btn)
        root.addWidget(self._pairwise_widget)

        # --- Ranking mode ---
        self._ranking_widget = QGroupBox("Rank genomes (drag to reorder, best at top)")
        rk_layout = QVBoxLayout(self._ranking_widget)
        self._rank_list = QListWidget()
        self._rank_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        rk_layout.addWidget(self._rank_list)
        self._submit_ranking_btn = QPushButton("Submit Ranking")
        self._submit_ranking_btn.clicked.connect(self._on_submit_ranking)
        rk_layout.addWidget(self._submit_ranking_btn)
        root.addWidget(self._ranking_widget)

        # Initially hide all input widgets
        self._rating_widget.setVisible(False)
        self._pairwise_widget.setVisible(False)
        self._ranking_widget.setVisible(False)

    # ------------------------------------------------------------------
    # Poll & display
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        if self._evaluator is None:
            return
        pending = self._evaluator.pending_genome_ids()
        if not pending:
            return
        registry = self._evaluator._genome_registry
        genomes = [registry[gid] for gid in pending if gid in registry]
        if not genomes:
            return

        self._pending_snapshot = genomes
        mode = self._evaluator.mode

        # Update stats
        self._stats_label.setText(
            f"Queries: {self._evaluator.query_count}  |  "
            f"Surrogate skips: {self._evaluator.surrogate_skips}"
        )

        if mode in ("rating", "implicit"):
            g = genomes[0]
            self._card_a.setText(self._genome_text(g))
            self._card_b.setText("")
        elif mode == "pairwise" and len(genomes) >= 2:
            self._card_a.setText(self._genome_text(genomes[0]))
            self._card_b.setText(self._genome_text(genomes[1]))
        elif mode == "ranking":
            self._rank_list.clear()
            for g in genomes:
                item = QListWidgetItem(f"Genome #{g._genome_id}")
                item.setData(Qt.ItemDataRole.UserRole, g._genome_id)
                self._rank_list.addItem(item)
            self._card_a.setText("\n".join(self._genome_text(g) for g in genomes))

        if mode == "implicit":
            import time
            self._implicit_start = time.time()

    @staticmethod
    def _genome_text(genome) -> str:
        gid = genome._genome_id
        n_nodes = len(genome.nodes)
        n_conn = sum(len(n.connections) for n in genome.nodes)
        fitness = getattr(genome, "fitness", None)
        fitness_str = f"{fitness:.4f}" if fitness is not None else "—"
        return (
            f"Genome #{gid}\n"
            f"Nodes: {n_nodes}   Connections: {n_conn}\n"
            f"Fitness: {fitness_str}"
        )

    # ------------------------------------------------------------------
    # Feedback handlers
    # ------------------------------------------------------------------

    def _on_submit_rating(self) -> None:
        if not self._pending_snapshot or self._evaluator is None:
            return
        g = self._pending_snapshot[0]
        mode = self._evaluator.mode
        if mode == "implicit":
            import time
            value = time.time() - self._implicit_start
        else:
            value = float(self._slider.value())
        self._evaluator.submit_feedback(g._genome_id, value)
        self._pending_snapshot.clear()

    def _on_pairwise(self, winner: int) -> None:
        """winner=0 → left genome wins, 1 → right genome wins."""
        if len(self._pending_snapshot) < 2 or self._evaluator is None:
            return
        g_a = self._pending_snapshot[0]
        g_b = self._pending_snapshot[1]
        if winner == 0:
            self._evaluator.submit_feedback(g_a._genome_id, 0)  # a won
        else:
            self._evaluator.submit_feedback(g_b._genome_id, 0)  # b won (raw Elo update)
            self._evaluator.submit_feedback(g_a._genome_id, 1)  # a lost
        self._pending_snapshot.clear()

    def _on_submit_ranking(self) -> None:
        if self._evaluator is None:
            return
        for rank in range(self._rank_list.count()):
            item = self._rank_list.item(rank)
            gid = item.data(Qt.ItemDataRole.UserRole)
            self._evaluator.submit_feedback(gid, float(rank + 1))
        self._pending_snapshot.clear()
