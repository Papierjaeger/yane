"""Inspect tab: test cases, manual forward pass, sequence runner for best genome."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QFrame, QLabel, QGroupBox, QCheckBox, QDoubleSpinBox,
    QSpinBox, QComboBox, QFormLayout,
)
from PySide6.QtCore import Qt

from yane.gui._helpers import _label, _divider
from yane.gui.canvas import SensitivityChart


def _fmt_value(v: float, denormalized: bool) -> str:
    """Format a single value for display. When denormalized and close to an integer,
    drop decimals so e.g. 5 → "5" instead of "5.000"."""
    if denormalized and abs(v - round(v)) < 0.01 and abs(v) < 1e9:
        return f"{int(round(v))}"
    return f"{v:.2f}" if denormalized else f"{v:.3f}"


def _fmt_list(vs: list[float], scale: list[float] | None, denormalized: bool) -> str:
    if denormalized and scale:
        vs = [v * s for v, s in zip(vs, scale)]
    return "[" + ", ".join(_fmt_value(v, denormalized) for v in vs) + "]"


_TOL_NORMALIZED = 0.5    # per-output threshold: prediction is in the correct half of [0,1]


class _TestCaseRow:
    """One persistent row in the test-cases table. Created once, updated in place."""

    _MONO = "font-family: monospace; font-size: 12px;"

    def __init__(self, layout, inputs: list[float], expected: list[float],
                 input_scale: list[float] | None = None,
                 output_scale: list[float] | None = None,
                 denormalized: bool = False) -> None:
        self._inputs        = inputs
        self._expected      = expected
        self._input_scale   = input_scale
        self._output_scale  = output_scale
        self._denormalized  = denormalized
        self._delta: float | None = None
        self._correct: bool | None = None

        row = QWidget()
        rlay = QHBoxLayout(row)
        rlay.setContentsMargins(0, 2, 0, 2)

        self._in_lbl    = QLabel();    self._in_lbl.setMinimumWidth(120);   self._in_lbl.setStyleSheet(self._MONO)
        self._exp_lbl   = QLabel();    self._exp_lbl.setMinimumWidth(80);   self._exp_lbl.setStyleSheet(self._MONO)
        self._out_lbl   = QLabel("—"); self._out_lbl.setMinimumWidth(90);   self._out_lbl.setStyleSheet(self._MONO)
        self._delta_lbl = QLabel("—"); self._delta_lbl.setMinimumWidth(60); self._delta_lbl.setStyleSheet(self._MONO + " color: #a6adc8;")
        self._tick      = QLabel("?"); self._tick.setFixedWidth(30)
        self._tick.setStyleSheet("color: #585b70; font-size: 16px; font-weight: bold;")

        self._refresh_static_labels()
        for w in (self._in_lbl, self._exp_lbl, self._out_lbl, self._delta_lbl, self._tick):
            rlay.addWidget(w)

        layout.addWidget(row)

    def _refresh_static_labels(self) -> None:
        self._in_lbl.setText(_fmt_list(self._inputs, self._input_scale, self._denormalized))
        self._exp_lbl.setText(_fmt_list(self._expected, self._output_scale, self._denormalized))

    def update(self, genome) -> None:
        if genome is None:
            self._delta = None
            self._correct = None
            self._out_lbl.setText("—")
            self._delta_lbl.setText("—")
            self._tick.setText("?")
            self._tick.setStyleSheet("color: #585b70; font-size: 16px; font-weight: bold;")
            return
        try:
            outputs = genome.forward(self._inputs)
        except Exception:
            self._delta = None
            self._correct = None
            self._out_lbl.setText("err")
            self._delta_lbl.setText("—")
            self._tick.setText("✗")
            self._tick.setStyleSheet("color: #f38ba8; font-size: 16px; font-weight: bold;")
            return

        # Correctness check matches the displayed units: when denormalized, the tick
        # is ✓ only if every output rounds to the same integer as the expected raw
        # value. Otherwise we keep the normalized tolerance the genome was trained on.
        if self._denormalized and self._output_scale:
            correct = all(
                round(o * s) == round(e * s)
                for o, e, s in zip(outputs, self._expected, self._output_scale)
            )
        else:
            correct = all(abs(o - e) < _TOL_NORMALIZED for o, e in zip(outputs, self._expected))
        self._out_lbl.setText(_fmt_list(list(outputs), self._output_scale, self._denormalized))

        # Δ shown in displayed units (raw if denormalized, normalized otherwise).
        diffs = [o - e for o, e in zip(outputs, self._expected)]
        if self._denormalized and self._output_scale:
            diffs = [d * s for d, s in zip(diffs, self._output_scale)]
        total = sum(abs(d) for d in diffs)
        self._delta = total
        self._correct = correct
        self._delta_lbl.setText(f"{total:.2f}" if self._denormalized else f"{total:.3f}")

        tick, color = ("✓", "#a6e3a1") if correct else ("✗", "#f38ba8")
        self._tick.setText(tick)
        self._tick.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")


class _SequenceStepRow:
    """One row in the sequence-trace table. Created once, updated or cleared in place."""

    _MONO = "font-family: monospace; font-size: 12px;"

    def __init__(self, layout, step_idx: int,
                 inputs: list[float], expected: list[float],
                 input_scale: list[float] | None = None,
                 output_scale: list[float] | None = None,
                 denormalized: bool = False) -> None:
        self._inputs        = inputs
        self._expected      = expected
        self._input_scale   = input_scale
        self._output_scale  = output_scale
        self._denormalized  = denormalized
        self._delta: float | None = None
        self._correct: bool | None = None

        row = QWidget()
        rlay = QHBoxLayout(row)
        rlay.setContentsMargins(2, 2, 2, 2)
        rlay.setSpacing(6)

        step_lbl = QLabel(f"{step_idx + 1:>2}.")
        step_lbl.setFixedWidth(24)
        step_lbl.setStyleSheet(self._MONO + " color: #585b70;")

        self._in_lbl    = QLabel(_fmt_list(inputs,   input_scale,  denormalized))
        self._exp_lbl   = QLabel(_fmt_list(expected, output_scale, denormalized))
        self._out_lbl   = QLabel("—")
        self._delta_lbl = QLabel("—")
        self._tick      = QLabel("?")

        for lbl, w in ((self._in_lbl, 75), (self._exp_lbl, 75),
                       (self._out_lbl, 75), (self._delta_lbl, 52)):
            lbl.setMinimumWidth(w)
            lbl.setStyleSheet(self._MONO)

        self._delta_lbl.setStyleSheet(self._MONO + " color: #a6adc8;")
        self._tick.setFixedWidth(22)
        self._tick.setStyleSheet("color: #585b70; font-size: 14px; font-weight: bold;")

        self._row_widget = row
        for w in (step_lbl, self._in_lbl, self._exp_lbl,
                  self._out_lbl, self._delta_lbl, self._tick):
            rlay.addWidget(w)
        rlay.addStretch()

        layout.addWidget(row)

    def set_result(self, outputs: list[float]) -> float:
        self._out_lbl.setText(_fmt_list(list(outputs), self._output_scale, self._denormalized))
        # Δ shown in displayed units (raw if denormalized, normalized otherwise).
        # When denormalized, the tick matches what the user sees: ✓ only if every
        # output rounds to the same integer as the expected raw value.
        norm_delta = sum(abs(o - e) for o, e in zip(outputs, self._expected))
        if self._denormalized and self._output_scale:
            shown = sum(abs((o - e) * s) for o, e, s
                        in zip(outputs, self._expected, self._output_scale))
            self._delta_lbl.setText(f"{shown:.2f}")
            correct = all(
                round(o * s) == round(e * s)
                for o, e, s in zip(outputs, self._expected, self._output_scale)
            )
        else:
            self._delta_lbl.setText(f"{norm_delta:.3f}")
            correct = norm_delta < _TOL_NORMALIZED * len(self._expected)
        self._delta = norm_delta
        self._correct = correct
        tick, color = ("✓", "#a6e3a1") if correct else ("✗", "#f38ba8")
        self._tick.setText(tick)
        self._tick.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        return self._delta

    def clear(self) -> None:
        self._delta = None
        self._correct = None
        self._out_lbl.setText("—")
        self._delta_lbl.setText("—")
        self._tick.setText("?")
        self._tick.setStyleSheet("color: #585b70; font-size: 14px; font-weight: bold;")

    def set_highlighted(self, on: bool) -> None:
        self._row_widget.setStyleSheet(
            "background-color: #313244; border-radius: 4px;" if on else ""
        )


class InspectTab(QWidget):
    """Shows the best genome's outputs for known test cases and manual inputs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._genome = None
        self._example = None
        self._ne = None
        self._test_rows: list[_TestCaseRow] = []
        self._seq_rows: list[_SequenceStepRow] = []
        self._seq_samples: list[tuple] = []
        self._seq_step: int = 0      # number of steps executed so far
        self._memory_labels: list[QLabel] = []
        self._mem_sig: tuple = ()    # (innovation,...) of current memory nodes
        self._pending_genome = None
        self._pending_mem: dict = {}

        # Outer layout + scroll area
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)
        scroll.setWidget(inner)

        # ── Test Cases ─────────────────────────────────────────────────────
        self._test_group = QGroupBox("Test Cases — best genome output vs. expected")
        self._test_inner = QVBoxLayout(self._test_group)
        # Denormalize toggle — visible only for examples with input/output scales.
        # When on, values are shown in raw units (e.g. 5 * 5 → 25 instead of 0.56 → 0.31).
        self._denorm_row = QWidget()
        denorm_layout = QHBoxLayout(self._denorm_row)
        denorm_layout.setContentsMargins(0, 0, 0, 4)
        self.chk_denormalize = QCheckBox("Denormalize values (show raw units)")
        self.chk_denormalize.setChecked(True)
        self.chk_denormalize.setToolTip(
            "Aus: Werte werden im normalisierten [0,1]-Bereich angezeigt.\n"
            "An: Werte werden in Roheinheiten dargestellt (z. B. 5 * 5 → 25)."
        )
        self.chk_denormalize.toggled.connect(self._on_denormalize_toggled)
        denorm_layout.addWidget(self.chk_denormalize)
        denorm_layout.addStretch()
        self._denorm_row.setVisible(False)
        self._test_inner.addWidget(self._denorm_row)
        self._placeholder = _label("Select an example to see test cases.", "sectionTitle")
        self._test_inner.addWidget(self._placeholder)
        self._test_sum_lbl: QLabel | None = None
        layout.addWidget(self._test_group)

        # ── Manual Forward Pass ────────────────────────────────────────────
        manual = QGroupBox("Manual Forward Pass")
        manual_layout = QVBoxLayout(manual)

        self._input_widgets: list[QDoubleSpinBox] = []
        self._inputs_form = QWidget()
        self._inputs_form_layout = QFormLayout(self._inputs_form)
        manual_layout.addWidget(self._inputs_form)

        self._no_genome_lbl = _label(
            "Start training first — the best genome will be used here.", "sectionTitle")
        self._no_genome_lbl.setWordWrap(True)
        manual_layout.addWidget(self._no_genome_lbl)

        # Button row: reset memory + forward pass
        btn_row = QWidget()
        btn_row_layout = QHBoxLayout(btn_row)
        btn_row_layout.setContentsMargins(0, 0, 0, 0)
        btn_row_layout.setSpacing(8)
        self.btn_reset_mem = QPushButton("↺  Reset memory")
        self.btn_reset_mem.setEnabled(False)
        self.btn_reset_mem.setToolTip(
            "Setzt den internen Zustand zurück (genome.reset()).\n"
            "Der Forward Pass setzt die aktuelle Sequenz fort — bei\n"
            "sequenziellen Netzen hier zurücksetzen um neu zu starten."
        )
        self.btn_reset_mem.clicked.connect(self._reset_memory)
        self.btn_run = QPushButton("▶  Forward Pass")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_manual)
        btn_row_layout.addWidget(self.btn_reset_mem)
        btn_row_layout.addWidget(self.btn_run)
        manual_layout.addWidget(btn_row)

        # Outputs
        self._output_group = QGroupBox("Outputs")
        self._output_layout = QFormLayout(self._output_group)
        self._output_labels: list[QLabel] = []
        manual_layout.addWidget(self._output_group)

        # Interpreted action (shown only when example provides action_display_fn)
        self._action_row = QWidget()
        action_row_layout = QHBoxLayout(self._action_row)
        action_row_layout.setContentsMargins(0, 0, 0, 0)
        action_row_layout.addWidget(_label("Aktion:", "sectionTitle"))
        self._action_lbl = QLabel("—")
        self._action_lbl.setStyleSheet(
            "font-family: monospace; font-size: 13px; font-weight: bold; color: #cdd6f4;"
        )
        action_row_layout.addWidget(self._action_lbl)
        action_row_layout.addStretch()
        self._action_row.setVisible(False)
        manual_layout.addWidget(self._action_row)

        # Memory state display (hidden until memory nodes exist)
        self._memory_group = QGroupBox("Memory state")
        self._memory_form  = QFormLayout(self._memory_group)
        self._memory_group.setVisible(False)
        manual_layout.addWidget(self._memory_group)

        layout.addWidget(manual)

        # ── Sequence Trace ─────────────────────────────────────────────────
        self._seq_group = QGroupBox("Sequence Trace")
        seq_outer_layout = QVBoxLayout(self._seq_group)
        self._seq_group.setVisible(False)

        # Sequence control buttons
        seq_btn_row = QWidget()
        seq_btn_layout = QHBoxLayout(seq_btn_row)
        seq_btn_layout.setContentsMargins(0, 0, 0, 0)
        seq_btn_layout.setSpacing(6)
        self.btn_seq_prev  = QPushButton("◀  Back")
        self.btn_seq_next  = QPushButton("▶  Next step")
        self.btn_seq_all   = QPushButton("⏭  Run all")
        self.btn_seq_reset = QPushButton("↺  Reset")
        for b in (self.btn_seq_prev, self.btn_seq_next,
                  self.btn_seq_all, self.btn_seq_reset):
            b.setEnabled(False)
            seq_btn_layout.addWidget(b)
        self.btn_seq_prev.clicked.connect(self._seq_prev)
        self.btn_seq_next.clicked.connect(self._seq_next)
        self.btn_seq_all.clicked.connect(self._seq_run_all)
        self.btn_seq_reset.clicked.connect(self._seq_reset)
        seq_outer_layout.addWidget(seq_btn_row)

        # Header row for sequence table
        seq_hdr = QWidget()
        seq_hdr_layout = QHBoxLayout(seq_hdr)
        seq_hdr_layout.setContentsMargins(2, 0, 2, 0)
        seq_hdr_layout.setSpacing(6)
        for txt, w, fixed in [("#", 24, True), ("Input", 75, False), ("Expected", 75, False),
                               ("Output", 75, False), ("Δ", 52, False), ("", 22, True)]:
            lbl = _label(txt, "sectionTitle")
            if fixed:
                lbl.setFixedWidth(w)
            else:
                lbl.setMinimumWidth(w)
            seq_hdr_layout.addWidget(lbl)
        seq_hdr_layout.addStretch()
        seq_outer_layout.addWidget(seq_hdr)

        # Sequence step rows added dynamically by _rebuild_sequence_table()
        self._seq_rows_container = QWidget()
        self._seq_rows_layout = QVBoxLayout(self._seq_rows_container)
        self._seq_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._seq_rows_layout.setSpacing(1)
        seq_outer_layout.addWidget(self._seq_rows_container)

        seq_outer_layout.addWidget(_divider())
        self._acc_fitness_lbl = _label("", "sectionTitle")
        self._acc_fitness_lbl.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #cdd6f4;"
            " font-weight: bold; padding-top: 2px;"
        )
        seq_outer_layout.addWidget(self._acc_fitness_lbl)

        layout.addWidget(self._seq_group)

        # ── Sensitivity Analysis ────────────────────────────────────────
        self._sens_group = QGroupBox("Sensitivitätsanalyse (Input-Einfluss + Tote Knoten)")
        sens_layout = QVBoxLayout(self._sens_group)
        self._sensitivity_chart = SensitivityChart()
        self._sensitivity_chart.setMinimumHeight(60)
        sens_layout.addWidget(self._sensitivity_chart)
        self._sens_info_lbl = _label(
            "Analysiert den Einfluss jedes Inputs und erkennt nie-aktive Hidden-Knoten.",
            "sectionTitle"
        )
        self._sens_info_lbl.setWordWrap(True)
        sens_layout.addWidget(self._sens_info_lbl)
        layout.addWidget(self._sens_group)

        # ── Ensemble ────────────────────────────────────────────────────────
        self._ensemble_group = QGroupBox("Ensemble Inference")
        self._ensemble_group.setVisible(False)
        ens_layout = QVBoxLayout(self._ensemble_group)

        ens_row = QWidget()
        ens_row_layout = QHBoxLayout(ens_row)
        ens_row_layout.setContentsMargins(0, 0, 0, 0)
        ens_row_layout.setSpacing(6)

        ens_row_layout.addWidget(QLabel("k:"))
        self.spin_ensemble_k = QSpinBox()
        self.spin_ensemble_k.setRange(1, 20)
        self.spin_ensemble_k.setValue(3)
        ens_row_layout.addWidget(self.spin_ensemble_k)

        ens_row_layout.addWidget(QLabel("mode:"))
        self.combo_ensemble_mode = QComboBox()
        self.combo_ensemble_mode.addItems(["mean", "vote", "weighted"])
        ens_row_layout.addWidget(self.combo_ensemble_mode)

        self.btn_ensemble_run = QPushButton("▶  Run Ensemble")
        self.btn_ensemble_run.clicked.connect(self._run_ensemble)
        self.btn_ensemble_run.setEnabled(False)
        ens_row_layout.addWidget(self.btn_ensemble_run)

        self.btn_ensemble_forward = QPushButton("▶  Forward (current inputs)")
        self.btn_ensemble_forward.clicked.connect(self._run_ensemble_forward)
        self.btn_ensemble_forward.setEnabled(False)
        ens_row_layout.addWidget(self.btn_ensemble_forward)

        ens_layout.addWidget(ens_row)

        self._ensemble_output = QLabel("—")
        self._ensemble_output.setWordWrap(True)
        self._ensemble_output.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #a6e3a1; padding: 4px;"
        )
        ens_layout.addWidget(self._ensemble_output)

        layout.addWidget(self._ensemble_group)

        layout.addStretch()

    # ------------------------------------------------------------------

    def set_example(self, example) -> None:
        self._example = example
        has_scales = bool(example and (example.input_scale or example.output_scale))
        self._denorm_row.setVisible(has_scales)
        has_action = bool(example and getattr(example, "action_display_fn", None))
        self._action_row.setVisible(has_action)
        self._rebuild_test_rows()
        self._rebuild_input_widgets(
            example.n_inputs  if example else 0,
            example.n_outputs if example else 0,
        )
        self._rebuild_sequence_table()

    def reset_genome(self) -> None:
        if self._genome is not None:
            self._genome._clear()
        self._genome = None
        self._no_genome_lbl.setVisible(True)
        self.btn_run.setEnabled(False)
        self.btn_reset_mem.setEnabled(False)
        for row in self._test_rows:
            row.update(None)
        self._update_test_sum()
        self._memory_group.setVisible(False)
        self._mem_sig = ()
        self._memory_labels.clear()
        if self._seq_rows:
            self._seq_step = 0
            for row in self._seq_rows:
                row.clear()
            self._acc_fitness_lbl.setText("")
        self._update_seq_buttons()
        self._sensitivity_chart.clear()

    def update_genome(self, genome, mem: dict) -> None:
        self._pending_genome = genome
        self._pending_mem = mem
        if not self.isVisible():
            return
        self._apply_genome_update(genome, mem)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pending_genome is not None:
            self._apply_genome_update(self._pending_genome, self._pending_mem)
            self._pending_genome = None
            self._pending_mem = {}

    def _apply_genome_update(self, genome, mem: dict) -> None:
        self._no_genome_lbl.setVisible(False)
        if self._genome is not None and self._genome is not genome:
            self._genome._clear()
        self._genome = genome
        self.btn_run.setEnabled(bool(self._input_widgets))
        self.btn_reset_mem.setEnabled(True)
        # Always start the test-case rollout from a clean state so the displayed
        # output matches the evaluator's first forward pass after env.reset() /
        # episode start. For stateless genomes this is a no-op; for stateful ones
        # the sequence below picks up the carry-over via memory neurons.
        if self._test_rows:
            genome.reset()
        for row in self._test_rows:
            row.update(genome)
        self._update_test_sum()
        if self._seq_samples:
            self._seq_run_all()
        else:
            self._update_memory_display()
        self._update_seq_buttons()
        self._update_sensitivity(genome)

    # ── NeuroEvolution reference (set by MainWindow) ─────────────────────────

    def set_ne(self, ne) -> None:
        """Set NeuroEvolution instance reference (called by MainWindow)."""
        self._ne = ne
        self._ensemble_group.setVisible(ne is not None)
        self.btn_ensemble_run.setEnabled(ne is not None)
        self.btn_ensemble_forward.setEnabled(ne is not None)

    # ── Ensemble ─────────────────────────────────────────────────────────────

    def _run_ensemble(self) -> None:
        """Build and run the ensemble on all test cases."""
        if self._ne is None:
            return
        k = self.spin_ensemble_k.value()
        mode = self.combo_ensemble_mode.currentText()
        try:
            ens = self._ne.make_ensemble(k=k, mode=mode)
            results = []
            for row in self._test_rows:
                try:
                    out = ens.forward(row._inputs)
                    results.append(list(out))
                except Exception:
                    results.append(None)
            self._ensemble_output.setText(
                f"Ensemble (k={k}, mode={mode}): "
                + ", ".join(
                    _fmt_list(r, None, False) if r is not None else "err"
                    for r in results
                )
            )
        except Exception as exc:
            self._ensemble_output.setText(f"❌ Ensemble failed: {exc}")

    def _run_ensemble_forward(self) -> None:
        """Run ensemble on the current manual input values."""
        if self._ne is None:
            return
        k = self.spin_ensemble_k.value()
        mode = self.combo_ensemble_mode.currentText()
        try:
            inputs = [w.value() for w in self._input_widgets]
            ens = self._ne.make_ensemble(k=k, mode=mode)
            out = list(ens.forward(inputs))
            self._ensemble_output.setText(
                f"Ensemble (k={k}, mode={mode}): {_fmt_list(out, None, False)}"
            )
        except Exception as exc:
            self._ensemble_output.setText(f"❌ Ensemble forward failed: {exc}")

    # ------------------------------------------------------------------

    def _rebuild_test_rows(self) -> None:
        self._test_rows.clear()
        self._test_sum_lbl = None
        # Strip everything except the persistent denormalize checkbox row at index 0.
        while self._test_inner.count() > 1:
            item = self._test_inner.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        tc = self._example.test_cases if self._example else None
        if not tc:
            self._placeholder = _label(
                "No fixed test cases for this example." if self._example
                else "Select an example to see test cases.", "sectionTitle")
            self._test_inner.addWidget(self._placeholder)
            return

        header = QWidget()
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(0, 0, 0, 0)
        for txt, w in [("Inputs", 120), ("Expected", 80), ("Output", 90), ("Δ", 60), ("", 40)]:
            lbl = _label(txt, "sectionTitle")
            lbl.setMinimumWidth(w)
            hlay.addWidget(lbl)
        self._test_inner.addWidget(header)

        in_scale  = self._example.input_scale
        out_scale = self._example.output_scale
        denorm    = self._denorm_active()
        for inputs, expected in tc:
            row = _TestCaseRow(self._test_inner, inputs, expected,
                               in_scale, out_scale, denorm)
            self._test_rows.append(row)

        sep = _divider()
        self._test_inner.addWidget(sep)
        self._test_sum_lbl = _label("", "sectionTitle")
        self._test_sum_lbl.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #cdd6f4;"
            " font-weight: bold; padding-top: 2px;"
        )
        self._test_inner.addWidget(self._test_sum_lbl)

    def _rebuild_input_widgets(self, n_inputs: int, n_outputs: int) -> None:
        while self._inputs_form_layout.rowCount():
            self._inputs_form_layout.removeRow(0)
        self._input_widgets.clear()

        in_scale = self._example.input_scale if self._example else None
        denorm   = self._denorm_active()
        for i in range(n_inputs):
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            if denorm and in_scale and i < len(in_scale):
                scale = in_scale[i]
                # Integer-style stepping when scale ≥ 5 and is itself integer-ish.
                if scale >= 5 and abs(scale - round(scale)) < 0.01:
                    spin.setDecimals(0)
                    spin.setSingleStep(1)
                else:
                    spin.setDecimals(2)
                    spin.setSingleStep(scale / 10)
            else:
                spin.setDecimals(4)
                spin.setSingleStep(0.1)
            self._inputs_form_layout.addRow(f"Input {i}:", spin)
            self._input_widgets.append(spin)

        while self._output_layout.rowCount():
            self._output_layout.removeRow(0)
        self._output_labels.clear()

        for i in range(n_outputs):
            lbl = _label("—", "statValue")
            self._output_layout.addRow(f"Output {i}:", lbl)
            self._output_labels.append(lbl)

        self.btn_run.setEnabled(n_inputs > 0 and self._genome is not None)

    def _rebuild_sequence_table(self) -> None:
        self._seq_rows.clear()
        while self._seq_rows_layout.count():
            item = self._seq_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        samples = (self._example.sequence_samples if self._example else None) or []
        if not samples:
            self._seq_group.setVisible(False)
            self._seq_samples = []
            self._seq_step = 0
            return

        self._seq_samples = samples
        self._seq_step = 0
        self._seq_group.setVisible(True)
        self._acc_fitness_lbl.setText("")

        in_scale  = self._example.input_scale
        out_scale = self._example.output_scale
        denorm    = self._denorm_active()
        for idx, (inp, exp) in enumerate(samples):
            row = _SequenceStepRow(self._seq_rows_layout, idx, inp, exp,
                                   in_scale, out_scale, denorm)
            self._seq_rows.append(row)

        self._update_seq_buttons()

    def _rebuild_memory_display(self, mem_nodes: list) -> None:
        while self._memory_form.rowCount():
            self._memory_form.removeRow(0)
        self._memory_labels.clear()

        all_hidden = [n for n in self._genome.nodes if n.type.value == "hidden"]
        for node in mem_nodes:
            idx = all_hidden.index(node) if node in all_hidden else -1
            act = node.activation.value[:3]
            lbl_text = f"H{idx}  {act}  b={node.bias:.2f}"
            val_lbl = _label(f"{node.value:.5f}", "statValue")
            self._memory_form.addRow(lbl_text + ":", val_lbl)
            self._memory_labels.append(val_lbl)

        n = len(mem_nodes)
        self._memory_group.setTitle(f"Memory state  ({n} nodes)")

    def _update_memory_display(self) -> None:
        if self._genome is None:
            self._memory_group.setVisible(False)
            return
        mem_nodes = [n for n in self._genome.nodes
                     if n.type.value == "hidden" and n.persist_value]
        if not mem_nodes:
            self._memory_group.setVisible(False)
            return
        self._memory_group.setVisible(True)
        curr_sig = tuple(n.innovation for n in mem_nodes)
        if curr_sig != self._mem_sig:
            self._mem_sig = curr_sig
            self._rebuild_memory_display(mem_nodes)
        else:
            for lbl, node in zip(self._memory_labels, mem_nodes):
                lbl.setText(f"{node.value:.5f}")

    def _update_seq_buttons(self) -> None:
        has_genome  = self._genome is not None
        has_samples = bool(self._seq_samples)
        at_start    = self._seq_step == 0
        at_end      = self._seq_step >= len(self._seq_samples)
        self.btn_seq_prev.setEnabled(has_genome and has_samples and not at_start)
        self.btn_seq_next.setEnabled(has_genome and has_samples and not at_end)
        self.btn_seq_all.setEnabled(has_genome and has_samples)
        self.btn_seq_reset.setEnabled(has_genome and has_samples)

    def _update_test_sum(self) -> None:
        if self._test_sum_lbl is None:
            return
        deltas = [r._delta for r in self._test_rows if r._delta is not None]
        if not deltas:
            self._test_sum_lbl.setText("")
            return
        total   = sum(deltas)
        correct = sum(1 for r in self._test_rows if r._correct)
        n       = len(self._test_rows)
        denorm  = bool(self._test_rows and self._test_rows[0]._denormalized)
        fmt     = f"{total:.2f}" if denorm else f"{total:.4f}"
        self._test_sum_lbl.setText(f"Σ Δ: {fmt}  |  {correct}/{n} correct")

    def _update_acc_fitness(self) -> None:
        if not self._seq_rows:
            return
        deltas = [r._delta for r in self._seq_rows if r._delta is not None]
        if not deltas:
            self._acc_fitness_lbl.setText("")
            return
        total   = sum(deltas)
        correct = sum(1 for r in self._seq_rows if r._correct)
        done    = len(deltas)
        self._acc_fitness_lbl.setText(
            f"Σ Δ: {total:.4f}  |  Fitness: {-total:.4f}  |  "
            f"{correct}/{done} correct"
        )

    # ── Sequence step navigation ───────────────────────────────────────

    def _seq_next(self) -> None:
        if self._genome is None or self._seq_step >= len(self._seq_samples):
            return
        if self._seq_step == 0:
            self._genome.reset()
        inp, _exp = self._seq_samples[self._seq_step]
        try:
            outputs = self._genome.forward(inp)
        except Exception:
            return
        self._seq_rows[self._seq_step].set_result(outputs)
        self._seq_step += 1
        self._update_memory_display()
        self._update_acc_fitness()
        self._update_seq_buttons()

    def _seq_prev(self) -> None:
        if self._genome is None or self._seq_step == 0:
            return
        self._seq_step -= 1
        self._seq_rows[self._seq_step].clear()
        self._genome.reset()
        for i in range(self._seq_step):
            try:
                self._genome.forward(self._seq_samples[i][0])
            except Exception:
                break
        self._update_memory_display()
        self._update_acc_fitness()
        self._update_seq_buttons()

    def _seq_run_all(self) -> None:
        if self._genome is None:
            return
        self._genome.reset()
        self._seq_step = 0
        n_out = self._example.n_outputs if self._example else 1
        for i, (inp, _exp) in enumerate(self._seq_samples):
            try:
                outputs = self._genome.forward(inp)
            except Exception:
                outputs = [0.0] * n_out
            self._seq_rows[i].set_result(outputs)
            self._seq_step = i + 1
        self._update_memory_display()
        self._update_acc_fitness()
        self._update_seq_buttons()

    def _seq_reset(self) -> None:
        if self._genome is not None:
            self._genome.reset()
        self._seq_step = 0
        for row in self._seq_rows:
            row.clear()
        self._acc_fitness_lbl.setText("")
        self._update_memory_display()
        self._update_seq_buttons()

    # ── Manual forward pass ────────────────────────────────────────────

    def _reset_memory(self) -> None:
        if self._genome is None:
            return
        self._genome.reset()
        if self._seq_rows:
            self._seq_step = 0
            for row in self._seq_rows:
                row.clear()
            self._acc_fitness_lbl.setText("")
            self._update_seq_buttons()
        self._update_memory_display()

    def _run_manual(self) -> None:
        if self._genome is None:
            return
        inputs = [w.value() for w in self._input_widgets]
        in_scale  = self._example.input_scale  if self._example else None
        out_scale = self._example.output_scale if self._example else None
        denorm    = self._denorm_active()
        # Spinbox holds raw values in denorm mode → convert to normalized for the genome.
        if denorm and in_scale:
            inputs = [v / s if s else v for v, s in zip(inputs, in_scale)]
        try:
            outputs = self._genome.forward(inputs)
            for i, lbl in enumerate(self._output_labels):
                if i >= len(outputs):
                    lbl.setText("—")
                    continue
                v = outputs[i]
                if denorm and out_scale and i < len(out_scale):
                    v *= out_scale[i]
                    lbl.setText(_fmt_value(v, True))
                else:
                    lbl.setText(f"{v:.5f}")
            # Show interpreted action if example provides action_display_fn
            action_fn = getattr(self._example, "action_display_fn", None) if self._example else None
            if action_fn is not None and self._action_row.isVisible():
                try:
                    self._action_lbl.setText(action_fn(list(outputs)))
                except Exception:
                    self._action_lbl.setText("—")
        except Exception as e:
            for lbl in self._output_labels:
                lbl.setText(f"Error: {e}")
            self._action_lbl.setText("—")
        self._update_memory_display()

    # ── Denormalize toggle ─────────────────────────────────────────────

    def _denorm_active(self) -> bool:
        """Denormalize is active only when the example exposes scales AND the box is checked."""
        if not self._example:
            return False
        if not (self._example.input_scale or self._example.output_scale):
            return False
        return self.chk_denormalize.isChecked()

    def _update_sensitivity(self, genome) -> None:
        """Run sensitivity analysis and dead-node detection; update chart."""
        tc = self._example.test_cases if self._example else None
        if not tc or genome is None:
            self._sensitivity_chart.clear()
            return
        try:
            scores = genome.sensitivity_analysis(tc, delta=0.1)
            dead = genome.dead_nodes(tc)
        except Exception:
            self._sensitivity_chart.clear()
            return

        n_inputs = self._example.n_inputs if self._example else len(scores)
        labels = [f"I{i}" for i in range(n_inputs)]

        dead_label = ""
        if dead:
            node_ids = ", ".join(f"#{nid}" for nid in sorted(dead))
            dead_label = f"Tote Hidden-Knoten: {node_ids}"

        self._sensitivity_chart.set_data(scores, labels, dead, dead_label)
        # Resize chart height to number of inputs
        self._sensitivity_chart.setMinimumHeight(max(60, len(scores) * 28 + 30))

    def _on_denormalize_toggled(self, _checked: bool) -> None:
        if not self._example:
            return
        self._rebuild_test_rows()
        # Re-run test rows against current genome so freshly created rows aren't empty.
        if self._genome is not None:
            self._genome.reset()
            for row in self._test_rows:
                row.update(self._genome)
        self._rebuild_input_widgets(self._example.n_inputs, self._example.n_outputs)
        self._rebuild_sequence_table()
        # Replay sequence so newly created rows show their last results.
        if self._genome is not None and self._seq_samples:
            self._seq_run_all()


# ---------------------------------------------------------------------------
# Server tab
# ---------------------------------------------------------------------------

