"""Features Tab — Layer-3 Research Feature Configuration.

Provides toggles and parameter widgets for all toggle-able research features.
Call ``apply_to_ne(yane)`` after ``yane.configure()`` to apply settings.

Deferred (require complex input or are full training loops):
  POET, meta_train, adversarial/cooperative, continual learning,
  Conv-NEAT (needs image input setup), HyperNEAT, Hybrid (needs PyTorch),
  Safety Constraints (require Python callables).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QGroupBox,
    QCheckBox, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QSizePolicy, QPushButton, QFrame,
)
from PySide6.QtCore import Qt

if TYPE_CHECKING:
    from yane.neuro_evolution import NeuroEvolution


# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------

def _row(label: str, widget: QWidget, tip: str = "") -> QHBoxLayout:
    """Return a label + widget HBox layout row."""
    lbl = QLabel(label)
    lbl.setFixedWidth(170)
    lbl.setWordWrap(True)
    if tip:
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
    row = QHBoxLayout()
    row.setContentsMargins(0, 2, 0, 2)
    row.addWidget(lbl)
    row.addWidget(widget, 1)
    return row


def _chk(label: str, default: bool = False, tip: str = "") -> QCheckBox:
    c = QCheckBox(label)
    c.setChecked(default)
    if tip:
        c.setToolTip(tip)
    return c


def _spin(lo: int, hi: int, default: int, tip: str = "") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(default)
    if tip:
        s.setToolTip(tip)
    return s


def _dspin(lo: float, hi: float, default: float, decimals: int = 3, tip: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(default)
    s.setDecimals(decimals)
    if tip:
        s.setToolTip(tip)
    return s


def _combo(options: list[str], default: str = "") -> QComboBox:
    c = QComboBox()
    for opt in options:
        c.addItem(opt)
    if default and default in options:
        c.setCurrentText(default)
    return c


def _separator() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: #313244;")
    return f


def _group(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    box = QGroupBox(title)
    layout = QVBoxLayout()
    layout.setSpacing(4)
    layout.setContentsMargins(8, 12, 8, 8)
    box.setLayout(layout)
    return box, layout


# ---------------------------------------------------------------------------
# FeaturesTab
# ---------------------------------------------------------------------------

class FeaturesTab(QWidget):
    """Tab for all Layer-3 research feature toggles."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(self._build_neural_arch())
        layout.addWidget(self._build_learning())
        layout.addWidget(self._build_evaluation())
        layout.addWidget(self._build_topology())
        layout.addWidget(self._build_deployment())
        layout.addWidget(self._build_analysis())
        layout.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------ groups

    def _build_neural_arch(self) -> QGroupBox:
        box, lay = _group("Neurale Architektur")

        # Attention
        self.chk_attention = _chk("Attention Heads", tip="Evolvable multi-head attention preprocessing.")
        lay.addWidget(self.chk_attention)
        self.spin_attention_head_dim = _spin(2, 32, 4, "Dimensionality of each attention head.")
        lay.addLayout(_row("Head-Dim", self.spin_attention_head_dim))
        self.spin_attention_num_heads = _spin(1, 8, 2, "Number of parallel attention heads.")
        lay.addLayout(_row("Num Heads", self.spin_attention_num_heads))
        lay.addWidget(_separator())

        # LTC
        self.chk_ltc = _chk("LTC Nodes (ODE)", tip="Liquid Time-Constant ODE neuron dynamics.")
        lay.addWidget(self.chk_ltc)
        lay.addWidget(_separator())

        # Neuromodulation
        self.chk_neuromodulation = _chk("Neuromodulation", tip="Dynamic modulation of connection weights via modulator nodes.")
        lay.addWidget(self.chk_neuromodulation)
        lay.addWidget(_separator())

        # STDP
        self.chk_stdp = _chk("STDP", tip="Spike-Timing-Dependent Plasticity (Hebbian learning).")
        lay.addWidget(self.chk_stdp)
        self.dspin_stdp_wmin = _dspin(-5.0, 0.0, -1.0, 2, "Min weight after STDP update.")
        lay.addLayout(_row("Weight Min", self.dspin_stdp_wmin))
        self.dspin_stdp_wmax = _dspin(0.0, 5.0, 1.0, 2, "Max weight after STDP update.")
        lay.addLayout(_row("Weight Max", self.dspin_stdp_wmax))
        lay.addWidget(_separator())

        # Probabilistic
        self.chk_probabilistic = _chk("Probabilistic (Bayesian)", tip="Gaussian output noise for uncertainty estimation.")
        lay.addWidget(self.chk_probabilistic)
        self.dspin_noise_std = _dspin(0.001, 2.0, 0.05, 3, "Standard deviation of per-output Gaussian noise.")
        lay.addLayout(_row("Noise σ", self.dspin_noise_std))
        self.chk_inference_mode = _chk("Inference mode (deterministisch)", tip="When checked, forward() is deterministic (no noise).")
        lay.addWidget(self.chk_inference_mode)

        return box

    def _build_learning(self) -> QGroupBox:
        box, lay = _group("Lernmechanismen")

        # Curiosity
        self.chk_curiosity = _chk("Curiosity (Intrinsic Reward)", tip="Prediction-error bonus encourages exploration.")
        lay.addWidget(self.chk_curiosity)
        self.dspin_curiosity_weight = _dspin(0.0, 5.0, 0.3, 2, "Weight of curiosity bonus added to fitness.")
        lay.addLayout(_row("Gewicht", self.dspin_curiosity_weight))
        lay.addWidget(_separator())

        # DARTS
        self.chk_darts = _chk("DARTS-Lite", tip="Differentiable architecture search via sigmoid gates.")
        lay.addWidget(self.chk_darts)
        self.dspin_darts_threshold = _dspin(0.01, 0.99, 0.1, 2, "Gate threshold for post-training pruning.")
        lay.addLayout(_row("Prune-Schwelle", self.dspin_darts_threshold))
        lay.addWidget(_separator())

        # Shared Weights
        self.chk_shared_weights = _chk("Shared Weights", tip="Weight sharing between connection groups.")
        lay.addWidget(self.chk_shared_weights)
        lay.addWidget(_separator())

        # Augmentation
        self.chk_augmentation = _chk("Input-Augmentierung", tip="Co-evolve augmentation pipelines alongside genomes.")
        lay.addWidget(self.chk_augmentation)
        self.spin_aug_pop = _spin(2, 50, 8, "Number of augmentation pipelines in the pool.")
        lay.addLayout(_row("Pool-Größe", self.spin_aug_pop))
        self.spin_aug_interval = _spin(1, 200, 20, "Generations between augmentation evolution steps.")
        lay.addLayout(_row("Evolutions-Interval", self.spin_aug_interval))

        return box

    def _build_evaluation(self) -> QGroupBox:
        box, lay = _group("Evaluierung & Pruning")

        # Anytime Eval
        self.chk_anytime = _chk("Anytime-Evaluation", tip="Progressive eval budget: cheap first-pass, expensive top-k.")
        lay.addWidget(self.chk_anytime)
        self.spin_anytime_min = _spin(1, 50, 1, "Minimum evaluations per genome.")
        lay.addLayout(_row("Min Evals", self.spin_anytime_min))
        self.spin_anytime_max = _spin(1, 200, 5, "Maximum evaluations per genome (top-k only).")
        lay.addLayout(_row("Max Evals", self.spin_anytime_max))
        self.dspin_anytime_promo = _dspin(0.05, 1.0, 0.3, 2, "Fraction of population promoted to full eval.")
        lay.addLayout(_row("Promotion", self.dspin_anytime_promo))
        self.combo_anytime_agg = _combo(["mean", "median", "min", "max"], "mean")
        lay.addLayout(_row("Aggregation", self.combo_anytime_agg))
        lay.addWidget(_separator())

        # Adaptive Recovery
        self.chk_recovery = _chk("Adaptive Recovery", tip="Automatic stagnation recovery (diversity injection, Lamarck burst, partial restart).")
        lay.addWidget(self.chk_recovery)
        lay.addWidget(_separator())

        # Post-Training Pruning
        self.chk_pruning = _chk("Post-Training Pruning", tip="Automatically prune weak connections after training.")
        lay.addWidget(self.chk_pruning)
        self.dspin_prune_thresh = _dspin(0.0, 0.5, 0.01, 3, "Weight magnitude below which connections are pruned.")
        lay.addLayout(_row("Schwelle", self.dspin_prune_thresh))
        self.dspin_prune_drop = _dspin(0.0, 0.5, 0.02, 3, "Max fitness drop before pruning is rolled back.")
        lay.addLayout(_row("Max Fitness-Verlust", self.dspin_prune_drop))

        return box

    def _build_topology(self) -> QGroupBox:
        box, lay = _group("Topologie-Erweiterungen")

        # Input Grouping
        self.chk_input_grouping = _chk("Input-Gruppierung", tip="Evolvable input aggregation groups (INIT-only — restart required).")
        lay.addWidget(self.chk_input_grouping)
        self.spin_ig_groups = _spin(1, 32, 2, "Number of initial input groups.")
        lay.addLayout(_row("Gruppen", self.spin_ig_groups))
        lay.addWidget(_separator())

        # Output Grouping
        self.chk_output_grouping = _chk("Output-Gruppierung", tip="Evolvable output expansion groups (INIT-only — restart required).")
        lay.addWidget(self.chk_output_grouping)

        return box

    def _build_deployment(self) -> QGroupBox:
        box, lay = _group("Deployment & Ressourcen")

        # Hardware constraints
        self.chk_hardware = _chk("Hardware-Constraints", tip="Penalise genomes that exceed target platform limits.")
        lay.addWidget(self.chk_hardware)
        _platforms = ["cortex-m4", "cortex-m7", "esp32",
                      "raspberry-pi-zero", "raspberry-pi-4", "desktop", "mobile-arm"]
        self.combo_hw_platform = _combo(_platforms, "cortex-m4")
        lay.addLayout(_row("Zielplattform", self.combo_hw_platform))
        self.dspin_hw_penalty = _dspin(0.1, 100.0, 1.0, 1, "Penalty scale for hardware constraint violations.")
        lay.addLayout(_row("Penalty-Skala", self.dspin_hw_penalty))
        lay.addWidget(_separator())

        # Budget
        self.chk_budget = _chk("Ressourcen-Budget", tip="Limit training by wall-clock time and memory.")
        lay.addWidget(self.chk_budget)
        self.combo_budget_time = _combo(["–", "30s", "1min", "5min", "10min", "30min", "1h", "2h"], "–")
        lay.addLayout(_row("Max Zeit", self.combo_budget_time))
        self.combo_budget_mem = _combo(["–", "80%", "1GB", "2GB", "4GB", "8GB", "16GB", "auto"], "–")
        lay.addLayout(_row("Max RAM", self.combo_budget_mem))
        lay.addWidget(_separator())

        # Tracking
        self.chk_wandb = _chk("WandB Tracking", tip="Log metrics to Weights & Biases.")
        lay.addWidget(self.chk_wandb)
        self.chk_mlflow = _chk("MLflow Tracking", tip="Log metrics to MLflow.")
        lay.addWidget(self.chk_mlflow)

        return box

    def _build_analysis(self) -> QGroupBox:
        box, lay = _group("Analyse & Experiment")

        # Phylogeny
        self.chk_phylogeny = _chk("Phylogenie-Tracking", tip="Record genome ancestry tree (Stammbaum der Innovationen).")
        lay.addWidget(self.chk_phylogeny)
        self.spin_phylogeny_max = _spin(100, 50000, 2000, "Maximum nodes to keep in the phylogeny tree.")
        lay.addLayout(_row("Max Nodes", self.spin_phylogeny_max))
        lay.addWidget(_separator())

        # Behaviour Cloning info label
        note = QLabel(
            "<b>Behaviour Cloning</b><br>"
            "Verfügbar via <code>NeuroEvolution.behaviour_clone(demos)</code><br>"
            "oder <code>POST /export/clone</code> (API)."
        )
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setStyleSheet("color: #a6adc8; font-size: 12px;")
        lay.addWidget(note)
        lay.addWidget(_separator())

        # Deferred features info
        deferred = QLabel(
            "<b>Weitere Features (nur via Code / API):</b><br>"
            "POET, Continual Learning, Meta-Learning, Reservoir Computing,<br>"
            "Adversarial/Cooperative Populations, Conv-NEAT, HyperNEAT,<br>"
            "Safety Constraints, Hybrid Backprop, GRN Encoding."
        )
        deferred.setWordWrap(True)
        deferred.setTextFormat(Qt.TextFormat.RichText)
        deferred.setStyleSheet("color: #585b70; font-size: 12px;")
        lay.addWidget(deferred)

        return box

    # ------------------------------------------------------------------ apply

    def apply_to_ne(self, yane: "NeuroEvolution") -> None:
        """Apply all feature settings to *yane* after it has been configured.

        Called from TrainingTab._apply_evolution_options() so that settings
        take effect at the start of each training run.  Init-stage features
        (input/output grouping) are re-applied here; the population is already
        built so they only affect the next configure() call — we still apply
        them to configure per-genome state.
        """
        self._apply_neural_arch(yane)
        self._apply_learning(yane)
        self._apply_evaluation(yane)
        self._apply_topology(yane)
        self._apply_deployment(yane)
        self._apply_analysis(yane)

    def _apply_neural_arch(self, yane: "NeuroEvolution") -> None:
        yane.set_attention(
            enabled=self.chk_attention.isChecked(),
            head_dim=self.spin_attention_head_dim.value(),
            num_heads=self.spin_attention_num_heads.value(),
        )
        yane.set_ltc(enabled=self.chk_ltc.isChecked())
        yane.set_neuromodulation(enabled=self.chk_neuromodulation.isChecked())
        if self.chk_stdp.isChecked():
            yane.set_stdp(
                enabled=True,
                weight_min=self.dspin_stdp_wmin.value(),
                weight_max=self.dspin_stdp_wmax.value(),
            )
        else:
            yane.set_stdp(enabled=False)
        yane.set_probabilistic(
            enabled=self.chk_probabilistic.isChecked(),
            noise_std=self.dspin_noise_std.value(),
            inference_mode=self.chk_inference_mode.isChecked(),
        )

    def _apply_learning(self, yane: "NeuroEvolution") -> None:
        yane.set_curiosity(
            enabled=self.chk_curiosity.isChecked(),
            weight=self.dspin_curiosity_weight.value(),
        )
        yane.set_darts_mode(
            enabled=self.chk_darts.isChecked(),
            prune_threshold=self.dspin_darts_threshold.value(),
        )
        yane.set_shared_weights(enabled=self.chk_shared_weights.isChecked())
        yane.set_evolutionary_augmentation(
            enabled=self.chk_augmentation.isChecked(),
            population_augmentations=self.spin_aug_pop.value(),
            evolution_interval=self.spin_aug_interval.value(),
        )

    def _apply_evaluation(self, yane: "NeuroEvolution") -> None:
        if self.chk_anytime.isChecked():
            yane.set_anytime_eval(
                enabled=True,
                min_evals=self.spin_anytime_min.value(),
                max_evals=self.spin_anytime_max.value(),
                promotion_frac=self.dspin_anytime_promo.value(),
                aggregation=self.combo_anytime_agg.currentText(),
            )
        else:
            yane.set_anytime_eval(enabled=False)
        if self.chk_recovery.isChecked():
            yane.set_adaptive_recovery(enabled=True)
        if self.chk_pruning.isChecked():
            yane.set_post_training_pruning(
                enabled=True,
                threshold=self.dspin_prune_thresh.value(),
                max_drop_frac=self.dspin_prune_drop.value(),
            )
        else:
            yane.set_post_training_pruning(enabled=False)

    def _apply_topology(self, yane: "NeuroEvolution") -> None:
        # Note: grouping features are init-stage — they shape the population
        # template. Re-applying here updates per-genome flags for future runs.
        if self.chk_input_grouping.isChecked():
            try:
                yane.set_input_grouping(
                    enabled=True,
                    n_groups=self.spin_ig_groups.value(),
                )
            except Exception:
                pass  # May fail if topology is incompatible; non-critical
        else:
            try:
                yane.set_input_grouping(enabled=False)
            except Exception:
                pass
        if self.chk_output_grouping.isChecked():
            try:
                yane.set_output_grouping(enabled=True)
            except Exception:
                pass
        else:
            try:
                yane.set_output_grouping(enabled=False)
            except Exception:
                pass

    def _apply_deployment(self, yane: "NeuroEvolution") -> None:
        if self.chk_hardware.isChecked():
            yane.set_hardware_constraints(
                target_platform=self.combo_hw_platform.currentText(),
                penalty_scale=self.dspin_hw_penalty.value(),
            )
        else:
            yane._hw_constraints = None

        if self.chk_budget.isChecked():
            time_str = self.combo_budget_time.currentText()
            mem_str = self.combo_budget_mem.currentText()
            try:
                yane.set_budget(
                    total_time=None if time_str == "–" else time_str,
                    max_memory=None if mem_str == "–" else mem_str,
                )
            except Exception:
                pass

        backends = []
        if self.chk_wandb.isChecked():
            try:
                from yane.evolution.tracking import WandbBackend
                backends.append(WandbBackend())
            except Exception:
                pass
        if self.chk_mlflow.isChecked():
            try:
                from yane.evolution.tracking import MlflowBackend
                backends.append(MlflowBackend())
            except Exception:
                pass
        if backends:
            yane.set_tracking_backend(*backends)

    def _apply_analysis(self, yane: "NeuroEvolution") -> None:
        if self.chk_phylogeny.isChecked():
            yane.enable_phylogeny(max_size=self.spin_phylogeny_max.value())
        else:
            yane.disable_phylogeny()

    # ------------------------------------------------------------------ state

    def collect_state(self) -> dict:
        """Return the current widget values as a serialisable dict (for presets)."""
        return {
            "attention": self.chk_attention.isChecked(),
            "attention_head_dim": self.spin_attention_head_dim.value(),
            "attention_num_heads": self.spin_attention_num_heads.value(),
            "ltc": self.chk_ltc.isChecked(),
            "neuromodulation": self.chk_neuromodulation.isChecked(),
            "stdp": self.chk_stdp.isChecked(),
            "stdp_wmin": self.dspin_stdp_wmin.value(),
            "stdp_wmax": self.dspin_stdp_wmax.value(),
            "probabilistic": self.chk_probabilistic.isChecked(),
            "probabilistic_noise_std": self.dspin_noise_std.value(),
            "probabilistic_inference_mode": self.chk_inference_mode.isChecked(),
            "curiosity": self.chk_curiosity.isChecked(),
            "curiosity_weight": self.dspin_curiosity_weight.value(),
            "darts": self.chk_darts.isChecked(),
            "darts_threshold": self.dspin_darts_threshold.value(),
            "shared_weights": self.chk_shared_weights.isChecked(),
            "augmentation": self.chk_augmentation.isChecked(),
            "augmentation_pool": self.spin_aug_pop.value(),
            "augmentation_interval": self.spin_aug_interval.value(),
            "anytime": self.chk_anytime.isChecked(),
            "anytime_min": self.spin_anytime_min.value(),
            "anytime_max": self.spin_anytime_max.value(),
            "anytime_promo": self.dspin_anytime_promo.value(),
            "anytime_agg": self.combo_anytime_agg.currentText(),
            "recovery": self.chk_recovery.isChecked(),
            "pruning": self.chk_pruning.isChecked(),
            "pruning_threshold": self.dspin_prune_thresh.value(),
            "pruning_max_drop": self.dspin_prune_drop.value(),
            "input_grouping": self.chk_input_grouping.isChecked(),
            "input_grouping_n": self.spin_ig_groups.value(),
            "output_grouping": self.chk_output_grouping.isChecked(),
            "hardware": self.chk_hardware.isChecked(),
            "hardware_platform": self.combo_hw_platform.currentText(),
            "hardware_penalty": self.dspin_hw_penalty.value(),
            "budget": self.chk_budget.isChecked(),
            "budget_time": self.combo_budget_time.currentText(),
            "budget_mem": self.combo_budget_mem.currentText(),
            "wandb": self.chk_wandb.isChecked(),
            "mlflow": self.chk_mlflow.isChecked(),
            "phylogeny": self.chk_phylogeny.isChecked(),
            "phylogeny_max": self.spin_phylogeny_max.value(),
        }

    def restore_state(self, d: dict) -> None:
        """Restore widget values from a state dict (from presets)."""
        def _b(key: bool) -> bool:
            return bool(d.get(key, False))
        def _i(key, default: int) -> int:
            return int(d.get(key, default))
        def _f(key, default: float) -> float:
            return float(d.get(key, default))

        self.chk_attention.setChecked(_b("attention"))
        self.spin_attention_head_dim.setValue(_i("attention_head_dim", 4))
        self.spin_attention_num_heads.setValue(_i("attention_num_heads", 2))
        self.chk_ltc.setChecked(_b("ltc"))
        self.chk_neuromodulation.setChecked(_b("neuromodulation"))
        self.chk_stdp.setChecked(_b("stdp"))
        self.dspin_stdp_wmin.setValue(_f("stdp_wmin", -1.0))
        self.dspin_stdp_wmax.setValue(_f("stdp_wmax", 1.0))
        self.chk_probabilistic.setChecked(_b("probabilistic"))
        self.dspin_noise_std.setValue(_f("probabilistic_noise_std", 0.05))
        self.chk_inference_mode.setChecked(_b("probabilistic_inference_mode"))
        self.chk_curiosity.setChecked(_b("curiosity"))
        self.dspin_curiosity_weight.setValue(_f("curiosity_weight", 0.3))
        self.chk_darts.setChecked(_b("darts"))
        self.dspin_darts_threshold.setValue(_f("darts_threshold", 0.1))
        self.chk_shared_weights.setChecked(_b("shared_weights"))
        self.chk_augmentation.setChecked(_b("augmentation"))
        self.spin_aug_pop.setValue(_i("augmentation_pool", 8))
        self.spin_aug_interval.setValue(_i("augmentation_interval", 20))
        self.chk_anytime.setChecked(_b("anytime"))
        self.spin_anytime_min.setValue(_i("anytime_min", 1))
        self.spin_anytime_max.setValue(_i("anytime_max", 5))
        self.dspin_anytime_promo.setValue(_f("anytime_promo", 0.3))
        agg = d.get("anytime_agg", "mean")
        idx = self.combo_anytime_agg.findText(str(agg))
        if idx >= 0:
            self.combo_anytime_agg.setCurrentIndex(idx)
        self.chk_recovery.setChecked(_b("recovery"))
        self.chk_pruning.setChecked(_b("pruning"))
        self.dspin_prune_thresh.setValue(_f("pruning_threshold", 0.01))
        self.dspin_prune_drop.setValue(_f("pruning_max_drop", 0.02))
        self.chk_input_grouping.setChecked(_b("input_grouping"))
        self.spin_ig_groups.setValue(_i("input_grouping_n", 2))
        self.chk_output_grouping.setChecked(_b("output_grouping"))
        self.chk_hardware.setChecked(_b("hardware"))
        plat = d.get("hardware_platform", "cortex-m4")
        pidx = self.combo_hw_platform.findText(str(plat))
        if pidx >= 0:
            self.combo_hw_platform.setCurrentIndex(pidx)
        self.dspin_hw_penalty.setValue(_f("hardware_penalty", 1.0))
        self.chk_budget.setChecked(_b("budget"))
        bt = d.get("budget_time", "–")
        btidx = self.combo_budget_time.findText(str(bt))
        if btidx >= 0:
            self.combo_budget_time.setCurrentIndex(btidx)
        bm = d.get("budget_mem", "–")
        bmidx = self.combo_budget_mem.findText(str(bm))
        if bmidx >= 0:
            self.combo_budget_mem.setCurrentIndex(bmidx)
        self.chk_wandb.setChecked(_b("wandb"))
        self.chk_mlflow.setChecked(_b("mlflow"))
        self.chk_phylogeny.setChecked(_b("phylogeny"))
        self.spin_phylogeny_max.setValue(_i("phylogeny_max", 2000))
