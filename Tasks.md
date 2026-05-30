# Tasks: YANE staerker machen

Roadmap und Implementierungshistorie. Alle Tasks abgeschlossen.

## Status

**Teststand:** `2345 passed, 23 skipped` — P0, P1, P2 vollständig.

## Legende

- `P0`: hoher Hebel, direkt nuetzlich
- `P1`: wichtiger Ausbau
- `P2`: experimentell / Forschung
- ✓: erledigt

---

## Architektur-Leitlinie

YANE ist in drei konzeptionelle Schichten gegliedert. Neue Features sollen klar einer Schicht zugeordnet werden.

**Layer 1 — Stable Core**
Population, Genome, Speciation, Mutation, Evaluation, Checkpoints.
Kleine, testbare, rueckwaertskompatible Basis. Keine experimentellen Abhaengigkeiten.

**Layer 2 — Adaptive Systems**
Recovery, Auto-Tuning, Scheduling, Surrogates, Diversity-Systeme, Policy-Orchestrierung, Meta-Adaptive Orchestration (MetaOptimizer, ParamRegistry, Knowledge Base, Feature Gating), ResourceBudget-System.
Nutzt ausschliesslich oeffentliche Layer-1-APIs.

**Layer 3 — Research Features**
Modular, per API an-/abschaltbar, standardmaessig deaktiviert, Zero-Cost bei Deaktivierung.

---

## Implementierte Features (Kompakt)

### P0 — Core & Adaptive Systems

| Feature | Modul / API |
|---|---|
| Core Evolution, Speciation, Mutation | `evolution/population.py`, `core/` |
| Worker-Pipeline, GUI, API, Logging | `gui/`, `api/`, `util/logger.py` |
| Multi-Objective, Quality Diversity | `evolution/multi_objective.py` |
| CMA-ES, NES, SA, Lamarck | `evolution/lamarck_refiner.py` |
| Adaptive Recovery System | `set_adaptive_recovery()` |
| Anytime-Evaluation | `set_anytime_eval()` |
| Self-Tuning Speciation (PI-Regler) | `set_target_species()` |
| AdaptiveController, OperatorScheduler | `evolution/adaptive_control.py` |
| Fitness Surrogate | `evolution/surrogate.py` |
| Checkpoint Rolling + Codec | `set_checkpoint_policy()`, `set_checkpoint_codec()` |
| Remote/Distributed Evaluation | `evolution/remote_eval.py` |
| **Meta-Adaptive Orchestration (6 Phasen)** | `evolution/meta_optimizer.py`, `evolution/param_registry.py`, `evolution/problem_profiler.py`, `evolution/knowledge_base.py`, `evolution/feature_gating.py`, `evolution/auto_train.py` |
| **`auto_train(evaluator)`** — Zero-Config | `NeuroEvolution.auto_train()` |
| ResourceBudget-System | `evolution/resource_budget.py` |

### P1 — Wichtige Erweiterungen

| Feature | Modul / API |
|---|---|
| Regression Benchmarking Suite | `benchmarks/`, `python -m yane.benchmarks --ci` |
| WandB / MLflow Integration | `evolution/tracking.py`, `set_tracking_backend()` |
| Interactive Evolution (Human-in-the-Loop) | `evolution/interactive_eval.py` |
| Hardware-Aware NEAT | `evolution/hardware_aware.py` |
| Evolutionary Data Augmentation | `evolution/augmentation.py` |
| Selection Strategy Plugin | `evolution/selection_strategy.py` |
| Evaluation Middleware Stack | `evolution/eval_middleware.py` |
| Adaptive Policy System | `evolution/policy_system.py` |
| Modular Compatibility Distance | `evolution/compatibility.py` |
| Experiment Tracking / Run Database | `util/run_database.py` |
| Fitness Landscape Visualisierung | `population_pca()`, GUI Scatterplot |
| Generations-Report / Postmortem | `util/report.py`, `export_run_report()` |
| Hyperparameter-Suche | `evolution/hyperparameter_search.py` |
| Island-Modell | `evolution/islands.py` |
| Ensemble Bewertung + Deployment | `make_ensemble()`, `EnsembleGenome` |
| Online-Hyperparameter-Adaptation | `evolution/online_tuning.py` |
| Weight-Inheritance beim Crossover | `set_weight_inheritance()` |
| GUI-Integration P0 (Auto-Train Panel) | `gui/`, `AutoSetupWorker` |

### P2 — Research Features

| Feature | Modul / API |
|---|---|
| Transfer Learning / Fine-Tuning | `warm_start_from_checkpoint()`, `fine_tune_genome()` |
| STDP / Hebbsches Lernen | `evolution/stdp.py`, `set_stdp()` |
| Neuromodulation | `evolution/neuromodulation.py`, `set_neuromodulation()` |
| Input-Gruppierung | `evolution/input_grouping.py`, `set_input_grouping()` |
| Output-Gruppierung | `evolution/output_grouping.py`, `set_output_grouping()` |
| Convolutional NEAT | `evolution/conv_neat.py`, `set_conv_neat()` |
| ES-HyperNEAT | `evolution/indirect_encoding.py`, `es_hyperneat_substrate()` |
| ONNX-Export | `evolution/onnx_export.py`, `genome_to_onnx()` |
| WebAssembly / JS-Export | `evolution/wasm_export.py`, `genome_to_js()` |
| C-Array / TFLite-Export | `evolution/tflite_export.py`, `genome_to_c_array()` |
| Population Distillation | `evolution/distillation.py`, `distill_ensemble()` |
| Gradient-NEAT-Hybrid | `evolution/hybrid_neat.py`, `set_hybrid_mode()` |
| Evolvable Attention Heads | `evolution/attention.py`, `set_attention()` |
| LTC Nodes (ODE-Neuronen) | `evolution/ltc.py`, `set_ltc()` |
| Temporal Speciation | `evolution/compatibility.py`, `TemporalDistance` |
| Self-Play / Adversarial Populations | `evolution/self_play.py`, `set_adversarial_populations()` |
| Hierarchical NEAT (H-NEAT) | `evolution/h_neat.py`, `HierarchicalGenome` |
| GRN Encoding | `evolution/grn_encoding.py`, `GRNGenome` |
| Developmental NEAT | `evolution/developmental.py`, `ParametricRule` |
| Continual / Lifelong Learning | `evolution/continual.py`, `set_continual_learning()` |
| Meta-Learning (Few-Shot) | `evolution/meta_learning.py`, `meta_train()` |
| Reservoir Computing | `evolution/reservoir.py`, `ReservoirGenome` |
| Open-Ended / Minimal Criterion | `evolution/minimal_criterion.py`, `set_minimal_criterion()` |
| Multi-Agent Cooperation | `evolution/cooperative.py`, `train_cooperative()` |
| Probabilistic / Bayesian NEAT | `evolution/bayesian_neat.py`, `bayesian_forward()` |
| Safe NEAT | `evolution/safety.py`, `set_safety_constraints()` |
| Sparse NEAT / Lottery Ticket | `evolution/sparse_neat.py`, `find_lottery_ticket()` |
| Symbolic Regression Export | `core/_symbolic.py`, `genome.to_symbolic()` |
| DARTS-Lite | `set_darts_mode()` |
| Intrinsic Curiosity | `set_curiosity()` |
| Shared Weights | `evolution/shared_weights.py`, `set_shared_weights()` |
| Behaviour Cloning Warm-Start | `evolution/behaviour_cloning.py`, `behaviour_clone()` |
| Genome-Phylogenie (Stammbaum) | `evolution/phylogeny.py`, `enable_phylogeny()` |
| POET / Co-Evolution | `evolution/poet.py`, `train_poet()` |
