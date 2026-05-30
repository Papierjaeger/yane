# Tasks: YANE staerker machen

Roadmap und Implementierungshistorie.

## Status

**Teststand:** `2392 passed, 23 skipped` — P0, P1, P2 vollständig. Automodus-Bugs behoben.

---

## Offene Tasks: Automodus-Verbesserungen

### P0 — Direkte Verbesserungen (hoher Hebel)

**Profiler: noise_level direkt für Multi-Eval nutzen** ✓ *erledigt*

**Profiler: reward_sparsity für Feature-Aktivierung verwenden** ✓ *erledigt*

**FeatureGating: max_concurrent von 2 auf 3 erhöhen**
Mit 2 concurrent Features und 12 lightweight registrierten Features kann ein Feature im schlechtesten Fall erst nach ~6 × test_interval getestet werden. max_concurrent=3 verbessert die Abdeckung ohne wesentlichen Overhead.
- Datei: `evolution/auto_train.py` (Zeile ~7135: `set_auto_features(max_concurrent=2, ...)`)
- Datei: `gui/worker.py` (`AutoSetupWorker.run()`: `set_auto_features(max_concurrent=2, ...)`)

**FeatureGating: Task-adaptive test_interval / test_duration**
Aktuell: `test_interval = max(50, expected_gens // 6)` — gleich für alle Tasks. Besser: RL-Tasks mit langer Episodendauer brauchen längere Fenster (`// 4`), kurze Dataset-Tasks können schneller testen (`// 8`).
- Datei: `evolution/auto_train.py` (Feature-Gating-Konfiguration in `auto_train()`)

### P1 — Wichtige Erweiterungen

**Problem Profiler: Eval-Zeit messen und direkt für Worker-Auswahl nutzen**
Der Profiler evaluiert bereits n_warmup Genomes, misst aber die Eval-Zeit nicht strukturiert. Diese Zeit sollte direkt in `auto_train()` für die automatische Worker-Wahl (`set_n_workers()`) verwendet werden, statt immer auf den TrainingWorker-internen Automodus zu verlassen.
- Datei: `evolution/problem_profiler.py`, `evolution/auto_train.py`

**Knowledge Base: Parameter-Gewichtung bei Suggestion**
Aktuell werden alle Parameter eines KB-Eintrags gleichwertig übergeben. Ein schlechter `recovery.cooldown`-Wert aus einem anderen Kontext kann gute `lamarck.mode`-Vorschläge "verwässern". Lösung: nur Parameter mit hoher Konfidenz (Varianz über ähnliche Einträge niedrig) weitergeben.
- Datei: `evolution/knowledge_base.py` (`suggest()`-Methode)

**FeatureGating: degrade_fn für weitere Features implementieren**
Aktuell hat nur `island_model` eine `degrade_fn` (stufenweise Merge). Für Features wie `curiosity`, `stdp`, `neuromodulation` wäre eine sanfte Deaktivierung (z.B. Weight auf 0 reduzieren statt hard-disable) sinnvoll, um Trainingsschocks zu vermeiden.
- Datei: `evolution/feature_gating.py` (`_register_known_features()`)

**AutoSetupWorker und `auto_train()` vereinheitlichen** *(neu)*
`AutoSetupWorker.run()` und `auto_train()` implementieren dieselbe Setup-Logik doppelt. Lösung: `auto_train()` in eine `auto_configure()`-Phase und eine `train()`-Phase aufteilen, sodass `AutoSetupWorker` nur `auto_configure()` aufruft. Verhindert künftige Drift zwischen GUI- und API-Pfad.
- Datei: `neuro_evolution.py`, `gui/worker.py`

**`task_type_confidence` für konservativere Defaults nutzen** *(neu)*
Der Profiler berechnet `task_type_confidence` (Zuverlässigkeit der Task-Klassifikation), aber kein Code verwendet diesen Wert. Wenn `task_type_confidence < 0.6`: weniger task-spezifische Parameter setzen, `target_species` auf Mittelwert (7–8) statt task-spezifisch, FeatureGating konservativer starten.
- Datei: `evolution/auto_train.py`, `neuro_evolution.py` (`auto_train()`)

**MetaOptimizer-GP mit KB-Werten vorseeden** *(neu)*
Wenn eine KB-Suggestion angewendet wird, startet MetaOptimizerS GP-Optimizer danach blind. Die KB-Parameterwerte sind gute initiale GP-Beobachtungspunkte (`gp.observe(value, 0.1)` als Warm-Prior). Würde die Explorations-Phase des MetaOptimizers deutlich verkürzen.
- Datei: `evolution/meta_optimizer.py`, `neuro_evolution.py` (Übergabe der KB-Werte an `set_meta_optimizer()`)

**FeatureGating: Feature-Konflikte registrieren** *(neu)*
Manche Features schließen sich aus: STDP ist inkompatibel mit matrix_forward (STDP braucht sequentielle Forward-Passes). island_model + adaptive_recovery können sich gegenseitig behindern. Wenn zwei konfliktierende Features gleichzeitig getestet werden, sind die Impact-Messungen beider verzerrt. Lösung: `conflicts_with: frozenset[str]` im `FeatureRecord`, FeatureGating skippt konflikthafte Kombinationen.
- Datei: `evolution/feature_gating.py`

**Checkpoint-Erkennung für Warm-Restart** *(neu)*
Wenn für dasselbe Example ein Checkpoint im Log-Verzeichnis existiert, könnte `auto_train()` / `AutoSetupWorker` ihn laden und weitertrainieren statt neu zu starten. Großer praktischer Vorteil für iterative Experimente.
- Datei: `neuro_evolution.py` (`auto_train()`), `gui/worker.py` (`AutoSetupWorker`)

**KB-Eintrags-Alterung** *(neu)*
Ältere KB-Einträge verwenden möglicherweise veraltete Hyperparameter-Semantiken. Einträge mit YANE-Version taggen und bei k-NN mit einem Zeit-Abschlag gewichten.
- Datei: `evolution/knowledge_base.py` (`KBEntry`, `suggest()`)

### P2 — Forschung / Experimentell

**Problem Profiler: Fitness-Landscape-Shape schätzen**
Aus den warmup-Evaluierungen könnte man grob abschätzen, ob die Fitnesslandschaft unimodal (einfach) oder multimodal (schwer) ist, indem man die Varianz zwischen Random-Genomes analysiert. Multimodale Probleme profitieren mehr von `diversity_injection` und `quality_diversity`.

**Problem Profiler: Echte temporale Autokorrelation messen**
`temporal_dependency` ist derzeit `min(1.0, noise_level * 2.0)` — ein reiner Proxy. Eine echte Sequenz-Analyse (Autokorrelation über mehrere Steps) würde zuverlässiger RL von stochastischen supervised-Tasks unterscheiden.

**Knowledge Base: Cross-Task Transfer via latente Embeddings**
Aktuell: k-NN direkt auf 10-D Feature-Vektor. Besser: Ein kleines gelerntes Embedding (z.B. PCA oder Autoencoder auf akkumulierten KB-Einträgen), das strukturelle Ähnlichkeiten zwischen verschiedenen Task-Typen lernt.

**Live-Status von FeatureGating im GUI** *(neu)*
Der Auto-Train-Bericht erscheint nur am Ende. Eine Statuszeile wie `"FeatureGating: testet curiosity (Gen 45/65)"` während des Trainings würde Nutzern erklären, warum das Training manchmal kurzzeitig langsamer wird. Die nötigen Daten liegen bereits in `population_memory_info()` vor.
- Datei: `gui/tabs/training_tab.py` (`_on_iteration()`), `neuro_evolution.py` (FeatureGating-Diagnostics in heartbeat)

**Fitness-Target automatisch schätzen** *(neu)*
Wenn kein `target_fitness` gesetzt ist, könnte `auto_train()` aus den Profiling-Daten einen Richtwert ableiten (z.B. `max_observed_fitness * 3.0` oder difficulty-basiert). Gibt sauberere Stopp-Bedingungen ohne Nutzer-Input.
- Datei: `evolution/auto_train.py`

---

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
