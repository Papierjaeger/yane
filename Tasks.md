# Tasks: YANE staerker machen

Diese Datei ist die aktuelle Roadmap fuer YANE. Offene und neue Tasks stehen
oben. Abgeschlossene Arbeit ist am Ende kompakt zusammengefasst.

## Status

**Aktueller Stand:** P0 komplett. P1 vollständig. P2: Transfer Learning (✓), Input-Gruppierung (✓), Output-Gruppierung (✓), Convolutional NEAT (✓), ES-HyperNEAT (✓), ONNX-Export (✓), Population Distillation (✓), Gradient-NEAT-Hybrid (✓), STDP (✓), Neuromodulation (✓), WebAssembly-Export (✓), Attention Heads (✓), LTC Nodes (✓), Temporal Speciation (✓), Self-Play (✓), H-NEAT (✓), GRN-Encoding (✓), Developmental NEAT (✓), Continual Learning (✓), Meta-Learning (✓), Reservoir Computing (✓), Open-Ended/Minimal Criterion (✓), Multi-Agent Cooperation (✓) fertig.
Teststand: `2157 passed, 23 skipped` (nach Multi-Agent Cooperation).

> **Roadmap:** 6 P1-Tasks (Benchmarking-Suite, WandB/MLflow, Interactive
> Evolution, Hardware-Aware, ResourceBudget-System, Data Augmentation) und
> 21 P2-Tasks offen. Mehrere P2-Forschungsfeatures besitzen aktuell nur
> isolierte Experimental-/Spike-Bausteine (`⚡` oder `□`) und müssen noch
> vollständig in Genome, Population, NeuroEvolution, Checkpoints und Tests
> integriert werden.

- Core-Evolution, Speciation, Mutation, Worker-Pipeline, GUI, API, Logging, Checkpoints: implementiert.
- Multi-Objective, Quality Diversity, CMA-ES, Backprop-/Matrix-Bausteine, Presets, Benchmark-Gates: implementiert.
- AdaptiveController, OperatorScheduler, Lamarck-Budget, Interspecies-Trigger (Novelty/Isolation/Schutz): implementiert.
- Adaptive Benchmark-Suite, GUI-Stability-Guard, Preset-Schema v2 mit `adaptive_policies`: implementiert.
- Matrix-Forward-Integration, Checkpoint-State fuer Adaptive-Objekte: implementiert.
- Checkpoint gehaertet: Fixture-Dateien, JSON-Metadaten in GUI+API, Pickle-Dokumentation: implementiert.
- Remote/Distributed Evaluation: HTTP-Protokoll, Client/Worker, Retry/Cancel, Benchmark: implementiert.
- P2-Forschungsfeatures: Modulbibliothek, CPPN, Meta-adaptive Policies und evolvierbare Descriptor-Gewichte sind implementiert; DARTS-Lite, Intrinsic Curiosity und Shared Weights vollstaendig integriert; STDP, Neuromodulation, Input-/Output-Gruppierung, Conv-NEAT und ES-HyperNEAT sind noch Spikes/offen.
- raw_fitness-Fix (Fitness-Komponenten verschmutzen nicht mehr genome.fitness fuer Ziel-Check und Diagnostics): implementiert.
- Event-System, Anomalie-Detektion, Fitness-Transformer, Genome-Export, Validierungs-Set, Konfigurationspersistenz, Gym-Inspect-Verbesserung: implementiert.
- **P0 Meta-Adaptive Orchestration Layer:** Alle 6 Phasen abgeschlossen (ParamRegistry, ProblemProfiler, KnowledgeBase, MetaOptimizer, Feature Gating, auto_train).
- **P1 GUI-Integration P0:** `⚡ Auto-Train`-Button, `AutoSetupWorker` (non-blocking Profiling), MetaOptimizer+FeatureGating-Live-Panel im Left-Panel, `auto_config_report`-Dialog nach Training. 4 neue Smoke-Tests.
- **auto_train Bugfixes (via PI-Beispiel-Testing):** (1) `raw_fitness` enthielt Curiosity-Bonus → Stop-Bedingung feuerte auf aufgeblähten Wert; (2) ungekappter Curiosity-Bonus ermöglichte Reward Hacking (Fitness ~10⁹) wenn DARTS numerisch instabile Outputs produzierte; (3) `max_time_seconds`-Budget ignorierte Lamarck-Overhead. Alle drei behoben.
- **Code-Qualität:** Tote Parameter/Funktionen entfernt (`total_entries` in `_confidence`, `_median`, `_prev_best`); UCB1-Reward-Attribution in EXPLOIT-Phase korrigiert (`best()` tracked jetzt `_last_idx`).
- **P1 WandB/MLflow Tracking:** `TrackingBackend`-Protocol, `WandbBackend`, `MlflowBackend`, `set_tracking_backend(*backends)`; Metrics werden einmal pro Generation geloggt.
- **P1 Regression Benchmarking Suite:** `RegressionDetector`, `BaselineStore`, `HistoryStore`, `TrendReport`, `BenchmarkReport`; CLI `python -m yane.benchmarks --ci` mit Exit-Codes 0/1/2.

## Legende

- `P0`: hoher Hebel, nah am aktuellen Code, direkt nuetzlich
- `P1`: wichtiger Ausbau, mittlerer Aufwand
- `P2`: experimentell, Forschungsarbeit oder groesserer Umbau
- ✓: erledigt
- ⚡: teilweise erledigt
- □: offen

---

## Architektur-Leitlinie

YANE ist in drei konzeptionelle Schichten gegliedert. Neue Features sollen klar einer Schicht zugeordnet werden.

**Layer 1 — Stable Core**
Population, Genome, Speciation, Mutation, Evaluation, Checkpoints.
Kleine, testbare, rueckwaertskompatible Basis. Aenderungen hier erfordern besondere Sorgfalt; keine experimentellen Abhaengigkeiten.

**Layer 2 — Adaptive Systems**
Recovery, Auto-Tuning, Scheduling, Surrogates, Diversity-Systeme, Policy-Orchestrierung, **Meta-Adaptive Orchestration (MetaOptimizer, ParamRegistry, Knowledge Base, Feature Gating)**, **ResourceBudget-System (ResourceDiscovery, BudgetEnforcer, Graceful Degradation)**.
Baut auf Layer 1 auf; nutzt ausschliesslich oeffentliche APIs. Ziel: Adaptive Policy System + MetaOptimizer + ResourceBudget als kohaerentes System aus Optimierung (MetaOptimizer) und Begrenzung (ResourceBudget).

**Layer 3 — Research Features**
DARTS, STDP, Neuromodulation, Curiosity, ES-HyperNEAT, Input-/Output-Gruppierung, Convolutional NEAT, Attention-Heads, LTC-Nodes, ONNX/WASM/TFLite-Export, Distillation, Gradient-NEAT-Hybrid, Self-Play, H-NEAT, GRN-Encoding, Developmental NEAT, Continual Learning, Meta-Learning, Reservoir Computing, Open-Endedness, Multi-Agent Cooperation, Bayesian NEAT, Safe NEAT, Sparse NEAT, Symbolic Regression.
Research-Features werden modular und per API an-/abschaltbar implementiert. Sie koennen in Genome, Population und NeuroEvolution integriert sein, muessen aber standardmaessig deaktiviert sein und duerfen bei Deaktivierung keine Laufzeitkosten verursachen.

Ziel: Layer 1 wartbar halten; Layer 2 durch das Policy-Interface erweiterbar machen; Layer 3 vollstaendig integriert aber sicher deaktivierbar.

---

## Offene Tasks

---

### ✓ P1 GUI-Integration für P0 Meta-Adaptive Orchestration Layer

**Implementiert:**

- `⚡ Auto-Train`-Button im Training-Tab: startet `AutoSetupWorker` (non-blocking,
  profilt Problem + konfiguriert KB/MetaOptimizer/FeatureGating im Hintergrund),
  dann normaler `TrainingWorker` mit allen vorhandenen Live-Updates.
- `auto_config_report`-Dialog erscheint nach Auto-Train-Ende automatisch.
- Collapsible „Meta-Adaptive (P0)"-Gruppe im Left-Panel: MetaOptimizer-Phase,
  Overhead %, Ticks/Skipped, letzte Param-Änderungen, aktive Features,
  Feature-Status mit Degradation-Level. Nur sichtbar wenn P0 aktiv ist.
- P0-Diagnostics (`meta_optimizer`, `feature_gating`) werden thread-safe
  im `TrainingWorker._emit_update()` eingesammelt und per Signal übertragen.
- 4 neue Smoke-Tests (Button-Existenz, Panel-Labels, Update aus mem-Dict,
  AutoSetupWorker profiles und signalisiert setup_done).

**Nicht implementiert** (bewusst weggelassen):
- Inspect-Tab-Erweiterung (profile_problem manuell auslösbar)
- ParamRegistry-Live-View mit ≥30 Parametern (komplexes Widget, kein klarer Mehrwert gegenüber Diagnostics)
- Checkboxen für manuelle MetaOptimizer/FeatureGating-Overrides

---
### ✓ P0 Meta-Adaptive Orchestration Layer — Das selbstoptimierende YANE

**Das zentrale Architektur-Feature der nächsten Evolutionsstufe.** YANE soll
ein „System der unendlichen Adaption" werden: kein manuelles Tuning von
Dutzenden Parametern mehr, keine geratenen Startwerte, keine Frage „welches
Feature soll ich aktivieren?".

**Problem:** Jedes Subsystem hat eigene adaptive Regler (Mutationsraten,
Speziations-PI, Recovery-Eskalation, Online-Tuning-Bandit, Pop-Size-Adaptation,
Fitness-Shaping-Autoerkennung, Anytime-Eval-Budget, Surrogate-Filter). Aber
diese Regler arbeiten **isoliert** — keiner weiss vom anderen. Sie können sich
gegenseitig behindern (z. B. Recovery injected Diversity während Online-Tuning
die Mutationsrate senkt). Und jedes NEUE Feature (Attention, LTC, H-NEAT, ...)
bringt 3–6 neue Parameter, die der Nutzer wieder manuell einstellen müsste.

**Lösung:** Ein Meta-Layer, der ALLE Parameter — existierende adaptive wie neue
P2 — als gemeinsamen Suchraum behandelt, kontinuierlich beobachtet und
koordiniert optimiert.

**Ziel:** `yane.auto_train(evaluator)` — KEIN einziger manueller Parameter mehr.

---

#### Architektur

```
┌──────────────────────────────────────────────────────────────────┐
│                    MetaOptimizer (evolution/meta_optimizer.py)     │
│                                                                    │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────────────┐   │
│  │ Problem      │  │ Parameter        │  │ Performance         │   │
│  │ Profiler     │─▶│ Explorer         │─▶│ Monitor             │   │
│  │              │  │                  │  │                     │   │
│  │ - Task-Typ   │  │ - UCB1 (diskret) │  │ - Fitness-Trend     │   │
│  │ - Schwierig- │  │ - BayesOpt (kont)│  │ - Konvergenzrate    │   │
│  │   keit       │  │ - Succ.Halving   │  │ - Stagnation        │   │
│  │ - Rauschen   │  │   (Feature-Gate) │  │ - Diversity         │   │
│  │ - Temporal   │  │                  │  │ - Param-Impact      │   │
│  └──────────────┘  └─────────────────┘  └────────────────────┘   │
│          │                  │                      │               │
│          ▼                  ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │         Unified Parameter Registry (alle Subsysteme)          │ │
│  │                                                               │ │
│  │  ┌─ Existing Adaptive ─┐  ┌─ New P2 Features ────────────┐  │ │
│  │  │ mutation.*_rate     │  │ attention.head_dim           │  │ │
│  │  │ species.target_n    │  │ attention.num_heads          │  │ │
│  │  │ recovery.cooldown   │  │ ltc.tau_initial              │  │ │
│  │  │ recovery.strategies │  │ ltc.dt                       │  │ │
│  │  │ lamarck.mode        │  │ hneat.n_sub_policies         │  │ │
│  │  │ lamarck.n_steps     │  │ grn.development_steps        │  │ │
│  │  │ novelty.weight      │  │ continual.lambda_ewc         │  │ │
│  │  │ pop.adaptive_mode   │  │ meta.adaptation_steps        │  │ │
│  │  │ anytime.promotion   │  │ reservoir.n_nodes            │  │ │
│  │  │ surrogate.frac      │  │ reservoir.spectral_radius    │  │ │
│  │  │ fitness.transform   │  │ safe.penalty                 │  │ │
│  │  │ crossover.tourn_k   │  │ bayesian.uncertainty_penalty │  │ │
│  │  │ ...                 │  │ sparse.target_sparsity       │  │ │
│  │  └─────────────────────┘  │ hw.max_flops                 │  │ │
│  │                           │ ...                          │  │ │
│  │                           └──────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │            Cross-Run Knowledge Base                          │ │
│  │  "CartPole-ähnlich → CMA-ES bringt +15% Fitness"            │ │
│  │  "XOR-artig → small pop, high mutation, no Lamarck"         │ │
│  │  k-NN über Problem-Profile im RunDatabase                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │            Feature Gating & Graceful Degradation              │ │
│  │  Auto-Auswahl: welche Research-Features sind aktiv?          │ │
│  │  Auto-Deaktivierung: Features die nichts bringen             │ │
│  │  Gradueller Rückbau statt hartem Aus                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

#### ✓ Phase 1: Unified Parameter Registry

**Abgeschlossen:** `evolution/param_registry.py` mit `ParamSpec`, `ParamRegistry`,
`build_default_registry()`. 38 Parameter aus 12 Subsystemen registriert.
`NeuroEvolution.set_param()`, `get_param_space()`, `get_param_registry()`.
55 Tests in `tests/test_param_registry.py`.

Jeder konfigurierbare Parameter in YANE wird in einer zentralen Registry
angemeldet — egal ob existierend-adaptiv oder neues P2-Feature.

**Design: `ParamSpec` und `ParamRegistry`**

```python
@dataclass
class ParamSpec:
    name: str                    # "lamarck.mode"
    type: str                    # "categorical" | "continuous" | "integer"
    domain: Any                  # ["hill_climb","cma_es","nes","sa","none"]
                                 # oder (0.0, 1.0) für continuous
    default: Any                 # Default-Wert
    stage: str                   # "init" | "runtime" | "both"
    impact_history: list[float]  # Fitness-Delta pro Änderung (gefüllt zur Laufzeit)
    subsystem: str               # "lamarck", "attention", "speciation", ...
```

- Alle existierenden `set_*()`-Methoden registrieren ihre Parameter automatisch
  in der `ParamRegistry`.
- Neue P2-Features registrieren ihre Parameter bei `__init__`.
- `NeuroEvolution.get_param_space()` → vollständiger Suchraum als Dict.
- `NeuroEvolution.set_param(name, value)` → universeller Parameter-Setter
  (umgeht spezifische `set_*()`-Methoden nicht, ruft sie intern auf).

**Akzeptanzkriterien Phase 1:**

- `get_param_space()` listet ≥30 Parameter (alle existierenden Adaptiven +
  mindestens Stubs für P2).
- `set_param("lamarck.mode", "cma_es")` funktioniert identisch zu
  `set_lamarck(mode="cma_es")`.
- Parameter-Änderungen werden in Diagnostics protokolliert.
- Tests: Registry-Vollständigkeit; Set/Get-Roundtrip; Typ-Validierung.

---

#### ✓ Phase 2: Problem Profiler

**Abgeschlossen:** `evolution/problem_profiler.py` mit `ProblemProfile`-Dataclass
und `ProblemProfiler`. Metriken: `noise_level`, `temporal_dependency`,
`reward_sparsity`, `estimated_difficulty`, `task_type` (4 Kategorien mit
Confidence). `NeuroEvolution.profile_problem(evaluator, n_warmup)`.
44 Tests in `tests/test_problem_profiler.py`.

Vor Trainingsstart analysiert YANE die Aufgabe automatisch.

**Design: `ProblemProfiler` in `evolution/problem_profiler.py`**

```python
profile = yane.profile_problem(evaluator, n_warmup=50)
# → ProblemProfile(
#     task_type="rl_continuous",    # classification, regression,
#                                   # rl_discrete, rl_continuous, ...
#     n_inputs=4,
#     n_outputs=2,
#     estimated_difficulty=0.72,    # 1 - (Random-Baseline / Target)
#     noise_level=0.08,             # std der Fitness über n_warmup Random-Genome
#     reward_sparsity=0.31,         # Anteil Episoden mit ≈0 Reward
#     temporal_dependency=0.83,     # Autokorrelation State[t] vs State[t-1]
#     state_dim_effective=3,        # PCA: 95%-Varianz-Dimensionen
#     action_distribution=...,      # Verteilung der optimalen Aktionen (geschätzt)
# )
```

**Analyse-Methoden:**

- `task_type`: heuristische Klassifikation basierend auf Input/Output-Dimensionen,
  Reward-Struktur (diskret/kontinuierlich), Episoden-Länge.
- `estimated_difficulty`: N Random-Genome evaluieren → mittlere Fitness als
  Baseline. Target-Fitness (falls gesetzt) als Obergrenze. Difficulty = 1 −
  (Baseline/Target).
- `noise_level`: Varianz der Fitness bei wiederholter Evaluation desselben
  Random-Genoms.
- `temporal_dependency`: Autokorrelation der Umgebungszustände (→ braucht das
  Netz Memory?). Berechnet aus N zufälligen Episoden.
- `state_dim_effective`: PCA über gesammelte States während Warmup → effektive
  Dimensionalität (95%-Varianz-Schwelle).

**Akzeptanzkriterien Phase 2:**

- `profile_problem()` gibt gültiges `ProblemProfile` zurück nach ≤2s Analyse.
- CartPole: `temporal_dependency > 0.5` (braucht Memory).
- XOR: `task_type == "classification"`, `noise_level ≈ 0`.
- LunarLander: `estimated_difficulty > 0.5`, `reward_sparsity > 0.5`.
- Tests: Task-Typ-Erkennung; Difficulty-Schätzung; Temporal-Dependency;
  PCA-Dimension; Noise-Level.

---

#### ✓ Phase 3: Parameter Explorer (Meta-Optimizer)

Der Meta-Optimizer wählt und justiert Parameter — vor dem Start (Initialisierung
via Knowledge Base + Problem-Profil) und während des Trainings (kontinuierliche
Adaption).

**Drei Optimierungsstrategien (je nach Parameter-Typ):**

#### 3a. UCB1-Bandit für kategoriale Parameter

```python
# Parameter: lamarck.mode ∈ {hill_climb, cma_es, nes, sa, none}
# Alle 20 Generationen: Meta-Optimizer wählt einen Mode via UCB1
# Reward = Fitness-Delta seit letztem Wechsel
```

- Existierender `UCB1Bandit` in `online_tuning.py` wird generalisiert.
- Jeder kategoriale Parameter bekommt einen eigenen Bandit.
- Exploration/Exploitation-Balance: `c=2.0` (konfigurierbar).
- Konvergenz: Bandit konvergiert auf besten Mode nach ~5-10 Trials.

#### 3b. Bayesian Optimization für kontinuierliche Parameter

```python
# Parameter: ltc.tau ∈ [0.1, 10.0], attention.head_dim ∈ [2, 16]
# Gaussian Process modelliert Fitness(tau, head_dim, ...)
# Expected Improvement wählt nächsten Testpunkt
```

- Leichte Implementation (kein scipy/gpytorch-Abhängigkeit):
  einfacher Gaussian Process mit RBF-Kernel (ca. 200 Zeilen).
- Optional: externes `scipy.optimize` für Acquisition-Funktion.
- Fallback: Random Search wenn GP nicht verfügbar.

#### 3c. Successive Halving für Feature-Gating

```python
# Welche Research-Features sind aktiv?
# Phase 1: alle Features testen (10 Generationen)
# Phase 2: beste 50% behalten (20 Generationen)
# Phase 3: beste 25% behalten (40 Generationen)
# → nur Features mit echtem Impact überleben
```

**Trainingsphasen (automatisch, datengetrieben):**

```
Phase EXPLORE   (0-20% des Budgets): große Pop, hohe Mutation, viele Features
Phase EXPLOIT   (20-60%): Lamarck aktiv, Surrogate aktiv, Feature-Selektion
Phase REFINE    (60-90%): CMA-ES, kleine Pop, nur bewährte Features
Phase CONVERGE  (90-100%): Fine-Tuning, Early-Stopping-Prüfung
```

- Phasenübergänge sind NICHT hartkodiert, sondern datengetrieben:
  Fitness-Plateau > 30 Generationen → nächste Phase.
- Jede Phase hat eigene Parameter-Priors (die aber auch adaptiv sind).

**Integration existierender adaptiver Systeme:**

| Existierendes System | Meta-Optimizer-Rolle |
|---|---|
| Mutationsraten (selbstadaptierend) | Initiale Priors setzen; `rate_mutation_rate` adaptiv justieren; bei Stagnation → temporärer Mutations-Burst |
| Speziations-PI-Regler | PI-Gains (`Kp`, `Ki`) tunen; Ziel-Spezies-Anzahl adaptiv je nach Problem-Komplexität |
| Recovery-Eskalation | Strategie-Reihenfolge optimieren; `cooldown` an Problem-Typ anpassen (lange Cooldowns für Stepping-Stone-Probleme) |
| Online-Tuning (UCB1) | Exploration/Exploitation-Balance (`c`) adaptiv; Tune-Intervall dynamisch |
| Pop-Size-Adaptation | Modus-Wahl (`linear_decay` vs. `performance_based`); `min_pop`/`max_pop` aus Problem-Profil |
| Fitness-Shaping | Transformations-Typ und -Aggressivität automatisch |
| Anytime-Eval | `promotion_frac` adaptiv; `max_evals` basierend auf Noise-Level |
| Surrogate | `surrogate_frac` und Update-Frequenz automatisch |

**Akzeptanzkriterien Phase 3:**

- UCB1-Bandit konvergiert auf besten kategorialen Wert innerhalb 10 Trials
  (Unit-Test mit simulierter Reward-Funktion).
- Bayesian Optimization findet Optimum eines 2D-Testfunktion (Branin) innerhalb
  30 Iterationen.
- Successive Halving eliminiert schlechte Features und behält gute.
- Phasenübergänge erfolgen datengetrieben (nicht hartkodiert).
- ≥5 existierende adaptive Systeme sind in die Registry integriert.
- Tests: UCB1-Konvergenz; BayesOpt auf Testfunktion; Successive-Halving;
  Phasen-Logik; Registry-Integration.

---

#### ✓ Phase 4: Cross-Run Knowledge Base

**Abgeschlossen:** `evolution/knowledge_base.py` mit `KBEntry`, `KnowledgeBase`,
`profile_to_vector()`, `cold_start_suggestion()`. Gewichtetes k-NN (10 Features,
Euclidean). JSON-Persistenz mit `save()`/`export_json()`/`import_json(merge)`.
`NeuroEvolution.set_knowledge_base()`, `knowledge_base`, `suggest_params()`.
Auto-Learn nach `train()` wenn `profile_problem()` zuvor aufgerufen. RunDatabase
um `problem_profile_json`-Spalte erweitert. 54 Tests in `tests/test_knowledge_base.py`.

YANE lernt über mehrere Runs hinweg: welche Parameter für welche Problem-Typen
funktionieren.

**Design: `KnowledgeBase` in `evolution/knowledge_base.py`**

```python
# Nach jedem Run:
kb = yane.knowledge_base
kb.learn(problem_profile, final_params, final_fitness)

# Vor dem nächsten Run:
suggestion = kb.suggest(problem_profile, top_k=5)
# → [
#     {"params": {...}, "expected_fitness": 195.0, "confidence": 0.85},
#     {"params": {...}, "expected_fitness": 188.0, "confidence": 0.72},
#     ...
#   ]
```

**Algorithmus:**

1. k-NN über Problem-Profile im RunDatabase (Features: task_type_onehot,
   n_inputs, n_outputs, difficulty, noise_level, temporal_dependency,
   state_dim_effective).
2. Distanzmetrik: gewichteter euklidischer Abstand (Gewichte aus
   historischer Feature-Importance).
3. Top-K ähnlichste Runs: gewichteter Median ihrer finalen Parameter.
4. Confidence: invers zur Distanz-Streuung der Top-K (enge Nachbarschaft →
   hohe Confidence).
5. Fallback (kalter Start): Default-Parameter basierend auf Task-Typ (z. B.
   „classification → Lamarck hill-climb, small pop").

**Speicherung:**

- RunDatabase wird um `problem_profile`-Spalte erweitert.
- Knowledge-Base-Indizes (k-NN-Baum) werden beim Laden vorgecached.
- Optional: Export/Import der KB für Team-Sharing (JSON).

**Akzeptanzkriterien Phase 4:**

- `suggest()` gibt sinnvolle Vorschläge nach ≥3 ähnlichen Runs.
- Confidence steigt mit Anzahl ähnlicher Runs in der KB.
- Kalter Start: Default-Vorschlag basierend auf Task-Typ.
- k-NN-Distanz verwendet gewichtete Features.
- Tests: k-NN-Korrektheit; Confidence-Berechnung; Kalter-Start-Fallback;
  KB-Import/Export.

---

#### ✓ Phase 5: Feature Gating & Graceful Degradation

**Abgeschlossen:** `evolution/feature_gating.py` mit `FeatureStatus`, `FeatureRecord`,
`FeatureGate`, `_register_known_features()`. UCB1-Selektion inaktiver Features,
Testfenster (`test_duration`), `degradation_level` in 0.2er-Schritten → disabled ab 0.8,
Auto-Reaktivierung nach `reactivation_delay` Generationen, explizites `reactivate()`.
5 Research-Features: curiosity, darts, shared_weights, novelty_search, diversity_injection.
`NeuroEvolution.set_auto_features()`, `get_feature_gating_diagnostics()`, `_tick_feature_gating()`
im `train()`-Loop. 53 Tests in `tests/test_feature_gating.py`.

Nicht jedes Research-Feature hilft bei jedem Problem. YANE soll automatisch
auswählen, welche Features aktiv sind — und schlechte graduell zurückfahren.

**Design: `FeatureGate` in `evolution/feature_gating.py`**

```python
yane.set_auto_features(max_concurrent=3, test_interval=50)
```

**Feature-Gating (Successive Halving, Phase 3c):**

1. Alle Research-Features (Attention, LTC, H-NEAT, GRN, Developmental,
   Curiosity, DARTS, STDP, Neuromodulation, ...) starten inaktiv.
2. Alle `test_interval` Generationen: wähle ein inaktives Feature via UCB1,
   aktiviere es für ein Testfenster.
3. Wenn Feature Fitness-Delta > Schwellwert → behalte aktiv.
4. Wenn Feature keinen Impact → deaktiviere und merke es als „nicht hilfreich".
5. `max_concurrent`: nie mehr als N Features gleichzeitig aktiv (Overhead-Schutz).

**Graceful Degradation (statt hartem Deaktivieren):**

```
Feature mit abnehmendem Impact:
  Schritt 1: Parameter reduzieren (head_dim 8→4, num_heads 2→1)
  Schritt 2: Feature-Gewicht in Ensemble/Fitness halbieren
  Schritt 3: Feature vollständig deaktivieren
```

- Jedes Feature hat einen `degradation_level` (0.0 = voll aktiv, 1.0 = deaktiviert).
- `degradation_level` steigt wenn Feature-Impact (gemessen via
  `ParamSpec.impact_history`) über 30 Generationen unter Schwellwert.
- Bei `degradation_level > 0.8`: Feature wird aus Registry entfernt
  (keine Laufzeitkosten mehr).
- Reaktivierung möglich wenn Umgebung sich ändert (Continual Learning).

**Feature-Impact-Messung:**

- A/B-Vergleich: jede 10. Generation wird ein Kontroll-Genom OHNE das Feature
  evaluiert. Fitness-Delta = Feature-Impact.
- Bei Features die nur initial wirken (z. B. gute Initialisierung durch GRN):
  Impact über erste 20 Generationen messen.

**Akzeptanzkriterien Phase 5:**

- Nach 200 Generationen sind ≤ `max_concurrent` Features aktiv.
- `degradation_level` steigt bei Features ohne Impact.
- Deaktivierte Features verursachen keinen messbaren Laufzeit-Overhead
  (<1% der Trainingszeit).
- Feature kann reaktiviert werden (z. B. in Continual-Learning-Phase).
- Tests: Feature-Selektion; Degradation-Mechanik; Impact-Messung;
  Reaktivierung; Overhead-Freiheit.

---

#### ✓ Phase 6: Zero-Config Start — `auto_train()`

**Abgeschlossen:** `evolution/auto_train.py` mit `AutoTrainResult`-Dataclass,
`pick_pop_size()`, `apply_cold_start_defaults()`, `build_report()`.
`NeuroEvolution.auto_train(evaluator, n_inputs, n_outputs, target_fitness,
max_time_seconds, problem_name, n_warmup)` orchestriert alle 6 Phasen:
Profile → KB-Suggest → Configure → MetaOptimizer + FeatureGating → Train →
KB-Learn → AutoTrainResult mit `auto_config_report`. `AutoTrainResult` in
`yane.__init__` exportiert. 38 Tests in `tests/test_auto_train.py`.

Das Endziel. Der Nutzer spezifiziert NUR die Aufgabe — YANE macht den Rest.

**Design:**

```python
yane = NeuroEvolution()

# Der einzige API-Call:
result = yane.auto_train(
    evaluator,
    target_fitness=None,          # optional, sonst auto-detektiert
    max_time_seconds=None,        # optional, sonst 30min Default
    problem_name=None             # optional, für Knowledge-Base-Lookup
)

# result = AutoTrainResult(
#     best_genome=...,
#     final_fitness=...,
#     total_generations=...,
#     wall_time=...,
#     active_features=["attention", "ltc"],
#     final_params={...},         # alle finalen Parameter
#     problem_profile=...,
#     auto_config_report=...      # "Warum diese Konfiguration?"
# )
```

**Ablauf `auto_train()`:**

1. **Profile** (Phase 2): `profile_problem(evaluator)` → ProblemProfile.
2. **Suggest** (Phase 4): `knowledge_base.suggest(problem_profile)` →
   initiale Parameter + Feature-Auswahl.
3. **Configure**: `configure()` + alle `set_*()` automatisch basierend auf
   Suggestion. `max_iterations` aus `max_time_seconds` und Problem-Schwierigkeit
   geschätzt.
4. **Train**: `train(evaluator)` mit Meta-Optimizer (Phase 3) der kontinuierlich
   Parameter justiert und Feature-Gating (Phase 5) das Features an-/abschaltet.
5. **Learn** (Phase 4): `knowledge_base.learn(problem_profile, final_params,
   final_fitness)` für zukünftige Runs.
6. **Return**: `AutoTrainResult` mit vollständiger Dokumentation aller
   automatischen Entscheidungen.

**`auto_config_report` (Transparenz):**

```python
print(result.auto_config_report)
# ─── Auto-Configuration Report ───
# Problem: CartPole-v1 (rl_continuous, 4→2)
# Difficulty: 0.72 | Noise: 0.08 | Temporal: 0.83
#
# Knowledge Base: 12 similar runs found (confidence: 0.85)
# Top suggestion: CMA-ES + pop=150, attention=on, ltc=off
#
# Phase Transitions:
#   Gen   0- 80: EXPLORE  (pop=150, features=[attention, hneat])
#   Gen  80-250: EXPLOIT  (pop=80,  lamarck=cma_es, features=[attention])
#   Gen 250-350: REFINE   (pop=50,  lamarck=cma_es, surrogate=on)
#   Gen 350-400: CONVERGE (pop=30,  early_stopping)
#
# Final active features: attention (impact: +12.3%)
# Degraded features: hneat (no impact detected)
# Total wall time: 4m32s | Generations: 400 | Best fitness: 500.0
```

**Fallback-Strategie (wenn KB kalt):**

- Keine ähnlichen Runs in KB → konservative Defaults:
  - Lamarck: hill-climb, 3 steps
  - Pop: 100, linear_decay
  - Keine Research-Features initial aktiv
  - Meta-Optimizer exploriert aggressiver (höheres `c` im UCB1)
- Nach diesem ersten Run → KB hat ersten Eintrag → nächster Run profitiert.

**Akzeptanzkriterien Phase 6:**

- `auto_train(evaluator)` läuft OHNE jegliche `set_*()`-Aufrufe.
- `AutoTrainResult` enthält alle erforderlichen Metadaten.
- `auto_config_report` dokumentiert jede automatische Entscheidung
  nachvollziehbar.
- Bei identischem Seed + KB-State: reproduzierbare Ergebnisse.
- Ohne KB (kalter Start): konservative Defaults, Training funktioniert.
- Mit KB (≥3 ähnliche Runs): Parameter-Vorschlag weicht <20% vom Optimum ab.
- Tests: End-to-End `auto_train()` auf XOR und CartPole; KB-Integration;
  Reproduzierbarkeit; Kalter-Start-Fallback; Report-Generierung.

---

#### Implementierungsreihenfolge

1. **Phase 1 (Unified Param Registry)**: Fundament für alles weitere.
   Betrifft `core/` und `neuro_evolution.py`. ~300 LOC + Tests.
2. **Phase 2 (Problem Profiler)**: Unabhängig, nur `evolution/`.
   ~400 LOC + Tests.
3. **Phase 4 (Knowledge Base)**: Baut auf RunDatabase (existiert) und
   Problem Profiler auf. ~500 LOC + Tests.
4. **Phase 3 (Meta-Optimizer)**: Baut auf Registry + Profiler + KB auf.
   Integration in `train()`. ~800 LOC + Tests.
5. **Phase 5 (Feature Gating)**: Baut auf Meta-Optimizer auf.
   ~400 LOC + Tests.
6. **Phase 6 (auto_train)**: Integration aller Phasen.
   ~300 LOC + Tests.

Geschätzte Gesamtgröße: ~2700 LOC + ~80 Tests.

---

#### Bekannte Probleme & Gegenmassnahmen

Diese Probleme wurden in der Design-Phase identifiziert und werden aktiv
adressiert — nicht als „das machen wir später", sondern als integrale
Bestandteile der Implementierung.

---

#### Problem 1: Credit Assignment — Wer war's?

**Szenario:** MetaOptimizer ändert gleichzeitig `lamarck.mode=cma_es`,
`pop_size=120` und aktiviert `attention`. Fitness springt um +15%. Welche
Änderung war verantwortlich?

Ohne korrekte Attribution lernt der MetaOptimizer falsche Kausalitäten.

**Gegenmassnahme: Shapley-Approximierte Counterfactuals**

```python
# Alle attribution_interval (Default: 100 Gen): Counterfactual-Rollouts
#   Group A: alle Änderungen zusammen
#   Group B: nur lamarck=cma_es
#   Group C: nur pop=120
#   Group D: nur attention=on
#   Group E: keine Änderung (Baseline)
# → Attribution via Shapley-Value-Approximation pro Parameter

# Kostenbegrenzung:
# - Nur bei |Fitness-Delta| > 2σ der Baseline-Varianz
# - Nur wenn Overhead-Budget es erlaubt (<5% Gesamtzeit)
# - Surrogate-Modell für günstige Counterfactual-Vorhersage (keine echten Evals)
```

**Akzeptanzkriterien:** Shapley-Werte summieren zum Gesamt-Delta (±5%);
Surrogate-Counterfactuals korrelieren mit echten (Pearson > 0.7).

---

#### Problem 2: Parameter-Interaktionen

**Szenario:** `lamarck=cma_es` bringt +10% wenn Population klein, −5% wenn
gross. UCB1 und univariate Bayesian Optimization modellieren keine
Interaktionen — lernen falschen Durchschnitt.

**Gegenmassnahme: Multivariate Bayesian Optimization mit ARD-Kernel**

```python
# Gaussian Process mit Matern-5/2-Kernel + Automatic Relevance Determination
# Modelliert: f(param1, param2, ..., paramN) → expected_fitness
# ARD erkennt automatisch irrelevante Parameter (hohe Lengthscale).

# Skalierung: GP mit 30 Parametern ist O(n³). Lösung: Additive GP
# f(x₁,...,x₃₀) ≈ f₁(x₁,x₂) + f₂(x₃,x₄,x₅) + ...  (linear skalierend)
# Nur Top-10 einflussreichste Parameter aktiv modellieren (via ARD-Selektion)
```

**Akzeptanzkriterien:** Additive GP findet Interaktion in synthetischer
Testfunktion (z. B. `f(x,y) = x*y`); ARD identifiziert irrelevante Parameter
(Lengthscale > 10× Median).

---

#### Problem 3: Delayed Rewards & Non-Stationarität

**Szenario:** MetaOptimizer erhöht `pop_size=50→200` in Gen 10. Wirkung zeigt
sich erst in Gen 50. Aber in Gen 25 wurde `pop_size` schon wieder geändert.
Der optimale Parameterwert ändert sich zudem ÜBER die Trainingszeit.

**Gegenmassnahme: Eligibility Traces + Schedule-Optimierung**

```python
# Eligibility Trace über Parameter-Änderungen:
#   trace[t] = trace[t-1] * lambda + delta_fitness[t]
#   lambda=0.9 → Reward über ~10 Gen "zurückpropagiert"

# Zusätzlich: MetaOptimizer optimiert SCHEDULES, nicht Punktwerte
# param = schedule(t, problem_phase)
# Schedule-Typen: constant, linear_decay, exponential_decay, step, cosine
# Nur 2-3 Schedule-Parameter statt einem Wert pro Generation
```

**Akzeptanzkriterien:** Eligibility-Trace korreliert Parameter-Änderung mit
verzögertem Fitness-Delta (Test: künstliche Verzögerung von 10 Gen);
Schedule-basierte Optimierung findet besseres Optimum als punktbasiert im
nicht-stationären Testfall.

---

#### Problem 4: Wer tuned den Tuner?

**Szenario:** MetaOptimizer hat eigene Hyperparameter (UCB1-`c`, GP-Kernel,
Acquisition-Funktion, Halving-Eta, Attributions-Intervall). Unendlicher Regress.

**Gegenmassnahme: Robuste Defaults + Selbst-Überwachung (Self-Throttling)**

```python
META_DEFAULTS = {
    "ucb1_c": 2.0,               # Standard, gut erforscht
    "gp_kernel": "matern52",     # Robust für Bayesian Opt
    "acquisition": "ei",         # Expected Improvement
    "halving_eta": 3,            # Standard Successive Halving
    "attribution_interval": 100, # Nur alle 100 Gen
    "max_overhead_pct": 5.0,     # Max 5% Overhead (Self-Throttling)
}

# Self-Throttling:
# - Overhead kontinuierlich messen
# - Bei >5%: Intervall erhöhen, Attribution aussetzen, GP vereinfachen
# - Bei <2%: aggressiver optimieren (falls Fitness stagniert)
# → "conservative by default, self-regulating"
```

**Akzeptanzkriterien:** Overhead bleibt unter `max_overhead_pct`; bei
Overhead-Überschreitung greift Throttling innerhalb von 2 Generationen.

---

#### Problem 5: Cold Start & KB-Drift

**Szenario:** Erster Run auf neuem Problem-Typ → Knowledge Base leer.
Oder: KB-Einträge aus YANE 0.1.x sind für 0.2.x (neue Features) irreführend.

**Gegenmassnahme: Tiered Fallback + KB-Versioning**

```python
# Drei Fallback-Ebenen:
# 1. KB: ähnliche Runs (confidence > 0.7) → KB-Suggestion
# 2. Heuristiken: Task-Typ-basierte Defaults
#    "classification → Lamarck hill-climb, pop=50, no memory"
# 3. Universelle Defaults: konservativ, funktionieren für alles
#    "Lamarck=off, pop=100, linear_decay, keine Research-Features"

# KB-Versioning:
kb_entry = {
    "yane_version": "0.2.0",
    "compatibility_hash": "a3f2...",  # Hash relevanter Code-Module
    "params": {...}, "fitness": 195.0
}
# Neuere Einträge höher gewichtet (exponentieller Decay)
# Major-Version-Sprung → alte Einträge nur als "weak prior"
```

**Akzeptanzkriterien:** Kalter Start wählt Ebene-3-Defaults; nach 3 ähnlichen
Runs wechselt Suggestion zu Ebene 1; KB-Versioning blockt inkompatible
Einträge.

---

#### Problem 6: Overhead-Kosten

**Szenario:** Bayesian Optimization mit 30 Parametern, Counterfactuals,
Profiling, KB-Abfragen → 10 Minuten Overhead bei 30-Minuten-Training.

**Gegenmassnahme: Overhead-Budget mit Lazy Evaluation**

```python
class OverheadBudget:
    def __init__(self, max_pct=5.0):
        self.max_pct = max_pct
        self.total_eval_time = 0.0
        self.meta_time = 0.0

    def can_run(self, cost_estimate):
        projected = (self.meta_time + cost_estimate) / max(self.total_eval_time, 1.0)
        return projected < self.max_pct

# Konsequenzen:
# - Counterfactuals nur bei Budget
# - GP mit reduziertem Parameter-Set (Top-10 statt Top-30)
# - KB-Update asynchron (nach Training)
# - Profiling auf max. 2% der geschätzten Trainingszeit begrenzt
# - MetaOptimizer-Operationen sind priorisiert:
#   1. (immer): Phasen-Erkennung, Fitness-Monitoring
#   2. (oft):   UCB1 für diskrete Parameter (billig)
#   3. (selten): GP-Optimierung, Counterfactuals, Attribution (teuer)
```

**Akzeptanzkriterien:** Overhead-Budget wird eingehalten; teure Operationen
werden ausgesetzt wenn Budget knapp; billige Operationen laufen immer.

---

#### Problem 7: Subsidiarität — Wer optimiert was?

**Szenario:** `mutation.shift_rate` ist evolvierbar. MetaOptimizer will sie
anpassen. Beide Regler können gegeneinander arbeiten.

**Gegenmassnahme: Hierarchische Regelung mit klaren Verantwortlichkeiten**

```python
# Prinzip: MetaOptimizer setzt ZIELBEREICHE, nicht Punktwerte.
# Die untere Ebene optimiert innerhalb ihres Bereichs.

# Konkret für Mutationsraten:
meta_optimizer.set_param("mutation.global_pressure", 1.5)
# → Alle Raten werden mit 1.5× skaliert (phasen-abhängig)

meta_optimizer.set_param("mutation.shift_rate_range", (0.01, 0.5))
# → Evolution darf shift_rate ∈ [0.01, 0.5] selbst optimieren

# rate_mutation_rate (wie schnell Raten mutieren):
# NICHT evolvierbar, sondern von MetaOptimizer kontrolliert
meta_optimizer.set_param("mutation.rate_mutation_rate", 0.05)
```

**Verantwortlichkeitsmatrix (Subsidiaritätsprinzip):**

| Parameter-Gruppe | Evolution (pro Genom) | MetaOptimizer (global) |
|---|---|---|
| Gewichte, Topologie, Bias | **Optimiert** | — |
| Aktivierungsfunktionen | **Optimiert** | — |
| `mutation.*_rate` (innerhalb Range) | **Optimiert** | Setzt Range |
| `mutation.global_pressure` | — | **Optimiert** (phasen-abhängig) |
| `mutation.rate_mutation_rate` | — | **Optimiert** (nicht evolvierbar) |
| `pop_size` | — | **Optimiert** |
| `lamarck.mode`, `lamarck.n_steps` | — | **Optimiert** |
| `novelty.weight` | — | **Optimiert** |
| `species.target_n` | — | **Optimiert** |
| `recovery.cooldown`, Strategie-Reihenfolge | — | **Optimiert** |
| `feature.enabled` | — | **Optimiert** |
| `attention.head_dim` | — | **Optimiert** |
| Alle P2-Parameter | — | **Optimiert** |

**Akzeptanzkriterien:** `mutation.global_pressure` skaliert alle Raten;
`mutation.*_rate` bleibt innerhalb des gesetzten Ranges; `rate_mutation_rate`
ist nicht mehr evolvierbar; MetaOptimizer-Änderungen und Evolution
konfligieren nicht (Test: Monitor über 200 Gen).

---

#### Problem 8: Problem-Profil-Fehler

**Szenario:** ProblemProfiler klassifiziert Task falsch → KB schlägt
unpassende Parameter vor → Training scheitert.

**Gegenmassnahme: Multi-Method-Profiling + Confidence-gewichtete KB-Suggestion**

```python
@dataclass
class ProblemProfile:
    task_type: str
    task_type_confidence: float     # 0.85 = 85% sicher
    alternative_types: list[str]    # ["rl_continuous"] bei Unsicherheit

# Drei Profiling-Methoden (Ensemble):
# 1. Heuristic: Input/Output/Action-Dimensionen, Reward-Range
# 2. Statistical: Fitness-Verteilung über Random-Genome
# 3. Behavioral: State-Transition-Analyse, Autokorrelation
# → Majority Vote oder gewichteter Average

# KB-Suggestion nutzt Confidence:
# confidence > 0.9 → KB-Suggestion mit geringer Exploration (c=1.0)
# confidence < 0.6 → KB-Suggestion mit hoher Exploration (c=3.0)
#                    + alternative_types als Secondary-Suggestions testen
```

**Akzeptanzkriterien:** Profiling-Confidence wird mitgeschrieben; niedrige
Confidence führt zu breiterer Exploration; Task-Typ-Erkennung ≥90% korrekt
über alle Gym-Standard-Umgebungen.

---

#### Problem 9: Reproduzierbarkeit

**Szenario:** `auto_train()` trifft Dutzende stochastische Entscheidungen.
Wie reproduziert man einen erfolgreichen Run?

**Gegenmassnahme: Vollständige Seed-Determination + Meta-State-Snapshot**

```python
result = yane.auto_train(evaluator, seed=42)

# seed=42 determiniert ALLES:
# - ProblemProfiler (Warmup-Genome, State-Sampling)
# - KB-Suggestion (k-NN-Tiebreaking)
# - MetaOptimizer (UCB1, BayesOpt-Init, Halving)
# - Feature-Gating (Test-Reihenfolge)
# - ALLE internen random-Entscheidungen

# Result enthält vollständigen Meta-State:
result.meta_state  # → Dict mit allen MetaOptimizer-Interna

# Bitgenaue Reproduktion:
yane.auto_train(evaluator, replay_from=result.meta_state)
```

**Akzeptanzkriterien:** Zwei `auto_train()`-Aufrufe mit gleichem Seed
produzieren identische Ergebnisse; `replay_from` reproduziert ohne
`evaluator`-Aufrufe (nur Config); Meta-State ist serialisierbar (JSON).

---

#### Problem 10: User Trust & Explainability

**Szenario:** `auto_train()` deaktiviert Attention. User: „Bug?". Der
`auto_config_report` ist Post-hoc — zu spät wenn das Training noch läuft.

**Gegenmassnahme: Live-Explainability + Override-API**

```python
# Live-Monitoring (GUI + API):
yane.auto_train(evaluator, interactive=True)

# GUI/API zeigt live:
#   "Gen 100: Attention impact = -3.2% → degrading (level 0.3)"
#   "Gen 150: Attention degraded (level 0.7) — no impact for 50 gen"
#   "Gen 200: Attention deactivated"
#
# User-Override (jederzeit):
yane.meta_override("attention.enabled", True)
#   → "User override: attention re-enabled. MetaOptimizer will respect."
#
# Lock/Unlock:
yane.meta_lock("lamarck.mode")    # „Lamarck bleibt wie es ist"
yane.meta_unlock("pop_size")      # „Pop-Size darf wieder optimiert werden"

# Erklärungs-API:
explanation = yane.explain_decision("attention.disabled")
# → {
#     "decision": "disabled",
#     "generation": 200,
#     "reason": "impact_below_threshold",
#     "impact_history": [0.03, -0.01, 0.005, ...],
#     "threshold": 0.01,
#     "confidence": 0.92,
#     "alternative_explanation": "noise (fitness_variance=2.1%)"
#   }
```

**Akzeptanzkriterien:** `explain_decision()` gibt nachvollziehbare Begründung;
Override wird respektiert (MetaOptimizer fasst Parameter nicht mehr an);
Lock/Unlock funktioniert pro Parameter; Live-GUI zeigt Degradations-Fortschritt.

---

#### Subsidiaritätsprinzip: Zusammenfassung

```
┌──────────────────────────────────────────────────────────────┐
│                    Wer optimiert was?                          │
│                                                               │
│  Evolution (pro Genom)       │  MetaOptimizer (global)        │
│  ───────────────────────     │  ──────────────────────        │
│  Gewichte                    │  pop_size                      │
│  Topologie (add/remove)      │  lamarck.mode, lamarck.n_steps │
│  Bias                        │  novelty.weight                │
│  Aktivierungsfunktionen      │  species.target_n              │
│  persist_value               │  recovery.*                    │
│  input_scale                 │  anytime.promotion_frac        │
│  mutation.*_rate (lokal)     │  surrogate.frac                │
│                               │  mutation.global_pressure      │
│  → Langsam, diversität-      │  mutation.rate_mutation_rate    │
│    fördernd, genom-spezifisch│  feature.* (alle P2)           │
│                               │                                │
│                               │  → Schnell, phasen-abhängig,  │
│                               │    global, datengetrieben      │
└──────────────────────────────────────────────────────────────┘
```

---

#### Auswirkungen auf bestehende Tasks

**Dieser Meta-Task ändert die Implementierung ALLER anderen offenen Tasks:**

- Jedes neue P2-Feature registriert seine Parameter in der `ParamRegistry`
  statt eigene `set_*()`-Methoden mit manuellen Defaults zu verwenden.
- Akzeptanzkriterien für P2-Features werden ergänzt um:
  „Parameter sind in der ParamRegistry registriert und via Meta-Optimizer
  auto-tuning-fähig."
- `set_*()`-Methoden bleiben als explizite Override-API erhalten, aber
  `auto_train()` nutzt sie nicht.

**Prioritätsverschiebung:** Dieser P0-Task sollte VOR den meisten P2-Tasks
implementiert werden, weil er das Fundament für deren Auto-Tuning-Integration
liefert.

---

#### Zusammenfassung: Was YANE nach diesem Task kann

```python
# VORHER (manuelles Tuning, raten, frustrieren):
yane = NeuroEvolution()
yane.configure(n_inputs=4, n_outputs=2, max_nodes=50, max_connections=200)
yane.set_min_fitness(195.0)
yane.set_max_iterations(50000)
yane.set_target_species(5)
yane.set_lamarck(mode="cma_es", n_steps=5)
yane.set_anytime_eval(enabled=True, promotion_frac=0.3)
yane.set_surrogate(enabled=True, surrogate_frac=0.5)
yane.set_attention(enabled=True, head_dim=4)  # <- geraten!
yane.set_ltc(enabled=False)                   # <- geraten!
yane.set_adaptive_recovery(enabled=True, ...)
# ... 15 weitere set_*()-Aufrufe ...
yane.train(evaluator)

# NACHHER (zero-config):
yane = NeuroEvolution()
result = yane.auto_train(evaluator)
print(result.auto_config_report)
```

---

---

### ✓ P1 Automated Regression Benchmarking Suite (CI-faehige Benchmark-Pipeline)

**Implementiert** in `benchmarks/regression.py`, `benchmarks/benchmarks.yaml`, `benchmarks/__main__.py`:

- `RegressionSeverity`: NONE / MINOR (<5%) / MAJOR (5–20%) / CRITICAL (>20% oder Success-Rate bricht um >20pp ein)
- `RegressionDetector`: prüft `median_fitness`, `success_rate` und `median_iterations`; Mann-Whitney-U via scipy (fallback: kein p-Wert); MINOR → NONE wenn p ≥ 0.05
- `BaselineStore`: JSON-Dateien in `benchmarks/baseline/`; `save/load/list_names`
- `HistoryStore`: JSONL-Zeitreihen in `benchmarks/history/`; `append/load`
- `TrendReport.ascii_plot()`: Unicode-Sparkline mit Richtungspfeil (↑↓→)
- `BenchmarkReport`: `exit_code()` → 0/1/2; `to_dict()` → JSON; `format_text()` → lesbar
- `run_benchmark_suite(path, update_baseline)`: lädt `benchmarks.yaml`, läuft alle Benchmarks, vergleicht Baselines, hängt an History an
- CLI `python -m yane.benchmarks --ci` → JSON-stdout + Exit-Code; `--update-baseline`; `--trend NAME`; `--list-baselines`
- 31 Tests, alle bestanden.

---

### ✓ P1 WandB / MLflow Integration (Experiment-Tracking-Backends)

**Implementiert** in `evolution/tracking.py`:
- `TrackingBackend` Protocol (duck-typed, keine Vererbung nötig)
- `WandbBackend`: `wandb.init/log/finish`, Metriken als `yane/<key>`
- `MlflowBackend`: `mlflow.start_run/log_params/log_metrics/end_run`
- `_scalar_metrics()`: filtert NaN und nicht-skalare Werte aus `mem`
- `NeuroEvolution.set_tracking_backend(*backends)`: variadic; kein Argument → löscht alle
- Hooks in `train()`: `init+log_config` einmalig, `log_metrics` einmal pro Generation,
  `finish` am Ende. Backend-Fehler brechen Training nie ab.
- 16 Tests (15 passed, 1 skipped wenn mlflow nicht installiert): Protocol, Dispatch,
  Mock-Backends für WandB und MLflow, BrokenBackend-Resilience.

---

### ✓ P1 Interactive Evolution — Human-in-the-Loop

**Implementiert** in `evolution/interactive_eval.py` und `gui/interactive_eval.py`:

- `EloRating`: K=32 Elo-Rating für paarweise Vergleiche (Summenerhaltung, Convergenz).
- `_RatingSurrogate`: Lineares Modell auf `genome_descriptor_vector`; nach `warmup_queries` aktiv; `confidence`-Schranke konfiguierbar.
- `InteractiveEvaluator(mode, surrogate_model, surrogate_warmup, surrogate_confidence_threshold)`:
  - Modi: `"rating"` (Slider 0-100), `"pairwise"` (Elo), `"ranking"` (Rank→Fitness), `"implicit"` (Verweildauer).
  - `set_feedback_source(fn, compare_fn)`: synchrones Oracle für Tests/Programmatic.
  - `submit_feedback(genome_id, value)`: GUI- oder Programmatic-Feedback; updated Elo/Rating, signalisiert wartende Threads.
  - `query_count` / `surrogate_skips`: Metriken für Surrogate-Benefit-Test.
  - `pending_genome_ids()`: ausstehende Genome für GUI-Poll.
  - Caching: bereits bewertete Genome werden nicht erneut angefragt.
  - Thread-sicher: `threading.Lock` + `threading.Event` für blockierenden GUI-Pfad.
- `NeuroEvolution.set_interactive_evaluation(evaluator, mode, surrogate_model, surrogate_update_interval)` → gibt `InteractiveEvaluator` zurück.
- `NeuroEvolution.submit_feedback(genome_id, value)` → delegiert an konfigurierten Evaluator.
- `InteractiveEvalPanel` (PySide6): Poll-Timer, zwei Genome nebeneinander, Rating-Slider, Pairwise-Buttons, Drag-to-Rank-Liste, Implicit-Dwell-Timer.
- `yane.InteractiveEvaluator` in `__init__.py` exportiert.
- 31 Tests in `tests/test_interactive_eval.py`, alle bestanden.

**Nicht implementiert** (bewusst weggelassen):
- GUI-Tests (PySide6 erfordert Display, keine headless-CI-Abdeckung)
- Gym-Render-Integration im GUI-Panel (kein direkter Mehrwert für Core-Logik)

---

### ✓ P1 Hardware-Aware NEAT (Deployment-Constraint-Evolution)

**Implementiert** in `evolution/hardware_aware.py`:
- `HardwareMetrics`: flops, memory_bytes, latency_us (alle deterministisch aus Topologie)
- `HardwareConstraints`: max_flops, max_memory_bytes, max_latency_us, target_platform, penalty_scale
- `PLATFORM_PROFILES`: cortex-m4, cortex-m7, esp32, raspberry-pi-zero, raspberry-pi-4, desktop, mobile-arm
- `compute_hardware_metrics()`: FLOPs = 2×n_conns + activation-FLOPs pro Node; Memory = n_nodes×8 + n_conns×8 (konfigurierbar); Latenz = FLOPs / (MHz/cycles_per_flop) × 1e6
- `compute_penalty()`: proportional zur Überschreitung × penalty_scale, summiert über alle verletzten Constraints
- `hw_pareto_front()`: nicht-dominierte Menge (maximise fitness, minimise cost), sortiert nach Fitness
- `NeuroEvolution.set_hardware_constraints()`, `.hardware_profile()`, `.hw_pareto_front()`
- Penalty in `_finalize_fitness()` integriert — fehlerresistent, bricht Training nie ab
- 29 Tests, alle bestanden.

---

### ✓ P1 ResourceBudget-System — Einheitliches Ressourcen-Management

**Implementiert** in `evolution/resource_budget.py`:

- `parse_time(s)`: `"30min"` → 1800s, `"1h"` → 3600s, `"45s"` → 45s, Zahl-Passthrough, None-safe.
- `parse_memory(s, available_bytes)`: `"4GB"` → 4e9, `"80%"` → 80 % von *available_bytes*, `"auto"` → 80 % freies RAM.
- `ResourceDiscovery`: CPU-Kerne, RAM total/verfügbar, GPU-Speicher (pynvml, None wenn fehlt), Disk, Battery — alle Sensoren wrapped, kein Crash bei fehlendem Gerät.
- `BudgetConfig`: Dataclass mit `total_time_seconds`, `max_memory_bytes`, `max_cpu_pct`, `target_platform`, `auto`.
- `GracefulDegradation`: 6 einstufige Eskalationen (monoton, kein Recovery):
  - Stufe 1 `reduce_pop`: `_population_size` und laufende Population halbieren.
  - Stufe 2 `skip_lamarck`: `steps=0`, `max_steps=0`.
  - Stufe 3 `simplify_topology`: `max_nodes * 0.75`.
  - Stufe 4 `disable_research`: curiosity/DARTS/shared_weights + FeatureGating abschalten.
  - Stufe 5 `reduce_eval_budget`: Anytime `max_evals=1`.
  - Stufe 6 `emergency_stop`: Checkpoint in `/tmp/yane_emergency_checkpoint.pkl` + `stop_requested=True`.
- `BudgetEnforcer(config, ne_ref, clock)`: Injectable Clock (Testbarkeit ohne Sleep), `start()`, `is_time_over()`, `elapsed_seconds()`, `check_memory(over_budget=None)`, `status()`.
- `NeuroEvolution.set_budget(preset, total_time, max_memory, max_cpu_pct, target_platform)`: `"auto"` → 80 % RAM; String-Limits werden geparst; propagiert `max_memory` an ResourceGuard.
- `NeuroEvolution.budget_status()`: gibt Budget-Snapshot-Dict zurück; leer wenn kein Budget.
- train()-Loop: Zeit-Check jede Iteration (billig); Memory-Check alle `_resource_check_interval` Iterationen; `stop_reason = "budget_exceeded"` bei Überschreitung.
- Alte APIs unverändert: `set_resource_limits()`, `set_efficiency_penalty()`.
- 53 Tests in `tests/test_resource_budget.py`, alle bestanden.

---

### ✓ P1 Evolutionary Data Augmentation

**Implementiert** in `evolution/augmentation.py`:
- `AugmentationGene(type, probability, magnitude)`: 5 Typen — `gaussian_noise`, `dropout_noise`, `scaling`, `translation`, `cutout`. Mutate (Gauss, clamped [0,1]), Serialisierung.
- `AugmentationPipeline`: geordnete Gene-Sequenz; `apply(inputs, rng)` transformiert Inputs in-series; Crossover (uniforme Gen-Ebene, Länge = max beider Eltern); UCB1-Score für Pool-Selektion.
- `AugmentationPool`: UCB1-Selektion; Evolution alle N Generationen (schlechteste Hälfte → Crossover+Mutation der besten Hälfte).
- Integration in `_run_evaluations()`: wraps `genome.forward` nach `input_transform`, vor Curiosity — Inputs werden vor dem Forward-Pass augmentiert.
- Reward = Fitness-Delta best-genome pro Generation; `set_evolutionary_augmentation()`, `get_augmentation_diagnostics()`.
- 33 Tests, alle bestanden.

---

### ✓ P2 Transfer Learning / Genome Fine-Tuning

Vollständig implementiert. `warm_start_from_checkpoint()`, `load_genome_as_seed()`,
`fine_tune_genome()`, `behaviour_clone()`, progressives Unfreeze und ein Benchmark-Demo
(`benchmarks/warm_start_demo.py`, CartPole→Acrobot) sind vorhanden. Tests in
`tests/test_neuro_evolution.py`.

- ✓ `yane.load_genome_as_seed(genome, freeze_layers=[...])`: bestimmte Verbindungsgruppen koennen eingefroren werden.
- ✓ Lamarck-Feinabstimmung auf neuer Aufgabe ohne Topologie-Aenderung als erste Phase.
- ✓ Dann schrittweise Entsperren eingefrorener Teile (progressive unfreeze).
- ✓ Benchmark: `benchmarks/warm_start_demo.py` (CartPole→Acrobot warm-start vs. cold-start).

---

### ⚡ P2 Offene Evolution / Co-Evolution von Aufgabe und Agent (POET-aehnlich)

**Aktueller Stand:** Experimenteller Spike in `evolution/experimental.py`.
`CoevolutionPool` und `CoevolutionPair` modellieren einfache Agent-/Environment-
Paare. Eine vollstaendige POET-aehnliche Integration mit `EnvironmentGenome`,
Archiv, Survival-Regeln, Transfer zwischen Aufgaben und Trainingsloop fehlt.

---

### ⚡ P2 Genome-Phylogenie (Stammbaum der Innovationen)

**Aktueller Stand:** Teilweise implementiert. Genome tragen IDs und Parent-IDs;
`InnovationTracker` kann Crossover, Innovationen und Ahnenketten erfassen.
Fitness-Delta-Attribution pro Innovation, Baum-/Graph-Analyse, Export und GUI-
Visualisierung fehlen noch.

---

### ⚡ P2 Verhaltensklonierung als Warm-Start

Evolution braucht viele Iterationen bis zu brauchbaren Loesungen; Demonstrationen koennen das beschleunigen.

**Aktueller Stand:** Teilweise implementiert. `behaviour_clone()` optimiert ein
Genom mit Demonstrationspaaren ueber eine einfache lokale Suche und gibt das
gekloente Genom zurueck. Es wird noch nicht automatisch als Population-Seed
verdrahtet; Backprop-/Torch-Training, Lamarck-Integration und Benchmarks fehlen.

- `yane.behaviour_clone(demonstrations, n_steps)`: supervised Vortraining des besten Genoms auf Demonstrations-Daten via Lamarck/Backprop.
- Demonstrationen als Liste von `(inputs, outputs)`-Paaren.
- Benchmark: BC-Warm-Start vs. random-init auf LunarLander.

---

### ✓ P2 Synaptische Plastizitaet (STDP / Hebbsches Lernen)

**Implementiert** in `evolution/stdp.py` + `core/connection.py`:

- `Connection.__slots__`: `hebb_a, hebb_b, hebb_c, hebb_d, _base_weight` hinzugefügt (alle 0.0/None default → Zero-Cost).
- Hebb-Regel: `Δw = A·pre + B·post + C·pre·post + D` (klassische 4-Koeffizienten-Form).
- `set_hebb_coeffs(genome, a,b,c,d, sigma)`: Initialisierung + Noise; `mutate_hebb_coeffs()`: NEAT-Mutation.
- `init_stdp_base_weights(genome)`: speichert `_base_weight` beim Evaluierungs-Start.
- `apply_stdp_update(genome, weight_min, weight_max)`: nach jedem `forward()`, nutzt `node.value` für pre/post — geclampt.
- `restore_stdp_weights(genome)`: setzt Gewichte auf Basiswerte zurück (episodenlokal).
- `genome.crossover()`: kopiert jetzt auch `hebb_a/b/c/d` für Matching-Genes (Bugfix).
- `NeuroEvolution.set_stdp(enabled, weight_min, weight_max, hebb_sigma)`: Wrapper in `_run_evaluations` setzt Input-`node.value` manuell (compiled forward path setzt nur Output-Knoten).
- 23 Tests in `tests/test_stdp.py`, alle bestanden.
- Benchmark (kein CI-Pfad): bewusst weggelassen.

---

### ✓ P2 Neuromodulation

**Implementiert** in `evolution/neuromodulation.py` + `core/node.py`:

- `Node.__slots__`: `is_modulator` (bool, default False) und `modulation_gain` (float, default 1.0) hinzugefügt.
- `node.copy()` überträgt `is_modulator` (Crossover/Checkpoint korrekt).
- `apply_modulation_to_weights(genome)`: multipliziert `conn.weight = _base_weight * target.modulation_gain` — One-step-delayed (Gain aus Call T wirkt auf T+1).
- `update_modulation_gains(genome)`: liest MODULATOR-`node.value`, schreibt `target.modulation_gain`.
- `restore_modulation_weights(genome)`: stellt Basisgewichte + neutrale Gains (1.0) wieder her.
- `make_node_modulator(genome, idx)`: markiert Knoten als MODULATOR.
- `mutate_modulator_flags(genome, add_prob, remove_prob)`: NEAT-Mutation für Modulator-Promotion/Demotion.
- `NeuroEvolution.set_neuromodulation(enabled)`: Wrapper in `_run_evaluations`.
- Kein neuer `NodeType.MODULATOR` (stattdessen `node.is_modulator` Flag — Zero Blast-Radius); dokumentiert.
- 22 Tests in `tests/test_neuromodulation.py`, alle bestanden.
- Benchmark (kein CI-Pfad): bewusst weggelassen.

---

### ✓ P2 Evolutionaere Input-Gruppierung (Evolvable Input Aggregation Layer)

**Implementiert** in `evolution/input_grouping.py`:

- `AggType` Enum: `MEAN`, `MAX`, `SUM`, `WEIGHTED_SUM`.
- `InputGroup(members, aggregation, weights, enabled)`: Dataclass mit `copy()`.
- `InputGrouper(n_raw, initial_groups)`: `transform(raw) -> list[float]`, `n_outputs` Property, alle 6 Mutations-Operatoren (`create_group`, `split_group`, `merge_groups`, `add_input_to_group`, `remove_input_from_group`, `change_aggregation`), `crossover(other)`, `copy()`.
- `apply_split_to_genome(genome, group_idx)`: splittet Gruppe und fuegt Input-Knoten hinzu.
- `apply_merge_to_genome(genome, idx_a, idx_b)`: merged Gruppen und entfernt Input-Knoten.
- Genome: `genome.grouper` Feld (None by default, geclonet in `copy()`, gecrossoverd in `crossover()`, in `__setstate__` backward-kompatibel).
- `NeuroEvolution.set_input_grouping(enabled, n_groups, n_raw, initial_groups)`: aktiviert Gruppierung; `configure()` erzeugt automatisch initiale Gruppen.
- `NeuroEvolution.get_input_grouping_diagnostics()`.
- Integration in `_run_evaluations()`: wraps `genome.forward` mit `grouper.transform()` — Zero-Cost-When-Disabled.
- 39 Tests in `tests/test_input_grouping.py`, alle bestanden.

---

### ✓ P2 Evolutionaere Output-Gruppierung (Evolvable Output Synergy Layer)

**Implementiert** in `evolution/output_grouping.py`:

- `ExpType` Enum: `COPY`, `SCALE`, `AFFINE`.
- `OutputGroup(targets, expansion, weights, enabled)`: Dataclass mit `copy()`.
- `OutputGrouper(n_outputs, initial_groups)`: `expand(proto) -> list[float]` (immer len=n_outputs), `n_proto` Property, alle 6 Mutations-Operatoren (`create_group`, `split_group`, `merge_groups`, `add_output_to_group`, `remove_output_from_group`, `change_expansion`), `crossover(other)`, `copy()`, `to_python_expand_block()`.
- `apply_split_to_genome(genome, group_idx)`: splittet Gruppe und fuegt Output-Knoten hinzu.
- `apply_merge_to_genome(genome, idx_a, idx_b)`: merged Gruppen und entfernt Output-Knoten.
- Genome: `genome.out_grouper` Feld (None by default, geclonet/gecrossoverd/backward-kompatibel).
- `NeuroEvolution.set_output_grouping(enabled, n_proto, n_outputs, initial_groups)`.
- `NeuroEvolution.get_output_grouping_diagnostics()`.
- Integration in `_run_evaluations()`: wraps `genome.forward` mit `out_grouper.expand()`.
- `genome_to_python()`: erzeugt korrekten `_ext`-Expand-Block wenn `out_grouper` gesetzt.
- 41 Tests in `tests/test_output_grouping.py`, alle bestanden.

---

### ✓ P2 Convolutional NEAT (CoDeepNEAT-inspiriert)

**Implementiert** in `evolution/conv_neat.py` als Wrapper-Layer (kein `NodeType.CONV2D` — bewusst, um die Core-Forward-Path unberührt zu lassen; Zero-Cost-When-Disabled):

- `ConvBlock(kernel_size, stride, in_channels, out_channels, activation)`: weight-shared Kernel (K²·in_c Gewichte/Kanal, unabhängig von Bildgröße); `forward(planes, h, w) -> list[float]` mit Global-Average-Pool; `mutate()`, `crossover()`, `copy()`.
- `ConvStack(blocks)`: geordnete Block-Sequenz; `forward_image(pixels, h, w, c) -> list[float]` (Länge = `n_outputs`); `n_outputs` = Summe der `out_channels` — konstant, unabhängig von Bildgröße; `crossover()`, `copy()`.
- `add_conv_block(stack, ...)`: fügt Block hinzu; `in_channels` automatisch aus Vorgänger; `mutate_conv_stack()`.
- `make_conv_stack(n_image_channels, n_blocks, kernel_size, out_channels, activation)`: Factory.
- Genome: `genome.conv_stack` Feld (None by default), `genome.forward_image(pixels, h, w, c)`.
- `NeuroEvolution.set_conv_neat(...)`, `conv_n_inputs()` → liefert `n_inputs` für `configure()`.
- 33 Tests in `tests/test_conv_neat.py`, alle bestanden.

**Nicht implementiert** (bewusst): `NodeType.CONV2D` (zu hohe Blast-Radius), MNIST-Benchmark (kein CI-Path).

---

### ✓ P2 ES-HyperNEAT (Evolvable Substrate HyperNEAT)

**Implementiert** in `evolution/indirect_encoding.py`:

- `_cppn_variance(weight_fn, x, y, eps)`: lokale CPPN-Ausgabevarianz via 4 Nachbar-Proben.
- `_quadtree_place_nodes(...)`: rekursiver Quadtree-Algorithmus — unterteilt Regionen bei Varianz > Schwellwert; platziert Knoten an Blättern mit ausreichender Varianz; bounded by `max_depth`.
- `es_hyperneat_substrate(cppn, n_inputs, n_outputs, variance_threshold, max_depth, initial_resolution, x_range, y_range)`: vollständige ES-HyperNEAT-Pipeline; Fallback (1 Mittelknoten) bei uniformem CPPN; Deduplizierung; gültige Paare.
- `hyperneat_substrate(evolve=True, cppn=..., ...)`: erweitertes API — ruft ES-HyperNEAT intern auf; `evolve=False` verhält sich identisch wie bisher.
- `generate_genome_from_cppn(..., evolve_substrate=True, es_variance_threshold, es_max_depth, es_initial_resolution)`: erweitertes API.
- `yane.es_hyperneat_substrate` exportiert.
- 23 Tests in `tests/test_es_hyperneat.py`, alle bestanden.

**Nicht implementiert** (kein CI-Pfad): Benchmark auf 2D-Navigations-Task und MNIST.

---

### ✓ P2 Genome-to-ONNX-Export (Produktions-Deployment)

**Implementiert** in `evolution/onnx_export.py`:

- `_activation_nodes(act_name, input, output)`: 15 Aktivierungsfunktionen → ONNX-Ops (linear, sigmoid, tanh, relu, leaky_relu, elu, swish, softplus, sine, cosine, abs, gaussian, binary, square, cube); unbekannte fallen auf Identity zurück.
- `genome_to_onnx(genome, path, opset_version, unroll_steps)`: acyclische Genome → direkter ONNX-Graph (topologische Reihenfolge, Slice/Flatten für Inputs, chained Add für Gewichtssummen, Bias + Aktivierung); zyklische Genome → Zeitunrolling für `unroll_steps` Schritte (Memory-Nodes ausgehend von 0 initialisiert).
- `NeuroEvolution.export_genome_onnx(path, opset_version, unroll_steps)`.
- `yane.genome_to_onnx` exportiert.
- Fehlt `onnx`-Paket: klarer `ImportError` mit `pip install onnx`-Hinweis.
- 15 Tests in `tests/test_onnx_export.py`: 3 pass (strukturell, ohne onnx), 12 skip wenn `onnx` nicht installiert.

---

### ✓ P2 Population Distillation (Ensemble → kompaktes Einzelgenom)

**Implementiert** in `evolution/distillation.py`:

- `_make_student(n_inputs, n_outputs, target_nodes, rng)`: kompaktes, voll verbundenes Student-Genom.
- `_teacher_outputs(teachers, probes)`: gemittelte Ensemble-Ausgabe als Supervisions-Signal.
- `_hill_climb_mse(student, teacher_outputs, probe_inputs, n_steps, sigma, ...)`: Hill-Climbing gegen MSE; akkumuliert Verlauf mit `log_interval`; nimmt nur Verbesserungen an → monotone Garantie.
- `distill_ensemble(teachers, student, probe_inputs, distillation_steps, sigma, sigma_decay, ...)`: standalone-Funktion; generiert Probe-Inputs wenn nicht angegeben; optional Sigma-Annealing.
- `DistillationResult`: `student`, `final_loss`, `initial_loss`, `loss_history`, `compression_ratio`, `loss_is_monotone`-Property.
- `NeuroEvolution.distill_ensemble(k, target_nodes, distillation_steps, ...)`: verwendet `population.get_top(k)`.
- `yane.distill_ensemble`, `yane.DistillationResult` exportiert.
- 23 Tests in `tests/test_distillation.py`, alle bestanden.

---

### ✓ P2 Gradient-NEAT-Hybrid-Modus (Backprop + Evolution interleaved)

**Implementiert** in `evolution/hybrid_neat.py`:

- `ReplayBuffer(max_size)`: Circular-Buffer (FIFO, `deque`); `add(inputs)`, `sample(n, rng)`, `clear()`.
- `genome_to_trainable_module(genome)`: wie `genome_to_torch_module()`, aber W und b als `nn.Parameter` (Gradienten möglich); raises `ImportError` ohne PyTorch.
- `sync_weights_back(genome, module)`: schreibt W/b aus Modul zurück in `conn.weight` / `node.bias`.
- `run_hybrid_backprop(genomes, inputs_batch, targets_batch, bp_epochs, bp_lr, bp_batch_size)`: Adam-Optimierung gegen MSE; sync-back nach jeder Genome; raises `ImportError` ohne PyTorch.
- `NeuroEvolution.set_hybrid_mode(enabled, bp_interval, bp_epochs, bp_lr, bp_batch_size, top_k, train_data, replay_buffer_size)`: konfiguriert Hybrid; kein PyTorch-Import beim Setup.
- Train-Loop-Hook: every `bp_interval` Generationen → `_run_hybrid_backprop()`.
- Replay-Buffer-Befüllung: automatisch durch fitness_fn-Wrapper der Inputs aufzeichnet.
- Ohne PyTorch: `set_hybrid_mode()` kein Crash; `run_hybrid_backprop()` klarer `ImportError`.
- 28 Tests in `tests/test_hybrid_neat.py`: 19 pass (strukturell), 9 skip wenn `torch` fehlt.

---

### ✓ P2 WebAssembly-Export (Browser-Deployment)

**Implementiert** in `evolution/wasm_export.py`:

- `_js_activation(act_name, expr)`: 15 Aktivierungsfunktionen → JS-Ausdrücke (Math.exp, Math.tanh, Math.max, etc.); unbekannte → Identity.
- `genome_to_js(genome, function_name, unroll_steps)`: Pure-JS-Transpilation acyclischer Genome (topologische Reihenfolge, identisch zu `genome_to_python()`); zyklische Genome → time-unrolled. Auto-detect cyclic wenn `_build_exec_order()` None zurückgibt.
- `genome_to_html(genome, path, title, function_name, unroll_steps, mode)`: Standalone HTML mit `<script>` (interaktives UI mit Input-Feldern), `mode="js"` → pure JS; `mode="wasm"` → klarer `ImportError` (Emscripten nicht verfügbar).
- `NeuroEvolution.export_genome_wasm(path, title, mode, unroll_steps)`.
- `yane.genome_to_js`, `yane.genome_to_html` exportiert.
- 19 Tests in `tests/test_wasm_export.py`: alle bestanden, inkl. Python↔JS-Vergleich via Node.js v24 (numerisch bis 1e-5).

**Nicht implementiert** (kein Emscripten): `mode="wasm"` → `ImportError` mit klarer Anweisung.

---

### ✓ P2 Evolvable Attention Heads (Transformer-inspirierte Architektursuche)

**Implementiert** in `evolution/attention.py` als Wrapper-Layer (kein `NodeType.ATTENTION` — bewusst, da NEAT keinen natürlichen Sequenz-Begriff hat):

- `_softmax(values)`: numerisch stabiles Softmax.
- `AttentionBlock(n_inputs, head_dim, num_heads, seed)`: W_Q/W_K/W_V-Matrizen [num_heads × head_dim × n_inputs]; `forward(inputs) -> list[float]` (Länge = `n_outputs = num_heads * head_dim`); Scaled Dot-Product Attention pro Head; `mutate(sigma)`, `crossover(other)` (per-Head, von `self.copy()` ausgehend), `copy()`.
- `genome.attention_block` Feld (None by default; copy/crossover/setstate backward-kompatibel).
- `NeuroEvolution.set_attention(enabled, head_dim, num_heads, n_inputs)`, `attention_n_inputs()`.
- Integration in `_run_evaluations()`: wraps `genome.forward` mit `attention_block.forward()`.
- 26 Tests in `tests/test_attention.py`, alle bestanden.

**Nicht implementiert** (kein NodeType-Enum-Eintrag, Wrapper-Ansatz): `NodeType.ATTENTION`; `genome_to_torch_module()`-Mapping.

---

### ✓ P2 Liquid Time-Constant (LTC) Nodes (ODE-basierte Neuronen)

**Implementiert** in `evolution/ltc.py` + `core/node.py`:

- `Node.__slots__`: `tau` (float, default `inf`) und `dt` (float, default 0.01).
- `node.copy()` überträgt `tau` und `dt`.
- `_SLOT_DEFAULTS` + Pickle/Checkpoint backward-kompatibel.
- `apply_ltc_update(genome)`: `x_{t+1} = x_t + dt*(-x_t/τ + node.value)` für alle Knoten mit `tau < inf`; geclampt; Standard-Knoten (tau=inf) werden übersprungen (Zero-Cost).
- `make_node_ltc(genome, idx, tau, dt)`: markiert Knoten, setzt `persist_value=True`.
- `mutate_ltc_params(genome, tau_sigma, dt_sigma)`: log-normale tau-Mutation, dt-Perturbation.
- `NeuroEvolution.set_ltc(enabled)`: Wrapper in `_run_evaluations`.
- τ-Extremwerte: τ→∞ langsam akkumulierender State, τ→0 instantane Antwort; beides getestet.
- 22 Tests in `tests/test_ltc.py`, alle bestanden.

---

### ✓ P2 Temporal Speciation (Verhaltensbasierte Spezies-Bildung)

**Implementiert** in `evolution/compatibility.py`:

- `_dtw(traj1, traj2)`: Dynamic Time Warping (DP, O(n·m)), Euclidean pointwise; identische Trajektorien → 0; verschiedene Längen unterstützt; symmetrisch.
- `_topology_hash(genome)`: schneller Fingerabdruck über Knoten/Connections/Gewichte für Cache-Invalidierung.
- `TemporalDistance(n_rollouts, rollout_len, time_weight, seed)`: Trajectory-Cache (dict[hash → traj]); `_cache_hits`/`_cache_misses` Metriken; `invalidate_cache()`; feste RNG-Sequenz für faire Genome-Vergleiche; `time_weight` skaliert DTW-Ergebnis.
- Implementiert `DistanceMetric`-Protokoll: `__call__(g1, g2) -> float`.
- Kombinierbar mit `ChainMetric`: `ChainMetric([TopologyDistance(), TemporalDistance()], weights=[0.5, 0.5])`.
- `yane.TemporalDistance`, `yane.ChainMetric`, `yane.TopologyDistance`, `yane.WeightDistance` exportiert.
- 24 Tests in `tests/test_temporal_speciation.py`, alle bestanden.

---

### □ P2 Self-Play / Adversarial Populations (Kompetitive Co-Evolution)

**Ziel:** `set_adversarial_populations()` teilt Population in gegnerische Sub-Populationen.

**Implementiert** in `evolution/self_play.py`:

- `EloRating` aus `interactive_eval.py` wiederverwendet (Single Source of Truth, Zero-Sum-Garantie durch gemeinsames Rating über alle Populationen).
- `AdversarialSystem(n_populations, pairing, n_matches, elo_k, seed)`: Pairing-Strategien `"round_robin"`, `"random"`, `"best_vs_rest"`; `apply_game_result()` + `apply_zero_sum_batch()`; `record_elo_snapshot()`; `arms_race_indicator` Property.
- `AdversarialResult(populations, elo_histories, n_generations)`: `arms_race_indicator` Property; `best_genome(pop_id)`.
- `train_adversarial(populations, game_fn, mutation_fn, n_generations, ...)`: Standalone-Evolutionsloop mit Elitismus + Mutation.
- `NeuroEvolution.set_adversarial_populations(...)`, `train_adversarial(game_fn, n_generations, pop_size)`.
- `yane.AdversarialSystem`, `yane.AdversarialResult`, `yane.train_adversarial` exportiert.
- 24 Tests in `tests/test_self_play.py`, alle bestanden.

---

### □ P2 Hierarchical NEAT (H-NEAT) — Mehrstufige Policy-Architektur

**Ziel:** Zweistufiges System: Manager-Genom waehlt Sub-Policy, Worker-Genome fuehren aus.

```
High-Level Manager → Sub-Policy Pool (N Low-Level Genome)
```

**Implementiert** in `evolution/h_neat.py`:

- `HierarchicalGenome(manager, workers, selection_mode)`: `forward(inputs)` — Manager wählt/gewichtet Worker via Softmax; `selection_mode="hard"` (argmax → ein Worker) oder `"soft"` (gewichtete Summe aller Worker-Ausgaben).
- `last_selected_idx`: welcher Worker zuletzt ausgewählt wurde.
- `selection_distribution(inputs_list)`: wie oft jeder Worker gewählt wird (Zustandsabhängigkeit).
- Mutations-Operatoren: `add_sub_policy(worker)`, `split_sub_policy(idx, rng)` (mutierter Clone), `merge_sub_policies(idx_a, idx_b)` (Gewichtsmittelung + Entfernen).
- `save(path)` / `HierarchicalGenome.load(path)`: Pickle-Checkpoint der kompletten Hierarchie.
- `copy()`: tiefe Kopie.
- `NeuroEvolution.configure_hierarchical(n_workers, selection_mode)`: erzeugt Manager + Workers aus aktueller Konfiguration.
- `yane.HierarchicalGenome` exportiert.
- 26 Tests in `tests/test_h_neat.py`, alle bestanden.

---

### ✓ P2 Gene Regulatory Network (GRN) Encoding

**Implementiert** in `evolution/grn_encoding.py`:

- `GRNGene(src_node, tgt_node, weight, activation, regulatory_sites, expression_rate)`: `copy()`, `mutate(sigma)`.
- `GRNGenome(genes)`: `develop(n_inputs, n_outputs, development_steps)` → Phänotyp-Genome; Entwicklungsalgorithmus: 1 Initialisierungsschritt + N reguläre Schritte → 20 Gene × (5+1) = 120 > 100 Connections.  Regulierungslogik: Gen ohne Regulatoren = konstitutiv; Gen mit Regulatoren = aktiv wenn mind. ein Regulator im Vorschritt aktiv war. `random(n_genes, n_nodes, regulatory_prob, seed)`, `copy()`, `mutate()`, `crossover(other)` (Gen-Index-Alignment).
- `GRNCodec(n_inputs, n_outputs, development_steps)`: implementiert `GenomeCodec`-Protokoll; `encode(grn)` → pickle; `decode(data)` → GRNGenome; `develop(grn)` → Phänotyp.
- `NeuroEvolution.set_genome_encoding("grn", development_steps, n_genes, n_nodes, seed)`, `develop_grn(grn)`.
- `yane.GRNGene`, `yane.GRNGenome`, `yane.GRNCodec` exportiert.
- 24 Tests in `tests/test_grn_encoding.py`, alle bestanden.

---

### ✓ P2 Developmental NEAT — Ontogenese waehrend Evaluation

**Implementiert** in `evolution/developmental.py` + `core/genome.py`:

- `DevelopmentalRule(trigger_condition, action, max_fires)`: abstrakte Regel; `should_fire(genome)`, `fire(genome)`, `reset_episode()`, `copy()`, `mutate()`.
- `ParametricRule(trigger_node_idx, threshold, trigger_mode, src_idx, tgt_idx, weight, max_fires)`: konkrete, evolvierbare Regel; Trigger: `node.value >= threshold` oder `<= threshold`; Aktion: fügt ephemere Verbindung hinzu.
- `make_threshold_rule(...)`: Factory-Funktion.
- `genome.dev_rules: list[DevelopmentalRule]` (leer = Zero-Cost).
- `genome._dev_added`: ephemere Verbindungen dieser Episode (in `__getstate__` ausgeschlossen).
- `genome._dev_frozen: bool`.
- `genome.developmental_forward(inputs)`: forward() + Regelauswertung.
- `genome.freeze_development()`: deaktiviert alle Regeln.
- `genome.reset()`: entfernt `_dev_added`-Verbindungen, setzt Regel-Zähler zurück.
- `genome.copy()` / `crossover()`: erben `dev_rules` (mit Gen-Alignment).
- `mutate_rules(genome, weight_sigma, threshold_sigma)`.
- `yane.DevelopmentalRule`, `yane.ParametricRule`, `yane.make_threshold_rule` exportiert.
- 24 Tests in `tests/test_developmental.py`, alle bestanden.

---

### ✓ P2 Continual / Lifelong Learning NEAT

**Implementiert** in `evolution/continual.py`:

- `TaskAnchor(name, best_genome, best_fitness, anchor_weights)`: speichert Ankerwerte nach Aufgabe.
- `compute_ewc_penalty(genome, anchors, lambda_ewc)`: `(λ/2) * Σ (w_i - w*_i)²`.
- `make_ewc_fitness(base_fn, anchors, lambda_ewc)`: wraps Fitness mit EWC-Penalty.
- `freeze_genome_weights(genome)`: setzt `spike_rate=0` auf alle Verbindungen (progressiv einfrieren).
- `progressive_expand(genome, n_new_nodes)`: fügt neue HIDDEN-Knoten + Verbindungen hinzu.
- `TaskMemory(name, max_size)`: Replay-Buffer für (Eingabe, Ausgabe)-Paare.
- `make_replay_fitness(base_fn, memories, replay_weight, replay_samples)`: MSE-Penalty auf alten Beispielen.
- `ContinualLearner(mode, lambda_ewc, replay_weight, ...)`: orchestriert alle 4 Modi (ewc, progressive, replay, hybrid); `start_task()`, `wrap_fitness()`, `finish_task()`, `forgetting_rate()`.
- `NeuroEvolution.set_continual_learning()`, `task_start()`, `task_finish(evaluator, sample_inputs)`, `evaluate_all_tasks()`.
- train()-Hook: `fitness_fn = learner.wrap_fitness(fitness_fn)` bei jedem Aufruf.
- `yane.ContinualLearner`, `yane.TaskAnchor`, `yane.TaskMemory`, `yane.compute_ewc_penalty` exportiert.
- 28 Tests in `tests/test_continual.py`, alle bestanden.

---

### ✓ P2 Meta-Learning NEAT — Few-Shot Adaptation

**Implementiert** in `evolution/meta_learning.py`:

- `MetaLearner(adaptation_steps, lamarck_sigma, track_deltas)`: Inner-Loop via `LamarckRefiner.refine()` (wiederverwendet); `compute_meta_fitness(genome, task_sampler)` evaluiert Pre-Fitness → Lamarck → Post-Fitness; `make_fitness_fn(task_sampler)` für NEAT-Outer-Loop; `adaptation_deltas` Liste.
- `MetaTrainResult(best_genome, best_meta_fitness, adaptation_deltas, meta_iterations)`: `mean_adaptation_delta` Property.
- `NeuroEvolution.meta_train(task_sampler, meta_iterations, adaptation_steps, lamarck_sigma)`: wraps NEAT train() mit meta_fn; gibt `MetaTrainResult` zurück.
- `make_fitness_fn()` nutzt `genome.copy()` → Originalgenom bleibt unverändert.
- Hill-Climbing-Garantie: Adaptation-Delta ≥ 0 immer (Lamarck akzeptiert nur Verbesserungen).
- `yane.MetaLearner`, `yane.MetaTrainResult` exportiert.
- 13 Tests in `tests/test_meta_learning.py`, alle bestanden.

---

### □ P2 Evolutionary Reservoir Computing

**Implementiert** in `evolution/reservoir.py`:

- `ReservoirGenome(n_inputs, n_reservoir, n_outputs, spectral_radius, input_scaling, leaking_rate, seed)`: Leaky ESN-Dynamik `x = (1-α)*x + α*tanh(W*x + W_in*u + b)`; `forward(inputs)`, `reset()`, `collect_states(inputs_sequence, washout)`.
- `actual_spectral_radius` Property; Power-Iteration-Approximation.
- `mutate_readout(sigma)`, `copy()`, `save(path)`, `load(path)`.
- `train_ridge_readout(reservoir, inputs, targets, lambda_ridge, washout)`: analytische Lösung W_out = Y·X^T·(X·X^T + λI)^{-1} via Gauß-Elimination; gibt `ReservoirTrainResult(reservoir, train_mse, n_samples)` zurück.
- `NeuroEvolution.configure_reservoir(n_reservoir, spectral_radius, input_scaling, leaking_rate, n_inputs, n_outputs, seed)`, `train_reservoir(inputs, targets, ...)`.
- `yane.ReservoirGenome`, `yane.ReservoirTrainResult`, `yane.train_ridge_readout` exportiert.
- 20 Tests in `tests/test_reservoir.py`: Determinismus; Echo-State-Property (SR<1, State bounded); Ridge löst linearen Task; Mutation; Checkpoint.

---

### □ P2 Open-Ended Evolution / Minimal Criterion

**Ziel:** `set_minimal_criterion(fn)` filtert Genome vor Selektion.

**Implementiert** in `evolution/minimal_criterion.py`:

- `MinimalCriterion(criterion_fn, min_viable_frac, penalty, viable_boost_factor)`: `apply(genome, base_fitness)` → `base_fitness` wenn viable, `penalty` wenn nicht; `wrap_fitness(base_fn)` setzt `genome.fitness = base` vor Criterion-Check; `reset_generation()`, `viable_fraction_history()`.
- Adaptive Lockerung: wenn `viable_frac < min_viable_frac` → `penalty *= viable_boost_factor` (weniger streng); `_relaxation_active` Flag.
- `make_novelty_with_criterion()`, `make_curiosity_with_criterion()`, `make_qd_with_criterion()`: kombinieren bestehende YANE-Features mit Minimal Criterion.
- `NeuroEvolution.set_minimal_criterion(criterion_fn, min_viable_frac, penalty, viable_boost_factor)`, `set_open_ended(mode, archive_size)`.
- train()-Hook: `fitness_fn = mc.wrap_fitness(fitness_fn)` bei jedem Aufruf.
- `yane.MinimalCriterion` exportiert.
- 20 Tests in `tests/test_minimal_criterion.py`, alle bestanden.

---

### □ P2 Multi-Agent Cooperation (kooperativ, nicht adversarial)

**Implementiert** in `evolution/cooperative.py`:

- `assign_shared/difference/individual/hierarchical()`: 4 Credit-Assignment-Funktionen; `difference` = Shapley-Approx: `credit_i = f(team) - f(team_without_i)`.
- `compute_role_similarity(agents, probe_inputs)`: durchschnittliche paarweise Kosinus-Ähnlichkeit der Agent-Ausgaben; `role_diversity_penalty(similarity, weight)`.
- `CooperativeSystem(n_agents, credit, role_specialization, diversity_weight)`: `evaluate_team(agents, team_fitness_fn, probe_inputs)` → setzt `agent.fitness` für alle Agenten; `role_similarity_history`, `team_fitness_history`.
- `train_cooperative(agents, team_fitness_fn, ...)`: Evolutions-Loop mit Credit-Assignment + Elitismus.
- `CooperativeResult(agents, team_fitness_history, role_similarity_history, n_generations)`: `best_agent`, `mean_final_fitness`.
- `NeuroEvolution.set_cooperative_population(...)`, `train_cooperative(...)`.
- `yane.CooperativeSystem`, `yane.CooperativeResult`, `yane.train_cooperative` exportiert.
- 21 Tests in `tests/test_cooperative.py`, alle bestanden.

---

### □ P2 Probabilistic / Bayesian NEAT

**Ziel:** `NodeType.PROBABILISTIC` mit dualem Output (μ, σ).

- Forward: `output = mu + sigma * epsilon` (Reparameterization).
- `genome.bayesian_forward(n=100)`: gibt `(mean, std)` zurueck.
- `NeuroEvolution.set_probabilistic(enabled=True, n_samples=5)`.

**Akzeptanzkriterien:**

- `bayesian_forward(n=100)` reduziert Output-Varianz vs. n=1.
- `inference_mode=True`: deterministischer Output.
- Tests: Reparameterization-Sampling; Uncertainty-Kalibrierung; Crossover.

---

### □ P2 Safety-Constrained Evolution (Safe NEAT)

**Ziel:** `set_safety_constraints(constraints)` — unverletzbare Regeln fuer sicherheitskritische Anwendungen.

```python
yane.set_safety_constraints([
    SafetyConstraint("joint_limit",
                     check=lambda state, action: max(action) < 1.0,
                     mode="hard"),
])
```

- `mode="hard"`: Evaluation sofort abbrechen, Fitness = −∞.
- `mode="soft"`: Fitness-Reduktion proportional zur Verletzungshaeufigkeit.
- `mode="barrier"`: logarithmische Barriere-Funktion.

**Akzeptanzkriterien:**

- Hard-Constraint-Verletzung → Fitness = penalty, keine weiteren Schritte.
- `min_safe_frac`-Mechanismus schuetzt sichere Genome vor Verdraengung.
- Tests: Hard-Constraint-Abbruch; Soft-Constraint-Penalty; Safe-Fraction-Schutz.

---

### □ P2 Sparse NEAT / Lottery Ticket Hypothesis

**Ziel:** `genome.find_lottery_ticket()` identifiziert sparsestes Subnetzwerk.

```python
ticket = genome.find_lottery_ticket(target_sparsity=0.1,
                                     max_fitness_drop=0.01,
                                     iterations=5)
genome.apply_ticket(ticket)
```

**Algorithmus (Iterative Magnitude Pruning):** Trainiere → Pruning p% → Fine-tune → Wiederholen.

**Akzeptanzkriterien:**

- Ticket-Fitness ≥ Original − `max_fitness_drop`.
- IMP findet sparsere Tickets als Random (p < 0.05 ueber 10 Runs).
- Tests: IMP-Iteration; Ticket-Sparsity; Fitness-Drop-Check; Ticket-Serialisierung.

---

### □ P2 Genome-to-TFLite / Embedded Export

**Ziel:** `genome.export_tflite("model.tflite", quantization="int8")`.

**Export-Pipeline:** `genome_to_torch_module()` → `torch.onnx.export()` → TFLite.
Alternativ: direkte TFLite-Modellkonstruktion via `tflite.Model`-API.

**Quantisierungs-Modi:** `"fp32"`, `"fp16"`, `"int8"`, `"hybrid"`.
Output-Formate: `model.tflite`, `model.cc` (C-Array), `model.h`.

**Akzeptanzkriterien:**

- XOR: TFLite-Inferenz innerhalb 1e-3 (fp32).
- INT8: Outputs innerhalb 5% relativen Fehlers.
- `model.cc` kompilierbar (syntaktisch korrektes C-Array).
- Tests: TFLite-Export; Quantisierungs-Vergleich; C-Array-Export.

---

### □ P2 Symbolic Regression Export (Genom → mathematische Formel)

**Ziel:** `genome.to_symbolic(format="latex")` — analytische, menschenlesbare Formel.

```python
formula = genome.to_symbolic(input_names=["x", "y"], format="latex")
# f(x,y) = sin(0.5*x - 0.3*y) + tanh(0.8*x + 0.2)
```

**Ausgabe-Formate:** `"latex"`, `"python"`, `"sympy"`, `"text"`.
**Konstanten-Faltung:** `0.0 * x` → entfernt, `1.0 * x` → `x`.

**Akzeptanzkriterien:**

- Symbolic-Output evaluiert zu identischen Werten wie `genome.forward()` (Toleranz 1e-6).
- LaTeX-Output ist kompilierbar.
- Tests: Format-Korrektheit; Output-Aequivalenz; Konstanten-Faltung; CSE-Reduktion.

---

## Erledigte Tasks (Zusammenfassung)

Vollstaendig abgeschlossene Tasks. Fuer Details siehe Git-History und Tests.

---

### ✓ P0 Strukturierte Evaluator-Komponenten / Task-Curriculum

**Abgeschlossen:** `EvaluatorSpec` mit `enabled_components` erlaubt Ablations-Vergleiche.
CliffWalking, FrozenLake und Taxi nutzen `EvaluatorSpec.combine()` mit
`rollout_score`, `oracle_score`, `subgoal_score`, `policy_score`. GUI-Komponenten-
Schalter. Diagnostics schreiben Komponentenwerte via `_component_scores`.
Tests: Ablations-Tests und MultiStartRollout-Tests in `test_p0_features.py`.

### ✓ P0 Self-Tuning Speziation (Automatischer Kompatibilitaetsschwellenwert)

**Abgeschlossen:** `set_target_species(n_min=4, n_max=8, tune_interval=10)` — PI-Regler
passt Schwellenwert alle N Schritte an Zielband an. `set_target_species(None)` deaktiviert.
Diagnostics: aktueller Schwellenwert, letzter Anpassungsschritt, Trend.

### ✓ P0 Adaptive Recovery System

**Abgeschlossen:** `set_adaptive_recovery(enabled, strategies, cooldown, escalate, ...)`.
Recovery via Diversity-Injektion, Partial-Restart, Lamarck-Burst. Eskalation,
Cooldown, konservatives Early Stopping (Patience + Diversity-Bedingung), persistente
`stopped_early`/`stop_reason`-Attribute. Stepping-Stone-Schutz via `warmup`.

### ✓ P0 Anytime-Evaluation (Adaptives Evaluations-Budget)

**Abgeschlossen:** `set_anytime_eval(enabled, min_evals, max_evals, promotion_frac, aggregation)`.
Phase 1: alle Genome minimal evaluieren. Phase 2: obere `promotion_frac` bis `max_evals`.
Diagnostics: gesparte Evals, Promotion-Rate, Fitness-Varianz promoteter Genome.

---

### ✓ P1 Adaptive Policy System

**Abgeschlossen:** `PolicyRegistry` mit `register()`, `set_order()`, `tick()` (observe → decide → apply).
`Action`-Typ mit `priority`/`conflict_group`. Konfliktaufloesung: hoehere Priority gewinnt
pro conflict_group. `register_policy()`/`set_policy_order()`/`get_policy_diagnostics()` API. 7 Tests.

### ✓ P1 Modular Compatibility Distance / Genome Descriptor

**Abgeschlossen:** `DistanceMetric`-Protokoll, `TopologyDistance`, `WeightDistance`,
`ActivationDistance`, `ChainMetric`. `set_compatibility_distance()` API.
Tests in `test_diagnostics_features.py`.

### ✓ P1 Experiment Tracking / Run Database

**Abgeschlossen:** `Run`, `Experiment`, `RunDatabase` in `util/run_database.py`.
SQLite-Persistenz, automatische Run-Erfassung via `set_run_database()`,
`experiment()`-Kontext, `start_run()`/`finish_run()`, `load_run()`, `list_runs()`,
`reproduce_run()`. `RunRecord` als CSV-/History-Layer erhalten.

### ✓ P1 Selektionsstrategie als Plugin

**Abgeschlossen:** `SelectionStrategy`-Protokoll in `evolution/selection_strategy.py`.
Eingebaute Strategien: `TournamentSelection`, `ElitistSelection`,
`FitnessProportionalSelection`, `RankSelection`, `NoveltyOnlySelection`.
`set_selection_strategy()` API, per-Species-Override, Diagnostics.

### ✓ P1 Evaluation-Middleware-Stack

**Abgeschlossen:** `EvalMiddleware`-Protokoll in `evolution/eval_middleware.py`.
`CachingMiddleware`, `TimingMiddleware`, `RetryMiddleware`, `NoiseMiddleware`,
`ComponentMiddleware`, `CaseBatchMiddleware`. `add_eval_middleware()` API, LIFO-Reihenfolge.
Diagnostics: Cache-Hit-Rate, Eval-Zeit, Retry-Rate, Komponenten-Rohwerte.

### ✓ P1 Generationsreport / Run-Postmortem

**Abgeschlossen:** `util/report.py` mit `export_run_report()`. Formate: HTML
(self-contained, inline SVG), JSON, Markdown. SVG-Fitnesskurve, Recovery-Events-Tabelle,
Beste-Genom-Topologie, Config-Dump. `set_report_autosave()` API. 11 Tests.

### ✓ P1 Automatisches Checkpoint-Rolling (Retention-Policy und Best-Tracking)

**Abgeschlossen:** `set_checkpoint_policy(interval, keep_best, max_keep, path_template)`.
Rolling-Checkpoints waehrend `train()`, Retention entfernt alte Dateien,
`get_best_checkpoint_path()` API. Diagnostics: letzter Auto-Save, Checkpoint-Anzahl.

### ✓ P1 Generationsanzeige in der GUI

**Abgeschlossen:** `population_memory_info()` enthaelt `generation` (= iteration // pop_size).
GUI zeigt "Generation" und separate "Evaluations". CSV/JSONL enthalten `generation`-Spalte.

### ✓ P1 Hybrid Feature-Extractor API (Input-Vorverarbeitung)

**Abgeschlossen:** `set_input_transform(fn)` mit Dimensionalitaets-Validierung,
Integration in `_run_evaluations()` und `_run_with_matrix_forward()`. 6 Tests in
`test_diagnostics_features.py`.

### ✓ P1 Fitness-Landscape-Visualisierung (PCA / t-SNE)

**Abgeschlossen:** `GenomeDescriptor` (12D-Featurevektor), `population_pca()` via Power-Iteration/SVD
(keine externen Abhaengigkeiten). CSV- und PNG-Export. GUI zeigt Snapshot als Scatterplot,
exportiert PNG/CSV. Tests in `test_landscape.py`.

### ✓ P1 Populations-Filter und -Aggregatoren API

**Abgeschlossen:** `population.filter()`, `map()`, `reduce()`, `group_by()`, `top_k()`.
Alle Methoden mit Typ-Annotationen; kein Overhead wenn nicht verwendet.
Tests in `test_population.py`.

### ✓ P1 Connection-Weight-Histogramm und Gewichtsgesundheit

**Abgeschlossen:** `WeightHistogram`-Widget in `gui/canvas.py`. `export_best_weights_npy()` API.
N-Generationen-Streak-Tracking in Diagnostics. GUI zeigt Histogramm/Gesundheitswerte.

### ✓ P1 Erweiterbare Aktivierungsfunktionen

**Abgeschlossen:** `CUSTOM_ACTIVATION_FNS`-Dict, `register_activation()`,
`resolve_activation_fn()`, `list_activations()`. GELU, Mish, SiLU als eingebaute
Erweiterungen. Pickle-Roundtrip via Rekonstruktion von `_activate_fn` in `__setstate__`.
14 Tests in `test_diagnostics_features.py`.

### ✓ P1 Multi-Population Inselmodell

**Abgeschlossen:** `IslandModel` in `evolution/islands.py`. N unabhaengige `Population`-Instanzen,
periodische Migration der Top-Genome zwischen zufaellig gepaarten Inseln.
`set_island_model()` API. Diagnostics: Fitness/Stagnation/Species pro Insel. 5 Tests.

### ✓ P1 Hyperparameter-Suche

**Abgeschlossen:** `hyperparameter_search()` in `evolution/hyperparameter_search.py`.
Grid-Search und Random-Search ueber Parameter-Grid. Mehrere Seeds pro Konfiguration.
Ranking nach medianer Fitness. RunDatabase-Integration. 3 Tests.

### ✓ P1 Ensemble-Bewertung und -Deployment

**Abgeschlossen:** `EnsembleGenome`-Wrapper mit `forward()`, `to_python()`.
Strategien: mean, vote, weighted. `make_ensemble(k, mode)` API. Export erzeugt
standalone Python. 7 Tests in `test_ensemble.py`.

### ✓ P1 Strukturierte / maschinenlesbare Protokollierung

**Abgeschlossen:** `set_log_format("jsonlines"|"csv"|"both")`, `set_tensorboard_logdir(path)`,
`set_log_callbacks(on_generation=fn)`. CSV enthaelt `validation_fitness`-Spalte.
JSONL enthaelt erweiterte Diagnostics.

### ✓ P1 Erweiterte Genome-Analyse im Inspect (Sensitivitaet / Attribution)

**Abgeschlossen:** `genome.sensitivity_analysis()` und `genome.dead_nodes()` in
`core/genome.py`. `SensitivityChart`-Widget in `gui/canvas.py`. Tests in
`test_diagnostics_features.py`.

### ✓ P1 Plugin-System fuer benutzerdefinierte Evaluatoren

**Abgeschlossen:** `PLUGIN_EXAMPLES`-Liste, `register_example()`, `load_plugins_from_directory()`,
`autoload_user_plugins()` (opt-in aus `~/.yane/plugins/`). Plugin-Dateien definieren
`register(reg)`-Funktion. `NeuroEvolution.register_example()` als Static-Method-API. 7 Tests.

### ✓ P1 Lernkurven-Vergleich (mehrere Runs)

**Abgeschlossen:** `ComparisonTab` in `gui/tabs/comparison_tab.py`. Laedt Runs, zeichnet
ueberlagerte Fitness-Kurven, berechnet Median/IQR-Baender, exportiert PNG/CSV.
Baut auf Experiment-Tracking und `RunRecord`-Kompatibilitaet auf.

### ✓ P1 Fitness-Surrogate-Modell (Billigfilter vor teurer Evaluierung)

**Abgeschlossen:** `FitnessSurrogate` in `evolution/surrogate.py`. Lineares Modell
(OLS mit Ridge) ueber 12D-Genome-Deskriptor-Vektoren. Warmup-Phase, adaptive
Filterung der unteren `surrogate_frac`. Spearman-Rho-Diagnostik. 5 Tests.

### ✓ P1 Automatische Fitness-Shaping-Erkennung

**Abgeschlossen:** `FitnessLandscapeAnalyzer` mit `analyze()` → `FitnessLandscapeReport`
(Sparsity, Plateau, Skewness, Cluster-Separability) und `recommend_transform()`.
Automatische Anwendung alle 50 Generationen via `set_auto_fitness_shaping()`. 8 Tests.

### ✓ P1 Online-Hyperparameter-Adaptation (Bandit-Tuning waehrend Training)

**Abgeschlossen:** `UCB1Bandit`-Klasse in `evolution/online_tuning.py`.
`set_online_tuning()` mit Parametern `mutation_rate` und `n_lamarck_steps`.
UCB1-Formel mit Exploration/Exploitation-Phase. Fitness-Delta als Reward. 8 Tests.

### ✓ P1 Weight-Inheritance beim Crossover

**Abgeschlossen:** `NeuroEvolution.set_weight_inheritance(enabled, blend_alpha)`.
Matching-Connections blended: `blend_alpha * w_fitter + (1-blend_alpha) * w_weaker`.
Bias ebenso blended. Default -1.0 = 50/50-Zufall. 7 Tests in `test_neuro_evolution.py`.

---

### ✓ P2 Genome-Codec-Protokoll (austauschbare Serialisierung)

**Abgeschlossen:** `GenomeCodec`-Protokoll, `PickleCodec`, `JsonCodec`,
`set_checkpoint_codec()`, `migrate_checkpoint()`, `detect_codec()`. 9 Tests.

### ✓ P2 Konfigurationsversionierung und Kompatibilitaets-Check

**Abgeschlossen:** Checkpoint-Metadaten mit Version, Topologie-Daten, Konfigurations-Hash.
`CompatibilityLevel` (`EXACT`, `COMPATIBLE`, `BREAKING`), `compatibility_report()`,
`check_compatibility()`. GUI-Metadaten zeigen Konfig-Hash. CLI-Diff via
`python -m yane.checkpoint --diff old.pkl new.pkl`.

### ✓ P2 YANE → PyTorch-Bruecke (NAS + Feinabstimmung)

**Abgeschlossen:** `genome_to_torch_module()` in `evolution/torch_bridge.py` exportiert
Genome als `torch.nn.Module` mit exakter Topologie und Gewichten. `forward_with_torch()`
mit Fallback. Memory-Knoten via GRUCell. Test (skipped ohne torch).

### ✓ P2 Population-Size-Adaptation (Dynamische Pop-Groesse)

**Abgeschlossen:** `set_adaptive_pop_size(min_pop, max_pop, schedule)`. `linear_decay`
senkt Groesse monoton; `performance_based` nutzt Stagnations- und Diversitaetssignale.
Diagnostics: `current_pop_size`, `last_resize_trigger`, `pop_size_history`. 8 Tests.

### ✓ P2 Gradient-gesteuerte Mutations-Richtung (Lamarck-Momentum)

**Abgeschlossen:** `LamarckRefiner` speichert Gewichts-Delta des letzten Schritts.
`set_lamarck_momentum(enabled, momentum_prob, decay)`. Momentum-Decay exponentiell
(`decay=0.9`). Tests: Momentum-Vektor-Update; `momentum_prob=0` kein Einfluss.

### ✓ P2 Automatisches Post-Training Pruning (Netzwerk-Komprimierung)

**Abgeschlossen:** `set_post_training_pruning(enabled, threshold, max_drop_frac)`.
`genome.prune()`, `genome.compress()`, `genome.prune_stats()`. Fitness-Rollback
wenn Drop > `max_drop_frac`. 12 Tests in `test_post_training_pruning.py`.

### ✓ P2 Differenzierbare Topologie-Suche (DARTS-Lite)

**Abgeschlossen:** Gate-Werte per Connection (`_darts_gates: dict[int, float]`)
werden jede Generation aus `sigmoid(|weight| * 2)` aktualisiert. Post-Training
Pruning entfernt Connections mit Gate < threshold vom besten Genom.
`NeuroEvolution.set_darts_mode(enabled, prune_threshold)`. Gates vererbt bei
`copy()` und `crossover()`. 12 Tests in `tests/test_darts.py`.

### ✓ P2 Intrinsische Belohnung / Curiosity-Modul

**Abgeschlossen:** `IntrinsicCuriosityModule` (2-Layer-Vorhersagenetz) in
`_run_evaluations()` integriert. Vorhersagefehler als Fitness-Bonus. Online-SGD
mit Gradient-Clipping. `NeuroEvolution.set_curiosity(enabled, weight, network_size, lr)`.
14 Tests in `tests/test_curiosity.py`.

### ✓ P2 Shared Weights (Weight-Sharing zwischen Verbindungen)

**Abgeschlossen:** `Connection.weight_group: str | None`. `genome.weight_groups: dict[str, float]`.
`genome.sync_shared_weights()`, `genome.set_weight_group()`. Lamarck-Deduplication via
`get_lamarck_connections()`. `mutate()` resynct automatisch. `NeuroEvolution.set_shared_weights(enabled)`.
`copy()`, `crossover()`, `__setstate__` rueckwaertskompatibel. 24 Tests in `tests/test_shared_weights.py`.
