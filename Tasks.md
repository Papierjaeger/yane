# Tasks: YANE staerker machen

Diese Datei ist die aktuelle Roadmap fuer YANE. Offene und neue Tasks stehen
oben. Abgeschlossene Arbeit ist am Ende kompakt zusammengefasst.

## Status

**Aktueller Stand:** Alle P0-Bausteine fertig. Die grossen P1-Architekturthemen
aus diesem Loop sind weitgehend implementiert (Adaptive Policy System,
RunDatabase/Experiment-Tracking, Selektionsstrategien, Middleware, Reports,
Inselmodell, Surrogates, Online-Tuning, Weight-Inheritance). P2-Features
Lamarck-Momentum, Post-Training Pruning, Population-Size-Adaptation,
Intrinsic Curiosity, DARTS-Lite und Shared Weights vollstaendig abgeschlossen.
Teststand: `1153 passed, 1 skipped`.

> **Paradigmenwechsel (2026-05-26):** Der P0 Meta-Adaptive Orchestration Layer
> ist das zentrale Architektur-Feature der naechsten Evolutionsstufe. Er
> integriert ALLE existierenden adaptiven Systeme (Mutationsraten, Speziation,
> Recovery, Online-Tuning, Pop-Size, Fitness-Shaping, Anytime-Eval, Surrogate)
> und ALLE zukuenftigen P2-Features unter einem gemeinsamen Meta-Optimizer.
> Ziel: `yane.auto_train(evaluator)` — kein einziger manueller Parameter mehr.
> Dieser Task sollte VOR den meisten anderen offenen Tasks implementiert werden.

> **Roadmap-Erweiterung (2026-05-26):** 28 neue Features geplant — 1 P0
> (Meta-Adaptive Orchestration), 6 P1 (Benchmarking-Suite, WandB/MLflow,
> Interactive Evolution, Hardware-Aware, ResourceBudget-System, Data
> Augmentation) und 21 P2 (ONNX-Export, Distillation, Gradient-Hybrid,
> WASM-Export, Attention-Heads, LTC-Nodes, Temporal Speciation, Self-Play,
> H-NEAT, GRN-Encoding, Developmental NEAT, Continual Learning, Meta-Learning,
> Reservoir Computing, Open-Endedness, Multi-Agent Cooperation, Bayesian NEAT,
> Safe NEAT, Sparse NEAT, TFLite-Export, Symbolic Regression).

> **Roadmap-Korrektur:** Mehrere P2-Forschungsfeatures besitzen aktuell nur
> isolierte Experimental-/Spike-Bausteine. Sie sind unten als `⚡` oder `□`
> markiert, bis sie sauber in `Genome`, `Population`, `NeuroEvolution`,
> Checkpoints, Crossover und Tests integriert sind.

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
- **P0 Meta-Adaptive Orchestration Layer:** Phase 1 (ParamRegistry, 55 Tests) ✓, Phase 2 (ProblemProfiler, 44 Tests) ✓, Phase 4 (KnowledgeBase, 54 Tests) ✓, Phase 3 (MetaOptimizer, 46 Tests) ✓, Phase 5 (Feature Gating, 53 Tests) ✓, Phase 6 (auto_train, 38 Tests) ✓ abgeschlossen. Teststand: `1443 passed, 1 skipped`.

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

### □ P0 Meta-Adaptive Orchestration Layer — Das selbstoptimierende YANE

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

### □ P1 Automated Regression Benchmarking Suite (CI-faehige Benchmark-Pipeline)

Benchmarks werden derzeit manuell gestartet und die Ergebnisse nur informell
in `benchmarks/` abgelegt. Es gibt keine automatisierte Erkennung von
Regressionen zwischen Code-Aenderungen.

**Ziel:** Eine CI-faehige Benchmark-Suite, die automatisch Regressionen erkennt,
Reports generiert und historische Trends verfolgt.

**Design: `BenchmarkRunner` und `RegressionDetector`**

- `BenchmarkRunner`: liest eine `benchmarks.yaml`-Konfiguration mit
  Beispielen, Seeds, Timeouts, Target-Fitness und Vergleichsbasislinie.
- `RegressionDetector`: vergleicht aktuelle Runs mit gespeicherten
  Baseline-Werten (Median-Fitness, Konvergenz-Iterationen, Erfolgsrate).
  Nutzt Mann-Whitney-U-Test fuer statistische Signifikanz (p < 0.05).
- `run_benchmark_suite(path="benchmarks.yaml")`: fuehrt alle definierten
  Benchmarks aus und gibt einen `BenchmarkReport` zurueck.
- `RegressionSeverity`: `NONE`, `MINOR` (<5% Verschlechterung), `MAJOR`
  (5-20%), `CRITICAL` (>20% oder Erfolgsrate bricht ein).
- `baseline/`-Verzeichnis: speichert Referenzwerte pro Benchmark als JSON.
  `--update-baseline` Flag aktualisiert sie nach manueller Pruefung.

**CI-Integration:**

- Exit-Code: 0 bei `NONE`/`MINOR`, 1 bei `MAJOR`, 2 bei `CRITICAL`.
- `python -m yane.benchmarks --ci` fuer CI-Pipelines.
- JSON-Report auf stdout fuer einfaches Parsen in GitHub Actions/GitLab CI.
- Optionale GitHub-Issue-Erstellung bei `CRITICAL`-Regression (via CLI-Flag).

**Trend-Tracking:**

- `benchmarks/history/` speichert Zeitreihen pro Benchmark (Datum, Commit,
  Median-Fitness, Konvergenz-Iterationen).
- `benchmark_trend(example_name) → TrendReport` mit Visualisierung (ASCII-Plot
  oder PNG via Matplotlib, falls verfuegbar).

**Akzeptanzkriterien:**

- Alle existierenden Benchmarks in `benchmarks/` sind in der Suite
  konfigurierbar.
- Regression-Erkennung meldet korrekt eine absichtlich verschlechterte
  Mutation (z. B. `sigma_global` auf Extremwert).
- CI-Mode gibt korrekten Exit-Code.
- Baseline-Update ueberschreibt nur mit explizitem Flag.
- Trend-Tracking speichert und laedt historische Daten korrekt.
- Tests: RegressionDetector mit synthetischen Fitness-Verlaeufen;
  CI-Exit-Codes; Baseline-Laden/Speichern.

---

### □ P1 WandB / MLflow Integration (Experiment-Tracking-Backends)

TensorBoard-Support existiert bereits (`set_tensorboard_logdir()`), aber viele
Teams nutzen WandB (Weights & Biases) oder MLflow fuer Experiment-Tracking,
Hyperparameter-Vergleiche und Team-Kollaboration.

**Ziel:** Alternative Tracking-Backends, die parallel zu TensorBoard und
RunDatabase laufen koennen.

**Design: `TrackingBackend`-Protokoll**

```python
class TrackingBackend:
    def init(self, config: dict) -> None: ...
    def log_metrics(self, metrics: dict, step: int) -> None: ...
    def log_artifact(self, path: str, artifact_type: str) -> None: ...
    def log_config(self, config: dict) -> None: ...
    def finish(self) -> None: ...
```

- `NeuroEvolution.set_tracking_backend(backend)` registriert einen
  TrackingBackend; mehrere Backends sind gleichzeitig moeglich (wandb +
  tensorboard + mlflow).
- `WandbBackend`: nutzt `wandb.init()`, `wandb.log()`,
  `wandb.log_artifact()`. API-Key aus Umgebungsvariable `WANDB_API_KEY`.
- `MlflowBackend`: nutzt `mlflow.start_run()`, `mlflow.log_metrics()`,
  `mlflow.log_artifact()`. Tracking-URI aus Umgebungsvariable
  `MLFLOW_TRACKING_URI`.
- Generations-Metriken (Best-Fitness, Mean-Fitness, Species-Count, IQR,
  Novelty, etc.) werden automatisch geloggt.
- Checkpoint-Dateien werden als Artifacts hochgeladen (optional).
- Run-Konfiguration wird als `config`-Dict geloggt.
- `finish()` wird automatisch am Ende von `train()` aufgerufen.

**Abgrenzung zu RunDatabase:**

- RunDatabase ist der lokale, YANE-eigene Persistenz-Layer (SQLite).
- Tracking-Backends sind externe Cloud-/Team-Dienste fuer Visualisierung und
  Kollaboration.
- Beide koennen parallel genutzt werden.

**Akzeptanzkriterien:**

- `WandbBackend` loggt Metriken ohne Fehler (Test mit `wandb.init(mode="disabled")`).
- `MlflowBackend` loggt Metriken in lokales Tracking-Verzeichnis.
- Mehrere Backends gleichzeitig aktiv: alle erhalten dieselben Metriken.
- Fehlende optionale Abhaengigkeiten (`wandb`, `mlflow`) werden mit klarem
  ImportError gemeldet, nicht als harte Abhaengigkeit.
- Tests: Mock-Backend verifiziert korrekte Metrik-Uebergabe; ImportError bei
  fehlenden Paketen; Multi-Backend-Dispatch.

---

### □ P1 Interactive Evolution — Human-in-the-Loop

Nicht jede Aufgabe hat eine objektive Fitness-Funktion. Bei kreativen oder
subjektiven Aufgaben (Design, Kunst, Musik, Praeferenzlernen) bewertet der
Mensch die Genome — interaktiv ueber die GUI.

**Ziel:** `InteractiveEvaluator` in der GUI, die menschliches Feedback
(Favoriten, Rankings, Slider) als Fitness-Signal verwendet.

**Design: `InteractiveEvaluator` in `gui/interactive_eval.py`**

**Feedback-Modi:**

1. **Paarweiser Vergleich:** GUI zeigt zwei Genome nebeneinander
   (Output-Visualisierung oder Verhaltens-Animation). Nutzer waehlt das
   bessere aus. Elo-Rating wird als Fitness verwendet.
2. **Rating:** Nutzer bewertet jedes Genom auf einer Skala (1-5 Sterne,
   Slider 0-100).
3. **Ranking:** Nutzer ordnet K Genome nach Praeferenz. Rank wird in
   Fitness umgerechnet (1. Platz = K, letzter Platz = 1).
4. **Implicit:** Nutzerverhalten (Verweildauer, Interaktionen) wird als
   implizites Fitness-Signal verwendet.

**API:**

```python
yane.set_interactive_evaluation(mode="pairwise", surrogate_model=True,
                                surrogate_update_interval=5)
yane.submit_feedback(genome_id, rating)  # programmatisch
```

**Akzeptanzkriterien:**

- Paarweiser Vergleich produziert konsistente Elo-Ratings (Test mit
  simuliertem Nutzer).
- Surrogate-Modell reduziert Anzahl benoetigter menschlicher Bewertungen.
- GUI zeigt mindestens zwei Genome gleichzeitig an.
- `submit_feedback()` aktualisiert Fitness korrekt.
- Tests: Elo-Update; Surrogate-Vorhersage; Feedback→Fitness-Konvertierung;
  Modus-Wechsel.

---

### □ P1 Hardware-Aware NEAT (Deployment-Constraint-Evolution)

Genome werden in der Cloud oder auf lokaler Workstation evolviert, aber auf
Edge-Geraeten deployed. YANE soll Genome direkt unter Hardware-Constraints
evolvieren: maximale FLOPs, Speicher, Latenz, Energie.

**Ziel:** `set_hardware_constraints()` erweitert `efficiency_score` um reale
HW-Metriken und bestraft Genome die HW-Budget ueberschreiten.

**Design: `HardwareConstraint` in `evolution/hardware_aware.py`**

```python
yane.set_hardware_constraints(
    max_flops=1_000_000,      # 1M FLOPs pro Forward-Pass
    max_memory_bytes=4096,    # 4KB
    max_latency_us=100,       # 100µs
    target_platform="cortex-m4"
)
```

**Metrik-Berechnung (deterministisch, aus Genom-Topologie):**

- `flops_per_forward()`: Zaehlt Multiplikationen + Additionen pro Forward.
- `memory_bytes()`: `n_nodes * sizeof(Node) + n_connections * sizeof(Connection)`.
- `latency_us()`: Schaetzung basierend auf FLOPs / Zielplattform-MHz * Safety-Factor.

**Vordefinierte Profile:** `"cortex-m4"`, `"cortex-m7"`, `"esp32"`,
`"raspberry-pi-zero"`, `"raspberry-pi-4"`, `"desktop"`.

> **Integration:** Dieser Task wird vom P1 ResourceBudget-System orchestriert.
> `HardwareConstraint` definiert die ZIEL-Hardware fuers Deployment;
> `ResourceBudget` managed die LAUFZEIT-Ressourcen waehrend des Trainings.

**Akzeptanzkriterien:**

- `hardware_profile()` gibt korrekte FLOPs/Memory/Latenz zurueck.
- Fitness-Penalty ist proportional zur Constraint-Ueberschreitung.
- Vordefinierte Profile laden korrekte Werte.
- `hw_pareto_front` enthaelt nicht-dominierte Genome.
- Tests: FLOPs-Zaehlung; Memory-Schaetzung; Penalty-Berechnung;
  Plattform-Profile; Pareto-Front.

---

### □ P1 ResourceBudget-System — Einheitliches Ressourcen-Management

**Das betriebswirtschaftliche Gegenstück zum MetaOptimizer.** Waehrend der
MetaOptimizer fragt „wie nutze ich Ressourcen optimal?", stellt das
ResourceBudget sicher, DASS Ressourcen eingehalten werden — und kalibriert
sich selbst.

**Ziel:** `yane.set_budget(total_time="30min", max_memory="auto")` — Budgets
in menschenlesbaren Einheiten, automatische Erkennung der Maschinenkapazitaet,
einheitliche Enforcement-Strategie.

**Architektur:**

```
┌──────────────────────────────────────────────────────────────────┐
│                  ResourceBudget System                            │
│  ┌───────────────────────┐   ┌──────────────────────────────┐   │
│  │  Resource Discovery    │   │  Budget Definition             │   │
│  │  (auto-calibration)    │   │  (user + auto)                │   │
│  │  - CPU cores/usage     │   │  total_time: "30min"          │   │
│  │  - RAM total/available │   │  max_memory: "auto" → 80% RAM │   │
│  │  - GPU memory (nvidia) │   │  max_cpu_pct: 75%             │   │
│  │  - Disk space/temp     │   │  per_genome_ms: "auto"        │   │
│  │  - Battery status      │   │  target_platform: None        │   │
│  └───────────────────────┘   └──────────────────────────────┘   │
│                                                                    │
│  Graceful Degradation Pipeline:                                    │
│  Stufe 1: reduce_pop()    → Pop-Größe halbieren                   │
│  Stufe 2: skip_lamarck()  → Lamarck deaktivieren                 │
│  Stufe 3: simplify_topology() → max_nodes senken                  │
│  Stufe 4: disable_research()  → P2-Features deaktivieren          │
│  Stufe 5: reduce_eval_budget() → Anytime-Budget kürzen            │
│  Stufe 6: emergency_stop()    → Checkpoint + beenden              │
└──────────────────────────────────────────────────────────────────┘
```

**API:**

```python
yane.set_budget(
    total_time="30min",        # Klar, lesbar
    max_memory="auto",         # 80% des verfügbaren RAMs
    target_platform="cortex-m4"  # Deployment-Ziel
)
# ODER:
yane.set_budget("auto")       # Alles automatisch
```

**Alte APIs bleiben als Kompatibilitaets-Wrapper:** `set_efficiency_penalty()`,
`set_resource_limits()`.

**Akzeptanzkriterien (Gesamt):**

- `set_budget("auto")` kalibriert alle Werte automatisch ohne Crash.
- `set_budget(total_time="30min")` stoppt Training nach 30 Minuten (±5%).
- Memory-Budget-Ueberschreitung → `trim_memory()` + Warnung.
- Graceful-Degradation eskaliert korrekt durch alle 6 Stufen.
- Budget-Status ist via `budget_status()` jederzeit abrufbar.
- ResourceDiscovery funktioniert auf Linux, macOS, Windows.
- Tests: ~25 (Auto-Kalibrierung, Einheiten-Parsing, Budget-Enforcement,
  Graceful-Degradation, MetaOptimizer-Integration, API-Kompatibilitaet).

---

### □ P1 Evolutionary Data Augmentation

Bei kleinen Trainingsdatensaetzen overfittet NEAT. Evolutionaere Data
Augmentation findet automatisch Augmentierungen die die Generalisierung
verbessern — die Augmentierungsparameter sind selbst evolvierbar.

**Ziel:** `set_evolutionary_augmentation()` evolviert Augmentierungs-Pipelines
parallel zur Genom-Evolution.

**Design: `AugmentationPipeline` in `evolution/augmentation.py`**

```python
yane.set_evolutionary_augmentation(enabled=True,
    augmentation_space=[
        "gaussian_noise", "dropout_noise", "scaling",
        "translation", "mixup", "cutout"
    ],
    population_augmentations=10
)
```

**Augmentierungstypen:** `gaussian_noise`, `dropout_noise`, `scaling`,
`translation`, `mixup`, `cutout`, `label_smoothing`. Jedes Augmentierungs-Gen
hat `(type, probability, magnitude)`.

**Evolutions-Mechanik:** Augmentierungsgenome haben eigene Population;
Fitness = NEAT-Test-Set-Performance; Crossover/Mutation evolvieren Pipeline-
Parameter parallel zur Hauptpopulation.

**Akzeptanzkriterien:**

- Augmentierungs-Pipeline-Parameter veraendern sich ueber Generationen.
- Test-Set-Fitness mit Augmentierung > ohne Augmentierung (bei Small Data).
- Augmentierungs-Crossover produziert valide Pipelines.
- Tests: Pipeline-Forward; Augmentierungs-Mutation; Crossover; Fitness-
  Attribution; Small-Data-Benchmark.

---

### ⚡ P2 Transfer Learning / Genome Fine-Tuning

Wissen aus einem trainierten Genom soll auf eine neue Aufgabe uebertragen werden.

**Aktueller Stand:** Teilweise implementiert. `warm_start_from_checkpoint(path)`
laedt Genome aus einem Checkpoint und passt Eingabe/Ausgabe-Dimension an.
`load_genome_as_seed()` seeded die Population tatsaechlich mit einem gegebenen
Genom. Freeze speichert die urspruenglichen Mutations-/Spike-Raten pro
Connection-Innovation und `set_transfer_unfreeze()` kann eingefrorene Teile
waehrend des Trainings schrittweise entsperren. `fine_tune_genome()` bietet
Lamarck-only-Feinabstimmung ohne Topologie-Aenderung; `behaviour_clone()` kann
das geklonte Genom direkt als Population-Seed setzen. Transfer-Benchmarks
fehlen noch.

- ✓ `yane.load_genome_as_seed(genome, freeze_layers=[...])`: bestimmte Verbindungsgruppen koennen eingefroren werden.
- ✓ Lamarck-Feinabstimmung auf neuer Aufgabe ohne Topologie-Aenderung als erste Phase.
- ✓ Dann schrittweise Entsperren eingefrorener Teile (progressive unfreeze).
- Benchmark: Transfer CartPole → LunarLander vs. Training from scratch.

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

### ⚡ P2 Synaptische Plastizitaet (STDP / Hebbsches Lernen)

Genome lernen derzeit nur durch Evolution, nicht durch Erfahrung innerhalb einer Episode.

**Aktueller Stand:** Experimenteller Spike. `STDPRule` existiert als isolierte
Regel fuer Gewichtsanpassungen. Genome speichern aber noch keine evolvierbaren
Plastizitaetskoeffizienten, `genome.forward()` veraendert Gewichte nicht
episoden-lokal und `genome.reset()` verwaltet keine Basis-/Arbeitsgewichte.

- Knoten/Verbindungen koennen evolvierte Hebb-Regel-Koeffizienten (A, B, C, D) tragen.
- Gewichte werden waehrend `genome.forward()` nach der STDP-Regel angepasst (intra-lifetime-learning).
- `genome.reset()` setzt Gewichte auf Basiswerte zurueck (Plastizitaet ist episoden-lokal).
- Benchmark: STDP vs. Lamarck auf Aufgaben mit veraenderlicher Umgebung.

---

### ⚡ P2 Neuromodulation

Modulatorische Signale erlauben kontextabhaengige Gewichtung ganzer Verbindungsgruppen.

**Aktueller Stand:** Experimenteller Spike. `Neuromodulator` existiert als
isolierter Skalierungshelfer. Es gibt noch keinen `MODULATOR`-Knotentyp, keine
evolvierbaren Modulationskanten und keine Integration in `genome.forward()`.

- Sonderknotentyp `Modulator`: sein Ausgabe-Wert skaliert eingehende Verbindungen anderer Knoten.
- Evolvierbar: welcher Knoten moduliert, welche Verbindungen beeinflusst werden, Staerke.
- Benchmark: Modulation vs. kein Modulation auf einem Aufgaben-Wechsel-Szenario.

---

### ⚡ P2 Evolutionaere Input-Gruppierung (Evolvable Input Aggregation Layer)

NEAT behandelt jeden Input als unabhaengigen Knoten; bei hochdimensionalen Eingaben entstehen so viele Verbindungen, dass der Suchraum explodiert.

**Aktueller Stand:** Experimenteller Spike. `InputGrouping` kann externe Inputs
standalone reduzieren und aggregieren. Es ist aber kein Teil von `Genome`, wird
nicht gecrossovert oder gecheckpointet, erzeugt keine dynamischen Input-Knoten
und hat keine `NeuroEvolution.set_input_grouping()`-Integration.

**Design: `InputGroup` und `InputGrouper`**

- `InputGroup`: `members: list[int]`, `aggregation: AggType` (mean / max / sum / weighted_sum), `weights: list[float]`, `enabled: bool`.
- `InputGrouper`: ist Teil des `Genome`-Objekts (wird beim Checkpoint mitgespeichert und gecrossoverd).
- Forward-Pass: `grouper.transform(raw_inputs) -> list[float]`.

**Mutations-Operatoren:** `create_group`, `split_group`, `merge_groups`,
`add_input_to_group`, `remove_input_from_group`, `change_aggregation`.

**`NeuroEvolution.set_input_grouping(enabled=True, initial_groups=None)`**.

**Tests:** `transform()` produziert korrekte Ausgabe-Dimension; `split_group()` fuegt korrekten Input-Knoten hinzu; Crossover zweier Genome mit unterschiedlichen Groupern laeuft ohne Fehler; Checkpoint-Round-Trip erhaelt Gruppen.

---

### ⚡ P2 Evolutionaere Output-Gruppierung (Evolvable Output Synergy Layer)

Symmetrisch zur Input-Gruppierung: von aussen sieht das Genome weiterhin N Ausgabe-Kanaele; intern werden Outputs die immer gemeinsam aktiviert werden unter einem geteilten Proto-Output-Knoten zusammengefasst.

**Aktueller Stand:** Experimenteller Spike. `OutputGrouping` kann Proto-Outputs
standalone auf externe Outputs expandieren. Es ist aber kein Teil von `Genome`,
veraendert keine interne Output-Knoten-Anzahl, wird nicht gecrossovert oder
gecheckpointet und ist nicht in `genome.forward()`/`genome_to_python()`
integriert.

**Design: `OutputGroup` und `OutputGrouper`**

- `OutputGroup`: `targets: list[int]`, `expansion: ExpType` (copy / scale / affine), `weights: list[float]`, `enabled: bool`.
- Forward-Pass: `grouper.expand(proto_outputs) -> list[float]` der Laenge N.

**Mutations-Operatoren:** `create_group`, `split_group`, `merge_groups`,
`add_output_to_group`, `remove_output_from_group`, `change_expansion`.

**`NeuroEvolution.set_output_grouping(enabled=True, initial_groups=None)`**.

**Tests:** `expand()` gibt stets N Werte zurueck; `genome.forward()` gibt unveraendert N Werte zurueck; `split_group()` erhoeh interne Output-Knoten-Anzahl um 1; Crossover fehlerfrei; `genome_to_python()` erzeugt korrekten Expand-Block.

---

### ⚡ P2 Convolutional NEAT (CoDeepNEAT-inspiriert)

NEAT sucht verbindungsweise; fuer Bildverarbeitung ist die sinnvolle Sucheinheit ein Conv-Block (Filter, Stride, Channels), nicht eine einzelne Gewichtsverbindung.

**Aktueller Stand:** Experimenteller Spike. `ConvModule` kann einfache
Faltungs-/Patch-Operationen standalone ausfuehren. Es gibt noch keinen
`NodeType.CONV2D`, keine `add_conv_block`-Mutation, kein `forward_image()`,
keine Shared-Weight-Integration und keinen Python-Export fuer Conv-Knoten.

- Neuer Knotentyp `CONV2D` mit evolvierbaren Parametern: `kernel_size`, `stride`, `out_channels`, `activation`.
- Weight-Sharing automatisch: alle raeumlichen Positionen eines Filters teilen dasselbe Gewicht.
- Mutations-Operator `add_conv_block`: fuegt einen vollstaendigen Conv-Block als Einheit hinzu.
- `genome.forward_image(pixels, height, width, channels)`.
- Benchmark: Convolutional NEAT vs. HyperNEAT vs. flaches NEAT auf MNIST.

---

### □ P2 ES-HyperNEAT (Evolvable Substrate HyperNEAT)

Das aktuelle HyperNEAT-Substrat wird vom Nutzer manuell als Gitter-Koordinaten definiert; die CPPN weiss nicht wo sinnvolle Knoten-Positionen im geometrischen Raum liegen.

**Aktueller Stand:** HyperNEAT-Bausteine fuer feste Substrate sind vorhanden.
ES-HyperNEAT im eigentlichen Sinn fehlt: keine CPPN-varianzbasierte
Quadtree-Suche, keine evolvierbaren Substrat-Koordinaten, kein
`evolve_substrate=True`-Flow und keine Benchmarks.

- ES-HyperNEAT: CPPN-Output-Varianz bestimmt automatisch ob an einer Koordinate ein Knoten sinnvoll ist.
- `hyperneat_substrate(evolve=True)`: Substrat-Koordinaten aus der CPPN-Aktivierungslandschaft.
- `generate_genome_from_cppn(cppn, substrate, evolve_substrate=True)`: erweiterte Signatur.
- Benchmark: ES-HyperNEAT vs. festes Substrat auf einem 2D-Navigations-Task und MNIST.

---

### □ P2 Genome-to-ONNX-Export (Produktions-Deployment)

Trainierte Genome sind nur innerhalb von YANE/Python nutzbar. ONNX (Open Neural
Network Exchange) ist der Standard fuer plattformunabhaengiges Deployment.

**Ziel:** `genome.export_onnx(path="model.onnx", opset_version=17)`.

- ONNX-Graph: Input-Nodes → ONNX-Inputs, Output-Nodes → ONNX-Outputs.
- Jede Connection: `Mul(weight)` → `Add(bias)` → `Activation`.
- Zyklische Netze: BFS-Forward-Pass in feste Anzahl entrollter Schritte.
- `genome_to_onnx()` als standalone-Funktion; Validierung via `onnx.checker.check_model()`.

**Akzeptanzkriterien:**

- XOR-Genom: ONNX-Inferenz liefert identische Outputs (Toleranz 1e-5).
- ONNX-Modell besteht `onnx.checker.check_model()`.
- Zyklisches Genom mit `unroll_steps=3`: Outputs innerhalb 1e-3.
- Tests: ONNX-Export fuer XOR, lineares Netz, rekurrentes Netz, custom Aktivierungen.

---

### □ P2 Population Distillation (Ensemble → kompaktes Einzelgenom)

Ein Ensemble aus Top-K-Genomen ist robuster als ein einzelnes Genom, aber
teurer in der Inferenz. Distillation uebertraegt das Wissen in ein kompaktes Genom.

**Ziel:** `yane.distill_ensemble(k=5, target_nodes=10, distillation_steps=500)`.

**Ablauf:** Ensemble aus Top-K → Student-Genom initialisieren → Probe-Inputs
forwarden → MSE-Loss → Student-Gewichte via Lamarck-Refinement optimieren.

**Akzeptanzkriterien:**

- XOR: Student (3 Nodes) loest XOR exakt.
- Student ist kleiner als durchschnittliches Ensemble-Mitglied.
- Distillation-Loss sinkt monoton.
- Tests: Distillation auf XOR; Kompressionsrate; Output-Korrelation.

---

### □ P2 Gradient-NEAT-Hybrid-Modus (Backprop + Evolution interleaved)

**Ziel:** `train()` mit hybridem Modus, der zwischen Evolutions- und
Backprop-Phasen wechselt.

```python
yane.set_hybrid_mode(enabled=True, bp_interval=10, bp_epochs=50,
                     bp_lr=0.01, bp_batch_size=32)
```

**Ablauf:** Alle `bp_interval` Generationen: Top-K-Genome via
`genome_to_torch_module()` konvertieren → Backprop-Phase → Gewichte zurueck.

**Akzeptanzkriterien:**

- XOR: Hybrid konvergiert in <50% der Iterationen von reinem NEAT.
- Ohne PyTorch: klarer ImportError, kein Crash.
- Tests: Hybrid auf XOR; Gewichts-Persistenz; Replay-Buffer-Sampling.

---

### □ P2 WebAssembly-Export (Browser-Deployment)

**Ziel:** `genome_to_wasm()` generiert eine standalone HTML/JS/WASM-Datei.

```python
genome.export_wasm("model.html", mode="wasm", allow_cyclic=True, inline=True)
```

**Export-Strategie:** Python→C→WASM (Emscripten), ONNX→WASM (Fallback),
Pure-JS-Transpilation (einfachste, keine WASM-Abhaengigkeit).

**Akzeptanzkriterien:**

- XOR-Genom als `.html`: im Browser Forward-Pass identisch (Toleranz 1e-5).
- Pure-JS-Modus funktioniert ohne Emscripten.
- Tests: WASM/Pure-JS-Export; Output-Vergleich Python↔WASM↔JS.

---

### □ P2 Evolvable Attention Heads (Transformer-inspirierte Architektursuche)

**Ziel:** `NodeType.ATTENTION` als neuer Knotentyp mit Key/Query/Value-Berechnung.

- `head_dim`, `num_heads`: evolvierbare Parameter.
- Forward: `softmax(Q @ K^T / sqrt(head_dim)) @ V`.
- `NeuroEvolution.set_attention(enabled=True)` aktiviert Attention-Nodes.
- `genome_to_torch_module()` mappt auf `nn.MultiheadAttention`.

**Akzeptanzkriterien:**

- Attention-Node produziert korrekte Softmax-gewichtete Outputs.
- Crossover: Genome mit unterschiedlichen Attention-Konfigurationen kreuzbar.
- Checkpoint-Roundtrip erhaelt `head_dim` und `num_heads`.
- Tests: Attention-Mathe; Multi-Head-Konkatenation; Crossover; Checkpoint.

---

### □ P2 Liquid Time-Constant (LTC) Nodes (ODE-basierte Neuronen)

**Ziel:** `NodeType.LTC` mit ODE-basierter Dynamik.

- ODE: `dx/dt = -(1/τ) * x + f(input, bias)`.
- Diskrete Approximation (Forward Euler): `x_{t+1} = x_t + dt * (-x_t/τ + activation(sum(inputs) + bias))`.
- `tau`, `dt`: evolvierbar. `persist_value` implizit True.
- `NeuroEvolution.set_ltc(enabled=True)`.

**Akzeptanzkriterien:**

- LTC mit τ→∞ naehert sich konstantem State; τ→0 naehert sich instantanem Node.
- `genome.reset()` setzt LTC-State korrekt zurueck.
- Tests: ODE-Mathe; Reset-Verhalten; τ-Extremwerte; Crossover.

---

### □ P2 Temporal Speciation (Verhaltensbasierte Spezies-Bildung)

**Ziel:** `TemporalDistance` als alternative Kompatibilitaetsmetrik.

```python
yane.set_compatibility_distance(TemporalDistance(
    n_rollouts=5, rollout_len=20, time_weight=0.5
))
```

- Behavior Descriptor: Trajektorie ueber `rollout_len` Schritte.
- Distanz: Dynamic Time Warping (DTW) ueber Output-Trajektorien.
- Implementiert `DistanceMetric`-Protokoll; kombinierbar mit `ChainMetric`.

**Akzeptanzkriterien:**

- DTW-Distanz zwischen identischen Trajektorien ist 0.
- Caching: Trajektorie wird nicht neu berechnet wenn Topologie unveraendert.
- Tests: DTW-Mathe; Protokoll-Kompatibilitaet; Caching-Invalidierung.

---

### □ P2 Self-Play / Adversarial Populations (Kompetitive Co-Evolution)

**Ziel:** `set_adversarial_populations()` teilt Population in gegnerische Sub-Populationen.

```python
yane.set_adversarial_populations(n_populations=2, pairing="round_robin")
```

- Fitness: Nullsumme. Elo-Ratings pro Genom.
- `arms_race_indicator`: Anstieg der Fitness in BEIDEN Populationen.

**Akzeptanzkriterien:**

- Fitness ist korrekt Nullsumme.
- Elo-Ratings steigen in beiden Populationen bei gesundem Arms Race.
- Tests: Nullsummen-Fitness; getrennte Spezies; Elo-Update; Pairing-Mechanismen.

---

### □ P2 Hierarchical NEAT (H-NEAT) — Mehrstufige Policy-Architektur

**Ziel:** Zweistufiges System: Manager-Genom waehlt Sub-Policy, Worker-Genome fuehren aus.

```
High-Level Manager → Sub-Policy Pool (N Low-Level Genome)
```

- `ManagerGenome`, `WorkerGenome`, Sub-Policy-Pool mit eigener Evolution.
- `selection_mode`: `"hard"` (eine aktiv) oder `"soft"` (gewichtete Mischung).
- Mutations-Operatoren: `add_sub_policy`, `split_sub_policy`, `merge_sub_policies`.

**Akzeptanzkriterien:**

- Manager waehlt unterschiedliche Sub-Policies fuer unterschiedliche Zustaende.
- Checkpoint speichert/laedt komplette Hierarchie.
- Tests: Manager-Output-Range; Sub-Policy-Selektion; Pool-Mutation; Checkpoint.

---

### □ P2 Gene Regulatory Network (GRN) Encoding

**Ziel:** `GRNCodec` als alternative Genom-Kodierung (kompakt, indirekt).

- GRN-Gen: `(input_gene, output_gene, weight, activation, regulatory_sites[])`.
- Entwicklung (Genotyp→Phaenotyp): GRN fuer N Schritte simulieren → Connections.
- `NeuroEvolution.set_genome_encoding("grn", development_steps=5)`.
- `GRNCodec` implementiert `GenomeCodec`-Protokoll.

**Akzeptanzkriterien:**

- GRN mit 20 Genen enkodiert Genom mit >100 Connections.
- Crossover zweier GRN-Genome funktioniert (Alignment der Gene).
- Tests: GRN-Entwicklung; Phaenotyp-Groessen-Korrelation; Crossover; Codec-Protokoll.

---

### □ P2 Developmental NEAT — Ontogenese waehrend Evaluation

**Ziel:** `genome.developmental_forward()` aendert Topologie waehrend der Evaluation.

- Entwicklungsregeln: `DevelopmentalRule` mit `trigger_condition` + `action`.
- Episoden-lokal: `genome.reset()` stellt Basis-Topologie wieder her.
- `genome.freeze_development()` unterdr uckt alle Regel-Anwendungen.

**Akzeptanzkriterien:**

- `developmental_forward()` fuegt waehrend Episode tatsaechlich Connections hinzu.
- Entwicklungsregeln werden korrekt vererbt und mutiert.
- Tests: Regel-Trigger; Episoden-Lokalitaet; freeze; Vererbung; Checkpoint.

---

### □ P2 Continual / Lifelong Learning NEAT

**Ziel:** `train()` mit aufgabenweisem Training ohne Catastrophic Forgetting.

```python
yane.set_continual_learning(mode="ewc", lambda_ewc=0.1, progressive=True)
yane.task_start("cartpole")
yane.train(cartpole_evaluator)
yane.task_start("lunarlander")
yane.train(lunarlander_evaluator)
print(yane.evaluate_all_tasks())
```

**Modi:** EWC, Progressive, Memory-Replay, Hybrid.

**Akzeptanzkriterien:**

- Nach Aufgabe 2: Fitness auf Aufgabe 1 ≥ 90% der urspruenglichen Fitness.
- Tests: EWC-Penalty; Progressive-Expansion; Replay-Buffer; Forgetting-Rate.

---

### □ P2 Meta-Learning NEAT — Few-Shot Adaptation

**Ziel:** `meta_train()` evolviert Genome, die sich in wenigen Episoden an neue Aufgaben anpassen.

```python
yane.meta_train(task_sampler, meta_iterations=1000, adaptation_steps=3)
```

- Inner Loop: Lamarck-Refinement als Adaptations-Mechanismus.
- Outer Loop: NEAT-Evolution auf Post-Adaptation-Fitness.

**Akzeptanzkriterien:**

- Post-Adaptation-Fitness > Pre-Adaptation-Fitness.
- Meta-trainiertes Genom adaptiert schneller als zufaellig initialisiertes.
- Tests: Inner-Loop-Lamarck; Adaptation-Delta; Task-Sampler-Integration.

---

### □ P2 Evolutionary Reservoir Computing

**Ziel:** `ReservoirGenome` mit fixiertem Reservoir und evolvierbaren Readout-Connections.

```python
yane.configure_reservoir(n_reservoir=100, spectral_radius=0.9,
                          input_scaling=0.5, leaking_rate=0.3)
```

- Nur Readout-Gewichte evolvieren; kein add_node/add_connection.
- `spectral_radius < 1`: Echo-State-Property garantiert.
- Optional: `RidgeRegressionReadout` (analytische Loesung).

**Akzeptanzkriterien:**

- Reservoir-State ist deterministisch bei gleichem Seed.
- Ridge-Readout loest XOR-nahen Task ohne Evolution.
- Tests: Determinismus; Echo-State-Property; Readout-Evolution; Checkpoint.

---

### □ P2 Open-Ended Evolution / Minimal Criterion

**Ziel:** `set_minimal_criterion(fn)` filtert Genome vor Selektion.

```python
yane.set_minimal_criterion(lambda g: g.fitness > -50.0)
yane.set_open_ended(mode="novelty_with_criterion", archive_size=200)
```

- `min_viable_frac` (Default: 0.1): adaptiver Schwellwert wenn zu wenige viable.
- Modi: `novelty_with_criterion`, `curiosity_with_criterion`, `quality_diversity_with_criterion`.

**Akzeptanzkriterien:**

- Genome unter Kriterium werden nicht zur Fortpflanzung zugelassen.
- Adaptive Lockerung greift wenn `viable_frac < min_viable_frac`.
- Tests: Kriteriums-Filter; adaptive Schwelle; viable-Boost; Archiv-Integration.

---

### □ P2 Multi-Agent Cooperation (kooperativ, nicht adversarial)

**Ziel:** `set_cooperative_population(n_agents)` trainiert N Genome kooperativ.

```python
yane.set_cooperative_population(n_agents=3, credit="shared")
```

- Credit-Assignment-Modi: `shared`, `difference` (Shapley-approx), `individual`, `hierarchical`.
- `role_specialization=True`: Agenten evolvieren in spezialisierte Rollen.

**Akzeptanzkriterien:**

- N Agenten erhalten korrekte Fitness nach den Modi.
- `role_similarity` sinkt ueber Generationen.
- Tests: Credit-Assignment-Modi; Rollen-Spezialisierung; Free-Rider-Erkennung.

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
