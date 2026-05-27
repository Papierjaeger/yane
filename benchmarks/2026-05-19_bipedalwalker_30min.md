# BipedalWalker-v3 — 3 Min + 30 Min Langzeittest

**Datum:** 2026-05-19  
**Branch/Commit:** major-update, 515ca96 (mit Species-Fix, Struct-Mut, Adaptivem Lamarck)  
**Zweck:** Langzeittest + Beobachtung Netzwerkgröße + Lamarck-Wirkung

## Konfiguration

| Parameter | Wert |
|---|---|
| Pop-Größe | 150 |
| n\_eval | 3 (mean) |
| early\_stop | -50 |
| max\_steps | 1.000 |
| n\_inputs / n\_outputs | 24 / 4 |
| n\_initial\_hidden | 4 |
| max\_nodes / max\_connections | 60 / 300 |
| Fitness-Funktion | kontinuierliche Aktionen ∈ [-1,1] |
| Ziel-Fitness | 0.0 |
| Umgebung | BipedalWalker-v3 |

## Ergebnisse 30-Min-Langzeittest

| Zeit | Iter | Sp | Stag | PoolBest | AllBest | Nodes | Conns | Lamarck |
|---|---|---|---|---|---|---|---|---|
| 2 Min | 854 | 6 | 208 | -19.91 | -19.91 | 33 | 116 | 52x |
| 4 Min | 1.619 | 5 | 114 | -7.19 | -7.19 | 33 | 117 | 81x |
| 8 Min | 2.890 | 5 | 349 | -2.88 | -2.88 | 33 | 116 | 121x |
| 12 Min | 3.851 | 5 | 105 | **-0.63** | -0.63 | 31 | 91 | 163x |
| 14 Min | 4.299 | 5 | 136 | **-0.07** | -0.07 | 31 | 90 | 193x |
| ~16 Min | ~4.551 | — | — | **>0.0 ✓** | 0.47 | — | — | — |
| 20 Min | 5.655 | 6 | 69 | 1.99 | 1.99 | 30 | 61 | 228x |
| 24 Min | 6.633 | 5 | 69 | 2.86 | 2.86 | **29** | **44** | 258x |
| 30 Min | 8.216 | **5** | 666 | 4.67 | **4.67** | 29 | 49 | 308x |

- **Gelöst:** Ja, bei Iter ~4.551 (~16 Min)
- **Top-3:** 4.67 / 4.61 / 4.30
- **Cross/Mut:** 2.647 / 5.511 (32% Crossover)
- **Diversity-Injections:** 57
- **Lamarck:** 308x angewendet, 768 Schritte gesamt (3.75% der Evaluierungen)

## Netzwerkgröße-Analyse

**Das Netz schrumpft während es besser wird — deutlichster Beweis bisher.**

| Phase | Nodes | Conns | Fitness |
|---|---|---|---|
| Anfang (2–8 Min) | 33 | 116–117 | -20 bis -3 |
| Annäherung (12–14 Min) | 31 | 90–91 | -0.6 bis -0.07 |
| Nach Lösung (20 Min) | 30 | 61 | +2.0 |
| Optimierung (24–30 Min) | **29** | **44–49** | +3–5 |

Vom Ausgangsnetz (33 Nodes / ~116 Connections) schrumpfte das beste Genom auf
29 Nodes / 49 Connections — bei gleichzeitig steigender Fitness. Evolution fand,
dass weniger Struktur bei BipedalWalker besser funktioniert.

## Beobachtungen

- **BipedalWalker gelöst in ~16 Min** — LunarLander (ähnliche Schwierigkeit) wurde
  in 30 Min nicht gelöst. Der Fortschritt ist auf die kombinierten Verbesserungen
  zurückzuführen (Lamarck + Strukturmutationen).
- **Species stabil bei 5** (Threshold 0.880) — Species-Fix funktioniert auch bei
  hochdimensionalen Eingaben (24 Inputs) korrekt.
- **Netz schrumpft mit zunehmender Fitness** — gewichtsbasiertes Remove +
  Rewiring entfernen irrelevante Verbindungen sobald sie nicht mehr benötigt werden.
  Kein expliziter Effizienz-Druck nötig.
- **Lamarck 308x / 3.75% Overhead** — konstant aktiv da Stagnation bei BipedalWalker
  häufig auftritt; gleichzeitig niedrig genug um den Eval-Durchsatz nicht zu gefährden.
