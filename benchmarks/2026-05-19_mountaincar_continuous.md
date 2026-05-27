# MountainCar (Continuous) — 3 Min + 30 Min Langzeittest

**Datum:** 2026-05-19  
**Branch/Commit:** major-update, c7bef0d (mit Species-Fix)  
**Zweck:** Langzeittest nach Species-Fix

## Konfiguration

| Parameter | Wert |
|---|---|
| Pop-Größe | 150 |
| n\_eval | 3 (mean) |
| max\_train\_steps | 1.000 |
| n\_inputs / n\_outputs | 2 / 1 |
| max\_nodes / max\_connections | 20 / 60 |
| Fitness-Funktion | max\_pos + 10 (wenn gelöst); max\_pos ∈ [-1.2, 0.45] |
| Ziel-Fitness | 10.0 |
| Umgebung | MountainCarContinuous-v0 |

## Ergebnisse 3-Min-Test

| Zeit | Iter | Species | Threshold | Stagnation | Best Fitness |
|---|---|---|---|---|---|
| 30 s | 2.782 | 11 | 1.500 | 1.113 | 0.077 |
| 1 Min | 5.850 | 9 | 1.500 | 910 | **10.501** ✓ |
| 2 Min | 13.803 | 6 | 1.500 | 4.543 | 10.514 |
| 3 Min | 22.166 | 6 | 1.500 | 8.699 | 10.514 |

- **Gelöst:** Ja, bei Iter 3.907 (~1 Min)
- **Top-3:** 10.514 / 10.513 / 10.513 (Pool konvergiert auf Lösung)
- **Diversity-Injections:** 184

## Ergebnisse 30-Min-Langzeittest

| Zeit | Iter | Species | Threshold | Stagnation | Best Fitness | Inj |
|---|---|---|---|---|---|---|
| 1 Min | 5.725 | 5 | 1.010 | 914 | 0.081 | 47 |
| 2 Min | 12.364 | 5 | 1.500 | 1.059 | **10.503** ✓ | 93 |
| 3 Min | 20.932 | 6 | 1.500 | 6.259 | 10.513 | 166 |
| 5 Min | 39.749 | 6 | 1.500 | 1.379 | 10.514 | 331 |
| 30 Min | 268.150 | **5** | 1.500 | 65.194 | 10.514 | 2.357 |

- **Gelöst:** Ja, bei Iter 7.255 (~90 Sek)
- **Top-3:** 10.514 / 10.514 / 10.514 (vollständig konvergiert)
- **Cross/Mut:** 119.361 / 146.431
- **Diversity-Injections:** 2.357

## Beobachtungen

- Trotz "notorisch schwer für NEAT" wird das Problem in ~1 Min gelöst
- Species stabil bei 6–11, pendelt sich auf 6 ein (Ziel: 5, Dead-band ±1)
- Threshold bleibt konstant bei 1.500 — Merging reguliert effektiv
