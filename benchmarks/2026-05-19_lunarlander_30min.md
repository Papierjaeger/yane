# LunarLander-v3 — 3 Min + 30 Min Langzeittest

**Datum:** 2026-05-19  
**Branch/Commit:** major-update, c7bef0d (mit Species-Fix)  
**Zweck:** Langzeittest + Beobachtung der Netzwerkgröße nach Fitness-Plateau

## Konfiguration

| Parameter | Wert |
|---|---|
| Pop-Größe | 150 |
| n\_eval | 3 (mean) |
| early\_stop | -200 |
| max\_steps | 1.000 |
| n\_inputs / n\_outputs | 8 / 4 |
| max\_nodes / max\_connections | 40 / 150 |
| Ziel-Fitness | 200.0 |

## Ergebnisse 3-Min-Test

- Bestes Fitness: 12.29 (Ziel: 200 — nicht erreicht, erwartet)
- Species: erst 37, dann durch Merging auf 5 (dauert länger als Acrobot wegen mehr struktureller Diversität)

## Ergebnisse 30-Min-Langzeittest

| Zeit | Iter | Sp | Stag | PoolBest | AllBest | Nodes | Conns | Inj |
|---|---|---|---|---|---|---|---|---|
| 1 Min | 9.370 | 5 | 864 | -3.44 | -3.44 | 12 | 0 | 78 |
| 2 Min | 17.867 | 23 | 6.691 | 7.57 | 7.57 | 14 | 2 | 152 |
| 3 Min | 26.129 | 5 | 667 | 34.57 | 34.57 | 14 | 5 | 225 |
| 6 Min | 50.087 | 5 | 6.461 | **92.18** | 92.18 | 14 | 5 | 438 |
| 11 Min | 89.417 | 5 | 4.189 | **103.50** | 103.50 | 15 | 7 | 786 |
| 15 Min | 117.517 | 5 | 881 | **109.39** | 109.39 | 17 | 9 | 1034 |
| 25 Min | 191.973 | 5 | 6.688 | **135.54** | 135.54 | **14** | **4** | 1695 |
| 30 Min | 229.157 | **5** | 36.466 | 135.54 | 135.54 | 14 | 4 | 1961 |

- **Gelöst:** Nein (135.54 / 200.0)
- **Top-3:** 135.54 / 120.88 / 109.39
- **Cross/Mut:** 45.317 / 181.813 (20% Crossover — niedriger als bei einfacheren Envs)

## Netzwerkgröße-Analyse

**Zentrale Beobachtung: Das Netz wächst mit der Fitness — aber schrumpft beim nächsten Durchbruch.**

| Fitness-Plateau | Nodes | Conns | Interpretation |
|---|---|---|---|
| -3 (Start) | 12 | 0 | 8 In + 4 Out, keine Verbindungen |
| 7 | 14 | 2 | 2 Hidden Nodes hinzugefügt |
| 92 | 14 | 5 | mehr Verbindungen, gleiche Struktur |
| 103 | 15 | 7 | 1 weiterer Hidden Node |
| 109 | 17 | 9 | 2 weitere Hidden Nodes |
| **135** | **14** | **4** | **Durchbruch mit kleinerem Netz!** |

Der beste Sprung von 109 → 135 wurde von einem **einfacheren** Genom erreicht
(14/4 statt 17/9). Das bedeutet: Evolution findet nicht nur größere, sondern
auch kompaktere Lösungen — ohne expliziten Effizienz-Druck (`efficiency_weight=0`
nach Stagnation).

## Beobachtungen

- **Species-Fix bewährt sich:** Stabil bei 5 für ~28 der 30 Minuten (kurze Delle
  auf 23 bei t=2min durch Bootstrap-Explosion, schnell gemergt)
- **Netzwerk wächst nicht monoton:** Trotz `efficiency_weight=0` nach Stagnation
  wurde die beste Lösung mit einem kleineren Netz gefunden. Evolution selbst
  macht Druck auf Kompaktheit wenn kleinere Strukturen besser passen.
- **LunarLander nicht gelöst in 30 Min** — deutlich schwerer als Acrobot/MountainCar.
  200 würde vermutlich 60–120+ Min oder größere Population brauchen.
