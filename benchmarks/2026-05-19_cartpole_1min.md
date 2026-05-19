# CartPole-v1 — 1 Min Kurztest

**Datum:** 2026-05-19  
**Branch/Commit:** major-update (vor Species-Fix)  
**Zweck:** Erster Sanity-Check der Evolutionslogik

## Konfiguration

| Parameter | Wert |
|---|---|
| Pop-Größe | 50 |
| n\_eval | 1 |
| max\_steps | 500 |
| Laufzeit | 1 Min |
| n\_inputs / n\_outputs | 4 / 2 |
| max\_nodes / max\_connections | 30 / 100 |

## Ergebnisse

| Zeit | Iter | Species | Stagnation | Best Fitness |
|---|---|---|---|---|
| 1 Min | ~14.800 | 12 | 12.000+ | 500.0 |

- **Gelöst:** Ja, bei Iter ~2300 (~15 Sek)
- **Eval-Zeit:** 0.1–0.3 ms (failed), 5–12 ms (500 Schritte)

## Beobachtungen

- Fitness springt von ~101 auf 500 (max\_steps-Cap) bei Iter 2300
- Species stabilisiert sich bei 12 — **noch ohne Species-Fix**
- `stagnation_count` wächst nach Solve unbegrenzt (bis 12.000+), da 500 der
  max\_steps-Cap ist und keine weitere Verbesserung möglich
- Dieses Verhalten ist korrekt: das System weiß nicht, dass 500 die Decke ist
- 225 Diversity-Injections ohne nennenswerten Effekt nach dem Solve
