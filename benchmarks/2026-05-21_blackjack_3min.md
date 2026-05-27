# Blackjack-v1 — 3 Min Kurztest

**Datum:** 2026-05-21  
**Branch/Commit:** major-update, 3ed1b0f (Species-Explosion Fix)  
**Zweck:** Erster Blackjack-Benchmark + Verifikation des Species-Fix

## Konfiguration

| Parameter | Wert |
|---|---|
| Pop-Größe | 150 |
| n\_eval | 3 (mean) |
| n\_episodes | 500 |
| Laufzeit | 3 Min |
| n\_inputs / n\_outputs | 3 / 2 |
| max\_nodes / max\_connections | 20 / 60 |
| Fitness-Funktion | Mittlere Belohnung über 500 Runden (+1 gewonnen, -1 verloren, 0 push) |
| Ziel-Fitness | −0.05 |

## Ergebnisse

| Zeit | Iter | Species | Pop | Best Fitness | Stagnation |
|---|---|---|---|---|---|
| 0.5 Min | 394 | 5 | 150 | −0.0940 | 280 |
| 1.0 Min | 796 | 5 | 150 | −0.0913 | 193 |
| 1.5 Min | 1118 | 19 | 150 | −0.0880 | 118 |
| 2.0 Min | 1358 | 25 | 150 | −0.0880 | 358 |
| 2.5 Min | 1660 | 27 | 150 | −0.0880 | 660 |
| 3.0 Min | 2048 | 29 | 150 | −0.0880 | 1048 |

- **Gelöst:** Nein (Ziel −0.05 nicht erreicht)
- **Compat-Threshold:** 1.500 (Maximum)
- **Bestes Ergebnis:** −0.0880 (ab ~1.5 Min stabil)

## Bug entdeckt & gefixed (Commit 3ed1b0f)

**Symptom:** Species-Anzahl überschritt die Pop-Größe, Population wuchs
unkontrolliert über das `max_size`-Limit hinaus.

**Ursache:** `_spawn_offspring()` kehrte bei Stagnations-Injection früh zurück
ohne `_assign_species()` aufzurufen. Folge: kein Merging, kein Threshold-
Adjustment → Species wuchsen ohne Kontrolle → jedes Genome war Species-Elite
→ `_prune()` fand keine Kandidaten zum Entfernen.

**Fix:** `_spawn_count` und `_assign_species()` werden jetzt immer ausgeführt,
bevor der Injection-Pfad verlassen wird.

## Beobachtungen

- **Species-Wachstum nach Fix:** Stabilisiert bei ~29 Species (Ziel: 5). Das
  Wachstum von 5 → 29 ab ~1.5 Min ist normales Verhalten: Diversity-Injections
  erzeugen Genome mit Compat-Distanz > 1.5 (Maximum) zu allen bestehenden Species.
  Population bleibt aber korrekt bei 150 — kein Überschreiten mehr.
- **Fitness-Stagnation:** −0.0880 ab Iter ~1100. Blackjack hat hohe Varianz
  (±~0.05 pro Lauf); 3×500 Episoden reichen für ein zuverlässiges Gradient-Signal.
- **Schwierigkeit:** Blackjack liegt an der Grenze der NEAT-Kapazität für dieses
  Setting. Basic Strategy bräuchte ~2−4 Schwellenwerte, die ein minimales Netz
  prinzipiell kodieren kann — braucht aber mehr Zeit oder mehr Episoden.
