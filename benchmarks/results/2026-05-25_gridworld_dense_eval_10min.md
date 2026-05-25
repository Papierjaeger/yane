# Langzeittest aller Beispiele

**Datum:** 2026-05-25T17:58:58
**Maximale Laufzeit pro Run:** 10.0 Min
**Wiederholungen bei geloestem Beispiel:** 3

## Zusammenfassung

| Beispiel | Runs | Geloest | Best | Mittel | Output-Check | Diagnose |
|---|---:|---:|---:|---:|---|---|
| Cliff Walking | 1 | 0 | -154.3958 | -154.3958 | n/a | verbessert, aber Target noch 154.4 entfernt |
| Frozen Lake | 3 | 3 | 1.3055 | 1.1567 | n/a | stabil geloest |
| Taxi | 1 | 0 | -183.0625 | -183.0625 | n/a | starke Stagnation / Fitness-Kollaps |

## Details

### Cliff Walking

- Erfolg: 0/1 Runs, beste Fitness -154.396, mittlere Fitness -154.396.
- Diagnose: verbessert, aber Target noch 154.4 entfernt.
- Seed 0 Run 1: nicht geloest, Fitness -154.396, 10455 Iter, 600.0s, Stop external.
- Empfehlung: Laenger laufen lassen oder Ziel/Reward-Shaping anhand stabiler Mehrfachlaeufe kalibrieren.

### Frozen Lake

- Erfolg: 3/3 Runs, beste Fitness 1.30545, mittlere Fitness 1.15667.
- Diagnose: stabil geloest.
- Seed 0 Run 1: geloest, Fitness 1.30545, 862 Iter, 11.3s, Stop target_reached.
- Seed 1 Run 2: geloest, Fitness 0.859091, 3791 Iter, 16.5s, Stop target_reached.
- Seed 2 Run 3: geloest, Fitness 1.30545, 13312 Iter, 416.4s, Stop target_reached.

### Taxi

- Erfolg: 0/1 Runs, beste Fitness -183.062, mittlere Fitness -183.062.
- Diagnose: starke Stagnation / Fitness-Kollaps.
- Seed 0 Run 1: nicht geloest, Fitness -183.062, 7308 Iter, 600.1s, Stop external.
- Empfehlung: Mehr explorative Struktur ist noetig: Inselmodell, Curiosity oder problem-spezifische State-/Reward-Zerlegung pruefen.
