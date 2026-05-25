# Langzeittest aller Beispiele

**Datum:** 2026-05-25T18:10:09
**Maximale Laufzeit pro Run:** 10.0 Min
**Wiederholungen bei geloestem Beispiel:** 3

## Zusammenfassung

| Beispiel | Runs | Geloest | Best | Mittel | Output-Check | Diagnose |
|---|---:|---:|---:|---:|---|---|
| Cliff Walking | 3 | 2 | 145.6042 | 62.4819 | n/a | loest, aber seed-abhaengig |
| Taxi | 1 | 0 | 32.1313 | 32.1313 | n/a | verbessert, aber Target noch 17.87 entfernt |

## Details

### Cliff Walking

- Erfolg: 2/3 Runs, beste Fitness 145.604, mittlere Fitness 62.4819.
- Diagnose: loest, aber seed-abhaengig.
- Seed 0 Run 1: geloest, Fitness 128.508, 256 Iter, 17.8s, Stop target_reached.
- Seed 1 Run 2: geloest, Fitness 145.604, 2405 Iter, 162.7s, Stop target_reached.
- Seed 2 Run 3: nicht geloest, Fitness -86.6667, 10192 Iter, 600.0s, Stop external.
- Empfehlung: Defaults robuster machen: mehr Population oder mehr strukturelle Diversitaet.

### Taxi

- Erfolg: 0/1 Runs, beste Fitness 32.1313, mittlere Fitness 32.1313.
- Diagnose: verbessert, aber Target noch 17.87 entfernt.
- Seed 0 Run 1: nicht geloest, Fitness 32.1313, 7454 Iter, 600.1s, Stop external.
- Empfehlung: Laenger laufen lassen oder Ziel/Reward-Shaping anhand stabiler Mehrfachlaeufe kalibrieren.
