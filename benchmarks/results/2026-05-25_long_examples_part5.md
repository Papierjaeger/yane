# Langzeittest aller Beispiele

**Datum:** 2026-05-25T15:50:32
**Maximale Laufzeit pro Run:** 30.0 Min
**Wiederholungen bei geloestem Beispiel:** 10

## Zusammenfassung

| Beispiel | Runs | Geloest | Best | Mittel | Output-Check | Diagnose |
|---|---:|---:|---:|---:|---|---|
| Acrobot | 10 | 10 | 11.6145 | 11.2772 | n/a | stabil geloest |
| MountainCar (Continuous) | 10 | 10 | 10.4750 | 10.4600 | n/a | stabil geloest |
| MountainCar (Discrete) | 10 | 10 | 10.5371 | 10.5150 | n/a | stabil geloest |
| Pendulum | 4 | 3 | -6.2961 | -196.8563 | n/a | loest, aber seed-abhaengig |
| LunarLander | 1 | 0 | 66.5228 | 66.5228 | n/a | stagnierend, Abstand zum Target 133.5 |
| BipedalWalker | 1 | 0 | 51.5065 | 51.5065 | n/a | verbessert, aber Target noch 48.49 entfernt |
| CarRacing | 2 | 1 | 127.3050 | 72.6331 | n/a | loest, aber seed-abhaengig |
| Blackjack | 10 | 10 | -0.0300 | -0.0406 | n/a | stabil geloest |
| Cliff Walking | 1 | 0 | -150.0000 | -150.0000 | n/a | starke Stagnation / Fitness-Kollaps |
| Frozen Lake | 1 | 0 | 0.5000 | 0.5000 | n/a | starke Stagnation / Fitness-Kollaps |
| Taxi | 1 | 0 | -147.0000 | -147.0000 | n/a | starke Stagnation / Fitness-Kollaps |

## Details

### Acrobot

- Erfolg: 10/10 Runs, beste Fitness 11.6145, mittlere Fitness 11.2772.
- Diagnose: stabil geloest.
- Seed 0 Run 1: geloest, Fitness 11.0228, 1225 Iter, 36.3s, Stop target_reached.
- Seed 1 Run 2: geloest, Fitness 11.1413, 1947 Iter, 55.6s, Stop target_reached.
- Seed 2 Run 3: geloest, Fitness 11.2368, 10765 Iter, 293.2s, Stop target_reached.
- Seed 3 Run 4: geloest, Fitness 11.4208, 12602 Iter, 368.8s, Stop target_reached.
- Seed 4 Run 5: geloest, Fitness 11.2342, 15737 Iter, 460.2s, Stop target_reached.
- Seed 5 Run 6: geloest, Fitness 11.2819, 7849 Iter, 238.4s, Stop target_reached.
- Seed 6 Run 7: geloest, Fitness 11.6145, 2169 Iter, 66.2s, Stop target_reached.
- Seed 7 Run 8: geloest, Fitness 11.4748, 6284 Iter, 195.8s, Stop target_reached.
- Seed 8 Run 9: geloest, Fitness 11.0953, 590 Iter, 18.1s, Stop target_reached.
- Seed 9 Run 10: geloest, Fitness 11.2498, 7238 Iter, 221.4s, Stop target_reached.

### MountainCar (Continuous)

- Erfolg: 10/10 Runs, beste Fitness 10.475, mittlere Fitness 10.46.
- Diagnose: stabil geloest.
- Seed 0 Run 1: geloest, Fitness 10.4519, 416 Iter, 3.2s, Stop target_reached.
- Seed 1 Run 2: geloest, Fitness 10.4676, 124432 Iter, 1006.9s, Stop target_reached.
- Seed 2 Run 3: geloest, Fitness 10.4617, 144350 Iter, 1528.2s, Stop target_reached.
- Seed 3 Run 4: geloest, Fitness 10.4577, 408 Iter, 4.3s, Stop target_reached.
- Seed 4 Run 5: geloest, Fitness 10.4503, 1231 Iter, 12.0s, Stop target_reached.
- Seed 5 Run 6: geloest, Fitness 10.4588, 6547 Iter, 62.7s, Stop target_reached.
- Seed 6 Run 7: geloest, Fitness 10.4672, 226 Iter, 2.0s, Stop target_reached.
- Seed 7 Run 8: geloest, Fitness 10.475, 27764 Iter, 223.9s, Stop target_reached.
- Seed 8 Run 9: geloest, Fitness 10.4515, 91934 Iter, 1031.2s, Stop target_reached.
- Seed 9 Run 10: geloest, Fitness 10.4582, 55685 Iter, 343.1s, Stop target_reached.

### MountainCar (Discrete)

- Erfolg: 10/10 Runs, beste Fitness 10.5371, mittlere Fitness 10.515.
- Diagnose: stabil geloest.
- Seed 0 Run 1: geloest, Fitness 10.5084, 98085 Iter, 646.9s, Stop target_reached.
- Seed 1 Run 2: geloest, Fitness 10.5369, 5847 Iter, 25.0s, Stop target_reached.
- Seed 2 Run 3: geloest, Fitness 10.5124, 7909 Iter, 54.1s, Stop target_reached.
- Seed 3 Run 4: geloest, Fitness 10.503, 3228 Iter, 15.6s, Stop target_reached.
- Seed 4 Run 5: geloest, Fitness 10.5074, 7508 Iter, 45.5s, Stop target_reached.
- Seed 5 Run 6: geloest, Fitness 10.5143, 18122 Iter, 113.0s, Stop target_reached.
- Seed 6 Run 7: geloest, Fitness 10.5185, 64252 Iter, 466.6s, Stop target_reached.
- Seed 7 Run 8: geloest, Fitness 10.5371, 23197 Iter, 168.0s, Stop target_reached.
- Seed 8 Run 9: geloest, Fitness 10.5059, 86200 Iter, 678.4s, Stop target_reached.
- Seed 9 Run 10: geloest, Fitness 10.5059, 41213 Iter, 352.4s, Stop target_reached.

### Pendulum

- Erfolg: 3/4 Runs, beste Fitness -6.29611, mittlere Fitness -196.856.
- Diagnose: loest, aber seed-abhaengig.
- Seed 0 Run 1: geloest, Fitness -6.29611, 32356 Iter, 284.9s, Stop target_reached.
- Seed 1 Run 2: geloest, Fitness -139.997, 27853 Iter, 291.9s, Stop target_reached.
- Seed 2 Run 3: geloest, Fitness -188.903, 20758 Iter, 237.9s, Stop target_reached.
- Seed 3 Run 4: nicht geloest, Fitness -452.23, 137083 Iter, 1800.0s, Stop external.
- Empfehlung: Defaults robuster machen: mehr Population oder mehr strukturelle Diversitaet.

### LunarLander

- Erfolg: 0/1 Runs, beste Fitness 66.5228, mittlere Fitness 66.5228.
- Diagnose: stagnierend, Abstand zum Target 133.5.
- Seed 0 Run 1: nicht geloest, Fitness 66.5228, 132376 Iter, 1800.0s, Stop external.
- Empfehlung: Default-Population, QD-Druck oder Lamarck-Budget erhoehen; bei Sparse Reward Feature-Task einplanen.

### BipedalWalker

- Erfolg: 0/1 Runs, beste Fitness 51.5065, mittlere Fitness 51.5065.
- Diagnose: verbessert, aber Target noch 48.49 entfernt.
- Seed 0 Run 1: nicht geloest, Fitness 51.5065, 11653 Iter, 1800.0s, Stop external.
- Empfehlung: Laenger laufen lassen oder Ziel/Reward-Shaping anhand stabiler Mehrfachlaeufe kalibrieren.

### CarRacing

- Erfolg: 1/2 Runs, beste Fitness 127.305, mittlere Fitness 72.6331.
- Diagnose: loest, aber seed-abhaengig.
- Seed 0 Run 1: geloest, Fitness 127.305, 93 Iter, 296.4s, Stop target_reached.
- Seed 1 Run 2: nicht geloest, Fitness 17.9612, 398 Iter, 1801.2s, Stop external.
- Empfehlung: Defaults robuster machen: mehr Population oder mehr strukturelle Diversitaet.

### Blackjack

- Erfolg: 10/10 Runs, beste Fitness -0.03, mittlere Fitness -0.0406.
- Diagnose: stabil geloest.
- Seed 0 Run 1: geloest, Fitness -0.05, 207 Iter, 6.7s, Stop target_reached.
- Seed 1 Run 2: geloest, Fitness -0.036, 593 Iter, 22.3s, Stop target_reached.
- Seed 2 Run 3: geloest, Fitness -0.034, 143 Iter, 4.8s, Stop target_reached.
- Seed 3 Run 4: geloest, Fitness -0.048, 227 Iter, 7.6s, Stop target_reached.
- Seed 4 Run 5: geloest, Fitness -0.03, 1329 Iter, 47.6s, Stop target_reached.
- Seed 5 Run 6: geloest, Fitness -0.042, 429 Iter, 14.2s, Stop target_reached.
- Seed 6 Run 7: geloest, Fitness -0.036, 60 Iter, 1.8s, Stop target_reached.
- Seed 7 Run 8: geloest, Fitness -0.038, 316 Iter, 11.6s, Stop target_reached.
- Seed 8 Run 9: geloest, Fitness -0.048, 1629 Iter, 56.7s, Stop target_reached.
- Seed 9 Run 10: geloest, Fitness -0.044, 512 Iter, 19.8s, Stop target_reached.

### Cliff Walking

- Erfolg: 0/1 Runs, beste Fitness -150, mittlere Fitness -150.
- Diagnose: starke Stagnation / Fitness-Kollaps.
- Seed 0 Run 1: nicht geloest, Fitness -150, 139184 Iter, 1800.1s, Stop external.
- Empfehlung: Mehr explorative Struktur ist noetig: Inselmodell, Curiosity oder problem-spezifische State-/Reward-Zerlegung pruefen.

### Frozen Lake

- Erfolg: 0/1 Runs, beste Fitness 0.5, mittlere Fitness 0.5.
- Diagnose: starke Stagnation / Fitness-Kollaps.
- Seed 0 Run 1: nicht geloest, Fitness 0.5, 100468 Iter, 1800.1s, Stop external.
- Empfehlung: Mehr explorative Struktur ist noetig: Inselmodell, Curiosity oder problem-spezifische State-/Reward-Zerlegung pruefen.

### Taxi

- Erfolg: 0/1 Runs, beste Fitness -147, mittlere Fitness -147.
- Diagnose: starke Stagnation / Fitness-Kollaps.
- Seed 0 Run 1: nicht geloest, Fitness -147, 59994 Iter, 1800.0s, Stop external.
- Empfehlung: Mehr explorative Struktur ist noetig: Inselmodell, Curiosity oder problem-spezifische State-/Reward-Zerlegung pruefen.
