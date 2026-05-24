# YANE Benchmarks

Langzeittests zur Verifikation der Evolutionslogik. Dienen als Referenz bei
zukünftigen Änderungen, um Regressionen und Verbesserungen zu erkennen.

## Dateistruktur

`YYYY-MM-DD_<umgebung>[_notiz].md` — ein File pro Testsession.

## Skripte

```bash
python -m yane.benchmarks.run_suite --fast
python -m yane.benchmarks.forward_bench --sizes 10 50 200 1000
python -m yane.benchmarks.compare_lamarck_modes --env Acrobot-v1 --modes hc nes sa cma_es
```

- `run_suite.py`: Standard-Suite über mehrere Seeds.
- `forward_bench.py`: Microbenchmarks für `forward()`.
- `compare_lamarck_modes.py`: Vergleich von Hill-Climbing, NES, SA und CMA-ES auf
  Acrobot/LunarLander.

## Standardkonfiguration (sofern nicht anders angegeben)

| Parameter | Wert |
|---|---|
| Pop-Größe | 150 |
| n\_eval | 3 (mean) |
| Laufzeit | 30 Min |
| Branch | major-update |

## Laufende Ergebnisse (Kurzübersicht)

| Datum | Umgebung | Dauer | Gelöst | Best Fitness | Commit |
|---|---|---|---|---|---|
| 2026-05-19 | CartPole-v1 | 1 Min | Ja (Iter ~2300) | 500.0 | vor benchmarks |
| 2026-05-19 | Taxi-v4 | 1 Min | Nein | -147.5 | vor benchmarks |
| 2026-05-19 | Acrobot-v1 | 30 Min | Ja (Iter 3269) | 11.804 | c7bef0d |
| 2026-05-19 | MountainCar Cont. | 3 Min | Ja (Iter 3907) | 10.514 | c7bef0d |
| 2026-05-19 | MountainCar Cont. | 30 Min | Ja (Iter 7255) | 10.514 | c7bef0d |
| 2026-05-19 | LunarLander-v3 | 30 Min | Nein | 135.54 | c7bef0d |
| 2026-05-19 | BipedalWalker-v3 | 30 Min | Ja (Iter ~4551, ~16 Min) | 4.67 | 515ca96 |
| 2026-05-21 | Blackjack-v1 | 3 Min | Nein | −0.0880 | 3ed1b0f |
