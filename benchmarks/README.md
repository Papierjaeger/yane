# YANE Benchmarks

Langzeittests zur Verifikation der Evolutionslogik. Dienen als Referenz bei
zukünftigen Änderungen, um Regressionen und Verbesserungen zu erkennen.

## Dateistruktur

`YYYY-MM-DD_<umgebung>[_notiz].md` — ein File pro Testsession.

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
