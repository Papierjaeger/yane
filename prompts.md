## Benchmark Prompt

Es ist Zeit, weitere Langzeittests auszuführen.
Zuerst, wie genau es funktioniert:
- Es läuft Sequenziell ab.
- Jedes Beispiel muss getestet werden
- Maximale Laufzeit sind 30 Minuten
- Wenn das Beispiel vor der maximalen Laufzeit die Targetfitness erreicht hat, stelle bei den Beispielen mit erwartetem output sicher, dass diese erreicht wurde, oder ob das target fitness angepasst werden muss.
- Beispiele die vor der maximalen Laufzeit ihr Ziel erreicht haben, sollen mindestens 10 mal durchlaufen, um Abweichungen festzustellen.
- Eine umfassende Diagnose zu beispielen stellen, die stark stagnieren und sich kaum verbessern
- Mögliche Probleme verbessern -> Default Einstellungen für bessere Ergebnisse optimieren.

Falls bei einigen Beispielen der Entschluss kommt, dass es gewisse Features gibt, die man noch implementieren müsste, um das Problem zu lösen, trage diese in die Tasks.md oder implementiere diese direkt.

Am Ende, möchte ich einen Bericht. Denke ebenfalls daran die Benchmarks bei [benchmarks](benchmarks/) festzuhalten.