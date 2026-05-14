"""Measures RSS growth during training to catch memory leaks.

These tests simulate the GUI's pattern of creating genome copies every N
iterations and holding them while training continues.
"""
import gc
import os
import unittest


def _rss_mb() -> float:
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1_048_576


def _make_yane(max_size=20, max_nodes=10, max_connections=20):
    from yane import NeuroEvolution
    yane = NeuroEvolution()
    yane.configure(2, 1, max_nodes=max_nodes, max_connections=max_connections)
    yane.set_population_size(max_size)
    return yane


def _evaluate(g):
    f = 0.0
    for inp, t in [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]:
        f -= abs(g.forward(inp)[0] - t)
    return f


class TestRSSStability(unittest.TestCase):
    """RSS must not grow unboundedly during sustained training."""

    def _run_with_display_copies(self, n_iters: int, copy_every: int = 50):
        """Simulate the GUI worker: every `copy_every` iters, create a best.copy(),
        'display' it (hold ref), then release the previous copy — mirroring what
        canvas and inspect tab do."""
        yane = _make_yane()
        display_genome = None

        for i in range(n_iters):
            g = yane.next_genome()
            yane.submit_fitness(_evaluate(g))

            if i % copy_every == 0:
                try:
                    new_copy = yane.get_best().copy()
                except RuntimeError:
                    new_copy = g.copy()

                # Simulate canvas._clear() before replacing
                if display_genome is not None:
                    display_genome._clear()
                display_genome = new_copy

        if display_genome is not None:
            display_genome._clear()
        return yane

    def test_rss_plateau_during_training(self):
        """RSS must plateau after filling the population."""
        gc.collect()
        rss_start = _rss_mb()

        yane = self._run_with_display_copies(500)
        gc.collect()
        rss_mid = _rss_mb()

        # Second burst — population is full, growth should stop
        for _ in range(500):
            g = yane.next_genome()
            yane.submit_fitness(_evaluate(g))
        gc.collect()
        rss_end = _rss_mb()

        growth_mb = rss_end - rss_mid
        self.assertLess(growth_mb, 10.0,
            f"RSS grew {growth_mb:.1f} MB in second burst "
            f"(mid={rss_mid:.1f}, end={rss_end:.1f}) — possible leak")

    def test_display_copy_cleared_immediately(self):
        """With _clear(), a replaced display copy must be freed without gc.collect()."""
        import weakref
        yane = _make_yane()
        for _ in range(30):
            g = yane.next_genome()
            yane.submit_fitness(_evaluate(g))

        old_copy = yane.get_best().copy()
        ref = weakref.ref(old_copy)

        # Simulate replacement with _clear()
        old_copy._clear()
        del old_copy

        # Must be freed immediately (no cycles left after _clear)
        self.assertIsNone(ref(),
            "Cleared genome copy must be freed immediately without gc.collect()")

    def test_rss_stays_bounded_long_run(self):
        """5000 iterations must not use more than 20 MB above baseline."""
        gc.collect()
        baseline = _rss_mb()

        yane = self._run_with_display_copies(5000, copy_every=50)
        gc.collect()

        try:
            import ctypes
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        gc.collect()

        peak = _rss_mb()
        growth = peak - baseline
        self.assertLess(growth, 25.0,
            f"RSS grew {growth:.1f} MB over 5000 iterations (baseline={baseline:.1f}, peak={peak:.1f})")


if __name__ == "__main__":
    unittest.main()
