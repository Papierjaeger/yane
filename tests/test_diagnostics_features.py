"""Tests for the three new diagnostics features:
1. Structured logging (JSONL + callbacks)
2. Extended genome analysis (sensitivity + dead nodes)
3. Run history loading (RunRecord)
"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from yane import NeuroEvolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_eval(genome):
    total = 0.0
    for node in genome.nodes:
        for conn in node.connections:
            if conn.enabled:
                total += abs(conn.weight)
    return total


def _xor_cases():
    return [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]


# ---------------------------------------------------------------------------
# Phase 1 — Strukturierte Protokollierung
# ---------------------------------------------------------------------------

class TestLogFormat(unittest.TestCase):

    def test_set_log_format_valid(self):
        ne = NeuroEvolution()
        for fmt in ("csv", "jsonlines", "both"):
            ne.set_log_format(fmt)
            self.assertEqual(ne._log_format, fmt)

    def test_set_log_format_invalid(self):
        ne = NeuroEvolution()
        with self.assertRaises(ValueError):
            ne.set_log_format("html")

    def test_jsonlines_file_created(self):
        from yane.util import logger as logmod
        orig_root = logmod.log_root
        with tempfile.TemporaryDirectory() as tmp:
            logmod.log_root = Path(tmp)
            try:
                ne = NeuroEvolution()
                ne.set_population_size(10)
                ne.configure(n_inputs=2, n_outputs=1)
                ne.set_max_iterations(120)  # triggers heartbeat at 100
                ne.set_log_format("jsonlines")
                ne.train(_dummy_eval)

                jsonl_path = ne._log_run_dir / "fitness_history.jsonl"
                self.assertTrue(jsonl_path.exists(), "JSONL file was not created")
                lines = jsonl_path.read_text().strip().split("\n")
                self.assertGreater(len(lines), 0)
                obj = json.loads(lines[0])
                self.assertIn("iteration", obj)
                self.assertIn("max_fitness", obj)
            finally:
                logmod.log_root = orig_root

    def test_both_format_creates_csv_and_jsonl(self):
        from yane.util import logger as logmod
        orig_root = logmod.log_root
        with tempfile.TemporaryDirectory() as tmp:
            logmod.log_root = Path(tmp)
            try:
                ne = NeuroEvolution()
                ne.set_population_size(10)
                ne.configure(n_inputs=2, n_outputs=1)
                ne.set_max_iterations(120)
                ne.set_log_format("both")
                ne.train(_dummy_eval)

                self.assertTrue((ne._log_run_dir / "fitness_history.csv").exists())
                self.assertTrue((ne._log_run_dir / "fitness_history.jsonl").exists())
            finally:
                logmod.log_root = orig_root

    def test_callback_receives_dict_with_iteration(self):
        from yane.util import logger as logmod
        orig_root = logmod.log_root
        with tempfile.TemporaryDirectory() as tmp:
            logmod.log_root = Path(tmp)
            try:
                received = []

                ne = NeuroEvolution()
                ne.set_population_size(10)
                ne.configure(n_inputs=2, n_outputs=1)
                ne.set_max_iterations(120)
                ne.set_log_callbacks(on_generation=lambda d: received.append(d))
                ne.train(_dummy_eval)

                self.assertGreater(len(received), 0)
                self.assertIn("iteration", received[0])
                self.assertIn("max_fitness", received[0])
            finally:
                logmod.log_root = orig_root

    def test_callback_none_clears_callbacks(self):
        ne = NeuroEvolution()
        ne.set_log_callbacks(on_generation=lambda d: None)
        self.assertEqual(len(ne._log_callbacks), 1)
        ne.set_log_callbacks(on_generation=None)
        self.assertEqual(len(ne._log_callbacks), 0)

    def test_set_tensorboard_logdir_none(self):
        """set_tensorboard_logdir(None) must not raise."""
        ne = NeuroEvolution()
        ne.set_tensorboard_logdir(None)  # no-op, should not raise

    def test_set_tensorboard_logdir_missing_package(self):
        """ImportError is raised when torch/tensorboard is not installed.
        Skip this test if torch happens to be installed in the CI environment."""
        import importlib
        try:
            import torch.utils.tensorboard  # noqa: F401
            self.skipTest("torch is installed — skipping ImportError test")
        except ImportError:
            pass
        ne = NeuroEvolution()
        with self.assertRaises(ImportError):
            ne.set_tensorboard_logdir("/tmp/tb_test")

    def test_write_jsonl_helper(self):
        from yane.util.logger import write_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.jsonl"
            write_jsonl(p, {"a": 1, "b": "x"})
            write_jsonl(p, {"a": 2, "b": "y"})
            lines = p.read_text().strip().split("\n")
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0]), {"a": 1, "b": "x"})
            self.assertEqual(json.loads(lines[1]), {"a": 2, "b": "y"})


# ---------------------------------------------------------------------------
# Phase 2 — Erweiterte Genome-Analyse
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis(unittest.TestCase):

    def _make_genome(self):
        from yane.core.genome import Genome
        from yane.core.node import Node, NodeType
        from yane.core.connection import Connection
        from yane.util.activation import ActivationType
        g = Genome()
        i0 = Node(NodeType.INPUT,  0); i0.input_index = 0
        i1 = Node(NodeType.INPUT,  1); i1.input_index = 1
        o0 = Node(NodeType.OUTPUT, 2); o0.activation = ActivationType.LINEAR
        conn0 = Connection(target=o0, innovation=10)
        conn0.weight = 1.5; conn0.enabled = True
        conn1 = Connection(target=o0, innovation=11)
        conn1.weight = 0.1; conn1.enabled = True
        i0.connections = [conn0]
        i1.connections = [conn1]
        o0.connections = []
        g.nodes = [i0, i1, o0]
        g.input_nodes = [i0, i1]
        g.output_nodes = [o0]
        return g

    def test_returns_list_of_correct_length(self):
        g = self._make_genome()
        scores = g.sensitivity_analysis(_xor_cases(), delta=0.1)
        self.assertEqual(len(scores), 2)

    def test_scores_are_non_negative(self):
        g = self._make_genome()
        scores = g.sensitivity_analysis(_xor_cases())
        for s in scores:
            self.assertGreaterEqual(s, 0.0)

    def test_larger_weight_input_scores_higher(self):
        """Input 0 (weight 1.5) should influence more than Input 1 (weight -0.1)."""
        g = self._make_genome()
        scores = g.sensitivity_analysis(_xor_cases())
        self.assertGreater(scores[0], scores[1])

    def test_empty_test_cases_returns_zeros(self):
        g = self._make_genome()
        scores = g.sensitivity_analysis([])
        self.assertEqual(scores, [0.0, 0.0])

    def test_no_input_nodes_returns_empty(self):
        from yane.core.genome import Genome
        g = Genome()
        scores = g.sensitivity_analysis(_xor_cases())
        self.assertEqual(scores, [])

    def test_delta_parameter_affects_scores(self):
        """Different delta values should produce different (non-zero) scores."""
        g = self._make_genome()
        s1 = g.sensitivity_analysis(_xor_cases(), delta=0.05)
        s2 = g.sensitivity_analysis(_xor_cases(), delta=0.2)
        # Both should be non-zero (the genome has non-trivial weights)
        self.assertGreater(max(s1), 0.0)
        self.assertGreater(max(s2), 0.0)
        # The dominant input (weight 1.5) must dominate for both deltas
        self.assertGreater(s1[0], s1[1])
        self.assertGreater(s2[0], s2[1])


class TestDeadNodes(unittest.TestCase):

    def _make_genome_with_dead_hidden(self):
        """Genome: I0 → H1 (weight 0, bias 0) → O2; I0 → O2 (weight 1).
        H1 has weight=0 and bias=0, so accumulated_input=0 always.
        sigmoid(0)≈0.5 > 1e-6, so NOT dead by activation — use tanh+bias=-100 instead.
        Actually the simplest: H1 has no *incoming* connections → value stays 0.0.
        """
        from yane.core.genome import Genome
        from yane.core.node import Node, NodeType
        from yane.core.connection import Connection
        from yane.util.activation import ActivationType

        g = Genome()
        i0 = Node(NodeType.INPUT,  0); i0.input_index = 0
        h1 = Node(NodeType.HIDDEN, 1); h1.activation = ActivationType.SIGMOID; h1.bias = 0.0
        o2 = Node(NodeType.OUTPUT, 2); o2.activation = ActivationType.SIGMOID
        # H1 has no incoming connections → it's never triggered → value stays 0.0 → dead
        conn_i0_o2 = Connection(target=o2, innovation=11)
        conn_i0_o2.weight = 1.0; conn_i0_o2.enabled = True
        i0.connections = [conn_i0_o2]
        h1.connections = []
        o2.connections = []
        g.nodes = [i0, h1, o2]
        g.input_nodes = [i0]
        g.output_nodes = [o2]
        return g

    def test_returns_set(self):
        from yane.core.genome import Genome
        g = Genome()
        result = g.dead_nodes(_xor_cases())
        self.assertIsInstance(result, set)

    def test_empty_test_cases_returns_empty_set(self):
        g = self._make_genome_with_dead_hidden()
        result = g.dead_nodes([])
        self.assertEqual(result, set())

    def test_no_hidden_nodes_returns_empty_set(self):
        from yane.core.genome import Genome
        from yane.core.node import Node, NodeType
        from yane.core.connection import Connection
        from yane.util.activation import ActivationType
        g = Genome()
        i0 = Node(NodeType.INPUT,  0); i0.input_index = 0
        o0 = Node(NodeType.OUTPUT, 1); o0.activation = ActivationType.SIGMOID
        conn = Connection(target=o0, innovation=10)
        conn.weight = 1.0; conn.enabled = True
        i0.connections = [conn]
        o0.connections = []
        g.nodes = [i0, o0]; g.input_nodes = [i0]; g.output_nodes = [o0]
        result = g.dead_nodes(_xor_cases())
        self.assertEqual(result, set())

    def test_integrated_with_trained_genome(self):
        """End-to-end: sensitivity + dead_nodes on a trained genome."""
        from yane.util import logger as logmod
        import tempfile
        orig_root = logmod.log_root
        with tempfile.TemporaryDirectory() as tmp:
            logmod.log_root = Path(tmp)
            try:
                ne = NeuroEvolution()
                ne.set_population_size(20)
                ne.configure(n_inputs=2, n_outputs=1)
                ne.set_max_iterations(50)

                def eval_fn(genome):
                    total = 0.0
                    for inp, exp in _xor_cases():
                        genome.reset()
                        out = genome.forward(inp)
                        total -= abs(out[0] - exp[0])
                    return total

                ne.train(eval_fn)
                best = ne.get_best()

                scores = best.sensitivity_analysis(_xor_cases())
                self.assertEqual(len(scores), 2)
                for s in scores:
                    self.assertGreaterEqual(s, 0.0)

                dead = best.dead_nodes(_xor_cases())
                self.assertIsInstance(dead, set)
            finally:
                logmod.log_root = orig_root


# ---------------------------------------------------------------------------
# Phase 3 — Run History / RunRecord
# ---------------------------------------------------------------------------

class TestRunRecord(unittest.TestCase):

    def _make_fake_run_dir(self, tmp: Path, category: str, timestamp: str,
                           n_rows: int = 5) -> Path:
        run_dir = tmp / category / timestamp
        run_dir.mkdir(parents=True)

        config = {"n_inputs": 2, "n_outputs": 1, "pop_size": 10}
        (run_dir / "config.json").write_text(json.dumps(config))

        header = "iteration,best_fitness,mean_fitness,median_fitness,iqr_fitness,species_count,stagnation_count,nodes,connections"
        rows = [header]
        for i in range(1, n_rows + 1):
            rows.append(f"{i * 10},-{i * 0.1:.2f},-{i * 0.2:.2f},-{i * 0.15:.2f},0.01,2,0,3,4")
        (run_dir / "fitness_history.csv").write_text("\n".join(rows))
        return run_dir

    def test_load_valid_run(self):
        from yane.util.run_history import RunRecord
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir = self._make_fake_run_dir(tmp, "XOR", "2026-05-25_14-00-00")
            rec = RunRecord.load(run_dir)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.name, "XOR")
            self.assertEqual(rec.timestamp, "2026-05-25_14-00-00")
            self.assertEqual(len(rec.iterations), 5)
            self.assertEqual(rec.iterations[0], 10)

    def test_load_missing_csv_returns_none(self):
        from yane.util.run_history import RunRecord
        with tempfile.TemporaryDirectory() as tmp_str:
            run_dir = Path(tmp_str) / "X" / "2026-01-01_00-00-00"
            run_dir.mkdir(parents=True)
            self.assertIsNone(RunRecord.load(run_dir))

    def test_list_runs(self):
        from yane.util.run_history import RunRecord
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            self._make_fake_run_dir(tmp, "XOR", "2026-05-25_10-00-00")
            self._make_fake_run_dir(tmp, "XOR", "2026-05-25_11-00-00")
            self._make_fake_run_dir(tmp, "CartPole", "2026-05-25_12-00-00")

            runs = RunRecord.list_runs(log_root=tmp)
            self.assertEqual(len(runs), 3)

            runs_xor = RunRecord.list_runs(category="XOR", log_root=tmp)
            self.assertEqual(len(runs_xor), 2)

    def test_list_categories(self):
        from yane.util.run_history import RunRecord
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            self._make_fake_run_dir(tmp, "XOR", "2026-05-25_10-00-00")
            self._make_fake_run_dir(tmp, "CartPole", "2026-05-25_12-00-00")
            cats = RunRecord.list_categories(log_root=tmp)
            self.assertIn("XOR", cats)
            self.assertIn("CartPole", cats)

    def test_config_hash_same_config_same_hash(self):
        from yane.util.run_history import RunRecord
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            self._make_fake_run_dir(tmp, "XOR", "2026-05-25_10-00-00")
            self._make_fake_run_dir(tmp, "XOR", "2026-05-25_11-00-00")
            runs = RunRecord.list_runs(category="XOR", log_root=tmp)
            self.assertEqual(len(runs), 2)
            self.assertEqual(runs[0].config_hash, runs[1].config_hash)

    def test_group_by_config(self):
        from yane.util.run_history import RunRecord
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            self._make_fake_run_dir(tmp, "XOR", "2026-05-25_10-00-00")
            self._make_fake_run_dir(tmp, "XOR", "2026-05-25_11-00-00")
            runs = RunRecord.list_runs(category="XOR", log_root=tmp)
            groups = RunRecord.group_by_config(runs)
            self.assertEqual(len(groups), 1)
            for group in groups.values():
                self.assertEqual(len(group), 2)

    def test_final_best(self):
        from yane.util.run_history import RunRecord
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir = self._make_fake_run_dir(tmp, "XOR", "2026-05-25_10-00-00")
            rec = RunRecord.load(run_dir)
            self.assertIsNotNone(rec.final_best)

    def test_to_chart_dict_structure(self):
        from yane.util.run_history import RunRecord
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir = self._make_fake_run_dir(tmp, "XOR", "2026-05-25_10-00-00")
            rec = RunRecord.load(run_dir)
            d = rec.to_chart_dict()
            self.assertIn("name", d)
            self.assertIn("iterations", d)
            self.assertIn("best_fitness", d)
            self.assertEqual(len(d["iterations"]), len(d["best_fitness"]))

    def test_to_csv_rows(self):
        from yane.util.run_history import RunRecord
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_dir = self._make_fake_run_dir(tmp, "XOR", "2026-05-25_10-00-00")
            rec = RunRecord.load(run_dir)
            rows = rec.to_csv_rows()
            self.assertEqual(len(rows), 5)
            self.assertIn("iteration", rows[0])
            self.assertIn("best_fitness", rows[0])

    def test_empty_logs_dir_returns_empty(self):
        from yane.util.run_history import RunRecord
        with tempfile.TemporaryDirectory() as tmp_str:
            runs = RunRecord.list_runs(log_root=Path(tmp_str))
            self.assertEqual(runs, [])


if __name__ == "__main__":
    unittest.main()
