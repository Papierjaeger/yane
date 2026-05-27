import unittest

from yane.benchmarks.run_suite import (
    BenchmarkResult,
    RunResult,
    evaluate_gates,
    format_gate_report,
)


class TestBenchmarkGates(unittest.TestCase):
    def test_evaluate_gates_checks_success_fitness_and_time(self):
        result = BenchmarkResult(
            name="XOR",
            runs=[
                RunResult(0, True, 10, 1.0, -0.05, "target"),
                RunResult(1, False, 20, 2.0, -0.3, "max"),
            ],
        )

        checks = evaluate_gates([result], {
            "XOR": {
                "min_success_rate": 0.5,
                "min_best_fitness": -0.1,
                "max_mean_elapsed_s": 2.0,
            }
        })

        self.assertTrue(all(c.passed for c in checks))

    def test_gate_report_is_markdown(self):
        checks = evaluate_gates([], {"missing": {"min_success_rate": 1.0}})
        report = format_gate_report(checks)

        self.assertIn("# Benchmark Gate Report", report)
        self.assertIn("| missing | present |", report)


if __name__ == "__main__":
    unittest.main()
