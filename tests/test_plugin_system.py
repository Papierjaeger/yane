"""Tests for the evaluator plugin system (ExamplePlugin, register_example, auto-load)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from yane.gui.examples import (
    ExampleConfig,
    PLUGIN_EXAMPLES,
    load_examples,
    load_plugins_from_directory,
    register_example,
)


def _dummy_make_eval():
    def eval_fn(genome):
        return 1.0
    return eval_fn


class TestPluginRegistration(unittest.TestCase):

    def setUp(self):
        self._saved = list(PLUGIN_EXAMPLES)
        PLUGIN_EXAMPLES.clear()

    def tearDown(self):
        PLUGIN_EXAMPLES.clear()
        PLUGIN_EXAMPLES.extend(self._saved)

    def test_register_plugin_appears_in_load_examples(self):
        plugin = ExampleConfig(
            name="TestPlugin",
            description="A test plugin.",
            n_inputs=2, n_outputs=1,
            max_nodes=10, max_connections=20,
            make_eval=_dummy_make_eval,
            target_fitness=0.0,
            category="Plugins",
        )
        register_example(plugin)
        examples = load_examples()
        names = [ex.name for ex in examples]
        self.assertIn("TestPlugin", names)

    def test_register_via_neuroevolution_api(self):
        from yane import NeuroEvolution
        plugin = ExampleConfig(
            name="NEAPI_Plugin",
            description="Registered via NeuroEvolution API.",
            n_inputs=1, n_outputs=1,
            max_nodes=5, max_connections=10,
            make_eval=_dummy_make_eval,
            target_fitness=0.0,
            category="API",
        )
        NeuroEvolution.register_example(plugin)
        examples = load_examples()
        names = [ex.name for ex in examples]
        self.assertIn("NEAPI_Plugin", names)

    def test_multiple_plugins(self):
        for i in range(3):
            register_example(ExampleConfig(
                name=f"Plugin_{i}",
                description=f"Plugin {i}",
                n_inputs=1, n_outputs=1,
                max_nodes=5, max_connections=10,
                make_eval=_dummy_make_eval,
                target_fitness=float(i),
                category="Multi",
            ))
        examples = load_examples()
        names = [ex.name for ex in examples]
        for i in range(3):
            self.assertIn(f"Plugin_{i}", names)

    def test_load_plugins_from_directory_imports_py_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin_dir = Path(tmp) / "plugins"
            plugin_dir.mkdir()
            # Create a valid plugin file
            plugin_file = plugin_dir / "my_env.py"
            plugin_file.write_text(
                "from yane.gui.examples import ExampleConfig\n"
                "def register(reg):\n"
                "    reg(ExampleConfig(\n"
                "        name='DirPlugin',\n"
                "        description='Loaded from directory.',\n"
                "        n_inputs=2, n_outputs=1,\n"
                "        max_nodes=10, max_connections=20,\n"
                "        make_eval=lambda: (lambda g: 1.0),\n"
                "        target_fitness=0.0,\n"
                "        category='Plugins',\n"
                "    ))\n"
            )
            count = load_plugins_from_directory(str(plugin_dir))
            self.assertEqual(count, 1)
            names = [ex.name for ex in PLUGIN_EXAMPLES]
            self.assertIn("DirPlugin", names)

    def test_load_plugins_skips_init_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin_dir = Path(tmp) / "plugins"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text("# empty")
            count = load_plugins_from_directory(str(plugin_dir))
            self.assertEqual(count, 0)

    def test_load_plugins_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            count = load_plugins_from_directory(str(tmp))
            self.assertEqual(count, 0)

    def test_plugin_preserves_builtin_examples(self):
        """Registering a plugin does not remove built-in examples."""
        plugin = ExampleConfig(
            name="PreserveTest",
            description="Should not clobber built-ins.",
            n_inputs=2, n_outputs=1,
            max_nodes=10, max_connections=20,
            make_eval=_dummy_make_eval,
            target_fitness=0.0,
            category="Plugins",
        )
        register_example(plugin)
        examples = load_examples()
        builtin_names = {ex.name for ex in examples}
        # At least the core dataset examples should be present
        self.assertIn("XOR", builtin_names)
        self.assertIn("Multiplication", builtin_names)
        self.assertIn("PreserveTest", builtin_names)


if __name__ == "__main__":
    unittest.main()
