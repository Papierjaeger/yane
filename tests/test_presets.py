import tempfile
import unittest
from pathlib import Path

from yane.util.presets import list_presets, load_preset, save_preset, ExperimentPreset


class TestPresets(unittest.TestCase):
    def test_save_load_and_list_presets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = save_preset("My Preset", {"population_size": 42}, "desc", preset_dir=root)
            loaded = load_preset(path)
            all_presets = list_presets(root)

            self.assertEqual(loaded.name, "My Preset")
            self.assertEqual(loaded.config["population_size"], 42)
            self.assertEqual(len(all_presets), 1)

    def test_adaptive_policies_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ap = {
                "adaptive_controller": True,
                "operator_scheduler": False,
                "interspecies_mode": "Adaptiv",
                "interspecies_min_rate": 0.01,
                "interspecies_max_rate": 0.15,
                "lamarck_schedule": "Adaptiv",
                "lamarck_optimizer": "NES",
                "lamarck_budget": 50,
            }
            path = save_preset(
                "Adaptive Test",
                {"population_size": 100},
                "test",
                adaptive_policies=ap,
                preset_dir=root,
            )
            loaded = load_preset(path)
            self.assertEqual(loaded.adaptive_policies["adaptive_controller"], True)
            self.assertEqual(loaded.adaptive_policies["operator_scheduler"], False)
            self.assertEqual(loaded.adaptive_policies["interspecies_mode"], "Adaptiv")
            self.assertAlmostEqual(loaded.adaptive_policies["interspecies_min_rate"], 0.01)
            self.assertAlmostEqual(loaded.adaptive_policies["interspecies_max_rate"], 0.15)
            self.assertEqual(loaded.adaptive_policies["lamarck_schedule"], "Adaptiv")
            self.assertEqual(loaded.adaptive_policies["lamarck_optimizer"], "NES")
            self.assertEqual(loaded.adaptive_policies["lamarck_budget"], 50)

    def test_adaptive_policies_empty_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = save_preset("No Policies", {"population_size": 50}, preset_dir=root)
            loaded = load_preset(path)
            self.assertEqual(loaded.adaptive_policies, {})

    def test_to_json_omits_empty_adaptive_policies(self):
        preset = ExperimentPreset(name="X", description="", config={"a": 1})
        data = preset.to_json()
        self.assertNotIn("adaptive_policies", data)

    def test_to_json_includes_adaptive_policies_when_set(self):
        preset = ExperimentPreset(
            name="X", description="", config={},
            adaptive_policies={"adaptive_controller": True},
        )
        data = preset.to_json()
        self.assertIn("adaptive_policies", data)
        self.assertTrue(data["adaptive_policies"]["adaptive_controller"])

    def test_schema_version_in_json(self):
        preset = ExperimentPreset(name="V", description="", config={})
        data = preset.to_json()
        self.assertIn("schema_version", data)
        self.assertIsInstance(data["schema_version"], int)

    def test_load_preset_without_adaptive_policies_key(self):
        """Old v1 preset files (without adaptive_policies) load cleanly."""
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v1 = {"name": "OldPreset", "description": "v1", "config": {"population_size": 80}}
            path = root / "old.json"
            path.write_text(json.dumps(v1), encoding="utf-8")
            loaded = load_preset(path)
            self.assertEqual(loaded.name, "OldPreset")
            self.assertEqual(loaded.adaptive_policies, {})

    def test_builtin_adaptive_presets_load_correctly(self):
        """The 4 bundled adaptive profile presets parse and have expected keys."""
        from yane.util.presets import PRESET_DIR
        for name in (
            "adaptive_konservativ",
            "adaptive_balanciert",
            "adaptive_aggressiv",
            "adaptive_analysefreundlich",
        ):
            preset = load_preset(name, preset_dir=PRESET_DIR)
            self.assertIn("adaptive_controller", preset.adaptive_policies, name)
            self.assertIn("operator_scheduler", preset.adaptive_policies, name)
            self.assertIn("interspecies_mode", preset.adaptive_policies, name)
            self.assertIn("lamarck_budget", preset.adaptive_policies, name)

    def test_konservativ_preset_has_adaptive_ctrl_false(self):
        from yane.util.presets import PRESET_DIR
        preset = load_preset("adaptive_konservativ", preset_dir=PRESET_DIR)
        self.assertFalse(preset.adaptive_policies["adaptive_controller"])
        self.assertFalse(preset.adaptive_policies["operator_scheduler"])

    def test_aggressiv_preset_has_high_max_rate(self):
        from yane.util.presets import PRESET_DIR
        preset = load_preset("adaptive_aggressiv", preset_dir=PRESET_DIR)
        self.assertGreater(preset.adaptive_policies["interspecies_max_rate"], 0.2)

    def test_analysefreundlich_preset_has_budget(self):
        from yane.util.presets import PRESET_DIR
        preset = load_preset("adaptive_analysefreundlich", preset_dir=PRESET_DIR)
        self.assertGreater(preset.adaptive_policies["lamarck_budget"], 0)


if __name__ == "__main__":
    unittest.main()
