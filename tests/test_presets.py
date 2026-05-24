import tempfile
import unittest
from pathlib import Path

from yane.util.presets import list_presets, load_preset, save_preset


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


if __name__ == "__main__":
    unittest.main()
