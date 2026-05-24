"""Experiment preset loading/saving for GUI and API."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PRESET_DIR = Path(__file__).resolve().parents[1] / "presets"


@dataclass
class ExperimentPreset:
    name: str
    description: str
    config: dict
    path: Path | None = None

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config,
        }


def list_presets(preset_dir: Path | None = None) -> list[ExperimentPreset]:
    root = preset_dir or PRESET_DIR
    if not root.exists():
        return []
    presets: list[ExperimentPreset] = []
    for path in sorted(root.glob("*.json")):
        presets.append(load_preset(path))
    return presets


def load_preset(path_or_name: str | Path, preset_dir: Path | None = None) -> ExperimentPreset:
    path = Path(path_or_name)
    if not path.suffix:
        root = preset_dir or PRESET_DIR
        path = root / f"{path_or_name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentPreset(
        name=data["name"],
        description=data.get("description", ""),
        config=dict(data.get("config", {})),
        path=path,
    )


def save_preset(
    name: str,
    config: dict,
    description: str = "",
    preset_dir: Path | None = None,
) -> Path:
    root = preset_dir or PRESET_DIR
    root.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    path = root / f"{slug or 'preset'}.json"
    payload = ExperimentPreset(name=name, description=description, config=config).to_json()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
