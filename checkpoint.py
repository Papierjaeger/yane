"""Command-line helpers for YANE checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from yane.evolution import checkpoint as _ckpt


def _read_config(path: str | Path) -> tuple[dict, str | None]:
    payload = _ckpt.read(path)
    cfg = payload.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return cfg, payload.get("config_hash")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect YANE checkpoints.")
    parser.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"),
                        help="show configuration differences between two checkpoints")
    args = parser.parse_args(argv)

    if args.diff:
        old_cfg, old_hash = _read_config(args.diff[0])
        new_cfg, _new_hash = _read_config(args.diff[1])
        report = _ckpt.compatibility_report(old_cfg, new_cfg, stored_hash=old_hash)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 1 if report["level"] == _ckpt.CompatibilityLevel.BREAKING.value else 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
