"""Structured training logger — writes rotating logs to ~/.yane/logs/."""
from __future__ import annotations
import logging
import logging.handlers
import os
from pathlib import Path


_LOG_DIR = Path.home() / ".yane" / "logs"
_LOG_FILE = _LOG_DIR / "training.log"
_MAX_BYTES = 5 * 1_048_576   # 5 MB per file
_BACKUP_COUNT = 3             # keep training.log + 3 rotated copies


def get_logger(name: str = "yane") -> logging.Logger:
    """Return the shared yane logger, creating it on first call."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def log_path() -> Path:
    return _LOG_FILE
