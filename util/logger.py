"""Structured training logger — writes timestamped logs to <project>/logs/.

Directory layout::

    logs/
      <category>/
        2026-05-24_14-10-00/
          run.log          — structured training log
          crash.log        — faulthandler output (stack trace on segfault/SIGABRT)
          config.json
          best_genome.json
          fitness_history.csv
        2026-05-24_15-30-00/
          ...

Each ``train()`` invocation creates one timestamped run directory.  Old runs
are automatically cleaned up (keeps at most ``max_log_dirs`` per category).
"""
from __future__ import annotations

import faulthandler
import json
import logging
import logging.handlers
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

# ---------------------------------------------------------------------------
# Project-relative log root — logs/ inside the yane package directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_ROOT = _PROJECT_ROOT / "logs"

# Global defaults (mutable so GUI / config can change them).
log_root: Path = _DEFAULT_LOG_ROOT
max_log_dirs: int = 20

# Shared logger instance (lazily created by get_logger()).
_logger: logging.Logger | None = None

# Open file handle for faulthandler (kept alive so the OS doesn't close it).
_crash_log_fh: IO[str] | None = None

# Whether sys.excepthook has already been patched.
_excepthook_installed: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(name: str, log_root_override: str | Path | None = None) -> Path:
    """Create a timestamped log directory and configure the file handler.

    Returns the absolute path to the new run directory
    (``<root>/<name>/<timestamp>/``).

    Args:
        name: Category name (e.g. ``"xor"``, ``"cartpole"``, ``"benchmarks"``).
        log_root_override: If given, used instead of the global ``log_root``.
            Always relative to the YANE project directory.  Pass ``None`` to
            use the default.
    """
    root = _resolve_root(log_root_override)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    # Sanitize each path component: colons are forbidden in NTFS/SMB directory names.
    safe_name = "/".join(_sanitize_dir_component(p) for p in name.split("/"))
    run_dir = root / safe_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure the shared logger to write into this run directory.
    _setup_file_handler(run_dir / "run.log")

    # Redirect faulthandler to crash.log and install the excepthook.
    _enable_crash_logging(run_dir)
    _install_excepthook()

    # Auto-cleanup old runs in this category.
    _cleanup_old_runs(root, safe_name)

    return run_dir


def get_logger(name: str = "yane") -> logging.Logger:
    """Return the shared yane logger, creating it on first call.

    If ``setup_logging()`` has not been called yet the logger will have no
    file handler — only console output (when configured externally).
    """
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    # No file handler yet — setup_logging() adds one.
    return _logger


def log_info(msg: str, *args: object) -> None:
    """Convenience wrapper for ``get_logger().info(msg, *args)``."""
    get_logger().info(msg, *args)


def log_warning(msg: str, *args: object) -> None:
    """Convenience wrapper for ``get_logger().warning(msg, *args)``."""
    get_logger().warning(msg, *args)


def log_error(msg: str, *args: object) -> None:
    """Convenience wrapper for ``get_logger().error(msg, *args)``."""
    get_logger().error(msg, *args)


def log_path() -> Path | None:
    """Return the path of the current log file, or *None* if unconfigured."""
    if _logger is None or not _logger.handlers:
        return None
    for h in _logger.handlers:
        if isinstance(h, logging.FileHandler):
            return Path(h.baseFilename)
    return None


# ---------------------------------------------------------------------------
# Helpers for writing structured run artefacts
# ---------------------------------------------------------------------------

def write_json(path: Path, data: object) -> None:
    """Atomically write *data* as JSON to *path*."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def write_csv(path: Path, header: str, row: str) -> None:
    """Append *row* to CSV *path*; write *header* if the file is new."""
    if not path.exists():
        path.write_text(header + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(row + "\n")


def write_jsonl(path: Path, data: dict) -> None:
    """Append *data* as a JSON line to *path*."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, default=str) + "\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enable_crash_logging(run_dir: Path) -> None:
    """Open crash.log in *run_dir* and redirect faulthandler to it.

    faulthandler writes a C-level + Python-level traceback on SIGSEGV,
    SIGFPE, SIGABRT, SIGBUS, and SIGILL — surviving segfaults that kill
    the process before any Python exception handler can run.
    """
    global _crash_log_fh
    if _crash_log_fh is not None:
        try:
            _crash_log_fh.close()
        except OSError:
            pass

    try:
        crash_path = run_dir / "crash.log"
        _crash_log_fh = crash_path.open("w", encoding="utf-8")
        _crash_log_fh.write(
            f"# YANE crash log\n"
            f"# created  : {datetime.now(timezone.utc).isoformat()}\n"
            f"# python   : {sys.version}\n"
            f"# platform : {platform.platform()}\n"
            f"# pid      : {os.getpid()}\n"
            f"# run_dir  : {run_dir}\n"
            f"#\n"
            f"# A traceback below means the process crashed after this point.\n"
            f"# An empty file (only this header) means a clean exit.\n\n"
        )
        _crash_log_fh.flush()
        faulthandler.enable(file=_crash_log_fh, all_threads=True)
    except Exception:
        pass  # never break training because crash logging failed


def _install_excepthook() -> None:
    """Patch sys.excepthook to log uncaught Python exceptions to run.log."""
    global _excepthook_installed
    if _excepthook_installed:
        return
    _excepthook_installed = True

    _orig = sys.excepthook

    def _hook(exc_type: type, exc_value: BaseException, exc_tb) -> None:
        try:
            tb_lines = "".join(traceback.format_tb(exc_tb))
            get_logger().critical(
                "Uncaught exception — %s: %s\nTraceback:\n%s",
                exc_type.__name__, exc_value, tb_lines,
            )
        except Exception:
            pass
        _orig(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def _resolve_root(override: str | Path | None) -> Path:
    if override is not None:
        p = Path(override)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p.resolve()
    return log_root.resolve()


def _setup_file_handler(log_file: Path) -> None:
    logger = get_logger()

    # Remove any previous file handler so the logger always points at the
    # current run's log.
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1_048_576, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def _sanitize_dir_component(part: str) -> str:
    """Replace characters forbidden in NTFS/SMB directory names with '-'."""
    # Colons conflict with NTFS alternate data streams and break Samba.
    return part.replace(":", "-")


def _cleanup_old_runs(root: Path, category: str) -> None:
    """Remove old timestamped directories in *root/category* so at most
    ``max_log_dirs`` remain (oldest first)."""
    cat_dir = root / category
    if not cat_dir.is_dir():
        return

    dirs = sorted(
        [d for d in cat_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,  # ISO timestamps sort chronologically
    )
    excess = len(dirs) - max_log_dirs
    for d in dirs[:excess]:
        shutil.rmtree(d, ignore_errors=True)

