"""
Logging configuration for LLM Quantization Research.

Architecture:
  - ``setup_unified_logger(log_dir)`` creates ONE pipeline-wide log file
    that captures every message from every module.  Called once by main.py.
  - ``setup_logger(name, log_dir)`` adds a per-module file handler ON TOP
    of the unified handler.  The unified log file continues to receive
    all messages, while the module-specific file receives only that module's
    output.  When called again for a different module, the previous
    per-module file handler is closed (but the unified handler stays).
  - ``get_logger(name)`` returns a named child that propagates to root.
  - Result: ONE unified log file for the whole pipeline + one smaller
    log file per module for focused debugging.

For standalone scripts (``scripts/run_module*.py``), calling
``setup_logger()`` without ``setup_unified_logger()`` works identically
to before — the module log file IS the single log file.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# ── Module-level state ──────────────────────────────────────────────────
_FILE_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_UNIFIED_HANDLER: Optional[logging.FileHandler] = None
_MODULE_HANDLER: Optional[logging.FileHandler] = None


class _ConsoleFormatter(logging.Formatter):
    """Minimal colour-coded formatter for terminal output."""

    _COLOURS = {
        "DEBUG":    "\033[36m",    # Cyan
        "INFO":     "\033[32m",    # Green
        "WARNING":  "\033[33m",    # Yellow
        "ERROR":    "\033[31m",    # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelname, self._RESET)
        record.levelname = f"{colour}{record.levelname:<7}{self._RESET}"
        return super().format(record)


# Bootstrap: give the root logger a console handler so messages emitted
# before setup_logger() (e.g. at import time) are not silently dropped.
_root = logging.getLogger()
_root.setLevel(logging.DEBUG)
if not _root.handlers:
    _boot_ch = logging.StreamHandler(sys.stdout)
    _boot_ch.setLevel(logging.INFO)
    _boot_ch.setFormatter(_ConsoleFormatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S",
    ))
    _root.addHandler(_boot_ch)


def _suppress_noisy_libs() -> None:
    """Set 3rd-party library loggers to WARNING to keep logs readable."""
    for lib in ("transformers", "datasets", "huggingface_hub",
                "accelerate", "urllib3", "filelock",
                "httpcore", "httpx", "hf_transfer"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def setup_unified_logger(
    log_dir: str = "outputs",
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Create the pipeline-wide unified log file.

    Call this ONCE at the start of ``main.py``.  The file handler
    persists across all modules so that every message is captured
    in a single ``pipeline_<timestamp>.log``.
    """
    global _UNIFIED_HANDLER

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove the bootstrap console handler and any stale handlers
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    # ── Unified file handler (persists for the whole run) ───────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        log_path / f"pipeline_{ts}.log", encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FILE_FMT)
    _UNIFIED_HANDLER = fh
    root.addHandler(fh)

    # ── Console handler ─────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(_ConsoleFormatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S",
    ))
    root.addHandler(ch)

    _suppress_noisy_libs()
    return logging.getLogger("pipeline")


def setup_logger(
    name: str,
    log_dir: str = "outputs/logs",
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Add a per-module file handler.

    If a unified handler already exists (main.py pipeline), this ADDS
    a module-specific file handler without removing the unified one.

    If NO unified handler exists (standalone script), this behaves like
    the old setup_logger — one file + one console handler.
    """
    global _MODULE_HANDLER

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # ── Close previous per-module handler (if any) ──────────────────────
    if _MODULE_HANDLER is not None:
        root.removeHandler(_MODULE_HANDLER)
        _MODULE_HANDLER.close()
        _MODULE_HANDLER = None

    # ── If NO unified handler → standalone script mode ──────────────────
    if _UNIFIED_HANDLER is None:
        # Clear everything (same as old behaviour)
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(_ConsoleFormatter(
            "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S",
        ))
        root.addHandler(ch)

    # ── Per-module file handler ─────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mfh = logging.FileHandler(
        log_path / f"{name}_{ts}.log", encoding="utf-8"
    )
    mfh.setLevel(logging.DEBUG)
    mfh.setFormatter(_FILE_FMT)
    _MODULE_HANDLER = mfh
    root.addHandler(mfh)

    _suppress_noisy_libs()
    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.  No handlers are attached — all messages
    propagate to the root logger configured by ``setup_logger()``.
    """
    return logging.getLogger(name)
