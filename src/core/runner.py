"""
ModuleRunner — base class for all pipeline modules.

Eliminates the repeated timing / try-except / status-dict / JSON-persistence
boilerplate that was copy-pasted across all 11 module runners.

Usage::

    class BF16BaselineModule(ModuleRunner):
        MODULE_NUM = 1
        MODULE_NAME = "BF16 Baseline"

        def execute(self):
            from src.quantization.bf16_baseline import BF16Baseline
            baseline = BF16Baseline(self.config)
            return baseline.run(cache_weights=True)

    # Then in pipeline.py:
    runner = BF16BaselineModule(config)
    result = runner.run()   # all boilerplate handled
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.core.logger import get_logger


class ModuleRunner(ABC):
    """Base class for all pipeline modules (research & mxmoe)."""

    MODULE_NUM: int = 0
    MODULE_NAME: str = "Unnamed Module"

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(f"module_{self.MODULE_NUM}")
        self.results: Dict[str, Any] = {
            "status": "UNKNOWN",
            "submodules": {},
            "total_time_sec": 0,
        }

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Template method: header → execute → status → persist."""
        t_start = time.time()
        self._log_header()
        try:
            self.execute()
            self._determine_status()
        except Exception as exc:
            self.results["status"] = "✗ FAILED"
            self.results["error"] = str(exc)
            self.logger.error(
                f"Module {self.MODULE_NUM} FAILED: {exc}", exc_info=True
            )
        finally:
            self.results["total_time_sec"] = round(time.time() - t_start, 2)
            self._persist_summary()
            self._log_footer()
        return self.results

    # ── Subclass implements this ──────────────────────────────────────────

    @abstractmethod
    def execute(self) -> None:
        """Run the actual module logic. Call self.run_submodule() for each step."""

    # ── Submodule helper ─────────────────────────────────────────────────

    def run_submodule(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        critical: bool = False,
    ) -> Any:
        """
        Run a submodule with standard timing, error handling, and status tracking.

        Args:
            name: Human-readable submodule name (used as dict key).
            fn: Zero-arg callable that does the work.
            critical: If True, re-raise on failure (abort module).

        Returns:
            Whatever ``fn()`` returns, or None on failure.
        """
        self.logger.info(f"  {name}")
        t = time.time()
        try:
            result = fn()
            elapsed = round(time.time() - t, 2)
            self.results["submodules"][name] = {
                "status": "✓ COMPLETED",
                "time_sec": elapsed,
            }
            self.logger.info(f"  [{name}] ✓ COMPLETED ({elapsed:.1f}s)")
            return result
        except Exception as exc:
            elapsed = round(time.time() - t, 2)
            self.results["submodules"][name] = {
                "status": "✗ FAILED",
                "error": str(exc),
                "time_sec": elapsed,
            }
            self.logger.error(f"  [{name}] ✗ FAILED: {exc}", exc_info=True)
            if critical:
                raise
            return None

    # ── Internals ────────────────────────────────────────────────────────

    def _log_header(self) -> None:
        self.logger.info("")
        self.logger.info("-" * 60)
        self.logger.info(
            f"  MODULE {self.MODULE_NUM}: {self.MODULE_NAME}"
        )
        self.logger.info("-" * 60)

    def _log_footer(self) -> None:
        completed = [
            k for k, v in self.results["submodules"].items()
            if str(v.get("status", "")).startswith("✓")
        ]
        failed = [
            k for k, v in self.results["submodules"].items()
            if str(v.get("status", "")).startswith("✗")
        ]
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(
            f"Module {self.MODULE_NUM}: "
            f"{len(completed)} succeeded, {len(failed)} failed"
        )
        if completed:
            self.logger.info(f"  ✓ {', '.join(completed)}")
        if failed:
            self.logger.info(f"  ✗ {', '.join(failed)}")
        self.logger.info(
            f"  Total time: {self.results['total_time_sec']:.1f}s"
        )
        self.logger.info("=" * 60)

    def _determine_status(self) -> None:
        """Set overall status based on submodule outcomes."""
        subs = self.results["submodules"]
        if not subs:
            # Module didn't use run_submodule — assume success if no error
            self.results["status"] = "✓ COMPLETED"
            return
        completed = [
            k for k, v in subs.items()
            if str(v.get("status", "")).startswith("✓")
        ]
        failed = [
            k for k, v in subs.items()
            if str(v.get("status", "")).startswith("✗")
        ]
        if not failed:
            self.results["status"] = "✓ COMPLETED"
        elif completed:
            self.results["status"] = "⚠ PARTIAL"
        else:
            self.results["status"] = "✗ FAILED"

    def _persist_summary(self) -> None:
        """Write module_N_summary.json to the results directory."""
        try:
            results_dir = Path(self.config.output.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            summary_path = results_dir / f"module_{self.MODULE_NUM}_summary.json"
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(self.results, fh, indent=2, default=str)
            self.logger.info(f"  Summary: {summary_path}")
        except Exception as exc:
            self.logger.warning(f"  Could not write summary: {exc}")
