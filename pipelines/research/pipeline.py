#!/usr/bin/env python3
"""
Research Pipeline — Clean orchestrator.

Replaces the legacy research orchestrator. All module logic is in modules.py.

Usage (from project root):
    python -m pipelines.research.pipeline                          # All modules
    python -m pipelines.research.pipeline --module 1               # BF16 only
    python -m pipelines.research.pipeline --module 2 --quantizer gptq
    python -m pipelines.research.pipeline --config configs/research.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from src.core.config import load_config
from src.core.logger import setup_unified_logger, setup_logger, get_logger
from src.core.device import get_device_info, set_primary_cuda_device
from src.core.paths import scope_config_to_module

from pipelines.research.modules import MODULE_MAP, MODULE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Quantisation Research Pipeline — sarvamai/sarvam-30b"
    )
    parser.add_argument(
        "--module", nargs="+", type=int, default=None,
        help="Module number(s) to run (1-6). Default: all.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/research.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--quantizer", nargs="+", type=str, default=None,
        help="Specific quantizer(s) to run (int8, fp8, nf4, gptq).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Set primary CUDA device to cuda:1 before any model loading ────
    hw = getattr(config, "hardware", None)
    primary_idx = getattr(hw, "primary_cuda_index", 1) if hw else 1
    set_primary_cuda_device(preferred_index=primary_idx)

    # Store quantizer filter for Modules 2, 4, 5
    config._quantizer_filter = (
        [q.lower() for q in args.quantizer] if args.quantizer else None
    )

    # Unified pipeline log
    setup_unified_logger(log_dir=config.output.base_dir)
    logger = get_logger("research.pipeline")
    logger.info("=" * 70)
    logger.info("LLM Quantisation Research — sarvamai/sarvam-30b")
    logger.info("=" * 70)
    get_device_info()

    modules: List[int] = args.module or [1, 2, 3, 4, 5, 6]
    overall: Dict[int, Dict] = {}
    t_total = time.time()

    for num in modules:
        ModuleClass = MODULE_MAP.get(num)
        if ModuleClass is None:
            logger.warning(f"Unknown module {num} — skipping")
            continue

        # Scope config paths to this module's directory
        scope_config_to_module(config, num)
        setup_logger(f"module_{num}", log_dir=config.output.logs_dir)

        # Run module
        runner = ModuleClass(config)
        result = runner.run()
        overall[num] = result

        # Optionally run tests after each module
        try:
            from tests.test_runner import run_module_tests
            if not result["status"].startswith("✗"):
                test_passed = run_module_tests(num, config, logger)
                if not test_passed:
                    logger.error(f"❌ Module {num} tests FAILED — aborting pipeline")
                    sys.exit(1)
                else:
                    logger.info(f"✅ Module {num} tests PASSED")
        except ImportError:
            pass  # tests not available

    # Pipeline summary
    total_time = time.time() - t_total
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIPELINE SUMMARY")
    logger.info("=" * 70)
    for num in modules:
        if num in overall:
            status = overall[num].get("status", "unknown")
            wall = overall[num].get("total_time_sec", 0)
            icon = "OK" if "COMPLETED" in status else (
                "PART" if "PARTIAL" in status else "FAIL"
            )
            logger.info(
                f"  [{icon:>4}]  Module {num} ({MODULE_NAMES.get(num, ''):<25s})  "
                f"{wall:>8.1f}s"
            )
    logger.info(f"  Total time: {total_time:.1f}s ({total_time / 3600:.2f}h)")
    logger.info("=" * 70)

    # Persist pipeline summary (merge with prior runs)
    _persist_pipeline_summary(config, modules, overall, total_time)


def _persist_pipeline_summary(config, modules, overall, total_time):
    """Save/merge pipeline_summary.json."""
    summary_path = Path(config.output.base_dir) / "pipeline_summary.json"

    existing: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
        except Exception:
            existing = {}

    prev_modules = existing.get("modules_run", [])
    merged_modules = sorted(set(prev_modules) | set(modules))

    prev_results = existing.get("results", {})
    new_results = {
        str(k): v.get("status", "unknown") for k, v in overall.items()
    }
    prev_results.update(new_results)

    prev_timing = existing.get("module_timing_sec", {})
    for num in modules:
        if num in overall:
            prev_timing[str(num)] = {
                "wall_time_sec": overall[num].get("total_time_sec", 0),
            }

    prev_total = existing.get("total_time_sec", 0)
    merged = {
        "modules_run": merged_modules,
        "total_time_sec": round(prev_total + total_time, 2),
        "module_timing_sec": prev_timing,
        "results": prev_results,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(merged, fh, indent=2)


if __name__ == "__main__":
    main()
