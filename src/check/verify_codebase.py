#!/usr/bin/env python3
"""Refactor-aware codebase verification script.

Checks repository structure and reports what is present vs missing without
hard-coded pass/fail claims.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def status_line(ok: bool, message: str) -> None:
    print(f"[{'OK' if ok else 'MISS'}] {message}")


def info_line(message: str) -> None:
    print(f"[INFO] {message}")


def count_test_functions(test_dir: Path) -> int:
    total = 0
    for test_file in sorted(test_dir.glob("test_module*.py")):
        total += test_file.read_text(encoding="utf-8").count("def test_")
    return total


def check_required_files(files: dict[str, str]) -> tuple[int, int]:
    passed = 0
    for rel_path, label in files.items():
        clean_path = rel_path[3:] if rel_path.startswith("../") else rel_path
        ok = (REPO_ROOT / clean_path).exists()
        status_line(ok, f"{label}: {clean_path}")
        passed += int(ok)
    return passed, len(files)


def resolve_output_roots() -> Tuple[Path, Path]:
    """Resolve output roots from runtime config, with safe fallbacks."""
    default_research = REPO_ROOT / "research" / "outputs"
    default_mxmoe = REPO_ROOT / "mxmoe" / "outputs"
    try:
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from src.core.config import load_config

        research_cfg = load_config(str(REPO_ROOT / "configs" / "research.yaml"))
        mxmoe_cfg = load_config(str(REPO_ROOT / "configs" / "mxmoe.yaml"))
        return Path(str(research_cfg.output.base_dir)), Path(str(mxmoe_cfg.output.base_dir))
    except Exception:
        return default_research, default_mxmoe


def resolve_module_maps() -> Tuple[Dict[int, str], Dict[int, str]]:
    """Resolve module directory names from the canonical source in src.core.paths."""
    fallback_research = {
        1: "module_1_baseline",
        2: "module_2_quantization",
        3: "module_3_analysis",
        4: "module_4_profiling",
        5: "module_5_evaluation",
        6: "module_6_visualization",
    }
    fallback_mxmoe = {
        1: "module_1_sensitivity",
        2: "module_2_synthesis",
        3: "module_3_evaluation",
        4: "module_4_deployment",
        5: "module_5_publication",
    }
    try:
        from src.core.paths import MODULE_DIR_NAMES, MXMOE_MODULE_DIR_NAMES

        return dict(MODULE_DIR_NAMES), dict(MXMOE_MODULE_DIR_NAMES)
    except Exception:
        return fallback_research, fallback_mxmoe


def select_modules(module_map: Dict[int, str], selected: list[int] | None) -> Dict[int, str]:
    """Filter module map by selected module numbers (or return all if None)."""
    if not selected:
        return module_map
    return {num: name for num, name in module_map.items() if num in selected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Refactor-aware repository verification")
    parser.add_argument(
        "--require-outputs",
        action="store_true",
        help="Treat missing module output directories/summaries as required checks.",
    )
    parser.add_argument(
        "--scope",
        choices=["all", "research", "mxmoe"],
        default="all",
        help="Which pipeline outputs to verify (default: all).",
    )
    parser.add_argument(
        "--research-modules",
        nargs="+",
        type=int,
        default=None,
        help="Optional research module numbers to verify (default: all research modules).",
    )
    parser.add_argument(
        "--mxmoe-modules",
        nargs="+",
        type=int,
        default=None,
        help="Optional MxMoE module numbers to verify (default: all MxMoE modules).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE CODEBASE VERIFICATION")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_checks = 0
    passed_checks = 0

    section("1) CORE REFACTOR FILES")
    core_files = {
        "../pyproject.toml": "Dependency manifest",
        "../main.py": "Unified orchestrator",
        "../config.yaml": "Top-level orchestrator config",
        "../configs/base.yaml": "Base shared config",
        "../configs/research.yaml": "Research pipeline config",
        "../configs/mxmoe.yaml": "MxMoE pipeline config",
        "../src/core/runner.py": "ModuleRunner base",
        "../src/core/artifacts.py": "ResearchArtifacts contract",
        "../src/core/calibration.py": "Shared calibration loader",
        "../src/visualization/core.py": "Shared plot utilities",
    }
    passed, checks = check_required_files(core_files)
    passed_checks += passed
    total_checks += checks

    section("2) PIPELINE ENTRYPOINTS")
    pipeline_files = {
        "../pipelines/research/pipeline.py": "Research orchestrator",
        "../pipelines/research/modules.py": "Research module runners",
        "../pipelines/mxmoe/pipeline.py": "MxMoE orchestrator",
        "../pipelines/mxmoe/modules.py": "MxMoE module runners",
    }
    passed, checks = check_required_files(pipeline_files)
    passed_checks += passed
    total_checks += checks

    section("3) LEGACY REQUIREMENTS CLEANUP")
    legacy_requirements = sorted(REPO_ROOT.glob("requirements*.txt"))
    no_legacy = len(legacy_requirements) == 0
    status_line(no_legacy, "No requirements*.txt files in repo root")
    if not no_legacy:
        for req in legacy_requirements:
            print(f"       -> {req}")
    passed_checks += int(no_legacy)
    total_checks += 1

    section("4) OUTPUT TREE STATUS")
    research_modules, mxmoe_modules = resolve_module_maps()

    if args.research_modules:
        invalid = [m for m in args.research_modules if m not in research_modules]
        if invalid:
            parser.error(
                f"Invalid values for --research-modules: {invalid}; allowed={sorted(research_modules.keys())}"
            )
    if args.mxmoe_modules:
        invalid = [m for m in args.mxmoe_modules if m not in mxmoe_modules]
        if invalid:
            parser.error(
                f"Invalid values for --mxmoe-modules: {invalid}; allowed={sorted(mxmoe_modules.keys())}"
            )

    selected_research_modules = select_modules(research_modules, args.research_modules)
    selected_mxmoe_modules = select_modules(mxmoe_modules, args.mxmoe_modules)

    research_root, mxmoe_root = resolve_output_roots()
    info_line(f"Research output root: {research_root}")
    info_line(f"MxMoE output root: {mxmoe_root}")

    output_checks = 0
    output_passed = 0

    if args.scope in {"all", "research"}:
        research_any = any((research_root / name).exists() for name in selected_research_modules.values())
        if not args.require_outputs and not research_any:
            info_line("Research outputs not found yet (expected before first research pipeline run).")
        else:
            for mod_num, mod_name in selected_research_modules.items():
                mod_dir = research_root / mod_name
                summary = mod_dir / "results" / f"module_{mod_num}_summary.json"
                dir_ok = mod_dir.exists()
                sum_ok = summary.exists()
                status_line(dir_ok, f"Research output dir exists: {mod_dir}")
                status_line(sum_ok, f"Research summary exists: {summary}")
                if args.require_outputs:
                    output_checks += 2
                    output_passed += int(dir_ok) + int(sum_ok)
    else:
        info_line("Research output checks skipped (--scope mxmoe).")

    if args.scope in {"all", "mxmoe"}:
        mxmoe_any = any((mxmoe_root / name).exists() for name in selected_mxmoe_modules.values())
        if not args.require_outputs and not mxmoe_any:
            info_line("MxMoE outputs not found yet (expected before first MxMoE pipeline run).")
        else:
            for mod_num, mod_name in selected_mxmoe_modules.items():
                mod_dir = mxmoe_root / mod_name
                summary = mod_dir / "results" / f"module_{mod_num}_summary.json"
                dir_ok = mod_dir.exists()
                sum_ok = summary.exists()
                status_line(dir_ok, f"MxMoE output dir exists: {mod_dir}")
                status_line(sum_ok, f"MxMoE summary exists: {summary}")
                if args.require_outputs:
                    output_checks += 2
                    output_passed += int(dir_ok) + int(sum_ok)
    else:
        info_line("MxMoE output checks skipped (--scope research).")

    if args.require_outputs:
        passed_checks += output_passed
        total_checks += output_checks

    section("5) TEST COVERAGE SNAPSHOT")
    research_test_dir = REPO_ROOT / "tests" / "research"
    mxmoe_test_dir = REPO_ROOT / "tests" / "mxmoe"
    research_test_count = count_test_functions(research_test_dir) if research_test_dir.exists() else 0
    mxmoe_test_count = count_test_functions(mxmoe_test_dir) if mxmoe_test_dir.exists() else 0
    status_line(research_test_dir.exists(), "Research test directory present")
    status_line(mxmoe_test_dir.exists(), "MxMoE test directory present")
    print(f"Research test functions: {research_test_count}")
    print(f"MxMoE test functions:    {mxmoe_test_count}")
    print(f"Total test functions:    {research_test_count + mxmoe_test_count}")

    section("6) SOURCE PACKAGE LAYOUT")
    layout_checks = {
        "../src/core": "Core package",
        "../src/quantization": "Quantization package",
        "../src/analysis": "Analysis package",
        "../src/evaluation": "Evaluation package",
        "../src/profiling": "Profiling package",
        "../src/visualization": "Visualization package",
        "../src/mxmoe/sensitivity": "MxMoE sensitivity package",
        "../src/mxmoe/recipe": "MxMoE recipe package",
        "../src/mxmoe/ablation": "MxMoE ablation package",
        "../src/mxmoe/deployment": "MxMoE deployment package",
    }
    passed, checks = check_required_files(layout_checks)
    passed_checks += passed
    total_checks += checks

    section("SUMMARY")
    print(f"Required checks passed: {passed_checks}/{total_checks}")
    if not args.require_outputs:
        print("Output artifact checks: informational only (use --require-outputs to enforce).")
    else:
        print(
            "Strict output scope: "
            f"{args.scope}"
            + (
                f" | research modules={sorted(selected_research_modules.keys())}"
                if args.scope in {"all", "research"} else ""
            )
            + (
                f" | mxmoe modules={sorted(selected_mxmoe_modules.keys())}"
                if args.scope in {"all", "mxmoe"} else ""
            )
        )
    if passed_checks == total_checks:
        print("Result: structural refactor checks are complete.")
    else:
        print("Result: some structural checks are missing. Review MISS lines above.")


if __name__ == "__main__":
    main()
