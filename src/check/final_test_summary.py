#!/usr/bin/env python3
"""Quick refactor-aware verification summary."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_state(tag: str, message: str) -> None:
    print(f"[{tag}]  {message}")


def _try_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        _print_state("OK", module_name)
        return True
    except Exception as exc:
        _print_state("ERR", f"{module_name}: {str(exc)[:120]}")
        return False


def _count_test_functions(test_dir: Path) -> int:
    total = 0
    for test_file in sorted(test_dir.glob("test_module*.py")):
        total += test_file.read_text(encoding="utf-8").count("def test_")
    return total


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_base_dir_from_yaml(config_path: str, default: str) -> Path:
    """Load output.base_dir from YAML with optional _base_ inheritance."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return Path(default)

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        base_ref = raw.pop("_base_", None)
        merged = raw

        if base_ref:
            base_path = cfg_path.parent / base_ref
            base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
            merged = _deep_merge(base_raw, raw)

        output_cfg = merged.get("output", {}) or {}
        return Path(str(output_cfg.get("base_dir", default)))
    except Exception:
        return Path(default)


def _load_output_roots() -> tuple[Path, Path]:
    """Resolve output roots without importing runtime-heavy packages."""
    research_base = _load_base_dir_from_yaml(str(REPO_ROOT / "configs" / "research.yaml"), str(REPO_ROOT / "research" / "outputs"))
    mxmoe_base = _load_base_dir_from_yaml(str(REPO_ROOT / "configs" / "mxmoe.yaml"), str(REPO_ROOT / "mxmoe" / "outputs"))
    return research_base, mxmoe_base


def _load_module_maps() -> tuple[dict[int, str], dict[int, str]]:
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
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from src.core.paths import MODULE_DIR_NAMES, MXMOE_MODULE_DIR_NAMES

        return dict(MODULE_DIR_NAMES), dict(MXMOE_MODULE_DIR_NAMES)
    except Exception:
        return fallback_research, fallback_mxmoe


def _select_modules(module_map: dict[int, str], selected: list[int] | None) -> dict[int, str]:
    if not selected:
        return module_map
    return {num: name for num, name in module_map.items() if num in selected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick refactor-aware verification summary")
    parser.add_argument(
        "--require-outputs",
        action="store_true",
        help="Treat missing output artifact directories as an error state.",
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

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    print("=" * 80)
    print("FINAL TEST EXECUTION SUMMARY")
    print("=" * 80)

    research_out, mxmoe_out = _load_output_roots()
    research_modules, mxmoe_modules = _load_module_maps()

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

    selected_research_modules = _select_modules(research_modules, args.research_modules)
    selected_mxmoe_modules = _select_modules(mxmoe_modules, args.mxmoe_modules)

    _print_header("1) CRITICAL MODULE IMPORTS")
    critical_imports = ["main", "tests.test_runner"]
    runtime_imports = [
        "pipelines.research.pipeline",
        "pipelines.mxmoe.pipeline",
        "src.core.config",
        "src.core.runner",
        "src.core.artifacts",
        "src.core.calibration",
    ]

    import_pass = sum(_try_import(name) for name in critical_imports)
    torch_available = importlib.util.find_spec("torch") is not None
    if torch_available:
        import_pass += sum(_try_import(name) for name in runtime_imports)
    else:
        _print_state("SKIP", "Runtime import checks skipped because 'torch' is not installed.")

    _print_header("2) CONFIGURATION LOADING")
    if torch_available:
        try:
            from src.core.config import load_config

            cfg_research = load_config(str(REPO_ROOT / "configs" / "research.yaml"))
            cfg_mxmoe = load_config(str(REPO_ROOT / "configs" / "mxmoe.yaml"))
            _print_state("OK", "configs/research.yaml")
            _print_state("OK", "configs/mxmoe.yaml")
            print(f"      model_id={cfg_research.model.model_id}")
            print(f"      research_output_base={cfg_research.output.base_dir}")
            print(f"      mxmoe_output_base={cfg_mxmoe.output.base_dir}")
            config_state = "OK"
        except Exception as exc:
            _print_state("ERR", f"Config load failed: {str(exc)[:160]}")
            config_state = "FAIL"
    else:
        _print_state("SKIP", "Config runtime load skipped because 'torch' is not installed.")
        print(f"      research_output_base={research_out}")
        print(f"      mxmoe_output_base={mxmoe_out}")
        config_state = "SKIP"

    _print_header("3) TEST FUNCTION COUNTS")
    research_tests = _count_test_functions(REPO_ROOT / "tests" / "research")
    mxmoe_tests = _count_test_functions(REPO_ROOT / "tests" / "mxmoe")
    total_tests = research_tests + mxmoe_tests
    _print_state("OK", f"Research tests: {research_tests}")
    _print_state("OK", f"MxMoE tests:    {mxmoe_tests}")
    _print_state("OK", f"Total tests:    {total_tests}")

    _print_header("4) OUTPUT TREE PRESENCE")
    outputs_ok = True

    if args.scope in {"all", "research"}:
        r_dirs = [research_out / mod_name for mod_name in selected_research_modules.values()]
        r_module_count = sum(1 for p in r_dirs if p.exists())
        r_summary_count = sum(
            1
            for mod_num, mod_name in selected_research_modules.items()
            if (research_out / mod_name / "results" / f"module_{mod_num}_summary.json").exists()
        )
        r_expected = len(selected_research_modules)

        if r_module_count == 0 and not args.require_outputs:
            _print_state("INFO", "Research module output dirs: 0 (expected before first research run)")
        else:
            _print_state(
                "OK" if r_module_count == r_expected else "ERR",
                f"Research module output dirs: {r_module_count}/{r_expected}",
            )
        _print_state("INFO", f"Research module summaries: {r_summary_count}/{r_expected}")

        if args.require_outputs and (r_module_count != r_expected or r_summary_count != r_expected):
            outputs_ok = False
    else:
        _print_state("INFO", "Research output checks skipped (--scope mxmoe).")

    if args.scope in {"all", "mxmoe"}:
        m_dirs = [mxmoe_out / mod_name for mod_name in selected_mxmoe_modules.values()]
        m_module_count = sum(1 for p in m_dirs if p.exists())
        m_summary_count = sum(
            1
            for mod_num, mod_name in selected_mxmoe_modules.items()
            if (mxmoe_out / mod_name / "results" / f"module_{mod_num}_summary.json").exists()
        )
        m_expected = len(selected_mxmoe_modules)

        if m_module_count == 0 and not args.require_outputs:
            _print_state("INFO", "MxMoE module output dirs:   0 (expected before first MxMoE run)")
        else:
            _print_state(
                "OK" if m_module_count == m_expected else "ERR",
                f"MxMoE module output dirs:   {m_module_count}/{m_expected}",
            )
        _print_state("INFO", f"MxMoE module summaries:   {m_summary_count}/{m_expected}")

        if args.require_outputs and (m_module_count != m_expected or m_summary_count != m_expected):
            outputs_ok = False
    else:
        _print_state("INFO", "MxMoE output checks skipped (--scope research).")

    _print_header("SUMMARY")
    total_imports = len(critical_imports) + (len(runtime_imports) if torch_available else 0)
    print(
        f"Imports: {import_pass}/{total_imports} passed | "
        f"Config load: {config_state}"
    )
    _print_state(
        "INFO",
        "Strict output scope: "
        f"{args.scope}"
        + (
            f" | research modules={sorted(selected_research_modules.keys())}"
            if args.scope in {"all", "research"} else ""
        )
        + (
            f" | mxmoe modules={sorted(selected_mxmoe_modules.keys())}"
            if args.scope in {"all", "mxmoe"} else ""
        ),
    )
    if args.require_outputs and not outputs_ok:
        _print_state("ERR", "Output artifacts are required but selected scope/modules are incomplete.")


if __name__ == "__main__":
    main()
