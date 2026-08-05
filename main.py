#!/usr/bin/env python3
"""Top-level orchestrator for research and MxMoE pipelines.

Examples:
    python main.py
    python main.py --pipeline research --module 1 2
    python main.py --pipeline research --quantizer gptq int8
    python main.py --pipeline mxmoe --module 1 2
    python main.py --pipeline all --research-module 2 --research-quantizer fp8 --mxmoe-module 1
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_orchestrator_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    orch = data.get("orchestrator", {})
    research = orch.get("research", {})
    mxmoe = orch.get("mxmoe", {})

    return {
        "default_pipeline": orch.get("default_pipeline", "all"),
        "stop_on_error": bool(orch.get("stop_on_error", True)),
        "research": {
            "entrypoint": research.get("entrypoint", "pipelines.research.pipeline"),
            "config_path": research.get("config_path", "configs/research.yaml"),
            "modules": research.get("modules"),
            "quantizers": research.get("quantizers"),
        },
        "mxmoe": {
            "entrypoint": mxmoe.get("entrypoint", "pipelines.mxmoe.pipeline"),
            "config_path": mxmoe.get("config_path", "configs/mxmoe.yaml"),
            "modules": mxmoe.get("modules"),
        },
    }


def _append_multi_arg(cmd: List[str], flag: str, values: Optional[List[Any]]) -> None:
    if values:
        cmd.append(flag)
        cmd.extend(str(v) for v in values)


def _build_research_cmd(
    entrypoint: str,
    config_path: str,
    modules: Optional[List[int]],
    quantizers: Optional[List[str]],
) -> List[str]:
    cmd = [sys.executable, "-m", entrypoint, "--config", config_path]
    _append_multi_arg(cmd, "--module", modules)
    _append_multi_arg(cmd, "--quantizer", quantizers)
    return cmd


def _build_mxmoe_cmd(
    entrypoint: str,
    config_path: str,
    modules: Optional[List[int]],
) -> List[str]:
    cmd = [sys.executable, "-m", entrypoint, "--config", config_path]
    _append_multi_arg(cmd, "--module", modules)
    return cmd


def _run(cmd: List[str], dry_run: bool) -> int:
    print("Running:", " ".join(cmd))
    if dry_run:
        return 0
    
    # Ensure current directory is in PYTHONPATH so subprocess can resolve local modules (like `pipelines`)
    env = os.environ.copy()
    cwd = os.getcwd()
    env["PYTHONPATH"] = f"{cwd}{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    completed = subprocess.run(cmd, check=False, env=env)
    return int(completed.returncode)


def _validate_entrypoint(entrypoint: str) -> None:
    pattern = r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
    if not re.match(pattern, entrypoint):
        raise ValueError(f"Invalid entrypoint format: {entrypoint}")


def _validate_modules(modules: Optional[List[int]], allowed: set[int], flag_name: str) -> None:
    if not modules:
        return
    bad = [m for m in modules if m not in allowed]
    if bad:
        raise ValueError(f"Invalid values for {flag_name}: {bad}; allowed={sorted(allowed)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified orchestrator for Sarvam research and MxMoE pipelines"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to top-level orchestrator config YAML.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["research", "mxmoe", "all"],
        default=None,
        help="Pipeline scope to run. Default comes from config.yaml orchestrator.default_pipeline.",
    )

    # Legacy shortcuts (apply only to single selected pipeline)
    parser.add_argument(
        "--module",
        nargs="+",
        type=int,
        default=None,
        help="Legacy shortcut for a single selected pipeline. Use --research-module/--mxmoe-module for --pipeline all.",
    )
    parser.add_argument(
        "--quantizer",
        nargs="+",
        type=str,
        default=None,
        help="Legacy shortcut for research quantizers (single selected pipeline).",
    )

    # Explicit per-pipeline controls
    parser.add_argument("--research-config", type=str, default=None)
    parser.add_argument("--mxmoe-config", type=str, default=None)
    parser.add_argument("--research-module", nargs="+", type=int, default=None)
    parser.add_argument("--mxmoe-module", nargs="+", type=int, default=None)
    parser.add_argument("--research-quantizer", nargs="+", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")

    args = parser.parse_args()
    cfg = _load_orchestrator_config(args.config)

    _validate_entrypoint(cfg["research"]["entrypoint"])
    _validate_entrypoint(cfg["mxmoe"]["entrypoint"])

    pipeline = args.pipeline or cfg["default_pipeline"]
    if pipeline not in {"research", "mxmoe", "all"}:
        raise ValueError(f"Invalid pipeline mode: {pipeline}")

    if pipeline == "all" and args.module:
        parser.error("--module is ambiguous with --pipeline all. Use --research-module and/or --mxmoe-module.")

    if pipeline != "research" and args.quantizer:
        parser.error("--quantizer only applies to research pipeline.")

    research_modules = args.research_module
    mxmoe_modules = args.mxmoe_module
    research_quantizers = args.research_quantizer

    if pipeline == "research":
        research_modules = research_modules or args.module or cfg["research"]["modules"]
        research_quantizers = research_quantizers or args.quantizer or cfg["research"]["quantizers"]
    elif pipeline == "mxmoe":
        mxmoe_modules = mxmoe_modules or args.module or cfg["mxmoe"]["modules"]
    else:
        research_modules = research_modules or cfg["research"]["modules"]
        research_quantizers = research_quantizers or cfg["research"]["quantizers"]
        mxmoe_modules = mxmoe_modules or cfg["mxmoe"]["modules"]

    _validate_modules(research_modules, {1, 2, 3, 4, 5, 6}, "research modules")
    _validate_modules(mxmoe_modules, {1, 2, 3, 4, 5}, "mxmoe modules")

    research_config = args.research_config or cfg["research"]["config_path"]
    mxmoe_config = args.mxmoe_config or cfg["mxmoe"]["config_path"]

    steps: List[List[str]] = []

    if pipeline in {"research", "all"}:
        steps.append(
            _build_research_cmd(
                entrypoint=cfg["research"]["entrypoint"],
                config_path=research_config,
                modules=research_modules,
                quantizers=research_quantizers,
            )
        )

    if pipeline in {"mxmoe", "all"}:
        steps.append(
            _build_mxmoe_cmd(
                entrypoint=cfg["mxmoe"]["entrypoint"],
                config_path=mxmoe_config,
                modules=mxmoe_modules,
            )
        )

    for i, cmd in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}]")
        code = _run(cmd, dry_run=args.dry_run)
        if code != 0 and cfg["stop_on_error"]:
            sys.exit(code)


if __name__ == "__main__":
    main()
