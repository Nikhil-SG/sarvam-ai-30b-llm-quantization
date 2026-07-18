"""
Module-wise output directory management.

Every module (1-6) gets its own output subdirectory:

    outputs/
    ├── module_1_baseline/       (plots/ logs/ results/)
    ├── module_2_quantization/   (plots/ logs/ results/)
    ├── module_3_analysis/       (plots/ logs/ results/)
    ├── module_4_profiling/      (plots/ logs/ results/)
    ├── module_5_evaluation/     (plots/ logs/ results/)
    ├── module_6_visualization/  (plots/ logs/ results/)
    ├── shared_weights/          (weight caches used across modules 1-6)
    └── pipeline_summary.json

MxMoE modules (1-5) get their own output subdirectory:

    mxmoe/outputs/
    ├── module_1_sensitivity/    (plots/ logs/ results/)
    ├── module_2_synthesis/      (plots/ logs/ results/)
    ├── module_3_evaluation/     (plots/ logs/ results/)
    ├── module_4_deployment/     (plots/ logs/ results/)
    ├── module_5_publication/    (plots/ logs/ results/)
    └── pipeline_summary.json

The ``scope_config_to_module`` function patches ``config.output.*``
in-place so that all downstream code reads the correct paths without
any constructor changes.

Weight caches are written to ``outputs/shared_weights/`` so that
modules 1 & 2 (producers) and modules 3 & 6 (consumers) share data.
"""

from pathlib import Path
from typing import Dict


# Human-readable folder names per module — Research Pipeline (Part 1)
MODULE_DIR_NAMES: Dict[int, str] = {
    1: "module_1_baseline",
    2: "module_2_quantization",
    3: "module_3_analysis",
    4: "module_4_profiling",
    5: "module_5_evaluation",
    6: "module_6_visualization",
}

# Human-readable folder names per module — MxMoE Pipeline (Part 2)
MXMOE_MODULE_DIR_NAMES: Dict[int, str] = {
    1: "module_1_sensitivity",
    2: "module_2_synthesis",
    3: "module_3_evaluation",
    4: "module_4_deployment",
    5: "module_5_publication",
}


def get_module_dir(base_dir: str, module_num: int) -> Path:
    """Return the root output directory for a given module number."""
    name = MODULE_DIR_NAMES.get(module_num, f"module_{module_num}")
    return Path(base_dir) / name


def get_mxmoe_module_dir(base_dir: str, module_num: int) -> Path:
    """Return the root output directory for a given MxMoE module number."""
    name = MXMOE_MODULE_DIR_NAMES.get(module_num, f"mxmoe_module_{module_num}")
    return Path(base_dir) / name


def get_shared_weights_dir(base_dir: str) -> str:
    """Return the shared weight-cache directory."""
    return str(Path(base_dir) / "shared_weights")


def get_module_paths(base_dir: str, module_num: int) -> Dict[str, str]:
    """
    Build the full set of output sub-directories for one module.

    Returns:
        Dict with keys: base_dir, plots_dir, logs_dir, results_dir, weights_dir
        ``weights_dir`` always points to the shared location.
    """
    root = get_module_dir(base_dir, module_num)
    return {
        "base_dir":    str(root),
        "plots_dir":   str(root / "plots"),
        "logs_dir":    str(root / "logs"),
        "results_dir": str(root / "results"),
        "weights_dir": get_shared_weights_dir(base_dir),
    }


def get_mxmoe_module_paths(base_dir: str, module_num: int) -> Dict[str, str]:
    """
    Build the full set of output sub-directories for one MxMoE module.

    Returns:
        Dict with keys: base_dir, plots_dir, logs_dir, results_dir, weights_dir
        ``weights_dir`` points to the MxMoE shared weight cache.
    """
    root = get_mxmoe_module_dir(base_dir, module_num)
    return {
        "base_dir":    str(root),
        "plots_dir":   str(root / "plots"),
        "logs_dir":    str(root / "logs"),
        "results_dir": str(root / "results"),
        "weights_dir": get_shared_weights_dir(base_dir),
    }


def ensure_module_dirs(base_dir: str, module_num: int) -> Dict[str, str]:
    """Create and return all output directories for a module."""
    paths = get_module_paths(base_dir, module_num)
    for p in paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return paths


def ensure_mxmoe_module_dirs(base_dir: str, module_num: int) -> Dict[str, str]:
    """Create and return all output directories for an MxMoE module."""
    paths = get_mxmoe_module_paths(base_dir, module_num)
    for p in paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return paths


def ensure_all_module_dirs(base_dir: str) -> None:
    """Create the base output directory only. Module dirs are created lazily.

    Previously pre-created all 6 module directories on config load,
    leaving empty folders even when modules weren't run.  Now only
    the top-level base_dir is created eagerly; per-module dirs are
    created on demand via ``ensure_module_dirs()`` when a module starts.
    """
    Path(base_dir).mkdir(parents=True, exist_ok=True)


def scope_config_to_module(config, module_num: int) -> None:
    """
    Patch ``config.output.*`` in-place to point at this module's directories.

    After this call, any code reading ``config.output.plots_dir`` etc.
    will get the module-specific path.  Calling this again with a
    different module_num re-scopes to that module.

    ``weights_dir`` always points to the shared weight cache.

    .. warning::
        This function mutates the **shared** ``config`` object.  Any class
        that captures ``config.output.*`` paths in its ``__init__`` (e.g.
        ``self.results_dir = Path(config.output.results_dir)``) **must** be
        constructed *after* this call has been made for the correct module.
        Constructing such a class before ``scope_config_to_module`` runs, or
        caching the path values across module boundaries, will silently write
        files to the wrong module directory.
    """
    base = config.output.base_dir
    paths = ensure_module_dirs(base, module_num)

    config.output.plots_dir   = paths["plots_dir"]
    config.output.logs_dir    = paths["logs_dir"]
    config.output.results_dir = paths["results_dir"]
    config.output.weights_dir = paths["weights_dir"]
    config.output._data["plots_dir"]   = paths["plots_dir"]
    config.output._data["logs_dir"]    = paths["logs_dir"]
    config.output._data["results_dir"] = paths["results_dir"]
    config.output._data["weights_dir"] = paths["weights_dir"]


def scope_config_to_mxmoe_module(config, module_num: int) -> None:
    """
    Patch ``config.output.*`` in-place to point at an MxMoE module's
    directories.

    Works identically to ``scope_config_to_module`` but uses the MxMoE
    output tree (``mxmoe/outputs/module_N_*/``).

    .. warning::
        This function mutates the **shared** ``config`` object.  Any class
        that captures ``config.output.*`` paths in its ``__init__`` (e.g.
        ``self.results_dir = Path(config.output.results_dir)``) **must** be
        constructed *after* this call has been made for the correct module.
        Constructing such a class before ``scope_config_to_mxmoe_module`` runs,
        or caching the path values across module boundaries, will silently write
        files to the wrong module directory.
    """
    base = config.output.base_dir
    paths = ensure_mxmoe_module_dirs(base, module_num)

    config.output.plots_dir   = paths["plots_dir"]
    config.output.logs_dir    = paths["logs_dir"]
    config.output.results_dir = paths["results_dir"]
    config.output.weights_dir = paths["weights_dir"]
    config.output._data["plots_dir"]   = paths["plots_dir"]
    config.output._data["logs_dir"]    = paths["logs_dir"]
    config.output._data["results_dir"] = paths["results_dir"]
    config.output._data["weights_dir"] = paths["weights_dir"]
