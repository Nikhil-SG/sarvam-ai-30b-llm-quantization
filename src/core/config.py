"""
Configuration management for LLM Quantization Research.

Loads YAML configuration, validates required keys, and provides
convenient dot-notation access to settings.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Thin wrapper around a dict that supports attribute-style access."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    # ── dict-like helpers ───────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        return f"Config({list(self._data.keys())})"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge ``override`` into ``base`` (override wins).

    Used by ``load_config()`` to implement ``_base_`` config inheritance.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from a YAML file.

    Supports ``_base_`` key for config inheritance: if the YAML contains
    ``_base_: "base.yaml"``, the base config is loaded first and the
    current config is merged on top (current wins).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object with validated settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path.absolute()}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    # ── Base config inheritance ──────────────────────────────────────────
    base_ref = raw.pop("_base_", None)
    if base_ref:
        base_path = path.parent / base_ref
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base config not found: {base_path} (referenced by _base_ in {path})"
            )
        with open(base_path, "r", encoding="utf-8") as fh:
            base_raw: Dict[str, Any] = yaml.safe_load(fh)
        raw = _deep_merge(base_raw, raw)

    _validate_repo_scoped_paths(raw)
    _ensure_output_dirs(raw)
    return Config(raw)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_repo_scoped_paths(config: Dict[str, Any]) -> None:
    """Reject path values that escape the repository root."""
    root = _repo_root().resolve()

    def _validate(path_value: Optional[str], key: str) -> None:
        if not path_value or not isinstance(path_value, str):
            return
        p = Path(path_value)
        resolved = p.resolve() if p.is_absolute() else (root / p).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Unsafe config path for {key}: {path_value} (resolves outside repo)"
            ) from exc

    storage_cfg = config.get("storage", {}) or {}
    model_cfg = config.get("model", {}) or {}
    output_cfg = config.get("output", {}) or {}

    _validate(storage_cfg.get("hf_home"), "storage.hf_home")
    _validate(storage_cfg.get("local_model_path"), "storage.local_model_path")
    _validate(model_cfg.get("cache_dir"), "model.cache_dir")
    _validate(output_cfg.get("base_dir"), "output.base_dir")
    _validate(output_cfg.get("quantized_models_dir"), "output.quantized_models_dir")


def _ensure_output_dirs(config: Dict[str, Any]) -> None:
    """Set default output path attributes. Directories created lazily."""
    from src.core.paths import ensure_all_module_dirs

    output_cfg = config.get("output", {})
    base_dir = output_cfg.get("base_dir", "outputs")

    # Fallback paths — overridden by scope_config_to_module() at runtime.
    # These are NOT created on disk here; they exist only so that
    # config.output.logs_dir etc. don't raise AttributeError before scoping.
    output_cfg.setdefault("plots_dir", str(Path(base_dir) / "plots"))
    output_cfg.setdefault("logs_dir", str(Path(base_dir) / "logs"))
    output_cfg.setdefault("results_dir", str(Path(base_dir) / "results"))
    output_cfg.setdefault("weights_dir", str(Path(base_dir) / "shared_weights"))

    # Only create the top-level base_dir eagerly.
    # Per-module dirs are created lazily when scope_config_to_module() runs.
    ensure_all_module_dirs(base_dir)
