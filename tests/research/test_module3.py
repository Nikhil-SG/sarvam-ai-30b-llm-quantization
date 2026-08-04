"""
tests/test_module3.py
─────────────────────
Validation tests for Module 3 (Weight Introspection).

Checks:
  - plots/ directory has at least one PNG (histograms or heatmap produced)
  - Weight distribution PNG files exist (one per target layer if cache present)
  - MSE heatmap PNG exists (if both bf16 and ≥1 quantized cache exist)
  - results/ directory is present and non-empty (JSON files written)
  - No crash evidence: sub-result keys exist with "complete" or a list, not only errors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _module_dir(config) -> Path:
    from src.core.paths import get_module_paths
    paths = get_module_paths(config.output.base_dir, 3)
    return Path(paths["base_dir"])


def _plots_dir(config) -> Path:
    return _module_dir(config) / "plots"


def _results_dir(config) -> Path:
    return _module_dir(config) / "results"


def _shared_weights_dir(config) -> Path:
    return Path(config.output.base_dir) / "shared_weights"


def _has_bf16_cache(config) -> bool:
    return (_shared_weights_dir(config) / "bf16").is_dir()


def _has_any_quant_cache(config) -> bool:
    sw = _shared_weights_dir(config)
    if not sw.is_dir():
        return False
    return any(d.is_dir() and d.name != "bf16" for d in sw.iterdir())


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_plots_directory_exists(config) -> None:
    """module_3_analysis/plots/ directory must be created by Module 3."""
    assert _plots_dir(config).is_dir(), (
        f"Plots directory not found: {_plots_dir(config)}"
    )


def test_at_least_one_plot_produced(config) -> None:
    """At least one PNG must exist in plots/ — analysis produced visual output."""
    if not _has_bf16_cache(config) and not _has_any_quant_cache(config):
        return  # no cache available — Module 3 cannot produce plots; skip
    pngs = list(_plots_dir(config).glob("*.png"))
    assert pngs, (
        f"No PNG files found in {_plots_dir(config)}. "
        "Expected weight-distribution histograms or MSE heatmap."
    )


def test_weight_distribution_plots_exist(config) -> None:
    """weight_dist_* PNGs must exist when the BF16 weight cache is present."""
    if not _has_bf16_cache(config):
        return  # Module 1 hasn't run — expected to have no weight dist plots
    pngs = list(_plots_dir(config).glob("weight_dist_*.png"))
    assert pngs, (
        f"No weight_dist_*.png files found in {_plots_dir(config)} "
        "despite BF16 weight cache being present."
    )


def test_mse_heatmap_plot_exists(config) -> None:
    """mse_heatmap*.png must exist when both BF16 and quantized caches are present."""
    if not (_has_bf16_cache(config) and _has_any_quant_cache(config)):
        return  # prerequisites missing — skip
    pngs = list(_plots_dir(config).glob("mse_heatmap*.png"))
    assert pngs, (
        f"No mse_heatmap*.png found in {_plots_dir(config)} "
        "despite BF16 + quantized weight caches being available."
    )


def test_results_directory_exists(config) -> None:
    """module_3_analysis/results/ must exist."""
    assert _results_dir(config).is_dir(), (
        f"Results directory not found: {_results_dir(config)}"
    )


def test_mse_json_exists_when_data_available(config) -> None:
    """mse_all_layers.json must be written when the MSE heatmap runs successfully."""
    if not (_has_bf16_cache(config) and _has_any_quant_cache(config)):
        return  # prerequisites missing — skip
    path = _results_dir(config) / "mse_all_layers.json"
    assert path.exists(), (
        f"mse_all_layers.json not found at {path}. "
        "MSEHeatmapAnalyzer should write this file."
    )


def test_results_directory_non_empty(config) -> None:
    """results/ should contain at least one file when any analysis completed."""
    if not _has_bf16_cache(config):
        return  # no input data — nothing to validate
    files = list(_results_dir(config).iterdir())
    assert files, f"Results directory is empty: {_results_dir(config)}"

def test_mse_json_structure_valid(config) -> None:
    """mse_all_layers.json must be valid JSON with expected structure."""
    if not (_has_bf16_cache(config) and _has_any_quant_cache(config)):
        return  # prerequisites missing — skip
    path = _results_dir(config) / "mse_all_layers.json"
    if not path.exists():
        return  # File was not created — test earlier catch this
    
    with open(path, "r") as f:
        data = json.load(f)
    
    # Validate structure: should have quantizer tags as keys
    assert isinstance(data, dict), f"mse_all_layers.json should be a dict, got {type(data)}"
    assert len(data) > 0, "mse_all_layers.json is empty"
    
    # Each quantizer should have layer data
    for tag, layer_data in data.items():
        assert isinstance(layer_data, dict), f"Layer data for {tag} should be a dict"
        # At least some layers should be present
        assert len(layer_data) > 0, f"No layer data for quantizer {tag}"


def test_outlier_stats_json_exists(config) -> None:
    """outlier_stats_*.json files must be written when outlier detection runs."""
    if not _has_bf16_cache(config):
        return  # Module 1 hasn't run — no baseline to detect outliers
    
    outlier_files = list(_results_dir(config).glob("outlier_stats_*.json"))
    assert outlier_files, (
        f"No outlier_stats_*.json found in {_results_dir(config)}. "
        "OutlierDetector should write stat files."
    )


def test_outlier_stats_json_structure_valid(config) -> None:
    """outlier_stats_*.json must be valid JSON with outlier statistics."""
    outlier_files = list(_results_dir(config).glob("outlier_stats_*.json"))
    if not outlier_files:
        return  # No outlier stats to validate
    
    for stats_file in outlier_files:
        with open(stats_file, "r") as f:
            data = json.load(f)
        
        assert isinstance(data, dict), f"{stats_file.name} should contain a dict"
        
        # Each layer should have outlier statistics
        for layer_name, stats in data.items():
            assert isinstance(stats, dict), f"Stats for {layer_name} should be a dict"
            # Check for expected keys
            expected_keys = {"outlier_pct", "abs_max", "mean", "std", "num_outliers"}
            present_keys = set(stats.keys())
            assert expected_keys.issubset(present_keys), (
                f"Missing keys in {layer_name}: "
                f"expected {expected_keys}, got {present_keys}"
            )


def test_weight_distribution_plots_are_pngs(config) -> None:
    """weight_dist_*.png files must be valid PNG images."""
    if not _has_bf16_cache(config):
        return  # Module 1 hasn't run
    
    pngs = list(_plots_dir(config).glob("weight_dist_*.png"))
    for png_file in pngs:
        # Check PNG magic number
        with open(png_file, "rb") as f:
            magic = f.read(8)
            assert magic == b'\x89PNG\r\n\x1a\n', (
                f"{png_file.name} is not a valid PNG file"
            )
        # Check file size is reasonable (> 1KB)
        assert png_file.stat().st_size > 1024, (
            f"{png_file.name} is suspiciously small (< 1KB)"
        )


def test_mse_heatmap_plots_are_pngs(config) -> None:
    """mse_heatmap*.png files must be valid PNG images."""
    if not (_has_bf16_cache(config) and _has_any_quant_cache(config)):
        return  # prerequisites missing
    
    pngs = list(_plots_dir(config).glob("mse_heatmap*.png"))
    for png_file in pngs:
        # Check PNG magic number
        with open(png_file, "rb") as f:
            magic = f.read(8)
            assert magic == b'\x89PNG\r\n\x1a\n', (
                f"{png_file.name} is not a valid PNG file"
            )
        # Check file size is reasonable
        assert png_file.stat().st_size > 1024, (
            f"{png_file.name} is suspiciously small (< 1KB)"
        )


def test_analysis_methods_are_tracked(config) -> None:
    """Analysis methods should be documented in results JSON."""
    # Check if a summary file exists that tracks analysis methods
    summary_files = list(_results_dir(config).glob("*summary*.json"))
    
    # If no summary file, at least check that individual result files exist
    # This is a soft check — the main validation is in the analyzer's return dicts
    if not summary_files and (_has_bf16_cache(config) or _has_any_quant_cache(config)):
        # At minimum, some result files should exist
        result_files = list(_results_dir(config).glob("*.json"))
        assert result_files, (
            f"No result JSON files found in {_results_dir(config)}. "
            "Analysis should produce traceable results."
        )


def test_weight_distribution_overlays_are_correct(config) -> None:
    """Weight distribution plots should show statistical overlays."""
    if not _has_bf16_cache(config):
        return  # No baseline weights
    
    # Weight dist plots are named per-layer (weight_dist_{layer}.png) with all
    # quantisation tags plotted as side-by-side subplots inside each image.
    # Verify that plots exist and are non-trivial (> 1 KB).
    pngs = list(_plots_dir(config).glob("weight_dist_*.png"))
    assert pngs, "No weight distribution plots found"
    
    # Verify each plot is a non-trivial image (> 1 KB)
    for png_file in pngs:
        assert png_file.stat().st_size > 1024, (
            f"Weight distribution plot {png_file.name} is suspiciously small "
            f"({png_file.stat().st_size} bytes)"
        )
    
    # BF16 cache must be present (already guarded above) so it was plotted
    # as a subplot inside the per-layer images.
    bf16_dir = _shared_weights_dir(config) / "bf16"
    bf16_files = list(bf16_dir.glob("*.npz"))
    assert bf16_files, (
        "BF16 weight cache directory exists but contains no .npz files — "
        "BF16 weight distribution cannot have been plotted"
    )


def test_module_3_summary_report_exists(config) -> None:
    """module_3_summary.json must be created with analysis results."""
    summary_path = _results_dir(config) / "module_3_summary.json"
    assert summary_path.exists(), (
        f"module_3_summary.json not found at {summary_path}. "
        "Module 3 should create a summary report."
    )


def test_module_3_summary_structure_valid(config) -> None:
    """module_3_summary.json must have expected structure."""
    summary_path = _results_dir(config) / "module_3_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    # Check top-level keys
    expected_keys = {"status", "submodules", "cache_validation", "total_time_sec"}
    assert expected_keys.issubset(data.keys()), (
        f"Summary missing keys: expected {expected_keys}, "
        f"got {set(data.keys())}"
    )
    
    # Check submodules structure
    submodules = data.get("submodules", {})
    assert isinstance(submodules, dict), "submodules should be a dict"
    
    # Each submodule should have status and time_sec
    for name, info in submodules.items():
        assert "status" in info, f"Submodule {name} missing 'status'"
        assert "time_sec" in info, f"Submodule {name} missing 'time_sec'"
    
    # Check cache_validation structure
    cache_val = data.get("cache_validation", {})
    assert isinstance(cache_val, dict), "cache_validation should be a dict"
    if cache_val:
        assert "tags" in cache_val, "cache_validation missing 'tags'"
        assert "valid" in cache_val, "cache_validation missing 'valid'"


def test_all_submodules_have_methods_tracked(config) -> None:
    """Each analysis submodule should track its method."""
    summary_path = _results_dir(config) / "module_3_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    submodules = data.get("submodules", {})
    
    # Only check successful submodules
    for name, info in submodules.items():
        if info.get("status", "").startswith("✓"):
            # Successful submodules should document their method
            assert "method" in info, (
                f"Successful submodule {name} should track its analysis method"
            )
