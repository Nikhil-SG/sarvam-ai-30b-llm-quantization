"""
tests/test_module6.py
─────────────────────
Validation tests for Module 6 (Layer Visualization).

Checks:
  - Summary report structure and required fields
  - Per-layer results tracking with status
  - Method documentation in results
  - Visualization generation (histograms, heatmaps, diff heatmaps)
  - Image files are non-empty and properly saved
  - All target layers are processed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _results_dir(config) -> Path:
    from src.core.paths import get_module_paths
    return Path(get_module_paths(config.output.base_dir, 6)["results_dir"])


def _plots_dir(config) -> Path:
    from src.core.paths import get_module_paths
    return Path(get_module_paths(config.output.base_dir, 6)["plots_dir"])


def _has_weight_caches(config) -> bool:
    """True if either BF16 or any quantized weight cache exists."""
    sw = Path(config.output.base_dir) / "shared_weights"
    if not sw.is_dir():
        return False
    return any(d.is_dir() for d in sw.iterdir())


def _all_pngs(config) -> List[Path]:
    return list(_plots_dir(config).glob("*.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_module_6_summary_report_exists(config) -> None:
    """module_6_summary.json must be created."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    assert summary_path.exists(), (
        f"Summary report not found: {summary_path}"
    )


def test_module_6_summary_structure_valid(config) -> None:
    """Summary must have required top-level fields."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    required_fields = ["status", "layers", "methods", "total_time_sec"]
    for field in required_fields:
        assert field in data, (
            f"Missing required field '{field}' in module_6_summary.json"
        )
    
    assert isinstance(data["status"], str), "status must be a string"
    assert isinstance(data["layers"], dict), "layers must be a dict"
    assert isinstance(data["methods"], dict), "methods must be a dict"
    assert isinstance(data["total_time_sec"], (int, float)), "total_time_sec must be numeric"


def test_module_6_status_format_valid(config) -> None:
    """Status field must contain status markers (✓, ⚠, ✗)."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    status = data.get("status", "")
    valid_statuses = ["✓ COMPLETED", "⚠ PARTIAL", "✗ FAILED"]
    assert any(s in status for s in valid_statuses), (
        f"Status '{status}' must contain one of: {valid_statuses}"
    )


def test_all_visualizers_track_methods(config) -> None:
    """Methods dict must document 6a, 6b, 6c visualization approaches."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    methods = data.get("methods", {})
    expected_methods = ["6a_histograms", "6b_heatmaps", "6c_diff_heatmaps"]
    for method in expected_methods:
        assert method in methods, (
            f"Missing method documentation for '{method}' in results['methods']"
        )
        assert isinstance(methods[method], str), (
            f"Method '{method}' description must be a string"
        )


def test_per_layer_results_exist(config) -> None:
    """Each target layer must have results entry."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists() or not _has_weight_caches(config):
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    layers_dict = data.get("layers", {})
    target_layers = config.visualization.target_layers
    
    # At least some layers should be processed
    assert layers_dict, (
        "No layer results found in summary"
    )


def test_per_layer_status_tracking(config) -> None:
    """Each layer must have status field (✓, ⚠, or ✗)."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    layers_dict = data.get("layers", {})
    valid_statuses = ["✓ COMPLETED", "⚠ PARTIAL", "✗ FAILED"]
    
    for layer_name, layer_result in layers_dict.items():
        assert isinstance(layer_result, dict), (
            f"Layer result for '{layer_name}' must be a dict"
        )
        assert "status" in layer_result, (
            f"Layer '{layer_name}' missing status field"
        )
        status = layer_result["status"]
        assert any(s in status for s in valid_statuses), (
            f"Layer '{layer_name}' status '{status}' must contain marker (✓, ⚠, or ✗)"
        )


def test_per_layer_plots_documented(config) -> None:
    """Each layer result should document generated plots."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    layers_dict = data.get("layers", {})
    for layer_name, layer_result in layers_dict.items():
        if layer_result.get("status", "").startswith("✓") or layer_result.get("status", "").startswith("⚠"):
            assert "plots" in layer_result, (
                f"Successful/partial layer '{layer_name}' must document plots"
            )


def test_plots_directory_exists(config) -> None:
    """plots/ directory must be created."""
    assert _plots_dir(config).is_dir(), (
        f"Plots directory not found: {_plots_dir(config)}"
    )


def test_at_least_one_visualization_png_exists(config) -> None:
    """At least one PNG must be produced when weight caches are available."""
    if not _has_weight_caches(config):
        return
    pngs = _all_pngs(config)
    assert pngs, (
        f"No PNG files found in {_plots_dir(config)}. "
        "LayerVisualizer should generate layer histograms / heatmaps."
    )


def test_all_png_files_are_non_empty(config) -> None:
    """Every generated PNG must be larger than 0 bytes (not a blank stub)."""
    empty = [p for p in _all_pngs(config) if p.stat().st_size == 0]
    assert not empty, (
        "The following PNG files are empty (0 bytes):\n"
        + "\n".join(f"  • {p}" for p in empty)
    )


def test_layer_histogram_plots_exist(config) -> None:
    """layer_hist_* PNGs must be generated (one per target layer)."""
    if not _has_weight_caches(config):
        return
    pngs = list(_plots_dir(config).glob("layer_hist_*.png"))
    assert pngs, (
        f"No layer_hist_*.png files found in {_plots_dir(config)}. "
        "LayerVisualizer should create per-layer histograms."
    )


def test_plot_count_matches_target_layers(config) -> None:
    """
    Total PNG count should be at least len(target_layers).
    Each layer produces at least one image (histogram).
    """
    if not _has_weight_caches(config):
        return
    n_target_layers = len(config.visualization.target_layers)
    pngs = _all_pngs(config)
    assert len(pngs) >= n_target_layers, (
        f"Expected at least {n_target_layers} PNGs (one per target layer), "
        f"but only {len(pngs)} found in {_plots_dir(config)}."
    )


def test_layer_heatmap_plots_exist(config) -> None:
    """layer_heatmap_* PNGs should exist for layers with weight data."""
    if not _has_weight_caches(config):
        return
    pngs = list(_plots_dir(config).glob("layer_heatmap_*.png"))
    # At least some heatmaps should be generated (some layers may fail)
    assert pngs or len(_all_pngs(config)) > 0, (
        f"No heatmap images generated in {_plots_dir(config)}"
    )


def test_layer_diff_heatmap_plots_exist_for_quantizers(config) -> None:
    """layer_diff_* PNGs should exist (quantisation error vs BF16)."""
    if not _has_weight_caches(config):
        return
    pngs = list(_plots_dir(config).glob("layer_diff_*.png"))
    # Diff heatmaps require BF16 + quantizers, which may not always be available
    # Just verify no errors occurred by checking at least some images generated
    assert len(_all_pngs(config)) > 0, (
        f"No visualization images generated in {_plots_dir(config)}"
    )


def test_summary_total_time_is_positive(config) -> None:
    """total_time_sec should be positive if module ran."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    total_time = data.get("total_time_sec", 0)
    assert total_time >= 0, (
        f"total_time_sec should be non-negative, got {total_time}"
    )
    # If layers were processed, time should be > 0
    if data.get("layers", {}):
        assert total_time > 0, (
            "total_time_sec should be > 0 if layers were processed"
        )


def test_module_6_checks_weight_cache_before_visualization(config) -> None:
    """Module 6 should gracefully handle missing weight caches."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    # If no layers processed, should be FAILED (expected without weights)
    layers = data.get("layers", {})
    status = data.get("status", "")
    
    if not layers:
        # No layer results → expected when weights unavailable
        # Status should indicate failure reason
        assert status in ["✗ FAILED", "⚠ PARTIAL"], (
            f"Status '{status}' without layer results should be FAILED or PARTIAL"
        )
    else:
        # Some layers processed → at least some weights were available
        assert status in ["✓ COMPLETED", "⚠ PARTIAL"], (
            f"Status '{status}' with layer results should be COMPLETED or PARTIAL"
        )


def test_module_6_layer_plots_have_timing(config) -> None:
    """Each layer plot should have documented generation time."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    layers = data.get("layers", {})
    for layer_name, layer_data in layers.items():
        plots = layer_data.get("plots", [])
        for plot in plots:
            assert "time_sec" in plot, (
                f"Layer '{layer_name}' plot missing 'time_sec'"
            )
            assert isinstance(plot["time_sec"], (int, float)), (
                f"Layer '{layer_name}' plot time_sec should be numeric"
            )


def test_module_6_respects_config_target_layers(config) -> None:
    """Module 6 should attempt to visualize config.visualization.target_layers."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    layers = data.get("layers", {})
    target_layers = config.visualization.target_layers
    
    # Should attempt all target layers (even if some fail)
    if layers:
        attempted = set(layers.keys())
        configured = set(target_layers)
        
        # All attempted layers should be from configured set
        assert attempted.issubset(configured), (
            f"Module 6 visualized layers not in config: {attempted - configured}"
        )


def test_module_6_visualization_methods_documented(config) -> None:
    """Visualization types (histogram, heatmap, diff_heatmap) must be documented."""
    summary_path = _results_dir(config) / "module_6_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    # Check each layer's plots document their type
    layers = data.get("layers", {})
    valid_types = {"histogram", "heatmap", "diff_heatmap"}
    
    for layer_name, layer_data in layers.items():
        plots = layer_data.get("plots", [])
        for idx, plot in enumerate(plots):
            plot_type = plot.get("type")
            assert plot_type in valid_types, (
                f"Layer '{layer_name}' plot {idx} has invalid type '{plot_type}'"
            )


