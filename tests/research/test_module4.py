"""
tests/test_module4.py
─────────────────────
Validation tests for Module 4 (Inference & Resource Profiling).

Checks:
  - latency_results.json, vram_results.json, disk_results.json exist
  - latency_results.json contains at least one quantizer entry
  - Each latency entry has tokens_per_second or mean_latency_ms
  - Comparison plots exist (latency, vram, disk PNGs)
  - No entry has a bare 'error' string as its entire value
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _paths(config) -> Dict[str, Path]:
    from src.core.paths import get_module_paths
    p = get_module_paths(config.output.base_dir, 4)
    return {
        "results": Path(p["results_dir"]),
        "plots":   Path(p["plots_dir"]),
    }


def _load_latency(config) -> Dict[str, Any]:
    path = _paths(config)["results"] / "latency_results.json"
    assert path.exists(), f"latency_results.json not found at {path}"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_latency_results_file_exists(config) -> None:
    """module_4_profiling/results/latency_results.json must exist."""
    path = _paths(config)["results"] / "latency_results.json"
    assert path.exists(), f"Missing: {path}"


def test_vram_results_file_exists(config) -> None:
    """module_4_profiling/results/vram_results.json must exist."""
    path = _paths(config)["results"] / "vram_results.json"
    assert path.exists(), f"Missing: {path}"


def test_disk_results_file_exists(config) -> None:
    """module_4_profiling/results/disk_results.json must exist."""
    path = _paths(config)["results"] / "disk_results.json"
    assert path.exists(), f"Missing: {path}"


def test_latency_results_non_empty(config) -> None:
    """latency_results.json must have at least one quantizer entry."""
    data = _load_latency(config)
    assert data, "latency_results.json is an empty JSON object — no profiling data recorded."


def test_latency_entries_have_throughput_key(config) -> None:
    """Each latency entry must record throughput data (tokens_per_sec nested in batch dicts)."""
    data = _load_latency(config)
    valid_keys = {"tokens_per_sec", "tokens_per_second", "mean_latency_ms", "latency_ms", "tps"}
    missing_entries = []
    for tag, entry in data.items():
        if not isinstance(entry, dict):
            continue
        # Check if the entry itself has a valid key (flat structure)
        if any(k in entry for k in valid_keys):
            continue
        # Check nested batch_size dicts (e.g. {"1": {"tokens_per_sec": ...}, ...})
        has_throughput = False
        for bs_key, bs_val in entry.items():
            if isinstance(bs_val, dict) and any(k in bs_val for k in valid_keys):
                has_throughput = True
                break
        if not has_throughput and "error" not in entry:
            # Also skip if all batch-size sub-entries are errors (e.g. all OOM)
            sub_dicts = [v for v in entry.values() if isinstance(v, dict)]
            all_sub_errors = sub_dicts and all("error" in d for d in sub_dicts)
            if not all_sub_errors:
                missing_entries.append(tag)
    assert not missing_entries, (
        f"Latency entries missing throughput keys {valid_keys}: {missing_entries}"
    )


def test_latency_comparison_plot_exists(config) -> None:
    """latency_comparison.png must be saved in plots/."""
    path = _paths(config)["plots"] / "latency_comparison.png"
    assert path.exists(), f"Missing latency comparison plot: {path}"


def test_vram_comparison_plot_exists(config) -> None:
    """vram_comparison.png must be saved in plots/."""
    path = _paths(config)["plots"] / "vram_comparison.png"
    assert path.exists(), f"Missing VRAM comparison plot: {path}"


def test_disk_comparison_plot_exists(config) -> None:
    """disk_comparison.png must be saved in plots/."""
    path = _paths(config)["plots"] / "disk_comparison.png"
    assert path.exists(), f"Missing disk comparison plot: {path}"


def test_no_latency_entry_is_pure_error(config) -> None:
    """No top-level latency entry should be just an error string (profiler crashed entirely)."""
    data = _load_latency(config)
    error_tags = [
        tag for tag, val in data.items()
        if isinstance(val, str) and "error" in val.lower()
    ]
    assert not error_tags, (
        f"Latency entries recorded as plain error strings: {error_tags}"
    )


def test_module_4_summary_report_exists(config) -> None:
    """module_4_summary.json must be created with profiling results."""
    summary_path = _paths(config)["results"] / "module_4_summary.json"
    assert summary_path.exists(), (
        f"module_4_summary.json not found at {summary_path}. "
        "Module 4 should create a summary report."
    )


def test_module_4_summary_structure_valid(config) -> None:
    """module_4_summary.json must have expected structure."""
    summary_path = _paths(config)["results"] / "module_4_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    # Check top-level keys
    expected_keys = {"status", "quantizers", "methods", "total_time_sec"}
    assert expected_keys.issubset(data.keys()), (
        f"Summary missing keys: expected {expected_keys}, "
        f"got {set(data.keys())}"
    )
    
    # Check quantizers structure
    quantizers = data.get("quantizers", {})
    assert isinstance(quantizers, dict), "quantizers should be a dict"
    
    # Each quantizer should have status and timing
    for tag, info in quantizers.items():
        assert "status" in info, f"Quantizer {tag} missing 'status'"
        assert "time_sec" in info, f"Quantizer {tag} missing 'time_sec'"
        
        # Each should have latency, vram, disk results
        assert "latency" in info, f"Quantizer {tag} missing 'latency'"
        assert "vram" in info, f"Quantizer {tag} missing 'vram'"
        assert "disk" in info, f"Quantizer {tag} missing 'disk'"
    
    # Check methods are documented
    methods = data.get("methods", {})
    expected_methods = {"latency", "vram", "disk"}
    if methods:
        assert expected_methods.issubset(methods.keys()), (
            f"Methods missing: expected {expected_methods}, got {set(methods.keys())}"
        )


def test_all_profilers_track_methods(config) -> None:
    """Each profiler should document its analysis method."""
    summary_path = _paths(config)["results"] / "module_4_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    quantizers = data.get("quantizers", {})
    
    # Check each quantizer's profilers
    for tag, info in quantizers.items():
        for profiler_name in ["latency", "vram", "disk"]:
            profiler_data = info.get(profiler_name, {})
            if profiler_data.get("status", "").startswith("✓"):
                assert "method" in profiler_data, (
                    f"Quantizer {tag}'s {profiler_name} profiler should track its method"
                )


def test_vram_results_structure_valid(config) -> None:
    """vram_results.json must have valid structure."""
    vram_path = _paths(config)["results"] / "vram_results.json"
    if not vram_path.exists():
        return  # File may not exist if profiling was skipped
    
    with open(vram_path, "r") as f:
        data = json.load(f)
    
    assert isinstance(data, dict), "vram_results.json should be a dict"
    
    # Each tag should have snapshot info
    for tag, vram_data in data.items():
        if isinstance(vram_data, dict) and "error" not in vram_data:
            assert "snapshot" in vram_data or "peak_gb" in vram_data, (
                f"VRAM data for {tag} should have snapshot or peak_gb"
            )


def test_disk_results_structure_valid(config) -> None:
    """disk_results.json must have valid structure."""
    disk_path = _paths(config)["results"] / "disk_results.json"
    if not disk_path.exists():
        return  # File may not exist if profiling was skipped
    
    with open(disk_path, "r") as f:
        data = json.load(f)
    
    assert isinstance(data, dict), "disk_results.json should be a dict"
    
    # Each tag should have size info
    for tag, disk_data in data.items():
        if isinstance(disk_data, dict) and "error" not in disk_data:
            assert "model_size_gb" in disk_data or "total_gb" in disk_data, (
                f"Disk data for {tag} should have model_size_gb or total_gb"
            )


def test_per_quantizer_profiling_exists(config) -> None:
    """Each quantizer should have profiling results in the summary."""
    summary_path = _paths(config)["results"] / "module_4_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    quantizers = data.get("quantizers", {})
    expected_tags = ["bf16", "int8", "fp8", "nf4", "gptq"]

    missing_tags = [tag for tag in expected_tags if tag not in quantizers]
    assert not missing_tags, (
        f"Module 4 summary is missing expected quantizer entries: {missing_tags}"
    )

    represented_tags = [
        tag for tag, info in quantizers.items()
        if info.get("status", "").startswith("✓") or info.get("status", "").startswith("⚠")
    ]
    assert represented_tags, "No quantizers were represented in Module 4 summary"


def test_skipped_quantizer_entries_remain_structured(config) -> None:
    """Skipped quantizer entries should still keep latency/vrAM/disk sub-objects."""
    summary_path = _paths(config)["results"] / "module_4_summary.json"
    if not summary_path.exists():
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    for tag, info in data.get("quantizers", {}).items():
        for profiler_name in ["latency", "vram", "disk"]:
            profiler_data = info.get(profiler_name, {})
            assert isinstance(profiler_data, dict), (
                f"Quantizer {tag} profiler {profiler_name} should be a dict"
            )
            if profiler_data.get("status", "").startswith("⚠"):
                assert "error" in profiler_data, (
                    f"Skipped profiler {profiler_name} for {tag} should explain why it was skipped"
                )


def test_profiling_plots_are_generated(config) -> None:
    """Comparison plots should be generated."""
    plots_dir = _paths(config)["plots"]
    
    expected_plots = {
        "latency_comparison.png",
        "vram_comparison.png",
        "disk_comparison.png",
    }
    
    generated_plots = {p.name for p in plots_dir.glob("*.png")}
    
    # At least some comparison plots should exist
    found = expected_plots.intersection(generated_plots)
    assert found, (
        f"No comparison plots found. Expected at least one of {expected_plots}, "
        f"but got {generated_plots}"
    )

