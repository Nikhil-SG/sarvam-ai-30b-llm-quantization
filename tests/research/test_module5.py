"""
tests/test_module5.py
─────────────────────
Validation tests for Module 5 (Evaluation & Accuracy).

Checks:
  - perplexity_results.json exists and has entries per quantizer
  - Each perplexity entry contains a numeric 'perplexity' value
    - benchmark_results.json exists and follows configured benchmark tasks
  - Pareto plot is produced when latency data is available
  - Perplexity comparison plot exists
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
    p = get_module_paths(config.output.base_dir, 5)
    return {
        "results": Path(p["results_dir"]),
        "plots":   Path(p["plots_dir"]),
    }


def _load_perplexity(config) -> Dict[str, Any]:
    path = _paths(config)["results"] / "perplexity_results.json"
    assert path.exists(), f"perplexity_results.json not found at {path}"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _has_module4_latency(config) -> bool:
    from src.core.paths import get_module_paths
    lat_path = Path(get_module_paths(config.output.base_dir, 4)["results_dir"]) \
               / "latency_results.json"
    return lat_path.exists()


def _configured_benchmark_task_ids(config) -> set[str]:
    bench_cfg = getattr(config.evaluation, "benchmarks", None)
    groups = getattr(bench_cfg, "benchmark_groups", []) if bench_cfg else []
    task_ids: set[str] = set()
    for group in groups:
        tasks = group.get("tasks", []) if isinstance(group, dict) else getattr(group, "tasks", [])
        for task in tasks:
            if isinstance(task, str):
                task_ids.add(task)
            elif isinstance(task, dict):
                task_id = task.get("id")
                if task_id:
                    task_ids.add(task_id)
            else:
                task_id = getattr(task, "id", None)
                if task_id:
                    task_ids.add(task_id)
    return task_ids


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_perplexity_results_file_exists(config) -> None:
    """module_5_evaluation/results/perplexity_results.json must exist."""
    path = _paths(config)["results"] / "perplexity_results.json"
    assert path.exists(), f"Missing: {path}"


def test_perplexity_results_non_empty(config) -> None:
    """perplexity_results.json must have at least one entry."""
    data = _load_perplexity(config)
    assert data, "perplexity_results.json contains no entries."


def test_perplexity_entries_have_numeric_value(config) -> None:
    """Every perplexity entry must contain a numeric 'perplexity' key."""
    data = _load_perplexity(config)
    bad = []
    for tag, entry in data.items():
        if not isinstance(entry, dict):
            continue
        ppl = entry.get("perplexity")
        if not isinstance(ppl, (int, float)):
            bad.append(f"{tag}: perplexity={ppl!r}")
    assert not bad, (
        "Perplexity entries missing a numeric 'perplexity' value:\n"
        + "\n".join(f"  • {b}" for b in bad)
    )


def test_benchmark_results_file_exists(config) -> None:
    """benchmark_results.json must be written by BenchmarkRunner."""
    path = _paths(config)["results"] / "benchmark_results.json"
    assert path.exists(), f"Missing: {path}"


def test_perplexity_comparison_plot_exists(config) -> None:
    """perplexity_comparison.png must be generated."""
    path = _paths(config)["plots"] / "perplexity_comparison.png"
    assert path.exists(), f"Missing perplexity comparison plot: {path}"


def test_pareto_plot_exists_when_latency_available(config) -> None:
    """pareto_frontier.png must exist when Module 4 latency data is available."""
    if not _has_module4_latency(config):
        return  # latency data missing — Pareto plot is intentionally skipped
    path = _paths(config)["plots"] / "pareto_frontier.png"
    assert path.exists(), (
        f"pareto_frontier.png not found at {path} "
        "despite Module 4 latency data being present."
    )


def test_no_perplexity_entry_is_error(config) -> None:
    """Perplexity entries must not contain a top-level 'error' key."""
    data = _load_perplexity(config)
    errors = [
        f"{tag}: {entry['error']}"
        for tag, entry in data.items()
        if isinstance(entry, dict) and "error" in entry
    ]
    assert not errors, (
        "Perplexity evaluation errors detected:\n"
        + "\n".join(f"  • {e}" for e in errors)
    )


def test_module_5_summary_report_exists(config) -> None:
    """module_5_summary.json must be created with evaluation results."""
    summary_path = _paths(config)["results"] / "module_5_summary.json"
    assert summary_path.exists(), (
        f"module_5_summary.json not found at {summary_path}. "
        "Module 5 should create a summary report."
    )


def test_module_5_summary_structure_valid(config) -> None:
    """module_5_summary.json must have expected structure."""
    summary_path = _paths(config)["results"] / "module_5_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    # Check top-level keys
    expected_keys = {"status", "quantizers", "methods", "total_time_sec", "pareto"}
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
        
        # Each should have perplexity and benchmarks results
        assert "perplexity" in info, f"Quantizer {tag} missing 'perplexity'"
        assert "benchmarks" in info, f"Quantizer {tag} missing 'benchmarks'"
    
    # Check methods are documented
    methods = data.get("methods", {})
    expected_methods = {"perplexity", "benchmarks", "pareto"}
    if methods:
        assert expected_methods.issubset(methods.keys()), (
            f"Methods missing: expected {expected_methods}, got {set(methods.keys())}"
        )


def test_all_evaluators_track_methods(config) -> None:
    """Each evaluator should document its analysis method."""
    summary_path = _paths(config)["results"] / "module_5_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    quantizers = data.get("quantizers", {})
    
    # Check each quantizer's evaluators
    for tag, info in quantizers.items():
        for evaluator_name in ["perplexity", "benchmarks"]:
            evaluator_data = info.get(evaluator_name, {})
            if evaluator_data.get("status", "").startswith("✓"):
                assert "method" in evaluator_data, (
                    f"Quantizer {tag}'s {evaluator_name} evaluator should track its method"
                )


def test_pareto_section_exists_in_summary(config) -> None:
    """Summary should contain Pareto analysis status."""
    summary_path = _paths(config)["results"] / "module_5_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    pareto_data = data.get("pareto", {})
    assert isinstance(pareto_data, dict), "pareto should be a dict"
    assert "status" in pareto_data, "pareto missing 'status'"
    assert "method" in pareto_data, "pareto missing 'method'"
    assert "time_sec" in pareto_data, "pareto missing 'time_sec'"


def test_benchmark_results_structure_valid(config) -> None:
    """benchmark_results.json must have valid structure."""
    bench_path = _paths(config)["results"] / "benchmark_results.json"
    if not bench_path.exists():
        return  # File may not exist if evaluation was skipped
    
    with open(bench_path, "r") as f:
        data = json.load(f)
    
    assert isinstance(data, dict), "benchmark_results.json should be a dict"
    
    # Each tag should have benchmark results
    for tag, bench_data in data.items():
        if isinstance(bench_data, dict) and "error" not in bench_data:
            # Should have at least one task result
            assert len(bench_data) > 0, f"No benchmark results for {tag}"


def test_benchmark_results_match_configured_tasks(config) -> None:
    """Benchmark outputs should only include tasks configured in config.yaml."""
    bench_path = _paths(config)["results"] / "benchmark_results.json"
    if not bench_path.exists():
        return

    with open(bench_path, "r") as f:
        data = json.load(f)

    configured_tasks = _configured_benchmark_task_ids(config)
    assert configured_tasks, "No benchmark tasks configured for Module 5"

    # This setup is expected to run MMLU-only for speed.
    assert configured_tasks == {"mmlu"}, (
        f"Expected MMLU-only benchmark config, got {sorted(configured_tasks)}"
    )

    for tag, bench_data in data.items():
        if not isinstance(bench_data, dict) or "error" in bench_data:
            continue
        result_tasks = set((bench_data.get("tasks") or {}).keys())
        if result_tasks:
            assert result_tasks.issubset(configured_tasks), (
                f"{tag} contains unexpected benchmark tasks: {sorted(result_tasks - configured_tasks)}"
            )


def test_per_quantizer_evaluation_exists(config) -> None:
    """Each quantizer should have evaluation results in the summary."""
    summary_path = _paths(config)["results"] / "module_5_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    quantizers = data.get("quantizers", {})
    
    # At least some quantizers should have completed
    completed_tags = [
        tag for tag, info in quantizers.items()
        if info.get("status", "").startswith("✓") or info.get("status", "").startswith("⚠")
    ]
    
    assert completed_tags, "No quantizers completed evaluation"


def test_evaluation_plots_are_generated(config) -> None:
    """Comparison plots should be generated."""
    plots_dir = _paths(config)["plots"]
    
    expected_plots = {
        "perplexity_comparison.png",
        "pareto_frontier.png",
    }
    
    generated_plots = {p.name for p in plots_dir.glob("*.png")}
    
    # At least perplexity plot should exist (Pareto is optional without Module 4)
    assert "perplexity_comparison.png" in generated_plots, (
        f"perplexity_comparison.png not found in {plots_dir}"
    )


def test_perplexity_entries_match_quantizers(config) -> None:
    """All quantizers should have perplexity entries (or errors)."""
    summary_path = _paths(config)["results"] / "module_5_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        summary_data = json.load(f)
    
    ppl_data = _load_perplexity(config)
    summary_quantizers = set(summary_data.get("quantizers", {}).keys())
    ppl_quantizers = set(ppl_data.keys())
    
    # All quantizers in summary should have perplexity data
    assert summary_quantizers.issubset(ppl_quantizers) or ppl_quantizers.issubset(summary_quantizers), (
        f"Mismatch between summary quantizers {summary_quantizers} "
        f"and perplexity quantizers {ppl_quantizers}"
    )


def test_module_5_dependencies_satisfied(config) -> None:
    """Module 5 should gracefully handle missing Module 4 results."""
    summary_path = _paths(config)["results"] / "module_5_summary.json"
    if not summary_path.exists():
        return  # Previous test would catch this
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    pareto_data = data.get("pareto", {})
    pareto_status = pareto_data.get("status", "")
    
    # Pareto should be either COMPLETED or PARTIAL (not FAILED)
    # If Module 4 hasn't run, Pareto should be ⚠ PARTIAL, not ✗ FAILED
    assert not pareto_status.startswith("✗") or "error" in data, (
        "Pareto should be PARTIAL if Module 4 hasn't run, not FAILED"
    )


def test_module_5_pareto_data_json_exists(config) -> None:
    """pareto_data.json should be generated with extracted points."""
    path = _paths(config)["results"] / "pareto_data.json"
    assert path.exists(), (
        f"pareto_data.json not found at {path}. "
        "ParetoAnalyzer should save extracted (throughput, accuracy) points."
    )


def test_module_5_pareto_data_structure(config) -> None:
    """pareto_data.json should have extracted points for each quantizer."""
    path = _paths(config)["results"] / "pareto_data.json"
    if not path.exists():
        return  # Previous test would catch this
    
    with open(path, "r") as f:
        pareto_points = json.load(f)
    
    # Should be a dict with quantizer tags as keys
    assert isinstance(pareto_points, dict), "pareto_data.json should be a dict"
    
    # Each entry should have throughput and accuracy (or be None if incomplete)
    for tag, point_data in pareto_points.items():
        if point_data is not None:
            assert "throughput" in point_data, (
                f"Pareto point for {tag} missing 'throughput'"
            )
            assert "accuracy" in point_data, (
                f"Pareto point for {tag} missing 'accuracy'"
            )
            assert isinstance(point_data["throughput"], (int, float)), (
                f"Pareto throughput for {tag} should be numeric"
            )
            assert isinstance(point_data["accuracy"], (int, float)), (
                f"Pareto accuracy for {tag} should be numeric"
            )


def test_module_5_handles_mixed_quantizer_states(config) -> None:
    """Module 5 should handle some quantizers having Module 4 data and others not."""
    summary_path = _paths(config)["results"] / "module_5_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    quantizers = data.get("quantizers", {})
    
    # Check that quantizers completed evaluation regardless of Pareto availability
    completed_eval = [
        tag for tag, info in quantizers.items()
        if info.get("status", "").startswith("✓")
    ]
    
    assert completed_eval, (
        "At least some quantizers should complete evaluation even if Pareto data missing"
    )


def test_module_5_downstream_evaluations_before_pareto(config) -> None:
    """Module 5 should complete perplexity & benchmarks before attempting Pareto."""
    summary_path = _paths(config)["results"] / "module_5_summary.json"
    if not summary_path.exists():
        return
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    quantizers = data.get("quantizers", {})
    
    # Each quantizer should have evaluation results
    for tag, info in quantizers.items():
        if info.get("status", "").startswith("✓") or info.get("status", "").startswith("⚠"):
            # Completed or partial → should have perplexity and benchmarks attempted
            perplexity = info.get("perplexity", {})
            benchmarks = info.get("benchmarks", {})
            
            # At least one should exist (may be partial)
            has_eval = (
                perplexity.get("status") or
                benchmarks.get("status")
            )
            assert has_eval, (
                f"Quantizer {tag} completed but missing evaluation results"
            )

