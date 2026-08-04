"""
Tests for MxMoE Module 3: Evaluation & Ablation.

Mirrors the research evaluation test coverage for:
  - Perplexity results
  - Benchmark results
  - Evaluation structure
  - Plots
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.core.paths import get_mxmoe_module_paths
from tests.test_runner import SkipTest


def _paths(config) -> Dict[str, Path]:
    base_dir = getattr(config.output, "base_dir", "mxmoe/outputs")
    p = get_mxmoe_module_paths(base_dir, 3)
    return {
        "results": Path(p["results_dir"]),
        "plots": Path(p["plots_dir"]),
    }


def _require(path: Path, msg: str) -> None:
    if not path.exists():
        raise SkipTest(msg)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def _eval_results_path(config) -> Path:
    return _paths(config)["results"] / "eval_results_full.json"


def test_eval_results_exist(config) -> None:
    res_dir = _paths(config)["results"]
    fp = res_dir / "eval_results_full.json"

    if not fp.exists():
        ab = res_dir / "ablation_results.json"
        if not ab.exists():
            raise SkipTest(
                f"Module 3 artifacts missing: run MxMoE module 3 first ({res_dir})"
            )
        data = _load_json(ab)
        assert (
            "base_evaluation" in data
            or "variants" in data
            or "variant_results" in data
            or "study_config" in data
        ), "ablation_results.json does not contain expected evaluation/variant keys"
        return

    data = _load_json(fp)
    assert "evaluations" in data, "eval_results_full.json missing evaluations key"


def test_eval_results_include_perplexity_and_benchmarks(config) -> None:
    fp = _eval_results_path(config)
    _require(fp, f"Module 3 eval missing: {fp}")
    data = _load_json(fp)
    evaluations = data.get("evaluations", {})

    has_ppl = any(key.startswith("perplexity_") for key in evaluations.keys())
    has_bench = any(key.startswith("benchmarks_") for key in evaluations.keys())
    assert has_ppl, "No perplexity_* entries found in evaluations"
    assert has_bench, "No benchmarks_* entries found in evaluations"


def test_perplexity_results_file_exists(config) -> None:
    fp = _eval_results_path(config)
    _require(fp, f"Module 3 eval missing: {fp}")
    path = _paths(config)["results"] / "perplexity_results.json"
    assert path.exists(), f"Missing: {path}"


def test_perplexity_entries_have_numeric_value(config) -> None:
    fp = _eval_results_path(config)
    _require(fp, f"Module 3 eval missing: {fp}")
    path = _paths(config)["results"] / "perplexity_results.json"
    _require(path, f"Perplexity results missing: {path}")

    data = _load_json(path)
    bad = []
    for tag, entry in data.items():
        if not isinstance(entry, dict):
            continue
        ppl = entry.get("perplexity")
        if not isinstance(ppl, (int, float)):
            bad.append(f"{tag}: perplexity={ppl!r}")
    assert not bad, (
        "Perplexity entries missing a numeric 'perplexity' value:\n"
        + "\n".join(f"  - {b}" for b in bad)
    )


def test_benchmark_results_file_exists(config) -> None:
    fp = _eval_results_path(config)
    _require(fp, f"Module 3 eval missing: {fp}")
    path = _paths(config)["results"] / "benchmark_results.json"
    assert path.exists(), f"Missing: {path}"


def test_benchmark_results_match_configured_tasks(config) -> None:
    fp = _eval_results_path(config)
    _require(fp, f"Module 3 eval missing: {fp}")
    bench_path = _paths(config)["results"] / "benchmark_results.json"
    _require(bench_path, f"Benchmark results missing: {bench_path}")

    data = _load_json(bench_path)
    configured_tasks = _configured_benchmark_task_ids(config)
    assert configured_tasks, "No benchmark tasks configured for Module 3"

    for _, bench_data in data.items():
        if not isinstance(bench_data, dict) or "error" in bench_data:
            continue
        result_tasks = set((bench_data.get("tasks") or {}).keys())
        if result_tasks:
            assert result_tasks.issubset(configured_tasks), (
                f"Benchmark results contain tasks not in config: {sorted(result_tasks - configured_tasks)}"
            )


def test_perplexity_comparison_plot_exists(config) -> None:
    fp = _eval_results_path(config)
    _require(fp, f"Module 3 eval missing: {fp}")
    path = _paths(config)["plots"] / "perplexity_comparison.png"
    assert path.exists(), f"Missing perplexity comparison plot: {path}"


def test_benchmark_summary_plot_exists(config) -> None:
    fp = _eval_results_path(config)
    _require(fp, f"Module 3 eval missing: {fp}")
    path = _paths(config)["plots"] / "benchmark_accuracy_table.png"
    assert path.exists(), f"Missing benchmark summary plot: {path}"