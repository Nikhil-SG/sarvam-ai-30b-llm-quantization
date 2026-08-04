"""
Tests for MxMoE Module 4: Deployment Readiness + Strategy Profiling.

Mirrors the research profiling coverage for:
  - latency_results.json, vram_results.json, disk_results.json
  - profiling plots
  - Pareto plot when latency + benchmarks are available
  - vLLM profiling + model card
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.core.paths import get_mxmoe_module_paths
from tests.test_runner import SkipTest


def _paths(config) -> Dict[str, Path]:
    base_dir = getattr(config.output, "base_dir", "mxmoe/outputs")
    p = get_mxmoe_module_paths(base_dir, 4)
    return {
        "results": Path(p["results_dir"]),
        "plots": Path(p["plots_dir"]),
    }


def _module3_results_dir(config) -> Path:
    base_dir = getattr(config.output, "base_dir", "mxmoe/outputs")
    p = get_mxmoe_module_paths(base_dir, 3)
    return Path(p["results_dir"])


def _has_any_module4_artifacts(res_dir: Path) -> bool:
    candidates = [
        res_dir / "vllm_profiling_results.json",
        res_dir / "MODEL_CARD.md",
        res_dir / "latency_results.json",
        res_dir / "vram_results.json",
        res_dir / "disk_results.json",
    ]
    return any(path.exists() for path in candidates)


def _require_file(path: Path, res_dir: Path) -> None:
    if not _has_any_module4_artifacts(res_dir):
        raise SkipTest(f"Module 4 artifacts missing: run MxMoE module 4 first ({res_dir})")
    assert path.exists(), f"Missing: {path}"


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_latency_results_file_exists(config) -> None:
    res_dir = _paths(config)["results"]
    path = res_dir / "latency_results.json"
    _require_file(path, res_dir)


def test_vram_results_file_exists(config) -> None:
    res_dir = _paths(config)["results"]
    path = res_dir / "vram_results.json"
    _require_file(path, res_dir)


def test_disk_results_file_exists(config) -> None:
    res_dir = _paths(config)["results"]
    path = res_dir / "disk_results.json"
    _require_file(path, res_dir)


def test_latency_entries_have_throughput_key(config) -> None:
    res_dir = _paths(config)["results"]
    path = res_dir / "latency_results.json"
    _require_file(path, res_dir)

    data = _load_json(path)
    valid_keys = {"tokens_per_sec", "tokens_per_second", "mean_latency_ms", "latency_ms", "tps"}
    missing_entries = []
    for tag, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if any(k in entry for k in valid_keys):
            continue
        has_throughput = False
        for _, bs_val in entry.items():
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
    res_dir = _paths(config)["results"]
    plot = _paths(config)["plots"] / "latency_comparison.png"
    _require_file(res_dir / "latency_results.json", res_dir)
    assert plot.exists(), f"Missing latency comparison plot: {plot}"


def test_vram_comparison_plot_exists(config) -> None:
    res_dir = _paths(config)["results"]
    plot = _paths(config)["plots"] / "vram_comparison.png"
    _require_file(res_dir / "vram_results.json", res_dir)
    assert plot.exists(), f"Missing VRAM comparison plot: {plot}"


def test_disk_comparison_plot_exists(config) -> None:
    res_dir = _paths(config)["results"]
    plot = _paths(config)["plots"] / "disk_comparison.png"
    _require_file(res_dir / "disk_results.json", res_dir)
    assert plot.exists(), f"Missing disk comparison plot: {plot}"


def test_pareto_plot_exists_when_latency_and_benchmarks_available(config) -> None:
    res_dir = _paths(config)["results"]
    latency_path = res_dir / "latency_results.json"
    bench_path = _module3_results_dir(config) / "benchmark_results.json"
    if not latency_path.exists() or not bench_path.exists():
        return
    plot = _paths(config)["plots"] / "pareto_frontier.png"
    assert plot.exists(), f"Missing Pareto plot: {plot}"


def test_pareto_data_exists_when_latency_and_benchmarks_available(config) -> None:
    res_dir = _paths(config)["results"]
    latency_path = res_dir / "latency_results.json"
    bench_path = _module3_results_dir(config) / "benchmark_results.json"
    if not latency_path.exists() or not bench_path.exists():
        return
    data_path = res_dir / "pareto_data.json"
    assert data_path.exists(), f"Missing Pareto data: {data_path}"


def test_vllm_profiling_results(config) -> None:
    res_dir = _paths(config)["results"]
    fp = res_dir / "vllm_profiling_results.json"
    _require_file(fp, res_dir)
    data = _load_json(fp)
    assert "metrics" in data, "vLLM profiling missing metrics"


def test_model_card_generated(config) -> None:
    res_dir = _paths(config)["results"]
    fp = res_dir / "MODEL_CARD.md"
    _require_file(fp, res_dir)

    content = fp.read_text(encoding="utf-8")
    assert "Model Card" in content or "sarvam-30b" in content