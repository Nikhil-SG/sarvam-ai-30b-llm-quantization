"""
Module 5c – Pareto Frontier Analysis.

Produce the final quality-vs-throughput scatter plot for the Sarvam-30B
quantization study, using the benchmark suite's configured primary metric.
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


class ParetoAnalyzer:
    """Combine evaluation + profiling results into a Pareto frontier plot."""

    TAG_MARKERS = {
        "bf16": "o", "gptq": "s", "int8": "D",
        "fp8_gptq": "^", "int8_gptq": "v",
    }
    TAG_COLORS = {
        "bf16": "#2196F3", "gptq": "#4CAF50", "int8": "#E91E63",
        "fp8_gptq": "#00ACC1", "int8_gptq": "#8E24AA",
    }

    def __init__(self, config):
        self.config = config
        self.plots_dir = Path(config.output.plots_dir)
        self.results_dir = Path(config.output.results_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config.visualization.dpi

    # ── build combined data ─────────────────────────────────────────────
    @staticmethod
    def _extract_point(
        tag: str,
        profiling: Dict,
        evaluation: Dict,
    ) -> Optional[Dict[str, Any]]:
        """Pull throughput and accuracy from raw results."""
        # Throughput: take batch_size=1 TPS (or first available)
        tps = None
        latency_data = profiling.get(tag, {})
        for bs_key in ("1", 1):
            entry = latency_data.get(bs_key, {})
            if isinstance(entry, dict) and "tokens_per_sec" in entry:
                tps = entry["tokens_per_sec"]
                break

        acc = None
        metric_name = "Composite Benchmark Score"
        bench = evaluation.get(tag, {})
        summary = bench.get("summary", {})
        suite = bench.get("suite", {})
        primary_metric = summary.get("primary_metric") or suite.get("primary_metric") or "composite"
        if primary_metric == "composite":
            acc = summary.get("composite_score")
            metric_name = "Composite Benchmark Score (%)"
        else:
            task_data = bench.get("tasks", {}).get(primary_metric, {}) or bench.get(primary_metric, {})
            a = task_data.get("accuracy")
            if a is not None:
                acc = float(a)
                metric_name = f"{task_data.get('label', primary_metric)} Score (%)"

        if acc is None:
            for task in ("mmlu", "gpqa_diamond", "humaneval", "mbpp"):
                task_data = bench.get("tasks", {}).get(task, {}) or bench.get(task, {})
                a = task_data.get("accuracy")
                if a is not None:
                    acc = float(a)
                    metric_name = f"{task_data.get('label', task)} Score (%)"
                    break

        if tps is None or acc is None:
            return None

        return {
            "tag": tag,
            "throughput": tps,
            "accuracy": acc,
            "metric_name": metric_name,
        }

    # ── compute Pareto front ────────────────────────────────────────────
    @staticmethod
    def _pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Given (throughput, accuracy) pairs, return the Pareto-optimal subset
        (maximise both).
        """
        # Sort by throughput descending
        pts = sorted(points, key=lambda p: -p[0])
        front = []
        max_acc = -1.0
        for t, a in pts:
            if a > max_acc:
                front.append((t, a))
                max_acc = a
        return sorted(front)

    # ── main plot ───────────────────────────────────────────────────────
    def plot(
        self,
        profiling_results: Dict[str, Dict],
        evaluation_results: Dict[str, Dict],
        tags: Optional[List[str]] = None,
    ) -> Path:
        """
        Generate the Pareto frontier scatter plot.

        Args:
            profiling_results: From ``LatencyProfiler.profile_all()``.
            evaluation_results: From ``BenchmarkRunner`` outputs.
            tags: Quantisation tags to include (default: all found).
        """
        if tags is None:
            tags = list(
                set(profiling_results.keys()) | set(evaluation_results.keys())
            )

        fig, ax = plt.subplots(figsize=(12, 8))
        all_points: List[Tuple[float, float]] = []
        y_label = "Quality Score (%)"

        for tag in tags:
            pt = self._extract_point(tag, profiling_results, evaluation_results)
            if pt is None:
                logger.warning(f"  Skipping {tag} — missing data")
                continue

            t, a = pt["throughput"], pt["accuracy"]
            all_points.append((t, a))
            y_label = pt.get("metric_name", y_label)

            ax.scatter(
                t, a,
                marker=self.TAG_MARKERS.get(tag, "o"),
                color=self.TAG_COLORS.get(tag, "#888"),
                s=220, zorder=5, edgecolors="white", linewidths=1.2,
                label=tag.upper(),
            )
            ax.annotate(
                f"  {tag.upper()}", (t, a),
                fontsize=10, fontweight="bold",
                color=self.TAG_COLORS.get(tag, "#888"),
            )

        # Pareto frontier line
        if len(all_points) >= 2:
            front = self._pareto_front(all_points)
            if len(front) >= 2:
                ax.plot(
                    [p[0] for p in front],
                    [p[1] for p in front],
                    "k--", alpha=0.5, linewidth=1.5,
                    label="Pareto Frontier",
                )

        ax.set_xlabel("Throughput (Tokens / sec) →", fontsize=14)
        ax.set_ylabel(f"{y_label} →", fontsize=14)
        ax.set_title(
            "Sarvam-30B Quantization Pareto Frontier",
            fontsize=16, fontweight="bold",
        )
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = self.plots_dir / "pareto_frontier.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        # Persist
        pareto_data = {
            tag: self._extract_point(tag, profiling_results, evaluation_results)
            for tag in tags
        }
        json_path = self.results_dir / "pareto_data.json"
        with open(json_path, "w") as fh:
            json.dump(pareto_data, fh, indent=2, default=str)

        logger.info(f"Pareto frontier: {out}")
        logger.info(f"Pareto data: {json_path}")
        return out
