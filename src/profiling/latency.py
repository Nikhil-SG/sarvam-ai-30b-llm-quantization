"""
Module 4a – Latency & Throughput Profiling.

Measure tokens-per-second (TPS) across batch sizes for each
quantisation variant.  Results include mean, std, and per-run timings.
"""

from __future__ import annotations

import time
import json
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger
from src.quantization.base import sanitize_generation_inputs

logger = get_logger(__name__)


class LatencyProfiler:
    """Benchmark token-generation speed across batch sizes."""

    def __init__(self, config):
        self.config = config
        self.prompt = config.profiling.prompt
        self.max_new_tokens = config.profiling.max_new_tokens
        self.batch_sizes: List[int] = config.profiling.batch_sizes
        self.warmup = config.profiling.warmup_steps
        self.num_runs = config.profiling.num_runs
        self.plots_dir = Path(config.output.plots_dir)
        self.results_dir = Path(config.output.results_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config.visualization.dpi

    # ── public API ──────────────────────────────────────────────────────
    def profile(
        self, model, tokenizer, tag: str
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run throughput benchmark for every configured batch size.

        Returns:
            Mapping batch_size → metrics dict.
        """
        logger.info(f"Latency profiling [{tag}] - batch sizes {self.batch_sizes}")
        results: Dict[int, Dict[str, Any]] = {}

        for bs in self.batch_sizes:
            logger.info(f"  batch_size={bs}")
            try:
                metrics = self._benchmark_batch(model, tokenizer, bs)
                results[bs] = metrics
                logger.info(
                    f"    TPS = {metrics['tokens_per_sec']:.1f}  "
                    f"({metrics['avg_time_sec']:.2f} s/batch)"
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"    OOM at batch_size={bs} — skipping")
                results[bs] = {"error": "OOM"}
                torch.cuda.empty_cache()
            except Exception as exc:
                logger.error(f"    Failed: {exc}")
                results[bs] = {"error": str(exc)}

        # If every batch size errored (e.g. OOM), mark the strategy-level result
        all_errored = results and all(
            isinstance(v, dict) and "error" in v and "tokens_per_sec" not in v
            for v in results.values()
        )
        if all_errored:
            results["error"] = "all_batch_sizes_failed"

        return results

    def profile_all(
        self,
        models: Dict[str, Any],
        tokenizer,
    ) -> Dict[str, Dict]:
        """
        Profile multiple (tag → model) pairs and produce a comparison plot.

        ``models`` is a dict like ``{"bf16": model_bf16, "gptq": model_gptq}``.
        """
        all_results: Dict[str, Dict] = {}

        for tag, model in models.items():
            all_results[tag] = self.profile(model, tokenizer, tag)

        # Persist
        path = self.results_dir / "latency_results.json"
        with open(path, "w") as fh:
            json.dump(all_results, fh, indent=2, default=str)
        logger.info(f"Latency results: {path}")

        # Plot
        self._plot_comparison(all_results)
        return all_results

    # ── per-batch benchmark ─────────────────────────────────────────────
    def _benchmark_batch(
        self, model, tokenizer, batch_size: int
    ) -> Dict[str, Any]:
        inputs = tokenizer(
            [self.prompt] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        # With device_map="auto" the model spans multiple GPUs; use the
        # embedding layer's device as the input device instead of model.device.
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        inputs = sanitize_generation_inputs(model, inputs)

        # Warm up
        for _ in range(self.warmup):
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=8, do_sample=False)

        torch.cuda.synchronize()

        times, tokens_generated = [], []
        for _ in range(self.num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            gen = (out.shape[1] - inputs["input_ids"].shape[1]) * batch_size
            times.append(elapsed)
            tokens_generated.append(gen)

        avg_t = float(np.mean(times))
        avg_tok = float(np.mean(tokens_generated))
        tps = avg_tok / avg_t if avg_t > 0 else 0

        return {
            "batch_size": batch_size,
            "avg_time_sec": round(avg_t, 4),
            "std_time_sec": round(float(np.std(times)), 4),
            "avg_tokens": int(avg_tok),
            "tokens_per_sec": round(tps, 2),
            "num_runs": self.num_runs,
        }

    # ── comparison chart ────────────────────────────────────────────────
    def _plot_comparison(self, all_results: Dict[str, Dict]) -> Path:
        tag_colors = {
            "bf16": "#2196F3", "gptq": "#4CAF50", "int8": "#E91E63",
            "fp8": "#00ACC1",
            "nf4": "#9C27B0",
        }

        fig, ax = plt.subplots(figsize=(12, 7))

        for tag, bs_data in all_results.items():
            # Skip strategies that are entirely errored
            if not isinstance(bs_data, dict) or "error" in bs_data:
                continue
            batch_sizes, tps_vals = [], []
            # Filter to only numeric keys (batch sizes); skip metadata keys
            numeric_items = {k: v for k, v in bs_data.items() if str(k).isdigit()}
            for bs, m in sorted(numeric_items.items(), key=lambda x: int(x[0])):
                if isinstance(m, dict) and "tokens_per_sec" in m:
                    batch_sizes.append(int(bs))
                    tps_vals.append(m["tokens_per_sec"])
            if batch_sizes:
                ax.plot(
                    batch_sizes, tps_vals,
                    marker="o", linewidth=2, markersize=8,
                    color=tag_colors.get(tag, "#888"),
                    label=tag.upper(),
                )

        ax.set_xlabel("Batch Size", fontsize=13)
        ax.set_ylabel("Tokens / Second", fontsize=13)
        ax.set_title(
            "Latency vs. Batch Size — Quantisation Comparison",
            fontsize=15, fontweight="bold",
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = self.plots_dir / "latency_comparison.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  saved: {out.name}")
        return out
