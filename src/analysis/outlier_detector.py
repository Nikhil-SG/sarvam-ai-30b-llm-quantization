"""
Module 3c – Activation Outlier Detection.

Run a short forward pass through the model and capture intermediate
activations for target layers.  Identify high-magnitude outliers that
typically break low-bit quantisation (the "kurtosis problem").
"""

from __future__ import annotations

import json
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger
from src.core.weight_io import WeightExtractor

logger = get_logger(__name__)


class OutlierDetector:
    """Capture activations via forward hooks and analyse outliers."""

    def __init__(self, config):
        self.config = config
        self.plots_dir = Path(config.output.plots_dir)
        self.results_dir = Path(config.output.results_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config.visualization.dpi

    # ── public API ──────────────────────────────────────────────────────
    def run(
        self,
        model,
        tokenizer,
        target_layers: Optional[List[str]] = None,
        sigma_threshold: float = 6.0,
        tag: str = "bf16",
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Capture activations, detect outliers, generate plots.

        Args:
            model: A loaded LLM.
            tokenizer: Matching tokenizer.
            target_layers: Dotted layer names to hook.
            sigma_threshold: Values beyond ±(threshold × σ) are outliers.
            tag: Label used in filenames (e.g. ``bf16``).
            prompts: Prompt suite used to probe activations.

        Returns:
            Dict with 'statistics', 'method', and metadata.
        """
        if target_layers is None:
            target_layers = self.config.visualization.target_layers

        prompts = self._resolve_prompts(prompts)
        activations: Dict[str, List[torch.Tensor]] = {}
        prompt_runs: List[Dict[str, Any]] = []

        logger.info(
            f"Running outlier detection [{tag}] across {len(prompts)} prompts"
        )
        for idx, prompt in enumerate(prompts, start=1):
            prompt_activations: Dict[str, torch.Tensor] = {}
            hooks = []

            for name in target_layers:
                try:
                    module = WeightExtractor.get_module_by_name(model, name)
                    hooks.append(
                        module.register_forward_hook(
                            self._make_hook(name, prompt_activations)
                        )
                    )
                except Exception as exc:
                    logger.warning(f"Cannot hook {name}: {exc}")

            inputs = tokenizer(prompt, return_tensors="pt")
            first_device = next(model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs)

            for h in hooks:
                h.remove()

            for name, act in prompt_activations.items():
                activations.setdefault(name, []).append(act)

            prompt_runs.append(
                {
                    "prompt_index": idx,
                    "prompt_preview": prompt[:120],
                    "layers_captured": len(prompt_activations),
                }
            )

        # ── analyse each captured activation ────────────────────────────
        stats: Dict[str, Any] = {}
        for name, acts in activations.items():
            try:
                layer_stats = self._analyse_activation(
                    name, acts, sigma_threshold, tag, prompts
                )
                stats[name] = layer_stats
                logger.info(
                    f"  {name}: outlier% = {layer_stats['outlier_pct']:.4f}%  "
                    f"max|act| = {layer_stats['abs_max']:.4f}"
                )
            except Exception as exc:
                logger.warning(f"  Analysis failed for {name}: {exc}")

        # ── summary plot ────────────────────────────────────────────────
        self._plot_summary(stats, tag)

        # ── persist ─────────────────────────────────────────────────────
        json_path = self.results_dir / f"outlier_stats_{tag}.json"
        with open(json_path, "w") as fh:
            json.dump(stats, fh, indent=2, default=str)
        logger.info(f"Outlier stats: {json_path}")

        return {
            "statistics": stats,
            "method": (
                f"forward hook activation capture across {len(prompts)} prompts "
                f"with {sigma_threshold}-sigma threshold"
            ),
            "tag": tag,
            "num_layers_analyzed": len(stats),
            "sigma_threshold": sigma_threshold,
            "target_layers": target_layers,
            "num_prompts": len(prompts),
            "prompts": prompt_runs,
        }

    def _resolve_prompts(self, prompts: Optional[List[str]]) -> List[str]:
        if prompts:
            resolved = [prompt.strip() for prompt in prompts if prompt and prompt.strip()]
            if resolved:
                return resolved

        cfg = getattr(self.config.visualization, "outlier_detection", None)
        configured = getattr(cfg, "prompts", None) if cfg is not None else None
        if configured:
            resolved = [prompt.strip() for prompt in configured if prompt and prompt.strip()]
            if resolved:
                return resolved

        return [self.config.profiling.prompt]

    # ── hook factory ────────────────────────────────────────────────────
    @staticmethod
    def _make_hook(name: str, store: Dict[str, torch.Tensor]):
        def fn(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            store[name] = out.detach().cpu()
        return fn

    # ── per-layer analysis + plotting ───────────────────────────────────
    def _analyse_activation(
        self,
        layer_name: str,
        activations: List[torch.Tensor],
        sigma: float,
        tag: str,
        prompts: List[str],
    ) -> Dict[str, Any]:
        prompt_stats: List[Dict[str, Any]] = []
        flat_acts: List[np.ndarray] = []
        for idx, activation in enumerate(activations, start=1):
            vals = activation.float().numpy().flatten()
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size:
                flat_acts.append(finite_vals)
            stats = self._compute_stats(vals, sigma)
            stats["prompt_index"] = idx
            stats["prompt_preview"] = prompts[idx - 1][:120]
            prompt_stats.append(stats)

        vals = np.concatenate(flat_acts) if flat_acts else np.array([], dtype=np.float32)
        combined = self._compute_stats(vals, sigma)
        mu = combined["mean"]
        sd = combined["std"]
        abs_max = combined["abs_max"]
        n_outliers = combined["num_outliers"]
        pct = combined["outlier_pct"]
        outlier_mask = np.abs(vals - mu) > sigma * sd if len(vals) else np.array([], dtype=bool)
        invalid_total = int(sum(p["num_invalid"] for p in prompt_stats))
        invalid_pct = (invalid_total / (invalid_total + len(vals)) * 100) if (invalid_total + len(vals)) else 0.0

        # ── plot ────────────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Full distribution
        if len(vals):
            ax1.hist(vals, bins=300, density=True, color="#2196F3",
                     edgecolor="none", alpha=0.85)
            ax1.axvline(mu + sigma * sd, color="red", ls="--", lw=1,
                        label=f"+{sigma}σ")
            ax1.axvline(mu - sigma * sd, color="red", ls="--", lw=1,
                        label=f"−{sigma}σ")
            ax1.legend(fontsize=9)
        else:
            ax1.text(
                0.5,
                0.5,
                "No finite activations captured",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
        ax1.set_title("Activation Distribution", fontsize=12)
        ax1.set_xlabel("Activation value")
        ax1.set_ylabel("Density")

        # Outliers only
        if n_outliers > 0:
            ax2.hist(vals[outlier_mask], bins=min(80, n_outliers),
                     color="#F44336", alpha=0.85, edgecolor="none")
        elif invalid_total > 0:
            ax2.text(
                0.5,
                0.5,
                f"Invalid activations detected\n{invalid_total:,} values ({invalid_pct:.2f}%)",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
        ax2.set_title(
            f"Outliers (>{sigma}σ): {pct:.4f}%  ({n_outliers:,} values)",
            fontsize=12,
        )
        ax2.set_xlabel("Activation value")
        ax2.set_ylabel("Count")

        short = layer_name.replace("model.layers.", "L")
        fig.suptitle(
            f"Outlier Detection – {short} [{tag.upper()} | {len(prompt_stats)} prompts]",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        safe = layer_name.replace(".", "_")
        out = self.plots_dir / f"outlier_{tag}_{safe}.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return {
            "layer": layer_name,
            "mean": mu,
            "std": sd,
            "abs_max": abs_max,
            "sigma_threshold": sigma,
            "num_outliers": n_outliers,
            "total_values": len(vals),
            "outlier_pct": pct,
            "num_invalid": invalid_total,
            "invalid_pct": invalid_pct,
            "is_finite": invalid_total == 0,
            "num_prompts": len(prompt_stats),
            "prompt_outlier_pct_mean": float(np.mean([p["outlier_pct"] for p in prompt_stats])),
            "prompt_outlier_pct_std": float(np.std([p["outlier_pct"] for p in prompt_stats])),
            "per_prompt": prompt_stats,
            "plot": str(out),
        }

    @staticmethod
    def _compute_stats(vals: np.ndarray, sigma: float) -> Dict[str, Any]:
        total_values = int(len(vals))
        finite_vals = vals[np.isfinite(vals)] if total_values else np.array([], dtype=np.float32)
        num_invalid = total_values - int(len(finite_vals))

        if len(finite_vals) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "abs_max": 0.0,
                "num_outliers": 0,
                "total_values": 0,
                "outlier_pct": 0.0,
                "num_invalid": num_invalid,
                "invalid_pct": 100.0 if total_values else 0.0,
                "is_finite": False,
            }

        mu = float(np.mean(finite_vals))
        sd = float(np.std(finite_vals))
        abs_max = float(np.max(np.abs(finite_vals)))
        outlier_mask = np.abs(finite_vals - mu) > sigma * sd
        n_outliers = int(np.sum(outlier_mask))
        pct = n_outliers / len(finite_vals) * 100
        return {
            "mean": mu,
            "std": sd,
            "abs_max": abs_max,
            "num_outliers": n_outliers,
            "total_values": int(len(finite_vals)),
            "outlier_pct": pct,
            "num_invalid": num_invalid,
            "invalid_pct": num_invalid / total_values * 100 if total_values else 0.0,
            "is_finite": num_invalid == 0,
        }

    # ── summary bar chart ───────────────────────────────────────────────
    def _plot_summary(self, stats: Dict[str, Any], tag: str) -> Path:
        if not stats:
            return self.plots_dir / f"outlier_summary_{tag}.png"

        names = list(stats.keys())
        pcts = [stats[n]["outlier_pct"] for n in names]
        maxes = [stats[n]["abs_max"] for n in names]
        short = [n.replace("model.layers.", "L").replace(".", "\n")
                 for n in names]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(names) * 1.2), 10))

        ax1.barh(short, pcts, color="#F44336", edgecolor="none")
        ax1.set_xlabel("Outlier %")
        ax1.set_title("Outlier Percentage by Layer", fontsize=13,
                      fontweight="bold")
        ax1.invert_yaxis()

        ax2.barh(short, maxes, color="#FF9800", edgecolor="none")
        ax2.set_xlabel("Max |activation|")
        ax2.set_title("Max Absolute Activation by Layer", fontsize=13,
                      fontweight="bold")
        ax2.invert_yaxis()

        fig.suptitle(f"Outlier Summary [{tag.upper()}]",
                     fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        out = self.plots_dir / f"outlier_summary_{tag}.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  saved: {out.name}")
        return out
