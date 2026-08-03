"""
Module 6 – Layer-level Visualisation & Analysis.

For each selected transformer layer, generate:
  • Side-by-side weight-distribution histograms across all quant types.
  • Side-by-side weight-value heatmaps showing clipping/rounding effects.
  • Difference heatmaps (quantised − BF16) highlighting per-element error.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from src.core.logger import get_logger
from src.core.weight_io import WeightCache

logger = get_logger(__name__)

TAG_COLORS: Dict[str, str] = {
    "bf16": "#2196F3",
    "int8": "#E91E63",
    "fp8": "#009688",
    "nf4": "#9C27B0",
    "gptq": "#4CAF50",
}

ALL_TAGS = ["bf16", "int8", "fp8", "nf4", "gptq"]


class LayerVisualizer:
    """Publication-ready side-by-side layer comparisons."""

    def __init__(self, config):
        self.config = config
        self.cache = WeightCache(
            cache_dir=config.output.weights_dir,
            sample_size=config.visualization.weight_sample_size,
        )
        self.plots_dir = Path(config.output.plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config.visualization.dpi

    # ── public API ──────────────────────────────────────────────────────
    def run(
        self,
        target_layers: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Generate all layer visualisation images.

        Returns:
            List of saved image paths.
        """
        if target_layers is None:
            target_layers = self.config.visualization.target_layers
        if tags is None:
            tags = ALL_TAGS

        saved: List[Path] = []

        for layer_name in target_layers:
            available = self._available_tags(layer_name, tags)
            if not available:
                logger.warning(f"No data for {layer_name} – skipping")
                continue

            try:
                # 1) Side-by-side histograms
                p = self._plot_histograms(layer_name, available)
                saved.append(p)

                # 2) Weight-value heatmaps
                p = self._plot_heatmaps(layer_name, available)
                saved.append(p)

                # 3) Difference heatmap vs BF16
                if "bf16" in available and len(available) > 1:
                    p = self._plot_diff_heatmaps(layer_name, available)
                    saved.append(p)

                logger.debug(f"  done: {layer_name}")
            except Exception as exc:
                logger.warning(f"  skip {layer_name}: {exc}")

        logger.info(f"Layer visualisations: {len(saved)} images saved")
        return saved

    # ────────────────────────────────────────────────────────────────────
    #  Side-by-side Histograms
    # ────────────────────────────────────────────────────────────────────
    def _plot_histograms(
        self, layer_name: str, tags: List[str]
    ) -> Path:
        n = len(tags)
        fig, axes = plt.subplots(
            1, n, figsize=(5.5 * n, 5), sharey=True, squeeze=False
        )
        axes = axes[0]

        for i, tag in enumerate(tags):
            data = self.cache.load_layer_weights(layer_name, tag)
            vals = data["values"]
            ax = axes[i]
            color = TAG_COLORS.get(tag, "#888")

            ax.hist(vals, bins=200, density=True, color=color,
                    alpha=0.85, edgecolor="none")

            mu, sigma = float(np.mean(vals)), float(np.std(vals))
            ax.set_title(tag.upper(), fontsize=13, fontweight="bold")
            ax.set_xlabel("Weight value")
            if i == 0:
                ax.set_ylabel("Density")

            # Statistics inset
            textstr = (
                f"μ = {mu:.5f}\n"
                f"σ = {sigma:.5f}\n"
                f"min = {np.min(vals):.5f}\n"
                f"max = {np.max(vals):.5f}"
            )
            ax.text(
                0.97, 0.97, textstr,
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                          alpha=0.5),
            )

        short = layer_name.replace("model.layers.", "L")
        fig.suptitle(
            f"Weight Distribution — {short}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        safe = layer_name.replace(".", "_")
        out = self.plots_dir / f"layer_hist_{safe}.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    # ────────────────────────────────────────────────────────────────────
    #  Weight-Value Heatmaps
    # ────────────────────────────────────────────────────────────────────
    def _plot_heatmaps(
        self, layer_name: str, tags: List[str]
    ) -> Path:
        n = len(tags)
        fig, axes = plt.subplots(
            1, n, figsize=(5 * n, 4.5), squeeze=False
        )
        axes = axes[0]

        for i, tag in enumerate(tags):
            data = self.cache.load_layer_weights(layer_name, tag)
            vals = data["values"]

            # Reshape into a small grid for visualisation
            grid_size = min(64, int(np.sqrt(len(vals))))
            grid = vals[: grid_size * grid_size].reshape(grid_size, grid_size)

            v_lo = np.percentile(vals, 1)
            v_hi = np.percentile(vals, 99)

            im = axes[i].imshow(
                grid, cmap="RdBu_r", aspect="auto",
                vmin=v_lo, vmax=v_hi,
            )
            axes[i].set_title(tag.upper(), fontsize=13, fontweight="bold")
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        short = layer_name.replace("model.layers.", "L")
        fig.suptitle(
            f"Weight Heatmap — {short}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        safe = layer_name.replace(".", "_")
        out = self.plots_dir / f"layer_heatmap_{safe}.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    # ────────────────────────────────────────────────────────────────────
    #  Difference Heatmaps  (quantised − BF16)
    # ────────────────────────────────────────────────────────────────────
    def _plot_diff_heatmaps(
        self, layer_name: str, tags: List[str]
    ) -> Path:
        ref_data = self.cache.load_layer_weights(layer_name, "bf16")
        ref_vals = ref_data["values"]

        quant_tags = [t for t in tags if t != "bf16"]
        n = len(quant_tags)
        if n == 0:
            return self.plots_dir / "placeholder.png"

        fig, axes = plt.subplots(
            1, n, figsize=(5 * n, 4.5), squeeze=False
        )
        axes = axes[0]

        for i, tag in enumerate(quant_tags):
            data = self.cache.load_layer_weights(layer_name, tag)
            vals = data["values"]

            min_len = min(len(ref_vals), len(vals))
            diff = vals[:min_len] - ref_vals[:min_len]

            grid_size = min(64, int(np.sqrt(min_len)))
            grid = diff[: grid_size * grid_size].reshape(grid_size, grid_size)

            abs_max = max(abs(np.percentile(diff, 1)),
                         abs(np.percentile(diff, 99)))

            im = axes[i].imshow(
                grid, cmap="bwr", aspect="auto",
                vmin=-abs_max, vmax=abs_max,
            )
            axes[i].set_title(
                f"{tag.upper()} − BF16", fontsize=13, fontweight="bold"
            )
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            # MSE annotation
            mse = float(np.mean(diff ** 2))
            axes[i].text(
                0.5, -0.08, f"MSE = {mse:.2e}",
                transform=axes[i].transAxes, fontsize=9,
                ha="center", fontweight="bold",
            )

        short = layer_name.replace("model.layers.", "L")
        fig.suptitle(
            f"Quantisation Error — {short}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        safe = layer_name.replace(".", "_")
        out = self.plots_dir / f"layer_diff_{safe}.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    # ── helpers ─────────────────────────────────────────────────────────
    def _available_tags(
        self, layer_name: str, tags: List[str]
    ) -> List[str]:
        found: List[str] = []
        for t in tags:
            try:
                self.cache.load_layer_weights(layer_name, t)
                found.append(t)
            except FileNotFoundError:
                pass
        return found
