"""
Module 3a – Weight Distribution Histograms.

For each target layer, plot the distribution of weights across all
quantisation formats side-by-side so the user can visually see the
bins / clusters introduced by quantisation.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from src.core.logger import get_logger
from src.core.weight_io import WeightCache

logger = get_logger(__name__)

# Consistent colours across all plots
TAG_COLORS: Dict[str, str] = {
    "bf16": "#2196F3",
    "int8": "#E91E63",
    "fp8": "#00ACC1",
    "nf4": "#9C27B0",
    "gptq": "#4CAF50",
}


class WeightDistributionAnalyzer:
    """Generate per-layer weight-distribution histograms."""

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
    ) -> dict:
        """
        Generate one histogram image per target layer.

        Returns:
            Dict with 'paths', 'count', 'method', and 'tags_used' keys.
        """
        if target_layers is None:
            target_layers = self.config.visualization.target_layers
        if tags is None:
            tags = list(TAG_COLORS.keys())

        saved: List[Path] = []
        for layer_name in target_layers:
            try:
                path = self._plot_layer(layer_name, tags)
                saved.append(path)
                logger.debug(f"  saved: {path.name}")
            except Exception as exc:
                logger.warning(f"  skip {layer_name}: {exc}")

        logger.info(f"Weight-distribution plots: {len(saved)} saved")
        return {
            "paths": [str(p) for p in saved],
            "count": len(saved),
            "method": "matplotlib histogram with gaussian overlay",
            "tags_used": tags,
            "target_layers": target_layers,
        }

    # ── per-layer plotting ──────────────────────────────────────────────
    def _plot_layer(self, layer_name: str, tags: List[str]) -> Path:
        available = self._available_tags(layer_name, tags)
        n = len(available)
        if n == 0:
            raise FileNotFoundError(f"No cached data for {layer_name}")

        fig, axes = plt.subplots(
            1, n, figsize=(5.5 * n, 5), sharey=True, squeeze=False
        )
        axes = axes[0]

        for i, tag in enumerate(available):
            data = self.cache.load_layer_weights(layer_name, tag)
            vals = data["values"]

            ax = axes[i]
            color = TAG_COLORS.get(tag, "#888888")
            ax.hist(vals, bins=200, alpha=0.85, density=True, color=color,
                    edgecolor="none")

            # Statistics overlay
            mu, sigma = float(np.mean(vals)), float(np.std(vals))
            ax.axvline(mu, color="red", ls="--", lw=1.0,
                       label=f"μ = {mu:.5f}")
            ax.axvline(mu + sigma, color="green", ls=":", lw=0.8,
                       label=f"σ = {sigma:.5f}")
            ax.axvline(mu - sigma, color="green", ls=":", lw=0.8)

            ax.set_title(tag.upper(), fontsize=13, fontweight="bold")
            ax.set_xlabel("Weight value")
            if i == 0:
                ax.set_ylabel("Density")
            ax.legend(fontsize=8, loc="upper right")
            ax.tick_params(labelsize=8)

        short = layer_name.replace("model.layers.", "L")
        fig.suptitle(
            f"Weight Distribution – {short}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        safe = layer_name.replace(".", "_")
        out = self.plots_dir / f"weight_dist_{safe}.png"
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
