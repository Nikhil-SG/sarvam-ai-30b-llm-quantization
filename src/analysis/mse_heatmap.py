"""
Module 3b – MSE Heatmap.

Compute the Mean Squared Error between BF16 reference weights and each
quantised variant for every layer × projection.  Render as a heatmap
where the Y-axis is model depth (0 → 79).
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger
from src.core.weight_io import WeightCache

logger = get_logger(__name__)


def _build_log_norm(matrix: np.ndarray) -> mcolors.LogNorm:
    """Return a stable log normalization for one heatmap matrix."""
    positive = matrix[~np.isnan(matrix) & (matrix > 0)]
    if positive.size == 0:
        return mcolors.LogNorm(vmin=1e-12, vmax=1e-8)

    vmin = max(float(np.nanmin(positive)), 1e-12)
    vmax = max(float(np.nanmax(positive)), vmin * 1.01)
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def _projection_label(name: str) -> str:
    """Return a compact axis label for a projection path."""
    if name == "attention.query_key_value":
        return "query_key_value"
    if name == "attention.dense":
        return "dense"
    if name == "mlp.shared_experts.down_proj":
        return "shared_down_proj"
    return name.split(".")[-1]


class MSEHeatmapAnalyzer:
    """Compute and visualise per-layer MSE across quantisation formats."""

    def __init__(self, config):
        self.config = config
        self.cache = WeightCache(
            cache_dir=config.output.weights_dir,
            sample_size=config.visualization.weight_sample_size,
        )
        self.plots_dir = Path(config.output.plots_dir)
        self.results_dir = Path(config.output.results_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config.visualization.dpi

    # ── public API ──────────────────────────────────────────────────────
    def run(
        self,
        ref_tag: str = "bf16",
        quant_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute MSE matrices and generate heatmap images.

        Returns:
            Dict with 'matrices', 'raw_data', 'method', and metadata.
        """
        if quant_tags is None:
            quant_tags = ["int8", "fp8", "nf4", "gptq"]

        projs = list(self.config.visualization.mse_heatmap.projections)
        include_moe = bool(
            getattr(
                self.config.visualization.mse_heatmap,
                "include_moe_projections",
                False,
            )
        )
        if include_moe:
            moe_projs = list(
                getattr(
                    self.config.visualization.mse_heatmap,
                    "moe_projections",
                    [],
                )
            )
            for proj in moe_projs:
                if proj not in projs:
                    projs.append(proj)
        n_layers = self.config.visualization.mse_heatmap.num_layers

        all_matrices: Dict[str, np.ndarray] = {}
        all_raw: Dict[str, Dict] = {}

        for tag in quant_tags:
            logger.info(f"Computing MSE: {ref_tag} vs {tag}")
            matrix = np.full((n_layers, len(projs)), np.nan)

            for li in range(n_layers):
                for pi, proj in enumerate(projs):
                    name = f"model.layers.{li}.{proj}"
                    try:
                        mse = self.cache.compute_mse(name, ref_tag, tag)
                        matrix[li, pi] = mse
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        logger.debug(f"  MSE error {name}: {exc}")

                if (li + 1) % 20 == 0:
                    logger.info(f"  {tag}: {li + 1}/{n_layers} layers done")

            all_matrices[tag] = matrix
            all_raw[tag] = {
                f"layer_{li}_{proj}": float(matrix[li, pi])
                for li in range(n_layers)
                for pi, proj in enumerate(projs)
                if not np.isnan(matrix[li, pi])
            }

        # Persist raw MSE data
        raw_path = self.results_dir / "mse_all_layers.json"
        with open(raw_path, "w") as fh:
            json.dump(all_raw, fh, indent=2)
        logger.info(f"Raw MSE data: {raw_path}")

        # ── individual heatmaps ─────────────────────────────────────────
        for tag, matrix in all_matrices.items():
            self._plot_single(matrix, tag, projs, n_layers)

        # ── combined comparison heatmap ─────────────────────────────────
        self._plot_combined(all_matrices, projs, n_layers)

        return {
            "matrices": all_matrices,
            "raw_data": all_raw,
            "method": "mean squared error with log-scale colormap",
            "reference_tag": ref_tag,
            "quantizer_tags": quant_tags,
            "num_layers": n_layers,
            "projections": projs,
        }

    # ── plotting helpers ────────────────────────────────────────────────
    def _plot_single(
        self,
        matrix: np.ndarray,
        tag: str,
        projs: List[str],
        n_layers: int,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(6, max(12, n_layers * 0.18)))

        # Use log-scale colours so small and large MSE are both visible
        norm = _build_log_norm(matrix)
        cmap = plt.colormaps["hot"].copy()
        cmap.set_bad(color="#d9d9d9")
        plot_matrix = np.ma.masked_invalid(matrix)

        im = ax.imshow(
            plot_matrix, aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest",
        )
        ax.set_xlabel("Projection", fontsize=12)
        ax.set_ylabel("Layer Index", fontsize=12)
        ax.set_xticks(range(len(projs)))
        ax.set_xticklabels([_projection_label(p) for p in projs], fontsize=9)

        # Y-ticks every 10 layers
        yticks = list(range(0, n_layers, 10))
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=8)

        fig.colorbar(im, ax=ax, label="MSE (log scale)", shrink=0.6)
        ax.set_title(
            f"MSE Heatmap: BF16 vs {tag.upper()}", fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

        out = self.plots_dir / f"mse_heatmap_{tag}.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  saved: {out.name}")
        return out

    def _plot_combined(
        self,
        matrices: Dict[str, np.ndarray],
        projs: List[str],
        n_layers: int,
    ) -> Path:
        tags = list(matrices.keys())
        n = len(tags)
        fig, axes = plt.subplots(
            1, n, figsize=(5.2 * n, max(12, n_layers * 0.18)),
            sharey=True, squeeze=False,
            constrained_layout=True,
        )
        axes = axes[0]

        valid_counts = [
            np.count_nonzero(~np.isnan(matrix) & (matrix > 0))
            for matrix in matrices.values()
        ]
        if not any(valid_counts):
            logger.warning("No valid MSE values for combined heatmap")
            plt.close(fig)
            return self.plots_dir / "mse_heatmap_combined.png"

        cmap = plt.colormaps["hot"].copy()
        cmap.set_bad(color="#d9d9d9")

        for i, tag in enumerate(tags):
            norm = _build_log_norm(matrices[tag])
            plot_matrix = np.ma.masked_invalid(matrices[tag])
            im = axes[i].imshow(
                plot_matrix, aspect="auto", cmap=cmap, norm=norm,
                interpolation="nearest",
            )
            positive = matrices[tag][~np.isnan(matrices[tag]) & (matrices[tag] > 0)]
            if positive.size:
                range_label = (
                    f"{positive.min():.1e} to {positive.max():.1e}"
                )
            else:
                range_label = "no data"

            axes[i].set_title(
                f"{tag.upper()}\n{range_label}",
                fontsize=12,
                fontweight="bold",
            )
            axes[i].set_xlabel("Projection")
            axes[i].set_xticks(range(len(projs)))
            axes[i].set_xticklabels(
                [_projection_label(p) for p in projs], fontsize=8,
                rotation=25, ha="right"
            )
            if i == 0:
                axes[i].set_ylabel("Layer Index")
            axes[i].set_yticks(list(range(0, n_layers, 10)))

            fig.colorbar(
                im,
                ax=axes[i],
                label="MSE (log scale)",
                shrink=0.6,
                pad=0.02,
            )

        fig.suptitle(
            "MSE Heatmap – BF16 vs All Quantisation Formats\nIndependent log scale per quantizer",
            fontsize=15, fontweight="bold",
        )

        out = self.plots_dir / "mse_heatmap_combined.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  saved: {out.name}")
        return out
