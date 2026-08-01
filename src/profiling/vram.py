"""
Module 4b – VRAM Utilisation Profiling.

Track both the current post-benchmark allocation snapshot and the peak GPU
memory reached since counters were reset before model load.
"""

from __future__ import annotations

import json
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any, Dict

from src.core.logger import get_logger
from src.core.memory import get_memory_snapshot, get_peak_memory

logger = get_logger(__name__)


class VRAMProfiler:
    """Capture and visualise per-GPU VRAM utilisation."""

    def __init__(self, config):
        self.config = config
        self.plots_dir = Path(config.output.plots_dir)
        self.results_dir = Path(config.output.results_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config.visualization.dpi

    # ── per-model snapshot ──────────────────────────────────────────────
    def snapshot(self, tag: str) -> Dict[str, Any]:
        """
        Capture current VRAM utilisation and peak stats.

        Call this while the model is loaded.  Peak memory reflects usage
        since the model was loaded (do NOT reset before reading).
        """
        snap = get_memory_snapshot()
        peak = get_peak_memory()

        current_total_allocated = round(sum(snap.gpu_allocated_gb.values()), 3)
        current_total_reserved = round(sum(snap.gpu_reserved_gb.values()), 3)
        peak_total = round(sum(peak.values()), 3) if peak else 0.0
        peak_max = round(max(peak.values()), 3) if peak else 0.0

        result = {
            "tag": tag,
            "snapshot": snap.to_dict(),
            "peak_gb": peak,
            "current_total_allocated_gb": current_total_allocated,
            "current_total_reserved_gb": current_total_reserved,
            "peak_total_gb": peak_total,
            "peak_max_gb": peak_max,
            "measurement_scope": (
                "current snapshot captured after latency benchmark; "
                "peak_* reflects memory since reset_peak_memory() before model load"
            ),
        }
        logger.info(
            f"[{tag}] VRAM snapshot — {snap.summary()}"
        )
        return result

    # ── comparison across all tags ──────────────────────────────────────
    def plot_comparison(
        self, vram_data: Dict[str, Dict[str, Any]]
    ) -> Path:
        """
        Generate a grouped bar chart of peak VRAM usage per GPU per tag.

        Args:
            vram_data: Mapping tag → snapshot dict (from ``snapshot()``).
        """
        tag_colors = {
            "bf16": "#2196F3", "gptq": "#4CAF50", "int8": "#E91E63",
            "fp8": "#00ACC1",
            "nf4": "#9C27B0",
        }

        tags = list(vram_data.keys())
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

        # Build data arrays
        allocated = {t: [] for t in tags}
        for t in tags:
            peak = vram_data[t].get("peak_gb", {})
            snap = vram_data[t].get("snapshot", {})
            alloc = peak or snap.get("gpu_allocated_gb", {})
            for g in range(num_gpus):
                allocated[t].append(alloc.get(str(g), alloc.get(g, 0)))

        fig, ax = plt.subplots(figsize=(max(8, len(tags) * 2.5), 7))

        x = np.arange(num_gpus)
        width = 0.8 / len(tags)

        for i, t in enumerate(tags):
            offset = (i - len(tags) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, allocated[t], width,
                label=t.upper(),
                color=tag_colors.get(t, "#888"),
                edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, allocated[t]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8,
                )

        # Total memory line
        if torch.cuda.is_available():
            for g in range(num_gpus):
                total = torch.cuda.get_device_properties(g).total_memory / (1024 ** 3)
                ax.axhline(
                    total, color="gray", ls="--", lw=0.8,
                    label=f"GPU {g} total" if g == 0 else None,
                )

        ax.set_xlabel("GPU Index", fontsize=13)
        ax.set_ylabel("Peak Allocated VRAM (GB)", fontsize=13)
        ax.set_title(
            "Peak VRAM Utilisation by Quantisation Type",
            fontsize=15, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"GPU {g}" for g in range(num_gpus)])
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = self.plots_dir / "vram_comparison.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        # Persist
        json_path = self.results_dir / "vram_results.json"
        with open(json_path, "w") as fh:
            json.dump(vram_data, fh, indent=2, default=str)
        logger.info(f"VRAM results: {json_path}")
        logger.debug(f"  saved: {out.name}")

        return out
