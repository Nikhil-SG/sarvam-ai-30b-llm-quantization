"""
Module 4c – Model-on-Disk Size Profiling.

Measure serialized checkpoint bytes on disk for each quantisation format.
For HuggingFace-cached models, sum the shard sizes from the resolved snapshot.
"""

from __future__ import annotations

import json
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class DiskProfiler:
    """Measure and compare model sizes on disk."""

    def __init__(self, config):
        self.config = config
        self.plots_dir = Path(config.output.plots_dir)
        self.results_dir = Path(config.output.results_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config.visualization.dpi

    # ── measure a single model directory ────────────────────────────────
    @staticmethod
    def measure_dir(model_dir: str) -> Dict[str, Any]:
        """
        Sum the sizes of .safetensors and .bin shard files in a directory.

        Returns:
            Dict with ``total_bytes``, ``total_gb``, ``num_shards``.
        """
        path = Path(model_dir)
        if not path.exists():
            return {"error": f"Path not found: {model_dir}"}

        extensions = (".safetensors", ".bin")
        shards: List[Path] = [
            f for f in path.rglob("*") if f.suffix in extensions
        ]

        total_bytes = sum(f.stat().st_size for f in shards)
        return {
            "path": str(path),
            "num_shards": len(shards),
            "total_bytes": total_bytes,
            "total_gb": round(total_bytes / (1024 ** 3), 3),
        }

    # ── measure from HF cache (snapshot_download layout) ────────────────
    @staticmethod
    def measure_hf_model(model_id: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Try to locate model shards in the HuggingFace cache and sum sizes.
        """
        candidate_dirs = []
        if cache_dir:
            cache_path = Path(cache_dir)
            candidate_dirs.append(cache_path)
            if cache_path.name != "hub":
                candidate_dirs.append(cache_path / "hub")

        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            hf_home_path = Path(hf_home)
            candidate_dirs.append(hf_home_path)
            candidate_dirs.append(hf_home_path / "hub")

        seen = set()
        for candidate in candidate_dirs:
            candidate = candidate.resolve()
            if candidate in seen or not candidate.exists():
                continue
            seen.add(candidate)

            try:
                from huggingface_hub import scan_cache_dir, HFCacheInfo

                cache_info: HFCacheInfo = scan_cache_dir(str(candidate))
                for repo in cache_info.repos:
                    if repo.repo_id == model_id:
                        revisions = [rev for rev in repo.revisions if rev.size_on_disk]
                        latest = max(
                            revisions,
                            key=lambda rev: getattr(rev, "last_modified", 0),
                            default=None,
                        )
                        if latest is None:
                            continue

                        snapshot_path = getattr(latest, "snapshot_path", None)
                        if snapshot_path and Path(snapshot_path).exists():
                            measured = DiskProfiler.measure_dir(str(snapshot_path))
                            if "error" not in measured:
                                measured.update(
                                    {
                                        "model_id": model_id,
                                        "source": "hf_cache_snapshot",
                                        "snapshot_path": str(snapshot_path),
                                    }
                                )
                                return measured

                        total = latest.size_on_disk
                        return {
                            "model_id": model_id,
                            "total_bytes": total,
                            "total_gb": round(total / (1024 ** 3), 3),
                            "source": "hf_cache_revision",
                        }
            except Exception as exc:
                logger.debug(f"HF cache scan failed at {candidate}: {exc}")

        return {"model_id": model_id, "error": "not found in cache"}

    def measure_model_storage(
        self,
        model_ref: Optional[str] = None,
        *,
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve and measure serialized model storage.

        Priority:
          1. Existing local directory / snapshot path.
          2. HuggingFace cache lookup by model id.
        """
        result: Dict[str, Any]

        if model_ref:
            path = Path(model_ref)
            if path.exists():
                result = self.measure_dir(str(path))
                if "error" not in result:
                    result["source"] = "filesystem"
            else:
                result = {
                    "error": f"Path not found: {model_ref}",
                    "path": str(path),
                }
        elif model_id:
            result = self.measure_hf_model(model_id, cache_dir)
        else:
            result = {"error": "No model reference provided"}

        if "error" in result and model_id and result.get("model_id") != model_id:
            hf_result = self.measure_hf_model(model_id, cache_dir)
            if "error" not in hf_result:
                result = hf_result

        if "error" not in result:
            result.setdefault("measurement_type", "serialized_checkpoint_bytes")
            if "total_gb" in result:
                result.setdefault("model_size_gb", result["total_gb"])

        return result

    # ── comparison chart ────────────────────────────────────────────────
    def plot_comparison(self, disk_data: Dict[str, Dict[str, Any]]) -> Path:
        """
        Bar chart of total model size (GB) per quantisation format.

        Args:
            disk_data: Mapping tag → measurement dict.
        """
        tag_colors = {
            "bf16": "#2196F3", "gptq": "#4CAF50", "int8": "#E91E63",
            "fp8": "#00ACC1",
            "nf4": "#9C27B0",
        }

        tags, sizes = [], []
        for t, d in disk_data.items():
            gb = d.get("total_gb", d.get("model_size_gb", 0))
            if gb and gb > 0:
                tags.append(t.upper())
                sizes.append(gb)

        if not tags:
            logger.warning("No disk-size data to plot")
            return self.plots_dir / "disk_comparison.png"

        fig, ax = plt.subplots(figsize=(10, 6))
        colours = [tag_colors.get(t.lower(), "#888") for t in tags]
        bars = ax.bar(tags, sizes, color=colours, edgecolor="white", width=0.6)

        for bar, val in zip(bars, sizes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f} GB",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        ax.set_ylabel("Model Size (GB)", fontsize=13)
        ax.set_title(
            "Serialized Checkpoint Size on Disk — Quantisation Comparison",
            fontsize=15, fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        out = self.plots_dir / "disk_comparison.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"  saved: {out.name}")

        return out
