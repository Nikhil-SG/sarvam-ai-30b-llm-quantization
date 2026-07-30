#!/usr/bin/env python3
"""
Module 1c — Importance Map Generator.

Combines Fisher Information scores (from Module 1a) with expert routing
statistics (from Module 1b) to produce a per-expert importance classification:
HIGH / MEDIUM / LOW.

This map drives the precision assignment in Module 2:
    HIGH   → FP8  (sensitive experts, frequently activated)
    MEDIUM → FP8  (moderate sensitivity/frequency)
    LOW    → INT4 (insensitive, rarely activated, aggressive compression OK)

Attention + shared experts always get FP8 (they're always active).

Usage:
    python -m src.mxmoe.sensitivity.importance_map \\
        --fisher_path mxmoe/outputs/module_1_sensitivity/results/fisher_scores.json \\
        --routing_path mxmoe/outputs/module_1_sensitivity/results/routing_stats.json

RUN THIS NEXT: After fisher_info.py and expert_router_stats.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
NUM_MOE_LAYERS = 18
FIRST_MOE_LAYER = 1
NUM_EXPERTS = 128


class ImportanceMapBuilder:
    """
    Build an expert importance map from Fisher scores + routing statistics.

    The combined score is a weighted average of:
        - Normalized Fisher score (sensitivity to quantization)
        - Normalized routing frequency (utilization / how often activated)

    Experts that are both sensitive AND frequently used must keep high precision.
    Experts that are insensitive AND rarely used can be aggressively compressed.
    """

    def __init__(
        self,
        config=None,
        output_dir: str = "mxmoe/outputs/module_1_sensitivity/results",
        high_threshold: float = 0.7,
        low_threshold: float = 0.2,
        fisher_weight: float = 0.6,
        routing_weight: float = 0.4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.fisher_weight = fisher_weight
        self.routing_weight = routing_weight

        # Override from config if provided
        if config is not None:
            self.output_dir = Path(getattr(config.output, "results_dir", str(self.output_dir)))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            sens_cfg = getattr(config, "sensitivity", None)
            if sens_cfg:
                imp_cfg = getattr(sens_cfg, "importance", None)
                if imp_cfg:
                    self.high_threshold = getattr(imp_cfg, "high_threshold", self.high_threshold)
                    self.low_threshold = getattr(imp_cfg, "low_threshold", self.low_threshold)

    def build(
        self,
        fisher_scores: Dict[str, Any],
        routing_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the importance map.

        Args:
            fisher_scores: Output from FisherInformationAnalyzer.run()
                           Expected shape: {"fisher_scores": {layer: {expert: score}}}
            routing_stats: Output from ExpertRoutingAnalyzer.run()
                           Expected shape: {"routing_frequency": {layer: {expert: count}}}

        Returns:
            Complete importance map with per-expert classifications and
            recommended precisions.
        """
        logger.info("Building importance map from Fisher scores + routing statistics")

        # Extract the nested scores
        fisher_data = fisher_scores.get("fisher_scores", fisher_scores)
        routing_data = routing_stats.get("routing_frequency", routing_stats)

        # ── Collect all scores for global normalization ───────────────────
        all_fisher: List[float] = []
        all_routing: List[float] = []

        for layer_idx in range(FIRST_MOE_LAYER, FIRST_MOE_LAYER + NUM_MOE_LAYERS):
            li = str(layer_idx)
            fisher_layer = fisher_data.get(li, {})
            routing_layer = routing_data.get(li, {})

            for eid in range(NUM_EXPERTS):
                eid_str = str(eid)
                all_fisher.append(float(fisher_layer.get(eid_str, 0.0)))
                all_routing.append(float(routing_layer.get(eid_str, 0)))

        fisher_arr = np.array(all_fisher, dtype=np.float64)
        routing_arr = np.array(all_routing, dtype=np.float64)

        # ── Percentile-rank normalization to [0, 1] ─────────────────────
        # This replaces min-max normalization which was collapsing the
        # distribution when outlier Fisher scores were present, causing
        # almost all experts to cluster near 0 and be classified as LOW.
        from scipy.stats import rankdata
        fisher_ranked = rankdata(fisher_arr) / len(fisher_arr)
        routing_ranked = rankdata(routing_arr) / len(routing_arr)

        logger.info(f"  Fisher range:  [{fisher_arr.min():.6f}, {fisher_arr.max():.6f}]")
        logger.info(f"  Routing range: [{routing_arr.min():.0f}, {routing_arr.max():.0f}]")

        # ── Build per-expert importance map ───────────────────────────────
        importance_map: Dict[str, Dict[str, Any]] = {}
        stats_by_class = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        idx = 0
        for layer_idx in range(FIRST_MOE_LAYER, FIRST_MOE_LAYER + NUM_MOE_LAYERS):
            li = str(layer_idx)
            fisher_layer = fisher_data.get(li, {})
            routing_layer = routing_data.get(li, {})
            layer_map: Dict[str, Any] = {}

            for eid in range(NUM_EXPERTS):
                eid_str = str(eid)
                raw_fisher = float(fisher_layer.get(eid_str, 0.0))
                raw_routing = float(routing_layer.get(eid_str, 0))

                # Use percentile-ranked scores (idx tracks position in flat array)
                norm_fisher = float(fisher_ranked[idx])
                norm_routing = float(routing_ranked[idx])

                # Weighted combined score
                combined = (
                    self.fisher_weight * norm_fisher
                    + self.routing_weight * norm_routing
                )

                # Classify
                if combined >= self.high_threshold:
                    importance = "HIGH"
                    recommended_precision = "fp8"
                elif combined <= self.low_threshold:
                    importance = "LOW"
                    recommended_precision = "int4"
                else:
                    importance = "MEDIUM"
                    recommended_precision = "fp8"  # Upgraded from int4

                stats_by_class[importance] += 1

                layer_map[eid_str] = {
                    "importance": importance,
                    "combined_score": round(combined, 6),
                    "fisher_score_raw": raw_fisher,
                    "fisher_score_norm": round(norm_fisher, 6),
                    "routing_count": int(raw_routing),
                    "routing_norm": round(norm_routing, 6),
                    "recommended_precision": recommended_precision,
                }
                idx += 1

            # Shared expert — always HIGH importance
            shared_fisher = float(fisher_layer.get("shared", 0.0))
            layer_map["shared"] = {
                "importance": "HIGH",
                "combined_score": 1.0,
                "fisher_score_raw": shared_fisher,
                "recommended_precision": "fp8",
                "note": "Shared expert: always active, always quantized conservatively (FP8)",
            }

            importance_map[li] = layer_map

        logger.info(f"  Classification: HIGH={stats_by_class['HIGH']}, "
                     f"MEDIUM={stats_by_class['MEDIUM']}, LOW={stats_by_class['LOW']}")
        logger.info(f"  Total experts classified: {sum(stats_by_class.values())} "
                     f"(expected: {NUM_MOE_LAYERS * NUM_EXPERTS})")

        # ── Build summary ────────────────────────────────────────────────
        result = {
            "importance_map": importance_map,
            "thresholds": {
                "high": self.high_threshold,
                "low": self.low_threshold,
            },
            "weights": {
                "fisher": self.fisher_weight,
                "routing": self.routing_weight,
            },
            "classification_summary": stats_by_class,
            "normalization": {
                "method": "percentile_rank",
                "fisher_min": float(fisher_arr.min()),
                "fisher_max": float(fisher_arr.max()),
                "routing_min": float(routing_arr.min()),
                "routing_max": float(routing_arr.max()),
            },
            "architecture": {
                "num_moe_layers": NUM_MOE_LAYERS,
                "first_moe_layer": FIRST_MOE_LAYER,
                "num_experts_per_layer": NUM_EXPERTS,
                "dense_layers": [0],
            },
        }

        return result

    def save(
        self,
        importance_map: Dict[str, Any],
        filename: str = "expert_importance_map.json",
    ) -> Path:
        """Persist the importance map to disk."""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(importance_map, fh, indent=4, ensure_ascii=False, default=str)
        logger.info(f"Importance map saved: {path}")
        return path

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """Load a previously saved importance map."""
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def build_from_files(
        self,
        fisher_path: str,
        routing_path: str,
    ) -> Dict[str, Any]:
        """
        Convenience: load JSON files from disk and build the importance map.
        """
        logger.info(f"Loading Fisher scores from: {fisher_path}")
        with open(fisher_path, encoding="utf-8") as fh:
            fisher_scores = json.load(fh)

        logger.info(f"Loading routing stats from: {routing_path}")
        with open(routing_path, encoding="utf-8") as fh:
            routing_stats = json.load(fh)

        return self.build(fisher_scores, routing_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Module 1c: Importance Map Generator for sarvam-30b MoE experts"
    )
    parser.add_argument(
        "--fisher_path", type=str,
        default="mxmoe/outputs/module_1_sensitivity/results/fisher_scores.json",
        help="Path to Fisher scores JSON from Module 1a",
    )
    parser.add_argument(
        "--routing_path", type=str,
        default="mxmoe/outputs/module_1_sensitivity/results/routing_stats.json",
        help="Path to routing stats JSON from Module 1b",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="mxmoe/outputs/module_1_sensitivity/results",
        help="Output directory for the importance map",
    )
    parser.add_argument(
        "--high_threshold", type=float, default=0.7,
        help="Combined score threshold for HIGH classification (default: 0.7)",
    )
    parser.add_argument(
        "--low_threshold", type=float, default=0.2,
        help="Combined score threshold for LOW classification (default: 0.2)",
    )
    parser.add_argument(
        "--fisher_weight", type=float, default=0.6,
        help="Weight for Fisher score in combined metric (default: 0.6)",
    )
    parser.add_argument(
        "--routing_weight", type=float, default=0.4,
        help="Weight for routing frequency in combined metric (default: 0.4)",
    )
    args = parser.parse_args()

    builder = ImportanceMapBuilder(
        output_dir=args.output_dir,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        fisher_weight=args.fisher_weight,
        routing_weight=args.routing_weight,
    )

    result = builder.build_from_files(args.fisher_path, args.routing_path)
    builder.save(result)

    summary = result.get("classification_summary", {})
    logger.info(f"Importance map complete: "
                f"HIGH={summary.get('HIGH', 0)}, "
                f"MEDIUM={summary.get('MEDIUM', 0)}, "
                f"LOW={summary.get('LOW', 0)}")


if __name__ == "__main__":
    main()
