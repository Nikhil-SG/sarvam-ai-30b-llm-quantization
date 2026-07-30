#!/usr/bin/env python3
"""
Module 2a — Dynamic llm-compressor Recipe Builder.

Reads the expert_importance_map.json from Module 1 and generates a
heterogeneous quantization recipe for llm-compressor. Each expert group
gets a different precision based on its importance classification:

    Component              | fp8_gptq    | int8_gptq   | Modifier
    ───────────────────────────────────────────────────────────────
    Attention (QKV, dense) | FP8_DYNAMIC | W8A16 INT8  | oneshot() call 2 (data-free)
    Dense layer 0 MLP      | FP8_DYNAMIC | W8A16 INT8  | oneshot() call 2 (data-free)
    Shared experts         | FP8_DYNAMIC | W8A16 INT8  | oneshot() call 2 (data-free)
    HIGH-importance routed | FP8_DYNAMIC | W8A16 INT8  | oneshot() call 2 (data-free)
    MEDIUM-importance      | FP8_DYNAMIC | W8A16 INT8  | oneshot() call 2 (data-free)
    LOW-importance         | W4A16 (GPTQ)| oneshot() call 1 (calibration)
    lm_head, gates (router) | Ignored     | —

Supports multiple strategies:
    - fp8_gptq:  FP8_DYNAMIC for non-LOW, GPTQ W4A16 for LOW
    - int8_gptq: W8A16 for non-LOW, GPTQ W4A16 for LOW

Architecture module paths (from modeling_sarvam_moe.py):
    model.layers.{0}.mlp                     → Dense (not MoE)
    model.layers.{i}.attention.query_key_value
    model.layers.{i}.attention.dense
    model.layers.{i}.mlp.experts.{j}.gate_proj
    model.layers.{i}.mlp.experts.{j}.up_proj
    model.layers.{i}.mlp.experts.{j}.down_proj
    model.layers.{i}.mlp.shared_experts.gate_proj
    model.layers.{i}.mlp.shared_experts.up_proj
    model.layers.{i}.mlp.shared_experts.down_proj

Usage:
    python -m src.mxmoe.recipe.recipe_builder \\
        --importance_map mxmoe/outputs/module_1_sensitivity/results/expert_importance_map.json

RUN THIS NEXT: After Module 1 (importance_map.py). Generates recipe for compressor.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
NUM_MOE_LAYERS = 18
FIRST_MOE_LAYER = 1
NUM_EXPERTS = 128

# Supported quantization strategies
SUPPORTED_STRATEGIES = ["fp8_gptq", "int8_gptq"]


class PrecisionRecipe:
    """Container for the heterogeneous precision recipe.

    Target lists:
        fp8_targets:  modules quantized with FP8_DYNAMIC (used by fp8_gptq strategy)
        int8_targets: modules quantized with W8A16 INT8 (used by int8_gptq strategy)
                      — same modules as fp8_targets, different scheme per strategy.
        gptq_targets: modules quantized with GPTQ W4A16 (shared across strategies)
        gptq_low_targets: legacy field, kept for backward compatibility (always empty)
    """

    def __init__(
        self,
        fp8_targets: List[str],
        gptq_targets: List[str],
        ignore_list: List[str],
        metadata: Dict[str, Any],
        int8_targets: Optional[List[str]] = None,
        gptq_low_targets: Optional[List[str]] = None,
    ):
        self.fp8_targets = fp8_targets
        self.gptq_targets = gptq_targets
        self.ignore_list = ignore_list
        self.metadata = metadata
        # int8_targets mirrors fp8_targets (same modules, different quant scheme)
        self.int8_targets = int8_targets if int8_targets is not None else list(fp8_targets)
        # Legacy — kept for backward compatibility
        self.gptq_low_targets = gptq_low_targets or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fp8_targets": self.fp8_targets,
            "int8_targets": self.int8_targets,
            "gptq_targets": self.gptq_targets,
            "gptq_low_targets": self.gptq_low_targets,
            "ignore_list": self.ignore_list,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrecisionRecipe":
        return cls(
            fp8_targets=data["fp8_targets"],
            gptq_targets=data["gptq_targets"],
            ignore_list=data["ignore_list"],
            metadata=data.get("metadata", {}),
            int8_targets=data.get("int8_targets", []),
            gptq_low_targets=data.get("gptq_low_targets", []),
        )

    def summary(self) -> str:
        lines = [
            "=== Precision Recipe Summary ===",
            f"  FP8 targets (fp8_gptq):  {len(self.fp8_targets)} module patterns",
            f"  INT8 targets (int8_gptq):{len(self.int8_targets)} module patterns",
            f"  GPTQ W4A16 targets:      {len(self.gptq_targets)} module patterns (LOW experts)",
            f"  Ignored modules:         {len(self.ignore_list)}",
            f"  Total quantized:         {len(self.fp8_targets) + len(self.gptq_targets)}",
        ]
        return "\n".join(lines)


class RecipeBuilder:
    """
    Generate a heterogeneous precision recipe from the importance map.

    The recipe uses llm-compressor's `targets` regex patterns so that each
    config_group applies to the correct subset of Linear layers.

    Supports multiple strategies via the `strategies` config list.
    """

    def __init__(
        self,
        config=None,
        output_dir: str = "mxmoe/outputs/module_2_synthesis/results",
        attention_precision: str = "fp8",
        shared_expert_precision: str = "fp8",
        high_precision: str = "fp8",
        medium_precision: str = "fp8",
        low_precision: str = "int4",
        gptq_group_size: int = 128,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_output_dir = Path("mxmoe/outputs")
        self.attention_precision = attention_precision
        self.shared_expert_precision = shared_expert_precision
        self.high_precision = high_precision
        self.medium_precision = medium_precision
        self.low_precision = low_precision
        self.gptq_group_size = gptq_group_size

        # Multi-strategy support
        self.strategies: List[str] = ["fp8_gptq", "int8_gptq"]

        if config is not None:
            self.base_output_dir = Path(
                getattr(config.output, "base_dir", str(self.base_output_dir))
            )
            self.output_dir = Path(getattr(config.output, "results_dir", str(self.output_dir)))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            recipe_cfg = getattr(config, "recipe", None)
            if recipe_cfg:
                self.attention_precision = getattr(recipe_cfg, "attention_precision", self.attention_precision)
                self.shared_expert_precision = getattr(recipe_cfg, "shared_expert_precision", self.shared_expert_precision)
                self.high_precision = getattr(recipe_cfg, "high_importance_precision", self.high_precision)
                self.medium_precision = getattr(recipe_cfg, "medium_importance_precision", self.medium_precision)
                self.low_precision = getattr(recipe_cfg, "low_importance_precision", self.low_precision)
                # Load strategies from config
                strategies_cfg = getattr(recipe_cfg, "strategies", None)
                if strategies_cfg:
                    self.strategies = list(strategies_cfg)

    def build(self, importance_map_path: Optional[str] = None) -> PrecisionRecipe:
        """
        Build a precision recipe from the importance map.

        Args:
            importance_map_path: Path to expert_importance_map.json.
                If None, looks in the Module 1 results directory.

        Returns:
            PrecisionRecipe with per-component target lists.
        """
        logger.info("=" * 60)
        logger.info("  MODULE 2a: Building Precision Recipe")
        logger.info("=" * 60)

        # ── Load importance map ──────────────────────────────────────────
        if importance_map_path is None:
            importance_map_path = str(
                self.base_output_dir
                / "module_1_sensitivity"
                / "results"
                / "expert_importance_map.json"
            )

        if not Path(importance_map_path).exists():
            raise FileNotFoundError(
                f"Importance map not found: {importance_map_path}\n"
                "Run Module 1 first: python -m pipelines.mxmoe.pipeline --module 1"
            )

        with open(importance_map_path, encoding="utf-8") as fh:
            importance_data = json.load(fh)

        imap = importance_data.get("importance_map", importance_data)

        # ── Build target lists ───────────────────────────────────────────
        # fp8_targets = modules for data-free quant (FP8 or INT8, depending on strategy)
        # gptq_targets = modules for GPTQ W4A16 (LOW experts only)
        fp8_targets: List[str] = []
        gptq_targets: List[str] = []
        ignore_list: List[str] = [
            "lm_head",
            # MoE gate/router MUST NOT be quantized — quantizing the router
            # destroys expert routing and causes degenerate output ("is is is...").
            # The research FP8 quantizer (fp8_quantizer.py) also excludes these.
            "re:.*mlp\\.gate$",
        ]

        stats = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "shared": 0}

        for layer_idx in range(FIRST_MOE_LAYER, FIRST_MOE_LAYER + NUM_MOE_LAYERS):
            li = str(layer_idx)
            layer_data = imap.get(li, {})

            for expert_idx in range(NUM_EXPERTS):
                eid = str(expert_idx)
                expert_info = layer_data.get(eid, {})
                importance = expert_info.get("importance", "MEDIUM")

                # Build exact module path patterns for this expert's Linear layers
                expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                expert_linears = [
                    f"{expert_prefix}.gate_proj",
                    f"{expert_prefix}.up_proj",
                    f"{expert_prefix}.down_proj",
                ]

                if importance == "HIGH":
                    fp8_targets.extend(expert_linears)
                    stats["HIGH"] += 1
                elif importance == "LOW":
                    gptq_targets.extend(expert_linears)
                    stats["LOW"] += 1
                else:  # MEDIUM → data-free quant (FP8 or INT8)
                    fp8_targets.extend(expert_linears)
                    stats["MEDIUM"] += 1

            # Shared experts → data-free quant (always)
            shared_prefix = f"model.layers.{layer_idx}.mlp.shared_experts"
            fp8_targets.extend([
                f"{shared_prefix}.gate_proj",
                f"{shared_prefix}.up_proj",
                f"{shared_prefix}.down_proj",
            ])
            stats["shared"] += 1

        # ── Attention layers → data-free quant (all 19 layers) ──
        for layer_idx in range(19):  # 0..18
            fp8_targets.extend([
                f"model.layers.{layer_idx}.attention.query_key_value",
                f"model.layers.{layer_idx}.attention.dense",
            ])

        # ── Dense layer 0's MLP → data-free quant ─────────────────────
        fp8_targets.extend([
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        ])

        # int8_targets = same modules as fp8_targets (different quant scheme per strategy)
        int8_targets = list(fp8_targets)

        logger.info(f"Recipe classification: {stats}")
        logger.info(f"  FP8 targets (fp8_gptq):  {len(fp8_targets)} modules (attention + shared + HIGH + MEDIUM)")
        logger.info(f"  INT8 targets (int8_gptq): {len(int8_targets)} modules (same modules, W8A16 scheme)")
        logger.info(f"  GPTQ targets (shared):    {len(gptq_targets)} modules (LOW only)")
        logger.info(f"  Active strategies: {self.strategies}")

        recipe = PrecisionRecipe(
            fp8_targets=fp8_targets,
            gptq_targets=gptq_targets,
            ignore_list=ignore_list,
            metadata={
                "precision_mapping": {
                    "attention": "fp8_or_int8",
                    "shared_experts": "fp8_or_int8",
                    "high_importance": "fp8_or_int8",
                    "medium_importance": "fp8_or_int8",
                    "low_importance": "int4",
                },
                "strategies": self.strategies,
                "gptq_group_size": self.gptq_group_size,
                "expert_classification_counts": stats,
            },
            int8_targets=int8_targets,
        )

        # ── Save recipe ─────────────────────────────────────────────────
        self.save_recipe(recipe)

        return recipe

    def save_recipe(
        self, recipe: PrecisionRecipe, filename: str = "precision_recipe.json"
    ) -> Path:
        """Persist the recipe to disk."""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(recipe.to_dict(), fh, indent=4, ensure_ascii=False, default=str)
        logger.info(f"Precision recipe saved: {path}")
        return path


def main():
    parser = argparse.ArgumentParser(
        description="Module 2a: Dynamic Precision Recipe Builder for sarvam-30b MxMoE"
    )
    parser.add_argument(
        "--importance_map", type=str,
        default="mxmoe/outputs/module_1_sensitivity/results/expert_importance_map.json",
        help="Path to expert_importance_map.json from Module 1",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="mxmoe/outputs/module_2_synthesis/results",
        help="Output directory for the recipe",
    )
    parser.add_argument(
        "--gptq_group_size", type=int, default=128,
        help="GPTQ group size for LOW experts (default: 128)",
    )
    args = parser.parse_args()

    builder = RecipeBuilder(
        output_dir=args.output_dir,
        gptq_group_size=args.gptq_group_size,
    )
    recipe = builder.build(args.importance_map)
    logger.info(recipe.summary())


if __name__ == "__main__":
    main()
