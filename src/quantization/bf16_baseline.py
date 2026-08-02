"""
Module 1 – BF16 Baseline ("The Gold Standard").

Loads the target model (sarvamai/sarvam-30b) in bfloat16 across
available GPUs and establishes upper-bound accuracy / resource metrics.

The baseline:
  - Auto-detects model architecture (layers, projections, MoE topology)
  - Measures static memory (param count, model size in GB)
  - Runs a short inference to measure dynamic memory (peak, KV-cache delta)
  - Caches weight samples for downstream analysis modules
  - Does NOT save a quantized checkpoint (the original model is the baseline)
"""

import json
import torch
from pathlib import Path
from typing import Any, Dict, Optional
from transformers import AutoModelForCausalLM

from src.core.logger import get_logger
from src.quantization.base import BaseQuantizer

logger = get_logger(__name__)


class BF16Baseline(BaseQuantizer):
    """Full-precision BF16 reference model."""

    QUANT_TAG = "bf16"

    def load_model(self) -> AutoModelForCausalLM:
        logger.info(f"Loading BF16 baseline (device_map={self.config.hardware.device_map})")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.config.hardware.device_map,
            max_memory=self.max_memory,
            token=self.hf_token,
            attn_implementation=getattr(
                self.config.model, "attn_implementation", "sdpa"
            ),
            cache_dir=getattr(self.config.model, "cache_dir", None),
            trust_remote_code=getattr(
                self.config.model, "trust_remote_code", False
            ),
        )
        self.model.eval()
        self._quantization_method = "bf16_native"

        logger.info("BF16 baseline loaded")
        return self.model

    def save_quantized_model(self) -> Optional[Path]:
        """
        BF16 is the full-precision baseline. Saving it to
        quantized_models/ is wasteful — it can always be loaded
        from HuggingFace Hub or local_model_path.
        """
        logger.info(
            f"[{self.QUANT_TAG}] Skipping save — BF16 is the reference "
            f"model, not a quantized checkpoint"
        )
        return None

    def run(self, cache_weights: bool = True) -> Dict[str, Any]:
        """
        Extended run pipeline for Module 1.

        Adds architecture info summary to the results JSON
        so downstream modules can reference it without reloading.
        """
        results = super().run(cache_weights=cache_weights)

        # Save architecture info separately for easy reference
        if self.arch_info:
            arch_path = self.results_dir / "model_architecture.json"
            arch_path.parent.mkdir(parents=True, exist_ok=True)
            with open(arch_path, "w", encoding="utf-8") as fh:
                json.dump(self.arch_info, fh, indent=2, default=str)
            logger.info(f"Architecture info saved: {arch_path}")

        return results
