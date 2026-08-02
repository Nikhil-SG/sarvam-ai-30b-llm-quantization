"""
NF4 Quantisation via bitsandbytes (4-bit NormalFloat).

Uses ``BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')``
which implements the NF4 data type from the QLoRA paper (Dettmers et al., 2023):
  - 4-bit NormalFloat quantization optimised for normally-distributed weights.
  - Double quantization compresses the quantization constants themselves.
  - Computation in BF16 for stable training / inference.
  - ~4× memory savings vs BF16 with minimal quality loss.

This is the most popular 4-bit format, widely used for:
  - QLoRA fine-tuning
  - Inference on consumer GPUs
  - Research on aggressive quantization

Requirements:
    pip install bitsandbytes>=0.41.0

References:
    Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Large Language
    Models", NeurIPS 2023. https://arxiv.org/abs/2305.14314
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.core.logger import get_logger
from src.quantization.base import BaseQuantizer

logger = get_logger(__name__)


class NF4Quantizer(BaseQuantizer):
    """4-bit NormalFloat quantisation via bitsandbytes (QLoRA format)."""

    QUANT_TAG = "nf4"

    def _assert_model_is_nf4_quantized(self, context: str) -> None:
        """Fail fast when a supposedly NF4 model is actually plain float."""
        if self.model is None:
            raise RuntimeError(f"[{context}] No model loaded to validate")

        linear4bit_layers = []
        quant_state_layers = []
        for name, module in self.model.named_modules():
            module_type = type(module).__name__.lower()
            if "linear4bit" in module_type:
                linear4bit_layers.append(name)

            weight = getattr(module, "weight", None)
            if hasattr(weight, "quant_state") and getattr(weight, "quant_state", None) is not None:
                quant_state_layers.append(name)

        if not linear4bit_layers and not quant_state_layers:
            raise RuntimeError(
                f"[{context}] Loaded model does not expose NF4 quantized layers"
            )

        logger.info(
            "[NF4] Validation [%s]: %d Linear4bit modules, %d quant_state weights",
            context,
            len(linear4bit_layers),
            len(quant_state_layers),
        )

    def _try_load_from_quantized_dir(self) -> bool:
        """Load cached NF4 checkpoint and reject it if it is not truly quantized."""
        if not super()._try_load_from_quantized_dir():
            return False

        try:
            self._assert_model_is_nf4_quantized("cached_checkpoint")
            return True
        except Exception as exc:
            logger.warning(
                "[NF4] Cached checkpoint validation failed: %s — will re-quantize",
                exc,
            )
            self.model = None
            return False

    def load_model(self) -> AutoModelForCausalLM:
        # ── Fast path: reload from a previously saved checkpoint ────────
        if self._try_load_from_quantized_dir():
            self._quantization_method = "cached"
            return self.model

        nf4_cfg = getattr(self.config.quantization, "nf4", None)

        # Configuration options with sensible defaults
        compute_dtype = torch.bfloat16
        double_quant = True
        if nf4_cfg is not None:
            double_quant = getattr(nf4_cfg, "double_quant", True)
            compute_str = getattr(nf4_cfg, "compute_dtype", "bfloat16")
            compute_dtype = getattr(torch, compute_str, torch.bfloat16)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=double_quant,
        )

        logger.info(
            f"Loading NF4 model (4-bit NormalFloat, "
            f"double_quant={double_quant}, compute_dtype={compute_dtype})"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.config.hardware.device_map,
            max_memory=self.max_memory,
            torch_dtype=torch.float16,
            token=self.hf_token,
            cache_dir=getattr(self.config.model, "cache_dir", None),
            trust_remote_code=getattr(
                self.config.model, "trust_remote_code", False
            ),
        )
        self.model.eval()
        self._assert_model_is_nf4_quantized("fresh_quantization")
        self._quantization_method = "BitsAndBytesConfig_nf4"

        logger.info("NF4 model loaded via bitsandbytes (4-bit NormalFloat)")
        return self.model
