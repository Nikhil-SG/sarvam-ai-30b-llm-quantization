"""
INT8 Quantisation via bitsandbytes (LLM.int8()).

The most widely deployed quantisation method in production.
Uses ``BitsAndBytesConfig(load_in_8bit=True)`` which implements
the LLM.int8() algorithm (Dettmers et al., 2022):
  - Decomposes weights into two streams: normal INT8 and outlier FP16.
  - Outlier features (> 6.0 threshold) are kept in FP16 to preserve quality.
  - Near-lossless compression (~2× memory savings vs BF16).

Requirements:
    pip install bitsandbytes>=0.41.0
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.core.logger import get_logger
from src.quantization.base import BaseQuantizer

logger = get_logger(__name__)


class INT8Quantizer(BaseQuantizer):
    """8-bit integer quantisation via bitsandbytes (LLM.int8())."""

    QUANT_TAG = "int8"

    def _get_int8_cfg(self):
        return getattr(self.config.quantization, "int8", None)

    @staticmethod
    def _coerce_module_list(value: Any, default: Iterable[str]) -> List[str]:
        if value is None:
            items = list(default)
        elif isinstance(value, str):
            items = [token.strip() for token in value.split(",") if token.strip()]
        elif isinstance(value, (list, tuple, set)):
            items = [str(token).strip() for token in value if str(token).strip()]
        else:
            items = list(default)

        deduped: List[str] = []
        for item in items:
            if item not in deduped:
                deduped.append(item)
        return deduped

    def _resolve_int8_settings(self) -> Dict[str, Any]:
        settings: Dict[str, Any] = {
            "llm_int8_threshold": 6.0,
            "skip_modules": ["lm_head"],
            "llm_int8_has_fp16_weight": False,
            "llm_int8_enable_fp32_cpu_offload": False,
            "strict_linear_int8": False,
            "allow_float_linear_modules": ["lm_head"],
            "quantize_lm_head": False,
            "non_quantized_dtype": "float16",
        }

        int8_cfg = self._get_int8_cfg()
        if int8_cfg is None:
            return settings

        settings["llm_int8_threshold"] = float(
            getattr(int8_cfg, "llm_int8_threshold", settings["llm_int8_threshold"])
        )
        settings["skip_modules"] = self._coerce_module_list(
            getattr(int8_cfg, "llm_int8_skip_modules", settings["skip_modules"]),
            settings["skip_modules"],
        )
        settings["llm_int8_has_fp16_weight"] = bool(
            getattr(
                int8_cfg,
                "llm_int8_has_fp16_weight",
                settings["llm_int8_has_fp16_weight"],
            )
        )
        settings["llm_int8_enable_fp32_cpu_offload"] = bool(
            getattr(
                int8_cfg,
                "llm_int8_enable_fp32_cpu_offload",
                settings["llm_int8_enable_fp32_cpu_offload"],
            )
        )
        settings["strict_linear_int8"] = bool(
            getattr(int8_cfg, "strict_linear_int8", False)
            or getattr(int8_cfg, "strict_int8", False)
        )
        settings["allow_float_linear_modules"] = self._coerce_module_list(
            getattr(
                int8_cfg,
                "allow_float_linear_modules",
                settings["allow_float_linear_modules"],
            ),
            settings["allow_float_linear_modules"],
        )
        settings["quantize_lm_head"] = bool(
            getattr(int8_cfg, "quantize_lm_head", settings["quantize_lm_head"])
        )
        settings["non_quantized_dtype"] = str(
            getattr(int8_cfg, "non_quantized_dtype", settings["non_quantized_dtype"])
        )

        if settings["quantize_lm_head"]:
            settings["skip_modules"] = [
                name for name in settings["skip_modules"] if name != "lm_head"
            ]
            settings["allow_float_linear_modules"] = [
                name
                for name in settings["allow_float_linear_modules"]
                if name != "lm_head"
            ]

        return settings

    @staticmethod
    def _is_allowed_linear_module(name: str, allowlist: List[str]) -> bool:
        for pattern in allowlist:
            if not pattern:
                continue
            if name == pattern or name.endswith(pattern) or pattern in name:
                return True
        return False

    def _collect_linear_module_stats(
        self,
        allow_float_linear_modules: List[str],
    ) -> Dict[str, List[str]]:
        stats: Dict[str, List[str]] = {
            "linear8bitlt": [],
            "int8_weight": [],
            "float_allowed": [],
            "float_unexpected": [],
            "other_dtype": [],
        }

        for name, module in self.model.named_modules():
            if not (hasattr(module, "in_features") and hasattr(module, "out_features")):
                continue

            module_type = type(module).__name__.lower()
            weight = getattr(module, "weight", None)

            if "linear8bitlt" in module_type:
                stats["linear8bitlt"].append(name)
                continue

            if isinstance(weight, torch.Tensor) and weight.dtype == torch.int8:
                stats["int8_weight"].append(name)
                continue

            if isinstance(weight, torch.Tensor) and weight.is_floating_point():
                if self._is_allowed_linear_module(name, allow_float_linear_modules):
                    stats["float_allowed"].append(name)
                else:
                    stats["float_unexpected"].append(name)
                continue

            dtype_repr = getattr(weight, "dtype", type(weight).__name__)
            stats["other_dtype"].append(f"{name}:{dtype_repr}")

        return stats

    def _build_bnb_config(self, settings: Dict[str, Any]) -> BitsAndBytesConfig:
        kwargs = {
            "load_in_8bit": True,
            "llm_int8_threshold": settings["llm_int8_threshold"],
            "llm_int8_skip_modules": settings["skip_modules"],
            "llm_int8_has_fp16_weight": settings["llm_int8_has_fp16_weight"],
            "llm_int8_enable_fp32_cpu_offload": settings[
                "llm_int8_enable_fp32_cpu_offload"
            ],
        }
        supported = set(inspect.signature(BitsAndBytesConfig.__init__).parameters)
        filtered = {key: value for key, value in kwargs.items() if key in supported}
        return BitsAndBytesConfig(**filtered)

    def _assert_model_is_int8_quantized(
        self,
        context: str,
        *,
        strict_linear_int8: bool,
        allow_float_linear_modules: List[str],
    ) -> None:
        """Fail fast when a supposedly INT8 model is actually plain float."""
        if self.model is None:
            raise RuntimeError(f"[{context}] No model loaded to validate")

        stats = self._collect_linear_module_stats(allow_float_linear_modules)
        linear8bit_layers = stats["linear8bitlt"]
        int8_weight_layers = stats["int8_weight"]
        float_unexpected = stats["float_unexpected"]
        float_allowed = stats["float_allowed"]

        if not linear8bit_layers and not int8_weight_layers:
            raise RuntimeError(
                f"[{context}] Loaded model does not expose INT8 quantized layers"
            )

        if strict_linear_int8 and float_unexpected:
            preview = ", ".join(float_unexpected[:8])
            raise RuntimeError(
                f"[{context}] strict_linear_int8 failed: "
                f"{len(float_unexpected)} linear modules remain floating-point "
                f"outside allowlist ({preview})"
            )

        if float_unexpected:
            preview = ", ".join(float_unexpected[:5])
            logger.warning(
                "[INT8] Validation [%s]: %d floating-point linear modules outside allowlist "
                "(showing up to 5): %s",
                context,
                len(float_unexpected),
                preview,
            )

        logger.info(
            "[INT8] Validation [%s]: %d Linear8bitLt, %d int8-weight, "
            "%d float-allowed linear modules",
            context,
            len(linear8bit_layers),
            len(int8_weight_layers),
            len(float_allowed),
        )

    def _try_load_from_quantized_dir(self) -> bool:
        """Load cached INT8 checkpoint and reject it if it is not truly quantized."""
        if not super()._try_load_from_quantized_dir():
            return False

        settings = self._resolve_int8_settings()

        try:
            self._assert_model_is_int8_quantized(
                "cached_checkpoint",
                strict_linear_int8=settings["strict_linear_int8"],
                allow_float_linear_modules=settings["allow_float_linear_modules"],
            )
            return True
        except Exception as exc:
            logger.warning(
                "[INT8] Cached checkpoint validation failed: %s — will re-quantize",
                exc,
            )
            self.model = None
            return False

    def load_model(self) -> AutoModelForCausalLM:
        # ── Fast path: reload from a previously saved checkpoint ────────
        if self._try_load_from_quantized_dir():
            self._quantization_method = "cached"
            return self.model

        settings = self._resolve_int8_settings()
        bnb_config = self._build_bnb_config(settings)
        dtype_name = settings["non_quantized_dtype"]
        torch_dtype = getattr(torch, dtype_name, torch.float16)
        if not isinstance(torch_dtype, torch.dtype):
            torch_dtype = torch.float16
            dtype_name = "float16"

        logger.info(
            "Loading INT8 model (LLM.int8 mixed backend, threshold=%s, "
            "skip_modules=%s, has_fp16_weight=%s, strict_linear_int8=%s)",
            settings["llm_int8_threshold"],
            settings["skip_modules"],
            settings["llm_int8_has_fp16_weight"],
            settings["strict_linear_int8"],
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.config.hardware.device_map,
            max_memory=self.max_memory,
            torch_dtype=torch_dtype,
            token=self.hf_token,
            cache_dir=getattr(self.config.model, "cache_dir", None),
            trust_remote_code=getattr(
                self.config.model, "trust_remote_code", False
            ),
        )
        self.model.eval()
        self._assert_model_is_int8_quantized(
            "fresh_quantization",
            strict_linear_int8=settings["strict_linear_int8"],
            allow_float_linear_modules=settings["allow_float_linear_modules"],
        )
        self._quantization_method = "BitsAndBytesConfig_int8"

        logger.info(
            "INT8 model loaded via bitsandbytes (LLM.int8 mixed); "
            "non-quantized dtype=%s",
            dtype_name,
        )
        return self.model
