from src.quantization.base import BaseQuantizer
from src.quantization.bf16_baseline import BF16Baseline
from src.quantization.fp8_quantizer import FP8Quantizer
from src.quantization.gptq_quantizer import GPTQQuantizer
from src.quantization.int8_quantizer import INT8Quantizer
from src.quantization.nf4_quantizer import NF4Quantizer

QUANTIZER_REGISTRY = {
    "bf16": BF16Baseline,
    "fp8": FP8Quantizer,
    "gptq": GPTQQuantizer,
    "int8": INT8Quantizer,
    "nf4": NF4Quantizer,
}

__all__ = [
    "BaseQuantizer",
    "BF16Baseline",
    "FP8Quantizer",
    "GPTQQuantizer",
    "INT8Quantizer",
    "NF4Quantizer",
    "QUANTIZER_REGISTRY",
]
