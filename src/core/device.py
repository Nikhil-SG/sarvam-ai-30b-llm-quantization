"""
GPU device management and hardware validation.

Primary GPU policy:
    This project targets 2× A100 80 GB systems and designates **cuda:1**
    as the primary GPU for all workloads.  cuda:0 is used as an overflow /
    secondary device.  Every entry-point should call ``set_primary_cuda_device()``
    early so that PyTorch defaults, CUDA context creation, and HuggingFace
    ``device_map="auto"`` all prefer cuda:1.
"""

import torch
from typing import Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Primary-device bootstrap
# ---------------------------------------------------------------------------

def set_primary_cuda_device(
    preferred_index: Optional[int] = None,
    default_index: int = 1,
) -> int:
    """
    Set the PyTorch default CUDA device to the primary GPU.

    Call this **once** at the very start of every entry-point (main.py) before any model loading.

    Returns the resolved primary index.
    """
    primary = resolve_primary_cuda_index(preferred_index, default_index)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(primary)
        logger.info(
            f"Primary CUDA device set to cuda:{primary} "
            f"(total GPUs: {torch.cuda.device_count()})"
        )
    return primary


def resolve_primary_cuda_index(
    preferred_index: Optional[int] = None,
    default_index: int = 1,
) -> int:
    """
    Resolve the primary CUDA index used for single-device-preferring flows.

    Defaults to cuda:1 when available, otherwise falls back to GPU index 0.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return 0

    try:
        idx = int(default_index if preferred_index is None else preferred_index)
    except (TypeError, ValueError):
        idx = default_index

    if 0 <= idx < torch.cuda.device_count():
        return idx
    return 0


def build_cuda_priority_order(
    preferred_index: Optional[int] = None,
    default_index: int = 1,
) -> List[int]:
    """
    Build CUDA retry/priority order with configured primary first.
    """
    if not torch.cuda.is_available():
        return []

    n = torch.cuda.device_count()
    primary = resolve_primary_cuda_index(preferred_index, default_index)
    return [primary] + [i for i in range(n) if i != primary]


def _parse_gib(value: str) -> Optional[float]:
    """Parse strings like '75GiB' into numeric GiB values."""
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw.lower().endswith("gib"):
        return None
    try:
        return float(raw[:-3])
    except ValueError:
        return None


def get_device_info() -> List[Dict]:
    """
    Enumerate CUDA devices and return their properties.

    Returns:
        List of dicts with keys: index, name, total_memory_gb,
        compute_capability, multi_processor_count.
    """
    devices: List[Dict] = []
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available – running on CPU only.")
        return devices

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info = {
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
        devices.append(info)
        logger.info(
            f"GPU {i}: {info['name']} | "
            f"{info['total_memory_gb']} GB | "
            f"CC {info['compute_capability']}"
        )
    return devices


def build_max_memory_map(
    max_memory_config: Optional[Dict[str, str]] = None,
    default_per_gpu: str = "75GiB",
    primary_cuda_index: Optional[int] = None,
    secondary_reduction_gib: int = 5,
) -> Dict:
    """
    Build a ``max_memory`` dict accepted by ``device_map="auto"``.

    The map always biases toward the primary GPU (default cuda:1) so that
    ``device_map="auto"`` places the majority of layers there.

    Args:
        max_memory_config: Mapping from GPU index (str) → memory string.
        default_per_gpu: Fallback value when config is absent.
        primary_cuda_index: Preferred GPU index (defaults to cuda:1 when available).
        secondary_reduction_gib: How much to reduce non-primary GPUs when
            auto-generating defaults to bias placement to the primary GPU.
    """
    primary = resolve_primary_cuda_index(primary_cuda_index)

    if max_memory_config:
        mem = {int(k): v for k, v in max_memory_config.items()}
        # Even with explicit config, enforce primary GPU bias: if all GPUs
        # have the same budget, reduce non-primary GPUs automatically.
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            parsed_values = {
                gpu_idx: _parse_gib(budget)
                for gpu_idx, budget in mem.items()
                if isinstance(gpu_idx, int) and _parse_gib(budget) is not None
            }
            if len(set(parsed_values.values())) == 1 and len(parsed_values) > 1:
                # All GPUs have equal budgets — reduce non-primary to bias placement
                for gpu_idx, parsed in parsed_values.items():
                    if gpu_idx != primary:
                        reduced = max(parsed - float(secondary_reduction_gib), 1.0)
                        mem[gpu_idx] = f"{reduced:.0f}GiB"
                logger.debug(
                    f"Biased max_memory toward cuda:{primary} "
                    f"(non-primary GPUs reduced by {secondary_reduction_gib} GiB)"
                )
    else:
        mem = {}
        if torch.cuda.is_available():
            parsed = _parse_gib(default_per_gpu)

            for i in range(torch.cuda.device_count()):
                mem[i] = default_per_gpu

            # When no explicit config is provided, bias auto placement toward
            # the primary GPU by giving non-primary GPUs a lower cap.
            if parsed is not None and torch.cuda.device_count() > 1:
                reduced = max(parsed - float(secondary_reduction_gib), 1.0)
                for i in range(torch.cuda.device_count()):
                    if i != primary:
                        mem[i] = f"{reduced:.0f}GiB"

    # Always include CPU offload as a safety net so device_map="auto"
    # can spill layers to RAM instead of OOM-ing during quantized loading.
    if "cpu" not in mem:
        mem["cpu"] = "100GiB"
    return mem


def validate_hardware(
    required_gpus: int = 1, min_memory_gb: float = 24.0
) -> bool:
    """
    Ensure the system meets minimum hardware requirements.

    Defaults are conservative: 1 GPU with 24 GB suffices for
    quantized variants of sarvam-30b.  The caller (main.py) can
    pass stricter thresholds when running the BF16 baseline.

    Returns:
        True if all checks pass, False otherwise.
    """
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        return False

    num_gpus = torch.cuda.device_count()
    if num_gpus < required_gpus:
        logger.error(f"Need {required_gpus} GPUs, found {num_gpus}")
        return False

    for i in range(num_gpus):
        mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        if mem_gb < min_memory_gb:
            logger.error(
                f"GPU {i} has {mem_gb:.1f} GB, need ≥ {min_memory_gb} GB"
            )
            return False

    logger.info(
        f"Hardware OK: {num_gpus} GPU(s), "
        f">= {min_memory_gb} GB each"
    )
    return True
