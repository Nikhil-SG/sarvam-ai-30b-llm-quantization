"""
GPU memory tracking and management utilities.

Provides snapshot capture, peak-memory tracking, automatic cleanup,
and a context-manager for measuring memory deltas during operations.
"""

import gc
import torch
import psutil
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

from src.core.logger import get_logger

logger = get_logger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class MemorySnapshot:
    """Point-in-time snapshot of GPU and CPU memory."""

    gpu_allocated_gb: Dict[int, float] = field(default_factory=dict)
    gpu_reserved_gb: Dict[int, float] = field(default_factory=dict)
    gpu_total_gb: Dict[int, float] = field(default_factory=dict)
    gpu_utilization_pct: Dict[int, float] = field(default_factory=dict)
    cpu_ram_used_gb: float = 0.0
    cpu_ram_total_gb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = []
        for gid in sorted(self.gpu_allocated_gb):
            parts.append(
                f"GPU{gid} {self.gpu_allocated_gb[gid]:.2f}/{self.gpu_total_gb[gid]:.0f}GB "
                f"({self.gpu_utilization_pct[gid]:.1f}%)"
            )
        parts.append(f"CPU {self.cpu_ram_used_gb:.1f}/{self.cpu_ram_total_gb:.0f}GB")
        return "  ".join(parts)


# ── Snapshot helpers ────────────────────────────────────────────────────────
def get_memory_snapshot() -> MemorySnapshot:
    """Capture current memory usage across all GPUs and CPU."""
    snap = MemorySnapshot()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
            resv = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)

            snap.gpu_allocated_gb[i] = round(alloc, 3)
            snap.gpu_reserved_gb[i] = round(resv, 3)
            snap.gpu_total_gb[i] = round(total, 3)
            snap.gpu_utilization_pct[i] = (
                round((alloc / total) * 100, 2) if total > 0 else 0.0
            )

    ram = psutil.virtual_memory()
    snap.cpu_ram_used_gb = round(ram.used / (1024 ** 3), 3)
    snap.cpu_ram_total_gb = round(ram.total / (1024 ** 3), 3)

    return snap


def get_peak_memory() -> Dict[int, float]:
    """Return peak allocated memory (GB) per GPU since last reset."""
    peaks: Dict[int, float] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            peaks[i] = round(
                torch.cuda.max_memory_allocated(i) / (1024 ** 3), 3
            )
    return peaks


def reset_peak_memory() -> None:
    """Reset the peak memory tracking counters on every GPU."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


# ── Cleanup ─────────────────────────────────────────────────────────────────
def cleanup_model(model) -> None:
    """Delete a model and aggressively free GPU memory."""
    if model is not None:
        # Remove accelerate dispatch hooks that hold internal GPU
        # tensor references and prevent the model from being collected.
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)
        except Exception:
            pass
        try:
            model.cpu()  # move tensors off GPU before deletion
        except Exception:
            pass
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # Second gc pass to catch weak-ref / destructor cycles
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("GPU memory released")


# ── Context manager ─────────────────────────────────────────────────────────
@contextmanager
def track_memory(label: str = "operation"):
    """
    Context manager that logs memory usage before and after a block.

    Usage::

        with track_memory("load_model"):
            model = load(...)
    """
    before = get_memory_snapshot()
    logger.debug(f"[{label}] before: {before.summary()}")

    yield before

    after = get_memory_snapshot()
    deltas = ", ".join(
        f"GPU{gid} {after.gpu_allocated_gb[gid] - before.gpu_allocated_gb.get(gid, 0):+.2f}GB"
        for gid in after.gpu_allocated_gb
    )
    logger.info(f"[{label}] memory delta: {deltas}")
