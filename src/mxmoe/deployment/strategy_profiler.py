#!/usr/bin/env python3
"""
Module 4x — MxMoE Strategy Profiling.

Profiles latency, VRAM, and disk usage for each quantized strategy
(e.g., fp8_gptq, int8_gptq) using the shared profiling utilities.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.core.logger import get_logger
from src.core.memory import reset_peak_memory
from src.profiling.latency import LatencyProfiler
from src.profiling.vram import VRAMProfiler
from src.profiling.disk import DiskProfiler

logger = get_logger(__name__)

# ── Monkeypatch ModelCompressor for older compressed-tensors compatibility ──
try:
    from compressed_tensors import ModelCompressor
    if not hasattr(ModelCompressor, "compress_model") and hasattr(ModelCompressor, "compress"):
        ModelCompressor.compress_model = lambda self, model, *args, **kwargs: self.compress(model, *args, **kwargs)
except ImportError:
    pass

DEFAULT_QUANTIZED_PATH = "mxmoe/quantized_models"
DEFAULT_STRATEGIES = ["fp8_gptq", "int8_gptq"]


class StrategyProfiler:
    """Profile latency/VRAM/disk for each MxMoE strategy directory."""

    def __init__(
        self,
        config=None,
        model_path: str = DEFAULT_QUANTIZED_PATH,
        strategies: Optional[List[str]] = None,
    ):
        self.config = config
        self.model_path = model_path
        self.strategies = list(strategies) if strategies else None

        if config is not None:
            out_cfg = getattr(config, "output", None)
            if out_cfg:
                self.model_path = getattr(out_cfg, "quantized_models_dir", self.model_path)
            recipe_cfg = getattr(config, "recipe", None)
            if recipe_cfg:
                cfg_strategies = list(getattr(recipe_cfg, "strategies", []) or [])
                if cfg_strategies:
                    self.strategies = cfg_strategies

    def _resolve_strategy_dirs(self) -> List[Tuple[str, str]]:
        base_dir = Path(self.model_path).parent
        model_name = Path(self.model_path).name
        strategies = self.strategies or list(DEFAULT_STRATEGIES)
        resolved: List[Tuple[str, str]] = []

        for strategy in strategies:
            strategy_dir = base_dir / f"{model_name}_{strategy}"
            if strategy_dir.exists():
                resolved.append((strategy, str(strategy_dir)))

        if not resolved and Path(self.model_path).exists():
            resolved.append(("default", self.model_path))

        return resolved

    def _resolve_max_memory(self) -> Optional[Dict]:
        """Resolve max_memory from config, ensuring integer keys for GPU devices."""
        hw_cfg = getattr(self.config, "hardware", None) if self.config else None
        if not hw_cfg or not getattr(hw_cfg, "max_memory", None):
            return None
        mm = hw_cfg.max_memory
        if isinstance(mm, dict) or hasattr(mm, "items"):
            resolved = {}
            for k, v in (mm if isinstance(mm, dict) else dict(mm)).items():
                # accelerate requires integer keys for GPU indices
                try:
                    resolved[int(k)] = v
                except (ValueError, TypeError):
                    resolved[k] = v  # "cpu", "disk" stay as strings
            return resolved
        return None

    def _load_model(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading quantized model from {model_path} ...")

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        max_memory = self._resolve_max_memory()
        if not max_memory and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Auto-balance: cap each GPU at ~70GB to leave headroom for inference
            num_gpus = torch.cuda.device_count()
            max_memory = {}
            for i in range(num_gpus):
                total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                cap_gb = min(total_gb * 0.85, 70.0)  # 85% or 70GB, whichever is lower
                max_memory[i] = f"{cap_gb:.0f}GiB"  # integer keys required by accelerate
            max_memory["cpu"] = "80GiB"
        if max_memory:
            logger.info(f"  max_memory: {max_memory}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
        )
        model.eval()

        self._dequantize_fp8_weights(model)
        return model, tokenizer

    @staticmethod
    def _dequantize_fp8_weights(model) -> None:
        """Cast FP8 weights to bf16 for A100 compatibility."""
        fp8_dtypes = set()
        try:
            fp8_dtypes.add(torch.float8_e4m3fn)
            fp8_dtypes.add(torch.float8_e5m2)
        except AttributeError:
            return

        num_cast = 0
        for _, param in model.named_parameters():
            if param.dtype in fp8_dtypes:
                param.data = param.data.to(torch.bfloat16)
                num_cast += 1

        if num_cast > 0:
            logger.info(f"  ✓ Dequantized {num_cast} FP8 parameters → bf16")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _unload_model(model, tokenizer) -> None:
        """Aggressively free GPU memory after profiling a strategy."""
        try:
            # Move model off GPU if possible
            if hasattr(model, 'cpu'):
                try:
                    model.cpu()
                except Exception:
                    pass
            del model
            del tokenizer
        except Exception:
            pass
        # Multi-pass GC to break reference cycles
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        time.sleep(2)  # let memory settle

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("  MODULE 4: Strategy Profiling (Latency/VRAM/Disk)")
        logger.info("=" * 60)

        strategies = self._resolve_strategy_dirs()
        if not strategies:
            msg = f"No strategy directories found for {self.model_path}"
            logger.warning(msg)
            return {"status": "skipped", "reason": msg}

        latency_profiler = LatencyProfiler(self.config)
        vram_profiler = VRAMProfiler(self.config)
        disk_profiler = DiskProfiler(self.config)

        all_latency: Dict[str, Dict[str, Any]] = {}
        all_vram: Dict[str, Dict[str, Any]] = {}
        all_disk: Dict[str, Dict[str, Any]] = {}

        for strategy, strategy_path in strategies:
            logger.info(f"Profiling strategy: {strategy} ({strategy_path})")
            t0 = time.time()

            reset_peak_memory()
            model, tokenizer = self._load_model(strategy_path)

            try:
                latency_data = latency_profiler.profile(model, tokenizer, strategy)
                vram_data = vram_profiler.snapshot(strategy)
                disk_data = disk_profiler.measure_model_storage(model_ref=strategy_path)

                all_latency[strategy] = latency_data
                all_vram[strategy] = vram_data
                all_disk[strategy] = disk_data

                logger.info(
                    f"  ✓ {strategy} profiling completed in {time.time() - t0:.1f}s"
                )
            finally:
                self._unload_model(model, tokenizer)

        # Persist results — save JSON files first before any plotting
        results_dir = Path(self.config.output.results_dir) if self.config else Path(".")
        results_dir.mkdir(parents=True, exist_ok=True)

        import json

        latency_path = results_dir / "latency_results.json"
        with open(latency_path, "w") as fh:
            json.dump(all_latency, fh, indent=2, default=str)
        logger.info(f"Latency results: {latency_path}")

        if all_vram:
            vram_path = results_dir / "vram_results.json"
            with open(vram_path, "w") as fh:
                json.dump(all_vram, fh, indent=2, default=str)
            logger.info(f"VRAM results: {vram_path}")

        if all_disk:
            disk_path = results_dir / "disk_results.json"
            with open(disk_path, "w") as fh:
                json.dump(all_disk, fh, indent=2, default=str)
            logger.info(f"Disk results: {disk_path}")

        # Generate plots — each wrapped so one failure doesn't block the rest
        if all_latency:
            try:
                latency_profiler._plot_comparison(all_latency)
            except Exception as exc:
                logger.warning(f"Latency plot failed: {exc}")
        if all_vram:
            try:
                vram_profiler.plot_comparison(all_vram)
            except Exception as exc:
                logger.warning(f"VRAM plot failed: {exc}")
        if all_disk:
            try:
                disk_profiler.plot_comparison(all_disk)
            except Exception as exc:
                logger.warning(f"Disk plot failed: {exc}")

        return {
            "status": "success",
            "latency": all_latency,
            "vram": all_vram,
            "disk": all_disk,
        }
