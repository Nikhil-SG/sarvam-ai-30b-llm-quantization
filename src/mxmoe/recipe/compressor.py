"""
Module 2b — Model Compressor (Mixed-Precision MxMoE).

Two-pass quantization:
    Pass 1: GPTQ W4A16 on LOW experts (calibration on GPU) — RUN ONCE, shared
    Pass 2: FP8_DYNAMIC or W8A16 on everything else (data-free on CPU)

KEY DESIGN DECISIONS (learned from 3 failed runs):

1. GPTQ is SHARED: Both fp8_gptq and int8_gptq use identical GPTQ targets.
   Run GPTQ once → save intermediate → reuse for all strategies.

2. GPU memory CANNOT be freed in-process after llmcompressor oneshot().
   llmcompressor keeps internal session/modifier state with live GPU tensor
   references. del + gc + empty_cache only frees CACHED memory, not ALLOCATED.
   Solution: run GPTQ once, then only CPU work after that.

3. llmcompressor's data-free pipeline ALWAYS calls dispatch_model() which
   moves the CPU model to GPU. The monkey-patch must target the IMPORT SITE
   (llmcompressor.pipelines.data_free.pipeline.dispatch_model), NOT the
   source module (compressed_tensors.offload.dispatch.dispatch_model).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.auth import configure_hf_home, resolve_hf_token, resolve_model_path
from src.core.calibration import load_calibration_data
from src.core.device import build_max_memory_map
from src.core.logger import get_logger

logger = get_logger(__name__)

MODEL_ID = "sarvamai/sarvam-30b"
DEFAULT_SAVE_DIR = "mxmoe/quantized_models"
DEFAULT_CALIB_DATASET = "dataset/sangraha_verified"
DEFAULT_CALIB_SPLIT = "train"
# Shared GPTQ intermediate (identical for all strategies)
SHARED_GPTQ_INTERMEDIATE = "mxmoe/quantized_models_gptq_shared_intermediate"


class ModelCompressor:
    """
    Two-pass heterogeneous quantization.

    Architecture (fixes all 3 OOM failure modes):
        1. GPTQ Pass: runs ONCE on GPU, saves shared intermediate
        2. Data-free Pass: runs per-strategy on CPU only (monkey-patched)
    """

    def __init__(self, config=None, model_id=MODEL_ID, save_dir=DEFAULT_SAVE_DIR,
                 output_dir="mxmoe/outputs/module_2_synthesis/results",
                 calib_samples=256, seq_length=2048):
        self.config = config
        self.model_id = model_id
        self.save_dir = Path(save_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calib_samples = calib_samples
        self.seq_length = seq_length
        self.calib_dataset = DEFAULT_CALIB_DATASET
        self.calib_dataset_config = None
        self.calib_split = DEFAULT_CALIB_SPLIT
        self.calib_seed = 42

        if config is not None:
            self.output_dir = Path(getattr(config.output, "results_dir", str(self.output_dir)))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_cfg = getattr(config, "output", None)
            if out_cfg:
                self.save_dir = Path(getattr(out_cfg, "quantized_models_dir", str(self.save_dir)))
            model_cfg = getattr(config, "model", None)
            if model_cfg:
                self.model_id = getattr(model_cfg, "model_id", self.model_id)
            recipe_cfg = getattr(config, "recipe", None)
            if recipe_cfg:
                cal_cfg = getattr(recipe_cfg, "calibration", None)
                if cal_cfg:
                    self.calib_samples = getattr(cal_cfg, "num_samples", self.calib_samples)
                    self.seq_length = getattr(cal_cfg, "seq_length", self.seq_length)
                    self.calib_dataset = getattr(cal_cfg, "dataset", self.calib_dataset)
                    self.calib_dataset_config = getattr(cal_cfg, "dataset_config", self.calib_dataset_config)
                    self.calib_split = getattr(cal_cfg, "split", self.calib_split)
                    self.calib_seed = getattr(cal_cfg, "seed", self.calib_seed)

    # ─── Resolve loading params ─────────────────────────────────────────

    def _resolve_params(self):
        if self.config is not None:
            configure_hf_home(self.config)
            model_source = resolve_model_path(self.config)
            hf_token = resolve_hf_token(self.config)
            model_cache_dir = getattr(getattr(self.config, "model", None), "cache_dir", None)
            trust = getattr(getattr(self.config, "model", None), "trust_remote_code", True)
            hw = getattr(self.config, "hardware", None)
            primary = getattr(hw, "primary_cuda_index", 1) if hw else 1
            mm_cfg = getattr(hw, "max_memory", None) if hw else None
            dmap = getattr(hw, "device_map", "auto") if hw else "auto"
        else:
            model_source, hf_token, model_cache_dir = self.model_id, None, None
            trust, primary, mm_cfg, dmap = True, 1, None, "auto"

        max_memory = build_max_memory_map(
            mm_cfg._data if hasattr(mm_cfg, "_data") else mm_cfg,
            primary_cuda_index=primary,
        )
        return {"model_source": model_source, "hf_token": hf_token,
                "model_cache_dir": model_cache_dir, "trust_remote_code": trust,
                "max_memory": max_memory, "device_map": dmap}

    # ─── GPTQ Pass (GPU, calibration) — runs ONCE ──────────────────────

    def run_gptq_pass(self, gptq_targets, ignore_list, params) -> Path:
        """Run GPTQ W4A16 on LOW experts. Returns path to saved intermediate.

        If the shared intermediate already exists (from a previous strategy),
        skip GPTQ entirely and return the existing path.
        """
        intermediate_dir = Path(SHARED_GPTQ_INTERMEDIATE)

        # Check if GPTQ intermediate already exists (reuse from previous run)
        if intermediate_dir.exists() and list(intermediate_dir.glob("*.safetensors")):
            logger.info(f"  GPTQ intermediate already exists at {intermediate_dir} — reusing")
            return intermediate_dir

        logger.info("=" * 50)
        logger.info("  PASS 1: GPTQ W4A16 (LOW experts, on BF16 model)")
        logger.info("=" * 50)

        # Load model onto GPU
        logger.info(f"Loading model onto GPU: {params['model_source']}")
        model = AutoModelForCausalLM.from_pretrained(
            str(params["model_source"]), torch_dtype="auto",
            device_map=params.get("device_map", "auto"),
            max_memory=params["max_memory"], low_cpu_mem_usage=True,
            token=params["hf_token"], cache_dir=params.get("model_cache_dir"),
            trust_remote_code=params["trust_remote_code"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(params["model_source"]), token=params["hf_token"],
            cache_dir=params.get("model_cache_dir"),
            trust_remote_code=params["trust_remote_code"],
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Build calibration data
        gptq_num = min(self.calib_samples, 256)
        gptq_seq = min(self.seq_length, 1024)
        logger.info(f"  GPTQ calibration: {gptq_num} samples, seq_len={gptq_seq}")
        logger.info(f"  Calibration dataset: {self.calib_dataset}")

        calibration_data = load_calibration_data(
            tokenizer=tokenizer, dataset_name=self.calib_dataset,
            dataset_config=self.calib_dataset_config, split=self.calib_split,
            num_samples=gptq_num, seq_length=gptq_seq, seed=self.calib_seed,
            text_column="text", streaming=False, min_text_length=200,
            fallback_datasets=[{"name": "wikitext", "config": "wikitext-2-raw-v1", "split": "train"}]
            if self.calib_dataset != "wikitext" else [],
            hf_token=params["hf_token"],
        )
        logger.info(f"  Tokenized {len(calibration_data)} sequences")

        from datasets import Dataset
        calib_ds = Dataset.from_dict({
            "input_ids": [d["input_ids"].tolist() for d in calibration_data],
            "attention_mask": [d["attention_mask"].tolist() for d in calibration_data],
        })
        del calibration_data
        gc.collect()

        # Build GPTQ modifier
        from llmcompressor import oneshot
        try:
            from llmcompressor.modifiers.quantization import GPTQModifier
            modifier = GPTQModifier(targets=gptq_targets, scheme="W4A16", ignore=ignore_list)
        except ImportError:
            from llmcompressor.modifiers.quantization import QuantizationModifier
            modifier = QuantizationModifier(targets=gptq_targets, scheme="W4A16", ignore=ignore_list)

        t0 = time.time()
        oneshot(model=model, dataset=calib_ds, recipe=modifier,
                num_calibration_samples=len(calib_ds),
                max_seq_length=gptq_seq,
                trust_remote_code_model=params["trust_remote_code"])
        elapsed = time.time() - t0
        logger.info(f"  GPTQ pass completed in {elapsed:.1f}s")

        # Offload to CPU and save
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        self._offload_and_save(model, tokenizer, intermediate_dir)

        # NOTE: GPU memory is leaked by llmcompressor internals.
        # We don't try to free it — Pass 2 runs on CPU and doesn't need GPU.
        del model, tokenizer, calib_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_gpu_state("After GPTQ save")

        return intermediate_dir

    # ─── Data-free Pass (CPU only) — runs per-strategy ──────────────────

    def run_datafree_pass(self, intermediate_dir: Path, strategy_name: str,
                          fp8_targets, gptq_targets, ignore_list, params) -> Path:
        """Apply FP8 or INT8 quantization on CPU. No GPU needed."""
        if strategy_name == "int8_gptq":
            scheme, label = "W8A16", "INT8 W8A16"
        else:
            scheme, label = "FP8_DYNAMIC", "FP8_DYNAMIC"

        strategy_save_dir = self.save_dir.parent / f"{self.save_dir.name}_{strategy_name}"

        logger.info("=" * 50)
        logger.info(f"  PASS 2: {label} (data-free, CPU only)")
        logger.info("=" * 50)

        # Load model to CPU only
        logger.info(f"Loading model onto CPU only: {intermediate_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            str(intermediate_dir), torch_dtype="auto", device_map="cpu",
            low_cpu_mem_usage=True, token=params["hf_token"],
            cache_dir=params.get("model_cache_dir"),
            trust_remote_code=params["trust_remote_code"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(intermediate_dir), token=params["hf_token"],
            cache_dir=params.get("model_cache_dir"),
            trust_remote_code=params["trust_remote_code"],
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Build ignore list: base ignores + GPTQ'd LOW experts
        datafree_ignore = list(ignore_list) + list(gptq_targets)
        logger.info(f"  {label} ignore: {len(datafree_ignore)} entries")

        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        modifier = QuantizationModifier(targets="Linear", scheme=scheme, ignore=datafree_ignore)

        strategy_save_dir.mkdir(parents=True, exist_ok=True)

        # CRITICAL: Monkey-patch dispatch_model at the IMPORT SITE.
        # llmcompressor/pipelines/data_free/pipeline.py does:
        #     from compressed_tensors.offload import dispatch_model
        # This creates a LOCAL reference. Patching the source module
        # (compressed_tensors.offload.dispatch) does NOT update this reference.
        # We must patch the name IN the pipeline module itself.
        patched_modules = []
        try:
            import llmcompressor.pipelines.data_free.pipeline as _dfp
            if hasattr(_dfp, "dispatch_model"):
                _original_dfp = _dfp.dispatch_model
                _dfp.dispatch_model = lambda model, **kw: logger.info(
                    "    [patch] dispatch_model → no-op (CPU)") or model
                patched_modules.append(("_dfp", _dfp, "dispatch_model", _original_dfp))
                logger.info("  ✓ Patched dispatch_model in data_free pipeline")
        except ImportError:
            logger.warning("  Could not import data_free pipeline for patching")

        # Also patch the source module as fallback
        try:
            import compressed_tensors.offload.dispatch as _ctd
            _original_ctd = _ctd.dispatch_model
            _ctd.dispatch_model = lambda model, **kw: model
            patched_modules.append(("_ctd", _ctd, "dispatch_model", _original_ctd))
        except ImportError:
            pass

        t0 = time.time()
        try:
            oneshot(model=model, recipe=modifier,
                    trust_remote_code_model=params["trust_remote_code"])
        finally:
            # Restore all patches
            for name, mod, attr, original in patched_modules:
                setattr(mod, attr, original)

        elapsed = time.time() - t0
        logger.info(f"  {label} pass completed in {elapsed:.1f}s")

        # Save from CPU
        logger.info(f"  Saving model to {strategy_save_dir}...")
        model.save_pretrained(strategy_save_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(strategy_save_dir)
        logger.info(f"  ✓ Model saved to {strategy_save_dir}")

        del model, tokenizer
        gc.collect()

        return strategy_save_dir

    # ─── Main entry point ───────────────────────────────────────────────

    def compress(self, recipe=None, recipe_path=None,
                 strategy_name="fp8_gptq") -> Dict[str, Any]:
        """Compress using the specified strategy."""
        if strategy_name == "int8_gptq":
            datafree_label = "INT8 W8A16"
        else:
            datafree_label = "FP8_DYNAMIC"

        logger.info("=" * 60)
        logger.info(f"  MODULE 2b: Model Compression ({strategy_name})")
        logger.info(f"    GPTQ W4A16 (LOW) + {datafree_label} (all others)")
        logger.info("=" * 60)

        t_start = time.time()

        # Load recipe
        if recipe is not None:
            recipe_dict = recipe.to_dict()
        elif recipe_path is not None:
            with open(recipe_path, encoding="utf-8") as fh:
                recipe_dict = json.load(fh)
        else:
            default = "mxmoe/outputs/module_2_synthesis/results/precision_recipe.json"
            if not Path(default).exists():
                raise FileNotFoundError("No recipe found. Run Module 2a first.")
            with open(default, encoding="utf-8") as fh:
                recipe_dict = json.load(fh)

        # Select data-free targets based on strategy
        if strategy_name == "int8_gptq" and recipe_dict.get("int8_targets"):
            datafree_targets = recipe_dict["int8_targets"]
        else:
            datafree_targets = recipe_dict["fp8_targets"]

        gptq_targets = recipe_dict["gptq_targets"]
        ignore_list = recipe_dict["ignore_list"]

        # Merge legacy fields (backward compat with old recipe files)
        gptq_targets = gptq_targets + recipe_dict.get("gptq_low_targets", [])
        fp8_targets = datafree_targets

        has_gptq = len(gptq_targets) > 0
        has_datafree = len(fp8_targets) > 0

        logger.info(f"  {datafree_label} targets: {len(fp8_targets)} modules")
        logger.info(f"  GPTQ targets: {len(gptq_targets)} modules")
        logger.info(f"  Ignore list:  {ignore_list}")

        params = self._resolve_params()
        strategy_save_dir = self.save_dir.parent / f"{self.save_dir.name}_{strategy_name}"

        result = {"model_id": self.model_id, "save_dir": str(strategy_save_dir),
                  "strategy": strategy_name, "passes": []}

        # ── Pass 1: GPTQ (shared, reused across strategies) ─────────────
        if has_gptq:
            t1 = time.time()
            intermediate = self.run_gptq_pass(gptq_targets, ignore_list, params)
            result["passes"].append({
                "type": "GPTQ_W4A16", "num_targets": len(gptq_targets),
                "calibration_dataset": self.calib_dataset,
                "time_sec": round(time.time() - t1, 2),
                "intermediate_dir": str(intermediate),
            })
        else:
            intermediate = Path(str(params["model_source"]))

        # ── Pass 2: Data-free (per-strategy, CPU only) ──────────────────
        if has_datafree:
            t2 = time.time()
            final_dir = self.run_datafree_pass(
                intermediate, strategy_name, fp8_targets,
                gptq_targets, ignore_list, params)
            result["passes"].append({
                "type": datafree_label, "scheme": "FP8_DYNAMIC" if "fp8" in strategy_name else "W8A16",
                "time_sec": round(time.time() - t2, 2),
            })
        elif has_gptq:
            # GPTQ-only: copy intermediate to final
            import shutil
            strategy_save_dir.mkdir(parents=True, exist_ok=True)
            for f in intermediate.iterdir():
                shutil.copy2(f, strategy_save_dir / f.name)
            final_dir = strategy_save_dir

        # Measure size
        total_size = sum(f.stat().st_size for f in strategy_save_dir.rglob("*") if f.is_file())
        result["compressed_model_size_gb"] = round(total_size / (1024**3), 2)
        result["total_time_sec"] = round(time.time() - t_start, 2)

        logger.info(f"Compressed model ({strategy_name}): {result['compressed_model_size_gb']:.2f} GB")

        # Save report
        report = self.output_dir / f"compression_report_{strategy_name}.json"
        with open(report, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=4, default=str)
        logger.info(f"Report: {report}")

        return result

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _offload_and_save(self, model, tokenizer, save_dir):
        """Move model to CPU and save. Bypasses from_accelerate OOM."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Offloading model to CPU for saving to {save_dir}...")
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)
        except Exception:
            pass

        for param in model.parameters():
            param.data = param.data.to("cpu", non_blocking=False)
        for buf in model.buffers():
            buf.data = buf.data.to("cpu", non_blocking=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info("  Saving model from CPU...")
        model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(save_dir)
        logger.info(f"  ✓ Model saved to {save_dir}")

    @staticmethod
    def _log_gpu_state(label=""):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used = (total - free) / (1024**3)
                logger.info(f"  [{label}] GPU {i}: {used:.1f}/{total/(1024**3):.1f} GiB used")


def main():
    parser = argparse.ArgumentParser(description="Module 2b: Model Compression")
    parser.add_argument("--recipe_path", type=str,
                        default="mxmoe/outputs/module_2_synthesis/results/precision_recipe.json")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--output_dir", type=str, default="mxmoe/outputs/module_2_synthesis/results")
    parser.add_argument("--calib_samples", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--strategy", type=str, default="fp8_gptq",
                        choices=["fp8_gptq", "int8_gptq"])
    args = parser.parse_args()

    compressor = ModelCompressor(save_dir=args.save_dir, output_dir=args.output_dir,
                                 calib_samples=args.calib_samples, seq_length=args.seq_length)
    result = compressor.compress(recipe_path=args.recipe_path, strategy_name=args.strategy)
    logger.info(f"Done: {result.get('total_time_sec', 0):.1f}s")


if __name__ == "__main__":
    main()
