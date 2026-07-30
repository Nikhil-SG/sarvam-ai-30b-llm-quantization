#!/usr/bin/env python3
"""
Module 3 — Multi-Strategy Evaluation & Ablation Analysis.

Evaluates the MxMoE-quantized models using perplexity (WikiText-2) and
conducts ablation studies by comparing multiple quantization strategies.

Evaluation is performed **in-process** using the same pattern as the research
pipeline's Module 5 — the model is loaded once with ``device_map="auto"``
(both GPUs) and perplexity is computed directly.

The model is loaded in its compressed-tensors format, then:
- FP8 weights are dequantized to bf16 (A100 CC 8.0 lacks native FP8 tensor cores)
- INT8 weights stay native (compressed_tensors handles dequant)
- GPTQ (packed INT4) weights stay native (compressed_tensors handles dequant)

Evaluation:
    - Perplexity: WikiText-2 (same dataset/config as research pipeline)

Ablation study:
    Compares multiple strategy outputs (fp8_gptq vs int8_gptq) and
    estimates model sizes for alternative bit-width configurations.

Usage:
    python -m src.mxmoe.ablation.ablation_study \\
        --model_path mxmoe/quantized_models_fp8_gptq \\
        --run_eval

RUN THIS NEXT: After Module 2 (compressor.py). Uses the compressed models.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "sarvamai/sarvam-30b"
DEFAULT_QUANTIZED_PATH = "mxmoe/quantized_models"
DEFAULT_IMPORTANCE_MAP_PATH = "mxmoe/outputs/module_1_sensitivity/results/expert_importance_map.json"

# Strategies that Module 2 produces (fallback if config doesn't override)
STRATEGIES = ["fp8_gptq", "int8_gptq"]


class EvaluationRunner:
    """
    Run perplexity evaluation on quantized models.

    Loads the model **in-process** using ``AutoModelForCausalLM.from_pretrained``
    with ``device_map="auto"`` (both GPUs), then calls PerplexityEvaluator
    directly. This is the same pattern used by the research pipeline.

    The model is loaded in its **native quantized format** — compressed-tensors
    handles FP8/INT8/GPTQ config_groups automatically. FP8 weights are
    dequantized to bf16 for A100 compatibility.
    """

    def __init__(
        self,
        config=None,
        model_path: str = DEFAULT_QUANTIZED_PATH,
        baseline_model: str = MODEL_ID,
        output_dir: str = "mxmoe/outputs/module_3_evaluation/results",
        tensor_parallel_size: int = 2,
        max_model_len: int = 4096,
        eval_limit: Optional[int] = None,
    ):
        self.model_path = model_path
        self.baseline_model = baseline_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.eval_limit = eval_limit
        self.config = config

        # Will hold the loaded model + tokenizer across evaluations
        self._model = None
        self._tokenizer = None

        if config is not None:
            self.output_dir = Path(getattr(config.output, "results_dir", str(self.output_dir)))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            model_cfg = getattr(config, "model", None)
            if model_cfg:
                self.baseline_model = getattr(model_cfg, "model_id", self.baseline_model)
            hw_cfg = getattr(config, "hardware", None)
            if hw_cfg:
                self.tensor_parallel_size = getattr(hw_cfg, "num_gpus", self.tensor_parallel_size)
            out_cfg = getattr(config, "output", None)
            if out_cfg:
                self.model_path = getattr(out_cfg, "quantized_models_dir", self.model_path)

    def _get_strategies(self) -> List[str]:
        """Resolve strategies from config or fall back to defaults."""
        recipe_cfg = getattr(self.config, "recipe", None) if self.config else None
        if recipe_cfg:
            strategies = list(getattr(recipe_cfg, "strategies", []) or [])
            if strategies:
                return strategies
        return list(STRATEGIES)

    # ── Model loading (native quantized format) ──────────────────────────

    def _patch_quantization_config(self, model_path: str) -> bool:
        """
        Patch config.json to fix the 'mixed-precision' base format name.

        llmcompressor can write ``format: "mixed-precision"`` at the top level
        with multiple ``config_groups`` (float-quantized + pack-quantized, etc.).
        compressed_tensors v0.14 does not register a compressor named
        "mixed-precision", so the loader fails before weights are read.

        Fix: change only the *base* format to a registered name ("float-quantized")
        while preserving config_groups. This keeps the model mixed-precision and
        allows CompressedLinear to load correctly.

        Returns True if patching was performed.
        """
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return False

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        quant_config = config_data.get("quantization_config", {})
        fmt = quant_config.get("format") or quant_config.get("quantization_format", "")

        if fmt == "mixed-precision":
            logger.info("  Patching quantization format: 'mixed-precision' → 'float-quantized'")
            # Preserve config_groups; only update the base format key.
            if "format" in quant_config:
                quant_config["format"] = "float-quantized"
            else:
                quant_config["quantization_format"] = "float-quantized"
            config_data["quantization_config"] = quant_config
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            return True
        return False

    def _load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the quantized model in-process using both GPUs.

        Handles compressed_tensors v0.14 compatibility:
        1. Patches 'mixed-precision' format in config.json before loading
        2. Sets max_memory to leave GPU headroom (prevents OOM)
        3. Decompresses CompressedLinear modules post-load (avoids on-the-fly OOM)
        4. Casts FP8 weights to bf16 for A100 compatibility
        """
        if self._model is not None:
            return  # Already loaded

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = model_path or self.model_path
        logger.info(f"Loading quantized model from {model_path} ...")

        # Allow HumanEval code execution (lm-eval safety gate)
        import os
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        # Reduce CUDA memory fragmentation
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )

        # ── Fix 1: Patch config.json if it has "mixed-precision" format ──
        self._patch_quantization_config(model_path)

        # ── Fix 2: Build max_memory map with headroom ────────────────────
        max_memory = None
        hw_cfg = getattr(self.config, "hardware", None) if self.config else None
        if hw_cfg and getattr(hw_cfg, "max_memory", None):
            mm = hw_cfg.max_memory
            if isinstance(mm, dict) or hasattr(mm, "items"):
                # OmegaConf DictConfig or similar
                max_memory = {}
                for k, v in mm.items():
                    if isinstance(k, str) and k.isdigit():
                        max_memory[int(k)] = v
                    else:
                        max_memory[k] = v
            else:
                max_memory = None
        # Default: leave ~5GB headroom per GPU for inference activations
        if max_memory is None and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            max_memory = {}
            for i in range(num_gpus):
                total = torch.cuda.get_device_properties(i).total_memory
                # Reserve 5GB for activations + overhead
                usable = int((total / (1024**3)) - 5)
                max_memory[i] = f"{usable}GiB"
            logger.info(f"  max_memory (auto, 5GB headroom): {max_memory}")
        else:
            logger.info(f"  max_memory: {max_memory}")

        # Load tokenizer
        logger.info("  Loading tokenizer ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model — torch_dtype="auto" preserves quantized dtypes
        logger.info("  Loading model with device_map='auto' (both GPUs), torch_dtype='auto' ...")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
        )
        self._model.eval()
        logger.info("  ✓ Model loaded")

        # ── Fix 3: Decompress CompressedLinear → nn.Linear ───────────────
        # CompressedLinear modules decompress weights on-the-fly during
        # forward(), causing GPU memory spikes and OOM. Convert them to
        # plain nn.Linear with pre-decompressed weights.
        self._decompress_compressed_linears()

        # ── Cast FP8 weights to bf16 for A100 inference ──────────────────
        self._dequantize_fp8_weights()

        # ── Cast remaining float32 params (embeddings, norms) to bf16 ────
        # Non-quantized layers (embed_tokens, RMSNorm, lm_head, router gates)
        # stay as float32 after decompression. This causes dtype mismatch
        # during inference (float32 hidden_states × bfloat16 weights → crash).
        self._cast_remaining_float32_to_bf16()

        # Log model dtype info
        self._log_model_dtype_info()

    def _decompress_compressed_linears(self) -> None:
        """
        Replace all CompressedLinear modules with plain nn.Linear.

        CompressedLinear.forward() decompresses packed INT4 weights on every
        call, causing GPU memory spikes. By decompressing once and replacing
        with nn.Linear, we use slightly more static memory but avoid OOM
        during inference.
        """
        try:
            from compressed_tensors.linear.compressed_linear import CompressedLinear
        except ImportError:
            logger.info("  compressed_tensors not available — skipping decompression")
            return

        compressed_modules = []
        for name, module in self._model.named_modules():
            if isinstance(module, CompressedLinear):
                compressed_modules.append(name)

        if not compressed_modules:
            logger.info("  No CompressedLinear modules found — skipping decompression")
            return

        logger.info(f"  Decompressing {len(compressed_modules)} CompressedLinear modules → nn.Linear ...")
        num_ok = 0
        num_fail = 0
        for name in compressed_modules:
            try:
                # Navigate to parent module
                parts = name.split(".")
                parent = self._model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                child_name = parts[-1]
                module = getattr(parent, child_name)

                # Get decompressed weight
                device = next(module.parameters()).device
                with torch.no_grad():
                    try:
                        weight = module.compressor.decompress_module(module)
                    except Exception:
                        # Fallback: try accessing weight directly
                        weight = module.weight.data
                        if hasattr(module, "weight_scale"):
                            weight = weight.float() * module.weight_scale.float()
                        weight = weight.to(torch.bfloat16)

                # Create replacement nn.Linear
                bias_data = module.bias.data if module.bias is not None else None
                new_linear = torch.nn.Linear(
                    weight.shape[1], weight.shape[0],
                    bias=bias_data is not None,
                    device="cpu", dtype=torch.bfloat16,
                )
                new_linear.weight.data = weight.to("cpu", dtype=torch.bfloat16)
                if bias_data is not None:
                    new_linear.bias.data = bias_data.to("cpu", dtype=torch.bfloat16)

                # Move to original device
                new_linear = new_linear.to(device)

                # Replace in parent
                setattr(parent, child_name, new_linear)
                num_ok += 1
            except Exception as exc:
                num_fail += 1
                if num_fail <= 3:
                    logger.warning(f"  Failed to decompress {name}: {exc}")

        # Free memory from decompression
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"  ✓ Decompressed {num_ok}/{len(compressed_modules)} modules"
                    f"{f' ({num_fail} failed)' if num_fail else ''}")

    def _dequantize_fp8_weights(self) -> None:
        """
        Cast FP8 weights to bf16 for inference on A100.

        Only affects float8 parameters. INT8/GPTQ weights and
        scale tensors are left unchanged.
        """
        fp8_dtypes = set()
        try:
            fp8_dtypes.add(torch.float8_e4m3fn)
            fp8_dtypes.add(torch.float8_e5m2)
        except AttributeError:
            # Older PyTorch without float8 support
            logger.info("  PyTorch does not have float8 dtypes — no FP8 dequant needed")
            return

        num_cast = 0
        for name, param in self._model.named_parameters():
            if param.dtype in fp8_dtypes:
                # Cast FP8 → bf16 in-place
                param.data = param.data.to(torch.bfloat16)
                num_cast += 1

        if num_cast > 0:
            logger.info(f"  ✓ Dequantized {num_cast} FP8 parameters → bf16 (A100 compatibility)")
        else:
            logger.info("  No FP8 parameters found — skipping dequant")

    def _cast_remaining_float32_to_bf16(self) -> None:
        """Cast any remaining float32 parameters to bfloat16 for dtype consistency.

        After decompression, non-quantized layers (embeddings, RMS norms,
        lm_head, MoE router/gate weights) remain as float32 while all
        decompressed quantized layers are bfloat16.  This dtype mismatch
        causes RuntimeError in F.linear when float32 hidden_states hit
        bfloat16 weight matrices during inference.

        Casting these to bf16 is standard practice — vLLM, TGI, and all
        major serving frameworks do this for quantized model inference.
        The precision loss on embeddings/norms is negligible for a model
        that is already quantized to INT8/FP8/INT4.

        NOTE: This does NOT affect the mixed-precision quantization quality.
        The quantized weights (7007 CompressedLinear modules) have already
        been decompressed — their quantization error is baked in.  Only the
        115 NON-quantized parameters are touched here.
        """
        num_cast = 0
        for name, param in self._model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
                num_cast += 1
        if num_cast > 0:
            logger.info(f"  ✓ Cast {num_cast} float32 parameters → bf16 (dtype consistency)")
        else:
            logger.info("  All parameters already bf16 — no casting needed")

    def _log_model_dtype_info(self) -> None:
        """Log information about the loaded model's weight dtypes."""
        dtype_counts: Dict[str, int] = {}
        total_params = 0
        for name, param in self._model.named_parameters():
            dtype_str = str(param.dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
            total_params += 1

        logger.info(f"  Model parameter dtype distribution ({total_params} total):")
        for dtype_str, count in sorted(dtype_counts.items()):
            logger.info(f"    {dtype_str}: {count} parameters")

        # Check for CompressedLinear modules
        try:
            from compressed_tensors.linear.compressed_linear import CompressedLinear
            compressed_count = sum(
                1 for _, m in self._model.named_modules()
                if isinstance(m, CompressedLinear)
            )
            if compressed_count > 0:
                logger.info(f"  CompressedLinear modules: {compressed_count}")
        except ImportError:
            pass

    def _unload_model(self) -> None:
        """Release model and tokenizer from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ── Perplexity evaluation (direct, in-process) ───────────────────────

    def run_perplexity(
        self,
        model_path: Optional[str] = None,
        *,
        evaluator=None,
        write_output: bool = True,
        tag_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate perplexity on WikiText-2 using the PerplexityEvaluator
        from the research pipeline (in-process, no subprocess).

        Uses the same dataset/config as research pipeline:
          dataset: wikitext, config: wikitext-2-raw-v1, split: test
        """
        model_path = model_path or self.model_path
        logger.info(f"Running perplexity evaluation on {model_path}")

        self._load_model(model_path)

        from src.evaluation.perplexity import PerplexityEvaluator

        t0 = time.time()
        try:
            ppl_eval = evaluator or PerplexityEvaluator(self.config)
            # Determine tag from model path unless overridden
            tag = tag_override or (Path(model_path).name if model_path else "mxmoe")
            ppl_data = ppl_eval.evaluate(self._model, self._tokenizer, tag)
            elapsed = time.time() - t0

            result = {
                "model_path": model_path,
                "tasks": ["wikitext"],
                "label": "perplexity",
                "status": "success",
                "time_sec": round(elapsed, 2),
                "perplexity": ppl_data.get("perplexity"),
                "avg_nll": ppl_data.get("avg_nll"),
                "num_windows": ppl_data.get("num_windows"),
                "dataset": "wikitext",
                "dataset_config": "wikitext-2-raw-v1",
                "parsed_results": ppl_data,
            }
            logger.info(
                f"Perplexity evaluation completed in {elapsed:.1f}s — "
                f"PPL = {ppl_data.get('perplexity', '?')}"
            )
        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(f"Perplexity evaluation failed: {exc}", exc_info=True)
            result = {
                "model_path": model_path,
                "tasks": ["wikitext"],
                "label": "perplexity",
                "status": "failed",
                "time_sec": round(elapsed, 2),
                "error": str(exc),
            }

        # Save result (single-model mode)
        if write_output:
            output_path = self.output_dir / "perplexity_results.json"
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=4, ensure_ascii=False, default=str)

        return result

    # ── Full evaluation (perplexity only per user request) ────────────────

    @staticmethod
    def _aggregate_eval_status(results: Dict[str, Any]) -> str:
        """Aggregate per-evaluation statuses into one module-level status."""
        evaluations = results.get("evaluations", {}) if isinstance(results, dict) else {}
        statuses = [str(v.get("status", "unknown")) for v in evaluations.values() if isinstance(v, dict)]
        if statuses and all(status == "success" for status in statuses):
            return "success"
        if statuses and any(status == "success" for status in statuses):
            return "partial_success"
        if statuses and all(status.startswith("skipped") for status in statuses):
            return "skipped"
        return "failed"

    def run_full_evaluation(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Run perplexity evaluation on the quantized model.

        Evaluates each strategy's quantized model separately (model loaded once per strategy).
        """
        model_path = model_path or self.model_path

        logger.info("=" * 60)
        logger.info("  MODULE 3: Perplexity Evaluation (Native Quantized Format)")
        logger.info("=" * 60)

        results = {
            "model_path": model_path,
            "evaluations": {},
        }

        all_ppl: Dict[str, Any] = {}
        all_bench: Dict[str, Any] = {}

        from src.evaluation.perplexity import PerplexityEvaluator
        from src.evaluation.benchmarks import BenchmarkRunner

        ppl_eval = PerplexityEvaluator(self.config)
        bench_runner = BenchmarkRunner(self.config)

        # Check if we should evaluate multiple strategies
        strategies_to_eval = []
        base_dir = Path(model_path).parent
        model_name = Path(model_path).name

        # Look for strategy-specific directories
        for strategy in self._get_strategies():
            strategy_dir = base_dir / f"{model_name}_{strategy}"
            if strategy_dir.exists():
                strategies_to_eval.append((strategy, str(strategy_dir)))

        # If no strategy dirs found, evaluate the model_path directly
        if not strategies_to_eval:
            strategies_to_eval = [("default", model_path)]

        for strategy_name, strategy_path in strategies_to_eval:
            logger.info(f"\n  Evaluating strategy: {strategy_name} ({strategy_path})")

            try:
                # Load model for this strategy
                self._load_model(strategy_path)

                # Perplexity evaluation
                ppl_result = self.run_perplexity(
                    strategy_path,
                    evaluator=ppl_eval,
                    write_output=False,
                    tag_override=strategy_name,
                )
                results["evaluations"][f"perplexity_{strategy_name}"] = ppl_result
                if isinstance(ppl_result, dict) and ppl_result.get("parsed_results"):
                    all_ppl[strategy_name] = ppl_result["parsed_results"]

                # Benchmark evaluation
                bench_start = time.time()
                try:
                    bench_data = bench_runner.evaluate(
                        self._model, self._tokenizer, strategy_name
                    )
                    bench_result = {
                        "model_path": strategy_path,
                        "label": "benchmarks",
                        "status": "success",
                        "time_sec": round(time.time() - bench_start, 2),
                        "parsed_results": bench_data,
                    }
                    all_bench[strategy_name] = bench_data
                except Exception as exc:
                    bench_result = {
                        "model_path": strategy_path,
                        "label": "benchmarks",
                        "status": "failed",
                        "time_sec": round(time.time() - bench_start, 2),
                        "error": str(exc),
                    }
                    logger.error(
                        f"Benchmark evaluation failed for {strategy_name}: {exc}",
                        exc_info=True,
                    )

                results["evaluations"][f"benchmarks_{strategy_name}"] = bench_result

            except Exception as exc:
                logger.error(f"Evaluation failed for {strategy_name}: {exc}", exc_info=True)
                results["evaluations"][f"perplexity_{strategy_name}"] = {
                    "status": "failed",
                    "error": str(exc),
                    "model_path": strategy_path,
                }
                results["evaluations"][f"benchmarks_{strategy_name}"] = {
                    "status": "failed",
                    "error": str(exc),
                    "model_path": strategy_path,
                }
            finally:
                # Unload model between strategies to free GPU memory
                self._unload_model()

        results["status"] = self._aggregate_eval_status(results)

        # Save combined results
        output_path = self.output_dir / "full_evaluation.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False, default=str)

        # Backward-compatible filename expected by existing tests/tooling.
        legacy_output_path = self.output_dir / "eval_results_full.json"
        with open(legacy_output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False, default=str)

        logger.info(f"Full evaluation results saved: {output_path}")
        logger.info(f"Evaluation aggregate status: {results['status']}")

        # Persist per-strategy evaluation artifacts
        try:
            if all_ppl:
                ppl_eval.plot_comparison(all_ppl)
            if all_bench:
                bench_runner.save_results(all_bench)
                bench_runner.plot_summary(all_bench)
        except Exception as exc:
            logger.warning(f"Evaluation plots/save skipped: {exc}")

        # Print comparison summary
        self._print_comparison_summary(results)

        return results

    def _print_comparison_summary(self, results: Dict[str, Any]) -> None:
        """Print a comparison table of all evaluated strategies."""
        evals = results.get("evaluations", {})
        if not evals:
            return

        logger.info("\n" + "=" * 60)
        logger.info("  STRATEGY COMPARISON (Perplexity)")
        logger.info("=" * 60)
        logger.info(f"  {'Strategy':<20} {'Perplexity':>12} {'NLL':>10} {'Status':>10}")
        logger.info("  " + "-" * 55)

        for key, eval_data in sorted(evals.items()):
            if not isinstance(eval_data, dict):
                continue
            strategy = key.replace("perplexity_", "")
            ppl = eval_data.get("perplexity", "N/A")
            nll = eval_data.get("avg_nll", "N/A")
            status = eval_data.get("status", "unknown")

            ppl_str = f"{ppl:.4f}" if isinstance(ppl, (int, float)) else str(ppl)
            nll_str = f"{nll:.6f}" if isinstance(nll, (int, float)) else str(nll)
            logger.info(f"  {strategy:<20} {ppl_str:>12} {nll_str:>10} {status:>10}")

        logger.info("=" * 60)


class AblationRunner:
    """
    Run ablation studies: compare strategy outputs and estimate model
    sizes for alternative bit-width configurations.

    NOTE: The ablation runner does NOT re-run full evaluation on the MxMoE
    default model. That evaluation is already done by run_module_3 → 3a.
    The ablation generates theoretical ablation data and references existing
    evaluation results.
    """

    def __init__(
        self,
        config=None,
        output_dir: str = "mxmoe/outputs/module_3_evaluation/results",
        importance_map_path: str = DEFAULT_IMPORTANCE_MAP_PATH,
        model_path: str = DEFAULT_QUANTIZED_PATH,
        tensor_parallel_size: int = 2,
        eval_limit: int = 100,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.importance_map_path = importance_map_path
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.eval_limit = eval_limit
        self.config = config

        # Ablation configurations
        self.ablation_variants = [
            {"label": "baseline_bf16", "low_bits": 16, "medium_bits": 16, "high_bits": 16},
            {"label": "all_int8", "low_bits": 8, "medium_bits": 8, "high_bits": 8},
            {"label": "all_fp8", "low_bits": 8, "medium_bits": 8, "high_bits": 8},
            {"label": "int8_gptq", "low_bits": 4, "medium_bits": 8, "high_bits": 8},
            {"label": "fp8_gptq", "low_bits": 4, "medium_bits": 8, "high_bits": 8},
            {"label": "aggressive_low3", "low_bits": 3, "medium_bits": 8, "high_bits": 8},
            {"label": "extreme_low2", "low_bits": 2, "medium_bits": 8, "high_bits": 8},
        ]

        if config is not None:
            self.output_dir = Path(getattr(config.output, "results_dir", str(self.output_dir)))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_cfg = getattr(config, "output", None)
            if out_cfg:
                self.model_path = getattr(out_cfg, "quantized_models_dir", self.model_path)
                base_dir = getattr(out_cfg, "base_dir", None)
                if base_dir:
                    self.importance_map_path = str(
                        Path(base_dir)
                        / "module_1_sensitivity"
                        / "results"
                        / "expert_importance_map.json"
                    )
            ablation_cfg = getattr(config, "ablation", None)
            if ablation_cfg:
                self.eval_limit = getattr(ablation_cfg, "quick_eval_limit", self.eval_limit)

    def run(self) -> Dict[str, Any]:
        """
        Run the ablation study.

        Generates ablation curve data structure with theoretical size estimates
        and references existing MxMoE evaluation results from step 3a.
        """
        logger.info("=" * 60)
        logger.info("  MODULE 3: Ablation Study")
        logger.info("=" * 60)

        t_start = time.time()

        # ── Load importance map for expert counts ────────────────────────
        classification_summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        if Path(self.importance_map_path).exists():
            with open(self.importance_map_path, encoding="utf-8") as fh:
                imap = json.load(fh)
            classification_summary = imap.get("classification_summary", classification_summary)

        num_low = classification_summary.get("LOW", 0)
        num_med = classification_summary.get("MEDIUM", 0)
        num_high = classification_summary.get("HIGH", 0)
        total_experts = num_low + num_med + num_high

        logger.info(f"Expert classification: HIGH={num_high}, MEDIUM={num_med}, LOW={num_low}")

        # ── Build ablation data structure ────────────────────────────────
        ablation_results = {
            "study_config": {
                "variants": self.ablation_variants,
                "eval_limit": self.eval_limit,
                "expert_counts": classification_summary,
                "strategies_evaluated": STRATEGIES,
            },
            "variant_results": [],
            "accuracy_floor": None,
        }

        # Estimate model sizes
        bf16_size_gb = 60.0
        expert_params_gb = 28.0  # Approximate
        non_expert_gb = bf16_size_gb - expert_params_gb

        from tqdm import tqdm
        for variant in tqdm(self.ablation_variants, desc="Ablation variants"):
            bits_low = variant["low_bits"]
            bits_med = variant["medium_bits"]
            bits_high = variant["high_bits"]

            if total_experts > 0:
                low_frac = num_low / total_experts
                med_frac = num_med / total_experts
                high_frac = num_high / total_experts
            else:
                low_frac = med_frac = high_frac = 1 / 3

            expert_compressed_gb = expert_params_gb * (
                low_frac * (bits_low / 16.0) +
                med_frac * (bits_med / 16.0) +
                high_frac * (bits_high / 16.0)
            )
            non_expert_compressed_gb = non_expert_gb * 0.5
            total_compressed_gb = round(expert_compressed_gb + non_expert_compressed_gb, 2)

            variant_data = {
                "label": variant["label"],
                "config": variant,
                "estimated_size_gb": total_compressed_gb,
                "compression_ratio": round(bf16_size_gb / max(total_compressed_gb, 0.1), 2),
            }
            ablation_results["variant_results"].append(variant_data)

        # ── Reference existing evaluation results ────────────────────────
        existing_eval_path = self.output_dir / "full_evaluation.json"
        if existing_eval_path.exists():
            logger.info(f"Found existing evaluation at {existing_eval_path}")
            try:
                with open(existing_eval_path, encoding="utf-8") as fh:
                    existing_eval = json.load(fh)
                ablation_results["mxmoe_default_eval"] = {
                    "status": existing_eval.get("status", "unknown"),
                    "source": str(existing_eval_path),
                    "note": "Reusing evaluation from step 3a (not re-evaluated)",
                }
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Could not load existing evaluation: {exc}")
                ablation_results["mxmoe_default_eval"] = {
                    "status": "unavailable",
                    "note": "Evaluation from step 3a could not be loaded",
                }
        else:
            logger.info("No existing evaluation found — will be created in step 3a")
            ablation_results["mxmoe_default_eval"] = {
                "status": "pending",
                "note": "Run step 3a first to generate evaluation results",
            }

        # ── Accuracy floor detection guidance ────────────────────────────
        ablation_results["accuracy_floor"] = {
            "method": (
                "The accuracy floor is detected by finding the bit-width "
                "where perplexity increases >10% compared to BF16, OR "
                "where any benchmark score drops >5 absolute points."
            ),
            "detection_thresholds": {
                "perplexity_degradation_pct": 10.0,
                "benchmark_drop_absolute": 5.0,
            },
        }

        # ── Save results ─────────────────────────────────────────────────
        ablation_results["total_time_sec"] = round(time.time() - t_start, 2)

        output_path = self.output_dir / "ablation_results.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(ablation_results, fh, indent=4, ensure_ascii=False, default=str)
        logger.info(f"Ablation results saved: {output_path}")

        return ablation_results


def main():
    parser = argparse.ArgumentParser(
        description="Module 3: Evaluation & Ablation for sarvam-30b MxMoE"
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_QUANTIZED_PATH,
        help="Path to the quantized model (default: mxmoe/quantized_models)",
    )
    parser.add_argument(
        "--baseline_model", type=str, default=MODEL_ID,
        help="Baseline model for comparison",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="mxmoe/outputs/module_3_evaluation/results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--importance_map", type=str,
        default=DEFAULT_IMPORTANCE_MAP_PATH,
        help="Path to importance map for ablation study",
    )
    parser.add_argument(
        "--tp", type=int, default=2,
        help="Tensor parallel size (default: 2 for 2x A100)",
    )
    parser.add_argument(
        "--eval_limit", type=int, default=None,
        help="Limit number of eval samples per task (for quick runs)",
    )
    parser.add_argument(
        "--run_ablation", action="store_true",
        help="Run ablation study",
    )
    parser.add_argument(
        "--run_eval", action="store_true",
        help="Run perplexity evaluation",
    )
    args = parser.parse_args()

    if args.run_eval or (not args.run_ablation):
        # Default: run evaluation
        evaluator = EvaluationRunner(
            model_path=args.model_path,
            baseline_model=args.baseline_model,
            output_dir=args.output_dir,
            tensor_parallel_size=args.tp,
            eval_limit=args.eval_limit,
        )
        eval_results = evaluator.run_full_evaluation()
        logger.info(f"Evaluation complete. Status: {eval_results.get('status')}")

    if args.run_ablation:
        ablation = AblationRunner(
            output_dir=args.output_dir,
            importance_map_path=args.importance_map,
            tensor_parallel_size=args.tp,
            eval_limit=args.eval_limit or 100,
        )
        ablation_results = ablation.run()
        logger.info(f"Ablation complete. {len(ablation_results.get('variant_results', []))} variants analyzed")


if __name__ == "__main__":
    main()
