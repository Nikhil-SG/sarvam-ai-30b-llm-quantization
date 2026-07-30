#!/usr/bin/env python3
"""
Module 4a — vLLM Inference Profiler.

Measures tokens/sec, time-to-first-token (TTFT), total throughput, and peak
VRAM usage for the MxMoE-quantized model using the vLLM engine.

Benchmarks at batch_size=1 (latency) and batch_size=32 (throughput) on
2× A100 80GB with tensor_parallel_size=2.

Usage:
    python -m src.mxmoe.deployment.vllm_profiler \\
        --model_path mxmoe/quantized_models \\
        --batch_sizes 1 4 8 16 32

RUN THIS NEXT: After Module 2 or Module 3. Produces profiling JSON + tables.
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

from src.core.logger import get_logger

logger = get_logger(__name__)

# ── Monkeypatch ModelCompressor for older compressed-tensors compatibility ──
try:
    from compressed_tensors import ModelCompressor
    if not hasattr(ModelCompressor, "compress_model") and hasattr(ModelCompressor, "compress"):
        ModelCompressor.compress_model = lambda self, model, *args, **kwargs: self.compress(model, *args, **kwargs)
except ImportError:
    pass

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "sarvamai/sarvam-30b"
DEFAULT_QUANTIZED_PATH = "mxmoe/quantized_models"

PROMPT_TEMPLATE = (
    "The future of artificial intelligence in healthcare is transforming "
    "how we approach diagnosis and treatment. In the coming years, we expect"
)


class VLLMProfiler:
    """Profile inference latency and throughput using vLLM engine."""

    def __init__(
        self,
        config=None,
        model_path: str = DEFAULT_QUANTIZED_PATH,
        output_dir: str = "mxmoe/outputs/module_4_deployment/results",
        tensor_parallel_size: int = 2,
        max_model_len: int = 4096,
        batch_sizes: Optional[List[int]] = None,
        warmup_steps: int = 2,
        num_runs: int = 3,
        max_new_tokens: int = 128,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.batch_sizes = batch_sizes or [1, 4, 8, 16, 32]
        self.warmup_steps = warmup_steps
        self.num_runs = num_runs
        self.max_new_tokens = max_new_tokens

        if config is not None:
            self.output_dir = Path(getattr(config.output, "results_dir", str(self.output_dir)))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            deploy_cfg = getattr(config, "deployment", None)
            if deploy_cfg:
                vllm_cfg = getattr(deploy_cfg, "vllm", None)
                if vllm_cfg:
                    self.tensor_parallel_size = getattr(vllm_cfg, "tensor_parallel_size", self.tensor_parallel_size)
                    self.max_model_len = getattr(vllm_cfg, "max_model_len", self.max_model_len)
                    self.batch_sizes = list(getattr(vllm_cfg, "batch_sizes", self.batch_sizes))
                    self.warmup_steps = getattr(vllm_cfg, "warmup_steps", self.warmup_steps)
                    self.num_runs = getattr(vllm_cfg, "num_runs", self.num_runs)
                    self.max_new_tokens = getattr(vllm_cfg, "max_new_tokens", self.max_new_tokens)
            out_cfg = getattr(config, "output", None)
            if out_cfg:
                self.model_path = getattr(out_cfg, "quantized_models_dir", self.model_path)

    def run(self) -> Dict[str, Any]:
        """
        Run inference profiling at all configured batch sizes.

        Returns:
            Dict with per-batch-size latency, throughput, TTFT, and VRAM.
        """
        logger.info("=" * 60)
        logger.info("  MODULE 4a: vLLM Inference Profiling")
        logger.info("=" * 60)

        t_start = time.time()

        # ── Import vLLM ──────────────────────────────────────────────────
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            logger.error(
                "vLLM not installed. Use the mxmoe_vllm_env:\n"
                "  pip install vllm==0.6.6 --extra-index-url "
                "https://download.pytorch.org/whl/cu121"
            )
            return {"status": "vllm_not_installed"}

        # ── Register SarvamMoEForCausalLM dynamically if needed ──────────
        try:
            try:
                from vllm.model_executor.models import ModelRegistry
            except ImportError:
                try:
                    from vllm.model_executor.models.registry import ModelRegistry
                except ImportError:
                    from vllm import ModelRegistry

            is_registered = False
            if hasattr(ModelRegistry, "get_supported_models"):
                is_registered = "SarvamMoEForCausalLM" in ModelRegistry.get_supported_models()
            elif hasattr(ModelRegistry, "_MODEL_REGISTRY"):
                is_registered = "SarvamMoEForCausalLM" in ModelRegistry._MODEL_REGISTRY

            if not is_registered:
                logger.info("SarvamMoEForCausalLM is not registered in vLLM. Registering dynamically...")
                from vllm.model_executor.models.bailing_moe import BailingMoeForCausalLM
                from typing import Iterable

                class SarvamMoEForCausalLM(BailingMoeForCausalLM):
                    """Same as BailingMoeForCausalLM, but normalizes gate expert_bias pre-load."""

                    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
                        def _is_gate_expert_bias_name(name: str) -> bool:
                            return name.endswith(".mlp.gate.e_score_correction_bias") or name.endswith(
                                ".gate.e_score_correction_bias"
                            )

                        def _zero_mean_tensor(t: torch.Tensor) -> torch.Tensor:
                            if t.numel() == 0:
                                return t
                            return t - t.mean()

                        def _normalized_weights(w_iterable):
                            for name, w in w_iterable:
                                if _is_gate_expert_bias_name(name):
                                    yield name, _zero_mean_tensor(w)
                                else:
                                    yield name, w

                        return super().load_weights(_normalized_weights(weights))

                ModelRegistry.register_model("SarvamMoEForCausalLM", SarvamMoEForCausalLM)
                logger.info("Successfully registered SarvamMoEForCausalLM dynamically in vLLM registry.")
            else:
                logger.info("SarvamMoEForCausalLM is already registered in vLLM.")
        except Exception as e:
            logger.warning(f"Could not register SarvamMoEForCausalLM dynamically: {e}")

        # Resolve strategies
        base_dir = Path(self.model_path).parent
        model_name = Path(self.model_path).name
        
        # Check strategy directories
        strategies = ["fp8_gptq", "int8_gptq"]
        resolved_strategies = []
        for s in strategies:
            s_dir = base_dir / f"{model_name}_{s}"
            if s_dir.exists():
                resolved_strategies.append((s, s_dir))
                
        if not resolved_strategies:
            if Path(self.model_path).exists():
                resolved_strategies.append(("default", Path(self.model_path)))
            else:
                logger.error(f"Quantized model not found: {self.model_path}")
                logger.info("Run Module 2 first to generate the compressed model")
                return {"status": "model_not_found", "model_path": self.model_path}

        overall_results = {}

        for strategy_name, strategy_path in resolved_strategies:
            logger.info(f"\nProfiling strategy under vLLM: {strategy_name} ({strategy_path})")

            # Patch quantization config for older compressed-tensors (vLLM 0.6.6)
            self._patch_quantization_config(strategy_path)

            # ── Initialize vLLM engine ───────────────────────────────────────
            logger.info(f"Loading model with vLLM: {strategy_path}")
            logger.info(f"  tensor_parallel={self.tensor_parallel_size}, "
                         f"max_model_len={self.max_model_len}")

            try:
                llm = LLM(
                    model=str(strategy_path),
                    tensor_parallel_size=self.tensor_parallel_size,
                    max_model_len=self.max_model_len,
                    trust_remote_code=True,
                    dtype="auto",
                    gpu_memory_utilization=0.55,
                    quantization="compressed-tensors",
                    enforce_eager=True,
                )
            except Exception as init_err:
                logger.error(f"vLLM engine init failed for {strategy_name}: {init_err}")
                overall_results[strategy_name] = {
                    "model_path": str(strategy_path),
                    "strategy": strategy_name,
                    "error": str(init_err),
                    "profile_results": {},
                }
                continue

            sampling_params = SamplingParams(
                temperature=0.0,  # Greedy for reproducibility
                max_tokens=self.max_new_tokens,
            )

            # ── Record VRAM usage ────────────────────────────────────────────
            vram_usage = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    vram_usage[f"gpu_{i}"] = {
                        "allocated_gb": round(allocated, 2),
                        "reserved_gb": round(reserved, 2),
                    }

            # ── Profile at each batch size ───────────────────────────────────
            profile_results = {}

            for batch_size in self.batch_sizes:
                logger.info(f"    Profiling batch_size={batch_size}")

                # Build batch of prompts
                prompts = [PROMPT_TEMPLATE] * batch_size

                # Warmup
                logger.info(f"      Warmup: {self.warmup_steps} runs")
                for _ in range(self.warmup_steps):
                    _ = llm.generate(prompts, sampling_params)

                # Timed runs
                latencies = []
                ttft_samples = []
                total_tokens_generated = 0

                for run_idx in range(self.num_runs):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    outputs = llm.generate(prompts, sampling_params)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0

                    latencies.append(elapsed)

                    # Count generated tokens
                    run_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                    total_tokens_generated += run_tokens

                    # Extract TTFT from vLLM metrics when available
                    try:
                        for o in outputs:
                            m = getattr(o, "metrics", None)
                            if m and hasattr(m, "first_token_time"):
                                ttft = m.first_token_time - m.arrival_time
                                ttft_samples.append(ttft)
                    except Exception:
                        pass

                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                avg_tokens_per_run = total_tokens_generated / self.num_runs
                tokens_per_sec = avg_tokens_per_run / avg_latency

                result_entry = {
                    "batch_size": batch_size,
                    "avg_latency_sec": round(avg_latency, 4),
                    "min_latency_sec": round(min_latency, 4),
                    "max_latency_sec": round(max_latency, 4),
                    "tokens_per_sec": round(tokens_per_sec, 2),
                    "avg_tokens_generated": round(avg_tokens_per_run, 1),
                    "time_per_token_ms": round(1000.0 / max(tokens_per_sec, 0.01), 2),
                    "num_runs": self.num_runs,
                }

                if ttft_samples:
                    avg_ttft = sum(ttft_samples) / len(ttft_samples)
                    result_entry["avg_ttft_sec"] = round(avg_ttft, 4)

                profile_results[str(batch_size)] = result_entry

                logger.info(f"      batch={batch_size}: {tokens_per_sec:.1f} tok/s, "
                             f"latency={avg_latency:.3f}s")

            strategy_results = {
                "model_path": str(strategy_path),
                "strategy": strategy_name,
                "vllm_config": {
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "max_model_len": self.max_model_len,
                    "max_new_tokens": self.max_new_tokens,
                },
                "vram_usage": vram_usage,
                "profile_results": profile_results,
            }

            # Save strategy-specific results
            output_path = self.output_dir / f"vllm_profiling_{strategy_name}.json"
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(strategy_results, fh, indent=4, ensure_ascii=False, default=str)
            logger.info(f"    Strategy results saved: {output_path}")

            overall_results[strategy_name] = strategy_results

            # ── Print summary table for this strategy ────────────────────────
            logger.info(f"\n    === vLLM Profiling Results ({strategy_name}) ===")
            logger.info(f"    {'Batch':>5}  {'Tok/s':>10}  {'Latency':>10}  {'ms/tok':>8}")
            logger.info(f"    {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}")
            for bs, data in profile_results.items():
                logger.info(
                    f"    {data['batch_size']:>5}  "
                    f"{data['tokens_per_sec']:>10.1f}  "
                    f"{data['avg_latency_sec']:>10.3f}s  "
                    f"{data['time_per_token_ms']:>8.1f}"
                )

            # Cleanup vLLM engine before running the next strategy
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)

        # Guard: if all strategies failed, write error results and return
        if not overall_results:
            error_result = {
                "status": "all_strategies_failed",
                "error": "All vLLM engine initializations failed",
                "profile_results": {},
                "all_strategies": {},
                "total_time_sec": round(time.time() - t_start, 2),
            }
            output_path = self.output_dir / "vllm_profiling.json"
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(error_result, fh, indent=4, ensure_ascii=False, default=str)
            legacy_results = dict(error_result)
            legacy_results["metrics"] = {}
            legacy_output_path = self.output_dir / "vllm_profiling_results.json"
            with open(legacy_output_path, "w", encoding="utf-8") as fh:
                json.dump(legacy_results, fh, indent=4, ensure_ascii=False, default=str)
            logger.error("All vLLM strategies failed. Error results saved.")
            return error_result

        # Determine default strategy results to expose to model_card and tests
        # We prefer int8_gptq if available, then fp8_gptq, then the first resolved strategy
        default_strategy = "int8_gptq" if "int8_gptq" in overall_results else (
            "fp8_gptq" if "fp8_gptq" in overall_results else list(overall_results.keys())[0]
        )
        
        default_results = dict(overall_results[default_strategy])
        default_results["all_strategies"] = overall_results
        default_results["total_time_sec"] = round(time.time() - t_start, 2)

        # Save unified vllm_profiling.json
        output_path = self.output_dir / "vllm_profiling.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(default_results, fh, indent=4, ensure_ascii=False, default=str)

        # Backward-compatible artifact for existing tests/tooling
        legacy_results = dict(default_results)
        legacy_results["metrics"] = default_results.get("profile_results", {})
        legacy_output_path = self.output_dir / "vllm_profiling_results.json"
        with open(legacy_output_path, "w", encoding="utf-8") as fh:
            json.dump(legacy_results, fh, indent=4, ensure_ascii=False, default=str)

        logger.info(f"\nProfiling results saved: {output_path}")
        return default_results

    @staticmethod
    def _patch_quantization_config(model_path: str) -> None:
        """Ensure quantization_config is compatible with older compressed-tensors.

        vLLM 0.6.6 ships compressed-tensors 0.8.x which may not understand
        multi-group configs written by compressed-tensors >=0.14.x.  This
        helper reads config.json and, if it contains a
        ``quantization_config`` with ``quant_method: compressed-tensors``,
        ensures the structure is loadable.
        """
        import json as _json
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                model_config = _json.load(fh)
        except Exception:
            return

        qc = model_config.get("quantization_config", {})
        if qc.get("quant_method") != "compressed-tensors":
            return

        # If config_groups exists and is a dict, the model was written by
        # a newer compressed-tensors.  Check that format_version is set;
        # older loaders ignore it gracefully.
        config_groups = qc.get("config_groups")
        if isinstance(config_groups, dict) and "format_version" not in qc:
            logger.info("  Patching quantization_config: adding format_version for compat")
            qc["format_version"] = "1.0"
            model_config["quantization_config"] = qc

            backup_path = config_path.with_suffix(".json.bak")
            if not backup_path.exists():
                import shutil
                shutil.copy2(config_path, backup_path)

            with open(config_path, "w", encoding="utf-8") as fh:
                _json.dump(model_config, fh, indent=2, ensure_ascii=False)
            logger.info(f"  Config patched: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Module 4a: vLLM Inference Profiling for sarvam-30b MxMoE"
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_QUANTIZED_PATH,
        help="Path to the quantized model",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="mxmoe/outputs/module_4_deployment/results",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--tp", type=int, default=2,
        help="Tensor parallel size (default: 2)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=4096,
        help="Maximum model length (default: 4096)",
    )
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32],
        help="Batch sizes to profile (default: 1 4 8 16 32)",
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Number of warmup runs per batch size (default: 2)",
    )
    parser.add_argument(
        "--num_runs", type=int, default=3,
        help="Number of timed runs per batch size (default: 3)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="Max new tokens to generate (default: 128)",
    )
    args = parser.parse_args()

    profiler = VLLMProfiler(
        model_path=args.model_path,
        output_dir=args.output_dir,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        batch_sizes=args.batch_sizes,
        warmup_steps=args.warmup,
        num_runs=args.num_runs,
        max_new_tokens=args.max_new_tokens,
    )
    results = profiler.run()

    if results.get("status") not in ("model_not_found", "vllm_not_installed"):
        logger.info(f"Profiling complete in {results.get('total_time_sec', 0):.1f}s")


if __name__ == "__main__":
    main()
