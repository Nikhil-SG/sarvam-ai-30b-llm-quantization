#!/usr/bin/env python3
"""
=============================================================================
vLLM Profiling with FusedMoE Weight Scale Patch
=============================================================================

This script fixes the vLLM 0.19.0 FusedMoE weight scale shape mismatch 
that prevents loading mixed-precision compressed-tensors MoE models.

THE BUG:
    RuntimeError: output with shape [4096, 1] doesn't match 
    the broadcast shape [4096, 8]
    
    at vllm/model_executor/layers/fused_moe/layer.py:863 
    in _load_per_channel_weight_scale

ROOT CAUSE:
    The MxMoE models use per-channel weight scales (shape [N, 1]) for the 
    8-bit quantized layers, but vLLM 0.19.0's FusedMoE Marlin kernel expects 
    per-group weight scales (shape [N, G] where G = hidden_size / group_size).
    
    When expert_data has shape [N, G] and loaded_weight has shape [N, 1],
    the copy_() call fails because PyTorch cannot broadcast [N, 1] into [N, G]
    via copy_ (copy_ requires exact shape match, unlike regular operations).

THE FIX:
    Monkeypatch _load_per_channel_weight_scale to detect the shape mismatch
    and expand the [N, 1] tensor to [N, G] before copying.

USAGE (on GPU server with 2x A100):
    python vllm_profiling_patched.py
    
    Or in JupyterLab:
    %run vllm_profiling_patched.py
    
    Or copy-paste cells into a notebook.

PREREQUISITES:
    Activate the mxmoe_vllm_env:
        conda activate mxmoe_vllm_env
        # or
        source /path/to/mxmoe_vllm_env/bin/activate
=============================================================================
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List



import torch

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these paths to match your GPU server layout
# ═══════════════════════════════════════════════════════════════════════════════

# Base project directory on the GPU server
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# Quantized model directories (relative to PROJECT_DIR)
MODEL_PATHS = {
    "int8_gptq": PROJECT_DIR / "mxmoe" / "quantized_models_int8_gptq",
    "fp8_gptq": PROJECT_DIR / "mxmoe" / "quantized_models_fp8_gptq",
}

# Output directory for results
OUTPUT_DIR = PROJECT_DIR / "mxmoe" / "outputs" / "module_4_deployment" / "results"

# vLLM engine settings
TENSOR_PARALLEL_SIZE = 2       # 2x A100
MAX_MODEL_LEN = 4096           # Context window
GPU_MEMORY_UTILIZATION = 0.55  # Conservative to avoid OOM
ENFORCE_EAGER = True           # Disable CUDA graphs for stability

# Profiling settings
BATCH_SIZES = [1, 4, 8, 16, 32]
WARMUP_STEPS = 2
NUM_RUNS = 3
MAX_NEW_TOKENS = 128

# Test prompt
PROMPT_TEMPLATE = (
    "The future of artificial intelligence in healthcare is transforming "
    "how we approach diagnosis and treatment. In the coming years, we expect"
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Apply the FusedMoE Weight Scale Monkeypatch
# ═══════════════════════════════════════════════════════════════════════════════

def apply_fusedmoe_patch():
    """
    Monkeypatch vLLM's FusedMoE._load_per_channel_weight_scale to handle
    the [N, 1] → [N, G] shape mismatch for per-channel quantized models.
    
    This patch intercepts the weight loading call, detects when the checkpoint
    provides per-channel scales (shape [N, 1]) but the FusedMoE kernel expects
    per-group scales (shape [N, G]), and broadcasts accordingly.
    
    The fix is mathematically correct because per-channel quantization is a 
    special case of per-group quantization where group_size = entire_row.
    Repeating the single scale across all groups is equivalent.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    
    # Store reference to the original method
    _original_load_per_channel = FusedMoE._load_per_channel_weight_scale
    
    def _patched_load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        """
        Patched version that handles per-channel [N,1] → per-group [N,G] broadcast.
        
        When the checkpoint has per-channel scales (shape [N, 1]) but the 
        FusedMoE kernel allocated per-group buffers (shape [N, G]):
        
        1. For w2: expert_data is the full target buffer. If shapes mismatch,
           we expand loaded_weight to match before copy.
        2. For w1/w3: delegates to _load_w13, but we still need to fix the 
           loaded_weight shape before it gets sliced.
        """
        if expert_data.shape != loaded_weight.shape:
            expert_ndim = expert_data.dim()
            loaded_ndim = loaded_weight.dim()
            
            # Only patch when dimensions match but sizes differ
            if expert_ndim == loaded_ndim and expert_ndim >= 2:
                # Case A: checkpoint [N, 1] → kernel expects [N, G]
                # Per-channel scale needs to be broadcast across groups
                if loaded_weight.shape[-1] == 1 and expert_data.shape[-1] > 1:
                    print(f"  [PATCH] Broadcasting weight scale for {shard_id}: "
                          f"{list(loaded_weight.shape)} → {list(expert_data.shape)}")
                    loaded_weight = loaded_weight.expand_as(expert_data).contiguous()
                    
                # Case B: checkpoint [N, G] → kernel expects [N, 1] (unlikely but safe)
                elif loaded_weight.shape[-1] > 1 and expert_data.shape[-1] == 1:
                    print(f"  [PATCH] Reducing weight scale for {shard_id}: "
                          f"{list(loaded_weight.shape)} → {list(expert_data.shape)}")
                    loaded_weight = loaded_weight.mean(dim=-1, keepdim=True)
        
        # Call the original method with the (potentially reshaped) tensor
        return _original_load_per_channel(
            self, expert_data, shard_dim, shard_id, loaded_weight, tp_rank
        )
    
    # Inject the patch
    FusedMoE._load_per_channel_weight_scale = _patched_load_per_channel_weight_scale
    print("✅ FusedMoE._load_per_channel_weight_scale monkeypatch applied successfully")
    print("   [N, 1] → [N, G] broadcast will be handled automatically during weight loading")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Register SarvamMoEForCausalLM in vLLM's Model Registry
# ═══════════════════════════════════════════════════════════════════════════════

def register_sarvam_model():
    """
    Dynamically register SarvamMoEForCausalLM in vLLM's ModelRegistry.
    
    sarvam-30b uses a custom architecture (SarvamMoEForCausalLM) that is 
    compatible with BailingMoeForCausalLM but requires gate bias normalization
    (zero-mean correction on e_score_correction_bias).
    """
    try:
        try:
            from vllm.model_executor.models import ModelRegistry
        except ImportError:
            try:
                from vllm.model_executor.models.registry import ModelRegistry
            except ImportError:
                from vllm import ModelRegistry

        # Check if already registered
        is_registered = False
        if hasattr(ModelRegistry, "get_supported_models"):
            is_registered = "SarvamMoEForCausalLM" in ModelRegistry.get_supported_models()
        elif hasattr(ModelRegistry, "_MODEL_REGISTRY"):
            is_registered = "SarvamMoEForCausalLM" in ModelRegistry._MODEL_REGISTRY

        if not is_registered:
            print("Registering SarvamMoEForCausalLM in vLLM...")
            from vllm.model_executor.models.bailing_moe import BailingMoeForCausalLM
            from typing import Iterable

            class SarvamMoEForCausalLM(BailingMoeForCausalLM):
                """BailingMoeForCausalLM with gate expert_bias zero-mean normalization."""

                def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
                    def _is_gate_expert_bias_name(name: str) -> bool:
                        return name.endswith(".mlp.gate.e_score_correction_bias") or \
                               name.endswith(".gate.e_score_correction_bias")

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
            print("✅ SarvamMoEForCausalLM registered successfully")
        else:
            print("✅ SarvamMoEForCausalLM already registered")
            
    except Exception as e:
        print(f"⚠️  Could not register SarvamMoEForCausalLM: {e}")
        print("   The model may still load with --trust-remote-code")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Patch quantization_config for compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def patch_quantization_config(model_path: Path) -> None:
    """
    Ensure quantization_config in config.json is compatible with the 
    compressed-tensors version bundled with vLLM.
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"⚠️  config.json not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as fh:
        model_config = json.load(fh)

    qc = model_config.get("quantization_config", {})
    if qc.get("quant_method") != "compressed-tensors":
        return

    config_groups = qc.get("config_groups")
    if isinstance(config_groups, dict) and "format_version" not in qc:
        print("  Patching quantization_config: adding format_version for compat")
        qc["format_version"] = "1.0"
        model_config["quantization_config"] = qc

        # Backup original config
        backup_path = config_path.with_suffix(".json.bak")
        if not backup_path.exists():
            import shutil
            shutil.copy2(config_path, backup_path)

        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(model_config, fh, indent=2, ensure_ascii=False)
        print(f"  Config patched: {config_path}")
    else:
        print(f"  Config already compatible: {config_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Profile a single strategy
# ═══════════════════════════════════════════════════════════════════════════════

def profile_strategy(
    strategy_name: str,
    model_path: Path,
    LLM,
    SamplingParams,
) -> Dict[str, Any]:
    """Run inference profiling for a single quantized model variant."""
    
    print(f"\n{'='*60}")
    print(f"  Profiling: {strategy_name}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")
    
    # Patch config if needed
    patch_quantization_config(model_path)
    
    # Initialize vLLM engine
    print(f"\nLoading model with vLLM...")
    print(f"  tensor_parallel={TENSOR_PARALLEL_SIZE}")
    print(f"  max_model_len={MAX_MODEL_LEN}")
    print(f"  gpu_memory_utilization={GPU_MEMORY_UTILIZATION}")
    print(f"  enforce_eager={ENFORCE_EAGER}")
    
    try:
        llm = LLM(
            model=str(model_path),
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=ENFORCE_EAGER,
        )
        print("✅ vLLM engine initialized successfully!")
    except Exception as e:
        print(f"❌ vLLM engine init failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model_path": str(model_path),
            "strategy": strategy_name,
            "error": str(e),
            "profile_results": {},
        }
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for reproducibility
        max_tokens=MAX_NEW_TOKENS,
    )
    
    # Record VRAM usage after loading
    vram_usage = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            vram_usage[f"gpu_{i}"] = {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
            }
        print(f"\nVRAM after loading:")
        for gpu, usage in vram_usage.items():
            print(f"  {gpu}: {usage['allocated_gb']:.2f} GB allocated, "
                  f"{usage['reserved_gb']:.2f} GB reserved")
    
    # Profile at each batch size
    profile_results = {}
    
    for batch_size in BATCH_SIZES:
        print(f"\n  Profiling batch_size={batch_size}")
        prompts = [PROMPT_TEMPLATE] * batch_size
        
        # Warmup
        print(f"    Warmup: {WARMUP_STEPS} runs")
        for _ in range(WARMUP_STEPS):
            _ = llm.generate(prompts, sampling_params)
        
        # Timed runs
        latencies = []
        ttft_samples = []
        total_tokens_generated = 0
        
        for run_idx in range(NUM_RUNS):
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
            
            # Extract TTFT from vLLM metrics
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
        avg_tokens_per_run = total_tokens_generated / NUM_RUNS
        tokens_per_sec = avg_tokens_per_run / avg_latency
        
        result_entry = {
            "batch_size": batch_size,
            "avg_latency_sec": round(avg_latency, 4),
            "min_latency_sec": round(min_latency, 4),
            "max_latency_sec": round(max_latency, 4),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "avg_tokens_generated": round(avg_tokens_per_run, 1),
            "time_per_token_ms": round(1000.0 / max(tokens_per_sec, 0.01), 2),
            "num_runs": NUM_RUNS,
        }
        
        if ttft_samples:
            avg_ttft = sum(ttft_samples) / len(ttft_samples)
            result_entry["avg_ttft_sec"] = round(avg_ttft, 4)
        
        profile_results[str(batch_size)] = result_entry
        print(f"    ✅ batch={batch_size}: {tokens_per_sec:.1f} tok/s, "
              f"latency={avg_latency:.3f}s, ms/tok={1000/max(tokens_per_sec, 0.01):.1f}")
    
    # Print summary table
    print(f"\n  {'='*55}")
    print(f"  vLLM Results: {strategy_name}")
    print(f"  {'='*55}")
    print(f"  {'Batch':>5}  {'Tok/s':>10}  {'Latency':>10}  {'ms/tok':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}")
    for bs, data in profile_results.items():
        print(f"  {data['batch_size']:>5}  "
              f"{data['tokens_per_sec']:>10.1f}  "
              f"{data['avg_latency_sec']:>10.3f}s  "
              f"{data['time_per_token_ms']:>8.1f}")
    
    # Also do a quick generation quality check
    print(f"\n  Sample generation:")
    test_outputs = llm.generate(["Hello, my name is"], sampling_params)
    print(f"    Input:  'Hello, my name is'")
    print(f"    Output: '{test_outputs[0].outputs[0].text[:200]}'")
    
    strategy_results = {
        "model_path": str(model_path),
        "strategy": strategy_name,
        "vllm_config": {
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "max_model_len": MAX_MODEL_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "enforce_eager": ENFORCE_EAGER,
        },
        "vram_usage": vram_usage,
        "profile_results": profile_results,
    }
    
    # Cleanup
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(3)
    print(f"\n  GPU memory freed. Waiting 3s before next strategy...")
    
    return strategy_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  vLLM Inference Profiling — MxMoE Quantized Sarvam-30B")
    print("  with FusedMoE Weight Scale Shape Mismatch Patch")
    print("=" * 70)
    
    t_start = time.time()
    
    # ── Step 1: Import vLLM ──────────────────────────────────────────────
    print("\n[1/4] Importing vLLM...")
    try:
        from vllm import LLM, SamplingParams
        import vllm
        print(f"  vLLM version: {vllm.__version__}")
    except ImportError:
        print("❌ vLLM not installed! Activate the mxmoe_vllm_env first.")
        print("   conda activate mxmoe_vllm_env")
        sys.exit(1)
    
    # ── Step 2: Apply the monkeypatch ────────────────────────────────────
    print("\n[2/4] Applying FusedMoE weight scale patch...")
    apply_fusedmoe_patch()
    
    # ── Step 3: Register the model architecture ──────────────────────────
    print("\n[3/4] Registering SarvamMoEForCausalLM...")
    register_sarvam_model()
    
    # ── Step 4: Profile each strategy ────────────────────────────────────
    print("\n[4/4] Starting profiling...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find available models
    available_models = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            available_models[name] = path
            print(f"  ✅ Found {name}: {path}")
        else:
            print(f"  ⚠️  Not found {name}: {path}")
    
    if not available_models:
        print("❌ No quantized models found! Check MODEL_PATHS configuration.")
        sys.exit(1)
    
    # Profile each strategy
    overall_results = {}
    
    for strategy_name, model_path in available_models.items():
        result = profile_strategy(strategy_name, model_path, LLM, SamplingParams)
        overall_results[strategy_name] = result
        
        # Save per-strategy results immediately
        output_path = OUTPUT_DIR / f"vllm_profiling_{strategy_name}.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=4, ensure_ascii=False, default=str)
        print(f"  Saved: {output_path}")
    
    # ── Save combined results ────────────────────────────────────────────
    # Prefer int8_gptq as default (Pareto optimal)
    default_strategy = "int8_gptq" if "int8_gptq" in overall_results else \
                       list(overall_results.keys())[0]
    
    combined = dict(overall_results[default_strategy])
    combined["all_strategies"] = overall_results
    combined["total_time_sec"] = round(time.time() - t_start, 2)
    combined["patch_applied"] = "FusedMoE._load_per_channel_weight_scale [N,1]→[N,G] broadcast"
    
    # Save main results file
    output_path = OUTPUT_DIR / "vllm_profiling.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=4, ensure_ascii=False, default=str)
    
    # Legacy compatibility file
    legacy = dict(combined)
    legacy["metrics"] = combined.get("profile_results", {})
    legacy_path = OUTPUT_DIR / "vllm_profiling_results.json"
    with open(legacy_path, "w", encoding="utf-8") as fh:
        json.dump(legacy, fh, indent=4, ensure_ascii=False, default=str)
    
    # ── Final Summary ────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("  PROFILING COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"  Strategies profiled: {list(overall_results.keys())}")
    
    # Print comparison table if both strategies succeeded
    successful = {k: v for k, v in overall_results.items() 
                  if "error" not in v and v.get("profile_results")}
    
    if len(successful) > 1 and "1" in list(successful.values())[0].get("profile_results", {}):
        print(f"\n  {'Strategy':<12} {'BS=1 TPS':>10} {'BS=8 TPS':>10} {'VRAM (GB)':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for name, res in successful.items():
            pr = res["profile_results"]
            bs1 = pr.get("1", {}).get("tokens_per_sec", "N/A")
            bs8 = pr.get("8", {}).get("tokens_per_sec", "N/A")
            vram = sum(v.get("allocated_gb", 0) for v in res.get("vram_usage", {}).values())
            print(f"  {name:<12} {bs1:>10} {bs8:>10} {vram:>10.1f}")
    
    for name, res in overall_results.items():
        if "error" in res:
            print(f"\n  ⚠️  {name} FAILED: {res['error']}")
    
    print(f"\n  Output files:")
    print(f"    {OUTPUT_DIR / 'vllm_profiling.json'}")
    for name in overall_results:
        print(f"    {OUTPUT_DIR / f'vllm_profiling_{name}.json'}")


if __name__ == "__main__":
    main()
