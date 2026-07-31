#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  HuggingFace Transformers Inference Profiling — INT8 GPTQ SarvamMoE-30B
═══════════════════════════════════════════════════════════════════════════════

Profiles the int8_gptq quantized SarvamMoE model using HuggingFace Transformers.
This is used as an alternative to vLLM profiling since vLLM's FusedMoE kernel
does not support the pack-quantized int8 format.

Usage (on GPU server):
    python notebooks/hf_profiling_int8.py

Output:
    mxmoe/outputs/module_4_deployment/results/hf_profiling_int8_gptq.json
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
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_DIR / "mxmoe" / "quantized_models_int8_gptq"
OUTPUT_DIR = PROJECT_DIR / "mxmoe" / "outputs" / "module_4_deployment" / "results"

# Profiling settings
BATCH_SIZES = [1, 4, 8, 16, 32]
WARMUP_STEPS = 2
NUM_RUNS = 3
MAX_NEW_TOKENS = 128

# Test prompt
PROMPT = (
    "The future of artificial intelligence in healthcare is transforming "
    "how we approach diagnosis and treatment. In the coming years, we expect"
)


def get_gpu_memory():
    """Get current GPU memory usage for all available GPUs."""
    mem = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        mem[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
        }
    return mem


def profile_batch(model, tokenizer, batch_size: int, device: str) -> Dict[str, Any]:
    """Profile inference for a given batch size."""
    prompts = [PROMPT] * batch_size
    
    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    inputs.pop("token_type_ids", None)
    
    input_len = inputs["input_ids"].shape[1]
    
    # Warmup
    print(f"    Warmup: {WARMUP_STEPS} runs")
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
    torch.cuda.synchronize()
    
    # Timed runs
    latencies = []
    total_tokens_list = []
    
    for run_idx in range(NUM_RUNS):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        # Count only new tokens generated (subtract input length)
        total_new_tokens = sum(
            len(seq) - input_len for seq in outputs
        )
        
        latencies.append(elapsed)
        total_tokens_list.append(total_new_tokens)
        
        tok_per_sec = total_new_tokens / elapsed
        print(f"    Run {run_idx + 1}/{NUM_RUNS}: {elapsed:.3f}s, "
              f"{total_new_tokens} tokens, {tok_per_sec:.1f} tok/s")
    
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(total_tokens_list) / len(total_tokens_list)
    avg_tok_per_sec = avg_tokens / avg_latency
    
    result = {
        "batch_size": batch_size,
        "avg_latency_sec": round(avg_latency, 4),
        "min_latency_sec": round(min(latencies), 4),
        "max_latency_sec": round(max(latencies), 4),
        "tokens_per_sec": round(avg_tok_per_sec, 2),
        "avg_tokens_generated": avg_tokens,
        "time_per_token_ms": round(1000.0 / avg_tok_per_sec, 2),
        "num_runs": NUM_RUNS,
    }
    
    print(f"    ✅ batch={batch_size}: {avg_tok_per_sec:.1f} tok/s, "
          f"latency={avg_latency:.3f}s, ms/tok={1000.0/avg_tok_per_sec:.1f}")
    
    return result


def main():
    print("=" * 70)
    print("  HuggingFace Transformers Profiling — INT8 GPTQ SarvamMoE-30B")
    print("=" * 70)
    
    total_start = time.time()
    
    # ── Step 1: Load model ──
    print(f"\n[1/3] Loading model from {MODEL_PATH}...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    load_start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        import accelerate
        has_accelerate = True
    except ImportError:
        has_accelerate = False
        print("  ⚠️  accelerate not installed, using manual GPU placement")
    
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        **({"device_map": "auto"} if has_accelerate else {}),
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    if not has_accelerate:
        model = model.cuda()
    
    model.eval()
    
    load_time = time.time() - load_start
    print(f"  ✅ Model loaded in {load_time:.1f}s")
    
    # Determine primary device
    if hasattr(model, "hf_device_map"):
        print(f"  Device map: {dict(list(model.hf_device_map.items())[:5])}...")
        device = "cuda:0"
    else:
        device = next(model.parameters()).device
        print(f"  Device: {device}")
    
    # ── Step 2: Memory snapshot ──
    print(f"\n[2/3] GPU memory after loading:")
    vram = get_gpu_memory()
    for gpu, mem in vram.items():
        print(f"  {gpu}: {mem['allocated_gb']:.2f} GB allocated, "
              f"{mem['reserved_gb']:.2f} GB reserved / {mem['total_gb']:.2f} GB total")
    
    # ── Step 3: Profile ──
    print(f"\n[3/3] Profiling inference...")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  Warmup: {WARMUP_STEPS}, Runs: {NUM_RUNS}")
    
    profile_results = {}
    
    for bs in BATCH_SIZES:
        print(f"\n  Profiling batch_size={bs}")
        try:
            result = profile_batch(model, tokenizer, bs, device)
            profile_results[str(bs)] = result
        except Exception as e:
            print(f"    ❌ batch={bs} failed: {e}")
            import traceback
            traceback.print_exc()
            profile_results[str(bs)] = {
                "batch_size": bs,
                "error": str(e),
            }
    
    # ── Sample generation ──
    print(f"\n  Sample generation:")
    sample_input = tokenizer("Hello, my name is", return_tensors="pt").to(device)
    sample_input.pop("token_type_ids", None)
    with torch.no_grad():
        sample_output = model.generate(
            **sample_input,
            max_new_tokens=50,
            do_sample=False,
        )
    sample_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    print(f"    Input:  'Hello, my name is'")
    print(f"    Output: '{sample_text}'")
    
    # ── Save results ──
    total_time = time.time() - total_start
    
    results = {
        "model_path": str(MODEL_PATH),
        "strategy": "int8_gptq",
        "engine": "huggingface_transformers",
        "config": {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "vram_usage": vram,
        "model_load_time_sec": round(load_time, 2),
        "profile_results": profile_results,
        "sample_output": sample_text[:200],
        "total_time_sec": round(total_time, 2),
    }
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "hf_profiling_int8_gptq.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    # ── Print summary ──
    print(f"\n{'=' * 70}")
    print(f"  HuggingFace Results: int8_gptq")
    print(f"{'=' * 70}")
    print(f"  {'Batch':>5}  {'Tok/s':>10}  {'Latency':>10}  {'ms/tok':>8}")
    print(f"  {'-----':>5}  {'----------':>10}  {'----------':>10}  {'--------':>8}")
    for bs_key, res in profile_results.items():
        if "error" not in res:
            print(f"  {res['batch_size']:>5}  {res['tokens_per_sec']:>10.1f}  "
                  f"{res['avg_latency_sec']:>9.3f}s  {res['time_per_token_ms']:>8.1f}")
        else:
            print(f"  {res['batch_size']:>5}  {'FAILED':>10}  {'—':>10}  {'—':>8}")
    
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results saved: {output_file}")
    
    # Also update the combined vllm_profiling_results.json with HF data
    combined_file = OUTPUT_DIR / "vllm_profiling_results.json"
    if combined_file.exists():
        with open(combined_file) as f:
            combined = json.load(f)
        # Add HF int8 results alongside vLLM data
        if "all_strategies" in combined:
            combined["all_strategies"]["int8_gptq_hf"] = results
        with open(combined_file, "w") as f:
            json.dump(combined, f, indent=4)
        print(f"  Updated combined results: {combined_file}")


if __name__ == "__main__":
    main()
