#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  Inference Script for Quantized SarvamMoE-30B (INT8 & FP8)
═══════════════════════════════════════════════════════════════════════════════

Loads the INT8 or FP8 quantized SarvamMoE models using HuggingFace Transformers
and generates a response to the prompt: "Explain Artificial Intelligence?".

Usage:
  1. Run INT8 model only:
     python notebooks/run_inference.py int8

  2. Run FP8 model only:
     python notebooks/run_inference.py fp8

  3. Run both one after the other:
     python notebooks/run_inference.py
"""

import gc
import sys
import time
from pathlib import Path
import torch

# Configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATHS = {
    "int8": PROJECT_DIR / "mxmoe" / "quantized_models_int8_gptq",
    "fp8": PROJECT_DIR / "mxmoe" / "quantized_models_fp8_gptq",
}

DEFAULT_PROMPT = "Explain Artificial Intelligence?"


def get_gpu_memory():
    """Print current GPU memory usage."""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def run_inference(model_type: str, prompt: str):
    print("=" * 60)
    print(f" Loading {model_type.upper()} Model...")
    print("=" * 60)
    
    model_path = MODEL_PATHS[model_type]
    
    if not model_path.exists():
        print(f"❌ Error: Model path {model_path} does not exist.")
        return
        
    start_time = time.time()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    
    load_time = time.time() - start_time
    print(f"✅ Loaded in {load_time:.1f} seconds")
    get_gpu_memory()
    
    # Run generation
    print(f"\nPrompt: '{prompt}'")
    print("Generating response...")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    inputs.pop("token_type_ids", None)  # Avoid ValueError in custom model forward
    
    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    gen_time = time.time() - gen_start
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response if it is included
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
        
    print("\nResponse:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    print(f"Generation took {gen_time:.1f} seconds.\n")
    
    # Clean up GPU memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 GPU memory cleared.\n")


def main():
    # Parse argument
    args = sys.argv[1:]
    model_choice = None
    
    if len(args) > 0:
        arg = args[0].lower()
        if arg in ["int8", "fp8"]:
            model_choice = arg
        else:
            print(f"Unknown argument '{args[0]}'. Use 'int8', 'fp8', or no arguments to run both.")
            sys.exit(1)
            
    # Check prompt argument if provided
    prompt = DEFAULT_PROMPT
    if len(args) > 1:
        prompt = " ".join(args[1:])
        
    if model_choice:
        run_inference(model_choice, prompt)
    else:
        print("No model specified. Running both INT8 and FP8 sequentially...\n")
        run_inference("int8", prompt)
        print("Waiting 5 seconds before loading FP8 model...")
        time.sleep(5)
        run_inference("fp8", prompt)


if __name__ == "__main__":
    main()
