#!/usr/bin/env python3
"""
Standalone script to check disk size of quantized models.
No dependencies on codebase — simple and direct.
"""

import os
from pathlib import Path


def get_directory_size(path):
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except Exception as e:
        print(f"  Error calculating size: {e}")
        return 0
    return total


def format_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def main():
    # Define model paths
    base_path = Path("./research/quantized_models")
    
    models = {
        "FP8 Quantized": base_path / "fp8_quantized",
        "INT8 Quantized": base_path / "int8_quantized",
        "NF4 Quantized": base_path / "nf4_quantized",
        "GPTQ Quantized": base_path / "gptq_quantized",
    }
    
    # BF16 baseline is NOT saved to disk (~140GB) — it's loaded from:
    # 1. Local path: ../model_registry (if available)
    # 2. HuggingFace Hub: meta-llama/Llama-3.1-70B (otherwise)
    bf16_path = Path("./model_registry")
    if bf16_path.exists():
        models["BF16 Baseline (Full Model)"] = bf16_path
    else:
        print("ℹ️  BF16 baseline not found at ../model_registry")
        print("   (will be loaded from HuggingFace Hub on demand)\n")
    
    print("=" * 60)
    print("  MODEL DISK SIZES")
    print("=" * 60)
    
    total_size = 0
    found_models = []
    
    for model_name, model_path in models.items():
        if model_path.exists():
            size_bytes = get_directory_size(str(model_path))
            size_formatted = format_size(size_bytes)
            print(f"{model_name:.<40} {size_formatted:>12}")
            total_size += size_bytes
            found_models.append((model_name, size_bytes))
        else:
            print(f"{model_name:.<40} {'[NOT FOUND]':>12}")
    
    print("=" * 60)
    print(f"{'TOTAL':.<40} {format_size(total_size):>12}")
    print("=" * 60)
    print()
    
    # Summary
    if found_models:
        print("Found Models Summary:")
        for name, size in sorted(found_models, key=lambda x: x[1], reverse=True):
            pct = (size / total_size * 100) if total_size > 0 else 0
            print(f"  {name:.<35} {format_size(size):>12} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
