#!/usr/bin/env python3
"""
Detailed Quantized Model Dtype & Size Auditor.

Inspects every safetensors shard in each quantized checkpoint and reports:
  1. Per-dtype tensor counts and total bytes
  2. Compression ratio vs theoretical BF16 baseline
  3. Percentage of weights truly quantized vs kept in higher precision
  4. Per-component breakdown (attention, experts, shared_experts, embeddings, norms)
  5. Quantization metadata inventory (scales, zeros, g_idx, etc.)

Usage:
    python check/check_model_dtypes.py
"""

import os
import sys
from collections import defaultdict
from safetensors import safe_open

# ── Configuration ──────────────────────────────────────────────────────
BASE_PATH = "research/quantized_models"

# Bytes per element for each dtype
DTYPE_BYTES = {
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e4m3fnuz": 1,
    "torch.float8_e5m2": 1,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.int16": 2,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.bool": 1,
}

# Component classification by tensor name patterns
def classify_tensor(name: str) -> str:
    """Classify a tensor into a semantic component."""
    n = name.lower()
    if "lm_head" in n or "embed" in n:
        return "embedding/lm_head"
    if "layernorm" in n or "norm" in n:
        return "layernorm"
    if "gate.weight" in n and "experts" not in n:
        return "router"
    if "shared_experts" in n:
        if "scb" in n or "scale" in n or "zero" in n or "g_idx" in n or "absmax" in n or "quant_state" in n or "weight_format" in n:
            return "shared_expert_meta"
        return "shared_expert_weight"
    if "experts." in n:
        if "scb" in n or "scale" in n or "zero" in n or "g_idx" in n or "absmax" in n or "quant_state" in n or "weight_format" in n:
            return "expert_meta"
        return "expert_weight"
    if "attention" in n:
        if "scb" in n or "scale" in n or "zero" in n or "g_idx" in n or "absmax" in n or "quant_state" in n or "weight_format" in n:
            return "attention_meta"
        if "weight" in n or "qweight" in n or "._data" in n:
            return "attention_weight"
        return "attention_other"
    if "mlp" in n:
        if "scb" in n or "scale" in n or "zero" in n or "g_idx" in n or "absmax" in n or "quant_state" in n or "weight_format" in n:
            return "mlp_meta"
        return "mlp_weight"
    return "other"


def classify_tensor_role(name: str) -> str:
    """Classify whether a tensor is a weight, scale/metadata, or other."""
    n = name.lower()
    if any(k in n for k in ["scb", "scale", "zero", "g_idx", "absmax", "quant_state", "weight_format", "nested_absmax", "nested_quant_map", "bitsandbytes"]):
        return "quant_metadata"
    if any(k in n for k in ["weight", "qweight", "._data"]):
        return "weight"
    if any(k in n for k in ["bias"]):
        return "bias"
    return "other"


def format_bytes(b: int) -> str:
    """Pretty-print bytes."""
    if b >= 1024**3:
        return f"{b / 1024**3:.3f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def analyze_folder(folder_path: str, folder_name: str):
    """Analyze all safetensors in a folder."""
    sf_files = sorted([
        f for f in os.listdir(folder_path) if f.endswith(".safetensors")
    ])
    if not sf_files:
        return

    print(f"\n{'=' * 72}")
    print(f"  {folder_name.upper()}")
    print(f"{'=' * 72}")

    # Accumulators
    dtype_counts = defaultdict(int)          # dtype -> count of tensors
    dtype_bytes = defaultdict(int)           # dtype -> total bytes
    dtype_elements = defaultdict(int)        # dtype -> total elements
    component_bytes = defaultdict(int)       # component -> total bytes
    component_counts = defaultdict(int)      # component -> tensor count
    role_bytes = defaultdict(int)            # role -> total bytes
    role_counts = defaultdict(int)           # role -> tensor count
    total_tensors = 0
    total_bytes = 0
    total_elements = 0

    # Specific dtype examples (first seen)
    dtype_examples = {}

    for filename in sf_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with safe_open(file_path, framework="pt") as f:
                keys = list(f.keys())
                for k in keys:
                    t = f.get_tensor(k)
                    dtype_str = str(t.dtype)
                    numel = t.numel()
                    elem_size = DTYPE_BYTES.get(dtype_str, t.element_size())
                    nbytes = numel * elem_size

                    dtype_counts[dtype_str] += 1
                    dtype_bytes[dtype_str] += nbytes
                    dtype_elements[dtype_str] += numel

                    comp = classify_tensor(k)
                    component_bytes[comp] += nbytes
                    component_counts[comp] += 1

                    role = classify_tensor_role(k)
                    role_bytes[role] += nbytes
                    role_counts[role] += 1

                    total_tensors += 1
                    total_bytes += nbytes
                    total_elements += numel

                    if dtype_str not in dtype_examples:
                        shape_str = str(tuple(t.shape))
                        dtype_examples[dtype_str] = f"{k}  {shape_str}"

        except Exception as e:
            print(f"  ⚠ Error reading {filename}: {e}")

    # ── 1. Overview ──────────────────────────────────────────────────
    print(f"\n  Files: {len(sf_files)}  |  Tensors: {total_tensors:,}  |  "
          f"Total: {format_bytes(total_bytes)}  |  Elements: {total_elements:,.0f}")

    # ── 2. Dtype breakdown ───────────────────────────────────────────
    print(f"\n  {'DTYPE BREAKDOWN':─^68}")
    print(f"  {'Dtype':<28} {'Tensors':>8} {'Elements':>14} {'Bytes':>12} {'Share':>7}")
    print(f"  {'─'*28} {'─'*8} {'─'*14} {'─'*12} {'─'*7}")
    for dtype_str in sorted(dtype_bytes, key=lambda d: dtype_bytes[d], reverse=True):
        pct = 100.0 * dtype_bytes[dtype_str] / total_bytes if total_bytes else 0
        print(f"  {dtype_str:<28} {dtype_counts[dtype_str]:>8,} "
              f"{dtype_elements[dtype_str]:>14,.0f} "
              f"{format_bytes(dtype_bytes[dtype_str]):>12} "
              f"{pct:>6.1f}%")
    print(f"  {'─'*28} {'─'*8} {'─'*14} {'─'*12} {'─'*7}")
    print(f"  {'TOTAL':<28} {total_tensors:>8,} "
          f"{total_elements:>14,.0f} "
          f"{format_bytes(total_bytes):>12} {'100.0':>6}%")

    # ── 3. Example tensor per dtype ──────────────────────────────────
    print(f"\n  {'EXAMPLE TENSORS':─^68}")
    for dtype_str in sorted(dtype_examples):
        print(f"  {dtype_str:<26} → {dtype_examples[dtype_str]}")

    # ── 4. Component breakdown ───────────────────────────────────────
    print(f"\n  {'COMPONENT BREAKDOWN':─^68}")
    print(f"  {'Component':<28} {'Tensors':>8} {'Bytes':>12} {'Share':>7}")
    print(f"  {'─'*28} {'─'*8} {'─'*12} {'─'*7}")
    for comp in sorted(component_bytes, key=lambda c: component_bytes[c], reverse=True):
        pct = 100.0 * component_bytes[comp] / total_bytes if total_bytes else 0
        print(f"  {comp:<28} {component_counts[comp]:>8,} "
              f"{format_bytes(component_bytes[comp]):>12} "
              f"{pct:>6.1f}%")

    # ── 5. Role breakdown (weights vs metadata) ──────────────────────
    print(f"\n  {'ROLE BREAKDOWN (weights vs quantization metadata)':─^68}")
    print(f"  {'Role':<28} {'Tensors':>8} {'Bytes':>12} {'Share':>7}")
    print(f"  {'─'*28} {'─'*8} {'─'*12} {'─'*7}")
    for role in sorted(role_bytes, key=lambda r: role_bytes[r], reverse=True):
        pct = 100.0 * role_bytes[role] / total_bytes if total_bytes else 0
        print(f"  {role:<28} {role_counts[role]:>8,} "
              f"{format_bytes(role_bytes[role]):>12} "
              f"{pct:>6.1f}%")

    # ── 6. Compression analysis ──────────────────────────────────────
    # Packed formats: GPTQ packs 8×int4 per int32, NF4 packs 2×nf4 per uint8
    # We need to estimate the REAL parameter count, not the packed element count.

    # Count quantized weight elements (1 element = 1 weight)
    quant_8bit_elems = dtype_elements.get("torch.int8", 0) + \
                       dtype_elements.get("torch.float8_e4m3fn", 0) + \
                       dtype_elements.get("torch.float8_e4m3fnuz", 0) + \
                       dtype_elements.get("torch.float8_e5m2", 0)

    # NF4: uint8 elements that are weights (2 nf4 values per byte)
    # vs uint8 that are metadata (absmax, quant_state, etc.)
    nf4_weight_bytes = 0
    nf4_meta_bytes = 0
    nf4_weight_elems = dtype_elements.get("torch.uint8", 0)
    # For NF4, estimate: weight uint8 tensors have shape [N, 1], meta tensors are smaller
    # Use role_bytes to separate: weight role vs metadata role
    # Simpler: if uint8 dominates and int8/fp8 are absent, this is NF4
    is_nf4 = (nf4_weight_elems > 1_000_000 and quant_8bit_elems == 0
              and dtype_elements.get("torch.int32", 0) < 1_000_000)

    # GPTQ: int32 elements contain packed weights (8 int4 per int32) AND metadata (g_idx, qzeros)
    int32_elems = dtype_elements.get("torch.int32", 0)
    is_gptq = (int32_elems > 1_000_000 and quant_8bit_elems == 0)

    fp16_elems = dtype_elements.get("torch.float16", 0) + dtype_elements.get("torch.bfloat16", 0)
    fp32_elems = dtype_elements.get("torch.float32", 0)

    # Estimate real parameter count
    if quant_8bit_elems > 0:
        # INT8 or FP8: 1 element = 1 weight parameter
        estimated_real_params = quant_8bit_elems + fp16_elems
    elif is_nf4:
        # NF4: each uint8 in weight tensors holds 2 NF4 values
        # weight role bytes / 1 byte per uint8 × 2 values per byte
        nf4_packed_weight_bytes = role_bytes.get("weight", 0) - (fp16_elems * 2)
        estimated_quantized_params = max(nf4_packed_weight_bytes * 2, 0)  # 2 values per byte
        estimated_real_params = estimated_quantized_params + fp16_elems
    elif is_gptq:
        # GPTQ: qweight int32 has 8 int4 values packed per element
        # But int32 also includes g_idx and qzeros (metadata)
        # Use: weight role bytes / 4 bytes per int32 × 8 values per int32
        gptq_weight_bytes_int32 = role_bytes.get("weight", 0) - (fp16_elems * 2)
        estimated_quantized_params = max(gptq_weight_bytes_int32 // 4 * 8, 0)
        estimated_real_params = estimated_quantized_params + fp16_elems
    else:
        estimated_real_params = total_elements

    # BF16 equivalent: if all real params were stored as BF16 (2 bytes each)
    bf16_equiv_bytes = estimated_real_params * 2
    compression = bf16_equiv_bytes / total_bytes if total_bytes else 0

    print(f"\n  {'COMPRESSION ANALYSIS':─^68}")
    print(f"  Estimated real parameters:                  {estimated_real_params:,.0f} ({estimated_real_params/1e9:.2f}B)")
    print(f"  Theoretical BF16 size (params × 2B):        {format_bytes(int(bf16_equiv_bytes))}")
    print(f"  Actual checkpoint size:                     {format_bytes(total_bytes)}")
    print(f"  Compression ratio vs BF16:                  {compression:.2f}×")

    if quant_8bit_elems > 0:
        quant_pct = 100.0 * quant_8bit_elems / estimated_real_params if estimated_real_params else 0
        print(f"  Quantized weight params (int8/fp8):         {quant_8bit_elems:,.0f} ({quant_pct:.1f}%)")
    if is_nf4:
        nf4_real = max(estimated_real_params - fp16_elems, 0)
        nf4_pct = 100.0 * nf4_real / estimated_real_params if estimated_real_params else 0
        print(f"  NF4-quantized params (2 per uint8):         {nf4_real:,.0f} ({nf4_pct:.1f}%)")
        print(f"  Packed uint8 elements on disk:              {nf4_weight_elems:,.0f}")
    if is_gptq:
        gptq_real = max(estimated_real_params - fp16_elems, 0)
        gptq_pct = 100.0 * gptq_real / estimated_real_params if estimated_real_params else 0
        print(f"  GPTQ-quantized params (8 per int32):        {gptq_real:,.0f} ({gptq_pct:.1f}%)")
        print(f"  Packed int32 elements on disk:              {int32_elems:,.0f}")
    if fp16_elems > 0:
        fp16_pct = 100.0 * fp16_elems / estimated_real_params if estimated_real_params else 0
        print(f"  FP16/BF16 params (unquantized):             {fp16_elems:,.0f} ({fp16_pct:.1f}%)")
    if fp32_elems > 0:
        print(f"  FP32 elements (scales/metadata only):       {fp32_elems:,.0f}")
    if int32_elems > 0 and not is_gptq:
        print(f"  INT32 elements:                             {int32_elems:,.0f}")

    print()


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.isdir(BASE_PATH):
        print(f"Error: {BASE_PATH} not found. Run from project root.")
        sys.exit(1)

    folders = sorted([
        d for d in os.listdir(BASE_PATH)
        if os.path.isdir(os.path.join(BASE_PATH, d))
    ])

    if not folders:
        print(f"No quantized model folders found in {BASE_PATH}")
        sys.exit(1)

    print(f"╔{'═'*70}╗")
    print(f"║{'QUANTIZED MODEL DTYPE & SIZE AUDIT':^70}║")
    print(f"╚{'═'*70}╝")

    for folder in folders:
        folder_path = os.path.join(BASE_PATH, folder)
        analyze_folder(folder_path, folder)

    print(f"\n{'═' * 72}")
    print(f"  AUDIT COMPLETE")
    print(f"{'═' * 72}")
