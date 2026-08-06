# Module 1: BF16 Baseline — Reference Metrics & Weight Cache Initialization

> **Pipeline**: Research Quantization Pipeline — Module 1 (BF16 Baseline)
> **Wall Time**: 160.8s (~2.7 min)

**Navigation**: [← Project README](../../README.md) · [Research Overview](../../research/README.md) · [Module 2 →](module2_quantization.md) · [Module 3](module3_analysis.md) · [Module 4](module4_profiling.md) · [Module 5](module5_evaluation.md) · [Module 6](module6_visualization.md)

---

## 1. Purpose

Module 1 is the foundation of the entire research pipeline. It loads the full-precision BF16 (Brain Floating Point 16) model, captures the complete architecture metadata, measures baseline memory consumption under both static and dynamic conditions, and — critically — builds the **shared weight cache** that all subsequent modules depend on for MSE comparison, outlier analysis, and visualization.

Without Module 1's outputs, Modules 3, 5, and 6 cannot function correctly because they require the BF16 reference weights as the ground truth for quantization error measurement.

---

## 2. Model Architecture

Sarvam-30B uses the `SarvamMoEForCausalLM` architecture — a custom Mixture-of-Experts transformer that requires `trust_remote_code=True` for loading. This is not a standard HuggingFace model architecture; it uses custom attention and MoE implementations.

| Property | Value | Notes |
|---|---|---|
| **Model Type** | `sarvam_moe` | Custom architecture, requires `trust_remote_code=True` |
| **Total Parameters** | 32.15B | All parameters across all experts |
| **Active Parameters/Token** | 4.52B | Only 6 of 128 experts fire per token |
| **Layers** | 19 | Layer 0 = dense MLP, Layers 1–18 = MoE |
| **Hidden Size** | 4,096 | |
| **Vocabulary Size** | 262,144 | Large vocabulary for multilingual Indic support |
| **Experts per MoE Layer** | 128 routed + 1 shared | Top-6 routing per token |
| **Top-K Routing** | 6 | Each token activates 6 of 128 experts |

### Architecture Layout

The model has a **heterogeneous layer structure** — layer 0 is a standard dense MLP, while layers 1–18 are full MoE layers:

```
Layer 0 (Dense):
├── attention.query_key_value     # Fused QKV projection (not separate q/k/v_proj)
├── attention.dense               # Output projection (not o_proj)
├── mlp.gate_proj                 # Standard dense MLP
├── mlp.up_proj
└── mlp.down_proj

Layers 1–18 (MoE):
├── attention.query_key_value
├── attention.dense
├── mlp.gate.weight               # Router (128-way softmax gating)
├── mlp.shared_experts.gate_proj  # Always-active shared expert
├── mlp.shared_experts.up_proj
├── mlp.shared_experts.down_proj
└── mlp.experts.{0-127}           # 128 routed experts
    ├── gate_proj
    ├── up_proj
    └── down_proj
```

> **Key architectural detail**: Sarvam-30B uses **fused attention projections**. `attention.query_key_value` is a single weight matrix that contains Q, K, and V concatenated, rather than the separate `q_proj`/`k_proj`/`v_proj` seen in LLaMA-style models. The output projection is called `attention.dense` (not `o_proj`). This matters for quantization because the fused QKV matrix is 3× larger and has different numerical properties than individual projections.

---

## 3. How Module 1 Executes

### Step 1: Model Loading

The BF16 model is loaded using HuggingFace's `AutoModelForCausalLM.from_pretrained()` with `device_map="auto"`, which distributes layers across both A100 GPUs using the `accelerate` library's automatic placement algorithm:

```python
# Effective loading configuration
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,          # Required for SarvamMoEForCausalLM
    attn_implementation="sdpa",      # Scaled Dot-Product Attention (PyTorch native)
    max_memory={"0": "70GiB", "1": "78GiB"}  # Biases placement toward GPU 1
)
```

The `max_memory` configuration intentionally gives GPU 1 more budget (78 GiB vs 70 GiB) so that `accelerate`'s automatic placement biases more layers toward GPU 1 as the primary compute device. This is controlled by `hardware.primary_cuda_index: 1` in the config.

**Load time**: 27.6 seconds from a local pre-downloaded snapshot at `model_registry/models--sarvamai--sarvam-30b/snapshots/`.

### Step 2: Static Memory Measurement

After loading, the module takes a CUDA memory snapshot on both GPUs using `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`:

| Metric | GPU 0 | GPU 1 | Total |
|---|---|---|---|
| **Allocated VRAM** | 26.98 GB | 32.91 GB | **59.89 GB** |
| **Reserved VRAM** | 26.99 GB | 32.92 GB | 59.91 GB |
| **GPU Utilization** | 34.1% | 41.6% | — |
| **CPU RAM Used** | — | — | 335.8 GB |

The gap between allocated and reserved memory is tiny (~10 MB), meaning PyTorch's caching allocator has minimal fragmentation overhead. The model is split roughly 45%/55% across the two GPUs.

### Step 3: Dynamic Memory Measurement

The module generates 64 tokens from a fixed prompt to measure KV cache overhead and peak memory during inference:

| Metric | GPU 0 | GPU 1 | Total |
|---|---|---|---|
| **Peak VRAM** | 27.01 GB | 32.94 GB | **59.95 GB** |
| **KV Cache Delta** | 0.008 GB | 0.008 GB | 0.016 GB |

> **Why KV cache is tiny**: With 19 layers, hidden_size=4096, and only 64 new tokens for a single sequence, the KV cache is approximately `19 × 2 × 4096 × 64 × 2 bytes ≈ 20 MB`. This is negligible compared to the model's 60 GB footprint. For production batch serving (e.g., batch_size=32, seq_len=4096), KV cache would grow to ~10 GB.

### Step 4: Weight Cache Construction

This is the most critical step. Module 1 extracts and saves weight samples from **97 target tensor locations** across the model. For each tensor, it:

1. Calls `w.dequantize()` to get the raw BF16 values (this is a no-op for BF16 but ensures consistent extraction across quantized models)
2. Samples up to 500,000 values from the tensor (controlled by `visualization.weight_sample_size: 500000`)
3. Saves the sample as a compressed `.npz` file to `research/outputs/shared_weights/bf16/`

The 97 tensors cover:
- **38 attention projections**: 19 layers × 2 projections (`query_key_value` + `dense`)
- **5 dense MLP projections**: Layer 0's `gate_proj`, `up_proj`, `down_proj`
- **54 shared expert projections**: 18 MoE layers × 3 projections (`gate_proj`, `up_proj`, `down_proj`)
- **Plus select routed experts**: Experts 0, 63, 127 at representative layers for spot-checking

Each `.npz` file is approximately 1.7 MB, totaling ~165 MB for the complete BF16 cache. These cached values are the **ground truth** that Module 3 uses to compute MSE against quantized weight caches.

---

## 4. Outputs

| File | Size | Description |
|---|---|---|
| `results/bf16_results.json` | 2.4 KB | Complete BF16 metrics (architecture, static/dynamic memory, timing) |
| `results/model_architecture.json` | 719 B | Model architecture metadata (layers, experts, projections) |
| `shared_weights/bf16/*.npz` | ~165 MB | 97 weight sample NPZ caches — **ground truth for all MSE comparisons** |

---

## 5. Dependency Graph

```
Module 1 (BF16 Baseline)
   ├── shared_weights/bf16/  ──→  Module 2 (comparison target for weight cache validation)
   ├── bf16_results.json     ──→  Module 3 (reference for MSE and outlier baselines)
   ├── model_architecture.json ──→ Module 6 (layer layout for visualization)
   └── bf16_results.json     ──→  Module 5 (baseline perplexity/MMLU for quality comparison)
```

*Module 1 must complete before Modules 2–6. Its weight caches and architecture metadata are the foundation that every downstream module depends on.*
