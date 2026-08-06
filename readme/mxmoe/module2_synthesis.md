# MxMoE Module 2 — Mixed-Precision Synthesis Results

> **Model**: `sarvamai/sarvam-30b` (MoE: 32B total, 2.4B active per token)
> **Hardware**: 2× NVIDIA A100 80GB PCIe (160 GB total VRAM)
> **Pipeline**: MxMoE Mixed-Precision Quantization — Module 2 (Mixed-Precision Synthesis)
> **Execution Date**: 2026-05-30 19:11:13 UTC → 2026-05-30 21:04:06 UTC (~1.88 hours)

**Navigation**: [← Project README](../../README.md) · [MxMoE Overview](../../mxmoe/README.md) · [← Module 1](module1_sensitivity.md) · [Module 3 →](module3_evaluation.md) · [Module 4](module4_deployment.md) · [Troubleshooting](troubleshooting_fp8_int8.md)

---

## Table of Contents

1. [Execution Summary](#1-execution-summary)
2. [Test Results — Verification](#2-test-results--verification)
3. [Sub-module 2a: Precision Recipe Generation](#3-sub-module-2a-precision-recipe-generation)
4. [Sub-module 2b: Model Compression](#4-sub-module-2b-model-compression)
5. [2-Pass Compression Architecture](#5-2-pass-compression-architecture)
6. [Compression Results](#6-compression-results)
7. [Detailed Execution Log Analysis](#7-detailed-execution-log-analysis)
8. [Output Artifacts](#8-output-artifacts)
9. [Key Findings & Downstream Impact](#9-key-findings--downstream-impact)

---

## 1. Execution Summary

Module 2 is the **Mixed-Precision Synthesis** stage. It takes the expert importance map from Module 1 and produces two quantized model variants using a 2-pass compression strategy.

| Parameter | Value |
|---|---|
| **Module** | 2 — Mixed-Precision Synthesis |
| **Status** | ✅ **SUCCESS** |
| **Total Wall Time** | 6,773.2 seconds (~1.88 hours) |
| **Sub-module 2a** | Precision Recipe Generation — ✅ SUCCESS (<0.1s) |
| **Sub-module 2b** | Model Compression (fp8_gptq) — ✅ SUCCESS (5,028.0s) |
| **Sub-module 2b** | Model Compression (int8_gptq) — ✅ SUCCESS (1,744.1s) |
| **Tests** | ✅ **ALL PASS** — 4/4 |
| **Strategies Produced** | `fp8_gptq`, `int8_gptq` |
| **Compressed Model Size** | 34.62 GB per strategy (vs ~60 GB BF16 baseline) |

### Pipeline Context (Modules 1→2)

| Module | Task | Wall Time | Status | Tests |
|---|---|---|---|---|
| Module 1 | Sensitivity Profiling | 1,246.9s (~20.8 min) | ✅ SUCCESS | 4/4 PASS |
| **Module 2** | **Mixed-Precision Synthesis** | **6,773.2s (~1.88 hr)** | **✅ SUCCESS** | **4/4 PASS** |
| **Total** | **Modules 1+2** | **8,020.2s (~2.23 hr)** | **✅ SUCCESS** | **8/8 PASS** |

---

## 2. Test Results — Verification

All **4 automated tests** for Module 2 passed:

| # | Test Name | Status | What It Verifies |
|---|---|---|---|
| 1 | `test_compressed_model_exists` | ✅ PASS | Quantized model checkpoints exist on disk |
| 2 | `test_compressor_uses_config_calibration_dataset` | ✅ PASS | GPTQ calibration used `sangraha_verified` dataset |
| 3 | `test_precision_recipe` | ✅ PASS | `precision_recipe.json` has correct structure and module counts |
| 4 | `test_recipe_builder_strategies` | ✅ PASS | Both `fp8_gptq` and `int8_gptq` strategies are present |

---

## 3. Sub-module 2a: Precision Recipe Generation

The recipe builder reads `expert_importance_map.json` from Module 1 and generates target lists for each quantization pass.

### 3.1 Recipe Classification

| Tier | Expert Count | Module Count | Quantization |
|---|---|---|---|
| **HIGH** | 1,013 experts | Part of 6,224 FP8/INT8 modules | 8-bit (FP8 or INT8) |
| **MEDIUM** | 1,030 experts | Part of 6,224 FP8/INT8 modules | 8-bit (FP8 or INT8) |
| **LOW** | 261 experts | 783 GPTQ modules | INT4 (GPTQ W4A16) |
| **Shared** | 18 shared experts | Part of 6,224 FP8/INT8 modules | 8-bit (FP8 or INT8) |
| **Attention** | All attention layers | Part of 6,224 FP8/INT8 modules | 8-bit (FP8 or INT8) |

### 3.2 Module Counts per Strategy

| Component | fp8_gptq | int8_gptq |
|---|---|---|
| FP8/INT8 targets (attention + shared + HIGH + MEDIUM) | 6,224 | 6,224 |
| GPTQ W4A16 targets (LOW only) | 783 | 783 |
| Ignore list | `lm_head`, `re:.*mlp\.gate$` | Same |

> **Why 783 GPTQ targets?** — 261 LOW experts × 3 linear projections per expert (`gate_proj`, `up_proj`, `down_proj`) = 783 modules.

---

## 4. Sub-module 2b: Model Compression

Module 2b executes a **2-pass compression** strategy for each variant. Both variants share the same GPTQ-quantized intermediate (Pass 1), then apply different 8-bit quantization in Pass 2.

### 4.1 Strategy: `fp8_gptq`

| Pass | Type | Targets | Time | Description |
|---|---|---|---|---|
| **Pass 1** | GPTQ W4A16 | 783 modules | 3,748.9s (~62.5 min) | INT4 weight quantization for LOW-importance experts |
| **Pass 2** | FP8 Dynamic | 6,224 modules | 1,279.0s (~21.3 min) | FP8 E4M3 dynamic quantization for all other modules |
| **Total** | — | — | **5,028.0s (~83.8 min)** | — |

### 4.2 Strategy: `int8_gptq`

| Pass | Type | Targets | Time | Description |
|---|---|---|---|---|
| **Pass 1** | GPTQ W4A16 | 783 modules | **0.0s** (reused) | Shared GPTQ intermediate from `fp8_gptq` |
| **Pass 2** | INT8 W8A16 | 6,224 modules | 1,744.1s (~29.1 min) | Per-channel symmetric INT8 weight quantization |
| **Total** | — | — | **1,744.1s (~29.1 min)** | — |

> **Efficiency note**: The INT8 strategy completed 3× faster because it reused the GPTQ intermediate checkpoint from the FP8 strategy's Pass 1, avoiding a second GPTQ calibration pass.

---

## 5. 2-Pass Compression Architecture

### 5.1 Why 2 Passes?

The MxMoE pipeline uses a 2-pass compression strategy because the quantization tools (`llm-compressor`) don't natively support applying two different quantization schemes to different subsets of modules in a single pass.

```
Pass 1: GPTQ W4A16 (LOW-importance experts only)
═══════════════════════════════════════════════════

BF16 Base Model
    │
    ├── Load full model onto GPU (60s)
    ├── Prepare calibration data: 256 samples × 1024 tokens
    │   └── Source: dataset/sangraha_verified
    ├── Apply GPTQ W4A16 to 783 LOW expert linear modules
    │   └── 261 experts × 3 projections (gate_proj, up_proj, down_proj)
    ├── Save intermediate checkpoint → mxmoe/quantized_models_gptq_shared_intermediate
    └── Checkpoint contains: BF16 weights (HIGH/MEDIUM) + INT4 packed (LOW)

Pass 2: FP8 or INT8 overlay (remaining modules)
═══════════════════════════════════════════════════

GPTQ Intermediate
    │
    ├── Load intermediate onto CPU only (104s)
    │   └── CPU-only loading avoids GPU memory conflicts
    ├── Apply FP8_DYNAMIC or INT8_W8A16 to 6,224 remaining modules
    │   ├── Attention layers → 8-bit
    │   ├── Shared experts → 8-bit
    │   ├── HIGH experts → 8-bit
    │   └── MEDIUM experts → 8-bit
    ├── GPTQ modules (783) in ignore list → not re-quantized
    └── Save final checkpoint → mxmoe/quantized_models_{fp8,int8}_gptq
```

### 5.2 Shared GPTQ Intermediate

Both strategies share the same Pass 1 output at `mxmoe/quantized_models_gptq_shared_intermediate`. When the second strategy (`int8_gptq`) runs, it detects this existing checkpoint and skips GPTQ calibration entirely:

```
INFO  GPTQ intermediate already exists at mxmoe/quantized_models_gptq_shared_intermediate — reusing
```

This saves ~62.5 minutes of redundant GPTQ computation.

### 5.3 CPU-Only Pass 2

Pass 2 runs entirely on CPU to avoid GPU memory conflicts:

```
INFO  Loading model onto CPU only: mxmoe/quantized_models_gptq_shared_intermediate
INFO  ✓ Patched dispatch_model in data_free pipeline
INFO  [patch] dispatch_model → no-op (CPU)
```

The `data_free` pipeline (both FP8_DYNAMIC and INT8_W8A16 are data-free — they don't need calibration data) is patched to prevent `accelerate.dispatch_model()` from trying to move the model to GPU.

---

## 6. Compression Results

### 6.1 Final Model Sizes

| Strategy | On-Disk Size | Compression Ratio | Shards |
|---|---|---|---|
| **fp8_gptq** | 34.62 GB | **1.73×** (vs ~60 GB BF16) | 8 |
| **int8_gptq** | 34.62 GB | **1.73×** (vs ~60 GB BF16) | 8 |

Both strategies produce **nearly identical** on-disk sizes because both use 8-bit quantization for HIGH/MEDIUM experts (FP8 and INT8 are both 8-bit) and INT4 GPTQ for LOW experts.

### 6.2 Quantization Format Details

| Component | fp8_gptq Format | int8_gptq Format |
|---|---|---|
| **Attention (all layers)** | FP8 E4M3 dynamic | INT8 per-channel symmetric |
| **Shared experts (18)** | FP8 E4M3 dynamic | INT8 per-channel symmetric |
| **HIGH experts (1,013)** | FP8 E4M3 dynamic | INT8 per-channel symmetric |
| **MEDIUM experts (1,030)** | FP8 E4M3 dynamic | INT8 per-channel symmetric |
| **LOW experts (261)** | GPTQ W4A16 (packed INT4) | GPTQ W4A16 (packed INT4) |
| **lm_head** | Unquantized (BF16) | Unquantized (BF16) |
| **MLP gates** | Unquantized (BF16) | Unquantized (BF16) |

### 6.3 GPTQ Calibration Details

| Parameter | Value |
|---|---|
| **Algorithm** | GPTQ (Generative Pre-trained Transformer Quantization) |
| **Weight Bits** | 4 (W4A16 — 4-bit weights, 16-bit activations) |
| **Calibration Dataset** | `dataset/sangraha_verified` |
| **Calibration Samples** | 256 sequences × 1,024 tokens |
| **Group Size** | Default (128) |
| **GPTQ Computation Time** | 3,173.4 seconds (~52.9 minutes) |
| **Total GPTQ Time (incl. save)** | 3,748.9 seconds (~62.5 minutes) |

---

## 7. Detailed Execution Log Analysis

### 7.1 Execution Timeline

```
19:11:13  Module 2 starts
19:11:13  2a: Recipe generation (instant)
           ├── Classification: HIGH=1013, MEDIUM=1030, LOW=261
           ├── FP8 targets: 6224, INT8 targets: 6224, GPTQ targets: 783
           └── Strategies: ['fp8_gptq', 'int8_gptq']
19:11:13  2b_fp8_gptq: Start
           ├── Pass 1: GPTQ W4A16
           │   ├── 19:12:13  Model loaded onto GPU (60s)
           │   ├── 19:12:15  Calibration tokenized (256 × 1024)
           │   ├── 20:05:11  GPTQ pass completed (3173.4s)
           │   ├── 20:07:11  Model offloaded to CPU (120s)
           │   └── 20:13:40  Saved to mxmoe/quantized_models_gptq_shared_intermediate
           └── Pass 2: FP8_DYNAMIC
               ├── 20:15:26  Loaded intermediate onto CPU
               ├── 20:19:56  FP8 pass completed (270.2s)
               └── 20:35:00  Saved to mxmoe/quantized_models_fp8_gptq (34.62 GB)
20:35:01  2b_fp8_gptq: COMPLETED (5028.0s)
20:35:01  GPU cleanup (0.5s)
20:35:01  2b_int8_gptq: Start
           ├── GPTQ intermediate already exists — reusing
           └── Pass 2: INT8 W8A16
               ├── 20:36:45  Loaded intermediate onto CPU
               ├── 20:41:14  INT8 pass completed (269.2s)
               └── 21:04:05  Saved to mxmoe/quantized_models_int8_gptq (34.62 GB)
21:04:05  2b_int8_gptq: COMPLETED (1744.1s)
21:04:06  Module 2 finished — success in 6773.2s
21:04:06  Tests: ALL PASS [4/4]
```

### 7.2 GPU Memory During GPTQ

| Checkpoint | GPU 0 | GPU 1 |
|---|---|---|
| After model load (GPTQ start) | ~71.9 GiB / 79.2 GiB | ~31.4 GiB / 79.2 GiB |
| After GPTQ save | 71.9 GiB | 31.4 GiB |

### 7.3 Warnings

| Warning | Status |
|---|---|
| `No HF token found` | ⚠️ Benign — model loaded from local cache |

---

## 8. Output Artifacts

All Module 2 outputs are stored under `mxmoe/outputs/module_2_synthesis/`:

```
mxmoe/outputs/module_2_synthesis/
├── logs/
│   └── mxmoe_module_2_20260530_191113.log        (14.2 KB, 118 lines)
├── plots/
│   └── (empty — Module 2 produces models, not visualizations)
└── results/
    ├── precision_recipe.json                       (675.0 KB) ← Full recipe with per-module targets
    ├── compression_report_fp8_gptq.json            (0.6 KB)   ← FP8+GPTQ compression summary
    └── compression_report_int8_gptq.json           (0.6 KB)   ← INT8+GPTQ compression summary
```

Additionally, the quantized model checkpoints:
- `mxmoe/quantized_models_fp8_gptq/` — FP8+GPTQ compressed model (~34.62 GB, 8 shards)
- `mxmoe/quantized_models_int8_gptq/` — INT8+GPTQ compressed model (~34.62 GB, 8 shards)
- `mxmoe/quantized_models_gptq_shared_intermediate/` — Shared GPTQ Pass 1 intermediate

---

## 9. Key Findings & Downstream Impact

### 9.1 Summary

1. **Both strategies achieved 1.73× compression** — From ~60 GB (BF16) to 34.62 GB, a 25.4 GB reduction per model variant.

2. **GPTQ intermediate sharing saved ~62.5 minutes** — The `int8_gptq` strategy reused the GPTQ checkpoint, reducing its total time from ~90 min to ~29 min.

3. **CPU-only Pass 2 is effective** — Both FP8 and INT8 overlays completed in ~4.5 minutes on CPU, proving that data-free quantization doesn't need GPU acceleration.

4. **The 2-pass architecture correctly preserves GPTQ integrity** — The 783 GPTQ-quantized modules are placed in the ignore list during Pass 2, ensuring they are not re-quantized.

### 9.2 How This Feeds Module 3

Module 3 (Evaluation & Ablation) loads both compressed model checkpoints and:

- Measures **perplexity** on WikiText-2 (sliding window)
- Runs **6 standard benchmarks** (MMLU, HellaSwag, Winogrande, ARC-Challenge, GSM8K, TruthfulQA)
- Performs **ablation analysis** to detect the accuracy floor

See [Module 3 — Evaluation & Ablation](module3_evaluation.md) for the evaluation results.

---

*Generated from [compression_report_fp8_gptq.json](../../mxmoe/outputs/module_2_synthesis/results/compression_report_fp8_gptq.json), [compression_report_int8_gptq.json](../../mxmoe/outputs/module_2_synthesis/results/compression_report_int8_gptq.json), and [mxmoe_module_2 log](../../mxmoe/outputs/module_2_synthesis/logs/mxmoe_module_2_20260530_191113.log).*
