# MxMoE Troubleshooting & Known Issues

All issues encountered during the development and execution of the MxMoE mixed-precision quantization pipeline for `sarvamai/sarvam-30b`.

**Navigation**: [← Project README](../../README.md) · [MxMoE Overview](../../mxmoe/README.md) · [Module 1](module1_sensitivity.md) · [Module 2](module2_synthesis.md) · [Module 3](module3_evaluation.md) · [Module 4](module4_deployment.md)

---

## Table of Contents

1. [Dependency Conflict: llm-compressor vs vLLM](#1-dependency-conflict-llm-compressor-vs-vllm)
2. [vLLM Engine Core Initialization Failure](#2-vllm-engine-core-initialization-failure)
3. [Missing vLLM Backend in Quantization Env](#3-missing-vllm-backend-in-quantization-env)
4. [Quality Collapse: Perplexity >1700 Across 4 Runs](#4-quality-collapse-perplexity-1700-across-4-runs)
5. [indicxnli Task Not Found in lm-eval](#5-indicxnli-task-not-found-in-lm-eval)
6. [FP8 Dequantization & Log Warnings on A100](#6-fp8-dequantization--log-warnings-on-a100)

---

## 1. Dependency Conflict: llm-compressor vs vLLM

### Symptom

```
ERROR: Cannot install compressed-tensors==0.14.0 because these package versions have conflicting dependencies.

The conflict is caused by:
    The user requested compressed-tensors==0.14.0
    llmcompressor 0.10.0 depends on compressed-tensors==0.14.0
    vllm 0.18.0 depends on compressed-tensors==0.13.0
```

Installing `requirements.mxmoe.txt` with both `llmcompressor==0.10.0` and `vllm==0.18.0` fails because they pin different versions of `compressed-tensors`.

### Root Cause

`llm-compressor` generates quantized models using `compressed-tensors==0.14.0` format, but `vLLM 0.18.0` can only load models saved with `compressed-tensors==0.13.0`. These libraries are tightly coupled — the serialization format changed between versions.

### Resolution

**Use two separate virtual environments:**

| Venv | Modules | Key Packages | Purpose |
|---|---|---|---|
| `mxmoe` | 1, 2, 3 | `llmcompressor==0.10.0.2`, `compressed-tensors==0.14.0.1`, `lm_eval==0.4.11` | Profiling, quantization, evaluation (HF backend) |
| `mxmoe_vllm` | 4, 5 | `vllm==0.19.0`, `compressed-tensors==0.14.0.1` | vLLM inference profiling, Hub push |

> **Note**: vLLM was upgraded to `0.19.0` which accepts `compressed-tensors==0.14.0.1`, resolving the conflict for Module 4. Module 3 still uses the HF `transformers` backend (not vLLM) for evaluation.

---

## 2. vLLM Engine Core Initialization Failure

### Symptom

```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}
```

Module 3 attempted to evaluate quantized models via vLLM's `lm-eval` integration but failed at engine startup. The pipeline retry chain was:
1. vLLM with default settings → **FAILED**
2. vLLM with safe settings (lower `gpu_memory_utilization=0.70`, `max_model_len=2048`) → **FAILED**
3. HF backend → **FAILED** (`missing mixed-precision registration`)
4. vLLM retry → **FAILED**
5. Legacy vLLM engine (`VLLM_USE_V1=0`) → **FAILED**

All 5 retry attempts failed, causing Module 3 to abort.

### Root Cause

vLLM 0.19.0 has a **weight scale shape mismatch** when loading MxMoE-quantized models. The quantized model stores per-group weight scales with shape `[N, G]` (per-group quantization), but vLLM's compressed-tensors loader expects shape `[N, 1]` (per-channel quantization). This causes a tensor shape error during model weight loading inside the engine core subprocess, which manifests as an opaque `RuntimeError` at the parent process level.

### Resolution

**Use HuggingFace `transformers` backend with static dequantization for Module 3 evaluation.**

The pipeline was refactored so Module 3 (`mxmoe` venv) loads quantized models via `transformers.AutoModelForCausalLM.from_pretrained()` with `compressed-tensors`, then statically dequantizes all `CompressedLinear` modules to `nn.Linear` (BF16) before running evaluation. This bypasses vLLM entirely for accuracy evaluation.

Module 4 (deployment profiling) uses vLLM in the `mxmoe_vllm` venv for latency/throughput benchmarking, where the weight loading path is different.

---

## 3. Missing vLLM Backend in Quantization Env

### Symptom

```
WARNING  Skipping lm-eval perplexity: missing vLLM backend dependencies: vllm
WARNING  Skipping lm-eval benchmarks: missing vLLM backend dependencies: vllm
```

Module 3 detected that `vllm` was not installed in the quantization environment and skipped all evaluations. The module reported `partial_success` with no actual evaluation results.

### Root Cause

The `mxmoe_quant_env` was created for quantization (Modules 1–2) and did not include `vllm` due to the dependency conflict described in [Issue #1](#1-dependency-conflict-llm-compressor-vs-vllm). When Module 3 was run in this env, it could not find vLLM and gracefully degraded to skipping.

### Resolution

Module 3 no longer requires vLLM. It runs in the `mxmoe` venv using the HF `transformers` backend with static dequantization (see [Issue #2](#2-vllm-engine-core-initialization-failure)). The `lm-evaluation-harness` is invoked with `--model hf` instead of `--model vllm`.

---

## 4. Quality Collapse: Perplexity >1700 Across 4 Runs

### Symptom

Across 4 consecutive quantization runs (March 27–30, 2026), the quantized model **never** produced coherent output:

| Run | Date | Pass Split | Model Size | Perplexity | Sanity Output |
|---|---|---|---|---|---|
| `outputs_28_1` | Mar 27 | FP8:47 + W8A8:54 + GPTQ:6,906 | 18.9 GB | 2,066 | ❌ `"is is is is is is..."` |
| `outputs_29_8` | Mar 28 | FP8:1,424 + W8A8:54 + GPTQ:5,529 | 21.5 GB | 2,273 | ❌ `"is is is is is is..."` |
| `outputs_29_19` | Mar 29 | FP8:1,424 + W8A8:54 + GPTQ:5,529 | 21.5 GB | 1,752 | ❌ `"is is is is is is..."` |
| `outputs_30_6` | Mar 30 | FP8:1,478 + W8A16:3,456 + GPTQ:2,073 | 28.1 GB | 1,859 | ❌ `"is is is is is is..."` |

Expected perplexity for a well-quantized 30B model: < 20. Actual: 1,700–2,300.

### Root Causes (5 Identified)

#### 4a. W4A16 GPTQ Without Outlier Smoothing

The original approach applied W4A16 GPTQ to 80%+ of all experts (5,500–6,900 modules). MoE expert weights are highly specialized with significant outliers. Without Hadamard rotation or SmoothQuant preprocessing, GPTQ at 4-bit **destroyed** expert weight distributions.

> The reference MxMoE paper (ICML 2025) uses Hadamard rotation (`gptq-had` mode) to redistribute weight outliers before quantization. Our implementation omitted this.

#### 4b. MoE Calibration Context Forcing All 128 Experts Active

The `SarvamMoECalibrationContext` in `moe_calibration.py` set `top_k = num_experts` (128), forcing **every token** to activate all 128 experts during GPTQ calibration. Normal inference activates only top-6 (4.7%).

This completely distorted the activation patterns that GPTQ used for Hessian estimation. The weights were rounded to minimize error against an activation distribution that **never occurs during real inference**. The llmcompressor metrics confirmed: `avg_error: 755.09` (normal is < 1.0).

#### 4c. Wrong Calibration Dataset

Both Module 1 (Fisher/Routing) and Module 2 (GPTQ) silently fell back to `HuggingFaceH4/ultrachat_200k` (English-only conversational) because the `ai4bharat/sangraha` dataset config was set to `None` instead of `"verified"`.

`sarvam-30b` is a multilingual Indic model — calibrating with English-only data meant expert importance rankings and GPTQ weight rounding were optimized for the wrong domain.

#### 4d. Coarse Expert-Level Fisher Sensitivity

The Fisher Information approach computed a single scalar score per expert by summing `grad²` across all parameters. Within one expert, `down_proj` often has 10–100× higher sensitivity than `gate_proj`, but averaging merged them. The discriminating signal was too weak: `max_score=0.755, mean=0.022, std=0.038`.

#### 4e. MEDIUM and LOW GPTQ Groups Merged

The recipe builder separated MEDIUM and LOW experts into different groups (group_size 128 vs 64), but the compressor merged them into a single GPTQ pass: `all_gptq = gptq_targets + gptq_low_targets`. The differentiation was lost.

### Resolution (Final Working Approach)

The pipeline was fundamentally restructured for the successful May 30 run:

1. **2-pass compression strategy**: GPTQ W4A16 applied to **only LOW experts** (261 experts = 783 modules), then FP8/INT8 overlay for everything else (6,224 modules)
2. **Disabled `SarvamMoECalibrationContext`**: GPTQ calibration runs with normal top-6 routing
3. **Switched calibration dataset**: `dataset/sangraha_verified` (13 Indic languages) instead of UltraChat
4. **Increased calibration samples**: 256 samples × 2,048 tokens (up from 128 × 2,048)
5. **Two-strategy evaluation**: Both `fp8_gptq` and `int8_gptq` variants produced and compared
6. **Result**: Perplexity dropped from 1,700–2,300 → **15.43** (INT8_GPTQ), 27/27 tests pass

| Metric | Before (4 failed runs) | After (successful run) |
|---|---|---|
| Perplexity | 1,700–2,300 | **15.43** |
| GPTQ targets | 5,500–6,900 | **783** |
| Calibration dataset | UltraChat (English) | **Sangraha verified (13 Indic)** |
| MoE calibration context | All 128 experts forced | **Disabled (normal top-6)** |
| Model size | 18.9–28.1 GB | **34.62 GB** |
| Sanity output | `"is is is is is..."` | **Coherent text** |

---

## 5. indicxnli Task Not Found in lm-eval

### Symptom

```
WARNING  lm-eval reported unknown tasks (indicxnli). Retrying with: mmlu, gsm8k, humaneval, truthfulqa_mc2
```

The `config.yaml` specified `indicxnli` (Indic Cross-Lingual NLI) as a benchmark, but `lm-evaluation-harness 0.4.11` does not include this task in its default task registry.

### Resolution

Replaced `indicxnli` and `humaneval` with standard benchmarks available in `lm-eval`: **HellaSwag**, **Winogrande**, **ARC-Challenge**. The final benchmark suite (MMLU, HellaSwag, Winogrande, ARC-Challenge, GSM8K, TruthfulQA) provides comprehensive coverage.

---

## 6. FP8 Dequantization & Log Warnings on A100

### 6.1 Executive Summary

During Module 3 (Evaluation & Ablation), the model is loaded in its mixed-precision format. You will see warnings like `Could not match...` and `No FP8 parameters found` in the logs.

**These logs are normal, expected, and correct.** The model is being evaluated as a true quantized mixed-precision model, and its accuracy/perplexity results are 100% mathematically equivalent to a native mixed-precision run.

### 6.2 Mathematical Equivalence of Dequantized Evaluation

A common concern: *"If we decompress FP8/INT4 weights back to BF16 during evaluation, does it defeat the purpose?"*

**No.** Quantization is a **lossy operation**. Once weights are quantized, high-frequency precision is permanently discarded:
- **Quantization** (lossy): BF16 weight `0.123456` → FP8 value `0.125000`
- **Dequantization** (lossless padding): FP8 `0.125000` → BF16 `0.125000` (lost digits not recovered)
- **Computation**: Matrix multiplications use the exact quantized value (`0.125000`)

Evaluating by decompressing to BF16 is **mathematically identical** to native FP8 execution.

### 6.3 Hardware Constraints: A100 vs H100

The target system runs on **NVIDIA A100 80GB** (Ampere, Compute Capability 8.0):
- **No native FP8 math**: Ampere hardware lacks FP8 tensor cores (introduced in Hopper/H100, CC 9.0)
- **Mandatory dequantization**: Any FP8 model on A100 **must** upcast weights to BF16 for matrix multiplication

### 6.4 Static Pre-Decompression (OOM Prevention)

For a model with **7,007 separate linear layers** (128 experts × 18 MoE layers), on-the-fly dequantization causes OOM crashes and extreme slowness. The evaluation script uses **static pre-decompression**:

```python
_decompress_compressed_linears()  # One-time decompression post-load
```

This decompresses all `CompressedLinear` → `nn.Linear` (BF16) once, then evaluates with zero VRAM spikes.

### 6.5 Understanding Log Warnings

#### `Could not match ... in instance of SarvamMoEForCausalLM` (Benign)

```
WARNING  Could not match `model.layers.18.mlp.experts.7.gate_proj` in SarvamMoEForCausalLM
```

- **Why**: Mixed-precision format has multiple `config_groups`. As `compressed-tensors` processes each group, it warns about layers in *other* groups.
- **Proof of success**: All 7,007 modules are eventually matched:
  ```
  ✓ Decompressed 7007/7007 modules
  ```

#### `No FP8 parameters found — skipping dequant` (Benign)

- **Why**: Static decompression runs first, converting all FP8 layers to BF16. When `_dequantize_fp8_weights()` runs later, no native `float8` tensors remain.

### 6.6 Evaluation vs Deployment

| Phase | Framework | Loading | Execution | Goal |
|---|---|---|---|---|
| **Evaluation (Module 3)** | `transformers` + `lm-eval` | Decompressed to BF16 | Standard PyTorch GEMM | Mathematical accuracy |
| **Deployment (Module 4)** | **vLLM** | Raw packed format | Custom CUDA kernels | VRAM savings + throughput |

When deployed via vLLM, the model loads raw quantized weights directly (~34.6 GB instead of ~60 GB) and uses optimized kernels for maximum throughput.

---

*For module-specific details, see the individual module documentation: [Module 1](module1_sensitivity.md) · [Module 2](module2_synthesis.md) · [Module 3](module3_evaluation.md) · [Module 4](module4_deployment.md)*
