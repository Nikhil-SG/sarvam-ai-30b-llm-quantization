# MxMoE — Mixed-Precision Mixture-of-Experts Quantization

> Sensitivity-aware heterogeneous mixed-precision quantization pipeline for [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b).

**Navigation**: [← Project README](../README.md)

---

## Pipeline Overview

The MxMoE pipeline profiles each expert's importance using Fisher Information and routing statistics, assigns per-expert quantization precision, evaluates the compressed models, and benchmarks them for deployment readiness.

```
Module 1              Module 2              Module 3               Module 4              Module 5
Sensitivity      →    Mixed-Precision   →   Evaluation         →   Deployment        →   Hub
Profiling             Synthesis             & Ablation              Readiness              Publication
(Fisher + Routing)    (GPTQ + FP8/INT8)    (PPL + Benchmarks)     (Latency + VRAM)       (HF Push)
```

### Module 1: Sensitivity-Aware Profiling & Fisher Information Analysis
- Compute Hessian-based sensitivity / Fisher Information for each expert
- Map expert routing statistics (heavy hitters vs long tail)
- Generate layer-wise Importance Map classifying experts as HIGH / MEDIUM / LOW

### Module 2: Heterogeneous Mixed-Precision Synthesis
- 2-pass compression strategy using `llm-compressor`:
  - **Pass 1 — GPTQ W4A16**: INT4 weight quantization for LOW-importance experts only
  - **Pass 2 — FP8/INT8 overlay**: 8-bit quantization for attention, shared experts, HIGH and MEDIUM experts
- Produces two model variants: `fp8_gptq` and `int8_gptq`

### Module 3: Multi-Objective Evaluation & Ablation Analysis
- Perplexity evaluation on WikiText-2 (sliding window)
- 6 standard benchmarks via `lm-evaluation-harness` (MMLU, HellaSwag, Winogrande, ARC-Challenge, GSM8K, TruthfulQA)
- Ablation study to detect the accuracy floor across bit-width variants

### Module 4: Deployment Readiness & Profiling
- Measure inference latency (tokens/sec) and VRAM efficiency on A100 via vLLM
- Pareto Efficiency Frontier analysis
- Draft Technical Model Card locally

### Module 5: Hub Publication & Dissemination
- Validate compressed artifacts and README
- Publish checkpoint to HuggingFace Hub

---

## Environment Setup

> [!IMPORTANT]
> Running all MxMoE modules in a single venv is **not possible** due to dependency conflicts between `llm-compressor` (requires `compressed-tensors==0.14.0`) and `vLLM` (requires `compressed-tensors==0.13.0`). Two separate venvs are required.

### 1. `mxmoe` — Modules 1–3 (Profiling + Quantization + Evaluation)

Key packages: `torch==2.10.0`, `transformers==4.57.6`, `llmcompressor==0.10.0.2`, `compressed-tensors==0.14.0.1`, `lm_eval==0.4.11`

```bash
python3 -m venv env/mxmoe
source env/mxmoe/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[mxmoe]"
```

### 2. `mxmoe_vllm` — Modules 4–5 (vLLM Inference + Hub Push)

Key packages: `torch==2.10.0`, `transformers==4.57.6`, `vllm==0.19.0`, `compressed-tensors==0.14.0.1`, `flashinfer-python==0.6.6`

```bash
python3 -m venv env/mxmoe_vllm
source env/mxmoe_vllm/bin/activate
pip install --upgrade pip setuptools wheel
pip install vllm==0.19.0
pip install --no-deps -e .
```

### Running Modules

```bash
# In mxmoe env (Modules 1–3)
source env/mxmoe/bin/activate
python -m pipelines.mxmoe.pipeline --config configs/mxmoe.yaml --module 1 2 3

# In mxmoe_vllm env (Modules 4–5)
source env/mxmoe_vllm/bin/activate
export VLLM_USE_V1=0  # Forces legacy engine to bypass HPC multiprocessing startup issues
python -m pipelines.mxmoe.pipeline --config configs/mxmoe.yaml --module 4 5
```

---

## Usage

Run from the **project root** directory:

```bash
python -m pipelines.mxmoe.pipeline                    # Run ALL modules (1-5)
python -m pipelines.mxmoe.pipeline --module 1         # Sensitivity profiling only
python -m pipelines.mxmoe.pipeline --module 1 2       # Profile + Quantize
python -m pipelines.mxmoe.pipeline --module 3 4       # Evaluate + Profile
python -m pipelines.mxmoe.pipeline --module 5         # Push to HuggingFace Hub
python -m pipelines.mxmoe.pipeline --config configs/mxmoe.yaml
```

---

## Key Results

### Pipeline Execution Summary

| Module | Document | Status | Wall Time | Tests |
|---|---|---|---|---|
| **Module 1** | [Sensitivity-Aware Profiling](../readme/mxmoe/module1_sensitivity.md) | ✅ SUCCESS | 1,246.9s (~20.8 min) | 4/4 PASS |
| **Module 2** | [Mixed-Precision Synthesis](../readme/mxmoe/module2_synthesis.md) | ✅ SUCCESS | 6,773.2s (~1.88 hr) | 4/4 PASS |
| **Module 3** | [Evaluation & Ablation](../readme/mxmoe/module3_evaluation.md) | ✅ SUCCESS | 123,204.1s (~34.2 hr) | 8/8 PASS |
| **Module 4** | [Deployment Readiness & Profiling](../readme/mxmoe/module4_deployment.md) | ✅ SUCCESS | 2,117.7s (~35.3 min) | 11/11 PASS |
| **Total** | **Full Pipeline (Modules 1–4)** | **✅ SUCCESS** | **~37.0 hours** | **27/27 PASS** |

### Model Compression

| Metric | Value |
|---|---|
| **Original Model Size** | ~60 GB (BF16) |
| **Compressed Model Size** | 34.62 GB (both strategies) |
| **Compression Ratio** | 1.73× |
| **Expert Classification** | HIGH: 1,013 · MEDIUM: 1,030 · LOW: 261 |
| **LOW Expert Quantization** | INT4 GPTQ (W4A16) |
| **HIGH/MEDIUM Quantization** | FP8 dynamic or INT8 W8A16 |

### Strategy Comparison — INT8_GPTQ vs FP8_GPTQ

| Metric | FP8_GPTQ | INT8_GPTQ | Winner |
|---|---|---|---|
| Perplexity (WikiText-2) | 16.13 | **15.43** | INT8 (−4.3%) |
| Composite Benchmark Score | 48.63% | **51.31%** | INT8 (+2.68 pts) |
| MMLU (General Reasoning) | 65.18% | **65.69%** | INT8 |
| GSM8K (Math) | 60.20% | **70.96%** | INT8 (+10.76 pts) |
| Throughput (BS=1) | 1.11 TPS | **3.05 TPS** | INT8 (2.75×) |
| VRAM Usage | 95.4 GB | **76.3 GB** | INT8 (−20%) |
| **Pareto Optimal?** | ❌ | **✅** | INT8 |

> **Recommendation**: INT8_GPTQ is the clear winner — it is faster, more accurate, and uses less VRAM than FP8_GPTQ at the same disk size.

---

## Reference Guides

| Document | Description |
|---|---|
| [Troubleshooting & Known Issues](../readme/mxmoe/troubleshooting_fp8_int8.md) | Dependency conflicts, vLLM failures, quality collapse root causes, FP8 dequantization on A100 |

---

## Configuration

Edit `configs/mxmoe.yaml` for MxMoE-specific settings.

Model storage is shared with the research pipeline via `./model_registry`, so Module 2 can reuse the same downloaded Sarvam-30B assets.

Backward compatibility: legacy local folders (`./Base_model` or `./Model`) are auto-migrated to `./model_registry` when possible.

---

## Outputs

```
mxmoe/outputs/
├── module_1_sensitivity/     # Fisher scores, routing stats, importance map
├── module_2_synthesis/       # Precision recipe, compression reports
├── module_3_evaluation/      # Perplexity, benchmarks, ablation, plots
├── module_4_deployment/      # Latency, VRAM, disk, Pareto, model card
├── module_5_publication/     # Hub push logs
├── shared_weights/           # Shared model weight caches
├── test_results/             # Aggregated test results across all modules
└── pipeline_summary.json     # Cross-module status summary

mxmoe/quantized_models/       # Saved mixed-precision checkpoints
```

---

## Shared Infrastructure

This pipeline reuses evaluation, profiling, and analysis tools from `src/`:
- `src/evaluation/` — Perplexity and benchmark evaluation
- `src/profiling/` — Latency, VRAM, disk profiling
- `src/analysis/` — MSE heatmaps, weight analysis
- `src/core/` — Config, logging, device management

---

### Hardware

| Property | Value |
|---|---|
| **GPU** | 2× NVIDIA A100 80GB PCIe (160 GB total VRAM) |
| **Calibration Dataset** | `dataset/sangraha_verified` (13 Indic languages) |
| **Evaluation Framework** | `lm-evaluation-harness` |
| **Quantization Tools** | `llm-compressor` 0.10.0 + `compressed-tensors` 0.14.0 |

*Pipeline executed 2026-05-30 to 2026-06-02 on 2× NVIDIA A100 80GB PCIe. See [pipeline_summary.json](outputs/pipeline_summary.json) for raw timing data.*
