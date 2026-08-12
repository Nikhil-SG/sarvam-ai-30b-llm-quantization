# Research Pipeline — Sarvam-30B Quantization Comparative Analysis

> Systematic quantization research across 4 methods (INT8, FP8, NF4, GPTQ) for [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b), with publication-ready comparative analysis.

**Navigation**: [← Project README](../README.md)

---

## Pipeline Overview

The research pipeline builds a BF16 reference, quantizes the model with 4 methods, then systematically compares them across weight fidelity, inference performance, and downstream accuracy.

```
Module 1         Module 2            Module 3          Module 4          Module 5          Module 6
BF16         →   Quantization    →   Weight        →   Inference     →   Evaluation    →   Layer-wise
Baseline         Matrix              Introspection     Profiling         & Pareto          Visualization
(Reference)      (INT8/FP8/NF4/GPTQ) (MSE/Outliers)   (Latency/VRAM)   (PPL/MMLU)        (Histograms)
```

---

## Pipeline Execution Summary

| Module | Document | Status | Wall Time |
|---|---|---|---|
| **Module 1** | [BF16 Baseline](../readme/research/module1_baseline.md) | ✅ COMPLETED | 160.8s (~2.7 min) |
| **Module 2** | [Quantization Matrix](../readme/research/module2_quantization.md) | ✅ COMPLETED | 17,299.5s (~4.8 hr) |
| **Module 3** | [Weight Introspection](../readme/research/module3_analysis.md) | ✅ COMPLETED | 468.1s (~7.8 min) |
| **Module 4** | [Inference Profiling](../readme/research/module4_profiling.md) | ✅ COMPLETED | 1,018.1s (~17.0 min) |
| **Module 5** | [Evaluation & Pareto](../readme/research/module5_evaluation.md) | ✅ COMPLETED | 71,139.2s (~19.8 hr) |
| **Module 6** | [Layer-wise Visualization](../readme/research/module6_visualization.md) | ✅ COMPLETED | 46.3s (~0.8 min) |
| **Total** | **Full Pipeline (Modules 1–6)** | **✅ COMPLETED** | **90,131.9s (~25.0 hr)** |
| **Troubleshooting** | [Issues & Resolutions](../readme/research/troubleshooting.md) | 9 issues documented | — |

---

## Key Results

### Quantization Comparison

| Method | Perplexity | MMLU | Model Size (GB) | Peak VRAM (GB) | Throughput BS=1 | Single-GPU? |
|---|---:|---:|---:|---:|---:|---|
| **INT8** | **11.95** | **68.20%** | 32.01 | 32.23 | 5.11 tok/s | ✅ |
| **BF16** | 12.07 | 66.12% | 119.78 | 60.25 | 17.72 tok/s | ❌ |
| **FP8** | 12.27 | 64.83% | 31.98 | 59.93 | 4.74 tok/s | ❌ |
| **NF4** | 12.40 | 67.11% | 18.44 | 18.66 | 8.97 tok/s | ✅ |
| **GPTQ** | 12.38 | 66.93% | 18.62 | 20.63 | 7.66 tok/s | ✅ |

### Weight Fidelity (MSE vs BF16)

| Method | Mean MSE | Fidelity Ranking |
|---|---:|---|
| **INT8** | 6.18×10⁻⁸ | 🥇 Best |
| **FP8** | 4.36×10⁻⁷ | 🥈 |
| **NF4** | 5.38×10⁻⁶ | 🥉 |
| **GPTQ** | 1.12×10⁻⁴ | 4th |

### Recommendation

> **INT8 is the best overall quantization method for Sarvam-30B** — it achieves lower perplexity and higher MMLU accuracy than the BF16 baseline while halving VRAM usage to fit on a single A100 80GB. NF4 is the best choice when maximum VRAM compression is needed (3.2× reduction with <1% quality loss).

---

## Usage

Run from the **project root** directory:

```bash
python -m pipelines.research.pipeline                          # Run ALL modules (1-6)
python -m pipelines.research.pipeline --module 1               # BF16 Baseline only
python -m pipelines.research.pipeline --module 2               # All quantizers
python -m pipelines.research.pipeline --module 2 --quantizer gptq int8
python -m pipelines.research.pipeline --module 1 2 3           # Baseline + Quantise + Analysis
python -m pipelines.research.pipeline --config configs/research.yaml
```

---

## Configuration

Edit `configs/research.yaml` for runtime configuration.

Model storage is shared with MxMoE via `./model_registry`, so once the Sarvam-30B model is downloaded it is reused across both parts.

Backward compatibility: legacy local folders (`./Base_model` or `./Model`) are auto-migrated to `./model_registry` when possible.

---

## Outputs

```
research/outputs/
├── module_1_baseline/        # BF16 results + model architecture
├── module_2_quantization/    # Per-method quantization results
├── module_3_analysis/        # Weight analysis, MSE heatmaps, outlier reports
├── module_4_profiling/       # Latency, VRAM, disk measurements
├── module_5_evaluation/      # Perplexity, benchmarks, Pareto analysis
├── module_6_visualization/   # Publication-ready layer histograms
├── shared_weights/           # NPZ weight caches (bf16, int8, fp8, nf4, gptq)
├── test_results/             # Aggregated test results
└── pipeline_summary.json     # Aggregate status across all runs

research/quantized_models/    # Saved quantized checkpoints
├── int8_quantized/
├── fp8_quantized/
├── nf4_quantized/
└── gptq_quantized/
```

---

### Hardware

| Property | Value |
|---|---|
| **GPU** | 2× NVIDIA A100 80GB PCIe (160 GB total VRAM) |
| **Evaluation Framework** | `lm-evaluation-harness` |
| **Quantization Libraries** | `bitsandbytes`, `optimum-quanto`, `auto-gptq` |

*Pipeline executed 2026-04-27 on 2× NVIDIA A100 80GB PCIe. See [pipeline_summary.json](outputs/pipeline_summary.json) for raw timing data.*
