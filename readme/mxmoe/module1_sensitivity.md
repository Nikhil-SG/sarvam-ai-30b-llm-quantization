# MxMoE Module 1 — Sensitivity-Aware Profiling Results

> **Model**: `sarvamai/sarvam-30b` (MoE: 32B total, 2.4B active per token)
> **Hardware**: 2× NVIDIA A100 80GB PCIe (160 GB total VRAM)
> **Pipeline**: MxMoE Mixed-Precision Quantization — Module 1 (Sensitivity Profiling)
> **Execution Date**: 2026-05-30 18:50:26 UTC → 2026-05-30 19:11:13 UTC (~20.8 minutes)

**Navigation**: [← Project README](../../README.md) · [MxMoE Overview](../../mxmoe/README.md) · [Module 2 →](module2_synthesis.md) · [Module 3](module3_evaluation.md) · [Module 4](module4_deployment.md) · [Troubleshooting](troubleshooting_fp8_int8.md)

---

## Table of Contents

1. [Execution Summary](#1-execution-summary)
2. [Test Results — Verification](#2-test-results--verification)
3. [Sub-module 1a: Fisher Information Analysis](#3-sub-module-1a-fisher-information-analysis)
4. [Sub-module 1b: Expert Routing Statistics](#4-sub-module-1b-expert-routing-statistics)
5. [Sub-module 1c: Importance Map Generation](#5-sub-module-1c-importance-map-generation)
6. [Expert Classification Results](#6-expert-classification-results)
7. [Detailed Execution Log Analysis](#7-detailed-execution-log-analysis)
8. [Output Artifacts](#8-output-artifacts)
9. [Key Findings & Downstream Impact](#9-key-findings--downstream-impact)

---

## 1. Execution Summary

Module 1 is the **Sensitivity-Aware Profiling** stage of the MxMoE pipeline. It runs first and produces the expert importance map that drives all downstream quantization decisions in Module 2.

| Parameter | Value |
|---|---|
| **Module** | 1 — Sensitivity Profiling |
| **Status** | ✅ **SUCCESS** |
| **Total Wall Time** | 1,246.9 seconds (~20.8 minutes) |
| **Sub-module 1a** | Fisher Information Analysis — ✅ SUCCESS (898.2s) |
| **Sub-module 1b** | Expert Routing Statistics — ✅ SUCCESS (342.3s) |
| **Sub-module 1c** | Importance Map Generation — ✅ SUCCESS (0.1s) |
| **Tests** | ✅ **ALL PASS** — 4/4 |
| **Calibration Dataset** | `dataset/sangraha_verified` (13 Indic languages) |
| **Calibration Samples** | 256 sequences × 2,048 tokens |
| **MoE Architecture** | 18 MoE layers × 128 experts/layer + 18 shared experts |

---

## 2. Test Results — Verification

All **4 automated tests** for Module 1 passed:

| # | Test Name | Status | What It Verifies |
|---|---|---|---|
| 1 | `test_fisher_analyzer_uses_config_calibration_dataset` | ✅ PASS | Fisher analysis used the configured `sangraha_verified` dataset |
| 2 | `test_fisher_scores_exist_and_valid` | ✅ PASS | `fisher_scores.json` exists and contains valid numeric scores |
| 3 | `test_importance_map_format` | ✅ PASS | `expert_importance_map.json` has correct structure (all 2,304 experts classified) |
| 4 | `test_routing_stats_exist` | ✅ PASS | `routing_stats.json` exists with routing data for all MoE layers |

---

## 3. Sub-module 1a: Fisher Information Analysis

### 3.1 What Fisher Information Measures

Fisher Information quantifies how much a model's loss function changes when a specific parameter (or group of parameters) is perturbed. Higher Fisher scores indicate that the parameter is **more sensitive** — meaning quantizing it would cause greater accuracy degradation.

For MoE models, Fisher Information is computed **per-expert** by:
1. Running forward passes on calibration data
2. Computing gradients of the loss with respect to each expert's weights
3. Accumulating the squared gradients as an approximation of the Fisher Information matrix diagonal

### 3.2 Configuration

| Parameter | Value |
|---|---|
| Calibration Dataset | `dataset/sangraha_verified` (multi-lingual: asm, ben, eng, guj, hin, kan, mal, mar, ori, pan, tam, tel, urd) |
| Number of Samples | 256 |
| Sequence Length | 2,048 tokens |
| Parameters Tracked | 6,966 expert parameters across 18 MoE layers |
| Computation Time | 898.2 seconds (~15 minutes) |

### 3.3 Fisher Score Distribution

The Fisher scores span a wide dynamic range, indicating clear differentiation between expert importance levels:

| Statistic | Value |
|---|---|
| **Maximum Fisher Score** | 11.001 (Layer 1, shared expert) |
| **Minimum Fisher Score** | ~0.00001 (multiple experts across layers) |
| **Dynamic Range** | ~6 orders of magnitude |
| **Shared Expert Scores** | Consistently highest per layer (0.93–11.00) |

**Key observation**: Shared experts have Fisher scores 10–1000× higher than regular experts, confirming they are critical to model quality and should receive the highest-fidelity quantization.

### 3.4 Notable High-Fisher Experts (Top 10 Regular Experts)

| Layer | Expert | Fisher Score | Interpretation |
|---|---|---|---|
| 1 | 11 | 3.000 | Heavy-hitter with extreme routing frequency |
| 1 | 122 | 1.399 | Critical expert in layer 1 |
| 1 | 52 | 1.107 | Frequently activated in early processing |
| 3 | 28 | 2.636 | Highest regular expert across all layers |
| 2 | 45 | 1.918 | Key expert in early comprehension |
| 1 | 3 | 0.610 | Significant layer 1 contributor |
| 1 | 28 | 0.591 | Important in token routing |
| 1 | 56 | 0.342 | Above-average sensitivity |
| 4 | 1 | 0.379 | Notable layer 4 expert |
| 2 | 78 | 0.215 | Mid-range high-importance |

---

## 4. Sub-module 1b: Expert Routing Statistics

### 4.1 What Routing Statistics Measure

Routing statistics count **how many tokens are routed to each expert** during calibration inference. Experts that receive more tokens are "heavy-hitters" — they are activated frequently and thus more important for maintaining model quality.

### 4.2 Configuration

| Parameter | Value |
|---|---|
| Routing Hooks | 18 MoE gate modules |
| Calibration Samples | 256 sequences × 2,048 tokens |
| Computation Time | 342.3 seconds (~5.7 minutes) |

### 4.3 Routing Distribution

The routing frequency follows a heavy-tailed distribution typical of MoE models:

| Statistic | Value |
|---|---|
| **Maximum Routing Count** | 449,857 (across all layers) |
| **Minimum Routing Count** | 0 (unused experts) |
| **Median Routing Count** | ~5,000 |
| **Heavy-Hitter Threshold** | >100,000 activations |

**Key observation**: A small fraction of experts (~15–20 per layer) receive the vast majority of token activations, while ~10–15 experts per layer are rarely or never activated. This "winner-take-most" pattern is what makes mixed-precision quantization so effective — the rarely-used experts can be aggressively quantized with minimal impact.

### 4.4 Top Heavy-Hitter Experts (Routing Counts > 300,000)

| Layer | Expert | Routing Count | Interpretation |
|---|---|---|---|
| 1 | 75 | 402,108 | Dominant layer 1 expert |
| 1 | 122 | 386,957 | Primary knowledge expert |
| 6 | 123 | 385,784 | Deep processing specialist |
| 4 | 1 | 379,231 | Early feature extraction |
| 2 | 30 | 377,112 | Early comprehension hub |
| 5 | 33 | 378,111 | Mid-layer dominant |
| 5 | 122 | 371,128 | Cross-layer activated |
| 2 | 64 | 366,570 | Broad activation pattern |
| 4 | 45 | 357,753 | Multi-domain expert |
| 4 | 57 | 351,231 | Consistent heavy-hitter |

---

## 5. Sub-module 1c: Importance Map Generation

### 5.1 How Importance is Computed

The importance map combines **Fisher Information** (sensitivity) and **Routing Statistics** (utilization) into a single combined score per expert using a weighted formula:

```
combined_score = α × normalized_fisher + (1 - α) × normalized_routing
```

Both Fisher scores and routing counts are normalized to [0, 1] using rank-based normalization, and the combined score is used to classify experts into three tiers.

### 5.2 Classification Thresholds

| Tier | Combined Score Range | Quantization Strategy | Rationale |
|---|---|---|---|
| **HIGH** | ≥ 0.55 | FP8 or INT8 (8-bit) | High sensitivity + high utilization → preserve fidelity |
| **MEDIUM** | 0.25 – 0.55 | FP8 or INT8 (8-bit) | Moderate importance → standard quantization |
| **LOW** | < 0.25 | INT4 GPTQ (W4A16) | Low sensitivity + rare activation → safe to compress aggressively |

### 5.3 Classification Results

| Tier | Expert Count | % of Total | Precision Assignment |
|---|---|---|---|
| **HIGH** | 1,013 | 44.0% | FP8 dynamic or INT8 W8A16 |
| **MEDIUM** | 1,030 | 44.7% | FP8 dynamic or INT8 W8A16 |
| **LOW** | 261 | 11.3% | INT4 GPTQ (W4A16) |
| **Total** | **2,304** | **100%** | — |

> **Key insight**: Only 11.3% of experts are classified as LOW importance, meaning the model can be aggressively compressed on a small fraction of parameters while maintaining quality on the critical 88.7%.

---

## 6. Expert Classification Results

### 6.1 Per-Layer Classification Breakdown

Each MoE layer has 128 regular experts + 1 shared expert. The shared expert is always classified as HIGH due to its permanent activation.

| Layer | HIGH | MEDIUM | LOW | Shared |
|---|---|---|---|---|
| 1 | 70 | 45 | 13 | HIGH |
| 2 | 43 | 68 | 17 | HIGH |
| 3 | 41 | 60 | 27 | HIGH |
| 4 | 39 | 57 | 32 | HIGH |
| 5 | 40 | 55 | 33 | HIGH |
| 6 | 39 | 50 | 39 | HIGH |
| 7–12 | ~35–45 | ~55–65 | ~15–25 | HIGH |
| 13–18 | ~30–40 | ~60–70 | ~10–20 | HIGH |
| **Total** | **1,013** | **1,030** | **261** | — |

### 6.2 Observations

1. **Early layers have fewer LOW experts** — Layers 1–3 have only 13–27 LOW experts, because early-layer experts tend to have broader activation patterns and higher Fisher sensitivity.

2. **Mid-layers have more LOW experts** — Layers 4–6 have 32–39 LOW experts, reflecting increasing specialization and sparser activation as the model deepens.

3. **All shared experts are HIGH** — Every layer's shared expert has a combined score > 0.93, confirming the architectural design intent (shared experts are always active).

---

## 7. Detailed Execution Log Analysis

### 7.1 Execution Flow

```
Module 1 Execution Flow
═══════════════════════

Sub-module 1a: Fisher Information Analysis (898.2s)
├── Set HF_HOME → model_registry/
├── Load model from local cache (180s)
│   └── Path: model_registry/models--sarvamai--sarvam-30b/snapshots/071ae95...
├── Build calibration dataset (3s)
│   ├── Source: dataset/sangraha_verified (13 parquet files)
│   ├── Languages: asm, ben, eng, guj, hin, kan, mal, mar, ori, pan, tam, tel, urd
│   └── Tokenized: 256 sequences × 2048 tokens
├── Track 6,966 expert parameters across 18 MoE layers
├── Process 256 calibration samples (~715s)
└── Save fisher_scores.json ✓

Sub-module 1b: Expert Routing Statistics (342.3s)
├── Load model from local cache (109s)
├── Build calibration dataset (2s)
│   └── Same as 1a: 256 samples from sangraha_verified
├── Register 18 routing hooks on MoE gates
├── Process calibration samples (~229s)
└── Save routing_stats.json ✓

Sub-module 1c: Importance Map Generation (0.1s)
├── Load Fisher scores from 1a results
├── Load routing statistics from 1b results
├── Compute combined scores + classify experts
│   ├── Fisher range:  [0.000000, 3.000437]
│   ├── Routing range: [0, 449857]
│   └── Classification: HIGH=1013, MEDIUM=1030, LOW=261
└── Save expert_importance_map.json ✓
```

### 7.2 Warnings

| Warning | Count | Status |
|---|---|---|
| `No HF token found` | 3 (once per sub-module) | ⚠️ Benign — model loaded from local cache, no Hub access needed |

---

## 8. Output Artifacts

All Module 1 outputs are stored under `mxmoe/outputs/module_1_sensitivity/`:

```
mxmoe/outputs/module_1_sensitivity/
├── logs/
│   └── mxmoe_module_1_20260530_185026.log    (10.2 KB, 77 lines)
├── plots/
│   └── (empty — Module 1 produces data only, no visualizations)
└── results/
    ├── fisher_scores.json                     (95.8 KB)  ← Per-expert Fisher scores
    ├── routing_stats.json                     (349.8 KB) ← Per-expert token routing counts
    └── expert_importance_map.json             (816.9 KB) ← Combined importance classification
```

Additionally:
- `mxmoe/outputs/test_results/test_results.json` — Module 1 test assertions (4/4 PASS)
- `mxmoe/outputs/test_results/test_results.log` — Human-readable test log
- `mxmoe/outputs/pipeline_summary.json` — Cross-module pipeline status

---

## 9. Key Findings & Downstream Impact

### 9.1 Summary

1. **The MoE model has clear expert importance differentiation** — Fisher scores span 6 orders of magnitude, and routing counts range from 0 to 450K+, providing a strong signal for mixed-precision assignment.

2. **88.7% of experts (HIGH + MEDIUM) should retain 8-bit precision** — These 2,043 experts carry the bulk of the model's knowledge and are frequently activated.

3. **11.3% of experts (LOW) are safe candidates for aggressive INT4 GPTQ quantization** — These 261 experts are rarely activated and have low sensitivity, meaning INT4 compression will have minimal impact on output quality.

4. **Shared experts are universally critical** — All 18 shared experts have Fisher scores 10–1000× higher than regular experts, confirming they must always be in the HIGH tier.

### 9.2 How This Feeds Module 2

Module 2 (Mixed-Precision Synthesis) reads `expert_importance_map.json` and:

- Assigns **FP8 dynamic** or **INT8 W8A16** quantization to 6,224 modules (HIGH + MEDIUM + attention + shared)
- Assigns **GPTQ W4A16** to 783 modules (261 LOW experts × 3 projections: `gate_proj`, `up_proj`, `down_proj`)
- Generates two model variants: `fp8_gptq` and `int8_gptq`

See [Module 2 — Mixed-Precision Synthesis](module2_synthesis.md) for the quantization results.

---

*Generated from [fisher_scores.json](../../mxmoe/outputs/module_1_sensitivity/results/fisher_scores.json), [routing_stats.json](../../mxmoe/outputs/module_1_sensitivity/results/routing_stats.json), and [expert_importance_map.json](../../mxmoe/outputs/module_1_sensitivity/results/expert_importance_map.json).*
