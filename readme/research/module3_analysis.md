# Module 3: Weight Introspection — Distribution Analysis, MSE Heatmaps, Activation Outliers

> **Pipeline**: Research Quantization Pipeline — Module 3 (Weight Analysis)
> **Wall Time**: 468.1s (~7.8 min) — dominated by outlier detection (424.7s)

**Navigation**: [← Project README](../../README.md) · [Research Overview](../../research/README.md) · [← Module 2](module2_quantization.md) · [Module 4 →](module4_profiling.md) · [Module 5](module5_evaluation.md) · [Module 6](module6_visualization.md)

---

## 1. Purpose

Module 3 performs **internal model introspection** — it looks inside the weights and activations to understand *how* quantization affects the model, not just *whether* quality changes (that's Module 5's job). It answers three questions:

1. **Weight Distribution** (Submodule 3a): How do weight value distributions change after quantization? Do they still look Gaussian?
2. **MSE Heatmap** (Submodule 3b): Which layers have the highest quantization error? Are there systematic patterns?
3. **Activation Outliers** (Submodule 3c): Does quantization create new activation outliers, or remove existing ones?

---

## 2. Prerequisites — Weight Cache Validation

Before any analysis begins, Module 3 validates the weight caches from Modules 1 and 2 using `WeightCache.validate_caches()`:

- Checks that the BF16 reference cache exists (required for all MSE calculations)
- Checks that quantized format caches exist (int8, fp8, nf4, gptq)
- Verifies layer counts are consistent across all formats (all should have 97 cached layers)
- Reports any missing or corrupted cache files

In this run, validation passed cleanly: all 5 tags (bf16, int8, fp8, nf4, gptq) present, 97 layers each.

---

## 3. Submodule 3a — Weight Distribution Histograms

**Time**: 27.1s · **Output**: 14 histogram plots

**How it works**: The `WeightDistributionAnalyzer` loads weight samples from the NPZ caches and generates matplotlib histograms for each of the 14 target layers defined in the config:

1. For each target layer (e.g., `model.layers.9.attention.query_key_value`):
   - Load the weight samples from each format's NPZ cache
   - Compute histogram bins using numpy (typically 100–200 bins spanning the value range)
   - Overlay a fitted Gaussian curve (μ, σ from sample statistics)
   - Plot all formats on the same axes for direct comparison

2. Save each plot as a PNG at 150 DPI, 14×8 inches

**What the histograms reveal**: Neural network weights typically follow a near-Gaussian distribution centered around zero. Quantization transforms these distributions in characteristic ways:
- **INT8**: Distribution closely tracks BF16 — the 256 quantization levels are dense enough to preserve the continuous shape
- **FP8**: Similar fidelity to INT8, with slightly more noise in the tails due to the reduced mantissa (3 bits vs INT8's linear spacing)
- **NF4**: Clearly stepped distribution — only 16 discrete levels, placed at positions optimized for a Gaussian distribution. The overall envelope is preserved but the histogram shows 16 clear peaks
- **GPTQ**: Most aggressive transformation — some layers show shifted centroids and altered variance because GPTQ's Hessian-based rounding optimizes for output accuracy rather than weight fidelity

---

## 4. Submodule 3b — MSE Heatmap Analysis

**Time**: 15.8s · **4 quantizers analyzed** · **Method**: Mean squared error with log-scale colormap

**How it works**: The `MSEHeatmapAnalyzer` computes element-wise MSE between each quantized format and the BF16 reference for every cached layer:

```
MSE(layer, format) = mean((W_quantized - W_bf16)²)
```

For all 97 layers × 4 formats = 388 MSE values, organized into a 2D heatmap where:
- **X-axis**: Layer index (0–18) × projection type (attention, shared expert, routed expert)
- **Y-axis**: Quantization format (INT8, FP8, NF4, GPTQ)
- **Color**: Log-scale MSE value (blue = low error, red = high error)

### Results — MSE Ranking

| Quantizer | Mean MSE | Worst Layer | Worst MSE | Error Ratio vs INT8 |
|---|---:|---|---:|---:|
| **INT8** | 6.179×10⁻⁸ | `layer_17_attention.dense` | 9.576×10⁻⁸ | **1×** (reference) |
| **FP8** | 4.355×10⁻⁷ | `layer_17_attention.dense` | 6.045×10⁻⁷ | 7.0× |
| **NF4** | 5.377×10⁻⁶ | `layer_17_attention.dense` | 7.245×10⁻⁶ | 87× |
| **GPTQ** | 1.119×10⁻⁴ | `layer_17_attention.dense` | 1.443×10⁻⁴ | **1,811×** |

> **Key pattern**: All methods show their worst MSE at `layer_17_attention.dense`. This attention output projection at the penultimate layer is the model's most quantization-sensitive weight matrix. This is consistent with the known pattern that later transformer layers carry more "specialized" information and are harder to compress.

> **Why GPTQ has highest MSE despite being "smarter"**: GPTQ optimizes for **output accuracy** (minimizing the change in layer output), not **weight fidelity** (minimizing ||W_q - W||²). It intentionally allows large weight perturbations if they cancel out in the layer's output space. This means GPTQ can have high per-weight MSE while still achieving good downstream accuracy — which is exactly what Module 5 confirms (GPTQ MMLU: 66.93% vs BF16: 66.12%).

---

## 5. Submodule 3c — Activation Outlier Detection

**Time**: 424.7s (~7.1 min) · **Method**: Forward hook activation capture with 6.0σ threshold · **62 layers analyzed across 5 tags**

**How it works**: This is the most compute-intensive submodule because it requires **live model inference**. For each quantization format:

1. **Load the quantized model** (or BF16 baseline) onto GPUs
2. **Register forward hooks** on all linear layers — these hooks capture the activation tensors (inputs/outputs) during inference
3. **Run 4 diverse prompts** through the model:
   - English AI/healthcare prompt
   - English MoE technical prompt
   - English Python coding prompt
   - Hindi (Devanagari) language model question — tests Indic language handling
4. **Count outlier activations**: For each layer, compute μ and σ of activation values. Any activation > μ + 6σ or < μ - 6σ is counted as an outlier
5. **Unload the model** and perform GPU cleanup before loading the next format

The 6.0σ threshold is deliberately high — under a perfect Gaussian, only 0.0002% of values would exceed 6σ. Real activation distributions have heavier tails, so outlier rates of 0.08–0.10% indicate genuine extreme values.

### Results — Outlier Ranking

| Method | Mean Outlier % | Highest-Outlier Layer | Peak Outlier % |
|---|---:|---|---:|
| **BF16** (baseline) | 0.1049% | `model.layers.18.mlp.shared_experts.down_proj` | 0.2907% |
| **FP8** | 0.0996% | `model.layers.0.attention.dense` | 0.2201% |
| **INT8** | 0.0987% | `model.layers.18.mlp.shared_experts.down_proj` | 0.2262% |
| **GPTQ** | 0.0910% | `model.layers.18.mlp.shared_experts.down_proj` | 0.2363% |
| **NF4** | 0.0826% | `model.layers.0.attention.dense` | 0.2096% |

> **Counter-intuitive finding**: All quantized methods have **fewer** outlier activations than BF16. This is not a bug — it's expected behavior. Quantization clips extreme weight values to the representable range, which in turn reduces the magnitude of activation spikes. The BF16 model retains full-precision weights that can produce more extreme activations.

> **Hotspot analysis**: `model.layers.18.mlp.shared_experts.down_proj` (the last MoE layer's shared expert output) is consistently the worst outlier layer across BF16, INT8, and GPTQ. This shared expert processes every token (unlike routed experts) and sits at the model's deepest layer, where representation errors compound.

---

## 6. Outputs

| File | Size | Description |
|---|---|---|
| `module_3_report.md` | 1.6 KB | Human-readable research summary with MSE + outlier tables |
| `module_3_report.json` | 7.6 KB | Structured findings for programmatic consumption |
| `module_3_summary.json` | 1.5 KB | Execution metadata (timing, status, cache validation) |
| `mse_all_layers.json` | 14.0 KB | Per-layer MSE for all 4 quantizers across 97 layers |
| `outlier_stats_bf16.json` | 33.6 KB | Per-layer outlier statistics for BF16 baseline |
| `outlier_stats_int8.json` | 33.0 KB | Per-layer outlier statistics for INT8 |
| `outlier_stats_fp8.json` | 33.0 KB | Per-layer outlier statistics for FP8 |
| `outlier_stats_nf4.json` | 34.7 KB | Per-layer outlier statistics for NF4 |
| `outlier_stats_gptq.json` | 34.0 KB | Per-layer outlier statistics for GPTQ |

### Plots

- 14 weight distribution histograms (per target layer, all formats overlaid)
- MSE heatmap with log-scale colormap (97 layers × 4 formats)

---

*Module 3 reveals that INT8 is the most weight-faithful method (lowest MSE) while GPTQ trades weight fidelity for output accuracy. All quantized methods reduce activation outliers compared to BF16. See Module 5 for how these weight-level differences translate to downstream task accuracy.*
