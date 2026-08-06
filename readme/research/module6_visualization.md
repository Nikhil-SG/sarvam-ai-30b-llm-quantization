# Module 6: Layer-Wise Visualization — Weight Distribution Histograms & BF16 Delta Plots

> **Pipeline**: Research Quantization Pipeline — Module 6 (Visualization)
> **Wall Time**: 46.3s (~0.8 min)

**Navigation**: [← Project README](../../README.md) · [Research Overview](../../research/README.md) · [← Module 5](module5_evaluation.md) · [Module 1](module1_baseline.md) · [Module 2](module2_quantization.md) · [Module 3](module3_analysis.md) · [Module 4](module4_profiling.md)

---

## 1. Purpose

Module 6 generates **publication-ready per-layer weight distribution visualizations** that show how each quantization method transforms the weight distributions compared to the BF16 baseline. While Module 3's histograms focus on the 14 target layers defined in the config, Module 6 produces a comprehensive sweep across **all cached layers** (97 tensors), producing side-by-side comparison plots suitable for inclusion in research reports.

---

## 2. How It Works

### Data Source

Module 6 reads from the **shared weight caches** (`research/outputs/shared_weights/`) built by Modules 1 and 2. Each NPZ file contains up to 500,000 weight values sampled from the corresponding tensor. No live model loading is required — this is a pure post-processing visualization step, which is why it completes in under 1 minute.

### Visualization Pipeline

For each cached layer across all formats (bf16, int8, fp8, nf4, gptq):

1. **Load weight samples**: Read the NPZ file, extract the `values` array (up to 500K float32 values)
2. **Compute statistics**: Mean (μ), standard deviation (σ), min, max, kurtosis
3. **Generate histogram**: Using matplotlib with the configured style (`seaborn-v0_8-whitegrid`):
   - Bin the weight values into 100–200 histogram bins
   - Overlay a fitted Gaussian curve using the computed μ and σ
   - All 5 formats plotted on the same axes with distinct colors for direct comparison
4. **BF16 delta overlay**: For quantized formats, compute and annotate the difference from BF16:
   - Mean shift: μ_quantized - μ_bf16
   - Variance change: σ_quantized / σ_bf16
   - Distribution shape change (kurtosis delta)
5. **Save as PNG**: 150 DPI, 14×8 inches (configured via `visualization.dpi` and `visualization.figsize_*`)

### Plot Configuration

```yaml
visualization:
  weight_sample_size: 500000    # Max values per layer
  dpi: 150                      # Publication resolution
  figsize_width: 14             # Inches
  figsize_height: 8             # Inches
  style: "seaborn-v0_8-whitegrid"  # Clean academic style
```

---

## 3. What the Visualizations Reveal

### INT8 — Near-Invisible Quantization

INT8 weight distributions are virtually indistinguishable from BF16. With 256 quantization levels spanning the weight range, the histogram bins align closely with the continuous BF16 distribution. The Gaussian overlay fits both distributions nearly identically.

**Why**: INT8 uses per-channel or per-tensor absmax scaling, placing 256 evenly-spaced quantization levels across the weight range. For a typical weight tensor with σ ≈ 0.01, the quantization step size is approximately 2×max_val/256 ≈ 0.0002, which is much finer than the histogram bin width.

### FP8 — Slight Tail Noise

FP8 (E4M3 format) distributions closely track BF16 but show slightly more noise in the tail regions (weights far from zero). The 3-bit mantissa provides 8 levels of precision between consecutive powers of 2, compared to INT8's uniform spacing.

**Why**: FP8 E4M3 has non-uniform quantization — fine granularity near zero (where most weights cluster) but coarser granularity for large values. This is generally beneficial (most information is preserved), but the tails show visible quantization artifacts in the histograms.

### NF4 — Clearly Stepped Distribution

NF4 distributions show 16 distinct peaks corresponding to the 16 NormalFloat quantization levels. The overall Gaussian envelope is preserved (this is by design — NF4 levels are optimized for normal distributions), but the continuous distribution becomes a discrete 16-point distribution.

**Why**: With only 4 bits, there are exactly 16 possible values per weight group. The NF4 lookup table places these values at positions that minimize information loss under a Gaussian assumption: approximately at the quantiles of a standard normal distribution. The histogram clearly shows these 16 peaks, with the peak heights following the expected Gaussian density.

### GPTQ — Shifted Centroids and Altered Variance

GPTQ distributions show the most dramatic changes. Some layers exhibit:
- **Centroid shifts**: The mean of the quantized distribution moves noticeably from the BF16 mean
- **Variance changes**: The spread of the distribution can increase or decrease
- **Asymmetric tail behavior**: One tail may be compressed more than the other

**Why**: GPTQ's Hessian-based rounding algorithm intentionally allows large per-weight perturbations if they minimize the layer's output error. The algorithm compensates for each quantized weight by adjusting all remaining weights in the column, which can shift the overall distribution. This is why GPTQ has the highest weight-level MSE (Module 3) but still achieves good downstream accuracy (Module 5) — the errors are coordinated to cancel in the output space.

The effect is most visible in `layer_17_attention.dense` — the same layer identified as the MSE hotspot in Module 3. Here, GPTQ's centroid shift is visible to the eye in the histogram overlay.

---

## 4. Layer Coverage

Module 6 produces visualizations for all 97 cached layers, organized by component type:

| Component | Layers | Count |
|---|---|---|
| Attention QKV | `layers.{0-18}.attention.query_key_value` | 19 |
| Attention Output | `layers.{0-18}.attention.dense` | 19 |
| Dense MLP (Layer 0) | `layers.0.mlp.{gate,up,down}_proj` | 3 |
| Shared Expert | `layers.{1-18}.mlp.shared_experts.{gate,up,down}_proj` | 54 |
| Routed Experts (sample) | `layers.9.mlp.experts.{0,63,127}.down_proj` | 3 |
| **Total** | | **97** (≈1 layer per format = 485 distribution curves) |

---

## 5. Outputs

```
research/outputs/module_6_visualization/
├── plots/          ← Per-layer distribution PNG files
├── results/        ← Visualization metadata and statistics
└── logs/           ← Module execution logs
```

---

*Module 6 provides the visual evidence supporting the quantitative findings from Modules 3 and 5. The histograms make the abstract MSE numbers concrete — showing exactly how each quantization method reshapes the weight landscape.*
