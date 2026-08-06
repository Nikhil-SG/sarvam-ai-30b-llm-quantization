# Module 2: Quantization Matrix — INT8, FP8, NF4, GPTQ

> **Pipeline**: Research Quantization Pipeline — Module 2 (Quantization)
> **Wall Time**: 17,299.5s (~4.8 hr) — dominated by GPTQ calibration (4.6 hr)

**Navigation**: [← Project README](../../README.md) · [Research Overview](../../research/README.md) · [← Module 1](module1_baseline.md) · [Module 3 →](module3_analysis.md) · [Module 4](module4_profiling.md) · [Module 5](module5_evaluation.md) · [Module 6](module6_visualization.md)

---

## 1. Purpose

Module 2 quantizes the BF16 Sarvam-30B model across 4 methods — INT8, FP8, NF4, and GPTQ — saves each as a reusable checkpoint, builds per-method weight caches for downstream MSE analysis, and records detailed memory/size metrics. It also performs **cross-format weight cache validation** at startup, using Pearson correlation to detect and purge stale caches where "quantized" values are near-identical to BF16 (indicating the previous run failed to properly dequantize).

---

## 2. Quantization Methods — Deep Technical Detail

### 2.1 INT8 — `LLM.int8()` via bitsandbytes

**Library**: `bitsandbytes` · **Config Class**: `BitsAndBytesConfig` · **Bits**: 8

**How it works**: `LLM.int8()` (Dettmers et al., 2022) is a mixed-precision decomposition scheme. It does **not** simply quantize all weights to INT8. Instead, it:

1. **Identifies outlier features** — any activation dimension whose magnitude exceeds a threshold (configured as `llm_int8_threshold: 6.0`) is marked as an "outlier feature"
2. **Decomposes the matrix multiply** into two parts:
   - **Main path** (≥99% of dimensions): Weights and activations are quantized to INT8, multiplied using INT8 GEMM, and the result is dequantized back to FP16
   - **Outlier path** (≤1% of dimensions): Outlier feature columns are kept in FP16 and multiplied in FP16 precision
3. **Results are merged**: The INT8 and FP16 partial results are added together

This decomposition preserves quality because the outlier features (which carry disproportionate information) remain in full precision, while the bulk of computation benefits from INT8 efficiency.

**On A100**: The A100 has INT8 tensor cores that can execute INT8 GEMM at 2× the throughput of FP16. However, the `bitsandbytes` `LLM.int8()` implementation adds overhead from the decomposition, dequantization, and FP16 outlier path, which offsets the raw throughput gain. Net result: **INT8 is slower than BF16 on A100** in this pipeline (5.11 vs 17.72 tok/s at BS=1) because the decomposition overhead dominates.

**Configuration used**:
```yaml
int8:
  llm_int8_threshold: 6.0       # σ threshold for outlier detection
  llm_int8_skip_modules: ["lm_head"]  # Keep lm_head in FP16
  llm_int8_has_fp16_weight: false     # No FP16 weight copy retained
```

**Save method — `save_pretrained`**: The quantized model uses HuggingFace's native `save_pretrained()`. bitsandbytes stores INT8 weights as `torch.int8` tensors alongside quantization state (absmax scaling factors per row/column block) in the safetensors checkpoint. On reload, `from_pretrained()` with the same `BitsAndBytesConfig` reconstructs the quantized layers automatically.

---

### 2.2 FP8 — Float8 via optimum-quanto

**Library**: `optimum-quanto` · **Method**: `optimum_quanto_fp8` · **Bits**: 8

**How it works**: `optimum-quanto` implements FP8 quantization using the E4M3 format (4-bit exponent, 3-bit mantissa). The process:

1. **Weight quantization**: Each weight tensor is converted from BF16 to FP8 E4M3 by finding a per-tensor scale factor that maps the weight range into the FP8 representable range (±448 for E4M3)
2. **Activation quantization**: Calibration data (64 samples from `sangraha_verified` Indic dataset) is run through the model to determine optimal input/output activation scales for each layer
3. **Runtime dequantization**: During inference, FP8 weights are dequantized back to BF16/FP16 for matrix multiplication, since A100 does **not** have native FP8 tensor cores

**On A100**: This is the critical limitation. The A100 (compute capability 8.0) supports INT8 and BF16/FP16 tensor cores but **not** FP8. FP8 tensor cores were introduced with H100 (compute capability 9.0). On A100, every FP8 matmul requires:
1. Load FP8 weights from memory
2. Dequantize to BF16 in registers
3. Execute BF16 GEMM
4. Store result

This means FP8 on A100 saves **memory** (half the weight footprint) but provides **no speed benefit** — in fact it's slower than native BF16 due to the dequantization overhead (4.74 vs 17.72 tok/s at BS=1).

**Calibration configuration**:
```yaml
fp8:
  weights: "float8"
  activations: "float8"
  cal_num_samples: 64     # 64 Indic-language calibration samples
  cal_seq_length: 1024    # 1024 tokens per sample
  batch_size: 4           # Process 4 samples per calibration batch
  streamline: true        # Skip unnecessary q/deq activation paths
```

**Save method — `optimum_quanto`**: quanto uses its own serialization format. It saves quantized weight tensors as FP8 values plus per-tensor scale factors in safetensors shards. The model is split across 7 shards (vs 26 for BF16) because the compressed weights are approximately half the size. On reload, `quanto.load()` reconstructs the quantized model with the correct scale factors.

---

### 2.3 NF4 — NormalFloat4 via bitsandbytes

**Library**: `bitsandbytes` · **Config Class**: `BitsAndBytesConfig` · **Bits**: 4

**How it works**: NF4 (Dettmers et al., 2023, QLoRA paper) exploits the fact that neural network weights follow approximately normal distributions. Instead of using uniformly-spaced quantization levels (like standard INT4), NF4 uses 16 quantization levels that are optimally placed for a standard normal distribution:

1. **Block quantization**: Weights are divided into blocks of 64 elements
2. **Absmax normalization**: Each block is normalized by its absolute maximum value to the range [-1, 1]
3. **NF4 lookup**: The normalized values are mapped to the nearest of 16 NormalFloat levels (precomputed for optimal information density under Gaussian assumption)
4. **Double quantization**: The absmax scaling factors themselves are quantized to FP8, saving additional memory (enabled by `double_quant: true`)

**On A100**: Like INT8, every forward pass requires dequantization: NF4 → FP16/BF16, then standard GEMM. A100 has no native 4-bit tensor cores. The dequantization path for NF4 involves lookup table operations that are memory-bound. Despite this, NF4 achieves better throughput than INT8/FP8 (8.97 vs 5.11/4.74 tok/s at BS=1) because the model fits entirely on a single GPU (18.5 GB), eliminating cross-GPU communication latency.

**Configuration used**:
```yaml
nf4:
  double_quant: true          # Quantize the quantization constants too
  compute_dtype: "bfloat16"   # Dequantize to BF16 for forward pass
```

**Save method — `save_pretrained`**: bitsandbytes NF4 models save via HuggingFace's `save_pretrained()`. The checkpoint contains packed 4-bit weight values (2 weights per byte) plus FP8 absmax scales. The `quantization_config` in `config.json` records the NF4 parameters so that `from_pretrained()` with the appropriate `BitsAndBytesConfig` can reconstruct the model. The model saves as a single shard because at ~18 GB it's below the default shard size threshold.

**Parameter count anomaly**: NF4 reports 17.15B total params and 3.34B active params (vs 32.15B/4.52B for BF16). This is because `model.num_parameters()` counts the packed 4-bit representation (2 weights per byte = half the parameter count) and the quantization metadata. The actual logical parameter count remains 32.15B.

---

### 2.4 GPTQ — Post-Training Quantization with Hessian-based Rounding

**Library**: `optimum` + `auto-gptq` · **Method**: `optimum_gptq_from_scratch` · **Bits**: 4

**How it works**: GPTQ (Frantar et al., 2023) is fundamentally different from the other methods. Instead of simply rounding weights to lower precision, it uses **second-order information** (Hessian/Fisher matrix) to make optimal rounding decisions that minimize the output error of each layer:

1. **Calibration data collection**: 32 samples of 512 tokens each from the `sangraha_verified` Indic dataset are forward-passed through the model. The Hessian of the layer's output with respect to its weights is approximated from the calibration activations.

2. **Layer-by-layer quantization**: For each of the model's 7,000+ weight matrices (19 layers × 128 experts × 3 projections + attention + shared experts):
   - Compute the approximate Hessian H = X^T X (where X is the calibration input matrix)
   - Use the **Optimal Brain Quantizer (OBQ)** algorithm to quantize each column of the weight matrix in order, compensating for quantization error in already-processed columns by adjusting the remaining unprocessed columns
   - Group weights into blocks of 128 (`group_size: 128`) for per-group scaling

3. **Asymmetric quantization fallback**: The config requests `sym: false` (asymmetric quantization with separate zero-point), but `auto-gptq` does not support asymmetric mode. The pipeline detects this and falls back to `sym: true` with a logged warning.

**On A100**: GPTQ quantization is extremely compute-intensive for MoE models. With 128 experts × 3 projections × 18 MoE layers = 6,912 individual weight matrices to process (plus attention and dense layers), the full calibration takes **4.6 hours**. Each matrix requires forward-passing all calibration samples through the preceding layers and computing the Hessian inverse.

**GPTQ calibration challenges observed in logs**:

Several MoE expert layers received **zero calibration batches** because those experts were never activated by the 32 calibration samples (with top-6 routing out of 128 experts, many experts see no traffic). The pipeline handles this gracefully:
```
WARNING [gptq_quantizer] auto_gptq received no calibration batches for this layer;
forcing nsamples=1 to avoid ZeroDivisionError and continue quantization.
```

This means some low-traffic experts are quantized with essentially random rounding (no Hessian guidance), which contributes to GPTQ's higher MSE compared to INT8/FP8.

Additionally, the calibration filter had to progressively relax its `min_tokens` threshold:
```
WARNING [gptq_quantizer] Calibration filter min_tokens=64 retained zero samples; relaxing threshold
WARNING [gptq_quantizer] Calibration filter min_tokens=32 retained zero samples; relaxing threshold
WARNING [gptq_quantizer] Calibration filter min_tokens=16 retained zero samples; relaxing threshold
```

**Configuration used**:
```yaml
gptq:
  bits: 4
  group_size: 128               # 128 weights per quantization group
  desc_act: false               # Disable activation reordering (better MoE compatibility)
  sym: false                    # Requested asymmetric, falls back to symmetric
  cal_num_samples: 32           # 32 samples (smaller than shared 128)
  cal_seq_length: 512           # 512 tokens (shorter than shared 2048)
  batch_size: 4
```

**Save method — `optimum_save`**: GPTQ models use the `optimum` library's save path, which writes the quantized weights in a format compatible with `auto-gptq`. The checkpoint contains packed INT4 weight values, per-group scale factors, and zero-points (for asymmetric mode, though zero-points are trivial when `sym=True`). The Exllamav2 backend warning in logs (`disable_exllama=True`) indicates that the save process explicitly disables the Exllama kernel to ensure weights are saved in the correct order.

**Parameter count anomaly**: GPTQ reports only 2.16B total params and 4.02 GB model size. This is because the packed INT4 representation stores 2 weights per byte, and the `auto-gptq` serialization only counts the compressed tensor elements. The logical model remains 32.15B parameters.

---

## 3. Cross-Format Weight Cache Validation

Before running any quantizer, Module 2 performs a **cache integrity check** using Pearson correlation:

1. Loads one layer's BF16 weight cache from Module 1
2. For each quantized format (int8, fp8, nf4, gptq), loads the corresponding cached values
3. Computes Pearson correlation r between BF16 and quantized values
4. If r > threshold **and** max absolute difference < threshold, the cache is flagged as "near-identical to BF16" — meaning the previous run failed to properly dequantize and just saved BF16 values as-is

The thresholds are tuned per format:
- **INT8**: r > 0.999999, max_diff < 0.001 (INT8 is genuinely near-lossless, so use very strict thresholds)
- **FP8/NF4/GPTQ**: r > 0.9999, max_diff < 0.05 (lower precision should show visible differences)

Flagged caches are purged and re-extracted with corrected dequantization paths.

---

## 4. Results Summary

| Method | Model Size (GB) | VRAM Allocated (GB) | Load Time | Compression vs BF16 | Single GPU? |
|---|---|---|---|---|---|
| **BF16** (ref) | 59.89 | 59.89 | 27.6s | 1.00× | ❌ (needs 2 GPUs) |
| **INT8** | 31.95 | 32.02 | 68.1s | 1.87× | ✅ |
| **FP8** | 31.95 | 32.00 | 169.3s | 1.87× | ✅ (disk only) |
| **NF4** | 17.99 | 18.45 | 84.2s | 3.33× | ✅ |
| **GPTQ** | 4.02 | 18.63 | 16,443.1s (~4.6h) | 14.90× | ✅ |

> **Note on GPTQ VRAM vs model size**: GPTQ's on-disk model is only 4.02 GB (packed INT4), but GPU memory is 18.63 GB because the weights are dequantized to FP16/BF16 during loading for compatibility with the `auto-gptq` runtime. The dequantized model occupies ~18 GB in memory.

---

## 5. Saved Checkpoints

| Method | Output Directory | Shards | Disk Size | Format |
|---|---|---|---|---|
| INT8 | `research/quantized_models/int8_quantized` | 1 | 32.01 GB | safetensors + bnb quant state |
| FP8 | `research/quantized_models/fp8_quantized` | 7 | 31.98 GB | safetensors + quanto scales |
| NF4 | `research/quantized_models/nf4_quantized` | 1 | 18.44 GB | safetensors + packed 4-bit + FP8 absmax |
| GPTQ | `research/quantized_models/gptq_quantized` | 1 | 18.62 GB | safetensors + packed INT4 + group scales |

---

## 6. Inter-Quantizer GPU Cleanup

After each quantizer completes, Module 2 performs aggressive cleanup to prevent OOM cascades:

```python
gc.collect()                    # Python garbage collection
torch.cuda.empty_cache()       # Return unused cached memory to CUDA
torch.cuda.synchronize()       # Wait for all GPU operations to complete
```

This is critical because the pipeline processes all 4 quantizers sequentially in the same process. Without cleanup, residual CUDA allocations from INT8 could cause FP8 to OOM, even though each quantized model individually fits in memory.

---

## 7. Outputs

| File | Description |
|---|---|
| `results/int8_results.json` | INT8 metrics (method, memory, timing, save path) |
| `results/fp8_results.json` | FP8 metrics |
| `results/nf4_results.json` | NF4 metrics |
| `results/gptq_results.json` | GPTQ metrics (includes 4.6h calibration time) |
| `shared_weights/{int8,fp8,nf4,gptq}/*.npz` | Per-method weight caches for MSE comparison |

---

*Module 2 produces the 4 quantized checkpoints and their weight caches. These feed into Module 3 (MSE + outlier analysis), Module 4 (latency/VRAM profiling), and Module 5 (perplexity/MMLU evaluation).*
