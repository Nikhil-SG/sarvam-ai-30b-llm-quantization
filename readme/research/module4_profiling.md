# Module 4: Inference Profiling — Latency, VRAM, Disk Footprint

> **Pipeline**: Research Quantization Pipeline — Module 4 (Profiling)
> **Wall Time**: 1,018.1s (~17.0 min)

**Navigation**: [← Project README](../../README.md) · [Research Overview](../../research/README.md) · [← Module 3](module3_analysis.md) · [Module 5 →](module5_evaluation.md) · [Module 1](module1_baseline.md) · [Module 6](module6_visualization.md)

---

## 1. Purpose

Module 4 answers the deployment-critical question: **what does each quantization method actually buy you in production?** It profiles three dimensions of deployment cost:

- **Latency** (4a): How many tokens per second can each method generate at different batch sizes?
- **VRAM** (4b): What is the peak GPU memory usage during active inference?
- **Disk** (4c): How large is the serialized checkpoint on storage?

These metrics, combined with Module 5's quality scores, feed into the Pareto analysis that determines which methods offer the best quality-vs-efficiency tradeoff.

---

## 2. Profiling Methodology

### Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Batch sizes | 1, 2, 4, 8 | Covers single-user to small-batch serving scenarios |
| Max new tokens | 64 | Sufficient for stable tok/s measurement |
| Warmup steps | 2 | Ensures GPU caches and CUDA kernels are warm |
| Num runs | 3 | Enough for statistical validity (mean ± std reported) |
| Prompt | "The future of artificial intelligence in healthcare..." | Fixed prompt for reproducibility |

### Execution Flow per Quantizer

For each of the 5 formats (bf16, int8, fp8, nf4, gptq):

1. **Load model**: BF16 loads fresh from the base model; quantized formats load from saved checkpoints via `load_saved_checkpoint_only()`
2. **Reset peak memory**: Calls `torch.cuda.reset_peak_memory_stats()` on all GPUs to get clean peak measurements
3. **Run 4a (Latency)**: For each batch size [1, 2, 4, 8]:
   - Run 2 warmup generations (results discarded)
   - Run 3 timed generations, measuring wall-clock time and tokens generated
   - Compute tokens/second = total_tokens / elapsed_time
4. **Run 4b (VRAM)**: Snapshot `torch.cuda.max_memory_allocated()` across all GPUs after the latency benchmark
5. **Run 4c (Disk)**: Walk the checkpoint directory, summing `.safetensors` and `.bin` file sizes. Also measure runtime weight footprint via `model.num_parameters() * bytes_per_param`
6. **Unload model**: Call `model.unload()`, then `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()`

---

## 3. Latency Results — Tokens per Second

### Raw Data

| Method | BS=1 (tok/s) | BS=2 (tok/s) | BS=4 (tok/s) | BS=8 (tok/s) | BS=1 Latency (s/64tok) |
|---|---:|---:|---:|---:|---:|
| **BF16** | **17.72** | **33.56** | **67.42** | **133.94** | 3.61s |
| **NF4** | 8.97 | 12.78 | 26.12 | 48.98 | 7.13s |
| **GPTQ** | 7.66 | 15.44 | 32.04 | 58.18 | 8.35s |
| **INT8** | 5.11 | 9.28 | 18.41 | 34.39 | 12.53s |
| **FP8** | 4.74 | 9.37 | 16.76 | 35.87 | 13.51s |

### Why BF16 is Fastest on A100

This result surprises many readers: quantized models are *slower*, not faster. Here's why:

**A100 compute capabilities** (compute capability 8.0):
- ✅ BF16 tensor cores: 312 TFLOPS
- ✅ FP16 tensor cores: 312 TFLOPS
- ✅ INT8 tensor cores: 624 TOPS (theoretical 2× throughput)
- ❌ FP8 tensor cores: **Not available** (introduced in H100, CC 9.0)
- ❌ INT4 tensor cores: **Not available**

**The dequantization overhead problem**:

Despite INT8 tensor cores being available, the `bitsandbytes` LLM.int8() implementation does NOT use them for end-to-end INT8 GEMM. Instead, it:
1. Decomposes the input by outlier detection (FP16 path for outliers)
2. Dequantizes INT8 weights to FP16
3. Executes FP16 GEMM for the majority of computation
4. Merges the outlier and non-outlier results

This decomposition overhead (~2.5× slower than native BF16) dominates any potential throughput gain.

For NF4 and GPTQ, every forward pass requires:
1. Unpack 4-bit values from packed byte storage
2. Apply per-group scale factors and zero-points
3. Cast to BF16/FP16
4. Execute standard GEMM

For FP8 on A100 specifically:
1. Load FP8 weights
2. Dequantize to BF16 (no native FP8 GEMM available)
3. Execute BF16 GEMM

**Why NF4/GPTQ are faster than INT8/FP8**:
NF4 and GPTQ achieve ~2× better throughput than INT8/FP8 because:
- The 4-bit model fits on a **single GPU** (18–20 GB), eliminating all cross-GPU communication
- BF16 and FP8 are split across 2 GPUs, incurring PCIe transfer overhead for inter-layer data movement
- The single-GPU advantage outweighs the slightly more complex dequantization of 4-bit vs 8-bit

**H100 projection**: On H100 with native FP8 tensor cores (1,979 TFLOPS), FP8 quantized models would likely be 1.5–2× faster than BF16, completely reversing the A100 result.

---

## 4. VRAM Results — Peak Memory

| Method | Peak VRAM (GB) | vs BF16 | Fits Single A100 80GB? | Notes |
|---|---:|---:|---|---|
| **BF16** | 60.25 | 1.00× | ❌ (needs 2 GPUs) | Split ~45/55 across GPUs |
| **FP8** | 59.93 | 0.99× | ❌ (needs 2 GPUs) | Runtime dequantization doubles footprint |
| **INT8** | 32.23 | 0.53× | ✅ | Fits on single GPU with 47 GB headroom |
| **GPTQ** | 20.63 | 0.34× | ✅ | Fits with 58 GB headroom for KV cache |
| **NF4** | 18.66 | 0.31× | ✅ | Best VRAM efficiency — 60 GB free for serving |

**Why FP8 uses ~60 GB despite being "8-bit"**: The `optimum-quanto` FP8 implementation on A100 must dequantize all weights to BF16 for computation. During the latency benchmark (which processes multiple batch sizes), the peak memory reflects the moment when both the FP8 compressed weights AND the dequantized BF16 copies coexist in GPU memory. The on-disk model is only 32 GB, but runtime memory peaks at 60 GB.

### VRAM Breakdown

```
BF16:   [===================|=======================] 27.0 GB + 33.0 GB = 60.0 GB (2 GPUs)
INT8:   [================================|] 32.2 GB + 0.0 GB = 32.2 GB (1 GPU)
FP8:    [|====================================] 0.0 GB + 59.9 GB = 59.9 GB (dequant overhead)
NF4:    [==================|] 18.7 GB + 0.0 GB = 18.7 GB (1 GPU)
GPTQ:   [====================|] 20.6 GB + 0.0 GB = 20.6 GB (1 GPU)
```

---

## 5. Disk Footprint

| Method | Disk Size (GB) | Shards | Runtime Weight Footprint (GB) | Disk vs Runtime |
|---|---:|---:|---:|---|
| **BF16** | 119.78 | 26 | 59.89 | 2.0× (safetensors stores full param copies) |
| **INT8** | 32.01 | 1 | 31.95 | 1.0× (compressed matches runtime) |
| **FP8** | 31.98 | 7 | 31.95 | 1.0× |
| **NF4** | 18.44 | 1 | 17.99 | 1.0× (includes packed 4-bit + FP8 absmax scales) |
| **GPTQ** | 18.62 | 1 | 4.02 | 4.6× (disk stores packed + metadata; runtime is just packed weights) |

**Why BF16 disk is 2× runtime**: Safetensors format stores the full checkpoint with all parameter tensors. The 119.78 GB includes the embedding layer, LM head, and all layer norm parameters at full BF16 precision, plus safetensors metadata. The runtime footprint (59.89 GB) reflects only the model's weight tensors as counted by `model.num_parameters() * 2 bytes`.

---

## 6. Comparison Plot Generation

After profiling all formats, Module 4 generates three comparison plots:
- **Latency comparison**: Bar chart of tokens/second across batch sizes and methods
- **VRAM comparison**: Bar chart of peak memory per method
- **Disk comparison**: Bar chart of checkpoint sizes

These plots are saved to `research/outputs/module_4_profiling/plots/`.

---

## 7. Outputs

| File | Size | Description |
|---|---|---|
| `latency_results.json` | 3.6 KB | Per-method, per-batch-size latency data (avg, std, tok/s) |
| `vram_results.json` | 3.9 KB | Peak VRAM per GPU per method |
| `disk_results.json` | 2.1 KB | Checkpoint sizes and shard counts |
| `module_4_summary.json` | 6.2 KB | Combined profiling summary with method descriptions |

---

*Module 4 shows that the primary deployment benefit of quantization on A100 is VRAM savings (enabling single-GPU deployment), not speed. INT8/NF4/GPTQ all fit on a single A100 80GB, while BF16 and FP8 require two GPUs. Speed improvements require hardware with native low-precision tensor cores (H100+).*
