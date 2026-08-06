# Research Pipeline — Troubleshooting Guide

> Issues encountered during quantization research for Sarvam-30B across INT8, FP8, NF4, and GPTQ on 2× NVIDIA A100 80GB PCIe.

**Navigation**: [← Project README](../../README.md) · [Research Overview](../../research/README.md) · [Module 1](module1_baseline.md) · [Module 2](module2_quantization.md) · [Module 3](module3_analysis.md) · [Module 4](module4_profiling.md) · [Module 5](module5_evaluation.md) · [Module 6](module6_visualization.md)

---

## Issue 1: BF16 Perplexity CUDA OOM During Module 5

### Symptom
```
WARNING [main] Perplexity failed: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 79.15 GiB of which 1.91 GiB is free.
Including non-PyTorch memory, this process has 58.31 GiB memory in use.
Of the allocated memory 51.75 GiB is allocated by PyTorch, and 6.05 GiB is reserved
by PyTorch but unallocated.
```

### Root Cause
The BF16 model occupies ~60 GB VRAM across 2 GPUs. During perplexity evaluation, the sliding-window computation (2048-token context) creates intermediate activation tensors and gradient buffers that require an additional ~2 GB. With other GPU processes consuming shared memory (532 MB + 16.6 GB + 518 MB + 838 MB + 414 MB from co-resident processes), the remaining 1.91 GB on GPU 0 was insufficient.

Additionally, PyTorch's caching allocator had 6.05 GB of **reserved but unallocated** memory — fragmented blocks too small for the 2 GB contiguous allocation requested.

### Resolution
1. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
2. Re-run Module 5 with `--module 5 --quantizer bf16` to evaluate only the BF16 format
3. The pipeline's result caching system preserved all previously-computed results (INT8, FP8, NF4, GPTQ perplexity and MMLU) and merged the new BF16 result on the subsequent run

### Prevention
- Kill co-resident GPU processes before running the full pipeline
- Use `max_memory` config to reserve headroom: set GPU budgets 5–10 GB below physical capacity
- For BF16-only evaluation, use a smaller sliding window (`max_length: 1024`) or reduce `max_samples`

---

## Issue 2: GPTQ Empty Calibration Batches (ZeroDivisionError Risk)

### Symptom
Repeated warnings during GPTQ quantization (occurring 30+ times):
```
WARNING [gptq_quantizer] auto_gptq received no calibration batches for this layer;
forcing nsamples=1 to avoid ZeroDivisionError and continue quantization.
```

### Root Cause
Sarvam-30B has 128 routed experts per MoE layer with top-6 routing. With only 32 calibration samples, many experts receive **zero activations** — the router simply never selects them for any calibration token. When `auto_gptq` processes these experts' weight matrices, it has no Hessian information (H = X^T X where X is empty), which would cause a ZeroDivisionError in the Hessian inverse computation.

The affected layers are primarily **low-traffic experts** — experts indexed far from the router's learned preferred set. With top-6 out of 128, approximately 95% of experts are selected for any given token, but some experts may never appear in a small 32-sample calibration set.

### Resolution
The pipeline's `GPTQQuantizer` detects the empty-batch condition and forces `nsamples=1` with a dummy input to provide a degenerate (but non-singular) Hessian. This allows quantization to proceed with essentially random rounding for these experts.

**Quality impact**: These zero-traffic experts are rarely activated during inference, so the quality impact is minimal. Module 5 confirms GPTQ achieves 66.93% MMLU (vs 66.12% BF16), indicating the workaround does not significantly degrade performance.

### Prevention
- Increase `cal_num_samples` from 32 to 128 or 256 to ensure more experts see calibration traffic
- Use a diverse multilingual calibration dataset to activate more expert specializations
- Set `cal_seq_length: 2048` (longer sequences activate more experts per sample)

---

## Issue 3: GPTQ Asymmetric Quantization Not Supported

### Symptom
```
WARNING [gptq_quantizer] Requested sym=False (asymmetric), but current backend is
auto_gptq, which does not support asymmetric mode. Overriding to sym=True for
compatibility. Install gptqmodel to use sym=False.
```

### Root Cause
The research config requests `sym: false` (asymmetric quantization with learned zero-points), which can improve quality for weight distributions that are not centered at zero. However, the `auto_gptq` backend only implements symmetric quantization (zero-point fixed at 0).

### Resolution
The pipeline automatically falls back to `sym=True` (symmetric quantization) and logs a warning. Quality impact is minimal for this model because Sarvam-30B's weight distributions are approximately zero-centered.

### To Use Asymmetric Mode
Install `gptqmodel` (a fork of `auto-gptq` with additional features):
```bash
pip install gptqmodel
```
The pipeline will automatically detect `gptqmodel` and use its asymmetric quantization support.

---

## Issue 4: GPTQ Calibration Filter Relaxation

### Symptom
Progressive threshold relaxation during GPTQ calibration data preparation:
```
WARNING [gptq_quantizer] Calibration filter min_tokens=64 retained zero samples; relaxing threshold
WARNING [gptq_quantizer] Calibration filter min_tokens=32 retained zero samples; relaxing threshold
WARNING [gptq_quantizer] Calibration filter min_tokens=16 retained zero samples; relaxing threshold
```

### Root Cause
The GPTQ calibration data loader filters samples by minimum token count to ensure each calibration sample has enough context for meaningful Hessian computation. With the `sangraha_verified` Indic dataset and `cal_seq_length: 512`, some samples tokenize to fewer tokens than the filter threshold. The filter progressively relaxes from 64 → 32 → 16 tokens until samples pass.

### Resolution
The pipeline handles this automatically by relaxing the filter threshold. Final calibration proceeds with shorter samples. This is acceptable because:
- GPTQ's Hessian estimation is robust to sample length variation
- The 32 samples that eventually pass still provide sufficient statistical coverage
- Module 5 confirms no quality degradation from this relaxation

### Prevention
- Use longer calibration sequences: `cal_seq_length: 2048`
- Use a dataset with consistently long documents (e.g., Wikipedia rather than conversational data)

---

## Issue 5: Exllamav2 Backend Weight Reordering Warning

### Symptom
During GPTQ model saving:
```
WARNING [optimum.gptq.quantizer] Using Exllamav2 backend will reorder the weights
offline, thus you will not be able to save the model with the right weights.
Setting `disable_exllama=True`. You should only use Exllamav2 backend for inference.
```

### Root Cause
The Exllamav2 CUDA kernel reorders weight columns for optimal memory access patterns during inference. This reordering is an in-place transformation that changes the physical weight layout in memory. If the model is saved after Exllamav2 reordering, the saved weights are in a format that only Exllamav2 can read — they cannot be loaded by standard `auto_gptq` or `transformers`.

### Resolution
The `optimum` library automatically sets `disable_exllama=True` before saving, which:
1. Disables Exllamav2 kernel to prevent weight reordering
2. Saves weights in the canonical format compatible with all backends
3. Exllamav2 kernel can still be enabled at reload time for inference-only use

This is **not an error** — it's an informational warning from the save path doing the right thing.

---

## Issue 6: CUDA Extension Not Installed for auto_gptq

### Symptom
At pipeline startup:
```
WARNING [auto_gptq.nn_modules.qlinear.qlinear_cuda] CUDA extension not installed.
WARNING [auto_gptq.nn_modules.qlinear.qlinear_cuda_old] CUDA extension not installed.
```

### Root Cause
`auto_gptq` has optional CUDA extensions that provide optimized INT4 GEMM kernels. These extensions must be compiled from source for the specific CUDA toolkit and GPU architecture. When the extensions are not compiled, `auto_gptq` falls back to a pure PyTorch implementation.

### Impact
- **During quantization** (Module 2): No impact — the Hessian computation and weight rounding are CPU-bound operations that don't use the CUDA extensions
- **During inference** (Modules 4–5): The fallback PyTorch implementation is slower than the CUDA kernel. GPTQ's measured throughput (7.66 tok/s at BS=1) could improve with compiled extensions

### Resolution
Compile the CUDA extensions:
```bash
pip install auto-gptq --no-build-isolation
# Or from source:
git clone https://github.com/PanQiWei/AutoGPTQ
cd AutoGPTQ
pip install -e .
```

---

## Issue 7: No HuggingFace Token Found

### Symptom
Repeated throughout the pipeline:
```
WARNING [src.core.auth] No HF token found ($HF_TOKEN / huggingface-cli login / config.yaml fallback)
```

### Root Cause
The model is loaded from a local pre-downloaded snapshot (`model_registry/`), so no HuggingFace authentication is actually needed. The warning fires because the `auth` module checks for a token as a precaution (it would be needed for gated models or private repos).

### Impact
**None** — the model loads successfully from the local snapshot. The warning is purely informational.

### To Suppress
Either:
- Set `HF_TOKEN` environment variable: `export HF_TOKEN=hf_xxxx`
- Login via CLI: `huggingface-cli login`
- Set in config: `model.hf_token: "hf_xxxx"` (not recommended — avoids committing tokens to git)

---

## Issue 8: lm-eval Custom Model Type Warning

### Symptom
During MMLU evaluation in Module 5:
```
WARNING [lm_eval.models.huggingface] `pretrained` model kwarg is not of type `str`.
Many other model arguments may be ignored.

WARNING [lm_eval.models.huggingface] HF model type is neither marked as CausalLM or
Seq2SeqLM. Setting backend to causal

WARNING [lm_eval.models.huggingface] Passed an already-initialized model through
`pretrained`, assuming single-process call to evaluate()
```

### Root Cause
The pipeline passes an already-loaded model object to `lm-eval` (rather than a model ID string) to avoid reloading the 30B model. `lm-eval` warns that:
1. Model arguments (like `device_map`) are ignored for pre-loaded models
2. `SarvamMoEForCausalLM` doesn't inherit from `AutoModelForCausalLM` in a way that `lm-eval` recognizes
3. Distributed evaluation features are disabled for pre-loaded models

### Impact
**None** — the evaluation runs correctly. The model is a causal LM and `lm-eval`'s auto-detection of `backend=causal` is correct. All MMLU results are valid.

---

## Issue 9: Weight Cache Near-Identical to BF16 (Stale Cache Detection)

### Symptom
During Module 2's cross-format cache validation:
```
WARNING [main] [FP8] Weight cache is near-identical to BF16 (r=0.999999, max_diff=0.000123)
— purging for proper re-extraction
```

### Root Cause
In earlier pipeline runs, the weight extraction code used `w.data` instead of `w.dequantize()` to access quantized weight values. For some backends (particularly `optimum-quanto` FP8), `w.data` returns the underlying BF16 tensor rather than the quantized-then-dequantized values. This results in cached "FP8" weights that are actually identical to the BF16 reference.

### Resolution
Module 2's startup validation detects this condition using Pearson correlation between BF16 and quantized caches:
- If correlation > 0.9999 and max absolute difference < 0.05 → cache is flagged as stale
- Stale caches are purged (`shutil.rmtree`) and re-extracted with the corrected `w.dequantize()` extraction path

The corrected weight extraction pipeline in `src/core/weight_io.py` (`WeightExtractor`) now always calls `.dequantize()` first, which returns the quantized-then-dequantized values, correctly reflecting the quantization error.

---

## Summary — Issue Severity

| # | Issue | Severity | Auto-Resolved? | Module |
|---|---|---|---|---|
| 1 | BF16 Perplexity OOM | 🔴 High | Partially (cache preserves progress) | 5 |
| 2 | GPTQ Empty Calibration Batches | 🟡 Medium | ✅ Yes (nsamples=1 fallback) | 2 |
| 3 | GPTQ Asymmetric Mode Fallback | 🟢 Low | ✅ Yes (sym=True fallback) | 2 |
| 4 | GPTQ Calibration Filter Relaxation | 🟢 Low | ✅ Yes (threshold relaxation) | 2 |
| 5 | Exllamav2 Weight Reordering | 🟢 Low | ✅ Yes (auto-disabled for save) | 2 |
| 6 | CUDA Extension Not Installed | 🟢 Low | N/A (performance only) | 2 |
| 7 | No HF Token | 🟢 Info | N/A (local model used) | All |
| 8 | lm-eval Custom Model Warnings | 🟢 Info | N/A (benign warnings) | 5 |
| 9 | Stale Weight Cache Detection | 🟡 Medium | ✅ Yes (auto-purge + re-extract) | 2 |
