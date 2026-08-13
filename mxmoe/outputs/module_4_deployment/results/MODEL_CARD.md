---
language:
  - en
  - hi
  - ta
  - te
  - kn
  - ml
  - mr
  - bn
  - gu
  - or
license: apache-2.0
base_model: sarvamai/sarvam-30b
tags:
  - "quantized"
  - "moe"
  - "mxmoe"
  - "mixed-precision"
  - "sarvam"
  - "compressed-tensors"
  - "vllm"
model-index:
  - name: sarvam-30b-MxMoE
    results: []
library_name: transformers
pipeline_tag: text-generation
quantized_by: MxMoE Pipeline
---

# sarvam-30b-MxMoE: Mixed-Precision Mixture-of-Experts Quantization (Model Card)

> **Heterogeneous mixed-precision quantization** of
> [`sarvamai/sarvam-30b`](https://huggingface.co/sarvamai/sarvam-30b)
> using the **MxMoE** pipeline. Each expert is quantized to a precision
> matched to its importance, achieving significant compression with
> minimal quality loss.

## Model Overview

| Property | Value |
|:--|:--|
| **Base model** | `sarvamai/sarvam-30b` |
| **Architecture** | Mixture-of-Experts (MoE): 32B total, 2.4B active per token |
| **MoE layers** | 18 layers (indices 1–18), 128 routed experts per layer |
| **Routing** | Top-6 sigmoid with expert bias |
| **Dense layer** | Layer 0 (non-MoE) |
| **Quantization method** | MxMoE heterogeneous mixed-precision |
| **Format** | `compressed-tensors` (vLLM compatible) |

## Precision Assignment

The MxMoE pipeline assigns precision per-expert based on Fisher
Information sensitivity analysis and routing frequency:

| Component | Precision | Method | Count |
|:--|:--|:--|--:|
| Attention (QKV, dense) | FP8 Dynamic | `oneshot()` call 1 | 38 modules |
| Shared experts | FP8 Dynamic | `oneshot()` call 1 | 54 modules |
| Dense layer 0 MLP | FP8 Dynamic | `oneshot()` call 1 | 3 modules |
| HIGH-importance routed | FP8 Dynamic | `oneshot()` call 1 | 1013 experts |
| MEDIUM-importance routed | FP8 Dynamic | `oneshot()` call 1 | 1030 experts |
| LOW-importance routed | W4A16 (GPTQ) | `oneshot()` call 2 | 261 experts |
| `lm_head`, gates | Ignored (BF16) | — | — |

## Evaluation Results

| Benchmark | FP8_GPTQ | INT8_GPTQ |
|:--|--:|--:|
| MMLU | 65.18 | 65.69 |
| HellaSwag | 39.95 | 40.79 |
| Winogrande | 51.30 | 52.57 |
| ARC-Challenge | 26.37 | 28.75 |
| GSM8K | 60.20 | 70.96 |
| TruthfulQA | 48.79 | 49.11 |

## Inference Performance

Benchmarked with vLLM (tensor_parallel=2):

| Strategy | Batch Size | Tokens/sec | Latency (s) | ms/token |
|:---|--:|--:|--:|--:|
| `fp8_gptq` | 1 | 27.5 | 4.662 | 36.4 |
| `fp8_gptq` | 4 | 125.0 | 4.096 | 8.0 |
| `fp8_gptq` | 8 | 245.6 | 4.170 | 4.1 |
| `fp8_gptq` | 16 | 478.4 | 4.281 | 2.1 |
| `fp8_gptq` | 32 | 962.4 | 4.256 | 1.0 |

## Hardware Requirements

| Requirement | Specification |
|:--|:--|
| **Minimum GPU VRAM** | ~40 GB (single GPU with offloading) |
| **Recommended setup** | 2× NVIDIA A100 80 GB |
| **Tensor parallel** | 2 (one model shard per GPU) |
| **Precision** | Mixed (FP8 Dynamic / W4A16 GPTQ) |
| **Framework** | vLLM ≥ 0.8.0 with compressed-tensors support |


## Usage

### vLLM (Recommended)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="your-username/sarvam-30b-MxMoE",
    tensor_parallel_size=2,
    max_model_len=4096,
    trust_remote_code=True,
    dtype="auto",
    gpu_memory_utilization=0.90,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain quantum entanglement:"], params)
print(outputs[0].outputs[0].text)
```

### Transformers (with compressed-tensors)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "your-username/sarvam-30b-MxMoE",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "your-username/sarvam-30b-MxMoE",
    trust_remote_code=True,
)

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


## Methodology: MxMoE Pipeline

The MxMoE (Mixed-precision Mixture-of-Experts) pipeline assigns
heterogeneous precision to each expert based on its measured importance:

1. **Sensitivity Profiling (Module 1)**
   - Diagonal Fisher Information estimation per expert
   - Expert routing frequency analysis across calibration data
   - Combined importance scoring → HIGH / MEDIUM / LOW classification

2. **Mixed-Precision Synthesis (Module 2)**
   - Dynamic recipe generation: each expert gets precision matching
     its importance tier
   - Compression via `llm-compressor` `oneshot` API
   - Output in `compressed-tensors` format for vLLM compatibility

3. **Evaluation & Ablation (Module 3)**
   - Perplexity (WikiText-2) and benchmark evaluation via `lm-eval`
   - Ablation study sweeping LOW-expert bit-widths to find accuracy floor

4. **Deployment Profiling (Module 4)**
   - vLLM inference benchmarks at multiple batch sizes
   - Technical model card and optional HuggingFace publication


## Citation

If you use this model or the MxMoE quantization pipeline, please cite:

```bibtex
@misc{mxmoe2026,
  title={MxMoE: Heterogeneous Mixed-Precision Quantization for Mixture-of-Experts Models},
  year={2026},
  note={Applied to sarvamai/sarvam-30b}
}
```

## Acknowledgments

- Base model: [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b)
- Quantization: [llm-compressor](https://github.com/vllm-project/llm-compressor)
- Inference: [vLLM](https://github.com/vllm-project/vllm)
- Evaluation: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
