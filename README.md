# Sarvam-30B Quantization Research & MxMoE Pipeline

> **Comprehensive quantization research and sensitivity-aware mixed-precision compression for the [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b) Mixture-of-Experts language model.**

| Property | Value |
|----------|-------|
| **Model** | `sarvamai/sarvam-30b` — MoE architecture, 32B total parameters, 2.4B active per token |
| **Hardware** | 2× NVIDIA A100 80GB PCIe (160 GB total VRAM) |
| **Python** | ≥ 3.10 |
| **Frameworks** | PyTorch 2.10.0, Transformers 4.57.6, llm-compressor, vLLM |

---

## Overview

This project is organized into **two complementary pipelines** that share a unified core library:

1. **Research Pipeline** — Systematically quantizes the model across 4 methods (INT8, FP8, NF4, GPTQ), profiles performance, and produces comparative analysis.
2. **MxMoE Pipeline** — Uses sensitivity-aware expert profiling (Fisher Information + routing statistics) to apply heterogeneous mixed-precision quantization at per-expert granularity, then validates, profiles, and publishes the compressed model.

Both pipelines are driven by a shared `ModuleRunner` base class, hierarchical YAML configs with `_base_` inheritance, and a common artifact contract.

---

## Project Structure

```
sarvam-ai-30b-llm-quantization/
│
├── pyproject.toml                    # Single source of truth for dependencies
├── config.yaml                       # Top-level orchestrator config
├── main.py                           # Unified CLI entry point
│
├── configs/
│   ├── base.yaml                     # Shared: model, hardware, storage
│   ├── research.yaml                 # Research pipeline overrides (_base_: base.yaml)
│   └── mxmoe.yaml                   # MxMoE pipeline overrides  (_base_: base.yaml)
│
├── src/                              # Shared core library
│   ├── core/                         # Config, logging, runner, artifacts, calibration
│   ├── quantization/                 # BaseQuantizer + 5 method implementations
│   ├── analysis/                     # Weight distribution, MSE heatmap, outlier detection
│   ├── evaluation/                   # Perplexity, benchmarks, Pareto analysis
│   ├── profiling/                    # Latency, VRAM, disk measurement
│   ├── visualization/                # Shared plot utilities + color palette
│   └── mxmoe/                       # MxMoE-specific: sensitivity, recipe, deployment
│       ├── sensitivity/              #   Fisher info, expert routing, importance map
│       ├── recipe/                   #   Recipe builder + model compressor
│       ├── ablation/                 #   Evaluation runner + ablation study
│       ├── deployment/               #   vLLM profiler, model card, HF publisher
│       └── visualization/            #   Pareto frontier, precision heatmap
│
├── pipelines/                        # Pipeline orchestrators + module runners
│   ├── research/
│   │   ├── modules.py                # 6 ModuleRunner subclasses
│   │   └── pipeline.py               # Research pipeline CLI + orchestrator
│   └── mxmoe/
│       ├── modules.py                # 5 ModuleRunner subclasses
│       └── pipeline.py               # MxMoE pipeline CLI + orchestrator
│
├── tests/                            # Test suite
│   ├── test_runner.py                # Cross-pipeline test coordinator
│   ├── research/                     # Research integration tests
│   └── mxmoe/                       # MxMoE integration tests
│
├── model_registry/                   # HuggingFace cache (gitignored)
├── research/outputs/                 # Research artifacts  (gitignored)
└── mxmoe/outputs/                   # MxMoE artifacts     (gitignored)
```

---

## Quick Start

### 1. Install

```bash
# Core dependencies
pip install -e .

# Research pipeline (INT8, FP8, NF4, GPTQ quantizers + benchmarks)
pip install -e ".[research]"

# MxMoE pipeline — requires two separate venvs due to dependency conflicts
# See mxmoe/README.md for full setup instructions

# Venv 1: mxmoe (Modules 1–3: profiling, quantization, evaluation)
pip install -e ".[mxmoe]"

# Venv 2: mxmoe_vllm (Modules 4–5: vLLM inference, Hub push)
pip install vllm==0.19.0 && pip install --no-deps -e .

# Development tools
pip install -e ".[dev]"
```

### 2. Configure

Edit `configs/base.yaml` to set your model path and hardware:

```yaml
model:
  model_id: "sarvamai/sarvam-30b"
  cache_dir: "./model_registry"
  hf_token: null                      # or set HF_TOKEN env var

hardware:
  num_gpus: 2
  device_map: "auto"
  primary_cuda_index: 1
  max_memory:
    "0": "75GiB"
    "1": "75GiB"
```

Pipeline-specific settings are in `configs/research.yaml` and `configs/mxmoe.yaml`, which inherit from `base.yaml` via the `_base_` key.

### 3. Run

```bash
# ── Run everything ──────────────────────────────────────
python main.py                                    # Both pipelines, all modules

# ── Research Pipeline ───────────────────────────────────
python main.py --pipeline research                # All 6 research modules
python main.py --pipeline research --module 1 2   # Specific modules
python main.py --pipeline research --module 2 --quantizer gptq fp8

# Or directly:
python -m pipelines.research.pipeline
python -m pipelines.research.pipeline --module 1 --config configs/research.yaml

# ── MxMoE Pipeline ─────────────────────────────────────
python main.py --pipeline mxmoe                   # All 5 MxMoE modules
python main.py --pipeline mxmoe --module 1 2      # Sensitivity + Synthesis

# Or directly:
python -m pipelines.mxmoe.pipeline
python -m pipelines.mxmoe.pipeline --module 1 --config configs/mxmoe.yaml

# ── Advanced ────────────────────────────────────────────
python main.py --pipeline all \
    --research-module 2 --research-quantizer fp8 \
    --mxmoe-module 1

python main.py --dry-run                          # Print commands without executing
```

---

## Research Pipeline

A 6-module pipeline that systematically quantizes and benchmarks Sarvam-30B across multiple methods. See [research/README.md](research/README.md) for full details, results, and module documentation.

| Module | Name | Purpose | Approx. Time |
|--------|------|---------|-------------|
| 1 | BF16 Baseline | Load model, measure memory, cache weights | ~2.7 min |
| 2 | Quantization Matrix | INT8, FP8, NF4, GPTQ quantization | ~4.8 hr |
| 3 | Weight Introspection | Distribution analysis, MSE heatmaps, outlier detection | ~7.8 min |
| 4 | Inference Profiling | Latency (tok/s), VRAM usage, disk footprint | ~17 min |
| 5 | Evaluation & Accuracy | Perplexity, MMLU, benchmarks, Pareto frontier | ~19.8 hr |
| 6 | Layer Visualization | Per-layer weight distribution histograms | ~0.8 min |

---

## MxMoE Pipeline

A 5-module production pipeline for heterogeneous mixed-precision quantization using sensitivity-aware expert profiling. See [mxmoe/README.md](mxmoe/README.md) for full details, environment setup, results, and module documentation.

| Module | Name | Purpose | Approx. Time |
|--------|------|---------|-------------|
| 1 | Sensitivity Profiling | Fisher Information, routing stats, importance map | ~20 min |
| 2 | Mixed-Precision Synthesis | 2-pass compression (GPTQ + FP8/INT8) via llm-compressor | ~1.9 hr |
| 3 | Evaluation & Ablation | Perplexity, 6 benchmarks, ablation over bit-widths | ~34 hr |
| 4 | Deployment Readiness | vLLM latency profiling, model card, Pareto analysis | ~35 min |
| 5 | Hub Publication | Push compressed model to HuggingFace Hub | ~30 min |

---

## Architecture

### Core Library (`src/core/`)

| Module | Purpose |
|--------|---------|
| `runner.py` | `ModuleRunner` base class — timing, error handling, status tracking, JSON persistence |
| `config.py` | YAML loader with `_base_` config inheritance and path validation |
| `artifacts.py` | `ResearchArtifacts` — cross-pipeline artifact contract (Research → MxMoE) |
| `calibration.py` | Shared calibration data loading (unified across all quantizers) |
| `logger.py` | Structured logging with per-module and unified pipeline logs |
| `memory.py` | GPU memory tracking, peak memory, model cleanup |
| `device.py` | Hardware detection, multi-GPU memory maps |
| `weight_io.py` | `WeightExtractor` + `WeightCache` for NPZ weight sampling |
| `auth.py` | HuggingFace token resolution plus model path management |
| `paths.py` | Per-module directory scoping for outputs, logs, plots, results |

### Config Inheritance

```
configs/base.yaml          ← Model, hardware, storage (shared)
    ↑ _base_
configs/research.yaml      ← Quantizers, benchmarks, visualization targets
configs/mxmoe.yaml         ← Sensitivity, recipe, deployment, ablation
```

Pipeline configs reference `_base_: "base.yaml"` which is deep-merged at load time — overrides win.

### ModuleRunner Pattern

Every pipeline module inherits from `ModuleRunner`, which provides:

```python
class MyModule(ModuleRunner):
    MODULE_NUM = 1
    MODULE_NAME = "My Module"

    def execute(self):
        result = self.run_submodule("step_1", lambda: do_step_1())
        self.run_submodule("step_2", lambda: do_step_2(result))
```

The base class handles: header/footer logging ▸ timing ▸ try/except per submodule ▸ status determination (✓/⚠/✗) ▸ JSON summary persistence.

---

## Hardware Requirements

| Requirement | Specification |
|-------------|---------------|
| **GPU** | 2× NVIDIA A100 80GB PCIe (160 GB VRAM total) |
| **RAM** | ≥ 64 GB system memory |
| **Disk** | ≥ 200 GB free (model downloads + quantized checkpoints) |
| **CUDA** | ≥ 12.0 |

The BF16 model requires ~64 GB VRAM to load. Quantized checkpoints require less VRAM but the pipeline loads models sequentially to avoid GPU memory fragmentation.

---

## Outputs

```
research/outputs/
├── shared_weights/           # NPZ weight caches (bf16, int8, fp8, nf4, gptq)
├── module_1_baseline/        # BF16 results + model architecture
├── module_2_quantization/    # Per-method quantization results
├── module_3_analysis/        # Weight analysis, MSE heatmaps, outlier reports
├── module_4_profiling/       # Latency, VRAM, disk measurements
├── module_5_evaluation/      # Perplexity, benchmarks, Pareto analysis
├── module_6_visualization/   # Publication-ready layer histograms
└── pipeline_summary.json     # Aggregate status across all runs

mxmoe/outputs/
├── module_1_sensitivity/     # Fisher scores, routing stats, importance map
├── module_2_synthesis/       # Recipe + compression summary
├── module_3_evaluation/      # Benchmarks, ablation study results
├── module_4_deployment/      # vLLM profiling, model card, visualizations
├── module_5_publication/     # Hub upload status
└── pipeline_summary.json
```

---

## Documentation

Detailed module-wise execution results, analysis, and troubleshooting guides:

| Pipeline | Documentation | Description |
|---|---|---|
| **Research** | [Research Overview & Results](research/README.md) | Quantization comparison (INT8/FP8/NF4/GPTQ), profiling, evaluation |
| **MxMoE** | [MxMoE Overview & Results](mxmoe/README.md) | Pipeline setup, module results, benchmarks, deployment analysis |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Lint
ruff check src/ pipelines/ tests/

# Run tests
pytest tests/
```

---

## License

Same as the base model: [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b)
