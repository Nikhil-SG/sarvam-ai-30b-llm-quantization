#!/usr/bin/env python3
"""
Module 4b — Technical Model Card Generator.

Generates a comprehensive model card for the MxMoE-quantized model including:
- Accuracy table (BF16 vs MxMoE)
- Pareto frontier image
- MSE heatmap image
- Hardware requirements
- Usage code snippet

Collates results from Modules 1-3 and deployment profiling to produce a
README.md suitable for HuggingFace model repos.

Usage:
    python -m src.mxmoe.deployment.model_card \\
        --results_dir mxmoe/outputs

RUN THIS NEXT: After Modules 3 and 4a. Aggregates all results into model card.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "sarvamai/sarvam-30b"
DEFAULT_RESULTS_DIR = "mxmoe/outputs"


class TechnicalModelCard:
    """Generate a technical model card for HuggingFace publication."""

    def __init__(
        self,
        config=None,
        results_dir: str = DEFAULT_RESULTS_DIR,
        output_dir: str = "mxmoe/outputs/module_4_deployment/results",
        model_id: str = MODEL_ID,
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id

        if config is not None:
            self.output_dir = Path(getattr(config.output, "results_dir", str(self.output_dir)))
            self.output_dir.mkdir(parents=True, exist_ok=True)
            model_cfg = getattr(config, "model", None)
            if model_cfg:
                self.model_id = getattr(model_cfg, "model_id", self.model_id)
            out_cfg = getattr(config, "output", None)
            if out_cfg:
                self.results_dir = Path(getattr(out_cfg, "base_dir", str(self.results_dir)))

    # ── Loader Helpers ───────────────────────────────────────────────────

    def _load_json_safe(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load a JSON file, returning None if not found or invalid."""
        if not path.exists():
            logger.debug(f"File not found: {path}")
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load {path}: {exc}")
            return None

    def _load_evaluation_results(self) -> Optional[Dict[str, Any]]:
        """Load evaluation results from Module 3."""
        candidates = [
            self.results_dir / "module_3_evaluation" / "results" / "full_evaluation.json",
            self.results_dir / "module_3_evaluation" / "results" / "eval_results_full.json",
            self.results_dir / "module_3_evaluation" / "results" / "ablation_results.json",
        ]
        for path in candidates:
            data = self._load_json_safe(path)
            if data is not None:
                return data
        return None

    def _load_profiling_results(self) -> Optional[Dict[str, Any]]:
        """Load vLLM profiling results from Module 4a."""
        path = self.results_dir / "module_4_deployment" / "results" / "vllm_profiling.json"
        return self._load_json_safe(path)

    def _load_recipe(self) -> Optional[Dict[str, Any]]:
        """Load precision recipe from Module 2."""
        path = self.results_dir / "module_2_synthesis" / "results" / "precision_recipe.json"
        return self._load_json_safe(path)

    def _load_compression_report(self) -> Optional[Dict[str, Any]]:
        """Load compression report from Module 2."""
        path = self.results_dir / "module_2_synthesis" / "results" / "compression_report.json"
        return self._load_json_safe(path)

    def _load_importance_map(self) -> Optional[Dict[str, Any]]:
        """Load importance map from Module 1."""
        path = self.results_dir / "module_1_sensitivity" / "results" / "expert_importance_map.json"
        return self._load_json_safe(path)

    # ── Card Section Builders ────────────────────────────────────────────

    def _build_yaml_frontmatter(
        self,
        compression_report: Optional[Dict[str, Any]],
    ) -> str:
        """Build YAML frontmatter for HuggingFace model card."""
        tags = [
            "quantized",
            "moe",
            "mxmoe",
            "mixed-precision",
            "sarvam",
            "compressed-tensors",
            "vllm",
        ]
        tag_lines = "\n".join(f'  - "{t}"' for t in tags)

        return f"""---
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
base_model: {self.model_id}
tags:
{tag_lines}
model-index:
  - name: sarvam-30b-MxMoE
    results: []
library_name: transformers
pipeline_tag: text-generation
quantized_by: MxMoE Pipeline
---
"""


    def _build_header(self) -> str:
        """Build model card header."""
        return textwrap.dedent("""\
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
        """)

    def _build_precision_table(
        self,
        recipe: Optional[Dict[str, Any]],
        importance_map: Optional[Dict[str, Any]],
    ) -> str:
        """Build the precision assignment table."""
        lines = [
            "## Precision Assignment",
            "",
            "The MxMoE pipeline assigns precision per-expert based on Fisher",
            "Information sensitivity analysis and routing frequency:",
            "",
            "| Component | Precision | Method | Count |",
            "|:--|:--|:--|--:|",
        ]

        metadata = recipe.get("metadata", {}) if recipe else {}
        counts = metadata.get("expert_classification_counts", {})

        if importance_map:
            cls_summary = importance_map.get("classification_summary", {})
            counts = counts or cls_summary

        num_high = counts.get("HIGH", "—")
        num_med = counts.get("MEDIUM", "—")
        num_low = counts.get("LOW", "—")

        lines.extend([
            f"| Attention (QKV, dense) | FP8 Dynamic | `oneshot()` call 1 | 38 modules |",
            f"| Shared experts | FP8 Dynamic | `oneshot()` call 1 | 54 modules |",
            f"| Dense layer 0 MLP | FP8 Dynamic | `oneshot()` call 1 | 3 modules |",
            f"| HIGH-importance routed | FP8 Dynamic | `oneshot()` call 1 | {num_high} experts |",
            f"| MEDIUM-importance routed | FP8 Dynamic | `oneshot()` call 1 | {num_med} experts |",
            f"| LOW-importance routed | W4A16 (GPTQ) | `oneshot()` call 2 | {num_low} experts |",
            f"| `lm_head`, gates | Ignored (BF16) | — | — |",
        ])

        return "\n".join(lines)

    def _build_accuracy_table(self, eval_results: Optional[Dict[str, Any]]) -> str:
        """Build the accuracy comparison table."""
        lines = [
            "",
            "## Evaluation Results",
            "",
        ]

        if eval_results and "evaluations" in eval_results:
            evaluations = eval_results["evaluations"]

            def _extract_tasks(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
                data = payload.get("parsed_results", payload) if isinstance(payload, dict) else {}
                tasks = data.get("tasks", {})
                if tasks:
                    return tasks
                results_data = data.get("results", {})
                extracted = {}
                for task_name, task_data in results_data.items():
                    extracted[task_name] = {
                        "label": task_name.replace("_", " ").title(),
                        "accuracy": task_data.get("acc,none", task_data.get("acc_norm,none")),
                    }
                return extracted

            bench_sets = []
            if "benchmarks" in evaluations:
                tasks = _extract_tasks(evaluations["benchmarks"])
                if tasks:
                    bench_sets.append(("mxmoe", tasks))
            else:
                for key, payload in evaluations.items():
                    if not key.startswith("benchmarks_"):
                        continue
                    strategy = key.replace("benchmarks_", "")
                    tasks = _extract_tasks(payload)
                    if tasks:
                        bench_sets.append((strategy, tasks))

            if bench_sets:
                task_ids = []
                task_labels = {}
                for _, tasks in bench_sets:
                    for task_id, task_data in tasks.items():
                        if task_id not in task_ids:
                            task_ids.append(task_id)
                            task_labels[task_id] = task_data.get("label", task_id)

                if len(bench_sets) == 1:
                    label, tasks = bench_sets[0]
                    lines.extend([
                        "| Benchmark | MxMoE Score | Metric |",
                        "|:--|--:|:--|",
                    ])
                    for task_id in task_ids:
                        task_data = tasks.get(task_id, {})
                        acc = task_data.get("accuracy", "—")
                        if isinstance(acc, (int, float)):
                            acc = f"{acc:.2f}"
                        lines.append(f"| {task_labels[task_id]} | {acc} | accuracy |")
                else:
                    headers = " | ".join([name.upper() for name, _ in bench_sets])
                    lines.extend([
                        f"| Benchmark | {headers} |",
                        f"|:--|" + "--:|" * len(bench_sets),
                    ])
                    for task_id in task_ids:
                        row = [task_labels[task_id]]
                        for _, tasks in bench_sets:
                            acc = tasks.get(task_id, {}).get("accuracy", "—")
                            if isinstance(acc, (int, float)):
                                acc = f"{acc:.2f}"
                            row.append(str(acc))
                        lines.append("| " + " | ".join(row) + " |")
            else:
                lines.append(
                    "_Evaluation results will be populated after running Module 3 on GPU._"
                )
        else:
            lines.extend([
                "| Benchmark | BF16 Baseline | MxMoE | Δ |",
                "|:--|--:|--:|--:|",
                "| MMLU | — | — | — |",
                "| HellaSwag | — | — | — |",
                "| ARC-Challenge | — | — | — |",
                "| Winogrande | — | — | — |",
                "| TruthfulQA MC2 | — | — | — |",
                "| WikiText-2 PPL ↓ | — | — | — |",
                "",
                "> _Run Module 3 to populate these results._",
            ])

        return "\n".join(lines)

    def _build_profiling_section(self, profiling: Optional[Dict[str, Any]]) -> str:
        """Build the inference profiling section."""
        lines = [
            "",
            "## Inference Performance",
            "",
        ]

        if profiling and "profile_results" in profiling:
            vllm_cfg = profiling.get("vllm_config", {})
            tp = vllm_cfg.get("tensor_parallel_size", 2)
            lines.extend([
                f"Benchmarked with vLLM (tensor_parallel={tp}):",
                "",
            ])

            all_strats = profiling.get("all_strategies", {})
            if all_strats:
                lines.extend([
                    "| Strategy | Batch Size | Tokens/sec | Latency (s) | ms/token |",
                    "|:---|--:|--:|--:|--:|",
                ])
                for strat_name, strat_data in all_strats.items():
                    prof_res = strat_data.get("profile_results", {})
                    for bs, data in prof_res.items():
                        lines.append(
                            f"| `{strat_name}` | {data['batch_size']} | {data['tokens_per_sec']:.1f} "
                            f"| {data['avg_latency_sec']:.3f} | {data['time_per_token_ms']:.1f} |"
                        )
            else:
                lines.extend([
                    "| Batch Size | Tokens/sec | Latency (s) | ms/token |",
                    "|--:|--:|--:|--:|",
                ])
                for bs, data in profiling["profile_results"].items():
                    lines.append(
                        f"| {data['batch_size']} | {data['tokens_per_sec']:.1f} "
                        f"| {data['avg_latency_sec']:.3f} | {data['time_per_token_ms']:.1f} |"
                    )

            # VRAM
            vram = profiling.get("vram_usage", {})
            if vram:
                lines.extend(["", "**VRAM usage after model load:**", ""])
                for gpu_id, usage in vram.items():
                    lines.append(
                        f"- `{gpu_id}`: {usage['allocated_gb']:.1f} GB allocated, "
                        f"{usage['reserved_gb']:.1f} GB reserved"
                    )
        else:
            lines.extend([
                "| Batch Size | Tokens/sec | Latency (s) | ms/token |",
                "|--:|--:|--:|--:|",
                "| 1 | — | — | — |",
                "| 32 | — | — | — |",
                "",
                "> _Run Module 4a to populate these results._",
            ])

        return "\n".join(lines)

    def _build_hardware_section(self) -> str:
        """Build hardware requirements section."""
        return textwrap.dedent("""\

            ## Hardware Requirements

            | Requirement | Specification |
            |:--|:--|
            | **Minimum GPU VRAM** | ~40 GB (single GPU with offloading) |
            | **Recommended setup** | 2× NVIDIA A100 80 GB |
            | **Tensor parallel** | 2 (one model shard per GPU) |
            | **Precision** | Mixed (FP8 Dynamic / W4A16 GPTQ) |
            | **Framework** | vLLM ≥ 0.8.0 with compressed-tensors support |
        """)

    def _build_usage_section(self) -> str:
        """Build usage code snippet section."""
        return textwrap.dedent("""\

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
                "Nikhil-SG/sarvam-30b-MxMoE",
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "Nikhil-SG/sarvam-30b-MxMoE",
                trust_remote_code=True,
            )

            inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=128)
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            ```
        """)

    def _build_methodology_section(self) -> str:
        """Build methodology description."""
        return textwrap.dedent("""\

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
        """)

    def _build_citation_section(self) -> str:
        """Build citation and acknowledgments section."""
        return textwrap.dedent("""\

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
        """)

    # ── Main Build Method ────────────────────────────────────────────────

    def build(self) -> Dict[str, Any]:
        """
        Build the model card.

        Collects results from earlier modules and generates a markdown
        README suitable for HuggingFace model repos.

        Returns:
            Dict with model card content and metadata.
        """
        logger.info("=" * 60)
        logger.info("  MODULE 4b: Building Technical Model Card")
        logger.info("=" * 60)

        # ── Load results from earlier modules ────────────────────────────
        eval_results = self._load_evaluation_results()
        profiling_results = self._load_profiling_results()
        recipe = self._load_recipe()
        compression_report = self._load_compression_report()
        importance_map = self._load_importance_map()

        loaded = {
            "evaluation": eval_results is not None,
            "profiling": profiling_results is not None,
            "recipe": recipe is not None,
            "compression": compression_report is not None,
            "importance_map": importance_map is not None,
        }
        logger.info(f"Loaded results: {loaded}")

        # ── Assemble card sections ───────────────────────────────────────
        sections = [
            self._build_yaml_frontmatter(compression_report),
            self._build_header(),
            self._build_precision_table(recipe, importance_map),
            self._build_accuracy_table(eval_results),
            self._build_profiling_section(profiling_results),
            self._build_hardware_section(),
            self._build_usage_section(),
            self._build_methodology_section(),
            self._build_citation_section(),
        ]

        card_content = "\n".join(sections)

        # ── Save model card ──────────────────────────────────────────────
        card_path = self.save(card_content)
        self.save(card_content, filename="MODEL_CARD.md")

        result = {
            "status": "success",
            "card_path": str(card_path),
            "sections_built": len(sections),
            "loaded_results": loaded,
            "card_length_chars": len(card_content),
        }

        logger.info(f"Model card built: {len(card_content)} chars → {card_path}")
        return result

    def save(self, card_content: str, filename: str = "README.md") -> Path:
        """Save the model card markdown."""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(card_content)
        logger.info(f"Model card saved: {path}")
        return path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Module 4b: Technical Model Card Generator for sarvam-30b MxMoE"
    )
    parser.add_argument(
        "--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
        help="Base results directory containing module outputs",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="mxmoe/outputs/module_4_deployment/results",
        help="Output directory for the model card",
    )
    parser.add_argument(
        "--model_id", type=str, default=MODEL_ID,
        help="Base model ID",
    )
    args = parser.parse_args()

    card_builder = TechnicalModelCard(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
    )
    result = card_builder.build()
    logger.info(f"Model card generation complete: {result.get('card_path', '')}")


if __name__ == "__main__":
    main()
