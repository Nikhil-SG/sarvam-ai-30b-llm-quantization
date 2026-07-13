#!/usr/bin/env python3
"""
MxMoE pipeline module runners.

Each class inherits from ModuleRunner, which handles timing,
error tracking, status determination, and JSON persistence.

Implements the execution logic for each module.
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.runner import ModuleRunner
from src.core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Module 1: Sensitivity-Aware Profiling
# ═══════════════════════════════════════════════════════════════════════════

class SensitivityProfilingModule(ModuleRunner):
    """Module 1 — Fisher Information, routing stats, importance map."""

    MODULE_NUM = 1
    MODULE_NAME = "Sensitivity-Aware Profiling"

    def execute(self):
        from src.mxmoe.sensitivity.fisher_info import FisherInformationAnalyzer
        from src.mxmoe.sensitivity.expert_router_stats import ExpertRoutingAnalyzer
        from src.mxmoe.sensitivity.importance_map import ImportanceMapBuilder

        # 1a — Fisher Information
        fisher_scores = self.run_submodule(
            "fisher_info",
            lambda: FisherInformationAnalyzer(self.config).run(),
        ) or {}

        # 1b — Expert Routing Statistics
        routing_stats = self.run_submodule(
            "routing_stats",
            lambda: ExpertRoutingAnalyzer(self.config).run(),
        ) or {}

        # 1c — Importance Map (depends on 1a + 1b)
        def _build_importance_map():
            builder = ImportanceMapBuilder(self.config)
            importance_map = builder.build(fisher_scores, routing_stats)
            builder.save(importance_map)
            return importance_map

        self.run_submodule("importance_map", _build_importance_map)


# ═══════════════════════════════════════════════════════════════════════════
# Module 2: Mixed-Precision Synthesis
# ═══════════════════════════════════════════════════════════════════════════

class MixedPrecisionSynthesisModule(ModuleRunner):
    """Module 2 — Recipe generation + model compression."""

    MODULE_NUM = 2
    MODULE_NAME = "Mixed-Precision Synthesis"

    def execute(self):
        from src.mxmoe.recipe.recipe_builder import RecipeBuilder
        from src.mxmoe.recipe.compressor import ModelCompressor

        # 2a — Build precision recipe
        recipe = self.run_submodule(
            "recipe_generation",
            lambda: RecipeBuilder(self.config).build(),
            critical=True,  # can't compress without a recipe
        )

        # 2b — Compress model
        def _compress():
            compressor = ModelCompressor(self.config)
            strategies = getattr(self.config.recipe, "strategies", ["fp8_gptq"])
            results = {}
            for strategy_name in strategies:
                logger.info(f"Starting model compression for strategy: {strategy_name}")
                results[strategy_name] = compressor.compress(recipe, strategy_name=strategy_name)
                # Cleanup GPU and CPU memory between strategies
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return results

        self.run_submodule("model_compression", _compress, critical=True)


# ═══════════════════════════════════════════════════════════════════════════
# Module 3: Evaluation & Ablation
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationAblationModule(ModuleRunner):
    """Module 3 — Perplexity, benchmarks, and ablation study."""

    MODULE_NUM = 3
    MODULE_NAME = "Evaluation & Ablation"

    def execute(self):
        from src.mxmoe.ablation.ablation_study import EvaluationRunner, AblationRunner

        # 3a — Perplexity & Benchmarks
        self.run_submodule(
            "evaluation",
            lambda: EvaluationRunner(self.config).run_full_evaluation(),
        )

        # 3b — Ablation Study
        self.run_submodule(
            "ablation",
            lambda: AblationRunner(self.config).run(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Module 4: Deployment Readiness
# ═══════════════════════════════════════════════════════════════════════════

class DeploymentReadinessModule(ModuleRunner):
    """Module 4 — vLLM profiling, model card, and visualization."""

    MODULE_NUM = 4
    MODULE_NAME = "Deployment Readiness"

    def execute(self):
        from src.mxmoe.deployment.vllm_profiler import VLLMProfiler
        from src.mxmoe.deployment.model_card import TechnicalModelCard
        from src.mxmoe.deployment.strategy_profiler import StrategyProfiler
        from src.evaluation.pareto import ParetoAnalyzer
        from src.core.paths import get_mxmoe_module_paths
        from pathlib import Path
        import json

        # 4a — Strategy Profiling (Latency/VRAM/Disk)
        self.run_submodule(
            "strategy_profiling",
            lambda: StrategyProfiler(self.config).run(),
        )

        # 4b — Pareto Analysis (quality vs throughput)
        def _run_pareto():
            base_dir = getattr(self.config.output, "base_dir", "mxmoe/outputs")
            module3_results = Path(get_mxmoe_module_paths(base_dir, 3)["results_dir"])

            bench_path = module3_results / "benchmark_results.json"
            latency_path = Path(self.config.output.results_dir) / "latency_results.json"

            def _load_json(path: Path):
                if not path.exists():
                    return {}
                try:
                    return json.loads(path.read_text())
                except Exception:
                    return {}

            if not bench_path.exists():
                self.logger.warning(
                    f"  ⚠ Module 3 benchmark results not found at {bench_path}. "
                    "Run Module 3 evaluation before Module 4 for a complete Pareto plot."
                )
            if not latency_path.exists():
                self.logger.warning(
                    f"  ⚠ Module 4 latency results not found at {latency_path}. "
                    "Run strategy profiling before Pareto analysis."
                )

            bench_data = _load_json(bench_path)
            latency_data = _load_json(latency_path)

            if not bench_data or not latency_data:
                return {
                    "status": "skipped",
                    "benchmarks_available": bool(bench_data),
                    "latency_available": bool(latency_data),
                }

            pareto = ParetoAnalyzer(self.config)
            tags = sorted(set(bench_data.keys()) | set(latency_data.keys()))
            plot_path = pareto.plot(latency_data, bench_data, tags=tags)
            return {
                "status": "success",
                "plot_path": str(plot_path),
                "benchmarks_available": True,
                "latency_available": True,
            }

        self.run_submodule(
            "pareto_analysis",
            _run_pareto,
        )

        # 4c — vLLM Profiling
        # Aggressively free GPU memory left from strategy_profiling
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Reset peak memory stats to get clean readings
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        import time as _time
        _time.sleep(3)  # let GPU memory settle

        self.run_submodule(
            "vllm_profiling",
            lambda: VLLMProfiler(self.config).run(),
        )

        # 4d — Model Card
        self.run_submodule(
            "model_card",
            lambda: TechnicalModelCard(self.config).build(),
        )

        # 4e — Visualization
        def _generate_viz():
            from src.mxmoe.visualization.pareto_frontier import plot_pareto_frontier
            from src.mxmoe.visualization.precision_heatmap import plot_precision_heatmap

            base_dir = getattr(self.config.output, "base_dir", "mxmoe/outputs")
            plots_dir = getattr(self.config.output, "plots_dir", None)

            pareto = plot_pareto_frontier(results_dir=base_dir, output_dir=plots_dir)
            heatmap = plot_precision_heatmap(results_dir=base_dir, output_dir=plots_dir)
            return {"pareto": pareto, "heatmap": heatmap}

        self.run_submodule("visualization", _generate_viz)


# ═══════════════════════════════════════════════════════════════════════════
# Module 5: Hub Publication
# ═══════════════════════════════════════════════════════════════════════════

class HubPublicationModule(ModuleRunner):
    """Module 5 — Push to HuggingFace Hub."""

    MODULE_NUM = 5
    MODULE_NAME = "Hub Publication"

    def execute(self):
        from src.mxmoe.deployment.hf_publisher import HFPublisher

        # 5a — Validation
        self.run_submodule(
            "validation_check",
            lambda: self._check_readiness(),
        )

        # 5b — Publish
        self.run_submodule(
            "hub_push",
            lambda: HFPublisher(self.config).publish(),
        )

    def _check_readiness(self):
        """Check that Module 4 outputs exist before publishing."""
        from pathlib import Path
        from src.core.paths import get_mxmoe_module_paths
        base_dir = getattr(self.config.output, "base_dir", "mxmoe/outputs")
        mod4_paths = get_mxmoe_module_paths(base_dir, 4)
        card_path = Path(mod4_paths["results_dir"])
        if not card_path.exists():
            self.logger.warning(
                f"Module 4 results not found at {card_path} — publishing may be incomplete"
            )
        return {"ready": card_path.exists()}


# ── Registry ────────────────────────────────────────────────────────────

MODULE_MAP = {
    1: SensitivityProfilingModule,
    2: MixedPrecisionSynthesisModule,
    3: EvaluationAblationModule,
    4: DeploymentReadinessModule,
    5: HubPublicationModule,
}

MODULE_NAMES = {
    1: "Sensitivity Profiling",
    2: "Mixed-Precision Synthesis",
    3: "Evaluation & Ablation",
    4: "Deployment Readiness",
    5: "Hub Publication",
}
