#!/usr/bin/env python3
"""
Research pipeline module runners.

Each class inherits from ModuleRunner, which handles timing,
error tracking, status determination, and JSON persistence.
The module only implements execute() with the actual domain logic.

Implements the execution logic for each module.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.core.runner import ModuleRunner
from src.core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Module 1: BF16 Baseline
# ═══════════════════════════════════════════════════════════════════════════

class BF16BaselineModule(ModuleRunner):
    """Module 1 — Load BF16 model, measure memory, cache weights."""

    MODULE_NUM = 1
    MODULE_NAME = "BF16 Baseline"

    def execute(self):
        from src.quantization.bf16_baseline import BF16Baseline

        baseline = BF16Baseline(self.config)
        result = baseline.run(cache_weights=True)
        self.results["bf16_results"] = result


# ═══════════════════════════════════════════════════════════════════════════
# Module 2: Quantisation Matrix (INT8, FP8, NF4, GPTQ)
# ═══════════════════════════════════════════════════════════════════════════

class QuantizationMatrixModule(ModuleRunner):
    """Module 2 — Run selected quantizers, cache weights, save checkpoints."""

    MODULE_NUM = 2
    MODULE_NAME = "Quantisation Matrix"

    def execute(self):
        from src.quantization.int8_quantizer import INT8Quantizer
        from src.quantization.fp8_quantizer import FP8Quantizer
        from src.quantization.nf4_quantizer import NF4Quantizer
        from src.quantization.gptq_quantizer import GPTQQuantizer

        # Cache cross-format validation (purge near-BF16 caches)
        self._validate_weight_caches()

        all_quantizers = [
            INT8Quantizer, FP8Quantizer, NF4Quantizer,
            GPTQQuantizer,
        ]

        # Apply quantizer filter if specified via --quantizer
        quantizer_filter = getattr(self.config, "_quantizer_filter", None)
        if quantizer_filter:
            all_quantizers = [
                Q for Q in all_quantizers if Q.QUANT_TAG in quantizer_filter
            ]
            if not all_quantizers:
                self.logger.warning(
                    f"No quantizers match filter: {quantizer_filter}. "
                    f"Valid: int8, fp8, nf4, gptq"
                )
                return
            self.logger.info(
                f"Quantizers to run: {[Q.QUANT_TAG for Q in all_quantizers]}"
            )

        for idx, QuantCls in enumerate(all_quantizers, 1):
            tag = QuantCls.QUANT_TAG

            def _run_quantizer(cls=QuantCls):
                q = cls(self.config)
                result = q.run(cache_weights=True)
                self.results.setdefault("quantizer_results", {})[cls.QUANT_TAG] = result
                return result

            self.run_submodule(f"{tag.upper()} quantization", _run_quantizer)

            # Aggressive inter-quantizer cleanup
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _validate_weight_caches(self):
        """Purge quantized caches that are near-identical to BF16."""
        try:
            import numpy as np
            import shutil

            shared_dir = Path(self.config.output.base_dir) / "shared_weights"
            bf16_dir = shared_dir / "bf16"
            quant_tags = ["int8", "fp8", "nf4", "gptq"]
            quantizer_filter = getattr(self.config, "_quantizer_filter", None)

            if not bf16_dir.is_dir():
                return

            bf16_files = sorted(bf16_dir.glob("*.npz"))[:1]
            if not bf16_files:
                return

            bf16_vals = np.load(bf16_files[0])["values"].astype(np.float64)
            fname = bf16_files[0].name

            for tag in quant_tags:
                tag_file = shared_dir / tag / fname
                if not tag_file.exists():
                    continue
                try:
                    tag_vals = np.load(tag_file)["values"].astype(np.float64)
                    if len(tag_vals) != len(bf16_vals):
                        continue

                    bm, tm = bf16_vals.mean(), tag_vals.mean()
                    bd, td = bf16_vals - bm, tag_vals - tm
                    denom = np.sqrt((bd ** 2).sum() * (td ** 2).sum())
                    r = float((bd * td).sum() / denom) if denom > 0 else 1.0
                    max_diff = float(np.abs(bf16_vals - tag_vals).max())

                    r_thresh = 0.999999 if tag == "int8" else 0.9999
                    d_thresh = 0.001 if tag == "int8" else 0.05

                    if r > r_thresh and max_diff < d_thresh:
                        if quantizer_filter and tag not in quantizer_filter:
                            continue
                        self.logger.warning(
                            f"[{tag.upper()}] Cache near-identical to BF16 "
                            f"(r={r:.8f}) — purging"
                        )
                        shutil.rmtree(shared_dir / tag)
                except Exception:
                    pass
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Module 3: Weight Introspection (histograms, MSE, outliers)
# ═══════════════════════════════════════════════════════════════════════════

class WeightIntrospectionModule(ModuleRunner):
    """Module 3 — Analyse weight distributions, MSE, and outliers."""

    MODULE_NUM = 3
    MODULE_NAME = "Weight Introspection"

    def execute(self):
        from src.analysis.weight_distribution import WeightDistributionAnalyzer
        from src.analysis.mse_heatmap import MSEHeatmapAnalyzer
        from src.analysis.outlier_detector import OutlierDetector
        from src.analysis.report import Module3ReportBuilder
        from src.core.weight_io import WeightCache

        # Pre-check weight caches
        _cache = WeightCache(
            cache_dir=str(self.config.output.weights_dir),
            sample_size=self.config.visualization.weight_sample_size,
        )
        cache_status = _cache.validate_caches()
        self.results["cache_validation"] = cache_status

        _has_bf16 = "bf16" in cache_status["tags"]
        if not _has_bf16:
            self.logger.warning(
                "⚠ No BF16 weight cache — run Module 1 first."
            )

        # 3a — Weight distributions
        weight_data = self.run_submodule(
            "weight_distribution",
            lambda: WeightDistributionAnalyzer(self.config).run(),
        )
        self._record_submodule_method(
            "weight_distribution",
            weight_data,
            fallback="matplotlib histogram with gaussian overlay",
        )

        # 3b — MSE heatmap
        mse_data = self.run_submodule(
            "mse_heatmap",
            lambda: MSEHeatmapAnalyzer(self.config).run(ref_tag="bf16"),
        )
        self._record_submodule_method(
            "mse_heatmap",
            mse_data,
            fallback="mean squared error with log-scale colormap",
        )

        # 3c — Outlier detection (needs live models)
        outlier_runs = self.run_submodule(
            "outlier_detection",
            lambda: self._run_outlier_detection(),
        )
        self._record_submodule_method(
            "outlier_detection",
            outlier_runs,
            fallback="forward hook activation capture across configured prompts",
        )

        # Research report
        try:
            results_dir = Path(self.config.output.results_dir)
            report_builder = Module3ReportBuilder(results_dir)
            report_builder.build(
                cache_validation=cache_status,
                mse_data=mse_data,
                outlier_runs=outlier_runs or {},
                module_summary=self.results,
            )
        except Exception as exc:
            self.logger.warning(f"Report generation failed: {exc}")

    def _record_submodule_method(
        self,
        submodule_name: str,
        payload: Any,
        *,
        fallback: str,
    ) -> None:
        """Store analysis method in the module summary for successful submodules."""
        submodule_info = self.results.get("submodules", {}).get(submodule_name)
        if not isinstance(submodule_info, dict):
            return
        if not str(submodule_info.get("status", "")).startswith("✓"):
            return

        method = None
        if isinstance(payload, dict):
            method = payload.get("method")
            if not method:
                for value in payload.values():
                    if isinstance(value, dict) and value.get("method"):
                        method = value["method"]
                        break

        submodule_info["method"] = str(method or fallback)

    def _run_outlier_detection(self) -> Dict[str, Any]:
        """Run outlier detection across all configured quantization tags."""
        from src.analysis.outlier_detector import OutlierDetector
        from src.quantization import QUANTIZER_REGISTRY

        detector = OutlierDetector(self.config)
        outlier_cfg = getattr(self.config.visualization, "outlier_detection", None)
        sigma_threshold = getattr(outlier_cfg, "sigma_threshold", 6.0)
        analyze_tags = list(getattr(outlier_cfg, "analyze_tags", ["bf16"]))
        prompt_suite = list(
            getattr(outlier_cfg, "prompts", [self.config.profiling.prompt])
        )
        skip_missing = bool(getattr(outlier_cfg, "skip_missing_checkpoints", True))

        results = {}
        for tag in analyze_tags:
            QuantCls = QUANTIZER_REGISTRY.get(tag)
            if QuantCls is None:
                continue

            quantizer = QuantCls(self.config)
            if tag != "bf16" and skip_missing and not quantizer.has_saved_checkpoint():
                self.logger.info(f"[3c] Skipping {tag.upper()} — no checkpoint")
                continue

            try:
                quantizer.load_tokenizer()
                if tag == "bf16":
                    quantizer.load_model()
                else:
                    loaded = quantizer.load_saved_checkpoint_only()
                    if not loaded:
                        continue

                result_tag = detector.run(
                    quantizer.model, quantizer.tokenizer,
                    sigma_threshold=sigma_threshold, tag=tag, prompts=prompt_suite,
                )
                results[tag] = result_tag
            except Exception as exc:
                self.logger.error(f"[3c] {tag.upper()} ✗: {exc}", exc_info=True)
            finally:
                try:
                    quantizer.unload()
                except Exception:
                    pass
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Module 4: Inference & Resource Profiling
# ═══════════════════════════════════════════════════════════════════════════

class InferenceProfilingModule(ModuleRunner):
    """Module 4 — Latency, VRAM, and disk size across all methods."""

    MODULE_NUM = 4
    MODULE_NAME = "Inference Profiling"

    def execute(self):
        from src.profiling.latency import LatencyProfiler
        from src.profiling.vram import VRAMProfiler
        from src.profiling.disk import DiskProfiler
        from src.core.memory import reset_peak_memory
        from src.quantization import QUANTIZER_REGISTRY

        latency_profiler = LatencyProfiler(self.config)
        vram_profiler = VRAMProfiler(self.config)
        disk_profiler = DiskProfiler(self.config)

        all_latency, all_vram, all_disk = {}, {}, {}

        ordered_tags = ["bf16", "int8", "fp8", "nf4", "gptq"]
        quantizer_filter = getattr(self.config, "_quantizer_filter", None)
        selected_tags = (
            {tag.lower() for tag in quantizer_filter}
            if quantizer_filter
            else set(ordered_tags)
        )

        self.results["methods"] = {
            "latency": "token-generation throughput benchmark across configured batch sizes",
            "vram": "post-inference VRAM snapshot plus peak-memory counters",
            "disk": "serialized checkpoint size measurement from filesystem/HF cache",
        }
        self.results["quantizers"] = {}

        for tag in ordered_tags:
            filtered = tag not in selected_tags
            status = "⚠ FILTERED" if filtered else "⚠ SKIPPED"
            error = (
                "Skipped by quantizer filter"
                if filtered
                else "Profiling not executed"
            )
            self.results["quantizers"][tag] = {
                "status": status,
                "time_sec": 0.0,
                "latency": {"status": status, "error": error},
                "vram": {"status": status, "error": error},
                "disk": {"status": status, "error": error},
            }

        quantizers = []
        for tag in ordered_tags:
            if tag not in selected_tags:
                continue
            quant_cls = QUANTIZER_REGISTRY.get(tag)
            if quant_cls is None:
                self.results["quantizers"][tag] = {
                    "status": "⚠ SKIPPED",
                    "time_sec": 0.0,
                    "latency": {"status": "⚠ SKIPPED", "error": "Quantizer not registered"},
                    "vram": {"status": "⚠ SKIPPED", "error": "Quantizer not registered"},
                    "disk": {"status": "⚠ SKIPPED", "error": "Quantizer not registered"},
                }
                self.logger.warning(f"[{tag.upper()}] Quantizer not registered — skipping")
                continue
            quantizers.append(quant_cls)

        for idx, QuantCls in enumerate(quantizers, 1):
            tag = QuantCls.QUANT_TAG

            def _profile_quantizer(cls=QuantCls, _tag=tag):
                q = cls(self.config)
                q.load_tokenizer()
                reset_peak_memory()

                if _tag == "bf16":
                    q.load_model()
                else:
                    if not q.has_saved_checkpoint():
                        self.logger.warning(f"[{_tag.upper()}] No checkpoint — skipping")
                        return None
                    if not q.load_saved_checkpoint_only():
                        self.logger.warning(f"[{_tag.upper()}] Cannot load checkpoint — skipping")
                        return None

                try:
                    # Latency
                    latency_data = latency_profiler.profile(q.model, q.tokenizer, _tag)
                    all_latency[_tag] = latency_data

                    # VRAM
                    vram_data = vram_profiler.snapshot(_tag)
                    all_vram[_tag] = vram_data

                    # Disk
                    static_memory = q.measure_static_memory()
                    if _tag == "bf16":
                        disk_data = disk_profiler.measure_model_storage(
                            model_ref=q.model_id,
                            model_id=self.config.model.model_id,
                            cache_dir=getattr(self.config.model, "cache_dir", None),
                        )
                    else:
                        disk_data = disk_profiler.measure_model_storage(
                            model_ref=str(q._saved_quantized_dir),
                            cache_dir=getattr(self.config.model, "cache_dir", None),
                        )

                    runtime_gb = static_memory.get("model_size_gb")
                    disk_data["runtime_exposed_weight_footprint_gb"] = runtime_gb
                    disk_data["runtime_total_parameters"] = static_memory.get("total_parameters")
                    if runtime_gb:
                        disk_data["disk_minus_runtime_gb"] = round(
                            disk_data["total_gb"] - runtime_gb, 3
                        )
                    all_disk[_tag] = disk_data

                    return {"latency": latency_data, "vram": vram_data, "disk": disk_data}
                finally:
                    try:
                        q.unload()
                    except Exception:
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()

            submodule_name = f"{tag.upper()} profiling"
            result = self.run_submodule(submodule_name, _profile_quantizer)

            submodule_meta = self.results["submodules"].get(submodule_name, {})
            quant_entry = self.results["quantizers"][tag]
            quant_entry["time_sec"] = submodule_meta.get("time_sec", 0.0)

            if isinstance(result, dict):
                quant_entry["status"] = "✓ COMPLETED"
                quant_entry["latency"] = {
                    "status": "✓ COMPLETED",
                    "method": self.results["methods"]["latency"],
                    **result.get("latency", {}),
                }
                quant_entry["vram"] = {
                    "status": "✓ COMPLETED",
                    "method": self.results["methods"]["vram"],
                    **result.get("vram", {}),
                }
                quant_entry["disk"] = {
                    "status": "✓ COMPLETED",
                    "method": self.results["methods"]["disk"],
                    **result.get("disk", {}),
                }
                continue

            failed = str(submodule_meta.get("status", "")).startswith("✗")
            warn_status = "⚠ FAILED" if failed else "⚠ SKIPPED"
            warn_error = submodule_meta.get(
                "error",
                "Checkpoint missing, load failed, or profiling was skipped",
            )
            quant_entry["status"] = warn_status
            quant_entry["latency"] = {
                "status": warn_status,
                "error": warn_error,
            }
            quant_entry["vram"] = {
                "status": warn_status,
                "error": warn_error,
            }
            quant_entry["disk"] = {
                "status": warn_status,
                "error": warn_error,
            }

        # Generate comparison plots
        self._safe_plot("latency", lambda: latency_profiler._plot_comparison(all_latency))
        self._safe_plot("vram", lambda: vram_profiler.plot_comparison(all_vram))
        self._safe_plot("disk", lambda: disk_profiler.plot_comparison(all_disk))

        # Persist results
        results_dir = Path(self.config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        for name, data in [("latency", all_latency), ("vram", all_vram), ("disk", all_disk)]:
            with open(results_dir / f"{name}_results.json", "w") as fh:
                json.dump(data, fh, indent=2, default=str)

    def _safe_plot(self, name: str, fn):
        try:
            fn()
            self.logger.info(f"  ✓ {name} comparison plot saved")
        except Exception as exc:
            self.logger.warning(f"  ✗ {name} plot failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Module 5: Evaluation (Perplexity, Benchmarks, Pareto)
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationModule(ModuleRunner):
    """Module 5 — Perplexity, benchmarks, and Pareto frontier."""

    MODULE_NUM = 5
    MODULE_NAME = "Evaluation & Accuracy"

    def execute(self):
        from src.evaluation.perplexity import PerplexityEvaluator
        from src.evaluation.benchmarks import BenchmarkRunner
        from src.evaluation.pareto import ParetoAnalyzer
        from src.quantization import QUANTIZER_REGISTRY

        ppl_eval = PerplexityEvaluator(self.config)
        bench_runner = BenchmarkRunner(self.config)

        # Load previously persisted results for partial-run merging
        results_dir = Path(self.config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        all_ppl = self._load_json(results_dir / "perplexity_results.json", {})
        all_bench = self._load_json(results_dir / "benchmark_results.json", {})

        ordered_tags = ["bf16", "int8", "fp8", "nf4", "gptq"]
        quantizer_filter = getattr(self.config, "_quantizer_filter", None)
        selected_tags = (
            {tag.lower() for tag in quantizer_filter}
            if quantizer_filter
            else set(ordered_tags)
        )

        self.results["methods"] = {
            "perplexity": "sliding-window language modeling perplexity",
            "benchmarks": "lm-evaluation-harness with grouped Sarvam-aligned benchmark configs",
            "pareto": "pareto-optimal analysis from benchmark quality score and throughput",
        }
        self.results["quantizers"] = {}
        self.results["pareto"] = {
            "status": "⚠ SKIPPED",
            "method": self.results["methods"]["pareto"],
            "time_sec": 0.0,
            "error": "Pareto analysis not executed",
        }

        for tag in ordered_tags:
            filtered = tag not in selected_tags
            status = "⚠ FILTERED" if filtered else "⚠ SKIPPED"
            reason = (
                "Skipped by quantizer filter"
                if filtered
                else "Evaluation not executed"
            )
            self.results["quantizers"][tag] = {
                "status": status,
                "time_sec": 0.0,
                "perplexity": {
                    "status": status,
                    "error": reason,
                },
                "benchmarks": {
                    "status": status,
                    "error": reason,
                },
            }

        quantizers = []
        for tag in ordered_tags:
            if tag not in selected_tags:
                continue
            quant_cls = QUANTIZER_REGISTRY.get(tag)
            if quant_cls is None:
                self.results["quantizers"][tag] = {
                    "status": "⚠ SKIPPED",
                    "time_sec": 0.0,
                    "perplexity": {
                        "status": "⚠ SKIPPED",
                        "error": "Quantizer not registered",
                    },
                    "benchmarks": {
                        "status": "⚠ SKIPPED",
                        "error": "Quantizer not registered",
                    },
                }
                self.logger.warning(
                    f"[{tag.upper()}] Quantizer not registered — skipping"
                )
                continue
            quantizers.append(quant_cls)

        for idx, QuantCls in enumerate(quantizers, 1):
            tag = QuantCls.QUANT_TAG
            submodule_name = f"{tag.upper()} evaluation"
            quant_entry = self.results["quantizers"][tag]

            # Skip if already evaluated from a prior run
            if (
                tag in all_ppl and "error" not in all_ppl.get(tag, {})
                and tag in all_bench and "error" not in all_bench.get(tag, {})
            ):
                self.logger.info(f"  {tag.upper()} — already evaluated, skipping")
                self.results["submodules"][submodule_name] = {
                    "status": "✓ COMPLETED (cached)",
                    "time_sec": 0.0,
                }
                quant_entry["status"] = "✓ COMPLETED (cached)"
                quant_entry["time_sec"] = 0.0
                quant_entry["perplexity"] = {
                    "status": "✓ COMPLETED (cached)",
                    "method": self.results["methods"]["perplexity"],
                    **all_ppl.get(tag, {}),
                }
                quant_entry["benchmarks"] = {
                    "status": "✓ COMPLETED (cached)",
                    "method": self.results["methods"]["benchmarks"],
                    **all_bench.get(tag, {}),
                }
                continue

            def _evaluate(cls=QuantCls, _tag=tag):
                q = cls(self.config)
                q.load_tokenizer()

                # BF16 loads the full model; quantized models load saved
                # checkpoints from Module 2 (seconds vs re-quantizing)
                if _tag == "bf16":
                    q.load_model()
                else:
                    if not q.has_saved_checkpoint():
                        self.logger.warning(
                            f"  [{_tag.upper()}] No saved checkpoint — skipping "
                            f"(run Module 2 first)"
                        )
                        return None
                    if not q.load_saved_checkpoint_only():
                        self.logger.warning(
                            f"  [{_tag.upper()}] Failed to load checkpoint — skipping"
                        )
                        return None

                try:
                    # Perplexity
                    ppl_data = ppl_eval.evaluate(q.model, q.tokenizer, _tag)
                    all_ppl[_tag] = ppl_data

                    # Benchmarks
                    bench_data = bench_runner.evaluate(q.model, q.tokenizer, _tag)
                    all_bench[_tag] = bench_data

                    return {"perplexity": ppl_data, "benchmarks": bench_data}
                finally:
                    q.unload()
                    gc.collect()
                    torch.cuda.empty_cache()

            result = self.run_submodule(submodule_name, _evaluate)

            submodule_meta = self.results["submodules"].get(submodule_name, {})
            quant_entry["time_sec"] = submodule_meta.get("time_sec", 0.0)

            if isinstance(result, dict):
                quant_entry["status"] = "✓ COMPLETED"
                quant_entry["perplexity"] = {
                    "status": "✓ COMPLETED",
                    "method": self.results["methods"]["perplexity"],
                    **result.get("perplexity", {}),
                }
                quant_entry["benchmarks"] = {
                    "status": "✓ COMPLETED",
                    "method": self.results["methods"]["benchmarks"],
                    **result.get("benchmarks", {}),
                }
            else:
                failed = str(submodule_meta.get("status", "")).startswith("✗")
                warn_status = "⚠ FAILED" if failed else "⚠ SKIPPED"
                warn_error = submodule_meta.get(
                    "error",
                    "Checkpoint missing, load failed, or evaluation was skipped",
                )
                quant_entry["status"] = warn_status
                quant_entry["perplexity"] = {
                    "status": warn_status,
                    "error": warn_error,
                }
                quant_entry["benchmarks"] = {
                    "status": warn_status,
                    "error": warn_error,
                }

            # ── Incremental save after each model (survive crashes) ──
            with open(results_dir / "perplexity_results.json", "w") as fh:
                json.dump(all_ppl, fh, indent=2, default=str)
            with open(results_dir / "benchmark_results.json", "w") as fh:
                json.dump(all_bench, fh, indent=2, default=str)

        # 5c: Pareto analysis
        def _pareto():
            from src.core.paths import get_module_paths
            mod4_results = get_module_paths(self.config.output.base_dir, 4)["results_dir"]
            latency_path = Path(mod4_results) / "latency_results.json"
            if not latency_path.exists():
                self.logger.warning(
                    f"  ⚠ Module 4 latency results not found at {latency_path}. "
                    "Run Module 4 before Module 5 for a complete Pareto frontier. "
                    "Pareto plot will be generated without throughput data."
                )
            latency_data = self._load_json(latency_path, {})

            ppl_eval.plot_comparison(all_ppl)
            bench_runner.save_results(all_bench)
            bench_runner.plot_summary(all_bench)

            pareto = ParetoAnalyzer(self.config)
            plot_path = pareto.plot(latency_data, all_bench)
            return {
                "plot_path": str(plot_path),
                "latency_available": bool(latency_data),
            }

        pareto_result = self.run_submodule("Pareto analysis", _pareto)
        pareto_meta = self.results["submodules"].get("Pareto analysis", {})
        pareto_failed = str(pareto_meta.get("status", "")).startswith("✗")

        if pareto_failed:
            self.results["pareto"] = {
                "status": "✗ FAILED",
                "method": self.results["methods"]["pareto"],
                "time_sec": pareto_meta.get("time_sec", 0.0),
                "error": pareto_meta.get("error", "Pareto analysis failed"),
            }
        else:
            latency_available = bool(
                isinstance(pareto_result, dict)
                and pareto_result.get("latency_available")
            )
            pareto_status = "✓ COMPLETED" if latency_available else "⚠ PARTIAL"
            self.results["pareto"] = {
                "status": pareto_status,
                "method": self.results["methods"]["pareto"],
                "time_sec": pareto_meta.get("time_sec", 0.0),
            }
            if isinstance(pareto_result, dict) and pareto_result.get("plot_path"):
                self.results["pareto"]["plot_path"] = pareto_result["plot_path"]
            if not latency_available:
                self.results["pareto"]["error"] = (
                    "Module 4 latency results not found; Pareto output may be incomplete"
                )

        # Persist results
        with open(results_dir / "perplexity_results.json", "w") as fh:
            json.dump(all_ppl, fh, indent=2, default=str)
        with open(results_dir / "benchmark_results.json", "w") as fh:
            json.dump(all_bench, fh, indent=2, default=str)

    def _load_json(self, path: Path, default):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return default


# ═══════════════════════════════════════════════════════════════════════════
# Module 6: Layer Visualisation
# ═══════════════════════════════════════════════════════════════════════════

class VisualizationModule(ModuleRunner):
    """Module 6 — Weight distribution histograms per target layer."""

    MODULE_NUM = 6
    MODULE_NAME = "Layer Visualisation"

    def execute(self):
        from src.visualization.layer_viz import LayerVisualizer

        viz = LayerVisualizer(self.config)
        target_layers = self.config.visualization.target_layers
        all_tags = ["bf16", "int8", "fp8", "nf4", "gptq"]

        self.results["methods"] = {
            "6a_histograms": "side-by-side per-layer weight distribution histograms across available tags",
            "6b_heatmaps": "per-layer weight-value heatmaps with percentile clipping for visual comparability",
            "6c_diff_heatmaps": "per-layer quantization error heatmaps computed as (quantized - bf16)",
        }
        self.results["layers"] = {}

        self.logger.info(f"Target layers: {len(target_layers)}")

        for idx, layer_name in enumerate(target_layers, 1):
            def _visualize_layer(name=layer_name):
                layer_start = time.time()
                available = viz._available_tags(name, all_tags)
                layer_result: Dict[str, Any] = {
                    "status": "✗ FAILED",
                    "time_sec": 0.0,
                    "available_tags": available,
                    "plots": [],
                }

                if not available:
                    layer_result["error"] = "No cached weights found for this layer"
                    layer_result["time_sec"] = round(time.time() - layer_start, 2)
                    self.results["layers"][name] = layer_result
                    raise RuntimeError(layer_result["error"])

                plot_records: List[Dict[str, Any]] = []
                plot_errors: List[str] = []

                # 6a: Histograms
                hist_start = time.time()
                try:
                    hist_path = viz._plot_histograms(name, available)
                    plot_records.append(
                        {
                            "type": "histogram",
                            "path": str(hist_path),
                            "time_sec": round(time.time() - hist_start, 2),
                        }
                    )
                except Exception as exc:
                    plot_errors.append(f"histogram: {exc}")

                # 6b: Heatmaps
                heat_start = time.time()
                try:
                    heat_path = viz._plot_heatmaps(name, available)
                    plot_records.append(
                        {
                            "type": "heatmap",
                            "path": str(heat_path),
                            "time_sec": round(time.time() - heat_start, 2),
                        }
                    )
                except Exception as exc:
                    plot_errors.append(f"heatmap: {exc}")

                # 6c: Diff heatmaps (requires BF16 and at least one quantized tag)
                if "bf16" in available and len(available) > 1:
                    diff_start = time.time()
                    try:
                        diff_path = viz._plot_diff_heatmaps(name, available)
                        plot_records.append(
                            {
                                "type": "diff_heatmap",
                                "path": str(diff_path),
                                "time_sec": round(time.time() - diff_start, 2),
                            }
                        )
                    except Exception as exc:
                        plot_errors.append(f"diff_heatmap: {exc}")

                if plot_records and not plot_errors:
                    layer_result["status"] = "✓ COMPLETED"
                elif plot_records:
                    layer_result["status"] = "⚠ PARTIAL"
                else:
                    layer_result["status"] = "✗ FAILED"

                layer_result["plots"] = plot_records
                layer_result["time_sec"] = round(time.time() - layer_start, 2)
                if plot_errors:
                    layer_result["error"] = "; ".join(plot_errors)

                self.results["layers"][name] = layer_result

                if layer_result["status"].startswith("✗"):
                    raise RuntimeError(layer_result.get("error", "No plots generated"))

                return layer_result

            self.run_submodule(
                f"layer_{idx}_{layer_name.split('.')[-1]}",
                _visualize_layer,
            )

    def _determine_status(self) -> None:
        """Module 6 status is based on per-layer outcomes."""
        layers = self.results.get("layers", {})
        if not isinstance(layers, dict) or not layers:
            self.results["status"] = "✗ FAILED"
            return

        completed = 0
        partial = 0
        failed = 0
        for layer_result in layers.values():
            status = str(layer_result.get("status", "")) if isinstance(layer_result, dict) else ""
            if status.startswith("✓"):
                completed += 1
            elif status.startswith("⚠"):
                partial += 1
            else:
                failed += 1

        if failed and not (completed or partial):
            self.results["status"] = "⚠ PARTIAL"
        elif failed or partial:
            self.results["status"] = "⚠ PARTIAL"
        else:
            self.results["status"] = "✓ COMPLETED"


# ── Registry ────────────────────────────────────────────────────────────

MODULE_MAP = {
    1: BF16BaselineModule,
    2: QuantizationMatrixModule,
    3: WeightIntrospectionModule,
    4: InferenceProfilingModule,
    5: EvaluationModule,
    6: VisualizationModule,
}

MODULE_NAMES = {
    1: "BF16 Baseline",
    2: "Quantisation Matrix",
    3: "Weight Introspection",
    4: "Inference Profiling",
    5: "Evaluation & Accuracy",
    6: "Layer Visualisation",
}
