"""
Module 5b – Sarvam-30B benchmark evaluation.

This runner keeps the evaluation set practical for a quantization study while
aligning the reporting structure with the Sarvam-30B model card:
    - grouped benchmark configs with per-group decoding settings
    - graceful skipping when a task is unavailable in the installed lm-eval build
    - composite accuracy score for cross-quantizer comparison and Pareto plots
    - publication-friendly summary table visualization
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)

# Tasks that use generate_until (autoregressive decoding — slow)
_GENERATIVE_TASKS = {
    "gsm8k", "humaneval", "mbpp", "triviaqa", "drop", "ifeval", "math",
}

_PREFERRED_METRIC_KEYS = (
    "acc,none",
    "acc_norm,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "inst_level_strict_acc,none",
    "prompt_level_strict_acc,none",
    "pass@1,none",
    "pass@1",
    "acc",
    "score,none",
)


class BenchmarkRunner:
    """Wrapper around ``lm_eval.simple_evaluate``."""

    def __init__(self, config):
        self.config = config
        bench_cfg = config.evaluation.benchmarks
        self.limit = getattr(bench_cfg, "limit", None)
        self.default_num_fewshot = getattr(bench_cfg, "num_fewshot", 0)
        self.default_batch_size = getattr(bench_cfg, "batch_size", 4)
        self.default_max_gen_toks = getattr(bench_cfg, "max_gen_toks", 256)
        self.primary_metric = getattr(bench_cfg, "pareto_metric", "composite")
        self.benchmark_groups = self._build_benchmark_groups(bench_cfg)
        self.task_specs = [
            task for group in self.benchmark_groups for task in group["tasks"]
        ]
        self.tasks: List[str] = [task["id"] for task in self.task_specs]

        self.results_dir = Path(config.output.results_dir)
        self.plots_dir = Path(config.output.plots_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config.visualization.dpi

    def _build_benchmark_groups(self, bench_cfg) -> List[Dict[str, Any]]:
        raw_groups = getattr(bench_cfg, "benchmark_groups", None)
        if raw_groups:
            groups = []
            for raw_group in raw_groups:
                group = raw_group if isinstance(raw_group, dict) else raw_group.to_dict()
                task_specs = [
                    self._normalize_task_spec(task, group)
                    for task in group.get("tasks", [])
                ]
                groups.append(
                    {
                        "name": group.get("name", "default"),
                        "display_name": group.get("display_name", group.get("name", "Default")),
                        "num_fewshot": group.get("num_fewshot", self.default_num_fewshot),
                        "batch_size": group.get("batch_size", self.default_batch_size),
                        "max_gen_toks": group.get("max_gen_toks", self.default_max_gen_toks),
                        "temperature": group.get("temperature"),
                        "top_p": group.get("top_p"),
                        "max_new_tokens": group.get("max_new_tokens"),
                        "tasks": task_specs,
                    }
                )
            return groups

        legacy_group = {
            "name": "legacy",
            "display_name": "Legacy Benchmarks",
            "num_fewshot": getattr(bench_cfg, "num_fewshot", 0),
            "batch_size": getattr(bench_cfg, "batch_size", 4),
            "max_gen_toks": getattr(bench_cfg, "max_gen_toks", 256),
            "temperature": None,
            "top_p": None,
            "max_new_tokens": getattr(bench_cfg, "max_gen_toks", 256),
            "tasks": [
                self._normalize_task_spec(task, {})
                for task in getattr(bench_cfg, "tasks", [])
            ],
        }
        return [legacy_group]

    def _normalize_task_spec(
        self,
        task: Any,
        group: Dict[str, Any],
    ) -> Dict[str, Any]:
        if isinstance(task, str):
            task = {"id": task, "label": task.upper()}

        task_id = task.get("id")
        label = task.get("label", task_id)
        generative = task.get("generative", task_id in _GENERATIVE_TASKS)

        return {
            "id": task_id,
            "label": label,
            "group": group.get("name", "default"),
            "group_display_name": group.get("display_name", group.get("name", "Default")),
            "weight": float(task.get("weight", 1.0)),
            "generative": generative,
            "num_fewshot": task.get("num_fewshot", group.get("num_fewshot", self.default_num_fewshot)),
            "batch_size": task.get("batch_size", group.get("batch_size", self.default_batch_size)),
            "max_gen_toks": task.get("max_gen_toks", group.get("max_gen_toks", self.default_max_gen_toks)),
            "temperature": task.get("temperature", group.get("temperature")),
            "top_p": task.get("top_p", group.get("top_p")),
            "max_new_tokens": task.get("max_new_tokens", group.get("max_new_tokens")),
            "limit": task.get("limit", group.get("limit", self.limit)),
        }

    @staticmethod
    def _candidate_batch_sizes(initial_batch_size: int) -> List[int]:
        candidates: List[int] = []
        for size in (initial_batch_size, 4, 2, 1):
            if isinstance(size, int) and size > 0 and size not in candidates:
                candidates.append(size)
        return candidates

    # ── resolve batch size ──────────────────────────────────────────────
    def _resolve_batch_size(self, spec: Dict[str, Any]) -> int:
        """
        Return a concrete integer batch size.

        ``batch_size="auto"`` from lm_eval probes exponentially and can
        hang on wrapped quantised tensors.  Replace it with a safe fixed
        value based on the workload type.
        """
        bs = spec.get("batch_size", self.default_batch_size)
        if isinstance(bs, int) and bs > 0:
            return bs
        # "auto" or invalid → sensible fixed defaults
        # Loglikelihood tasks process many short sequences → larger batch
        # Generative tasks hold full KV-cache → smaller batch
        return 4 if spec.get("generative") else 8

    @staticmethod
    def _extract_accuracy(metrics: Dict[str, Any]) -> Optional[float]:
        for key in _PREFERRED_METRIC_KEYS:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value) * 100 if value <= 1.0 else float(value)
        return None

    @staticmethod
    def _build_gen_kwargs(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not spec.get("generative"):
            return None

        gen_kwargs: Dict[str, Any] = {}
        if spec.get("temperature") is not None:
            gen_kwargs["temperature"] = spec["temperature"]
        if spec.get("top_p") is not None:
            gen_kwargs["top_p"] = spec["top_p"]
        if spec.get("max_new_tokens") is not None:
            gen_kwargs["max_new_tokens"] = spec["max_new_tokens"]

        if gen_kwargs and "temperature" in gen_kwargs and gen_kwargs["temperature"] > 0:
            gen_kwargs["do_sample"] = True

        return gen_kwargs or None

    @staticmethod
    def _available_tasks() -> Optional[set[str]]:
        try:
            from lm_eval.tasks import TaskManager

            manager = TaskManager()
            if hasattr(manager, "all_tasks"):
                tasks = manager.all_tasks
                if callable(tasks):
                    tasks = tasks()
                return set(tasks)
            if hasattr(manager, "task_index"):
                return set(manager.task_index.keys())
        except Exception:
            return None
        return None

    @staticmethod
    def _call_simple_evaluate(lm_eval_mod, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return lm_eval_mod.simple_evaluate(**kwargs)
        except TypeError:
            fallback = dict(kwargs)
            fallback.pop("gen_kwargs", None)
            return lm_eval_mod.simple_evaluate(**fallback)

    # ── run a single task group ─────────────────────────────────────────
    def _run_task(
        self, lm_eval_mod, lm, spec: Dict[str, Any], tag: str,
    ) -> Dict[str, Any]:
        """Run a single task and return parsed results."""
        from lm_eval.models.huggingface import HFLM

        initial_bs = self._resolve_batch_size(spec)
        kind = "generative" if spec.get("generative") else "loglikelihood"
        task_limit = spec.get("limit")
        gen_kwargs = self._build_gen_kwargs(spec)
        last_oom: Optional[Exception] = None

        for bs in self._candidate_batch_sizes(initial_bs):
            logger.info(
                f"  [{tag}] Running {spec['label']} ({spec['id']}, {kind}) "
                f"(batch_size={bs}, fewshot={spec['num_fewshot']}, limit={task_limit})"
            )

            lm_obj = HFLM(
                pretrained=lm.model if hasattr(lm, "model") else lm,
                tokenizer=lm.tokenizer if hasattr(lm, "tokenizer") else None,
                batch_size=bs,
                max_gen_toks=spec.get("max_gen_toks") if spec.get("generative") else None,
            )

            t0 = time.time()
            try:
                raw = self._call_simple_evaluate(
                    lm_eval_mod,
                    {
                        "model": lm_obj,
                        "tasks": [spec["id"]],
                        "num_fewshot": spec["num_fewshot"],
                        "limit": task_limit,
                        "gen_kwargs": gen_kwargs,
                        "confirm_run_unsafe_code": True,
                    },
                )
            except torch.cuda.OutOfMemoryError as exc:
                last_oom = exc
                logger.warning(
                    f"  [{tag}] {spec['id']} OOM at batch_size={bs}; retrying with smaller batch size"
                )
                torch.cuda.empty_cache()
                continue

            elapsed = time.time() - t0
            logger.info(f"  [{tag}] {spec['label']} done in {elapsed:.1f}s")

            metrics = raw.get("results", {}).get(spec["id"], {})
            metrics = {k: v for k, v in metrics.items() if not k.startswith("alias")}
            accuracy = self._extract_accuracy(metrics)

            logger.info(f"    [{tag}] {spec['label']}: score = {accuracy}")
            return {
                "status": "✓ COMPLETED",
                "task_id": spec["id"],
                "label": spec["label"],
                "group": spec["group"],
                "accuracy": accuracy,
                "weight": spec["weight"],
                "kind": kind,
                "num_fewshot": spec["num_fewshot"],
                "batch_size": bs,
                "max_gen_toks": spec.get("max_gen_toks"),
                "limit": task_limit,
                "gen_kwargs": gen_kwargs,
                "time_sec": round(elapsed, 2),
                "raw": metrics,
            }

        if last_oom is not None:
            raise last_oom
        raise RuntimeError(f"Task {spec['id']} did not produce a result")

    def _summarize_results(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        task_entries = parsed.get("tasks", {})
        completed = []
        skipped = []
        failed = []
        weighted_sum = 0.0
        total_weight = 0.0
        primary_score = None

        for task_id, task_data in task_entries.items():
            status = task_data.get("status", "")
            if status.startswith("✓"):
                completed.append(task_id)
                score = task_data.get("accuracy")
                weight = float(task_data.get("weight", 1.0))
                if isinstance(score, (int, float)):
                    weighted_sum += float(score) * weight
                    total_weight += weight
                    if task_id == self.primary_metric:
                        primary_score = float(score)
            elif status.startswith("⚠"):
                skipped.append(task_id)
            else:
                failed.append(task_id)

        composite_score = round(weighted_sum / total_weight, 2) if total_weight else None
        if self.primary_metric == "composite":
            primary_score = composite_score

        return {
            "completed_tasks": completed,
            "skipped_tasks": skipped,
            "failed_tasks": failed,
            "composite_score": composite_score,
            "primary_metric": self.primary_metric,
            "primary_score": round(primary_score, 2) if isinstance(primary_score, (int, float)) else None,
            "num_tasks_total": len(task_entries),
            "num_tasks_completed": len(completed),
        }

    # ── run benchmarks for one model ────────────────────────────────────
    def evaluate(
        self, model, tokenizer, tag: str
    ) -> Dict[str, Any]:
        """
        Run configured benchmark tasks, splitting loglikelihood and
        generative workloads so each gets an optimal batch size.

        Returns:
            Dict with per-task accuracy and metadata.
        """
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
        except ImportError:
            raise ImportError(
                "lm-evaluation-harness is not installed.  "
                "Install with:  pip install lm-eval"
            )

        logger.info(
            f"Running Sarvam benchmark suite [{tag}]: {self.tasks}  "
            f"(primary_metric={self.primary_metric}, limit={self.limit})"
        )

        # Lightweight sentinel wrapper — holds model+tokenizer for _run_tasks
        class _ModelRef:
            pass
        ref = _ModelRef()
        ref.model = model
        ref.tokenizer = tokenizer

        available_tasks = self._available_tasks()
        parsed: Dict[str, Any] = {
            "tag": tag,
            "tasks": {},
            "suite": {
                "task_ids": self.tasks,
                "primary_metric": self.primary_metric,
                "groups": [group["name"] for group in self.benchmark_groups],
            },
        }

        for spec in self.task_specs:
            task_id = spec["id"]
            if available_tasks is not None and task_id not in available_tasks:
                parsed["tasks"][task_id] = {
                    "status": "⚠ SKIPPED",
                    "task_id": task_id,
                    "label": spec["label"],
                    "group": spec["group"],
                    "weight": spec["weight"],
                    "reason": "task_not_available_in_lm_eval",
                }
                parsed[task_id] = {
                    "accuracy": None,
                    "status": "⚠ SKIPPED",
                    "label": spec["label"],
                    "group": spec["group"],
                }
                logger.warning(f"  [{tag}] Skipping {task_id} — task unavailable in lm-eval")
                continue

            try:
                result = self._run_task(lm_eval, ref, spec, tag)
            except Exception as exc:
                result = {
                    "status": "✗ FAILED",
                    "task_id": task_id,
                    "label": spec["label"],
                    "group": spec["group"],
                    "weight": spec["weight"],
                    "error": str(exc),
                }
                logger.warning(f"  [{tag}] {task_id} failed: {exc}")

            parsed["tasks"][task_id] = result
            parsed[task_id] = {
                "accuracy": result.get("accuracy"),
                "status": result.get("status"),
                "label": result.get("label"),
                "group": result.get("group"),
                "raw": result.get("raw", {}),
            }

        parsed["summary"] = self._summarize_results(parsed)
        return parsed

    # ── persist results ─────────────────────────────────────────────────
    def save_results(self, all_results: Dict[str, Dict]) -> Path:
        path = self.results_dir / "benchmark_results.json"
        with open(path, "w") as fh:
            json.dump(all_results, fh, indent=2, default=str)
        logger.info(f"Benchmark results: {path}")
        return path

    def plot_summary(self, all_results: Dict[str, Dict]) -> Path:
        task_ids: List[str] = []
        labels: List[str] = []
        for spec in self.task_specs:
            if spec["id"] not in task_ids:
                task_ids.append(spec["id"])
                labels.append(spec["label"])

        tags = list(all_results.keys())
        if not tags or not task_ids:
            return self.plots_dir / "benchmark_accuracy_table.png"

        matrix = np.full((len(tags), len(task_ids)), np.nan)
        for row_idx, tag in enumerate(tags):
            tag_results = all_results.get(tag, {})
            for col_idx, task_id in enumerate(task_ids):
                score = tag_results.get("tasks", {}).get(task_id, {}).get("accuracy")
                if isinstance(score, (int, float)):
                    matrix[row_idx, col_idx] = float(score)

        summary_scores = [
            all_results.get(tag, {}).get("summary", {}).get("composite_score")
            for tag in tags
        ]

        fig_h = max(4.2, 0.8 + 0.8 * len(tags))
        fig_w = max(10.5, 1.2 * len(task_ids) + 3.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        masked = np.ma.masked_invalid(matrix)
        cmap = plt.cm.get_cmap("YlGnBu").copy()
        cmap.set_bad(color="#f2f2f2")
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)

        ax.set_xticks(range(len(task_ids)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        composite_labels = [
            f"{tag.upper()}\nComp {score:.1f}" if isinstance(score, (int, float)) else f"{tag.upper()}\nComp n/a"
            for tag, score in zip(tags, summary_scores)
        ]
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels(composite_labels, fontsize=10)
        ax.set_title(
            "Sarvam-30B Quantization Accuracy Table",
            fontsize=15,
            fontweight="bold",
        )

        for row_idx in range(len(tags)):
            for col_idx in range(len(task_ids)):
                value = matrix[row_idx, col_idx]
                text = "--" if np.isnan(value) else f"{value:.1f}"
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#0b0b0b",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Accuracy / Score (%)")
        fig.tight_layout()

        out = self.plots_dir / "benchmark_accuracy_table.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Benchmark summary table: {out}")
        return out
